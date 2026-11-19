"""Generate a fixed-size resident state ABI with an optional score transport study."""

import argparse
import ctypes as ffi
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import scalar_transition_standalone as transition
from scar.codegen.scalar_table_standalone import _dependencies_for, _FLAGS, _LINK_FLAGS, _local, _ref, _require
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.scalar-transition-resident-library'
SYMBOL = 'scar_resident_transition_v1'
_SONAME = 'libscar_scalar_transition_resident_v1.so'
_LIBRARY = 'lib/' + _SONAME
_HEADER = 'include/scar_scalar_transition_resident_v1.h'
_SOURCE = 'src/transition.cpp'
_MAP = 'exports.map'
_NATIVE = ('-std=c++20', *_FLAGS, '-shared', '-fPIC', '-fvisibility=hidden', '-fvisibility-inlines-hidden',
           *_LINK_FLAGS, '-Wl,-soname,' + _SONAME, '-Wl,--version-script,exports.map')
_MEMBERS = {_SOURCE: 'scar.cpp-source', _HEADER: 'scar.c-header', _MAP: 'scar.elf-export-map'}
_TEMPLATE = Path(__file__).with_name('scalar_transition_resident_v1.cpp.in')
_HEADER_TEMPLATE = Path(__file__).with_name('scalar_transition_resident_v1.h.in')
_STATUS = dict(native=0, transported=1, declined=2, internal_error=3)


def _layout(ports, records):
    fields = transition._schema(ports)
    rows = []
    scalars = lengths = strings = 0
    for record in records:
        _require(record.keys() == fields.keys() and all(transition._valid(record[name], kind)
                  for name,kind in fields.items()), 'resident shape requires exact original typed records')
    for name, kind in fields.items():
        if kind == 'str':
            capacity = max(len(record[name]) for record in records)
            _require(capacity <= 2**32 - 1, 'resident text extent exceeds the ABI')
            rows.append(dict(name=name, type=kind, slot=lengths, offset=strings, capacity=capacity))
            lengths += 1
            strings += capacity
        else:
            rows.append(dict(name=name, type=kind, slot=scalars))
            scalars += 1
    return dict(ports=rows, scalars=max(1, scalars), lengths=max(1, lengths), strings=max(1, strings))


def _abi(plan, transport):
    observed = plan['samples']
    _require(observed and {sample['table_version'] for sample in observed} == plan['table_versions'].keys(),
             'resident bounds need original samples for every table version')
    return dict(version=1, symbol=SYMBOL, input=_layout(plan['inputs'], [row['inputs'] for row in observed]),
                output=_layout(plan['outputs'], [row['outputs'] for row in observed]),
                table_slots=sorted(plan['table_versions']), statuses=_STATUS,
                bounds='EXACT_TYPED_ASCII_WITH_OBSERVED_PER_PORT_MAXIMUM_LENGTH',
                output_effect='FIXED_POD_PUBLISHED_ONLY_AFTER_ALL_COMPUTATION_AND_BOUNDS_CHECKS',
                score_study=transport is not None,
                receipt_tag=0 if transport is None else int(transport['sha256'][:16], 16),
                source_producer_guard='REQUIRED_FROM_ISOLATED_CALLER_NOT_PROVEN_BY_POD_SNAPSHOT')


def resident_types(abi):
    """Construct ctypes mirrors of the published native PODs for external hosts."""
    def layout(name, value):
        fields = [('scalars', ffi.c_int64 * value['scalars']), ('lengths', ffi.c_uint32 * value['lengths']),
                  ('strings', ffi.c_uint8 * value['strings'])]
        return type(name, (ffi.Structure,), {'_fields_': fields})

    inputs = layout('Input', abi['input'])
    outputs = layout('Output', abi['output'])
    token = type('Score', (ffi.Structure,), {'_fields_': [('receipt_tag', ffi.c_uint64), ('table_slot', ffi.c_uint32),
                 ('delta', ffi.c_int64), ('source_inputs', inputs)]})
    return inputs, outputs, token


def pack(layout, structure, values):
    """Initialize one resident POD from exact source leaves at a wider boundary."""
    _require(values.keys() == {row['name'] for row in layout['ports']}, 'resident fields differ from the declared ports')
    result = structure()
    for port in layout['ports']:
        kind, slot = port['type'], port['slot']
        value = values[port['name']]
        _require(transition._valid(value, kind), 'resident value differs from the exact port type')
        if kind == 'str':
            size, start = len(value), port['offset']
            _require(size <= port['capacity'], 'resident string length exceeds the compiled capacity')
            result.lengths[slot] = size
            result.strings[start:start+size] = value.encode('ascii')
        else:
            result.scalars[slot] = value
    return result


def load(path, types):
    library = ffi.CDLL(str(Path(path).resolve(strict=True)))
    function = getattr(library, SYMBOL)
    inputs, outputs, score = types
    function.argtypes = [ffi.c_uint32, ffi.POINTER(inputs), ffi.POINTER(score), ffi.POINTER(outputs)]
    function.restype = ffi.c_int
    return function


def _computation(plan, scorer):
    rendered = transition._render(plan, scorer).decode()
    boundary = '[[maybe_unused]] void append_field(std::string& output, const std::string& value) {\n'
    _require(rendered.count(boundary) == 1, 'transition lowering has no unique typed return boundary')
    return rendered.split(boundary, 1)[0]


def _transport(plan, scorer, proof, abi, args):
    native = f'evaluate_transition(table, {args})'
    if proof is None:
        return '', f'        (void)score;\n        constexpr bool use_score = false;\n        const auto state = {native};'
    lowered = dict(plan, inputs=proof['inputs'], body=proof['body'])
    variant = transition._TransitionEmitter(lowered, scorer).evaluate(lowered['body']).split('\n\n', 1)[1]
    variant = variant.replace('State evaluate_transition(', 'State evaluate_transported(', 1)
    guard = ['bool score_matches(const scar_resident_score_v1& token, uint32_t slot, const scar_resident_input_v1& input) {',
             '    if (token.receipt_tag != SCAR_RESIDENT_SCORE_RECEIPT_TAG_V1 || token.table_slot != slot) return false;']
    for row in abi['input']['ports']:
        slot = row['slot']
        if row['type'] == 'str':
            offset = row['offset']
            guard.append(f'    if (token.source_inputs.lengths[{slot}] != input.lengths[{slot}] ||'
                         f' std::memcmp(token.source_inputs.strings + {offset}, input.strings + {offset}, '
                         f'input.lengths[{slot}]) != 0) return false;')
        else:
            guard.append(f'    if (token.source_inputs.scalars[{slot}] != input.scalars[{slot}]) return false;')
    guard.extend(['    return true;', '}'])
    chosen = ('        const bool use_score = score && score_matches(*score, table_slot, *input);\n'
              f'        const auto state = use_score ? evaluate_transported(table, {args}, score->delta) : {native};')
    return variant + '\n\n' + '\n'.join(guard), chosen


def _render(plan, scorer, abi, proof):
    inputs, outputs = [], []
    for i, row in enumerate(abi['input']['ports']):
        slot = row['slot']
        if row['type'] == 'str':
            text = f'read_text(input->strings + {row["offset"]}, input->lengths[{slot}], {row["capacity"]})'
        else:
            text = f'input->scalars[{slot}]'
            if row['type'] == 'bool':
                text = f'read_bool({text})'
        inputs.append(f'        const auto in_{i} = {text};')
    for i, row in enumerate(abi['output']['ports']):
        slot = row['slot']
        if row['type'] == 'str':
            outputs.append(f'        write_text(staged.strings + {row["offset"]}, staged.lengths[{slot}], '
                           f'{row["capacity"]}, std::get<{i}>(state));')
        else:
            outputs.append(f'        staged.scalars[{slot}] = std::get<{i}>(state);')
    args = ', '.join(f'in_{i}' for i in range(len(inputs)))
    variant, chosen = _transport(plan, scorer, proof, abi, args)
    source = _TEMPLATE.read_text()
    fields = dict(COMPUTATION=_computation(plan, scorer), TRANSPORT=variant, INPUT_SETUP='\n'.join(inputs),
                  OUTPUT_SETUP='\n'.join(outputs), EVALUATION=chosen)
    for key, value in fields.items():
        source = source.replace('@SCAR_' + key + '@', value)
    header = _HEADER_TEMPLATE.read_text()
    for direction in ('input', 'output'):
        for field in ('scalars', 'lengths', 'strings'):
            header = header.replace('@SCAR_' + direction.upper() + '_' + field.upper() + '@', str(abi[direction][field]))
    header = header.replace('@SCAR_SCORE_STUDY@', str(int(abi['score_study']))).replace('@SCAR_SCORE_TAG@', str(abi['receipt_tag']))
    return {_SOURCE: source.encode(), _HEADER: header.encode(), _MAP: f'{{ global: {SYMBOL}; local: *; }};\n'.encode()}


def _proof(path, plan_ref, original):
    if path is None:
        return None, None
    from scar.optimization import scalar_score_reuse_plan as transport

    ref = _ref(path, transport.SCHEMA)
    proof = transport.check(ref['uri'])
    expected = [*original['inputs'], {'name': 'producer_score', 'type': 'int'}]
    _require(proof == read_artifact(ref) and proof['transition_plan'] == plan_ref and proof['inputs'] == expected
             and proof['disposition'] == 'WAIT_FOR_LIVE_VALUE_PROVENANCE' and proof['removed_score_calls'] == 1
             and proof['verified_sample_counts'] and proof['native_score_bypass_qualified'] is False
             and proof['deployment_authorized'] is False, 'resident score study needs the exact guarded transport plan')
    return ref, proof


def _transport_samples(proof):
    from scar.optimization import scalar_transition_ir as ir

    evidence = read_artifact(proof['evidence'])
    capture = read_artifact(proof['capture'])
    fields = {row['name']: ir._KINDS[row['kind']] for row in capture['state_fields']}
    for witness in evidence['samples']:
        observed = capture['samples'][witness['sample']]
        inputs = ir._flatten(fields, observed['receiver_fields'])
        inputs.update({f'arg_{i}': value for i, value in enumerate(observed['argument'])})
        yield observed['table_version'], inputs, ir._flatten(fields, observed['returned_fields']), observed['score_reuse']['first_result']


def _verify(path, plan, abi, proof):
    types = resident_types(abi)
    inputs, outputs, token = types
    action = load(path, types)
    state = outputs()
    native = []
    for sample in plan['samples']:
        value = pack(abi['input'], inputs, sample['inputs'])
        version = abi['table_slots'].index(sample['table_version'])
        status = action(version, ffi.byref(value), None, ffi.byref(state))
        expected = pack(abi['output'], outputs, sample['outputs'])
        _require(status == 0 and bytes(state) == bytes(expected), 'resident native transition differs from original state')
        native.append(bytes(state))
    transported = []
    if proof is not None:
        for identity, input_values, output_values, delta in _transport_samples(proof):
            value = pack(abi['input'], inputs, input_values)
            version = abi['table_slots'].index(identity)
            witness = token(abi['receipt_tag'], version, delta, value)
            expected = pack(abi['output'], outputs, output_values)
            status = action(version, ffi.byref(value), ffi.byref(witness), ffi.byref(state))
            _require(status == 1 and bytes(state) == bytes(expected), 'resident score transport differs from original return')
            transported.append(bytes(state))
        witness.receipt_tag ^= 1
        status = action(version, ffi.byref(value), ffi.byref(witness), ffi.byref(state))
        _require(status == 0 and bytes(state) == bytes(expected), 'resident score receipt mismatch did not recompute')
        witness.receipt_tag ^= 1
        first = abi['input']['ports'][0]
        if first['type'] == 'str':
            witness.source_inputs.lengths[first['slot']] ^= 1
        else:
            witness.source_inputs.scalars[first['slot']] ^= 1
        status = action(version, ffi.byref(value), ffi.byref(witness), ffi.byref(state))
        _require(status == 0 and bytes(state) == bytes(expected), 'resident stale producer inputs did not recompute')

    ffi.memset(ffi.addressof(state), 0xA5, ffi.sizeof(state))
    unchanged = bytes(state)
    status = action(len(abi['table_slots']), ffi.byref(value), None, ffi.byref(state))
    _require(status == 2 and bytes(state) == unchanged, 'resident unknown version changed output')
    invalid = next((row for row in abi['input']['ports'] if row['type'] == 'str'), None)
    if invalid is not None and invalid['capacity'] < 2**32 - 1:
        value.lengths[invalid['slot']] = invalid['capacity'] + 1
        status = action(version, ffi.byref(value), None, ffi.byref(state))
        _require(status == 2 and bytes(state) == unchanged, 'resident oversized input changed output')
    return b''.join(native), b''.join(transported)


def _links(path):
    result = _dependencies_for(path)
    export = subprocess.check_output(['/usr/bin/nm', '-D', '--defined-only', '--format=posix', str(path)], env={}, text=True)
    _require([line.split()[0] for line in export.splitlines()] == [SYMBOL], 'resident library has unexpected public symbols')
    return result


def _validation(root, abi, samples):
    size = ffi.sizeof(resident_types(abi)[1])
    rows = {name: _local(root, name + '.pod', 'application/octet-stream') for name in ('native', 'transported')}
    _require(rows['native']['size_bytes'] == size * samples and rows['transported']['size_bytes'] % size == 0,
             'resident sample extents differ from their POD ABI')
    return dict(native_samples=samples, transported_samples=rows['transported']['size_bytes'] // size,
                validated_relocated_library=True, output_struct_bytes=size, records=rows)


def export(plan_path, out, *, transport_path=None):
    """Build the resident ABI, optionally enabling the explicitly isolated score study."""
    ref, plan, scorer = transition._plan(plan_path)
    transport_ref, proof = _proof(transport_path, ref, plan)
    abi = _abi(plan, transport_ref)
    rendered = _render(plan, scorer, abi, proof)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    version = subprocess.check_output([*compiler, '--version'], env=env, text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-resident-transition-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'package'
        for name, data in rendered.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (stage / _LIBRARY).parent.mkdir()
        subprocess.run([*compiler, *_NATIVE, '-Iinclude', _SOURCE, '-o', _LIBRARY], cwd=stage, env=env, check=True)
        relocated = Path(temporary) / 'relocated resident library.so'
        copy2(stage / _LIBRARY, relocated)
        linkage = _links(relocated)
        for name, data in zip(('native', 'transported'), _verify(relocated, plan, abi, proof), strict=True):
            (stage / (name + '.pod')).write_bytes(data)
        record = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, plan=ref, score_plan=plan['score_plan'],
                      transport=transport_ref, abi=abi, scope='EXPLICIT_ISOLATED_RESIDENT_STATE_STUDY',
                      validation=_validation(stage, abi, len(plan['samples'])), linkage=linkage,
                      compiler=dict(command=compiler, version=version, flags=list(_NATIVE)),
                      library=_local(stage, _LIBRARY, 'scar.native-library-file'),
                      header=_local(stage, _HEADER, 'scar.c-header'),
                      members={name: _local(stage, name, kind) for name, kind in _MEMBERS.items()},
                      live_producer_guard_implemented=False, deployment_authorized=False, whole_program_cpp=False)
        (stage / 'artifact.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def check(path):
    """Cold-read the original plan, deterministic POD lowering and returned states."""
    ref = _ref(path, SCHEMA)
    record = read_artifact(ref)
    root = Path(ref['uri']).parent
    plan_ref, plan, scorer = transition._plan(record['plan']['uri'])
    transport = record['transport']
    transport_ref, proof = _proof(transport['uri'] if transport else None, plan_ref, plan)
    abi = _abi(plan, transport_ref)
    _require(record['schema_version'] == 1 and record['cxx_standard'] == 20
             and record['scope'] == 'EXPLICIT_ISOLATED_RESIDENT_STATE_STUDY' and record['plan'] == plan_ref
             and record['score_plan'] == plan['score_plan'] and transport == transport_ref and record['abi'] == abi
             and all(record[flag] is False for flag in ('live_producer_guard_implemented', 'deployment_authorized', 'whole_program_cpp')),
             'resident artifact differs from its exact isolated source study')
    source = _render(plan, scorer, abi, proof)
    library = root / _LIBRARY
    _require(record['members'] == {name: _local(root, name, kind) for name, kind in _MEMBERS.items()}
             and all((root / name).read_bytes() == data for name, data in source.items())
             and record['library'] == _local(root, _LIBRARY, 'scar.native-library-file')
             and record['header'] == record['members'][_HEADER] and record['compiler']['flags'] == list(_NATIVE)
             and record['linkage'] == _links(library), 'resident generated code or native linkage differs')
    actual = _verify(library, plan, abi, proof)
    _require(record['validation'] == _validation(root, abi, len(plan['samples']))
             and all((root / (name + '.pod')).read_bytes() == data
                     for name, data in zip(('native', 'transported'), actual, strict=True)),
             'resident POD sample records differ from fresh verification')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--transport')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.plan, args.transport, args.out)):
        result = check(args.check)
    elif args.plan and args.out and not args.check:
        result = export(args.plan, args.out, transport_path=args.transport)
    else:
        parser.error('provide --plan and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
