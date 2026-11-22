"""Compose exact ordered and lazy state regions in one native C++20 DSO."""

import argparse
import ctypes as ffi
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import scalar_sorted_library as sorted_library
from scar.codegen import scalar_transition_resident as resident
from scar.codegen.scalar_standalone import _dependencies_for, _FLAGS, _LINK_FLAGS, _local, _ref, _require
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.scalar-sorted-transition-library'
SYMBOL = 'scar_scalar_lazy_transition_v1'
_SONAME = 'libscar_scalar_sorted_transition_v1.so'
_LIBRARY = 'lib/' + _SONAME
_SOURCE = 'src/sorted_transition.cpp'
_HEADER = 'include/scar_scalar_sorted_transition_v1.h'
_MAP = 'exports.map'
_TEMPLATE = Path(__file__).with_name('scalar_sorted_transition_library_v1.cpp.in')
_HEADER_TEMPLATE = Path(__file__).with_name('scalar_sorted_transition_library_v1.h.in')
_MEMBERS = {_SOURCE: 'scar.cpp-source', _HEADER: 'scar.c-header', _MAP: 'scar.elf-export-map',
            sorted_library._HEADER: 'scar.c-header', resident._HEADER: 'scar.c-header'}
_NATIVE = ('-std=c++20', *_FLAGS, '-shared', '-fPIC', '-fvisibility=hidden', '-fvisibility-inlines-hidden',
           *_LINK_FLAGS, '-Wl,-soname,' + _SONAME, '-Wl,--version-script,exports.map')


def _by_name(layout):
    return {row['name']: row for row in layout['ports']}


def _bridge(abi):
    state = _by_name(abi['state'])
    move = _by_name(abi['move'])
    score = move.pop('score')
    original = _by_name(abi['input'])
    _require(score['type'] == 'int' and state.keys().isdisjoint(move) and original.keys() == state.keys() | move.keys(),
             'lazy input must consist of the exact current state and record argument ports')
    copy = []
    for name, port in original.items():
        owner = 'state' if name in state else 'move'
        received = (state if owner == 'state' else move)[name]
        slot, other = port['slot'], received['slot']
        _require(port['type'] == received['type'], 'ranked record or state field type differs from original transition')
        if port['type'] == 'str':
            length = f'{owner}->lengths[{other}]'
            capacity = min(received['capacity'], port['capacity'])
            copy.extend([f'    if ({length} > {capacity}) return SCAR_RESIDENT_DECLINED_V1;',
                         f'    input.lengths[{slot}] = {length};',
                         f'    std::memcpy(input.strings + {port["offset"]}, {owner}->strings + {received["offset"]}, {length});'])
        else:
            copy.append(f'    input.scalars[{slot}] = {owner}->scalars[{other}];')
    return '\n'.join(copy), str(score['slot'])


def _local_source(root, ref):
    return read_artifact({**ref, 'uri': str(root / ref['uri'])}, json_document=False)


def _verified(path, schema, verify):
    ref = _ref(path, schema)
    doc = verify(ref['uri'])
    _require(doc == read_artifact(ref), 'source native package changed during joint qualification')
    return ref, doc


def _basis(sort_path, resident_path):
    sorted_ref, eager = _verified(sort_path, sorted_library.SCHEMA, sorted_library.check)
    resident_ref, lazy = _verified(resident_path, resident.SCHEMA, resident.check)
    sort_plan = read_artifact(eager['region'])
    state_plan = read_artifact(lazy['plan'])
    _require(sort_plan['scorer'] == lazy['score_plan'] and read_artifact(sort_plan['transition']) == state_plan
             and eager['abi']['table_slots'] == lazy['abi']['table_slots'],
             'ordered and state regions require exactly the same score plan, state plan and version table')
    abi = dict(sort=eager['abi'], transition={**lazy['abi'], 'symbol': SYMBOL, 'state': lazy['abi']['output'],
                    'move': eager['abi']['record'], 'dispatch': 'EXPLICIT_CALLER_SELECTED_RECORD_ONLY',
                    'record_score_guard': 'AGREES_WITH_CALLER_ATTESTED_ORIGINAL_PRODUCER_TOKEN'})
    roots = [Path(ref['uri']).parent for ref in (sorted_ref, resident_ref)]
    source = [_local_source(base, record['members'][member]) for base, record, member in zip(
              roots, (eager, lazy), (sorted_library._SOURCE, resident._SOURCE), strict=True)]
    headers = {record['header']['uri']: _local_source(base, record['header'])
               for base, record in zip(roots, (eager, lazy), strict=True)}
    captures = read_artifact(eager['capture'])['samples']
    names = [field['name'] for field in eager['abi']['record']['ports']]
    sorted_samples = [(row, [dict(zip(names, result, strict=True)) for result in row['records']]) for row in captures]
    transport = read_artifact(lazy['transport']) if lazy['transport'] is not None else None
    return (sorted_ref, resident_ref), abi, source, headers, state_plan, sorted_samples, transport


def _section(source, marker):
    _require(source.count(marker) == 1, 'combined C++ lowering requires its exact distinct publication boundary')
    return source.split(marker, 1)


def _render(source, headers, abi):
    sorted_source, residency = (row.decode() for row in source)
    state_alias = 'using State = std::tuple<'
    _, state_body = _section(residency, state_alias)
    computation, _ = _section(state_body, '[[maybe_unused]] std::string read_text(')
    _, guarded_output = _section(state_body, '[[maybe_unused]] void write_text(')
    output, signature = _section(guarded_output, '\n}  // namespace\n\nextern "C" __attribute__((visibility("default")))\n')
    old, bridge = _section(signature, 'int scar_resident_transition_v1(')
    _require(not old, 'resident lowering function has a different ABI boundary')
    copy, score = _bridge(abi['transition'])
    fields = dict(SORTED_IMPLEMENTATION=sorted_source, RESIDENT_COMPUTATION=state_alias + computation,
                  RESIDENT_BRIDGE='[[maybe_unused]] void write_text(' + output + '\nstatic int evaluate_resident(' + bridge,
                  INPUT_SETUP=copy, SCORE_SLOT=score)
    generated = _TEMPLATE.read_text()
    for name, value in fields.items():
        generated = generated.replace('@SCAR_' + name + '@', value)
    return {**headers, _SOURCE: generated.encode(), _HEADER: _HEADER_TEMPLATE.read_bytes(),
            _MAP: f'{{ global: {sorted_library.SYMBOL}; {SYMBOL}; local: *; }};\n'.encode()}


def transition_types(abi):
    """Return ctypes mirrors of current state, ranked record and proof token."""
    _, state, token = resident.resident_types(abi['transition'])
    _, move = sorted_library.pod_types(abi['sort'])
    return state, move, token


def load_transition(path, types):
    library = ffi.CDLL(str(Path(path).resolve(strict=True)))
    function = getattr(library, SYMBOL)
    state, move, token = types
    function.argtypes = [ffi.c_uint32, ffi.POINTER(state), ffi.POINTER(move), ffi.POINTER(token), ffi.POINTER(state)]
    function.restype = ffi.c_int
    return function


def _verify(path, abi, plan, sorted_samples, transport):
    sorted_outputs = sorted_library._verify(path, abi['sort'], sorted_samples)
    scope = abi['transition']
    types = transition_types(abi)
    state, move, token = types
    flat = dict(token._fields_)['source_inputs']
    action = load_transition(path, types)
    slots = scope['table_slots']
    state_names, move_names = _by_name(scope['state']), _by_name(scope['move'])
    outcome = state()

    def parts(data, delta):
        current = resident.pack(scope['state'], state, {name: data[name] for name in state_names})
        ticket = resident.pack(scope['move'], move, {name: delta if name == 'score' else data[name] for name in move_names})
        return current, ticket

    def expected(data):
        return bytes(resident.pack(scope['state'], state, data))

    native = []
    for sample in plan['samples']:
        # Deliberately ignore the record score in the absence of a token.
        current, ticket = parts(sample['inputs'], 0)
        status = action(slots.index(sample['table_version']), ffi.byref(current), ffi.byref(ticket), None, ffi.byref(outcome))
        _require(status == 0 and bytes(outcome) == expected(sample['outputs']), 'fused lazy native state differs')
        native.append(bytes(outcome))

    transported, joined = [], set()
    sort_types = sorted_library.pod_types(abi['sort'])
    eager = sorted_library.load(path, sort_types)
    witnesses = resident._transport_samples(transport) if transport is not None else ()
    for version, data, value, delta in witnesses:
        slot = slots.index(version)
        current, ticket = parts(data, delta)
        producer = token(scope['receipt_tag'], slot, delta, resident.pack(scope['input'], flat, data))
        status = action(slot, ffi.byref(current), ffi.byref(ticket), ffi.byref(producer), ffi.byref(outcome))
        _require(status == 1 and bytes(outcome) == expected(value), 'fused lazy score transport differs')
        transported.append(bytes(outcome))
        if version not in joined:
            for sample, records in sorted_samples:
                if sample['table_version'] != version or any(data[name] != observed for name, observed in sample['inputs'].items()):
                    continue
                key = {name: delta if name == 'score' else data[name] for name in move_names}
                if key not in records:
                    continue
                argument = resident.pack(abi['sort']['input'], sort_types[0], sample['inputs'])
                array = (sort_types[1] * len(records))()
                count = ffi.c_size_t()
                result = eager(slot, ffi.byref(argument), sample['threshold'], sample['nonzero_capture'],
                               array, len(records), ffi.byref(count))
                _require(result == 0 and count.value == len(records), 'fused sorted stage did not publish the complete array')
                ptr = ffi.cast(ffi.byref(array, records.index(key) * ffi.sizeof(move)), ffi.POINTER(move))
                for offered, wanted in ((None, 0), (ffi.byref(producer), 1)):
                    result = action(slot, ffi.byref(current), ptr, offered, ffi.byref(outcome))
                    _require(result == wanted and bytes(outcome) == expected(value), 'fused ranked-record to lazy-state handoff differs')
                joined.add(version)
                break

    if transport is not None:
        _require(transported and joined == set(slots), 'score study needs a joined ranked-record/state witness per table')
        score_slot = move_names['score']['slot']
        ticket.scalars[score_slot] ^= 1
        result = action(slot, ffi.byref(current), ffi.byref(ticket), ffi.byref(producer), ffi.byref(outcome))
        _require(result == 0 and bytes(outcome) == expected(value), 'disagreeing record score did not recompute')
    ffi.memset(ffi.addressof(outcome), 0xA5, ffi.sizeof(outcome))
    untouched = bytes(outcome)
    status = action(len(slots), ffi.byref(current), ffi.byref(ticket), None, ffi.byref(outcome))
    _require(status == 2 and bytes(outcome) == untouched, 'invalid version changed the immutable-state result')
    long_text = next((row for row in move_names.values() if row['type'] == 'str'), None)
    if long_text is not None and long_text['capacity'] < 2**32 - 1:
        ticket.lengths[long_text['slot']] = long_text['capacity'] + 1
        status = action(0, ffi.byref(current), ffi.byref(ticket), None, ffi.byref(outcome))
        _require(status == 2 and bytes(outcome) == untouched, 'invalid move extent changed the state result')
    return (sorted_outputs, b''.join(native), b''.join(transported)), sorted(joined)


def _links(path):
    linked = _dependencies_for(path)
    listing = subprocess.check_output(['/usr/bin/nm', '-D', '--defined-only', '--format=posix', str(path)], env={}, text=True)
    _require(sorted(row.split()[0] for row in listing.splitlines()) == sorted([SYMBOL, sorted_library.SYMBOL]),
             'fused native library exports a different public ABI')
    return linked


_RECORDS = ('sorted.pod', 'native.pod', 'transported.pod')


def _validation(root, abi, eager, native, joined):
    records = {name: _local(root, name, 'application/octet-stream') for name in _RECORDS}
    size = ffi.sizeof(transition_types(abi)[0])
    _require(records['native.pod']['size_bytes'] == native * size and records['transported.pod']['size_bytes'] % size == 0,
             'combined state samples differ from their POD ABI')
    return dict(sorted_samples=len(eager), sorted_records=sum(len(rows) for _, rows in eager), native_states=native,
                transported_states=records['transported.pod']['size_bytes'] // size,
                ranked_record_to_lazy_state_versions=joined, relocated=True, records=records)


def export(sorted_path, resident_path, out):
    """Build a fused DSO after both exact package checkers pass."""
    refs, abi, source, headers, plan, samples, proof = _basis(sorted_path, resident_path)
    rendered = _render(source, headers, abi)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    version = subprocess.check_output([*compiler, '--version'], env=env, text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-sorted-transition-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'package'
        for name, data in rendered.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (stage / _LIBRARY).parent.mkdir()
        subprocess.run([*compiler, *_NATIVE, '-Iinclude', _SOURCE, '-o', _LIBRARY], cwd=stage, env=env, check=True)
        moved = Path(temporary) / 'relocated sorted and lazy region.so'
        copy2(stage / _LIBRARY, moved)
        links = _links(moved)
        records, joined = _verify(moved, abi, plan, samples, proof)
        for name, data in zip(_RECORDS, records, strict=True):
            (stage / name).write_bytes(data)
        document = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, sorted=refs[0], resident=refs[1], abi=abi,
                        compiler=dict(command=compiler, version=version, flags=list(_NATIVE)), linkage=links,
                        library=_local(stage, _LIBRARY, 'scar.native-library-file'),
                        header=_local(stage, _HEADER, 'scar.c-header'),
                        headers=dict(sort=_local(stage, sorted_library._HEADER, 'scar.c-header'),
                                     resident=_local(stage, resident._HEADER, 'scar.c-header')),
                        members={name: _local(stage, name, kind) for name, kind in _MEMBERS.items()},
                        validation=_validation(stage, abi, samples, len(plan['samples']), joined),
                        live_producer_guard_implemented=False, deployment_authorized=False, whole_program_cpp=False)
        (stage / 'artifact.json').write_text(json.dumps(document, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def check(path):
    """Rebuild the fused source from both checked packages and replay the two native boundaries."""
    ref = _ref(path, SCHEMA)
    result = read_artifact(ref)
    root = Path(ref['uri']).parent
    refs, abi, source, headers, plan, samples, proof = _basis(result['sorted']['uri'], result['resident']['uri'])
    rendered = _render(source, headers, abi)
    library = root / _LIBRARY
    _require(result['schema_version'] == 1 and result['cxx_standard'] == 20
             and (result['sorted'], result['resident']) == refs and result['abi'] == abi
             and all(result[key] is False for key in ('live_producer_guard_implemented', 'deployment_authorized', 'whole_program_cpp'))
             and result['members'] == {name: _local(root, name, kind) for name, kind in _MEMBERS.items()}
             and result['header'] == result['members'][_HEADER]
             and result['headers'] == dict(sort=result['members'][sorted_library._HEADER], resident=result['members'][resident._HEADER])
             and all((root / name).read_bytes() == data for name, data in rendered.items())
             and result['library'] == _local(root, _LIBRARY, 'scar.native-library-file')
             and result['compiler']['flags'] == list(_NATIVE) and result['linkage'] == _links(library),
             'fused C++ library differs from its exact parent packages and output ABI')
    data, joined = _verify(library, abi, plan, samples, proof)
    _require(result['validation'] == _validation(root, abi, samples, len(plan['samples']), joined)
             and all((root / name).read_bytes() == content for name, content in zip(_RECORDS, data, strict=True)),
             'fused library outputs differ from fresh native replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sorted')
    parser.add_argument('--resident')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.sorted, args.resident, args.out)):
        result = check(args.check)
    elif args.sorted and args.resident and args.out and not args.check:
        result = export(args.sorted, args.resident, args.out)
    else:
        parser.error('provide --sorted, --resident and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
