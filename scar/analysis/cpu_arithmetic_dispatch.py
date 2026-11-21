"""Bind immutable repeat shapes to the checked grouped native arithmetic tape."""

from dataclasses import dataclass, replace
from hashlib import sha256
import re
import struct
from subprocess import CalledProcessError, TimeoutExpired

from scar.analysis.cpu_fp_guard_groups import _ARITHMETIC, _HOOK, _ROUND, _table
from scar.analysis.replay_transaction import detached_replay
from scar.codegen import native_cpu_fp_guard_groups as grouped, native_numpy_complex as stock
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-arithmetic-dispatch-proof'
_ROW = re.compile(r'    \{Op::([a-z]+), (\d+), (\d+), \{(-?\d+, -?\d+, -?\d+)\}, '
                  r'\{(-?\d+, -?\d+, -?\d+)\}, \{(-?\d+, -?\d+, -?\d+)\}, (\d+)\},')
_TAPE = 'src/tape_0.inc'
_PROFILE = 'src/profile_0.cpp'
_LOAD = 'rax,QWORD PTR [rsp+0x10]'
_STORE = 'QWORD PTR [rsp+0x10],rax'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU arithmetic dispatch: ' + detail)


@dataclass(frozen=True, slots=True)
class _Step:
    operator: str
    count: int
    repeat: int
    offsets: tuple
    strides: tuple
    advances: tuple
    argument: int

    def shape(self):
        return self.operator, self.count, self.repeat, self.strides, self.advances


def _steps(reference):
    text = read_artifact(reference, json_document=False).decode('ascii')
    rows = _table(text, 'constexpr Step steps[]{\n', _ROW)
    triples = lambda text: tuple(map(int, text.split(', ')))
    steps = [_Step(op, int(count), int(repeat), triples(offsets), triples(strides), triples(advances), int(arg))
             for op, count, repeat, offsets, strides, advances, arg in rows]
    rounds = [tuple(map(int, row)) for row in _table(text, 'constexpr std::size_t rounds[][2]{\n', _ROUND)]
    _need(all(step.count > 0 and step.repeat > 0 for step in steps), 'unknown emitted step geometry')
    return steps, rounds


def _schedule(schedule, steps):
    starts, origins = [0], []
    offsets = schedule['storage']['offset_bytes']
    for index, operation in enumerate(schedule['operations']):
        kind = operation['kind']
        start = starts[-1]
        if start == len(steps):
            break
        extent = len(operation['groups']) if kind == 'COPY' else 1
        _need(extent > 0 and start + extent <= len(steps), 'compiled rows split an original operation')
        starts.append(start + extent)
        origins.extend([index] * extent)
        if kind in ('ADD', 'MULTIPLY'):
            row = operation['group']
            repeat = row.get('repeat', dict(count=1, offsets=[0, 0, 0]))
            expected = _Step(kind.lower(), row['count'], repeat['count'], tuple(
                offsets[operand['root']] + operand['offset'] for operand in row['operands']), tuple(
                operand['stride'] for operand in row['operands']), tuple(repeat['offsets']), 0)
            _need(expected == steps[start], 'compiled arithmetic row differs from original storage and schedule')
        else:
            _need(all(step.operator not in _ARITHMETIC for step in steps[start:start + extent]),
                  'arithmetic row has a non-arithmetic source')
    _need(len(origins) == len(steps), 'emitted table does not have complete original operation coordinates')
    return starts, origins


def _windows(profile, proof, steps, rounds, periodic, starts):
    text = read_artifact(profile, json_document=False).decode('ascii')
    raw_hooks = [(int(index), kind) for index, kind, _ in _table(text, 'constexpr Hook hooks[]{\n', _HOOK)]
    hooks = dict(raw_hooks)
    _need(list(hooks) == sorted(hooks) and len(hooks) == len(raw_hooks)
          and all(0 <= index < len(steps) and steps[index].operator ==
                  ('literal' if kind == 'coefficient' else 'normalize') for index, kind in raw_hooks),
          'coefficient or normalizer hook moved from the compiled step')
    _need(rounds == [tuple(starts[index] for index in periodic[key]['operations'])
                      for key in ('first', 'steady')] and len(proof['phases']) == 2,
          'emitted first/steady rounds differ from original periodic layout')
    result, coordinates = [], {}
    for label, (start, end), proved in zip(('FIRST', 'STEADY'), rounds, proof['phases'], strict=True):
        if label == 'STEADY':
            ingress = periodic['carry_into_steady']['operation']
            _need(starts[ingress] == start and starts[ingress + 1] == start + 1 and steps[start].operator == 'copy',
                  'steady dispatch must start after its original carry handoff')
            start = starts[ingress + 1]
        normalize = starts[periodic[label.lower()]['normalize_operation']]
        checkpoint = starts[periodic[label.lower()]['checkpoint_operation']]
        _need((label, start, end, normalize, checkpoint) == tuple(proved[key] for key in (
              'phase', 'begin_compiled_step', 'end_compiled_step', 'normalization_compiled_step',
              'checkpoint_compiled_step')) and hooks.get(normalize) == 'normalization'
              and start < normalize < checkpoint < end, 'active phase differs from sticky guard proof')
        cursor = start
        for boundary, kind in raw_hooks:
            if not start <= boundary <= normalize:
                continue
            for index in range(cursor, boundary):
                if steps[index].operator in _ARITHMETIC:
                    coordinates[index] = label, (cursor, boundary)
            cursor = boundary + 1
            if kind == 'normalization':
                _need(boundary == normalize, 'dispatch crosses the original normalizer')
                break
        phase = [steps[index] for index, (owner, _) in coordinates.items() if owner == label]
        repeats = sum(step.repeat > 1 for step in phase)
        calls = sum(step.repeat for step in phase)
        _need((len(phase), calls) == (proved['arithmetic_steps'], proved['stock_provider_calls']),
              'whole phase does not cover the original provider call count')
        result.append(dict(phase=label, begin_compiled_step=start, end_compiled_step=end,
            normalization_compiled_step=normalize, checkpoint_compiled_step=checkpoint,
            arithmetic_steps=len(phase), repeated_steps=repeats, singleton_steps=len(phase) - repeats,
            stock_provider_calls=calls))
    _need(len(coordinates) == sum(step.operator in _ARITHMETIC for step in steps),
          'unqualified arithmetic outside the production dispatch windows')
    return result, coordinates


def _guards(baseline, steps, original, coordinates):
    owners = {}
    for ordinal, group in enumerate(baseline['groups']):
        begin, end = group['compiled_steps']
        _need(begin + 1 < end and begin in coordinates and coordinates[begin][0] == group['phase']
              and all(coordinates.get(index) == coordinates[begin] and index not in owners
                      for index in range(begin, end)), 'sticky guard group crosses a production dispatch window')
        for index in range(begin, end):
            owners[index] = ordinal, int(index != begin) | (int(index != end - 1) << 1)
    for index, (step, prior) in enumerate(zip(steps, original, strict=True)):
        expected = owners.get(index, (None, 0))[1] if index in coordinates else prior.argument
        _need((prior.argument == 0 or index not in coordinates) and
              step == replace(prior, argument=expected),
              'grouped tape changed an original operation, pointer, or FP guard')
    return owners


def _machine(baseline, providers):
    """Read the narrow linked x86 indirect-loop idiom, when present."""
    executable = baseline['executable']
    fallback = dict(status='SOURCE_DISPATCH_ONLY', executable=executable)
    try:
        image = stock._Image(executable)
        symbols = stock._symbols(image)
        matches = [(name, start, extent) for name, values in symbols.items()
                   if name.startswith('_ZN12_GLOBAL__N_18run_bodyE')
                   for start, extent, visibility in values if visibility == 'LOCAL']
        if len(matches) != 1:
            return fallback
        name, start, extent = matches[0]
        rows = stock._instructions(image, start, extent)
        indirect = [(index, address) for index, (address, raw, op, args) in enumerate(rows)
                    if (raw, op, args) == (b'\xff\xd0', 'call', 'rax')]
        if len(indirect) != 1:
            return fallback
        index, call = indirect[0]
        loads = [i for i, (_, _, op, args) in enumerate(rows) if i < index and (op, args) == ('mov', _LOAD)]
        stores = [i for i, (_, _, op, args) in enumerate(rows) if (op, args) == ('mov', _STORE)]
        if len(loads) != 1 or len(stores) != 2:
            return fallback
        load = loads[0]
        if index - load > 16 or any(args.split(',')[0] in ('rax', 'eax', 'ax', 'al', 'ah')
                                    or op.startswith(('j', 'call', 'ret')) for _, _, op, args in rows[load+1:index]):
            return fallback
        by_address = {provider['address']: provider['symbol'] for provider in providers.values()}
        writes = {}
        for store in stores:
            address, raw, op, args = rows[store - 2]
            if (op != 'lea' or not args.startswith('rax,[rip') or raw[:3] != b'\x48\x8d\x05'
                    or len(raw) != 7 or rows[store-1][2:] != ('mov', 'rdx,QWORD PTR [rbx+0x58]')):
                return fallback
            target = address + len(raw) + struct.unpack('<i', raw[3:])[0]
            if target not in by_address:
                return fallback
            writes[by_address[target]] = rows[store][0]
        if len(writes) != 2:
            return fallback
        branches = [address for address, raw, op, args in rows[index+1:index+9] if op == 'jne'
                    and stock._displacement(address, raw, args) == rows[load][0]]
        if len(branches) != 1:
            return fallback
        return dict(status='BASELINE_X86_INDIRECT_PROVIDER_IN_REPEAT_LOOP', executable=executable,
                    runner=dict(symbol=name, address=start, extent=extent, sha256=sha256(
                        image.read(start, extent, 5)).hexdigest()),
                    provider_pointer_writes=writes, pointer_reload=rows[load][0],
                    indirect_call=call, repeat_backedge=branches[0])
    except (ValueError, CalledProcessError, TimeoutExpired):
        return fallback


def derive_cpu_arithmetic_dispatch(grouped_native_ref):
    """Prove repeat dispatch candidates without changing arithmetic or running the entry."""
    identity = tuple(grouped_native_ref[key] for key in ('schema', 'uri', 'sha256', 'size_bytes'))
    return detached_replay('scar.cpu-arithmetic-dispatch.v1', identity, lambda: _derive(grouped_native_ref))


def _derive(grouped_native_ref):
    baseline = grouped.verify_native_cpu_fp_guard_groups(grouped_native_ref)
    runtime_ref = baseline['runtime']
    runtime = read_artifact(runtime_ref)
    proof = read_artifact(baseline['proof'])
    arena = read_artifact(proof['arena_proof'])
    storage = arena['evidence']['storage']
    periodic = read_artifact(storage['periodic_memory'])
    schedule = read_artifact(storage['schedule'])
    _need(proof['runtime'] == runtime_ref and arena['runtime'] == runtime_ref
          and baseline['scope'] == runtime['scope'] == proof['scope']
          and proof['proof']['flags_are_sticky_within_group'] is True
          and proof['proof']['fp_operation_order_unchanged'] is True,
          'baseline does not replay the same sticky guard and original numerical contract')
    steps, rounds = _steps(baseline['files'][_TAPE])
    original, original_rounds = _steps(runtime['files'][_TAPE])
    _need(rounds == original_rounds, 'grouped tape moved an original round boundary')
    starts, origins = _schedule(schedule, original)
    phases, coordinates = _windows(baseline['files'][_PROFILE], proof, steps, rounds, periodic, starts)
    groups = _guards(baseline, steps, original, coordinates)
    providers = {role: baseline['linkage']['numerical']['stock']['functions'][role]
                 for role in sorted(_ARITHMETIC)}
    shapes, repeated, indices = [], [], {}
    for index, (phase, window) in coordinates.items():
        step = steps[index]
        if step.repeat == 1:
            continue
        key = step.shape()
        if key not in indices:
            indices[key] = len(shapes)
            shapes.append(dict(id=len(shapes), operator=step.operator.upper(), count=step.count, repeat=step.repeat,
                               strides=list(step.strides), advances=list(step.advances),
                               provider=providers[step.operator]['symbol']))
        repeated.append(dict(compiled_step=index, original_operation=origins[index], phase=phase, shape=indices[key],
            offsets=list(step.offsets), guard_bits=step.argument, guard_group=groups.get(index, (None, 0))[0],
            dispatch_window=list(window)))
    summary = {key: sum(phase[key] for phase in phases) for key in
                ('arithmetic_steps', 'repeated_steps', 'singleton_steps', 'stock_provider_calls')}
    summary['shapes'] = len(shapes)
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_STATIC_ARITHMETIC_DISPATCH',
        baseline=grouped_native_ref, runtime=runtime_ref, scope=baseline['scope'], phases=phases, shapes=shapes,
        steps=repeated, summary=summary,
        evidence=dict(guard_proof=baseline['proof'], guard_plan=baseline['plan'],
            schedule=storage['schedule'], periodic_memory=storage['periodic_memory'],
            baseline_tape=baseline['files'][_TAPE], baseline_profile=baseline['files'][_PROFILE],
            original_tape=runtime['files'][_TAPE], provider_functions=providers),
        dispatch=dict(scope='PRODUCTION_GROUPED_RUN_BODY_ONLY', machine=_machine(baseline, providers)),
        proof=dict(immutable_shapes=True, arithmetic_provider_calls_and_order_preserved=True,
            offsets_and_aliases_preserved=True, guard_bits_preserved=True, no_crossing_dispatch_windows=True,
            private_probe_unchanged_required=True),
        target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_arithmetic_dispatch(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved == derive_cpu_arithmetic_dispatch(saved['baseline']),
          'saved repeat dispatch proof differs on cold replay')
    return saved
