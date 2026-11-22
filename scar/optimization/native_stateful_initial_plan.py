"""Plan fixed initial materializations around a source-replayed stateful time loop."""

import argparse
from hashlib import sha256
import json
from math import isfinite, prod
from pathlib import Path

from scar.analysis import source_stateful_initial as source
from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-stateful-initial-plan'
OBSERVATION = 'scar.stateful-entry-observation'
_MAX_STEPS = 100_000
_MAX_RATE = 1.0
_FRAME = 'frame_000000.png'
_ORDER = ('PRE_STEP_DUMP', 'PRE_STEP_REPORT', 'EULER', 'GHOST_U', 'GHOST_V',
          'ADVANCE_TIME_USING_CURRENT_DT', 'CLIP_NEXT_DT', 'INCREMENT_STEP')


def _require(condition, detail):
    if not condition:
        raise ValueError('stateful initial placement: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _finite(value):
    return type(value) in (int, float) and isfinite(value)


def evaluate_time(end_time, dump_freq, *, initial_dt, report_frequency=50, max_steps=_MAX_STEPS):
    """Project the exact source clock and preflight the one-frame success path."""
    _require(_finite(end_time) and _finite(initial_dt) and end_time > 0 and initial_dt > 0
             and all(type(value) is int and value > 0 for value in (dump_freq, report_frequency, max_steps)),
             'expected positive finite end time and dt with valid step controls')
    time = 0.0
    dt = float(initial_dt)
    step = 0
    reports = []
    while time < end_time:
        _require(step < dump_freq and step < max_steps, 'time path exceeds the one-frame bounded loop')
        if step % report_frequency == 0:
            reports.append(dict(step=step, time=time))
        advanced = time + dt
        _require(isfinite(advanced) and advanced > time, 'binary64 source time must make positive progress')
        time = advanced
        remaining = end_time - time
        if remaining < dt:
            dt = remaining
        step += 1
    return dict(updates=step, dumps=1, final_time=time, final_dt=dt, reports=reports)


def _arguments(ports, argv):
    _require(type(argv) is list and all(type(arg) is str for arg in argv), 'expected original argv strings')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    converters = {'float': float, 'int': int, 'str': str}
    for name, port in ports.items():
        policy = (dict(action='store_true') if port['parser'] == 'bool' else dict(type=converters[port['parser']]))
        parser.add_argument(*port['flags'], dest=name, default=port['default'], **policy)
    parsed, unknown = parser.parse_known_args(argv)
    actual = vars(parsed)
    _require(not unknown and actual['demo'] is actual['movie'] is False
             and actual['outdir'] and all(_finite(actual[key]) and 0 < actual[key] <= _MAX_RATE
                                          for key in ('feed_rate', 'death_rate')),
             'expected ordinary positive-rate Euler CLI and a concrete output location')
    return actual


def _ordered_source(proof):
    constructor, time = proof['constructor'], proof['integration']
    shape = constructor['arrays']['u']['shape']
    _require(len(shape) == 2 and all(type(extent) is int and extent > 2 for extent in shape)
             and set(constructor['arrays']) == {'x', 'y', 'u', 'v'}
             and all(row['shape'] == shape and row['dtype'] == '<f8' and row['argument_independent'] is True
                     for row in constructor['arrays'].values())
             and time['time_order'] == list(_ORDER) and time['step_start'] == time['start_time'] == 0
             and type(time['report_frequency']) is int and time['report_frequency'] > 0
             and constructor['scalars']['second_order'] is False and constructor['scalars']['dump_count'] == 0,
             'expects the homogeneous binary64 constructor and its distance-one Euler transition')
    return shape


def _capture(reference, proof, shape):
    record = read_artifact(reference)
    _require(record['schema_version'] == 1 and record['instrumented'] is True
             and record['performance_measurement'] is False and record['payload_order'] == ['u', 'v']
             and record['entry'] == proof['entry'] and record['source'] == proof['source'],
             'observation must come from the exact original source, not staged preparation')
    read_artifact(record['launcher'], json_document=False)
    phases = record['phases']
    args = _arguments(proof['ports'], record['argv'])
    initialize, finalize = phases['initial'], phases['final']
    _require(initialize['code'] == proof['constructor']['code']
             and finalize['code'] == proof['integration']['code'], 'constructor or loop CodeID differs')
    scalars = proof['constructor']['scalars']
    clock = evaluate_time(args['end_time'], args['dump_freq'], initial_dt=scalars['dt'],
                          report_frequency=proof['integration']['report_frequency'])
    runtime_fields = dict(F=args['feed_rate'], kappa=args['death_rate'], outdir=args['outdir'], movie=False)
    for phase, fields in ((initialize, dict(scalars, **runtime_fields)),
                          (finalize, dict(scalars, **runtime_fields, dt=clock['final_dt'], dump_count=1))):
        _require(all(phase['scalars'][name] == value for name, value in fields.items()),
                 'observed receiver does not preserve the exact source scalar state')
    field_bytes = prod(shape) * 8
    array_names = set(proof['constructor']['arrays'])
    for phase in (initialize, finalize):
        arrays = phase['arrays']
        _require({'u', 'v'} <= set(arrays) <= array_names
                 and all(row['shape'] == shape and row['dtype'] == 'float64' and row['byte_order'] == 'little'
                         and row['size_bytes'] == field_bytes for row in arrays.values()),
                 'observed arrays differ from the source binary64 storage geometry')

    payload = read_artifact(record['initial_payload'], json_document=False)
    _require(len(payload) == 2 * field_bytes and all(
        sha256(payload[index * field_bytes:(index + 1) * field_bytes]).hexdigest() == initialize['arrays'][field]['sha256']
        for index, field in enumerate(('u', 'v'))), 'initial payload is not exact ordered u and v bytes')
    files = record['effects']['files']
    expected_file = (Path(record['cwd']) / args['outdir'] / _FRAME).resolve(strict=True)
    _require(len(files) == 1 and files[0]['name'] == _FRAME and files[0]['artifact']['uri'] == str(expected_file),
             'observed initial frame must belong to the original runtime output path')
    frame = read_artifact(files[0]['artifact'], json_document=False)
    stdout = ''.join(f"step={row['step']}; time={row['time']:e}\n" for row in clock['reports']).encode('ascii')
    _require(record['updates'] == clock['updates'] and record['dumps'] == clock['dumps']
             and frame.startswith(b'\x89PNG\r\n\x1a\n')
             and read_artifact(record['stdout'], json_document=False) == stdout
             and read_artifact(record['stderr'], json_document=False) == b'',
             'original update count, staged frame or ordered progress differs from the source clock')
    return record, args, clock, payload, frame


def _graph(initial, time, shape):
    field_bytes = prod(shape) * 8
    return dict(stages=dict(admit=dict(kind='CPU_CONTROL'),
        initialize=dict(kind='READ_ONLY_IMAGE_TO_MUTABLE_CPU_FIELDS', logical_bytes=2 * field_bytes),
        publish_initial_frame=dict(kind='FILESYSTEM_WRITE', logical_bytes=initial['frame']['size_bytes']),
        report=dict(kind='STDOUT'), euler=dict(kind='CPU_STENCIL'), ghosts=dict(kind='CPU_BOUNDARY_WRITE'),
        advance=dict(kind='CPU_SCALAR'), exit=dict(kind='CONTROL')),
        pre_step_order=['INITIAL_FRAME_IF_STEP_ZERO', 'REPORT_IF_STEP_MULTIPLE', 'EULER', 'GHOSTS', 'CLOCK'],
        loop=dict(entry='report', end='advance', order=time['time_order'], condition='TIME_LT_RUNTIME_END_TIME',
            iteration_distance=1, unrolled=False, carried=['u', 'v', 't', 'dt', 'step'],
            invariant=['feed_rate', 'death_rate', 'fa', 'fs', 'end_time'],
            stencil_state='READ_OLD_U_V_AND_COMMIT_BOTH_VERSIONS_BEFORE_GHOSTS'),
        layout=dict(field_bytes=field_bytes, field_payload_bytes=2 * field_bytes,
            physical_peak_bytes=None, physical_dram_residency=None),
        controls=dict(no_movie=True, no_second_frame=True, preserve_pre_step_report=True))


def _provider(path):
    if path is None:
        return None
    from scar.analysis import numpy_broadcast_square

    receipt = numpy_broadcast_square.check(path)
    reference = _ref(path, receipt['schema'])
    _require(receipt == read_artifact(reference), 'numeric provider differs from cold replay')
    return reference


@replay_transaction()
def propose(source_proof, captures, *, provider=None):
    """Bind initial images to independent original runs and replay one symbolic loop."""
    reference = _ref(source_proof, source.SCHEMA)
    proof = source.check(reference['uri'])
    _require(proof == read_artifact(reference), 'source proof differs from its checked bytes')
    shape = _ordered_source(proof)
    capture_refs = [_ref(path, OBSERVATION) for path in captures]
    _require(len({item['sha256'] for item in capture_refs}) == len(capture_refs) >= 2,
             'needs distinct captures from at least two original invocations')
    observed = [_capture(item, proof, shape) for item in capture_refs]
    primary, _, _, payload, frame = observed[0]
    _require(all(item[3] == payload and item[4] == frame for item in observed)
             and all(len({item[1][port] for item in observed}) >= 2
                     for port in ('feed_rate', 'death_rate', 'end_time', 'outdir'))
             and all(row['environment'] == primary['environment'] and row['interpreter'] == primary['interpreter']
                     and row['launcher'] == primary['launcher']
                     for row, *_ in observed),
             'initial images must match across distinct rates, time and output paths under one provider context')
    runtime_context = {name: primary[name] for name in ('environment', 'interpreter')}
    scalars = proof['constructor']['scalars']
    first = primary['phases']['initial']
    initial = dict(payload=primary['initial_payload'], frame=primary['effects']['files'][0]['artifact'],
                   arrays={field: first['arrays'][field] for field in ('u', 'v')}, scalars=scalars,
                   encoding='U_THEN_V_CONTIGUOUS_LITTLE_ENDIAN_BINARY64')
    time = proof['integration']
    scope = dict(shape=shape, dtype='<f8', start_time=time['start_time'], initial_dt=scalars['dt'],
        report_frequency=time['report_frequency'], max_steps=_MAX_STEPS, frame=_FRAME,
        rates=dict(minimum_exclusive=0.0, maximum=_MAX_RATE, finite=True),
        output_mode='EXACTLY_ONE_PRE_STEP_FRAME', full_original_effect_equivalence=False)
    if 'ghost_assignments' in time:
        scope['ghost_assignments'] = time['ghost_assignments']
    qualification = [dict(capture=ref, argv=record['argv'], arguments=args, clock=clock,
                           final_arrays=record['phases']['final']['arrays'])
                     for ref, (record, args, clock, _, _) in zip(capture_refs, observed, strict=True)]
    return dict(schema=SCHEMA, schema_version=1, source_proof=reference, entry=proof['entry'], source=proof['source'],
        captures=capture_refs, ports=proof['ports'], scope=scope, initial=initial, time=time,
        qualification=qualification, original_context=runtime_context, numeric_provider=_provider(provider),
        graph=_graph(initial, time, shape), effects=proof['effects'],
        cost=dict(status='UNKNOWN', prior_wall_times_used=False, cpu_compute_ns=None, memory_move_ns=None,
                  directory_and_frame_write_ns=None, startup_ns=None, whole_entry_ns=None,
                  basis='FRESH_PAIRED_DIRECT_FULL_ENTRY_VALIDATION_REQUIRED'),
        decision=dict(selected='RETAIN_ORIGINAL_ENTRY', native_experiment_authorized=True,
                      deployment_authorized=False,
                      success_path_only=True, strict_numeric_backend_validation_required=True))


def check(path):
    """Cold-replay all pinned source, payload, effect and time evidence."""
    original = read_artifact(_ref(path, SCHEMA))
    provider = original['numeric_provider']
    result = propose(original['source_proof']['uri'], [ref['uri'] for ref in original['captures']],
                     provider=provider['uri'] if provider else None)
    _require(json.dumps(original, sort_keys=True, allow_nan=False)
             == json.dumps(result, sort_keys=True, allow_nan=False), 'stored plan differs from cold replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-proof', 'out', 'check', 'provider'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--capture', action='append', type=Path, default=[])
    args = parser.parse_args(argv)
    if args.check and not any((args.source_proof, args.capture, args.out, args.provider)):
        result = check(args.check)
    elif not args.check and all((args.source_proof, args.capture, args.out)):
        result = propose(args.source_proof, args.capture, provider=args.provider)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf8') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        parser.error('provide --source-proof, repeated --capture, --out and optional --provider, or --check PLAN')
    print(json.dumps({name: result[name] for name in ('schema', 'scope', 'decision')}, sort_keys=True))


if __name__ == '__main__':
    main()
