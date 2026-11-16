"""Source-faithful paired canonical costs for a certified copy window."""
import argparse
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scar.analysis.target_graph import TargetHardwareGraph
from scar.codegen import stateful
from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
from scar.compiler.calibrate_stateful_boundary import _json, _now, _receipt
from scar.compiler.prepare_window_rewrite import replay_window_rewrite, SCHEMA as REWRITE_SCHEMA
from scar.compiler.stateful_model import stateful_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.target_hardware import HardwareThread
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.copy-window-calibration'
_ROOT = Path(__file__).resolve().parents[1]
_CLOCK = dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns')
CONDITIONS = (
    'EXACT_CERTIFIED_SAME_CALLER_SINGLE_CPU_RAW_COPY_WINDOW',
    'ORIGINAL_WINDOW_AND_FUSED_WINDOW_EACH_HAVE_ONE_WHOLE_WINDOW_CLOCK_CUT',
    'ALTERNATING_AB_BA_CANONICAL_PAIRED_SAMPLES_NOT_TARGET_VARIANT_SEARCH',
    'ORIGINAL_ARRAY_ORDER_AND_SLOT_CAPACITIES_WITH_INDEPENDENT_CANONICAL_INPUTS',
    'WARM_REUSED_BUFFERS_DESTINATIONS_RESET_OUTSIDE_EACH_CLOCK_CUT',
    'FULL_POST_CLOCK_BITWISE_OUTPUT_CHECK_AND_CHECKSUM_CAN_WARM_CACHES',
    'EXACT_COMPILER_OPTIONS_TARGET_AND_SELECTED_CALLER_CPU_CONTEXT',
    'EMPIRICAL_SAMPLE_RANGE_IS_NOT_A_GUARANTEED_PHYSICAL_BOUND',
)
MISSING = (
    'WHOLE_FUNCTION_COMPILATION_REGISTER_VECTOR_AND_MEMORY_CONTEXT_DELTA',
    'TARGET_FIRST_TOUCH_NUMA_CACHE_AND_INTERFERENCE_REGIME_NOT_VALIDATED',
    'INSTRUMENTATION_PERTURBATION_NOT_BOUNDED',
    'COMPONENT_MODEL_IMPROVEMENT_IS_NOT_WHOLE_PROGRAM_PROFITABILITY',
)
_FILES = ('compiler/calibrate_copy_window.py', 'codegen/copy_window_probe.py',
    'codegen/copy_window_probe_v1.cpp.in', 'codegen/stateful_stage_probe.py',
    'codegen/stateful_stage_capture.py', 'codegen/stateful.py',
    'codegen/stateful_v1.cpp.in', 'codegen/loop_v1.cpp.in',
    'codegen/scar_abi_v1.h', 'codegen/stateful_abi_v1.h')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pins():
    return {name: sha256((_ROOT/name).read_bytes()).hexdigest() for name in _FILES}


def _counts(repetitions, warmup):
    _need(type(repetitions) is int and 3 <= repetitions <= 101 and repetitions % 2 == 1
        and type(warmup) is int and 0 <= warmup <= 100, 'bounded odd repetitions and nonnegative warmup required')


def _statistics(values):
    return dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
        stdev_ns=statistics.stdev(values))


def _spec(reference, max_payload_bytes):
    from scar.codegen.copy_window_probe import render_copy_window_probe
    report, graph, resources, costs, certificate, context = replay_window_rewrite(reference['uri'])
    _need(report == read_artifact(reference), 'copy calibration must retain the exact source rewrite receipt')
    recipe = context['recipe']
    region, plan, binding = stateful._admit(read_artifact(recipe['artifacts']['region']),
        read_artifact(recipe['artifacts']['stage_plan']))
    runtime, _, owners, _ = stateful._providers(region)
    _need(runtime == recipe['implementation']['backend_capability_record']['provider_runtime'],
        'original provider binary differs from source recipe')
    contract = dict(region_sha256=stateful_digest(region), plan_sha256=semantic_sha256(plan),
        provider_runtime=runtime)
    capture = capture_stateful_stage_sources(region, plan, binding, contract)
    _need(sha256(stateful._source(region, plan, binding, contract).encode()).hexdigest()
        == capture['full_source_sha256'], 'original stage capture differs from source renderer')
    stages = {row['id']: row for row in plan['stages']}
    bodies = {row['index']: row for row in capture['stages']}
    original_bodies = [dict(task=key, body=bodies[stages[key]['index']]['body'])
        for key in certificate['original_task_ids']]
    lines = [row['body'].split('\n') for row in original_bodies]
    _need(lines and all(len(row) == 5 and row[:2] == lines[0][:2] and row[-2:] == lines[0][-2:]
        for row in lines), 'certified copies must have the original shared two-loop skeleton')
    fused_body = '\n'.join([*lines[0][:2], *(row[2] for row in lines), *lines[0][-2:]])
    ids = set(certificate['original_task_ids'])
    old_activities = {row.id for row in context['original_schedule'].temporal.activities if row.task in ids}
    old_occupancies = {row.id for row in context['original_schedule'].temporal.occupancies
        if row.activity in old_activities}
    bindings = [row for row in context['original_resources'].executions if row.occupancy in old_occupancies]
    _need(bindings and all(row.exact_thread is not None and row.units == 1
        and row.exact_thread == bindings[0].exact_thread for row in bindings),
        'copy window needs its original exact same-caller resource domain')
    inventory = TargetHardwareGraph(Path(resources.target.uri).parent)
    records = {row.id: row for row in inventory.resources()}
    thread = records[bindings[0].exact_thread.resource]
    _need(type(thread) is HardwareThread and list(binding.cpu_ids) == [thread.cpu_id]
        and bindings[0].exact_thread.snapshot == resources.target, 'selected CPU differs from retained hardware record')
    compiler = deepcopy(recipe['implementation']['backend_capability_record']['compiler'])
    cpus = list(binding.cpu_ids)
    resource_context = dict(cpu_ids=cpus, target_snapshot=encode(resources.target),
        exact_thread_refs=[encode(bindings[0].exact_thread)], compiler=compiler, provider_runtime=runtime)
    harness_plan = {**plan, 'copy_window_array_order': [row.id for row in region.inputs if row.kind == 'array']}
    source, provenance = render_copy_window_probe(harness_plan, cpus, original_bodies, fused_body,
        max_payload_bytes=max_payload_bytes)
    source_context = dict(parent_schedule_sha256=certificate['parent_schedule_sha256'],
        rewritten_schedule_sha256=certificate['rewritten_schedule_sha256'],
        certificate_sha256=semantic_sha256(certificate), resource_context=resource_context,
        original_bodies=[dict(task=row['task'], body_sha256=sha256(row['body'].encode()).hexdigest()) for row in original_bodies],
        fused_body_sha256=sha256(fused_body.encode()).hexdigest(), original_capture_sha256=capture['full_source_sha256'],
        array_order=harness_plan['copy_window_array_order'], source_recipe=recipe['recipe_id'])
    return dict(source=source, provenance=provenance, compiler=compiler, cpus=cpus,
        source_context=source_context, graph=graph, resources=resources, costs=costs,
        certificate=certificate, context=context, rewrite=reference, owners=owners,
        max_payload_bytes=max_payload_bytes)


def _build(spec, directory):
    directory.mkdir()
    compiler = spec['compiler']
    identity = subprocess.run([*compiler['argv'], '--version'], text=True, capture_output=True, check=True).stdout.splitlines()[0]
    _need(identity == compiler['identity'] and '-fopenmp' not in compiler['options'], 'original serial compiler profile required')
    source = directory/'probe.cpp'; source.write_text(spec['source'])
    headers = {}
    for name in ('scar_abi_v1.h', 'stateful_abi_v1.h'):
        (directory/name).write_bytes((_ROOT/'codegen'/name).read_bytes())
        headers[name] = _receipt(directory/name)
    library = directory/'copy_window_probe.so'
    command = [*compiler['argv'], *compiler['options'], str(source), '-o', str(library), '-lm']
    result = subprocess.run(command, text=True, capture_output=True)
    (directory/'build.log').write_text(result.stdout+result.stderr)
    _need(result.returncode == 0, 'copy-window probe compilation failed: '+result.stderr[-2000:])
    handle = ctypes.CDLL(str(library)); run = handle.scar_copy_window_probe_v1
    run.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
    run.restype = ctypes.c_int32
    return handle, run, dict(source=_receipt(source), library=_receipt(library), headers=headers,
        compiler=compiler, compiler_command=command, implementation_files=_pins(),
        source_provenance=spec['provenance'], max_payload_bytes=spec['max_payload_bytes'])


def calibrate_copy_window(rewrite, out, *, repetitions=7, warmup=2, max_payload_bytes=256*2**20):
    from scar.runtime.compiled_deployment import thread_environment
    _counts(repetitions, warmup)
    _need(sys.gettrace() is None and sys.getprofile() is None, 'component sampling requires no Python trace/profile hook')
    directory = Path(out).resolve(); _need(not directory.exists(), 'fresh copy calibration output required')
    reference = artifact_reference(Path(rewrite).resolve(), REWRITE_SCHEMA)
    spec = _spec(reference, max_payload_bytes)
    before = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    _need(set(spec['cpus']) <= set(before), 'original component CPU is unavailable to this caller')
    libc = ctypes.CDLL(None); libc.fegetround.restype = ctypes.c_int
    rounding = libc.fegetround(); started = time.monotonic(); observed_from = _now()
    directory.mkdir(parents=True)
    library, run, probe = _build(spec, directory/'native')
    elapsed = (ctypes.c_uint64*(2*repetitions))()
    checksums = (ctypes.c_uint64*(2*repetitions))(); orders = (ctypes.c_int32*repetitions)()
    status = run(repetitions, warmup, elapsed, checksums, orders)
    _need(status == 0, 'canonical copy probe declined: '+str(status))
    after = sorted(os.sched_getaffinity(0))
    _need(before == after and libc.fegetround() == rounding and thread_environment() == environment,
        'copy probe did not preserve caller affinity, rounding or thread environment')
    samples = []
    for index in range(repetitions):
        _need(orders[index] == (index+warmup) % 2 and checksums[2*index] == checksums[2*index+1],
            'paired order or canonical output checksum differs')
        samples.append(dict(index=index, order=['ORIGINAL', 'FUSED'] if orders[index] == 0 else ['FUSED', 'ORIGINAL'],
            original_ns=int(elapsed[2*index]), fused_ns=int(elapsed[2*index+1]),
            original_checksum=int(checksums[2*index]), fused_checksum=int(checksums[2*index+1]), checksums_equal=True))
    document = dict(schema=SCHEMA, schema_version=1, rewrite=reference,
        **{key:spec['source_context'][key] for key in ('parent_schedule_sha256',
            'rewritten_schedule_sha256','certificate_sha256','resource_context')},
        source_context=spec['source_context'],
        samples=samples, statistics={name:_statistics([row[name+'_ns'] for row in samples]) for name in ('original', 'fused')},
        repetitions=repetitions, warmup=warmup, conditions=list(CONDITIONS), missing_terms=list(MISSING), clock=_CLOCK,
        probe=probe, observed_from=observed_from, observed_until=_now(), seconds=time.monotonic()-started,
        caller_affinity_before=before, caller_affinity_after=after, caller_rounding_before=rounding,
        caller_rounding_after=libc.fegetround(), native_thread_environment=environment,
        host_isolated=False, canonical_components_measured=True, target_program_executed=False,
        whole_target_call_measured=False, target_variants_ranked=False, clock_overhead_subtracted=False,
        physical_optimality_proven=False, deployment_authorized=False)
    path = directory/'observations.json'; path.write_bytes(_json(document))
    result = dict(schema='scar.copy-window-calibration-result', schema_version=1,
        observations=artifact_reference(path, SCHEMA), statistics=document['statistics'], deployment_authorized=False)
    (directory/'result.json').write_bytes(_json(result))
    read_artifact(reference)
    return result


def read_copy_window_calibration(path):
    """Replay source and raw statistics without executing a component or target."""
    reference = path if type(path) is dict else artifact_reference(Path(path).resolve(), SCHEMA)
    doc = read_artifact(reference)
    _need((doc.get('schema'), doc.get('schema_version')) == (SCHEMA, 1), 'versioned copy calibration required')
    _counts(doc['repetitions'], doc['warmup'])
    spec = _spec(doc['rewrite'], doc['probe']['max_payload_bytes'])
    _need(doc['source_context'] == spec['source_context']
        and all(doc[key] == spec['source_context'][key] for key in ('parent_schedule_sha256',
            'rewritten_schedule_sha256','certificate_sha256','resource_context')), 'canonical copy source/CPU/certificate context differs')
    probe = doc['probe']
    _need(probe['source_provenance'] == spec['provenance'] and probe['implementation_files'] == _pins()
        and probe['compiler'] == spec['compiler'] and set(probe['headers']) == {'scar_abi_v1.h','stateful_abi_v1.h'},
        'copy probe source implementation or compiler differs')
    for receipt in (probe['source'], probe['library'], *probe['headers'].values()):
        _need(_receipt(receipt['path']) == receipt, 'retained copy probe artifact changed')
    _need(Path(probe['source']['path']).read_text() == spec['source']
        and all(Path(ref['path']).read_bytes() == (_ROOT/'codegen'/name).read_bytes() for name,ref in probe['headers'].items())
        and probe['compiler_command'] == [*spec['compiler']['argv'], *spec['compiler']['options'],
            probe['source']['path'], '-o', probe['library']['path'], '-lm'], 'copy probe source/header/command differs from source replay')
    _need(doc['clock'] == _CLOCK and doc['conditions'] == list(CONDITIONS) and doc['missing_terms'] == list(MISSING)
        and len(doc['samples']) == doc['repetitions'], 'paired clock/applicability/residual protocol differs')
    for index,row in enumerate(doc['samples']):
        _need(row['index'] == index and row['order'] == (['ORIGINAL','FUSED'] if (index+doc['warmup'])%2 == 0 else ['FUSED','ORIGINAL'])
            and all(type(row[name]) is int and row[name] >= 0 for name in ('original_ns','fused_ns','original_checksum','fused_checksum'))
            and row['original_checksum'] == row['fused_checksum'] and row['checksums_equal'] is True,
            'retained paired sample, order or checksum differs')
    _need(doc['statistics'] == {name:_statistics([row[name+'_ns'] for row in doc['samples']]) for name in ('original','fused')},
        'copy window statistics differ from raw paired integer observations')
    _need(doc['caller_affinity_before'] == doc['caller_affinity_after']
        and set(spec['cpus']) <= set(doc['caller_affinity_before'])
        and doc['caller_rounding_before'] == doc['caller_rounding_after'] and doc['canonical_components_measured'] is True,
        'copy probe caller context/restoration differs')
    for name in ('target_program_executed','whole_target_call_measured','target_variants_ranked',
            'clock_overhead_subtracted','physical_optimality_proven','deployment_authorized'):
        _need(doc[name] is False, 'canonical window evidence cannot claim '+name)
    read_artifact(reference)
    context = {**spec['context'], 'rewrite_reference':doc['rewrite'], 'rewritten_schedule':spec['graph'],
        'rewritten_resources':spec['resources'], 'rewritten_table':spec['costs'],
        'certificate':spec['certificate'], 'calibration_document':doc, 'calibration_reference':reference}
    return doc, context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--rewrite'); source.add_argument('--check')
    parser.add_argument('--out'); parser.add_argument('--repetitions',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--max-payload-bytes',type=int,default=256*2**20)
    args = parser.parse_args(argv)
    if args.check:
        if args.out: parser.error('check consumes only saved calibration evidence')
        doc, _ = read_copy_window_calibration(args.check)
        result = dict(statistics=doc['statistics'], target_program_executed=False)
    else:
        if not args.out: parser.error('calibration requires --out')
        result = calibrate_copy_window(args.rewrite,args.out,repetitions=args.repetitions,
            warmup=args.warmup,max_payload_bytes=args.max_payload_bytes)
    print(json.dumps(result,sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
