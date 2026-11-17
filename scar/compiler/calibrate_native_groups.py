"""Freeze and sample canonical inputs in the original native stage frame.

Groups observe actual predecessors and slot/scalar dependencies. They do not
omit the untimed prefix/suffix or invent intermediate port values. This is a
diagnostic component experiment, not target-program benchmarking or a profile
qualification. The original native binary is only a canonical-output oracle.
"""
import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

from scar.ir.execution_recipe import artifact_reference, read_artifact

DESIGN = 'scar.native-stage-group-calibration-design'
OBSERVATIONS = 'scar.native-stage-group-calibration'
_ROOT = Path(__file__).resolve().parents[1]
_FILES = ('compiler/calibrate_native_groups.py', 'codegen/native_stage_group.py',
    'optimization/native_stage_context.py', 'compiler/native_lowering_view.py',
    'codegen/stateful_stage_capture.py', 'codegen/stateful.py', 'codegen/cpp.py',
    'codegen/stateful_v1.cpp.in', 'codegen/loop_v1.cpp.in',
    'codegen/stateful_abi_v1.h', 'codegen/scar_abi_v1.h',
    'compiler/collect_native_optimization.py', 'analysis/gcc_optimization.py',
    'analysis/gcc_cost_features.py')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _pins():
    return {name:sha256((_ROOT/name).read_bytes()).hexdigest() for name in _FILES}


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _ref(value, schema=None):
    if type(value) is dict:
        reference = deepcopy(value)
    else:
        path = Path(value).absolute()
        reference = artifact_reference(path, schema or json.loads(path.read_bytes())['schema'])
    read_artifact(reference, json_document=False)
    return reference


def _write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return artifact_reference(path, document['schema'])


def _runtime_protocol():
    return dict(single_python_thread_required=True,
        data_regime='RESET_CANONICAL_ARRAY_CONTENTS_BEFORE_EACH_CALL_OUTSIDE_TIMING',
        scalar_regime='DECLARED_FLOAT_SCALARS_WITH_ZERO_INTEGER_TAGS',
        arrays_have_independent_owning_allocations=True,
        first_touch='UNBOUND_CALLER_RESET_BEFORE_ORIGINAL_NATIVE_BIND',
        physical_numa_residency='UNKNOWN', cache_residency='UNKNOWN', exclusive_cpu_reservation=False)


def _timer():
    return dict(api='CLOCK_MONOTONIC', unit='INTEGER_NANOSECONDS',
        inner='ONLY_SELECTED_STAGE_BODIES_WITH_OBSERVER_CLOCKS_AND_CONSTRAINTS',
        outer='CTYPES_FUNCTION_POINTER_CALL_TO_RETURN_WITH_PREPACKED_DESCRIPTORS',
        reader_descriptor_build_reset_and_hash_outside_timing=True,
        outer_is_not_whole_python_method=True, clock_overhead_subtracted=False)


def _derive(request):
    from scar.codegen.native_stage_group import render_native_stage_group
    from scar.optimization.native_stage_context import derive_native_stage_context
    _need(type(request['repetitions']) is int and 3 <= request['repetitions'] <= 10000
        and type(request['warmup']) is int and 1 <= request['warmup'] <= 10000,
        'bounded repetitions and warmup required')
    _need(all(type(request[key]) is float and math.isfinite(request[key])
        for key in ('array_value', 'scalar_value')), 'explicit finite canonical float values required')
    _need(type(request['max_external_bytes']) is int and request['max_external_bytes'] > 0,
        'positive explicit canonical allocation budget required')
    source, provenance = render_native_stage_group(request['native_manifest'], groups=request['groups'])
    context = derive_native_stage_context(request['native_manifest'], groups=request['groups'])
    view = provenance['parent_view']; plan = read_artifact(view['plan'])
    _need(plan['external_array_bytes'] <= request['max_external_bytes'], 'canonical external inputs exceed budget')
    return source, provenance, context, view


def prepare_native_groups(native, out, *, groups=None, repetitions=7, warmup=2,
                           array_value=1.25, scalar_value=0.75, max_external_bytes=64*2**20):
    from scar.runtime.compiled_deployment import thread_environment
    request = dict(native_manifest=_ref(native), groups=groups, repetitions=repetitions, warmup=warmup,
        array_value=array_value, scalar_value=scalar_value, max_external_bytes=max_external_bytes)
    _need(type(max_external_bytes) is int and max_external_bytes > 0, 'positive canonical allocation budget required')
    source, provenance, context, view = _derive(request)
    directory = Path(out).absolute()
    _need(not directory.exists(), 'fresh native group calibration output required')
    directory.mkdir(parents=True)
    path = directory/'observer.cpp'; path.write_text(source)
    headers = {}
    for name, reference in view['headers'].items():
        copied = directory/name; copied.write_bytes(read_artifact(reference, json_document=False))
        headers[name] = artifact_reference(copied, 'scar.cpp-header')
    document = dict(schema=DESIGN, schema_version=1, created_at=_now(), request=request,
        implementation_pins=_pins(), observer=provenance, stage_context=context,
        source=artifact_reference(path, 'scar.cpp-source'), headers=headers,
        runtime_context=dict(caller_affinity_cpu_ids=sorted(os.sched_getaffinity(0)),
            native_thread_environment=thread_environment(), **_runtime_protocol()),
        timer=_timer(),
        output_oracle='UNCHANGED_ORIGINAL_NATIVE_ON_ONE_INDEPENDENT_CANONICAL_INPUT',
        target_program_executed=False, compiled=False, native_loaded=False,
        latency_qualified=False, deployment_authorized=False)
    reference = _write(directory/'design.json', document)
    return reference


def check_native_group_design(design, *, expected_sha256):
    reference = _ref(design, DESIGN)
    _need(reference['sha256'] == expected_sha256, 'frozen native group design identity differs')
    document = read_artifact(reference)
    _need((document.get('schema'), document.get('schema_version')) == (DESIGN, 1)
        and document['implementation_pins'] == _pins(), 'group design schema or implementation differs')
    runtime = document['runtime_context']; cpus = runtime['caller_affinity_cpu_ids']
    _need(type(cpus) is list and cpus and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
        and cpus == sorted(set(cpus)) and type(runtime['native_thread_environment']) is dict
        and _same(runtime, dict(caller_affinity_cpu_ids=cpus,
            native_thread_environment=runtime['native_thread_environment'], **_runtime_protocol()))
        and _same(document['timer'], _timer())
        and document['output_oracle'] == 'UNCHANGED_ORIGINAL_NATIVE_ON_ONE_INDEPENDENT_CANONICAL_INPUT',
        'frozen timer, runtime input regime or output-oracle scope differs')
    source, provenance, context, view = _derive(document['request'])
    _need(_same(document['observer'], provenance) and _same(document['stage_context'], context)
        and read_artifact(document['source'], json_document=False) == source.encode(),
        'group observer source, context or original native replay differs')
    _need(set(document['headers']) == set(view['headers']), 'original ABI headers required')
    for name, row in document['headers'].items():
        _need(read_artifact(row, json_document=False) == read_artifact(view['headers'][name], json_document=False),
            'copied group ABI header differs')
    return document


def native_group_statistics(document):
    """Check each joint same-clock sample before aggregating group durations."""
    groups = document['group_layout']; samples = document['samples']
    _need(type(samples) is list and len(samples) == document['repetitions'] and samples,
        'all planned native group samples required')
    columns = [[] for _ in groups]; outer_values = []; complementary = []
    for index, sample in enumerate(samples):
        _need(type(sample['ordinal']) is int and sample['ordinal'] == index
            and type(sample['native_status']) is int and sample['native_status'] == 0,
            'ordered successful native calls required')
        cuts = sample['cuts_ns']; start, end = sample['ffi_start_ns'], sample['ffi_end_ns']
        _need(type(cuts) is list and len(cuts) == 2*len(groups)
            and all(type(value) is int and value > 0 for value in (start, *cuts, end))
            and all(a <= b for a, b in zip((start, *cuts), (*cuts, end))),
            'group and outer clocks must be integer, same-clock and ordered')
        durations = [cuts[g['end_cut']]-cuts[g['begin_cut']] for g in groups]
        _need(all(value >= 0 for value in durations) and sum(durations) <= end-start,
            'disjoint group intervals must fit their joint FFI window')
        for values, duration in zip(columns, durations): values.append(duration)
        outer_values.append(end-start); complementary.append(end-start-sum(durations))

    def stats(values):
        return dict(samples_ns=values, mean_ns=statistics.mean(values),
            exact_mean_ns=dict(numerator=sum(values), denominator=len(values)),
            median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
            sample_variance_ns2=statistics.variance(values) if len(values) > 1 else None)

    return dict(groups=[dict(index=g['index'], stage_indices=g['stage_indices'], **stats(values))
            for g, values in zip(groups, columns)], ffi_window=stats(outer_values),
        joint_unattributed_outside_groups=stats(complementary),
        complement_is_not_independent_cleanup_or_guard_measurement=True,
        sum_of_means_equals_mean_joint_partition=True, component_medians_summed=False)


def _runtime_matches(design):
    from scar.runtime.compiled_deployment import thread_environment
    _need(threading.active_count() == 1 and sys.gettrace() is None and sys.getprofile() is None,
        'single Python thread without profile/trace required for diagnostic sampling')
    context = design['runtime_context']
    _need(sorted(os.sched_getaffinity(0)) == context['caller_affinity_cpu_ids']
        and thread_environment() == context['native_thread_environment'], 'frozen caller context differs')


def run_native_groups(design, *, expected_sha256):
    from scar.codegen import stateful, cpp
    from scar.compiler.collect_native_optimization import collect_gcc_optimization
    from scar.analysis.gcc_cost_features import project_gcc_vector_costs
    from scar.compiler.stateful_model import stateful_from_dict
    import numpy as np

    design_ref = _ref(design, DESIGN)
    frozen = check_native_group_design(design_ref, expected_sha256=expected_sha256)
    _runtime_matches(frozen)
    request = frozen['request']; observer = frozen['observer']; view = observer['parent_view']
    directory = Path(design_ref['uri']).parent
    _need(not (directory/'observations.json').exists(), 'one sampling run per frozen group design required')
    spans = [dict(id=row['id'], **span) for row in observer['stages'] for span in row['source_line_spans']]
    collection = collect_gcc_optimization(frozen['source'], view['compiler'], directory/'compiled',
        headers=list(frozen['headers'].values()), stage_spans=spans,
        subject=dict(kind='INSTRUMENTED_ORIGINAL_NATIVE_GROUP_FRAME', design=design_ref,
            observer_identity=observer['observer_identity'], original_native_manifest=view['native_manifest']))
    collection_ref = artifact_reference(directory/'compiled/result.json', collection['schema'])
    vector_features = project_gcc_vector_costs(collection_ref)
    features_ref = _write(directory/'compiler-cost-features.json', vector_features)
    library = ctypes.CDLL(collection['files']['library']['uri'])
    identity = getattr(library, observer['identity_symbol']); identity.argtypes=[]; identity.restype=ctypes.c_char_p
    _need(identity() == observer['observer_identity'].encode(), 'loaded diagnostic observer identity differs')
    getter = library.scar_get_stateful_table_v1; getter.argtypes=[]; getter.restype=ctypes.POINTER(stateful._Table)
    table = getter()
    _need(table and table.contents.abi_version == 1 and table.contents.struct_size == ctypes.sizeof(stateful._Table)
        and table.contents.run, 'diagnostic native table ABI differs')
    manifest = read_artifact(view['native_manifest'])
    for name in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        _need(getattr(table.contents, name) == manifest[name].encode(), 'parent mathematical contract differs')
    region = stateful_from_dict(read_artifact(view['region'])); plan = read_artifact(view['plan'])
    arrays = {row.id:np.full(plan['input_shapes'][row.id], request['array_value'], dtype=np.float64)
        for row in region.inputs if row.kind == 'array'}
    scalars = {row.id:request['scalar_value'] for row in region.inputs if row.kind == 'scalar'}
    native_providers, providers, owners, power = stateful._providers(region)
    _need(native_providers == manifest['provider_runtime'], 'diagnostic original provider runtime differs')
    _need((table.contents.array_count, table.contents.scalar_count, table.contents.provider_count) ==
        (len(arrays), len(scalars), len(region.providers)), 'diagnostic ABI port count differs')
    descriptors = (cpp._Array*len(arrays))(*[cpp._descriptor(array) for array in arrays.values()])
    native_scalars = (stateful._Scalar*len(scalars))(*[stateful._Scalar(value, 0) for value in scalars.values()])
    reader = getattr(library, observer['read_symbol'])
    reader.argtypes=[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]; reader.restype=ctypes.c_int32
    cuts = (ctypes.c_uint64*observer['cut_count'])()
    run = table.contents.run

    # Independent original buffers; no target Python function or episode is imported.
    original = stateful.load(Path(view['native_manifest']['uri']).parent)
    oracle_arrays = {key:array.copy() for key,array in arrays.items()}
    original.run(oracle_arrays, scalars)
    _need(run(descriptors, len(arrays), native_scalars, len(scalars), providers, len(region.providers)) == 0,
        'diagnostic observer declined canonical correctness input')
    oracle = {}
    for key,array in arrays.items():
        _need(array.tobytes() == oracle_arrays[key].tobytes(), 'observer changed canonical output: '+key)
        oracle[key] = dict(shape=list(array.shape), dtype=array.dtype.str,
            observer_output_sha256=sha256(array.tobytes()).hexdigest(),
            original_output_sha256=sha256(oracle_arrays[key].tobytes()).hexdigest(),
            bitwise_equal_to_original=True)
    _runtime_matches(frozen)
    samples = []; observed_from = _now(); load_before = list(os.getloadavg())
    try:
        for index in range(request['warmup']+request['repetitions']):
            for array in arrays.values(): array.fill(request['array_value'])
            _runtime_matches(frozen)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            status = run(descriptors, len(arrays), native_scalars, len(scalars), providers, len(region.providers))
            end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            _need(status == 0 and reader(cuts, len(cuts)) == 0, 'native group or diagnostic cut collection failed')
            _runtime_matches(frozen)
            if index >= request['warmup']:
                samples.append(dict(ordinal=index-request['warmup'], native_status=status,
                    ffi_start_ns=start, cuts_ns=list(cuts), ffi_end_ns=end))
    finally:
        after = sorted(os.sched_getaffinity(0))
        if after != frozen['runtime_context']['caller_affinity_cpu_ids']:
            os.sched_setaffinity(0, frozen['runtime_context']['caller_affinity_cpu_ids'])
    _need(after == frozen['runtime_context']['caller_affinity_cpu_ids'], 'diagnostic source did not restore caller affinity')
    _need(frozen['implementation_pins'] == _pins(), 'group implementation changed during sampling')
    _runtime_matches(frozen)
    document = dict(schema=OBSERVATIONS, schema_version=1, design=design_ref,
        frozen_design_sha256=expected_sha256, compiler_collection=collection_ref,
        compiler_features=features_ref, group_layout=observer['groups'],
        warmup=request['warmup'], repetitions=request['repetitions'], samples=samples,
        canonical_output_oracle=oracle, observed_from=observed_from, observed_until=_now(),
        host_load=dict(before=load_before, after=list(os.getloadavg()), isolated=False),
        original_source_stages_compiled_and_executed=True, full_native_prefix_suffix_and_commits_executed=True,
        original_native_loaded_as_canonical_oracle=True, diagnostic_native_loaded=True,
        caller_context_restored=True, target_python_program_imported=False, target_episode_executed=False,
        target_variants_ranked=False, target_latency_prediction_published=False,
        clock_overhead_subtracted=False, qualified_profile=None, deployment_authorized=False,
        residuals=observer['retained_residuals']+['DECLARED_CANONICAL_INPUT_DATA_NOT_TARGET_DISTRIBUTION',
            'RESET_FIRST_TOUCH_ALLOCATOR_AND_CACHE_REGIME', 'COMPONENT_TO_TARGET_TRANSFER_NOT_VALIDATED'])
    document['statistics'] = native_group_statistics(document)
    return _write(directory/'observations.json', document)


def check_native_group_observations(observations):
    from scar.compiler.collect_native_optimization import check_gcc_optimization
    from scar.analysis.gcc_cost_features import project_gcc_vector_costs
    reference = _ref(observations, OBSERVATIONS); document = read_artifact(reference)
    _need((document.get('schema'), document.get('schema_version')) == (OBSERVATIONS, 1), 'versioned native group observations required')
    design = check_native_group_design(document['design'], expected_sha256=document['frozen_design_sha256'])
    _need(_same(document['group_layout'], design['observer']['groups'])
        and document['warmup'] == design['request']['warmup']
        and document['repetitions'] == design['request']['repetitions'], 'frozen group plan differs')
    collection = check_gcc_optimization(document['compiler_collection'])
    spans = [dict(id=row['id'], **span) for row in design['observer']['stages']
        for span in row['source_line_spans']]
    _need(collection['source'] == design['source'] and collection['compiler'] == design['observer']['parent_view']['compiler']
        and collection['headers'] == list(design['headers'].values())
        and _same(collection['stage_spans'], spans) and collection['link_options'] == ['-lm']
        and collection['subject'] == dict(kind='INSTRUMENTED_ORIGINAL_NATIVE_GROUP_FRAME',
            design=document['design'], observer_identity=design['observer']['observer_identity'],
            original_native_manifest=design['observer']['parent_view']['native_manifest']),
        'group observer compilation belongs to a different frozen source/context')
    _need(read_artifact(document['compiler_features']) == project_gcc_vector_costs(document['compiler_collection']),
        'group compiler features differ from original GCC record')
    _need(document['statistics'] == native_group_statistics(document), 'native group statistics differ from joint samples')
    from scar.compiler.stateful_model import stateful_from_dict
    view = design['observer']['parent_view']
    region = stateful_from_dict(read_artifact(view['region'])); plan = read_artifact(view['plan'])
    expected_arrays = {row.id for row in region.inputs if row.kind == 'array'}
    _need(set(document['canonical_output_oracle']) == expected_arrays, 'complete canonical array oracle required')
    for key, row in document['canonical_output_oracle'].items():
        _need(row['shape'] == plan['input_shapes'][key] and row['dtype'] == '<f8'
            and row['bitwise_equal_to_original'] is True
            and type(row['observer_output_sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', row['observer_output_sha256']) is not None
            and row['observer_output_sha256'] == row['original_output_sha256'],
            'retained independent canonical-output oracle differs')
    for key in ('original_source_stages_compiled_and_executed', 'full_native_prefix_suffix_and_commits_executed',
                'original_native_loaded_as_canonical_oracle', 'diagnostic_native_loaded', 'caller_context_restored'):
        _need(document[key] is True, 'canonical native observation must disclose '+key)
    _need(document['residuals'] == design['observer']['retained_residuals']+
        ['DECLARED_CANONICAL_INPUT_DATA_NOT_TARGET_DISTRIBUTION',
         'RESET_FIRST_TOUCH_ALLOCATOR_AND_CACHE_REGIME', 'COMPONENT_TO_TARGET_TRANSFER_NOT_VALIDATED'],
        'component transfer residuals cannot be removed')
    for key in ('target_python_program_imported', 'target_episode_executed', 'target_variants_ranked',
                'target_latency_prediction_published', 'clock_overhead_subtracted', 'deployment_authorized'):
        _need(document[key] is False, 'diagnostic component evidence cannot claim '+key)
    _need(document['qualified_profile'] is None, 'group observations cannot grant a qualified profile')
    read_artifact(reference)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native'); parser.add_argument('--out'); parser.add_argument('--groups-json')
    parser.add_argument('--design'); parser.add_argument('--sha256'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--check'); parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args(argv)
    if args.check:
        document = check_native_group_observations(args.check)
        print(json.dumps(dict(cold_replayed=True, statistics=document['statistics'], qualified_profile=None)))
    elif args.run:
        if not args.design or not args.sha256: parser.error('--run requires --design and --sha256')
        print(json.dumps(run_native_groups(args.design, expected_sha256=args.sha256)))
    else:
        if not args.native or not args.out: parser.error('--native and --out required to freeze a design')
        print(json.dumps(prepare_native_groups(args.native, args.out,
            groups=None if args.groups_json is None else json.loads(args.groups_json),
            repetitions=args.repetitions, warmup=args.warmup)))


if __name__ == '__main__':
    main()
