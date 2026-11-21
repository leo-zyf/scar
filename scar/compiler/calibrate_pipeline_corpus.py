"""Calibrate complete guarded calls on generated, fixed-topology arithmetic.

Canonical programs are calibration inputs, not production F4 admissions. Only
canonical TRAIN rows fit coefficients. Mixed programs and the real source
region validate frozen parameters; no target strategy is searched or ranked.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.pipeline-corpus-calibration'
_SYMBOL = dict(add='+', sub='-', mul='*', div='/')
_OPS = ('add', 'sub', 'mul', 'div', 'neg')


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(Path(path).resolve(), value['schema'])


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_source(pipeline, active):
    groups = {}
    for call in pipeline.calls:
        groups.setdefault(call.callee.lookup_path[0], call)
    specs = [(op+'-depth-'+str(depth), 'TRAIN', op, depth)
        for op in active for depth in (1, 3)]
    specs += [('mixed-'+str(offset), 'CANONICAL_HOLDOUT', None, 2+offset) for offset in (0, 1)]
    text, cases = [], []
    for index, (identity, role, op, depth) in enumerate(specs):
        leaves = {}
        for group, (key, call) in enumerate(groups.items()):
            name = 'c'+str(index)+'_leaf'+str(group)
            parameters = ['p'+str(i) for i in range(len(call.inputs))]
            operation = active[(group+index) % len(active)] if op is None else op
            expression = parameters[0]
            if operation != 'neg':
                for parameter in parameters[1:]:
                    expression = '('+expression+_SYMBOL[operation]+parameter+')'
            for _ in range(depth):
                expression = '-('+expression+')' if operation == 'neg' else '('+expression+_SYMBOL[operation]+'1.125)'
            text.append('def '+name+'('+','.join(parameters)+'):\n    return '+expression+'\n')
            leaves[key] = name
        name = 'c'+str(index)+'_entry'
        parameters = ['x'+str(i) for i in range(len(pipeline.parameters))]
        values = dict(zip(pipeline.input_values, parameters))
        lines = ['def '+name+'('+','.join(parameters)+'):']
        for i, call in enumerate(pipeline.calls):
            local = 'v'+str(i)
            lines.append('    '+local+' = '+leaves[call.callee.lookup_path[0]]+'('+','.join(values[v] for v in call.inputs)+')')
            values[call.output] = local
        lines.append('    return '+values[pipeline.output])
        text.append('\n'.join(lines)+'\n')
        cases.append(dict(id=identity, role=role, name=name, parameters=parameters, leaves=leaves))
    return '\n'.join(text), cases


def _case_pipeline(original, case, namespace, regions, path, budget):
    from scar.compiler.model import SourceRef
    from scar.compiler.pipeline_model import RegionCall, CalleeGuard, compose_pipeline, prepare_pipeline_leaf
    from scar.optimization.optimizer import optimize_region
    from scar.trace.code_snapshot import code_key
    code = namespace[case['name']].__code__
    source = SourceRef(str(path), sha256(path.read_bytes()).hexdigest(), case['name'],
        code.co_firstlineno, 0, code_key(code),
        (dict(kind='GENERATED_CANONICAL_CALIBRATION_NOT_PRODUCTION_F4_ADMISSION'),))
    plans = {name:prepare_pipeline_leaf(optimize_region(regions[name], memory_budget_bytes=budget))
        for name in case['leaves'].values()}
    calls = []
    for call in original.calls:
        name = case['leaves'][call.callee.lookup_path[0]]
        plan = plans[name]
        calls.append(RegionCall(call.id, plan, call.inputs, call.output, CalleeGuard(plan.region.source, (name,))))
    return compose_pipeline(source, case['parameters'], calls, original.output, input_values=original.input_values)


def _measure(original, candidate, arrays, *, warmup, repetitions):
    import numpy as np
    expected = original(*arrays)
    before = [sha256(a.tobytes()).hexdigest() for a in arrays]
    for _ in range(warmup):
        original(*arrays)
        actual = candidate(*arrays)
        if not np.array_equal(expected.view(np.uint64), actual.view(np.uint64)):
            raise ValueError('canonical warm output differs from its source')
    stats = dict(candidate.scar_stats)
    samples = dict(original=[], fused=[])
    for index in range(repetitions):
        for side in (('original', 'fused') if index % 2 == 0 else ('fused', 'original')):
            function = original if side == 'original' else candidate
            start = time.perf_counter_ns()
            value = function(*arrays)
            elapsed = time.perf_counter_ns()-start
            samples[side].append(elapsed)
            if not np.array_equal(expected.view(np.uint64), value.view(np.uint64)):
                raise ValueError('clean full-call output differs from original arithmetic')
            del value
    if (candidate.scar_stats['native_calls']-stats['native_calls'] != repetitions
            or candidate.scar_stats['fallback_calls'] != stats['fallback_calls']):
        raise ValueError('full protected-call calibration fell back instead of executing native')
    if before != [sha256(a.tobytes()).hexdigest() for a in arrays]:
        raise ValueError('calibration changed read-only source input contents')
    return {side:dict(median_ns=statistics.median(values), samples_ns=values,
        min_ns=min(values), max_ns=max(values)) for side, values in samples.items()}


def calibrate(inputs, out, *, repetitions=9, warmup=3, build_jobs=4,
              maximum_holdout_relative_error=.30):
    from scar.compiler.project_pipeline_temporal import INPUTS
    from scar.compiler.macro_pipeline import _derive
    from scar.compiler.frontend import discover_regions
    from scar.compiler.pipeline_fusion_delivery import derive_fusion_blueprint, build_fusion_blueprint, _guarded_fusion_entry
    from scar.optimization.pipeline_memory import plan_pipeline_memory
    from scar.optimization.pipeline_demand import derive_pipeline_demand
    from scar.optimization.macro_pass import fusion_pass_cost_context, COST_SCOPE
    from scar.optimization.deployment_cost import _digest
    import numpy as np
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise ValueError('clean full-call calibration cannot use Python profile or trace hooks')
    if (type(repetitions) is not int or not 3 <= repetitions <= 101 or repetitions % 2 != 1
            or type(warmup) is not int or not 1 <= warmup <= 100
            or type(build_jobs) is not int or not 1 <= build_jobs <= 8):
        raise ValueError('bounded odd repetitions, positive warmup and 1..8 compilation jobs required')
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh canonical calibration output required')
    reference = inputs if type(inputs) is dict else artifact_reference(Path(inputs).resolve(), INPUTS)
    cpus = sorted(os.sched_getaffinity(0))
    started = time.monotonic()
    pipeline, region, blueprint, _, budget, _ = _derive(reference, None, cpus, None, .05)
    demand = derive_pipeline_demand(pipeline, blueprint)
    shapes = list(blueprint['input_shapes'].values())
    if not shapes or any(shape != shapes[0] for shape in shapes) or not all(shapes[0]) or np.prod(shapes[0]) > 1024*1024:
        raise ValueError('canonical fixed-shape profile requires equal nonempty arrays of at most one million elements')
    counts = [demand[side]['features']['arithmetic.'+op] for side in ('original','fused') for op in _OPS]
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError('source arithmetic must be resolved before canonical calibration')
    active = [op for op in _OPS if any(demand[side]['features']['arithmetic.'+op] for side in ('original','fused'))]
    if not active:
        raise ValueError('canonical arithmetic calibration requires a known nonempty operation basis')
    output.mkdir(parents=True)
    files = {name:_write(output/(name+'.json'), value) for name,value in
        (('region',region), ('blueprint',blueprint), ('demand',demand))}
    source, cases = _canonical_source(pipeline, active)
    source_path = output/'canonical.py'; source_path.write_text(source)
    namespace = _module(source_path, 'scar_canonical_costs').__dict__
    discovered, _ = discover_regions(source_path)
    regions = {row.source.qualname:row for row in discovered}
    for case in cases:
        case['pipeline'] = _case_pipeline(pipeline, case, namespace, regions, source_path, budget)
    refs = (dict(role='GENERATED_CALIBRATION_SOURCE_NOT_PRODUCTION_F4', path=str(source_path),
        sha256=sha256(source_path.read_bytes()).hexdigest()),)

    def build_case(row):
        case_pipeline = row['pipeline']
        memory = plan_pipeline_memory(case_pipeline,
            input_shapes={name:shapes[0] for name in row['parameters']}, memory_budget_bytes=budget)
        bp = derive_fusion_blueprint(case_pipeline, memory, blueprint['target'],
            available_cpu_ids=cpus, source_refs=refs)
        artifact = build_fusion_blueprint(case_pipeline, bp, output/'canonical-native'/row['id'])
        counts = derive_pipeline_demand(case_pipeline, bp)
        candidate = _guarded_fusion_entry(namespace[row['name']], artifact, bp, exclusive_inputs=True)
        return dict(row=row, blueprint=bp, demand=counts, artifact=artifact, candidate=candidate)

    # Compilation is parallel; every worker is joined before guarded timing.
    with ThreadPoolExecutor(max_workers=build_jobs) as pool:
        prepared = list(pool.map(build_case, cases))
    target_artifact = build_fusion_blueprint(pipeline, blueprint, output/'target-native')
    target_module = _module(Path(pipeline.source.path), 'scar_cost_target')
    original = target_module
    for part in pipeline.source.qualname.split('.'):
        original = getattr(original, part)
    target_candidate = _guarded_fusion_entry(original, target_artifact, blueprint, exclusive_inputs=True)
    rows = []
    rng = np.random.default_rng(83)
    for item in prepared:
        row = item['row']
        arrays = [rng.uniform(.5, 1., size=shapes[0]) for _ in row['parameters']]
        for array in arrays: array.flags.writeable = False
        timing = _measure(namespace[row['name']], item['candidate'], arrays,
            warmup=warmup, repetitions=repetitions)
        rows.append(dict(id=row['id'], role=row['role'],
            **{side:dict(features=item['demand'][side]['features'], timing=timing[side]) for side in ('original','fused')},
            native_manifest=artifact_reference(item['artifact'].directory/'artifact.json', item['artifact'].report['schema']),
            source_qualification='GENERATED_CANONICAL_CALIBRATION_NOT_PRODUCTION_F4_ADMISSION',
            outputs_bitwise_equal=True, native_calls=item['candidate'].scar_stats['native_calls']))
    arrays = [rng.uniform(.5, 1., size=shape) for shape in shapes]
    for array in arrays: array.flags.writeable = False
    timing = _measure(original, target_candidate, arrays, warmup=warmup, repetitions=repetitions)
    rows.append(dict(id='target-source', role='TARGET_HOLDOUT',
        **{side:dict(features=demand[side]['features'], timing=timing[side]) for side in ('original','fused')},
        outputs_bitwise_equal=True, native_calls=target_candidate.scar_stats['native_calls']))
    if cpus != sorted(os.sched_getaffinity(0)):
        raise ValueError('caller CPU affinity changed during calibration')
    observations = dict(schema='scar.pipeline-corpus-observations', schema_version=1,
        context=fusion_pass_cost_context(region, blueprint), demand_sha256=_digest(demand),
        measurement=dict(scope=COST_SCOPE, boundary='WARM_SESSION', reuse_count=1,
            original_entry='ORIGINAL_PYTHON_NO_OP', fused_entry='FULL_PROTECTED_FUSION_CALL',
            compilation_included=False, native_load_included=False,
            internal_instrumentation=False, final_output_release_included=False),
        qualification=dict(topology_fixed=True, input_arity_fixed=True,
            unique_callee_count_fixed=True, clean_protected_call=True),
        cases=rows, warmup=warmup, repetitions=repetitions, build_jobs=build_jobs,
        collector_implementation=dict(path=str(Path(__file__).resolve()),
            sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        canonical_source=dict(path=str(source_path), sha256=sha256(source_path.read_bytes()).hexdigest()),
        original_inputs=reference, source_files=dict(files),
        target_region_executed=True, full_target_program_executed=False,
        workload_versions_ranked=False, mathematical_implementations_compared=2,
        original_affinity_restored=True, input_values_retained=False, output_values_retained=False)
    files['observations'] = _write(output/'observations.json', observations)
    result = fit_observations(files['observations'], output/'model',
        maximum_holdout_relative_error=maximum_holdout_relative_error)
    result.update(collection_files=files, elapsed_seconds=time.monotonic()-started)
    _write(output/'result.json', result)
    return result


def fit_observations(observations, out, *, maximum_holdout_relative_error=.30):
    """Freeze and replay retained samples; no native loading or target rerun."""
    from scar.optimization.pipeline_corpus_model import fit_pipeline_corpus, profile_for_corpus_fit
    from scar.optimization.pipeline_forward_cost import predict_pipeline_forward_cost, deployment_profile_for_prediction
    from scar.optimization.macro_pass import run_fusion_pass
    reference = (observations if type(observations) is dict else
        artifact_reference(Path(observations).resolve(), 'scar.pipeline-corpus-observations'))
    original = read_artifact(reference)
    region, blueprint, demand = (read_artifact(original['source_files'][name])
        for name in ('region', 'blueprint', 'demand'))
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh corpus model output required')
    fit = fit_pipeline_corpus(region, blueprint, demand, original,
        maximum_holdout_relative_error=maximum_holdout_relative_error)
    output.mkdir(parents=True)
    files = dict(fit=_write(output/'fit.json', fit), observations=reference)
    coefficients = profile_for_corpus_fit(files['fit'], region, blueprint, demand, original)
    files['coefficients'] = _write(output/'coefficients.json', coefficients)
    prediction = predict_pipeline_forward_cost(region, blueprint, demand, coefficients)
    files['prediction'] = _write(output/'prediction.json', prediction)
    profile = deployment_profile_for_prediction(files['prediction'], region_document=region,
        blueprint=blueprint, demand=demand, coefficient_profile=coefficients)
    files['deployment_profile'] = _write(output/'profile.json', profile)
    decision = run_fusion_pass(region, blueprint, cost_profile=files['deployment_profile']['uri'],
        available_cpu_ids=blueprint['execution']['allowed_cpu_ids'],
        memory_budget_bytes=blueprint['selected_memory_plan']['memory_budget_bytes'])
    files['pass_report'] = _write(output/'pass.json', decision)
    result = dict(schema=SCHEMA, schema_version=1, files=files,
        status=fit['status'], profile_available=fit['profile_available'], decision=decision['decision'],
        role_counts=fit['role_counts'], basis=fit['basis'],
        maximum_holdout_relative_error=maximum_holdout_relative_error,
        collection_target_region_executed=True, fit_target_program_executed=False,
        full_target_program_executed=False,
        new_backend=False, physical_optimality_proven=False, deployment_authorized=False,
        cost_basis='CALIBRATED_DECLARED_EFFECTIVE_MODEL_NOT_OBSERVED_COMPONENTS')
    _write(output/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--inputs'); sources.add_argument('--observations', help='refit retained samples without executing a program')
    parser.add_argument('--out', required=True)
    parser.add_argument('--repetitions', type=int, default=9); parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--build-jobs', type=int, default=4)
    parser.add_argument('--maximum-holdout-relative-error', type=float, default=.30)
    args = parser.parse_args(argv)
    result = (fit_observations(args.observations, args.out,
        maximum_holdout_relative_error=args.maximum_holdout_relative_error) if args.observations else
        calibrate(args.inputs, args.out, repetitions=args.repetitions, warmup=args.warmup,
        build_jobs=args.build_jobs, maximum_holdout_relative_error=args.maximum_holdout_relative_error))
    print(json.dumps({name:result[name] for name in ('status','profile_available','decision','role_counts','basis')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
