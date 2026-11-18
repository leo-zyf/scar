"""Plan the cost-selected semantic graph and bind its existing C++ lowering.

Physical obligations, logical planning, explicit native validation, and
deployment authority are independently recorded.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import problem_document, digest, validate_problem
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.semantic-region-planning'
BUILD = 'scar.semantic-region-build'
_PRODUCERS = ('compiler/plan_semantic_region.py', 'compiler/macro_pipeline.py',
    'compiler/semantic_revision.py', 'analysis/source_pipeline_graph.py',
    'optimization/semantic_rewrite.py', 'optimization/semantic_region_demand.py',
    'optimization/region_implementation.py', 'optimization/constructive_planner.py',
    'optimization/joint_planner.py', 'ir/implementation_plan.py',
    'compiler/pipeline_fusion_delivery.py')


def _pins():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in _PRODUCERS}


def _reference(value, schema):
    return value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _forward_context(parent, model_reference):
    from scar.compiler.model_pipeline_cost import check
    from scar.compiler.semantic_revision import _selection

    model, context = check(model_reference, include_macro_context=True)
    if context is None or model['macro_preparation'] != parent:
        raise ValueError('forward model must replay the same macro preparation')
    saved, _, _, blueprint, _ = context
    candidate = model['candidate_binding']
    if (candidate['candidate_revision'] != saved['files']['candidate_revision']
            or candidate['candidate_demand'] != saved['files']['candidate_demand']):
        raise ValueError('forward model must use the published candidate graph and demand')
    revision = read_artifact(candidate['candidate_revision'])
    demand = read_artifact(candidate['candidate_demand'])
    if (revision['decision'] != 'REWRITE'
            or revision['after_revision'] != candidate['graph_revision']
            or semantic_sha256(revision) != candidate['semantic_revision_sha256']
            or demand['graph_revision'] != revision['after_revision']
            or demand['implementation_lowering_demand']['plan_sha256']
                != candidate['optimized_plan_sha256']
                or candidate['optimized_plan_sha256'] != blueprint['optimized_plan_sha256']):
        raise ValueError('forward cost differs from the actual candidate graph or lowering')
    pass_report = read_artifact(model['files']['pass_report'])
    selection, alternative, rewrite = _selection(pass_report['optimization_selection'])
    if (model['decision'] != pass_report['decision'] or
            (rewrite and (alternative.delta.as_dict() != revision['delta'] or
                {key: value for key, value in selection.as_dict().items() if key != 'reason'} !=
                {key: value for key, value in revision['selection'].items() if key != 'reason'}))):
        raise ValueError('cost pass selects a different semantic rewrite')
    binding = dict(model=model_reference, pass_report=model['files']['pass_report'],
        decision=model['decision'], candidate_revision=candidate['candidate_revision'],
        candidate_demand=candidate['candidate_demand'])
    return context, pass_report, binding


def _retain_original(saved, pipeline, report, *, output, reference):
    """Reuse the parent before graph if newer cost evidence retracts a rewrite."""
    from scar.compiler.project_pipeline_temporal import open_temporal_inputs
    from scar.compiler.semantic_revision import _successor

    original = read_artifact(saved['files']['semantic_revision'])['files']
    with open_temporal_inputs(saved['inputs'], include_source=True) as (*_, values, _raw):
        _, _, revision, _, _ = _successor(pipeline, report['optimization_selection'],
            values, original['before'], None)
    revision['files'] = dict(before=original['before'], after=original['before'], body=None,
        before_analysis=original['before_analysis'], after_analysis=original['before_analysis'])
    if output is not None:
        reference = _write(output/'kept-revision.json', revision)
    elif read_artifact(reference) != revision:
        raise ValueError('new KEEP must retain the original source graph from the macro input')
    return reference


def _derive(parent, *, logical_analysis, model_reference=None, output=None, kept_revision=None):
    from scar.compiler.macro_pipeline import check
    from scar.optimization.semantic_region_demand import derive_semantic_region_demand
    from scar.optimization.region_implementation import project_region_implementation

    cost_binding = None
    if model_reference is None:
        saved, pipeline, region, blueprint, report = check(parent)
        original_report = report
    else:
        context, report, cost_binding = _forward_context(parent, model_reference)
        saved, pipeline, region, blueprint, original_report = context
    if output is not None:
        output.mkdir(parents=True)
    revision_ref = saved['files']['semantic_revision']
    if cost_binding is not None:
        if report['decision'] == 'REWRITE':
            revision_ref = cost_binding['candidate_revision']
        elif original_report['decision'] == 'REWRITE':
            revision_ref = _retain_original(saved, pipeline, report,
                output=output, reference=kept_revision)
    revision = read_artifact(revision_ref)
    demand = (read_artifact(cost_binding['candidate_demand'])
        if cost_binding is not None and report['decision'] == 'REWRITE' else
        derive_semantic_region_demand(revision, pipeline,
            (original_report if revision_ref == saved['files']['semantic_revision']
                else report)['optimization_selection'], blueprint))
    problem = project_region_implementation(region, blueprint, report, logical_analysis=logical_analysis)
    if len(problem.operations) != 1 or problem.operations[0].predecessors or problem.data_edges:
        raise ValueError('complete selected source region must project as one entry operation')
    binding = dict(schema='scar.semantic-successor-planning-reference',
        semantic_revision=revision_ref, graph=revision['files']['after'],
        graph_revision=revision['after_revision'], root=revision['root'],
        selected_oir=deepcopy(revision['selection']), demand_sha256=semantic_sha256(demand))
    if cost_binding is not None:
        binding['forward_cost'] = cost_binding
    old = problem.operations[0]
    parameters = deepcopy(old.options[0].parameters)
    parameters.update(semantic_successor=binding,
        active_members=[row['operation'] for row in demand['active']['operations']],
        implementation_lowering_demand=deepcopy(demand['implementation_lowering_demand']),
        old_call_schedule_reused=False,
        semantic_demand_is_physical_traffic=False)
    parameters['lowering_binding'] = (dict(
        qualification='SOURCE_FUSION_AND_EXISTING_OPTIMIZER_REPLAYED_NOT_NATIVE_ADMISSION',
        raw_semantic_body=revision['files']['body'],
        optimized_plan_sha256=blueprint['optimized_plan_sha256'],
        strategy_sha256=blueprint['strategy_sha256']) if report['decision'] == 'REWRITE' else None)
    mode = replace(old.options[0], proof_refs=old.options[0].proof_refs+(binding,), parameters=parameters)
    operation = replace(old, id=revision['root']['wire'], source_refs=old.source_refs+(binding,), options=(mode,))
    problem = validate_problem(replace(problem, source_refs=problem.source_refs+(binding,), operations=(operation,)))
    if problem_document(problem)['target'] != blueprint['target']:
        raise ValueError('successor planning lost the original hardware snapshot')
    return pipeline, region, blueprint, report, revision_ref, revision, demand, problem, cost_binding


def _metadata(parent, *, logical_analysis, max_seconds, files, execution, problem, cost_binding):
    return dict(schema=SCHEMA, schema_version=2, macro_preparation=parent,
        logical_analysis=logical_analysis, max_seconds=max_seconds, files=files,
        problem_sha256=digest(problem), solver_status=execution['status'], producer_pins=_pins(),
        actual_semantic_successor_consumed=True, old_call_schedule_reused=False,
        target_hardware=problem_document(problem)['target'],
        physical_feasibility_proven=False, execution_authorized=False,
        native_build_qualification='EXPLICIT_GUARDED_VALIDATION_ONLY',
        performance_measured=False, deployment_authorized=False, default_backend='NO_OP',
        forward_cost=cost_binding)


def _native_metadata(artifact_ref, blueprint):
    """Replay the existing serial artifact receipts without opening its ELF.

    ABI table admission belongs to explicit build/runtime loading. A namespace
    below supplies only data to existing receipt serializers, never a callable
    CppArtifact or a substitute for runtime admission.
    """
    from scar.codegen import cpp, fused_pipeline
    from scar.compiler.model import plan_from_dict
    from scar.compiler.pipeline_model import pipeline_from_dict

    metadata = read_artifact(artifact_ref)
    directory = Path(artifact_ref['uri']).parent
    if ((metadata.get('schema'), metadata.get('schema_version')) != (fused_pipeline.SCHEMA, 1)
            or metadata['files'] != fused_pipeline._FILES or len(metadata['operators']) != 1):
        raise ValueError('one existing serial fusion artifact required')
    for name, expected in (('pipeline', blueprint['pipeline']), ('strategy', blueprint['strategy']),
            ('memory_plan', blueprint['selected_memory_plan'])):
        path = directory/metadata['files'][name]
        if sha256(path.read_bytes()).hexdigest() != metadata['sha256'][name] or json.loads(path.read_bytes()) != expected:
            raise ValueError('native fusion sidecar differs: '+name)
    pipeline = pipeline_from_dict(blueprint['pipeline'])
    strategy, expected_plan, memory = fused_pipeline._admit(pipeline, blueprint['strategy'])
    fused_pipeline._sources(pipeline)
    binding = metadata['operators'][0]
    operator_dir = Path(binding['artifact_directory']).resolve()
    manifest = operator_dir/'artifact.json'
    operator = json.loads(manifest.read_bytes())
    files = operator['files']
    if (set(files) != {'plan', 'source', 'abi_header', 'library'}
            or any(type(name) is not str or Path(name).name != name for name in files.values())
            or set(operator['sha256']) != {'source', 'abi_header', 'library'}
            or (operator.get('schema'), operator.get('schema_version'), operator.get('operator_abi_version'),
                operator.get('cxx_standard')) != ('scar.native-artifact', 1, 1, 20)):
        raise ValueError('supported untimed C++20 ABI1 serial operator required')
    plan_wire = json.loads((operator_dir/files['plan']).read_bytes())
    if plan_wire != blueprint['optimized_plan']:
        raise ValueError('native operator plan differs from selected optimized lowering')
    plan = plan_from_dict(plan_wire)
    if (plan.resource_binding is not None or plan.cpu_threads != 1
            or cpp._plan_digest(expected_plan) != cpp._plan_digest(plan)):
        raise ValueError('fusion operator must retain its replayed serial caller policy')
    backend = blueprint['backend']
    for name in ('source', 'abi_header', 'library'):
        if sha256((operator_dir/files[name]).read_bytes()).hexdigest() != operator['sha256'][name]:
            raise ValueError('native operator member bytes differ: '+name)
    source = cpp._source(plan, cpp._supported(plan)).encode()
    header = (operator_dir/files['abi_header']).read_bytes()
    if (sha256(source).hexdigest() != operator['sha256']['source']
            or operator['sha256']['source'] != backend['generated_source_sha256']
            or operator['sha256']['abi_header'] != backend['abi_header_sha256']
            or operator['plan_sha256'] != cpp._plan_digest(plan)
            or operator['runtime_implementation_sha256'] != backend['operator_runtime_implementation_sha256']
            or operator['compiler_options'] != backend['compiler']['options']
            or operator['compiler'] != backend['compiler']['command']
            or operator['build_id'] != sha256(source+header+repr(tuple(operator['compiler_options'])).encode()).hexdigest()):
        raise ValueError('native source, compiler, ABI or build identity differs')
    receipt_data = SimpleNamespace(report=operator, directory=operator_dir, plan=plan, metadata_path=manifest)
    if fused_pipeline._operator_receipt(receipt_data) != binding:
        raise ValueError('native operator receipt differs from retained member bytes')
    expected = fused_pipeline._manifest(pipeline, strategy, memory, receipt_data,
        binding, backend['implementation_sha256'], directory)
    if metadata != expected:
        raise ValueError('native fusion manifest differs from complete replay')
    return artifact_reference(manifest, operator['schema'])


def _native_binding(artifact_ref, blueprint, revision_ref, revision, demand, problem, execution,
        *, cost_binding, study_only):
    operator_ref = _native_metadata(artifact_ref, blueprint)
    lowering = demand['implementation_lowering_demand']
    if semantic_sha256(blueprint['optimized_plan']) != lowering['plan_sha256']:
        raise ValueError('native body is not the demand-qualified optimized lowering')
    return dict(schema='scar.semantic-region-native-binding', schema_version=2,
        semantic_revision=revision_ref, forward_cost=cost_binding, study_only=study_only,
        graph_revision=revision['after_revision'], root=revision['root'],
        semantic_graph=revision['files']['after'], raw_semantic_body=revision['files']['body'],
        selected_oir=revision['selection'], demand_sha256=semantic_sha256(demand),
        optimized_plan_sha256=lowering['plan_sha256'],
        strategy_sha256=blueprint['strategy_sha256'], native_artifact=artifact_ref,
        generated_cpp_sha256=blueprint['backend']['generated_source_sha256'],
        operator_manifest=operator_ref,
        cpp_standard='c++20', operator_abi_version=1,
        execution=deepcopy(blueprint['execution']),
        planning_problem_sha256=digest(problem), planning_status=execution['status'],
        planning_problem_is_candidate=not study_only,
        unresolved_resources_retained=True, logical_schedule_authorizes_native=False,
        cold_verification='MANIFEST_BYTES_PLAN_AND_CPP_ONLY_NO_CDLL_OR_ABI_GETTER',
        runtime_abi_admission_performed_by_cold_check=False,
        qualification='EXPLICIT_SOURCE_GUARDED_VALIDATION_CANDIDATE',
        automatic_deployment_authorized=False)


def _build(preparation, decision, revision_ref, native, binding, compile_validation, status):
    return dict(schema=BUILD, schema_version=2, preparation=preparation,
        decision=decision, semantic_revision=revision_ref,
        native_artifact=native, native_binding=binding, native_compiled=native is not None,
        native_library_loaded_at_build=native is not None,
        runtime_abi_admission_at_build=native is not None,
        compile_validation_requested=compile_validation, planning_status=status,
        whole_program_cpp=False, target_program_executed=False, performance_measured=False,
        deployment_authorized=False, default_backend='NO_OP')


def _native_subject(revision_ref, revision, demand, cost_binding):
    if revision['decision'] == 'REWRITE':
        return revision_ref, revision, demand
    reference = cost_binding['candidate_revision']
    return reference, read_artifact(reference), read_artifact(cost_binding['candidate_demand'])


def prepare(parent, out, *, logical_analysis=False, max_seconds=30, compile_validation=False,
        forward_model=None):
    from scar.compiler.macro_pipeline import SCHEMA as PARENT
    from scar.compiler.model_pipeline_cost import SCHEMA as COST
    from scar.optimization.constructive_planner import construct_implementations
    from scar.optimization.joint_planner import verify_execution_plan

    reference = _reference(parent, PARENT)
    model = None if forward_model is None else _reference(forward_model, COST)
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh semantic region planning output required')
    if type(compile_validation) is not bool:
        raise ValueError('native validation compilation must be explicitly selected')
    pipeline, _, blueprint, report, revision_ref, revision, demand, problem, cost_binding = _derive(
        reference, logical_analysis=logical_analysis, model_reference=model, output=directory)
    execution = construct_implementations(problem, max_seconds=max_seconds)
    verification = verify_execution_plan(problem, execution) if execution['selected'] else None
    study_only = report['decision'] == 'KEEP'
    reuse_demand = cost_binding is not None and not study_only
    files = dict(demand=(cost_binding['candidate_demand'] if reuse_demand else
            _write(directory/'demand.json', demand)),
        problem=_write(directory/'problem.json', problem_document(problem)),
        execution_plan=_write(directory/'execution-plan.json', execution))
    if verification is not None:
        files['schedule_verification'] = _write(directory/'schedule-verification.json',
            dict(schema='scar.semantic-region-schedule-verification', schema_version=1, **verification))
    metadata = _metadata(reference, logical_analysis=logical_analysis, max_seconds=max_seconds,
        files=files, execution=execution, problem=problem, cost_binding=cost_binding)
    preparation = _write(directory/'preparation.json', metadata)
    native = binding = None
    if compile_validation and (not study_only or cost_binding is not None):
        from scar.compiler.pipeline_fusion_delivery import build_fusion_blueprint
        artifact = build_fusion_blueprint(pipeline, blueprint, directory/'native')
        native = artifact_reference(artifact.directory/'artifact.json', artifact.report['schema'])
        binding = _write(directory/'native-binding.json',
            _native_binding(native, blueprint, *_native_subject(revision_ref, revision, demand,
                cost_binding), problem, execution, cost_binding=cost_binding, study_only=study_only))
    result = _build(preparation, report['decision'], revision_ref, native, binding,
        compile_validation, execution['status'])
    _write(directory/'result.json', result)
    return result


def check(result):
    reference = _reference(result, BUILD)
    build = read_artifact(reference)
    saved = read_artifact(build['preparation'])
    if ((build.get('schema'), build.get('schema_version')) != (BUILD, 2)
            or (saved.get('schema'), saved.get('schema_version')) != (SCHEMA, 2)
            or saved['producer_pins'] != _pins()):
        raise ValueError('semantic region planning version or producer differs')
    model = None if saved['forward_cost'] is None else saved['forward_cost']['model']
    pipeline, region, blueprint, report, revision_ref, revision, demand, problem, cost_binding = _derive(
        saved['macro_preparation'], logical_analysis=saved['logical_analysis'],
        model_reference=model, kept_revision=build['semantic_revision'])
    files = saved['files']
    study_only = report['decision'] == 'KEEP'
    if cost_binding is not None and not study_only and files['demand'] != cost_binding['candidate_demand']:
        raise ValueError('selected planning demand must reference the costed candidate demand')
    if read_artifact(files['demand']) != demand or read_artifact(files['problem']) != problem_document(problem):
        raise ValueError('successor demand or implementation problem differs from source replay')
    execution = read_artifact(files['execution_plan'])
    from scar.optimization.joint_planner import verify_execution_plan
    if execution['selected']:
        verification = verify_execution_plan(problem, execution)
        if read_artifact(files['schedule_verification']) != dict(
                schema='scar.semantic-region-schedule-verification', schema_version=1, **verification):
            raise ValueError('successor schedule verification differs')
    else:
        from scar.optimization.constructive_planner import construct_implementations
        expected = construct_implementations(problem, max_seconds=saved['max_seconds'])
        if ({k:v for k,v in execution.items() if k != 'elapsed_seconds'} !=
                {k:v for k,v in expected.items() if k != 'elapsed_seconds'}):
            raise ValueError('unresolved construction differs from the retained problem')
    if saved != _metadata(saved['macro_preparation'], logical_analysis=saved['logical_analysis'],
            max_seconds=saved['max_seconds'], files=files, execution=execution, problem=problem,
            cost_binding=cost_binding):
        raise ValueError('successor planning summary differs')
    if build != _build(build['preparation'], report['decision'], revision_ref,
            build['native_artifact'], build['native_binding'], build['compile_validation_requested'], execution['status']):
        raise ValueError('successor build lost its source selection or qualification')
    requested = build['compile_validation_requested']
    if (type(requested) is not bool or
            (build['native_artifact'] is not None) != (requested and (not study_only or cost_binding is not None))):
        raise ValueError('candidate exists only for an explicit source-qualified validation build')
    if build['native_artifact'] is not None:
        if read_artifact(build['native_binding']) != _native_binding(
                build['native_artifact'], blueprint, *_native_subject(revision_ref, revision, demand,
                    cost_binding), problem, execution, cost_binding=cost_binding, study_only=study_only):
            raise ValueError('native artifact differs from the actual semantic successor lowering')
    elif build['native_binding'] is not None:
        raise ValueError('native binding exists without an artifact')
    return build, pipeline, region, blueprint, report, revision, demand, problem, execution


def _module_binding(module, pipeline):
    name = pipeline.source.qualname
    if type(module) is not ModuleType or '.' in name:
        raise ValueError('direct source module function required for controlled validation')
    return name, vars(module)[name]


def _trial(original, build, region, blueprint, report, *, exclusive_inputs, allow_study=False):
    if build['native_artifact'] is None:
        raise ValueError('explicit validation trial requires a bound native artifact')
    from scar.codegen.fused_pipeline import load
    from scar.compiler.pipeline_fusion_delivery import (
        _guarded_fusion_entry, select_fusion_callable,
    )

    if build['decision'] != 'REWRITE' and not allow_study:
        raise ValueError('a cost-KEEP candidate may only run isolated validation')
    artifact = load(Path(build['native_artifact']['uri']).parent)
    if build['decision'] == 'REWRITE':
        return select_fusion_callable(original, report, artifact=artifact, trial=True,
            exclusive_inputs=exclusive_inputs, region_document=region, blueprint=blueprint)
    # The public call reaches here only after replaying source, candidate graph,
    # demand, cost and native binding. KEEP remains the active selection.
    return _guarded_fusion_entry(original, artifact, blueprint,
        exclusive_inputs=exclusive_inputs)


@contextmanager
def install(module, result, *, trial=False, exclusive_inputs=False):
    """Install an explicitly trialed REWRITE temporarily; KEEP preserves Python."""
    build, pipeline, region, blueprint, report, *_ = check(result)
    name, original = _module_binding(module, pipeline)
    if not trial or build['decision'] == 'KEEP':
        yield original
        return
    selected = _trial(original, build, region, blueprint, report, exclusive_inputs=exclusive_inputs)
    setattr(module, name, selected)
    try:
        yield selected
    finally:
        setattr(module, name, original)


def validate_call(module, result, args=(), kwargs=None, *, exclusive_inputs=False):
    """Compare one original call against its bound C++ candidate without installation."""
    if exclusive_inputs is not True:
        raise ValueError('controlled validation requires explicit exclusive inputs')
    import numpy as np

    reference = _reference(result, BUILD)
    build, pipeline, region, blueprint, report, *_ = check(reference)
    _, original = _module_binding(module, pipeline)
    kwargs = {} if kwargs is None else kwargs
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    values = [bound.arguments[name] for name in pipeline.parameters]
    if any(type(value) is not np.ndarray or value.dtype != np.float64
            or value.shape != tuple(blueprint['input_shapes'][name]) or not value.flags.c_contiguous
            for name, value in zip(pipeline.parameters, values)):
        raise ValueError('controlled validation requires the exact modeled ndarray inputs')
    selected = _trial(original, build, region, blueprint, report,
        exclusive_inputs=exclusive_inputs, allow_study=True)

    def input_digests():
        return tuple(sha256(memoryview(value)).digest() for value in values)

    before = input_digests()
    baseline = original(*args, **kwargs)
    if before != input_digests():
        raise ValueError('original call changed input bytes under its read-only contract')
    candidate = selected(*args, **kwargs)
    inputs_unchanged = before == input_digests()
    matching_layout = (type(baseline) is np.ndarray and type(candidate) is np.ndarray
        and baseline.shape == candidate.shape and baseline.dtype == candidate.dtype == np.float64
        and baseline.flags.c_contiguous and candidate.flags.c_contiguous)
    bitwise_equal = matching_layout and np.array_equal(baseline.view(np.uint8), candidate.view(np.uint8))
    fresh_output = (matching_layout and baseline is not candidate
        and not np.shares_memory(baseline, candidate)
        and all(output is not value and not np.shares_memory(output, value)
            for output in (baseline, candidate) for value in values))
    stats = selected.scar_stats
    return dict(schema='scar.semantic-region-call-validation', schema_version=1,
        build=reference, native_binding=build['native_binding'],
        decision=build['decision'], native_calls=stats['native_calls'],
        fallback_calls=stats['fallback_calls'], fallback_reasons=deepcopy(stats['reasons']),
        exact_output_bytes_match=bitwise_equal, inputs_unchanged=inputs_unchanged, fresh_output=fresh_output,
        validated=(bool(bitwise_equal and inputs_unchanged and fresh_output)
            and stats['native_calls'] == 1 and stats['fallback_calls'] == 0),
        scope='ONE_GUARDED_SOURCE_REGION_CALL_UNDER_EXPLICIT_EXCLUSIVE_INPUT_ASSUMPTION',
        performance_measured=False, deployment_authorized=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--macro-preparation'); parser.add_argument('--out')
    parser.add_argument('--forward-model', help='checked forward cost of this macro candidate')
    parser.add_argument('--logical-analysis', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=30)
    parser.add_argument('--compile-validation', action='store_true')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        build, *_ = check(args.check)
        result = dict(source_and_successor_replayed=True, native_binding_checked=build['native_compiled'],
            decision=build['decision'], planning_status=build['planning_status'],
            deployment_authorized=False)
    elif not args.macro_preparation or not args.out:
        parser.error('--macro-preparation and --out are required')
    else:
        result = prepare(args.macro_preparation, args.out, logical_analysis=args.logical_analysis,
            max_seconds=args.max_seconds, compile_validation=args.compile_validation,
            forward_model=args.forward_model)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
