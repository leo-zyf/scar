"""Deliver the existing full-fusion lowering under its original caller guards.

Blueprint derivation/replay emits no native code and runs no workload. Target
CPU IDs describe the cost applicability domain, never a pin or reservation.
KEEP returns the original callable before importing any compiler backend.
"""
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import shutil

SCHEMA = 'scar.pipeline-fusion-blueprint'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _source_receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest())


def _target_domain(target, available_cpu_ids):
    from scar.ir.execution_recipe import read_artifact
    from scar.ir.program import target_hardware as H
    from scar.ir.record_codec import encode
    from scar.optimization.hardware import _target_materials
    if isinstance(target, dict):
        _need(target.get('schema') == 'scar.target-hardware-graph',
            'retained target hardware graph reference required')
        read_artifact(target)
        target = Path(target['uri']).parent
    elif hasattr(target, 'uri'):
        return _target_domain(encode(target), available_cpu_ids)
    elif not hasattr(target, 'path'):
        target = Path(target).resolve()
        if target.is_file():
            target = target.parent
    graph, reference, resources, _ = _target_materials(target)
    _need(type(available_cpu_ids) in (tuple, list) and bool(available_cpu_ids),
        'explicit nonempty cost-applicable CPU domain required')
    cpus = list(available_cpu_ids)
    _need(all(type(cpu) is int and cpu >= 0 for cpu in cpus)
        and len(set(cpus)) == len(cpus), 'distinct nonnegative CPU IDs required')
    threads = {row.cpu_id:row for row in resources if type(row) is H.HardwareThread}
    _need(all(cpu in threads and threads[cpu].online is not False for cpu in cpus),
        'cost-applicable CPU absent or offline in retained target inventory')
    return encode(reference), _copy(graph.report['inventory']), sorted(cpus)


def _backend(plan):
    from scar.codegen import cpp, fused_pipeline
    from scar.compiler import pipeline_runtime
    command = shlex.split(os.environ.get('CXX', 'g++'))
    executable = shutil.which(command[0]) if command else None
    _need(executable is not None, 'original C++ compiler unavailable')
    source = cpp._source(plan, cpp._supported(plan)).encode()
    header = Path(cpp.__file__).with_name('scar_abi_v1.h').read_bytes()
    return dict(family='EXISTING_CPP_ABI1_FULL_PIPELINE_FUSION',
        implementation_sha256=fused_pipeline.fused_implementation_sha256(),
        operator_runtime_implementation_sha256=cpp.runtime_implementation_sha256(),
        caller_guard_source=_source_receipt(pipeline_runtime.__file__),
        delivery_adapter_source=_source_receipt(__file__),
        compiler=dict(command=command, executable=str(Path(executable).resolve()),
            executable_sha256=sha256(Path(executable).read_bytes()).hexdigest(),
            options=list(cpp.COMPILER_OPTIONS), link_libraries=['m']),
        generated_source_sha256=sha256(source).hexdigest(),
        abi_header_sha256=sha256(header).hexdigest())


def derive_fusion_blueprint(pipeline, memory, target, *, available_cpu_ids, source_refs=()):
    """Reconstruct one conditional unbound fusion; do not compile or load it."""
    from scar.compiler.model import document as plan_document
    from scar.compiler.pipeline_model import document, pipeline_digest, validate_pipeline
    from scar.ir.record_codec import encode
    from scar.optimization.pipeline_memory import verify_pipeline_memory
    from scar.optimization.pipeline_strategy import plan_pipeline_strategy, fused_execution_plan
    pipeline = validate_pipeline(pipeline)
    memory = verify_pipeline_memory(pipeline, memory)
    _need(type(source_refs) in (tuple, list)
        and all(type(row) is dict and row for row in source_refs),
        'original source reference records required')
    target_ref, inventory, cpus = _target_domain(target, available_cpu_ids)
    strategy = plan_pipeline_strategy(pipeline, input_shapes=memory['input_shapes'],
        memory_budget_bytes=memory['memory_budget_bytes'])
    _need(strategy['selected_strategy'] == 'FULL_FUSION'
        and strategy['fusion_frontier'] == [], 'complete supported fusion strategy required')
    plan = fused_execution_plan(pipeline, strategy)
    _need(plan.resource_binding is None and plan.cpu_threads == 1,
        'fusion delivery must inherit the caller CPU without workers or pinning')
    selected_memory = strategy['selected_memory_plan']
    backend = _backend(plan)
    return _copy(dict(schema=SCHEMA, schema_version=1,
        pipeline=document(pipeline), pipeline_sha256=pipeline_digest(pipeline),
        source=asdict(pipeline.source), source_refs=encode(source_refs),
        original_memory_plan=memory, original_memory_plan_sha256=_digest(memory),
        target=target_ref, target_inventory=inventory, available_cpu_ids=cpus,
        strategy=strategy, strategy_sha256=_digest(strategy),
        optimized_plan=plan_document(plan), optimized_plan_sha256=_digest(plan_document(plan)),
        selected_memory_plan=selected_memory, memory_plan_sha256=_digest(selected_memory),
        input_shapes=selected_memory['input_shapes'], backend=backend,
        execution=dict(cpu_policy='INHERIT_CALLER_CPU', cpu_threads=1, resource_binding=None,
            required_cpu_ids=[], allowed_cpu_ids=cpus, creates_workers=False,
            cpu_domain_is_exclusive_reservation=False,
            affinity_guard='EXACT_CALLER_AFFINITY_EQUALS_COST_APPLICABILITY_DOMAIN'),
        qualification=dict(native_lowering_applicable=True,
            semantic_scope='NORMAL_COMPLETION_UNDER_ORIGINAL_GUARDED_ARRAY_CONTRACT',
            memory_budget_scope=selected_memory['budget_scope'],
            native_payload_scope='INCREMENTAL_PAYLOAD_FRESH_FINAL_ONLY',
            budget_excludes=selected_memory['budget_excludes'],
            total_workspace_bytes=0, final_bytes=selected_memory['final_bytes'],
            original_contract=dict(pipeline_assumptions=list(pipeline.assumptions),
                leaf_assumptions={call.id:list(call.plan.region.assumptions) for call in pipeline.calls},
                fused_assumptions=list(plan.region.assumptions),
                source_fusion_semantics=strategy['fusion_certificate']['semantics'],
                optimized_proof=plan.proof),
            runtime_guards_executed=False,
            source_admission='QUALIFIED_PARENT_SOURCE_PRODUCER_REPLAY_REQUIRED',
            external_assumptions=['SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS']),
        source_replayed_by_parent_required=True, target_program_executed=False,
        performance_measured=False, runtime_improvement_proven=False,
        deployment_authorized=False, automatic_caller_installation_supported=False,
        default_backend='NO_OP'))


def verify_fusion_blueprint(pipeline, memory, target, blueprint, *, available_cpu_ids):
    """Recompute the blueprint without CDLL, native compilation or workload execution."""
    value = _copy(blueprint)
    _need((value.get('schema'), value.get('schema_version')) == (SCHEMA, 1),
        'versioned fusion blueprint required')
    expected = derive_fusion_blueprint(pipeline, memory, target,
        available_cpu_ids=available_cpu_ids, source_refs=value['source_refs'])
    _need(value == expected, 'fusion blueprint differs from complete original lowering replay')
    return expected


def _replay_embedded(pipeline, blueprint):
    return verify_fusion_blueprint(pipeline, blueprint['original_memory_plan'],
        blueprint['target'], blueprint, available_cpu_ids=blueprint['execution']['allowed_cpu_ids'])


def _admit_artifact(artifact, blueprint):
    from scar.codegen import cpp, fused_pipeline
    from scar.compiler.model import document as plan_document
    from scar.compiler.pipeline_model import document, pipeline_digest
    _need(type(artifact) is fused_pipeline.FusedPipelineArtifact,
        'actual existing fused pipeline artifact required')
    _need(_copy(document(artifact.pipeline)) == blueprint['pipeline']
        and _copy(artifact._strategy) == blueprint['strategy']
        and artifact.memory_plan == blueprint['selected_memory_plan'],
        'native artifact source, fusion strategy or memory specialization differs')
    operator = artifact._operator
    backend = blueprint['backend']
    report = operator.report
    _need(_copy(plan_document(operator.plan)) == blueprint['optimized_plan']
        and operator.plan.resource_binding is None and operator.plan.cpu_threads == 1,
        'actual native body differs from the unbound fused blueprint')
    _need(report['runtime_implementation_sha256'] == backend['operator_runtime_implementation_sha256']
        and report['compiler'] == backend['compiler']['command']
        and report['compiler_options'] == backend['compiler']['options']
        and report['sha256']['source'] == backend['generated_source_sha256']
        and report['sha256']['abi_header'] == backend['abi_header_sha256']
        and sha256(operator.source_path.read_bytes()).hexdigest() == backend['generated_source_sha256']
        and sha256(operator.abi_header_path.read_bytes()).hexdigest() == backend['abi_header_sha256'],
        'actual native compiler, guards, C++ body or ABI header differs')
    binding = fused_pipeline._operator_receipt(operator)
    expected = fused_pipeline._manifest(artifact.pipeline, blueprint['strategy'],
        artifact.memory_plan, operator, binding, backend['implementation_sha256'], artifact.directory)
    identity = artifact.report
    # report adds convenience paths to the on-disk manifest.
    for name in ('artifact_directory', 'metadata_path', 'source_path', 'library_path'):
        identity.pop(name, None)
    _need(identity == expected and identity['pipeline_sha256'] == pipeline_digest(artifact.pipeline)
        and identity['strategy_sha256'] == blueprint['strategy_sha256']
        and identity['memory_plan_sha256'] == blueprint['memory_plan_sha256'],
        'actual fused artifact identity differs from the selected blueprint')
    _need(report['plan_sha256'] == cpp._plan_digest(operator.plan), 'native math plan identity differs')
    return artifact


def build_fusion_blueprint(pipeline, blueprint, out):
    """Compile the replayed existing operator and fused adapter; never execute it."""
    from scar.codegen import cpp, fused_pipeline
    from scar.compiler.model import plan_from_dict
    value = _replay_embedded(pipeline, blueprint)
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh fusion delivery output directory required')
    directory.mkdir(parents=True)
    operator = cpp.build(plan_from_dict(value['optimized_plan']), directory/'operator')
    artifact = fused_pipeline.build(pipeline, directory/'artifact',
        strategy=value['strategy'], operator=operator)
    _admit_artifact(artifact, value)
    (directory/'blueprint.json').write_text(json.dumps(value, sort_keys=True,
        indent=2, allow_nan=False)+'\n')
    return artifact


def select_fusion_callable(original, pass_report, *, artifact=None, trial=False,
        exclusive_inputs=False, region_document=None, blueprint=None):
    """Keep Python unchanged, or explicitly trial the selected guarded artifact."""
    if type(pass_report) is dict and pass_report.get('decision') == 'KEEP':
        return original
    _need(type(pass_report) is dict and pass_report.get('decision') == 'REWRITE',
        'replayed fusion pass decision required')
    _need(trial is True, 'native fusion delivery requires an explicit trial')
    _need(region_document is not None and blueprint is not None and artifact is not None,
        'original region, fusion blueprint and actual artifact required for trial delivery')
    from scar.optimization.macro_pass import verify_fusion_pass
    from scar.compiler.pipeline_runtime import guarded_pipeline_function
    verify_fusion_pass(pass_report, region_document, blueprint)
    _replay_embedded(artifact.pipeline, blueprint)
    _admit_artifact(artifact, blueprint)
    guarded = guarded_pipeline_function(original, artifact,
        exclusive_inputs=exclusive_inputs, trial=True)
    allowed = frozenset(blueprint['execution']['allowed_cpu_ids'])

    @wraps(original)
    def run(*args, **kwargs):
        try:
            applicable = hasattr(os, 'sched_getaffinity') and os.sched_getaffinity(0) == allowed
        except OSError:
            applicable = False
        if not applicable:
            stats = guarded.scar_stats
            stats['fallback_calls'] += 1
            reason = 'caller_cpu_domain_differs_from_cost_context'
            stats['reasons'][reason] = stats['reasons'].get(reason, 0)+1
            return original(*args, **kwargs)
        return guarded(*args, **kwargs)

    run.scar_stats = guarded.scar_stats
    run.scar_artifact = artifact
    return run


__all__ = ['SCHEMA', 'derive_fusion_blueprint', 'verify_fusion_blueprint',
    'build_fusion_blueprint', 'select_fusion_callable']
