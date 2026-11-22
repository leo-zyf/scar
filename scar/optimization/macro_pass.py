"""One guarded whole-pipeline fusion pass over the existing OIR and cost gate.

This is a metadata decision.  Source admission belongs to the parent producer;
native construction, live guards and deployment remain separate obligations.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import shutil
from typing import TYPE_CHECKING

from scar.ir.execution_recipe import artifact_reference
from scar.optimization.scope_accounting import _digest, plan_deployment, verify_deployment_plan
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN

if TYPE_CHECKING:
    from scar.compiler.pipeline_frontend import PipelinePlan
    from scar.ir.v2.optimization import OptimizationGraph, PlanAlternative


SCHEMA = 'scar.macro-pass-report'
BLUEPRINT_SCHEMA = 'scar.pipeline-fusion-blueprint'
COST_SCOPE = 'PROTECTED_SOURCE_REGION_CALL_ENTRY_TO_RETURN_WARM'


@dataclass(frozen=True, slots=True)
class PassSpec:
    name: str
    version: int
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    invalidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisFact:
    scope: str
    revision: str
    name: str
    qualification: str
    evidence: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class _AdmittedFusion:
    region: dict
    blueprint: dict
    pipeline: PipelinePlan
    graph: OptimizationGraph
    alternative: PlanAlternative | None
    legal: bool


FUSION_PASS = PassSpec('pipeline-full-fusion', 1,
    ('original_source', 'call_membership', 'public_ports', 'conditional_effect_Q',
     'fusion_legal', 'shape', 'final_budget', 'lowering_capability'),
    ('complete_region_OIR_selection', 'source_qualified_fusion_delta'),
    ('preserved_call_pool_assignment', 'preserved_call_liveness',
     'leaf_and_host_component_costs', 'execution_resource_schedule',
     'prior_lowering_and_native_identity'))

_ENTRY_SOURCES = (
    'compiler/pipeline_runtime.py',
    'codegen/fused_pipeline.py', 'codegen/cpp.py', 'codegen/kernel_v1.cpp.in',
    'codegen/scar_abi_v1.h', 'trace/code_snapshot.py', 'runtime/array_copy_guard.py',
)
_PRODUCERS = ('optimization/macro_pass.py', 'optimization/scope_accounting.py',
    'analysis/source_pipeline_graph.py', 'optimization/pipeline_strategy.py',
    'optimization/pipeline_fusion.py')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _document(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_bytes())
    _need(type(value) is dict, 'versioned metadata dictionary required')
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sources(names):
    root = Path(__file__).resolve().parents[1]
    return {name: artifact_reference(root / name, 'scar.backend-implementation-source')
            for name in names}


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
    return encode(reference), _document(graph.report['inventory']), sorted(cpus)


def _backend(plan):
    from scar.codegen import cpp, fused_pipeline
    command = shlex.split(os.environ.get('CXX', 'g++'))
    executable = shutil.which(command[0]) if command else None
    _need(executable is not None, 'original C++ compiler unavailable')
    source = cpp._source(plan, cpp._supported(plan)).encode()
    header = Path(cpp.__file__).with_name('scar_abi_v1.h').read_bytes()
    runtime = _source_receipt(Path(__file__).resolve().parents[1] / 'compiler/pipeline_runtime.py')
    return dict(family='EXISTING_CPP_ABI1_FULL_PIPELINE_FUSION',
        implementation_sha256=fused_pipeline.fused_implementation_sha256(),
        operator_runtime_implementation_sha256=cpp.runtime_implementation_sha256(),
        caller_guard_source=runtime, delivery_adapter_source=runtime,
        blueprint_producer_source=_source_receipt(__file__),
        compiler=dict(command=command, executable=str(Path(executable).resolve()),
            executable_sha256=sha256(Path(executable).read_bytes()).hexdigest(),
            options=list(cpp.COMPILER_OPTIONS), link_libraries=['m']),
        generated_source_sha256=sha256(source).hexdigest(),
        abi_header_sha256=sha256(header).hexdigest())


def derive_fusion_blueprint(pipeline, memory, target, *, available_cpu_ids, source_refs=()):
    """Reconstruct one conditional unbound fusion; do not compile or load it."""
    from scar.compiler.model import document as plan_document
    from scar.compiler.pipeline_frontend import document, pipeline_digest, validate_pipeline
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
    return _document(dict(schema=BLUEPRINT_SCHEMA, schema_version=1,
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
    value = _document(blueprint)
    _need((value.get('schema'), value.get('schema_version')) == (BLUEPRINT_SCHEMA, 1),
        'versioned fusion blueprint required')
    expected = derive_fusion_blueprint(pipeline, memory, target,
        available_cpu_ids=available_cpu_ids, source_refs=value['source_refs'])
    _need(value == expected, 'fusion blueprint differs from complete original lowering replay')
    return expected


def replay_fusion_blueprint(pipeline, blueprint):
    return verify_fusion_blueprint(pipeline, blueprint['original_memory_plan'],
        blueprint['target'], blueprint, available_cpu_ids=blueprint['execution']['allowed_cpu_ids'])


def _admit(region_document, blueprint):
    from scar.analysis.source_pipeline_graph import read_fusion_region
    from scar.compiler.pipeline_frontend import pipeline_from_dict, pipeline_digest

    region, blueprint = _document(region_document), _document(blueprint)
    graph = read_fusion_region(region)
    pipeline = pipeline_from_dict(blueprint['pipeline'])
    blueprint = replay_fusion_blueprint(pipeline, blueprint)
    strategy = blueprint['strategy']
    _need(_digest(region['strategy']) == blueprint['strategy_sha256'],
        'pipeline strategy differs from complete representation/source/shape replay')
    q = region['qualification']; memory = blueprint['selected_memory_plan']
    _need(q['pipeline_sha256'] == pipeline_digest(pipeline) == blueprint['pipeline_sha256']
        and q['memory_plan_sha256'] == _digest(blueprint['original_memory_plan'])
        and q['source'] == blueprint['source'],
        'region and lowering blueprint differ from the same original pipeline/strategy/pool')
    _need(q['input_shapes'] == memory['input_shapes']
        and q['output_shape'] == memory['output_shape']
        and q['input_dtype'] == 'float64' and q['input_layout'] == 'C_CONTIGUOUS'
        and q['memory_budget_bytes'] == memory['memory_budget_bytes']
        and q['memory_budget_scope'] == memory['budget_scope'],
        'original source shape/layout/budget obligations differ from the fusion blueprint')
    contract = q['Q']
    _need(all(contract.get(name) is True for name in (
        'full_original_source_required', 'readonly_stable_inputs',
        'original_fp_status_and_fallback_required', 'operand_order_preserved',
        'fresh_output_required', 'no_input_writes'))
        and contract['conditional_when'] == q['required_runtime_guards']
        and q['normal_completion_only'] is True
        and q['runtime_guards_discharged'] is False,
        'explicit guarded normal-completion source Q must remain conditional')
    execution = blueprint['execution']; cpus = execution['allowed_cpu_ids']
    _need(execution['cpu_policy'] == 'INHERIT_CALLER_CPU'
        and execution['required_cpu_ids'] == [] and execution['cpu_threads'] == 1
        and execution['resource_binding'] is None
        and execution['cpu_domain_is_exclusive_reservation'] is False
        and type(cpus) is list and cpus and cpus == sorted(set(cpus))
        and all(type(cpu) is int and cpu >= 0 for cpu in cpus),
        'one serial inherited-caller lowering and its exact nonexclusive CPU cost domain required')
    originals = [alternative for alternative in graph.alternatives.values() if alternative.original]
    _need(len(originals) == 1 and originals[0].delta.is_empty,
        'one real unchanged original Python alternative required')
    fusion_id = region['alternative_ids'].get('FULL_FUSION')
    fusion = next((a for a in graph.alternatives.values() if a.id.wire == fusion_id), None)
    legal = (fusion is not None and bool(fusion.proofs)
        and all(proof.status.value == 'PROVEN' for proof in fusion.proofs)
        and strategy['selected_strategy'] == 'FULL_FUSION'
        and not strategy['fusion_frontier']
        and memory['final_bytes'] <= q['memory_budget_bytes'])
    return _AdmittedFusion(region, blueprint, pipeline, graph, fusion, legal)


def _cost_context(region, blueprint, graph):
    q = region['qualification']; memory = blueprint['selected_memory_plan']
    entries = _sources(_ENTRY_SOURCES)
    guards = list(dict.fromkeys([*q['required_runtime_guards'],
        'CALLER_AFFINITY_EQUALS_COST_DOMAIN_EACH_CALL',
        'FULL_SOURCE_AND_NATIVE_IDENTITY_REVALIDATED_BEFORE_EXECUTION']))
    return dict(schema='scar.fusion-pass-cost-context', schema_version=1,
        boundary='WARM_SESSION', cost_scope=COST_SCOPE,
        workload_id=q['pipeline_sha256'], artifact_id=_digest(blueprint),
        region_document_sha256=_digest(region), region_id=deepcopy(region['region_id']),
        alternative_ids=deepcopy(region['alternative_ids']),
        source=deepcopy(q['source']), source_parents=deepcopy(region['projection']['source_parents']),
        public_ports=[port.as_dict() for port in graph.ports.values()],
        Q=deepcopy(q['Q']), contract=deepcopy(q['contract']),
        input_shapes=deepcopy(q['input_shapes']), output_shape=deepcopy(q['output_shape']),
        input_dtype=q['input_dtype'], input_layout=q['input_layout'],
        target=deepcopy(blueprint['target']), blueprint_sha256=_digest(blueprint),
        guarded_entry_implementation_sources=entries,
        guarded_entry_implementation_sha256=_digest(entries),
        original_entry=dict(kind='ORIGINAL_PYTHON_NO_OP', source=deepcopy(q['source'])),
        candidate_entry=dict(kind='FULL_FUSION', implementation='EXISTING_GUARDED_PIPELINE_CALL',
            backend=deepcopy(blueprint['backend'])),
        cpu_ids=deepcopy(blueprint['execution']['allowed_cpu_ids']),
        cpu_policy='INHERIT_CALLER_CPU', cpu_domain_is_exclusive_reservation=False,
        cpu_domain_guard='CALLER_AFFINITY_EQUALS_COST_DOMAIN_EACH_CALL',
        required_memory_bytes=memory['final_bytes'], memory_scope=memory['budget_scope'],
        memory_accounting=dict(required_payload='ONE_FRESH_FINAL_ONLY', workspace_bytes=0,
            external_input_payload_charged=False, opaque_memory_demand='UNKNOWN',
            process_RSS_proven=False),
        external_assumptions=['NO_FOREIGN_CONCURRENT_INPUT_WRITERS', 'SYNCHRONOUS_CPU_EXECUTION'],
        runtime_guards=guards,
        measurement_contract=dict(reuse_count=1, compilation_included=False,
            native_load_included=False, full_protected_call_included=True,
            leaf_ABI_or_unguarded_kernel_scope_is_not_this_boundary=True))


def fusion_pass_cost_context(region_document, blueprint):
    """Exact warm protected-call context, without loading or executing native code."""
    admitted = _admit(region_document, blueprint)
    return _cost_context(admitted.region, admitted.blueprint, admitted.graph)


def run_fusion_pass(region_document, blueprint, *, cost_profile=None,
                    available_cpu_ids, memory_budget_bytes, minimum_gain=DEFAULT_MINIMUM_GAIN,
                    strict_minimum_gain=True):
    """Replay one fusion's legality and reuse the existing exact deployment gate."""
    return _run_admitted_fusion_pass(_admit(region_document, blueprint), cost_profile=cost_profile,
        available_cpu_ids=available_cpu_ids, memory_budget_bytes=memory_budget_bytes,
        minimum_gain=minimum_gain, strict_minimum_gain=strict_minimum_gain)


def _run_admitted_fusion_pass(admitted, *, cost_profile, available_cpu_ids, memory_budget_bytes,
                              minimum_gain=DEFAULT_MINIMUM_GAIN, strict_minimum_gain=True):
    from scar.analysis.source_pipeline_graph import select_fusion_region

    region, blueprint = admitted.region, admitted.blueprint
    fusion, legal = admitted.alternative, admitted.legal
    context = _cost_context(region, blueprint, admitted.graph)
    profitability = plan_deployment({} if cost_profile is None else cost_profile,
        context=context, reuse_count=1, available_cpu_ids=available_cpu_ids,
        memory_budget_bytes=memory_budget_bytes, minimum_gain=minimum_gain,
        strict_minimum_gain=strict_minimum_gain)
    verify_deployment_plan(profitability)
    rewrite = legal and profitability['decision'] == 'LOAD_NATIVE_CANDIDATE'
    reason = ('Existing fusion is legal under the retained source Q and meets the protected-call cost threshold.'
        if rewrite else 'Keep original Python: ' + ('fusion legality is unresolved.' if not legal
        else '; '.join(profitability['reasons'])))
    selection = select_fusion_region(region, rewrite=rewrite, reason=reason)
    source_revision = _digest(region)
    evidence = dict(region_document_sha256=source_revision, blueprint_sha256=_digest(blueprint))
    locators = dict(original_source='#/qualification/source', call_membership='#/projection/operations',
        public_ports='#/projection/boundary_ports', conditional_effect_Q='#/qualification/Q',
        fusion_legal='#/strategy', shape='#/qualification/input_shapes',
        final_budget='#/strategy/selected_memory_plan', lowering_capability='BLUEPRINT_STATIC_REPLAY')
    requires = [asdict(AnalysisFact(region['region_id']['wire'], source_revision, name,
        'CONDITIONAL_SOURCE_AND_STATIC_MODEL_REPLAY' if name != 'fusion_legal' or legal else 'UNRESOLVED',
        ({**evidence, 'locator': locators[name]},))) for name in FUSION_PASS.requires]
    delta = deepcopy(fusion.delta.as_dict()) if rewrite else None
    optimized_revision = (_digest(dict(source_revision=source_revision,
        alternative_id=fusion.id.wire, delta=delta)) if rewrite else source_revision)
    invalidation = [asdict(AnalysisFact(region['region_id']['wire'], source_revision, name,
        'INVALIDATED_BY_FUSION_DELTA', (dict(optimized_revision=optimized_revision),)))
        for name in FUSION_PASS.invalidates] if rewrite else []
    request = dict(available_cpu_ids=profitability['request']['available_cpu_ids'],
                   memory_budget_bytes=memory_budget_bytes, minimum_gain=minimum_gain)
    if strict_minimum_gain:
        request['strict_minimum_gain'] = True
    return dict(schema=SCHEMA, schema_version=1, decision='REWRITE' if rewrite else 'KEEP',
        region_document_sha256=source_revision, blueprint_sha256=_digest(blueprint),
        spec=asdict(FUSION_PASS), requires=requires,
        legality=dict(qualified=legal, runtime_guards_discharged=False,
            original_python_always_available=True, original_F4_replayed_here=False,
            source_admission='QUALIFIED_PARENT_F4_REPLAY_REQUIRED',
            full_strategy_and_static_lowering_replayed=True,
            frontier=[] if legal else ['FULL_FUSION_LEGALITY_NOT_QUALIFIED']),
        cost_context=context, profitability=profitability, optimization_selection=selection,
        revision=dict(source_revision=source_revision, optimized_revision=optimized_revision,
            source_graph_unchanged=True, delta_summary=delta,
            optimized_revision_is_delta_summary=rewrite, new_full_graph_emitted=False),
        invalidation=invalidation,
        sourcepins=dict(original_source=deepcopy(region['qualification']['source']),
            source_parents=deepcopy(region['projection']['source_parents']),
            source_refs=deepcopy(region['source_refs']), target=deepcopy(blueprint['target']),
            implementations=_sources((*_PRODUCERS, *_ENTRY_SOURCES))),
        request=request,
        deployment_authorized=False, target_program_executed=False,
        native_artifact_built=False, physical_optimality_proven=False)


def replay_fusion_pass(report, region_document, blueprint):
    """Verify the pass and return its admitted material for the consuming stage."""
    report = _document(report)
    _need((report.get('schema'), report.get('schema_version')) == (SCHEMA, 1),
        'versioned single fusion pass report required')
    plan = verify_deployment_plan(report['profitability'])
    source = plan['profile_source']
    profile = (source['path'] if source['kind'] == 'PATH' else
        source['reference'] if source['kind'] == 'FACT_REFERENCE' else plan['profile'])
    request = dict(report['request'])
    request.setdefault('strict_minimum_gain', False)
    admitted = _admit(region_document, blueprint)
    expected = _run_admitted_fusion_pass(admitted, cost_profile=profile, **request)
    _need(_digest(expected) == _digest(report), 'fusion pass differs from exact legality/cost/selection replay')
    return admitted, expected


def verify_fusion_pass(report, region_document, blueprint):
    """Replay the source/model correspondence and original cost evidence exactly."""
    return replay_fusion_pass(report, region_document, blueprint)[1]


__all__ = ['SCHEMA', 'COST_SCOPE', 'PassSpec', 'AnalysisFact', 'FUSION_PASS',
    'BLUEPRINT_SCHEMA', 'derive_fusion_blueprint', 'verify_fusion_blueprint', 'replay_fusion_blueprint',
    'fusion_pass_cost_context', 'run_fusion_pass', 'verify_fusion_pass', 'replay_fusion_pass']
