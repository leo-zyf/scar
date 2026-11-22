"""One guarded whole-pipeline fusion pass over the existing OIR and cost gate.

This is a metadata decision.  Source admission belongs to the parent producer;
native construction, live guards and deployment remain separate obligations.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference
from scar.optimization.scope_accounting import _digest, plan_deployment, verify_deployment_plan
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN


SCHEMA = 'scar.macro-pass-report'
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


FUSION_PASS = PassSpec('pipeline-full-fusion', 1,
    ('original_source', 'call_membership', 'public_ports', 'conditional_effect_Q',
     'fusion_legal', 'shape', 'final_budget', 'lowering_capability'),
    ('complete_region_OIR_selection', 'source_qualified_fusion_delta'),
    ('preserved_call_pool_assignment', 'preserved_call_liveness',
     'leaf_and_host_component_costs', 'execution_resource_schedule',
     'prior_lowering_and_native_identity'))

_ENTRY_SOURCES = (
    'compiler/pipeline_runtime.py', 'compiler/pipeline_runtime.py',
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


def _admit(region_document, blueprint):
    from scar.analysis.source_pipeline_graph import read_fusion_region
    from scar.compiler.pipeline_frontend import pipeline_from_dict, pipeline_digest
    from scar.compiler.pipeline_runtime import verify_fusion_blueprint
    from scar.optimization.pipeline_strategy import verify_pipeline_strategy

    region, blueprint = _document(region_document), _document(blueprint)
    graph = read_fusion_region(region)
    pipeline = pipeline_from_dict(blueprint['pipeline'])
    blueprint = verify_fusion_blueprint(pipeline, blueprint['original_memory_plan'],
        blueprint['target'], blueprint,
        available_cpu_ids=blueprint['execution']['allowed_cpu_ids'])
    strategy = verify_pipeline_strategy(pipeline, region['strategy'])
    q = region['qualification']; memory = blueprint['selected_memory_plan']
    _need(q['pipeline_sha256'] == pipeline_digest(pipeline) == blueprint['pipeline_sha256']
        and q['memory_plan_sha256'] == _digest(blueprint['original_memory_plan'])
        and q['source'] == blueprint['source']
        and strategy == blueprint['strategy'],
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
    return region, blueprint, graph, fusion, legal


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
    region, blueprint, graph, _, _ = _admit(region_document, blueprint)
    return _cost_context(region, blueprint, graph)


def run_fusion_pass(region_document, blueprint, *, cost_profile=None,
                    available_cpu_ids, memory_budget_bytes, minimum_gain=DEFAULT_MINIMUM_GAIN,
                    strict_minimum_gain=True):
    """Replay one fusion's legality and reuse the existing exact deployment gate."""
    from scar.analysis.source_pipeline_graph import select_fusion_region

    region, blueprint, graph, fusion, legal = _admit(region_document, blueprint)
    context = _cost_context(region, blueprint, graph)
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


def verify_fusion_pass(report, region_document, blueprint):
    """Replay the source/model correspondence and original cost evidence exactly."""
    report = _document(report)
    _need((report.get('schema'), report.get('schema_version')) == (SCHEMA, 1),
        'versioned single fusion pass report required')
    plan = verify_deployment_plan(report['profitability'])
    source = plan['profile_source']
    profile = (source['path'] if source['kind'] == 'PATH' else
        source['reference'] if source['kind'] == 'FACT_REFERENCE' else plan['profile'])
    request = dict(report['request'])
    request.setdefault('strict_minimum_gain', False)
    expected = run_fusion_pass(region_document, blueprint, cost_profile=profile, **request)
    _need(_digest(expected) == _digest(report), 'fusion pass differs from exact legality/cost/selection replay')
    return expected


__all__ = ['SCHEMA', 'COST_SCOPE', 'PassSpec', 'AnalysisFact', 'FUSION_PASS',
    'fusion_pass_cost_context', 'run_fusion_pass', 'verify_fusion_pass']
