"""Project selected source composites into the existing implementation model.

Known complete-entry time and unresolved physical resources are independent.
Neither an inherited CPU mask nor an internal stage binding is promoted to a
whole-entry reservation.  Optional logical analysis is explicitly outside the
physical execution domain and cannot authorize a lowering.
"""
from copy import deepcopy
from fractions import Fraction

from scar.ir.implementation_plan import (CostEstimate, ImplementationOption, PlanningOperation,
    PlanningProblem, PlanningResource, ResourceObligation, digest, validate_problem)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode, encode
from scar.ir.v2.optimization import OptimizationGraph
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.region-implementation-projection'
_PHYSICAL_FIELDS = frozenset(('whole_entry_cpu_execution_domain',
    'whole_entry_opaque_memory_and_residency'))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _refs(rows):
    result, seen = [], set()
    for row in rows:
        _need(type(row) is dict and bool(row), 'nonempty original source/reference records required')
        key = semantic_sha256(row)
        if key not in seen: seen.add(key); result.append(deepcopy(row))
    return tuple(result)


def _cost(value):
    return value if type(value) is CostEstimate else decode(CostEstimate, value)


def _obligations(rows):
    return tuple(row if type(row) is ResourceObligation else decode(ResourceObligation, row) for row in rows)


def _assemble(region_id, alternative_id, lowering, *, target, entry_cost, source_refs,
        metadata, payload_budget_bytes, payload_bytes, payload_scope, resource_obligations,
        logical_analysis):
    _need(type(logical_analysis) is bool, 'logical analysis must be explicitly selected')
    target = target if type(target) is ProviderArtifact else decode(ProviderArtifact, target)
    entry_cost = _cost(entry_cost)
    references = _refs((*source_refs, *entry_cost.source_refs,
        dict(schema=SCHEMA, region_id=region_id, alternative_id=alternative_id,
            source_projection_sha256=semantic_sha256(metadata))))
    unresolved = list(_obligations(resource_obligations))
    fields = {row.field for row in unresolved}
    for field, reason in (
        ('whole_entry_cpu_execution_domain',
            'No admitted whole-entry physical execution demand: inherited caller affinity and internal stage bindings are not whole-entry CPU reservations.'),
        ('whole_entry_opaque_memory_and_residency',
            'Formal payload bytes do not establish opaque runtime memory, original Python peak memory or physical memory residency.')):
        if field not in fields: unresolved.append(ResourceObligation(field, reason, references))
    resources, reservations = [], {}
    if payload_budget_bytes is not None or payload_bytes is not None:
        _need(type(payload_budget_bytes) is int and payload_budget_bytes >= 0
            and type(payload_bytes) is int and payload_bytes >= 0
            and type(payload_scope) is str and bool(payload_scope),
            'formal payload projection requires exact bytes, budget and original scope')
        if payload_budget_bytes == 0:
            if payload_bytes:
                unresolved.append(ResourceObligation('formal_payload_budget',
                    'Required formal payload exceeds the explicit zero-byte budget.', references))
        else:
            pool = 'formal:region-payload:' + semantic_sha256(dict(region=region_id, scope=payload_scope))
            resources.append(PlanningResource(pool, 'MEMORY', payload_budget_bytes,
                references + (dict(formal_budget_scope=payload_scope, physical_memory_location_proven=False,
                    external_inputs_not_materialized_here=True),)))
            reservations[pool] = payload_bytes
    demands = {}
    if logical_analysis:
        caller = 'model:logical-region-caller:' + semantic_sha256(region_id)
        resources.append(PlanningResource(caller, 'EXECUTION', 1, references + (
            dict(logical_serial_caller_order_only=True, physical_cpu_demand=False,
                cpu_mask_is_reservation=False),)))
        demands[caller] = 1
    active = tuple(row for row in unresolved if not logical_analysis or row.field not in _PHYSICAL_FIELDS)
    has_oir_selection = metadata.get('oir_schema') == OptimizationGraph.SCHEMA and 'selection' in metadata
    parameters = dict(schema=SCHEMA, schema_version=1, source_region=deepcopy(metadata),
        entry_cost_scope='COMPLETE_SELECTED_SOURCE_ENTRY_TO_RETURN' if has_oir_selection else
            'COMPLETE_SOURCE_ENTRY_TO_RETURN_NO_OIR_SELECTION',
        analysis_scope='LOGICAL_ENTRY_TIME_AND_FORMAL_PAYLOAD_ONLY' if logical_analysis else 'STRICT_PHYSICAL_ENTRY',
        cpu_policy='SOURCE_QUALIFIED_ONLY_NO_INFERRED_BINDING',
        physical_resource_obligations=encode(tuple(unresolved)),
        physical_feasibility_proven=False, physical_cpu_assignment_proven=False,
        logical_analysis=logical_analysis, lowering_authorized=False,
        formal_payload=dict(budget_bytes=payload_budget_bytes, required_bytes=payload_bytes,
            scope=payload_scope, physical_residency_proven=False),
        public_ports_are_not_materialization_or_slot_identities=True,
        new_source_or_backend_admission_performed=False)
    mode = ImplementationOption(alternative_id, 'LOGICAL_ENTRY_ANALYSIS_ONLY_NO_BACKEND'
        if logical_analysis else lowering, references, demands, {}, {}, {},
        entry_cost, parameters, active)
    operation = PlanningOperation(region_id, references, (), (mode,), alternative_id)
    return validate_problem(PlanningProblem(target, references, tuple(resources), (operation,), (),
        assumptions=('SELECTED_COMPOSITE_SOURCE_DELTA_PUBLIC_PORTS_AND_Q_RETAINED' if has_oir_selection else
            'EXISTING_STATEFUL_SOURCE_ENTRY_REFERENCE_NO_OIR_SELECTION_OR_DELTA',
            'WHOLE_ENTRY_TIME_NOT_LEAF_ABI_OR_INTERNAL_STAGE_TIME',
            'LOGICAL_ANALYSIS_CANNOT_AUTHORIZE_PHYSICAL_EXECUTION_OR_LOWERING' if logical_analysis else
            'UNRESOLVED_PHYSICAL_RESOURCE_OBLIGATIONS_PROHIBIT_MODE_SELECTION',
            'SOURCE_PUBLIC_PORTS_REQUIRE_SEPARATE_MATERIALIZATION_AND_LOCATION_QUALIFICATION'),
        reserved_memory=reservations))


def project_selected_region(graph, *, target, entry_cost, source_refs=(), metadata=None,
        payload_budget_bytes=None, payload_bytes=None, payload_scope=None,
        resource_obligations=(), logical_analysis=False):
    """Consume a selected existing OIR composite; do not retain its old CALLs.

    Current complete-entry adapters have unresolved physical demand.  They may
    preserve known timing without making up physical CPU or RAM units.  Public
    source ports remain provenance until actual materializations are admitted.
    """
    _need(type(graph) is OptimizationGraph, 'existing typed OptimizationGraph required')
    graph.assert_valid()
    _need(len(graph.plans) == 1, 'one selected complete source composite required')
    selection = next(iter(graph.plans.values()))
    _need(selection.selected in graph.alternatives
        and selection.disposition.value in ('KEEP', 'REWRITE'), 'selected OIR KEEP or REWRITE alternative required')
    alternative = graph.alternatives[selection.selected]
    region = graph.regions[selection.region]
    _need(alternative.region == region.id, 'source alternative belongs to another region')
    if alternative.transform.value == 'fuse':
        _need(set(alternative.delta.removed_definitions) == set(region.definitions)
            and set(alternative.delta.removed_instances) == set(region.instances)
            and len(alternative.delta.added_operations) == 1,
            'fusion projection must cover all original members with one composite')
    source = dict(oir_schema=graph.SCHEMA, oir_sha256=semantic_sha256(graph.to_dict()),
        region=region.as_dict(), alternative=alternative.as_dict(), selection=selection.as_dict(),
        public_ports=[graph.ports[name].as_dict() for name in region.ports],
        source_projection=deepcopy(metadata or {}), source_bodies_copied=False,
        runtime_logical_versions_created=0)
    return _assemble(region.id.wire, alternative.id.wire, 'SELECTED_OIR_COMPLETE_SOURCE_ENTRY',
        target=target, entry_cost=entry_cost, source_refs=source_refs, metadata=source,
        payload_budget_bytes=payload_budget_bytes, payload_bytes=payload_bytes,
        payload_scope=payload_scope, resource_obligations=resource_obligations,
        logical_analysis=logical_analysis)


def _pipeline_entry_cost(pass_report, original):
    plan = pass_report['profitability']; context = pass_report['cost_context']
    _need(context['boundary'] == 'WARM_SESSION'
        and context['cost_scope'] == 'PROTECTED_SOURCE_REGION_CALL_ENTRY_TO_RETURN_WARM'
        and context['measurement_contract']['full_protected_call_included'] is True
        and plan['request']['reuse_count'] == 1, 'complete warm single protected-entry cost required')
    name = ('original_per_call' if original else 'native_per_call') if plan['mode'] == 'COMPONENTS' else (
        'original_total' if original else 'native_total')
    source = plan['profile'].get('costs', {}).get(name)
    failures = [row for row in plan['frontier'] if not row['field'].startswith('costs.')
        or row['field'] == 'costs.' + name or row['field'].startswith('costs.' + name + '.')]
    refs = _refs((dict(schema='scar.region-complete-entry-cost-reference',
        macro_pass_sha256=semantic_sha256(pass_report), context_sha256=semantic_sha256(context),
        locator='#/profitability/profile/costs/' + name),
        *((source or {}).get('evidence', ()))))
    key = 'complete-region-entry:' + semantic_sha256(dict(context=context, component=name))
    if source is None or failures or name not in plan['cost_bases']:
        reasons = tuple(row['reason'] for row in failures) or ('NO_QUALIFIED_COMPLETE_ENTRY_COST',)
        return CostEstimate(None, None, None, 'UNKNOWN', key, refs, reasons)
    nanoseconds = Fraction(str(source['value'])) * 10**9
    if nanoseconds.denominator != 1:
        return CostEstimate(None, None, None, 'UNKNOWN', key, refs,
            ('COMPLETE_ENTRY_COST_NOT_EXACT_INTEGER_NANOSECONDS',))
    return CostEstimate(nanoseconds.numerator, None, None,
        'DECLARED_MODEL' if source['basis'] == 'DECLARED_MODEL' else 'EMPIRICAL_MODEL', key, refs)


def project_region_implementation(region_document, blueprint, pass_report, *, logical_analysis=False):
    """Replay the existing pipeline pass/blueprint and project its whole entry."""
    from scar.analysis.source_pipeline_graph import read_fusion_region
    from scar.optimization.macro_pass import _admit, verify_fusion_pass
    region, blueprint, _, _, _ = _admit(region_document, blueprint)
    report = verify_fusion_pass(pass_report, region, blueprint)
    graph = read_fusion_region(report['optimization_selection'])
    selection = next(iter(graph.plans.values()))
    original = graph.alternatives[selection.selected].original
    q = region['qualification']
    references = _refs((*region['source_refs'], dict(schema=region['schema'],
        semantic_sha256=semantic_sha256(region), locator='#'),
        dict(schema=blueprint['schema'], semantic_sha256=semantic_sha256(blueprint), locator='#'),
        dict(schema=report['schema'], semantic_sha256=semantic_sha256(report), locator='#')))
    obligations = () if not original else (ResourceObligation('whole_entry_opaque_memory_and_residency',
        'The native fresh-final payload budget is not an original Python/NumPy peak-memory proof.', references),)
    return project_selected_region(graph, target=blueprint['target'],
        entry_cost=_pipeline_entry_cost(report, original), source_refs=references,
        metadata=dict(original_region_document_sha256=semantic_sha256(region),
            source_projection=deepcopy(region['projection']), Q=deepcopy(q['Q']),
            contract=deepcopy(q['contract']), runtime_guards=deepcopy(q['required_runtime_guards']),
            source=deepcopy(q['source']), blueprint_execution=deepcopy(blueprint['execution']),
            complete_entry_context=deepcopy(report['cost_context']),
            prior_call_pool_invalidated=deepcopy(report['invalidation'])),
        payload_budget_bytes=None if original else blueprint['selected_memory_plan']['memory_budget_bytes'],
        payload_bytes=None if original else blueprint['selected_memory_plan']['final_bytes'],
        payload_scope=None if original else blueprint['selected_memory_plan']['budget_scope'],
        resource_obligations=obligations, logical_analysis=logical_analysis)


def verify_region_implementation(region_document, blueprint, pass_report, problem, *, logical_analysis=False):
    expected = project_region_implementation(region_document, blueprint, pass_report,
        logical_analysis=logical_analysis)
    _need(digest(validate_problem(problem)) == digest(expected),
        'region implementation problem differs from selected source/entry/resource replay')
    return expected


def project_stateful_entry_implementation(domain, candidate_id, entry_report, *, profiles=(),
        graph=None, logical_analysis=False):
    """Bridge existing Gray/stateful whole-entry cost, never an internal stage.

    An optional already-admitted OIR selection uses the common projection.
    Without it only the existing stateful source reference is projected;
    no v2 structural selection or synthetic OIR delta is claimed.
    Internal CPU bindings remain evidence only for their original stage scope.
    """
    from scar.optimization.stateful_entry_cost import verify_stateful_entry_cost
    report = verify_stateful_entry_cost(entry_report, domain, candidate_id, profiles)
    refs = _refs((*report['source_refs'], dict(schema=report['schema'],
        semantic_sha256=semantic_sha256(report), locator='#')))
    obligations = list(_obligations(report['unresolved_resources']))
    metadata = dict(stateful_region=deepcopy(report['region']), candidate_id=candidate_id,
        complete_entry_context=deepcopy(report['context']), entry_cost_query=deepcopy(report['query']),
        resource_qualification=deepcopy(report['resource_qualification']),
        candidate_qualification=deepcopy(report['candidate_qualification']),
        source_refs=deepcopy(report['source_refs']),
        Q=dict(guards=deepcopy(report['context']['guards']),
            external_assumptions=deepcopy(report['context']['external_assumptions'])),
        scope='ONE_ORIGINAL_METHOD_CALL_SITE_NOT_INTERNAL_STAGE_OR_SESSION_REPEAT',
        oir_selection_supplied=graph is not None,
        new_structural_rewrite_claimed=False,
        internal_stage_binding_is_whole_entry_binding=False)
    qualification = report['candidate_qualification']
    _need(qualification['legal_under_guards'] and not qualification['legality_frontier'],
        'stateful entry projection requires an already source/capability-admitted candidate')
    if graph is not None:
        return project_selected_region(graph, target=report['target'], entry_cost=report['cost'],
            source_refs=refs, metadata=metadata, resource_obligations=obligations,
            logical_analysis=logical_analysis)
    return _assemble('stateful-entry:' + report['context']['region_sha256'], candidate_id,
        'EXISTING_STATEFUL_WHOLE_ENTRY_SOURCE_REFERENCE', target=report['target'],
        entry_cost=report['cost'], source_refs=refs, metadata=metadata,
        payload_budget_bytes=None, payload_bytes=None, payload_scope=None,
        resource_obligations=obligations, logical_analysis=logical_analysis)


def verify_stateful_entry_implementation(domain, candidate_id, entry_report, problem, *,
        profiles=(), graph=None, logical_analysis=False):
    expected = project_stateful_entry_implementation(domain, candidate_id, entry_report,
        profiles=profiles, graph=graph, logical_analysis=logical_analysis)
    _need(digest(validate_problem(problem)) == digest(expected),
        'stateful implementation problem differs from whole-entry/source/resource replay')
    return expected


__all__ = ['SCHEMA', 'project_selected_region', 'project_region_implementation',
    'verify_region_implementation', 'project_stateful_entry_implementation',
    'verify_stateful_entry_implementation']
