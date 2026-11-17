"""Adapt admitted whole-stateful-call profiles to the common cost record.

Stage timings and repeated-session totals are not entry costs.  A cost known
under the original profile contract does not establish whole-entry physical
CPU demands, memory residency, source admission or deployment permission.
"""
from copy import deepcopy
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import encode
from scar.optimization.deployment_cost import _digest
from scar.optimization.implementation_costs import CostQuery, query_key
from scar.optimization.stateful_cpu_choices import (
    _cost, _domain, stateful_cpu_cost_context,
)


SCHEMA = 'scar.stateful-entry-cost'
SCOPE = 'ONE_ORIGINAL_METHOD_CALL_SITE'
_DOMAIN_SCHEMA = 'scar.stateful-cpu-domain'


def _refs(rows):
    result, seen = [], set()
    for row in rows:
        key = _digest(row)
        if key not in seen:
            seen.add(key); result.append(deepcopy(row))
    return result


def _domain_material(value):
    reference = None
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        if path.is_dir():
            path = path/'domain.json'
        reference = artifact_reference(path, _DOMAIN_SCHEMA)
        value = read_artifact(reference)
    return _domain(value), reference


def derive_stateful_entry_cost(domain, candidate_id, profiles=()):
    """Replay existing whole-call cost admission without timing or loading code.

    Profiles may contain entries for other candidates; those remain rejected
    offers.  Multiple distinct applicable offers are ambiguous, never searched
    for the fastest estimate.  The shared stateful domain reader may query the
    compiler identity, but this adapter never compiles or invokes target code.
    """
    domain, domain_reference = _domain_material(domain)
    context = stateful_cpu_cost_context(domain, candidate_id)
    scope = context['scope']
    if (scope.get('boundary') != 'WARM_SESSION' or scope.get('call_scope') != SCOPE
            or type(scope.get('reuse_count')) is not int or scope['reuse_count'] != 1
            or scope.get('includes') != 'ALL_IMPLEMENTATION_WORK_FROM_CALL_DISPATCH_THROUGH_RETURN'):
        raise ValueError('exact warm single whole-method entry scope required')
    if type(profiles) not in (tuple, list):
        raise ValueError('explicit list or tuple of whole-stateful-call profiles required')
    base = read_artifact(domain['base_stage_plan'])
    candidate = next((row for row in domain['candidates'] if row['id'] == candidate_id), None)
    sources = [dict(artifact=domain[name], locator='#') for name in
        ('region', 'base_stage_plan', 'target_snapshot', 'inventory')]
    if domain_reference is not None:
        sources.append(dict(artifact=domain_reference, locator='#'))
    sources.append(dict(domain_id=domain['domain_id'], candidate_id=candidate_id,
        pointer='#/original_no_op' if candidate is None else
            '#/candidates/' + str(domain['candidates'].index(candidate))))
    sources = _refs(sources)
    query = CostQuery(SCOPE, context, tuple(sources))
    offers, accepted = [], {}
    for index, profile in enumerate(profiles):
        cost, issues, document = _cost(profile, context)
        issues = list(issues)
        if cost is not None:
            if cost['tick_seconds'] != 1e-9:
                issues.append('whole-call profile must use the existing integer-nanosecond writer convention')
            if type(cost['duration_ticks']) is not int or not 0 <= cost['duration_ticks'] < 2**60:
                issues.append('whole-call duration must fit the common nonnegative integer-nanosecond contract')
        admitted = cost is not None and not issues
        offers.append(dict(index=index, admitted=admitted,
            profile_sha256=_digest(document) if document is not None else None,
            source=deepcopy(cost['source']) if cost is not None else None,
            frontier=issues))
        if admitted:
            accepted.setdefault(_digest(document), (cost, document))
    frontier = []
    if not accepted:
        frontier.append(dict(field='cost', reason='NO_ADMITTED_EXACT_CONTEXT_WHOLE_METHOD_CALL_PROFILE'))
        for offer in offers:
            frontier.extend(dict(field='profiles.' + str(offer['index']), reason=reason)
                for reason in offer['frontier'])
    elif len(accepted) > 1:
        frontier.append(dict(field='cost', reason='MULTIPLE_DISTINCT_APPLICABLE_WHOLE_METHOD_PROFILES'))
    accepted_cost, accepted_profile = next(iter(accepted.values())) if len(accepted) == 1 else (None, None)
    if accepted_cost is not None:
        for key in ('context_evidence',):
            sources.extend(accepted_profile.get(key, ()))
        sources.extend(accepted_profile['estimate'].get('evidence', ()))
        sources.extend(accepted_profile['derivation'].get('evidence', ()))
        for name in ('source_profile', 'rebase_proof'):
            if name in accepted_profile:
                sources.append(dict(artifact=accepted_profile[name], locator='#'))
        sources = _refs(sources)
        # Canonical composition and observed-original forward expectations
        # remain declared models; neither creates observed target components.
        estimate = CostEstimate(accepted_cost['duration_ticks'], None, None,
            'DECLARED_MODEL', query_key(query), tuple(sources), ())
    else:
        estimate = CostEstimate(None, None, None, 'UNKNOWN', query_key(query), tuple(sources),
            tuple(item['reason'] for item in frontier))
    obligations = [
        dict(field='whole_entry_cpu_execution_domain',
            reason=('Original caller affinity and native-thread environment do not establish a whole-entry CPU demand or reservation.'
                if candidate is None else
                'Candidate binding covers internal native stages; caller guards, packing, allocation, providers and return do not have a proven whole-entry CPU demand.'),
            source_refs=deepcopy(sources)),
        dict(field='whole_entry_opaque_memory_and_residency',
            reason='Original payload budget does not establish opaque runtime memory, original NumPy peak memory, process RSS or physical NUMA residency.',
            source_refs=deepcopy(sources)),
    ]
    return dict(schema=SCHEMA, schema_version=1, candidate_id=candidate_id,
        domain_id=domain['domain_id'], domain_sha256=_digest(domain),
        region=deepcopy(domain['region']), target=deepcopy(domain['target_snapshot']),
        context=deepcopy(context), query=dict(scope=SCOPE, context=deepcopy(context)),
        cost=encode(estimate), source_refs=deepcopy(sources), frontier=frontier,
        profiles=offers, admitted_profile_sha256=_digest(accepted_profile) if accepted_profile is not None else None,
        derivation=deepcopy(accepted_cost['derivation']) if accepted_cost is not None else None,
        candidate_qualification=dict(original_no_op=candidate is None,
            legal_under_guards=True if candidate is None else candidate['legal_under_guards'],
            legality_frontier=[] if candidate is None else deepcopy(candidate['legality_frontier']),
            lowering_admitted=False if candidate is None else candidate.get('lowering_admitted', False)),
        resource_qualification=dict(status='UNRESOLVED',
            candidate_internal_stage_binding=None if candidate is None else deepcopy(candidate['binding']),
            stage_binding_is_whole_entry_binding=False,
            caller_affinity_cpu_ids=deepcopy(domain['live_constraints']['process_affinity_cpu_ids']),
            caller_affinity_is_exclusive_reservation=False,
            known_native_payload=dict(external_array_bytes=base['external_array_bytes'],
                workspace_payload_bytes=base['workspace_payload_bytes'],
                total_region_array_payload_bytes=base['total_region_array_payload_bytes'],
                memory_budget_bytes=base['memory_budget_bytes'], budget_scope=base['budget_scope']),
            original_numpy_peak_memory='UNKNOWN', opaque_runtime_memory='UNKNOWN',
            physical_numa_residency='UNKNOWN', whole_entry_physical_feasibility_proven=False),
        unresolved_resources=obligations,
        scope_interpretation='WHOLE_METHOD_PROFILE_REPLAY_NOT_STAGE_OR_SESSION_REPEAT_COST',
        source_admission='RETAINED_DOMAIN_REPLAY_NOT_NEW_F4_OR_NATIVE_ARTIFACT_ADMISSION',
        cost_interpretation='DECLARED_FORWARD_EXPECTATION_NOT_OBSERVED_COMPONENTS_OR_PHYSICAL_BOUNDS',
        target_program_executed=False, native_loaded=False, compiled=False,
        deployment_authorized=False)


def verify_stateful_entry_cost(report, domain, candidate_id, profiles=()):
    """Recompute domain, profile evidence, query identity and resource qualifiers."""
    expected = derive_stateful_entry_cost(domain, candidate_id, profiles)
    if _digest(report) != _digest(expected):
        raise ValueError('stateful entry cost differs from whole-call profile/domain replay')
    return expected


__all__ = ['SCHEMA', 'SCOPE', 'derive_stateful_entry_cost', 'verify_stateful_entry_cost']
