"""Device cost bounds."""

from dataclasses import asdict, replace
from statistics import median
from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import CostEstimate, problem_document, problem_from_dict
from scar.optimization.cpu_sgemm_probe import SCHEMA
from scar.optimization.constructive_planner import construct_implementations


# Compare closed service sample envelopes for native placement treatments.
def _envelope(row):
    interval = row.get('service_interval') or {}
    low, mid, high = (interval.get(name) for name in ('lower_ns', 'estimate_ns', 'upper_ns'))
    if (row['status'] == 'VERIFIED_SCHEDULE' and interval.get('status') == 'BOUNDED'
            and all(type(value) is int for value in (low, mid, high)) and 0 <= low <= mid <= high):
        return low, high
    return 0, None


def _comparison_blocker(ordinal, upper, other, lower):
    reason = 'SAMPLE_ENVELOPES_TOUCH' if upper == lower else 'SAMPLE_ENVELOPES_OVERLAP'
    return dict(candidate_ordinal=ordinal, other_ordinal=other, candidate_upper_ns=upper,
                other_lower_ns=lower, reason=reason)


def choose_native_device_interval(evaluated, baseline_ordinal=0, *, enumeration_complete=True):
    """Nominate a GPU only when its worst sample-model cost beats all rivals."""
    if type(baseline_ordinal) is not int or baseline_ordinal < 0 or type(enumeration_complete) is not bool:
        raise ValueError('expected a nonnegative baseline ordinal and an explicit enumeration status')
    bounds, seen, selected = {}, set(), None
    for row in evaluated:
        ordinal = row['ordinal']
        if type(ordinal) is not int or ordinal < 0 or ordinal in seen:
            raise ValueError('placement ordinals must be unique nonnegative integers')
        seen.add(ordinal)
        if row['status'] != 'PLACEMENT_CAPACITY_EXCEEDED':
            bounds[ordinal] = _envelope(row)
        if ordinal == baseline_ordinal and row['status'] == 'VERIFIED_SCHEDULE':
            selected = baseline_ordinal

    incumbent = bounds.get(baseline_ordinal, (0, None))
    blockers = [dict(ordinal=ordinal, lower_ns=low, upper_ns=high, reason='UNBOUNDED_SERVICE_INTERVAL')
                for ordinal, (low, high) in bounds.items() if high is None]
    if baseline_ordinal not in bounds:
        blockers.append(dict(ordinal=baseline_ordinal, reason='CPU_BASELINE_NOT_AVAILABLE'))
    if not enumeration_complete:
        blockers.append(dict(reason='PLACEMENT_BUDGET_EXHAUSTED'))

    if incumbent[1] is None:
        decision = 'CPU_INCUMBENT_UNRESOLVED'
    elif not enumeration_complete:
        decision = 'PLACEMENT_BUDGET_EXHAUSTED'
    elif blockers:
        decision = 'PLACEMENT_DOMAIN_UNRESOLVED'
    else:
        alternatives = [(ordinal, interval) for ordinal, interval in bounds.items() if ordinal != baseline_ordinal]
        if all(incumbent[1] <= interval[0] for _, interval in alternatives):
            decision = 'CPU_INTERVAL_DOMINANCE'
        else:
            # The least upper endpoint is the only possible strict winner.
            best, best_interval = min(alternatives, key=lambda row: (row[1][1], row[0]))
            blockers = [_comparison_blocker(best, best_interval[1], ordinal, interval[0])
                        for ordinal, interval in bounds.items() if ordinal != best and best_interval[1] >= interval[0]]
            if blockers:
                decision = 'INTERVALS_OVERLAP'
            else:
                decision, selected = 'STRICT_INTERVAL_GAIN', best

    return dict(schema='scar.native-device-interval-choice', schema_version=1, decision=decision,
                baseline_ordinal=baseline_ordinal, selected_ordinal=selected, blockers=blockers,
                scope='OBSERVED_SERVICE_SAMPLE_ENVELOPES_ONLY', statistical_confidence_claimed=False,
                physical_optimality_proven=False, deployment_authorized=False)


# Transfer matched NVIDIA SGEMM observations into the scoped CPU point model.
def sgemm_body_costs(projection, references, *, stack, target, cpus):
    observations = {}
    library = stack['libraries']['cpu_blas']
    identity = {key: library[key] for key in ('path', 'sha256', 'size_bytes')}
    read_artifact(asdict(target), json_document=False)
    for reference in references:
        report = read_artifact(reference)
        if (reference['schema'] != SCHEMA or report['schema_version'] != 1 or report['target'] != asdict(target)
                or report['cpu_ids'] != list(cpus) or report['dtype'] != 'FP32'
                or report['layout'] != 'CONTIGUOUS_ROW_MAJOR' or report['operator'] != 'SGEMM'
                or (report['alpha'], report['beta']) != (1, 0)
                or report['sample_scope'] != 'WARM_PREALLOCATED_SINGLE_OPERATOR_CALL'):
            raise ValueError('SGEMM observation differs from the source ABI, CPU cohort or hardware target')
        rows = [(index, row) for index, row in enumerate(report['providers'])
                if row['library'] == identity and row['symbol'] == 'cblas_sgemm' and row['integer_bits'] == 32
                and row['state'] == 'MEASURED' and row['threads'] == len(cpus)]
        dimensions = tuple(report['shape_mnk'])
        if len(rows) != 1 or dimensions in observations:
            raise ValueError('each observed SGEMM shape needs one unique pinned NVIDIA LP64 provider sample')
        index, row = rows[0]
        samples = row['samples_ns']
        if (len(samples) != report['repetitions'] or len(samples) < 3 or report['warmups'] < 1
                or any(type(ns) is not int or ns <= 0 for ns in samples) or int(median(samples)) < 1
                or (row['lower_sample_ns'], row['median_ns'], row['upper_sample_ns']) !=
                   (min(samples), median(samples), max(samples))):
            raise ValueError('invalid SGEMM cost samples')
        observations[dimensions] = CostEstimate(int(median(samples)), None, None, 'EMPIRICAL_MODEL',
            'MATCHED_WARM_CPU_SGEMM_POINT_FORECAST', (dict(artifact=reference, locator=f'#/providers/{index}'),))
    return {row['id']: observations.get((*row['shape'], row['reduction']), CostEstimate(None, None, None,
            'UNKNOWN', 'CPU_SGEMM_COMPONENT_UNPRICED', missing_terms=('MATCHED_WARM_SGEMM_OBSERVATION',)))
            for row in projection['operations']}


# Apply exact treatment costs to a fixed device placement before ranking.
_PAGEABLE = 'cpu:pageable-host'
_PINNED = 'cuda:pinned-host'
_DEVICE = 'cuda:device-memory'
_ENDPOINTS = ('CUDA_COPY_RETURN', 'REGION_VALUE_HANDOFF')
_PAGEABLE_TERM = 'CPU_PAGEABLE_REQUIRES_MATCHED_CALIBRATION'
_PINNED_TERM = 'CPU_PINNED_REQUIRES_MATCHED_ATEN_OUT'
_SGEMM_TERM = 'CUDA_REQUIRES_SGEMM'
_HYBRID_TERMS = {
    'LIBTORCH_ATEN_CPU_OUT_F32_V1': _PINNED_TERM,
    'CUBLAS_SGEMM_F32_COLUMN_MAJOR_V1': _SGEMM_TERM,
}
_COPY_TERMS = {
    ('h2d', _PINNED, _DEVICE): 'CUDA_REQUIRES_H2D',
    ('d2h', _DEVICE, _PINNED): 'CUDA_REQUIRES_D2H',
}


def _required_cost(mode, hybrid):
    if mode.lowering in _ENDPOINTS:
        endpoint = CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT')
        if (mode.cost != endpoint or mode.demands or mode.workspace_bytes or mode.output_locations
                or mode.unresolved_resources):
            raise ValueError('placement endpoint must be a zero-cost logical handoff')
        return None
    if hybrid:
        if mode.lowering == 'CUDA_MEMCPY_ASYNC':
            route = (mode.id, mode.parameters.get('source_memory'), mode.parameters.get('destination_memory'))
            term = _COPY_TERMS.get(route)
        else:
            term = _HYBRID_TERMS.get(mode.lowering)
        if term is not None:
            return term
    elif (mode.lowering == 'NATIVE_ATEN_CALL' and mode.input_locations and mode.output_locations
            and set((*mode.input_locations.values(), *mode.output_locations.values())) == {_PAGEABLE}):
        return _PAGEABLE_TERM
    raise ValueError('no empirical cost scope is defined for the fixed placement activity')


def _known(cost):
    return (type(cost) is CostEstimate and cost.basis == 'EMPIRICAL_MODEL'
            and type(cost.estimate_ns) is int and cost.estimate_ns > 0
            and cost.source_refs and all(type(ref) is dict and ref for ref in cost.source_refs)
            and not cost.missing_terms)


def _replace_costs(problem, costs):
    return tuple(replace(op, options=(replace(op.options[0], cost=costs[op.id]),))
                 if op.id in costs else op for op in problem.operations)


def _matched_hybrid(problem, bindings, terms, treatment):
    wire, schedule, observed_bindings = treatment
    observed = problem_from_dict(wire)
    if observed_bindings != bindings or any(len(op.options) != 1 for op in observed.operations):
        raise ValueError('matched hybrid costs must preserve the fixed placement bindings')
    original = {op.id: op.options[0].cost for op in problem.operations}
    restored = replace(observed, operations=_replace_costs(observed, original), assumptions=problem.assumptions,
                       timing_constraints=problem.timing_constraints)
    retained_order = observed.timing_constraints[:len(problem.timing_constraints)]
    service_order = observed.timing_constraints[len(problem.timing_constraints):]
    if (restored != problem or retained_order != problem.timing_constraints
            or any(row.id != f'native-service-order/{index}' or row.before_endpoint != 'END'
                   or row.after_endpoint != 'START' or row.minimum_ns != 0 or row.maximum_ns is not None
                   or row.when or not row.source_refs for index, row in enumerate(service_order))):
        raise ValueError('matched hybrid costs must preserve the declared physical problem and fixed service order')

    missing = {}
    for op in observed.operations:
        cost = op.options[0].cost
        if op.id not in terms:
            if cost != original[op.id]:
                raise ValueError('observed hybrid costs cannot modify a logical endpoint')
        elif cost.basis == 'UNKNOWN' and cost.missing_terms:
            missing[op.id] = replace(cost, missing_terms=tuple(dict.fromkeys((terms[op.id], *cost.missing_terms))))
        elif not _known(cost):
            raise ValueError('observed hybrid costs require exact empirical point evidence or a named unknown')
    if missing:
        updated = replace(observed, operations=_replace_costs(observed, missing))
        return problem_document(updated), None, bindings
    return treatment


def empirical_placement_candidate(placement, built, *, pageable_costs=None, matched_hybrid=None,
                                  unavailable_reason=None):
    """Return a ranking candidate, with no schedule when a service is unpriced.

    The caller qualifies byte-identical pageable evidence or replays an exact
    hybrid treatment. A matched hybrid schedule carries the emitted FIFO and
    wait relationships. Declared resource capacities remain in either model.
    """
    wire, _, bindings = built
    problem = problem_from_dict(wire)
    if (problem.activities or any(len(op.options) != 1 for op in problem.operations)
            or any(edge.transfers for edge in problem.data_edges)):
        raise ValueError('empirical placement expects fixed explicitly materialized operations')
    hybrid = 'CUDA' in placement.values()
    terms = {}
    for op in problem.operations:
        term = _required_cost(op.options[0], hybrid)
        if term is not None:
            terms[op.id] = term
    assigned = {name: 'CUDA' if term == _SGEMM_TERM else 'CPU' for name, term in terms.items()
                if term in (_PAGEABLE_TERM, _PINNED_TERM, _SGEMM_TERM)}
    if not assigned or assigned != placement:
        raise ValueError('the empirical cost candidate must preserve the assigned source operations')
    if matched_hybrid is not None:
        if not hybrid:
            raise ValueError('hybrid treatment costs cannot price the pageable CPU incumbent')
        return _matched_hybrid(problem, bindings, terms, matched_hybrid)

    if pageable_costs is not None and (type(pageable_costs) is not dict
            or pageable_costs and (hybrid or pageable_costs.keys() - terms.keys()
                                   or not all(map(_known, pageable_costs.values())))):
        raise ValueError('pageable CPU costs must be keyed by the fixed pageable ATen operations')
    costs = dict(pageable_costs or {})
    absent = 'NO_MATCHED_HYBRID_TREATMENT' if hybrid else 'NO_MATCHED_PAGEABLE_CPU_TREATMENT'
    reason = unavailable_reason or absent
    unknown = terms.keys() - costs.keys()
    for name in unknown:
        term = terms[name]
        costs[name] = CostEstimate(None, None, None, 'UNKNOWN', f'EMPIRICAL_PLACEMENT/{term}',
                                   missing_terms=tuple(dict.fromkeys((term, reason))))
    assumptions = ('DECLARED_RESOURCE_CAPACITIES_NOT_DEVICE_MEASUREMENTS', 'MATCHED_EMPIRICAL_COSTS_ONLY')
    if not hybrid:
        assumptions += ('PAGEABLE_ATEN_SERVICE_EXCLUDES_PROCESS_BOUNDARY',)
    prior = tuple(row for row in problem.assumptions
                  if row not in ('OFFLINE_SYNTHETIC_ONLY', 'ACTUAL_DEVICE_TIMING_UNKNOWN'))
    updated = replace(problem, operations=_replace_costs(problem, costs), assumptions=(*prior, *assumptions))
    schedule = None if unknown else construct_implementations(updated)
    return problem_document(updated), schedule, bindings
