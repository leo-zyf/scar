"""Apply exact treatment costs to a fixed device placement before ranking."""

from dataclasses import replace

from scar.ir.implementation_plan import CostEstimate, problem_document, problem_from_dict
from scar.optimization.constructive_planner import construct_implementations


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
