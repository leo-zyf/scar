"""Expand a fixed-placement data edge into explicit, budgeted copy hops."""

from dataclasses import replace
from heapq import heappop, heappush

from scar.ir.implementation_plan import (
    ImplementationOption, PlanningDataEdge, PlanningOperation, validate_problem,
)


def expand_material_route(problem, edge_id, qualified_hops):
    """Use calibrated copy capabilities, with no implied route from topology.

    Hop costs must be valid for the exact byte count and representation of the
    selected edge. This nominates the least isolated service time route within
    a fixed source/destination placement. The shared planner then checks the
    full dependency and resource calendar, including the staging leases.
    """
    problem = validate_problem(problem)
    edge, = (row for row in problem.data_edges if row.id == edge_id)
    operations = {row.id: row for row in problem.operations}
    consumer = operations[edge.consumer]
    sources = ({edge.external_location} if edge.producer is None else
               {mode.output_locations[edge.value] for mode in operations[edge.producer].options})
    destinations = {mode.input_locations[edge.value] for mode in consumer.options}
    if len(sources) != 1 or len(destinations) != 1:
        raise ValueError('route expansion requires fixed endpoint placement')
    source, = sources
    destination, = destinations
    if source == destination:
        return problem

    resources = {resource.id: resource for resource in problem.resources}
    eligible = {}
    for hop in qualified_hops:
        if (hop.source_memory not in resources or hop.destination_memory not in resources
                or any(resources[loc].kind != 'MEMORY' for loc in (hop.source_memory, hop.destination_memory))
                or hop.source_memory == hop.destination_memory or not hop.proof_refs):
            raise ValueError('copy capability needs distinct known memory endpoints and proof')
        target = resources[hop.destination_memory]
        duration = hop.cost.estimate_ns
        if (type(duration) is int and 0 <= duration < 2**60 and hop.cost.basis != 'UNKNOWN'
                and target.capacity - problem.reserved_memory.get(target.id, 0) >= edge.nbytes
                and all(name in resources and type(units) is int and 0 <= units <= resources[name].capacity
                        for name, units in hop.demands.items())):
            eligible.setdefault(hop.source_memory, []).append(hop)

    frontier = [(0, (), source)]
    best = {}
    while frontier:
        estimate, path, position = heappop(frontier)
        if position in best:
            continue
        best[position] = path
        if position == destination:
            break
        for ordinal, hop in enumerate(eligible.get(position, ())):
            if hop.destination_memory not in best:
                heappush(frontier, (estimate + hop.cost.estimate_ns, (*path, (position, ordinal)), hop.destination_memory))
    if destination not in best:
        raise ValueError('no calibrated copy path for the fixed material endpoints')

    copies, data = [], []
    value, producer, location = edge.value, edge.producer, source
    for index, (position, ordinal) in enumerate(best[destination]):
        hop = eligible[position][ordinal]
        name = f'$copy:{edge.id}:{index}'
        output = f'{name}:material'
        refs = (*edge.source_refs, *hop.proof_refs)
        data.append(PlanningDataEdge(f'{name}:input', value, producer, name, edge.nbytes,
            location if producer is None else None, source_refs=refs))
        mode = ImplementationOption(hop.id, hop.lowering, refs, hop.demands, {}, {value: location},
            {output: hop.destination_memory}, hop.cost,
            dict(logical_value=edge.value, source_memory=location, destination_memory=hop.destination_memory,
                 completion='DESTINATION_READY'))
        copies.append(PlanningOperation(name, refs, (), (mode,), mode.id))
        value, producer, location = output, name, hop.destination_memory

    def retarget(mode):
        inputs = dict(mode.input_locations)
        inputs.pop(edge.value)
        inputs[value] = destination
        return replace(mode, input_locations=inputs)

    updated = replace(consumer, options=tuple(retarget(mode) for mode in consumer.options))
    data.append(replace(edge, value=value, producer=producer, external_location=None, transfers=()))
    return validate_problem(replace(problem,
        operations=(*[updated if row.id == consumer.id else row for row in problem.operations], *copies),
        data_edges=(*[row for row in problem.data_edges if row.id != edge.id], *data)))
