"""Bound graph-held Tensor storage across all finishes of fixed CPU lanes."""

from collections import deque

from scar.optimization.aten_tensor_lifetime import _bounded_sum, _byte_vector, _bytes


_LANE_COUNT = 3
_MAX_PREFIX_STATES = 65_536
_INITIAL_PREFIX = (0, 0, 0)


def _validate(dependencies, lanes, output_indices, result_bytes, persistent_input_bytes):
    if type(dependencies) not in (list, tuple) or not dependencies:
        raise ValueError('ready lifetime requires a nonempty Tensor dependency graph')
    count = len(dependencies)
    if (type(lanes) not in (list, tuple) or len(lanes) != count or
            any(type(lane) is not int or not 0 <= lane < _LANE_COUNT for lane in lanes)):
        raise ValueError('ready lifetime requires one CPU lane per operation')
    for op, parents in enumerate(dependencies):
        if (type(parents) not in (list, tuple) or
                any(type(parent) is not int or not 0 <= parent < op for parent in parents) or
                any(left >= right for left, right in zip(parents, parents[1:]))):
            raise ValueError('ready lifetime dependencies must be unique earlier operations in source order')
    if (type(output_indices) not in (list, tuple, set, frozenset) or
            any(type(index) is not int or not 0 <= index < count for index in output_indices)):
        raise ValueError('ready lifetime outputs must name produced operations')
    _byte_vector(result_bytes, count, 'result')
    if not _bytes(persistent_input_bytes):
        raise ValueError('ready lifetime requires a bounded persistent input byte count')


def _satisfies(prefix, dependency):
    return (prefix[0] >= dependency[0] and prefix[1] >= dependency[1] and
            prefix[2] >= dependency[2])


def _operation_order(dependencies, supplied):
    count = len(dependencies)
    if supplied is None:
        return range(count)
    if (type(supplied) not in (list, tuple) or len(supplied) != count
            or any(type(op) is not int for op in supplied) or sorted(supplied) != list(range(count))):
        raise ValueError('ready lifetime operation order must be an exact permutation')
    seen = set()
    for op in supplied:
        if not all(parent in seen for parent in dependencies[op]):
            raise ValueError('ready lifetime operation order must follow Tensor dependencies')
        seen.add(op)
    return supplied


def ready_lifetime_bound(dependencies, lanes, output_indices, result_bytes, persistent_input_bytes, *,
                         operation_order=None):
    """Return a safe storage envelope independent of sampled operation durations.

    Each lane traverses the supplied topological order, or source order by
    default. A finished operation has published its result and dropped the
    handles it is last to consume. At a reachable completed prefix, every
    ready lane head can have allocated its output before any further operation
    finishes. That cut bounds storage across execution timings.
    """
    _validate(dependencies, lanes, output_indices, result_bytes, persistent_input_bytes)
    full_retention = _bounded_sum((persistent_input_bytes, *result_bytes))
    exports = set(output_indices)
    by_lane = [[], [], []]
    ranks = [0] * len(lanes)
    for op in _operation_order(dependencies, operation_order):
        lane = lanes[op]
        by_lane[lane].append(op)
        ranks[op] = len(by_lane[lane])

    # The latest consumer on each lane subsumes smaller ranks on that lane.
    # The same compression makes checking prerequisites independent of fan-in.
    last_uses = [[0] * _LANE_COUNT for _ in lanes]
    requirements = [[0] * _LANE_COUNT for _ in lanes]
    for op, parents in enumerate(dependencies):
        lane = lanes[op]
        for parent in parents:
            last_uses[parent][lane] = max(last_uses[parent][lane], ranks[op])
            source_lane = lanes[parent]
            requirements[op][source_lane] = max(requirements[op][source_lane], ranks[parent])
    retained_results = [op in exports or any(uses) for op, uses in enumerate(last_uses)]

    frontier = deque([_INITIAL_PREFIX])
    completed_storage = {_INITIAL_PREFIX: persistent_input_bytes}
    peak = -1
    while frontier:
        prefix = frontier.popleft()
        ready = [ops[finished] for ops, finished in zip(by_lane, prefix, strict=True)
                 if finished < len(ops) and _satisfies(prefix, requirements[ops[finished]])]
        live = completed_storage[prefix]
        cut = live + sum(result_bytes[op] for op in ready)
        if cut > peak:
            peak, witness, in_flight = cut, prefix, ready
        for op in ready:
            lane = lanes[op]
            advanced = prefix[:lane] + (prefix[lane] + 1,) + prefix[lane + 1:]
            if advanced in completed_storage:
                continue
            if len(completed_storage) == _MAX_PREFIX_STATES:
                return dict(peak_tensor_storage_bytes=full_retention, proof='CONSERVATIVE_FULL_RETENTION',
                            reachable_prefixes=None)
            # Only a handle consumed by this operation can leave the live set
            # when this prefix advances. The completed source set determines
            # its live bytes, regardless of which path reached the prefix.
            reclaimed = sum(result_bytes[parent] for parent in dependencies[op]
                            if parent not in exports and _satisfies(advanced, last_uses[parent]))
            produced = result_bytes[op] if retained_results[op] else 0
            completed_storage[advanced] = live + produced - reclaimed
            frontier.append(advanced)

    return dict(peak_tensor_storage_bytes=peak, proof='COMPLETE_LANE_PREFIX_FRONTIER',
                reachable_prefixes=len(completed_storage),
                witness_completed=sorted(op for ops, finished in zip(by_lane, witness, strict=True)
                                         for op in ops[:finished]),
                witness_in_flight=sorted(in_flight))
