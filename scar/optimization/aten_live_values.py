"""Aten live values."""

from collections import deque
from scar.compiler.aten_native_recipe import SCHEMA, SCHEMA_VERSION
from math import prod


# Derive Tensor handle lifetimes and a backing-storage envelope for native ATen graphs.
_MAX_BYTES = (1 << 63) - 1


def tensor_releases(recipe):
    """Return result indices to reset after each serial operator completes.

    The ordered recipe already passed the native ABI and alias signature checks.
    Inputs belong to the caller and outputs remain live through return.
    An unconsumed result remains live until its producer completes.
    """
    if (type(recipe) is not dict or recipe.get('schema') != SCHEMA
            or type(recipe.get('schema_version')) is not int or recipe['schema_version'] != SCHEMA_VERSION):
        raise ValueError('normalized native ATen recipe required for Tensor liveness')
    inputs, operations, outputs = (recipe.get(field) for field in ('inputs', 'operations', 'outputs'))
    if any(type(rows) is not list or not rows for rows in (inputs, operations, outputs)):
        raise ValueError('native ATen recipe requires inputs, operations and outputs for Tensor liveness')

    definitions = {}
    for row in inputs:
        name = row['id']
        if type(name) is not str or not name or name in definitions:
            raise ValueError('Tensor liveness requires unique native input references')
        definitions[name] = None

    last_use = list(range(len(operations)))
    for index, op in enumerate(operations):
        name = op['id']
        if type(name) is not str or not name or name in definitions:
            raise ValueError('Tensor liveness requires a unique reference per native result')
        for operand in op['arguments']:
            if 'value' not in operand:
                continue
            before = operand['value']
            if type(before) is not str or before not in definitions:
                raise ValueError('Tensor liveness requires every operand reference to follow its definition')
            producer = definitions[before]
            if producer is not None:
                last_use[producer] = index
        definitions[name] = index

    for name in outputs:
        if type(name) is not str or definitions.get(name) is None:
            raise ValueError('Tensor liveness requires every export to be a produced result')
        last_use[definitions[name]] = None
    release_after = [[] for _ in operations]
    for producer, end in enumerate(last_use):
        if end is not None:
            release_after[end].append(producer)
    return release_after


def _bytes(value):
    return type(value) is int and 0 <= value <= _MAX_BYTES


def _byte_vector(values, count, kind):
    if type(values) not in (tuple, list) or len(values) != count or not all(map(_bytes, values)):
        raise ValueError(f'exact nonnegative native {kind} backing-storage byte counts required')


def _bounded_sum(values):
    total = sum(values)
    if total > _MAX_BYTES:
        raise ValueError('native Tensor backing-storage envelope exceeds signed byte range')
    return total


def tensor_footprint(recipe, *, input_bytes, output_bytes, candidates, budget_bytes=None):
    """Account for graph-held backing storage at serial calls and fork/join windows.

    Each byte count describes one observed input or operator result in recipe
    order. A segment conservatively holds its outputs and all previously live
    handles until the call or the fork/join completes. Release means dropping
    the generated graph's handle, with no claim about allocator reclamation.
    """
    releases = tensor_releases(recipe)
    ops = recipe['operations']
    _byte_vector(input_bytes, len(recipe['inputs']), 'input')
    _byte_vector(output_bytes, len(ops), 'result')
    if budget_bytes is not None and not _bytes(budget_bytes):
        raise ValueError('native Tensor backing-storage budget must be a bounded nonnegative byte count')

    persistent = _bounded_sum(input_bytes)
    live = persistent
    entries, operation_peaks = [], []
    for count, end in zip(output_bytes, releases, strict=True):
        entries.append(live)
        peak = _bounded_sum((live, count))
        operation_peaks.append(peak)
        live = peak - sum(output_bytes[producer] for producer in end)
    serial_peak = max(operation_peaks)
    if budget_bytes is not None and serial_peak > budget_bytes:
        raise ValueError(f'native Tensor budget {budget_bytes} bytes is below the serial backing-storage peak '
                         f'of {serial_peak} bytes')

    window_peaks, eligible_starts = {}, []
    for candidate in candidates:
        first, second = candidate['pair_indices']
        if (type(first) is not int or type(second) is not int or not 0 <= first < len(ops) - 1
                or second != first + 1 or str(first) in window_peaks
                or any(arg.get('value') == ops[first]['id'] for arg in ops[second]['arguments'])):
            raise ValueError('Tensor footprint requires distinct independent adjacent CPU windows')
        peak = _bounded_sum((entries[first], output_bytes[first], output_bytes[second]))
        window_peaks[str(first)] = peak
        if budget_bytes is None or peak <= budget_bytes:
            eligible_starts.append(first)

    return dict(scope='GRAPH_HELD_TENSOR_BACKING_STORAGE', release_after=releases,
        persistent_input_bytes=persistent, serial_peak_bytes=serial_peak,
        operation_peak_bytes=operation_peaks, window_peak_bytes=window_peaks,
        budget_bytes=budget_bytes, candidate_starts_within_budget=eligible_starts,
        accounting='PER_GRAPH_VALUE_CONSERVATIVE_SUM_UNTIL_SEGMENT_COMPLETION',
        excluded_bytes=['PROVIDER_INTERNAL_WORKSPACE', 'ALLOCATOR_RESERVES', 'OTHER_PROCESS_MEMORY'],
        allocator_reclamation_verified=False)


# Bound graph-held Tensor storage across all finishes of fixed CPU lanes.
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


# Minimize backing-storage peaks over qualified recurrence step orders.
_MAX_PREFIXES = 32768
_ELEMENT_BYTES = {'FLOAT32': 4, 'FLOAT64': 8}


def _source_peak(base, sizes, allocations, consumers, output):
    release = [0] * len(sizes)
    for i, size in enumerate(sizes):
        if i != output:
            last = consumers[i].bit_length() - 1 if consumers[i] else i
            release[last] += size
    peak = live = base
    for size, allocation, freed in zip(sizes, allocations, release, strict=True):
        peak = max(peak, _bounded_sum((live, allocation)))
        live += size - freed
    return peak


def _live_bytes(done, pending, consumers, sizes, output_bit):
    live = 0
    while done:
        bit = done & -done
        i = bit.bit_length() - 1
        if bit == output_bit or consumers[i] & pending:
            live += sizes[i]
        done ^= bit
    return live


def plan_step_lifetime_order(operations, output, *, raw_backing, carry, rhs_temporary) -> dict:
    """Find the minimum scoped peak of the already-qualified functional body.

    Source-ordered steps define fresh results, with nonlocal inputs borrowed.
    RHS anchors keep their original relative order. The nested RHS envelope
    includes its result, and releasing a handle does not promise allocator
    reclamation. Prefixes at the source peak cannot yield an improvement.
    """
    names = [row['id'] for row in operations]
    indices = {name: i for i, name in enumerate(names)}
    if (len(indices) != len(names) or output not in indices
            or not all(type(name) is str and name for name in names)):
        raise ValueError('recurrence order requires unique result IDs and a produced output')
    if not all(map(_bytes, (raw_backing, carry, rhs_temporary))):
        raise ValueError('recurrence order requires exact nonnegative storage byte counts')
    if any(row['dtype'] not in _ELEMENT_BYTES for row in operations):
        raise ValueError('recurrence order requires qualified Float32 or Float64 results')
    sizes = [prod(row['shape']) * _ELEMENT_BYTES[row['dtype']] for row in operations]
    if not all(map(_bytes, sizes)):
        raise ValueError('recurrence results require exact nonnegative storage byte counts')
    base = _bounded_sum((raw_backing, carry, 8))
    count, output_index = len(names), indices[output]
    inputs, consumers, allocations = [0] * count, [0] * count, sizes.copy()
    last_rhs = 0
    for i, row in enumerate(operations):
        bit = 1 << i
        for arg in row['arguments']:
            producer = indices.get(arg.get('value'))
            if producer is None:
                continue
            if producer >= i:
                raise ValueError('recurrence order requires source-topological result definitions')
            inputs[i] |= 1 << producer
            consumers[producer] |= bit
        if row['op'] == 'RHS':
            if rhs_temporary < sizes[i]:
                raise ValueError('nested RHS storage must include the produced result')
            allocations[i] = rhs_temporary
            inputs[i] |= last_rhs
            last_rhs = bit

    source = _source_peak(base, sizes, allocations, consumers, output_index)
    source_order = dict(order=names, original_peak_bytes=source, peak_bytes=source, status='KEEP_SOURCE_ORDER')
    full, output_bit = (1 << count) - 1, 1 << output_index
    layer, parents, explored = {0: base}, {}, 1
    for _ in names:
        following = {}
        for done, peak in layer.items():
            pending = full ^ done
            live = base + _live_bytes(done, pending, consumers, sizes, output_bit)
            remaining = pending
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                i = bit.bit_length() - 1
                if inputs[i] & pending:
                    continue
                state, candidate = done | bit, max(peak, live + allocations[i])
                if candidate >= following.get(state, source):
                    continue
                if state not in following:
                    if explored == _MAX_PREFIXES:
                        return dict(source_order, explored_states=explored, reason='SEARCH_BUDGET_EXHAUSTED')
                    explored += 1
                following[state] = candidate
                parents[state] = done, i
        if not following:
            return dict(source_order, explored_states=explored)
        layer = following

    order, state = [], full
    while state:
        state, i = parents[state]
        order.append(names[i])
    return dict(order=order[::-1], original_peak_bytes=source, peak_bytes=layer[full],
                status='IMPROVED', explored_states=explored)
