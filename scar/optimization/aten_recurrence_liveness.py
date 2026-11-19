"""Minimize backing-storage peaks over qualified recurrence step orders."""

from math import prod

from scar.optimization.aten_tensor_lifetime import _bounded_sum, _bytes


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
