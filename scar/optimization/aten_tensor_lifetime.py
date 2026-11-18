"""Derive Tensor handle lifetimes and a backing-storage envelope for native ATen graphs."""

from scar.compiler.aten_native_recipe import SCHEMA, SCHEMA_VERSION


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
