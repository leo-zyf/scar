"""Validate caller-side timings for Tensor consumers after parallel joins."""

from statistics import median_low

from scar.optimization.aten_team_numa_locality import _BUDGET_FRONTIER, _legal_contexts
from scar.optimization.aten_tensor_lifetime import tensor_releases


def _operation_parents(recipe):
    tensor_releases(recipe)
    operations = recipe['operations']
    producers = {operation['id']: index for index, operation in enumerate(operations)}
    return [tuple(producers[arg['value']] for arg in operation['arguments'] if arg.get('value') in producers)
            for operation in operations]


def _consumer_indices(parents, selection):
    reached = {i for start, width in selection for i in range(start, start + width)}
    consumers = []
    for i, dependencies in enumerate(parents):
        if i not in reached and any(parent in reached for parent in dependencies):
            reached.add(i)
            consumers.append(i)
    return consumers


def read_aten_post_join_evidence(evidence, recipe, schedule, warmup):
    """Return exact ordered serial-descendant samples for each legal context."""
    if evidence is None:
        return None
    if (type(evidence) is not dict or type(evidence.get('schema_version')) is not int
            or evidence['schema_version'] != 1):
        raise ValueError('team post-join timing evidence version is unsupported')

    expected = _legal_contexts(schedule, recipe)
    status = evidence.get('status')
    if status == _BUDGET_FRONTIER:
        if evidence.keys() != {'schema_version', 'status'} or expected is not None:
            raise ValueError('team post-join timing budget frontier differs from the legal window set')
        return dict(status=status, contexts=[])

    if (status != 'MEASURED' or evidence.keys() != {'schema_version', 'status', 'contexts'}
            or expected is None or type(evidence['contexts']) is not list
            or len(evidence['contexts']) != len(expected)):
        raise ValueError('team post-join timings require the complete bounded legal context set')
    if type(warmup) is not int or not 3 <= warmup <= 100:
        raise ValueError('team post-join timings require three to one hundred calibration rows')

    parents = _operation_parents(recipe)
    contexts = []
    for selection, raw in zip(expected, evidence['contexts'], strict=True):
        if type(raw) is not dict or raw.keys() != {'windows', 'operation_indices', 'consumer_samples_ns'}:
            raise ValueError('team post-join timing context requires windows, consumers and timing samples')
        windows, indices, samples = raw['windows'], raw['operation_indices'], raw['consumer_samples_ns']
        if (type(windows) is not list or any(type(window) is not list or len(window) != 2
                or any(type(value) is not int for value in window) for window in windows)
                or tuple(map(tuple, windows)) != selection):
            raise ValueError('team post-join timings differ from the complete ordered context set')
        if (type(indices) is not list or any(type(i) is not int for i in indices)
                or indices != _consumer_indices(parents, selection)):
            raise ValueError('team post-join timings must cover exactly the serial Tensor descendants')
        if (type(samples) is not list or len(samples) != warmup
                or any(type(row) is not list or len(row) != len(indices)
                       or any(type(duration) is not int or not 0 < duration < 2**63 for duration in row)
                       for row in samples)):
            raise ValueError('team post-join timing requires one positive bounded duration per consumer and row')
        contexts.append(dict(windows=windows, operation_indices=indices, consumer_samples_ns=samples,
                             consumer_median_ns=[median_low(column) for column in zip(*samples, strict=True)]))
    return dict(status=status, contexts=contexts)
