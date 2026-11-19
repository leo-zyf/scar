"""Validate native fork/join timing evidence for a bounded ATen team."""

from statistics import median_low

from scar.optimization.aten_team_numa_locality import _BUDGET_FRONTIER, _legal_contexts


def read_aten_team_group_evidence(evidence, recipe, schedule, warmup):
    """Return samples and medians in the same ordered contexts as NUMA evidence."""
    if evidence is None:
        return None
    if (type(evidence) is not dict or type(evidence.get('schema_version')) is not int
            or evidence['schema_version'] != 1):
        raise ValueError('team group timing evidence version is unsupported')

    expected = _legal_contexts(schedule, recipe)
    status = evidence.get('status')
    if status == _BUDGET_FRONTIER:
        if evidence.keys() != {'schema_version', 'status'} or expected is not None:
            raise ValueError('team group timing budget frontier differs from the legal window set')
        return dict(status=status, contexts=[])

    if (status != 'MEASURED' or evidence.keys() != {'schema_version', 'status', 'contexts'}
            or expected is None or type(evidence['contexts']) is not list
            or len(evidence['contexts']) != len(expected)):
        raise ValueError('team group timings require the complete bounded legal context set')
    if type(warmup) is not int or not 3 <= warmup <= 100:
        raise ValueError('team group timings require three to one hundred calibration rows')

    contexts = []
    for selection, raw in zip(expected, evidence['contexts'], strict=True):
        if type(raw) is not dict or raw.keys() != {'windows', 'group_samples_ns'}:
            raise ValueError('team group timing context requires windows and their elapsed samples')
        windows, samples = raw['windows'], raw['group_samples_ns']
        if (type(windows) is not list or any(type(window) is not list or len(window) != 2
                or any(type(value) is not int for value in window) for window in windows)
                or tuple(map(tuple, windows)) != selection):
            raise ValueError('team group timings differ from the complete ordered context set')
        if (type(samples) is not list or len(samples) != warmup
                or any(type(row) is not list or len(row) != len(windows)
                       or any(type(duration) is not int or not 0 < duration < 2**63 for duration in row)
                       for row in samples)):
            raise ValueError('team group timing requires one positive bounded duration per window and row')
        contexts.append(dict(windows=windows, group_samples_ns=samples,
                             group_median_ns=[median_low(column) for column in zip(*samples, strict=True)]))
    return dict(status=status, contexts=contexts)
