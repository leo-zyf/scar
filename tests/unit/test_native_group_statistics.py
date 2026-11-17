from copy import deepcopy
from fractions import Fraction

import pytest

from scar.compiler.calibrate_native_groups import native_group_statistics


def test_joint_partition_uses_sample_means_and_keeps_complement_unattributed():
    samples = []
    for index, (left, right) in enumerate(((0, 10), (10, 0), (10, 10))):
        start = 1000*(index+1)
        cuts = [start+5, start+5+left, start+12+left, start+12+left+right]
        samples.append(dict(ordinal=index, native_status=0, ffi_start_ns=start,
            cuts_ns=cuts, ffi_end_ns=cuts[-1]+3))
    document = dict(repetitions=3, samples=samples, group_layout=[
        dict(index=0, stage_indices=[0], begin_cut=0, end_cut=1),
        dict(index=1, stage_indices=[1], begin_cut=2, end_cut=3)])
    result = native_group_statistics(document)
    def exact(row):
        mean = row['exact_mean_ns']
        return Fraction(mean['numerator'], mean['denominator'])
    assert sum(exact(row) for row in result['groups']) == Fraction(40, 3)
    assert sum(row['median_ns'] for row in result['groups']) == 20
    assert exact(result['joint_unattributed_outside_groups']) == 15
    assert sum(exact(row) for row in result['groups'])+15 == exact(result['ffi_window'])
    assert result['complement_is_not_independent_cleanup_or_guard_measurement'] is True
    bad = deepcopy(document)
    bad['samples'][0]['cuts_ns'][1] = bad['samples'][0]['ffi_end_ns']+1
    with pytest.raises(ValueError, match='ordered'):
        native_group_statistics(bad)
    bad = deepcopy(document)
    bad['samples'][0]['native_status'] = False
    with pytest.raises(ValueError, match='successful'):
        native_group_statistics(bad)
