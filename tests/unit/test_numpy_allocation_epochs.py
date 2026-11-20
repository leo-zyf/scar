"""Successful NumPy reallocations start a new epoch even for one array object."""
import shutil
import sys
import sysconfig

import numpy as np
import pytest

from scar.trace.numpy_allocation_epochs import AllocationEpochTracker, build_tracker


@pytest.mark.skipif(sys.platform != 'linux' or sysconfig.get_config_var('Py_GIL_DISABLED')
                    or np.__version__.split('.')[0] != '2' or shutil.which('g++') is None,
                    reason='native allocator test requires NumPy 2.x and a C++ compiler on Linux')
def test_array_owner_epoch_survives_value_updates_and_changes_on_resize(tmp_path):
    old = np.zeros((3, 3))
    tracker = AllocationEpochTracker(build_tracker(tmp_path / 'tracker'))
    try:
        assert tracker(old) is None
        owner = np.zeros((3, 3))
        before = tracker(owner)
        assert before is not None
        owner[:] += 2
        assert tracker(owner)['epoch'] == before['epoch']

        owner.resize((4, 4), refcheck=False)
        resized = tracker(owner)
        assert resized['epoch'] > before['epoch']
        assert resized['cursor'] >= resized['epoch']
    finally:
        tracker.close()
    assert tracker(owner) is None
