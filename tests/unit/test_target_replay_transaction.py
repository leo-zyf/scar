"""Retained hardware reuse is detached, content guarded and transaction local."""
from collections import Counter
from copy import deepcopy

import pytest

from test_cpu_hardware_placement import hardware
from scar.analysis import target_graph
from scar.analysis.replay_transaction import replay_transaction
from scar.optimization.hardware import _target_materials


def test_hardware_decode_and_topology_reuse_keep_content_and_cold_boundaries(hardware, monkeypatch):
    calls = Counter()
    original_resources = target_graph._resources
    original_decode = target_graph.TargetHardwareGraph._decode_table

    def resources(*args, **kwargs):
        calls['topology'] += 1
        return original_resources(*args, **kwargs)

    def decode(path, receipt, cls):
        calls[str(path)] += 1
        return original_decode(path, receipt, cls)

    monkeypatch.setattr(target_graph, '_resources', resources)
    monkeypatch.setattr(target_graph.TargetHardwareGraph, '_decode_table', staticmethod(decode))
    with replay_transaction():
        first = _target_materials(hardware)
        expected_report, expected_inventory = deepcopy(first[0].report), deepcopy(first[0].inventory)
        first[0].report['nodes'].clear()
        first[0].inventory['cpu']['online_cpu_ids'].clear()
        second = _target_materials(hardware)
        direct = target_graph.TargetHardwareGraph(hardware)
        assert second[0].report == direct.report == expected_report
        assert second[0].inventory == direct.inventory == expected_inventory
        assert second[2] == direct.resources()
        assert calls['topology'] == 1
        assert all(value == 1 for key, value in calls.items() if key != 'topology')
    with replay_transaction():
        _target_materials(hardware)
        assert calls['topology'] == 2
        assert all(value == 2 for key, value in calls.items() if key != 'topology')

    receipt = expected_report['nodes']['CpuExecutor']
    path = hardware/receipt['path']
    original = path.read_bytes()
    try:
        with pytest.raises(ValueError):
            with replay_transaction():
                _target_materials(hardware)
                path.write_bytes(original + b'changed')
                _target_materials(hardware)
    finally:
        path.write_bytes(original)
    with replay_transaction():
        assert _target_materials(hardware)[0].report == expected_report
