from pathlib import Path

from scar.analysis.platform_inventory import _accelerators, _constraints, _pci_category


def test_pci_discovery_survives_hidden_backend_and_self_limits_keep_raw_authority(monkeypatch):
    assert _pci_category(0x050200) == 'MEMORY'
    assert _pci_category(0x110100) == 'PERFORMANCE_COUNTER_BOUNDARY'
    assert _pci_category(0x110000) == 'SIGNAL_PROCESSING_INTERFACE'
    assert _pci_category(0x118000) == 'SIGNAL_PROCESSOR'
    assert _pci_category(0x400000) == 'COPROCESSOR'
    assert _pci_category(0x0b4000) == 'COPROCESSOR'
    assert _pci_category(0x100000) == 'CRYPTOGRAPHIC_PROCESSOR'
    assert _pci_category(0x040300) == 'MULTIMEDIA_PROCESSOR'
    class Sources:
        def add(self, *args, **kwargs): return 'fixture-source'
        def file(self, path, selected_fields=()):
            assert path == '/proc/self/status'
            return 'Cpus_allowed_list:\t2-3\nMems_allowed_list:\t0\n', 'fixture-status'
    class Probe:
        sources = Sources()
        def entries(self, path): return [], 'fixture-directory', True
        def file(self, path):
            assert path == '/proc/self/cgroup'
            return '0::/scope', 'fixture-cgroup'
        def fields(self, path, fields):
            values = {'cpu.max':'200000 100000','cpuset.cpus.effective':'2-3',
                      'cpuset.mems.effective':'0','memory.max':'4096','memory.high':'max'}
            return ({k:values[k] if str(path).endswith('/scope') else None for k in fields}, ['fixture-limit'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    probe = Probe()
    pci = {'devices':[dict(pci_bus_id='0000:01:00.0', category='DISPLAY', name='reported PCI device',
        driver=None, numa_node=0, source_ids=['fixture-pci'])]}
    candidate = _accelerators(probe, pci, [])[0]
    assert candidate['pci_bus_id'] == '0000:01:00.0'
    assert candidate['discovery_status'] == 'DISCOVERED_PCI_CLASS'
    assert candidate['legacy_gpu_record_index'] is None and candidate['memory_total_bytes'] is None
    assert candidate['accessibility_status'] == 'UNKNOWN' and candidate['backend_status'] == 'UNKNOWN_UNVALIDATED'
    pci['devices'] = [dict(pci['devices'][0], category=_pci_category(0x110100))]
    assert _accelerators(probe, pci, []) == []
    pci['devices'] = [dict(pci['devices'][0], category=_pci_category(0x110000))]
    assert _accelerators(probe, pci, []) == []
    original = Path.read_text
    def mountinfo(path, *args, **kwargs):
        return ('1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n' if str(path) == '/proc/self/mountinfo'
                else original(path, *args, **kwargs))
    monkeypatch.setattr(Path, 'read_text', mountinfo)
    monkeypatch.setattr('os.sched_getaffinity', lambda _: {2,3})
    limits = _constraints(probe)
    row = limits['cgroup_limits'][0]
    assert row['cpu_max'] == '200000 100000' and row['memory_max'] == '4096'
    assert row['memory_high'] == 'max' and row['parsed']['memory_high']['mode'] == 'UNLIMITED'
    assert row['parsed']['cpu_max']['quota_us'] == 200000
    assert limits['cgroup_limits'][-1]['parsed']['memory_max']['mode'] == 'UNKNOWN'
    assert limits['visibility_environment']['CUDA_VISIBLE_DEVICES'] == '-1'
    assert limits['hardware_filtered_by_execution_limits'] is False
    assert limits['limits_are_reserved_allocation_budget'] is False
