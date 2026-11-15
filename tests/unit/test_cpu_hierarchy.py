from hashlib import sha256
import json

from scar.analysis.cpu_hierarchy import inspect_cpu_hierarchy
from scar.analysis.target_graph import build_target_graph
from scar.ir.program.target_hardware import TargetResourceRef
from scar.ir.record_codec import decode


def test_cache_sharing_receipts_and_conditional_original_target_join(tmp_path, monkeypatch):
    inventory = dict(schema='scar.target-inventory', schema_version=1, sources=[], diagnostics=[],
        cpu=dict(online_cpu_ids=[0, 1, 2, 3], logical_cpu_count=4, model_names=['synthetic'],
            processors=[dict(cpu_id=cpu, package_id=cpu // 2, core_id=0) for cpu in range(4)],
            numa_nodes=[dict(node_id=node, cpu_ids=[2 * node, 2 * node + 1], memory_total_bytes=1024)
                        for node in range(2)]),
        dram=dict(memory_total_bytes=2048), gpus=[], storage=dict(disks=[], block_devices=[]), paths=[])
    inventory_path = tmp_path / 'inventory.json'
    inventory_path.write_text(json.dumps(inventory))
    target = tmp_path / 'target'
    build_target_graph(target, inventory=inventory_path)
    root = tmp_path / 'cpu'
    for cpu in range(4):
        for index in range(2):
            cache = root / ('cpu' + str(cpu)) / 'cache' / ('index' + str(index))
            cache.mkdir(parents=True)
            fields = dict(level=str(index + 1), type='Data' if index == 0 else 'Unified',
                size='32K' if index == 0 else '256K', coherency_line_size='64',
                shared_cpu_list=('0-1' if cpu < 2 else '2-3') if index == 0 else '0-3',
                id=str(cpu // 2) if index == 0 else '0')
            # A conflicting replica must not turn into a guessed line size.
            if cpu == 1 and index == 0:
                fields['coherency_line_size'] = '128'
            for field, value in fields.items():
                (cache / field).write_text(value + '\n')
            (cache / 'unrelated').write_text('must not read this')
    monkeypatch.setattr('scar.analysis.cpu_hierarchy.os.sched_getaffinity', lambda _: {0})
    report = inspect_cpu_hierarchy(target / 'result.json', sysfs_root=root)
    assert report['schema'] == 'scar.cpu-hierarchy'
    assert report['collection']['files_read'] == 48
    assert report['collection']['cache_index_observations'] == 8
    assert report['collection']['deduplicated_cache_nodes'] == 3
    assert report['affinity']['cpu_ids'] == [0]
    assert len(report['executors']) == 2  # Affinity never deletes the other NUMA node.
    assert [e['core_smt_groups'][0]['cpu_ids'] for e in report['executors']] == [[0, 1], [2, 3]]
    for executor in report['executors']:
        reference = decode(TargetResourceRef, executor['target'])
        assert reference.snapshot.sha256 == report['target_snapshot']['sha256']
        assert executor['physical_cores'] == 1
        assert len(executor['host_memory_refs']) == 1
    assert report['target_join']['same_machine_proven'] is False
    assert report['target_join']['status'] == 'CPU_ID_SET_MATCHES_NOT_MACHINE_IDENTITY'
    assert report['unknowns']['target_snapshot_timestamp'] is True
    assert report['cache_is_allocatable_storage'] is False
    conflict = next(c for c in report['caches'] if c['shared_cpu_ids'] == [0, 1])
    assert conflict['capacity_bytes'] == 32768
    assert conflict['line_size_bytes'] is None
    assert any(g['field'].endswith('.line_size_bytes') for g in report['gaps'])
    shared = next(c for c in report['caches'] if c['level'] == 2)
    assert len(shared['observations']) == 4
    assert len(shared['source_receipts']) == 6  # One receipt per raw attribute, four paths.
    for cache in report['caches']:
        assert cache['allocatable'] is False and cache['allocation_budget_bytes'] is None
        for receipt in cache['source_receipts']:
            assert sha256(receipt['text'].encode()).hexdigest() == receipt['sha256']
            assert len(receipt['text'].encode()) == receipt['size_bytes']
            for index in receipt['observations']:
                path = cache['observations'][index]['path'] + '/' + receipt['field']
                assert sha256(open(path, 'rb').read()).hexdigest() == receipt['sha256']
    assert len([e for e in report['relations'] if e['kind'] == 'SHARES_CACHE']) == 4
    refill = [e for e in report['relations'] if e['kind'] == 'REFILL_CANDIDATE']
    assert len(refill) == 4
    assert all('CURRENT_SYSFS_CPU_NUMBERING_MATCHES_RETAINED_TARGET' in e['conditions']
               for e in report['relations'])
    assert report['physical_use_established'] is False and report['transfers_observed'] is False
