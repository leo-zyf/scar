"""Hardware coverage must survive task limits and absent compute drivers."""
import json

from scar.analysis.build_target_resources import target_kind_role
from scar.analysis.target_graph import build_target_graph,TargetHardwareGraph
from scar.ir.program import target_hardware as H
from scar.ir.program.resources import ResourceRole


def test_discovered_hardware_is_retained_and_typed_independently_of_access(tmp_path):
    inventory=dict(schema='scar.target-inventory',schema_version=1,sources=[],diagnostics=[],
        cpu=dict(online_cpu_ids=[0,1],model_names=['fixture'],
            processors=[dict(cpu_id=i,package_id=0,core_id=0) for i in (0,1)],
            numa_nodes=[dict(node_id=0,cpu_ids=[0,1],memory_total_bytes=2048)]),
        dram=dict(memory_total_bytes=2048),gpus=[],storage=dict(disks=[],block_devices=[]),paths=[],
        cpu_topology=dict(processors=[dict(cpu_id=i,online=True,package_id=0,core_id=0,
            die_id=0,thread_siblings=[0,1]) for i in (0,1)],
            isa_groups=[dict(cpu_ids=[0,1],flags=['sse2'])],caches=[dict(id='shared',level=2,
            type='Unified',shared_cpu_ids=[0,1],capacity_bytes=1024,line_size_bytes=64)]),
        pci=dict(devices=[dict(pci_bus_id='0000:01:00.0',planning_role='ENDPOINT',category='DISPLAY',
            class_code='0x030000',parent=None,numa_node=0),
            dict(pci_bus_id='0000:00:01.0',planning_role='BOUNDARY',category='CONTROL',parent=None)]),
        accelerators=[dict(id='pci:0000:01:00.0',family='DISPLAY',pci_bus_id='0000:01:00.0',
            legacy_gpu_record_index=None,discovery_status='DISCOVERED',visibility_status='SYSFS_VISIBLE',
            accessibility_status='UNKNOWN_UNPROBED',backend_status='UNKNOWN_UNVALIDATED')],
        network=dict(interfaces=[dict(name='eth0',physical=True,pci_bus_id=None)],rdma=[]),
        execution_constraints=dict(affinity_cpu_ids=[0],allowed_cpu_ids=[0],allowed_memory_nodes=[0],
            visibility_environment={'CUDA_VISIBLE_DEVICES':'-1'},cgroup_limits=[]))
    path=tmp_path/'inventory.json';path.write_text(json.dumps(inventory))
    report=build_target_graph(tmp_path/'target',inventory=path)
    graph=TargetHardwareGraph(tmp_path/'target');nodes=graph.resources();edges=graph.connections()
    assert report['schema_version']==2
    assert {n.cpu_id for n in nodes if type(n) is H.HardwareThread}=={0,1}
    assert len([n for n in nodes if type(n) is H.DiscoveredAccelerator])==1
    assert not any(type(n) is H.MemoryDomain and n.kind is H.MemoryKind.DEVICE for n in nodes)
    assert graph.constraints()[0].affinity_cpu_ids==(0,)
    assert graph.availability()[0].backend_status=='UNKNOWN_UNVALIDATED'
    assert report['platform_coverage']['pci_discovered_functions']==2
    assert report['platform_coverage']['missing_pci_functions']==[]
    assert all(e.bandwidth_bytes_per_second is None and e.latency_seconds is None for e in edges)
    assert any(e.kind is H.ConnectionKind.DEVICE_INTERFACE for e in edges)
    cache=next(n for n in nodes if type(n) is H.CacheDomain)
    assert target_kind_role(cache)[1] is ResourceRole.OPAQUE_RESOURCE
    assert all(target_kind_role(n) for n in nodes)
