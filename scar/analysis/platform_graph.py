"""Project retained platform facts into shared, typed hardware topology.

No live reads, allocation budgets, bandwidth estimates or backend admission.
Control/register-only PCI functions remain grouped, referenceable boundaries.
"""
from collections import defaultdict

from scar.ir.program import target_hardware as H
from scar.ir.program.providers import ProviderFactReference


def platform_resources(document, artifact, existing):
    nodes, links = [], []
    evidence = lambda pointer: ProviderFactReference(artifact, pointer)
    identity = lambda value: H.TargetResourceID(value)
    seq = lambda value: None if value is None else tuple(value)

    def connect(a, b, kind, pointer, *, conditions=(), speed=None, width=None, locality=None):
        if a != b:
            links.append(H.ResourceConnection(a,b,kind,locality,None,None,speed,width,
                evidence(pointer),tuple(conditions),False))

    executors = [r for r in existing if type(r) is H.CpuExecutor]
    memories = [r for r in existing if type(r) is H.MemoryDomain and r.kind is H.MemoryKind.HOST]
    topology = document.get('cpu_topology', {})
    rows = topology.get('processors', ())
    flags = {cpu:(tuple(g['flags']) if g.get('flags') is not None else None,
        evidence('#/cpu_topology/isa_groups/'+str(i)))
        for i,g in enumerate(topology.get('isa_groups', ())) for cpu in g['cpu_ids']}
    packages, cores, threads = defaultdict(list), defaultdict(list), {}
    for i, row in enumerate(rows):
        cpu = row['cpu_id']; pointer = '#/cpu_topology/processors/'+str(i)
        tid = identity('cpu-thread/'+str(cpu)); threads[cpu] = tid
        isa,isa_evidence = flags.get(cpu,(None,None))
        nodes.append(H.HardwareThread(tid,cpu,row.get('online'),seq(row.get('thread_siblings')),
            isa,isa_evidence,evidence(pointer)))
        if row.get('package_id') is not None:
            packages[row['package_id']].append(cpu)
            if row.get('core_id') is not None:
                cores[row['package_id'],row.get('die_id'),row['core_id']].append(cpu)
        for executor in executors:
            if cpu in executor.cpu_ids:
                connect(executor.id,tid,H.ConnectionKind.CONTAINS,pointer)
    for package, cpu_ids in sorted(packages.items()):
        pid = identity('cpu-package/'+str(package))
        nodes.append(H.CpuPackage(pid,package,tuple(sorted(cpu_ids)),evidence('#/cpu_topology/processors')))
    for (package, die, core), cpu_ids in sorted(cores.items(),key=lambda item:str(item[0])):
        cid = identity('cpu-core/'+str(package)+'/'+str(die)+'/'+str(core))
        nodes.append(H.CpuCore(cid,package,die,core,tuple(sorted(cpu_ids)),evidence('#/cpu_topology/processors')))
        connect(identity('cpu-package/'+str(package)),cid,H.ConnectionKind.CONTAINS,'#/cpu_topology/processors')
        for cpu in cpu_ids:
            connect(cid,threads[cpu],H.ConnectionKind.CONTAINS,'#/cpu_topology/processors')
    caches = []
    for i, row in enumerate(topology.get('caches', ())):
        pointer = '#/cpu_topology/caches/'+str(i)
        cache = H.CacheDomain(identity('cache/'+row['id']),row['level'],row['type'],
            tuple(row['shared_cpu_ids']),row['capacity_bytes'],row['line_size_bytes'],evidence(pointer))
        caches.append((cache,pointer)); nodes.append(cache)
        for cpu in cache.shared_cpu_ids:
            if cpu in threads:
                connect(threads[cpu],cache.id,H.ConnectionKind.SHARES_CACHE,pointer)
    for cache,pointer in caches:
        sharing = set(cache.shared_cpu_ids)
        higher = [c for c,_ in caches if c.level > cache.level
            and c.cache_type in ('Unified',cache.cache_type) and sharing.issubset(c.shared_cpu_ids)]
        nearest = min((c.level for c in higher),default=None)
        destinations = [c.id for c in higher if c.level == nearest] if higher else [m.id for m in memories
            if any(e.id == m.owner and sharing.intersection(e.cpu_ids) for e in executors)]
        for destination in destinations:
            connect(cache.id,destination,H.ConnectionKind.REFILL_CANDIDATE,pointer,
                conditions=('LEVEL_AND_SHARING_DO_NOT_PROVE_REFILL_PATH_OR_INCLUSION',))

    pci_nodes, boundary_groups = {}, defaultdict(list)
    pci_rows = document.get('pci',{}).get('devices',())
    for i,row in enumerate(pci_rows):
        pointer = '#/pci/devices/'+str(i); bdf = row['pci_bus_id']
        if row['planning_role'] == 'BOUNDARY':
            boundary_groups[row['category']].append(bdf)
            continue
        pid = identity('pci/'+bdf); pci_nodes[bdf] = pid
        pcie = row.get('pcie') or {}
        nodes.append(H.PciFunction(pid,bdf,row.get('class_code'),row.get('vendor_id'),row.get('device_id'),
            row['category'],row['planning_role'],row.get('driver'),row.get('numa_node'),
            seq(row.get('local_cpu_ids')),pcie.get('current_link_speed'),pcie.get('current_link_width'),
            pcie.get('max_link_speed'),pcie.get('max_link_width'),evidence(pointer)))
    for category,bdfs in sorted(boundary_groups.items()):
        bid = identity('hardware-boundary/pci/'+category)
        reason = ('UNCLASSIFIED_CAPABILITIES_RETAINED_NOT_A_COMPUTE_IMPLEMENTATION'
            if category == 'OTHER_PCI_BOUNDARY' else 'CONTROL_OR_MONITOR_INTERFACE_NOT_SCHEDULED_AS_PROGRAM_COMPUTE')
        nodes.append(H.HardwareBoundary(bid,category,tuple(sorted(bdfs)),
            reason,evidence('#/pci/devices')))
        pci_nodes.update((bdf,bid) for bdf in bdfs)
    # Sysfs parent edges retain physical topology; no transfer is implied.
    for i,row in enumerate(pci_rows):
        pointer = '#/pci/devices/'+str(i); child = pci_nodes[row['pci_bus_id']]
        parent = pci_nodes.get(row.get('parent'))
        pcie = row.get('pcie') or {}
        if parent is not None:
            connect(parent,child,H.ConnectionKind.PCI_PARENT,pointer,
                speed=pcie.get('max_link_speed'),width=pcie.get('max_link_width'),
                conditions=('TOPOLOGY_NOT_MEASURED_LINK_BANDWIDTH_OR_TRANSFER',))
        elif row['planning_role'] != 'BOUNDARY':
            for executor in executors:
                if row.get('numa_node') in (None,-1,executor.numa_node):
                    connect(executor.id,child,H.ConnectionKind.FABRIC_ACCESS,pointer,
                        locality='local' if executor.numa_node == row.get('numa_node') else None,
                        conditions=('PCI_LOCALITY_NOT_COMPLETE_ROOT_COMPLEX_OR_LINK_TIMING',))

    legacy_gpus = document.get('gpus',())
    for i,row in enumerate(document.get('accelerators',())):
        pointer = '#/accelerators/'+str(i); legacy = row.get('legacy_gpu_record_index')
        if legacy is not None:
            aid = identity('accelerator/'+legacy_gpus[legacy]['uuid'])
        else:
            aid = identity('accelerator/'+row['id'])
            nodes.append(H.DiscoveredAccelerator(aid,row['family'],row.get('name'),row.get('pci_bus_id'),
                row.get('driver'),row.get('numa_node'),(row.get('capabilities') or {}).get('compute_capability'),
                row.get('memory_total_bytes'),evidence(pointer)))
            # An accelerator can use shared host memory. Discovery alone is
            # not evidence of a separate physical VRAM/HBM memory domain.
            if row.get('memory_total_bytes') is not None:
                mid = identity('memory/'+row['id'])
                nodes.append(H.MemoryDomain(mid,H.MemoryKind.DEVICE,aid,row.get('numa_node'),
                    row['memory_total_bytes'],None,None,evidence(pointer)))
                connect(mid,aid,H.ConnectionKind.MEMORY_ATTACHED,pointer)
        if row.get('pci_bus_id') in pci_nodes:
            connect(pci_nodes[row['pci_bus_id']],aid,H.ConnectionKind.DEVICE_INTERFACE,pointer)

    for i,row in enumerate(document.get('storage',{}).get('disks',())):
        if row.get('pci_bus_id') in pci_nodes:
            connect(pci_nodes[row['pci_bus_id']],identity('storage/'+row['kname']),
                H.ConnectionKind.DEVICE_INTERFACE,'#/storage/disks/'+str(i))
    network = document.get('network',{})
    net_ids = {r['name']:identity('network/'+r['name']) for r in network.get('interfaces',())}
    for i,row in enumerate(network.get('interfaces',())):
        pointer = '#/network/interfaces/'+str(i); nid = net_ids[row['name']]
        nodes.append(H.NetworkInterface(nid,row['name'],row['physical'],row.get('pci_bus_id'),
            row.get('numa_node'),row.get('operstate'),row.get('link_rate_mbps'),row.get('mtu'),evidence(pointer)))
        if row.get('pci_bus_id') in pci_nodes:
            connect(pci_nodes[row['pci_bus_id']],nid,H.ConnectionKind.DEVICE_INTERFACE,pointer)
        for lower in row.get('lower_interfaces',()):
            if lower in net_ids:
                connect(nid,net_ids[lower],H.ConnectionKind.NETWORK_LOWER,pointer)
    for i,row in enumerate(network.get('rdma',())):
        pointer = '#/network/rdma/'+str(i); rid = identity('rdma/'+row['name'])
        ports = tuple(H.RdmaPort(p['port'],p.get('state'),p.get('physical_state'),p.get('rate'),
            p.get('link_layer')) for p in row.get('ports',()))
        nodes.append(H.RdmaAdapter(rid,row['name'],row.get('pci_bus_id'),ports,evidence(pointer)))
        if row.get('pci_bus_id') in pci_nodes:
            connect(pci_nodes[row['pci_bus_id']],rid,H.ConnectionKind.DEVICE_INTERFACE,pointer)

    return nodes,list(dict.fromkeys(links))


def platform_metadata(document, artifact):
    evidence = lambda pointer: ProviderFactReference(artifact,pointer)
    seq = lambda value: None if value is None else tuple(value)
    availability, constraints = [], []
    for i,row in enumerate(document.get('accelerators',())):
        legacy = row.get('legacy_gpu_record_index')
        key = 'accelerator/'+(document['gpus'][legacy]['uuid'] if legacy is not None else row['id'])
        availability.append(H.ResourceAvailability(H.TargetResourceID(key),row['discovery_status'],row['visibility_status'],
            row['accessibility_status'],row['backend_status'],evidence('#/accelerators/'+str(i))))
    if 'execution_constraints' in document:
        row = document['execution_constraints']
        limits = tuple(H.CgroupLimit(p['path'],p.get('cpu_max'),p.get('cpuset_cpus'),p.get('cpuset_mems'),
            p.get('memory_max'),p.get('memory_high')) for p in row.get('cgroup_limits',()))
        environment = tuple(sorted(row.get('visibility_environment',{}).items()))
        constraints.append(H.ExecutionConstraints(seq(row.get('affinity_cpu_ids')),seq(row.get('allowed_cpu_ids')),
            seq(row.get('allowed_memory_nodes')),row.get('cgroup_mode'),row.get('cgroup_path'),limits,
            environment,evidence('#/execution_constraints')))
    return availability,constraints


def platform_coverage(document, nodes):
    """Coverage is relative to discovered namespaces, never all hidden hardware."""
    represented = {r.pci_bus_id for r in nodes if type(r) is H.PciFunction}
    represented.update(bdf for r in nodes if type(r) is H.HardwareBoundary for bdf in r.pci_bus_ids)
    discovered = {r['pci_bus_id'] for r in document.get('pci',{}).get('devices',())}
    return dict(scope='CURRENT_OS_VISIBLE_PROVIDER_NAMESPACES',
        observed_at=document.get('observed_at'),observation_window=document.get('observation_window'),
        inventory_filters_by_workload_or_device_visibility=False,
        pci_discovered_functions=len(discovered),pci_represented_functions=len(represented),
        missing_pci_functions=sorted(discovered-represented),
        cpu_topology=document.get('cpu_topology',{}).get('coverage'),
        pci=document.get('pci',{}).get('coverage'),network=document.get('network',{}).get('coverage'),
        hidden_or_unexposed_physical_hardware_known=False,
        cache_capacity_is_allocation_budget=False,link_specs_are_measured_bandwidth=False,
        inventory_implies_backend_support=False)
