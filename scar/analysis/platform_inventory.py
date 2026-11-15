"""Bounded current-platform metadata discovery, independent of any workload.

No device is removed by process affinity or framework visibility. All links are
reported topology/candidates, never observed transfers or allocation budgets.
Only selected sysfs hardware fields and this process's limits are inspected.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import resource
import shlex

from .target_inventory import _cpu_list, _integer, _pci_address


class _Probe:
    def __init__(self, sources):
        self.sources, self.files = sources, {}

    def file(self, path):
        path = str(path)
        if path not in self.files: self.files[path] = self.sources.file(path)
        text, ref = self.files[path]
        return (None if text is None else text.strip()), ref

    def fields(self, path, fields):
        values, refs = {}, []
        for field in fields:
            value, ref = self.file(Path(path) / field); values[field] = value; refs.append(ref)
        return values, refs

    def entries(self, path):
        path = Path(path)
        try:
            rows = sorted(path.iterdir(), key=lambda p:p.name)
            ref = self.sources.add('directory', json.dumps([p.name for p in rows]),
                path=str(path), status='observed', representation='entry_names_only')
            return rows, ref, True
        except OSError as error:
            ref = self.sources.add('directory', None, path=str(path), status='unavailable',
                error_type=type(error).__name__)
            return [], ref, False

    def link(self, path):
        path = Path(path)
        try:
            target = os.readlink(path)
            resolved = str(path.resolve())
            ref = self.sources.add('symlink', target, path=str(path), resolved_path=resolved, status='observed')
            return resolved, ref
        except OSError as error:
            ref = self.sources.add('symlink', None, path=str(path), status='unavailable',
                error_type=type(error).__name__)
            return None, ref


def _pci_in_path(path):
    return next((address for part in reversed(Path(path).parts)
                 if (address := _pci_address(part)) is not None), None) if path else None


def _nonnegative(text):
    value = _integer(text)
    return value if value is not None and value >= 0 else None


def _pci_category(code):
    if code is None: return 'OTHER_PCI_BOUNDARY'
    if code >> 8 == 0x1101: return 'PERFORMANCE_COUNTER_BOUNDARY'
    if code >> 8 == 0x1180: return 'SIGNAL_PROCESSOR'
    if code >> 8 == 0x0b40 or code >> 16 == 0x40: return 'COPROCESSOR'
    return {0x01:'STORAGE',0x02:'NETWORK',0x03:'DISPLAY',0x04:'MULTIMEDIA_PROCESSOR',
            0x05:'MEMORY',0x10:'CRYPTOGRAPHIC_PROCESSOR',0x11:'SIGNAL_PROCESSING_INTERFACE',
            0x12:'PROCESSING_ACCELERATOR'}.get(code >> 16,'OTHER_PCI_BOUNDARY')


def _pci_class_source(probe):
    path = '/usr/share/hwdata/pci.ids'
    selected = {'04','05','0b','10','11','12','40'}
    lines, numbers, keep = [], [], False
    try:
        with open(path, encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                match = re.match(r'^C ([0-9a-f]{2})  ', line)
                if match: keep = match[1] in selected
                if keep and line.strip() and not line.startswith('#'):
                    lines.append(line); numbers.append(number)
        return probe.sources.add('pci_class_definitions', ''.join(lines), path=path,
            status='observed', representation='selected_class_definitions',
            selected_class_codes=sorted(selected), line_numbers=numbers)
    except (OSError, UnicodeError) as error:
        return probe.sources.add('pci_class_definitions', None, path=path,
            status='unavailable', error_type=type(error).__name__)


def _cpu(probe, cpu):
    from .cpu_hierarchy import _numbered_dirs, _read, _receipts, _positive, _size
    root = Path('/sys/devices/system/cpu')
    values, refs = probe.fields(root, ('present', 'possible', 'online'))
    online = _cpu_list(values['online'])
    processors, observations, groups, gaps = [], [], defaultdict(list), []
    directories, directory_ref, discovered = probe.entries(root)
    refs.append(directory_ref)
    stats = dict(files_attempted=0, files_read=0, bytes_read=0)
    for directory in directories:
        if re.fullmatch(r'cpu[0-9]+', directory.name) is None: continue
        identity = int(directory.name[3:])
        fields, source_ids = probe.fields(directory / 'topology', ('physical_package_id', 'core_id',
            'die_id', 'thread_siblings_list', 'core_siblings_list'))
        processors.append(dict(cpu_id=identity, online=None if online is None else identity in online,
            package_id=_nonnegative(fields['physical_package_id']), core_id=_nonnegative(fields['core_id']),
            die_id=_nonnegative(fields['die_id']), thread_siblings=_cpu_list(fields['thread_siblings_list']),
            core_siblings=_cpu_list(fields['core_siblings_list']), source_ids=source_ids))
        indices = _numbered_dirs(directory / 'cache', 'index')
        if not indices: gaps.append(dict(path=str(directory / 'cache'), reason='NO_READABLE_CACHE_INDICES'))
        for index, path in indices:
            fields = {field:_read(path / field, stats) for field in
                      ('level','type','size','coherency_line_size','shared_cpu_list','id')}
            texts = {field:None if row['text'] is None else row['text'].strip() for field,row in fields.items()}
            shared, level, kind = _cpu_list(texts['shared_cpu_list']), _positive(texts['level']), texts['type']
            ordinal = len(observations)
            observations.append(dict(cpu_id=identity,index=index,path=str(path),_fields=fields))
            if shared and identity in shared and level is not None and kind in ('Data','Instruction','Unified'):
                groups[tuple(shared),level,kind].append(ordinal)
            else: gaps.append(dict(observation=ordinal,reason='CACHE_FIELDS_UNAVAILABLE_OR_INVALID'))
    caches = []
    for (shared,level,kind), ordinals in sorted(groups.items()):
        scalar = lambda field:{None if observations[i]['_fields'][field]['text'] is None else
                               observations[i]['_fields'][field]['text'].strip() for i in ordinals}
        def consensus(field, parse):
            values = scalar(field)
            return parse(next(iter(values))) if len(values)==1 else None
        caches.append(dict(id='cache:'+str(len(caches)),level=level,type=kind,shared_cpu_ids=list(shared),
            capacity_bytes=consensus('size',_size),line_size_bytes=consensus('coherency_line_size',_positive),
            sysfs_cache_ids=sorted(int(v) for v in scalar('id') if v is not None and v.isdecimal()),
            observations=ordinals,source_ids=[],allocatable=False,allocation_budget_bytes=None,effective_available_bytes=None))
    for receipt in _receipts(observations):
        source_id = probe.sources.add('cache_sysfs_files', receipt['text'],
            status='observed' if receipt['status']=='READ' else 'unavailable',field=receipt['field'],
            paths=[observations[i]['path']+'/'+receipt['field'] for i in receipt['observations']],
            original_size_bytes=receipt.get('size_bytes'),error_type=receipt.get('error_type'))
        seen = set(receipt['observations'])
        for cache in caches:
            if seen.intersection(cache['observations']): cache['source_ids'].append(source_id)
    flags, flag_source = probe.sources.file('/proc/cpuinfo', ('processor','flags','Features'))
    isa = defaultdict(list)
    for paragraph in (flags or '').strip().split('\n\n'):
        fields = dict(line.split(':',1) for line in paragraph.splitlines() if ':' in line)
        fields = {k.strip():v.strip() for k,v in fields.items()}; identity = _integer(fields.get('processor'))
        if identity is not None: isa[tuple(sorted((fields.get('flags') or fields.get('Features') or '').split()))].append(identity)
    distances = []
    nodes = sorted(row['node_id'] for row in cpu['numa_nodes'])
    for node in nodes:
        text, source = probe.file('/sys/devices/system/node/node'+str(node)+'/distance')
        row = None if text is None else [_integer(v) for v in text.split()]
        distances.append(dict(node_id=node,target_node_order=nodes if row is not None and len(row)==len(nodes) else None,
                              distances=row,source_ids=[source],units='RELATIVE_KERNEL_NUMA_DISTANCE_NOT_MEASURED_LATENCY'))
    return dict(present_cpu_ids=_cpu_list(values['present']),possible_cpu_ids=_cpu_list(values['possible']),
        online_cpu_ids=online,processors=processors,isa_groups=[dict(cpu_ids=ids,flags=list(flags),source_ids=[flag_source])
            for flags,ids in isa.items()],numa_distances=distances,caches=caches,cache_observations=observations,
        cache_collection=stats,cache_gaps=gaps,source_ids=refs,
        coverage=dict(cpu_directory_read=discovered,scope='VISIBLE_SYSFS_ALL_CPU_DIRECTORIES_NOT_PROCESS_AFFINITY',
                      cache_refill_inclusion_and_available_capacity_proven=False))


def _pci(probe):
    root = Path('/sys/bus/pci/devices'); entries, directory_ref, observed = probe.entries(root)
    class_ref = _pci_class_source(probe)
    text, name_ref = probe.sources.command(['lspci','-D','-mm','-nn'])
    names = {}
    for line in (text or '').splitlines():
        try: fields = shlex.split(line)
        except ValueError: continue
        if len(fields)>=4 and _pci_address(fields[0]): names[_pci_address(fields[0])] = fields[2]+' '+fields[3]
    rows = []
    for entry in entries:
        address = _pci_address(entry.name)
        if address is None: continue
        fields, source_ids = probe.fields(entry, ('class','vendor','device'))
        try: code = int(fields['class'],16)
        except (TypeError,ValueError): code = None
        category = _pci_category(code)
        resolved = str(entry.resolve()); parts = [_pci_address(p) for p in Path(resolved).parts]
        chain = [p for p in parts if p is not None]
        complex_name = next((p for p in Path(resolved).parts if re.fullmatch(r'pci[0-9a-f]{4}:[0-9a-f]{2}',p)),None)
        rows.append(dict(pci_bus_id=address,class_code=fields['class'],vendor_id=fields['vendor'],device_id=fields['device'],
            name=names.get(address),sysfs_device_path=resolved,parent=chain[-2] if len(chain)>1 else None,
            root_complex=complex_name,planning_role='BOUNDARY' if category in
                ('OTHER_PCI_BOUNDARY','PERFORMANCE_COUNTER_BOUNDARY') else 'ENDPOINT',
            category=category,driver=None,numa_node=None,local_cpu_ids=None,pcie=None,
            source_ids=[directory_ref,name_ref,class_ref,*source_ids]))
    mapping = {row['pci_bus_id']:row for row in rows}
    for row in rows:
        if row['planning_role']!='ENDPOINT': continue
        parent = row['parent']; visited=set()
        while parent in mapping and parent not in visited:
            visited.add(parent); mapping[parent]['planning_role']='FABRIC'; parent=mapping[parent]['parent']
    for row in rows:
        if row['planning_role']=='BOUNDARY': continue
        path = root / row['pci_bus_id']; fields, source_ids = probe.fields(path,
            ('numa_node','local_cpulist','current_link_speed','current_link_width','max_link_speed','max_link_width'))
        driver, driver_ref = probe.link(path/'driver')
        row.update(driver=Path(driver).name if driver else None,numa_node=_nonnegative(fields['numa_node']),
            local_cpu_ids=_cpu_list(fields['local_cpulist']),pcie=dict(
                current_link_speed=fields['current_link_speed'],current_link_width=_nonnegative(fields['current_link_width']),
                max_link_speed=fields['max_link_speed'],max_link_width=_nonnegative(fields['max_link_width']),
                bandwidth_bytes_per_second=None,latency_seconds=None,measured=False))
        row['source_ids'].extend([*source_ids,driver_ref])
    links = [dict(source=row['parent'],target=row['pci_bus_id'],kind='PCI_PARENT_CANDIDATE',source_ids=row['source_ids'])
             for row in rows if row['planning_role']!='BOUNDARY' and row['parent'] is not None]
    return dict(devices=rows,links=links,coverage=dict(directory_read=observed,
        scope='ALL_VISIBLE_PCI_FUNCTIONS_CORE_ENDPOINTS_AND_REQUIRED_ANCESTORS_ONLY',hidden_devices_unknown=True),
        source_ids=[directory_ref,name_ref,class_ref])


def _accelerators(probe, pci, gpus):
    result = {}; legacy={row['pci_bus_id']:(i,row) for i,row in enumerate(gpus) if row['pci_bus_id'] is not None}
    for device in pci['devices']:
        if device['category'] not in ('DISPLAY','PROCESSING_ACCELERATOR','COPROCESSOR',
                'MULTIMEDIA_PROCESSOR','CRYPTOGRAPHIC_PROCESSOR','SIGNAL_PROCESSOR'): continue
        address = device['pci_bus_id']; index,gpu = legacy.get(address,(None,{}))
        result[address] = dict(id='pci:'+address,pci_bus_id=address,family=device['category'],
            name=device['name'],driver=device['driver'],numa_node=device['numa_node'],
            memory_total_bytes=gpu.get('memory_total_bytes'),legacy_gpu_record_index=index,
            management_gpu_index=gpu.get('index'),capabilities=dict(compute_capability=gpu.get('compute_capability'),
                management_interface='NVIDIA_SMI' if gpu else None,execution_backend=None),
            discovery_status='DISCOVERED_PCI_CLASS',visibility_status='VISIBLE_IN_SYSFS',
            accessibility_status='MANAGEMENT_QUERY_SUCCEEDED_NOT_COMPUTE_PROOF' if gpu else 'UNKNOWN',
            backend_status='UNKNOWN_UNVALIDATED',interfaces=[],source_ids=list(device['source_ids']))
    for directory,pattern,family in (('/sys/class/drm',r'card[0-9]+','DRM_DISPLAY'),
                                      ('/sys/class/accel',r'accel[0-9]+','ACCELERATOR_CLASS')):
        entries, source, _ = probe.entries(directory)
        for entry in entries:
            if re.fullmatch(pattern,entry.name) is None: continue
            path, ref = probe.link(entry/'device'); address = _pci_in_path(path); identity=address or family+':'+entry.name
            if identity not in result:
                result[identity]=dict(id='sysfs:'+identity,pci_bus_id=address,family=family,name=None,driver=None,numa_node=None,
                    memory_total_bytes=None,legacy_gpu_record_index=None,management_gpu_index=None,
                    capabilities=dict(compute_capability=None,management_interface=None,execution_backend=None),
                    discovery_status='DISCOVERED_DEVICE_CLASS',visibility_status='VISIBLE_IN_SYSFS',accessibility_status='UNKNOWN',
                    backend_status='UNKNOWN_UNVALIDATED',interfaces=[],source_ids=[])
            result[identity]['interfaces'].append(str(entry));result[identity]['source_ids'].extend([source,ref])
    return list(result.values())


def _network(probe):
    interfaces, adapters = [], []
    entries, source, observed = probe.entries('/sys/class/net')
    for entry in entries:
        physical = (entry/'device').exists(); path, ref = probe.link(entry/'device') if physical else (None,source)
        source_ids = [source,ref]; fields = {}
        if physical:
            fields, ids = probe.fields(entry,('operstate','speed','duplex','mtu','dev_port','phys_port_name'));source_ids.extend(ids)
            numa, nr = probe.file(entry/'device/numa_node');source_ids.append(nr)
        else: numa=None
        lower = sorted(p.name.removeprefix('lower_') for p in entry.glob('lower_*'))
        interfaces.append(dict(name=entry.name,physical=physical,pci_bus_id=_pci_in_path(path),device_path=path,
            numa_node=_nonnegative(numa),operstate=fields.get('operstate'),link_rate_mbps=_nonnegative(fields.get('speed')),
            duplex=fields.get('duplex'),mtu=_nonnegative(fields.get('mtu')),dev_port=_nonnegative(fields.get('dev_port')),
            phys_port_name=fields.get('phys_port_name'),lower_interfaces=lower,source_ids=source_ids,
            rate_is_advertised_not_measured=True,accessibility_status='UNKNOWN',backend_status='UNKNOWN_UNVALIDATED'))
    entries, rdma_source, rdma_observed = probe.entries('/sys/class/infiniband')
    for entry in entries:
        path, ref = probe.link(entry/'device'); ports, pr, port_observed=probe.entries(entry/'ports');rows=[]
        for port in ports:
            if not port.name.isdecimal():continue
            fields, ids = probe.fields(port,('state','phys_state','rate','link_layer'))
            rows.append(dict(port=int(port.name),state=fields['state'],physical_state=fields['phys_state'],
                rate=fields['rate'],link_layer=fields['link_layer'],source_ids=ids))
        adapters.append(dict(name=entry.name,pci_bus_id=_pci_in_path(path),device_path=path,ports=rows,
            source_ids=[rdma_source,ref,pr],ports_discovered=port_observed,accessibility_status='UNKNOWN',backend_status='UNKNOWN_UNVALIDATED'))
    return dict(interfaces=interfaces,rdma=adapters,source_ids=[source,rdma_source],
        coverage=dict(net_directory_read=observed,rdma_directory_read=rdma_observed,hidden_devices_unknown=True))


def _storage(probe, storage):
    root=Path('/sys/class/block');entries,source,observed=probe.entries(root);rows=[]
    for entry in entries:
        resolved=str(entry.resolve()); slave_entries,sr,_=probe.entries(entry/'slaves')
        fields, refs=probe.fields(entry,('queue/logical_block_size','queue/physical_block_size'))
        parent=Path(resolved).parent.name if (entry/'partition').exists() else None
        path,dr=probe.link(entry/'device') if (entry/'device').exists() else (None,source)
        numa,nr=probe.file(entry/'device/numa_node') if path else (None,source)
        rows.append(dict(kname=entry.name,sysfs_device_path=path or resolved,pci_bus_id=_pci_in_path(path or resolved),
            numa_node=_nonnegative(numa),partition_parent=parent,slaves=[p.name for p in slave_entries],
            logical_block_size_bytes=_nonnegative(fields['queue/logical_block_size']),
            physical_block_size_bytes=_nonnegative(fields['queue/physical_block_size']),source_ids=[source,sr,dr,nr,*refs]))
    mapping={r['kname']:r for r in rows};physical={r['kname'] for r in storage['disks']}
    def backing(name,active=()):
        if name in active:return []
        if name in physical:return [name]
        row=mapping.get(name)
        return [] if row is None else sorted({disk for base in (*row['slaves'], *((row['partition_parent'],) if row['partition_parent'] else ()))
            for disk in backing(base,(*active,name))})
    for row in rows:row['physical_backing_candidates']=backing(row['kname'])
    for disk in storage['disks']:
        row=mapping.get(disk['kname'])
        if row:
            for key in ('sysfs_device_path','pci_bus_id','numa_node'):disk[key]=row[key]
            disk['source_ids']=list(dict.fromkeys([*disk['source_ids'],*row['source_ids']]))
    storage.update(block_topology=rows,block_topology_coverage=dict(directory_read=observed,
        physical_extents_or_io_observed=False),logical_backing_is_candidate=True)


def _limit(text):
    return dict(mode='UNKNOWN',bytes=None) if text is None else dict(mode='UNLIMITED',bytes=None) if text=='max' else dict(mode='LIMIT',bytes=_nonnegative(text))


def _constraints(probe):
    status, status_ref = probe.sources.file('/proc/self/status', ('Cpus_allowed_list','Mems_allowed_list'))
    status_fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (status or '').splitlines() if ':' in line}
    cgroup,cgroup_ref=probe.file('/proc/self/cgroup')
    try:
        lines=Path('/proc/self/mountinfo').read_text().splitlines()
        selected=[line for line in lines if ' - cgroup ' in line or ' - cgroup2 ' in line]
        mount_ref=probe.sources.add('file','\n'.join(selected)+'\n',path='/proc/self/mountinfo',
            status='observed',representation='selected_cgroup_mount_lines')
    except OSError as error:
        selected=[];mount_ref=probe.sources.add('file',None,path='/proc/self/mountinfo',status='unavailable',error_type=type(error).__name__)
    limits=[];mode='UNKNOWN';group_path=None;source_ids=[status_ref,cgroup_ref,mount_ref]
    for line in selected:
        head,tail=line.split(' - ',1);fields=head.split();provider=tail.split()
        if provider[0]!='cgroup2':continue
        mode='CGROUP_V2'; mount=Path(fields[4]);mount_root=fields[3]
        group_path=next((line.split(':',2)[2] for line in (cgroup or '').splitlines() if line.startswith('0::')),None)
        if group_path is None:continue
        try:relative=Path(group_path).relative_to(mount_root)
        except ValueError:continue
        current=mount/relative
        for _ in range(64):
            values,refs=probe.fields(current,('cpu.max','cpuset.cpus.effective','cpuset.mems.effective','memory.max','memory.high'))
            parts=(values['cpu.max'] or '').split()
            quota=dict(mode='UNKNOWN',quota_us=None,period_us=None)
            if len(parts)==2:quota=dict(mode='UNLIMITED' if parts[0]=='max' else 'QUOTA',
                quota_us=None if parts[0]=='max' else _nonnegative(parts[0]),period_us=_nonnegative(parts[1]))
            limits.append(dict(path=str(current),cpu_max=values['cpu.max'],
                cpuset_cpus=values['cpuset.cpus.effective'],cpuset_mems=values['cpuset.mems.effective'],
                memory_max=values['memory.max'],memory_high=values['memory.high'],source_ids=refs,
                parsed=dict(cpu_max=quota,cpuset_cpus=_cpu_list(values['cpuset.cpus.effective']),
                    cpuset_mems=_cpu_list(values['cpuset.mems.effective']),memory_max=_limit(values['memory.max']),
                    memory_high=_limit(values['memory.high']))))
            if current==mount:break
            current=current.parent
    if any(' - cgroup ' in line for line in selected) and mode=='UNKNOWN':mode='CGROUP_V1_UNRESOLVED_CONTROLLER_LIMITS'
    try:affinity=sorted(os.sched_getaffinity(0))
    except (AttributeError,OSError):affinity=None
    rlimits={name:list(resource.getrlimit(getattr(resource,name))) for name in ('RLIMIT_AS','RLIMIT_DATA','RLIMIT_CPU')}
    own_ref=probe.sources.add('self_process_limits',json.dumps(dict(affinity_cpu_ids=affinity,rlimits=rlimits),sort_keys=True),
        status='observed',scope='QUERYING_PROCESS_ONLY')
    environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES',
        'HIP_VISIBLE_DEVICES','ZE_AFFINITY_MASK','ONEAPI_DEVICE_SELECTOR')}
    environment_ref=probe.sources.add('visibility_environment',json.dumps(environment,sort_keys=True),status='observed',
        scope='QUERYING_PROCESS_ONLY_NOT_A_HARDWARE_DISCOVERY_FILTER')
    devices=[]
    candidates=[*Path('/dev').glob('nvidia*'),*Path('/dev/dri').glob('renderD*'),*Path('/dev/accel').glob('accel*')]
    for path in sorted(candidates):
        if not path.is_char_device():continue
        devices.append(dict(path=str(path),readable_by_querying_process=os.access(path,os.R_OK),
            writable_by_querying_process=os.access(path,os.W_OK),functional_access_proven=False))
    device_ref=probe.sources.add('device_node_metadata',json.dumps(devices,sort_keys=True),status='observed',
        scope='VISIBLE_ACCELERATOR_DEVICE_NODE_PERMISSIONS_ONLY_NO_OPEN_OR_EXECUTION')
    return dict(affinity_cpu_ids=affinity,allowed_cpu_ids=_cpu_list(status_fields.get('Cpus_allowed_list')),
        allowed_memory_nodes=_cpu_list(status_fields.get('Mems_allowed_list')),cgroup_mode=mode,cgroup_path=group_path,
        cgroup_limits=limits,rlimits=rlimits,visibility_environment=environment,device_nodes=devices,
        source_ids=[*source_ids,own_ref,environment_ref,device_ref],hardware_filtered_by_execution_limits=False,
        limits_are_reserved_allocation_budget=False,functional_device_access_proven=False,
        unknown_limit_interfaces_remain_unknown=True)


def collect_platform_inventory(sources, cpu, dram, gpus, storage):
    probe=_Probe(sources)
    topology=_cpu(probe,cpu);pci=_pci(probe);accelerators=_accelerators(probe,pci,gpus)
    _storage(probe,storage)
    return dict(cpu_topology=topology,pci=pci,accelerators=accelerators,network=_network(probe),
        execution_constraints=_constraints(probe),discovery_scope=dict(
            platform='CURRENT_VISIBLE_SYSFS_PROC_SELF_AND_PROVIDER_TOOLS',framework_imported=False,
            historical_inventory_reused=False,visibility_environment_filters_hardware=False,
            discovered_not_accessible_not_backend_supported=True,unseen_or_failed_interfaces_are_unknown=True,
            capacities_are_not_available_budgets=True,bandwidth_or_latency_measured=False))
