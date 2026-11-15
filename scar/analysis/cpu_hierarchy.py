"""Current Linux cache topology, conditionally joined to a retained CPU target.

Cache capacities are shared reported totals, not allocatable memory or budgets.
Level/sharing relationships suggest refill candidates; they prove neither an
inclusive hierarchy nor traffic, latency, bandwidth, placement, or cache misses.
"""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time

from scar.analysis.target_graph import TargetHardwareGraph, _resources
from scar.analysis.target_inventory import _cpu_list
from scar.ir.program import target_hardware as H
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode


SCHEMA = 'scar.cpu-hierarchy'
_FIELDS = ('level', 'type', 'size', 'coherency_line_size', 'shared_cpu_list', 'id')
_JOIN = 'CURRENT_SYSFS_CPU_NUMBERING_MATCHES_RETAINED_TARGET'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _artifact(path, schema):
    raw = path.read_bytes()
    return ProviderArtifact(str(path), sha256(raw).hexdigest(), len(raw), schema)


def _target(path):
    path = Path(path).resolve()
    if path.name == 'result.json':
        path = path.parent
    snapshot = _artifact(path / 'result.json', 'scar.target-hardware-graph')
    graph = TargetHardwareGraph(path)
    inventory = _artifact(Path(graph.report['inventory']['uri']), 'scar.target-inventory')
    for source in graph.inventory['sources']:
        if source['text'] is not None and sha256(source['text'].encode()).hexdigest() != source['sha256']:
            raise ValueError('retained target source receipt differs')
    resources = graph.resources()
    expected, _, _ = _resources(graph.inventory, inventory)
    if (len({r.id for r in resources}) != len(resources)
            or {r.id: r for r in resources} != {r.id: r for r in expected}):
        raise ValueError('target resource records differ from retained inventory')
    if snapshot != _artifact(path / 'result.json', snapshot.schema):
        raise ValueError('target snapshot changed while reading')
    return graph, snapshot, inventory, resources


def _positive(text):
    return int(text) if text is not None and re.fullmatch(r'[0-9]+', text) and int(text) > 0 else None


def _size(text):
    match = re.fullmatch(r'([0-9]+)([KMG]?)(?:B)?', text or '')
    return int(match[1]) * 1024 ** ('', 'K', 'M', 'G').index(match[2]) if match else None


def _read(path, stats):
    stats['files_attempted'] += 1
    try:
        with path.open('rb') as stream:
            raw = stream.read(4097)
        stats['bytes_read'] += len(raw)
        if len(raw) > 4096:
            return dict(status='TOO_LARGE', text=None)
        text = raw.decode('ascii')
        stats['files_read'] += 1
        return dict(status='READ', text=text, sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    except (OSError, UnicodeError) as error:
        return dict(status='UNAVAILABLE', text=None, error_type=type(error).__name__)


def _numbered_dirs(root, prefix):
    try:
        return sorted(((int(p.name[len(prefix):]), p) for p in root.iterdir()
            if re.fullmatch(re.escape(prefix) + r'[0-9]+', p.name) and p.is_dir()), key=lambda pair: pair[0])
    except OSError:
        return []


def _receipts(observations):
    # Each original file is (observations[i].path / field). Identical raw
    # contents retain one receipt plus occurrence indices, not SMT copies.
    groups = {}
    for index, observation in enumerate(observations):
        for field, receipt in observation.pop('_fields').items():
            key = (field, json.dumps(receipt, sort_keys=True))
            if key not in groups:
                groups[key] = dict(field=field, **receipt, observations=[])
            groups[key]['observations'].append(index)
    return list(groups.values())


def _executor_records(graph, snapshot, resources, gaps):
    memories = [r for r in resources if type(r) is H.MemoryDomain and r.kind is H.MemoryKind.HOST]
    processors = graph.inventory['cpu']['processors']
    result = []
    for executor in (r for r in resources if type(r) is H.CpuExecutor):
        groups, found = {}, set()
        for index, processor in enumerate(processors):
            cpu = processor['cpu_id']
            if cpu not in executor.cpu_ids:
                continue
            package, core = processor['package_id'], processor['core_id']
            if package is None or core is None:
                continue
            group = groups.setdefault((package, core), dict(package_id=package, core_id=core,
                cpu_ids=[], inventory_processor_refs=[]))
            group['cpu_ids'].append(cpu)
            group['inventory_processor_refs'].append('#/cpu/processors/' + str(index))
            found.add(cpu)
        missing = sorted(set(executor.cpu_ids) - found)
        if missing:
            gaps.append(dict(field='executors.' + str(executor.id) + '.core_smt_groups',
                reason='RETAINED_PACKAGE_OR_CORE_ID_MISSING', cpu_ids=missing))
        result.append(dict(target=encode(H.TargetResourceRef(snapshot, executor.id)),
            cpu_ids=list(executor.cpu_ids), numa_node=executor.numa_node,
            physical_cores=executor.physical_cores, core_smt_groups=list(groups.values()),
            inventory_evidence=encode(executor.evidence),
            host_memory_refs=[encode(H.TargetResourceRef(snapshot, m.id))
                              for m in memories if m.owner == executor.id]))
    return result


def inspect_cpu_hierarchy(target, *, sysfs_root='/sys/devices/system/cpu'):
    """Return a referenceable cache subgraph without altering the target graph.

    ``target`` is a retained target directory or its result.json. Only the six
    selected cache attributes are read. Affinity describes this querying task;
    it never removes inventory CPUs or caches. All source-file receipts share
    the observation window; the sysfs read is explicitly not atomic.
    """
    started, observed_from = time.monotonic(), _now()
    graph, snapshot, inventory, resources = _target(target)
    root = Path(sysfs_root).resolve()
    stats = dict(files_attempted=0, files_read=0, bytes_read=0)
    gaps, grouped, unresolved = [], defaultdict(list), []
    cpu_dirs = _numbered_dirs(root, 'cpu')
    for cpu, directory in cpu_dirs:
        indices = _numbered_dirs(directory / 'cache', 'index')
        if not indices:
            gaps.append(dict(field=str(directory / 'cache'), reason='NO_CACHE_INDICES_READABLE'))
        for index, cache in indices:
            fields = {field: _read(cache / field, stats) for field in _FIELDS}
            values = {field: row['text'].strip() if row['text'] is not None else None
                      for field, row in fields.items()}
            level, shared, kind = _positive(values['level']), _cpu_list(values['shared_cpu_list']), values['type']
            observation = dict(cpu_id=cpu, index=index, path=str(cache), _fields=fields)
            if level is None or not shared or kind not in ('Data', 'Instruction', 'Unified') or cpu not in shared:
                observation['reason'] = 'INVALID_OR_MISSING_CACHE_LEVEL_TYPE_OR_SHARING'
                unresolved.append(observation)
            else:
                grouped[(tuple(shared), level, kind)].append(observation)
    target_cpus = set(c for r in resources if type(r) is H.CpuExecutor for c in r.cpu_ids)
    current_cpus = set(cpu for cpu, _ in cpu_dirs)
    if current_cpus != target_cpus:
        gaps.append(dict(field='target_join.cpu_ids', reason='CURRENT_AND_RETAINED_CPU_ID_SETS_DIFFER',
            current_only=sorted(current_cpus - target_cpus), retained_only=sorted(target_cpus - current_cpus)))
    executors = _executor_records(graph, snapshot, resources, gaps)
    caches = []
    for (shared, level, kind), observations in sorted(grouped.items()):
        values = {field: {row['_fields'][field]['text'].strip() if row['_fields'][field]['text'] is not None
                         else None for row in observations} for field in _FIELDS}
        identity = sha256(json.dumps([shared, level, kind], separators=(',', ':')).encode()).hexdigest()[:24]
        node = dict(id='cpu_cache:' + identity, level=level, type=kind, shared_cpu_ids=list(shared),
            capacity_bytes=None, line_size_bytes=None, sysfs_cache_ids=[], observations=observations,
            allocatable=False, allocation_budget_bytes=None, effective_available_bytes=None)
        for field, output, parse in (('size', 'capacity_bytes', _size),
                                     ('coherency_line_size', 'line_size_bytes', _positive)):
            distinct = values[field]
            parsed = parse(next(iter(distinct))) if len(distinct) == 1 else None
            node[output] = parsed if parsed is not None and parsed > 0 else None
            if node[output] is None:
                gaps.append(dict(field='caches.' + node['id'] + '.' + output,
                    reason='CACHE_ATTRIBUTE_MISSING_INVALID_OR_REPLICA_CONFLICT'))
        node['sysfs_cache_ids'] = sorted(int(value) for value in values['id']
            if value is not None and re.fullmatch(r'[0-9]+', value))
        node['source_receipts'] = _receipts(observations)
        caches.append(node)
    unresolved_records = []
    for observation in unresolved:
        record = dict(reason=observation.pop('reason'), observations=[observation])
        record['source_receipts'] = _receipts(record['observations'])
        unresolved_records.append(record)
    if not caches:
        gaps.append(dict(field='caches', reason='NO_VALID_CACHE_TOPOLOGY_OBSERVED'))
    edges = []
    for ordinal, cache in enumerate(caches):
        cache_ref = dict(cache=cache['id'])
        shared = set(cache['shared_cpu_ids'])
        sharing = [e for e in executors if shared.intersection(e['cpu_ids'])]
        for executor in sharing:
            edges.append(dict(kind='SHARES_CACHE', source=dict(target=executor['target']), target=cache_ref,
                cpu_ids=sorted(shared.intersection(executor['cpu_ids'])),
                conditions=[_JOIN], evidence_ref='#/caches/' + str(ordinal)))
        compatible = {'Unified', cache['type']}
        higher = [other for other in caches if other['level'] > cache['level']
                  and other['type'] in compatible and shared.issubset(other['shared_cpu_ids'])]
        nearest = min((other['level'] for other in higher), default=None)
        destinations = [dict(cache=other['id']) for other in higher if other['level'] == nearest]
        if not destinations:
            destinations = list({json.dumps(ref, sort_keys=True): dict(target=ref)
                for executor in sharing for ref in executor['host_memory_refs']}.values())
        for destination in destinations:
            edges.append(dict(kind='REFILL_CANDIDATE', source=cache_ref, target=destination,
                conditions=[_JOIN, 'LEVEL_AND_CPU_SHARING_DO_NOT_PROVE_REFILL_PATH_OR_INCLUSION'],
                evidence_ref='#/caches/' + str(ordinal)))
    try:
        affinity = dict(cpu_ids=sorted(os.sched_getaffinity(0)), status='OBSERVED_QUERYING_TASK',
                        observed_at=_now(), restricts_inventory=False)
    except (AttributeError, OSError) as error:
        affinity = dict(cpu_ids=None, status='UNAVAILABLE', error_type=type(error).__name__,
                        restricts_inventory=False)
    result = dict(schema=SCHEMA, schema_version=1, target_snapshot=encode(snapshot),
        inventory_receipt=encode(inventory), inventory_cpu_ref='#/cpu',
        target_timestamp=graph.inventory.get('observed_at'),
        observation_window=dict(started_at=observed_from, finished_at=_now(), atomic=False),
        sysfs_root=str(root), fields_read=list(_FIELDS), affinity=affinity,
        target_join=dict(status='CPU_ID_SET_MATCHES_NOT_MACHINE_IDENTITY' if current_cpus == target_cpus
                        else 'CPU_ID_SET_MISMATCH', conditions=[_JOIN], same_machine_proven=False),
        executors=executors, caches=caches, relations=edges, unresolved_observations=unresolved_records, gaps=gaps,
        unknowns=dict(refill_paths=True, inclusion_policy=True, replacement_policy=True,
            latency_seconds=True, bandwidth_bytes_per_second=True, cache_miss_or_dram_traffic=True,
            effective_available_capacity=True, target_snapshot_timestamp=graph.inventory.get('observed_at') is None),
        cache_is_allocatable_storage=False, allocation_budgets_established=False,
        physical_use_established=False, transfers_observed=False, workload_executed=False,
        collector_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        collection=dict(**stats, cache_index_observations=sum(len(c['observations']) for c in caches)
                        + len(unresolved), deduplicated_cache_nodes=len(caches), seconds=time.monotonic() - started))
    return result
