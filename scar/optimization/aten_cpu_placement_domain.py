"""Nominate deterministic CPU pairs from retained hardware topology."""

from collections import Counter, defaultdict

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.program.target_hardware import CacheDomain, CpuCore, CpuExecutor, HardwareThread
from scar.ir.record_codec import encode


SCHEMA = 'scar.aten-cpu-placement-domain'
_MAX_CLASSES = 64


def _cpu_memberships(resources):
    executors, cores, caches, threads = defaultdict(list), {}, defaultdict(list), {}
    for resource in resources:
        kind = type(resource)
        if kind is CpuExecutor:
            for cpu in resource.cpu_ids:
                executors[cpu].append(resource)
        elif kind is CpuCore:
            for cpu in resource.cpu_ids:
                cores[cpu] = resource
        elif kind is HardwareThread:
            threads[resource.cpu_id] = resource
        elif kind is CacheDomain and resource.cache_type in ('Data', 'Unified'):
            for cpu in resource.shared_cpu_ids:
                caches[cpu].append(resource)
    return executors, cores, caches, threads


def _unique(rows):
    return rows[0] if len(rows) == 1 else None


def _last_level(caches):
    level = max((cache.level for cache in caches), default=None)
    return _unique([cache for cache in caches if cache.level == level])


def _lane(cpu, executors, cores, caches, threads):
    return (threads.get(cpu), cores.get(cpu), _unique(executors.get(cpu, ())), _last_level(caches.get(cpu, ())))


def _signature(lane):
    thread, core, executor, cache = lane
    return (executor.numa_node, core.package_id, core.die_id, executor.model_names,
            tuple(sorted(thread.isa_flags)), cache.level, cache.cache_type, cache.capacity_bytes, cache.line_size_bytes)


def _topology(lane):
    thread, core, executor, cache = lane
    return dict(cpu_id=thread.cpu_id, package_id=core.package_id, die_id=core.die_id, core=encode(core.id),
                numa_node=None if executor is None else executor.numa_node,
                model_names=None if executor is None else list(executor.model_names),
                last_level_cache=None if cache is None else encode(cache.id),
                evidence=dict(thread=encode(thread.evidence), core=encode(core.evidence),
                              numa=None if executor is None else encode(executor.evidence),
                              isa=None if thread.isa_evidence is None else encode(thread.isa_evidence),
                              cache=None if cache is None else encode(cache.evidence)))


def _pair(lanes, caller, worker, count):
    left, right = lanes[caller], lanes[worker]
    shared = None if left[3] is None or right[3] is None else left[3].id == right[3].id
    return dict(cpus=[caller, worker], represented_ordered_pairs=count,
                topology=dict(caller=_topology(left), worker=_topology(right), shared_last_level_cache=shared))


def _representatives(lanes, exact):
    if len(lanes) < 2:
        return [], None
    if exact:
        caller, worker = lanes
        return [_pair(lanes, caller, worker, 1)], None

    signatures, lane_groups = {}, {}
    for cpu, lane in lanes.items():
        signature = _signature(lane)
        lane_groups[cpu] = signatures.setdefault(signature, len(signatures)), lane[3].id

    classes = {}
    for caller, (caller_group, caller_cache) in lane_groups.items():
        for worker, (worker_group, worker_cache) in lane_groups.items():
            if caller == worker:
                continue
            key = caller_group, worker_group, caller_cache == worker_cache
            if key not in classes:
                if len(classes) == _MAX_CLASSES:
                    return [], 'TOPOLOGY_CLASS_BUDGET_EXCEEDED'
                classes[key] = [caller, worker, 0]
            classes[key][2] += 1
    return [_pair(lanes, caller, worker, count) for caller, worker, count in classes.values()], None


def aten_cpu_placement_domain(target, *, allowed_cpu_ids):
    """Retain one calibration representative per ordered topology class.

    Membership is a nomination rule. The recorded topology does not give
    per-core timings or establish where an application's pages are allocated.
    """
    from scar.optimization.aten_native_schedule import recorded_cpu_domain

    launch_ids = tuple(allowed_cpu_ids)
    if any(type(cpu) is not int or cpu < 0 for cpu in launch_ids):
        raise ValueError('launch affinity requires nonnegative Linux CPU IDs')
    launch = set(launch_ids)
    graph = TargetHardwareGraph(target)
    constraint, recorded = recorded_cpu_domain(graph)
    exact = (len(recorded) == 2
             and tuple(sorted(constraint.affinity_cpu_ids)) == tuple(sorted(constraint.allowed_cpu_ids)))
    memberships = _cpu_memberships(graph.resources())
    eligible, selected_cores, excluded = {}, set(), Counter()
    for cpu in recorded:
        thread, core, executor, cache = lane = _lane(cpu, *memberships)
        if cpu not in launch:
            reason = 'OUTSIDE_LAUNCH_AFFINITY'
        elif cpu >= 1024:
            reason = 'OUTSIDE_NATIVE_CPU_ID_RANGE'
        elif thread is None or thread.online is not True:
            reason = 'ONLINE_THREAD_UNAVAILABLE'
        elif core is None:
            reason = 'PHYSICAL_CORE_UNAVAILABLE'
        elif core.id in selected_cores:
            reason = 'SMT_SIBLING_NOT_NOMINATED'
        elif not exact and (executor is None or executor.numa_node is None or cache is None
                            or thread.isa_flags is None or thread.isa_evidence is None):
            reason = 'CPU_TOPOLOGY_INCOMPLETE'
        else:
            selected_cores.add(core.id)
            eligible[cpu] = lane
            continue
        excluded[reason] += 1

    representatives, reason = _representatives(eligible, exact)
    if not representatives and reason is None:
        reason = 'DISTINCT_ELIGIBLE_PHYSICAL_CORES_UNAVAILABLE'
    pairs = len(eligible) * (len(eligible) - 1)
    return dict(schema=SCHEMA, schema_version=1, target=encode(graph._snapshot),
                constraint_fact=encode(constraint.evidence), recorded_cpu_ids=list(recorded),
                launch_cpu_ids=sorted(launch), eligible_cpus=list(eligible),
                policy='RECORDED_EXACT_PAIR' if exact else 'ONE_REPRESENTATIVE_PER_ORDERED_TOPOLOGY_CLASS',
                representatives=representatives, reason=reason, excluded_cpu_counts=dict(excluded),
                domain_counts=dict(recorded_logical_cpus=len(recorded), eligible_physical_cores=len(eligible),
                                   eligible_ordered_pairs=pairs // 2 if exact else pairs,
                                   topology_class_limit=_MAX_CLASSES, representatives=len(representatives)))
