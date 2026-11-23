"""Linux hardware discovery for the graph's placement domain."""

import os
from pathlib import Path
import platform
import re

from scar.contract import Edge, Resource


def _text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _integer(path):
    value = _text(path)
    return int(value) if value and re.fullmatch(r"-?\d+", value) else None


def _numbers(text):
    result = set()
    for part in (text or "").split(","):
        start, separator, finish = part.partition("-")
        if start.isdecimal():
            result.update(
                range(int(start), int(finish) + 1) if separator else (int(start),)
            )
    return result


def _memory(text):
    match = re.search(r"\bMemTotal:\s+(\d+)\s+kB", text or "")
    return int(match[1]) * 1024 if match else None


def _cpu_isa():
    by_cpu, profiles = {}, {}
    for section in (_text(Path("/proc/cpuinfo")) or "").split("\n\n"):
        fields = {
            key.strip(): value.strip()
            for line in section.splitlines()
            for key, separator, value in (line.partition(":"),)
            if separator
        }
        number = fields.get("processor", "")
        flags = fields.get("flags", fields.get("Features", ""))
        if number.isdecimal() and flags:
            features = tuple(sorted(set(flags.split())))
            by_cpu[int(number)] = profiles.setdefault(features, f"isa:{len(profiles)}")
    return by_cpu, {identity: list(features) for features, identity in profiles.items()}


def _cache_size(text):
    match = re.fullmatch(r"(\d+)([KMG]?)", text or "")
    if match is None:
        return None
    power = {"": 0, "K": 1, "M": 2, "G": 3}
    return int(match[1]) * 1024 ** power[match[2]]


def _caches(root, allowed):
    domains = {}
    for cpu in sorted(allowed):
        for location in sorted((root / f"cpu{cpu}" / "cache").glob("index*")):
            level, kind = _integer(location / "level"), _text(location / "type")
            shared = sorted(_numbers(_text(location / "shared_cpu_list")))
            if level is None or kind is None or cpu not in shared:
                continue
            identity = f"cpu-cache:{level}:{kind.lower()}:{shared[0]}"
            if identity not in domains:
                domains[identity] = Resource(
                    identity,
                    "CPU_CACHE",
                    _cache_size(_text(location / "size")),
                    {
                        "level": level,
                        "type": kind,
                        "shared_cpu_ids": shared,
                        "capacity_unit": "bytes",
                        "cache_line_bytes": _integer(location / "coherency_line_size"),
                        "ways": _integer(location / "ways_of_associativity"),
                    },
                )
    edges = [
        Edge(f"cpu:{cpu}", identity, "RESOURCE", "SHARED_CPU_CACHE")
        for identity, cache in domains.items()
        for cpu in cache.attributes["shared_cpu_ids"]
        if cpu in allowed
    ]
    return tuple(domains.values()), edges


def _mount(path):
    selected = None
    for line in (_text(Path("/proc/self/mountinfo")) or "").splitlines():
        prefix, _, suffix = line.partition(" - ")
        fields, filesystem = prefix.split(), suffix.split()
        if len(fields) < 5 or len(filesystem) < 2:
            continue
        mount = Path(
            re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
        )
        if path.is_relative_to(mount) and (
            selected is None or len(mount.parts) >= len(selected[0].parts)
        ):
            selected = mount, fields[2], filesystem[0], filesystem[1]
    return selected


def snapshot(path):
    resources, edges = [], []
    cpu_root = Path("/sys/devices/system/cpu")
    online = _numbers(_text(cpu_root / "online"))
    affinity = getattr(os, "sched_getaffinity", None)
    allowed = (
        set(affinity(0)) if affinity else online or set(range(os.cpu_count() or 1))
    )
    isa_by_cpu, isa_profiles = _cpu_isa()
    architecture = platform.machine()
    by_cpu = {}
    nodes = {
        int(row.name[4:]): row
        for row in Path("/sys/devices/system/node").glob("node[0-9]*")
    }
    for number, node in sorted(nodes.items()):
        identity = f"dram:numa:{number}"
        cpus = _numbers(_text(node / "cpulist"))
        metric = (_text(node / "distance") or "").split()
        distances = (
            dict(zip(map(str, sorted(nodes)), map(int, metric)))
            if all(row.isdecimal() for row in metric)
            else {}
        )
        resources.append(
            Resource(
                identity,
                "NUMA_MEMORY",
                _memory(_text(node / "meminfo")),
                {
                    "node": number,
                    "logical_cpus": sorted(cpus),
                    "capacity_unit": "bytes",
                    "relative_distances": distances,
                },
            )
        )
        by_cpu.update({cpu: number for cpu in cpus})
    if not by_cpu:
        resources.append(
            Resource(
                "dram:host",
                "DRAM",
                _memory(_text(Path("/proc/meminfo"))),
                {"capacity_unit": "bytes"},
            )
        )
    for cpu in sorted(allowed):
        topology = cpu_root / f"cpu{cpu}" / "topology"
        number = by_cpu.get(cpu)
        resources.append(
            Resource(
                f"cpu:{cpu}",
                "CPU",
                1,
                {
                    "logical_cpu": cpu,
                    "package_id": _integer(topology / "physical_package_id"),
                    "core_id": _integer(topology / "core_id"),
                    "numa_node": number,
                    "architecture": architecture,
                    "isa_profile": isa_by_cpu.get(cpu),
                    "smt_siblings": sorted(
                        _numbers(_text(topology / "thread_siblings_list"))
                    ),
                    "capacity_unit": "logical_thread",
                },
            )
        )
        memory = f"dram:numa:{number}" if number is not None else "dram:host"
        if number is not None or not by_cpu:
            edges.append(Edge(f"cpu:{cpu}", memory, "RESOURCE", "HOST_MEMORY_ACCESS"))
    caches, cache_edges = _caches(cpu_root, allowed)
    resources.extend(caches)
    edges.extend(cache_edges)

    storage = {}
    for device in (
        sorted(Path("/sys/block").iterdir()) if Path("/sys/block").is_dir() else ()
    ):
        if not (device / "device").exists():
            continue
        sectors = _integer(device / "size")
        rotational = _integer(device / "queue/rotational")
        kind = (
            "NVME"
            if device.name.startswith("nvme")
            else "SSD" if rotational == 0 else "STORAGE"
        )
        identity = storage[device.name] = f"block:{device.name}"
        resources.append(
            Resource(
                identity,
                kind,
                sectors * 512 if sectors is not None else None,
                {
                    "device": device.name,
                    "rotational": rotational,
                    "capacity_unit": "bytes",
                },
            )
        )
    mount = _mount(path)
    if mount:
        mountpoint, major_minor, filesystem, source = mount
        stat = os.statvfs(path)
        resources.append(
            Resource(
                "mount:input",
                "STORAGE_MOUNT",
                stat.f_blocks * stat.f_frsize,
                {
                    "mountpoint": str(mountpoint),
                    "filesystem": filesystem,
                    "source": source,
                    "capacity_unit": "bytes",
                },
            )
        )
        device_path = Path("/sys/dev/block") / major_minor
        if device_path.exists():
            device = device_path.resolve()
            backing = (
                device.parent.name if (device / "partition").exists() else device.name
            )
            if backing in storage:
                edges.append(
                    Edge(
                        "mount:input",
                        storage[backing],
                        "RESOURCE",
                        "MOUNT_BACKING_DEVICE",
                    )
                )

    render_devices = {
        node.joinpath("device").resolve(): node.name
        for node in Path("/sys/class/drm").glob("renderD*")
        if node.joinpath("device").exists()
    }
    for device in (
        sorted(Path("/sys/bus/pci/devices").iterdir())
        if Path("/sys/bus/pci/devices").is_dir()
        else ()
    ):
        device_class = _text(device / "class") or ""
        driver = device / "driver"
        driver_name = driver.resolve().name if driver.exists() else None
        if (
            driver_name in {"nvidia", "amdgpu", "nouveau", "xe", "i915"}
            or device.resolve() in render_devices
        ):
            kind = "GPU"
        elif device_class.startswith("0x12"):
            kind = "ACCELERATOR"
        else:
            continue
        memory = _integer(device / "mem_info_vram_total")
        identity = f"pci:{device.name}"
        resources.append(
            Resource(
                identity,
                kind,
                None,
                {
                    "pci_class": device_class,
                    "vendor": _text(device / "vendor"),
                    "driver": driver_name,
                    "render_node": render_devices.get(device.resolve()),
                    "numa_node": _integer(device / "numa_node"),
                },
            )
        )
        if memory is not None:
            memory_id = f"{identity}:device-memory"
            resources.append(
                Resource(memory_id, "DEVICE_MEMORY", memory, {"capacity_unit": "bytes"})
            )
            edges.append(Edge(identity, memory_id, "RESOURCE", "DEVICE_MEMORY_ACCESS"))
    evidence = {
        "provider": "linux-proc-sysfs",
        "online_cpu_ids": sorted(online),
        "affinity_cpu_ids": sorted(allowed),
        "cpu_constraint": "sched_getaffinity" if affinity else "inventory",
        "architecture": architecture,
        "isa_profiles": isa_profiles,
    }
    return tuple(resources), tuple(edges), evidence
