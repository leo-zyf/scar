"""Linux hardware discovery for the graph's placement domain."""

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re

from scar.contract import Edge, Resource
from scar.host import array_sdk, cpu_availability


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


def _caches(root, visible):
    domains = {}
    for cpu in sorted(visible):
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
        if cpu in visible
    ]
    return tuple(domains.values()), edges


_PSEUDO_FS = frozenset(
    "autofs binfmt_misc bpf cgroup cgroup2 configfs debugfs devpts devtmpfs "
    "efivarfs fusectl mqueue nfsd nsfs proc pstore resctrl rpc_pipefs securityfs "
    "selinuxfs smackfs sysfs tracefs usbfs".split()
)
_NETWORK_FS = frozenset(
    "afs beegfs ceph cifs coda fuse.ceph fuse.glusterfs fuse.sshfs "
    "glusterfs lustre nfs nfs4 smb3 smbfs".split()
)
_MEMORY_FS = {"tmpfs", "ramfs", "hugetlbfs"}


@dataclass(frozen=True)
class _Mount:
    id: int
    parent: int
    device: str
    root: str
    point: Path
    filesystem: str
    source: str
    read_only: bool

    @property
    def data_mount(self):
        if self.filesystem in _PSEUDO_FS:
            return False
        if self.filesystem not in {"tmpfs", "ramfs"}:
            return True
        return self.point.is_relative_to("/dev/shm") or not any(
            self.point.is_relative_to(p) for p in ("/dev", "/sys", "/proc")
        )


def _mount_table(source):
    try:
        data = source.read_bytes()
    except OSError:
        return {}

    def name(field):
        unescaped = re.sub(
            rb"\\(040|011|012|134)", lambda m: bytes([int(m[1], 8)]), field
        )
        return os.fsdecode(unescaped)

    mounts = {}
    for line in data.split(b"\n"):
        left, separator, right = line.partition(b" - ")
        fields, fs = left.split(b" "), right.split(b" ")
        if (
            not separator
            or len(fields) < 6
            or len(fs) < 3
            or not fields[0].isdigit()
            or not fields[1].isdigit()
            or not re.fullmatch(rb"\d+:\d+", fields[2])
        ):
            continue
        record = _Mount(
            int(fields[0]),
            int(fields[1]),
            fields[2].decode("ascii"),
            name(fields[3]),
            Path(name(fields[4])),
            name(fs[0]),
            name(fs[1]),
            b"ro" in fields[5].split(b",") or b"ro" in fs[2].split(b","),
        )
        mounts[record.id] = record
    return mounts


def _visible_mount(path, children):
    """Walk mount attachments in path order, including overmounts of the same path."""
    current = None
    visited = set()
    for prefix in (*reversed(path.parents), path):
        if current is not None and (None, prefix) in children:
            return None, "unlinked_mount_hierarchy"
        while candidates := children.get((current, prefix)):
            if len(candidates) != 1 or candidates[0] in visited:
                return None, "ambiguous_mount_hierarchy"
            current = candidates[0]
            visited.add(current)
    return (current, "mount_hierarchy") if current is not None else (None, "unresolved")


def _block_topology(numbers, root, storage):
    by_device, identities, expanded = {}, storage.copy(), set()
    resources, edges, pending = [], [], []

    def identity(device):
        if device.name not in identities:
            key = identities[device.name] = f"block:{device.name}"
            sectors = _integer(device / "size")
            resources.append(
                Resource(
                    key,
                    "BLOCK_VOLUME",
                    sectors * 512 if sectors is not None else None,
                    {"device": device.name, "capacity_unit": "bytes"},
                )
            )
        return identities[device.name]

    for number in sorted(numbers):
        try:
            device = (root / number).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        by_device[number] = identity(device)
        pending.append((device, frozenset()))
    # Only follow kernel-published dependencies reachable from these mounts.
    while pending:
        device, ancestors = pending.pop()
        if device in expanded:
            continue
        expanded.add(device)
        ancestors = ancestors | {device}
        links = (
            [device.parent]
            if (device / "partition").is_file()
            else sorted((device / "slaves").glob("*"))
        )
        inputs = set()
        for link in links:
            try:
                backing = link.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if backing not in ancestors and backing not in inputs:
                inputs.add(backing)
                edges.append(
                    Edge(
                        identity(device),
                        identity(backing),
                        "RESOURCE",
                        "BLOCK_BACKED_BY",
                    )
                )
                pending.append((backing, ancestors))
    return by_device, resources, edges


def _storage_mounts(
    path,
    storage,
    *,
    mountinfo=Path("/proc/self/mountinfo"),
    device_root=Path("/sys/dev/block"),
    namespace_link=Path("/proc/self/ns/mnt"),
):
    mounts = _mount_table(mountinfo)
    try:
        namespace = os.readlink(namespace_link)
    except OSError:
        namespace = None
    children = {}
    for m in mounts.values():
        parent = m.parent if m.parent in mounts and m.parent != m.id else None
        children.setdefault((parent, m.point), []).append(m.id)
    try:
        selected, resolution = _visible_mount(Path(path).resolve(strict=True), children)
    except (OSError, RuntimeError):
        selected, resolution = None, "input_path_unavailable"
    kept = {i: m for i, m in mounts.items() if m.data_mount}
    capacity = None
    if selected in kept:
        try:
            stat = os.statvfs(path)
            capacity = stat.f_blocks * stat.f_frsize
        except OSError:
            pass
    elif selected is not None:
        selected, resolution = None, "outside_data_filesystems"
    numbers = {
        m.device
        for m in kept.values()
        if m.filesystem not in _MEMORY_FS and m.filesystem not in _NETWORK_FS
    }
    by_device, resources, edges = _block_topology(numbers, device_root, storage)
    for i, m in sorted(kept.items()):
        identity = f"mount:{i}"
        if m.parent in kept and m.parent != i:
            edges.append(
                Edge(identity, f"mount:{m.parent}", "RESOURCE", "MOUNT_PARENT")
            )
        if m.filesystem in _MEMORY_FS:
            backing = "host_memory"
        elif m.filesystem in _NETWORK_FS:
            backing = "network_filesystem"
        else:
            block = by_device.get(m.device)
            backing = "block_device" if block is not None else "unresolved"
            if block is not None:
                edges.append(Edge(identity, block, "RESOURCE", "MOUNT_BACKING_DEVICE"))
        top, _ = _visible_mount(m.point, children)
        resources.append(
            Resource(
                identity,
                "STORAGE_MOUNT",
                capacity if i == selected else None,
                dict(
                    mount_id=i,
                    parent_id=m.parent,
                    mountpoint=str(m.point),
                    root=m.root,
                    major_minor=m.device,
                    filesystem=m.filesystem,
                    source=m.source,
                    read_only=m.read_only,
                    backing_kind=backing,
                    **(
                        {"page_policy": "huge_pages"}
                        if m.filesystem == "hugetlbfs"
                        else {}
                    ),
                    path_visibility=(
                        "unresolved"
                        if top is None
                        else "visible" if top == i else "covered"
                    ),
                    capture_namespace=namespace,
                    capacity_unit="bytes",
                ),
            )
        )
    return (
        resources,
        edges,
        dict(
            input_mount=f"mount:{selected}" if selected is not None else None,
            input_mount_resolution=resolution,
            mount_namespace=namespace,
            mount_scope="capture_process_mount_namespace",
        ),
    )


def _block_kind(device, rotational):
    try:
        subsystem = Path(os.readlink(device / "device/subsystem")).name
    except OSError:
        subsystem = None
    if subsystem in {"nvme", "nvme-subsystem"}:
        return "NVME"
    return "SSD" if rotational == 0 else "STORAGE"


def snapshot(path):
    cpu_root = Path("/sys/devices/system/cpu")
    online_mask = _text(cpu_root / "online")
    online = _numbers(online_mask)
    present_mask = _text(cpu_root / "present")
    present = _numbers(present_mask)
    availability = cpu_availability()
    allowed = set(availability.cpu_ids)
    visible = present | online | allowed
    if not present_mask:
        visible.update(
            int(cpu.name[3:])
            for cpu in cpu_root.glob("cpu[0-9]*")
            if cpu.name[3:].isdecimal()
        )
    budget_evidence = availability.as_record()
    pool = Resource(
        "cpu-pool:process",
        "CPU_POOL",
        availability.worker_budget,
        {**budget_evidence, "capacity_unit": "conservative_worker_budget"},
    )
    resources = [pool]
    edges = [
        Edge(pool.id, f"cpu:{cpu}", "RESOURCE", "CPU_POOL_MEMBER")
        for cpu in sorted(allowed)
    ]
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
    if not nodes:
        resources.append(
            Resource(
                "dram:host",
                "DRAM",
                _memory(_text(Path("/proc/meminfo"))),
                {"capacity_unit": "bytes"},
            )
        )
    physical_cores = {}
    for cpu in sorted(visible):
        topology = cpu_root / f"cpu{cpu}" / "topology"
        number = by_cpu.get(cpu)
        package = _integer(topology / "physical_package_id")
        core = _integer(topology / "core_id")
        if package is not None and package >= 0 and core is not None and core >= 0:
            identity = f"core:{package}:{core}"
            physical_cores.setdefault(identity, []).append(cpu)
            edges.append(Edge(f"cpu:{cpu}", identity, "RESOURCE", "THREAD_ON_CORE"))
        resources.append(
            Resource(
                f"cpu:{cpu}",
                "CPU",
                1,
                {
                    "logical_cpu": cpu,
                    "online": cpu in online if online_mask is not None else None,
                    "package_id": package,
                    "core_id": core,
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
        if number is not None or not nodes:
            edges.append(Edge(f"cpu:{cpu}", memory, "RESOURCE", "HOST_MEMORY_ACCESS"))
    for identity, cpus in physical_cores.items():
        resources.append(
            Resource(
                identity,
                "CPU_CORE",
                1,
                {"logical_cpus": cpus, "capacity_unit": "physical_core"},
            )
        )
    caches, cache_edges = _caches(cpu_root, visible)
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
        kind = _block_kind(device, rotational)
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
    mounts, mount_edges, mount_evidence = _storage_mounts(path, storage)
    resources.extend(mounts)
    edges.extend(mount_edges)

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
        **mount_evidence,
        "provider": "linux-proc-sysfs",
        "present_cpu_ids": sorted(present) if present_mask is not None else None,
        "online_cpu_ids": sorted(online) if online_mask is not None else None,
        "affinity_cpu_ids": sorted(allowed),
        "cpu_constraint": pool.id,
        "cpu_availability": budget_evidence,
        "architecture": architecture,
        "isa_profiles": isa_profiles,
    }
    sdk = array_sdk()
    if sdk:
        evidence["operator_libraries"] = {"ndarray": sdk}
    return tuple(resources), tuple(edges), evidence
