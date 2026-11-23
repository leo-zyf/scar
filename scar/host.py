"""Read execution-host CPU constraints without importing any compiler stage."""

from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys


def array_sdk(*, required=False):
    """Inspect the fixed CPU array provider without provisioning or executing it."""
    explicit = os.environ.get("SCAR_CPU_SDK")
    prefixes = [explicit] if explicit else []
    if not explicit:
        prefixes.extend(
            os.environ.get("CMAKE_PREFIX_PATH", "")
            .replace(";", os.pathsep)
            .split(os.pathsep)
        )
        prefixes.append(str(Path.home() / ".local/share/scar/sdk/xtensor-0.27.1"))
    issues = []
    for candidate in dict.fromkeys(p for p in prefixes if p):
        prefix = Path(candidate).expanduser().resolve()
        receipt = prefix / "scar-sdk.json"
        try:
            raw = receipt.read_bytes()
            record = json.loads(raw)
            if (record.get("schema"), record.get("schema_version")) != (
                "scar.cpu-sdk",
                1,
            ):
                raise ValueError("invalid CPU SDK receipt")
            packages = record["packages"]
            if {key: packages[key]["version"] for key in ("xtensor", "xtl")} != {
                "xtensor": "0.27.1",
                "xtl": "0.8.0",
            }:
                raise ValueError("CPU SDK versions do not match the supported profile")
            files = record["files"]
            required_files = {
                "include/xtensor/containers/xarray.hpp",
                "include/xtensor/containers/xadapt.hpp",
                "share/cmake/xtensor/xtensorConfig.cmake",
                "share/cmake/xtl/xtlConfig.cmake",
            }
            if not required_files <= files.keys():
                raise ValueError("incomplete CPU array SDK")
            for name, expected in files.items():
                relative = Path(name)
                path = (prefix / relative).resolve(strict=True)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not path.is_relative_to(prefix)
                ):
                    raise ValueError("CPU SDK member escapes its prefix")
                if sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"CPU SDK member changed: {name}")
            return dict(
                provider="xtensor-cpu",
                prefix=str(prefix),
                packages=packages,
                receipt_sha256=sha256(raw).hexdigest(),
                validated_files=len(files),
                evidence="observed installed package identity; not a kernel timing",
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"{prefix}: {error}")
    if required:
        raise ValueError(
            "CPU ndarray SDK unavailable; run tools/provision_cpu_sdk.py: "
            + "; ".join(issues)
        )
    return None


@dataclass(frozen=True)
class CPUQuota:
    source: str
    quota_us: int | None
    period_us: int


@dataclass(frozen=True)
class CPUAvailability:
    cpu_ids: tuple[int, ...]
    quotas: tuple[CPUQuota, ...]
    issues: tuple[str, ...] = ()

    @property
    def capacity_upper_bound(self):
        """Sustained core-time bound; quotas do not forbid wider short-lived teams."""
        capacities = [Fraction(len(self.cpu_ids))]
        capacities.extend(
            Fraction(q.quota_us, q.period_us)
            for q in self.quotas
            if q.quota_us is not None
        )
        return min(capacities)

    @property
    def worker_budget(self):
        # This is a conservative tool budget, not a hardware thread-count claim.
        return 1 if self.issues else max(1, int(self.capacity_upper_bound))

    def as_record(self):
        return dict(
            provider="host-affinity-cgroup",
            evidence="observed_constraints",
            cpu_ids=list(self.cpu_ids),
            quotas=[asdict(q) for q in self.quotas],
            constraint_scope="visible cgroup hierarchy and process affinity",
            capacity_upper_bound_cores=float(self.capacity_upper_bound),
            capacity_scope="sustained core-time, not instantaneous concurrency or compute throughput",
            capacity_evidence="inferred upper bound from affinity and visible quota/period",
            worker_budget=self.worker_budget,
            budget_evidence="conservative policy, not measured optimal parallelism",
            budget_basis="floor of visible capacity, at least one worker; unresolved constraints use one",
            guarantee="upper bound; excludes contention, SMT efficiency and hidden ancestors",
            issues=list(self.issues),
        )


def _read(path):
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return None


def _unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def _quota_files(membership, mounts):
    groups = {}
    for line in membership.splitlines():
        _, controllers, group = line.split(":", 2)
        if not controllers:
            groups["cgroup2"] = Path(group)
        elif "cpu" in controllers.split(","):
            groups["cgroup"] = Path(group)
    matched = set()
    seen = set()
    for line in mounts.splitlines():
        before, after = line.split(" - ", 1)
        kind, _, options = after.split()
        if kind not in groups or kind == "cgroup" and "cpu" not in options.split(","):
            continue
        mount_root, mount = (Path(_unescape(p)) for p in before.split()[3:5])
        group = groups[kind]
        if not all(p.is_absolute() for p in (mount_root, mount, group)):
            raise ValueError("CPU controller paths must be absolute")
        relative = (
            group.relative_to(mount_root)
            if group.is_relative_to(mount_root)
            else group.relative_to("/")
        )
        mount = mount.resolve()
        folder = (mount / relative).resolve()
        if not folder.is_relative_to(mount):
            raise ValueError("CPU controller membership is outside its visible mount")
        matched.add(kind)
        while folder.is_relative_to(mount):
            quota = folder / ("cpu.max" if kind == "cgroup2" else "cpu.cfs_quota_us")
            if quota not in seen:
                seen.add(quota)
                yield quota, None if kind == "cgroup2" else folder / "cpu.cfs_period_us"
            if folder == mount:
                break
            folder = folder.parent
    if groups.keys() - matched:
        raise ValueError("a CPU controller mount is not visible")


def cpu_availability():
    """Sample affinity and all visible ancestor bandwidth limits once per consumer."""
    issues = []
    affinity = getattr(os, "sched_getaffinity", None)
    try:
        cpus = tuple(sorted(affinity(0))) if affinity else ()
    except OSError:
        cpus = ()
    if not cpus:
        cpus = tuple(range(os.cpu_count() or 1))
        issues.append("process affinity unavailable; CPU inventory is an upper bound")
    quotas = []
    if sys.platform.startswith("linux"):
        try:
            membership, mounts = _read("/proc/self/cgroup"), _read(
                "/proc/self/mountinfo"
            )
            if membership is None or mounts is None:
                raise ValueError("CPU controller metadata is unavailable")
            for source, period_source in _quota_files(membership, mounts):
                amount = _read(source)
                if amount is None:
                    continue
                if period_source is None:
                    amount, period = amount.split()
                else:
                    period = _read(period_source)
                if period is None:
                    raise ValueError(f"CPU controller period is unavailable: {source}")
                limit = None if amount.strip() in {"max", "-1"} else int(amount)
                period = int(period)
                if period <= 0 or limit is not None and limit <= 0:
                    raise ValueError(f"invalid CPU bandwidth constraint: {source}")
                quotas.append(CPUQuota(str(source), limit, period))
        except (OSError, ValueError) as error:
            issues.append(str(error))
    return CPUAvailability(cpus, tuple(quotas), tuple(issues))
