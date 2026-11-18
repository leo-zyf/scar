"""Optional, out-of-band resource observations for a local sparse worker pool."""

from multiprocessing.reduction import ForkingPickler
import os
from pathlib import Path
import resource


def _fields(path):
    return {key: value.strip() for key, _, value in (line.partition(':')
        for line in path.read_text().splitlines())}


def _kib(fields, name):
    value = fields.get(name)
    return None if value is None else int(value.split()[0]) * 1024


def connection_frame_bytes(message):
    """Reproduce Connection.send's serialized message length, excluding kernel overhead."""
    payload = len(ForkingPickler.dumps(message))
    framing = 4 if payload <= 0x7FFFFFFF else 12
    return dict(pickle_bytes=payload, framing_bytes=framing, total_bytes=payload + framing)


def snapshot_worker(pid, *, pss=False):
    """Read process resources for this process or its own live child."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('positive local worker PID required')
    proc = Path('/proc') / str(pid)
    try:
        status = _fields(proc / 'status')
        if pid != os.getpid() and int(status['PPid']) != os.getpid():
            raise ValueError('resource observation requires an owned child')
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        clock_hz = os.sysconf('SC_CLK_TCK')
        residence = _fields(proc / 'smaps_rollup') if pss and fields[0] != 'Z' else {}
        fd_entries = tuple((proc / 'fd').iterdir())
        fds = sum(os.path.lexists(entry) for entry in fd_entries)
        caps = list(resource.prlimit(pid, resource.RLIMIT_AS))
        fd_caps = list(resource.prlimit(pid, resource.RLIMIT_NOFILE))
        return dict(pid=pid, alive=fields[0] != 'Z', start_ticks=int(fields[19]),
            cpu_ids=sorted(os.sched_getaffinity(pid)), last_cpu=int(fields[36]),
            threads=int(status['Threads']), user_cpu_ns=int(fields[11]) * 10**9 // clock_hz,
            system_cpu_ns=int(fields[12]) * 10**9 // clock_hz,
            minor_faults=int(fields[7]), major_faults=int(fields[9]),
            rss_bytes=_kib(status, 'VmRSS'), rss_peak_bytes=_kib(status, 'VmHWM'),
            virtual_bytes=_kib(status, 'VmSize'), virtual_peak_bytes=_kib(status, 'VmPeak'),
            pss_bytes=_kib(residence, 'Pss'), private_dirty_bytes=_kib(residence, 'Private_Dirty'),
            fd_count=fds, fd_limits=fd_caps, address_space_limits=caps)
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, alive=False)


def snapshot_pool(pool, *, pss=False):
    """Sample only the owner's process and the child PIDs published by the pool."""
    return dict(owner=snapshot_worker(os.getpid(), pss=pss),
        workers=[snapshot_worker(worker['pid'], pss=pss) for worker in pool.stats['workers']
            if worker is not None])
