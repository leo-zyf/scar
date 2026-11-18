"""Isolate speculative SuperLU solves and consume them in source order."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
from multiprocessing.connection import Connection
import os
from pathlib import Path
import resource
import socket
import struct
import subprocess
import sys
from threading import Lock, get_ident
from time import perf_counter_ns, thread_time_ns

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.sparse_lu import _Timings, load_symbolic_solve


_MIB = 1024**2
_TIMEOUT = 30


def _floating_control():
    # Linux x86-64 glibc fenv_t keeps these control words at offsets 0 and 28.
    # Sticky exception flags are results of arithmetic, outside the control.
    fegetenv = ctypes.CDLL('libm.so.6').fegetenv
    fegetenv.argtypes, fegetenv.restype = [ctypes.c_void_p], ctypes.c_int
    storage = ctypes.create_string_buffer(32)

    def default():
        return (fegetenv(storage) == 0 and struct.unpack_from('=H', storage)[0] == 0x037F
            and struct.unpack_from('=I', storage, 28)[0] & ~0x3F == 0x1F80)

    return default


class _Worker:
    def __init__(self, manifest, cpu, limit, extra, budget):
        parent, child = socket.socketpair()
        with parent, child:
            self.connection = Connection(os.dup(parent.fileno()))
            self.process = None
            try:
                env = {key: value for key, value in os.environ.items() if key not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
                env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', GOTO_NUM_THREADS='1')
                self.process = subprocess.Popen(
                    [sys.executable, '-B', '-S', '-E', '-m', __spec__.name, str(child.fileno()), str(cpu),
                     str(limit or 0), str(extra), str(budget),
                     manifest['uri'], manifest['sha256'], str(manifest['size_bytes'])],
                    pass_fds=(child.fileno(),), env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).resolve().parents[2])
            except OSError:
                self.connection.close()
                raise

    def exchange(self, value):
        self.connection.send(value)
        if not self.connection.poll(_TIMEOUT):
            self.process.kill()
            raise TimeoutError('parallel SuperLU worker did not return')
        return self.connection.recv()

    def close(self):
        self.connection.close()
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class _ParallelSolver:
    def __init__(self, native, cpus, limit, extra, budget):
        self.manifest = artifact_reference(Path(native).resolve() / 'artifact.json', 'scar.superlu-symbolic-study')
        self.serial = load_symbolic_solve(native, budget_bytes=budget)
        self.blas = ctypes.CDLL(read_artifact(self.manifest)['blas']['uri'])
        self.blas_threads = self.blas.scipy_openblas_get_num_threads
        if self.blas.scipy_openblas_get_parallel() != 1:
            raise ValueError('pthread OpenBLAS required in the original solve scope')
        self.default_floating_control = _floating_control()
        self.cpus, self.limit, self.extra, self.budget = cpus, limit, extra, budget
        self.owner = get_ident()
        self.lock = Lock()
        self.executor = None
        self.workers = []
        self.pending = {}
        self.structure = None
        self.stats = dict(startup_ns=0, owner_validation_ns=0, workers=[], tasks=[], publications=[],
            submissions=0, native_calls=0, fallback_calls=0, peak_snapshot_bytes=0,
            pending_snapshot_bytes=0, pending_result_bound_bytes=0, structure_bytes=0, closed=False, available=False)

    def _check_owner(self):
        if get_ident() != self.owner:
            raise ValueError('parallel SuperLU issue and publication require their owner thread')

    def _capture(self, args, options):
        if (not self.default_floating_control() or self.blas_threads() != 1
                or self.serial._arguments(args, options) is None):
            return None
        n, nnz, values, indices, offsets, rhs, _ = args
        return (n, nnz, indices.tobytes(), offsets.tobytes()), (values.tobytes(), rhs.tobytes())

    def prepare(self, args, options):
        self._check_owner()
        if self.stats['closed']:
            return False
        captured = self._capture(args, options)
        if captured is None:
            return False
        structure, _ = captured
        if self.executor is not None:
            return self.stats['available'] and structure == self.structure
        started = perf_counter_ns()
        if self.serial(args, options) is None:
            return False
        self.serial.close()
        self.stats['owner_validation_ns'] = perf_counter_ns() - started
        started = perf_counter_ns()
        self.executor = ThreadPoolExecutor(max_workers=len(self.cpus), thread_name_prefix='scar-superlu-ipc')
        try:
            for cpu in self.cpus:
                self.workers.append(_Worker(self.manifest, cpu, self.limit, self.extra, self.budget))
            ordered = [self.executor.submit(worker.exchange, structure) for worker in self.workers]
            self.stats['workers'] = [future.result() for future in ordered]
            if any(worker is None for worker in self.stats['workers']):
                self.close()
                return False
        except (OSError, EOFError, RuntimeError):
            self.close()
            return False
        finally:
            self.stats['startup_ns'] = perf_counter_ns() - started
        self.structure = structure
        self.stats['structure_bytes'] = len(structure[2]) + len(structure[3])
        self.stats['pending_result_bound_bytes'] = len(self.workers) * structure[0] * 8
        self.stats['available'] = True
        return True

    def _execute(self, worker, buffers, issue, submitted):
        exchange_started = perf_counter_ns()
        try:
            numeric, metrics = worker.exchange(buffers)
        except (OSError, EOFError, RuntimeError):
            numeric, metrics = None, dict(pid=worker.process.pid, status='WORKER_UNAVAILABLE')
        metrics.update(issue=issue, submitted_ns=submitted, exchange_started_ns=exchange_started,
            exchange_complete_ns=perf_counter_ns())
        with self.lock:
            self.stats['tasks'].append(metrics)
        if numeric is None or len(numeric) != self.structure[0] * 8:
            return None
        result = self.serial.empty(self.structure[0], dtype=self.serial.f64)
        ctypes.memmove(result.__array_interface__['data'][0], numeric, len(numeric))
        return result, 0

    def submit(self, args, options):
        """Return a private future. Retire it with consume() or discard()."""
        self._check_owner()
        if self.stats['closed'] or self.executor is None and not self.prepare(args, options):
            return None
        captured = self._capture(args, options)
        if captured is None or captured[0] != self.structure:
            return None
        busy = {item[0] for item in self.pending.values()}
        worker = next((item for item in self.workers if item not in busy and item.process.poll() is None), None)
        if worker is None:
            return None
        buffers = captured[1]
        issue = self.stats['submissions']
        future = self.executor.submit(self._execute, worker, buffers, issue, perf_counter_ns())
        self.pending[future] = (worker, buffers, issue)
        self.stats['submissions'] += 1
        self.stats['pending_snapshot_bytes'] += sum(map(len, buffers))
        self.stats['peak_snapshot_bytes'] = max(self.stats['peak_snapshot_bytes'], self.stats['pending_snapshot_bytes'])
        return future

    def _retire(self, future):
        if not self.pending or next(iter(self.pending)) is not future:
            return None
        _, buffers, issue = self.pending.pop(future)
        self.stats['pending_snapshot_bytes'] -= sum(map(len, buffers))
        return buffers, issue

    def consume(self, future, args, options):
        """Publish in issue order only after matching the source carrier inputs."""
        self._check_owner()
        retired = self._retire(future)
        if retired is not None:
            buffers, issue = retired
            captured = self._capture(args, options)
            started = perf_counter_ns()
            answer = future.result()
            ready = captured == (self.structure, buffers) and answer is not None
            self.stats['publications'].append(dict(issue=issue, consume_wait_ns=perf_counter_ns() - started,
                native=ready))
            if ready:
                self.stats['native_calls'] += 1
                return answer
        self.stats['fallback_calls'] += 1
        return None

    def discard(self, future):
        self._check_owner()
        if self._retire(future) is not None:
            future.result()

    def close(self):
        self._check_owner()
        if self.stats['closed']:
            return
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        for worker in self.workers:
            worker.close()
        self.workers.clear()
        self.pending.clear()
        self.structure = None
        self.serial.close()
        self.stats.update(closed=True, available=False, pending_snapshot_bytes=0,
            pending_result_bound_bytes=0, structure_bytes=0, retained_after_close_bytes=0)


def load_parallel_symbolic_solve(native_dir, *, worker_cpu_ids, address_space_limit=None,
        private_bytes_per_worker=64 * _MIB, budget_bytes=8 * _MIB):
    """Load a bounded speculative pool for the admitted native CSR artifact."""
    if sys.platform != 'linux' or os.uname().machine != 'x86_64' or not os.confstr('CS_GNU_LIBC_VERSION'):
        raise ValueError('parallel SuperLU requires the audited Linux x86-64/glibc floating-control ABI')
    cpus = tuple(worker_cpu_ids)
    if (not cpus or any(type(cpu) is not int for cpu in cpus) or len(set(cpus)) != len(cpus)
            or not set(cpus) <= os.sched_getaffinity(0)
            or type(private_bytes_per_worker) is not int or private_bytes_per_worker <= 0
            or address_space_limit is not None and (type(address_space_limit) is not int or address_space_limit <= 0)):
        raise ValueError('distinct available worker CPUs and positive SuperLU memory limits required')
    return _ParallelSolver(native_dir, cpus, address_space_limit, private_bytes_per_worker, budget_bytes)


def _private_space_limit(limit, extra):
    pages = int(Path('/proc/self/statm').read_text().split()[0])
    baseline = pages * os.sysconf('SC_PAGE_SIZE')
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = limit or baseline + extra
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if limit <= baseline:
        raise ValueError('worker address space limit is below loaded runtime size')
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return baseline, limit


def _native(manifest):
    record = read_artifact(manifest)
    for reference in (record['library'], record['blas']):
        read_artifact(reference, json_document=False)
    blas = ctypes.CDLL(record['blas']['uri'])
    if blas.scipy_openblas_get_parallel() != 1 or blas.scipy_openblas_get_num_threads() != 1:
        raise ValueError('one-thread pthread OpenBLAS required in a sparse worker')
    library = ctypes.CDLL(record['library']['uri'])
    if library.scar_superlu_abi_v1() != 1:
        raise ValueError('worker SuperLU ABI differs')
    pointer, number = ctypes.c_void_p, ctypes.c_int
    timing = ctypes.POINTER(_Timings)
    create = library.scar_superlu_symbolic_v1
    create.argtypes, create.restype = [number, number, pointer, pointer, ctypes.c_size_t, timing], pointer
    apply = library.scar_superlu_symbolic_solve_v1
    apply.argtypes, apply.restype = [pointer, number, number, *([pointer] * 5), timing], number
    size = library.scar_superlu_symbolic_bytes_v1
    size.argtypes, size.restype = [pointer], ctypes.c_size_t
    destroy = library.scar_superlu_symbolic_destroy_v1
    destroy.argtypes, destroy.restype = [pointer], None
    return create, apply, size, destroy


def _child(fd, cpu, limit, extra, budget, manifest):
    os.sched_setaffinity(0, {cpu})
    with Connection(fd) as connection:
        create, solve, size, destroy = _native(manifest)
        if not _floating_control()():
            raise ValueError('child floating point control differs')
        baseline, cap = _private_space_limit(limit, extra)
        order, nnz, indices, offsets = connection.recv()
        index_buffer, offset_buffer = ctypes.create_string_buffer(indices), ctypes.create_string_buffer(offsets)
        timing = _Timings()
        started = perf_counter_ns()
        handle = create(order, nnz, index_buffer, offset_buffer, budget, ctypes.byref(timing))
        if not handle:
            connection.send(None)
            return
        try:
            connection.send(dict(pid=os.getpid(), cpu=cpu, symbolic_bytes=size(handle),
                symbolic_build_ns=perf_counter_ns() - started, phases=timing.measurements(),
                address_space_baseline_bytes=baseline, address_space_limit_bytes=cap))
            while True:
                values, rhs = connection.recv()
                data, load = ctypes.create_string_buffer(values), ctypes.create_string_buffer(rhs)
                output = (ctypes.c_double * order)()
                timing = _Timings()
                started = perf_counter_ns()
                cpu_started = thread_time_ns()
                status = solve(handle, order, nnz, index_buffer, offset_buffer,
                    data, load, output, ctypes.byref(timing))
                cpu_elapsed = thread_time_ns() - cpu_started
                ended = perf_counter_ns()
                connection.send((bytes(output) if status == 0 else None,
                    dict(pid=os.getpid(), cpu=cpu, status=status, started_ns=started, ended_ns=ended,
                        native_thread_cpu_ns=cpu_elapsed,
                        worker_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        phases=timing.measurements())))
        finally:
            destroy(handle)


if __name__ == '__main__':
    descriptor, cpu_id, limit_bytes, extra_bytes, budget, path, fingerprint, length = sys.argv[1:]
    try:
        manifest = dict(uri=path, sha256=fingerprint, size_bytes=int(length), schema='scar.superlu-symbolic-study')
        _child(int(descriptor), int(cpu_id), int(limit_bytes), int(extra_bytes), int(budget), manifest)
    except (OSError, ValueError, EOFError, MemoryError):
        sys.exit(1)
