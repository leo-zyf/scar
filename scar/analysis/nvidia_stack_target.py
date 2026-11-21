"""Observe the effective NVIDIA driver and CUDA device identities on Linux.

The isolated driver query initializes discovery only. A separate backend
trial is required for allocation, synchronization, execution, and timing.
"""

import argparse
import ctypes as C
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import UUID


SCHEMA = 'scar.nvidia-target-probe'
# Match the standalone launch environment in compile_exported_aten_cuda.
CUDA_ENVIRONMENT = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MPS_PIPE_DIRECTORY')
_LIBRARY = 'libcuda.so.1'
_NO_DEVICE = 100
_COMPUTE_ATTRIBUTES = dict(multiprocessors=16, compute_capability_major=75, compute_capability_minor=76)
_MAX_DEVICES = 1024


class _CudaError(Exception):
    def __init__(self, operation, status, name):
        self.operation = operation
        self.status = status
        self.name = name


def _function(library, name, result, *arguments):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


def _call(library, name, signatures, *arguments):
    result = _function(library, name, C.c_int, *signatures)(*arguments)
    if result != 0:
        status_name = C.c_char_p()
        lookup = _function(library, 'cuGetErrorName', C.c_int, C.c_int, C.POINTER(C.c_char_p))
        name_bytes = status_name.value if lookup(result, C.byref(status_name)) == 0 else None
        raise _CudaError(name, result, None if name_bytes is None else name_bytes.decode('ascii', errors='replace'))


def _stamp(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def _loaded_library(library):
    class DlInfo(C.Structure):
        _fields_ = [('path', C.c_char_p), ('base', C.c_void_p),
                    ('symbol', C.c_char_p), ('address', C.c_void_p)]

    info = DlInfo()
    dladdr = _function(C.CDLL(None), 'dladdr', C.c_int, C.c_void_p, C.POINTER(DlInfo))
    if not dladdr(C.cast(library.cuInit, C.c_void_p), C.byref(info)) or not info.path:
        raise ValueError('CUDA driver loader did not identify the loaded library')
    loaded = Path(os.fsdecode(info.path)).resolve(strict=True)
    return dict(path=str(loaded), stat=_stamp(loaded.stat()))


def _scalar(library, name, kind, signature=(), values=()):
    result = kind()
    _call(library, name, (C.POINTER(kind), *signature), C.byref(result), *values)
    return result.value


def _device(library, ordinal):
    handle = _scalar(library, 'cuDeviceGet', C.c_int, (C.c_int,), (ordinal,))
    name = C.create_string_buffer(256)
    pci = C.create_string_buffer(32)
    for operation, buffer in (('cuDeviceGetName', name), ('cuDeviceGetPCIBusId', pci)):
        _call(library, operation, (C.POINTER(C.c_char), C.c_int, C.c_int), buffer, len(buffer), handle)
    uuid = (C.c_ubyte * 16)()
    # The v2 ABI distinguishes MIG instances that share a PCI parent.
    uuid_interface = 'cuDeviceGetUuid_v2' if hasattr(library, 'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid'
    _call(library, uuid_interface, (C.POINTER(type(uuid)), C.c_int), C.byref(uuid), handle)
    caps = {key: _scalar(library, 'cuDeviceGetAttribute', C.c_int, (C.c_int, C.c_int), (value, handle))
        for key, value in _COMPUTE_ATTRIBUTES.items()}
    memory = _scalar(library, 'cuDeviceTotalMem_v2', C.c_size_t, (C.c_int,), (handle,))
    return dict(ordinal=ordinal, uuid=str(UUID(bytes=bytes(uuid))), uuid_interface=uuid_interface,
        name=name.value.decode('utf-8', errors='replace'), pci_bus_id=_pci_address(pci.value.decode('ascii')),
        memory_total_bytes=memory, **caps)


def _pci_address(text):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', text or '')
    if match is None:
        return None
    return f'{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}'


def _driver_worker():
    report = dict(status='DRIVER_UNAVAILABLE', driver_library=None, driver_api_version=None,
        device_count=None, visible_devices=[], failure=None)
    try:
        library = C.CDLL(_LIBRARY)
    except OSError as error:
        report['failure'] = dict(operation='LOAD_CUDA_DRIVER', detail=str(error))
        return report
    try:
        report['driver_library'] = _loaded_library(library)
        _call(library, 'cuInit', (C.c_uint,), 0)
        report['driver_api_version'] = _scalar(library, 'cuDriverGetVersion', C.c_int)
        count = _scalar(library, 'cuDeviceGetCount', C.c_int)
        if not 0 <= count <= _MAX_DEVICES:
            raise ValueError('CUDA driver reported a device count outside the observation bound')
        report['device_count'] = count
        for ordinal in range(count):
            report['visible_devices'].append(_device(library, ordinal))
        report['status'] = 'DEVICE_QUERY_SUCCEEDED' if count else 'NO_VISIBLE_DEVICES'
    except _CudaError as error:
        if error.operation in ('cuInit', 'cuDeviceGetCount') and error.status == _NO_DEVICE:
            report.update(status='NO_VISIBLE_DEVICES', device_count=0)
        elif error.operation != 'cuInit':
            report['status'] = 'QUERY_FAILED'
        report['failure'] = dict(operation=error.operation, cuda_status=error.status, cuda_name=error.name)
    except (AttributeError, OSError, ValueError) as error:
        report.update(status='QUERY_FAILED', failure=dict(operation='READ_DRIVER_IDENTITY', detail=str(error)))
    return report


def _observe_driver(environment):
    from scar.runtime.aten_provider_identity import verify_library_file

    command = [str(Path(sys.executable).resolve()), '-I', '-S', str(Path(__file__).resolve()), '--driver-worker']
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=10,
            env=dict(environment, CUDA_FORCE_PRELOAD_LIBRARIES='0'))
        facts = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return dict(status='QUERY_FAILED', driver_library=None, driver_api_version=None,
            device_count=None, visible_devices=[], failure=dict(operation='DRIVER_PROBE_SUBPROCESS', detail=str(error)))
    lock = facts['driver_library']
    try:
        if lock is not None:
            path = Path(lock['path'])
            with path.open('rb') as file:
                if _stamp(os.fstat(file.fileno())) != lock['stat']:
                    raise ValueError('loaded CUDA driver library changed before pinning')
                lock['bytes'] = verify_library_file(file)
                if _stamp(path.stat()) != lock['stat'] or _stamp(os.fstat(file.fileno())) != lock['stat']:
                    raise ValueError('loaded CUDA driver library changed during pinning')
    except (OSError, ValueError) as error:
        return dict(status='QUERY_FAILED', driver_library=None, driver_api_version=None,
            device_count=None, visible_devices=[], failure=dict(operation='PIN_LOADED_DRIVER', detail=str(error)))
    devices = facts['visible_devices']
    if (facts['status'] == 'DEVICE_QUERY_SUCCEEDED' and
            (facts['driver_api_version'] <= 0 or facts['device_count'] != len(devices)
             or [row['ordinal'] for row in devices] != list(range(len(devices)))
             or len({row['uuid'] for row in devices}) != len(devices)
             or any(row['pci_bus_id'] is None or row['memory_total_bytes'] <= 0
                    or row['multiprocessors'] <= 0 or row['compute_capability_major'] <= 0
                    or row['compute_capability_minor'] < 0 for row in devices))):
        facts.update(status='QUERY_FAILED', failure=dict(operation='VALIDATE_DEVICE_IDENTITY',
            detail='CUDA driver reported an incomplete device identity'))
    return facts


def _target(target):
    from scar.analysis.target_graph import TargetHardwareGraph
    from scar.ir.execution_recipe import artifact_reference, read_artifact

    if target is None:
        return None, ()
    if isinstance(target, dict):
        if target.get('schema') != 'scar.target-hardware-graph':
            raise ValueError('NVIDIA target requires the original target-graph receipt')
        read_artifact(target)
        graph = TargetHardwareGraph(Path(target['uri']).parent)
    elif isinstance(target, TargetHardwareGraph):
        graph = target
    else:
        path = Path(target).resolve()
        graph = TargetHardwareGraph(path.parent if path.name == 'result.json' else path)
    reference = artifact_reference(graph.path/'result.json', 'scar.target-hardware-graph')
    if ((isinstance(target, dict) and target != reference) or read_artifact(reference) != graph.report):
        raise ValueError('retained target graph changed during NVIDIA target binding')
    return reference, graph.resources()


def _join(devices, resources, reference):
    from scar.ir.program.target_hardware import Accelerator, DiscoveredAccelerator, MemoryDomain, PciFunction

    for device in devices:
        pci = device['pci_bus_id']
        exact = [row.id for row in resources if type(row) is Accelerator and row.family == 'cuda'
                 and row.uuid.lower().removeprefix('gpu-').removeprefix('mig-') == device['uuid']
                 and pci is not None and _pci_address(row.pci_bus_id) == pci]
        parent = [row.id.wire for row in resources
                  if type(row) in (Accelerator, DiscoveredAccelerator, PciFunction)
                  and pci is not None and _pci_address(row.pci_bus_id) == pci and row.id not in exact]
        if reference is None:
            status = 'TARGET_GRAPH_NOT_SUPPLIED'
        elif len(exact) == 1:
            status = 'UUID_AND_PCI_MATCH'
        elif len(exact) > 1:
            status = 'AMBIGUOUS_RESOURCES'
        else:
            status = 'PCI_PARENT_ONLY' if parent else 'MISSING_FROM_RETAINED_TARGET'
        device['retained_target'] = dict(status=status, accelerator_ids=[identity.wire for identity in exact],
            parent_pci_ids=parent, memory_ids=[row.id.wire for row in resources
                if type(row) is MemoryDomain and row.owner in exact])


def probe_nvidia_target(*, target=None):
    """Return direct driver evidence and an optional exact target-graph join."""
    reference, resources = _target(target)
    env = {name: os.environ[name] for name in CUDA_ENVIRONMENT if name in os.environ}
    host = os.uname()
    scope = dict(system=host.sysname, architecture=host.machine, kernel_release=host.release,
        cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_environment=env)
    facts = _observe_driver(env)
    _join(facts['visible_devices'], resources, reference)
    return dict(schema=SCHEMA, schema_version=1, target=reference, scope=scope, **facts,
        probe_implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        device_context_created=False, workload_executed=False, backend_qualified=False,
        allocation_budget_bytes=None, performance_measured=False)


def verify_nvidia_target(record, *, target=None):
    """Replay current effective device scope and original target identity."""
    actual = probe_nvidia_target(target=target)
    if actual != record:
        changed = sorted(key for key in actual.keys() | record.keys() if actual.get(key) != record.get(key))
        raise ValueError('NVIDIA target evidence changed in ' + ', '.join(changed))
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', help='retained target hardware graph directory')
    args = parser.parse_args(argv)
    print(json.dumps(probe_nvidia_target(target=args.target), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    if sys.argv[1:] == ['--driver-worker']:
        print(json.dumps(_driver_worker(), allow_nan=False))
    else:
        raise SystemExit(main())
