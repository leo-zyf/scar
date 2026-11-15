"""Inventory presence and metadata applicability never authorize execution."""
from copy import deepcopy
import ctypes
import subprocess

from scar.codegen.capabilities import (describe_stateful_capability,
    query_stateful_capability, retain_stateful_resources)


def test_stateful_domain_is_explicit_and_unsupported_resources_are_retained(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('capability query attempted native loading or a subprocess')
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    provider = dict(numpy_version='2.5.2', ndarray_c_abi=33554432,
        capsule_abi='numpy_1.24_ufunc_call_info', dtype_signature=['float64']*3,
        fixed_strides=[8, 0, 8], requires_pyapi=False,
        API_stability='EXPERIMENTAL_EXACT_NUMPY_VERSION_AND_BINARY_PIN',
        execution='ORIGINAL_NUMPY_UFUNC_STRIDED_LOOP_SERIAL_PROVIDER_STAGE',
        loaded_binary=dict(path='/declared/provider.so', sha256='a'*64, bytes=1))
    implementation = dict(schema='scar.stateful-backend-capability', schema_version=1,
        family='CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION', cxx_standard=20,
        stateful_abi_version=1, operator_abi_version=1, implementation_sha256='b'*64,
        compiler=dict(argv=['declared-compiler'], identity='DECLARED_NOT_EXECUTED', options=[
            '-std=c++20', '-fno-fast-math', '-ffp-contract=off', '-frounding-math',
            '-ftrapping-math', '-fopenmp']), provider_runtime=provider)
    request = dict(executor_kind='CPU', architecture='x86_64', os='linux', pointer_bits=64,
        byteorder='little', required_isa='BASELINE_X86_64', cpu_ids=[2, 3], cpu_threads=2,
        executor_cpu_ids=[2, 3, 4], available_cpu_ids=[2, 3, 4], memory_kind='HOST',
        memory_policy='UNBOUND', dtype='float64', rank=2, layout='C_CONTIGUOUS',
        sync_mode='SYNCHRONOUS', operations=['input', 'slice', 'power2', 'add'],
        write_operations=['add'], slice_semantics='FIXED_UNIT_STEP_2D_BOUNDS',
        memory_budget_bytes=1024, required_payload_bytes=512, provider_runtime=provider)
    before = deepcopy((request, implementation))
    result = query_stateful_capability(request, implementation=implementation)
    assert result['status'] == 'SUPPORTED_UNDER_GUARDS'
    assert not result['lowering_admitted'] and not result['deployment_authorized']
    assert not describe_stateful_capability()['isa']['emitted_isa_proven']
    assert (request, implementation) == before
    missing = deepcopy(request); missing.pop('available_cpu_ids')
    assert query_stateful_capability(missing, implementation=implementation)['status'] == 'UNRESOLVED'
    for field, value in [('executor_kind', 'GPU'), ('dtype', 'complex128'),
            ('memory_policy', 'NUMA_BIND'), ('sync_mode', 'ASYNCHRONOUS'),
            ('required_isa', 'AVX2'), ('cpu_ids', [2, 5]), ('operations', ['matmul']),
            ('memory_budget_bytes', 511), ('provider_runtime', {**provider, 'numpy_version': 'other'})]:
        bad = deepcopy(request); bad[field] = value
        assert query_stateful_capability(bad, implementation=implementation)['status'] == 'UNSUPPORTED'
    resources = [dict(id='cpu', resource_type='CpuExecutor'),
        dict(id='ram', resource_type='MemoryDomain', kind='HOST'),
        dict(id='gpu', resource_type='Accelerator'),
        dict(id='vram', resource_type='MemoryDomain', kind='DEVICE'),
        dict(id='disk', resource_type='StorageDevice'),
        dict(id='net', resource_type='NetworkInterface'),
        dict(id='other', resource_type='FutureDevice')]
    rows = retain_stateful_resources(resources)
    assert [row['resource_id'] for row in rows] == [row['id'] for row in resources]
    assert all(row['retained'] and not row['lowering_admitted'] for row in rows)
    assert all(row['status'] == 'NO_EXECUTABLE_IMPLEMENTATION_IN_THIS_PROFILE' for row in rows[2:])
    encoded = dict(id=dict(kind='target_resource', value='cpu/0', wire='target_resource:cpu/0'),
        resource_type='CpuExecutor')
    assert retain_stateful_resources([encoded])[0]['resource_id'] == 'target_resource:cpu/0'
