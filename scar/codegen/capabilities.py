"""Metadata-only applicability of the existing transactional CPU backend.

An inventory resource is not an executable implementation. These queries do
not probe hardware, resolve NumPy capsules, invoke a compiler, load a library,
or admit a source region. Exact implementation records come from the existing
``stateful_backend_capability(plan)`` producer and still require recipe replay.
"""
from copy import deepcopy
from enum import Enum
from pathlib import PurePath

SCHEMA = 'scar.executable-implementation-capability'
FAMILY = 'CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION'
PROFILE = 'cpu-stateful-array-region-v1'
_OPS = ('input', 'const', 'slice', 'add', 'sub', 'mul', 'div', 'neg', 'power2')
_WRITES = ('assign', 'add', 'sub', 'mul', 'div')
_GUARDS = (
    'ORIGINAL_REGION_SOURCE_CALLEE_PROVIDER_AND_ORDERED_PLAN_REPLAY',
    'ORIGINAL_TARGET_SNAPSHOT_AND_CPU_MEMORY_ACCESS_RECEIPTS',
    'EXACT_IMPLEMENTATION_COMPILER_AND_PROVIDER_BINARY_CONTEXT',
    'GENERATED_LIBRARY_AND_PROVIDER_ISA_COMPATIBLE_WITH_EVERY_SELECTED_CPU',
    'EXACT_BASE_NDARRAY_NATIVE_FLOAT64_RANK2_C_CONTIGUOUS_ALIGNED_BOUNDED_OWNER',
    'EXACT_INPUT_SHAPES_WRITABLE_TARGETS_AND_DISJOINT_EXTERNAL_ROOTS',
    'FINITE_INPUTS_AND_TAGGED_EXACT_SCALARS',
    'ALL_REGION_ARRAY_PAYLOAD_WITHIN_EXPLICIT_BUDGET_AND_ALLOCATION_SUCCEEDS',
    'LIVE_CPU_MASK_OPENMP_TEAM_SIZE_AND_TEMPORARY_AFFINITY_RESTORE',
    'FENV_AND_PROVIDER_SUCCESS_BEFORE_EXTERNAL_COMMIT',
    'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS',
    'FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT',
)


def _implementation(value):
    if value is None:
        return None
    if (type(value) is not dict or value.get('schema') != 'scar.stateful-backend-capability'
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('family') != FAMILY or value.get('cxx_standard') != 20
            or value.get('stateful_abi_version') != 1 or value.get('operator_abi_version') != 1):
        raise ValueError('existing stateful backend capability record required')
    digest = value.get('implementation_sha256')
    if (type(digest) is not str or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or type(value.get('compiler')) is not dict
            or type(value.get('provider_runtime')) is not dict):
        raise ValueError('pinned implementation/compiler/provider metadata required')
    return deepcopy(value)


def describe_stateful_capability(*, implementation=None):
    """Describe implemented roles; an optional existing record pins context.

    Supplying a record does not authenticate it. The established stateful
    recipe verifier/build and runtime admission remain the authority for that.
    """
    return dict(schema=SCHEMA, schema_version=1, profile=PROFILE, family=FAMILY,
        implementation_record=_implementation(implementation),
        execution=dict(executor_kind='CPU', mode='SYNCHRONOUS', cxx_standard=20,
            stateful_abi_version=1, parallelism='OPENMP_STATIC_ROWS',
            cpu_placement='ONE_TEMPORARILY_PINNED_THREAD_PER_RECORDED_CPU',
            provider_parallelism='SERIAL_PROVIDER_STAGE',
            temporal_fusion=False, process_parallelism=False),
        process=dict(os='linux', architecture='x86_64', pointer_bits=64, byteorder='little'),
        isa=dict(profile='BASELINE_X86_64', explicit_packed_lowering=False,
            emitted_isa_proven=False, compiler_flags_are_isa_proof=False,
            provider_dispatch_isa_proven=False),
        arrays=dict(dtype='float64', native_byteorder=True, rank=2,
            layout='C_CONTIGUOUS', alignment_bytes=8, array_broadcast=False,
            scalar_broadcast=True, slice_semantics='FIXED_UNIT_STEP_2D_BOUNDS'),
        scalars=dict(kinds=['PYTHON_FLOAT', 'PYTHON_INT_ABS_LE_2_POW_53'],
            integer_only_arithmetic='CHECKED_EXACT_PRECOMMIT', finite=True),
        operations=list(_OPS), write_operations=list(_WRITES),
        provider=dict(operation='power2', module='numpy', symbol='power',
            numpy_version='2.5.2', capsule_abi='numpy_1.24_ufunc_call_info',
            ndarray_c_abi=33554432, dtype_signature=['float64']*3,
            fixed_strides=[8, 0, 8], requires_pyapi=False,
            policy='EXACT_BINARY_AND_UFUNC_IDENTITY_PIN_NOT_MULTIPLICATION_REWRITE'),
        memory=dict(kind='HOST', policy='UNBOUND',
            budget_scope='ALL_REGION_ARRAY_PAYLOAD_EXTERNAL_STATE_PLUS_FIXED_WORKSPACE',
            allocation='PREALLOCATED_FIXED_POOL_PRIVATE_SHADOWS_THEN_ORDERED_ROI_COMMIT',
            numa_page_placement=False, rss_limit=False, compiler_stack_cap=False),
        unsupported_execution=['GPU', 'DEVICE_MEMORY', 'STORAGE_IO', 'NETWORK_IO',
            'COLLECTIVE_COMMUNICATION', 'ASYNC_COPY', 'MPI', 'PROCESS_PARTITIONING'],
        admission_guards=list(_GUARDS), lowering_admitted=False,
        deployment_authorized=False, performance_estimated=False)


def _cpu_mask(value):
    return (type(value) in (list, tuple) and bool(value)
            and all(type(cpu) is int and 0 <= cpu < 2**20 for cpu in value)
            and len(set(value)) == len(value))


def _provider_supported(value):
    if type(value) is not dict: return False
    expected = dict(numpy_version='2.5.2', ndarray_c_abi=33554432,
        capsule_abi='numpy_1.24_ufunc_call_info', dtype_signature=['float64']*3,
        fixed_strides=[8, 0, 8], requires_pyapi=False,
        API_stability='EXPERIMENTAL_EXACT_NUMPY_VERSION_AND_BINARY_PIN',
        execution='ORIGINAL_NUMPY_UFUNC_STRIDED_LOOP_SERIAL_PROVIDER_STAGE')
    if any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items()):
        return False
    binary = value.get('loaded_binary')
    return (type(binary) is dict and type(binary.get('path')) is str
        and PurePath(binary['path']).is_absolute() and type(binary.get('bytes')) is int
        and binary['bytes'] > 0 and type(binary.get('sha256')) is str
        and len(binary['sha256']) == 64 and all(c in '0123456789abcdef' for c in binary['sha256']))


def query_stateful_capability(request, *, implementation=None):
    """Check a proposed implementation domain without executing or probing.

    Missing applicability facts are UNRESOLVED; contradictory or unsupported
    requests are UNSUPPORTED. SUPPORTED_UNDER_GUARDS permits consideration as a
    candidate, never deployment or bypassing semantic/source/target admission.
    ``available_cpu_ids`` must be supplied by the caller's live-affinity scope.
    """
    capability = describe_stateful_capability(implementation=implementation)
    expected = dict(executor_kind='CPU', architecture='x86_64', os='linux',
        pointer_bits=64, byteorder='little', required_isa='BASELINE_X86_64',
        memory_kind='HOST', memory_policy='UNBOUND', dtype='float64', rank=2,
        layout='C_CONTIGUOUS', sync_mode='SYNCHRONOUS')
    allowed = set(expected) | {'cpu_ids', 'executor_cpu_ids', 'available_cpu_ids', 'cpu_threads',
        'operations', 'write_operations', 'slice_semantics', 'provider_runtime',
        'memory_budget_bytes', 'required_payload_bytes'}
    if type(request) is not dict or set(request) - allowed:
        raise ValueError('bounded stateful capability request with known fields required')
    unsupported, unresolved = [], []
    def reject(field, reason): unsupported.append(dict(field=field, reason=reason))
    def unknown(field, reason): unresolved.append(dict(field=field, reason=reason))
    for field, required in expected.items():
        if field not in request or request[field] is None:
            unknown(field, 'APPLICABILITY_FACT_REQUIRED')
        elif type(request[field]) is not type(required) or request[field] != required:
            reject(field, 'NOT_IMPLEMENTED_BY_THIS_PROFILE')
    for field in ('cpu_ids', 'executor_cpu_ids', 'available_cpu_ids'):
        if field not in request or request[field] is None:
            unknown(field, 'EXPLICIT_CPU_MASK_REQUIRED')
        elif not _cpu_mask(request[field]):
            reject(field, 'NONEMPTY_UNIQUE_BOUNDED_CPU_IDS_REQUIRED')
    threads = request.get('cpu_threads')
    if threads is None:
        unknown('cpu_threads', 'EXPLICIT_WORKER_COUNT_REQUIRED')
    elif type(threads) is not int or threads < 1:
        reject('cpu_threads', 'POSITIVE_EXACT_INTEGER_REQUIRED')
    elif _cpu_mask(request.get('cpu_ids')) and threads != len(request['cpu_ids']):
        reject('cpu_threads', 'ONE_WORKER_PER_RECORDED_CPU_REQUIRED')
    for field in ('executor_cpu_ids', 'available_cpu_ids'):
        if (_cpu_mask(request.get('cpu_ids')) and _cpu_mask(request.get(field))
                and not set(request['cpu_ids']) <= set(request[field])):
            reject('cpu_ids', 'REQUESTED_CPU_NOT_WITHIN_' + field.upper())
    for field, supported in (('operations', _OPS), ('write_operations', _WRITES)):
        values = request.get(field)
        if values is None:
            unknown(field, 'EXPLICIT_OPERATION_DOMAIN_REQUIRED')
        elif type(values) not in (list, tuple) or not all(type(v) is str for v in values):
            reject(field, 'OPERATION_NAMES_REQUIRED')
        elif set(values) - set(supported):
            reject(field, 'UNSUPPORTED_OPERATIONS:' + ','.join(sorted(set(values)-set(supported))))
    operations = request.get('operations')
    operations = operations if type(operations) in (list, tuple) else ()
    if 'slice' in operations:
        if request.get('slice_semantics') is None:
            unknown('slice_semantics', 'FIXED_UNIT_STEP_ALIAS_BOUNDS_REQUIRED')
        elif request['slice_semantics'] != 'FIXED_UNIT_STEP_2D_BOUNDS':
            reject('slice_semantics', 'ONLY_FIXED_UNIT_STEP_2D_BOUNDS_IMPLEMENTED')
    for field in ('memory_budget_bytes', 'required_payload_bytes'):
        value = request.get(field)
        if value is None:
            unknown(field, 'EXPLICIT_COMPLETE_REGION_PAYLOAD_BUDGET_REQUIRED')
        elif type(value) is not int or not 0 <= value <= 2**63-1:
            reject(field, 'NONNEGATIVE_PTRDIFF_BOUNDED_BYTE_COUNT_REQUIRED')
    budget, payload = request.get('memory_budget_bytes'), request.get('required_payload_bytes')
    if type(budget) is int and type(payload) is int and payload > budget:
        reject('memory_budget_bytes', 'COMPLETE_REGION_PAYLOAD_EXCEEDS_BUDGET')
    record = capability['implementation_record']
    if record is None:
        unknown('implementation', 'PINNED_EXISTING_BACKEND_RECORD_REQUIRED')
    if 'power2' in operations:
        provider = request.get('provider_runtime')
        if provider is None:
            unknown('provider_runtime', 'EXACT_NUMPY_PROVIDER_RUNTIME_RECORD_REQUIRED')
        elif not _provider_supported(provider):
            reject('provider_runtime', 'UNSUPPORTED_NUMPY_VERSION_ABI_OR_BINARY_PIN')
        elif record is not None and provider != record['provider_runtime']:
            reject('provider_runtime', 'PROVIDER_CONTEXT_DIFFERS_FROM_IMPLEMENTATION')
    if record is not None:
        if not _provider_supported(record['provider_runtime']):
            reject('implementation.provider_runtime', 'UNSUPPORTED_PINNED_PROVIDER_CONTEXT')
        options = record['compiler'].get('options', ())
        if type(options) not in (list, tuple) or not all(type(v) is str for v in options):
            reject('implementation.compiler', 'COMPILER_OPTIONS_RECORD_REQUIRED')
        elif type(threads) is int and (('-fopenmp' in options) != (threads > 1)):
            reject('implementation.compiler', 'OPENMP_OPTIONS_DO_NOT_MATCH_WORKER_DOMAIN')
        elif not {'-std=c++20', '-fno-fast-math', '-ffp-contract=off',
                  '-frounding-math', '-ftrapping-math'} <= set(options):
            reject('implementation.compiler', 'STRICT_FP_CXX20_OPTIONS_REQUIRED')
    return dict(schema='scar.implementation-applicability', schema_version=1,
        family=FAMILY, profile=PROFILE,
        status='UNSUPPORTED' if unsupported else 'UNRESOLVED' if unresolved else 'SUPPORTED_UNDER_GUARDS',
        request=deepcopy(request), unsupported=unsupported, unresolved=unresolved,
        admission_guards=capability['admission_guards'], lowering_admitted=False,
        deployment_authorized=False, performance_estimated=False,
        scope='IMPLEMENTATION_DOMAIN_ONLY_NOT_SOURCE_PLAN_RECEIPT_OR_BINARY_ISA_ADMISSION')


def retain_stateful_resources(resources):
    """Retain every resource ID, including unsupported and unfamiliar types.

    Accept typed target resources or encoded mappings. This is a role view of
    an existing graph, not topology verification, discovery or device pruning.
    """
    result, seen = [], set()
    for resource in resources:
        if isinstance(resource, dict):
            kind, identifier = resource.get('resource_type', 'UNKNOWN'), resource.get('id')
            memory_kind = resource.get('kind')
        else:
            kind, identifier = type(resource).__name__, getattr(resource, 'id', None)
            memory_kind = getattr(resource, 'kind', None)
        if isinstance(memory_kind, Enum): memory_kind = memory_kind.value
        if isinstance(identifier, dict): identifier = identifier.get('wire')
        if identifier is None or str(identifier) in seen:
            raise ValueError('unique original resource IDs required; none may be discarded')
        identifier = str(identifier); seen.add(identifier)
        if kind == 'CpuExecutor':
            role, status = 'CPU_EXECUTOR', 'IMPLEMENTED_ROLE_REQUIRES_APPLICABILITY_AND_ADMISSION'
        elif kind == 'MemoryDomain' and memory_kind == 'HOST':
            role, status = 'UNBOUND_HOST_ARRAY_STORAGE', 'IMPLEMENTED_ROLE_NO_PAGE_PLACEMENT'
        else:
            role, status = 'RETAINED_INVENTORY_BOUNDARY', 'NO_EXECUTABLE_IMPLEMENTATION_IN_THIS_PROFILE'
        result.append(dict(resource_id=identifier, resource_type=kind, role=role,
            status=status, retained=True, lowering_admitted=False))
    return result


__all__ = ['describe_stateful_capability', 'query_stateful_capability', 'retain_stateful_resources']
