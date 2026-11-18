"""Project the versioned ordered sparse stream from an admitted resource plan."""

import re
from struct import calcsize

from .native_ordered_resources import SCHEMA as RESOURCE_PLAN
from .resource_solver import semantic_sha256
from .sparse_parallel_recipe import DETACHED as DETACHED_RECIPE


SCHEMA = 'scar.native-ordered-stream'
DETACHED_STREAM = 'scar.native-ordered-detached-stream'
DETACHED_BLAS_SYMBOL = 'scipy_cblas_domatcopy'
_HEADER = '<8s32s4I'
_REQUEST = '<Q'
_RESPONSE = '<QIi'


def project_native_ordered_stream(resource_plan, *, recipe_id=None):
    """Describe the raw native wire, without encoding payloads or timing transfers.

    The caller must have verified the native resource plan against its recipe.
    An independent recipe ID, when supplied, must match the projected plan.
    """
    if (resource_plan.get('schema') != RESOURCE_PLAN
            or type(resource_plan.get('schema_version')) is not int
            or resource_plan['schema_version'] != 1):
        raise ValueError('a verified native ordered resource plan is required')
    identity = resource_plan.get('recipe_id')
    if (type(identity) is not str or re.fullmatch(r'recipe-v1:[0-9a-f]{64}', identity) is None
            or recipe_id is not None and recipe_id != identity):
        raise ValueError('stream and resource plan must name the same canonical recipe')
    case, dispatch, memory = (resource_plan[key] for key in ('case', 'dispatch', 'memory'))
    size, nnz, count = (case[key] for key in ('size', 'nnz', 'call_count'))
    window = dispatch['window']
    if (any(type(value) is not int or not 0 < value < 2**31 - 1 for value in (size, nnz, count, window))
            or not 1 < window <= count or dispatch['one_inflight_per_worker'] is not True
            or dispatch['issue_order'] != 'ASCENDING_ZERO_BASED'
            or dispatch['retirement_order'] != 'OLDEST_OUTSTANDING_ONLY'
            or dispatch['output'] != 'PRIVATE_UNTIL_ORDERED_SUCCESS'):
        raise ValueError('the stream requires its bounded ordered task scope')

    header_bytes, request_header_bytes, response_header_bytes = map(calcsize, (_HEADER, _REQUEST, _RESPONSE))
    indices_bytes, offsets_bytes = 4 * nnz, 4 * (size + 1)
    values_bytes, rhs_bytes = 8 * nnz, 8 * size
    structure_bytes = indices_bytes + offsets_bytes
    request_payload_bytes = values_bytes + rhs_bytes
    response_payload_bytes = rhs_bytes
    if (structure_bytes != memory['owner_structure_bytes']
            or window * request_payload_bytes != memory['owner_pending_input_bytes']
            or window * response_payload_bytes != memory['owner_pending_result_bound_bytes']):
        raise ValueError('wire payload does not match the projected CSR and pending-frame budgets')
    request_bytes = request_header_bytes + request_payload_bytes
    success_bytes = response_header_bytes + response_payload_bytes
    terminals = dict(worker_failure=dict(kind=1, status='EXACT_NONZERO_WORKER_STATUS'),
        transport_failure=dict(kind=2, status=0), protocol_failure=dict(kind=3, status=0),
        end=dict(kind=4, status=0), host_failure=dict(kind=5, status=0))
    return dict(schema=SCHEMA, schema_version=1, recipe_id=identity,
        resource_plan_sha256=semantic_sha256(resource_plan), byte_order='LITTLE_ENDIAN',
        header=dict(format=_HEADER, fields=['magic', 'recipe_digest', 'size', 'nnz', 'count', 'flags'],
            recipe_digest_hex=identity.removeprefix('recipe-v1:'), recipe_digest_encoding='RAW_32_BYTES',
            size=size, nnz=nnz, count=count, flags=0),
        input=dict(magic='SCARCSR1', structure=[dict(name='indices', dtype='<i4', count=nnz),
                dict(name='offsets', dtype='<i4', count=size + 1)], structure_contract='CANONICAL_CSR',
            request_header=dict(format=_REQUEST, fields=['seq']),
            request_payload=[dict(name='values', dtype='<f8', count=nnz), dict(name='rhs', dtype='<f8', count=size)],
            sequence='ASCENDING_ZERO_BASED', framing='REQUIRE_EOF_BEFORE_LAST_REQUEST_ADMISSION'),
        output=dict(magic='SCAROUT1', record_header=dict(format=_RESPONSE, fields=['seq', 'kind', 'status']),
            success=dict(kind=0, status=0, payload=dict(name='solution', dtype='<f8', count=size)),
            sequence='CONTIGUOUS_SUCCESS_PREFIX', terminal_sequence='NUMBER_OF_PUBLISHED_SUCCESSES',
            terminal_payload_bytes=0, complete_terminal_kind=terminals['end']['kind']),
        terminal_kinds=terminals,
        bytes=dict(header=header_bytes, indices=indices_bytes, offsets=offsets_bytes,
            common_structure=structure_bytes, request_header=request_header_bytes, values=values_bytes, rhs=rhs_bytes,
            request_payload=request_payload_bytes, request_record=request_bytes, output_record_header=response_header_bytes,
            solution=response_payload_bytes, success_record=success_bytes, terminal_record=response_header_bytes,
            input_complete=header_bytes + structure_bytes + count * request_bytes,
            output_complete=header_bytes + count * success_bytes + response_header_bytes,
            output_prefix=dict(fixed=header_bytes + response_header_bytes, per_published=success_bytes,
                applies_to='WELL_FORMED_STREAM_WITH_HEADER_AND_TERMINAL')),
        cost=dict(basis='UNKNOWN', host_startup_ns=None, host_framing_ns=None, provider_ns=None,
            transport_ns=None, close_ns=None, whole_scope_ns=None))


def validate_native_ordered_stream(resource_plan, projection, *, recipe_id=None, detached=None):
    """Replay wire metadata from the same resource plan; no cost is inferred."""
    expected = (project_native_ordered_stream(resource_plan, recipe_id=recipe_id) if detached is None
        else project_native_detached_stream(resource_plan, detached, recipe_id=recipe_id))
    if semantic_sha256(expected) != semantic_sha256(projection):
        raise ValueError('native ordered wire projection differs from its resource plan')
    return expected


def project_native_detached_stream(resource_plan, detached, *, recipe_id=None):
    """Move prepared invariant operands to ingress and scalar-dependent work to workers."""
    raw = project_native_ordered_stream(resource_plan, recipe_id=recipe_id)
    header, sizes = raw['header'], raw['bytes']
    n, nnz, count = (header[key] for key in ('size', 'nnz', 'count'))
    window = resource_plan['dispatch']['window']
    if (len(resource_plan['placement']['workers']) != window
            or detached.get('schema') != DETACHED_RECIPE or detached.get('schema_version') != 1
            or set(detached['roles']) != {'matrix', 'load', 'boundary', 'count'}):
        raise ValueError('compact ingress requires the ordered recipe and its detached snapshot contract')

    # Slot pairs are bounded by the number of rows. Worker request bytes exclude the sequence number.
    fixed_input = sizes['header'] + sizes['common_structure'] + sizes['request_payload'] + calcsize('<I')
    ticket, slot = calcsize('<Qd'), calcsize('<ii')
    inherited_maximum = fixed_input - sizes['header'] + n * slot
    pending_input = window * calcsize('<d')
    pending_result = resource_plan['memory']['owner_pending_result_bound_bytes']
    headroom = resource_plan['memory']['minimum_launch_headroom_bytes']
    if inherited_maximum + pending_input + pending_result > (
            headroom - resource_plan['memory']['worker_address_space_sum_bytes']):
        raise ValueError('compact snapshot and pending payloads exceed the retained memory gate')

    return dict(schema=DETACHED_STREAM, schema_version=1, recipe_id=raw['recipe_id'],
        resource_plan_sha256=raw['resource_plan_sha256'], byte_order=raw['byte_order'], header=header,
        source=dict(body_code_key=detached['body_code_key'], original_scope=detached['source'],
            ingress_roles=detached['roles'], expression=detached['scale'], spans=detached['source_spans'],
            contract='EXPLICIT_CALLER_PREPARED_SNAPSHOT_AND_SCALES', original_effects_closed=False),
        input=dict(magic='SCARDSR1', structure=raw['input']['structure'],
            template_values=dict(dtype='<f8', count=nnz, selected_rows='POSITIVE_ZERO'),
            base_rhs=dict(dtype='<f8', count=n, range='FINITE_NORMAL_OR_SIGNED_ZERO_WITH_ABS_AT_MOST_DBL_MAX/2'),
            slot_count=dict(format='<I', maximum=n), slots=dict(format='<ii', fields=['rhs_row', 'matrix_diagonal_slot'],
                rows='SORTED_UNIQUE', slot='STORED_DIAGONAL_IN_SELECTED_ROW'),
            request=dict(format='<Qd', fields=['seq', 'scale'], scale='BINARY64_1_INCLUSIVE_TO_2_EXCLUSIVE'),
            sequence=raw['input']['sequence'], framing=raw['input']['framing']),
        implementation=dict(placement='SAME_PINNED_PROCESS_COHORT', immutable_inputs='INHERITED_BEFORE_FORK',
            matrix='WORKER_PRIVATE_TEMPLATE_PATCHED_AT_SELECTED_DIAGONALS',
            rhs=dict(implementation=DETACHED_BLAS_SYMBOL, policy='SCALE_MASKED_BASE_OUT_OF_PLACE',
                runtime='SAME_PROVIDER_LP64_SINGLE_THREAD_AND_DEFAULT_FP_ENVIRONMENT'),
            solve='SAME_SUPERLU_ABI', publication='SAME_ORDERED_SUCCESS_PREFIX'),
        output=raw['output'], terminal_kinds=raw['terminal_kinds'],
        bytes=dict(input_fixed=fixed_input, selected_row_slot_pair=slot, ticket=ticket,
            input_complete_base=fixed_input + count * ticket, input_complete_max=fixed_input + count * ticket + n * slot,
            output_complete=sizes['output_complete'], raw_input_complete=sizes['input_complete'],
            raw_request_bytes_per_worker=sizes['request_payload'], compact_request_bytes_per_worker=calcsize('<d')),
        memory=dict(inherited_payload_bound_bytes=inherited_maximum, owner_pending_input_bound_bytes=pending_input,
            owner_pending_result_bound_bytes=pending_result, worker_matrix_and_rhs_bytes=sizes['request_payload'],
            provider_scratch_bound_bytes=None, minimum_launch_headroom_bytes=headroom, physical_residency_proven=False),
        cost=raw['cost'])
