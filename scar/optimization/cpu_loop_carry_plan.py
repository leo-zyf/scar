"""Choose a scoped read-only loop carry replacement without pricing observations."""

from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-loop-carry-plan'
ELIGIBLE = 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
TRANSFORMATION = 'REDIRECT_READ_ONLY_LOOP_CARRY_PREFIX'
REDIRECT = 'REDIRECT_PREFIX_READS'
_COPY = 'KEEP_COPY'
_SAME = 'KEEP_SAME_SPAN_TRANSFER'
_PROOFS = ('all_unwritten_destination_reads_accounted', 'source_stable_through_last_redirect_read',
           'write_sink_unchanged', 'unchanged_publication_and_provider_guards',
           'unqualified_route_retains_original_handoff', 'fixed_round_scope')


def _require(ok, detail):
    if not ok:
        raise ValueError('CPU loop carry plan: ' + detail)


def _require_prefix(carry):
    operation, overwrite = carry['operation'], carry['first_destination_write_operation']
    handoff, compiled_end = carry['compiled_step'], carry['first_destination_write_compiled_step']
    destination, size = carry['destination_offset'], carry['bytes']
    reads = carry['redirects']
    _require(all(type(value) is int for value in (operation, overwrite, destination, size, handoff, compiled_end))
             and 0 <= operation < overwrite and 0 <= handoff < compiled_end and destination >= 0 and size > 0
             and type(reads) is list and bool(reads), 'bounded typed carry and prefix required')
    slots = set()
    for row in reads:
        op, index, operand, offset, count, stride, item = (row[key] for key in (
            'operation', 'compiled_step', 'operand', 'relative_offset', 'count', 'stride', 'item_bytes'))
        _require(all(type(value) is int for value in (op, index, operand, offset, count, stride, item))
                 and operation < op < overwrite and handoff < index < compiled_end and operand == 0 and offset >= 0
                 and row['kind'] in ('COPY', 'MULTIPLY')
                 and count > 0 and stride >= 0 and item > 0 and offset + (count - 1) * stride + item <= size
                 and (index, operand) not in slots,
                 'redirect must name a unique typed read inside the live carry span')
        slots.add((index, operand))


def _routes(carry, arena_bytes):
    destination, size = carry['destination_offset'], carry['bytes']
    _require(destination + size <= arena_bytes, 'destination outside the retained arena')
    routes = carry['routes']
    identities = {(row['predecessor'], row['status']) for row in routes}
    first = {status for phase, status in identities if phase == 'FIRST'}
    steady = {status for phase, status in identities if phase == 'STEADY'}
    _require(len(identities) == len(routes) and first and first == steady and len(identities) == 2 * len(first),
             'all proved normal statuses must have one route in both predecessor phases')
    eligible, retained = set(), set()
    for row in routes:
        offset = row['source_offset']
        same = offset == destination
        _require(type(row['status']) is int and type(offset) is int and 0 <= offset <= arena_bytes - size
                 and type(row['redirect_eligible']) is bool and row['destination_offset'] == destination
                 and row['bytes'] == size and row['mode'] == ('TRANSFER_SAME_SPAN' if same else 'MEMCPY')
                 and row['physical_copy_bytes'] == (0 if same else size)
                 and not row['independent_live_versions']
                 and (same or offset + size <= destination or destination + size <= offset)
                 and (not same or row['redirect_eligible'] is False), 'retained pointer route or extent differs')
        (eligible if row['redirect_eligible'] else retained).add(offset)
    # A single offset guard covers both predecessor phases. Keep a copy when
    # either phase has an open proof for the same offset.
    redirected_offsets = eligible - retained
    selected = []
    for row in routes:
        offset = row['source_offset']
        action = _SAME if offset == destination else REDIRECT if offset in redirected_offsets else _COPY
        original = row['physical_copy_bytes']
        remaining = 0 if action == REDIRECT else original
        selected.append(dict(row, action=action, candidate_copy_bytes=remaining,
                             conditional_copy_bytes_saved=original - remaining))
    return selected, sorted(redirected_offsets)


def _work(routes, tasks, rounds):
    counts = {'FIRST': tasks * int(rounds > 1), 'STEADY': tasks * max(0, rounds - 2)}
    phases = {}
    for phase, count in counts.items():
        candidates = [row for row in routes if row['predecessor'] == phase]
        _require(candidates or count == 0, 'positive transition count requires proved status routes')
        savings = [row['conditional_copy_bytes_saved'] for row in candidates] or [0]
        copies = [int(value > 0) for value in savings]
        phases[phase] = dict(transition_count=count, route_counts=None,
            copy_requests_removed_lower=min(copies) * count, copy_requests_removed_upper=max(copies) * count,
            copy_payload_bytes_saved_lower=min(savings) * count, copy_payload_bytes_saved_upper=max(savings) * count)
    totals = {key: sum(row[key] for row in phases.values()) for key in (
        'transition_count', 'copy_requests_removed_lower', 'copy_requests_removed_upper',
        'copy_payload_bytes_saved_lower', 'copy_payload_bytes_saved_upper')}
    return dict(scope='ALL_TASKS_COMPLETE_SUCCESSFULLY_FOR_UNKNOWN_NORMAL_STATUS_SEQUENCE',
        accounting='REMOVED_SOURCE_COPY_REQUESTS_AND_LOGICAL_PAYLOAD_BYTES', per_predecessor=phases, totals=totals,
        observed_route_counts_used=False, physical_memory_traffic_bytes=None, elapsed_gain_ns=None, profitable=None)


def derive_cpu_loop_carry_plan(proof_ref, *, round_count=None):
    """Derive a guarded prefix candidate with conditional copy work savings."""
    from scar.analysis.cpu_loop_carry_redirect import verify_cpu_loop_carry_redirect

    proof = verify_cpu_loop_carry_redirect(proof_ref)
    scope = proof['scope']
    original_rounds, tasks = scope['round_count'], scope['task_count']
    rounds = original_rounds if round_count is None else round_count
    _require(type(rounds) is type(original_rounds) is type(tasks) is int
             and rounds > 0 and original_rounds > 0 and tasks > 0, 'positive exact task and round count required')
    report = dict(schema=SCHEMA, schema_version=1, proof=proof_ref, runtime=proof['runtime'],
        scope=scope, requested_round_count=rounds, worker_cpu_ids=proof['worker_cpu_ids'], arena=proof['arena'],
        status='REJECTED', reason='NORMAL_SUCCESS_PREFIX_REDIRECTION_PROOF_OPEN', transformation=None,
        carry=None, conditional_work=None, effect_contract=None, elapsed_gain_ns=None, profitable=None,
        execution_default='NO_OP', whole_entry_equivalence_proven=False, deployment_authorized=False)
    if rounds == 1:
        return dict(report, reason='NO_LOOP_CARRY_TRANSITIONS', conditional_work=_work((), tasks, rounds))
    if rounds != original_rounds:
        return dict(report, reason='ROUND_COUNT_OUTSIDE_EXACT_COMPILED_SCOPE')
    if proof['status'] != 'PROVEN_SCOPED_NORMAL_SUCCESS':
        return report
    if not all(proof['proof'].get(name) is True for name in _PROOFS):
        return report
    carry = proof['carry']
    _require_prefix(carry)
    arena = proof['arena']
    _require(type(arena['bytes']) is int and type(arena['alignment_bytes']) is int and arena['bytes'] > 0
             and arena['alignment_bytes'] > 0, 'carry plan requires a retained arena and alignment')
    routes, offsets = _routes(carry, arena['bytes'])
    if offsets and any(row['kind'] == 'MULTIPLY' for row in carry['redirects']):
        provider = proof['evidence']['provider']
        _require(provider is not None and provider['preserves_fp_operations'] is True
                 and provider['contract'] == 'DISJOINT_CONTIGUOUS_BROADCAST_RIGHT_COUNT_MULTIPLE_OF_FOUR',
                 'redirected arithmetic lacks the exact provider operand contract')
        alignment = provider['variable_alignment_bytes']
        _require(type(alignment) is int and alignment > 0 and arena['alignment_bytes'] % alignment == 0
                 and all((offset - carry['destination_offset']) % alignment == 0 for offset in offsets),
                 'redirected pointer crosses the proved provider alignment class')
    work = _work(routes, tasks, rounds)
    report['conditional_work'] = work
    if not offsets or work['totals']['copy_requests_removed_upper'] == 0:
        return dict(report, reason='NO_QUALIFIED_COPY_ROUTE_IN_SCOPED_TRANSITIONS')
    return dict(report, status=ELIGIBLE, reason='READ_ONLY_PREFIX_HAS_A_GUARDED_COPY_REMOVAL',
        transformation=TRANSFORMATION,
        carry=dict(carry, routes=routes, guard_source_offsets=offsets, guard_false_action='ORIGINAL_HANDOFF',
                   strategy='OVERRIDE_ONLY_PROVED_PREFIX_READ_OPERANDS', extra_arena_bytes=0),
        effect_contract=dict(scope='PROVED_NORMAL_SUCCESS_WITH_EXPLICIT_NATIVE_VALIDATION',
            preserve_original_task_and_iteration_order=True, preserve_selected_pointer_for_checkpoints_and_terminal=True,
            preserve_write_sinks_and_provider_guards=True, same_span_transfers_keep_existing_meaning=True,
            handoff_diagnostic_counts_describe_rewritten_operations=True,
            new_dispatch_overhead_ns=None, full_failure_effect_equivalence_proven=False))


def verify_cpu_loop_carry_plan(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU loop carry plan')
    replay = derive_cpu_loop_carry_plan(saved['proof'], round_count=saved['requested_round_count'])
    _require(replay == saved, 'saved carry plan differs at cold replay')
    return replay
