"""Plan a fused coefficient traversal and independently proved literal elisions."""

from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.cpu-coefficient-dispatch-plan'
ELIGIBLE = 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
TRANSFORMATION = 'FUSE_COEFFICIENT_HOOK_DISPATCH_AND_ELIDE_PROVEN_LITERAL_WRITES'
_PROOF = 'scar.cpu-coefficient-dispatch-proof'
_PROOF_STATUS = 'PROVEN_SCOPED_COEFFICIENT_DISPATCH'
_BASELINE = 'scar.native-cpu-fp-guard-groups'
_PHASES = ('FIRST', 'STEADY')
_COORDINATES = ('phase', 'begin_compiled_step', 'end_compiled_step',
                'normalization_compiled_step', 'checkpoint_compiled_step')
_FACTS = ('prepared_immutable_within_task', 'runtime_coefficient_writes_preserved',
          'normalizer_fences_preserved', 'checkpoint_publication_unchanged',
          'static_literal_equivalence_byte_exact', 'sticky_guard_bits_unchanged',
          'original_arithmetic_order_and_geometry_unchanged', 'private_probe_unchanged_required')
_INT64 = 2**63


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU coefficient dispatch plan: ' + detail)


def _span(row, arena_bytes):
    offset, count = row['destination_offset'], row['bytes']
    return type(offset) is type(count) is int and 0 <= offset < offset + count <= arena_bytes


def _phase_rows(phase, grouped, arena_bytes):
    coordinates = tuple(phase[key] for key in _COORDINATES)
    _require(coordinates == tuple(grouped[key] for key in _COORDINATES),
             'coefficient phase moved a grouped guard or terminal boundary')
    _, begin, end, normalize, checkpoint = coordinates
    _require(all(type(value) is int for value in (begin, end, normalize, checkpoint))
             and 0 <= begin < normalize < checkpoint < end, 'invalid original phase window')

    hooks, elisions = phase['hooks'], phase['elidable_literal_writes']
    hook_steps = [row['compiled_step'] for row in hooks]
    _require(hook_steps and hook_steps == sorted(set(hook_steps)) and hook_steps[-1] == normalize
             and hooks[-1]['kind'] == 'NORMALIZATION'
             and hooks[-1]['slot'] is None and hooks[-1]['original_literal_index'] is None
             and all(_span(row, arena_bytes) for row in hooks),
             'hook table needs one ordered terminal normalizer with proved storage')
    coefficients = hooks[:-1]
    for row in coefficients:
        _require(row['kind'] == 'COEFFICIENT' and begin <= row['compiled_step'] < normalize
                 and type(row['slot']) is int and row['slot'] >= 0
                 and type(row['original_literal_index']) is int and row['original_literal_index'] >= 0,
                 'prepared coefficient copy lost its original tape slot or literal index')

    omitted = {row['compiled_step'] for row in elisions}
    _require(len(omitted) == len(elisions) and omitted.isdisjoint(hook_steps),
             'literal elision cannot erase a prepared coefficient, normalizer or duplicate coordinate')
    excluded_producers = omitted | set(hook_steps)
    for row in elisions:
        index, producer, literal = (row[key] for key in ('compiled_step', 'producer_step', 'literal_index'))
        _require(type(index) is type(producer) is type(literal) is int
                 and begin <= producer < index < normalize and producer not in excluded_producers
                 and literal >= 0 and _span(row, arena_bytes),
                 'elided literal needs a distinct surviving producer in the same phase')
    return coefficients


def _work(source, baseline, grouped):
    scope, phases, grouped_phases = source['scope'], source['phases'], grouped['phases']
    tasks, rounds = scope['task_count'], scope['round_count']
    arena_bytes = baseline['decision']['memory']['private_arena_bytes_per_numerical_worker']
    _require(type(tasks) is type(rounds) is type(arena_bytes) is int
             and 0 < tasks < _INT64 and 0 < rounds < _INT64 and 0 < arena_bytes < _INT64
             and [row['phase'] for row in phases] == list(_PHASES)
             and phases[0]['end_compiled_step'] <= phases[1]['begin_compiled_step'],
             'exact positive task, round and private arena bounds required')
    phase_rows, hooks, elisions, costs = [], [], [], []
    for phase, original, invocations in zip(phases, grouped_phases, (tasks, tasks * (rounds - 1)), strict=True):
        coefficients = _phase_rows(phase, original, arena_bytes)
        phase_hooks, writes = phase['hooks'], phase['elidable_literal_writes']
        phase_rows.append(dict(
            {key: phase[key] for key in _COORDINATES}, normalization=phase['normalization'],
            hook_rows=[len(hooks), len(hooks) + len(phase_hooks)],
            literal_elision_rows=[len(elisions), len(elisions) + len(writes)]))
        hooks.extend(phase_hooks)
        elisions.extend(writes)
        copies = invocations * len(coefficients)
        prepared_bytes = invocations * sum(row['bytes'] for row in coefficients)
        stock_calls = invocations * original['stock_provider_calls']
        costs.append(dict(
            phase=phase['phase'], invocations=invocations,
            baseline_hook_table_searches=invocations, candidate_hook_table_searches=0,
            baseline_tape_dispatch_windows=invocations * len(phase_hooks), candidate_tape_dispatch_windows=invocations,
            baseline_prepared_coefficient_copies=copies, candidate_prepared_coefficient_copies=copies,
            baseline_prepared_coefficient_payload_bytes=prepared_bytes,
            candidate_prepared_coefficient_payload_bytes=prepared_bytes,
            baseline_stock_provider_calls=stock_calls, candidate_stock_provider_calls=stock_calls,
            baseline_normalizer_calls=invocations, candidate_normalizer_calls=invocations,
            removed_static_literal_writes=invocations * len(writes),
            removed_static_literal_payload_bytes=invocations * sum(row['bytes'] for row in writes)))
    totals = {name: sum(row[name] for row in costs) for name in costs[0] if name != 'phase'}
    totals['removed_hook_table_searches'] = totals['baseline_hook_table_searches']
    totals['removed_tape_dispatch_windows'] = (
        totals['baseline_tape_dispatch_windows'] - totals['candidate_tape_dispatch_windows'])
    totals['removed_prepared_coefficient_copies'] = 0
    return phase_rows, hooks, elisions, costs, totals


def derive_cpu_coefficient_dispatch_plan(proof_ref):
    """Require the proved hook order and surviving literal producers before lowering."""
    from scar.analysis.cpu_coefficient_dispatch import verify_cpu_coefficient_dispatch

    _require(proof_ref['schema'] == _PROOF, 'exact coefficient dispatch proof required')
    source = verify_cpu_coefficient_dispatch(proof_ref)
    baseline_ref = source['baseline']
    baseline = read_artifact(baseline_ref)
    grouped = read_artifact(baseline['proof'])
    _require(baseline_ref['schema'] == _BASELINE and baseline['runtime'] == grouped['runtime'] == source['runtime']
             and baseline['scope'] == grouped['scope'] == source['scope'] and source['status'] == _PROOF_STATUS
             and all(source['proof'].get(name) is True for name in _FACTS),
             'candidate differs from the frozen grouped FP runtime or effect proof')
    phases, hooks, elisions, work, totals = _work(source, baseline, grouped)
    return dict(
        schema=SCHEMA, schema_version=1, proof=proof_ref, baseline=baseline_ref,
        runtime=source['runtime'], scope=source['scope'], worker_cpu_ids=baseline['decision']['worker_cpu_ids'],
        status=ELIGIBLE, reason='FUSE_PROVED_PRODUCTION_TAPE_TRAVERSAL',
        transformation=TRANSFORMATION, phases=phases, hooks=hooks, elidable_literal_writes=elisions,
        summary=totals,
        conditional_work=dict(scope='ALL_SCOPED_TASKS_AND_ROUNDS_COMPLETE_NORMALLY',
            accounting='TYPED_TAPE_AND_SEGMENTED_SOURCE_CONTROL', per_phase=work,
            observed_counts_used=False, arena_payload_bytes_are_physical_traffic=False),
        effect_contract=dict(
            prepared_coefficient_copies_and_preparation_unchanged=True,
            eliminate_only_static_literal_writes_with_surviving_producer=True,
            coefficient_and_arithmetic_execution_order_unchanged=True,
            numeric_provider_calls_and_pointer_abi_unchanged=True,
            grouped_fp_guards_normalizer_and_publication_unchanged=True,
            original_private_probe_and_fallback_required=True),
        coverage_frontier=[
            dict(role='CONTROL_LAYOUT', status='SOURCE_WINDOWS_COUNTED', blocker='CANDIDATE_MACHINE_LAYOUT_AND_DISPATCH'),
            dict(role='ARENA_DATA', status='PROVED_STATIC_LITERAL_PAYLOAD_SAVINGS',
                 blocker='ACTUAL_CACHE_AND_MEMORY_TRAFFIC_UNKNOWN'),
            dict(role='PROFITABILITY', status='UNRESOLVED',
                 blockers=['LOWERED_CODE_LAYOUT_COST', 'CLEAN_WHOLE_ENTRY_PAIRED_BENCHMARK'])],
        target_executed=False, runtime_validated=False, original_default_retained=True,
        execution_default='NO_OP', transformation_authorized=False, whole_entry_equivalence_proven=False,
        deployment_authorized=False, forward_cost_proven=False, elapsed_gain_ns=None, profitable=None)


def verify_cpu_coefficient_dispatch_plan(reference):
    """Cold-check the retained proof and deterministic control and elision counts."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU coefficient dispatch plan')
    replay = derive_cpu_coefficient_dispatch_plan(saved['proof'])
    _require(replay == saved, 'saved coefficient dispatch plan differs from replay')
    return replay
