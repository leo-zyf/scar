"""Plan direct calls for repeated, statically qualified arithmetic tape shapes."""

from collections import Counter

from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.cpu-arithmetic-dispatch-plan'
ELIGIBLE = 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
TRANSFORMATION = 'SPECIALIZE_REPEATED_NATIVE_ARITHMETIC_DISPATCH'
_PROOF = 'scar.cpu-arithmetic-dispatch-proof'
_PROOF_STATUS = 'PROVEN_STATIC_ARITHMETIC_DISPATCH'
_BASELINE = 'scar.native-cpu-fp-guard-groups'
_BASELINE_INDIRECT = 'BASELINE_X86_INDIRECT_PROVIDER_IN_REPEAT_LOOP'
_PHASES = ('FIRST', 'STEADY')
_PROVIDERS = {'ADD': 'scar_numpy_complex_add', 'MULTIPLY': 'scar_numpy_complex_multiply'}
_INT64 = 2**63
_FACTS = ('immutable_shapes', 'arithmetic_provider_calls_and_order_preserved',
          'offsets_and_aliases_preserved', 'guard_bits_preserved',
          'no_crossing_dispatch_windows', 'private_probe_unchanged_required')


def _require(ok, detail):
    if not ok:
        raise ValueError('CPU arithmetic dispatch plan: ' + detail)


def _triple(value, *, signed=True):
    lower = -_INT64 + 1 if signed else 0
    return (type(value) is list and len(value) == 3
            and all(type(item) is int and lower <= item < _INT64 for item in value))


def _shape_table(rows):
    shapes, identities = {}, set()
    for row in rows:
        index, count, repeat = (row[name] for name in ('id', 'count', 'repeat'))
        _require(type(index) is int and index >= 0 and index not in shapes
                 and type(count) is type(repeat) is int and 0 < count < _INT64 and 2 <= repeat < _INT64
                 and _PROVIDERS.get(row['operator']) == row['provider']
                 and _triple(row['strides']) and _triple(row['advances']),
                 'shape requires one exact provider and three physical operand strides and advances')
        key = row['provider'], count, repeat, tuple(row['strides']), tuple(row['advances'])
        _require(key not in identities, 'multiple shape IDs would duplicate a specializable helper')
        shapes[index] = row
        identities.add(key)
    return shapes


def _bindings(steps, shapes, phases):
    sites = {row['phase']: [] for row in phases}
    phase_table = {row['phase']: row for row in phases}
    seen = set()
    for row in steps:
        index, operation, phase_name, shape, guard = (row[key] for key in
            ('compiled_step', 'original_operation', 'phase', 'shape', 'guard_bits'))
        phase = phase_table[phase_name]
        begin, end = row['dispatch_window']
        guard_group = row['guard_group']
        _require(type(index) is type(operation) is type(begin) is type(end) is int
                 and index not in seen and operation >= 0
                 and 0 <= begin <= index < end <= phase['end_compiled_step']
                 and phase['begin_compiled_step'] <= index < phase['normalization_compiled_step']
                 and type(shape) is int and shape in shapes and _triple(row['offsets'], signed=False)
                 and type(guard) is int and 0 <= guard <= 3
                 and (guard_group is None and guard == 0 or type(guard_group) is int and guard_group >= 0),
                 'compiled binding changed its original dispatch or floating-point guard boundary')
        sites[phase_name].append(row)
        seen.add(index)
    return sites


def _conditional_work(source):
    phases = source['phases']
    scope = source['scope']
    tasks, rounds = scope['task_count'], scope['round_count']
    _require(type(tasks) is type(rounds) is int and 0 < tasks < _INT64 and 0 < rounds < _INT64
             and [row['phase'] for row in phases] == list(_PHASES)
             and phases[0]['end_compiled_step'] <= phases[1]['begin_compiled_step'],
             'fixed positive tasks and ordered first/steady windows required')
    shapes = _shape_table(source['shapes'])
    sites = _bindings(source['steps'], shapes, phases)
    selected, invocations_by_shape, phase_work = [], Counter(), []
    for info, invocations in zip(phases, (tasks, tasks * (rounds - 1)), strict=True):
        rows = sites[info['phase']]
        repeated_calls = sum(shapes[row['shape']]['repeat'] for row in rows)
        singleton_steps, provider_calls = info['singleton_steps'], info['stock_provider_calls']
        _require(len(rows) == info['repeated_steps'] and singleton_steps >= 0
                 and info['arithmetic_steps'] == singleton_steps + len(rows)
                 and provider_calls == singleton_steps + repeated_calls,
                 'shape and singleton calls must reconstruct every original arithmetic provider occurrence')
        if invocations:
            selected.extend(rows)
            invocations_by_shape.update({shape: count * invocations for shape, count in
                                        Counter(row['shape'] for row in rows).items()})
        phase_work.append(dict(
            phase=info['phase'], invocations=invocations, provider_calls_per_invocation=provider_calls,
            repeated_provider_calls_per_invocation=repeated_calls, specialized_steps_per_invocation=len(rows),
            original_provider_calls=provider_calls * invocations, candidate_provider_calls=provider_calls * invocations,
            indirect_provider_calls_switched_to_direct=repeated_calls * invocations,
            specialized_step_dispatches=len(rows) * invocations))
    _require({row['shape'] for row in source['steps']} == shapes.keys(), 'proved shape has no original binding')
    selected.sort(key=lambda row: row['compiled_step'])
    short_list = [row for index, row in sorted(shapes.items()) if index in invocations_by_shape]
    work = dict(
        scope='ALL_TASKS_AND_ROUNDS_COMPLETE_NORMALLY',
        accounting='STATIC_GROUPED_FP_TAPE_ADD_AND_MULTIPLY_CALLS',
        replacement='ONE_SHAPE_DISPATCH_PER_SELECTED_STEP_THEN_DIRECT_PROVIDER_CALLS',
        per_phase=phase_work,
        per_shape=[dict(shape=row['id'], specialized_step_dispatches=invocations_by_shape[row['id']],
                        direct_provider_calls=invocations_by_shape[row['id']] * row['repeat'])
                   for row in short_list],
        observed_counts_used=False, machine_indirect_instruction_count_proven=False,
        elapsed_gain_ns=None, profitable=None)
    totals = dict(
        baseline_stock_provider_calls=sum(row['original_provider_calls'] for row in phase_work),
        candidate_stock_provider_calls=sum(row['candidate_provider_calls'] for row in phase_work),
        indirect_provider_calls_switched_to_direct=sum(
            row['indirect_provider_calls_switched_to_direct'] for row in phase_work),
        specialized_step_dispatches=sum(row['specialized_step_dispatches'] for row in phase_work),
        selected_shape_instantiations=len(short_list), selected_compiled_steps=len(selected),
        removed_arithmetic_provider_calls=0)
    totals['fewer_modeled_indirect_dispatches'] = (
        totals['indirect_provider_calls_switched_to_direct'] - totals['specialized_step_dispatches'])
    return short_list, selected, work, totals


def derive_cpu_arithmetic_dispatch_plan(proof_ref):
    """Choose scoped repeated shapes while retaining every original provider call."""
    from scar.analysis.cpu_arithmetic_dispatch import verify_cpu_arithmetic_dispatch

    _require(proof_ref['schema'] == _PROOF, 'exact arithmetic dispatch proof required')
    source = verify_cpu_arithmetic_dispatch(proof_ref)
    grouped = source['baseline']
    baseline = read_artifact(grouped)
    _require(grouped['schema'] == _BASELINE and baseline['runtime'] == source['runtime']
             and baseline['scope'] == source['scope'] and source['status'] == _PROOF_STATUS
             and all(source['proof'].get(key) is True for key in _FACTS),
             'arithmetic shapes must bind the exact grouped FP baseline and its unchanged effects')
    shapes, bindings, work, totals = _conditional_work(source)
    machine = source['dispatch']['machine']['status'] == _BASELINE_INDIRECT
    work.update(baseline_machine_indirect_provider_loop_proven=machine,
                baseline_dispatch_evidence='#/dispatch/machine')
    selected = bool(shapes)
    return dict(
        schema=SCHEMA, schema_version=1, proof=proof_ref, baseline=grouped,
        runtime=source['runtime'], scope=source['scope'], worker_cpu_ids=baseline['decision']['worker_cpu_ids'],
        status=ELIGIBLE if selected else 'REJECTED',
        reason='REPEATED_INDIRECT_ARITHMETIC_SOURCE_DISPATCH' if selected else 'NO_ACTIVE_REPEATED_SHAPE',
        transformation=TRANSFORMATION if selected else None,
        shapes=shapes, bindings=bindings, summary=totals, conditional_work=work,
        code_growth=dict(baseline_executable=baseline['executable'],
            candidate_shape_helpers_upper=len(shapes), candidate_lookup_rows_upper=len(bindings),
            express_each_shape_as_one_loop=True, expand_repeats_in_generated_source=False,
            arithmetic_provider_bytes_changed=False, candidate_machine_text_bytes=None),
        effect_contract=dict(original_arithmetic_provider_and_pointer_abi_unchanged=True,
            provider_call_count_and_order_unchanged=True, operand_addresses_and_strides_unchanged=True,
            fenv_check_suppression_and_boundaries_unchanged=True,
            hooks_checkpoints_and_publication_unchanged=True,
            scoped_grouped_baseline_effect_contract_retained=True),
        coverage_frontier=[
            dict(role='ARITHMETIC_PROVIDER', status='CALL_SEQUENCE_RETAINED',
                 blocker='EMITTED_DIRECT_CALL_SITE_AND_LINKAGE_VERIFICATION'),
            dict(role='COST', status='UNMEASURED',
                 blockers=['CANDIDATE_MACHINE_DISPATCH', 'MACHINE_CODE_FOOTPRINT', 'CLEAN_PAIRED_WHOLE_ENTRY_BENCHMARK'])],
        target_executed=False, runtime_validated=False, original_default_retained=True,
        execution_default='NO_OP', transformation_authorized=False,
        whole_entry_equivalence_proven=False, deployment_authorized=False,
        forward_cost_proven=False, elapsed_gain_ns=None, profitable=None)


def verify_cpu_arithmetic_dispatch_plan(reference):
    """Cold-check the arithmetic proof, frozen baseline and deterministic shortlist."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported arithmetic dispatch plan')
    replay = derive_cpu_arithmetic_dispatch_plan(saved['proof'])
    _require(saved == replay, 'saved arithmetic dispatch plan differs from replay')
    return replay
