"""Plan contiguous floating-point guard groups on proved native tape windows."""

from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fp-guard-plan'
ELIGIBLE = 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
TRANSFORMATION = 'GROUP_CONTIGUOUS_FP_ENVIRONMENT_GUARDS'
_RUNTIME = 'scar.native-seeded-array-task-entry'
_PHASES = ('FIRST', 'STEADY')
_DIAGNOSTICS = ('SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT')
_INTERPOSITION = ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH')
_MONITORED = ('FE_DIVBYZERO', 'FE_INVALID', 'FE_OVERFLOW', 'FE_UNDERFLOW')
_SUCCESS_FACTS = ('fp_operation_order_unchanged', 'normal_publication_unchanged',
                  'flags_are_sticky_within_group', 'traps_masked', 'guard_provider_exact',
                  'no_owner_publication_on_failure', 'diagnostics_preserved_on_normal_success')


def _require(ok, detail):
    if not ok:
        raise ValueError('CPU FP guard plan: ' + detail)


def _phase(phase, invocations):
    begin, end = (phase[name] for name in ('begin_compiled_step', 'end_compiled_step'))
    baseline, candidate = (phase[name] for name in ('baseline_guard_pairs', 'candidate_guard_pairs'))
    arithmetic, groups = (phase[name] for name in ('arithmetic_steps', 'all_groups'))
    _require(all(type(value) is int for value in (begin, end, baseline, candidate, arithmetic, groups))
             and 0 <= begin < end and 0 <= groups == candidate <= baseline == arithmetic <= end - begin
             and type(phase['groups']) is list, 'invalid proved phase window or guard count')
    last, covered, selected = begin, 0, []
    for row in phase['groups']:
        first, stop = row['begin'], row['end']
        _require(type(first) is type(stop) is int and last <= first < stop <= end and stop - first > 1,
                 'group must be contiguous and disjoint inside one original phase window')
        covered += stop - first
        if invocations:
            selected.append(dict(phase=phase['phase'], compiled_steps=[first, stop],
                                 baseline_guard_pairs=stop - first, candidate_guard_pairs=1))
        last = stop
    _require(covered <= baseline and baseline - covered + len(phase['groups']) == candidate,
             'grouping must account for every original and retained guard pair')
    return selected, dict(phase=phase['phase'], compiled_steps=[begin, end], invocations=invocations,
        baseline_guard_pairs_per_invocation=baseline, candidate_guard_pairs_per_invocation=candidate,
        original_guard_pairs=invocations * baseline, candidate_guard_pairs=invocations * candidate,
        removed_guard_pairs=invocations * (baseline - candidate))


def _work(phases, tasks, rounds):
    _require(type(phases) is list and [phase['phase'] for phase in phases] == list(_PHASES)
             and phases[0]['end_compiled_step'] <= phases[1]['begin_compiled_step'],
             'ordered first and steady tape windows required')
    groups, counts = [], []
    for phase, invocations in zip(phases, (tasks, tasks * (rounds - 1)), strict=True):
        selected, accounted = _phase(phase, invocations)
        groups.extend(selected)
        counts.append(accounted)
    baseline, candidate = (sum(row[key] for row in counts) for key in ('original_guard_pairs', 'candidate_guard_pairs'))
    saved = baseline - candidate
    return groups, dict(scope='ALL_TASKS_COMPLETE_SUCCESSFULLY',
        accounting='ONE_CLEAR_AND_ONE_TEST_PER_ORIGINAL_GUARDED_STEP_OR_CANDIDATE_GROUP',
        provider_repeats_and_other_guards_unchanged=True, per_phase=counts,
        totals=dict(original_guard_pairs=baseline, candidate_guard_pairs=candidate, removed_guard_pairs=saved,
                    original_guard_calls=2 * baseline, candidate_guard_calls=2 * candidate,
                    removed_clear_calls=saved, removed_test_calls=saved, removed_guard_calls=2 * saved),
        observed_counts_used=False, elapsed_gain_ns=None, profitable=None)


def derive_cpu_fp_guard_plan(proof_ref):
    """Choose proved local flag checks and count normal-path work, without running code."""
    from scar.analysis.cpu_fp_guard_groups import verify_cpu_fp_guard_groups

    source = verify_cpu_fp_guard_groups(proof_ref)
    runtime_ref, scope = source['runtime'], source['scope']
    report = dict(schema=SCHEMA, schema_version=1, proof=proof_ref, runtime=runtime_ref, scope=scope,
        worker_cpu_ids=source.get('worker_cpu_ids'), status='REJECTED', reason='MASKED_STICKY_FP_PROOF_OPEN',
        transformation=None, groups=[], conditional_work=None, effect_contract=None, elapsed_gain_ns=None,
        profitable=None, execution_default='NO_OP', whole_entry_equivalence_proven=False, deployment_authorized=False)
    if source['status'] != 'PROVEN_SCOPED_MASKED_STICKY_FLAGS':
        return report
    facts = source['proof']
    if (not all(facts.get(name) is True for name in _SUCCESS_FACTS)
            or facts.get('rounding_mode') != 'FE_TONEAREST'
            or facts.get('monitored_exceptions') != list(_MONITORED)):
        return report
    if (facts.get('diagnostic_environment_variables') != list(_DIAGNOSTICS)
            or facts.get('failure_qualification') != 'CLEAN_NO_DIAGNOSTICS_FIXED_OPENMP_LAUNCH'
            or type(facts.get('same_failure_code')) is not int or facts['same_failure_code'] != 3):
        return dict(report, reason='DIAGNOSTIC_OR_FAILURE_EFFECT_CONTRACT_OPEN')
    runtime = read_artifact(runtime_ref)
    count, rounds = scope['task_count'], scope['round_count']
    _require(runtime_ref['schema'] == _RUNTIME and runtime['scope'] == scope
             and runtime['decision']['mode'] == 'ORDERED_OPENMP'
             and runtime['decision']['worker_cpu_ids'] == source['worker_cpu_ids']
             and type(count) is type(rounds) is int and 0 < count < 2**63 and 0 < rounds < 2**63
             and runtime['original_default_retained'] is True and runtime['deployment_authorized'] is False,
             'group counts must bind the exact original native provider and positive execution bounds')
    environment = dict(read_artifact(runtime['recipe'])['environment'], **runtime['launch_environment'])
    unset = (*_DIAGNOSTICS, *_INTERPOSITION)
    if any(name in environment for name in unset):
        return dict(report, reason='DIAGNOSTIC_OR_INTERPOSITION_ACTIVE_IN_FROZEN_LAUNCH')
    groups, work = _work(source['phases'], count, rounds)
    if work['totals']['removed_guard_pairs'] == 0:
        return dict(report, reason='NO_REDUCIBLE_GUARD_PAIRS_IN_SCOPED_EXECUTION', conditional_work=work)
    return dict(report, status=ELIGIBLE, reason='FEWER_SCOPED_GUARD_CALLS_WITH_PROVED_NORMAL_EFFECTS',
        transformation=TRANSFORMATION, groups=groups, conditional_work=work,
        effect_contract=dict(scope='QUALIFIED_MASKED_STICKY_FLAGS_NORMAL_SUCCESS',
            rounding_mode=facts['rounding_mode'], monitored_exceptions=facts['monitored_exceptions'],
            preserve_arithmetic_provider_repeat_order=True, preserve_unguarded_steps=True,
            preserve_coefficient_hooks_normalization_and_checkpoints=True, preserve_original_publication=True,
            preserve_runtime_entry_guard=True, group_failure_requires_original_report_status=True,
            failure_status=facts['same_failure_code'], owner_publication_on_failure=False,
            exact_failure_prefix_equivalence_proven=False, whole_entry_fenv_equivalence_proven=False,
            clean_environment_requires_unset=list(unset), added_control_cost_ns=None))


def verify_cpu_fp_guard_plan(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU FP guard plan')
    replay = derive_cpu_fp_guard_plan(saved['proof'])
    _require(replay == saved, 'saved guard plan differs at cold replay')
    return replay
