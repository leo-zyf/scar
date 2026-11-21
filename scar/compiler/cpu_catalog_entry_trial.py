"""Emit a catalog-selected CPU entry and retain its provider-scoped trial."""

from pathlib import Path
from subprocess import SubprocessError

from scar.compiler import compile_cpu_entry, cpu_entry_replay
from scar.compiler.entry_delivery import HANDOFF, _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact


_ACQUISITION_RUNS_ENTRY = frozenset(('scar.cpu-fixed-grid-source-provider', 'scar.triangle-profile-family'))
_ERRORS = (OSError, ValueError, SubprocessError)


def _plan_scope(plan, scope):
    source = artifact_reference(scope['entry'], 'scar.python-source')
    if (source != plan['entry'] or source['sha256'] != scope['entry_sha256']
            or plan['argv'] != scope['argv'] or plan['target'] != scope['target']
            or plan['catalog'] != scope.get('cpu_provider_catalog', plan['catalog'])
            or plan['source_f4_program_sha256'] != scope['program_sha256']
            or plan['effect_obligations_sha256'] != scope['entry_obligations_sha256']):
        raise ValueError('catalog plan differs from the original source, argv, F4 proof or target')
    authority, provider = compile_cpu_entry._resolve((), source, plan['catalog'])
    if (authority != plan['authority'] or authority['schema'] != plan['provider_family']
            or provider.result_schema != plan['provider_result_schema']):
        raise ValueError('catalog provider differs from the selected authority')


def _compilation(reference, plan):
    record = read_artifact(reference)
    if (reference['schema'] != compile_cpu_entry.SCHEMA or record['schema_version'] != 1
            or record['provider'] != plan['authority'] or record['entry'] != plan['entry']
            or record['argv'] != plan['argv'] or record.get('catalog') not in (None, plan['catalog'])
            or record['compilation']['schema'] != plan['provider_result_schema']):
        raise ValueError('CPU compilation differs from the selected catalog source, argv or provider')
    profile = plan['contract'].get('profile')
    if profile is not None and read_artifact(record['compilation']) != profile:
        raise ValueError('CPU receipt differs from the selected retained provider profile')
    return record


def _provider_scope(spec, plan, scope):
    original = spec['arms']['original']
    context = plan['context']
    if (original['executable'] != context['interpreter'] or spec['cpu_ids'] != context['cpu_ids']
            or spec['cwd'] != context['cwd'] or original['environment'] != context['original_environment']):
        raise ValueError('CPU provider replay differs from its selected original process context')
    invoked = original['argv']
    if (spec['entry'] != plan['entry'] or spec['transcript']
            or invoked[:3] != [original['executable']['uri'], '-B', plan['entry']['uri']]
            or len(invoked[3:]) != len(plan['argv'])):
        raise ValueError('CPU provider replay does not recover its admitted argv entry')
    slots = []
    for index, (request, replay) in enumerate(zip(plan['argv'], invoked[3:], strict=True)):
        if request == replay:
            continue
        leaf = replay.removeprefix(cpu_entry_replay._OUTPUT + '/')
        if (replay == leaf or leaf not in spec['files'] or Path(leaf).name != leaf
                or cpu_entry_replay._OUTPUT in request):
            raise ValueError('CPU provider replay substituted a non-publication argument')
        slots.append(dict(argv_index=index, requested_argument=request, replay_template=replay))
    if len(slots) > 1 or ('output_slot' in original and
            (len(slots) != 1 or slots[0]['argv_index'] + 3 != original['output_slot'])):
        raise ValueError('CPU provider replay has a different publication slot contract')
    publication = plan['contract'].get('publication')
    normalized_fields = spec.get('json_report', {}).get('runtime_fields', [])
    expected_slots = [] if publication is None else [publication['output_index']]
    expected_fields = [] if publication is None else publication.get('normalized_fields', [])
    if [row['argv_index'] for row in slots] != expected_slots or normalized_fields != expected_fields:
        raise ValueError('CPU provider replay normalization differs from the selected contract')
    matches = dict(cwd=spec['cwd'] == scope['working_directory'],
        resolved_interpreter=str(Path(original['executable']['uri']).resolve()) == scope['python_executable'],
        affinity=spec['cpu_ids'] == scope['process_affinity_cpu_ids'],
        environment=_environment_digest(original['environment']) == scope['environment_sha256'])
    if matches != context['scope_matches']:
        raise ValueError('CPU provider replay no longer has the selected scope relation')
    return dict(requested_boundary=scope['boundary'], kind='PROVIDER_SCOPED_PAIRED_PROCESSES',
        cpu_ids=spec['cpu_ids'], working_directory=spec['cwd'], original=original,
        candidate=spec['arms']['candidate'], scope_matches=matches, publication_slots=slots,
        normalized_runtime_fields=normalized_fields,
        exact_original_argv_observed=not slots, exact_whole_entry_effects_proven=False)


def replay_cpu_catalog_observation(reference, plan, scope):
    """Cold-check a retained provider trial and its separate launch scope."""
    _plan_scope(plan, scope)
    report = read_artifact(reference)
    if reference['schema'] != cpu_entry_replay.SCHEMA:
        raise ValueError('CPU catalog review expects a provider-paired observation')
    record = _compilation(report['compilation'], plan)
    if cpu_entry_replay.check(reference['uri']) != report:
        raise ValueError('CPU provider replay changed while checking the retained observation')
    context = _provider_scope(report['invocation'], plan, scope)
    read_artifact(reference, json_document=False)
    return dict(reference=reference, compilation=report['compilation'], provider_compilation=record['compilation'],
        scope=context, selection=report['selection'], same_boundary_whole_entry_timing=False,
        deployment_authorized=False)


def _trial_result(result, review, *, retained):
    return dict(result, final_action='KEEP_ORIGINAL',
        reason='provider-scoped observations replayed while the original whole-entry path remains selected',
        provider_replay=dict(review, retained=retained),
        validation=dict(reference=review['reference'], timing=None, exact_declared_effects_replayed=False,
                        provider_boundary_replayed=True))


def emit_cpu_catalog_entry(option, scope, out, *, plan_only, verify_entry):
    """Materialize the exact provider choice or replay its original sealed trial."""
    plan = option['proof']['emission_plan']
    result = dict(schema=HANDOFF, schema_version=1, selected_id=option['id'], scope=scope,
        selected_emission=plan, final_action='NATIVE_COMPILATION_DEFERRED',
        reason='plan-only preserved the selected CPU catalog plan', native_compiled=False,
        target_executed=False, validation_attempted=False, qualification=None, validation=None, launch_authority=None,
        deployment_authorized=False)
    try:
        _plan_scope(plan, scope)
    except _ERRORS as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason='CPU catalog plan scope failed: ' + str(error))
    if plan_only:
        return result

    if verify_entry and scope.get('cpu_paired_observation') is not None:
        try:
            replay = replay_cpu_catalog_observation(scope['cpu_paired_observation'], plan, scope)
        except _ERRORS as error:
            return dict(result, final_action='KEEP_ORIGINAL', reason='retained CPU catalog trial failed: ' + str(error))
        result['native'] = dict(compilation=replay['compilation'], artifact=replay['provider_compilation'],
                                executable=replay['scope']['candidate']['executable'], reused=True)
        result['qualification'] = dict(status='QUALIFIED_PROVIDER_BOUNDARY', scope=replay['scope'])
        return _trial_result(result, replay, retained=True)

    acquisition = plan['provider_family'] in _ACQUISITION_RUNS_ENTRY
    if acquisition and not verify_entry:
        return dict(result, final_action='CANDIDATE_VALIDATION_REQUIRED',
                    reason='this CPU provider requires an explicit entry trial for its fresh acquisition')
    out = Path(out).resolve()
    try:
        compilation = compile_cpu_entry.compile_cpu_entry((), plan['entry']['uri'], plan['argv'], out/'native-entry',
                                                          catalog=plan['catalog']['uri'])
        record = _compilation(compilation, plan)
    except _ERRORS as error:
        return dict(result, final_action='KEEP_ORIGINAL', target_executed=None if acquisition else False,
                    reason='CPU catalog emission failed: ' + str(error))
    material = read_artifact(record['compilation'])
    reused = material.get('runtime_reused') is True
    result.update(native_compiled=not reused, target_executed=acquisition, native=dict(compilation=compilation,
        artifact=record['compilation'], reused=reused))

    def admit(spec):
        context = _provider_scope(spec, plan, scope)
        result['native']['executable'] = context['candidate']['executable']
        result['qualification'] = dict(status='QUALIFIED_PROVIDER_BOUNDARY', scope=context)

    if not verify_entry:
        try:
            admit(cpu_entry_replay._spec(compilation, None, None))
        except _ERRORS as error:
            return dict(result, final_action='KEEP_ORIGINAL', reason='CPU catalog admission failed: ' + str(error))
        return dict(result, final_action='CANDIDATE_VALIDATION_REQUIRED',
                    reason='CPU catalog receipt replayed; provider trial required')

    validation = out/'native-validation'
    try:
        cpu_entry_replay.measure(compilation['uri'], validation, _admit_invocation=admit)
        reference = artifact_reference(validation/'result.json', cpu_entry_replay.SCHEMA)
        replay = replay_cpu_catalog_observation(reference, plan, scope)
    except _ERRORS as error:
        return dict(result, final_action='KEEP_ORIGINAL', validation_attempted=True,
                    target_executed=True if acquisition else None, reason='CPU catalog trial failed: ' + str(error))
    return _trial_result(dict(result, target_executed=True, validation_attempted=True), replay, retained=False)
