"""Emit a catalog-selected CPU entry and retain its provider-scoped trial."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
from subprocess import SubprocessError

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import compile_cpu_entry, cpu_entry_replay
from scar.compiler.entry_delivery import HANDOFF, _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


_ACQUISITION_RUNS_ENTRY = frozenset(('scar.cpu-fixed-grid-source-provider', 'scar.triangle-profile-family'))
_STREAM = 'scar.cpu-recursive-stream-source-provider'
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
    if plan['provider_family'] == _STREAM:
        material = read_artifact(record['compilation'])
        contract = plan['contract']
        if (material['normalization'] != contract['normalization']
                or material['invocation']['stdin'] != contract['stream_contract']
                or material['invocation']['stdin']['protocol'] != contract['protocol']
                or read_artifact(material['runtime'])['parameterized_plan'] != contract['parameterized_plan']):
            raise ValueError('CPU compilation differs from the selected stream protocol or runtime plan')
    return record


def _replay_inputs(plan, scope):
    capture = scope.get('cpu_stream_capture')
    if plan['provider_family'] != _STREAM:
        if capture is not None:
            raise ValueError('stdin evidence cannot qualify an argv-only catalog entry')
        return None, None, None
    from scar.analysis.scalar_recursive_driver import CAPTURE

    contract = plan['contract']
    cpu = contract['cpu']
    if (capture != contract['capture'] or not isinstance(capture, dict) or capture.get('schema') != CAPTURE
            or type(cpu) is not int or plan['context']['cpu_ids'] != [cpu]
            or scope['process_affinity_cpu_ids'] != [cpu] or plan['argv'] != []):
        raise ValueError('selected stream plan differs from the original capture or CPU scope')
    observed = read_artifact(capture)
    if (observed['source'] != plan['entry'] or observed['stdin'] != contract['stdin']
            or observed['stdout'] != contract['stdout']):
        raise ValueError('selected stream capture differs from the exact stdin or stdout evidence')
    launch_reference = observed['captured_launch']
    launch = json.loads(read_artifact(launch_reference, json_document=False))
    if launch['command'][:3] != ['/usr/bin/taskset', '-c', str(cpu)]:
        raise ValueError('original stream capture was not acquired on the selected CPU')
    if (launch['stdin'] != contract['stdin']['uri'] or launch.get('protocol_plan') != contract['parameterized_plan']
            or launch.get('pacing') != contract['pacing']
            or launch.get('normalized_stdout_sha256') != contract['eligible_stdout_digest']):
        raise ValueError('original stream launch differs from the selected protocol, pacing or normalized stdout')
    return capture, cpu, launch_reference


def _provider_scope(spec, plan, scope):
    original = spec['arms']['original']
    context = plan['context']
    if (original['executable'] != context['interpreter'] or spec['cpu_ids'] != context['cpu_ids']
            or spec['cwd'] != context['cwd'] or original['environment'] != context['original_environment']):
        raise ValueError('CPU provider replay differs from its selected original process context')
    invoked = original['argv']
    stream = plan['provider_family'] == _STREAM
    flags = ['-B', '-u'] if stream else ['-B']
    prefix = [original['executable']['uri'], *flags, plan['entry']['uri']]
    if (spec['entry'] != plan['entry'] or bool(spec['transcript']) != stream or invoked[:len(prefix)] != prefix
            or len(invoked[len(prefix):]) != len(plan['argv'])):
        raise ValueError('CPU provider replay does not recover its admitted source invocation')
    slots = []
    for index, (request, replay) in enumerate(zip(plan['argv'], invoked[len(prefix):], strict=True)):
        if request == replay:
            continue
        leaf = replay.removeprefix(cpu_entry_replay._OUTPUT + '/')
        if (replay == leaf or leaf not in spec['files'] or Path(leaf).name != leaf
                or cpu_entry_replay._OUTPUT in request):
            raise ValueError('CPU provider replay substituted a non-publication argument')
        slots.append(dict(argv_index=index, requested_argument=request, replay_template=replay))
    if len(slots) > 1 or ('output_slot' in original and
            (len(slots) != 1 or slots[0]['argv_index'] + len(prefix) != original['output_slot'])):
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
    result = dict(requested_boundary=scope['boundary'],
        kind='PACED_SOURCE_COMMAND_STREAM' if stream else 'PROVIDER_SCOPED_PAIRED_PROCESSES',
        cpu_ids=spec['cpu_ids'], working_directory=spec['cwd'], original=original,
        candidate=spec['arms']['candidate'], scope_matches=matches, publication_slots=slots,
        normalized_runtime_fields=normalized_fields,
        exact_original_argv_observed=not slots, exact_whole_entry_effects_proven=False)
    if stream:
        capture, cpu, launch = _replay_inputs(plan, scope)
        contract = plan['contract']
        depths = [depth for effect, depth in spec['ordered_effects'] if effect == 'SEARCH']
        if (spec.get('scope') != 'PACED_SOURCE_COMMAND_STREAM' or spec['transcript'] != contract['transcript']
                or spec['eligible_stdout_digest'] != contract['eligible_stdout_digest']
                or depths != contract['depths'] or original['stdout'] != contract['stdout']):
            raise ValueError('CPU provider replay differs from selected command pacing or normalized stdout')
        result['stream'] = dict(capture=capture, captured_launch=launch, cpu=cpu, stdin=contract['stdin'],
            protocol=contract['protocol'],
            pacing=dict(method=contract['pacing'], transcript=spec['transcript'], ordered_effects=spec['ordered_effects'],
                        paired_live_timestamps_retained=False),
            stdout=dict(normalization=contract['normalization'], canonical_sha256=spec['eligible_stdout_digest']))
    return result


def _observation_summary(reference, report, record, plan, scope):
    if (reference['schema'] != cpu_entry_replay.SCHEMA or report['schema_version'] != 1
            or report['whole_program_equivalence_proven'] is not False or report['deployment_authorized'] is not False):
        raise ValueError('CPU catalog review expects a provider-paired observation')
    captured, cpu, _ = _replay_inputs(plan, scope)
    if report['capture'] != captured or report['cpu_override'] != cpu:
        raise ValueError('retained CPU trial differs from the selected stdin or CPU binding')
    context = _provider_scope(report['invocation'], plan, scope)
    read_artifact(reference, json_document=False)
    summary = dict(reference=reference, compilation=report['compilation'], provider_compilation=record['compilation'],
        scope=context, selection=report['selection'], same_boundary_whole_entry_timing=False,
        deployment_authorized=False)
    if 'resource_observation' in report:
        summary['resource_observation'] = report['resource_observation']
    if 'capacity_contract' in report:
        summary['capacity_contract'] = report['capacity_contract']
    return summary


def replay_cpu_catalog_observation(reference, plan, scope):
    """Cold-check a retained provider trial and its separate launch scope."""
    _plan_scope(plan, scope)
    return _replay_cpu_catalog_observation(reference, plan, scope)


def _replay_cpu_catalog_observation(reference, plan, scope, *, strict_binding=False, checked_pair=None):
    report = read_artifact(reference)
    record = _compilation(report['compilation'], plan)
    if strict_binding and record.get('catalog') != plan['catalog']:
        raise ValueError('retained CPU trial does not bind the selected catalog')
    if checked_pair is not None and type(checked_pair) is not cpu_entry_replay.CheckedCpuPair:
        raise ValueError('expected a checked CPU pair for this review')
    checked = cpu_entry_replay.check(reference['uri']) if checked_pair is None else checked_pair.report(reference)
    if checked != report:
        raise ValueError('CPU provider replay changed while checking the retained observation')
    review = _observation_summary(reference, report, record, plan, scope)
    if strict_binding and not all(review['scope']['scope_matches'].values()):
        raise ValueError('retained CPU trial differs from the selected original process context')
    return review


def _retained_capacity(review, requested_ref, requested):
    observed_ref = review.get('capacity_contract')
    if observed_ref is None:
        raise ValueError('retained CPU trial did not enforce a candidate address-space limit')
    # The paired checker replays the historical capacity contract and checks
    # candidate-only readbacks for every arm. A newly attached graph can move
    # the ledger while preserving every fact used to constrain the launch.
    observed = read_artifact(observed_ref)
    if dict(observed, ledger=None) != dict(requested, ledger=None):
        raise ValueError('retained CPU trial enforced a different capacity or launch contract')
    runner = (review.get('resource_observation') or {}).get('runner') or {}
    guard = runner.get('native_capacity_guard')
    source = Path(__file__).resolve().parents[1]/'native/cpu_capacity_exec.cpp'
    if (guard is None or guard['cxx_standard'] != 20
            or read_artifact(guard['source'], json_document=False) != source.read_bytes()):
        raise ValueError('retained CPU trial requires the current C++20 address-space guard')
    return dict(requested=requested_ref, observed=observed_ref, guard=guard, enforcement_reexecuted=False)


def _trial_result(result, review, *, retained):
    return dict(result, final_action='KEEP_ORIGINAL',
        reason='provider-scoped observations replayed while the original whole-entry path remains selected',
        provider_replay=dict(review, retained=retained),
        validation=dict(reference=review['reference'], timing=None, exact_declared_effects_replayed=False,
                        provider_boundary_replayed=True))


def emit_cpu_catalog_entry(option, scope, out, *, plan_only, verify_entry, capacity_contract=None,
                           retained_pair_ref=None, checked_pair=None):
    """Materialize the exact provider choice or replay its original sealed trial."""
    plan = option['proof']['emission_plan']
    result = dict(schema=HANDOFF, schema_version=1, selected_id=option['id'], scope=scope,
        selected_emission=plan, final_action='NATIVE_COMPILATION_DEFERRED',
        reason='plan-only preserved the selected CPU catalog plan', native_compiled=False,
        target_executed=False, validation_attempted=False, qualification=None, validation=None, launch_authority=None,
        deployment_authorized=False)
    capacity = None
    try:
        _plan_scope(plan, scope)
        capture, cpu, _ = _replay_inputs(plan, scope)
        if capacity_contract is not None:
            from scar.optimization.cpu_entry_capacity_contract import verify_cpu_entry_capacity_contract

            capacity = verify_cpu_entry_capacity_contract(capacity_contract)
            if (capacity['option_id'] != option['id'] or capacity['scope_sha256'] != semantic_sha256(scope)
                    or plan_only or not verify_entry):
                raise ValueError('CPU capacity contract differs from its native trial nomination')
            result['capacity_contract'] = capacity_contract
            if capacity['status'] != 'CANDIDATE_GUARD_ADMISSIBLE':
                return dict(result, final_action='KEEP_ORIGINAL', reason='candidate address-space contract is unresolved')
    except _ERRORS as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason='CPU catalog plan scope failed: ' + str(error))
    if plan_only:
        return result

    pinned_pair = scope.get('cpu_paired_observation')
    pair = retained_pair_ref if retained_pair_ref is not None else pinned_pair
    if verify_entry and pair is not None:
        try:
            explicit = retained_pair_ref is not None
            # A historical unguarded pair can pin the original scope while
            # requesting a fresh guarded trial. A pair supplied for this
            # particular guarded review must qualify.
            if explicit or capacity is None or read_artifact(pair).get('capacity_contract') is not None:
                if pinned_pair not in (None, pair):
                    raise ValueError('retained CPU trial conflicts with the pair pinned in the original scope')
                if explicit and option['id'] != 'planned:' + semantic_sha256(
                        dict(scope_sha256=semantic_sha256(scope), plan=plan)):
                    raise ValueError('retained CPU trial does not match the scoped catalog option ID')
                replay = _replay_cpu_catalog_observation(pair, plan, scope, strict_binding=explicit,
                                                        checked_pair=checked_pair)
                if capacity is not None:
                    replay['capacity_replay'] = _retained_capacity(replay, capacity_contract, capacity)
                result['native'] = dict(compilation=replay['compilation'], artifact=replay['provider_compilation'],
                                        executable=replay['scope']['candidate']['executable'], reused=True)
                result['qualification'] = dict(status='QUALIFIED_PROVIDER_BOUNDARY', scope=replay['scope'])
                return _trial_result(result, replay, retained=True)
        except _ERRORS as error:
            return dict(result, final_action='KEEP_ORIGINAL', reason='retained CPU catalog trial failed: ' + str(error))

    acquisition = plan['provider_family'] in _ACQUISITION_RUNS_ENTRY
    if acquisition and not verify_entry:
        return dict(result, final_action='CANDIDATE_VALIDATION_REQUIRED',
                    reason='this CPU provider requires an explicit entry trial for its fresh acquisition')
    if verify_entry and not set(plan['context']['cpu_ids']) <= os.sched_getaffinity(0):
        return dict(result, final_action='KEEP_ORIGINAL', reason='qualified CPU mask is unavailable')
    out = Path(out).resolve()
    replay_scope = replay_transaction() if capture is not None else nullcontext()
    phase = 'emission'
    try:
        with replay_scope:
            compilation = compile_cpu_entry.compile_cpu_entry((), plan['entry']['uri'], plan['argv'], out/'native-entry',
                                                              catalog=plan['catalog']['uri'])
            record = _compilation(compilation, plan)
            material = read_artifact(record['compilation'])
            reused = material.get('runtime_reused') is True
            result.update(native_compiled=not reused, target_executed=acquisition, native=dict(compilation=compilation,
                artifact=record['compilation'], reused=reused))
            phase = 'admission'
            spec = cpu_entry_replay._spec(compilation, capture, cpu)
            context = _provider_scope(spec, plan, scope)
        result['native']['executable'] = context['candidate']['executable']
        result['qualification'] = dict(status='QUALIFIED_PROVIDER_BOUNDARY', scope=context)
    except _ERRORS as error:
        if acquisition and 'native' not in result:
            result['target_executed'] = None
        return dict(result, final_action='KEEP_ORIGINAL', reason=f'CPU catalog {phase} failed: {error}')

    if not verify_entry:
        return dict(result, final_action='CANDIDATE_VALIDATION_REQUIRED',
                    reason='CPU catalog receipt replayed; provider trial required')

    validation = out/'native-validation'
    try:
        observed = cpu_entry_replay._measure_checked(compilation, capture, cpu, spec, validation,
                                                    capacity_contract=capacity_contract)
        reference = artifact_reference(validation/'result.json', cpu_entry_replay.SCHEMA)
        if read_artifact(reference) != observed or observed['compilation'] != compilation:
            raise ValueError('fresh provider observation differs from its admitted compilation or saved receipt')
        replay = _observation_summary(reference, observed, record, plan, scope)
    except _ERRORS as error:
        return dict(result, final_action='KEEP_ORIGINAL', validation_attempted=True,
                    target_executed=True if acquisition else None, reason='CPU catalog trial failed: ' + str(error))
    return _trial_result(dict(result, target_executed=True, validation_attempted=True), replay, retained=False)
