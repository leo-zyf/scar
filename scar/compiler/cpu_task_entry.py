"""Cpu task entry."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
from subprocess import SubprocessError, CalledProcessError
from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import compile_cpu_entry, cpu_entry_replay
from scar.compiler.entry_delivery import HANDOFF, _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256
from dataclasses import asdict
from hashlib import sha256
from scar.compiler.compile_cpu_entry import (
    CATALOG_SCHEMA,
    _PROVIDERS,
    _match,
    _read_catalog,
    _reference,
)
from scar.compiler.plan_aten_boundary import _target


# Emit a catalog-selected CPU entry and retain its provider-scoped trial.
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


# Match sealed CPU provider evidence to an original F4 invocation before build.
SCHEMA = 'scar.cpu-catalog-entry-variants'
FAMILY = 'CATALOG_CPU_ENTRY'
_ARRAY = 'scar.cpu-array-task-source-provider'
_GRID = 'scar.cpu-fixed-grid-source-provider'
_SPARSE = 'scar.triangle-profile-family'
STREAM_CAPTURE_SCHEMA = 'scar.scalar-recursive-driver-capture'


class _Frontier(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _require(condition, reason, detail):
    if not condition:
        raise _Frontier(reason, detail)


def _launch(scope, *, interpreter, cpu_ids, cwd, original_environment):
    _require(bool(cpu_ids) and set(cpu_ids) <= set(scope['process_affinity_cpu_ids']), 'CPU_COHORT_OUTSIDE_F4_SCOPE',
             'provider CPU lanes are outside the original process affinity')
    try:
        target, lanes = _target(Path(scope['target']['uri']).parent, cpu_ids, allow_cpu_subset=True)
    except ValueError as error:
        raise _Frontier('CPU_HARDWARE_COHORT_OPEN', str(error)) from error
    _require(asdict(target) == scope['target'], 'CPU_HARDWARE_SCOPE_DIFFERS',
             'original F4 hardware graph differs from provider admission')
    environment_sha256 = sha256(json.dumps(sorted(original_environment.items()), separators=(',', ':')).encode()).hexdigest()
    matches = dict(cwd=scope['working_directory'] == cwd, affinity=scope['process_affinity_cpu_ids'] == cpu_ids,
        resolved_interpreter=Path(interpreter['uri']).resolve(strict=True) == Path(scope['python_executable']),
        environment=scope['environment_sha256'] == environment_sha256)
    return dict(interpreter=interpreter, cpu_ids=cpu_ids, cwd=cwd, original_environment=original_environment,
        scope_matches=matches, hardware_lanes={str(cpu): [row.id for row in lanes[cpu]] for cpu in cpu_ids})


def _array_memory(scope, decision):
    memory = decision['memory']
    scope_kind = memory['bound_scope']
    owner = memory['explicit_owner_payload_bytes']
    worker = memory['explicit_worker_payload_bytes']
    owner_cap = memory['coordinator_address_space_limit_bytes']
    worker_cap = memory['worker_address_space_limit_bytes']
    workers = decision['worker_count']
    if scope_kind == 'SINGLE_PROCESS_VAS_CAP_INCLUDING_PRIVATE_ARRAYS_AND_OPENMP_CHILD_STACKS':
        processes = 0
    elif scope_kind == 'AGGREGATE_PROCESS_VAS_CAPS_PLUS_EXPLICIT_PAYLOAD_SUBACCOUNT':
        processes = workers
    else:
        raise _Frontier('ARRAY_TASK_ADDRESS_SPACE_CONTRACT_OPEN', 'unsupported native array address-space accounting')
    budget = scope['memory_budget_bytes']
    required = owner_cap + processes * worker_cap
    payload = owner + processes * worker
    _require(all(type(value) is int and value >= 0 for value in (owner, worker, processes))
             and all(type(value) is int and value > 0 for value in (owner_cap, worker_cap, budget))
             and owner < owner_cap and (not processes or worker < worker_cap)
             and required == memory['required_headroom_bytes'] and memory['physical_reservation'] is False,
             'ARRAY_TASK_ADDRESS_SPACE_CONTRACT_DIFFERS', 'retained explicit payloads and process address-space caps differ')
    _require(payload <= budget and required <= budget, 'ARRAY_TASK_ADDRESS_SPACE_BUDGET_EXCEEDED',
             'retained array-task virtual address-space requirement exceeds the requested execution budget')
    return dict(scope=scope_kind, declared_budget_bytes=budget, required_address_space_headroom_bytes=required,
        explicit_payload_bytes=payload, capacity_kind='VIRTUAL_ADDRESS_SPACE_CAPS', physical_dram_reserved=False,
        memory=memory)


def _array(scope, authority, entry, argv):
    from scar.compiler.array_task_source_provider import _read, _result

    _, profiles = _read(authority)
    _require(sum(row['argv'] == argv for row in profiles) == 1, 'ARRAY_TASK_ARGV_NOT_QUALIFIED',
             'array provider requires an exact retained original argv')
    profile = _result(authority, entry, argv)
    invocation = profile['invocation']
    context = _launch(scope, interpreter=invocation['original_executable'], cpu_ids=invocation['cpu_ids'],
                      cwd=invocation['cwd'], original_environment=invocation['original_environment'])
    retained = read_artifact(profile['retention'])
    hardware = read_artifact(retained['schedule'])['hardware']
    historical = {row['cpu_id']: row for row in read_artifact(hardware)['processors']}
    for cpu in invocation['cpu_ids']:
        row = historical.get(cpu)
        _require(row is not None and row.get('permitted') is True and
                 'target_resource:cpu-core/'+'/'.join(row[key] for key in ('package_id', 'die_id', 'core_id'))
                 in context['hardware_lanes'][str(cpu)], 'ARRAY_HOST_TOPOLOGY_DIFFERS',
                 'retained array-task placement differs from the current F4 hardware lanes')
    runtime = read_artifact(profile['runtime'])
    holdout = read_artifact(profile['holdout'])
    return context, dict(profile=profile, parameters=holdout['parameters'], legacy_hardware=hardware,
        native_memory=_array_memory(scope, runtime['decision']), publication=dict(output_index=invocation['output_index'],
            normalized_fields=profile['normalization'])), [dict(reason='RETAINED_PROCESS_SAMPLES_ARE_OBSERVATIONS',
            artifact=profile['retention'], locator='#/retained_vs_original')]


def _fixed_grid(scope, authority, entry, argv):
    from scar.compiler.fixed_grid_source_provider import _argv, _ENV, check

    provider = check(authority['uri'])
    try:
        _argv(provider, argv, require_fresh=True)
    except (OSError, ValueError) as error:
        raise _Frontier('FIXED_GRID_ARGV_NOT_QUALIFIED', str(error)) from error
    selected = provider['source_project']
    manifest = read_artifact(selected)
    root = Path(selected['uri']).parent

    def member(name):
        row = manifest['files'][name]
        return json.loads(read_artifact(dict(uri=str(root / name), schema='application/json', **row), json_document=False))

    execution, rhs = member('plan/execution.json'), member('plan/rhs-recipe.json')
    _require(execution['target'] == scope['target'], 'FIXED_GRID_TARGET_DIFFERS',
             'fixed-grid provider retained a different hardware target')
    shapes = [row['shape'] for row in rhs['inputs']]
    common = scope.get('input_shapes', {}).get('common_shape')
    _require(common is None or all(shape == common for shape in shapes), 'FIXED_GRID_SHAPE_DIFFERS',
             'fixed-grid matrix inputs differ from the declared F4 scope')
    _require(execution['peak_tensor_storage_bytes'] <= scope['memory_budget_bytes'], 'FIXED_GRID_STORAGE_BUDGET_EXCEEDED',
             'fixed-grid retained graph payload exceeds the original requested budget')
    invocation = provider['context']
    _require((Path(invocation['cwd']) / argv[2]).parent.is_dir(), 'FIXED_GRID_OUTPUT_PARENT_REQUIRED',
             'fixed-grid output requires an existing parent directory')
    environment = dict(_ENV, PYTHONPATH=':'.join(invocation['pythonpath']))
    context = _launch(scope, interpreter=invocation['interpreter'], cpu_ids=manifest['cpus'],
                      cwd=invocation['cwd'], original_environment=environment)
    step_ns = manifest['alternatives'][manifest['rhs_mode']]['body_estimate_ns']
    return context, dict(project=selected, inputs=provider['inputs'], input_shapes=shapes, solver=provider['source_binding'],
        steps=int(argv[1]), storage=execution['storage'], body_placement=execution['placement'],
        publication=dict(output_index=2, output_contract='EXACT_FINAL_F32_FILE_DISTINCT_STDOUT_CONTRACTS')), [
            dict(reason='RETAINED_FIXED_GRID_BODY_ONLY', body_step_estimate_ns=step_ns,
                 modeled_body_ns=step_ns*int(argv[1]), artifact=selected)]


def _sparse(scope, authority, entry, argv):
    from scar.compiler.compile_triangle_profile import _launch_environment
    from scar.compiler.triangle_profile_family import check

    _require(len(argv) == 1 and argv[0].isascii() and argv[0].isdecimal() and 0 < len(argv[0]) <= 9
             and int(argv[0]) > 1 and argv[0] == str(int(argv[0])), 'SPARSE_PROFILE_ARGV_OUTSIDE_CONTRACT',
             'iterative sparse provider needs one canonical iteration count greater than one')
    family = check(authority['uri'])
    launch = family['context']
    context = _launch(scope, interpreter=family['interpreter_executable'], cpu_ids=family['launch_cpu_ids'],
                      cwd=launch['cwd'], original_environment=_launch_environment(launch))
    historical = read_artifact(family['release'])
    return context, dict(anchor=family['anchor'], source_model=family['model'], count=int(argv[0]),
        historical_count=historical['count'], historical_placement=family['release'],
        historical_problem=historical['problem'], historical_memory_floor_bytes=historical['memory_floor_bytes'],
        historical_costs_transferred=False, fresh_source_acquisition_required=True), [
            dict(reason='FRESH_SPARSE_PROFILE_AND_SCHEDULING_REQUIRED')]


def _stream_capture(scope, entry, argv):
    reference = scope.get('cpu_stream_capture')
    _require(type(reference) is dict and reference.get('schema') == STREAM_CAPTURE_SCHEMA,
             'LIVE_STDIN_PROTOCOL_REQUIRES_INVOCATION_SCOPE',
             'the live command provider requires a pinned bound original stream capture')
    capture = read_artifact(reference)
    launch = capture['launch']
    _require(capture['schema_version'] == 1 and capture['source'] == entry and launch['argv'] == argv
             and launch['entry_path'] == entry['uri'] and launch['exit_code'] == 0
             and launch['installed_bindings'] == 0 and not capture['frontiers']
             and capture['original_execution'] is True and capture['performance_measurement'] is False,
             'STREAM_CAPTURE_SCOPE_DIFFERS', 'captured source, argv or execution effects differ from the original entry')
    raw = read_artifact(capture['raw_capture'])
    _require(all(capture.get(key) == value for key, value in raw.items()), 'STREAM_CAPTURE_SCOPE_DIFFERS',
             'bound stream record differs from the raw original observation')
    context = json.loads(read_artifact(capture['captured_launch'], json_document=False))
    command = context['command']
    _require(type(command) is list and all(type(arg) is str and '\x00' not in arg for arg in command)
             and len(command) >= 3 and command[:2] == ['/usr/bin/taskset', '-c']
             and command[2].isascii() and command[2].isdecimal() and len(command[2]) <= 10
             and command[2] == str(int(command[2])), 'STREAM_CAPTURE_CPU_UNBOUND',
             'original capture launch did not pin one canonical taskset CPU')
    cpu = int(command[2])
    _require(scope['process_affinity_cpu_ids'] == [cpu] == context.get('cpu_ids', [cpu]), 'STREAM_CAPTURE_CPU_SCOPE_DIFFERS',
             'captured stream CPU must equal the exact original invocation cohort')
    expected = {'--design': capture['design']['uri'], '--out': capture['raw_capture']['uri']}
    _require(all(command.count(arg) == 1 and command.index(arg) + 1 < len(command)
                 and command[command.index(arg) + 1] == value for arg, value in expected.items())
             and context['stdin'] == capture['stdin']['uri'], 'STREAM_CAPTURE_SCOPE_DIFFERS',
             'stream capture launch does not bind the original observation and input files')
    return reference, capture, cpu, context


def _stream(scope, authority, entry, argv):
    from scar.compiler.cpu_entry_replay import _recursive
    from scar.compiler.recursive_stream_source_provider import _checked_provider, _invocation

    reference, capture, cpu, captured_launch = _stream_capture(scope, entry, argv)
    provider, seed = _checked_provider(authority)
    _require(argv == provider['argv'], 'STREAM_PROVIDER_ARGV_DIFFERS',
             'original argv differs from the sealed live-command provider')
    runtime = provider['source_artifact']
    invocation = _invocation(runtime, seed, provider, seed, entry, argv)
    try:
        spec = _recursive(dict(entry=entry, argv=argv), dict(runtime=runtime, invocation=invocation), reference, cpu)
    except ValueError as error:
        raise _Frontier('STREAM_CAPTURE_PACING_OR_PUBLICATION_OPEN', str(error)) from error
    parameterized = provider['parameterized_plan']
    _require(captured_launch.get('protocol_plan') == parameterized
             and captured_launch.get('pacing') == 'WAIT_FOR_MATCHED_STDOUT_PREFIX'
             and captured_launch.get('normalized_stdout_sha256') == spec['eligible_stdout_digest'],
             'STREAM_CAPTURE_PACING_OR_PUBLICATION_OPEN', 'the original capture did not preserve the verified pacing and stdout proof')
    original = spec['arms']['original']
    context = _launch(scope, interpreter=original['executable'], cpu_ids=[cpu], cwd=spec['cwd'],
                      original_environment=original['environment'])
    resources = read_artifact(parameterized)['resources']
    peak = resources['peak_owner_memory_bytes']
    budget = scope['memory_budget_bytes']
    _require(type(budget) is int and budget > 0 and (peak is None or type(peak) is int and 0 < peak <= budget),
             'STREAM_STORAGE_BUDGET_EXCEEDED', 'retained owner memory bound exceeds the requested execution budget')
    memory = dict(scope='PROVIDER_OWNER_MEMORY', peak_bytes=peak, declared_budget_bytes=budget,
                  status='UNKNOWN' if peak is None else 'RETAINED_PROVIDER_BOUND')
    return context, dict(capture=reference, cpu=cpu, parameterized_plan=parameterized, stdin=capture['stdin'],
        stdout=capture['stdout'], normalization=provider['normalization'], protocol=provider['stdin']['protocol'],
        stream_contract=provider['stdin'], pacing=captured_launch['pacing'],
        depths=[value for kind, value in spec['ordered_effects'] if kind == 'SEARCH'], transcript=spec['transcript'],
        eligible_stdout_digest=spec['eligible_stdout_digest'], memory=memory), [
            dict(reason='PACED_SOURCE_CAPTURE_HAS_NO_CLEAN_PROCESS_TIME')]


_PROFILE = {_ARRAY: _array, _GRID: _fixed_grid, _SPARSE: _sparse, _STREAM: _stream}


def derive_cpu_catalog_entry(scope, catalog_ref):
    """Derive a build candidate or retain the exact authority and the open frontier."""
    result = dict(schema=SCHEMA, schema_version=1, family=FAMILY, scope_sha256=semantic_sha256(scope),
        options=[], coverage_frontier=[], compiled_source_entry=False, target_executed=False)
    try:
        _require(all(scope.get(key) for key in ('program_sha256', 'source_admission_sha256', 'entry_obligations_sha256')),
                 'ORIGINAL_F4_ENTRY_OBLIGATIONS_REQUIRED', 'source, admission and effect obligations must be pinned')
        _require(scope.get('boundary') == 'PROCESS_LAUNCH_THROUGH_EXIT' and scope.get('native_output_contract') == 'STANDALONE_CPP',
                 'CPU_CATALOG_ENTRY_SCOPE_DIFFERS', 'standalone whole-process entry scope required')
        entry = _reference(scope['entry'], 'scar.python-source')
        _require((entry['uri'], entry['sha256']) == (scope['entry'], scope['entry_sha256']),
                 'ORIGINAL_F4_SOURCE_BYTES_DIFFER', 'original F4 source path or bytes changed')
        authorities = _read_catalog(catalog_ref)
        count = sum(document['entry'] == entry for _, document in authorities)
        _require(count == 1, 'NO_UNIQUE_CPU_CATALOG_AUTHORITY', 'the source path and bytes need one matching authority')
        authority, _ = _match(authorities, entry)
        result['matched_authority'] = authority
        kind = authority['schema']
        _require(kind == _STREAM or 'cpu_stream_capture' not in scope, 'STREAM_CAPTURE_PROVIDER_DIFFERS',
                 'a captured stdin stream cannot qualify an argv-only CPU provider')
        argv = scope['argv']
        context, contract, costs = _PROFILE[kind](scope, authority, entry, argv)
        timing_frontier = [*costs, dict(reason='SAME_ENTRY_FORWARD_LIFECYCLE_COSTS_OPEN')]
        if not all(context['scope_matches'].values()):
            timing_frontier.append(dict(reason='PROVIDER_LAUNCH_CONTEXT_DIFFERS', matches=context['scope_matches']))
        plan = dict(catalog=catalog_ref, authority=authority, entry=entry, argv=argv, provider_family=kind,
            provider_result_schema=_PROVIDERS[kind].result_schema, context=context, contract=contract,
            target=scope['target'], timing_frontier=timing_frontier, full_original_effects_closed=False,
            source_f4_program_sha256=scope['program_sha256'], effect_obligations_sha256=scope['entry_obligations_sha256'])
        option_id = 'planned:'+semantic_sha256(dict(scope_sha256=result['scope_sha256'], plan=plan))
        result['options'].append(dict(id=option_id, implementation_family=FAMILY, legal_status='UNRESOLVED',
            reason='CATALOG_TRIAL_REQUIRES_ORIGINAL_EFFECTS_AND_CLEAN_PROCESS_VALIDATION', proof=dict(emission_plan=plan)))
    except _Frontier as error:
        result['coverage_frontier'].append(dict(reason=error.reason, detail=str(error)))
    except (ValueError, OSError, KeyError, TypeError, CalledProcessError) as error:
        result['coverage_frontier'].append(dict(reason='CPU_CATALOG_PROVIDER_REPLAY_OPEN', detail=str(error)))
    return result
