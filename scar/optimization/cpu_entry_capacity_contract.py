"""Derive a candidate-only CPU address-space guard from a replayed planner ledger."""

from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import read_artifact
from .cpu_entry_decision_ledger import verify_cpu_entry_decision_ledger
from .deployment_cost import _pointer


SCHEMA = 'scar.cpu-entry-capacity-contract'
ADMISSIBLE = 'CANDIDATE_GUARD_ADMISSIBLE'
_FAMILY = 'CATALOG_CPU_ENTRY'
_ARRAY = 'scar.cpu-array-task-source-provider'
_RUNTIME = 'scar.native-seeded-array-task-entry'
_TEAM_PLAN = 'scar.independent-array-task-team-schedule'
_THREAD_PROOF = 'scar.native-array-thread-safety'
_SINGLE = 'SINGLE_PROCESS_VAS_CAP_INCLUDING_PRIVATE_ARRAYS_AND_OPENMP_CHILD_STACKS'
_AGGREGATE = 'AGGREGATE_PROCESS_VAS_CAPS_PLUS_EXPLICIT_PAYLOAD_SUBACCOUNT'


def _require(condition, message):
    if not condition:
        raise ValueError('CPU entry capacity contract: ' + message)


def _array_runtime(plan, witness):
    authority = read_artifact(plan['authority'])
    compiled = plan['contract']['profile']
    native_ref = compiled['runtime']
    retained = read_artifact(compiled['retention'])
    native = read_artifact(native_ref)
    decision = native['decision']
    memory = decision['memory']
    profiles = [row for row in authority['profiles'] if row['argv'] == plan['argv']]
    _require(len(profiles) == 1 and authority['entry'] == compiled['entry'] == plan['entry']
             and all(compiled[key] == value for key, value in profiles[0].items())
             and compiled['provider'] == plan['authority'] and native_ref['schema'] == _RUNTIME
             and retained['action'] == 'RETAIN_VERIFIED_LOCAL_BINARY' and retained['retained_release'] == native_ref
             and retained['selected'] == decision and native['schema_version'] == 1
             and native['deployment_authorized'] is False and native['python_runtime_required'] is False
             and native['cpu_ids'] == decision['launch_cpu_ids'] == plan['context']['cpu_ids']
             and witness['evidence'] == dict(artifact=native_ref, locator='#/decision/memory/required_headroom_bytes')
             and witness['bytes'] == memory['required_headroom_bytes'] and memory['physical_reservation'] is False,
             'provider runtime does not reproduce the exact candidate capacity witness')
    return compiled, native, decision


def _accounting(mode, memory, workers, aggregate_bytes):
    owner = memory['coordinator_address_space_limit_bytes']
    worker = memory['worker_address_space_limit_bytes']
    _require(type(owner) is int and 0 < owner < 2**63 and type(worker) is int and 0 < worker < 2**63
             and type(workers) is int and workers >= 1, 'positive exact per-role limits and worker count required')
    roles = [dict(role='COORDINATOR', count=1, per_process_limit_bytes=owner)]
    if memory['bound_scope'] == _SINGLE:
        _require(mode == 'ORDERED_OPENMP' and workers > 1 and aggregate_bytes == owner,
                 'OpenMP threads must share exactly one address space')
        return dict(kind='SINGLE_PROCESS_INCLUDING_THREADS', roles=roles, configured_threads=workers,
                    process_count=1, aggregate_limit_bytes=owner)
    _require(memory['bound_scope'] == _AGGREGATE and worker <= owner, 'unknown or non-inheritable address-space caps')
    if mode == 'ORDERED_LOCAL_PROCESSES':
        _require(workers > 1, 'a process team requires separate worker processes')
        roles.append(dict(role='WORKER', count=workers, per_process_limit_bytes=worker))
    else:
        _require(mode == 'IN_PROCESS_SERIAL' and workers == 1, 'unsupported aggregate process layout')
    _require(aggregate_bytes == sum(row['count'] * row['per_process_limit_bytes'] for row in roles),
             'aggregate cap must sum the address-space limits of every process role')
    return dict(kind='AGGREGATE_PER_PROCESS_CAP_ACCOUNTING', roles=roles, configured_threads=None,
                process_count=sum(row['count'] for row in roles), aggregate_limit_bytes=aggregate_bytes)


def derive_cpu_entry_capacity_contract(ledger_ref, option_id):
    """Return the exact guard admission or an unresolved reason for one candidate."""
    ledger = verify_cpu_entry_decision_ledger(ledger_ref)
    matches = [row for row in ledger['options'] if row['option_id'] == option_id and row['original'] is False]
    _require(len(matches) == 1, 'one exact native candidate option ID required')
    option, = matches
    report = dict(schema=SCHEMA, schema_version=1, ledger=ledger_ref, option_id=option_id,
        scope_sha256=ledger['scope_sha256'], legality=option['legality'], status='UNRESOLVED',
        reason='CANDIDATE_HAS_NO_SCOPED_VIRTUAL_ADDRESS_SPACE_CAP', native_launch=None,
        candidate_address_space_limit_bytes=None, provider_process_count=None, guard_scope=None,
        guard=None, accounting=None, evidence=None, whole_process_dram_peak_bytes=None,
        physical_dram_reservation_proven=False, execution_default='NO_OP', deployment_authorized=False)
    witness = option['memory']['virtual_address_space_headroom']
    if option['legality'] == 'INELIGIBLE':
        return dict(report, reason='CANDIDATE_LEGALITY_INELIGIBLE')
    if witness['status'] != 'VIRTUAL_ADDRESS_SPACE_CAPS':
        return report
    fact_ref, = option['legality_evidence']
    legality = _pointer(read_artifact(fact_ref['artifact']), fact_ref['locator'])
    generated = legality.get('generated_option', {})
    if generated.get('implementation_family') != _FAMILY:
        return dict(report, reason='NO_SCOPED_NATIVE_PROVIDER')
    plan = generated['proof']['emission_plan']
    if plan['provider_family'] != _ARRAY:
        return dict(report, reason='NATIVE_CAPACITY_INTERFACE_UNSUPPORTED')
    compiled, native, decision = _array_runtime(plan, witness)
    runtime_ref = compiled['runtime']
    memory = decision['memory']
    cap_bytes = witness['bytes']
    accounting = _accounting(decision['mode'], memory, decision['worker_count'], cap_bytes)
    report.update(accounting=accounting, provider_process_count=accounting['process_count'],
                  evidence=dict(provider=plan['authority'], runtime=runtime_ref, memory=witness['evidence']))
    if accounting['kind'] != 'SINGLE_PROCESS_INCLUDING_THREADS':
        return dict(report, reason='PROVIDER_LAYOUT_REQUIRES_A_SEPARATE_PROCESS_ROLE_GUARD')

    context = plan['context']
    if not all(context['scope_matches'].values()):
        return dict(report, reason='PROVIDER_LAUNCH_CONTEXT_DIFFERS')
    target = read_artifact(plan['target'])
    inventory_ref = dict(target['inventory'], schema='scar.target-inventory')
    if read_artifact(inventory_ref)['platform']['system'] != 'linux':
        return dict(report, reason='LINUX_RLIMIT_AS_TARGET_REQUIRED')
    invocation = compiled['invocation']
    binary = native['files'][native['binary']]
    environment = invocation['environment']
    _require(invocation['executable'] == binary and invocation['cwd'] == context['cwd']
             and invocation['cpu_ids'] == context['cpu_ids'] and invocation['original_executable'] == context['interpreter']
             and invocation['original_environment'] == context['original_environment']
             and environment == dict(context['original_environment'], **native['launch_environment'])
             and invocation['output_index'] == plan['contract']['publication']['output_index'] == len(plan['argv']) - 1
             and Path(binary['uri']).resolve(strict=True) == Path(runtime_ref['uri']).parent / native['binary']
             and read_artifact(binary, json_document=False).startswith(b'\x7fELF'),
             'address-space guard must bind the exact original scope, native ELF and public argv')
    schedule_ref = native['schedule']
    schedule = read_artifact(schedule_ref)
    thread_ref = native['thread_contract']
    threads = read_artifact(thread_ref)
    _require(schedule_ref['schema'] == _TEAM_PLAN and thread_ref['schema'] == _THREAD_PROOF
             and schedule['thread_contract'] == thread_ref and decision in schedule['alternatives']
             and native['scope'] == schedule['scope'] and native['family'] == schedule['family'] == threads['family']
             and native['native'] == schedule['native'] == threads['native']
             and 0 < cap_bytes <= option['memory']['request_budget_bytes'],
             'single-process cap must bind the admitted OpenMP schedule and request budget')
    report['evidence'].update(schedule=schedule_ref, thread_contract=thread_ref, target_inventory=inventory_ref)
    environment_sha256 = sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()
    return dict(report, status=ADMISSIBLE, reason='EXACT_SINGLE_PROCESS_PROVIDER_AND_DECLARED_ADDRESS_SPACE_CAP',
        native_launch=dict(executable=binary, argv=[binary['uri'], *plan['argv']],
                           cwd=invocation['cwd'], cpu_ids=invocation['cpu_ids'],
                           environment_sha256=environment_sha256, output_index=invocation['output_index']),
        candidate_address_space_limit_bytes=cap_bytes, guard_scope='PER_PROCESS_VIRTUAL_ADDRESS_SPACE',
        guard=dict(interface='LINUX_RLIMIT_AS', soft_limit_bytes=cap_bytes, hard_limit_bytes=cap_bytes,
                   applies_before='CANDIDATE_ELF_EXEC', candidate_only=True, may_raise_inherited_limits=False,
                   pre_exec_enforcement_observed=False, guarded_candidate_execution_verified=False))


def verify_cpu_entry_capacity_contract(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported capacity contract')
    replay = derive_cpu_entry_capacity_contract(saved['ledger'], saved['option_id'])
    _require(replay == saved, 'retained capacity contract differs at cold replay')
    return replay
