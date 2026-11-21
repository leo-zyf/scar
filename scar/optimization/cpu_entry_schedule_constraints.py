"""Project the fixed resource and order constraints of a qualified CPU entry."""

from dataclasses import asdict
from pathlib import Path

from scar.analysis.cpu_entry_lifecycle_graph import verify_cpu_entry_lifecycle_graph
from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.target_hardware import CpuCore, HardwareThread
from .cpu_entry_capacity_contract import ADMISSIBLE


SCHEMA = 'scar.cpu-entry-schedule-constraints'
EXACT = 'EXACT_ORDINAL_LANE_CONSTRAINTS'
_NORMAL = 'SELECTED_GUARDED_CANDIDATE_NORMAL_COMPLETION'
_CONDITION = 'QUALIFIED_OPENMP_PLACEMENT_AND_NORMAL_COMPLETION'
_HEADER = 'include/scar_generated/native_ordered_openmp.hpp'


def _require(condition, message):
    if not condition:
        raise ValueError('CPU entry schedule constraints: ' + message)


def _fact(artifact, locator):
    return dict(artifact=artifact, locator=locator)


def _operation(lane, kind):
    matches = [row for row in lane['operations'] if row['kind'] == kind]
    _require(len(matches) == 1, 'expected one scoped lifecycle operation: ' + kind)
    return matches[0]


def _provider(capacity, work):
    from scar.analysis.cpu_native_phase_sites import _retained_runtime
    from scar.codegen import native_seeded_array_task_entry as emitter
    from .independent_array_task_team_schedule import MODE, SCHEMA as SCHEDULE

    runtime_ref = capacity['evidence']['runtime']
    schedule_ref = capacity['evidence']['schedule']
    runtime, schedule = map(read_artifact, (runtime_ref, schedule_ref))
    decision = runtime['decision']
    cpus, width = decision['worker_cpu_ids'], decision['worker_count']
    count = schedule['scope']['task_count']
    threads = schedule['thread_domain']
    _require(runtime_ref['schema'] == emitter.SCHEMA and schedule_ref['schema'] == SCHEDULE
             and runtime['schedule'] == schedule_ref and runtime['scope'] == schedule['scope']
             and runtime['thread_contract'] == schedule['thread_contract'] == capacity['evidence']['thread_contract']
             and decision in schedule['alternatives'] and not decision['capacity_reasons'] and decision['mode'] == MODE
             and threads['assignment'] == 'ORDINAL_MOD_WORKER_COUNT' and threads['owner_participates'] is True
             and threads['publication'] == 'ORDERED_AFTER_ALL_LANES_JOIN'
             and threads['shared_result_slots_disjoint'] is True and runtime['no_worker_visible_output'] is True
             and schedule['dispatch']['terminal_publication'] == 'AFTER_ALL_SUCCEEDED_IN_ORDINAL_ORDER',
             'the native entry lacks a fixed ordinal-to-lane dispatch and owner join')
    _require(type(count) is type(width) is int and 1 < width <= count < 2**63
             and len(cpus) == len(set(cpus)) == width and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
             and cpus[0] == decision['owner_cpu_id'] == work['owner_cpu']
             and cpus == work['worker_cpu_ids'] and count == work['task_count']
             and width == work['configured_threads'] == capacity['accounting']['configured_threads']
             and runtime['cpu_ids'] == decision['launch_cpu_ids'] == capacity['native_launch']['cpu_ids']
             and set(cpus) <= set(runtime['cpu_ids']), 'compiled lane counts or CPU placement differ from the entry graph')
    # Re-render the admitted main so a provider decision cannot be paired with
    # an unrelated executable shape. The dispatcher is a frozen source member.
    source, _, _, _, _, _ = _retained_runtime(capacity)
    header = runtime['files'][_HEADER]
    _require(Path(header['uri']) == Path(runtime_ref['uri']).parent / _HEADER
             and read_artifact(header, json_document=False) == emitter._OPENMP.read_bytes(),
             'retained OpenMP dispatcher differs from the admitted lane protocol')
    return cpus, count, dict(runtime=runtime_ref, schedule=schedule_ref, main_source=source, dispatcher=header,
        dispatcher_template=artifact_reference(emitter._OPENMP, 'scar.native-cpp-source-template'),
        provider_decision=_fact(runtime_ref, '#/decision'), assignment=_fact(schedule_ref, '#/thread_domain/assignment'))


def _lanes(target_ref, placement, cpus, count):
    target = TargetHardwareGraph(Path(target_ref['uri']).parent)
    _require(asdict(target._snapshot) == target_ref, 'CPU resources must belong to the exact retained target')
    inventory = target.resources()
    threads = {row.cpu_id: row.id.wire for row in inventory if type(row) is HardwareThread and row.online is True}
    cores = {cpu: row.id.wire for row in inventory if type(row) is CpuCore for cpu in row.cpu_ids}
    endpoints = placement['endpoints']
    _require(all(cpu in threads and cpu in cores and endpoints.get(str(cpu)) == [threads[cpu], cores[cpu]] for cpu in cpus)
             and len({cores[cpu] for cpu in cpus}) == len(cpus), 'planned lanes need distinct original physical cores')
    width = len(cpus)
    quotient, remainder = divmod(count, width)
    return [dict(lane_index=index, cpu_id=cpu, target_thread=threads[cpu], target_core=cores[cpu],
                 first_ordinal=index, stride=width, task_count=quotient + int(index < remainder))
            for index, cpu in enumerate(cpus)]


def _normal_order(graph_ref, lane_index, lane, work):
    guard = _operation(lane, 'PRE_EXEC_CAPACITY_GUARD')
    join = _operation(lane, 'WORKER_TEAM_JOIN')
    publication = _operation(lane, 'FINAL_OWNER_PUBLICATION')
    expected = [(guard['id'], work['id']), (work['id'], join['id']), (join['id'], publication['id'])]
    edges = {}
    for index, row in enumerate(lane['edges']):
        relation = row['source'], row['target']
        if row['kind'] == 'HAPPENS_BEFORE' and row['when'] == _NORMAL and relation in expected:
            _require(relation not in edges, 'duplicate normal-entry ordering proof')
            edges[relation] = dict(source=row['source'], target=row['target'], when=row['when'],
                                   evidence=_fact(graph_ref, f'#/lanes/{lane_index}/edges/{index}'))
    _require(set(edges) == set(expected), 'normal guard, work, join and publication order required')
    return [edges[key] for key in expected]


def derive_cpu_entry_schedule_constraints(lifecycle_graph_ref):
    """Build a schedule constraint receipt without unrolling tasks or timing them."""
    graph = verify_cpu_entry_lifecycle_graph(lifecycle_graph_ref)
    capacity_ref = graph['capacity_contract']
    capacity = None if capacity_ref is None else read_artifact(capacity_ref)
    option_id = None if capacity is None else capacity['option_id']
    report = dict(schema=SCHEMA, schema_version=1, lifecycle_graph=lifecycle_graph_ref,
        scope_sha256=graph['scope_sha256'], option_id=option_id, status='UNRESOLVED',
        reason='NO_ADMITTED_FIXED_OPENMP_CANDIDATE', task_count=None, worker_count=None, lanes=[],
        evidence=None, normal_path_edges=[], logical_order=None, resource_occupancy=None, body_work_lower_bound=None,
        whole_entry_time_bound_ns=None, complete_lifecycle_proven=False, deployment_authorized=False)
    if capacity is None or capacity['status'] != ADMISSIBLE:
        return report
    candidates = [(index, lane) for index, lane in enumerate(graph['lanes'])
                  if lane['option_id'] == option_id and lane['role'] == 'CONDITIONAL_NATIVE_CANDIDATE']
    _require(len(candidates) == 1 and graph['scope_sha256'] == capacity['scope_sha256']
             and capacity['provider_process_count'] == 1, 'schedule constraints require one exact candidate lane')
    (lane_index, lane), = candidates
    work = _operation(lane, 'SCHEDULED_NUMERICAL_TEAM')
    cpus, count, evidence = _provider(capacity, work)
    ledger_ref = capacity['ledger']
    ledger = read_artifact(ledger_ref)
    option, = (row for row in ledger['options'] if row['option_id'] == option_id)
    placement = option['cpu_placement']
    _require(placement['cpu_ids'] == capacity['native_launch']['cpu_ids'] and ledger['target']['graph'] == graph['target'],
             'ledger placement differs from the admitted launch and target')
    lanes = _lanes(graph['target'], placement, cpus, count)
    _require(sum(row['task_count'] for row in lanes) == count, 'closed-form lane partition must cover every ordinal')
    evidence.update(capacity_contract=capacity_ref, decision_ledger=ledger_ref, target=graph['target'])
    return dict(report, status=EXACT, reason='FIXED_ORDINAL_DISPATCH_ON_QUALIFIED_CPU_LANES',
        task_count=count, worker_count=len(cpus), lanes=lanes, evidence=evidence,
        normal_path_edges=_normal_order(lifecycle_graph_ref, lane_index, lane, work),
        logical_order=dict(condition=_CONDITION, lane_order='ASCENDING_ORDINAL_ON_EACH_LANE',
                           assignment='ORDINAL_MOD_WORKER_COUNT', cross_lane_order='UNSPECIFIED',
                           join_requires_all_lane_tails=True),
        resource_occupancy=dict(condition=_CONDITION, task_slots_per_lane=1, maximum_active_task_lanes=len(cpus),
                                physical_cores_distinct=True, simultaneous_execution_required=False,
                                uncontended_service_proven=False),
        body_work_lower_bound=dict(kind='MAX_SERIAL_LANE_SERVICE_SUM',
            max_tasks_on_lane=max(row['task_count'] for row in lanes), service_times_ns=None, lower_ns=None))


def verify_cpu_entry_schedule_constraints(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU entry schedule receipt')
    replay = derive_cpu_entry_schedule_constraints(saved['lifecycle_graph'])
    _require(replay == saved, 'retained CPU schedule constraints differ at cold replay')
    return replay
