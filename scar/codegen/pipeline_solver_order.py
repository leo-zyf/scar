"""Render only certified CPU-lane order over the original ready-host envelope.

Full source/domain/cost replay belongs to pipeline._admit_execution_schedule.
These structural checks also prevent a partial or inconsistent lane policy
from being emitted if this renderer is called directly.
"""
from .pipeline_ready import render_ready_source


SOLVER_ORDER_FAMILY = 'SOLVER_ORDER_CPU_SERIAL_LEAVES_V1'


_READY_POLICY = '''    auto next_for_cpu = [&](int cpu) {
        for (size_t i = 0; i < call_plans.size(); ++i)
            if (state[i] == 0 && call_cpu_ids[i] == cpu && predecessors_complete(i)) return i;
        return call_plans.size();
    };
    auto any_ready = [&]() {
        for (size_t i = 0; i < call_plans.size(); ++i)
            if (state[i] == 0 && predecessors_complete(i)) return true;
        return false;
    };
'''


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def render_solver_order_source(serial_source, pipeline, memory, report):
    _require(report.get('schema') == 'scar.pipeline-execution-refinement'
             and type(report.get('schema_version')) is int and report['schema_version'] == 1
             and report.get('execution_family') == SOLVER_ORDER_FAMILY
             and report.get('requires_serial_leaf_operators') is True
             and report.get('dispatch_policy') ==
             'FIXED_CPU_LANE_HEAD_ONLY_WAIT_TRUE_PREDECESSORS_NO_TIMESTAMP_WAIT'
             and report.get('full_model_timeline_implemented') is False
             and report.get('task_model_makespan_is_native_runtime_prediction') is False,
             'a compute-only solver execution refinement is required')
    indexes = {call.id: index for index, call in enumerate(pipeline.calls)}
    cpus, rows = report['cpu_ids'], report['model_compute_order']
    _require(type(cpus) is list and bool(cpus) and
             all(type(cpu) is int and cpu >= 0 for cpu in cpus) and len(set(cpus)) == len(cpus),
             'solver CPU lanes must be distinct explicit CPU IDs')
    _require(len(indexes) == len(pipeline.calls) and type(rows) is list
             and len(rows) == len(indexes) and
             len({row['call'] for row in rows}) == len(rows) and
             {row['call'] for row in rows} == set(indexes) and
             set(report['call_cpu_ids']) == set(indexes),
             'solver order must cover each original call exactly once')
    lanes = {str(cpu): [] for cpu in cpus}
    for row in rows:
        cpu = report['call_cpu_ids'][row['call']]
        _require(type(cpu) is int and cpu in cpus and row['cpu_id'] == cpu,
                 'solver call placement differs from its fixed CPU lane')
        lanes[str(cpu)].append(row['call'])
    _require(report['lane_call_order'] == lanes,
             'solver compute order and certified per-lane order differ')
    ordered = [indexes[row['call']] for row in rows]
    source = render_ready_source(serial_source, pipeline, memory, report)
    declaration = ('constexpr std::array<size_t, ' + str(len(ordered))
        + '> solver_compute_order = {{' + ','.join(map(str, ordered)) + '}};\n')
    marker = 'constexpr int32_t worker_guard = 16;\n'
    _require(source.count(marker) == 1 and source.count(_READY_POLICY) == 1,
             'original ready-host lane policy and declaration marker must be unique')
    source = source.replace(marker, marker + declaration, 1)
    policy = '''    // Resource order is a certified implementation refinement. Never
    // bypass a pending lane head merely because a later call is ready.
    auto next_for_cpu = [&](int cpu) {
        for (const size_t i : solver_compute_order) {
            if (state[i] != 0 || call_cpu_ids[i] != cpu) continue;
            return predecessors_complete(i) ? i : call_plans.size();
        }
        return call_plans.size();
    };
    auto any_ready = [&]() {
        for (const int cpu : lane_cpu_ids)
            if (next_for_cpu(cpu) != call_plans.size()) return true;
        return false;
    };
'''
    return source.replace(_READY_POLICY, policy, 1)


__all__ = ['SOLVER_ORDER_FAMILY', 'render_solver_order_source']
