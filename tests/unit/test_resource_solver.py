"""The solver optimizes a declared model without executing a target program."""

from itertools import product

from scar.optimization.resource_solver import solve_resource_schedule
from scar.optimization.schedule import LogicalValue, Materialization, ResourceSchedule, Task, TaskKind


def transfer_schedule():
    value = LogicalValue("value")
    source = Materialization("host", "value", "host-address-space", external_ready=True, ownership="EXTERNAL")
    destination = Materialization("device", "value", "device-address-space",
                                  ready_task="copy", allocated_by="allocate", bytes_expr="8")
    tasks = (Task("allocate", TaskKind.ALLOCATION, writes=("device",)),
             Task("copy", TaskKind.COPY, ("host",), ("device",), ("allocate",), attrs={"non_executable": True}),
             Task("compute", TaskKind.COMPUTE, ("host",)),
             Task("consume", TaskKind.COMPUTE, ("device",), depends_on=("copy",)),
             Task("release", TaskKind.RELEASE, ("device",), depends_on=("consume",)))
    schedule = ResourceSchedule("transfer", (value,), (source, destination), tasks,
                                unresolved=({"reason": "COPY_BACKEND_NOT_IMPLEMENTED"},))
    arguments = dict(task_costs={"allocate": 0, "copy": 3, "compute": 5, "consume": 2, "release": 0},
                     task_placements={"allocate": {}, "copy": {"dma": 1}, "compute": {"cpu": 1},
                                      "consume": {"cpu": 1}, "release": {}},
                     resource_capacities={"cpu": 1, "dma": 1}, memory_capacities={"device-pool": 8},
                     materialization_sizes={"device": 8}, memory_locations={"device": "device-pool"})
    return schedule, arguments


def test_compute_and_copy_overlap_and_allocation_lifetime_is_budgeted():
    schedule, arguments = transfer_schedule()
    result = solve_resource_schedule(schedule, **arguments)
    assert result["status"] == "OPTIMAL_WITHIN_MODEL"
    assert result["solution"]["makespan_ticks"] == 7
    timed = {task["task"]: task for task in result["solution"]["tasks"]}
    assert timed["compute"]["start_ticks"] == timed["copy"]["start_ticks"] == 0
    assert timed["consume"]["start_ticks"] >= timed["copy"]["end_ticks"]
    assert result["solution"]["allocation_lifetimes"][0]["end_ticks"] == timed["release"]["end_ticks"]
    assert result["execution_unresolved"] and not result["target_program_executed"]
    impossible = solve_resource_schedule(schedule, **{**arguments, "memory_capacities": {"device-pool": 7}})
    assert impossible["status"] == "INFEASIBLE_WITHIN_MODEL" and impossible["solution"] is None


def test_missing_cost_and_budget_are_unresolved_instead_of_assumed_zero():
    schedule, arguments = transfer_schedule()
    missing_cost = {key: value for key, value in arguments["task_costs"].items() if key != "copy"}
    result = solve_resource_schedule(schedule, **{**arguments, "task_costs": missing_cost, "memory_capacities": None})
    fields = {item["field"] for item in result["unresolved"]}
    assert "task_costs.copy.duration_ticks" in fields and "memory_capacities.device-pool" in fields
    assert result["status"] == "UNRESOLVED" and result["solution"] is None
    assert result["lower_bound_ticks"] is None


def test_exact_search_matches_small_integer_oracle_and_reports_search_budget():
    tasks = (Task("a_short", TaskKind.COMPUTE), Task("b_critical", TaskKind.COMPUTE),
             Task("c_transfer", TaskKind.COMPUTE, depends_on=("b_critical",)),
             Task("d_tail", TaskKind.COMPUTE, depends_on=("c_transfer",)))
    schedule = ResourceSchedule("oracle", (), (), tasks)
    durations = {"a_short": 2, "b_critical": 3, "c_transfer": 2, "d_tail": 1}
    placement = {"a_short": ("cpu",), "b_critical": ("cpu",),
                 "c_transfer": ("dma",), "d_tail": ("cpu",)}
    oracle = None
    for starts in product(range(9), repeat=len(tasks)):
        start = dict(zip((task.id for task in tasks), starts))
        if any(start[task.id] < start[dep] + durations[dep] for task in tasks for dep in task.depends_on):
            continue
        if any(placement[left.id] == placement[right.id]
               and start[left.id] < start[right.id] + durations[right.id]
               and start[right.id] < start[left.id] + durations[left.id]
               for index, left in enumerate(tasks) for right in tasks[index + 1:]):
            continue
        makespan = max(start[key] + duration for key, duration in durations.items())
        oracle = min(oracle, makespan) if oracle is not None else makespan
    arguments = dict(task_costs=durations, task_placements=placement, resource_capacities={"cpu": 1, "dma": 1})
    result = solve_resource_schedule(schedule, **arguments)
    assert result["status"] == "OPTIMAL_WITHIN_MODEL"
    assert result["solution"]["makespan_ticks"] == oracle == 6
    assert result["lower_bound_ticks"] == 6 and result["gap_ticks"] == 0
    limited = solve_resource_schedule(schedule, **arguments, search_budget=10)
    assert limited["status"] == "SEARCH_BUDGET_EXHAUSTED"
    assert limited["lower_bound_ticks"] <= oracle
    if limited["solution"] is not None:
        assert limited["solution"]["makespan_ticks"] >= oracle
        assert limited["gap_ticks"] == limited["solution"]["makespan_ticks"] - limited["lower_bound_ticks"]


def test_versioned_schedule_uses_physical_slot_events_and_exact_capacities():
    # Declared integer toy durations prove solver/IR integration only. No
    # hardware rate or target-program performance is measured by this model.
    from test_versioned_schedule import ordered_state_schedule
    from scar.optimization.resource_solver import semantic_sha256
    from scar.optimization.schedule import schedule_document

    schedule = ordered_state_schedule()
    durations = dict(allocate=1, snapshot=2, compute=3, write=1, reuse=2, fence=1, commit=2, release=1)
    arguments = dict(task_costs=durations,
        task_placements={task.id: ({"cpu": 1} if task.id in ("compute", "write", "reuse") else {})
                         for task in schedule.tasks}, resource_capacities={"cpu": 1},
        memory_capacities={"workspace-budget": 32},
        materialization_sizes={"shadow": 16, "temp": 16},
        memory_locations={"shadow": "workspace-budget", "temp": "workspace-budget"})
    result = solve_resource_schedule(schedule, **arguments)
    assert result["status"] == "OPTIMAL_WITHIN_MODEL" and result["gap_ticks"] == 0
    assert result["solution"]["makespan_ticks"] == sum(durations.values()) == 13
    assert result["schedule_sha256"] == semantic_sha256(schedule_document(schedule))
    assert result["model"]["allocation_identity"] == "PHYSICAL_STORAGE_SLOT_ID"
    assert set(result["model"]["allocation_bytes_and_pools"]) == {"shadow", "temp"}
    lifetimes = result["solution"]["allocation_lifetimes"]
    assert {row["storage_slot"] for row in lifetimes} == {"shadow", "temp"}
    assert sum(row["bytes"] for row in lifetimes) == 32
    assert all(row["start_ticks"] == 0 and row["end_ticks"] == 13 for row in lifetimes)
    assert any(row["kind"] == "COMMIT" for row in result["solution"]["tasks"])
    assert result["storage_accounting"]["source_graph_budget_bytes"] == 48
    assert result["model"]["memory_capacities"] == {"workspace-budget": 32}
    too_small = solve_resource_schedule(schedule, **{**arguments, "memory_capacities": {"workspace-budget": 31}})
    assert too_small["status"] == "INFEASIBLE_WITHIN_MODEL" and too_small["solution"] is None
    compressed = solve_resource_schedule(schedule, **{**arguments, "materialization_sizes": {"shadow": 15, "temp": 16}})
    assert compressed["status"] == "UNRESOLVED"
    assert any(row["field"] == "materialization_sizes.shadow" and "capacity_bytes" in row["reason"]
               for row in compressed["unresolved"])
    mixed = solve_resource_schedule(schedule, **{**arguments,
        "materialization_sizes": {"shadow": 16, "temp": 16, "rhs0": 16}})
    assert mixed["status"] == "UNRESOLVED" and any(row["field"] == "materialization_sizes.rhs0"
                                                    for row in mixed["unresolved"])
    unknown = solve_resource_schedule(schedule, **{**arguments, "task_costs": {}})
    assert unknown["status"] == "UNRESOLVED" and unknown["solution"] is None
    assert all(row["field"].startswith("task_costs.") for row in unknown["unresolved"])
