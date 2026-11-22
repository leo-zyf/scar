"""Exact fixed-placement scheduling and sourced implementation choices.

No timings are sampled and no costs are inferred from operation names. Durations
are caller-declared integer ticks; executor/channel capacities and incremental
allocation budgets must also be explicit. Optimality refers only to this model,
not to physical hardware, unmodeled transfers, or a production implementation.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

from .schedule import ResourceSchedule, TaskKind, schedule_document, validate_schedule
from scar.ir.program.providers import ProviderFactReference
from scar.ir.record_codec import decode, encode, loads
from scar.ir.versioned_schedule import VersionedResourceSchedule, _encode_validated_schedule


_MODEL_ASSUMPTIONS = [
    "EXPLICIT_INTEGER_TIME_TICKS_AND_FIXED_NONPREEMPTIVE_TASK_DURATIONS",
    "FIXED_TASK_PLACEMENT_AND_CONSTANT_RENEWABLE_RESOURCE_CAPACITIES",
    "TASK_PLACEMENTS_ARE_INPUTS_NOT_JOINTLY_SOLVED",
    "RESOURCE_CAPACITY_ONE_MEANS_EXCLUSIVE_EXECUTOR_OR_CHANNEL",
    "ALLOCATION_RESERVES_BYTES_AT_START_RELEASE_RETURNS_BYTES_AT_COMPLETION",
    "INCREMENTAL_MEMORY_BUDGET_EXCLUDES_EXTERNALLY_OWNED_INPUT_ALLOCATIONS",
    "UNRELEASED_AND_ESCAPING_STORAGE_REMAINS_LIVE_THROUGH_MAKESPAN",
    "NO_UNMODELED_CONTENTION_TRANSFER_LATENCY_OR_DYNAMIC_ALLOCATION_COST",
    "OPTIMALITY_IS_ONLY_WITHIN_THE_GIVEN_DISCRETE_MODEL",
]


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def semantic_sha256(document) -> str:
    """Digest canonical semantic JSON, independently of any raw-file receipt."""
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _shape_instantiation(plan, input_shapes):
    unknown = object()
    issues, declared = [], {}
    if isinstance(input_shapes, (list, tuple)):
        requested = {parameter: input_shapes for parameter in plan.region.parameters}
    elif isinstance(input_shapes, Mapping):
        requested = input_shapes
    else:
        requested = {}
    for parameter in plan.region.parameters:
        shape = requested.get(parameter)
        if not (isinstance(shape, (tuple, list)) and len(shape) == 2
                and all(_integer(axis) and axis <= 2**63 - 1 for axis in shape)):
            issues.append({"field": f"input_shapes.{parameter}",
                           "reason": "explicit rank-two nonnegative shape required for byte instantiation"})
        else:
            declared[parameter] = tuple(shape)
    shapes = {}
    for node in plan.region.nodes:
        if node.op == "input":
            shapes[node.id] = declared.get(node.attrs["name"], unknown)
        elif node.op == "const":
            shapes[node.id] = None
        elif node.op == "slice":
            parent = shapes[node.inputs[0]]
            shapes[node.id] = (unknown if parent is unknown else
                               tuple(max(0, stop - start) for start, stop, _ in
                                     (slice(*bounds).indices(extent) for bounds, extent in
                                      zip(node.attrs["bounds"], parent))))
        elif node.op == "copy":
            parent = shapes[node.inputs[0]]
            shapes[node.id] = parent
            if parent is None:
                issues.append({"field": f"input_shapes.operation.{node.id}",
                               "reason": "exact ndarray copy requires an array operand"})
        else:
            operands = [shapes[key] for key in node.inputs]
            if any(shape is unknown for shape in operands):
                shapes[node.id] = unknown
                continue
            arrays = [shape for shape in operands if shape is not None]
            if any(shape != arrays[0] for shape in arrays[1:]):
                issues.append({"field": f"input_shapes.operation.{node.id}",
                               "reason": "array operand shapes differ; the ABI does not broadcast arrays"})
                shapes[node.id] = unknown
            else:
                shapes[node.id] = arrays[0] if arrays else None
    output = shapes[plan.region.output]
    if output is unknown or output is None:
        return declared, None, issues
    return declared, output, issues


def hardware_model_skeleton(plan, schedule, *, input_shapes=None) -> dict:
    """Derive known fixed-CPU constraints, leaving costs and budgets unknown.

    Shapes are instantiated by the original DAG's unit-step slice rules. The
    host pool is an incremental process-address-space budget, never a claim of
    NUMA page placement. This helper does not jointly optimize placements.
    """
    from scar.compiler.model import document
    from scar.compiler.model import validate_plan
    from scar.ir.record_codec import encode
    from .hardware import validate_cpu_placement

    validate_plan(plan)
    validate_schedule(schedule)
    if schedule.region_id != plan.region.id or schedule.loop_axes != plan.schedule:
        raise ValueError("model skeleton schedule differs from plan region or loop axes")
    unresolved = []
    binding = plan.resource_binding
    if binding is None:
        unresolved.append({"field": "plan.resource_binding", "reason": "recorded CPU IDs required; CPU count is not a placement"})
    else:
        validate_cpu_placement(binding)
    capacities, placements, memory_caps, sizes, locations = {}, {}, {}, {}, {}
    if binding is not None:
        capacities[binding.executor_id] = len(binding.cpu_ids)
    for task in schedule.tasks:
        unresolved.append({"field": f"task_costs.{task.id}.duration_ticks",
                           "reason": "hardware topology and work counts do not establish task duration"})
        if task.kind in (TaskKind.ALLOCATION, TaskKind.EXPORT):
            placements[task.id] = {}
        elif task.kind is TaskKind.COMPUTE and binding is not None:
            if (task.attrs.get("region_id") != plan.region.id
                    or task.attrs.get("cpu_threads") != plan.cpu_threads
                    or task.attrs.get("executor_id") != binding.executor_id
                    or semantic_sha256(task.attrs.get("nodes")) != semantic_sha256([encode(n) for n in plan.region.nodes])):
                unresolved.append({"field": f"task_placements.{task.id}",
                                   "reason": "compute payload or executor differs from fixed region binding"})
            else:
                placements[task.id] = {binding.executor_id: plan.cpu_threads}
        else:
            unresolved.append({"field": f"task_placements.{task.id}",
                               "reason": "COPY/WAIT/RELEASE or unbound task placement is not inferred or implemented"})
    declared_shapes, output_shape, shape_issues = _shape_instantiation(plan, input_shapes)
    unresolved.extend(shape_issues)
    output_elements = output_shape[0] * output_shape[1] if output_shape is not None else None
    budget = plan.cost.get("memory_budget_bytes")
    for material in schedule.materializations:
        if material.ownership != "SCHEDULE" or material.allocated_by is None:
            continue
        if material.location != "host-address-space:$CALL_PROCESS":
            unresolved.append({"field": f"memory_locations.{material.id}",
                               "reason": "device/NUMA physical pools are not derived from UNBOUND host policy"})
            continue
        pool = material.location
        locations[material.id] = pool
        if budget is None:
            unresolved.append({"field": f"memory_capacities.{pool}",
                               "reason": "available incremental budget is unknown; reported total capacity is not a budget"})
        else:
            memory_caps[pool] = budget
        if material.bytes_expr == "8 * output_elements" and output_elements is not None:
            sizes[material.id] = 8 * output_elements
        else:
            unresolved.append({"field": f"materialization_sizes.{material.id}",
                               "reason": "symbolic bytes need supported expression and an instantiated output shape"})
    schedule_hash = semantic_sha256(schedule_document(schedule))
    instantiation = {
        "region_id": plan.region.id, "plan_sha256": semantic_sha256(document(plan)),
        "schedule_sha256": schedule_hash,
        "input_shapes": {parameter: list(shape) for parameter, shape in declared_shapes.items()},
        "output_shape": list(output_shape) if output_shape is not None else None,
        "output_elements": output_elements,
        "output_bytes": 8 * output_elements if output_elements is not None else None,
        "shape_rule": "EXACT_SCALAR_DAG_UNIT_STEP_SLICE_SHAPES_NO_BROADCAST",
        "shape_authority": "CALLER_DECLARED_INPUT_SHAPES_WITH_EXACT_DAG_SLICE_RULES",
        "complete": not shape_issues and output_shape is not None,
    }
    return {
        "schema": "scar.resource-model-skeleton", "schema_version": 1,
        "status": "PARTIAL_MODEL", "placement_optimization": "FIXED_INPUT_PLACEMENTS_NOT_JOINT",
        "schedule_sha256": schedule_hash, "task_costs": {}, "task_placements": placements,
        "resource_capacities": capacities, "memory_capacities": memory_caps,
        "materialization_sizes": sizes, "memory_locations": locations,
        "model_instantiation": instantiation, "unresolved": unresolved,
        "provenance": {"input_shapes": instantiation["input_shapes"],
                       "output_shape": instantiation["output_shape"],
                       "shape_authority": instantiation["shape_authority"],
                       "plan_sha256": instantiation["plan_sha256"],
                       "schedule_sha256": schedule_hash},
        "resource_evidence": {
            "target_path": binding.target_path if binding else None,
            "target_sha256": binding.target_sha256 if binding else None,
            "executor_id": binding.executor_id if binding else None,
            "cpu_ids": list(binding.cpu_ids) if binding else [],
            "cpu_capacity_basis": "BOUND_CPU_IDS_COUNT_ONLY",
            "memory_policy": "UNBOUND", "memory_pool_scope": "HOST_PROCESS_INCREMENTAL_ALLOCATIONS",
            "numa_residency_proven": False, "costs_inferred": False,
        },
    }


def _allocation_view(schedule):
    """Adapt storage events, leaving the common makespan search unchanged.

    V1 allocation identities are materializations. V2 identities are physical
    slots: multiple immutable content versions/replicas never allocate a slot
    again. The original public graph is still validated and hashed as-is.
    """
    if isinstance(schedule, VersionedResourceSchedule):
        records = {slot.id: slot for slot in schedule.storage_slots
                   if slot.ownership == "SCHEDULE" and slot.allocated_by is not None}
        return records, {task.id: task.allocates for task in schedule.tasks}, {
            task.id: task.releases for task in schedule.tasks}
    records = {material.id: material for material in schedule.materializations
               if material.ownership == "SCHEDULE" and material.allocated_by is not None}
    return records, {
        task.id: tuple(key for key in task.writes if key in records)
                 if task.kind is TaskKind.ALLOCATION else () for task in schedule.tasks}, {
        task.id: task.reads if task.kind is TaskKind.RELEASE else () for task in schedule.tasks}


@dataclass(frozen=True, slots=True)
class _FixedSchedule:
    schedule: ResourceSchedule | VersionedResourceSchedule
    document: dict
    sha256: str
    tasks: dict
    task_ids: tuple[str, ...]
    order: tuple[str, ...]
    dependencies: dict[str, frozenset[str]]
    allocation_records: dict
    allocation_writes: dict
    release_reads: dict


def _prepare_fixed_schedule(schedule):
    """Admit the graph once for all costs and placements in a fixed domain."""
    order = tuple(validate_schedule(schedule)['topological_order'])
    document = (_encode_validated_schedule(schedule)
                if isinstance(schedule, VersionedResourceSchedule) else schedule_document(schedule))
    tasks = {task.id: task for task in schedule.tasks}
    allocation_records, allocation_writes, release_reads = _allocation_view(schedule)
    return _FixedSchedule(
        schedule, document, semantic_sha256(document), tasks, tuple(sorted(tasks)), order,
        {key: frozenset(task.depends_on) for key, task in tasks.items()},
        allocation_records, allocation_writes, release_reads,
    )


def _critical_path_bound(graph, durations, completed=(), running=(), now=0):
    finish = dict.fromkeys(completed, 0)
    finish.update((key, end - now) for key, end in running)
    for key in graph.order:
        if key not in finish:
            finish[key] = durations[key] + max((finish[parent] for parent in graph.dependencies[key]), default=0)
    return max(finish.values(), default=0)


def _problem(graph, task_costs, task_placements, resource_capacities,
             memory_capacities, materialization_sizes, memory_locations):
    unresolved = []

    def missing(field, reason):
        unresolved.append({"field": field, "reason": reason})

    def mapping(value, field):
        if not isinstance(value, Mapping):
            missing(field, "explicit mapping required")
            return {}
        return value

    costs = mapping(task_costs, "task_costs")
    placements = mapping(task_placements, "task_placements")
    capacities = mapping(resource_capacities, "resource_capacities")
    memory_caps = {} if memory_capacities is None else mapping(memory_capacities, "memory_capacities")
    sizes = {} if materialization_sizes is None else mapping(materialization_sizes, "materialization_sizes")
    locations = {} if memory_locations is None else mapping(memory_locations, "memory_locations")
    for resource, capacity in capacities.items():
        if not isinstance(resource, str) or not resource or not _integer(capacity, 1):
            missing(f"resource_capacities.{resource}", "resource names and positive integer capacities required")
    for pool, budget in memory_caps.items():
        if not isinstance(pool, str) or not pool or not _integer(budget):
            missing(f"memory_capacities.{pool}", "pool names and nonnegative incremental budgets required")
    durations, demands = {}, {}
    for task in graph.tasks.values():
        raw = costs.get(task.id)
        duration = raw.get("duration_ticks") if isinstance(raw, Mapping) else raw
        if isinstance(raw, Mapping) and raw.get("status") in ("UNKNOWN", "UNRESOLVED"):
            missing(f"task_costs.{task.id}.status", "declared cost remains unknown")
        if not _integer(duration):
            missing(f"task_costs.{task.id}.duration_ticks", "nonnegative integer ticks required; unknown is not zero")
        else:
            durations[task.id] = duration
        raw = placements.get(task.id)
        if isinstance(raw, (list, tuple)) and all(isinstance(r, str) and r for r in raw):
            if len(set(raw)) != len(raw):
                missing(f"task_placements.{task.id}", "resource IDs must be unique")
                continue
            raw = {r: 1 for r in raw}
        if not isinstance(raw, Mapping):
            missing(f"task_placements.{task.id}", "explicit resource demands required; use {} for control-only tasks")
            continue
        demands[task.id] = {}
        for resource, units in raw.items():
            if not isinstance(resource, str) or not resource or not _integer(units, 1):
                missing(f"task_placements.{task.id}.{resource}", "nonempty resource ID and positive integer demand required")
                continue
            capacity = capacities.get(resource)
            if not _integer(capacity, 1):
                missing(f"resource_capacities.{resource}", "explicit positive integer capacity required")
            demands[task.id][resource] = units
    allocations = graph.allocation_records
    versioned = isinstance(graph.schedule, VersionedResourceSchedule)
    if versioned:
        for field, values in (("materialization_sizes", sizes), ("memory_locations", locations)):
            for key in values.keys() - allocations.keys():
                missing(f"{field}.{key}",
                        "v2 accepts scheduled physical slot IDs only, not materialization or external slot IDs")
    allocation_model = {}
    for key, material in allocations.items():
        size, location = sizes.get(key), locations.get(key)
        valid_size = _integer(size)
        if not _integer(size):
            missing(f"materialization_sizes.{key}",
                    "explicit physical slot capacity_bytes required" if versioned else
                    "explicit nonnegative byte size required; symbolic bytes need instantiation")
        elif versioned and size != material.capacity_bytes:
            valid_size = False
            missing(f"materialization_sizes.{key}", "v2 size must equal the original physical slot capacity_bytes")
        if not isinstance(location, str) or not location:
            missing(f"memory_locations.{key}", "explicit budget pool required; UNBOUND is not a NUMA placement")
        elif not _integer(memory_caps.get(location)):
            missing(f"memory_capacities.{location}", "explicit incremental byte budget required; total capacity is not free budget")
        if valid_size and isinstance(location, str) and location and _integer(memory_caps.get(location)):
            allocation_model[key] = (location, size)
    return durations, demands, dict(capacities), dict(memory_caps), allocation_model, unresolved


def solve_resource_schedule(schedule: ResourceSchedule | VersionedResourceSchedule, *, task_costs=None,
                            task_placements=None, resource_capacities=None,
                            memory_capacities=None, materialization_sizes=None,
                            memory_locations=None, model_instantiation=None,
                            search_budget=100000) -> dict:
    """Minimize makespan in a declared task/resource/allocation model.

    task_costs maps task IDs to nonnegative integer durations (or records with
    duration_ticks). task_placements maps IDs to resource->units records, or
    lists of exclusive resource names. Every task needs both fields, including
    explicit zero-duration/control-only tasks. Each owned allocation also needs
    instantiated bytes, a memory pool, and that pool's incremental byte budget.
    For a v2 graph, the legacy argument names materialization_sizes and
    memory_locations refer exclusively to SCHEDULE-owned physical slot IDs;
    sizes must exactly equal slot.capacity_bytes. They do not count logical
    versions or external roots. The v2 graph's separate all-region budget is
    enforced by validate_schedule, not substituted for these pool budgets.

    Search enumerates eligible task starts and advances to completion events.
    For constant, nonpreemptive durations/resources and allocation lifetimes,
    a feasible schedule can be left-shifted to these events without increasing
    makespan. Memoization merges identical running/completed/live-storage
    states; critical-path and resource-work lower bounds prune worse branches.
    """
    graph = _prepare_fixed_schedule(schedule)
    return _solve_fixed_schedule(
        graph, task_costs=task_costs, task_placements=task_placements, resource_capacities=resource_capacities,
        memory_capacities=memory_capacities, materialization_sizes=materialization_sizes,
        memory_locations=memory_locations, model_instantiation=model_instantiation, search_budget=search_budget,
    )


def _solve_fixed_schedule(graph, *, task_costs, task_placements, resource_capacities,
                          memory_capacities, materialization_sizes, memory_locations,
                          model_instantiation, search_budget):
    schedule = graph.schedule
    if not _integer(search_budget, 1):
        raise ValueError("search_budget must be a positive integer state budget")
    durations, demands, capacities, memory_caps, allocation_model, unresolved = _problem(
        graph, task_costs, task_placements, resource_capacities,
        memory_capacities, materialization_sizes, memory_locations)
    if isinstance(schedule, VersionedResourceSchedule) and schedule.temporal is not None:
        unresolved.append({"field": "temporal", "reason":
            "explicit event, occupancy and recurrence constraints require a temporal solver adapter; "
            "the legacy task-only solver must not silently discard them"})
    if model_instantiation is not None:
        if not isinstance(model_instantiation, Mapping):
            unresolved.append({"field": "model_instantiation", "reason": "explicit shape/source receipt must be a mapping"})
        else:
            try:
                semantic_sha256(dict(model_instantiation))
            except (TypeError, ValueError):
                unresolved.append({"field": "model_instantiation", "reason": "receipt must be finite semantic JSON"})
    result = {
        "schema": "scar.resource-solver", "schema_version": 1,
        "region_id": schedule.region_id, "objective": "MINIMIZE_MAKESPAN",
        "placement_optimization": "FIXED_INPUT_PLACEMENTS_NOT_JOINT",
        "schedule_sha256": graph.sha256, "model_sha256": None,
        "time_unit": "CALLER_DECLARED_TICKS", "status": "UNRESOLVED",
        "optimality": "NOT_ESTABLISHED", "physical_optimality_proven": False,
        "model_assumptions": list(_MODEL_ASSUMPTIONS),
        "unresolved": unresolved, "execution_unresolved": list(schedule.unresolved),
        "source_schedule_executable": schedule.executable, "target_program_executed": False,
        "solution": None, "lower_bound_ticks": None, "gap_ticks": None,
        "relative_gap": None, "states_explored": 0, "search_budget": search_budget,
    }
    versioned = isinstance(schedule, VersionedResourceSchedule)
    if versioned:
        result["storage_accounting"] = {
            "allocation_identity": "SCHEDULE_OWNED_PHYSICAL_STORAGE_SLOT_ID",
            "sizes_argument": "materialization_sizes", "pools_argument": "memory_locations",
            "size_authority": "EXACT_ORIGINAL_SLOT_CAPACITY_BYTES",
            "memory_capacities_scope": "EXPLICIT_SCHEDULED_ALLOCATION_POOL_BUDGETS",
            "external_storage_in_solver_allocation_model": False,
            "source_graph_budget_scope": schedule.budget_scope,
            "source_graph_budget_bytes": schedule.budget_bytes,
            "source_graph_budget_validated_separately": True,
        }
        result["model_assumptions"] += [
            "VERSIONED_GRAPH_MEMORY_IS_COUNTED_PER_PHYSICAL_SLOT_NOT_PER_MATERIALIZATION",
            "SOURCE_GRAPH_REGION_BUDGET_INCLUDING_EXTERNAL_STORAGE_IS_VALIDATED_SEPARATELY",
        ]
    if unresolved:
        return result
    result["model"] = {"task_durations_ticks": durations, "resource_demands": demands,
                       "resource_capacities": capacities, "memory_capacities": memory_caps,
                       "allocation_bytes_and_pools": {key: {"memory_pool": pool, "bytes": size}
                                                      for key, (pool, size) in allocation_model.items()}}
    if model_instantiation is not None:
        result["model"]["instantiation"] = deepcopy(dict(model_instantiation))
    if versioned:
        result["model"]["allocation_identity"] = "PHYSICAL_STORAGE_SLOT_ID"
    result["model_sha256"] = semantic_sha256(result["model"])
    tasks, ids, dependencies = graph.tasks, graph.task_ids, graph.dependencies
    allocation_records = graph.allocation_records
    allocation_writes, release_reads = graph.allocation_writes, graph.release_reads

    def lower_bound(time, completed, running):
        bound = _critical_path_bound(graph, durations, completed, running.items(), time)
        remaining_work = {}
        for key in ids:
            if key in completed:
                continue
            duration = running[key] - time if key in running else durations[key]
            for resource, units in demands[key].items():
                remaining_work[resource] = remaining_work.get(resource, 0) + duration * units
        for resource, work in remaining_work.items():
            bound = max(bound, (work + capacities[resource] - 1) // capacities[resource])
        return time + bound

    initial_lower_bound = lower_bound(0, frozenset(), {})
    result["lower_bound_ticks"] = initial_lower_bound
    violations = []
    for key in ids:
        for resource, units in demands[key].items():
            if units > capacities[resource]:
                violations.append({"field": f"task_placements.{key}.{resource}",
                                   "reason": "task demand exceeds fixed capacity"})
    for key, (location, size) in allocation_model.items():
        if size > memory_caps[location]:
            violations.append({"field": f"materialization_sizes.{key}",
                               "reason": "allocation exceeds explicit pool budget"})
    if violations:
        result.update(status="INFEASIBLE_WITHIN_MODEL", optimality="INFEASIBILITY_ESTABLISHED",
                      constraint_violations=violations)
        return result

    best_time, best_starts = None, None
    explored, exhausted = 0, False
    visited = set()

    frontier = [(0, frozenset(), {}, frozenset(), {})]
    while frontier:
        if explored >= search_budget:
            exhausted = True
            break
        time, completed, running, live, starts = frontier.pop()
        signature = (time, tuple(sorted(completed)), tuple(sorted(running.items())), tuple(sorted(live)))
        if signature in visited:
            continue
        visited.add(signature)
        explored += 1
        bound = lower_bound(time, completed, running)
        if best_time is not None and bound >= best_time:
            continue
        if len(completed) == len(tasks):
            best_time, best_starts = time, dict(starts)
            if best_time == initial_lower_bound:
                break  # Matching global lower/upper bounds already prove model optimality.
            continue
        used_resources = {}
        for key in running:
            for resource, units in demands[key].items():
                used_resources[resource] = used_resources.get(resource, 0) + units
        used_memory = {}
        for material in live:
            location, size = allocation_model[material]
            used_memory[location] = used_memory.get(location, 0) + size
        successors = []
        for key in ids:
            if key in completed or key in running or not dependencies[key].issubset(completed):
                continue
            if any(used_resources.get(r, 0) + units > capacities[r] for r, units in demands[key].items()):
                continue
            extra = {}
            for material in allocation_writes[key]:
                location, size = allocation_model[material]
                extra[location] = extra.get(location, 0) + size
            if any(used_memory.get(pool, 0) + size > memory_caps[pool] for pool, size in extra.items()):
                continue
            next_live = live | set(allocation_writes[key])
            next_starts = {**starts, key: time}
            if durations[key] == 0:
                next_live -= set(release_reads[key])
                successors.append((time, completed | {key}, running, next_live, next_starts))
            else:
                successors.append((time, completed, {**running, key: time + durations[key]}, next_live, next_starts))
        # Deliberate idling is necessary for exact search. A ready allocation or
        # executor task may otherwise block a shorter critical chain later.
        if running:
            next_time = min(running.values())
            finishing = {key for key, end in running.items() if end == next_time}
            next_live = set(live)
            for key in finishing:
                next_live -= set(release_reads[key])
            successors.append((next_time, completed | finishing,
                               {key: end for key, end in running.items() if key not in finishing},
                               next_live, starts))
        frontier.extend(reversed(successors))
    result.update(states_explored=explored,
                  search_budget_exhausted=exhausted,
                  status="SEARCH_BUDGET_EXHAUSTED" if exhausted else
                         "OPTIMAL_WITHIN_MODEL" if best_time is not None else "INFEASIBLE_WITHIN_MODEL",
                  optimality="NOT_ESTABLISHED" if exhausted else
                             "MODEL_OPTIMUM_ESTABLISHED" if best_time is not None else "INFEASIBILITY_ESTABLISHED")
    if best_time is None:
        return result
    if not exhausted:
        result["lower_bound_ticks"] = best_time
    timed = [{"task": key, "kind": tasks[key].kind.value, "start_ticks": best_starts[key],
              "end_ticks": best_starts[key] + durations[key], "resources": demands[key]}
             for key in sorted(ids, key=lambda key: (best_starts[key], key))]
    release_task = {material: key for key in ids for material in release_reads[key]}
    lifetimes = []
    for material, (pool, size) in allocation_model.items():
        allocation = allocation_records[material].allocated_by
        release = release_task.get(material)
        lifetimes.append({"storage_slot" if versioned else "materialization": material,
                          "memory_pool": pool, "bytes": size,
                          "start_ticks": best_starts[allocation],
                          "end_ticks": best_starts[release] + durations[release] if release else best_time,
                          "retained_through_makespan": release is None})
    gap = best_time - result["lower_bound_ticks"]
    result.update(solution={"makespan_ticks": best_time, "tasks": timed,
                            "allocation_lifetimes": lifetimes}, gap_ticks=gap,
                  relative_gap=gap / best_time if best_time else 0.0)
    return result


def task_implementation_contract(schedule, task_id) -> dict:
    """Bind a choice to exact work/effects/dependencies and unchanged storage."""
    validate_schedule(schedule)
    task = next((task for task in schedule.tasks if task.id == task_id), None)
    if task is None:
        raise ValueError('unknown task ID')
    return _contract_reader(schedule)(task)


def _contract_reader(schedule):
    # Keep declaration order without walking unrelated graph values per task.
    materials_by_id = {record.id: (position, record) for position, record in enumerate(schedule.materializations)}
    values_by_id = {record.id: (position, record) for position, record in enumerate(schedule.values)}

    def contract(task):
        materials = [record for _, record in sorted(materials_by_id[key] for key in set(task.reads + task.writes))]
        value_ids = {material.value for material in materials}
        values = [record for _, record in sorted(values_by_id[key] for key in value_ids)]
        return dict(schema='scar.task-implementation-contract', schema_version=1,
                    region_id=schedule.region_id, call_scope=schedule.call_scope,
                    task=encode(task), materializations=encode(materials), values=encode(values),
                    loop_axes=encode(schedule.loop_axes))

    return contract


def solve_implementation_choices(schedule, *, task_options, resource_capacities,
                                 memory_capacities=None, materialization_sizes=None,
                                 memory_locations=None, model_instantiation=None,
                                 search_budget=100000) -> dict:
    """Search a finite, sourced option domain plus its resource-task schedules.

    Each option has id, duration_ticks, resource_demands, contract_sha256,
    cost_evidence, legality_conditions and legality_evidence. The cost evidence
    must reference matching integer ticks. At least one legality record must
    bind the exact contract SHA, resource_demands and nonempty conditions.
    Conditions are retained external/model obligations, not global proofs.

    Materialization locations, memory sizes/budgets, task data and dependencies
    remain fixed. No COPY or other task is inserted. The global search budget
    counts choice branches plus fixed-solver states. Unknown options cannot
    yield full-domain optimality, even when a known-subset incumbent is found.
    """
    from .cost_model import _pointer

    graph = _prepare_fixed_schedule(schedule)
    if type(search_budget) is not int or search_budget < 1:
        raise ValueError('search_budget must be a positive integer')
    tasks, ids = graph.tasks, graph.task_ids
    contract_for = _contract_reader(schedule)
    frontiers, model_issues, cache = [], [], {}
    if not isinstance(task_options, Mapping):
        task_options = {}
        model_issues.append(dict(field='task_options', reason='explicit finite option mapping required'))
    if not isinstance(resource_capacities, Mapping):
        resource_capacities = {}
        model_issues.append(dict(field='resource_capacities', reason='explicit resource capacities required'))
    for resource, capacity in resource_capacities.items():
        if not isinstance(resource, str) or not resource or type(capacity) is not int or capacity < 1:
            model_issues.append(dict(field='resource_capacities.' + str(resource), reason='positive integer capacity required'))
    for key in task_options:
        if key not in tasks:
            model_issues.append(dict(field='task_options.' + str(key), reason='option domain refers to an absent task'))

    def evidence(records, matcher):
        if not isinstance(records, (tuple, list)) or not records:
            return False
        matched = False
        for record in records:
            reference = decode(ProviderFactReference, record)
            artifact = reference.artifact
            key = (artifact.uri, artifact.sha256, artifact.size_bytes, artifact.schema)
            if key not in cache:
                path = Path(artifact.uri)
                if not path.is_absolute():
                    raise ValueError('absolute retained artifact required')
                raw = path.read_bytes()
                if len(raw) != artifact.size_bytes or sha256(raw).hexdigest() != artifact.sha256:
                    raise ValueError('retained option artifact SHA/size differs')
                data = loads(raw)
                if isinstance(data, dict) and data.get('schema') != artifact.schema:
                    raise ValueError('option artifact schema differs')
                cache[key] = data
            matched = matcher(_pointer(cache[key], reference.locator)) or matched
        return matched

    choices, incomplete = {}, set()
    allowed = {'id', 'duration_ticks', 'resource_demands', 'contract_sha256',
               'cost_evidence', 'legality_conditions', 'legality_evidence'}
    for key in ids:
        options = task_options.get(key)
        choices[key] = []
        if not isinstance(options, (tuple, list)) or not options:
            frontiers.append(dict(task=key, field='task_options.' + key, reason='nonempty finite option domain required'))
            incomplete.add(key)
            continue
        seen = set()
        contract_hash = semantic_sha256(contract_for(tasks[key]))
        for index, option in enumerate(options):
            field = f'task_options.{key}.{index}'
            reasons = []
            if not isinstance(option, Mapping):
                frontiers.append(dict(task=key, field=field, reason='option record required'))
                incomplete.add(key)
                continue
            option = deepcopy(dict(option))
            identifier = option.get('id')
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                reasons.append('nonempty unique option ID required')
            elif identifier:
                seen.add(identifier)
            if set(option) - allowed:
                reasons.append('options cannot add data, locations, effects, tasks or other unsupported constraints')
            duration = option.get('duration_ticks')
            if type(duration) is not int or duration < 0:
                reasons.append('explicit nonnegative duration_ticks required; unknown is not zero')
            demands = option.get('resource_demands')
            if not isinstance(demands, Mapping) or any(not isinstance(r, str) or not r or type(u) is not int or u < 1
                                                        for r, u in demands.items()):
                reasons.append('explicit resource-to-positive-units demands required')
            elif any(r not in resource_capacities for r in demands):
                reasons.append('each demanded resource needs an explicit capacity')
            if option.get('contract_sha256') != contract_hash:
                reasons.append('option must preserve the exact original task/storage/transport contract')
            attrs = tasks[key].attrs
            if 'cpu_threads' in attrs or 'executor_id' in attrs:
                executor, workers = attrs.get('executor_id'), attrs.get('cpu_threads')
                if (not isinstance(executor, str) or not executor or type(workers) is not int
                        or workers < 1 or not isinstance(demands, Mapping)
                        or dict(demands) != {executor: workers}):
                    reasons.append('bound CPU task attributes are fixed; resource alternatives require an explicitly unbound semantic task')
            for reference in tasks[key].resource_refs:
                if reference.get('kind') == 'cpu_executor_binding':
                    cpus = reference.get('cpu_ids')
                    executor, workers = attrs.get('executor_id'), attrs.get('cpu_threads')
                    refs = reference.get('refs', ())
                    declared_ids = [ref.get('resource', {}).get('wire') for ref in refs if isinstance(ref, Mapping)]
                    if (not isinstance(cpus, list) or not cpus or len(cpus) != workers
                            or executor not in declared_ids or resource_capacities.get(executor) != len(cpus)):
                        reasons.append('bound CPU resource references/capacity must preserve the recorded CPU allocation')
            conditions = option.get('legality_conditions')
            if not isinstance(conditions, list) or not conditions or any(not isinstance(c, str) or not c.strip() for c in conditions):
                reasons.append('explicit nonempty legality conditions required')
            try:
                def cost_matches(record):
                    value = record.get('duration_ticks') if isinstance(record, Mapping) else record
                    return type(value) is int and type(duration) is int and value == duration

                if not evidence(option.get('cost_evidence'), cost_matches):
                    reasons.append('matching retained duration-model evidence required')
                expected = dict(contract_sha256=contract_hash, resource_demands=demands, conditions=conditions)
                def legality_matches(record):
                    return (isinstance(record, Mapping) and all(name in record for name in expected)
                            and semantic_sha256({name: record[name] for name in expected}) == semantic_sha256(expected))

                if not evidence(option.get('legality_evidence'), legality_matches):
                    reasons.append('matching retained contract/resource/conditions evidence required')
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                reasons.append('invalid retained option evidence: ' + str(error))
            if reasons:
                incomplete.add(key)
                frontiers.extend(dict(task=key, option=identifier, field=field, reason=reason) for reason in reasons)
            else:
                choices[key].append(option)
        choices[key].sort(key=lambda option: option['id'])

    # A zero here is a mathematical nonnegativity relaxation for an unknown
    # frontier, never an imputed cost and never submitted to the fixed solver.
    relaxed = {key: (0 if key in incomplete or not choices[key] else
                     min(option['duration_ticks'] for option in choices[key])) for key in ids}
    global_lower = _critical_path_bound(graph, relaxed)
    constraints = dict(resource_capacities=deepcopy(dict(resource_capacities)),
                       memory_capacities=deepcopy(memory_capacities),
                       materialization_sizes=deepcopy(materialization_sizes),
                       memory_locations=deepcopy(memory_locations),
                       model_instantiation=deepcopy(model_instantiation))
    try:
        domain_hash = semantic_sha256(task_options)
        model_hash = semantic_sha256(dict(schedule=graph.document,
                                          option_domain=task_options, fixed_constraints=constraints))
    except (TypeError, ValueError):
        domain_hash, model_hash = None, None
        model_issues.append(dict(field='option_domain', reason='finite semantic JSON required'))
    result = dict(schema='scar.implementation-choice-solver', schema_version=1,
                  region_id=schedule.region_id, schedule_sha256=graph.sha256,
                  status='UNRESOLVED_OPTION_DOMAIN', optimality='NOT_ESTABLISHED',
                  objective='MINIMIZE_MAKESPAN', physical_optimality_proven=False,
                  time_unit='CALLER_DECLARED_COMMON_TICKS',
                  model_assumptions=['ALL_OPTION_COSTS_USE_THE_SAME_DECLARED_TICK_BASIS',
                                     'NONPREEMPTIVE_TASKS_AND_FIXED_MATERIALIZATION_LOCATIONS',
                                     'ONLY_SOURCED_DECLARED_LEGALITY_CONDITIONS_ARE_ASSUMED',
                                     'RESOURCE_CAPACITIES_AND_ALLOCATION_LIFETIMES_FOLLOW_THE_FIXED_SOLVER_MODEL'],
                  option_domain_sha256=domain_hash, global_model_sha256=model_hash,
                  option_domain=deepcopy(dict(task_options)), fixed_resource_constraints=constraints,
                  target_program_executed=False, deployment_authorized=False,
                  scope='FINITE_DECLARED_IMPLEMENTATION_RESOURCE_CHOICES_FIXED_MATERIALIZATIONS',
                  materialization_placement_optimized=False, implicit_transfers_inserted=False,
                  execution_plan_rewritten=False, backend_lowering_performed=False,
                  chosen_option_lowering_requires='TYPED_BACKEND_LEGALITY_AND_MATERIALIZATION_LOCATION_ADMISSION',
                  execution_unresolved=deepcopy(schedule.unresolved),
                  frontier=frontiers, unresolved=model_issues, chosen_option_ids=None,
                  chosen_options=None, fixed_solver_result=None, solution=None,
                  global_lower_bound_ticks=global_lower, gap_ticks=None, relative_gap=None,
                  lower_bound_basis='TASK_DAG_MINIMUM_DURATIONS_WITH_EXPLICIT_NONNEGATIVITY_RELAXATION_FOR_FRONTIERS_NOT_COST_IMPUTATION',
                  relaxed_duration_bounds=relaxed, states_explored=0, choice_branches=0,
                  search_budget=search_budget, search_budget_exhausted=False,
                  external_assumptions=deepcopy(schedule.proof.get('external_assumptions', [])),
                  optimality_scope='GIVEN_FINITE_OPTION_COST_LEGALITY_RESOURCE_AND_LIFETIME_MODEL_ONLY',
                  legality_authority='RETAINED_DECLARATIONS_UNDER_EXPLICIT_CONDITIONS_NOT_GLOBAL_EFFECT_PROOFS')
    if model_issues or any(not choices[key] for key in ids):
        if model_issues:
            result['status'] = 'UNRESOLVED_MODEL'
        return result
    best, used, branches, exhausted = None, 0, 0, False
    for combo in product(*(choices[key] for key in ids)):
        if used >= search_budget:
            exhausted = True
            break
        used += 1
        branches += 1
        selected = dict(zip(ids, combo))
        durations = {key: option['duration_ticks'] for key, option in selected.items()}
        if best is not None and _critical_path_bound(graph, durations) >= best['solution']['makespan_ticks']:
            continue
        if used >= search_budget:
            exhausted = True
            break
        fixed = _solve_fixed_schedule(
            graph, task_costs=durations,
            task_placements={key: option['resource_demands'] for key, option in selected.items()},
            resource_capacities=resource_capacities, memory_capacities=memory_capacities,
            materialization_sizes=materialization_sizes, memory_locations=memory_locations,
            model_instantiation=model_instantiation, search_budget=search_budget - used)
        used += fixed['states_explored']
        if fixed['status'] == 'UNRESOLVED':
            result['unresolved'] = fixed['unresolved']
            break
        if fixed['solution'] is not None and (best is None or fixed['solution']['makespan_ticks'] < best['solution']['makespan_ticks']):
            best = fixed
            result.update(chosen_option_ids={key: option['id'] for key, option in selected.items()},
                          chosen_options=deepcopy(selected))
        if fixed['status'] == 'SEARCH_BUDGET_EXHAUSTED':
            exhausted = True
            break
    result.update(states_explored=used, choice_branches=branches, search_budget_exhausted=exhausted)
    if best is not None:
        result.update(fixed_solver_result=best, solution=best['solution'])
    if result['unresolved']:
        result['status'] = 'UNRESOLVED_MODEL'
    elif frontiers:
        result['status'] = 'UNRESOLVED_OPTION_DOMAIN'
    elif exhausted:
        result['status'] = 'SEARCH_BUDGET_EXHAUSTED'
    elif best is None:
        result.update(status='INFEASIBLE_WITHIN_MODEL', optimality='INFEASIBILITY_ESTABLISHED')
    else:
        global_lower = best['solution']['makespan_ticks']
        result.update(status='OPTIMAL_WITHIN_FINITE_MODEL', optimality='FINITE_MODEL_OPTIMUM_ESTABLISHED')
    result['global_lower_bound_ticks'] = global_lower
    if best is not None:
        gap = best['solution']['makespan_ticks'] - global_lower
        result.update(gap_ticks=gap, relative_gap=gap / best['solution']['makespan_ticks'] if best['solution']['makespan_ticks'] else 0.0)
    return result


__all__ = [
    "solve_resource_schedule", "solve_implementation_choices", "task_implementation_contract",
    "hardware_model_skeleton", "semantic_sha256",
]
