"""One shared temporal recipe consumed by costing and existing CPU lowering.

The semantic boundary stays resource-independent. Allocation establishes
storage, the complete recurrence establishes final readiness, and export keeps
the fresh final alive. No task or expression is expanded by the trip count.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from scar.compiler.loop_admission import admit_loop, check_loop_sources
from scar.compiler.loop_admission import document as loop_document, loop_from_dict
from scar.compiler.model import document as plan_document, plan_from_dict
from scar.compiler.recipe_boundary import build_loop_boundary
from scar.ir.execution_recipe import (artifact_reference, make_recipe, read_artifact,
    read_recipe, validate_recipe, write_recipe)
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from .hardware import _target_materials
from .loop_layout import verify_loop_layout
from .resource_solver import semantic_sha256
from .schedule import LogicalValue, Materialization, ResourceSchedule, Task, TaskKind

PROFILE = 'cpu-temporal-loop-v1'
_ROLES = ('loop', 'strategy', 'execution_plan')
_EXCLUDES = ['EXTERNAL_INPUTS', 'ALLOCATOR_METADATA_AND_ALIGNMENT_OVERHEAD', 'THREAD_SCRATCH', 'PROCESS_RSS']


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return artifact_reference(path, value['schema'])


def _parts(recipe):
    artifacts = recipe['artifacts']
    if any(role not in artifacts for role in (*_ROLES, 'target_snapshot', 'target_inventory')):
        raise ValueError('complete temporal loop and original hardware artifacts required')
    allowed = {*_ROLES, 'layout', 'target_snapshot', 'target_inventory', 'cpu_hierarchy'}
    if any(role not in allowed and not role.startswith(('source_python:', 'source_artifact:', 'target_table:'))
           for role in artifacts):
        raise ValueError('unknown temporal recipe artifact roles')
    loop = loop_from_dict(read_artifact(artifacts['loop']))
    strategy = read_artifact(artifacts['strategy'])
    plan = plan_from_dict(read_artifact(artifacts['execution_plan']))
    layout = read_artifact(artifacts['layout']) if 'layout' in artifacts else None
    strategy, plan = admit_loop(loop, strategy, plan); check_loop_sources(loop)
    if layout is not None:
        layout = verify_loop_layout(loop, strategy, plan, layout)
        if layout['status'] != 'PLANNED_UNDER_MODEL' or layout['selected'] is None:
            raise ValueError('temporal recipe requires an admitted SIMD layout or no layout')
    return loop, strategy, plan, layout


def _target(plan):
    binding = plan.resource_binding
    graph, snapshot, resources, connections = _target_materials(binding.target_path)
    if snapshot.sha256 != binding.target_sha256:
        raise ValueError('recipe target snapshot differs from original selected binding')
    artifacts = dict(target_snapshot=encode(snapshot),
        target_inventory=artifact_reference(graph.report['inventory']['uri'], 'scar.target-inventory'))
    tables = {**graph.report['nodes'], 'connections':graph.report['connections'], 'paths':graph.report['paths']}
    for name, receipt in tables.items():
        artifacts['target_table:' + name] = artifact_reference(graph.path / receipt['path'], 'scar.target-table-jsonl-gzip')
    return resources, connections, snapshot, artifacts


def _source_refs(loop, artifacts):
    result = []
    sources = (loop.source, loop.body.source, *(call.callee.source for call in loop.body.calls))
    seen = set()
    for source in sources:
        if source.path in seen: continue
        seen.add(source.path)
        matches = [role for role, ref in artifacts.items() if ref['uri'] == source.path and
                   ref['sha256'] == source.sha256 and role.startswith(('source_python:', 'source_artifact:'))]
        if not matches: raise ValueError('original loop/body/leaf source artifact receipt missing')
        result.append(sorted(matches)[0])
    return result


def _assemble(loop, strategy, plan, layout, artifacts):
    from scar.codegen.loop import describe_capability

    boundary = build_loop_boundary(loop, strategy, plan)
    inventory, connections, snapshot, target_artifacts = _target(plan)
    if any(artifacts.get(role) != ref for role, ref in target_artifacts.items()):
        raise ValueError('complete retained target inventory/table references differ')
    if set(role for role in artifacts if role.startswith('target_table:')) != set(role for role in target_artifacts if role.startswith('target_table:')):
        raise ValueError('unsupported additional hardware table reference')
    source_roles = _source_refs(loop, artifacts)
    if 'cpu_hierarchy' in artifacts:
        hierarchy = read_artifact(artifacts['cpu_hierarchy'])
        if hierarchy.get('schema') != 'scar.cpu-hierarchy' or hierarchy.get('target_snapshot') != encode(snapshot):
            raise ValueError('cache hierarchy must reference this original target snapshot')
    binding = plan.resource_binding
    refs = [encode(H.TargetResourceRef(snapshot, resource.id)) for resource in inventory]
    hosts = [encode(H.TargetResourceRef(snapshot, resource.id)) for resource in inventory
             if isinstance(resource, H.MemoryDomain) and resource.kind is H.MemoryKind.HOST]
    executor = next(resource for resource in inventory if str(resource.id) == binding.executor_id)
    selected = encode(H.TargetResourceRef(snapshot, executor.id))
    shape = list(boundary['input']['shape']); size = 8 * shape[0] * shape[1]
    budget = plan.cost['memory_budget_bytes']
    resources = dict(schema='scar.cpu-temporal-recipe-resources', schema_version=1,
        inventory=dict(snapshot_role='target_snapshot', inventory_role='target_inventory', all_resources=refs,
            topology=[encode(connection) for connection in connections], original_program_use_does_not_filter_resources=True),
        selected_cpu=dict(executor=selected, cpu_ids=list(binding.cpu_ids), cpu_threads=plan.cpu_threads,
            binding_ref=dict(artifact_role='execution_plan', pointer='#/resource_binding')),
        host_allocation=dict(location='host-address-space:$CALL_PROCESS', memory_policy='UNBOUND',
            candidates=hosts, planning_domain_id=binding.memory_domain_id, physical_numa_residency_proven=False),
        memory=dict(required_fresh_final_bytes=size, workspace_bytes=0, explicit_budget_bytes=budget,
            budget_scope='INCREMENTAL_FRESH_FINAL_ARRAY_PAYLOAD', budget_excludes=list(_EXCLUDES),
            capacity_is_available_budget=False, allocation_success_is_runtime_guard=True),
        cache_hierarchy=dict(artifact_role='cpu_hierarchy' if 'cpu_hierarchy' in artifacts else None,
            allocatable=False, allocation_budget_bytes=None, effective_available_bytes=None,
            interpreted_as='OPTIONAL_TOPOLOGY_ONLY_NOT_CACHE_MALLOC_OR_DRAM_TRAFFIC'),
        unselected_resources_retained=True, unselected_devices_have_no_lowering_claim=True)
    control = dict(schema='scar.cpu-temporal-recipe-control', schema_version=1,
        spatial_axes=_copy([asdict(axis) for axis in plan.schedule]),
        partition='OPENMP_STATIC_DISJOINT_ROWS' if plan.cpu_threads > 1 else 'SERIAL_ROWS',
        temporal=dict(axis='time', count=boundary['iteration']['count'], order='STRICT_SAME_COORDINATE_RANGE',
            carry=boundary['state']['id'], entry=boundary['input']['id'], final=boundary['output']['id'],
            unrolled=False, shared_body_instances=1, body_artifact_role='execution_plan',
            semantic_state_ref='#/semantic_boundary/state', semantic_iteration_ref='#/semantic_boundary/iteration'),
        layout_artifact_role='layout' if layout is not None else None,
        no_global_iteration_barrier_inferred=True)
    inputs = dict(loop='loop', strategy='strategy', plan='execution_plan')
    if layout is not None: inputs['layout'] = 'layout'
    implementation = dict(family='CPP_SYNCHRONOUS_CPU_TEMPORAL_LOOP', backend_inputs=inputs,
        operator_abi_version=plan.operator_abi_version,
        document_sha256={role:semantic_sha256(read_artifact(artifacts[role])) for role in inputs.values()},
        semantic_materials=dict(loop='loop', strategy='strategy', canonical_execution_plan='execution_plan'),
        canonical_plan_alias_scope='IDENTICAL_REFERENCED_REGION_AND_EFFECTS_SUBTREES_ONLY',
        source_artifact_roles=source_roles,
        numerical_contract=dict(boundary_ref='#/semantic_boundary/Q', status='EXISTING_GUARDED_STRICT_CPU_CONTRACT',
            source='STAGE1_BOUNDARY_Q_AND_ORIGINAL_LOOP_STRATEGY', device_fp_relaxation_authorized=False),
        backend_capability='EXACT_ELEMENTWISE_TEMPORAL_FUSION',
        backend_capability_record=describe_capability(plan, layout), program_body_copies=1)
    input_id, output_id = boundary['input']['id'], boundary['output']['id']
    input_node = next(node.id for node in plan.region.nodes if node.op == 'input')
    value_attrs = dict(dtype='float64', shape=shape, call_instance_identity=True)
    values = (LogicalValue(input_id, source_nodes=(input_node,), attrs=dict(value_attrs,
        semantic_ref='#/semantic_boundary/input', identity='$CALL_INSTANCE:' + input_id)),
        LogicalValue(output_id, producer='compute:temporal_loop', escape=True, source_nodes=(plan.region.output,),
            attrs=dict(value_attrs, semantic_ref='#/semantic_boundary/output', fresh_each_call=True)))
    material_refs = ({'kind':'original_target_snapshot', 'artifact_role':'target_snapshot'},)
    materials = (Materialization('host:state:entry', input_id, 'host-address-space:$CALL_PROCESS', material_refs,
        external_ready=True, ownership='EXTERNAL', bytes_expr='8 * state_elements'),
        Materialization('host:state:return', output_id, 'host-address-space:$CALL_PROCESS', material_refs,
            ready_task='compute:temporal_loop', allocated_by='allocate:final', bytes_expr='8 * output_elements'))
    tasks = (Task('allocate:final', TaskKind.ALLOCATION, writes=(materials[1].id,), attrs=dict(
        bytes=size, memory_budget_bytes=budget, establishes_value_ready=False, memory_policy='UNBOUND')),
        Task('compute:temporal_loop', TaskKind.COMPUTE, (materials[0].id,), (materials[1].id,), ('allocate:final',),
            resource_refs=({'kind':'selected_cpu', 'target':selected, 'cpu_ids':list(binding.cpu_ids)},),
            attrs=dict(profile=PROFILE, cpu_threads=plan.cpu_threads, executor_id=binding.executor_id,
                implementation_ref='#/implementation', body_artifact_role='execution_plan',
                body_sha256=implementation['document_sha256']['execution_plan'],
                temporal_control_ref='#/control/temporal', trip_count=boundary['iteration']['count'],
                strict_time_carry=True, shared_body_instances=1, unrolled=False, final_ready_only_after_complete_call=True)),
        Task('export:final', TaskKind.EXPORT, (materials[1].id,), depends_on=('compute:temporal_loop',),
            attrs=dict(ownership='CALLER', escape=True, release=False)))
    graph = ResourceSchedule(loop.id, values, materials, tasks, loop_axes=plan.schedule, executable=True,
        proof=dict(profile=PROFILE, semantic_boundary_id=boundary['id'], scope='ONE_COMPLETE_TEMPORAL_CALL',
            whole_program_schedule_optimized=False, external_assumptions=boundary['external_assumptions']))
    obligations = [dict(id='semantic_contract', status='PROVEN', detail='Closed profile relation replayed under its retained guards; no global effect or purity proof.'),
        dict(id='numerical_contract', status='RUNTIME_GUARD', detail='Preserve original f64 DAG/order/count; withhold final on finite/fenv failure and fall back once.'),
        dict(id='source_and_callees', status='RUNTIME_GUARD', detail='Original source snapshots and live nested callable/range identities must still match.'),
        dict(id='cpu_and_payload', status='RUNTIME_GUARD', detail='Bound CPUs remain available; fresh output allocation succeeds within the explicit payload budget.'),
        dict(id='backend_abi_isa', status='LOWERING_CHECK', detail='Existing CPU backend must admit exact ABI, compiler/ISA/layout and recipe before use.')]
    obligations += [dict(id='external:' + str(index), status='EXTERNAL_ASSUMPTION', detail=item)
                    for index,item in enumerate(boundary['external_assumptions'])]
    return make_recipe(PROFILE, semantic_boundary=boundary, control=control, task_graph=graph,
        resources=resources, implementation=implementation, artifacts=artifacts, obligations=obligations)


def make_loop_recipe(loop, strategy, plan, *, layout=None, source_artifacts=(), cpu_hierarchy=None, out):
    """Persist source components once, then create the common execution recipe."""
    strategy, plan = admit_loop(loop, strategy, plan); check_loop_sources(loop)
    if layout is not None: layout = verify_loop_layout(loop, strategy, plan, layout)
    out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh loop recipe directory required')
    artifacts = dict(loop=_save(out/'components/loop.json', loop_document(loop)),
        strategy=_save(out/'components/strategy.json', strategy),
        execution_plan=_save(out/'components/execution-plan.json', plan_document(plan)))
    if layout is not None: artifacts['layout'] = _save(out/'components/layout.json', layout)
    extras = source_artifacts.values() if isinstance(source_artifacts, dict) else source_artifacts
    for index, reference in enumerate(extras):
        reference = encode(reference); read_artifact(reference, json_document=False)
        artifacts['source_artifact:' + str(index)] = reference
    for source in (loop.source, loop.body.source, *(call.callee.source for call in loop.body.calls)):
        if any(ref['uri'] == source.path for ref in artifacts.values()): continue
        role = 'source_python:' + str(len([key for key in artifacts if key.startswith('source_python:')]))
        reference = artifact_reference(source.path, 'scar.python-source')
        if reference['sha256'] != source.sha256: raise ValueError('original source snapshot differs')
        artifacts[role] = reference
    _, _, _, target_artifacts = _target(plan); artifacts.update(target_artifacts)
    if cpu_hierarchy is not None:
        artifacts['cpu_hierarchy'] = (artifact_reference(cpu_hierarchy, 'scar.cpu-hierarchy')
            if isinstance(cpu_hierarchy, (str, Path)) else encode(cpu_hierarchy))
    recipe = _assemble(loop, strategy, plan, layout, artifacts)
    write_recipe(recipe, out/'recipe.json')
    return recipe


def verify_loop_recipe(value):
    recipe = read_recipe(value) if isinstance(value, (str, Path)) else validate_recipe(value)
    if recipe['profile'] != PROFILE: raise ValueError('supported CPU temporal loop recipe profile required')
    parts = _parts(recipe)
    expected = _assemble(*parts, recipe['artifacts'])
    if semantic_sha256(expected) != semantic_sha256(recipe):
        raise ValueError('temporal loop recipe differs from complete source/control/resource replay')
    return recipe


def unpack_loop_recipe(value):
    """Return exactly the four existing backend inputs after profile replay."""
    return _parts(verify_loop_recipe(value))


def estimate_loop_recipe(value, worker_profile=None):
    """Independent recipe-keyed estimate; whole-call cost is not task decomposition."""
    recipe = verify_loop_recipe(value); loop, strategy, plan, layout = _parts(recipe)
    size = recipe['resources']['memory']['required_fresh_final_bytes']
    prediction = None; unresolved = []
    if worker_profile is None:
        unresolved.append(dict(field='worker_profile', reason='temporal whole-call cost is unknown; no single-step estimate or zero substituted'))
    elif layout is None:
        unresolved.append(dict(field='layout', reason='current temporal worker cost profile requires its exact SIMD layout'))
    else:
        from scar.optimization.scope_accounting import _profile
        from .loop_workers import predict_loop_worker_costs
        profile, _ = _profile(worker_profile)
        prediction = predict_loop_worker_costs(loop, strategy, plan, layout, profile)
        unresolved.extend(prediction['unresolved'])
    known = prediction is not None and prediction['prediction_available']
    return dict(schema='scar.loop-recipe-estimate', schema_version=1, recipe_id=recipe['recipe_id'],
        objective='COMPOSE_TOWARD_FULL_PROGRAM_MAKESPAN', scope='ONE_COMPLETE_TEMPORAL_CALL_NOT_FULL_PROGRAM_OPTIMUM',
        payload=dict(status='EXACT_UNDER_SHAPE_CONTRACT', external_input_bytes=size, fresh_final_bytes=size,
            workspace_bytes=0, incremental_payload_bytes=size, explicit_budget_bytes=plan.cost['memory_budget_bytes'],
            budget_excludes=list(_EXCLUDES)),
        timing=dict(status='ESTIMATED' if known else 'UNKNOWN', scope='WHOLE_CALL',
            duration_ticks=prediction['duration_ticks'] if known else None,
            tick_seconds=prediction['tick_seconds'] if known else None,
            per_task={task['id']:dict(status='UNKNOWN', duration_ticks=None) for task in recipe['task_graph']['tasks']},
            decomposition_inferred=False),
        prediction=prediction, unresolved=unresolved,
        task_timing_frontier='Whole-call alloc/guards/sync cost does not establish separate ALLOCATION/COMPUTE/EXPORT durations.',
        full_program_makespan_ticks=None, physical_optimality_proven=False, recipe_modified=False,
        target_executed=False, deployment_authorized=False)


__all__ = ['PROFILE', 'make_loop_recipe', 'verify_loop_recipe', 'unpack_loop_recipe', 'estimate_loop_recipe']
