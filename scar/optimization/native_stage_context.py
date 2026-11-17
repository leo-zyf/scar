"""Source-derived contexts for contiguous groups in the original native frame.

The existing recipe owns values, physical materializations and root epochs.
This projection adds no SSA, backend, execution history or duration model.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.native-stage-context'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _materializations(schedule):
    values = {row.id:row for row in schedule.values}
    result = {}
    for index, row in enumerate(schedule.materializations):
        value = values[row.value]
        epoch = {name:deepcopy(value.attrs[name]) for name in ('input_id', 'epoch', 'epoch_scope',
            'original_epoch_before', 'original_epoch_after', 'source_original_epoch_before',
            'source_original_epoch_after') if name in value.attrs}
        result[row.id] = dict(id=row.id, logical_value=row.value, call_scope=value.call_scope,
            slot_id=row.region.slot, offset_bytes=row.region.offset_bytes, logical_span_bytes=row.region.nbytes,
            storage_version=row.version, ready_task=row.ready_task, external_ready=row.external_ready,
            local_content_epoch=epoch if epoch else None,
            local_content_epoch_status='SOURCE_ROOT_EPOCH_METADATA' if epoch else 'NOT_A_SOURCE_ROOT_EPOCH',
            logical_value_attrs=deepcopy(value.attrs), graph_pointer='#/versioned_schedule/materializations/'+str(index),
            storage_version_is_source_epoch=False, content_identity_from_storage_id=False)
    return result


def _accesses(stage, task, materials):
    result = []
    for mode, names in (('READ', task.reads), ('WRITE', task.writes)):
        for name in names:
            material = materials[name]
            preservation = (stage['kind'] == 'COMMIT' and mode == 'READ'
                and material['slot_id'] == 'external:'+stage['target'])
            roi = stage['kind'] in ('WRITE_SHADOW', 'COMMIT')
            footprint = dict(kind='LOGICAL_MATERIALIZATION_BOUND_NOT_ELEMENT_TRAFFIC',
                span_upper_bound_bytes=material['logical_span_bytes'])
            if roi:
                footprint = dict(kind='ORIGINAL_WRITE_ROI', normalized_bounds=deepcopy(stage['normalized_bounds']),
                    shape=deepcopy(stage['shape']), payload_bytes=8*stage['shape'][0]*stage['shape'][1])
            if preservation:
                footprint = dict(kind='PRESERVED_EXTERNAL_COMPLEMENT_NOT_AN_EMITTED_READ')
            result.append(dict(mode=mode, materialization=name, slot_id=material['slot_id'],
                logical_value=material['logical_value'], storage_version=material['storage_version'],
                emitted_element_access=not preservation, footprint=footprint,
                machine_access_count=None, dram_bytes=None))
    return result


def _project(schedule, plan, region, observer, view):
    stages = {row['id']:row for row in plan['stages']}
    tasks = {row.id:row for row in schedule.tasks}
    positions = {row.id:index for index,row in enumerate(schedule.tasks)}
    materials = _materializations(schedule)
    consumers = defaultdict(list)
    for task in schedule.tasks:
        for name in task.reads:
            consumers[name].append(task.id)
    node_positions = {node.id:index for index,node in enumerate(region.nodes)}
    nodes = {node.id:node for node in region.nodes}
    inputs = {row.id:(index,row) for index,row in enumerate(region.inputs)}
    scalar_producers = {row['root_expr']:row['id'] for row in plan['stages'] if row['kind'] == 'SCALAR'}
    scalar_consumers = defaultdict(list)
    for stage in plan['stages']:
        for expression in stage['scalar_inputs']:
            _need(expression in scalar_producers, 'ready scalar has no original producer stage')
            scalar_consumers[expression].append(stage['id'])
    activities = defaultdict(list)
    if schedule.temporal is not None:
        for index,row in enumerate(schedule.temporal.activities):
            activities[row.task].append(dict(id=row.id, start=row.start, completion=row.completion,
                duration_basis=row.duration_basis.value,
                graph_pointer='#/versioned_schedule/temporal/activities/'+str(index)))

    def live(cursor, after=False):
        result = []
        for name, material in materials.items():
            ready = -1 if material['external_ready'] else positions[material['ready_task']]
            available = ready <= cursor if after else ready < cursor
            later = any(positions[user] > cursor if after else positions[user] >= cursor for user in consumers[name])
            if available and later:
                result.append(name)
        return result

    def ready_live(cursor, after=False):
        return [expression for expression,producer in scalar_producers.items()
            if (positions[producer] <= cursor if after else positions[producer] < cursor)
            and any(positions[user] > cursor if after else positions[user] >= cursor
                for user in scalar_consumers[expression])]

    def slot_contents(cursor, after=False):
        result = []
        for slot in schedule.storage_slots:
            available = [row for row in materials.values() if row['slot_id'] == slot.id and (
                row['external_ready'] or (positions[row['ready_task']] <= cursor if after
                    else positions[row['ready_task']] < cursor))]
            current = max(available, key=lambda row:row['storage_version']) if available else None
            result.append(dict(slot_id=slot.id, ownership=slot.ownership, capacity_bytes=slot.capacity_bytes,
                current_materialization=None if current is None else current['id'],
                current_logical_value=None if current is None else current['logical_value'],
                storage_version=None if current is None else current['storage_version'],
                live_content_required=current is not None and current['id'] in live(cursor, after),
                state='NO_EMITTED_MATERIALIZATION_READY' if current is None else 'LAST_SOURCE_EMITTED_WRITE',
                physical_cache_state='UNKNOWN', physical_numa_residency='UNKNOWN'))
        return result

    edges = []
    for stage in plan['stages']:
        task = tasks[stage['id']]
        for parent in stage['depends_on']:
            edges.append(dict(kind='ORIGINAL_STAGE_DEPENDENCY', before=parent, after=stage['id'], materialization=None))
        for material in task.reads:
            producer = materials[material]['ready_task']
            if producer is not None:
                edges.append(dict(kind='MATERIAL_READ_AFTER_WRITE', before=producer, after=stage['id'],
                    materialization=material, logical_value=materials[material]['logical_value']))
        for scalar in stage['scalar_inputs']:
            edges.append(dict(kind='READY_SCALAR_DEPENDENCY', before=scalar_producers[scalar], after=stage['id'],
                expression=scalar, materialization=None))
    by_slot = defaultdict(list)
    for row in materials.values():
        by_slot[row['slot_id']].append(row)
    for slot, versions in by_slot.items():
        versions.sort(key=lambda row:row['storage_version'])
        for old,new in zip(versions, versions[1:]):
            writer = new['ready_task']
            if writer not in stages:
                continue
            if old['ready_task'] in stages:
                edges.append(dict(kind='PHYSICAL_SLOT_OVERWRITE_ORDER_NOT_VALUE_FLOW', before=old['ready_task'],
                    after=writer, slot_id=slot, previous_materialization=old['id'], next_materialization=new['id'],
                    logical_values_equal=old['logical_value'] == new['logical_value'],
                    equality_basis='EXISTING_LOGICAL_VALUE_IDENTITY_NOT_SLOT_ID'))
            for reader in consumers[old['id']]:
                if reader in stages and reader != writer:
                    edges.append(dict(kind='READ_BEFORE_PHYSICAL_SLOT_OVERWRITE', before=reader, after=writer,
                        slot_id=slot, previous_materialization=old['id'], next_materialization=new['id']))
    first = {}
    stage_rows = []
    view_stages = {row['index']:row for row in view['stages']}
    for stage in plan['stages']:
        task = tasks[stage['id']]; cursor = positions[task.id]
        accesses = _accesses(stage, task, materials)
        for access in accesses:
            footprint = access['footprint']
            size = footprint.get('payload_bytes', footprint.get('span_upper_bound_bytes', 0))
            if access['emitted_element_access'] and size > 0 and access['slot_id'] not in first:
                first[access['slot_id']] = dict(stage_id=stage['id'], mode=access['mode'],
                    materialization=access['materialization'], source_event='FIRST_EMITTED_STAGE_PAYLOAD_ACCESS',
                    scope='STAGE_SEQUENCE_ONLY_HELPER_OR_ALLOCATOR_TOUCHES_MAY_PRECEDE',
                    physical_first_touch_observed=False, physical_page_placement='UNKNOWN')
        runtime = []
        for expression in stage['expr_ids']:
            node = nodes[expression]
            if node.op == 'input' and node.attrs.get('kind') == 'scalar':
                index, source = inputs[node.attrs['name']]
                runtime.append(dict(input_id=source.id, expression=expression, path=list(source.path),
                    source_ref=dict(artifact=view['region'], locator='#/inputs/'+str(index)),
                    value=None, integer_tag=None, value_status='CALLER_SUPPLIED_RUNTIME_VALUE'))
        source = view_stages.get(stage['index'])
        stage_rows.append(dict(id=stage['id'], index=stage['index'], kind=stage['kind'],
            original_write_barrier_epoch=stage['epoch'], target=stage['target'],
            source_action_ids=deepcopy(stage['source_action_ids']),
            source_stage_ref=dict(artifact=view['plan'], locator='#/stages/'+str(stage['index'])),
            source_body=None if source is None else dict(body_sha256=source['body_sha256'],
                line_spans=deepcopy(source['source_line_spans']), mapping_status=source['mapping_status']),
            source_body_status='OMITTED_BY_EXISTING_VIEW' if source is None else source['mapping_status'],
            predecessor_stage_ids=deepcopy(stage['depends_on']), reads=list(task.reads), writes=list(task.writes),
            live_before=live(cursor), live_after=live(cursor, True),
            live_through=[name for name in live(cursor) if name in live(cursor, True)],
            required_ready_scalars=deepcopy(stage['scalar_inputs']),
            scalar_live_before=ready_live(cursor), scalar_live_after=ready_live(cursor, True),
            produced_scalar=stage['root_expr'] if stage['kind'] == 'SCALAR' else None,
            runtime_scalar_inputs=runtime, array_reads=deepcopy(stage['array_reads']), emitted_accesses=accesses,
            semantic_time_activities=activities[stage['id']],
            cost_join_key=dict(stage_id=stage['id'], source_stage_ref=dict(artifact=view['plan'],
                locator='#/stages/'+str(stage['index'])), duration_basis='UNKNOWN_NO_COST_PROFILE_CONSUMED')))
    stage_map = {row['index']:row for row in stage_rows}
    groups = []
    for observed in observer['groups']:
        members = [stage_map[index] for index in observed['stage_indices']]
        member_ids = {row['id'] for row in members}
        begin, end = positions[members[0]['id']], positions[members[-1]['id']]
        read = list(dict.fromkeys(name for row in members for name in row['reads']))
        write = list(dict.fromkeys(name for row in members for name in row['writes']))
        incoming = [edge for edge in edges if edge['after'] in member_ids and edge['before'] not in member_ids]
        outgoing = [edge for edge in edges if edge['before'] in member_ids and edge['after'] not in member_ids]
        identity = semantic_sha256(dict(native=view['identity'], groups=observed['stage_indices'],
            observer_identity=observer['observer_identity']))
        groups.append(dict(id='native-stage-group:'+identity, index=observed['index'],
            stage_indices=deepcopy(observed['stage_indices']), stage_ids=[row['id'] for row in members],
            construction='CONTIGUOUS_ORIGINAL_DEPENDENCY_ORDER_NO_REORDER_OR_STAGE_OMISSION',
            source_order_witness=[dict(before=left['id'], after=right['id']) for left,right in zip(members,members[1:])],
            incoming_dependencies=incoming, outgoing_dependencies=outgoing,
            reads=read, writes=write,
            live_in_payloads=[name for name in read if materials[name]['ready_task'] not in member_ids],
            live_out_payloads=[name for name in write if any(user not in member_ids for user in consumers[name])],
            live_before=live(begin), live_after=live(end, True),
            live_through=[name for name in live(begin) if name in live(end, True)],
            entry_slot_contents=slot_contents(begin), exit_slot_contents=slot_contents(end, True),
            required_ready_scalar_live_in=list(dict.fromkeys(scalar for row in members for scalar in row['required_ready_scalars']
                if scalar_producers[scalar] not in member_ids)),
            ready_scalar_live_out=[scalar for scalar,producer in scalar_producers.items() if producer in member_ids
                and any(user not in member_ids for user in scalar_consumers[scalar])],
            original_frame_entry=deepcopy(observed),
            potential_first_accesses=[dict(slot_id=slot, **event) for slot,event in first.items()
                if event['stage_id'] in member_ids],
            runtime_scalar_inputs=list(dict.fromkeys(port['input_id'] for row in members for port in row['runtime_scalar_inputs'])),
            begin_cut=observed['begin_cut'], end_cut=observed['end_cut'],
            timed_scope=observer['timed_scope'], excluded_from_clock=deepcopy(observer['excluded_from_group_clock']),
            physical_cache_state='UNKNOWN', physical_numa_residency='UNKNOWN',
            numeric_cost=None, timing_qualified=False))
    return dict(materializations=materials, stages=stage_rows, groups=groups, dependency_edges=edges,
        potential_first_accesses=[dict(slot_id=slot, **event) for slot,event in first.items()],
        scalar_producers={expression:dict(stage_id=producer,
            source_ref=dict(artifact=view['region'], locator='#/nodes/'+str(node_positions[expression])))
            for expression,producer in scalar_producers.items()})


def derive_native_stage_context(native_manifest, stage_indices=None, *, groups=None, stage_capture=None):
    """Replay existing native source/graph and project its selected group context."""
    from scar.analysis.schedule_time import attach_task_time
    from scar.codegen.native_stage_group import render_native_stage_group
    from scar.compiler.stateful_boundary import build_stateful_boundary
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.ir.program.providers import ProviderArtifact
    from scar.ir.versioned_schedule import schedule_document, validate_versioned_schedule
    from scar.optimization.stateful_recipe import _graph

    _, observer = render_native_stage_group(native_manifest, stage_indices, groups=groups, stage_capture=stage_capture)
    view = observer['parent_view']
    plan = read_artifact(view['plan'])
    region = stateful_from_dict(read_artifact(view['region']))
    boundary = build_stateful_boundary(region, plan)
    snapshot = ProviderArtifact(**plan['resource_context']['target_snapshot'])
    original = _graph(region, plan, boundary, snapshot, physical_release_order=True)
    validate_versioned_schedule(original)
    schedule = attach_task_time(original, source_refs=(dict(artifact=view['plan'], locator='#/stages'),))
    projection = _project(schedule, plan, region, observer, view)
    return dict(schema=SCHEMA, schema_version=1,
        source_refs=dict(native_manifest=view['native_manifest'], source=view['source'], plan=view['plan'],
            region=view['region'], stage_capture=view['capture']['artifact']),
        native_identity=deepcopy(view['identity']), semantic_boundary_id=boundary['id'],
        compiler_boundary=dict(kind='ORIGINAL_FULL_NATIVE_RUN_FRAME_WITH_EXPLICIT_OBSERVER_INSERTIONS',
            compiler=deepcopy(view['compiler']), original_source=view['source'],
            observer_identity=observer['observer_identity'], observer_source_sha256=observer['generated_source_sha256'],
            renderer_source=observer['specification']['renderer'],
            observer_clock=deepcopy(observer['clock']), observer_cut_count=observer['cut_count'],
            source_bytes_preserved=observer['all_original_source_bytes_preserved'],
            all_original_stages_execute=observer['all_original_stages_execute'],
            executed_stage_indices=deepcopy(observer['executed_stage_indices']),
            machine_code_identity_claimed=False, instrumentation_residuals=deepcopy(observer['retained_residuals'])),
        runtime_context=dict(original_native_binding=deepcopy(view['actual_resource_binding']),
            binding_is_whole_entry_cpu_reservation=False, memory_policy='UNBOUND',
            original_wrapper=deepcopy(view['wrapper']), wrapper_in_timed_group=False,
            physical_cache_state='UNKNOWN', physical_page_residency='UNKNOWN',
            numeric_input_values='CALLER_SUPPLIED_NOT_OBSERVED_BY_STATIC_CONTEXT',
            guard_contract=deepcopy(plan['guard_contract']), external_assumptions=deepcopy(plan['external_assumptions'])),
        payload_budget={key:deepcopy(plan[key]) for key in ('budget_scope', 'memory_budget_bytes',
            'external_array_bytes', 'workspace_payload_bytes', 'total_region_array_payload_bytes', 'budget_excludes')},
        versioned_schedule=schedule_document(schedule), **projection,
        existing_adapters=dict(materializations='scar.optimization.stateful_recipe._graph',
            time='scar.analysis.schedule_time.attach_task_time',
            cost='JOIN_EXISTING_CPU_DEMAND_OR_ACTIVITY_COSTS_BY_EXACT_SOURCE_STAGE_REF_NO_COSTS_IMPORTED'),
        implementation_source=artifact_reference(__file__, 'scar.python-source'),
        qualifications=dict(source_renderer_replayed=True, existing_stage_plan_replayed=True,
            original_F4_admission_replayed=False, global_content_identity_inferred=False,
            storage_ids_establish_value_equality=False, first_physical_touch_observed=False,
            allocation_establishes_readiness=False, logical_release_is_physical_free=False,
            group_costs_additive=False, cost_profile_qualified=False, physical_feasibility_proven=False,
            target_program_executed=False, compiled=False, native_loaded=False, deployment_authorized=False))


def verify_native_stage_context(report, native_manifest, stage_indices=None, *, groups=None, stage_capture=None):
    expected = derive_native_stage_context(native_manifest, stage_indices, groups=groups, stage_capture=stage_capture)
    if semantic_sha256(report) != semantic_sha256(expected):
        raise ValueError('native stage group context differs from source/plan/materialization replay')
    return expected


__all__ = ['derive_native_stage_context', 'verify_native_stage_context']
