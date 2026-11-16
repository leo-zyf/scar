"""Keep private capacities outside a shared loop without retaining their values.

The macro graph has an explicit last-committed-iteration join. It is not a v3
schedule: that IR has no equivalent cross-scope last-iteration lifetime port.
"""
from copy import deepcopy

from scar.compiler.stateful_boundary import build_stateful_boundary
from scar.compiler.stateful_model import validate_stateful_region, stateful_digest
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_document, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_loop_schedule import _loop_model, LOOP_GUARDS
from scar.optimization.stateful_recipe import _graph

SCHEMA = 'scar.stateful-workspace-reuse-plan'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def plan_workspace_reuse(region, step_plan, trip_count, *, loop_plan=None, source_refs=()):
    """Plan lifetime hoisting only; source caller authority stays with its reader."""
    region = validate_stateful_region(region); step_plan = deepcopy(step_plan)
    _need(type(trip_count) is int and 0 <= trip_count < 2**64, 'exact nonnegative uint64 trip count required')
    _need(type(source_refs) in (tuple, list) and all(type(row) is dict for row in source_refs),
          'original source artifact/graph references required')
    # This existing producer independently verifies the original public stage
    # plan, shapes, slots, source bytes and guarded transaction contract.
    boundary = build_stateful_boundary(region, step_plan)
    digest = stateful_digest(region); plan_sha = semantic_sha256(step_plan)
    mapping = {name: name for name in region.parameter_names}
    source_authority = dict(mode='EXPLICIT_FIXED_INPUT_COMPOSITION', automatic_caller_replacement_authorized=False,
        original_f4_caller_loop_proven_by_this_planner=False)
    loop_reference = None
    if loop_plan is not None:
        _need(type(loop_plan) is dict and (loop_plan.get('schema'), loop_plan.get('schema_version'))
              == ('scar.stateful-loop-plan', 1), 'original source-loop plan required')
        raw = dict(loop_plan); identity = raw.pop('id')
        _need(identity == 'stateful-loop-plan:' + semantic_sha256(raw)
              and loop_plan['step_sha256'] == digest and loop_plan['trip_count'] == trip_count
              and semantic_sha256(loop_plan['step_plan']) == plan_sha
              and loop_plan['input_shapes'] == step_plan['input_shapes'],
              'source loop step, shape, count or original plan identity differs')
        mapping = loop_plan['parameter_map']
        _need(type(mapping) is dict and set(mapping) == set(region.parameter_names)
              and all(value in loop_plan['parameters'] for value in mapping.values())
              and loop_plan['trip_count_parameter'] in loop_plan['parameters']
              and loop_plan['call_evidence'] and loop_plan['origins'] and loop_plan['producer_pins'],
              'original exact caller/CALL parameter mapping and source evidence required')
        _need(loop_plan['loop_model'] == _loop_model(region, mapping, trip_count),
              'original shared-body content recurrence differs')
        loop_reference = dict(schema=loop_plan['schema'], id=identity, semantic_sha256=semantic_sha256(loop_plan),
            proof_id=loop_plan['proof_id'], proof_sha256=loop_plan['proof_sha256'],
            caller_unit=loop_plan['caller_unit'], function_ordinal=loop_plan['function_ordinal'],
            interface_ordinal=loop_plan['interface_ordinal'], step_reference=deepcopy(loop_plan['step_reference']),
            call_evidence_ref=dict(artifact_role='loop_plan', pointer='#/call_evidence'),
            origins_ref=dict(artifact_role='loop_plan', pointer='#/origins'))
        source_authority = dict(mode='ORIGINAL_SOURCE_LOOP_PLAN_BOUND', loop_plan=loop_reference,
            full_f4_source_proof_and_loop_plan_reader_required_at_publication_and_lowering=True,
            source_validator='scar.compiler.stateful_loop_frontend.validate_stateful_loop_proof',
            original_f4_caller_loop_proven_by_this_planner=False,
            automatic_caller_replacement_authorized=False)
    snapshot = ProviderArtifact(**step_plan['resource_context']['target_snapshot'])
    body = _graph(region, step_plan, boundary, snapshot, physical_release_order=True)
    validate_versioned_schedule(body)
    slots = {row.id: row for row in body.storage_slots}
    private = {key: row for key, row in slots.items() if row.ownership == 'SCHEDULE'}
    materials = {row.id: row for row in body.materializations}
    positions = {row.id: i for i, row in enumerate(body.tasks)}
    definitions = {key: [] for key in private}; reads = []; first_reads = {}
    def coverage(slot, start, end):
        point = start; witnesses = []
        for lower, upper, task, material in sorted(definitions[slot]):
            if upper <= point or lower > point:
                continue
            point = max(point, upper); witnesses.append(dict(task=task, materialization=material))
            if point >= end:
                break
        return point >= end, witnesses
    for task in body.tasks:
        if task.kind in (K.ALLOCATE, K.RELEASE):
            continue
        for name in task.reads:
            material = materials[name]; region_ = material.region
            if region_.slot not in private:
                continue
            _need(material.ready_task is not None and positions[material.ready_task] < positions[task.id],
                  'a private read must have its actual producer earlier in this iteration')
            start, end = region_.offset_bytes, region_.offset_bytes + region_.nbytes
            covered, witnesses = coverage(region_.slot, start, end)
            _need(covered, 'private read would consume undefined bytes or a previous-iteration workspace value')
            record = dict(task=task.id, materialization=name, slot=region_.slot,
                extent=dict(offset_bytes=start, nbytes=region_.nbytes), producer=material.ready_task,
                current_iteration_definitions=witnesses, first_required_slot_read=region_.slot not in first_reads,
                material_version=material.version, version_scope='ORIGINAL_STEP_WITHIN_ITERATION_i')
            reads.append(record); first_reads.setdefault(region_.slot, record)
        for name in task.writes:
            material = materials[name]; region_ = material.region
            if region_.slot not in private:
                continue
            start, end = region_.offset_bytes, region_.offset_bytes + region_.nbytes
            if task.kind is K.STATE_WRITE:
                # Its full-root material is a resulting value, not a full-root
                # physical store. The ROI only preserves previously defined halo.
                _need(coverage(region_.slot, start, end)[0],
                      'partial state write requires this iteration to have defined the complete prior root')
            else:
                _need(task.kind in (K.SNAPSHOT, K.COMPUTE), 'unsupported private definition/escape effect')
                definitions[region_.slot].append((start, end, task.id, name))
    _need(not any(task.kind is K.EXPORT for task in body.tasks)
          and all(row.region.slot not in private for task in body.tasks if task.kind is K.COMMIT
                  for row in (materials[name] for name in task.writes)),
          'workspace pointers or uncommitted private values cannot escape')
    order = [row['id'] for row in step_plan['workspace_slots']]
    release_order = list(reversed(order))
    body_ids = [row.id for row in body.tasks if row.kind not in (K.ALLOCATE, K.RELEASE)]
    root_id, loop_id = 'workspace:root', 'workspace:loop'
    allocation = {key: 'workspace:allocate:' + key for key in order}
    release = {key: 'workspace:release:' + key for key in order}
    nodes = [dict(id=root_id, kind='ROOT', entry_port='workspace:entry', exit_port='workspace:exit')]
    loop_node = dict(id=loop_id, kind='LOOP_REGION', parent=root_id, iteration_symbol='i', trip_count=trip_count,
        lower_inclusive=0, upper_exclusive=trip_count, active=trip_count > 0,
        body_pointer='#/shared_step', body_task_ids=body_ids,
        storage_boundary_map=[dict(slot=key, original_allocate=private[key].allocated_by,
            original_release=private[key].released_by, outer_allocate=allocation[key] if trip_count else None,
            outer_release=release[key] if trip_count else None) for key in order],
        per_iteration_content='RESET_DEFINITION_KNOWLEDGE_NOT_BUFFER_BYTES',
        private_value_reuse_across_iterations_authorized=False)
    nodes.append(loop_node); edges = []
    ports = [dict(id='workspace:entry', owner=root_id, kind='ENTRY'),
        dict(id='workspace:exit', owner=root_id, kind='EXIT'),
        dict(id='workspace:failed_step', owner=loop_id, kind='PRECOMMIT_FAILURE', completed_symbol='k'),
        dict(id='workspace:original_suffix', owner=root_id, kind='ORIGINAL_FALLBACK_SUFFIX',
            start='k', stop=trip_count)]
    if trip_count:
        previous = 'workspace:entry'
        for key in order:
            nodes.append(dict(id=allocation[key], kind='ALLOCATE', parent=root_id, slot=key,
                capacity_bytes=private[key].capacity_bytes, establishes_content_ready=False))
            edges.append(dict(source=previous, target=allocation[key], kind='BEFORE_LOOP_ALLOCATION'))
            previous = allocation[key]
        edges.append(dict(source=previous, target=loop_id, kind='ALL_CAPACITIES_HELD_BEFORE_FIRST_ITERATION'))
        previous = loop_id
        for index, key in enumerate(release_order):
            nodes.append(dict(id=release[key], kind='RELEASE', parent=root_id, slot=key))
            edges.append(dict(source=previous, target=release[key],
                kind='AFTER_LAST_COMMITTED_ITERATION' if index == 0 else 'REVERSE_ARRAY_DESTRUCTION_ORDER',
                **(dict(iteration_symbol='i', iteration=trip_count-1, condition='NORMAL_COMPLETE') if index == 0 else {})))
            previous = release[key]
        edges.append(dict(source=previous, target='workspace:exit', kind='ALL_PRIVATE_CAPACITIES_RELEASED', condition='NORMAL_COMPLETE'))
        edges.append(dict(source='workspace:failed_step', target=release[release_order[0]],
            kind='AFTER_FAILED_STEP_QUIESCENT', completed_symbol='k', condition='PRECOMMIT_FAILURE'))
        edges.append(dict(source=previous, target='workspace:original_suffix',
            kind='RELEASE_BEFORE_UNCOMMITTED_ORIGINAL_SUFFIX', condition='PRECOMMIT_FAILURE'))
        edges.append(dict(source='workspace:original_suffix', target='workspace:exit',
            kind='AFTER_ORIGINAL_SUFFIX_COMPLETES', replay_committed_prefix=False))
    else:
        edges.append(dict(source='workspace:entry', target='workspace:exit', kind='ZERO_TRIP_NO_FIELDS_NO_ALLOCATION'))
    loop_model = _loop_model(region, mapping, trip_count)
    for row in loop_model['inputs']:
        ports.append(dict(id='input:'+row['input_id'], owner=root_id, kind='INPUT', input_id=row['input_id'],
            source_ref=dict(artifact_role='region', pointer='#/inputs/'+str(next(i for i,v in enumerate(region.inputs) if v.id == row['input_id'])))))
        edges.append(dict(source='input:'+row['input_id'], target=loop_id,
            kind='GUARDED_FIXED_SCALAR_LIVE_IN' if row['kind'] == 'scalar' else 'GUARDED_FIXED_ARRAY_DESCRIPTOR_LIVE_IN',
            body_input=row['input_id'], shape=step_plan['input_shapes'].get(row['input_id']),
            enabled=trip_count > 0, binding_is_not_content_identity=True))
        if row['role'] == 'ORDERED_ARRAY_CONTENT_CARRY':
            edges.append(dict(source=loop_id, target=loop_id, kind='COMMITTED_ARRAY_CONTENT_CARRY',
                input_id=row['input_id'], iteration_symbol='i', from_coordinate=dict(iteration='i', local_epoch=row['exit']['local_epoch']),
                to_coordinate=dict(iteration='i + 1', local_epoch=0), enabled_when='i + 1 < trip_count',
                runtime_logical_version_inferred=False))
    for key in order:
        edges.append(dict(source=allocation[key] if trip_count else 'workspace:entry', target=loop_id,
            kind='BORROW_FIXED_CAPACITY_ADDRESS_NOT_CONTENT', slot=key, enabled=trip_count > 0,
            address_observed=False, invariant_content=False))
    endpoints = {row['id'] for row in (*nodes, *ports)}
    _need(all(edge['source'] in endpoints and edge['target'] in endpoints for edge in edges),
          'workspace macro graph has an undeclared node/port endpoint')
    nonzero = sum(private[key].capacity_bytes > 0 for key in order)
    slot_events = len(order) if trip_count else 0
    result = dict(schema=SCHEMA, schema_version=1, trip_count=trip_count,
        step_region_sha256=digest, step_plan_sha256=plan_sha, input_shapes=deepcopy(step_plan['input_shapes']),
        cpu_threads=step_plan['cpu_threads'], resource_binding=deepcopy(step_plan['resource_binding']),
        slots=[dict(id=key, capacity_bytes=private[key].capacity_bytes) for key in order],
        allocation_order=order, release_order=release_order, loop_region=loop_node,
        graph=dict(representation='ROOT_WITH_SHARED_LOOP_REGION_AND_LAST_COMMITTED_ITERATION_JOIN',
            nodes=nodes, ports=ports, edges=edges, is_versioned_schedule_v3=False, shared_body_count=1, iterations_unrolled=False),
        shared_step=schedule_document(body), loop_model=loop_model,
        source_f4_authority=source_authority, source_refs=encode(tuple(source_refs)),
        sources=dict(region=dict(artifact_role='region', pointer='#', semantic_sha256=digest),
            step_plan=dict(artifact_role='step_plan', pointer='#', semantic_sha256=plan_sha),
            original_source_ref=dict(artifact_role='region', pointer='#/source')),
        initialization_proof=dict(rule='CURRENT_ITERATION_FULL_DEFINITION_BEFORE_EVERY_PRIVATE_READ',
            reads=reads, first_required_slot_reads=list(first_reads.values()),
            state_write_material_is_not_full_physical_store=True,
            original_slot_alias_versions_and_stage_order_retained=True,
            previous_iteration_bytes_may_remain_but_are_never_inputs=True),
        memory=dict(budget_scope=step_plan['budget_scope'], memory_budget_bytes=step_plan['memory_budget_bytes'],
            external_array_bytes=step_plan['external_array_bytes'], workspace_payload_bytes=step_plan['workspace_payload_bytes'],
            total_region_array_payload_bytes=step_plan['total_region_array_payload_bytes'],
            budget_satisfied=step_plan['budget_satisfied'], peak_payload_budget_unchanged=True,
            memory_lifetime_extended=trip_count > 1, numa_residency_proven=False, process_rss_bound_proven=False),
        allocation_counts=dict(slot_records=len(order), nonzero_heap_slots=nonzero,
            original_slot_allocation_events=trip_count*len(order), reused_slot_allocation_events=slot_events,
            original_slot_release_events=trip_count*len(order), reused_slot_release_events=slot_events,
            original_heap_allocation_calls=trip_count*nonzero, reused_heap_allocation_calls=nonzero if trip_count else 0,
            original_heap_release_calls=trip_count*nonzero, reused_heap_release_calls=nonzero if trip_count else 0,
            scope='NORMAL_COMPLETION_ONLY_NOT_ALLOCATION_FAILURE_PATH'),
        execution_contract=dict(normal_completion='ORIGINAL_PER_STEP_TRANSACTION_AND_EXTERNAL_COMMITS',
            preserve_each_iteration=['SNAPSHOT','COMPUTE','SCALAR_AND_ARRAY_GUARDS','PROVIDERS','SUCCESS_FENCE',
                'COMMIT','CALLER_AND_WORKER_AFFINITY_FENV_TEAM','HELPER_CLEANUP'],
            only_hoisted=['PRIVATE_WORKSPACE_ALLOCATION','PRIVATE_WORKSPACE_FINAL_RELEASE'],
            zero_trip='RETURN_BEFORE_FIELDS_DESCRIPTORS_GUARDS_PROVIDERS_OR_ALLOCATIONS',
            completed='NUMBER_OF_SUCCESSFULLY_COMMITTED_ITERATIONS',
            failure=dict(failing_iteration='k', completed='k', failed_step_external_state_unchanged=True,
                release_workspace_after_failed_step_quiescent=True, resume_original_suffix=dict(start='k', stop=trip_count),
                replay_committed_prefix=False),
            allocation_events_and_oom_exception_sequence_preserved=False,
            interstep_observers_hooks_signals_or_foreign_readers_writers_supported=False),
        required_live_guards=(list(loop_plan['required_live_guards']) if loop_plan is not None else
            [guard for guard in LOOP_GUARDS if not guard.startswith('ORIGINAL_CALLER_CODE_')] + list(step_plan['guard_contract'])),
        external_assumptions=list(step_plan['external_assumptions']),
        default_runtime_decision='NO_OP', cost=dict(status='UNKNOWN', duration_ticks=None, profitable=None),
        specialized_lowerer_required=True, shared_body_native_implemented_by_this_planner=False,
        selected=False, automatic_caller_replacement_authorized=False,
        physical_optimality_proven=False, deployment_authorized=False, target_program_executed=False)
    result['id'] = 'stateful-workspace-plan:' + semantic_sha256(result)
    return result


def verify_workspace_reuse(region, step_plan, plan, *, loop_plan=None):
    _need(type(plan) is dict and (plan.get('schema'), plan.get('schema_version')) == (SCHEMA, 1),
          'versioned shared workspace reuse plan required')
    expected = plan_workspace_reuse(region, step_plan, plan['trip_count'], loop_plan=loop_plan,
                                    source_refs=tuple(plan['source_refs']))
    _need(semantic_sha256(plan) == semantic_sha256(expected),
          'workspace graph differs from original source step, lifetime and shared-loop replay')
    return plan


__all__ = ['SCHEMA', 'plan_workspace_reuse', 'verify_workspace_reuse']
