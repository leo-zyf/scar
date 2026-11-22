"""Barrier-preserving CPU stages for a closed ordered stateful array region.

Computed arrays are immutable payloads; alias descriptors read the use epoch.
Writes occur on private state shadows. External ROI copies begin only after
all computation, floating-environment and CPU restoration checks succeed.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from scar.compiler.model import CpuPlacement
from scar.compiler.stateful_model import (read_stateful, stateful_digest,
    validate_stateful_region, POWER2_SEMANTICS)
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode, loads
from .hardware import _target_materials, validate_cpu_placement
from scar.optimization.pipeline_memory import _shape, _solve
from .resource_solver import semantic_sha256

SCHEMA = 'scar.stateful-stage-plan'
_DESCRIPTORS = {'input', 'const', 'slice'}


def _normalized(bounds, shape):
    bounds = ((None, None), (None, None)) if bounds is None else bounds
    rows = [slice(*axis).indices(extent) for axis, extent in zip(bounds, shape)]
    return [[start, stop] for start, stop, _ in rows], [max(0, stop-start) for start, stop, _ in rows]


def _metadata(region, supplied):
    inputs = {row.id:row for row in region.inputs}; nodes = {node.id:node for node in region.nodes}
    arrays = {key for key,row in inputs.items() if row.kind == 'array'}
    issues = []; declared = {}; shapes = {}; kinds = {}
    if not isinstance(supplied, dict): supplied = {}
    for name in supplied.keys() - arrays:
        issues.append(dict(field='input_shapes.'+name, reason='only original array input IDs accept shape declarations'))
    for name in sorted(arrays):
        shape = _shape(supplied.get(name))
        if shape is None: issues.append(dict(field='input_shapes.'+name, reason='explicit nonnegative rank-two array shape required'))
        else: declared[name] = shape
    for node in region.nodes:
        if node.op == 'input':
            kinds[node.id] = node.attrs['kind']; shapes[node.id] = declared.get(node.attrs['name'])
        elif node.op == 'const': kinds[node.id] = 'scalar'; shapes[node.id] = None
        elif node.op == 'slice':
            kinds[node.id] = 'array'; parent = shapes[node.inputs[0]]
            shapes[node.id] = None if parent is None else _normalized(node.attrs['bounds'], parent)[1]
        else:
            operands = [shapes[key] for key in node.inputs if kinds[key] == 'array']
            kinds[node.id] = 'array' if operands else 'scalar'
            shapes[node.id] = operands[0] if operands and all(shape is not None and shape == operands[0] for shape in operands) else None
            if operands and shapes[node.id] is None:
                issues.append(dict(field='expressions.'+node.id, reason='array broadcasting or unresolved shape is outside this profile'))
    positions = {action.value:index for index,action in enumerate(region.actions) if action.kind == 'evaluate'}
    action_epochs = []; epoch = 0
    for action in region.actions:
        action_epochs.append(epoch)
        if action.kind == 'write': epoch += 1
    uses = defaultdict(list)
    for index,action in enumerate(region.actions):
        operands = nodes[action.value].inputs if action.kind == 'evaluate' else (action.value,)
        for operand in operands:
            if operand in positions: uses[operand].append(index)
    forced = defaultdict(set)
    for value,index in positions.items():
        if kinds[value] != 'array': continue
        if not uses[value]: forced[value].add('ORIGINAL_UNUSED_ARITHMETIC_MUST_STILL_EXECUTE_AND_GUARD')
        if any(action_epochs[user] != action_epochs[index] for user in uses[value]):
            forced[value].add('IMMUTABLE_COMPUTED_VALUE_USED_ACROSS_WRITE_BARRIER')
        if nodes[value].op == 'power2':
            forced[value].add('ORIGINAL_NUMPY_PROVIDER_REQUIRES_MATERIALIZED_OUTPUT')
            operand = nodes[value].inputs[0]
            if nodes[operand].op not in _DESCRIPTORS:
                forced[operand].add('ORIGINAL_PROVIDER_REQUIRES_ACTUAL_ARRAY_OPERAND')
    for action in region.actions:
        if action.kind == 'write' and kinds[action.value] == 'array' and nodes[action.value].op not in _DESCRIPTORS:
            forced[action.value].add('WRITE_RHS_READY_BEFORE_SHADOW_MUTATION')
    return inputs, nodes, declared, shapes, kinds, positions, action_epochs, forced, issues


def plan_stateful_region(region, *, input_shapes, cpu_threads, memory_budget_bytes,
                         resource_binding, search_budget=100000):
    """Build a fixed-order transactional stage plan; resource choice is explicit."""
    region = read_stateful(region) if isinstance(region, (str, Path)) else validate_stateful_region(region)
    if type(cpu_threads) is not int or cpu_threads < 1: raise ValueError('explicit positive CPU thread count required')
    if type(memory_budget_bytes) is not int or memory_budget_bytes < 0: raise ValueError('explicit nonnegative array payload budget required')
    if type(search_budget) is not int or search_budget < 1: raise ValueError('positive allocation search budget required')
    if type(resource_binding) is not CpuPlacement: raise ValueError('original typed CPU/memory allocation required')
    validate_cpu_placement(resource_binding)
    if len(resource_binding.cpu_ids) != cpu_threads or resource_binding.memory_budget_bytes != memory_budget_bytes:
        raise ValueError('CPU allocation or explicit payload budget differs from binding')
    graph, snapshot, inventory, connections = _target_materials(resource_binding.target_path)
    inputs,nodes,declared,shapes,kinds,positions,action_epochs,forced,frontier = _metadata(region,input_shapes)
    base = dict(schema=SCHEMA, schema_version=1, region_id=region.id, region_sha256=stateful_digest(region),
        input_shapes=declared, cpu_threads=cpu_threads, resource_binding=encode(resource_binding),
        memory_budget_bytes=memory_budget_bytes, search_budget=search_budget,
        status='UNRESOLVED', frontier=frontier, stages=[], values=[], materialized_exprs={}, shadows={},
        assignment=None, workspace_slots=[], workspace_payload_bytes=None, external_array_bytes=None,
        total_region_array_payload_bytes=None, budget_satisfied=False, commit_rows=[], state_epochs=[],
        resource_context=dict(target_snapshot=encode(snapshot),
            inventory=[encode(H.TargetResourceRef(snapshot, row.id)) for row in inventory],
            topology=[encode(row) for row in connections], selected_executor=resource_binding.executor_id,
            cpu_ids=list(resource_binding.cpu_ids), memory_policy='UNBOUND',
            memory_domain_id=resource_binding.memory_domain_id, physical_numa_residency_proven=False,
            capacity_or_free_snapshot_is_reserved_budget=False),
        budget_scope='ALL_REGION_ARRAY_PAYLOAD_EXTERNAL_STATE_PLUS_FIXED_WORKSPACE',
        allocation_policy='ENTRY_ONCE_FIXED_SLOT_CAPACITY_HELD_UNTIL_REGION_RETURN',
        logical_release='REUSE_PERMISSION_ONLY_NOT_PHYSICAL_FREE',
        alias_contract='PAIRWISE_DISJOINT_ARRAY_INPUT_STORAGE_AND_PRIVATE_WORKSPACE',
        budget_excludes=['OTHER_UNREFERENCED_PROGRAM_ARRAYS','ALLOCATOR_METADATA_AND_ALIGNMENT_OVERHEAD','THREAD_SCRATCH','PROCESS_RSS'],
        guard_contract=list(region.guards)+['C_CONTIGUOUS_NATIVE_FLOAT64_ARRAY_ROOTS',
            'FINITE_TAGGED_PYTHON_INT_FLOAT_OR_NUMPY_FLOAT64_SCALARS',
            'INTEGER_ONLY_INTERMEDIATES_EXACT_OR_PRECOMMIT_FALLBACK',
            'ALL_SHAPES_ALIAS_BOUNDS_AND_WORKSPACE_CHECKED_BEFORE_EXTERNAL_WRITE',
            'PROVIDER_NATIVE_LOOP_IDENTITY_ABI_AND_REQUIRES_PYAPI_FALSE',
            'ALL_CPU_AND_FENV_RESTORATION_SUCCEEDS_BEFORE_COMMIT'],
        external_assumptions=['SYNCHRONOUS_CPU_EXECUTION','NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS',
                              'FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'],
        transaction=dict(external_writes_before_success=False, failure='ORIGINAL_METHOD_ONCE_FROM_UNCHANGED_EXTERNAL_STATE',
            commit='ORIGINAL_WRITE_ROIS_IN_SOURCE_ORDER_FROM_FINAL_PRIVATE_SHADOWS',
            no_failing_operations_or_GuardFailure_after_first_external_commit=True),
        numerical_contract=dict(arithmetic='ORIGINAL_OPERANDS_AND_DAG_ORDER_NO_REASSOCIATION_OR_CONTRACTION',
            power2=POWER2_SEMANTICS, original_providers=encode(region.providers)),
        objective='LEGAL_BARRIER_PRESERVING_FUSION_AND_FIXED_HOST_BUFFER_REUSE',
        allocation_optimality_scope='FIXED_EMITTED_STAGES_AND_PREALLOCATED_HOST_PAYLOAD_POOL_ONLY',
        time_predicted=False, time_optimality_proven=False, physical_optimality_proven=False,
        iterations_unrolled=False, target_executed=False, deployment_authorized=False)
    if frontier: return base
    stages=[]; values={}; materialized={}; scalar_done=set(); shadow_ready={}; commits=[]; epoch_rows=[]
    shadows={name:'shadow:'+name for name in sorted({action.target for action in region.actions if action.kind=='write'})}
    scalar_shape_inputs=[row.id for row in region.inputs if row.kind=='scalar']
    base['scalar_inputs']=scalar_shape_inputs
    def add(kind, *, epoch, shape=None, root_expr=None, output=None, input_payloads=(),
            expr_ids=(), scalar_inputs=(), array_reads=(), action_ids=(), target=None, bounds=None,
            normalized_bounds=None, op=None, reason=()):
        identifier='stage:'+str(len(stages)); dependencies=[]
        if stages: dependencies.append(stages[-1]['id'])
        for payload in input_payloads:
            producer=values[payload]['producer_stage']
            if producer not in dependencies: dependencies.append(producer)
            root=next((name for name,shadow in shadows.items() if shadow==payload),None)
            if root is not None and shadow_ready[root] not in dependencies:
                dependencies.append(shadow_ready[root])
        row=dict(id=identifier,index=len(stages),kind=kind,epoch=epoch,shape=shape,root_expr=root_expr,
            expr_ids=list(expr_ids),scalar_inputs=list(scalar_inputs),array_reads=list(array_reads),
            input_payloads=list(dict.fromkeys(input_payloads)),output_payload=output,target=target,
            bounds=None if bounds is None else [list(axis) for axis in bounds],normalized_bounds=normalized_bounds,
            op=op,depends_on=dependencies,source_action_ids=list(action_ids),reason=list(reason),
            requires_precommit_validation=kind=='COMMIT')
        stages.append(row)
        if output is not None:
            values[output]=dict(id=output,kind='SHADOW_BUFFER' if kind=='SNAPSHOT_ROOT' else 'IMMUTABLE_TEMPORARY',
                identity_kind='PHYSICAL_STATE_SHADOW_STORAGE_NOT_A_CONTENT_VALUE' if kind=='SNAPSHOT_ROOT' else 'COMPUTED_EXPR_PAYLOAD',
                shape=list(shape),nbytes=8*shape[0]*shape[1],producer_index=row['index'],producer_stage=identifier,
                consumer_indices=[],last_use=row['index'],ready_after=identifier,release_after=None,reason=list(reason))
        return row
    for name,payload in shadows.items():
        row=add('SNAPSHOT_ROOT',epoch=0,shape=declared[name],output=payload,target=name,
            array_reads=[dict(input_id=name,epoch=0,storage='EXTERNAL',ready_after='REGION_ENTRY')],reason=['TRANSACTIONAL_FULL_ROOT_COPY_PRESERVES_UNWRITTEN_HALO'])
        shadow_ready[name]=row['id']
    def closure(root):
        required=set(); payloads=set(); scalars=set(); reads=[]
        def visit(value):
            if value in materialized:
                payloads.add(materialized[value]); return
            if value in scalar_done:
                scalars.add(value); return
            if value in required: return
            node=nodes[value]
            for operand in node.inputs: visit(operand)
            required.add(value)
            if node.op=='input' and node.attrs['kind']=='array':
                name=node.attrs['name']; storage='SHADOW' if name in shadows else 'EXTERNAL'
                reads.append(dict(expression=value,input_id=name,epoch=node.attrs['epoch'],storage=storage,
                                  ready_after=shadow_ready[name] if name in shadows else 'REGION_ENTRY'))
                if name in shadows: payloads.add(shadows[name])
        visit(root)
        return ([node.id for node in region.nodes if node.id in required],sorted(payloads),sorted(scalars),reads)
    for index,action in enumerate(region.actions):
        epoch=action_epochs[index]
        if action.kind=='evaluate':
            if kinds[action.value]=='scalar':
                expr_ids,payloads,scalars,reads=closure(action.value)
                add('SCALAR',epoch=epoch,root_expr=action.value,expr_ids=expr_ids,scalar_inputs=scalars,
                    input_payloads=payloads,array_reads=reads,action_ids=[action.id])
                scalar_done.add(action.value)
            elif action.value in forced:
                expr_ids,payloads,scalars,reads=closure(action.value)
                kind='PROVIDER' if nodes[action.value].op=='power2' else 'COMPUTE'; payload='expr:'+action.value
                row=add(kind,epoch=epoch,shape=shapes[action.value],root_expr=action.value,output=payload,
                    input_payloads=payloads,expr_ids=expr_ids,scalar_inputs=scalars,array_reads=reads,
                    action_ids=[region.actions[positions[value]].id for value in expr_ids if value in positions],reason=sorted(forced[action.value]))
                if kind=='PROVIDER': row['provider_id']=nodes[action.value].attrs['provider']
                materialized[action.value]=payload
        else:
            normalized,target_shape=_normalized(action.bounds,declared[action.target])
            if kinds[action.value]=='array' and shapes[action.value]!=target_shape:
                frontier.append(dict(field='actions.'+action.id,reason='write RHS shape differs from target; array broadcasting unsupported')); continue
            payload=materialized.get(action.value)
            if payload is None:
                expr_ids,payloads,scalars,reads=closure(action.value); payload='rhs:'+action.id
                add('COMPUTE',epoch=epoch,shape=target_shape,root_expr=action.value,output=payload,
                    input_payloads=payloads,expr_ids=expr_ids,scalar_inputs=scalars,array_reads=reads,
                    action_ids=[action.id],reason=['RHS_DESCRIPTOR_OR_SCALAR_SNAPSHOT_BEFORE_SHADOW_MUTATION'])
            row=add('WRITE_SHADOW',epoch=epoch,shape=target_shape,root_expr=action.value,
                input_payloads=[payload,shadows[action.target]],action_ids=[action.id],target=action.target,
                bounds=action.bounds,normalized_bounds=normalized,op=action.op,
                reason=['RHS_COMPLETE_AND_READY_BEFORE_ANY_NEIGHBOR_INPUT_IS_OVERWRITTEN'])
            shadow_ready[action.target]=row['id']
            barrier=dict(action_id=action.id,source_action_index=index,target=action.target,
                epoch_before=action.epoch_before,epoch_after=action.epoch_after,
                world_before=deepcopy(action.world_before),world_after=deepcopy(action.world_after),
                stage_id=row['id'],origins=deepcopy(list(action.origins)))
            epoch_rows.append(barrier)
            commits.append(dict(action_id=action.id,target=action.target,bounds=None if action.bounds is None else [list(axis) for axis in action.bounds],
                normalized_bounds=normalized,shape=target_shape,epoch_before=action.epoch_before,epoch_after=action.epoch_after))
    for commit in commits:
        row=add('COMMIT',epoch=len(epoch_rows),shape=commit['shape'],target=commit['target'],
            input_payloads=[shadows[commit['target']]],action_ids=[commit['action_id']],bounds=commit['bounds'],
            normalized_bounds=commit['normalized_bounds'],op='assign',reason=['EXTERNAL_ROI_COPY_ONLY_AFTER_ALL_PRECOMMIT_CHECKS_SUCCEED'])
        commit['stage_id']=row['id']
    for stage in stages:
        for payload in stage['input_payloads']: values[payload]['consumer_indices'].append(stage['index'])
    for value in values.values():
        value['last_use']=max(value['consumer_indices'],default=value['producer_index'])
        value['release_after']=value['last_use']
    records=list(values.values()); external=sum(8*shape[0]*shape[1] for shape in declared.values())
    base.update(stages=stages,values=records,materialized_exprs=materialized,shadows=shadows,
        commit_rows=commits,state_epochs=epoch_rows,external_array_bytes=external,
        source_operations=[dict(index=index,**encode(action)) for index,action in enumerate(region.actions)],
        expression_stages={node.id:[stage['id'] for stage in stages if node.id in stage['expr_ids']] for node in region.nodes})
    if frontier: return base
    slots,indices,lower,explored,complete=_solve(records,len(stages),search_budget)
    workspace=sum(capacity for capacity,_ in slots); total=external+workspace
    assignment={value['id']:dict(slot_id='slot:'+str(slot)) for value,slot in zip(records,indices)}
    slot_rows=[dict(id='slot:'+str(index),capacity_bytes=capacity,
        values=[value['id'] for value,slot in zip(records,indices) if slot==index]) for index,(capacity,_) in enumerate(slots)]
    for stage in stages:
        stage['release_values_after']=[value['id'] for value in records if value['release_after']==stage['index']]
    base.update(assignment=assignment,workspace_slots=slot_rows,workspace_payload_bytes=workspace,
        total_region_array_payload_bytes=total,budget_satisfied=total<=memory_budget_bytes,
        lower_bound_workspace_bytes=lower,allocation_gap_bytes=workspace-lower,states_explored=explored,
        allocation_search_complete=complete,search_budget_exhausted=not complete,
        status='READY' if complete and total<=memory_budget_bytes else
               'INFEASIBLE_WITHIN_FIXED_POOL_MODEL' if external+lower>memory_budget_bytes else 'SEARCH_BUDGET_EXHAUSTED')
    return base


def verify_stateful_plan(region,plan):
    """Recompute epochs, barriers, shapes, stage DAG and physical slot lifetimes."""
    value=loads(Path(plan).read_bytes()) if isinstance(plan,(str,Path)) else deepcopy(plan)
    if value.get('schema')!=SCHEMA or type(value.get('schema_version')) is not int or value['schema_version']!=1:
        raise ValueError('versioned stateful stage plan required')
    binding=dict(value['resource_binding']); binding['cpu_ids']=tuple(binding['cpu_ids']); binding['evidence']=tuple(binding['evidence'])
    expected=plan_stateful_region(region,input_shapes=value['input_shapes'],cpu_threads=value['cpu_threads'],
        memory_budget_bytes=value['memory_budget_bytes'],resource_binding=CpuPlacement(**binding),search_budget=value['search_budget'])
    if semantic_sha256(value)!=semantic_sha256(expected): raise ValueError('stateful plan differs from original barrier/epoch/lifetime replay')
    return value


__all__=['SCHEMA','plan_stateful_region','verify_stateful_plan']
