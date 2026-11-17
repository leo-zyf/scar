"""Logical access and MAY-hazard facts for an existing ordered stateful region.

Original input/Expr/action IDs and region-local epochs remain source metadata.
They are neither physical storage identities nor closed Python effect facts.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from scar.analysis.stateful_dependencies import project_stateful_dependencies
from scar.compiler.stateful_model import read_stateful, validate_stateful_region, stateful_digest
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_schedule import _metadata, _normalized


SCHEMA = 'scar.stateful-region-effects'


def _empty(bounds):
    return any(stop <= start for start, stop in bounds)


def _overlap(left, right):
    if _empty(left['bounds']) or _empty(right['bounds']):
        return 'EMPTY_ELEMENT_FOOTPRINT', 'NO_ELEMENT_PAYLOAD_OVERLAP'
    if left['input_id'] != right['input_id']:
        return 'DISTINCT_LOGICAL_INPUTS', 'UNKNOWN_DISTINCT_INPUT_STORAGE_ALIAS'
    if any(a[1] <= b[0] or b[1] <= a[0] for a, b in zip(left['bounds'], right['bounds'])):
        return 'DISJOINT_LOGICAL_COORDINATES', 'UNKNOWN_ROOT_LAYOUT_INJECTIVITY'
    return 'OVERLAPPING_LOGICAL_COORDINATES', 'CONDITIONAL_SAME_ROOT_COORDINATE_OVERLAP'


def derive_stateful_region_effects(region, *, input_shapes):
    """Project source accesses without removing order or discharging guards.

    Shapes must instantiate the existing equal-shape stateful profile. The
    validator rejects an old alias descriptor reused after its root epoch has
    advanced; this API does not silently turn that descriptor into a snapshot.
    """
    region = read_stateful(region) if isinstance(region, (str, Path)) else validate_stateful_region(region)
    inputs, nodes, declared, shapes, kinds, _, _, _, issues = _metadata(region, input_shapes)
    if issues:
        raise ValueError('stateful effect shape profile is unresolved: '+str(issues))
    for action in region.actions:
        if action.kind == 'write':
            _, shape = _normalized(action.bounds, declared[action.target])
            if kinds[action.value]=='array' and shapes[action.value]!=shape:
                raise ValueError('write RHS shape differs from the existing exact stateful profile')
    projection = project_stateful_dependencies(region)
    digest = stateful_digest(region)
    node_refs = {node.id:'#/nodes/'+str(index) for index, node in enumerate(region.nodes)}
    expression_at = {reference:key for key, reference in node_refs.items()}
    action_refs = {action.id:'#/actions/'+str(index) for index, action in enumerate(region.actions)}
    action_at = {reference:key for key, reference in action_refs.items()}
    descriptors, provenance, scalar_inputs, expressions = {}, {}, {}, []

    def ref(pointer):
        return dict(region_sha256=digest, locator=pointer)

    for node in region.nodes:
        scalar_inputs[node.id] = set()
        if node.op=='input' and node.attrs['kind']=='array':
            shape = declared[node.attrs['name']]
            descriptors[node.id] = dict(input_id=node.attrs['name'], epoch=node.attrs['epoch'],
                bounds=[[0, extent] for extent in shape], shape=list(shape), slice_chain=[])
        elif node.op=='slice':
            parent = descriptors[node.inputs[0]]
            bounds, shape = _normalized(node.attrs['bounds'], parent['shape'])
            descriptors[node.id] = dict(input_id=parent['input_id'], epoch=parent['epoch'],
                bounds=[[base[0]+start, base[0]+start+extent]
                    for base, (start, _), extent in zip(parent['bounds'], bounds, shape)],
                shape=shape, slice_chain=[*parent['slice_chain'], deepcopy(node.attrs['bounds'])])
        if node.id in descriptors:
            descriptor = descriptors[node.id]
            provenance[node.id] = {node.id}
        else:
            descriptor = None
            provenance[node.id] = set().union(*(provenance[key] for key in node.inputs))
        if node.op=='input' and node.attrs['kind']=='scalar':
            scalar_inputs[node.id].add(node.attrs['name'])
        else:
            scalar_inputs[node.id].update(set().union(*(scalar_inputs[key] for key in node.inputs)))
        expressions.append(dict(id=node.id, op=node.op, source_ref=ref(node_refs[node.id]),
            value_kind=kinds[node.id], shape=deepcopy(shapes[node.id]), operands=list(node.inputs),
            descriptor=deepcopy(descriptor), input_provenance=[dict(expression=key,
                source_ref=ref(node_refs[key]), **deepcopy(descriptors[key])) for key in sorted(provenance[node.id])],
            scalar_input_dependencies=sorted(scalar_inputs[node.id]),
            provenance_is_runtime_reload=False,
            content_read_semantics='AT_CONSUMING_ACTION' if descriptor is not None else
                'IMMUTABLE_VALUE_PROVENANCE_NOT_A_REPEAT_INPUT_READ',
            physical_storage_identity=None, physical_content_identity=None))

    accesses, per_action = [], defaultdict(lambda: dict(reads=[], writes=[]))

    def access(action_id, mode, role, descriptor, *, expression=None, source_operand_index=None, **extra):
        index = len(accesses)
        value = dict(index=index, action_id=action_id, mode=mode, role=role,
            source_action_ref=ref(action_refs[action_id]), expression=expression,
            source_expr_ref=None if expression is None else ref(node_refs[expression]),
            source_operand_index=source_operand_index, **deepcopy(descriptor), **extra,
            epoch_scope='ONE_REGION_INVOCATION_UNDER_UNDISCHARGED_SOURCE_AND_ALIAS_GUARDS',
            input_epoch_is_storage_or_global_content_identity=False,
            element_access_count=None, physical_bytes=None, duration_seconds=None)
        accesses.append(value)
        per_action[action_id]['reads' if mode=='READ' else 'writes'].append(index)
        return index

    read_edges = [edge for edge in projection['edges'] if edge['kind']=='ARRAY_CONTENT_READ']
    if len(read_edges)!=len(projection['reads']):
        raise ValueError('original dependency projection lost its descriptor read ledger')
    for read, edge in zip(projection['reads'], read_edges):
        action_id = action_at[read['action']]
        expression = expression_at[read['expression']]
        descriptor = descriptors[expression]
        if (descriptor['input_id'], descriptor['epoch']) != (read['input_id'], read['epoch']):
            raise ValueError('original content read and descriptor-local epoch differ')
        access(action_id, 'READ', 'ORIGINAL_DESCRIPTOR_OPERAND', descriptor,
            expression=expression, source_operand_index=edge['operand_index'])
    actions, dependencies, epochs = [], [], {row.id:0 for row in region.inputs if row.kind=='array'}
    evaluated = {}
    for index, action in enumerate(region.actions):
        operands = nodes[action.value].inputs if action.kind=='evaluate' else (action.value,)
        value_dependencies = []
        for operand in operands:
            if operand in evaluated:
                before = evaluated[operand]
                value_dependencies.append(dict(expression=operand, producer_action=before))
                dependencies.append(dict(kind='IMMUTABLE_VALUE_DEPENDENCE', before=before, after=action.id,
                    expression=operand, source_ref=ref(node_refs[operand]), strength='SOURCE_VALUE_DEPENDENCE'))
        if action.kind=='evaluate':
            evaluated[action.value] = action.id
        else:
            bounds, shape = _normalized(action.bounds, declared[action.target])
            bounds = [[start, start+extent] for (start, _), extent in zip(bounds, shape)]
            descriptor = dict(input_id=action.target, epoch=action.epoch_before,
                bounds=bounds, shape=shape, slice_chain=[])
            if action.op!='assign':
                access(action.id, 'READ', 'READ_MODIFY_WRITE_DESTINATION', descriptor)
            access(action.id, 'WRITE', 'ORIGINAL_TARGET_ROI', descriptor,
                epoch_before=action.epoch_before, epoch_after=action.epoch_after, op=action.op)
            epochs[action.target] = action.epoch_after
        if index:
            dependencies.append(dict(kind='ORIGINAL_ACTION_ORDER', before=region.actions[index-1].id,
                after=action.id, strength='RETAINED_SOURCE_TOTAL_ORDER', removable=False))
        footprint = None if action.kind!='write' else accesses[per_action[action.id]['writes'][0]]
        actions.append(dict(id=action.id, index=index, kind=action.kind, expression=action.value,
            source_ref=ref(action_refs[action.id]), reads=list(per_action[action.id]['reads']),
            writes=list(per_action[action.id]['writes']), immutable_value_dependencies=value_dependencies,
            epoch_after_action=dict(epochs),
            world_before_ref=None if action.world_before is None else ref(action_refs[action.id]+'/world_before'),
            world_after_ref=None if action.world_after is None else ref(action_refs[action.id]+'/world_after'),
            state_preservation=None if footprint is None else dict(input_id=action.target,
                epoch_before=action.epoch_before, epoch_after=action.epoch_after,
                unchanged_complement=footprint['bounds']!=[[0, extent] for extent in declared[action.target]],
                complement_is_emitted_payload_read=False),
            intra_action_element_access_order='ORIGINAL_OPERATOR_SEMANTICS_NOT_INFERRED_OR_PARALLELIZED'))

    positions = {action.id:index for index, action in enumerate(region.actions)}
    accesses.sort(key=lambda row:(positions[row['action_id']], row['mode']=='WRITE', row['index']))
    access_indices = {row['index']:index for index,row in enumerate(accesses)}
    for index,row in enumerate(accesses):
        row['index'] = index
    for row in actions:
        for field in ('reads','writes'):
            row[field] = [access_indices[index] for index in row[field]]
    for left in accesses:
        for right in accesses:
            if positions[left['action_id']]>=positions[right['action_id']] or (left['mode'],right['mode'])==('READ','READ'):
                continue
            logical, physical = _overlap(left, right)
            if physical=='NO_ELEMENT_PAYLOAD_OVERLAP':
                continue
            kind = ('RAW' if left['mode']=='WRITE' and right['mode']=='READ' else
                'WAR' if left['mode']=='READ' else 'WAW')
            dependencies.append(dict(kind=kind, before=left['action_id'], after=right['action_id'],
                before_access=left['index'], after_access=right['index'], strength='MAY_UNDER_PROFILE',
                logical_overlap=logical, physical_overlap=physical,
                disjointness_proven=False, source_order_may_be_removed=False))

    roots = [row for row in region.inputs if row.kind=='array']
    result = dict(schema=SCHEMA, schema_version=1, region_id=region.id, region_sha256=digest,
        source={key:value for key,value in asdict(region.source).items() if key!='origins'},
        profile=region.profile, status='CONDITIONAL_SOURCE_PROFILE_FACTS', input_shapes=deepcopy(declared),
        original_dependency_projection_sha256=semantic_sha256(projection),
        producer=artifact_reference(__file__, 'scar.python-producer'),
        identity_scope='ORIGINAL_INPUT_EXPR_ACTION_IDS_AND_REGION_LOCAL_LOGICAL_EPOCHS',
        inputs=[dict(id=row.id, path=list(row.path), kind=row.kind,
            source_ref=ref('#/inputs/'+str(index)), shape=deepcopy(declared.get(row.id)))
            for index,row in enumerate(region.inputs)],
        expressions=expressions, actions=actions, accesses=accesses, dependencies=dependencies,
        array_root_aliases=[dict(left=left.id, right=right.id,
            same_logical_input=left.id==right.id,
            physical_alias='SAME_ROOT_REFERENCE_CONDITIONAL_ON_SOURCE_MAPPING' if left.id==right.id else 'UNKNOWN',
            layout_injectivity='UNKNOWN', distinct_input_ids_prove_disjoint_storage=False,
            disjoint_logical_coordinates_prove_physical_disjointness=False)
            for index,left in enumerate(roots) for right in roots[index:]],
        epoch_transitions=[dict(action_id=action.id, source_ref=ref(action_refs[action.id]),
            input_id=action.target, epoch_before=action.epoch_before, epoch_after=action.epoch_after)
            for action in region.actions if action.kind=='write'],
        contracts=[dict(condition=condition, status='UNDISCHARGED_PROFILE_REQUIREMENT') for condition in region.guards],
        callees=deepcopy(projection['callees']), providers=deepcopy(projection['providers']),
        unknown_effects={field:dict(status='UNKNOWN') for field in (
            'caller_and_outside_region', 'source_calls_beyond_retained_profile', 'hooks_and_callbacks',
            'rng', 'exceptions_and_termination', 'allocation_and_release_callbacks',
            'physical_aliases_and_layout', 'whole_python_function_effects')},
        profile_limits=[dict(field='held_alias_descriptors_across_root_write', status='UNSUPPORTED_PROFILE',
            reason='Existing validator rejects consumption when the descriptor encoded epoch is stale; it is not a snapshot.')],
        frontier=[dict(field='physical_nonoverlap', status='UNRESOLVED',
                reason='No live storage/stride/injective-layout evidence or discharged disjoint-input guard is supplied.')],
        source_total_order_retained=True, source_f4_replayed=False, runtime_guards_discharged=False,
        whole_python_function_purity_proven=False, physical_resource_feasibility_proven=False,
        transformation_authorized=False, parallelization_authorized=False, hoisting_authorized=False,
        cost_or_timing_inferred=False, target_executed=False)
    result['id'] = 'stateful-region-effects:'+semantic_sha256(result)
    return result


def verify_stateful_region_effects(region, report):
    """Recompute every logical access and MAY hazard; never execute a target."""
    expected = derive_stateful_region_effects(region, input_shapes=report['input_shapes'])
    if semantic_sha256(expected)!=semantic_sha256(report):
        raise ValueError('stateful region effects differ from original action/epoch/shape replay')
    return deepcopy(expected)


__all__ = ['derive_stateful_region_effects', 'verify_stateful_region_effects']
