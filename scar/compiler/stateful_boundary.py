"""Source-replayed semantic ports for one transactional stateful array call.

Epochs are local to the original ordered writes, not global content versions.
The original source, expressions, worlds and providers remain referenced once
in the region artifact; no buffer or CPU choice enters this boundary.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .stateful_model import (read_stateful, stateful_from_dict, stateful_digest,
    validate_stateful_region)
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_schedule import verify_stateful_plan

SCHEMA = 'scar.stateful-semantic-boundary'


def _ref(role, pointer):
    return dict(artifact_role=role, pointer=pointer)


def _source(source):
    return {key:getattr(source,key) for key in ('path','sha256','qualname','line','column','code_key')}


def admit_stateful_inputs(region, plan):
    """Replay the public model/planner and original source bytes, without loading code."""
    if isinstance(region,(str,Path)): region=read_stateful(region)
    elif isinstance(region,dict): region=stateful_from_dict(region)
    else: region=validate_stateful_region(region)
    plan=verify_stateful_plan(region,plan)
    if (plan['status']!='READY' or plan['frontier'] or not plan['budget_satisfied']
            or not plan['allocation_search_complete']):
        raise ValueError('complete budget-satisfied stateful stage plan required')
    for source in (region.source,*(row.source for row in region.callees)):
        if sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256:
            raise ValueError('stateful original source snapshot differs')
    return region,plan


def boundary_digest(boundary):
    if boundary.get('schema')!=SCHEMA or boundary.get('schema_version')!=1:
        raise ValueError('versioned stateful semantic boundary required')
    value=dict(boundary); value.pop('id',None)
    return semantic_sha256(value)


def build_stateful_boundary(region, plan):
    region,plan=admit_stateful_inputs(region,plan)
    inputs=[]
    for index,row in enumerate(region.inputs):
        item=dict(id=row.id,path=list(row.path),kind=row.kind,kind_authority=row.kind_authority,
            source_ref=_ref('region',f'#/inputs/{index}'))
        if row.kind=='array':
            item.update(shape=plan['input_shapes'][row.id],dtype='float64',layout='C',
                ownership='CALLER_OWNED_MUTABLE' if row.id in plan['shadows'] else 'CALLER_OWNED_READ_ONLY',
                shape_authority='EXPLICIT_PLAN_SPECIALIZATION_GUARDED_AT_CALL_NOT_SOURCE_CONSTANT')
        else: item.update(type_contract='FINITE_TAGGED_BOUNDED_PYTHON_INT_OR_FLOAT')
        inputs.append(item)
    actions=[]
    for index,row in enumerate(region.actions):
        item=dict(id=row.id,kind=row.kind,expression=row.value,
            source_ref=_ref('region',f'#/actions/{index}'))
        if row.kind=='write':
            item.update(target=row.target,op=row.op,bounds=None if row.bounds is None else [list(b) for b in row.bounds],
                epoch_before=row.epoch_before,epoch_after=row.epoch_after,
                world_before_ref=_ref('region',f'#/actions/{index}/world_before'),
                world_after_ref=_ref('region',f'#/actions/{index}/world_after'))
        actions.append(item)
    value=dict(schema=SCHEMA,schema_version=1,profile=region.profile,
        source=dict(region_id=region.id,region_sha256=stateful_digest(region),source=_source(region.source),
            source_ref=_ref('region','#/source'),
            callees=[dict(path=list(row.path),kind=row.kind,source=_source(row.source),
                source_ref=_ref('region',f'#/callees/{index}')) for index,row in enumerate(region.callees)],
            providers=[dict(id=row.id,module=row.module,symbol=row.symbol,lookup_path=list(row.lookup_path),
                semantics=row.semantics,source_ref=_ref('region',f'#/providers/{index}'))
                for index,row in enumerate(region.providers)]),
        parameters=list(region.parameter_names),inputs=inputs,actions=actions,
        return_kind=region.return_kind,
        epochs=dict(scope='THIS_CALL_ORIGINAL_ORDERED_ROOT_WRITES_ONLY',
            global_content_identity_inferred=False,alias_descriptors='READ_AT_THEIR_ORIGINAL_CONSUMPTION_EPOCH',
            computed_values='IMMUTABLE_ORIGINAL_EVALUATION_RESULTS_CAN_CROSS_BARRIERS'),
        effects=dict(ordered_mutations=True,roots=sorted(plan['shadows']),
            external_write_policy=deepcopy(plan['transaction']),
            isolation='PRIVATE_FULL_ROOT_SHADOWS_THEN_ORIGINAL_ROI_COMMITS',
            whole_program_effects_closed=False),
        dependencies=dict(source_action_order=True,all_rhs_ready_before_shadow_write=True,
            all_precommit_checks_before_first_external_write=True,
            external_commit_order='ORIGINAL_WRITE_ROI_ORDER',device_difference_removes_dependencies=False),
        Q=dict(status='EXISTING_GUARDED_STRICT_CPU_CONTRACT',dtype='float64',
            arithmetic=plan['numerical_contract']['arithmetic'],
            power2=plan['numerical_contract']['power2'],
            integer_only_intermediates='EXACT_TAGGED_OR_PRECOMMIT_FALLBACK',
            reassociation=False,contraction=False,approximate_substitution=False,
            original_providers_ref=_ref('region','#/providers'),
            full_numpy_status_flag_equivalence=False,device_fp_relaxation_authorized=False),
        guards=list(plan['guard_contract']),external_assumptions=list(plan['external_assumptions']),
        provenance=dict(expressions_ref=_ref('region','#/nodes'),actions_ref=_ref('region','#/actions'),
            original_origins_ref=_ref('region','#/source/origins'),
            original_graph_mutated=False,iteration_instances_expanded=False,
            physical_slots_are_not_logical_values=True),
        scope='ONE_CLOSED_STATEFUL_METHOD_CALL_NOT_WHOLE_PROGRAM')
    value['id']='stateful-semantic-boundary:'+boundary_digest(value)
    return value


def verify_stateful_boundary(region,plan,boundary):
    expected=build_stateful_boundary(region,plan)
    if boundary!=expected: raise ValueError('stateful semantic boundary differs from original replay')
    return expected


__all__=['admit_stateful_inputs','build_stateful_boundary','verify_stateful_boundary','boundary_digest']
