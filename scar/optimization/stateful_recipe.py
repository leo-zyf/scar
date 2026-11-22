"""One source-replayed transactional recipe for costing and existing lowering.

The public versioned graph distinguishes array content epochs from reusable
physical slots. Costs are separate sourced records; no target is executed and
no missing duration is replaced with zero.
"""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from scar.compiler.model import CpuPlacement
from scar.compiler.stateful_boundary import admit_stateful_inputs, build_stateful_boundary
from scar.compiler.stateful_model import document, stateful_from_dict
from scar.ir.execution_recipe import (artifact_reference, make_recipe, read_artifact,
    read_recipe, validate_recipe, write_recipe)
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from .hardware import _target_materials
from .resource_solver import semantic_sha256

PROFILE = 'cpu-stateful-array-region-v1'
CPP_POOL_RELEASE_ORDER = 'CPP_STD_ARRAY_REVERSE_DESTRUCTION'


def _ref(role,pointer): return dict(artifact_role=role,pointer=pointer)


def _binding(plan):
    row=dict(plan['resource_binding']); row['cpu_ids']=tuple(row['cpu_ids']); row['evidence']=tuple(row['evidence'])
    return CpuPlacement(**row)


def _save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    return artifact_reference(path,value['schema'])


def stateful_backend_capability(plan):
    """Describe the frozen backend's compiler/ABI/provider applicability.

    This does not retrieve a provider loop or load a generated library. The
    existing backend still checks the actual capsule and requires_pyapi flag.
    """
    from scar.codegen import cpp,stateful
    import numpy as np
    if sys.platform!='linux' or np.__version__!='2.5.2' or sys.maxsize!=2**63-1:
        raise ValueError('stateful recipe requires the existing Linux/NumPy2.5.2/64-bit backend profile')
    compiler=shlex.split(os.environ.get('CXX','g++'))
    if not compiler or not shutil.which(compiler[0]): raise ValueError('C++20 compiler unavailable')
    identity=subprocess.run([*compiler,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    module=importlib.import_module('numpy._core._multiarray_umath')
    if type(np.power) is not np.ufunc or np.power is not module.power:
        raise ValueError('original NumPy power ufunc identity differs')
    binary=Path(module.__file__).resolve(); raw=binary.read_bytes()
    runtime=dict(numpy_version=np.__version__,ndarray_c_abi=int(module._get_ndarray_c_version()),
        capsule_abi='numpy_1.24_ufunc_call_info',loaded_binary=dict(path=str(binary),sha256=sha256(raw).hexdigest(),bytes=len(raw)),
        dtype_signature=['float64','float64','float64'],fixed_strides=[8,0,8],requires_pyapi=False,
        API_stability='EXPERIMENTAL_EXACT_NUMPY_VERSION_AND_BINARY_PIN',
        execution='ORIGINAL_NUMPY_UFUNC_STRIDED_LOOP_SERIAL_PROVIDER_STAGE')
    return dict(schema='scar.stateful-backend-capability',schema_version=1,
        family='CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION',cxx_standard=20,
        stateful_abi_version=1,operator_abi_version=1,implementation_sha256=stateful.implementation_sha256(),
        compiler=dict(argv=compiler,identity=identity,options=[*cpp.COMPILER_OPTIONS,*(['-fopenmp'] if plan['cpu_threads']>1 else [])]),
        provider_runtime=runtime)


def _target(plan):
    binding=_binding(plan)
    graph,snapshot,inventory,connections=_target_materials(binding.target_path)
    if snapshot.sha256!=binding.target_sha256: raise ValueError('stateful recipe target snapshot differs')
    artifacts=dict(target_snapshot=encode(snapshot),
        target_inventory=artifact_reference(graph.report['inventory']['uri'],'scar.target-inventory'))
    for name,receipt in {**graph.report['nodes'],'connections':graph.report['connections'],'paths':graph.report['paths']}.items():
        artifacts['target_table:'+name]=artifact_reference(graph.path/receipt['path'],'scar.target-table-jsonl-gzip')
    return binding,snapshot,inventory,artifacts


def _final_commit_proofs(region,plan):
    """Prove full-result identity from a snapshot and complete deferred ROI coverage.

    An individual ROI copy need not produce the complete final root. Only the
    last root commit can reuse the shadow's logical value: all locations ever
    changed by a shadow write have then been copied, and the complement still
    equals the original full snapshot under the exclusive-state contract.
    """
    if 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS' not in plan['external_assumptions']:
        raise ValueError('final root identity requires exclusive region state')
    proofs={}
    for target in plan['shadows']:
        snapshot=[row for row in plan['stages'] if row['kind']=='SNAPSHOT_ROOT' and row['target']==target]
        writes=[row for row in plan['stages'] if row['kind']=='WRITE_SHADOW' and row['target']==target]
        commits=[row for row in plan['stages'] if row['kind']=='COMMIT' and row['target']==target]
        actions=[row for row in region.actions if row.kind=='write' and row.target==target]
        if (len(snapshot)!=1 or snapshot[0]['shape']!=plan['input_shapes'][target]
                or len(writes)!=len(actions) or len(commits)!=len(actions) or not actions
                or [row['source_action_ids'] for row in writes]!=[[row.id] for row in actions]
                or [row['source_action_ids'] for row in commits]!=[[row.id] for row in actions]
                or [row['normalized_bounds'] for row in writes]!=[row['normalized_bounds'] for row in commits]):
            raise ValueError('final root identity needs full snapshot and complete original ordered ROI commit coverage')
        proofs[target]=dict(kind='FINAL_ROOT_COMMIT_RESTORES_COMPLETE_PRIVATE_RESULT',
            target=target,shape=plan['input_shapes'][target],
            nbytes=8*plan['input_shapes'][target][0]*plan['input_shapes'][target][1],
            initial_snapshot=snapshot[0]['id'],shadow_writes=[row['id'] for row in writes],
            ordered_root_commits=[row['id'] for row in commits],last_commit=commits[-1]['id'],
            write_roi_refs=[_ref('stage_plan',f"#/stages/{row['index']}/normalized_bounds") for row in writes],
            commit_roi_refs=[_ref('stage_plan',f"#/stages/{row['index']}/normalized_bounds") for row in commits],
            coverage='EVERY_SHADOW_WRITE_ROI_COMMITTED_FROM_FINAL_SHADOW_IN_ORIGINAL_ORDER',
            outside_written_union='UNCHANGED_FROM_THE_SAME_FULL_ROOT_ENTRY_SNAPSHOT',
            external_state='NO_EXTERNAL_WRITES_BEFORE_COMMIT_NO_FOREIGN_READERS_OR_WRITERS',
            source_transaction_ref=_ref('stage_plan','#/transaction'),
            source_exclusive_contract_ref=_ref('stage_plan','#/external_assumptions'),
            valid_only_after_last_root_commit=True,earlier_commits_are_not_original_intermediate_epochs=True)
    return proofs


def _graph(region,plan,boundary,snapshot,*,physical_release_order=False):
    from scar.ir.versioned_schedule import (VersionedLogicalValue as Value,StorageSlot,
        StorageRegion,VersionedMaterialization as Material,VersionedTask as Task,
        VersionedTaskKind as Kind,OrderConstraint,VersionedResourceSchedule)

    binding=_binding(plan); location='host-address-space:$CALL_PROCESS'
    refs=({'kind':'selected_cpu','executor_id':binding.executor_id,'cpu_ids':list(binding.cpu_ids),
           'snapshot':encode(snapshot)},)
    slots=[]; tasks=[]; values=[]; materials=[]; current={}; versions={}; orders=[]
    sizes={row['id']:row['nbytes'] for row in plan['values']}
    source_slots={row['id']:row for row in plan['workspace_slots']}
    def add_task(task):
        tasks.append(task)
        if len(tasks)>1: orders.append(OrderConstraint(tasks[-2].id,task.id,'SOURCE_ORDER'))
    def material(identifier,value,slot,size,ready=None,external=False):
        version=versions.get(slot,-1)+1; versions[slot]=version
        item=Material(identifier,value,StorageRegion(slot,0,size),version,ready_task=ready,external_ready=external)
        materials.append(item); return identifier
    for name,shape in plan['input_shapes'].items():
        size=8*shape[0]*shape[1]; slot='external:'+name; value='entry:'+name
        slots.append(StorageSlot(slot,location,size,ownership='EXTERNAL',device='cpu',
            resource_refs=({'kind':'original_external_state','input_id':name},),budget_refs=(_ref('stage_plan','#/budget_scope'),)))
        values.append(Value(value,source_refs=(_ref('region',f'#/inputs/{next(i for i,r in enumerate(region.inputs) if r.id==name)}'),),
            attrs=dict(input_id=name,epoch=0,shape=shape,identity='CALL_SCOPED_ORIGINAL_ROOT_CONTENT')))
        current[slot]=material('mat:'+value,value,slot,size,external=True)
    for row in plan['workspace_slots']:
        task_id='allocate:'+row['id']; release='release:'+row['id']
        slots.append(StorageSlot(row['id'],location,row['capacity_bytes'],allocated_by=task_id,
            released_by=release,device='cpu',resource_refs=refs,budget_refs=(_ref('stage_plan','#/memory_budget_bytes'),)))
        add_task(Task(task_id,Kind.ALLOCATE,depends_on=(tasks[-1].id,) if tasks else (),allocates=(row['id'],),
            capability='CPU_ALLOCATE',resource_refs=refs,
            attrs=dict(capacity_bytes=row['capacity_bytes'],establishes_value_ready=False,
                allocation_policy=plan['allocation_policy'],source_slot_ref=_ref('stage_plan',
                    '#/workspace_slots/'+str(list(source_slots).index(row['id']))))))
    action_indices={row.id:index for index,row in enumerate(region.actions)}
    epochs={name:0 for name in plan['shadows']}
    scalar_producers={}; fence_id='success:precommit'; fence_added=False
    identity_proofs=_final_commit_proofs(region,plan)
    for stage in plan['stages']:
        kind=stage['kind']; identifier=stage['id']
        if kind=='COMMIT' and not fence_added:
            guarded=[task.id for task in tasks if task.kind in (Kind.ALLOCATE,Kind.SNAPSHOT,Kind.COMPUTE,Kind.STATE_WRITE)]
            add_task(Task(fence_id,Kind.SUCCESS_FENCE,depends_on=tuple(guarded),validates=tuple(guarded),
                capability='CPU_SUCCESS_FENCE',resource_refs=refs,
                attrs=dict(checks=['ALL_COMPUTE_FINITE_AND_FENV_SUCCESS','CPU_AND_FENV_RESTORATION_SUCCESS'],
                    failure='WITHHOLD_ALL_EXTERNAL_WRITES_ORIGINAL_ONCE',no_failing_operations_after_commit=True)))
            fence_added=True
        dependencies=list(stage['depends_on'])
        if tasks and tasks[-1].id not in dependencies: dependencies.append(tasks[-1].id)
        for scalar in stage['scalar_inputs']:
            if scalar in scalar_producers and scalar_producers[scalar] not in dependencies: dependencies.append(scalar_producers[scalar])
        reads=[current[payload] for payload in stage['input_payloads']]
        for read in stage['array_reads']:
            if read['storage']=='EXTERNAL':
                key=current['external:'+read['input_id']]
                if key not in reads: reads.append(key)
        writes=[]; source_refs=[_ref('stage_plan',f"#/stages/{stage['index']}")]
        source_refs += [_ref('region',f'#/actions/{action_indices[key]}') for key in stage['source_action_ids']]
        attrs=dict(stage_kind=kind,source_stage_ref=source_refs[0],
            original_expression_refs=[_ref('region',f'#/nodes/{next(i for i,n in enumerate(region.nodes) if n.id==key)}') for key in stage['expr_ids']],
            original_root_expression=stage['root_expr'],shape=stage['shape'],epoch=stage['epoch'],
            cpu_threads=plan['cpu_threads'],executor_id=binding.executor_id,
            runtime_validation='EXISTING_STATEFUL_BACKEND_TRANSACTION_CONTRACT')
        world_before=(); world_after=()
        if stage['output_payload'] is not None:
            payload=stage['output_payload']; slot=plan['assignment'][payload]['slot_id']
            if kind=='SNAPSHOT_ROOT':
                logical='entry:'+stage['target']; reads=[current['external:'+stage['target']]]
            else:
                logical='computed:'+payload
                values.append(Value(logical,producer=identifier,source_refs=tuple(source_refs),
                    attrs=dict(expression=stage['root_expr'],shape=stage['shape'],immutable=True)))
            writes=[material('mat:'+payload,logical,slot,sizes[payload],identifier)]
            current[payload]=writes[0]
        if kind in ('WRITE_SHADOW','COMMIT'):
            target=stage['target']; shadow=plan['shadows'][target]
            action=next(row for row in region.actions if row.id==stage['source_action_ids'][0])
            ai=action_indices[action.id]
            world_before=(_ref('region',f'#/actions/{ai}/world_before'),)
            world_after=(_ref('region',f'#/actions/{ai}/world_after'),)
            final_copy=kind=='COMMIT' and identifier==identity_proofs[target]['last_commit']
            if kind=='WRITE_SHADOW':
                epochs[target]+=1; payload=shadow; slot=plan['assignment'][shadow]['slot_id']
                logical=f'shadow-state:{target}:epoch:{epochs[target]}'
            else:
                slot='external:'+target; payload=slot
                # The full resulting content includes preserved bytes outside
                # the ROI. This is not a claim that the whole root is copied.
                reads=[current[slot],current[shadow]]
                logical=(next(row.value for row in materials if row.id==current[shadow]) if final_copy
                         else f'committed-state:{target}:{identifier}')
                if fence_id not in dependencies: dependencies.append(fence_id)
            epoch_attrs=(dict(epoch_scope='ORIGINAL_LOCAL_WRITE_ORDER',
                original_epoch_before=action.epoch_before,original_epoch_after=action.epoch_after)
                if kind=='WRITE_SHADOW' else dict(
                    epoch_scope='DEFERRED_COMMIT_OCCURRENCE_NOT_ORIGINAL_INTERMEDIATE_STATE',
                    source_original_epoch_before=action.epoch_before,source_original_epoch_after=action.epoch_after))
            if not final_copy:
                values.append(Value(logical,producer=identifier,source_refs=tuple(source_refs),
                    attrs=dict(input_id=target,**epoch_attrs)))
            size=8*plan['input_shapes'][target][0]*plan['input_shapes'][target][1]
            material_id='mat:external-final:'+target if final_copy else 'mat:'+logical
            writes=[material(material_id,logical,slot,size,identifier)]; current[payload]=writes[0]
            attrs.update(op=stage['op'],target=target,normalized_bounds=stage['normalized_bounds'],
                write_footprint='ORIGINAL_ROI_ONLY_FULL_ROOT_MATERIALIZATION_DESCRIBES_RESULT',
                unchanged_outside_roi=True)
            if kind=='COMMIT': attrs['world_scope']='SOURCE_WORLD_REFERENCES_NOT_ORIGINAL_INTERMEDIATE_EXTERNAL_CONTENT'
            if final_copy:
                pointer=target.replace('~','~0').replace('/','~1')
                attrs.update(logical_identity='VALUE_PRESERVING_FINAL_ROOT_COMMIT',
                    identity_source_material=current[shadow],
                    identity_proof_ref=dict(schedule_pointer='#/proof/final_root_commit_identity/'+pointer))
                identity_proofs[target].update(source_material=current[shadow],source_value=logical,
                    final_external_material=writes[0])
        mapped={'SNAPSHOT_ROOT':(Kind.SNAPSHOT,'CPU_HOST_SNAPSHOT'),'SCALAR':(Kind.COMPUTE,'CPU_SCALAR_SETUP'),
            'COMPUTE':(Kind.COMPUTE,'CPU_SCALAR_DAG'),'PROVIDER':(Kind.COMPUTE,'CPU_NUMPY_UFUNC_PROVIDER'),
            'WRITE_SHADOW':(Kind.STATE_WRITE,'CPU_STATE_WRITE'),'COMMIT':(Kind.COMMIT,'CPU_COMMIT')}[kind]
        if kind=='PROVIDER': attrs.update(provider_id=stage['provider_id'],provider_ref=_ref('region',
            '#/providers/'+str(next(i for i,p in enumerate(region.providers) if p.id==stage['provider_id']))),
            serial_original_native_provider=True)
        add_task(Task(identifier,mapped[0],reads=tuple(dict.fromkeys(reads)),writes=tuple(writes),
            depends_on=tuple(dependencies),capability=mapped[1],resource_refs=refs,source_refs=tuple(source_refs),
            world_before=world_before,world_after=world_after,attrs=attrs))
        if kind=='SCALAR': scalar_producers[stage['root_expr']]=identifier
    release_slots = reversed(plan['workspace_slots']) if physical_release_order else plan['workspace_slots']
    for row in release_slots:
        add_task(Task('release:'+row['id'],Kind.RELEASE,depends_on=(tasks[-1].id,),releases=(row['id'],),
            capability='CPU_RELEASE',attrs=dict(logical_last_use_ref=_ref('stage_plan','#/values'),
                physical_release='ONLY_AFTER_ALL_ORIGINAL_EXTERNAL_COMMITS_COMPLETE')))
    return VersionedResourceSchedule(region.id,tuple(values),tuple(materials),tuple(slots),tuple(tasks),
        order_constraints=tuple(orders),resource_refs=refs,budget_bytes=plan['memory_budget_bytes'],
        budget_scope=plan['budget_scope'],budget_refs=(_ref('stage_plan','#/memory_budget_bytes'),),
        proof=dict(profile=PROFILE,semantic_boundary_id=boundary['id'],fixed_stage_order=True,
            allocation_never_establishes_readiness=True,logical_release_is_not_physical_free=True,
            transaction_ref=_ref('stage_plan','#/transaction'),source_epochs_ref=_ref('stage_plan','#/state_epochs'),
            final_root_commit_identity=identity_proofs,
            **(dict(pool_release_order=CPP_POOL_RELEASE_ORDER,
                physical_release_slot_order=[row['id'] for row in reversed(plan['workspace_slots'])])
                if physical_release_order else {}),
            physical_slots_are_not_logical_values=True,whole_program_makespan_optimized=False),executable=True)


def _assemble(region,plan,artifacts,*,physical_release_order=False):
    from .schedule import schedule_document
    boundary=build_stateful_boundary(region,plan)
    binding,snapshot,inventory,target_artifacts=_target(plan)
    if any(artifacts.get(key)!=value for key,value in target_artifacts.items()):
        raise ValueError('stateful recipe retained hardware references differ')
    if {k for k in artifacts if k.startswith('target_table:')}!={k for k in target_artifacts if k.startswith('target_table:')}:
        raise ValueError('unsupported additional hardware table reference')
    capability=stateful_backend_capability(plan)
    binary=capability['provider_runtime']['loaded_binary']
    expected_provider=artifact_reference(binary['path'],'scar.numpy-native-provider-binary')
    if artifacts.get('provider_binary')!=expected_provider: raise ValueError('original NumPy provider binary reference differs')
    source_roles=[]
    for source in (region.source,*(row.source for row in region.callees)):
        roles=[key for key,row in artifacts.items() if key.startswith(('source_python:','source_artifact:'))
               and row['uri']==source.path and row['sha256']==source.sha256]
        if not roles: raise ValueError('original stateful source artifact missing')
        if roles[0] not in source_roles: source_roles.append(roles[0])
    if 'cpu_hierarchy' in artifacts:
        hierarchy=read_artifact(artifacts['cpu_hierarchy'])
        if hierarchy.get('schema')!='scar.cpu-hierarchy' or hierarchy.get('target_snapshot')!=encode(snapshot):
            raise ValueError('CPU hierarchy differs from retained target snapshot')
    refs=[encode(H.TargetResourceRef(snapshot,row.id)) for row in inventory]
    resources=dict(schema='scar.stateful-recipe-resources',schema_version=1,
        inventory=dict(snapshot_role='target_snapshot',inventory_role='target_inventory',all_resources=refs,
            topology_ref=_ref('stage_plan','#/resource_context/topology'),original_program_use_does_not_filter_resources=True),
        selected_cpu=dict(executor_id=binding.executor_id,cpu_ids=list(binding.cpu_ids),cpu_threads=plan['cpu_threads'],
            binding_ref=_ref('stage_plan','#/resource_binding'),selection='EXPLICIT_UNRANKED_ALLOCATION'),
        host_allocation=dict(location='host-address-space:$CALL_PROCESS',memory_policy='UNBOUND',
            memory_domain_id=binding.memory_domain_id,physical_numa_residency_proven=False),
        memory=dict(external_array_bytes=plan['external_array_bytes'],workspace_payload_bytes=plan['workspace_payload_bytes'],
            total_region_array_payload_bytes=plan['total_region_array_payload_bytes'],explicit_budget_bytes=plan['memory_budget_bytes'],
            budget_scope=plan['budget_scope'],budget_excludes=plan['budget_excludes'],
            slot_assignment_ref=_ref('stage_plan','#/assignment'),allocation_success_is_runtime_guard=True,
            capacity_or_free_snapshot_is_reserved_budget=False),
        cache_hierarchy=dict(artifact_role='cpu_hierarchy' if 'cpu_hierarchy' in artifacts else None,
            allocatable=False,allocation_budget_bytes=None,effective_available_bytes=None),
        unselected_resources_retained=True,unselected_devices_have_no_lowering_claim=True)
    control=dict(schema='scar.stateful-recipe-control',schema_version=1,
        action_order_ref=_ref('region','#/actions'),stage_order_ref=_ref('stage_plan','#/stages'),
        world_epoch_ref=_ref('stage_plan','#/state_epochs'),fixed_source_order=True,
        spatial_partition='OPENMP_STATIC_DISJOINT_ROWS' if plan['cpu_threads']>1 else 'SERIAL_ROWS',
        provider_partition='SERIAL_ORIGINAL_NUMPY_STRIDED_LOOP_ROWS',
        transaction='ALL_PRIVATE_COMPUTE_AND_RESTORATION_SUCCEED_BEFORE_EXTERNAL_ROI_COMMIT',
        iteration_expansion=False,shared_body_instances=1)
    if physical_release_order: control['pool_release_order']=CPP_POOL_RELEASE_ORDER
    implementation=dict(family=capability['family'],backend_inputs=dict(region='region',plan='stage_plan'),
        backend_capability_record=capability,source_artifact_roles=source_roles,
        document_sha256=dict(region=semantic_sha256(read_artifact(artifacts['region'])),
            stage_plan=semantic_sha256(read_artifact(artifacts['stage_plan']))),
        lowering='EXACT_PUBLIC_STAGE_PLAN_TO_EXISTING_TRANSACTIONAL_CPP20_BACKEND',
        numerical_contract_ref='#/semantic_boundary/Q',program_body_copies=1)
    obligations=[dict(id='closed_semantic_boundary',status='PROVEN',detail='Original model, epochs, sources and emitted stage lifetimes replayed within the guarded profile.'),
        dict(id='runtime_state_and_scalars',status='RUNTIME_GUARD',detail='Exact receiver/source/provider/array/alias/scalar/shape guards remain mandatory.'),
        dict(id='transaction_success',status='RUNTIME_GUARD',detail='Withhold external writes until all compute, FP environment and CPU restoration succeeds; fallback once.'),
        dict(id='cpu_and_payload',status='RUNTIME_GUARD',detail='Selected CPU IDs remain available and all fixed slots allocate within the declared payload scope.'),
        dict(id='provider_capsule',status='LOWERING_CHECK',detail='Existing backend must acquire exact original NumPy native loop/capsule with requires_pyapi false.'),
        dict(id='compiler_and_backend',status='LOWERING_CHECK',detail='Native backend/ABI/compiler/provider binary must equal the recorded capability.')]
    obligations += [dict(id='external:'+str(i),status='EXTERNAL_ASSUMPTION',detail=item)
                    for i,item in enumerate(plan['external_assumptions'])]
    return make_recipe(PROFILE,semantic_boundary=boundary,control=control,
        task_graph=schedule_document(_graph(region,plan,boundary,snapshot,physical_release_order=physical_release_order)),resources=resources,
        implementation=implementation,artifacts=artifacts,obligations=obligations)


def make_stateful_recipe(region,plan,*,source_artifacts=(),cpu_hierarchy=None,out):
    region,plan=admit_stateful_inputs(region,plan)
    out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh stateful recipe directory required')
    artifacts=dict(region=_save(out/'components/region.json',document(region)),
        stage_plan=_save(out/'components/stage-plan.json',plan))
    extras=source_artifacts.values() if isinstance(source_artifacts,dict) else source_artifacts
    for i,reference in enumerate(extras):
        reference=encode(reference);read_artifact(reference,json_document=False);artifacts['source_artifact:'+str(i)]=reference
    for source in (region.source,*(row.source for row in region.callees)):
        if any(row['uri']==source.path for row in artifacts.values()): continue
        reference=artifact_reference(source.path,'scar.python-source')
        if reference['sha256']!=source.sha256: raise ValueError('original stateful source snapshot differs')
        artifacts['source_python:'+str(len(artifacts))]=reference
    artifacts.update(_target(plan)[3])
    capability=stateful_backend_capability(plan)
    artifacts['provider_binary']=artifact_reference(capability['provider_runtime']['loaded_binary']['path'],'scar.numpy-native-provider-binary')
    if cpu_hierarchy is not None:
        artifacts['cpu_hierarchy']=(artifact_reference(cpu_hierarchy,'scar.cpu-hierarchy')
            if isinstance(cpu_hierarchy,(str,Path)) else encode(cpu_hierarchy))
    recipe=_assemble(region,plan,artifacts,physical_release_order=True);write_recipe(recipe,out/'recipe.json')
    return recipe


def _parts(recipe):
    artifacts=recipe['artifacts']
    required={'region','stage_plan','target_snapshot','target_inventory','provider_binary'}
    if not required<=artifacts.keys(): raise ValueError('complete stateful recipe artifacts required')
    if any(key not in required|{'cpu_hierarchy'} and not key.startswith(('source_python:','source_artifact:','target_table:')) for key in artifacts):
        raise ValueError('unsupported stateful recipe artifact role')
    return admit_stateful_inputs(stateful_from_dict(read_artifact(artifacts['region'])),read_artifact(artifacts['stage_plan']))


def verify_stateful_recipe(value):
    recipe=read_recipe(value) if isinstance(value,(str,Path)) else validate_recipe(value)
    if recipe['profile']!=PROFILE: raise ValueError('supported stateful recipe profile required')
    release_order=recipe['control'].get('pool_release_order')
    if 'pool_release_order' in recipe['control'] and release_order!=CPP_POOL_RELEASE_ORDER:
        raise ValueError('unsupported stateful physical pool release order')
    expected=_assemble(*_parts(recipe),recipe['artifacts'],physical_release_order=release_order==CPP_POOL_RELEASE_ORDER)
    if semantic_sha256(recipe)!=semantic_sha256(expected):
        raise ValueError('stateful recipe differs from complete source/epoch/slot/transaction replay')
    return recipe


def unpack_stateful_recipe(value): return _parts(verify_stateful_recipe(value))


def estimate_stateful_recipe(value,task_costs=None):
    """Replay independent direct-task costs, never split a complete-call observation."""
    recipe=verify_stateful_recipe(value)
    from scar.optimization.scope_accounting import _fact, _profile
    context=dict(recipe_id=recipe['recipe_id'],task_graph_sha256=semantic_sha256(recipe['task_graph']),
        backend_capability_sha256=semantic_sha256(recipe['implementation']['backend_capability_record']),
        budget_scope=recipe['resources']['memory']['budget_scope'])
    issues=[]; parsed={}; costs={}; tick=None; source=None; matched_sources=set()
    def issue(field,reason): issues.append(dict(field=field,reason=reason))
    if task_costs is not None:
        try: parsed,source=_profile(task_costs)
        except (OSError,ValueError,TypeError,KeyError) as error: issue('task_costs',str(error))
    if task_costs is not None:
        if not isinstance(parsed,dict): parsed={}
        if (parsed.get('schema'),parsed.get('schema_version'))!=('scar.recipe-task-costs',1) or type(parsed.get('schema_version')) is not int:
            issue('task_costs.schema','scar.recipe-task-costs v1 required')
        if set(parsed)-{'schema','schema_version','context','context_evidence','tick_seconds','task_costs','model_assumptions'}:
            issue('task_costs','unsupported cost constraints cannot be ignored')
        if parsed.get('context')!=context: issue('task_costs.context','exact recipe/graph/backend/budget context required')
        references=parsed.get('context_evidence',[])
        if not isinstance(references,list): references=[];issue('context_evidence','original FactReference list required')
        for i,reference in enumerate(references):
            try:
                observed,artifact=_fact(reference)
                if observed!=context: raise ValueError('original cost context differs')
                matched_sources.add(artifact)
            except (OSError,ValueError,TypeError,KeyError,IndexError) as error: issue('context_evidence.'+str(i),str(error))
        if not matched_sources: issue('context_evidence','original matching cost context FactReference required')
    def quantity(record,field,unit,key,positive=False,task_id=None):
        if not isinstance(record,dict): issue(field,'unknown cost is not zero');return None
        if set(record)-{key,'unit','basis','evidence','measurement_scope','task_id'}:
            issue(field,'unsupported cost constraints cannot be ignored');return None
        if task_id is not None and record.get('task_id')!=task_id:
            issue(field+'.task_id','quantity must identify this exact recipe task');return None
        number=record.get(key);basis=record.get('basis')
        valid=(type(number) in (int,float) and (type(number) is int or math.isfinite(number))
               and number>=0 and (not positive or number>0))
        if unit=='ticks': valid=valid and type(number) is int
        if not valid or record.get('unit')!=unit or basis not in ('OBSERVED','DECLARED_MODEL'):
            issue(field,'explicit sourced nonnegative ticks or positive seconds/tick with basis required');return None
        refs=record.get('evidence')
        if not isinstance(refs,list) or not refs: issue(field+'.evidence','original matching quantity required');return None
        for i,reference in enumerate(refs):
            try:
                observed,artifact=_fact(reference)
                if artifact not in matched_sources: raise ValueError('cost source lacks matching recipe context')
                if (not isinstance(observed,dict) or type(observed.get(key)) is not type(number) or observed.get(key)!=number
                        or observed.get('unit')!=unit or observed.get('basis')!=basis): raise ValueError('source quantity/unit/basis differs')
                if task_id is not None and observed.get('task_id')!=task_id:
                    raise ValueError('source task identity differs')
                if basis=='OBSERVED' and observed.get('measurement_scope')!='DIRECT_TASK_TIMING':
                    raise ValueError('direct task timing required; whole-call totals cannot be split')
            except (OSError,ValueError,TypeError,KeyError,IndexError) as error: issue(field+'.evidence.'+str(i),str(error));return None
        return number
    if task_costs is not None: tick=quantity(parsed.get('tick_seconds'),'tick_seconds','seconds/tick','value',True)
    assumptions=parsed.get('model_assumptions',[])
    if not isinstance(assumptions,list) or any(type(row) is not str or not row for row in assumptions):
        issue('model_assumptions','explicit nonempty-string assumption list required');assumptions=[]
    if task_costs is not None and not assumptions:
        issue('model_assumptions','sourced task composition assumptions must remain explicit')
    rows=parsed.get('task_costs',{}) if isinstance(parsed,dict) else {}
    if not isinstance(rows,dict): rows={};issue('task_costs','task ID mapping required')
    task_ids={row['id'] for row in recipe['task_graph']['tasks']}
    if rows.keys()-task_ids: issue('task_costs','unknown task IDs cannot be ignored')
    for task in recipe['task_graph']['tasks']:
        duration=quantity(rows.get(task['id']),'task_costs.'+task['id'],'ticks','duration_ticks',task_id=task['id'])
        costs[task['id']]=dict(status='UNKNOWN' if duration is None else 'SOURCED',duration_ticks=duration,
            evidence=deepcopy(rows.get(task['id'])))
    complete=not issues and tick is not None
    ends={}; timed=[]
    if complete:
        # This profile deliberately emits a fixed total order. This is a replay
        # of that order, not a placement search or a physical optimum claim.
        for task in recipe['task_graph']['tasks']:
            start=max((ends[key] for key in task['depends_on']),default=0)
            end=start+costs[task['id']]['duration_ticks'];ends[task['id']]=end
            timed.append(dict(task=task['id'],start_ticks=start,end_ticks=end))
    memory=deepcopy(recipe['resources']['memory'])
    return dict(schema='scar.stateful-recipe-estimate',schema_version=1,recipe_id=recipe['recipe_id'],
        status='FIXED_ORDER_COST_MODEL_REPLAYED' if complete else 'UNKNOWN',scope='FIXED_ORDER_EMITTED_STAGE_TASKS',
        timing_coverage=dict(semantic_boundary='ONE_STATEFUL_METHOD_CALL',
            whole_native_call_cost_verified=False,whole_python_boundary_cost_verified=False,
            additional_implementation_work=['PYTHON_BINDING_AND_LIVE_INPUT_GUARDS',
                'NATIVE_METADATA_AND_FINITE_INPUT_PREFLIGHT','CALL_AND_PROVIDER_DISPATCH'],
            note='Task-source costs alone do not prove coverage of all boundary/implementation overhead.'),
        objective='COMPOSE_TOWARD_FULL_PROGRAM_MAKESPAN',payload=dict(status='EXACT_UNDER_SHAPE_CONTRACT',**memory),
        timing=dict(status='ESTIMATED' if complete else 'UNKNOWN',task_costs=costs,
            duration_ticks=max(ends.values(),default=0) if complete else None,tick_seconds=tick,
            fixed_order_timeline=timed),cost_source=source,cost_context=context,unresolved=issues,
        model_assumptions=deepcopy(assumptions),whole_call_observation_split=False,
        placement_jointly_optimized=False,makespan_optimality_proven=False,physical_optimality_proven=False,
        target_executed=False,deployment_authorized=False)


plan_stateful_recipe=make_stateful_recipe
__all__=['PROFILE','make_stateful_recipe','plan_stateful_recipe','verify_stateful_recipe',
    'unpack_stateful_recipe','estimate_stateful_recipe','stateful_backend_capability']
