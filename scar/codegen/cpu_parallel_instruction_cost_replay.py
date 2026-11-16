"""Independent static replay of exact-mask parallel instruction costs.

The frozen serial replay supplies only generic compiler/MCA receipt checking;
this reader uses its own parallel model schema, registry, barrier composition
and exact worker mask. No target or calibration timing is rerun.
"""
from copy import deepcopy
from . import cpu_instruction_cost as C
from . import cpu_parallel_instruction_cost as P
from .cpu_instruction_cost_replay import _raw,_replay_compiler,_replay_analysis as _generic_replay_analysis
from .cpu_fused_cost import _require
from scar.ir.execution_recipe import artifact_reference,read_artifact


def _replay_analysis(assembly,analysis,tool,**kwargs):
    _require(analysis['instruction_lexer']==P._lexer_evidence(assembly),
        'instruction lexical source/rule/implementation differs')
    base={k:v for k,v in analysis.items() if k!='instruction_lexer'}
    result=_generic_replay_analysis(P._lex_assembly(assembly),base,tool,**kwargs)
    _require(result==base,'normalized natural instruction-loop replay differs')
    return analysis


def verify_parallel_instruction_projection(domain,candidate_id,projection):
    """Return exact admitted component rows after static source/tool replay."""
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand,component_key
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region
    from scar.codegen import stateful

    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id)
    reference=artifact_reference(projection,'scar.stateful-resource-calibration');document=read_artifact(reference)
    _require(document['schema_version']==1 and document['resource_context']==demand['resource_context']
        and document.get('target_workload_executed') is False and document.get('target_variants_ranked') is False
        and document.get('target_dag_compiled_for_static_analysis') is True,'projection context/authority differs')
    model_ref=document['source_model'];read_artifact(model_ref)
    checked_ref,model,raw,calibrated=P._verify_model(model_ref['uri'])
    _require(checked_ref==model_ref and model['resource_context']==demand['resource_context'],'canonical model/context differs')
    tool=model['tool'];cache={};assembly=_raw(model['native']['assembly'])
    for key,analysis in calibrated.items():
        case=P._registry()[int(key)//2];group=int(key)%2+1
        _replay_analysis(assembly,analysis,tool,function=case['id']+'_g'+str(group),cache=cache)

    lowering_ref=document['static_lowering'];lowering=read_artifact(lowering_ref)
    _require(lowering['schema']=='scar.stateful-static-instruction-lowering' and lowering['schema_version']==1
        and lowering['target_dag_compiled_for_static_analysis'] is True
        and lowering['target_workload_executed'] is False and lowering['target_variants_ranked'] is False
        and lowering['native_library_built'] is False,'static lowering scope differs')
    _require(lowering['region']==domain['region'] and lowering['compiler']==demand['resource_context']['backend']['compiler']
        and lowering['backend_implementation']==stateful.implementation_sha256(),'static source/backend dependency differs')
    region=stateful_from_dict(read_artifact(domain['region']))
    candidate=next(r for r in domain['candidates'] if r['id']==candidate_id)
    binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    placement=CpuPlacement(**binding)
    plan=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],cpu_threads=len(placement.cpu_ids),
        memory_budget_bytes=placement.memory_budget_bytes,resource_binding=placement)
    region,plan,placement=stateful._admit(region,plan,placement)
    _require(plan==lowering['plan'] and lowering['resource_binding']==plan['resource_binding'],'static stage plan or placement differs')
    contract=dict(region_sha256=stateful_digest(region),plan_sha256=stateful._digest(plan),
        provider_runtime=demand['resource_context']['backend']['provider_runtime'])
    source,assembly=_replay_compiler(lowering)
    _require(source==stateful._source(region,plan,placement,contract),'original backend source differs from source/plan replay')
    order={n.id:i for i,n in enumerate(region.nodes)};terms=[r for r in demand['terms'] if r['component']=='FUSED_SCALAR_DAG_ROW_STAGE']
    _require(len(document['components'])==len(terms) and not document['unsupported_components']
        and not document['proposed_estimates'] and document['all_fused_components_covered'] is True,
        'complete admitted fused projection required')
    maximum_stack=max(r['stack_memory_references'] for a in calibrated.values() for r in a['loops'])
    checked=[]
    for index,(row,term) in enumerate(zip(document['components'],terms)):
        p=term['parameters'];stage_index=int(term['source_ref']['pointer'].rsplit('/',1)[1]);stage=plan['stages'][stage_index]
        _require(stage['kind']=='COMPUTE' and p['workers']>1,'unsupported stage boundary')
        output=plan['assignment'][stage['output_payload']]['slot_id']
        inputs=[plan['assignment'][k]['slot_id'] for k in stage['input_payloads'] if k in plan['assignment']]
        _require(output not in inputs,'static output allocation aliases live input')
        expected=C.V2._layouts(p['array_loads'],p['cols'],p['backing_storage_groups']);sort=lambda r:(r['storage_group'],r['stride'],r['offset'])
        _require(sorted(p['operand_layouts'],key=sort)==sorted(expected,key=sort),'memory proxy geometry differs')
        capacity=model['cache_geometry']['reported_capacity_bytes'];_,geometry=C.V2.fused_features(p,capacity)
        _require(256<=p['rows']<=768 and 320<=p['cols']<=832
            and geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,'memory proxy applicability differs')
        span=C._stage_source_range(source,order[stage['root_expr']])
        analysis=_replay_analysis(assembly,row['instruction_analysis'],tool,source_range=span,
            source_path=lowering['source']['path'],cache=cache)
        _require(max(r['stack_memory_references'] for r in analysis['loops'])<=maximum_stack,'spill scope differs')
        qualified=model['local_models'][str(p['backing_storage_groups'])]
        _require(qualified['admitted'] is True,'fresh holdout does not authorize this component')
        cost=P._predict(dict(groups=p['backing_storage_groups'],rows=p['rows'],cols=p['cols'],workers=p['workers'],
            elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],array_loads=p['array_loads']),analysis,model['coefficients'])
        expected_row=dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=cost['duration_ns'],
            prediction=cost,source_model=dict(artifact=model_ref,locator='#/coefficients'),
            source_validation=dict(artifact=model_ref,locator='#/local_models/'+str(p['backing_storage_groups'])+'/holdout'),
            source_provenance=term['source_ref'],instruction_analysis=analysis,static_lowering=lowering_ref,
            workspace_alias_authority=dict(output_slot=output,input_slots=inputs,live_slots_distinct=True),
            precision_policy=P._POLICY,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
            target_dag_compiled_for_static_analysis=True,measured_target_component=False,
            physical_optimality_proven=False,assumptions=P._ASSUMPTIONS)
        _require(row==expected_row,'projected component estimate, parameters or dependencies differ')
        checked.append(dict(index=index,key=component_key(row['component'],row['parameters']),
            fact=dict(artifact=reference,locator=f'#/components/{index}'),component=deepcopy(row)))
    return dict(schema='scar.cpu-parallel-instruction-cost-projection-replay',schema_version=1,
        projection=reference,source_profile=model_ref,static_lowering=lowering_ref,
        frozen_design=model['frozen_design'],raw_observations=model['raw_observations'],instruction_analysis=model['instruction_analysis'],
        tool=tool['executable'],openmp_runtime=tool['openmp_runtime'],
        compiler_openmp_runtime=tool['compiler_openmp_runtime'],partition_witnesses=model['partition_witnesses'],
        qualification_design=model.get('qualification_design'),components=checked,
        compiler_reexecuted_static_only=True,mca_reexecuted_static_only=True,
        calibration_library_loaded_for_runtime_symbol_identity_only=True,
        timing_observations_remeasured=False,target_workload_executed=False,target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=True,model_authority='DECLARED_FORWARD_COMPONENT_COST_NOT_PHYSICAL_BOUND')
