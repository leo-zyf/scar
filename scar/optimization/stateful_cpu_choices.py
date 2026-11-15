"""Finite CPU-mask/worker choices for one unchanged stateful method call.

The domain is hardware-derived and includes every worker count per supported
executor. It is not every CPU subset, a process/MPI implementation, or a timing
experiment. Unknown costs remain options and prevent full-domain selection.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import sys

from scar.compiler.stateful_model import document,read_stateful,stateful_from_dict,stateful_digest
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode,loads
from .deployment_cost import _fact,_profile
from .hardware import _target_materials,_affinity,_boundaries,propose_cpu_allocations,validate_cpu_placement
from .implementation_solver import solve_implementation_choices,task_implementation_contract
from .resource_solver import semantic_sha256
from .schedule import ResourceSchedule,Task,TaskKind
from .stateful_recipe import stateful_backend_capability
from .stateful_schedule import plan_stateful_region,verify_stateful_plan

_SCOPE=dict(boundary='WARM_SESSION',call_scope='ONE_GUARDED_STATEFUL_METHOD_CALL',reuse_count=1,
    includes='GUARDS_DISPATCH_ALL_ARRAY_ALLOCATION_STAGES_RESTORATION_COMMIT_RELEASE',
    excludes='PROCESS_START_IMPORT_COMPILE_INSTALL_ONE_TIME_LOAD')


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(path,value['schema'])


def _family(plan):
    return {key:value for key,value in plan.items() if key not in ('cpu_threads','resource_binding','resource_context')}


def _construct(region_ref,target,request,base_plan_ref):
    from scar.codegen.capabilities import query_stateful_capability,retain_stateful_resources
    region=stateful_from_dict(read_artifact(region_ref));base=verify_stateful_plan(region,read_artifact(base_plan_ref))
    for source in (region.source,*(row.source for row in region.callees)):
        if sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256: raise ValueError('original stateful source differs')
    graph,snapshot,resources,connections=_target_materials(target)
    inventory_ref=artifact_reference(graph.report['inventory']['uri'],'scar.target-inventory')
    fact=lambda pointer:dict(artifact=inventory_ref,locator=pointer)
    topology=graph.inventory.get('cpu_topology',{})
    hardware_context=dict(execution_constraints=dict(
        raw=graph.inventory.get('execution_constraints'),
        evidence=fact('#/execution_constraints') if 'execution_constraints' in graph.inventory else None,
        status='RETAINED_SNAPSHOT_NOT_RESERVATION' if 'execution_constraints' in graph.inventory else 'UNKNOWN'),
        cpu_isa=dict(evidence=fact('#/cpu_topology/isa_groups') if 'isa_groups' in topology else None,
            group_count=len(topology.get('isa_groups',[])),binary_isa_admission_proven=False),
        cpu_core_smt=dict(evidence=fact('#/cpu_topology/processors') if topology.get('processors') else fact('#/cpu/processors'),
            grouping='RETAINED_PACKAGE_DIE_CORE_IDS_WITHIN_EACH_EXECUTOR'),
        cpu_cache=dict(evidence=fact('#/cpu_topology/caches') if 'caches' in topology else None,
            resource_ids=[str(r.id) for r in resources if type(r).__name__=='CacheDomain'],
            allocation_budget=False,measured_effective_capacity=False),
        availability=encode(graph.availability()),inventory_platform=graph.inventory.get('platform',{}),
        runtime_abi=dict(pointer_bits=8*struct.calcsize('P'),byteorder=sys.byteorder,
            authority='CURRENT_PLANNER_PROCESS_REQUIRES_MATCH_AT_EXECUTION'),
        quota_is_worker_cap=False,quota_is_measured_throughput=False,
        topology_or_capacity_is_task_cost=False,hardware_filtered_by_idle_or_access=False)
    allowed,origin=_affinity(request['affinity'])
    seeds=propose_cpu_allocations(target,affinity=allowed,memory_budget_bytes=request['memory_budget_bytes'],
                                  max_threads=request['max_threads'])
    largest={}
    for binding in seeds:
        if binding.executor_id not in largest or len(binding.cpu_ids)>len(largest[binding.executor_id].cpu_ids):
            largest[binding.executor_id]=binding
    capabilities={key:stateful_backend_capability({**base,'cpu_threads':threads})
                  for key,threads in (('serial',1),('parallel',2))}
    processors=topology.get('processors') or graph.inventory['cpu']['processors'];rows=[];gaps=[]
    for executor_id,seed in sorted(largest.items()):
        validate_cpu_placement(seed)
        executor=next(r for r in resources if str(r.id)==executor_id)
        cpus=sorted(set(executor.cpu_ids)&set(allowed));groups={};missing=[]
        for cpu in cpus:
            p=next((p for p in processors if p['cpu_id']==cpu),None)
            if p is None or p.get('package_id') is None or p.get('core_id') is None: missing.append(cpu)
            else: groups.setdefault((p['package_id'],p.get('die_id'),p['core_id']),[]).append(cpu)
        orders={'LOGICAL_ID_PREFIX':cpus}
        if not missing:
            groups=[sorted(group) for _,group in sorted(groups.items(),key=lambda pair:
                (pair[0][0],-1 if pair[0][1] is None else pair[0][1],pair[0][2]))]
            orders['CORE_FIRST_THEN_SMT']=[group[level] for level in range(max(map(len,groups),default=0))
                                          for group in groups if level<len(group)]
            orders['CORE_GROUPED_SMT_FIRST']=[cpu for group in groups for cpu in group]
        else: gaps.append(dict(field=executor_id+'.core_smt',reason='retained package/core IDs incomplete',cpu_ids=missing))
        masks={};limit=len(seed.cpu_ids)
        for policy,order in orders.items():
            for count in range(1,limit+1): masks.setdefault(tuple(sorted(order[:count])),[]).append(policy)
        for mask,policies in sorted(masks.items(),key=lambda pair:(len(pair[0]),pair[0])):
            # All other receipt-bearing fields remain the validated seed's.
            # A sorted subset cannot add CPUs, alter memory or change target.
            binding=replace(seed,cpu_ids=mask)
            identifier='cpu-'+semantic_sha256(dict(executor=executor_id,cpu_ids=mask))[:20]
            issues=[]
            if binding.memory_budget_bytes!=request['memory_budget_bytes']:
                issues.append('effective hardware-domain budget differs from explicit region budget')
            if base['status']!='READY': issues.append('numeric stage plan is not budget-feasible and complete')
            capability=capabilities['serial' if len(mask)==1 else 'parallel']
            applicability=query_stateful_capability(dict(executor_kind='CPU',
                architecture=hardware_context['inventory_platform'].get('machine'),
                os=hardware_context['inventory_platform'].get('system'),
                pointer_bits=hardware_context['runtime_abi']['pointer_bits'],byteorder=sys.byteorder,
                required_isa='BASELINE_X86_64',memory_kind='HOST',memory_policy='UNBOUND',
                dtype='float64',rank=2,layout='C_CONTIGUOUS',sync_mode='SYNCHRONOUS',
                cpu_ids=list(mask),executor_cpu_ids=list(executor.cpu_ids),available_cpu_ids=list(allowed),
                cpu_threads=len(mask),operations=sorted({n.op for n in region.nodes}),
                write_operations=sorted({a.op for a in region.actions if a.kind=='write'}),
                slice_semantics='FIXED_UNIT_STEP_2D_BOUNDS',provider_runtime=capability['provider_runtime'],
                memory_budget_bytes=request['memory_budget_bytes'],
                required_payload_bytes=base['total_region_array_payload_bytes']),implementation=capability)
            issues += [f"capability {row['field']}: {row['reason']}" for row in
                       applicability['unsupported']+applicability['unresolved']]
            rows.append(dict(id=identifier,binding=encode(binding),worker_count=len(mask),
                representative_policies=policies,numa_node=binding.numa_node,
                legal_under_guards=not issues,legality_frontier=issues,
                capability_applicability=applicability,lowering_admitted=False,
                machine_fp_isa_admission='UNDISCHARGED_LOWERING_AND_RUNTIME_OBLIGATIONS',
                payload_bytes=base['total_region_array_payload_bytes']))
    inventory=[encode(H.TargetResourceRef(snapshot,r.id)) for r in resources]
    return dict(schema='scar.stateful-cpu-domain',schema_version=1,region=region_ref,base_stage_plan=base_plan_ref,
        target_path=str(graph.path),target_snapshot=encode(snapshot),
        inventory=inventory_ref,hardware_context=hardware_context,
        all_resources=inventory,unsupported_backend_boundary=_boundaries(resources,connections),
        retained_resource_roles=retain_stateful_resources(resources),
        backend_capabilities=capabilities,
        request=request,live_constraints=dict(affinity_cpu_ids=list(allowed),origin=origin,
            process_affinity_cpu_ids=list(_affinity(None)[0]),
            native_thread_environment={key:os.environ.get(key) for key in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            affinity_is_reservation=False,memory_capacity_is_available_budget=False),
        semantic_family_sha256=semantic_sha256(_family(base)),region_sha256=stateful_digest(region),
        candidates=rows,topology_frontier=gaps,
        enumeration=dict(worker_counts='EVERY_INTEGER_1_THROUGH_EACH_EXECUTOR_CAP',
            masks='DEDUPLICATED_LOGICAL_PREFIX_CORE_FIRST_AND_CORE_GROUPED_PREFIXES',
            all_cpu_subsets_enumerated=False,cross_executor_cpu_teams_supported=False,
            unsupported=['MULTIPROCESS','MPI','GPU_OFFLOAD','PHYSICAL_NUMA_MEMORY_BINDING'],
            optimality_scope='EXPLICIT_REPRESENTATIVE_MASK_FAMILY_OF_CURRENT_CPP_OPENMP_BACKEND_ONLY'),
        original_no_op=dict(id='NO_OP',available=True,cost_known=False),
        target_variants_compiled=False,target_variants_executed=False,physical_optimality_proven=False)


def build_stateful_cpu_domain(region,target,*,input_shapes,memory_budget_bytes,affinity=None,max_threads=None,out):
    """Persist one source graph and enumerate all W in the supported mask family."""
    if type(memory_budget_bytes) is not int or memory_budget_bytes<8: raise ValueError('explicit payload budget of at least 8 bytes required')
    out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh stateful CPU domain directory required')
    region=read_stateful(region) if isinstance(region,(str,Path)) else region
    allowed,_=_affinity(affinity)
    proposals=propose_cpu_allocations(target,affinity=allowed,memory_budget_bytes=memory_budget_bytes,max_threads=max_threads)
    if not proposals: raise ValueError('no supported CPU/host allocation from original target and live constraints')
    seed=proposals[0]
    plan=plan_stateful_region(region,input_shapes=input_shapes,cpu_threads=len(seed.cpu_ids),
        memory_budget_bytes=memory_budget_bytes,resource_binding=seed)
    region_ref=_write(out/'region.json',document(region));plan_ref=_write(out/'base-stage-plan.json',plan)
    request=dict(input_shapes=plan['input_shapes'],memory_budget_bytes=memory_budget_bytes,
        affinity=list(allowed),max_threads=max_threads)
    domain=_construct(region_ref,target,request,plan_ref)
    domain['domain_id']='stateful-cpu-domain:'+semantic_sha256(domain)
    _write(out/'domain.json',domain)
    return domain


def _domain(value):
    value=loads(Path(value).read_bytes()) if isinstance(value,(str,Path)) else deepcopy(value)
    if (value.get('schema'),value.get('schema_version'))!=('scar.stateful-cpu-domain',1): raise ValueError('stateful CPU domain required')
    expected=_construct(value['region'],value['target_path'],value['request'],value['base_stage_plan'])
    expected['domain_id']='stateful-cpu-domain:'+semantic_sha256(expected)
    if semantic_sha256(value)!=semantic_sha256(expected): raise ValueError('stateful CPU domain differs from retained hardware/source replay')
    return value


def stateful_cpu_cost_context(domain,candidate_id):
    """Exact context authority for future independent whole-call forward models."""
    base=read_artifact(domain['base_stage_plan'])
    row=next((row for row in domain['candidates'] if row['id']==candidate_id),None)
    if candidate_id!='NO_OP' and row is None: raise ValueError('unknown stateful CPU implementation ID')
    variant='serial' if row is None or row['worker_count']==1 else 'parallel'
    backend=domain['backend_capabilities'][variant]
    binding=None if row is None else {key:value for key,value in row['binding'].items() if key!='evidence'}
    shared=lambda pointer:dict(domain_id=domain['domain_id'],pointer=pointer)
    return dict(region_sha256=domain['region_sha256'],semantic_family_sha256=domain['semantic_family_sha256'],
        input_shapes=domain['request']['input_shapes'],memory_budget_bytes=domain['request']['memory_budget_bytes'],
        budget_scope=base['budget_scope'],scope=deepcopy(_SCOPE),candidate_id=candidate_id,
        target_sha256=domain['target_snapshot']['sha256'],
        cpu_binding=binding,
        live_constraints_sha256=semantic_sha256(domain['live_constraints']),live_constraints_ref=shared('#/live_constraints'),
        hardware_context_sha256=semantic_sha256(domain['hardware_context']),hardware_context_ref=shared('#/hardware_context'),
        backend=dict(family='ORIGINAL_PYTHON_NUMPY' if row is None else backend['family'],
            descriptor_sha256=semantic_sha256(backend),descriptor_ref=shared('#/backend_capabilities/'+variant)),
        guards=base['guard_contract'],external_assumptions=base['external_assumptions'])


def _cost(profile,context):
    """Validate an already-derived, sourced whole-call estimate; do not fit costs."""
    if profile is None: return None,['missing exact-context whole-stateful-call forward cost'],None
    issues=[]
    try:
        profile,source=_profile(profile)
        if (profile.get('schema'),profile.get('schema_version'))!=('scar.stateful-cpu-cost-profile',1): raise ValueError('stateful whole-call cost profile required')
        if set(profile)-{'schema','schema_version','context','context_evidence','estimate','derivation'}: raise ValueError('unsupported cost constraints')
        if semantic_sha256(profile['context'])!=semantic_sha256(context): raise ValueError('full scope/CPU/backend/source/shape/budget context differs')
        context_artifacts=set()
        for ref in profile['context_evidence']:
            observed,artifact=_fact(ref)
            if semantic_sha256(observed)!=semantic_sha256(context): raise ValueError('source cost context differs')
            context_artifacts.add(artifact)
        if not context_artifacts: raise ValueError('original cost context evidence required')
        estimate=profile['estimate']
        if (type(estimate['duration_ticks']) is not int or estimate['duration_ticks']<0
                or type(estimate['tick_seconds']) not in (int,float) or not 0<estimate['tick_seconds']<float('inf')):
            raise ValueError('finite sourced integer ticks and positive common tick_seconds required')
        matches=False
        for ref in estimate['evidence']:
            observed,artifact=_fact(ref)
            if (artifact not in context_artifacts or not isinstance(observed,dict)
                    or type(observed.get('duration_ticks')) is not int
                    or any(observed.get(k)!=estimate[k] for k in ('duration_ticks','tick_seconds'))):
                raise ValueError('source complete-call estimate or units differ')
            matches=True
        derivation=profile['derivation']
        if (not matches or derivation.get('origin') not in ('INDEPENDENT_PRIMITIVE_COMPOSITION','DECLARED_FORWARD_MODEL')
                or derivation.get('target_variant_trials') is not False or not derivation.get('assumptions')):
            raise ValueError('explicit independent/declared forward derivation and no target variant trials required')
        expected={k:v for k,v in derivation.items() if k!='evidence'}
        if not any(value==expected and artifact in context_artifacts
                   for value,artifact in (_fact(ref) for ref in derivation.get('evidence',[]))):
            raise ValueError('original context-bound derivation evidence required')
        return dict(duration_ticks=estimate['duration_ticks'],tick_seconds=estimate['tick_seconds'],
            derivation=derivation,source=source),[],profile
    except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError) as error: issues.append(str(error))
    return None,issues,profile


def _evaluate(domain,profiles,no_op_profile,out,write,search_budget):
    base=read_artifact(domain['base_stage_plan'])
    semantic=ResourceSchedule(domain['region_sha256'],(),(),(Task('stateful-call',TaskKind.COMPUTE,
        attrs=dict(semantic_family_sha256=domain['semantic_family_sha256'],scope=_SCOPE)),),
        proof=dict(external_assumptions=base['external_assumptions']))
    contract=semantic_sha256(task_implementation_contract(semantic,'stateful-call'))
    candidates=domain['candidates']+[dict(id='NO_OP',legal_under_guards=True,binding=None,legality_frontier=[])]
    records={};options=[];rows=[];caps={};ticks=None
    for row in candidates:
        identifier=row['id'];context=stateful_cpu_cost_context(domain,identifier)
        cost,issues,profile=_cost(no_op_profile if identifier=='NO_OP' else profiles.get(identifier),context)
        issues=list(row['legality_frontier'])+issues
        if cost is not None:
            if ticks is None: ticks=cost['tick_seconds']
            elif cost['tick_seconds']!=ticks: issues.append('cost tick basis differs across domain')
        binding=row['binding'];demands={} if binding is None else {binding['executor_id']:len(binding['cpu_ids'])}
        if binding is not None: caps.setdefault(binding['executor_id'],set()).update(binding['cpu_ids'])
        conditions=['UNCHANGED_ORIGINAL_STATEFUL_NUMERIC_DAG_AND_SOURCE_EFFECTS',
            'ONE_GUARDED_WARM_METHOD_CALL_COMPLETE_COST_SCOPE','LIVE_CPU_AND_EXCLUSIVE_STATE_RUNTIME_GUARDS',
            'EXPLICIT_REGION_PAYLOAD_BUDGET_AND_FIXED_HOST_UNBOUND_BACKEND'] if binding is not None else [
                'UNCHANGED_ORIGINAL_SOURCE_EXECUTION_REMAINS_AVAILABLE',
                'MATCHED_GUARDED_WARM_METHOD_CALL_COST_SCOPE',
                'ORIGINAL_RUNTIME_AFFINITY_AND_THREAD_ENVIRONMENT_MATCH_COST_CONTEXT',
                'NATIVE_WORKSPACE_BUDGET_IS_NOT_A_CLAIM_ABOUT_ORIGINAL_NUMPY_PEAK_MEMORY']
        duration=cost['duration_ticks'] if cost is not None and not issues else None
        records[identifier]=dict(duration_ticks=duration,context=context,cost=cost,frontier=issues,
            legality=dict(contract_sha256=contract,resource_demands=demands,conditions=conditions))
        rows.append(dict(id=identifier,cpu_binding=binding,cost_status='KNOWN' if duration is not None else 'UNRESOLVED',
            frontier=issues,context=context,profile=profile))
        options.append(dict(id=identifier,duration_ticks=duration,resource_demands=demands,
            contract_sha256=contract,legality_conditions=conditions))
    evidence=dict(schema='scar.stateful-cpu-choice-evidence',schema_version=1,options=records)
    path=out/'option-evidence.json'
    if write: receipt=_write(path,evidence)
    else:
        receipt=artifact_reference(path,evidence['schema'])
        if read_artifact(receipt)!=evidence: raise ValueError('stateful CPU option evidence differs')
    for option in options:
        pointer=option['id'].replace('~','~0').replace('/','~1')
        option['cost_evidence']=[dict(artifact=receipt,locator='#/options/'+pointer+'/duration_ticks')]
        option['legality_evidence']=[dict(artifact=receipt,locator='#/options/'+pointer+'/legality')]
    solver=solve_implementation_choices(semantic,task_options={'stateful-call':options},
        resource_capacities={key:len(value) for key,value in caps.items()},search_budget=search_budget)
    complete=(solver['status']=='OPTIMAL_WITHIN_FINITE_MODEL' and not solver['frontier']
              and not solver['unresolved'] and not domain['topology_frontier'])
    chosen=(solver.get('chosen_option_ids') or {}).get('stateful-call') if complete else None
    domain_ref=artifact_reference(Path(domain['base_stage_plan']['uri']).parent/'domain.json',domain['schema'])
    if read_artifact(domain_ref)!=domain: raise ValueError('original stateful CPU domain artifact differs')
    return dict(schema='scar.stateful-cpu-choice',schema_version=1,domain=domain_ref,profiles=profiles,
        no_op_profile=no_op_profile,output_directory=str(out),search_budget=search_budget,
        variants=rows,solver=solver,complete_cost_domain=complete,selected_id=chosen,
        decision='NO_OP' if chosen is None or chosen=='NO_OP' else 'SELECT_NATIVE_IMPLEMENTATION_FOR_LOWERING',
        status='MODEL_OPTIMUM_WITHIN_DECLARED_CPU_MASK_FAMILY' if complete else 'UNRESOLVED',tick_seconds=ticks,
        original_no_op_available=True,known_subset_incumbent=solver.get('chosen_option_ids'),
        scope='ONE_CALL_CPP_OPENMP_REPRESENTATIVE_MASK_FAMILY_PLUS_ORIGINAL_NO_OP',
        comparison_model='AUXILIARY_SINGLE_CALL_CHOICES_NOT_THE_ORIGINAL_PROGRAM_DATA_GRAPH',
        resource_policy='WHOLE_CALL_CPU_RESERVATION_FOR_SINGLE_CALL_COMPARISON_ONLY',
        all_cpu_subsets_enumerated=False,whole_program_makespan_optimized=False,
        deployment_authorized=False,physical_optimality_proven=False,
        target_variants_compiled=False,target_variants_executed=False)


def plan_stateful_cpu_choices(domain,*,cost_profiles=None,no_op_profile=None,search_budget=100000,out):
    domain=_domain(domain);out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh stateful CPU choice output required')
    profiles={key:_profile(value)[0] for key,value in (cost_profiles or {}).items()}
    if set(profiles)-{row['id'] for row in domain['candidates']}: raise ValueError('profile refers to absent CPU implementation')
    if no_op_profile is not None: no_op_profile=_profile(no_op_profile)[0]
    report=_evaluate(domain,profiles,no_op_profile,out,True,search_budget)
    _write(out/'choice.json',report);return report


def verify_stateful_cpu_choices(value):
    report=loads(Path(value).read_bytes()) if isinstance(value,(str,Path)) else deepcopy(value)
    if (report.get('schema'),report.get('schema_version'))!=('scar.stateful-cpu-choice',1): raise ValueError('stateful CPU choice report required')
    expected=_evaluate(_domain(read_artifact(report['domain'])),report['profiles'],report['no_op_profile'],
        Path(report['output_directory']),False,report['search_budget'])
    if semantic_sha256(report)!=semantic_sha256(expected): raise ValueError('stateful CPU choice differs from complete replay')
    return report


def selected_stateful_plan(value):
    report=verify_stateful_cpu_choices(value)
    if report['selected_id'] in (None,'NO_OP'): return None
    domain=read_artifact(report['domain']);row=next(row for row in domain['candidates'] if row['id']==report['selected_id'])
    from scar.compiler.model import CpuPlacement
    binding=dict(row['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    region=stateful_from_dict(read_artifact(domain['region']))
    plan=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],cpu_threads=row['worker_count'],
        memory_budget_bytes=domain['request']['memory_budget_bytes'],resource_binding=CpuPlacement(**binding))
    if semantic_sha256(_family(plan))!=domain['semantic_family_sha256']: raise ValueError('selected numeric plan differs from original family')
    return region,plan


__all__=['build_stateful_cpu_domain','stateful_cpu_cost_context','plan_stateful_cpu_choices',
    'verify_stateful_cpu_choices','selected_stateful_plan']
