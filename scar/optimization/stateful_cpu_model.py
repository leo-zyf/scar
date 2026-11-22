"""Stateful cpu model."""

from copy import deepcopy
import math
import sys
import json
import os
import struct
from pathlib import Path
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.compiler.stateful_model import (
    stateful_from_dict,
    document,
    read_stateful,
    stateful_digest,
)
from scar.optimization.scope_accounting import _fact, _profile
from dataclasses import replace
from hashlib import sha256
from scar.runtime.compiled_deployment import thread_environment
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode, loads
from scar.optimization.hardware import (
    _target_materials,
    _affinity,
    _boundaries,
    propose_cpu_allocations,
    validate_cpu_placement,
)
from scar.optimization.resource_solver import (
    semantic_sha256,
    solve_implementation_choices,
    task_implementation_contract,
)
from scar.optimization.schedule import ResourceSchedule, Task, TaskKind
from scar.optimization.stateful_recipe import stateful_backend_capability
from scar.optimization.stateful_schedule import plan_stateful_region, verify_stateful_plan


# Finite CPU-mask/worker choices for one unchanged stateful method call.
#
# The domain is hardware-derived and includes every worker count per supported
# executor. It is not every CPU subset, a process/MPI implementation, or a timing
# experiment. Unknown costs remain options and prevent full-domain selection.
_SCOPE=dict(boundary='WARM_SESSION',call_scope='ONE_ORIGINAL_METHOD_CALL_SITE',reuse_count=1,
    includes='ALL_IMPLEMENTATION_WORK_FROM_CALL_DISPATCH_THROUGH_RETURN',
    argument_expressions='ALREADY_EVALUATED_BEFORE_TIMING',
    implementation_components='ORIGINAL_AND_NATIVE_HAVE_DIFFERENT_INTERNAL_WORK',
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
            native_thread_environment=thread_environment(),
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
    region_path=Path(region).absolute() if isinstance(region,(str,Path)) else None
    region=read_stateful(region_path) if region_path is not None else region
    allowed,_=_affinity(affinity)
    proposals=propose_cpu_allocations(target,affinity=allowed,memory_budget_bytes=memory_budget_bytes,max_threads=max_threads)
    if not proposals: raise ValueError('no supported CPU/host allocation from original target and live constraints')
    seed=proposals[0]
    plan=plan_stateful_region(region,input_shapes=input_shapes,cpu_threads=len(seed.cpu_ids),
        memory_budget_bytes=memory_budget_bytes,resource_binding=seed)
    region_ref=(artifact_reference(region_path,'scar.stateful-array-region') if region_path is not None
        else _write(out/'region.json',document(region)))
    plan_ref=_write(out/'base-stage-plan.json',plan)
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
    return _cpu_cost_context(domain, candidate_id, row, base)


def _cpu_cost_context(domain, candidate_id, row, base):
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
        if (profile.get('schema'),profile.get('schema_version'))==('scar.rebased-stateful-cpu-cost-profile',1):
            from scar.optimization.stateful_cost_rebase import verify_rebased_stateful_cpu_cost
            replay=verify_rebased_stateful_cpu_cost(profile,expected_context=context)
            cost=dict(replay['original_cost'])
            cost['source']=dict(kind='VERIFIED_DOMAIN_EVIDENCE_REBASE',profile=source,
                original=profile['source_profile'],proof=profile['rebase_proof'])
            return cost,[],profile
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


# Forward component composition for a fixed stateful CPU implementation.
#
# Canonical resource measurements are not target-program trials. Composition is
# an explicit effective model, not an additivity theorem or a physical optimum.
# Unknown components are retained and prevent exporting a complete cost profile.
def component_key(component,parameters):
    return semantic_sha256(dict(component=component,parameters=parameters))


def stateful_resource_context(domain,candidate_id):
    """Hardware/backend applicability, independent of program/region/domain IDs."""
    row=next((r for r in domain['candidates'] if r['id']==candidate_id),None)
    return _resource_context(domain, candidate_id, row)


def _resource_context(domain, candidate_id, row):
    if row is None and candidate_id!='NO_OP':raise ValueError('unknown CPU candidate')
    backend=domain['backend_capabilities']['serial' if row is None or row['worker_count']==1 else 'parallel']
    return dict(target_sha256=domain['target_snapshot']['sha256'],
        executor_id=None if row is None else row['binding']['executor_id'],
        worker_cpu_ids=[] if row is None else row['binding']['cpu_ids'],
        caller_affinity_cpu_ids=domain['live_constraints']['process_affinity_cpu_ids'],
        native_thread_environment=domain['live_constraints']['native_thread_environment'],
        hardware_context=domain['hardware_context'],backend=backend,
        language_runtime=dict(version=sys.version,cache_tag=sys.implementation.cache_tag),
        dtype='float64',layout='RANK2_UNIT_STEP_ROW_STRIDES',memory_policy='UNBOUND',
        regime='CANONICAL_FINITE_REPEATED_BUFFERS_CACHE_AND_NUMA_RESIDENCY_NOT_PROVEN')


def stateful_cpu_demand(domain,candidate_id):
    """Count original stages and boundary work without deriving hardware time."""
    region = stateful_from_dict(read_artifact(domain['region']))
    plan = read_artifact(domain['base_stage_plan'])
    row = next((candidate for candidate in domain['candidates'] if candidate['id'] == candidate_id), None)
    workers = 1 if row is None else row['worker_count']
    terms = _demand_terms(region, plan, original=row is None, workers=workers)
    return _demand_document(domain, candidate_id, row, plan, terms)


def _demand_document(domain, candidate_id, row, plan, terms):
    context = _cpu_cost_context(domain, candidate_id, row, plan)
    return dict(schema='scar.stateful-cpu-demand',schema_version=1,context=context,
        resource_context=_resource_context(domain, candidate_id, row),terms=terms,
        logical_bytes_are_dram_bytes=False,source_graph_unrolled=False,
        scope='ONE_ORIGINAL_METHOD_CALL_SITE',costs_inferred=False)


def _boundary_parameters(region, plan):
    arrays = [item for item in region.inputs if item.kind == 'array']
    return dict(arrays=len(arrays), scalars=sum(item.kind == 'scalar' for item in region.inputs),
        callees=len(region.callees), providers=len(region.providers),
        input_path_depths=[len(item.path)-1 for item in region.inputs],
        callee_path_depths=[len(item.path)-1 for item in region.callees],
        provider_lookup_depths=[len(item.lookup_path)-1 for item in region.providers],
        array_shapes=[plan['input_shapes'][item.id] for item in arrays],
        written_arrays=[item.id in plan['shadows'] for item in arrays],
        workspace_slot_capacities=[slot['capacity_bytes'] for slot in plan['workspace_slots']])


def _demand_terms(region, plan, *, original, workers):
    nodes={n.id:n for n in region.nodes}
    shapes={};views={};origins={};available={};available_payloads={};terms=[]
    def term(component,parameters,source_ref=None):
        terms.append(dict(id='term:'+str(len(terms)),component=component,parameters=parameters,
            key=component_key(component,parameters),count=1,source_ref=source_ref))
    def view(key):
        if key in available: return available[key]
        return views.get(key)
    for node in region.nodes:
        if node.op=='input':
            shapes[node.id]=plan['input_shapes'].get(node.attrs['name'])
            if node.attrs['kind']=='array':
                views[node.id]=dict(stride=shapes[node.id][1],offset=0)
                origins[node.id]=(node.attrs['name'],node.attrs['epoch'])
        elif node.op=='const': shapes[node.id]=None
        elif node.op=='slice':
            parent=shapes[node.inputs[0]];starts=[];shape=[]
            for bound,extent in zip(node.attrs['bounds'],parent):
                start,stop,_=slice(*bound).indices(extent);starts.append(start);shape.append(max(0,stop-start))
            source=views[node.inputs[0]];shapes[node.id]=shape
            views[node.id]=dict(stride=source['stride'],offset=source['offset']+starts[0]*source['stride']+starts[1])
            origins[node.id]=origins[node.inputs[0]]
        else: shapes[node.id]=next((shapes[i] for i in node.inputs if shapes[i] is not None),None)
    if original:
        term('ORIGINAL_METHOD_BINDINGS_SCALARS_AND_VIEWS',dict(inputs=len(region.inputs),
            descriptors=sum(n.op=='slice' for n in region.nodes)))
        for index,action in enumerate(region.actions):
            node=nodes[action.value]
            parameters=dict(op=node.op if action.kind=='evaluate' else action.op,shape=shapes[action.value],
                operands=[view(i) if shapes[i] is not None else dict(scalar=True) for i in node.inputs])
            if action.kind=='write':
                shape=plan['input_shapes'][action.target];bounds=action.bounds or ((None,None),(None,None))
                spans=[slice(*b).indices(n) for b,n in zip(bounds,shape)]
                parameters.update(shape=[max(0,b-a) for a,b,_ in spans],target_stride=shape[1],
                    target_offset=spans[0][0]*shape[1]+spans[1][0],rhs=view(action.value) if shapes[action.value] is not None else dict(scalar=True))
            component=('NUMPY_EAGER_OPERATION_WITH_OWN_ALLOCATION_RELEASE' if shapes[action.value] is not None
                       else 'PYTHON_TAGGED_SCALAR_OPERATION') if action.kind=='evaluate' else 'NUMPY_INPLACE_WRITE'
            term(component,parameters,
                dict(artifact_role='region',pointer=f'#/actions/{index}'))
            if action.kind=='evaluate' and shapes[action.value] is not None:
                available[action.value]=dict(stride=shapes[action.value][1],offset=0)
    else:
        parameters = _boundary_parameters(region, plan)
        term('GUARDED_METHOD_AND_NATIVE_ARTIFACT_BOUNDARY', parameters)
        capacities = parameters['workspace_slot_capacities']
        term('ALLOCATE_FIXED_POOL',dict(capacities=capacities))
        for shape in plan['input_shapes'].values(): term('INPUT_FINITE_SCAN_SERIAL',dict(rows=shape[0],cols=shape[1]))
        term('CALLER_WORKER_AFFINITY_FENV_TEAM',dict(workers=workers))
        for stage in plan['stages']:
            kind=stage['kind'];shape=stage['shape'];root=stage['root_expr']
            ref=dict(artifact_role='base_stage_plan',pointer=f"#/stages/{stage['index']}")
            if kind=='SNAPSHOT_ROOT':
                term('ROW_COPY_STAGE',dict(rows=shape[0],cols=shape[1],source_stride=shape[1],source_offset=0,
                    destination_stride=shape[1],destination_offset=0,workers=workers),ref)
            elif kind=='PROVIDER':
                operand=view(nodes[root].inputs[0])
                term('NUMPY_POWER2_NATIVE_ROW_STAGE',dict(rows=shape[0],cols=shape[1],
                    source_stride=operand['stride'],source_offset=operand['offset'],workers=workers),ref)
            elif kind=='WRITE_SHADOW':
                target=stage['target'];r,c=[axis[0] for axis in stage['normalized_bounds']]
                term('INPLACE_ROW_OPERATION_STAGE',dict(op=stage['op'],rows=shape[0],cols=shape[1],
                    source_stride=shape[1],source_offset=0,destination_stride=plan['input_shapes'][target][1],
                    destination_offset=r*plan['input_shapes'][target][1]+c,workers=workers),ref)
            elif kind=='COMMIT':
                target=stage['target'];r,c=[axis[0] for axis in stage['normalized_bounds']];stride=plan['input_shapes'][target][1]
                term('ROW_COPY_SERIAL',dict(rows=shape[0],cols=shape[1],source_stride=stride,
                    source_offset=r*stride+c,destination_stride=stride,destination_offset=r*stride+c,workers=1),ref)
            elif kind=='SCALAR':
                term('TAGGED_SCALAR_SETUP_STAGE',dict(operations=[nodes[i].op for i in stage['expr_ids']
                    if nodes[i].op not in ('input','const','slice')],workers=workers),ref)
            else:
                operations=[nodes[i] for i in stage['expr_ids'] if nodes[i].op not in ('input','const','slice')]
                if len(operations)==1 and all(view(i) is not None for i in nodes[root].inputs):
                    operands=[view(i) for i in nodes[root].inputs]
                    term('ROW_OPERATION_STAGE',dict(op=nodes[root].op,rows=shape[0],cols=shape[1],
                        operands=operands,workers=workers),ref)
                else:
                    loads={};depth={};groups={}
                    def visit(key):
                        if key in depth:return depth[key]
                        descriptor=view(key)
                        if descriptor is not None:
                            identity=(str(origins.get(key,('computed',key))),descriptor['stride'],descriptor['offset'])
                            storage=str(origins.get(key,('computed',key)))
                            group=groups.setdefault(storage,len(groups))
                            loads[identity]=dict(**descriptor,storage_group=group);depth[key]=0
                        elif shapes[key] is None:depth[key]=0
                        else:depth[key]=1+max((visit(i) for i in nodes[key].inputs),default=0)
                        return depth[key]
                    span=visit(root)
                    layouts=sorted(loads.values(),key=lambda d:(d['stride'],d['offset'],d['storage_group']))
                    canonical_groups={}
                    layouts=[{**layout,'storage_group':canonical_groups.setdefault(layout['storage_group'],len(canonical_groups))}
                             for layout in layouts]
                    term('FUSED_SCALAR_DAG_ROW_STAGE',dict(rows=shape[0],cols=shape[1],workers=workers,
                        arithmetic={op:sum(n.op==op and shapes[n.id] is not None for n in operations)
                                    for op in ('add','sub','mul','div','neg')},
                        array_loads=len(loads),output_stores=1,dependent_operation_depth=span,
                        backing_storage_groups=len(canonical_groups),operand_layouts=layouts),ref)
            if stage['output_payload'] is not None and root is not None:
                available[root]=dict(stride=shape[1],offset=0)
                available_payloads[root]=stage['output_payload']
            # Match codegen's stage-local residency. A released workspace
            # value is not an available load in a later fused expression.
            released=set(stage.get('release_values_after',()))
            available={key:value for key,value in available.items()
                if available_payloads[key] not in released}
            available_payloads={key:value for key,value in available_payloads.items() if key in available}
        term('SUCCESS_CHECK_AND_CALLER_RESTORATION',dict(workers=workers))
        term('RELEASE_FIXED_POOL',dict(capacities=capacities))
    return terms


def _save(path,value):
    import json
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(path,value['schema'])


def predict_stateful_cpu_model(domain,candidate_id,calibrations,*,out):
    """Compose exact matching resource signatures; leave unsupported fusion unknown."""
    domain=_domain(domain)
    return _predict_stateful_cpu_model(domain,candidate_id,calibrations,out=out)


def _predict_stateful_cpu_model(domain,candidate_id,calibrations,*,out):
    """Internal composition within an already admitted domain session."""
    result = _evaluate_stateful_cpu_components(domain, candidate_id, calibrations)
    return _write_stateful_cpu_model(result, out)


def evaluate_stateful_cpu_components(domain,candidate_id,calibrations):
    """Replay component provenance without publishing another profile or model.

    Consumers may map these occurrence-specific terms to execution activities;
    canonical timings remain conditional models, not observed task durations.
    """
    return _evaluate_stateful_cpu_components(_domain(domain),candidate_id,calibrations)


def _evaluate_stateful_cpu_components(domain,candidate_id,calibrations):
    demand = stateful_cpu_demand(domain, candidate_id)
    return _evaluate_demand(demand, _prepare_calibrations(calibrations), {})


def _prepare_calibrations(calibrations):
    prepared = []
    for value in calibrations:
        document, source = _profile(value)
        if (document.get('schema'), document.get('schema_version')) != ('scar.stateful-resource-calibration', 1):
            raise ValueError('stateful independent resource calibration required')
        prepared.append((document, source, semantic_sha256(document['resource_context'])))
    return prepared


def _calibration_reference(document, source, receipts):
    if source['kind'] != 'PATH':
        raise ValueError('retained calibration JSON path required')
    path = source['path']
    if path not in receipts:
        receipts[path] = artifact_reference(path, document['schema'])
    reference = receipts[path]
    if (reference['sha256'], reference['size_bytes'], reference['schema']) != (
        source['sha256'], source['bytes'], document['schema'],
    ):
        raise ValueError('retained calibration changed during cost evaluation')
    return reference


def _evaluate_demand(demand, calibrations, receipts):
    sources, records, issues, rejected = [], {}, [], {}
    inapplicable, rejected_calibrations = [], []
    context_sha256 = semantic_sha256(demand['resource_context'])
    for document, source, calibration_sha256 in calibrations:
        if calibration_sha256 != context_sha256:
            inapplicable.append(dict(source=source,reason='ORIGINAL_RESOURCE_CONTEXT_MISMATCH',
                offered_context_sha256=calibration_sha256, required_context_sha256=context_sha256))
            continue
        artifact = _calibration_reference(document, source, receipts)
        sources.append(artifact)
        independence=('target_workload_executed','target_variants_ranked')
        invalid=[name for name in independence if document.get(name) is not False]
        if invalid:
            reason='independent calibration requires explicit False: '+', '.join(invalid)
            rejected_calibrations.append(dict(artifact=artifact,reason=reason,
                missing_fields=[name for name in invalid if name not in document],
                observed_flags={name:document.get(name) for name in invalid}))
            # Preserve offered estimates only as rejected diagnostic evidence;
            # they do not enter records, component sums or complete profiles.
            for index,row in enumerate(document.get('components',())):
                key=component_key(row['component'],row['parameters'])
                rejected.setdefault(key,[]).append(dict(point_estimate_ns=row.get('median_ns'),
                    reason=reason,evidence=dict(artifact=artifact,locator=f'#/components/{index}')))
            continue
        for index,row in enumerate(document.get('proposed_estimates',())):
            if row.get('selectable') is not False:
                raise ValueError('diagnostic estimates cannot authorize selection')
            key=component_key(row['component'],row['parameters'])
            rejected.setdefault(key,[]).append(dict(
                point_estimate_ns=row.get('point_estimate_ns'),reason=row.get('rejection_reason'),
                evidence=dict(artifact=artifact,locator=f'#/proposed_estimates/{index}')))
        for index,record in enumerate(document['components']):
            key=component_key(record['component'],record['parameters'])
            if key in records: raise ValueError('ambiguous resource component observations')
            if (record['basis'] not in ('DIRECT_COMPONENT_TIMING','DECLARED_COMPONENT_MODEL')
                    or record.get('selectable') is False or record.get('precision_admitted') is False
                    or record.get('exportable_as_complete_cost') is False
                    or type(record['median_ns']) not in (int,float)
                    or not math.isfinite(record['median_ns']) or record['median_ns']<0):
                raise ValueError('sourced direct or explicitly declared nonnegative resource cost required')
            records[key]=dict(duration_ns=record['median_ns'],basis=record['basis'],
                evidence=dict(artifact=artifact,locator=f'#/components/{index}'))
    evaluated=[];total=0
    for term in demand['terms']:
        record=records.get(term['key'])
        proposals=rejected.get(term['key'],[])
        if record is None: issues.append(dict(term=term['id'],component=term['component'],parameters=term['parameters'],
            reason='independent component model rejected by its admission policy' if proposals else
                'no exact-context independent component model or direct measurement',rejected_estimates=proposals))
        else:total+=record['duration_ns']*term['count']
        evaluated.append(dict(**term,status=('MODEL_REJECTED' if proposals else 'UNKNOWN')
            if record is None else 'SOURCED_EFFECTIVE_COMPONENT',measurement=record,rejected_estimates=proposals))
    return dict(demand=demand,source_calibrations=sources,components=evaluated,unresolved=issues,
        known_component_sum_ns=total,rejected_calibrations=rejected_calibrations,
        inapplicable_calibrations=inapplicable)


def _write_stateful_cpu_model(result, out):
    demand = result['demand']
    context = demand['context']
    evaluated = result['components']
    issues = result['unresolved']
    total = result['known_component_sum_ns']
    out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh stateful CPU model output required')
    report=dict(schema='scar.stateful-cpu-model',schema_version=1,context=context,demand=demand,
        resource_context=demand['resource_context'],components=evaluated,unresolved=issues,known_component_sum_ns=total,
        complete=not issues,duration_ticks=math.ceil(total) if not issues else None,tick_seconds=1e-9,
        source_calibrations=result['source_calibrations'],rejected_calibrations=result['rejected_calibrations'],profile=None,
        assumptions=['EFFECTIVE_COMPONENT_SUM_IS_A_MODEL_NOT_AN_ADDITIVITY_THEOREM',
            'STAGE_MEASUREMENTS_INCLUDE_THEIR_OWN_IMPLICIT_BARRIER_NO_DOUBLE_CHARGE',
            'CACHE_NUMA_ALIGNMENT_CONTENTION_REGIMES_REMAIN_APPLICABILITY_ASSUMPTIONS',
            'CANONICAL_OPERAND_VALUES_ARE_NOT_THE_PROGRAM_DATA_DISTRIBUTION',
            'RESOURCE_CONTENTION_IS_RECORDED_NOT_ASSUMED_ISOLATED',
            'NO_LINEAR_DIVISION_BY_WORKER_COUNT_NO_DRAM_TRAFFIC_INFERENCE'],
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    if not issues:
        facts=dict(schema='scar.stateful-cpu-model-facts',schema_version=1,context=context,
            estimate=dict(duration_ticks=report['duration_ticks'],tick_seconds=1e-9),
            derivation=dict(origin='INDEPENDENT_PRIMITIVE_COMPOSITION' if all(
                r['measurement']['basis']=='DIRECT_COMPONENT_TIMING' for r in evaluated) else 'DECLARED_FORWARD_MODEL',target_variant_trials=False,
                assumptions=report['assumptions'],component_refs=[r['measurement']['evidence'] for r in evaluated]))
        artifact=_save(out/'facts.json',facts)
        fact=lambda pointer:[dict(artifact=artifact,locator=pointer)]
        profile=dict(schema='scar.stateful-cpu-cost-profile',schema_version=1,context=context,
            context_evidence=fact('#/context'),estimate={**facts['estimate'],'evidence':fact('#/estimate')},
            derivation={**facts['derivation'],'evidence':fact('#/derivation')})
        _save(out/'profile.json',profile);report['profile']=str(out/'profile.json')
    _save(out/'model.json',report)
    return report


def build_stateful_cpu_profiles(domain,calibrations,*,out):
    """Write the mapping consumed directly by choose-stateful-cpu, including gaps."""
    domain=_domain(domain);out=Path(out).absolute()
    if out.exists():raise ValueError('fresh CPU profile-set output required')
    region = stateful_from_dict(read_artifact(domain['region']))
    plan = read_artifact(domain['base_stage_plan'])
    prepared = _prepare_calibrations(calibrations)
    receipts, demand_terms = {}, {}
    mapping={};reports=[];baseline=None
    for row in (*domain['candidates'], None):
        identifier = 'NO_OP' if row is None else row['id']
        workers = 1 if row is None else row['worker_count']
        demand_key = (row is None, workers)
        # The stage graph and worker count determine the logical demand.
        # Candidate resource contexts still carry their exact CPU masks.
        if demand_key not in demand_terms:
            demand_terms[demand_key] = _demand_terms(region, plan, original=row is None, workers=workers)
        demand = _demand_document(domain, identifier, row, plan, demand_terms[demand_key])
        report = _write_stateful_cpu_model(_evaluate_demand(demand, prepared, receipts), out / identifier)
        reports.append(dict(id=identifier,model=str(out/identifier/'model.json'),complete=report['complete'],
            known_components=sum(r['status']=='SOURCED_EFFECTIVE_COMPONENT' for r in report['components']),missing_components=len(report['unresolved'])))
        if report['profile'] is not None:
            if identifier=='NO_OP':baseline=report['profile']
            else:mapping[identifier]=report['profile']
    # Recheck the original graph, hardware and cost evidence at publication.
    _domain(domain)
    for reference in receipts.values():
        if artifact_reference(reference['uri'], reference['schema']) != reference:
            raise ValueError('retained calibration changed before profile-set publication')
    result=dict(schema='scar.stateful-cpu-profile-set',schema_version=1,cost_profiles=mapping,
        no_op_profile=baseline,reports=reports,complete_domain=len(mapping)==len(domain['candidates']) and baseline is not None,
        target_workload_executed=False,physical_optimality_proven=False)
    import json
    out.mkdir(parents=True,exist_ok=True)
    with (out/'profile-map.json').open('x') as stream: json.dump(mapping,stream,sort_keys=True,indent=2);stream.write('\n')
    result['cost_profiles_path']=str(out/'profile-map.json')
    _save(out/'profiles.json',result);return result


def wrap_original_call_cost(domain,observation,*,out):
    """Replay an observed original call as a forward expectation at the same site.

    This is not a native variant trial or a primitive decomposition. No load,
    setup or per-stage cost is inferred from the original complete-call total.
    """
    domain=_domain(domain);observed,source=_profile(observation)
    expected=stateful_cpu_cost_context(domain,'NO_OP')
    if (observed.get('schema'),observed.get('schema_version'))!=('scar.original-stateful-call-observation',1):
        raise ValueError('original method-call observation required')
    if (semantic_sha256(observed['context'])!=semantic_sha256(expected)
            or observed['measurement_scope']!='ONE_ORIGINAL_METHOD_CALL_SITE'
            or observed.get('target_variants_ranked') is not False
            or observed['observed_caller_affinity_cpu_ids']!=domain['live_constraints']['process_affinity_cpu_ids']
            or observed['observed_native_thread_environment']!=domain['live_constraints']['native_thread_environment']):
        raise ValueError('original method boundary/runtime context differs')
    samples=observed['samples_ns']
    import statistics
    if (not isinstance(samples,list) or len(samples)<3 or any(type(v) not in (int,float) or
            not math.isfinite(v) or v<=0 for v in samples) or observed['median_ns']!=statistics.median(samples)):
        raise ValueError('retained original samples/median differ')
    if source['kind']=='PATH': ref=artifact_reference(source['path'],observed['schema'])
    elif source['kind']=='FACT_REFERENCE':ref=source['reference']['artifact']
    else:raise ValueError('original observation must have retained artifact provenance')
    facts=dict(schema='scar.original-stateful-forward-cost',schema_version=1,context=expected,
        estimate=dict(duration_ticks=math.ceil(observed['median_ns']),tick_seconds=1e-9),
        derivation=dict(origin='DECLARED_FORWARD_MODEL',basis='OBSERVED_ORIGINAL_SAME_SCOPE_MEDIAN_AS_FORWARD_EXPECTATION',
            target_variant_trials=False,source_observation=ref,
            assumptions=['ORIGINAL_CALL_SAME_SCOPE_RUNTIME_CONTEXT_AND_SAMPLE_DISTRIBUTION_REMAIN_APPLICABLE',
                'ORIGINAL_HAS_NO_NATIVE_GUARD_DISPATCH_SHADOW_OR_COMMIT_COMPONENTS',
                'COMPLETE_CALL_OBSERVATION_NOT_SPLIT_OR_SUBTRACTED_FROM_ANOTHER_SCOPE']))
    out=Path(out).absolute()
    if out.exists():raise ValueError('fresh original cost profile output required')
    artifact=_save(out/'facts.json',facts);fact=lambda pointer:[dict(artifact=artifact,locator=pointer)]
    profile=dict(schema='scar.stateful-cpu-cost-profile',schema_version=1,context=expected,context_evidence=fact('#/context'),
        estimate={**facts['estimate'],'evidence':fact('#/estimate')},derivation={**facts['derivation'],'evidence':fact('#/derivation')})
    _save(out/'profile.json',profile);return str(out/'profile.json')


__all__ = [
    'component_key',
    'stateful_resource_context',
    'stateful_cpu_demand',
    'predict_stateful_cpu_model',
    'evaluate_stateful_cpu_components',
    'build_stateful_cpu_profiles',
    'wrap_original_call_cost',
    'build_stateful_cpu_domain',
    'stateful_cpu_cost_context',
    'plan_stateful_cpu_choices',
    'verify_stateful_cpu_choices',
    'selected_stateful_plan',
]
