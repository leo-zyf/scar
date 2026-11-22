"""Software resources."""

import argparse
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from scar.analysis.python_binding_query import PythonBindingGraph
from scar.analysis.target_graph import TargetHardwareGraph, _artifact
from scar.analysis.source_stage import _durable_bytes, _durable_mkdir
from scar.analysis.source_templates import _json, _stream
from scar.ir.program import (
    native_contracts as S,
    target_hardware as H,
    native_contracts as R,
    python_bindings as B,
    source_productions as P,
)
from scar.ir.program.providers import ProviderFactReference, ProviderArtifact
from scar.ir.record_codec import decode, encode
from copy import deepcopy
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.temporal_resources import (
    ExecutionBinding,
    StorageBinding,
    OriginalBudget,
    ResourceProjection,
    validate_resource_projection,
)
from scar.ir.versioned_schedule import (
    TemporalOccupancyKind as K,
    schedule_document,
    validate_versioned_schedule,
)
from scar.optimization.resource_solver import semantic_sha256


# Connect software interfaces and artifacts to a retained target description.
#
# Candidates express compatible locations/referents, not allocations, selected
# devices, resource consumption, or transfer events. Source routes express
# normalization dependencies, never equality with native values or storage.
def _records(directory,receipt,cls):
    payload=(directory/receipt['path']).read_bytes()
    if len(payload)!=receipt['bytes'] or sha256(payload).hexdigest()!=receipt['sha256']:raise ValueError('resource declaration table differs')
    raw=gzip.decompress(payload)
    if len(raw)!=receipt['decoded_bytes'] or sha256(raw).hexdigest()!=receipt['decoded_sha256'] or raw.count(b'\n')!=receipt['records']:
        raise ValueError('decoded resource declaration table differs')
    return tuple(decode(cls,json.loads(line),record_tags=P.PRODUCTION_RECORD_TAGS) for line in raw.splitlines())


def _input_sources(definition,operand,native_name,*,result,positional,keywords,positional_pack,keyword_pack,
        receiver,receiver_mode,conditions,semantics=S.NormalizationDependencyKind.CALL):
    if definition.implicit_receiver and native_name==definition.implicit_receiver:
        kind=S.OperandSourceKind.RECEIVER if receiver is not None else S.OperandSourceKind.RECEIVER_OPEN
        actuals=() if receiver is None else (receiver,)
        if receiver is None and receiver_mode=='UNBOUND' and positional_pack is not None:
            kind=S.OperandSourceKind.PACK_DEPENDENCY;actuals=(positional_pack,)
        yield S.OperandSource(result,definition.id,operand.boundary,operand.port,definition.implicit_receiver,
            actuals,kind,None,(*conditions,'METHOD_RECEIVER_DEPENDENCY_NOT_STORAGE_ALIAS'),semantics)
        return
    params={a.parser_index:a for a in definition.arguments};found=False
    for conversion in definition.conversions:
        if conversion.stage is not B.ConversionStage.PYTHON_PARSE or native_name not in conversion.schema_arguments:continue
        if conversion.parser_index not in params:continue
        argument=params[conversion.parser_index];found=True;named=dict(keywords)
        actuals=();kind=S.OperandSourceKind.CONVERSION_OPEN;expression=None
        if argument.name in named:
            kind=S.OperandSourceKind.KEYWORD;actuals=(named[argument.name],)
        elif keyword_pack is not None or (not argument.keyword_only and positional_pack is not None):
            # CALL_FUNCTION_EX retains whole evaluated packs. Either pack can
            # supply a positional-or-keyword parameter; **kwargs does not
            # erase the *args dependency. This is not a binding/alias proof.
            kind=S.OperandSourceKind.PACK_DEPENDENCY
            actuals=tuple(p for p in (None if argument.keyword_only else positional_pack,keyword_pack) if p is not None)
        elif argument.position is not None and argument.position+(1 if receiver_mode=='UNBOUND' else 0)<len(positional):
            shift=1 if receiver_mode=='UNBOUND' else 0
            kind=S.OperandSourceKind.POSITIONAL;actuals=(positional[argument.position+shift],)
        elif argument.default is not None or argument.default_init is not None:
            kind=S.OperandSourceKind.DEFAULT_EXPRESSION;expression=argument.default_init or argument.default
        yield S.OperandSource(result,definition.id,operand.boundary,operand.port,argument.name,actuals,kind,expression,
            (*conditions,'PARSER_SIGNATURE_ACCEPTS_ACTUAL_ARGUMENTS','NORMALIZATION_DEPENDENCY_NOT_NATIVE_VALUE_ALIAS'),semantics)
    if not found:
        kind=S.OperandSourceKind.RECEIVER_OPEN if definition.implicit_receiver and native_name=='self' else S.OperandSourceKind.CONVERSION_OPEN
        yield S.OperandSource(result,definition.id,operand.boundary,operand.port,None,(),kind,None,
            (*conditions,'RECEIVER_OR_CONVERSION_BINDING_OPEN'),semantics)


def _operand_sources(definition,call,operand,native_name):
    site=call.site
    yield from _input_sources(definition,operand,native_name,result=site.result,positional=site.positional,
        keywords=site.keywords,positional_pack=site.positional_pack,keyword_pack=site.keyword_pack,
        receiver=call.receiver,receiver_mode=call.receiver_mode,conditions=call.conditions)


def _operator_sources(definition,call,operand,native_name):
    # These are the implicit protocol's parser dependencies, not a fabricated
    # source CALL/SourceCallSite. The language operation keeps its original ABI.
    yield from _input_sources(definition,operand,native_name,result=call.result,positional=call.arguments,
        keywords=(),positional_pack=None,keyword_pack=None,receiver=call.receiver,
        receiver_mode=B.ReceiverMode.BOUND,conditions=call.conditions,semantics=S.NormalizationDependencyKind.LANGUAGE_PROTOCOL)


def build_software_resources(bindings,declarations,target,out):
    started=time.monotonic();view=PythonBindingGraph(bindings);hardware=TargetHardwareGraph(target)
    declarations=Path(declarations).absolute();out=Path(out).absolute()
    if out.exists():raise ValueError('fresh software/target output required')
    raw=(declarations/'result.json').read_bytes();report=json.loads(raw)
    if report['schema']!='scar.native-resource-declarations' or report['schema_version']!=1:raise ValueError('resource declarations version differs')
    if report['bindings_result_sha256']!=sha256((view.path/'result.json').read_bytes()).hexdigest():raise ValueError('resource declaration binding scope differs')
    if report['types_sha256']!=sha256(Path(R.__file__).read_bytes()).hexdigest():raise ValueError('resource declaration types differ')
    operands=_records(declarations,report['resource_operands'],R.ResourceOperand)
    resources=hardware.resources();snapshot=_artifact(hardware.path/'result.json','scar.target-hardware-graph')
    evidence=_artifact(declarations/report['resource_operands']['path'],'scar.native-resource-operands+gzip-jsonl')
    placements=[];by_boundary=defaultdict(list)
    for index,operand in enumerate(operands,1):
        by_boundary[operand.boundary].append(operand)
        declaration=ProviderFactReference(evidence,'jsonl:'+str(index))
        for resource in resources:
            if operand.category is R.ResourceOperandKind.TENSOR_ARGUMENT and type(resource) is H.MemoryDomain:
                relation=H.ResourceCandidateKind.TENSOR_STORAGE_POOL
                conditions=('PRESENT_MATERIALIZED_TENSOR','SOURCE_DEVICE_AND_STORAGE_PLACEMENT_MATCH','TARGET_ACCESS_POLICY_MATCHES')
            elif operand.category is R.ResourceOperandKind.DEVICE_ARGUMENT and type(resource) in (H.CpuExecutor,H.Accelerator):
                relation=H.ResourceCandidateKind.DEVICE_REFERENT
                conditions=('DEVICE_VALUE_PRESENT','FRAMEWORK_VISIBILITY_AND_TARGET_MAPPING_MATCHES')
            elif operand.category is R.ResourceOperandKind.GENERATOR_ARGUMENT and type(resource) in (H.CpuExecutor,H.Accelerator):
                relation=H.ResourceCandidateKind.GENERATOR_DEVICE
                conditions=('GENERATOR_PRESENT','GENERATOR_DEVICE_AND_TARGET_MAPPING_MATCHES','NO_RNG_CONSUMPTION_CLAIM')
            else:continue
            placements.append(S.OperandTargetCandidate(operand.boundary,operand.port,declaration,H.TargetResourceRef(snapshot,resource.id),relation,conditions))
    definitions={d.id:d for d in view.definitions()};sources=[];port_names={};materials={}
    def interface_sources(call,definition,operator=False):
        if definition.native is None:return
        boundary=definition.native.boundary
        if boundary not in port_names:
            graph=view.expand_native(definition);native=next(b for b in graph.native_boundaries if b.id==boundary)
            artifact=native.signature.artifact;key=(artifact.uri,artifact.sha256,artifact.size_bytes)
            if key not in materials:
                payload=Path(artifact.uri).read_bytes()
                if len(payload)!=artifact.size_bytes or sha256(payload).hexdigest()!=artifact.sha256:
                    raise ValueError('native normalization materials differ')
                materials[key]=json.loads(payload)
            material=materials[key]
            index=int(native.signature.locator.split('/')[2]);names=material['declarations'][index]['inputs']
            port_names[boundary]={slot.port:row['name'] for slot,row in zip(native.inputs,names,strict=True)}
        resolver=_operator_sources if operator else _operand_sources
        for operand in by_boundary[boundary]:sources.extend(resolver(definition,call,operand,port_names[boundary][operand.port]))
    for call in view.native_calls():interface_sources(call,definitions[call.binding])
    for call in view.declared_operators():
        if call.binding is not None:interface_sources(call,definitions[call.binding],True)
    endpoints=[r for r in resources if type(r) is H.StorageEndpoint]
    locations=[];facts=_artifact(view.source.facts/'result.json','scar.normal-source-link-facts')
    for index,row in enumerate(view.source.inputs['units']):
        path=Path(row['path']).resolve();matches=[e for e in endpoints if path.is_relative_to(Path(e.mount_point))]
        endpoint=max(matches,key=lambda e:len(e.mount_point)) if matches else None
        # Match actual filesystem IDs at this platform observation. The IDs
        # are routing helpers, not OS objects/nodes in the optimization graph.
        try:matches_device=endpoint is not None and path.stat().st_dev==Path(endpoint.mount_point).stat().st_dev
        except OSError:matches_device=False
        locations.append(S.SoftwareArtifactLocation(row['unit'],row['path'],str(path),
            H.TargetResourceRef(snapshot,endpoint.id) if matches_device else None,
            ProviderFactReference(facts,'#/units/'+str(index)+'/path')))
    _durable_mkdir(out)
    pt=_stream(out/'operand-targets.jsonl.gz',placements);st=_stream(out/'operand-sources.jsonl.gz',sources)
    at=_stream(out/'artifact-locations.jsonl.gz',locations)
    result=dict(schema='scar.software-target-resource-graph',schema_version=2,status='CONDITIONAL_RESOURCE_RELATIONSHIPS_BUILT',
        bindings=str(view.path),bindings_result_sha256=sha256((view.path/'result.json').read_bytes()).hexdigest(),
        declarations=str(declarations),declaration_result_sha256=sha256(raw).hexdigest(),
        target=str(hardware.path),target_result_sha256=snapshot.sha256,
        operand_targets=pt,operand_sources=st,artifact_locations=at,source_kinds=dict(Counter(row.kind.value for row in sources)),
        source_semantics=dict(Counter(row.semantics.value for row in sources)),
        source_artifacts_with_endpoints=sum(row.endpoint is not None for row in locations),
        actual_placement_established=False,resource_uses_created=0,transfers_observed=0,target_program_executed=False,first_stage_complete=False,
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),types_sha256=sha256(Path(S.__file__).read_bytes()).hexdigest(),
        seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json',_json(result),replace=False);return result


class SoftwareResourceGraph:
    def __init__(self,path):
        self.path=Path(path).absolute();self.report=json.loads((self.path/'result.json').read_bytes())
        if self.report['schema']!='scar.software-target-resource-graph' or self.report['schema_version']!=2:raise ValueError('software resource graph version differs')
        if self.report['implementation_sha256']!=sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('software resource builder differs')
        if self.report['types_sha256']!=sha256(Path(S.__file__).read_bytes()).hexdigest():raise ValueError('software resource types differ')
        self.bindings=PythonBindingGraph(self.report['bindings']);self.hardware=TargetHardwareGraph(self.report['target'])
        for key,path_ in (('bindings_result_sha256',self.bindings.path),('target_result_sha256',self.hardware.path),
                ('declaration_result_sha256',Path(self.report['declarations']))):
            if sha256((path_/'result.json').read_bytes()).hexdigest()!=self.report[key]:raise ValueError('software resource parent differs')

    def candidates(self,boundary=None):
        rows=_records(self.path,self.report['operand_targets'],S.OperandTargetCandidate)
        return tuple(row for row in rows if boundary is None or row.boundary==boundary)

    def sources(self,source=None):
        unit=self.bindings.source.units_for_source(source)
        rows=_records(self.path,self.report['operand_sources'],S.OperandSource)
        return tuple(row for row in rows if unit is None or row.call.unit in unit)

    def artifact_locations(self):
        return _records(self.path,self.report['artifact_locations'],S.SoftwareArtifactLocation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bindings','declarations','target','out'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();r=build_software_resources(args.bindings,args.declarations,args.target,args.out)
    print(json.dumps({k:r[k] for k in ('status','source_kinds','source_artifacts_with_endpoints','seconds')},indent=2))


# Bind existing time occupancies to retained CPU/host topology candidates.
#
# No live hardware query, resource reservation, page placement or cost inference
# occurs here. Candidate domains and source-selected CPU requirements differ.
def _cpu_ids(value,field):
    if value is None:return None
    if (type(value)not in(list,tuple)or any(type(x)is not int or x<0 for x in value)
            or len(set(value))!=len(value)):
        raise ValueError(field+' must be an original distinct CPU-id sequence or UNKNOWN')
    return tuple(value)


def _scope_domains(schedule,graph,target_ref):
    """Replay new source-scoped domains; absence keeps legacy behavior exact."""
    from scar.analysis.scope_demands import SCHEMA,derive_serial_scope_demands
    proofs=[(i,row)for i,row in enumerate(schedule.temporal.source_refs)if row.get('schema')==SCHEMA]
    if not proofs:return None
    if len(proofs)!=1:raise ValueError('one original serial scope demand proof required')
    ordinal,proof=proofs[0]
    if (proof.get('schema_version')!=1 or proof.get('target')!=target_ref
            or type(proof.get('recipe'))is not dict):
        raise ValueError('serial scope demand recipe/target provenance differs')
    recipe=read_artifact(proof['recipe'])
    if (recipe.get('recipe_id')!=proof.get('recipe_id')
            or recipe.get('artifacts',{}).get('target_snapshot')!=target_ref
            or recipe['artifacts'].get('stage_plan')!=proof.get('stage_plan')):
        raise ValueError('serial scope demand proof is not this original recipe/target')
    if derive_serial_scope_demands(schedule,recipe,graph)!=schedule:
        raise ValueError('serial scope demands differ from complete source replay')
    rows=proof.get('occupancies')
    if type(rows)is not list or len({row['occupancy']for row in rows})!=len(rows):
        raise ValueError('unique source scope execution domains required')
    return ordinal,proof,{row['occupancy']:row for row in rows}


def _workspace_domains(schedule,graph,target_ref):
    """Replay workspace domains from the exact original untimed-cost baseline."""
    from scar.analysis.workspace_demands import SCHEMA,derive_workspace_demands
    from scar.ir.versioned_schedule import schedule_from_dict
    proofs=[(i,row)for i,row in enumerate(schedule.temporal.source_refs)if row.get('schema')==SCHEMA]
    if not proofs:return None
    if len(proofs)!=1:raise ValueError('one original workspace CPU demand proof required')
    ordinal,proof=proofs[0]
    if proof.get('schema_version')!=1 or proof.get('target')!=target_ref:
        raise ValueError('workspace source demand target/version differs')
    parent=schedule_from_dict(read_artifact(proof['parent_schedule']))
    step_plan=read_artifact(proof['stage_plan']);native=read_artifact(proof['native_manifest'])
    replay=derive_workspace_demands(parent,step_plan,native,
        native_reference=proof['native_manifest'],cpp_reference=proof['cpp_source'],
        workspace_plan_reference=proof['workspace_plan'],parent_schedule_reference=proof['parent_schedule'],target=graph)
    if replay!=schedule:raise ValueError('workspace CPU domains differ from complete original source replay')
    rows=proof.get('occupancies')
    if type(rows)is not list or len({row['occupancy']for row in rows})!=len(rows):
        raise ValueError('unique workspace source execution domains required')
    return ordinal,proof,{row['occupancy']:row for row in rows}


def derive_temporal_resources(schedule,target,*,source_context):
    """Return one typed binding per original occupancy, with guarded domains."""
    validate_versioned_schedule(schedule)
    if schedule.temporal is None:raise ValueError('original schema3 temporal schedule required')
    if type(source_context)is not dict:raise TypeError('explicit source resource context required')
    if type(target)is TargetHardwareGraph:
        graph=target
    else:
        path=Path(target).resolve();graph=TargetHardwareGraph(path.parent if path.is_file()else path)
    target_ref=artifact_reference(graph.path/'result.json','scar.target-hardware-graph')
    if encode(source_context.get('target_reference'))!=target_ref:
        raise ValueError('source context and original hardware snapshot differ')
    if read_artifact(target_ref)!=graph.report:raise ValueError('original target report changed')
    snapshot=ProviderArtifact(**target_ref);resources=graph.resources();links=graph.connections()
    by_id={r.id:r for r in resources};by_wire={str(r.id):r for r in resources}
    threads={r.cpu_id:r for r in resources if type(r)is H.HardwareThread}
    caller=_cpu_ids(source_context.get('caller_affinity_cpu_ids'),'caller_affinity_cpu_ids')
    recipe=_cpu_ids(source_context.get('recipe_binding_cpu_ids'),'recipe_binding_cpu_ids')
    refs=source_context.get('source_refs',())
    if type(refs)not in(list,tuple)or any(type(row)is not dict for row in refs):
        raise ValueError('source_refs must retain original reference dictionaries')
    refs=tuple(encode(refs));digest=semantic_sha256(schedule_document(schedule))
    scope_domains=_scope_domains(schedule,graph,target_ref)
    workspace_domains=_workspace_domains(schedule,graph,target_ref)
    if scope_domains is not None and workspace_domains is not None:
        raise ValueError('standalone and workspace source CPU-domain authorities cannot be combined')
    if workspace_domains is not None:scope_domains=workspace_domains
    reference=lambda row:H.TargetResourceRef(snapshot,row.id)
    unresolved=[]
    def frontier(field,reason,**facts):unresolved.append(dict(field=field,reason=reason,**facts))
    # Executor/thread membership and host access are original target edges.
    contains={};access={}
    for ordinal,edge in enumerate(links):
        left,right=by_id[edge.source],by_id[edge.target]
        if edge.kind is H.ConnectionKind.CONTAINS and type(right)is H.HardwareThread:
            contains.setdefault(right.id,[]).append((ordinal,left))
        if (edge.kind is H.ConnectionKind.MEMORY_ACCESS and type(left)is H.CpuExecutor
                and type(right)is H.MemoryDomain and right.kind is H.MemoryKind.HOST):
            access.setdefault(left.id,[]).append((ordinal,right))
    reachable_memories={memory.id:memory for rows in access.values()for _,memory in rows}
    memory_rows=tuple(sorted(reachable_memories.values(),key=lambda r:str(r.id)))
    memory_relations=[]
    for executor_id,rows in access.items():
        for ordinal,memory in rows:
            edge=links[ordinal]
            memory_relations.append(dict(connection_ordinal=ordinal,executor=str(executor_id),
                memory=str(memory.id),locality=edge.locality,conditions=list(edge.conditions),
                evidence=encode(edge.evidence)))
    for relation in memory_relations:
        if relation['conditions']:
            frontier('target.connections/'+str(relation['connection_ordinal']),
                'ORIGINAL_HOST_ACCESS_POLICY_CONDITIONS_NOT_DISCHARGED',conditions=relation['conditions'])
    caller_threads=tuple(threads[cpu]for cpu in(caller or())
        if cpu in threads and threads[cpu].online is not False)
    if caller is not None:
        missing=[cpu for cpu in caller if cpu not in threads or threads[cpu].online is False]
        if missing:frontier('source_context.caller_affinity_cpu_ids',
            'DECLARED_CPU_IDS_MISSING_OR_OFFLINE_IN_ORIGINAL_TARGET',cpu_ids=missing)
    executions=[];storages=[]
    slots={slot.id:slot for slot in schedule.storage_slots};tasks={task.id:task for task in schedule.tasks}
    for ordinal,occupancy in enumerate(schedule.temporal.occupancies):
        source=({'kind':'ORIGINAL_TEMPORAL_OCCUPANCY','source_schedule_sha256':digest,
            'locator':'#/temporal/occupancies/'+str(ordinal)},)
        if occupancy.kind is K.EXECUTION_DEMAND:
            index=len(executions);record=by_wire.get(occupancy.resource)
            exact=record if type(record)is H.HardwareThread else None
            domain_source=None
            if scope_domains is None:
                candidates=(exact,)if exact is not None else caller_threads if record is None else()
            else:
                proof_ordinal,proof,domains=scope_domains
                if occupancy.id not in domains:raise ValueError('execution occupancy lacks its source scope domain')
                domain=domains[occupancy.id];cpu_ids=_cpu_ids(domain['cpu_ids'],'scope.cpu_ids')
                if domain['resource']!=occupancy.resource or domain['units']!=occupancy.units:
                    raise ValueError('execution demand differs from source scope proof')
                if domain['exact_cpu_id']is not None:
                    if (exact is None or exact.cpu_id!=domain['exact_cpu_id']
                            or cpu_ids!=(exact.cpu_id,)):
                        raise ValueError('source selected scope CPU identity differs')
                elif exact is not None:raise ValueError('unknown source scope cannot acquire an exact CPU')
                candidates=tuple(threads[cpu]for cpu in(cpu_ids or())
                    if cpu in threads and threads[cpu].online is not False)
                if cpu_ids is not None and len(candidates)!=len(cpu_ids):
                    raise ValueError('source scope CPU domain is missing/offline in original target')
                domain_source=dict(kind='SOURCE_REPLAYED_EXECUTION_CPU_DOMAIN',
                    source_schedule_sha256=digest,locator='#/temporal/source_refs/'+str(proof_ordinal)
                        +'/occupancies/'+str(proof['occupancies'].index(domain)),
                    domain_state=domain['domain_state'],
                    original_caller_mask_not_selected_mask=True,
                    actual_thread_execution_proven=False,global_reservation=False)
                if workspace_domains is None:domain_source['recipe']=proof['recipe']
                else:domain_source.update(native_manifest=proof['native_manifest'],
                    workspace_plan=proof['workspace_plan'],parent_schedule=proof['parent_schedule'])
            if exact is not None and exact.online is False:
                frontier('executions/'+str(index),'SOURCE_SELECTED_CPU_OFFLINE_IN_TARGET_SNAPSHOT')
            if record is not None and exact is None:
                frontier('executions/'+str(index),'EXECUTION_RESOURCE_IS_NOT_A_CPU_HARDWARE_THREAD',
                    original_resource=occupancy.resource)
            if exact is None and not candidates:
                reason=('OPAQUE_SCOPE_CPU_DOMAIN_UNPROVEN'if scope_domains is not None
                    and domain['domain_state']=='OPAQUE_UNKNOWN'else'ORIGINAL_CALLER_CPU_DOMAIN_UNSPECIFIED')
                frontier('executions/'+str(index)+'/candidate_threads',reason)
            if occupancy.units is None:
                frontier('executions/'+str(index)+'/units','ORIGINAL_EXECUTION_DEMAND_UNITS_UNKNOWN')
            core_rows=tuple(r for r in resources if type(r)is H.CpuCore
                and any(t.cpu_id in r.cpu_ids for t in candidates))
            executors={row.id:row for thread in candidates for _,row in contains.get(thread.id,())
                if type(row)is H.CpuExecutor and thread.cpu_id in row.cpu_ids}
            connection_ordinals=tuple(i for thread in candidates for i,_ in contains.get(thread.id,()))
            access_ordinals=tuple(i for executor in executors for i,_ in access.get(executor,()))
            if candidates and not core_rows:
                frontier('executions/'+str(index)+'/core_refs','ORIGINAL_PHYSICAL_CORE_TOPOLOGY_UNAVAILABLE')
            topology=dict(kind='ORIGINAL_CPU_TOPOLOGY_CANDIDATES',
                contains_connection_ordinals=connection_ordinals,memory_access_connection_ordinals=access_ordinals,
                executor_candidates=tuple(str(key)for key in executors),
                numa_memory_candidates=tuple(str(row.id)for executor in executors for _,row in access.get(executor,())),
                smt_sibling_cpu_ids={str(t.id):t.thread_siblings for t in candidates},
                actual_thread_execution_proven=False,exclusive_physical_core_reservation=False)
            if domain_source is not None:source=(*source,encode(domain_source))
            executions.append(ExecutionBinding(occupancy.id,occupancy.resource,
                None if exact is None else reference(exact),tuple(reference(r)for r in core_rows),
                tuple(reference(r)for r in candidates),occupancy.units,(*source,encode(topology))))
        elif occupancy.kind is K.PHYSICAL_SLOT_BYTES:
            slot=slots[occupancy.storage_slot];index=len(storages)
            allocation=tasks[slot.allocated_by]
            supported=slot.device=='cpu'and allocation.capability=='CPU_ALLOCATE'
            candidates=memory_rows if supported else()
            if not candidates:frontier('storages/'+str(index)+'/candidate_memories',
                'NO_ORIGINAL_HOST_ACCESS_DOMAIN_FOR_THIS_SLOT_LOCATION')
            frontier('storages/'+str(index)+'/exact_memory','PHYSICAL_PAGE_OR_NUMA_RESIDENCY_UNPROVEN')
            storages.append(StorageBinding(occupancy.id,slot.id,tuple(reference(r)for r in candidates),
                None,slot.capacity_bytes,(*source,encode(dict(kind='ORIGINAL_HOST_LOCATION_CANDIDATES',
                    slot_location=slot.location,memory_access_connection_ordinals=tuple(
                        relation['connection_ordinal']for relation in memory_relations),
                    reported_capacity_is_available_budget=False,allocation_or_page_binding_authorized=False)))))
        else:raise ValueError('unsupported original temporal occupancy kind')
    connections=graph.report['connections']
    context=dict(kind='ORIGINAL_SOURCE_RESOURCE_CONTEXT',caller_affinity_cpu_ids=caller,
        recipe_binding_cpu_ids=recipe,
        recipe_binding_authority='ONLY_SELECTED_COMPUTE_DOMAIN_NO_HOST_CONTROL_AUTHORITY',
        candidate_domains_are_runtime_binding=False,
        target_connection_table=dict(uri=str(graph.path/connections['path']),sha256=connections['sha256'],
            size_bytes=connections['bytes'],decoded_sha256=connections['decoded_sha256']),
        memory_access_relations=memory_relations,
        source_budget_interpretation='PRESERVE_ORIGINAL_SCOPE_NOT_RSS_OR_NEW_AVAILABLE_MEMORY_BUDGET')
    if scope_domains is not None:
        ordinal,proof,_=scope_domains
        domain_context=dict(source_schedule_sha256=digest,
            locator='#/temporal/source_refs/'+str(ordinal),
            global_caller_context_is_not_per_scope_domain_authority=True)
        if workspace_domains is None:
            domain_context['recipe']=proof['recipe'];context['serial_scope_cpu_domains']=domain_context
        else:
            domain_context.update(native_manifest=proof['native_manifest'],
                workspace_plan=proof['workspace_plan'],parent_schedule=proof['parent_schedule'])
            context['workspace_scope_cpu_domains']=domain_context
    result=ResourceProjection(snapshot,digest,tuple(executions),tuple(storages),
        OriginalBudget(schedule.budget_bytes,schedule.budget_scope,deepcopy(schedule.budget_refs)),
        tuple(unresolved),(*refs,encode(context)))
    validate_resource_projection(result,schedule,resources=resources)
    read_artifact(target_ref)
    return result


__all__ = [
    'derive_temporal_resources',
    'build_software_resources',
    'SoftwareResourceGraph',
    'main',
]


if __name__=='__main__':main()
