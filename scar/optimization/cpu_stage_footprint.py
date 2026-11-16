"""Mechanical access-set geometry over retained CPU/cache topology.

Intervals denote elements touched by a whole static row chunk. They are neither
the peak live working set nor a cache-residency requirement. Unknown allocation
base addresses produce cache-line phase bounds; no timing or hit/miss is inferred.
"""
from dataclasses import asdict,dataclass,is_dataclass
import math
from pathlib import Path

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode


@dataclass(frozen=True,slots=True,order=True)
class ElementInterval:
    start: int
    end: int


@dataclass(frozen=True,slots=True)
class ThreadRows:
    worker: int
    cpu_id: int
    first_row: int
    end_row: int


def _merge(intervals):
    result=[]
    for first,end in sorted(intervals):
        if end<=first:continue
        if result and first<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((first,end))
    return tuple(result)


def _partition(rows,cpu_ids):
    q,r=divmod(rows,len(cpu_ids))
    return tuple(ThreadRows(i,cpu,i*q+min(i,r),i*q+min(i,r)+q+(i<r))
        for i,cpu in enumerate(cpu_ids))


def _line_bounds(intervals,element_bytes,line_bytes,base_alignment_bytes):
    """Exact extrema for one storage under its unknown allowed base phase."""
    phases=range(0,line_bytes,math.gcd(base_alignment_bytes,line_bytes))
    counts=[]
    for phase in phases:
        lines=_merge(((a*element_bytes+phase)//line_bytes,
            ((b*element_bytes+phase-1)//line_bytes)+1) for a,b in intervals)
        counts.append(sum(b-a for a,b in lines))
    return dict(lower=min(counts),upper=max(counts),line_size_bytes=line_bytes,
        allowed_base_phase_count=len(counts),base_alignment_bytes=base_alignment_bytes,
        base_address_known=False,authority='EXACT_EXTREMA_OVER_ALLOWED_STORAGE_BASE_PHASES')


def _access_set(partitions,layouts,cols):
    groups={}
    for partition in partitions:
        for layout in layouts:
            key=layout['storage_group']
            group=groups.setdefault(key,[])
            group.extend((layout['offset']+row*layout['stride'],
                layout['offset']+row*layout['stride']+cols)
                for row in range(partition.first_row,partition.end_row))
    return {key:_merge(intervals) for key,intervals in groups.items()}


def _set_summary(groups,element_bytes,line_bytes,alignment):
    storage=[];unique_bytes=0
    for key,intervals in sorted(groups.items(),key=lambda item:(type(item[0]).__name__,repr(item[0]))):
        elements=sum(end-first for first,end in intervals);unique_bytes+=elements*element_bytes
        bounds=_line_bounds(intervals,element_bytes,line_bytes,alignment) if line_bytes else None
        storage.append(dict(storage_group=key,element_intervals=[list(r) for r in intervals],
            unique_elements=dict(lower=elements,upper=elements,exact=True),
            unique_bytes=elements*element_bytes,cache_line_bounds=bounds))
    physical=None;scoped=None
    if line_bytes:
        lower=sum(row['cache_line_bounds']['lower'] for row in storage)
        upper=sum(row['cache_line_bounds']['upper'] for row in storage)
        scoped=dict(lower=lower,upper=upper,
            identity='STORAGE_GROUP_AND_LOCAL_LINE_OFFSET_NOT_PHYSICAL_LINE_ADDRESS')
        # Distinct live byte regions can still share boundary cache lines.
        # Without their relative base addresses, summing per-storage minima is
        # not a proved lower bound on the physical union.
        physical=dict(lower=max([math.ceil(unique_bytes/line_bytes),
                *(row['cache_line_bounds']['lower'] for row in storage)]),upper=upper,
            authority='CONSERVATIVE_UNION_BOUNDS_WITH_UNKNOWN_RELATIVE_BASE_PLACEMENT',
            premise='DISTINCT_STORAGE_GROUPS_DESCRIBE_DISJOINT_LIVE_BYTE_REGIONS',
            relative_storage_placement_known=False)
    elements=unique_bytes//element_bytes
    return dict(storage_groups=storage,unique_elements=dict(lower=elements,upper=elements,exact=True),
        unique_bytes=unique_bytes,storage_scoped_cache_line_bounds=scoped,
        distinct_physical_cache_line_bounds=physical)


def _materials(binding,inventory):
    graph=inventory if isinstance(inventory,TargetHardwareGraph) else TargetHardwareGraph(binding['target_path'])
    target=artifact_reference(graph.path/'result.json','scar.target-hardware-graph')
    inventory_ref=artifact_reference(graph.report['inventory']['uri'],'scar.target-inventory')
    if isinstance(inventory,(str,Path)):
        supplied=read_artifact(artifact_reference(inventory,'scar.target-inventory'))
    elif isinstance(inventory,dict):
        supplied=read_artifact(inventory) if 'uri' in inventory else inventory
    else:supplied=graph.inventory
    if supplied!=graph.inventory:raise ValueError('supplied inventory differs from retained target')
    if binding.get('target_sha256',target['sha256'])!=target['sha256']:
        raise ValueError('candidate target snapshot differs')
    return graph,target,inventory_ref,graph.resources()


def derive_cpu_stage_footprint(shape,operand_layouts,binding,inventory,*,element_bytes=8,
    base_alignment_bytes=1,layout_source=None,output_layouts=(),output_source=None):
    """Return conditional geometry, with no cache behavior or duration claims.

    ``storage_group`` identifies a declared disjoint physical byte region;
    multiple operands in that group share its unknown base address. The optional
    output layouts must explicitly identify their own or aliased storage group.
    Alignment is a caller-supplied premise, defaulting to an unknown byte phase.
    ``inventory`` accepts a retained inventory ref/path/document or an already
    admitted TargetHardwareGraph, so one graph admission can serve many stages.
    """
    binding=asdict(binding) if is_dataclass(binding) else dict(binding)
    shape=tuple(shape);layouts=[dict(r) for r in operand_layouts];outputs=[dict(r) for r in output_layouts]
    if len(shape)!=2 or any(type(v) is not int or v<1 for v in shape):
        raise ValueError('positive rank2 shape required')
    if not layouts or len(layouts)>8:raise ValueError('one to8 operand layouts required')
    if type(element_bytes) is not int or element_bytes<1 or type(base_alignment_bytes) is not int or base_alignment_bytes<1:
        raise ValueError('positive element and base-alignment byte sizes required')
    for layout in layouts+outputs:
        if not isinstance(layout.get('storage_group'),(str,int)) or isinstance(layout['storage_group'],bool):
            raise ValueError('explicit scalar storage-group identity required')
        if any(type(layout.get(k)) is not int or layout[k]<0 for k in ('stride','offset')):
            raise ValueError('nonnegative element stride and offset required')
    cpu_ids=list(binding['cpu_ids'])
    if not cpu_ids or len(set(cpu_ids))!=len(cpu_ids) or any(type(c) is not int or c<0 for c in cpu_ids):
        raise ValueError('distinct selected CPU ids required')
    graph,target,inventory_ref,resources=_materials(binding,inventory)
    executors=[r for r in resources if isinstance(r,H.CpuExecutor)]
    selected_executor=next((r for r in executors if str(r.id)==binding['executor_id']),None)
    if selected_executor is None or not set(cpu_ids)<=set(selected_executor.cpu_ids):
        raise ValueError('selected CPUs outside retained executor')
    rows,cols=shape;partitions=_partition(rows,cpu_ids)
    thread_sets={p.cpu_id:_access_set((p,),layouts+outputs,cols) for p in partitions}
    all_sets=_access_set(partitions,layouts+outputs,cols)
    threads=[dict(**asdict(p),row_count=p.end_row-p.first_row,
        access_set=_set_summary(thread_sets[p.cpu_id],element_bytes,None,base_alignment_bytes)) for p in partitions]
    cores=[];covered=set()
    for core in (r for r in resources if isinstance(r,H.CpuCore)):
        selected=sorted(set(core.cpu_ids)&set(cpu_ids))
        if not selected:continue
        covered.update(selected)
        cores.append(dict(resource=encode(core,record_tags=H.TARGET_RECORD_TAGS),selected_cpu_ids=selected,
            selected_thread_count=len(selected),retained_sibling_count=len(core.cpu_ids),
            selected_additional_smt_siblings=max(0,len(selected)-1),
            selection_is_measured_cpu_utilization=False))
    caches=[];unselected=[]
    for cache in (r for r in resources if isinstance(r,H.CacheDomain)):
        selected=sorted(set(cache.shared_cpu_ids)&set(cpu_ids))
        if not selected:unselected.append(str(cache.id));continue
        applicable=cache.cache_type in ('Data','Unified')
        row=dict(resource=encode(cache,record_tags=H.TARGET_RECORD_TAGS),selected_cpu_ids=selected,
            data_access_footprint_applicable=applicable,reported_capacity_bytes=cache.capacity_bytes,
            capacity_is_available_budget=False,cache_inclusion_policy_assumed=False)
        if applicable:
            group={}
            for cpu in selected:
                for key,intervals in thread_sets[cpu].items():group.setdefault(key,[]).extend(intervals)
            group={key:_merge(intervals) for key,intervals in group.items()}
            summary=_set_summary(group,element_bytes,cache.line_size_bytes,base_alignment_bytes)
            row['access_set']=summary
            physical=summary['distinct_physical_cache_line_bounds']
            if physical and cache.capacity_bytes:
                row['geometric_line_bytes_to_reported_capacity_ratio']=dict(
                    lower=physical['lower']*cache.line_size_bytes/cache.capacity_bytes,
                    upper=physical['upper']*cache.line_size_bytes/cache.capacity_bytes,
                    meaning='WHOLE_CHUNK_TOUCHED_SET_NOT_SIMULTANEOUS_RESIDENCY_OR_CAPACITY_REQUIREMENT')
            else:row['geometric_line_bytes_to_reported_capacity_ratio']=None
        else:row['access_set']=None
        caches.append(row)
    return dict(schema='scar.cpu-stage-footprint',schema_version=1,shape=list(shape),operand_layouts=layouts,
        output_layouts=outputs,element_bytes=element_bytes,base_alignment_bytes=base_alignment_bytes,
        layout_source=layout_source,output_source=output_source,binding=binding,target=target,
        inventory=inventory_ref,executor=encode(selected_executor,record_tags=H.TARGET_RECORD_TAGS),
        partition_rule='GOMP_STATIC_CONTIGUOUS_FIRST_REMAINDER',thread_partitions=threads,
        physical_cores=cores,physical_core_topology_unknown_cpu_ids=sorted(set(cpu_ids)-covered),
        whole_team_access_set=_set_summary(all_sets,element_bytes,None,base_alignment_bytes),
        cache_domains=caches,unselected_cache_ids_retained_in_target=unselected,
        footprint_scope='READS_AND_EXPLICIT_OUTPUTS' if outputs else 'READ_OPERANDS_ONLY',
        authority='CONDITIONAL_EXACT_ELEMENT_UNIONS_AND_CONSERVATIVE_CACHE_LINE_GEOMETRY',
        premises=['LAYOUT_STORAGE_GROUP_IDENTITIES_AND_BYTE_DISJOINTNESS_ARE_DECLARED',
            'ELEMENT_STRIDE_OFFSET_AND_SHAPE_DESCRIBE_THE_ACTUAL_ACCESS_PATTERN',
            'RETAINED_TARGET_CPU_NUMBERING_MATCHES_THE_CANDIDATE',
            'EACH_SELECTED_CPU_EXECUTES_ITS_DECLARED_STATIC_ROW_PARTITION'],
        footprint_is_peak_live_set=False,footprint_is_reuse_distance=False,
        cache_residency_or_hit_miss_proven=False,physical_traffic_estimated=False,
        cost_estimated=False,physical_bottleneck_identified=False)
