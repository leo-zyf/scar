"""Read canonical layout observations without loading or timing native code."""
import argparse
import json
import math
from pathlib import Path
import statistics

from . import cpu_layout_probe as P
from .cpu_fused_cost import _receipt,_require,_save
from scar.ir.execution_recipe import artifact_reference,read_artifact


def audit_layout_observations(observations,out):
    reference=artifact_reference(observations,'scar.cpu-layout-diagnostic-observations')
    raw=read_artifact(reference);pack=read_artifact(raw['pack']);design=read_artifact(raw['design'])
    _require(raw['schema_version']==pack['schema_version']==design['schema_version']==1
        and pack['schema']==P.PACK and pack['registry']==P._registry()
        and pack['configurations']==P._configurations(),'original canonical layout design differs')
    for receipt in (*pack['implementation'],pack['source'],pack['assembly'],pack['library']):
        _require(_receipt(Path(receipt['path']))==receipt,'layout source/native receipt differs')
    _require(Path(pack['source']['path']).read_text()==P._source(pack['resource_context']['caller_affinity_cpu_ids']),
        'retained canonical layout source differs from original rendering')
    _require(design['pack']==raw['pack'] and design['policy']==pack['policy']
        and design['configurations']==pack['configurations'] and len(raw['records'])==len(pack['configurations'])
        and raw['exploratory_diagnostic_only'] is True and raw['holdout_qualification'] is False
        and raw['model_fit_performed'] is False and raw['target_workload_executed'] is False
        and raw['target_variants_ranked'] is False and raw['cost_selection_authorized'] is False,
        'diagnostic observations cannot authorize a cost model or transformation')
    cpus=pack['resource_context']['worker_cpu_ids'];page=raw['page_size'];rows=[]
    _require(raw['actual_worker_cpu_ids']==cpus and type(page)is int and page>0 and page&(page-1)==0,
        'recorded worker/page metadata differs')
    for record,configuration in zip(raw['records'],pack['configurations']):
        _require(all(record[k]==v for k,v in configuration.items()),'raw configuration order/metadata differs')
        times=record['samples_ns'];worker=record['per_worker_ns'];addresses=record['allocation_addresses']
        _require(len(times)==len(worker)==pack['policy']['repetitions'] and len(addresses)==9
            and all(type(x)in(int,float) and math.isfinite(x) and x>0 for x in times)
            and all(len(w)==len(cpus) and all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in w)
                and max(w)==t for w,t in zip(worker,times))
            and statistics.median(times)==record['median_ns'] and record['actual_worker_cpu_ids']==cpus,
            'raw per-worker elapsed/maximum/median differs')
        _require(all(type(a)is int and 0<=a<2**64 and a%8==0 for a in addresses)
            and record['page_offsets']==[a%page if a else None for a in addresses]
            and record['cache_line_offsets_64']==[a%64 if a else None for a in addresses]
            and record['allocation_identity_scope']=='THIS_PROBE_CALL_ONLY_NOT_CROSS_CALL_STORAGE_ID',
            'recorded allocation alignment/identity differs')
        sizes={}
        for layout in record['layouts']:
            upper=(record['rows']-1)*layout['stride']+layout['offset']+record['cols']
            group=layout['storage_group'];sizes[group]=max(sizes.get(group,0),upper*8)
        _require(all((address>0)==(i in sizes or i==8) for i,address in enumerate(addresses)),
            'unallocated or missing input/output allocation')
        sizes[8]=record['rows']*record['cols']*8
        intervals=sorted((addresses[i],addresses[i]+size) for i,size in sizes.items())
        _require(all(0<start<end<2**64 for start,end in intervals)
            and all(left[1]<=right[0] for left,right in zip(intervals,intervals[1:])),
            'canonical fresh backing/output allocations overlap')
        rows.append(dict(configuration=record['id'],case=pack['registry'][record['case_index']]['id'],
            role=record['role'],shape=[record['rows'],record['cols']],array_loads=record['array_loads'],
            backing_groups=record['groups'],median_ns=record['median_ns'],samples_ns=times,
            repeat_span_fraction=(max(times)-min(times))/record['median_ns'],
            maximum_worker_span_fraction=max((max(w)-min(w))/max(w) for w in worker),
            page_offsets=record['page_offsets'],offsets_modulo_64=record['cache_line_offsets_64']))
    result=dict(schema='scar.cpu-layout-diagnostic-audit',schema_version=1,observations=reference,
        pack=raw['pack'],design=raw['design'],configuration_count=len(rows),records=rows,
        source_and_raw_statistics_verified=True,compiler_or_native_executed=False,
        addresses_are_per_call_only=True,worker_clock_scope='WHOLE_KERNEL_BATCH_INCLUDING_COMPLETION_BARRIERS',
        compute_and_barrier_wait_separated=False,physical_cache_cause_established=False,
        model_fit_performed=False,holdout_qualification=False,cost_selection_authorized=False,
        interpretation='EXPLORATORY_OBSERVATIONS_NOT_A_PAIRED_TARGET_BENCHMARK_OR_PHYSICAL_CAUSAL_PROOF')
    out=Path(out).resolve();_require(not out.exists(),'fresh diagnostic audit required')
    _save(out,result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('observations');parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);r=audit_layout_observations(args.observations,args.out)
    print(json.dumps(dict(configuration_count=r['configuration_count'],source_and_raw_statistics_verified=True,
        cost_selection_authorized=False)));return 0


if __name__=='__main__':raise SystemExit(main())
