"""Measure canonical stateful resource components, never a target implementation."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from scar.ir.execution_recipe import artifact_reference
from scar.ir.record_codec import loads
from scar.optimization.stateful_cpu_choices import _domain
from scar.optimization.stateful_cpu_model import (stateful_cpu_demand,component_key,
    predict_stateful_cpu_model)


def _now(): return datetime.now(timezone.utc).isoformat()


def _load():
    try:return list(os.getloadavg())
    except OSError:return None


def calibrate_stateful_cpu(domain,candidate_id,out,*,repetitions=7,warmup=2):
    """Sample one exact CPU mask's independent primitives and emit a partial model.

    The helper compiles canonical resource kernels only. Its native-clock
    timings exclude Python receiver guards, descriptor packing and FFI; absent
    boundary/fused-DAG measurements remain explicit unsatisfied components.
    """
    from scar.codegen.cpu_resource_probe import build_resource_probe
    if sys.gettrace() is not None or sys.getprofile() is not None:
        raise ValueError('resource calibration requires no Python profile/trace hook')
    if type(repetitions) is not int or repetitions<3 or type(warmup) is not int or warmup<1:
        raise ValueError('at least three samples and one warmup required')
    domain=_domain(domain);out=Path(out).absolute()
    if out.exists():raise ValueError('fresh stateful calibration output required')
    row=next((r for r in domain['candidates'] if r['id']==candidate_id),None)
    if row is None or not row['legal_under_guards']:raise ValueError('supported CPU candidate required')
    demand=stateful_cpu_demand(domain,candidate_id)
    implementation=domain['backend_capabilities']['serial' if row['worker_count']==1 else 'parallel']
    context=dict(cpu_ids=row['binding']['cpu_ids'],compiler=implementation['compiler'],
        provider_runtime=implementation['provider_runtime'],implementation_sha256=implementation['implementation_sha256'],
        target_sha256=domain['target_snapshot']['sha256'],executor_id=row['binding']['executor_id'],
        caller_affinity_cpu_ids=domain['live_constraints']['process_affinity_cpu_ids'])
    observed_from=_now();load_before=_load();started=time.perf_counter()
    probe=build_resource_probe(context,out/'native')
    components=[];unresolved=[];extras=[];seen=set()
    for term in demand['terms']:
        if term['key'] in seen:continue
        seen.add(term['key'])
        try:
            measured=probe.sample(term['component'],term['parameters'],repetitions=repetitions,warmup=warmup)
            if measured.get('supported') is False or measured.get('applicable_to_component') is False:
                raise NotImplementedError(measured.get('reason','canonical proxy does not cover this complete component'))
            samples=measured['samples_ns']
            if (len(samples)!=repetitions or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in samples)):
                raise ValueError('invalid direct resource timing samples')
            components.append(dict(component=term['component'],parameters=term['parameters'],
                basis='DIRECT_COMPONENT_TIMING',median_ns=statistics.median(samples),
                min_ns=min(samples),max_ns=max(samples),stdev_ns=statistics.stdev(samples),
                samples_ns=samples,measurement=measured))
        except NotImplementedError as error:
            unresolved.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
    # Diagnostic only: stage observations already include their own implicit
    # barriers, so this extra primitive must never be added to those terms.
    try:
        measured=probe.sample('OMP_BARRIER_ONLY',dict(workers=row['worker_count']),repetitions=repetitions,warmup=warmup)
        extras.append(dict(component='OMP_BARRIER_ONLY',measurement=measured,
            used_in_composition=False,reason='stage timers already include implicit barrier'))
    except NotImplementedError:pass
    report=probe.report
    document=dict(schema='scar.stateful-resource-calibration',schema_version=1,
        resource_context=demand['resource_context'],probe_context=context,components=components,
        unsupported_components=unresolved,extra_observations=extras,
        repetitions=repetitions,warmup=warmup,observed_from=observed_from,observed_until=_now(),
        seconds=time.perf_counter()-started,host_load=dict(before=load_before,after=_load(),
            isolated=False,contention_not_controlled=True),
        probe=report,timing_scope='DIRECT_INDEPENDENT_RESOURCE_COMPONENTS_NOT_COMPLETE_METHOD_OR_PROGRAM',
        implicit_stage_barrier_cost_already_included=True,
        no_scope_median_subtraction=True,logical_bytes_are_dram_bytes=False,
        numa_page_residency_proven=False,caller_placement='ORIGINAL_CALLER_MASK_SEPARATE_FROM_PINNED_WORKER_TEAM',
        target_workload_executed=False,target_variants_compiled=False,target_variants_ranked=False,
        physical_optimality_proven=False)
    out.mkdir(parents=True,exist_ok=True)
    path=out/'observations.json'
    with path.open('x') as stream:json.dump(document,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    model=predict_stateful_cpu_model(domain,candidate_id,[path],out=out/'model')
    result=dict(schema='scar.stateful-resource-calibration-result',schema_version=1,
        observations=artifact_reference(path,document['schema']),candidate_id=candidate_id,
        measured_component_signatures=len(components),missing_component_signatures=len(unresolved),
        model=str(out/'model/model.json'),cost_profile=model['profile'],
        complete_method_cost=model['complete'],known_component_sum_ns=model['known_component_sum_ns'],
        target_workload_executed=False,target_variants_ranked=False)
    with (out/'result.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True)
    p.add_argument('--repetitions',type=int,default=7);p.add_argument('--warmup',type=int,default=2)
    args=p.parse_args(argv)
    print(json.dumps(calibrate_stateful_cpu(args.domain,args.candidate,args.out,
        repetitions=args.repetitions,warmup=args.warmup),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
