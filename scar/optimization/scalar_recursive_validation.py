"""Qualify a measured serial native recursion at its exact prepared boundary."""

import argparse
import json
from pathlib import Path
from statistics import median, stdev

from scar.codegen import scalar_recursive_standalone as native
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_recursive_scope_plan as plan


SCHEMA = 'scar.scalar-recursive-native-assessment'
TIMING = 'scar.scalar-recursive-clean-paired-region'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _profile(record, timing, cases):
    ordinal = record['case']
    _require(type(ordinal) is int and 0 <= ordinal < len(cases),'measured case is outside the admitted native replay')
    case = cases[ordinal]
    result = case['result']
    samples = {'original':[], 'cpp20':[]}
    rounds, repetitions = timing['measured_pairs'], timing['repetitions_per_arm']
    rows = record['raw']
    _require(type(result) is int and len(rows)==2*(1+rounds),'paired native timing requires a matched normal completion')
    for i,row in enumerate(rows):
        round_, index = divmod(i,2)
        mode = tuple(samples)[index if round_%2==0 else 1-index]
        _require(row['mode']==mode and row['round']==round_ and row['stops']==0 and row['checksum']==result*repetitions
                 and type(row['scope_ns']) is int and row['scope_ns']>0,'invalid timing observation or result check')
        if round_:
            samples[mode].append(row['scope_ns']/repetitions)
    summary = {mode:dict(samples_ns=values, median_ns=median(values), min_ns=min(values),
                         max_ns=max(values), stdev_ns=stdev(values)) for mode,values in samples.items()}
    left, right = summary['original'],summary['cpp20']
    gain = left['median_ns']/right['median_ns']
    _require(summary==record['measurements'] and gain==record['median_speedup'],
             'reported native timings differ from independent pairing')
    return dict(case=ordinal, measurements=summary, median_speedup=gain,
        separates_measured_envelopes=right['max_ns']<left['min_ns'],
        original_case_kind=case['oracle_kind'])


def assess(plan_path, timing_path):
    plan_ref=artifact_reference(Path(plan_path).resolve(strict=True),plan.SCHEMA)
    timing_ref=artifact_reference(Path(timing_path).resolve(strict=True),TIMING)
    model=plan.check(plan_ref['uri'])
    timing=read_artifact(timing_ref)
    native_result=native.check(timing['artifact']['uri'])
    _require(native_result==read_artifact(timing['artifact']) and native_result['cases']==timing['cases']
             and native_result['region']==model['artifacts']['recursive_region']
             and native_result['native_recursion'] and not native_result['whole_driver_cpp']
             and model['decision']['experimental_nomination']=='COLOCATE_RECURSIVE_OWNER_REGION_IN_CPP',
             'measured native scope does not match the nominated original region')
    _require(timing['scope']=='PREPARED_RECURSIVE_BODY_WITH_NATIVE_CONTAINERS_AND_CLOCK_CHECK'
             and timing['schema_version']==1 and timing['warmup_pairs']==1
             and timing['include_program_setup_wire_and_uci'] is False and timing['whole_search_proven'] is False
             and type(timing['measured_pairs']) is int and timing['measured_pairs']>=3
             and type(timing['repetitions_per_arm']) is int and timing['repetitions_per_arm']>=1
             and len(timing['cpu'])==1 and type(timing['cpu'][0]) is int and timing['cpu'][0]>=0,
             'timing evidence must isolate the same admitted single-CPU recursive boundary')
    cases=read_artifact(timing['cases'])['cases']
    _require(len(timing['profiles'])>=1 and len({row['case'] for row in timing['profiles']})==len(timing['profiles']),
             'each retained depth scenario needs one paired profile')
    profiles=[_profile(row,timing,cases) for row in timing['profiles']]
    return dict(schema=SCHEMA,schema_version=1,scope_plan=plan_ref,timing=timing_ref,
        validated_native_cases=native_result['validated_cases'], profiles=profiles, cpu=timing['cpu'],
        warmup_pairs=timing['warmup_pairs'], measured_pairs=timing['measured_pairs'],
        repetitions_per_arm=timing['repetitions_per_arm'],
        prepared_selection='CPP20' if all(row['separates_measured_envelopes'] for row in profiles) else 'ORIGINAL',
        comparison_scope='PREPARED_SAME_OWNER_RECURSIVE_BODY', whole_program_effects_closed=False,
        whole_program_performance_status='UNKNOWN', deployment_authorized=False)


def check(path):
    recorded=read_artifact(artifact_reference(Path(path).resolve(strict=True),SCHEMA))
    _require(recorded==assess(recorded['scope_plan']['uri'],recorded['timing']['uri']),
             'prepared native assessment differs from its cold replay')
    return recorded


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','timing','out','check'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args(argv)
    inputs=(args.plan,args.timing,args.out)
    if args.check and not any(inputs):
        result=check(args.check)
    elif not args.check and all(inputs):
        result=assess(args.plan,args.timing)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --plan, --timing, and --out, or --check')
    print(json.dumps({key:result[key] for key in ('prepared_selection','validated_native_cases',
                                                 'whole_program_performance_status','deployment_authorized')}))


if __name__=='__main__':
    main()
