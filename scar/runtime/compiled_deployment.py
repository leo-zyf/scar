"""Replay scoped entry measurements without loading the compiler or graph."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys


def thread_environment():
    names={'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'}
    names.update(name for name in os.environ if name.startswith(('OMP_','GOMP_','KMP_')))
    return {name:os.environ.get(name) for name in sorted(names)}


def entry_context(entry,argv):
    path=Path(entry).resolve()
    return dict(entry_path=str(path),entry_sha256=sha256(path.read_bytes()).hexdigest(),argv=list(argv),
        working_directory=str(Path.cwd().resolve()),python_executable=str(Path(sys.executable).resolve()),
        python_version=sys.version,python_cache_tag=sys.implementation.cache_tag,
        process_affinity_cpu_ids=sorted(os.sched_getaffinity(0)),
        native_thread_environment=thread_environment())


def runner_identity():
    root=Path(__file__).resolve().parents[1]
    names=('entrypoint.py','__main__.py','runtime/compiled_entry.py','runtime/compiled_deployment.py',
        'runtime/entry_runner.py','runtime/native_temporal_invariant.py','runtime/native_loop.py',
        'runtime/loop_guard.py','runtime/native_stateful.py','runtime/stateful_guard.py',
        'runtime/stateful_contract.py','runtime/array_copy_guard.py','trace/code_snapshot.py',
        'optimization/paired_entry_selection.py')
    values=[(name,sha256((root/name).read_bytes()).hexdigest()) for name in names]
    return sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


def selected_rows(compiled):
    loops=[row for row in compiled.get('invariant_loops',()) if row['status']=='MODEL_SELECTED_LOOP_COMPILED']
    composites=[row for row in compiled.get('composite_invariant_loops',())
        if row['status']=='MODEL_SELECTED_COMPOSITE_LOOP_COMPILED']
    states=[row for row in compiled.get('stateful_regions',()) if row['status']=='MEASURED_STATEFUL_BUNDLE_READY']
    return [*loops,*composites,*states]


def _raw(reference):
    path=Path(reference['uri']);raw=path.read_bytes()
    if len(raw)!=reference['size_bytes'] or sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('deployment evidence receipt differs')
    return raw


def deployment_gate(path,pipeline,entry,argv):
    """A receipt authorizes only the same measured entry and implementation."""
    rejected=lambda reason:dict(decision='NO_OP',deployment_authorized=False,reason=reason)
    if path is None:return rejected('deployment_measurement_not_provided')
    try:
        receipt=json.loads(Path(path).read_bytes())
        if (receipt.get('schema'),receipt.get('schema_version'))!=('scar.compiled-program-deployment',1):
            raise ValueError('unsupported deployment receipt')
        if receipt['pipeline']!=pipeline or receipt['entry_context']!=entry_context(entry,argv):
            raise ValueError('entry or pipeline differs from measured context')
        if receipt['runner_identity']!=runner_identity():raise ValueError('runtime implementation changed')
        if receipt['decision']!='LOAD_MODEL_SELECTED_NATIVE':return rejected('measured_choice_retained_original')
        for evidence in receipt['evidence']:_raw(evidence)
        designs=[e for e in receipt['evidence'] if e['schema']=='scar.compiled-program-validation-design']
        if len(designs)!=1:raise ValueError('one frozen validation design required')
        design=json.loads(_raw(designs[0]))
        if any(design[key]!=receipt[key] for key in ('pipeline','entry_context','runner_identity','policy','specializations')):
            raise ValueError('measured policy differs from frozen design')
        compiled=json.loads(_raw(pipeline))
        keys=('id','proof','schedule','bundle_export','bundle','input_shapes','trip_count','memory_budget_bytes','selected_kind')
        specializations=[{key:row[key] for key in keys if key in row} for row in selected_rows(compiled)]
        if specializations!=receipt['specializations']:raise ValueError('measured specialization differs from selected pipeline')
        policy=receipt['policy']
        expected={('WARMUP',i) for i in range(policy['warmup'])}|{('MEASUREMENT',i) for i in range(policy['repetitions'])}
        coordinates=[(sample['phase'],sample['index']) for sample in receipt['samples']]
        if len(coordinates)!=len(expected) or set(coordinates)!=expected or policy.get('frozen_before_measurement') is not True:
            raise ValueError('complete frozen warmup and measurement pairs required')
        measured=[]
        for sample in receipt['samples']:
            original=json.loads(_raw(sample['sample_ref']))
            if original!={key:value for key,value in sample.items() if key!='sample_ref'}:
                raise ValueError('retained sample differs from receipt')
            baseline,optimized=sample['baseline'],sample['optimized']
            stdout=_raw(baseline['stdout'])==_raw(optimized['stdout'])
            stderr=_raw(baseline['stderr'])==_raw(optimized['stderr'])
            if not (sample['valid'] and stdout and stderr and baseline['returncode']==optimized['returncode']==0
                    and sample['native_calls']>0 and sample['fallback_calls']==0):
                raise ValueError('entry validation or native-only execution failed')
            execution=json.loads(_raw(optimized['execution_report']))
            if execution['pipeline']!=pipeline:raise ValueError('execution observed a different pipeline')
            counts=[stats for attachment in execution['attachments'] for stats in attachment['statistics']]
            if (sum(c['native_calls'] for c in counts)!=sample['native_calls']
                    or sum(c['fallback_calls'] for c in counts)!=0):
                raise ValueError('native execution counts differ from retained observations')
            if sample['phase']=='MEASUREMENT':measured.append(sample)
        threshold=policy['minimum_gain'];strict=policy.get('strict_minimum_gain',False)
        if (type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<=threshold<1
                or type(strict)is not bool
                or len(measured)!=policy['repetitions'] or len(measured)<3):
            raise ValueError('complete measured policy required')
        values={kind:[row[kind]['wall_ns'] for row in measured] for kind in ('baseline','optimized')}
        if any(type(v) is not int or v<=0 for group in values.values() for v in group):
            raise ValueError('positive complete-process samples required')
        gain=1-statistics.median(values['optimized'])/statistics.median(values['baseline'])
        if strict:
            from scar.optimization.paired_entry_selection import strict_entry_gain_met

            profitable=strict_entry_gain_met(values['baseline'],values['optimized'],threshold)
        else:profitable=gain>=threshold
        if not profitable:return rejected('measured_gain_below_policy')
        return dict(decision='LOAD_MODEL_SELECTED_NATIVE',deployment_authorized=True,
            gain_fraction=gain,scope='EXACT_MEASURED_ENTRY_AND_REGION_CONTRACTS',
            arbitrary_external_effect_validation=False)
    except (OSError,ValueError,KeyError,TypeError) as error:
        return rejected('deployment_context_or_evidence_invalid:'+str(error))
