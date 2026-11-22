"""Measure locked CPU cases and qualify native delivery or runtime attachment."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scar.artifacts import file_reference
from scar.ir.execution_recipe import read_artifact
from scar.runtime.cpu_paired_runner import compare_native_outputs
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN, strict_entry_gain_met


def executable_receipt(path):
    ref = file_reference(path, 'binary')
    return dict(path=str(path), sha256=ref['sha256'], size_bytes=ref['size_bytes'])


class NativeCases:
    def __init__(self, request, cpu, seconds):
        self.baseline = request['baseline']
        self.cases = request['cases']
        self.cpu = cpu
        self.seconds = seconds
        self.binary = executable_receipt(self.baseline['argv'][0])
        self.stdin = {case['id']: Path(case['stdin']).resolve(strict=True) if case.get('stdin') else None
                      for case in self.cases}
        self.digests = self.input_digests()

    def input_digests(self):
        return {name: file_reference(path, 'stdin')['sha256'] if path is not None else sha256(b'').hexdigest()
                for name, path in self.stdin.items()}

    def check_locks(self, observation):
        current = executable_receipt(self.baseline['argv'][0])
        if current != observation['baseline'] or self.digests != observation['case_inputs']:
            raise ValueError('baseline executable or case input differs from the observation')
        if self.input_digests() != self.digests:
            raise ValueError('input changed during qualification')

    def run(self, case, command, cwd, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('native case exceeded its time budget')
        argv = ['taskset', '-c', str(self.cpu), *command, *case.get('argv', ())]
        path = self.stdin[case['id']]
        with path.open('rb') if path is not None else nullcontext(subprocess.DEVNULL) as source:
            start = time.perf_counter_ns()
            result = subprocess.run(argv, cwd=cwd, stdin=source, capture_output=True, timeout=remaining)
            return result, time.perf_counter_ns() - start

    @property
    def original(self):
        return self.baseline['argv'], self.baseline['cwd']

    def observe(self, request):
        case = self.cases[0]
        result, elapsed = self.run(case, *self.original, time.monotonic() + self.seconds)
        observation = dict(schema='scar.native-entry-cost', schema_version=1, request=request,
                           scope='ONE_BASELINE_LAUNCH', cpu=self.cpu, elapsed_ns=elapsed,
                           exit=result.returncode, baseline=self.binary, case=case['id'], case_inputs=self.digests)
        self.check_locks(observation)
        return observation

    def paired(self, executable, out, observation, repetitions):
        self.check_locks(observation)
        rows = []
        for case in self.cases:
            name = case['id']
            folder = out / name
            folder.mkdir(parents=True)
            deadline = time.monotonic() + self.seconds
            arms = {'original': self.original, 'generated': ([str(executable)], folder)}
            records = []
            for index in range(repetitions + 1):
                order = tuple(arms) if index % 2 == 0 else tuple(reversed(arms))
                results, measurements = {}, {}
                for arm in order:
                    result, elapsed = self.run(case, *arms[arm], deadline)
                    results[arm] = result
                    measurements[arm] = dict(ns=elapsed, exit=result.returncode)
                    self.retain(folder, f'{index}-{arm}', result)
                matches = compare_native_outputs(case, results['original'], results['generated'])
                records.append(dict(warmup=index == 0, order=order, arms=measurements, matches=matches))
                if not matches:
                    break
            timing = {}
            for arm in arms:
                samples = [row['arms'][arm]['ns'] for row in records if not row['warmup']]
                if samples:
                    timing[arm] = dict(median_ns=statistics.median(samples), min_ns=min(samples), max_ns=max(samples))
            rows.append(dict(case=name, matches=records[-1]['matches'], stdin_sha256=self.digests[name],
                             comparison=case.get('comparison', 'exact'), pairs=records, timing=timing))
        self.check_locks(observation)
        return rows

    @staticmethod
    def retain(folder, label, result):
        for stream in ('stdout', 'stderr'):
            (folder / f'{label}.{stream}').write_bytes(getattr(result, stream))

    def calibrate(self, observation, directory, repetitions, build_variant):
        count = max(3, repetitions)
        observation.update(schema_version=2, scope='PAIRED_CPP20_VARIANTS', measurement_pairs=count,
                           variants={}, cases={})
        self.check_locks(observation)
        original, original_ref = build_variant('original')
        observation['variants']['original'] = original_ref
        try:
            canonical, canonical_ref = build_variant('canonical')
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            observation['calibration_error'] = f'canonical build failed: {error}'
            return
        observation['variants']['canonical'] = canonical_ref
        routes = {'original': original, 'canonical': canonical}
        for case in self.cases:
            name = case['id']
            folder = directory / name
            folder.mkdir(parents=True)
            deadline = time.monotonic() + self.seconds

            def measure(label, command, cwd):
                result, elapsed = self.run(case, command, cwd, deadline)
                self.retain(folder, label, result)
                return dict(ns=elapsed, exit=result.returncode, **{
                    stream: file_reference(folder / f'{label}.{stream}', f'scar.native-{stream}')
                    for stream in ('stdout', 'stderr')})

            try:
                reference = measure('reference', *self.original)
                pairs = []
                for index in range(count + 1):
                    order = tuple(routes) if index % 2 == 0 else tuple(reversed(routes))
                    pair = {}
                    for mode in order:
                        label = f'{index}-{mode}'
                        cwd = folder / label
                        cwd.mkdir()
                        pair[mode] = measure(label, [str(routes[mode])], cwd)
                    pairs.append(pair)
                observation['cases'][name] = dict(reference=reference, pairs=pairs)
            except (TimeoutError, subprocess.TimeoutExpired) as error:
                observation['calibration_error'] = f'incomplete paired calibration at {name}: {error}'
                break
        self.check_locks(observation)
        for mode, reference in observation['variants'].items():
            binary = read_artifact(reference)['binary']
            if file_reference(routes[mode], binary['schema']) != binary:
                raise ValueError('C++20 trial executable changed during comparison')


def _reference(path, schema):
    return file_reference(Path(path).resolve(), schema)


def _save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return _reference(path,value['schema'])


def _equal_files(left,right):
    with Path(left).open('rb') as a,Path(right).open('rb') as b:
        while True:
            x=a.read(1024*1024);y=b.read(1024*1024)
            if x!=y:return False
            if not x:return True


def _statistics(samples):
    return dict(samples_ns=samples,median_ns=statistics.median(samples) if samples else None,
        min_ns=min(samples) if samples else None,max_ns=max(samples) if samples else None,
        stdev_ns=statistics.stdev(samples) if len(samples)>1 else 0. if samples else None)


def _native_counts(report,pipeline,entry,argv):
    if (report.get('schema'),report.get('schema_version'))!=('scar.compiled-program-execution',1):
        return 0,0,False
    if (report.get('mode')!='MEASURE_SELECTED_MODEL_IMPLEMENTATION' or report.get('pipeline')!=pipeline
            or report.get('entry')!=str(entry) or report.get('argv')!=list(argv)):
        return 0,0,False
    native=fallback=0; attached=False; clean=True
    for attachment in report.get('attachments',()):
        clean=clean and not attachment.get('frontier')
        for row in attachment.get('statistics',()):
            a,b=row.get('native_calls'),row.get('fallback_calls')
            if type(a) is not int or type(b) is not int or a<0 or b<0:clean=False;continue
            native+=a;fallback+=b;attached=attached or attachment.get('bindings',0)>0
    return native,fallback,bool(clean and attached and native>0 and fallback==0)


def validate_build(pipeline,entry,argv,out,*,warmup=1,repetitions=3,minimum_gain=DEFAULT_MINIMUM_GAIN):
    """Measure original versus the model-selected pipeline, never rank variants.

    The declaration and runner identity are frozen before the first child.
    A deployment receipt qualifies exactly this entry/argv/cwd/interpreter,
    inherited CPU-affinity/thread-environment and pipeline, not other inputs.
    """
    from scar.runtime.compiled_deployment import entry_context,runner_identity,selected_rows
    from scar.runtime.compiled_entry import read_pipeline

    entry=Path(entry).resolve();out=Path(out).resolve();argv=list(argv)
    if (type(warmup) is not int or warmup<0 or type(repetitions) is not int or repetitions<3
            or type(minimum_gain) not in (int,float) or not math.isfinite(minimum_gain)
            or not 0<=minimum_gain<1 or any(type(value) is not str for value in argv)):
        raise ValueError('nonnegative warmup, at least three repetitions, finite gain in [0,1) and string argv required')
    if out.exists():raise ValueError('fresh compiled-program validation output required')
    if entry.suffix!='.py' or not entry.is_file():raise ValueError('original Python script entry required')
    pipeline_ref, compiled = read_pipeline(pipeline)
    pipeline = Path(pipeline_ref['uri'])
    selected=selected_rows(compiled)
    if not selected:raise ValueError('an already-model-selected compiled loop implementation is required')
    specializations=[]
    for row in selected:
        specializations.append({key:row[key] for key in ('id','proof','schedule','bundle_export','bundle',
            'input_shapes','trip_count','memory_budget_bytes','selected_kind') if key in row})
        for key in ('proof','schedule','bundle_export'):
            if key in row:read_artifact(row[key])
    context=entry_context(entry,argv);runner=runner_identity();environment=dict(os.environ)
    policy=dict(warmup=warmup,repetitions=repetitions,minimum_gain=minimum_gain,strict_minimum_gain=True,
        frozen_before_measurement=True,admission='EXACT_OUTPUTS_NATIVE_ONLY_AND_MEDIAN_GAIN_ABOVE_DECLARED_THRESHOLD')
    out.mkdir(parents=True);samples_folder=out/'samples';samples_folder.mkdir()
    design=dict(schema='scar.compiled-program-validation-design',schema_version=1,pipeline=pipeline_ref,
        entry_context=context,runner_identity=runner,policy=policy,specializations=specializations,
        pair_order='ALTERNATE_ORIGINAL_FIRST_AND_SELECTED_FIRST',
        process_timing_scope='PARENT_MONOTONIC_LAUNCH_THROUGH_CHILD_EXIT_INCLUDES_IMPORTS_NATIVE_LOADING_APPLICATION_AND_REPORT_IO',
        comparison_timing_scope='OUTPUT_COMPARISONS_AND_RECEIPT_WRITING_OUTSIDE_CHILD_PROCESS_CLOCKS',
        selected_implementations_compared=1,target_variants_ranked=False)
    producer=_reference(Path(__file__),'scar.validation-raw')
    design['validator_implementation']=producer
    evidence=[_save(out/'design.json',design),producer];samples=[];failure_reasons=[]
    def process(command,folder,label):
        stdout=folder/(label+'.stdout');stderr=folder/(label+'.stderr')
        # Both children use the same inherited affinity, environment and cwd.
        # Open observers before timing; command completion includes all child I/O.
        with stdout.open('xb') as output,stderr.open('xb') as errors:
            start=time.perf_counter_ns()
            completed=subprocess.run(command,stdout=output,stderr=errors,env=environment,cwd=context['working_directory'])
            elapsed=time.perf_counter_ns()-start
        record=dict(command=command,wall_ns=elapsed,returncode=completed.returncode,
            stdout=_reference(stdout,'scar.validation-raw'),stderr=_reference(stderr,'scar.validation-raw'))
        evidence.extend((record['stdout'],record['stderr']))
        return record
    all_pairs=[('WARMUP',i) for i in range(warmup)]+[('MEASUREMENT',i) for i in range(repetitions)]
    for ordinal,(phase,index) in enumerate(all_pairs):
        folder=samples_folder/(phase.lower()+'-'+str(index));folder.mkdir()
        report_path=folder/'candidate-execution.json'
        commands=dict(baseline=[sys.executable,str(entry),*argv],optimized=[sys.executable,'-m','scar','run',
            '--pipeline',str(pipeline),'--measure-candidate','--exclusive-inputs','--report',str(report_path),'--',str(entry),*argv])
        records={}
        for label in (('baseline','optimized') if ordinal%2==0 else ('optimized','baseline')):
            records[label]=process(commands[label],folder,label)
        baseline,optimized=records['baseline'],records['optimized']
        native=fallback=0;native_only=False
        if report_path.is_file():
            try:
                raw=json.loads(report_path.read_bytes());ref=_reference(report_path,raw['schema'])
                optimized['execution_report']=ref;evidence.append(ref)
                native,fallback,native_only=_native_counts(raw,pipeline_ref,entry,argv)
            except (ValueError,KeyError,TypeError):
                optimized['execution_report_invalid']=True
                evidence.append(_reference(report_path,'scar.validation-raw'))
        stdout_match=_equal_files(baseline['stdout']['uri'],optimized['stdout']['uri'])
        stderr_match=_equal_files(baseline['stderr']['uri'],optimized['stderr']['uri'])
        returncode_match=baseline['returncode']==optimized['returncode']
        successful=baseline['returncode']==optimized['returncode']==0
        stable=entry_context(entry,argv)==context and runner_identity()==runner and _reference(pipeline,pipeline_ref['schema'])==pipeline_ref
        sample=dict(schema='scar.compiled-program-validation-sample',schema_version=1,phase=phase,index=index,
            baseline=baseline,optimized=optimized,stdout_match=stdout_match,stderr_match=stderr_match,
            returncode_match=returncode_match,successful_returncodes=successful,native_calls=native,
            fallback_calls=fallback,native_only=native_only,entry_pipeline_and_runner_context_unchanged=stable,
            valid=stdout_match and stderr_match and returncode_match and successful and native_only and stable)
        sample_ref=_save(folder/'sample.json',sample);evidence.append(sample_ref)
        samples.append(dict(**sample,sample_ref=sample_ref))
        if not sample['valid']:
            failure_reasons.append('OUTPUT_EXIT_NATIVE_COVERAGE_OR_CONTEXT_VALIDATION_FAILED');break
    measurements=[row for row in samples if row['phase']=='MEASUREMENT']
    baseline_stats=_statistics([row['baseline']['wall_ns'] for row in measurements])
    optimized_stats=_statistics([row['optimized']['wall_ns'] for row in measurements])
    original_median=baseline_stats['median_ns'];native_median=optimized_stats['median_ns']
    gain=(original_median-native_median)/original_median if original_median and native_median is not None else None
    speedup=original_median/native_median if original_median is not None and native_median else None
    complete=len(measurements)==repetitions and len(samples)==warmup+repetitions
    validated=complete and all(row['valid'] for row in samples)
    profitable=validated and strict_entry_gain_met(baseline_stats['samples_ns'], optimized_stats['samples_ns'], minimum_gain)
    if not complete:failure_reasons.append('REQUESTED_MEASUREMENTS_INCOMPLETE')
    if validated and not profitable:failure_reasons.append('NOT_PROFITABLE_AT_FROZEN_MINIMUM_GAIN')
    receipt=dict(schema='scar.compiled-program-deployment',schema_version=1,
        decision='LOAD_MODEL_SELECTED_NATIVE' if profitable else 'NO_OP',
        pipeline=pipeline_ref,entry_context=context,runner_identity=runner,policy=policy,
        specializations=specializations,samples=samples,evidence=evidence,
        performance=dict(baseline=baseline_stats,optimized=optimized_stats,speedup=speedup,gain_fraction=gain,
            clock='PARENT_MONOTONIC_FULL_PROCESS_LIFETIME',warmup_pairs_excluded_from_statistics=warmup),
        validation=dict(exact_stdout_stderr_and_returncode=all(row['stdout_match'] and row['stderr_match'] and row['returncode_match'] for row in samples),
            all_candidate_native_only=all(row['native_only'] for row in samples),
            native_calls_total=sum(row['native_calls'] for row in samples),fallback_calls_total=sum(row['fallback_calls'] for row in samples),
            completed_measurement_pairs=len(measurements),complete=complete,validated=validated,
            scope='EXACT_STDOUT_STDERR_AND_SUCCESSFUL_EXIT_PLUS_ORIGINAL_REGIONAL_F4_AND_RUNTIME_GUARD_BASIS',
            arbitrary_file_io_hidden_state_or_episode_equivalence_proven=False),
        reasons=failure_reasons,selected_implementations_compared=1,target_variants_ranked=False,
        full_program_cpp=False,physical_optimality_proven=False,profitability_is_future_runtime_guarantee=False)
    _save(out/'deployment.json',receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--warmup',type=int,default=1);parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--minimum-gain',type=float,default=DEFAULT_MINIMUM_GAIN)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args(argv)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('original Python script entry required after --')
    result=validate_build(args.pipeline,command[0],command[1:],args.out,warmup=args.warmup,
        repetitions=args.repetitions,minimum_gain=args.minimum_gain)
    print(json.dumps(dict(receipt=str(Path(args.out).resolve()/'deployment.json'),decision=result['decision'],
        performance=result['performance'],validation=result['validation'],reasons=result['reasons']),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
