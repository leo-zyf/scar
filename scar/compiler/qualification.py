"""Run locked cases on one CPU and retain observations for plan replay."""

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
import statistics
import subprocess
import time

from scar.artifacts import file_reference
from scar.ir.execution_recipe import read_artifact
from scar.runtime.cpu_paired_runner import compare_native_outputs


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
