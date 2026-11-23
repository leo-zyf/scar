"""Realize a sealed SCAR plan as a C++20 project and validate its entry."""

from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import os
from pathlib import Path
import re
import statistics
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter_ns

from ..contract import Member, artifact, open_plan, read_artifact, verify_artifact, write_json
from .project import build_project


_TIMING = re.compile(rb'(time:) +[0-9]+\.[0-9]+')
_DECIMAL = re.compile(rb'[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)[DEde][+-]?[0-9]+')


def _verify_exports(executable, exports):
    if not exports:
        return
    symbols = subprocess.check_output(['nm', '-P', '--defined-only', str(executable)], text=True, timeout=30)
    missing = set(exports) - {line.split()[0] for line in symbols.splitlines() if line.split()}
    if missing:
        raise ValueError(f'C++20 executable did not preserve link symbols: {sorted(missing)}')


def _scientific_output(data):
    values = [Decimal(value.upper().replace(b'D', b'E').decode()) for value in _DECIMAL.findall(data)]
    return _DECIMAL.sub(b'<decimal>', data).split(), values


def _matches(comparison, original, generated):
    if original.returncode or generated.returncode or original.stderr != generated.stderr:
        return False
    if comparison == 'timed-fields':
        return _TIMING.sub(rb'\1 <timing>', original.stdout) == _TIMING.sub(rb'\1 <timing>', generated.stdout)
    if comparison == 'scientific-text':
        return _scientific_output(original.stdout) == _scientific_output(generated.stdout)
    return original.stdout == generated.stdout


def _case_runs(validation, bundle, executable, out, cpus, seconds, repetitions):
    cases = validation.get('cases', ())
    if not cases:
        return []
    baseline = validation['baseline']
    original_ref = Member(**baseline['executable']).resolve(bundle)
    original = [original_ref.path, *baseline['argv']]
    verify_artifact(original_ref)
    if any(type(cpu) is not int or cpu not in os.sched_getaffinity(0) for cpu in cpus):
        raise ValueError('selected CPU is unavailable for native validation')
    affinity = ['taskset', '-c', ','.join(map(str, cpus))] if cpus else []
    validated = []
    for case in cases:
        name = case['id']
        comparison = case.get('comparison', 'exact')
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name):
            raise ValueError(f'invalid native case name: {name}')
        if comparison not in ('exact', 'timed-fields', 'scientific-text'):
            raise ValueError(f'unsupported native case comparison: {comparison}')
        folder = out / 'validation' / name
        folder.mkdir(parents=True)
        stdin = read_artifact(Member(**case['input']).resolve(bundle)) if 'input' in case else b''
        arguments = case.get('argv', ())
        arms = {'original': (original, baseline['cwd']), 'generated': ([str(executable), *original[1:]], folder)}
        env = dict(os.environ, LC_ALL='C')
        deadline = monotonic() + seconds
        pairs = []
        for index in range(repetitions + 1):
            outputs, measurements = {}, {}
            order = tuple(arms) if index % 2 == 0 else tuple(reversed(arms))
            for label in order:
                command, cwd = arms[label]
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'native case exceeded its time budget: {name}')
                start = perf_counter_ns()
                run = subprocess.run([*affinity, *command, *arguments], cwd=cwd, env=env, input=stdin,
                                     capture_output=True, timeout=remaining)
                measurements[label] = dict(ns=perf_counter_ns() - start, exit=run.returncode)
                outputs[label] = run
                (folder / f'{index}-{label}.stdout').write_bytes(run.stdout)
                (folder / f'{index}-{label}.stderr').write_bytes(run.stderr)
            equal = _matches(comparison, outputs['original'], outputs['generated'])
            pairs.append(dict(warmup=index == 0, order=order, arms=measurements, matches=equal))
            if not equal:
                break
        timing = {}
        if all(pair['matches'] for pair in pairs):
            for label in arms:
                samples = [pair['arms'][label]['ns'] for pair in pairs if not pair['warmup']]
                timing[label] = dict(median_ns=statistics.median(samples), min_ns=min(samples), max_ns=max(samples),
                                     stddev_ns=statistics.pstdev(samples))
        validated.append(dict(case=name, matches=all(pair['matches'] for pair in pairs), comparison=comparison,
                              stdin_sha256=sha256(stdin).hexdigest(), pairs=pairs, timing=timing))
    verify_artifact(original_ref)
    return validated


def compile_plan(path: Path, out: Path, *, case_seconds: int = 30, repetitions: int = 1) -> dict:
    """Read a SCAR plan and publish its backend result without loading another compiler stage."""
    path, out = Path(path), Path(out).resolve()
    if path.is_dir():
        path /= 'plan.json'
    if type(case_seconds) is not int or case_seconds < 1 or type(repetitions) is not int or repetitions < 1:
        raise ValueError('native delivery requires a positive case time budget and repetition count')
    plan_ref, plan = open_plan(path)
    if out.exists():
        raise FileExistsError(f'backend output already exists: {out}')
    if not plan.lowering.bodies:
        out.mkdir(parents=True)
        outcome = dict(schema='scar.cpp20-delivery', schema_version=1, plan=asdict(plan_ref), status='KEEP_ORIGINAL',
                       decision=plan.decision, validation=[])
        write_json(out / 'result.json', outcome)
        return outcome

    from .lowering import lower_project

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-cpp20-', dir=out.parent) as temporary:
        staging = Path(temporary) / 'delivery'
        staging.mkdir()
        project = staging / 'project'
        configuration, exports, invocation = lower_project(plan.lowering, project, path.parent.resolve())
        executable, compiler = build_project(project, configuration)
        _verify_exports(executable, exports)
        cpus = plan.lowering.cpu_ids
        cases = ([] if invocation else
                 _case_runs(plan.validation, path.parent.resolve(), executable, staging, cpus, case_seconds, repetitions))
        result = asdict(artifact(executable))
        result['path'] = str(out / executable.relative_to(staging))
        status = 'TRIAL_COMPILED' if invocation else 'BUILT'
        if cases:
            status = 'VALIDATED' if all(case['matches'] for case in cases) else 'VALIDATION_FAILED'
        outcome = dict(schema='scar.cpp20-delivery', schema_version=1, plan=asdict(plan_ref), status=status,
                       project=str(out / 'project'), executable=result, cpus=list(cpus), compiler=compiler,
                       exports=exports, decision=plan.decision, validation=cases)
        if invocation:
            outcome.update(invocation=invocation, deployment_authorized=False)
        write_json(staging / 'result.json', outcome)
        verify_artifact(plan_ref)
        staging.rename(out)
    return outcome
