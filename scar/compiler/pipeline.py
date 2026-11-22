"""Compile a locked native build target through SCAR's shared ProgramGraph."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PLANNING_SCHEMA
from .frontend.graph import build_graph
from .middle.plan import plan_entry, simplify_copies


_LANGUAGES = {'c', 'c++', 'fortran'}
_TIMED_FIELD = re.compile(rb'(time:) +[0-9]+\.[0-9]+')


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')


def _receipt(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=sha256(data).hexdigest(), size_bytes=len(data))


def _manifest(path):
    manifest = json.loads(path.read_bytes())
    if (manifest.get('schema'), manifest.get('schema_version')) != ('scar.native-entry', 1):
        raise ValueError('expected scar.native-entry schema v1')
    if manifest.get('language') not in _LANGUAGES | {'mixed'}:
        raise ValueError('unsupported native target language')
    root = Path(manifest['source_root']).resolve(strict=True)
    manifest['source_root'] = str(root)
    locks = manifest['source_lock']['files']
    for name, expected in locks.items():
        path = (root / name).resolve(strict=True)
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f'source path outside of root: {name}')
        current = _receipt(path)
        if any(current[key] != expected[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source differs: {name}')
    units = manifest['units']
    if not units or len({unit['path'] for unit in units}) != len(units):
        raise ValueError('target requires uniquely named translation units')
    for unit in units:
        if unit.get('language', manifest['language']) not in _LANGUAGES:
            raise ValueError(f'translation unit has no language profile: {unit["path"]}')
        if unit['path'] not in locks:
            raise ValueError(f'translation unit missing source lock: {unit["path"]}')
    return manifest


def _run(command, cwd, stdin, cpus, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('native case exceeded its time budget')
    argv = ['taskset', '-c', ','.join(map(str, cpus)), *command]
    start = time.perf_counter_ns()
    result = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, timeout=remaining)
    return result, time.perf_counter_ns() - start


def _compare(case, original, generated):
    if original.returncode or generated.returncode or original.stderr != generated.stderr:
        return False
    if original.stdout == generated.stdout:
        return True
    if case.get('comparison') == 'timed-fields':
        return (_TIMED_FIELD.sub(rb'\1 <timing>', original.stdout)
                == _TIMED_FIELD.sub(rb'\1 <timing>', generated.stdout))
    if case.get('comparison') == 'scientific-text':
        # The contract compares exact formatted decimal values, accepting the
        # exponent-letter/padding representational differences of format runtimes.
        from decimal import Decimal
        notation = rb'[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)[DEde][+-]?[0-9]+'
        token = re.compile(notation)
        def normalize(raw):
            labels = token.sub(b'<decimal>', raw).split()
            values = [Decimal(value.upper().replace(b'D', b'E').decode()) for value in token.findall(raw)]
            return labels, values
        return normalize(original.stdout) == normalize(generated.stdout)
    return False


def _paired(manifest, executable, out, cpus, repetitions, case_seconds):
    baseline = manifest['baseline']
    rows = []
    for case in manifest['cases']:
        stdin_path = case.get('stdin')
        stdin = b'' if stdin_path is None else Path(stdin_path).read_bytes()
        case_dir = out / case['id']
        case_dir.mkdir()
        deadline = time.monotonic() + case_seconds
        arguments = case.get('argv', [])
        arms = {'original': (baseline['argv'] + arguments, baseline['cwd']),
                'generated': ([str(executable), *arguments], str(case_dir))}
        matched, records = True, []
        for index in range(repetitions + 1):
            pair, record = {}, {}
            order = tuple(arms) if index % 2 == 0 else tuple(reversed(arms))
            for arm in order:
                command, working_dir = arms[arm]
                result, elapsed = _run(command, working_dir, stdin, cpus, deadline)
                pair[arm] = result
                (case_dir / f'{index}-{arm}.stdout').write_bytes(result.stdout)
                (case_dir / f'{index}-{arm}.stderr').write_bytes(result.stderr)
                record[arm] = dict(ns=elapsed, exit=result.returncode)
            records.append(dict(warmup=index == 0, order=order, arms=record,
                                matches=_compare(case, pair['original'], pair['generated'])))
            matched &= records[-1]['matches']
            if not matched:
                break
        samples = {arm: [record['arms'][arm]['ns'] for record in records if not record['warmup']]
                   for arm in arms}
        timing = {arm: dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values))
                  for arm, values in samples.items() if values}
        rows.append(dict(case=case['id'], matches=matched, stdin_sha256=sha256(stdin).hexdigest(),
                         comparison=case.get('comparison', 'exact'), pairs=records, timing=timing))
    return rows


def compile_native(manifest_path, out, *, jobs, repetitions, case_seconds, graph_only=False):
    manifest = _manifest(manifest_path)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    _write(out / 'request.json', manifest)
    frontend = build_graph(manifest, out, jobs=jobs)
    graph = frontend.report
    artifacts = {'graph_evidence': asdict(frontend.evidence), 'hardware': asdict(frontend.hardware)}
    if frontend.program is not None:
        artifacts['program'] = asdict(frontend.program)
    report = dict(schema='scar.native-entry-result', schema_version=1, graph={
        key: value for key, value in graph.items() if key not in ('anchors', 'frontiers', 'frontend_frontiers')},
        frontend_frontiers=len(graph['frontend_frontiers']), cfg_frontiers=len(graph['frontiers']),
        status='GRAPH_PARTIAL', selection='KEEP_ORIGINAL', artifacts=artifacts)
    if frontend.program is not None and not graph_only:
        rewrites = simplify_copies(frontend.program, graph['anchors'], out / 'simplified.json')
        _write(out / 'rewrites.json', rewrites)
        report['graph_rewrites'] = len(rewrites)
        core = min(os.sched_getaffinity(0))
        report['cpus'] = [core]
        if manifest.get('baseline') and manifest.get('cases'):
            case = manifest['cases'][0]
            baseline = manifest['baseline']
            fixture = b'' if not case.get('stdin') else Path(case['stdin']).read_bytes()
            result, cost = _run(baseline['argv'] + case.get('argv', []), baseline['cwd'], fixture,
                                [core], time.monotonic() + case_seconds)
            if result.returncode == 0:
                problem, schedule = plan_entry(
                    graph['entry'], frontend.program, frontend.hardware, core, cost)
                problem_path = out / 'planning-problem.json'
                schedule_path = out / 'plan.json'
                _write(problem_path, problem)
                _write(schedule_path, schedule)
                artifacts['planning_problem'] = artifact_reference(problem_path, PLANNING_SCHEMA)
                artifacts['execution_plan'] = artifact_reference(schedule_path, PLAN_SCHEMA)
                report['planning_scope'] = 'BASELINE_ONLY'
    _write(out / 'result.json', report)
    if graph_only or manifest['language'] == 'mixed':
        return report

    try:
        from .backend.cpp20 import build_reference_project

        executable, dynamic = build_reference_project(manifest, out / 'project')
        report['executable'] = _receipt(executable)
        report['elf'] = dynamic
        comparison = out / 'validation'
        comparison.mkdir()
        rows = _paired(manifest, executable, comparison, report.get('cpus', [min(os.sched_getaffinity(0))]),
                       repetitions, case_seconds)
        report['validation'] = rows
        report['delivery'] = 'COMPARED' if all(case['matches'] for case in rows) else 'MISMATCH'
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        report['delivery'] = 'INCOMPLETE'
        report['delivery_error'] = str(error)
    _write(out / 'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--jobs', type=int, choices=range(1, 13), default=4)
    parser.add_argument('--repetitions', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--case-seconds', type=int, default=3600)
    parser.add_argument('--graph-only', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.case_seconds <= 3600:
        parser.error('--case-seconds must be between 1 and 3600')
    try:
        report = compile_native(args.manifest, args.out, jobs=args.jobs, repetitions=args.repetitions,
                                case_seconds=args.case_seconds, graph_only=args.graph_only)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(report, indent=2))
    return 1 if report.get('delivery') in ('INCOMPLETE', 'MISMATCH') else 0


if __name__ == '__main__':
    raise SystemExit(main())
