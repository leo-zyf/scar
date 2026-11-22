"""Compile a locked native build target through SCAR's shared ProgramGraph."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from scar.artifacts import file_reference
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PLANNING_SCHEMA
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA
from .frontend.graph import build_graph
from .middle.plan import entry_cpus, plan_entry, simplify_copies


_LANGUAGES = {'c', 'c++', 'fortran'}
_TIMED_FIELD = re.compile(rb'(time:) +[0-9]+\.[0-9]+')


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')


def _receipt(path):
    reference = file_reference(path, 'binary')
    return dict(path=str(path), sha256=reference['sha256'], size_bytes=reference['size_bytes'])


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
    if baseline := manifest.get('baseline'):
        cwd = Path(baseline['cwd']).resolve(strict=True)
        argv = baseline['argv']
        if not argv:
            raise ValueError('baseline requires an executable')
        argv[0] = str((cwd / argv[0]).resolve() if '/' in argv[0]
                      else Path(shutil.which(argv[0]) or cwd / argv[0]).resolve())
        baseline['cwd'] = str(cwd)
    case_ids = [case['id'] for case in manifest.get('cases', ())]
    if (len(set(case_ids)) != len(case_ids)
            or any(not re.fullmatch(r'[\w.-]+', name) or name in ('.', '..') for name in case_ids)):
        raise ValueError('case IDs must be unique directory names')
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


def _case_input(case):
    return b'' if not case.get('stdin') else Path(case['stdin']).read_bytes()


def _paired(manifest, executable, out, cpus, observation, repetitions, case_seconds):
    baseline = manifest['baseline']
    if _receipt(baseline['argv'][0]) != observation['baseline']:
        raise ValueError('baseline executable differs from calibration')
    rows = []
    for case in manifest['cases']:
        stdin = _case_input(case)
        input_digest = sha256(stdin).hexdigest()
        if input_digest != observation['case_inputs'][case['id']]:
            raise ValueError(f'case input differs from calibration: {case["id"]}')
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
        rows.append(dict(case=case['id'], matches=matched, stdin_sha256=input_digest,
                         comparison=case.get('comparison', 'exact'), pairs=records, timing=timing))
    return rows


def _observe_entry(manifest, artifacts, case_seconds):
    baseline = manifest['baseline']
    cases = manifest['cases']
    available = entry_cpus(artifacts['hardware'])
    if not available:
        raise ValueError('no online CPU is available in the inventoried execution scope')
    cpu = min(available)
    executable = _receipt(baseline['argv'][0])
    case = cases[0]
    fixture = _case_input(case)
    inputs = {row['id']: sha256(fixture if row is case else _case_input(row)).hexdigest()
              for row in cases}
    result, elapsed = _run(baseline['argv'] + case.get('argv', []), baseline['cwd'], fixture,
                           [cpu], time.monotonic() + case_seconds)
    return dict(schema='scar.native-entry-cost', schema_version=1, request=artifacts['request'],
                scope='ONE_BASELINE_LAUNCH', cpu=cpu, elapsed_ns=elapsed, exit=result.returncode,
                baseline=executable, case=case['id'], case_inputs=inputs)


def compile_native(manifest_path, out, *, jobs, repetitions, case_seconds, graph_only=False):
    manifest = _manifest(manifest_path)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    request_path = out / 'request.json'
    _write(request_path, manifest)
    request = file_reference(request_path, 'scar.native-entry')
    frontend = build_graph(manifest, out, jobs=jobs, request=request)
    graph = frontend.report
    artifacts = dict(request=request, graph_evidence=asdict(frontend.evidence),
                     calls=asdict(frontend.calls), hardware=asdict(frontend.hardware))
    if frontend.program is not None:
        artifacts['program'] = asdict(frontend.program)
    layers = dict(source='LOCKED', compiler='CAPTURED',
                  graph='PARTIAL' if frontend.program else 'BLOCKED',
                  calls='PARTIAL' if graph['call_frontiers'] or not frontend.program else 'BOUND',
                  hardware='OBSERVED')
    report = dict(schema='scar.native-entry-result', schema_version=1, graph={
        key: graph[key] for key in ('entry', 'regions', 'operations', 'values', 'locations',
                                    'excluded_control', 'call_frontiers', 'call_sites', 'call_links',
                                    'reachable_regions')},
        frontend_frontiers=len(graph['frontend_frontiers']), cfg_frontiers=len(graph['frontiers']),
        status='GRAPH_PARTIAL', selection='KEEP_ORIGINAL', artifacts=artifacts, layers=layers)
    if graph_only:
        if frontend.program is None:
            report.update(delivery='INCOMPLETE', delivery_error='no validated entry region')
        _write(out / 'result.json', report)
        return report

    layer = 'rewrite'
    try:
        if frontend.program is None:
            raise ValueError('no validated entry region')
        simplified = out / 'simplified.json'
        rewrites = simplify_copies(frontend.program, graph['anchors'], simplified)
        artifacts['simplified'] = file_reference(simplified, PROGRAM_SCHEMA)
        rewrite_path = out / 'rewrites.json'
        _write(rewrite_path, dict(schema='scar.native-rewrites', schema_version=1, rules=rewrites,
                                  program=artifacts['program'], result=artifacts['simplified']))
        artifacts['rewrites'] = file_reference(rewrite_path, 'scar.native-rewrites')
        report['graph_rewrites'] = len(rewrites)
        layers[layer] = 'IR_VALIDATED'
        layer = 'cost'
        if not manifest.get('baseline') or not manifest.get('cases'):
            raise ValueError('baseline executable and cases are required to qualify native delivery')
        observation = _observe_entry(manifest, artifacts, case_seconds)
        cost_path = out / 'cost-observation.json'
        _write(cost_path, observation)
        artifacts['cost'] = file_reference(cost_path, observation['schema'])
        if observation['exit'] != 0:
            raise ValueError(f'baseline exited with status {observation["exit"]}')
        layers[layer] = 'OBSERVED'
        layer = 'schedule'
        problem, schedule = plan_entry(graph['entry'], artifacts, observation)
        for role, path, document, schema in (
            ('planning_problem', out / 'planning-problem.json', problem, PLANNING_SCHEMA),
            ('execution_plan', out / 'plan.json', schedule, PLAN_SCHEMA),
        ):
            _write(path, document)
            artifacts[role] = file_reference(path, schema)
        if not schedule.get('selected'):
            raise ValueError('no feasible native execution schedule')
        report['planning_scope'] = 'BASELINE_ONLY'
        layers[layer] = 'SCHEDULED'
        layer = 'admission'
        from .backend.cpp20 import admit_entry, build_reference_project

        manifest, observation, delivery = admit_entry(artifacts, graph['entry'])
        report['cpus'] = delivery['cpus']
        layers[layer] = 'REPLAYED'
        layer = 'build'
        executable, dynamic = build_reference_project(manifest, out / 'project')
        report['executable'] = _receipt(executable)
        report['elf'] = dynamic
        delivery_path = out / 'project' / 'delivery.json'
        _write(delivery_path, dict(delivery, executable=report['executable']))
        artifacts['delivery'] = file_reference(delivery_path, delivery['schema'])
        layers[layer] = 'BUILT'
        layer = 'validation'
        comparison = out / 'validation'
        comparison.mkdir()
        rows = _paired(manifest, executable, comparison, delivery['cpus'], observation,
                       repetitions, case_seconds)
        report['validation'] = rows
        report['delivery'] = 'COMPARED' if all(case['matches'] for case in rows) else 'MISMATCH'
        layers[layer] = 'MATCHED' if report['delivery'] == 'COMPARED' else 'MISMATCH'
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        layers[layer] = 'BLOCKED'
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
