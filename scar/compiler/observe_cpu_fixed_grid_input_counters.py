"""Measure process fault and block-input counters at the verified native input-route cuts."""

import os
from pathlib import Path
import shutil
from statistics import median

from scar.compiler.calibrate_array_entry import _write
from scar.compiler.observe_cpu_fixed_grid_native_lifecycle import (
    USAGE_FIELDS, WINDOWS, _capture, _need, _project, _raw, _ref, _sample, _validate_inputs,
)
from scar.ir.execution_recipe import read_artifact
from scar.optimization.cpu_fixed_grid_mapped_input_plan import verify_cpu_fixed_grid_mapped_input_plan


SCHEMA = 'scar.cpu-fixed-grid-input-counters'
ROUTES = ('OWNED_COPY', 'PROVIDER_MAP')
CUTS = ('input_ingestion', 'recurrence')


def _summarize(samples):
    result = {}
    for window in CUTS:
        start, end = WINDOWS[window]
        quantities = {'owner_wall_ns': [row['markers_ns'][end] - row['markers_ns'][start] for row in samples]}
        for field in USAGE_FIELDS:
            quantities[field] = [row['process_usage'][end][field] - row['process_usage'][start][field]
                                 for row in samples]
        result[window] = {name: dict(min=min(values), median=median(values), max=max(values), values=values)
                          for name, values in quantities.items()}
    return result


def _record(plan_ref, root, pairs, compiler, *, capture=False):
    plan = verify_cpu_fixed_grid_mapped_input_plan(plan_ref)
    _need(plan['observation'] is not None and type(pairs) is int and 1 <= pairs <= 12,
          'bounded independent diagnostic for a validated mapped input route required')
    compared = read_artifact(plan['observation'])
    control = read_artifact(compared['control_delivery'])
    input_files, argv = compared['file_inputs'], compared['argv']
    expected_mode = compared['expected_mode']
    _raw(compiler)
    _validate_inputs(input_files)
    if capture:
        _need(set(compared['cpu_ids']) <= os.sched_getaffinity(0), 'exact diagnostic CPU cohort unavailable')
        root.mkdir(parents=True, exist_ok=False)

    owner = dict(working_directory=compared['working_directory'], python_provider_environment=compared['environment'])
    mapped = dict(control, project=compared['mapped_project'], native_executable=compared['mapped_executable'],
                  native_launcher=compared['launchers']['PROVIDER_MAP'])
    routes = {}
    for name, source, proof in zip(ROUTES, (control, mapped), (None, plan['proof']), strict=True):
        sibling = _project(root/name, source, compiler, build=capture, usage=True, mapped_proof=proof)
        routes[name] = dict(original_project=source['project'], original_executable=source['native_executable'],
                            sibling=sibling, samples=[])

    publication = Path(argv[-1])
    ordered = []
    for pair in range(pairs):
        for name in ROUTES if pair % 2 == 0 else reversed(ROUTES):
            row = routes[name]
            folder = root/f'{pair:02}-{name}'
            command = [row['sibling']['diagnostic_launcher']['uri'], *argv]
            if capture:
                _capture(folder, command, owner, publication)
            sample = _sample(folder, command, control, pair, expected_mode, usage=True)
            row['samples'].append(sample)
            ordered.append(sample)
    _need(all(first['process_reaped_ns'] <= second['launcher_start_ns'] for first, second in zip(ordered, ordered[1:])),
          'route diagnostic process windows overlap')
    _validate_inputs(input_files)
    for row in routes.values():
        row['windows'] = _summarize(row['samples'])

    return dict(schema=SCHEMA, schema_version=1, mapped_plan=plan_ref, observation=plan['observation'],
        control_delivery=compared['control_delivery'], source=compared['source'], target=compared['target'],
        argv=argv, cpu_ids=compared['cpu_ids'], working_directory=compared['working_directory'],
        environment_sha256=compared['environment_sha256'], compiler=compiler, sample_pairs=pairs, routes=routes,
        scope='PROCESS_WIDE_RUSAGE_SELF_AT_OWNER_SOURCE_PHASE_BOUNDARIES',
        normal_completion_stdout_stderr_publication_and_mode_verified=True,
        clean_benchmark=False, physical_first_touch_attributed=False, physical_io_bytes_measured=False,
        page_cache_condition_controlled=False, forward_duration_bound_proven=False, deployment_authorized=False)


def measure(plan_ref, out, *, pairs=3, compiler='g++'):
    out = Path(out).absolute()
    executable = shutil.which(compiler)
    _need(executable is not None and out.resolve() == out and not out.exists() and not out.is_symlink(),
          'available compiler and fresh canonical output directory required')
    compiler_ref = _ref(Path(executable).resolve(strict=True), 'application/x-executable')
    report = _record(plan_ref, out, pairs, compiler_ref, capture=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_input_counters(reference):
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path == path.resolve(strict=True),
          'exact route diagnostic report required')
    expected = _record(saved['mapped_plan'], path.parent, saved['sample_pairs'], saved['compiler'])
    _need(expected == saved, 'retained paired source counter observations differ')
    return saved
