"""Compare one current stateful recipe with its original CPU driver."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

from scar.compiler.stateful_holdout import TIMER, _distribution, _environment, _episode, _owner, _stats
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN, compare_entry_pairs
from scar.optimization.stateful_recipe import verify_stateful_recipe

SCHEMA = 'scar.current-stateful-entry-benchmark'
MODES = ('ORIGINAL', 'GUARDED_NATIVE_TRIAL')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def receipt(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def planned(out, phase, pair, label, argv, token):
    root = out / phase / str(pair) / label
    files = str(root / 'program-output')
    return dict(directory=str(root), program_output_directory=files,
                argv=[files if value == token else value for value in argv])


def content_receipt(value):
    return dict(bytes=len(value), sha256=sha256(value).hexdigest())


def clean_entry(planned_run, entry, owner, name, descriptor, guard, native, design):
    from scar.runtime.entry_runner import run_original_script

    Path(planned_run['directory']).mkdir(parents=True)
    files = Path(planned_run['program_output_directory'])
    stdout, stderr = io.StringIO(), io.StringIO()
    before = _stats(guard)
    installed = staticmethod(guard) if type(descriptor) is staticmethod else guard
    if native:
        setattr(owner, name, installed)
    try:
        _environment(design)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            start = time.perf_counter_ns()
            execution = run_original_script(entry, planned_run['argv'])
            elapsed = time.perf_counter_ns() - start
        _environment(design)
    finally:
        if native:
            require(vars(owner)[name] is installed, 'entry changed the installed source boundary')
            setattr(owner, name, descriptor)
    after = _stats(guard)
    return dict(
        elapsed=elapsed, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
        files={str(path.relative_to(files)): path.read_bytes()
               for path in sorted(files.rglob('*')) if path.is_file()},
        native_calls=after['native_calls'] - before['native_calls'],
        fallback_calls=after['fallback_calls'] - before['fallback_calls'],
        returncode=execution['exit_code'],
    )


def run(args):
    from scar.codegen.stateful import SCHEMA as NATIVE, load
    from scar.compiler.stateful_runtime import install_stateful_regions
    from scar.runtime.stateful_guard import _function

    setup_start = time.perf_counter_ns()
    argv = json.loads(args.argv_json)
    require(type(argv) is list and all(type(value) is str for value in argv)
            and argv.count(args.output_token) == 1, 'one complete output argv token required')
    recipe_ref = receipt(args.recipe, 'scar.execution-recipe')
    recipe = verify_stateful_recipe(recipe_ref['uri'])
    lowering_ref = receipt(args.lowering, 'scar.execution-recipe-lowering')
    lowering = read_artifact(lowering_ref)
    directory = Path(lowering['native_directory']).resolve(strict=True)
    native_ref = receipt(directory / 'artifact.json', NATIVE)
    native = read_artifact(native_ref)
    correctness_ref = receipt(args.prior_correctness, 'scar.reissued-array-region-validation')
    correctness = read_artifact(correctness_ref)
    entry_ref = receipt(args.entry, 'scar.python-source')
    entry = entry_ref['uri']
    require(native_ref['sha256'] == lowering['native_manifest_sha256']
            and lowering['recipe_id'] == recipe['recipe_id']
            and lowering['recipe_sha256'] == recipe_ref['sha256']
            and correctness['lowering'] == lowering_ref and correctness['fallback_calls'] == 0,
            'current recipe, native lowering and exact trajectory validation must correspond')
    pins = {'entry': entry_ref, 'recipe': recipe_ref, 'lowering': lowering_ref,
            'native': native_ref, 'prior_trajectory_validation': correctness_ref}
    for source, digest in correctness['source_hashes'].items():
        source_ref = receipt(source, 'scar.python-source')
        require(source_ref['sha256'] == digest, 'original source differs from exact trajectory validation')
        pins['source:' + source] = source_ref
    require('source:' + entry in pins, 'entry is absent from exact trajectory validation')
    pins['harness'] = receipt(__file__, 'scar.python-source')
    pins['timer_harness'] = receipt(inspect.getsourcefile(_episode), 'scar.python-source')
    pins['selection'] = receipt(inspect.getsourcefile(compare_entry_pairs), 'scar.python-source')
    for key, name in native['files'].items():
        item = receipt(directory / name, 'scar.native-member')
        require(item['sha256'] == native['sha256'][key], 'native member does not match its manifest')
        if key == 'library':
            require(item['sha256'] == lowering['native_library_sha256'], 'native library differs from lowering')
        pins['native_member:' + key] = item

    affinity = sorted(os.sched_getaffinity(0))
    require(affinity == native['cpu_ids'], 'invoke with the native recipe CPU set for both arms')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(out / 'mpl')
    design = dict(schema=SCHEMA + '.design', schema_version=1,
        request=dict(caller_cpu_ids=affinity, capture_epoch=recipe['recipe_id']), material=pins,
        argv=argv, output_token=args.output_token, paired_order='ALTERNATING_AB_BA',
        warmup_pairs=2, measurement_pairs=7, minimum_gain=DEFAULT_MINIMUM_GAIN, method_timer=TIMER,
        entry_timer='ORIGINAL_SCRIPT_EXECUTION_FROM_PRELOADED_SOURCE_BOUNDARY',
        python=dict(executable=sys.executable, version=sys.version),
        environment={key: os.environ.get(key) for key in
                     ('MPLBACKEND', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MPLCONFIGDIR')})
    write(out / 'design.json', design)
    design_ref = receipt(out / 'design.json', design['schema'])

    _environment(design)
    artifact = load(directory)
    region = artifact.region
    sys.path.insert(0, str(Path(region.source.path).parent))
    module = importlib.import_module(Path(region.source.path).stem)
    require(Path(module.__file__).resolve() == Path(region.source.path), 'ordinary import selected another source')
    owner, name = _owner(module, region)
    descriptor = vars(owner)[name]
    original = descriptor.__func__ if type(descriptor) is staticmethod else descriptor
    _function(region.source, original)
    signature = inspect.signature(original)
    with install_stateful_regions(module, [artifact], trial=True, exclusive_inputs=True) as guards:
        guard, = guards
    setup_ns = time.perf_counter_ns() - setup_start

    raw, selection, counts = {}, {}, []
    for cohort in ('method', 'entry'):
        rows, comparisons = [], []
        base = out / cohort
        for pair in range(9):
            order = MODES if pair % 2 == 0 else MODES[::-1]
            expected = {}
            for mode in order:
                path = planned(base, 'warmup' if pair < 2 else 'measurement', pair, mode, argv, args.output_token)
                Path(path['directory']).parent.mkdir(parents=True, exist_ok=True)
                if cohort == 'entry':
                    run = clean_entry(path, entry, owner, name, descriptor, guard, mode == MODES[1], design)
                else:
                    selected = guard if mode == MODES[1] else original
                    run = _episode(entry, module, region, signature, selected, guard, design, path)
                    steps = run['steps']
                    run.update(elapsed=sum(step['elapsed_ns'] for step in steps),
                               native_calls=sum(step['native_calls'] for step in steps),
                               fallback_calls=sum(step['fallback_calls'] for step in steps),
                               returncode=run['execution']['exit_code'])
                expected[mode] = run
                rows.append(dict(pair=pair, mode=mode, wall_ns=run['elapsed'], returncode=run['returncode'],
                    stdout_sha256=content_receipt(run['stdout'].encode())['sha256'],
                    stderr_sha256=content_receipt(run['stderr'].encode())['sha256']))
            a, b = (expected[mode] for mode in MODES)
            require(all(a[key] == b[key] for key in ('stdout', 'stderr', 'files')),
                    'paired output bytes differ')
            if cohort == 'method':
                require(a['final'] == b['final'] and [s['inputs'] for s in a['steps']] == [s['inputs'] for s in b['steps']],
                        'paired numeric state or input context differs')
                counts.append(len(a['steps']))
                require(counts[-1] == counts[0] > 0, 'unexpected boundary multiplicity change')
            require(a['native_calls'] == a['fallback_calls'] == b['fallback_calls'] == 0
                    and b['native_calls'] == counts[0], 'guarded trial fell back or changed multiplicity')
            comparisons.append(dict(pair=pair, order=list(order), native_calls=b['native_calls'],
                fallback_calls=b['fallback_calls'], files={name: content_receipt(data) for name, data in a['files'].items()},
                final_numeric_state_equal=True if cohort == 'method' else None,
                method_calls_ns={mode: [s['elapsed_ns'] for s in expected[mode]['steps']] for mode in MODES}
                    if cohort == 'method' else None))
        selection[cohort] = compare_entry_pairs(rows, incumbent=MODES[0], challenger=MODES[1],
            warmup_pairs=2, measurement_pairs=7, minimum_gain=design['minimum_gain'])
        raw[cohort] = dict(rows=rows, correctness=comparisons, measured_variance_ns2={
            mode: statistics.variance(selection[cohort]['metrics'][mode]['samples_ns']) for mode in MODES})
    require(vars(owner)[name] is descriptor, 'source boundary was not restored')
    for reference in (*pins.values(), design_ref):
        read_artifact(reference, json_document=False)
    write(out / 'result.json', dict(schema=SCHEMA, schema_version=1, design=design_ref,
        recipe_id=recipe['recipe_id'], calls_per_episode=counts[0], result=selection, raw=raw,
        setup=dict(offline_admission_native_load_and_source_preload_ns=setup_ns, included_in_timers=False),
        method_distributions={mode: _distribution([value for row in raw['method']['correctness'][2:]
            for value in row['method_calls_ns'][mode]]) for mode in MODES},
        independent_correctness_reused_for_per_step_bytes=True, clean_entry_output_files_equal=True,
        sys_profile_and_sys_monitoring_enabled=False, native_call_local_resource_setup_in_method_span=True,
        separate_native_compute_only_cost_established=False,
        fresh_process_startup_cost_established=False, model_calibration_or_deployment_authorized=False))
    print(json.dumps({scope: dict(verdict=row['verdict'], gain=row['gain_fraction'], metrics=row['metrics'])
                      for scope, row in selection.items()}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recipe', 'lowering', 'prior-correctness', 'entry', 'argv-json', 'output-token', 'out'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
