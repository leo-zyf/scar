"""Compare matched fresh-process Python and NVHPC entries after ABI replay."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

from scar.codegen.native_provider_bundle import _content
from scar.compiler.compile_exported_aten import _calibration_paths, _write_record
from scar.compiler.compile_exported_nvhpc import _compare, check_nvhpc_project
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.nvhpc-source-process-observation'
EVIDENCE = 'scar.nvhpc-source-entry-evidence'
_TIMING_FIELDS = ('pair', 'mode', 'wall_ns', 'returncode', 'stdout_sha256', 'stderr_sha256')
_PROBE = '''
import json
import numpy
import torch
print(json.dumps(dict(torch=torch.__file__, torch_native=torch._C.__file__,
                     numpy=numpy.__file__, torch_version=torch.__version__,
                     numpy_version=numpy.__version__)))
'''


def _receipt(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), **_content(path))


def _context(cpus):
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('original and native process pair requires its admitted CPU affinity')
    python = str(Path(sys.executable).resolve())
    environment = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS=str(len(cpus)), MKL_NUM_THREADS=str(len(cpus)),
                       OPENBLAS_NUM_THREADS=str(len(cpus)), OMP_DYNAMIC='FALSE')
    if 'PYTHONPATH' in os.environ:
        environment['PYTHONPATH'] = os.environ['PYTHONPATH']
    prefix = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
    provider = json.loads(subprocess.check_output([*prefix, python, '-B', '-c', _PROBE],
                                                  env=environment, text=True, timeout=60))
    for key in ('torch', 'torch_native', 'numpy'):
        provider[key] = _receipt(provider[key])
    return dict(cpu_ids=cpus, prefix=prefix, python=_receipt(python), environment=environment,
                providers=provider, working_directory=str(Path.cwd().resolve()))


def _arm(command, context, path, publication, pair, mode):
    path.mkdir()
    start = perf_counter_ns()
    process = subprocess.run(command, capture_output=True, env=context['environment'], timeout=120)
    wall_ns = perf_counter_ns() - start
    stdout, stderr = path/'stdout', path/'stderr'
    stdout.write_bytes(process.stdout)
    stderr.write_bytes(process.stderr)
    outputs = []
    for index, file in enumerate(publication):
        if not file.is_file() or file.is_symlink():
            outputs.append(None)
            continue
        retained = path/f'output-{index}.f32'
        file.rename(retained)
        outputs.append(_receipt(retained))
    return dict(pair=pair, mode=mode, wall_ns=wall_ns, returncode=process.returncode,
        stdout_sha256=sha256(process.stdout).hexdigest(), stderr_sha256=sha256(process.stderr).hexdigest(),
        stdout=_receipt(stdout), stderr=_receipt(stderr), outputs=outputs)


def _assess(arms, contract, sizes, torch):
    original, native = (arms[key] for key in ('ORIGINAL', 'NATIVE'))
    same_streams = all(original[key]['sha256'] == native[key]['sha256']
                       for key in ('stdout', 'stderr'))
    checks = []
    for left, right, size in zip(original['outputs'], native['outputs'], sizes, strict=True):
        if left is None or right is None or left['size_bytes'] != size or right['size_bytes'] != size:
            checks.append(dict(accepted=False, exact_bytes=False, max_abs_error=None))
        else:
            data = (Path(row['path']).read_bytes() for row in (left, right))
            checks.append(_compare(*data, contract, torch))
    accepted = same_streams and original['returncode'] == native['returncode'] == 0 and all(
        row['accepted'] for row in checks)
    return dict(accepted=accepted, same_stdout_stderr=same_streams, outputs=checks)


def _timing(runs, warmup, count, *, minimum_gain=0.05):
    return compare_entry_pairs([{key: run[key] for key in _TIMING_FIELDS} for run in runs],
        incumbent='ORIGINAL', challenger='NATIVE', warmup_pairs=warmup, measurement_pairs=count,
        minimum_gain=minimum_gain)


def validate_source_process(project, manifest_sha256, arguments, out, *, warmup_pairs=1, measurement_pairs=3,
                            entry_scope=None):
    """Retain comparable process observations, separate from forward costs and release."""
    import torch

    if (type(warmup_pairs) is not int or not 0 <= warmup_pairs <= 10
            or type(measurement_pairs) is not int or not 3 <= measurement_pairs <= 30):
        raise ValueError('a bounded warmup and at least three timed process pairs are required')
    root, project_ref, execution, _, projection = check_nvhpc_project(project, manifest_sha256)
    manifest = read_artifact(project_ref)
    if 'plan/source-entry.json' not in manifest['files']:
        raise ValueError('a replayed original Python source entry proof is required')
    contract = json.loads((root/'plan/source-entry.json').read_bytes())
    source = Path(contract['source']['path'])
    original = _receipt(source)
    native_contract = execution['numerical_contract']
    paths = _calibration_paths(arguments, execution['inputs'])
    inputs = [_receipt(path) for path in paths]
    if [row['size_bytes'] for row in inputs] != [row['nbytes'] for row in contract['inputs']]:
        raise ValueError('source process input differs from its exact FP32 file shape')
    sizes = [row['nbytes'] for row in contract['outputs']]
    if len(sizes) != len(projection['outputs']):
        raise ValueError('source publication count differs from the native ABI')
    original_cpus = execution['cpu_ids'] if entry_scope is None else entry_scope['process_affinity_cpu_ids']
    if not set(execution['cpu_ids']).issubset(original_cpus):
        raise ValueError('native CPU cohort exceeds the original entry affinity')
    context = _context(original_cpus)
    out = Path(out).resolve()
    if entry_scope is not None:
        from scar.compiler.nvhpc_entry_variant import matrix_entry_inputs
        from scar.compiler.entry_delivery import _environment_digest

        argv = entry_scope['argv']
        facts = matrix_entry_inputs(contract, argv)
        if ([row['uri'] for row in facts] != list(map(str, paths))
                or _environment_digest(context['environment']) != entry_scope['environment_sha256']
                or context['working_directory'] != entry_scope['working_directory']
                or context['python']['path'] != entry_scope['python_executable']
                or context['cpu_ids'] != entry_scope['process_affinity_cpu_ids']):
            raise ValueError('source process inputs or provider environment differ from the admitted trial scope')
        publish = list(map(Path, argv[len(paths):]))
    else:
        publish = [out/f'output-{index}.f32' for index in range(len(sizes))]
        argv = [*map(str, paths), *map(str, publish)]
    out.mkdir(parents=True, exist_ok=False)
    if len(argv) != contract['argv_count']:
        raise ValueError('source process arguments differ from the structural ABI')
    commands = dict(ORIGINAL=[*context['prefix'], context['python']['path'], '-B', str(source), *argv],
                    NATIVE=[str(root/'launch.sh'), *argv])
    runs, comparisons = [], []
    labels = tuple(commands)
    for pair in range(warmup_pairs + measurement_pairs):
        order = labels if pair % 2 == 0 else labels[::-1]
        arms = {mode: _arm(commands[mode], context, out/f'{pair}-{mode}', publish, pair, mode) for mode in order}
        runs.extend(arms.values())
        comparisons.append(_assess(arms, native_contract, sizes, torch))
        if not comparisons[-1]['accepted']:
            break
    accepted = all(row['accepted'] for row in comparisons) and len(comparisons) == warmup_pairs + measurement_pairs
    timing = _timing(runs, warmup_pairs, measurement_pairs) if accepted else None
    contract_ref = artifact_reference(root/'plan/source-entry.json', contract['schema'])
    if (_receipt(source) != original or [_receipt(path) for path in paths] != inputs
            or _context(original_cpus) != context or read_artifact(project_ref) != manifest):
        raise ValueError('source, input or provider identity changed during the paired process observations')
    check_nvhpc_project(root, manifest_sha256)
    report = dict(schema=SCHEMA, schema_version=1, artifact=project_ref, source_contract=contract_ref, source=original,
        scope='PROCESS_LAUNCH_THROUGH_EXIT', context=context, argv=argv, commands=commands, inputs=inputs,
        numerical_contract=native_contract, runs=runs, comparisons=comparisons, timing=timing,
        warmup_pairs=warmup_pairs, measurement_pairs=measurement_pairs, accepted=accepted,
        stability_regime='PAIRED_FRESH_PROCESSES_WITH_REUSED_INPUT_FILES', deployment_authorized=False)
    _write_record(out/'result.json', report)
    return report


def replay_source_process(reference, evidence, scope, *, minimum_gain):
    """Cold-check the retained ordered process outputs and original scope evidence."""
    import torch

    saved = read_artifact(reference)
    context = _context(scope['process_affinity_cpu_ids'])
    env = sha256(json.dumps(sorted(context['environment'].items()), separators=(',', ':')).encode()).hexdigest()
    argv = scope['argv']
    input_records = [dict(path=ref['uri'], sha256=ref['sha256'], size_bytes=ref['size_bytes'])
                     for ref in evidence['file_inputs']]
    contract = evidence['numerical_contract']
    source = scope['entry']
    expected = dict(ORIGINAL=[*context['prefix'], context['python']['path'], '-B', source, *argv],
                    NATIVE=[evidence['launcher']['uri'], *argv])
    if (reference['schema'] != SCHEMA or saved['schema_version'] != 1 or saved['artifact'] != evidence['artifact']
            or saved['source_contract'] != evidence['source_contract'] or saved['source'] != _receipt(source)
            or saved['numerical_contract'] != contract or saved['inputs'] != input_records
            or saved['scope'] != scope['boundary'] or saved['argv'] != argv or saved['commands'] != expected
            or saved['context'] != context or env != scope['environment_sha256']
            or context['working_directory'] != scope['working_directory']
            or context['python']['path'] != scope['python_executable']
            or saved['accepted'] is not True or saved['deployment_authorized'] is not False):
        raise ValueError('original and native process observations differ from the admitted scope')
    runs = saved['runs']
    warmup, count = saved['warmup_pairs'], saved['measurement_pairs']
    sizes = evidence['output_sizes']
    folder = Path(reference['uri']).parent
    if len(runs) != 2 * (warmup + count) or len(saved['comparisons']) * 2 != len(runs):
        raise ValueError('incomplete retained source process pairs')
    for index, run in enumerate(runs):
        pair, slot = divmod(index, 2)
        mode = ('ORIGINAL', 'NATIVE')[(pair+slot) % 2]
        arm = folder/f'{pair}-{mode}'
        if (run['pair'], run['mode'], len(run['outputs'])) != (pair, mode, len(sizes)):
            raise ValueError('source process observation order differs from its pair')
        for name in ('stdout', 'stderr'):
            if run[name] != _receipt(arm/name) or run[name]['sha256'] != run[name+'_sha256']:
                raise ValueError('source process stream receipt changed')
        for ordinal, file in enumerate(run['outputs']):
            if file != _receipt(arm/f'output-{ordinal}.f32'):
                raise ValueError('source process ordered output receipt changed')
        if slot:
            assessed = _assess({row['mode']: row for row in runs[index-1:index+1]}, contract, sizes, torch)
            if not assessed['accepted'] or assessed != saved['comparisons'][pair]:
                raise ValueError('source and native process output contracts did not replay')
    if _timing(runs, warmup, count) != saved['timing']:
        raise ValueError('retained process timing summary changed')
    read_artifact(reference, json_document=False)
    return _timing(runs, warmup, count, minimum_gain=minimum_gain), context


def replay_source_entry_evidence(reference, evidence, scope, *, minimum_gain):
    """Attach diagnostic effects and a scoped component forecast to the clean comparison."""
    if reference['schema'] == SCHEMA:
        return replay_source_process(reference, evidence, scope, minimum_gain=minimum_gain)
    document = read_artifact(reference)
    if (reference['schema'] != EVIDENCE or document['schema_version'] != 1
            or set(document) != {'schema', 'schema_version', 'process', 'effects', 'lifecycle'}):
        raise ValueError('a versioned NVIDIA source evidence bundle is required')
    process = document['process']
    timing, context = replay_source_process(process, evidence, scope, minimum_gain=minimum_gain)
    diagnostics = {}
    effects = document['effects']
    if effects is not None:
        from collections import Counter
        from scar.analysis.nvhpc_source_effects import verify_nvhpc_source_effects

        journal = verify_nvhpc_source_effects(effects, process=process)
        diagnostics['effects'] = dict(reference=effects, status=journal['status'],
            uncovered=dict(Counter(row['kind'] for row in journal['projection']['uncovered_effects'])),
            global_effect_closure=False)
        if 'effect_coverage' in journal:
            coverage = journal['effect_coverage']
            diagnostics['effects']['coverage'] = dict(
                source_contract=coverage['source_contract'],
                logical_boundary_witnessed=coverage['logical_boundary_witnessed'],
                static_boundary_effects_closed=coverage['static_boundary_effects_closed'],
                obligations_by_status=dict(Counter(row['status'] for row in coverage['obligations'])))
    lifecycle = document['lifecycle']
    if lifecycle is not None:
        from scar.optimization.nvhpc_source_lifecycle import replay_nvhpc_source_lifecycle

        model = replay_nvhpc_source_lifecycle(lifecycle)
        raw = read_artifact(process)
        output_devices = {Path(path).parent.stat().st_dev for path in scope['argv'][len(raw['inputs']):]}
        if (model['artifact'] != evidence['artifact'] or model['source_contract'] != evidence['source_contract']
                or model['context'] != context or model['inputs'] != raw['inputs']
                or model['target'] != evidence['target'] or output_devices != {model['comparison_scratch_device']}):
            raise ValueError('lifecycle probe and clean run differ in source, inputs, provider or output device')
        phases = {mode: {name: measure['median_ns'] for name, measure in rows.items()}
                  for mode, rows in model['summary'].items()}
        diagnostics['lifecycle_forecast'] = dict(reference=lifecycle,
            basis='SUM_OF_MATCHED_INSTRUMENTED_COMPONENT_MEDIANS', median_components_ns=phases,
            predicted_process_ns={mode: sum(row.values()) for mode, row in phases.items()},
            limitations=model['limitations'], physical_bounds_proven=False)
    read_artifact(reference, json_document=False)
    return timing, dict(context, diagnostics=diagnostics)
