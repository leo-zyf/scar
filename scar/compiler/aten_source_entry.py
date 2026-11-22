"""Bind a source-proved CLI to one replayed standalone ATen recurrence."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.codegen.native_provider_bundle import _content


ABI_FILE = 'plan/source-entry-abi.json'
BINDING_FILE = 'plan/source-entry-physical-binding.json'
LAUNCHER = 'launch-source.sh'
_ENTRY_SOURCES = {'src/main.cpp', 'src/runtime/file_io.hpp'}


def _read(path):
    return json.loads(path.read_bytes())


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def source_entry_description(manifest_sha256, *, source_file_publication=None, mapped_input_proof=None):
    from scar.codegen.aten_recurrence_source import FRESH_SOURCE_FILE

    if source_file_publication not in (None, FRESH_SOURCE_FILE):
        raise ValueError('unsupported source-entry file creation contract')
    result = dict(abi=ABI_FILE, physical_binding=BINDING_FILE, launcher=LAUNCHER,
        parent_manifest_sha256=manifest_sha256,
        scope='PROVEN_SOURCE_CLI_NORMAL_COMPLETION_WITH_FRESH_OUTPUT',
        source_cli_performance_measured=False, deployment_authorized=False)
    if source_file_publication is not None:
        result['source_file_publication'] = source_file_publication
    if mapped_input_proof is not None:
        result['mapped_input_proof'] = mapped_input_proof
    return result


def verify_mapped_input_emission(reference, parent_sha256, publication):
    from scar.analysis.cpu_fixed_grid_mapped_inputs import verify_cpu_fixed_grid_mapped_inputs
    from scar.ir.execution_recipe import read_artifact

    proof = verify_cpu_fixed_grid_mapped_inputs(reference)
    incumbent = read_artifact(proof['project'])['source_entry_cli']
    if (proof['status'] != 'PROVEN_SCOPED_MAPPED_INPUT_MATERIALIZATION'
            or proof['parent_manifest_sha256'] != parent_sha256
            or incumbent.get('source_file_publication') != publication
            or 'mapped_input_proof' in incumbent):
        raise ValueError('mapped input proof differs from the original schedule or unchanged publication policy')


def render_source_launcher(cpus):
    from scar.compiler.compile_aten_recurrence import _render_recurrence_launch

    original = _render_recurrence_launch(cpus)
    marker = ' "$@"\n'
    if not original.endswith(marker):
        raise ValueError('original native launcher argument boundary differs')
    return original.removesuffix(marker) + ' --source-entry "$@"\n'


def bind_source_entry(project, manifest_sha256, out, *, compiler='g++', preserve_source_file_mode=False,
                      mapped_input_proof=None):
    """Rebuild one verified physical schedule with a source-derived argument mode."""
    from scar.analysis.fixed_grid_entry_abi import derive_fixed_grid_file_entry
    from scar.codegen.aten_recurrence_source import FRESH_SOURCE_FILE, render_recurrence
    from scar.codegen.native_project import build_native_project
    from scar.compiler.aten_source_entry_binding import verify_source_entry_binding
    from scar.compiler.tensor_execution import scheduled_tensor_body
    from scar.runtime.validate_native_project import verify_native_project

    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh source-entry project required')
    root, receipt = verify_native_project(project, manifest_sha256)
    manifest = _read(root/'artifact.json')
    if (manifest['schema'] != 'scar.aten-recurrence-delivery' or 'source_entry_cli' in manifest
            or not manifest.get('source_entry_replayable') or 'cpu_linkage' not in receipt):
        raise ValueError('a verified CPU-only recurrence with its original archived source is required')
    source = _read(root/'plan/source-entry-binding.json')
    original_entry = _read(root/'plan/source-entry.json')
    abi = derive_fixed_grid_file_entry(source, original_entry['entry'],
        source_bytes=(root/'plan/source-entry.py').read_bytes())
    execution = _read(root/'plan/execution.json')
    recipe = _read(root/'plan/rhs-recipe.json')
    binding = verify_source_entry_binding(abi, execution, recipe, manifest=manifest)
    lowered = scheduled_tensor_body(_read(root/'plan/problem.json'), _read(root/'plan/schedule.json'), execution)
    placement = execution['placement']
    publication = FRESH_SOURCE_FILE if preserve_source_file_mode else None
    mapping = None
    if mapped_input_proof is not None:
        from scar.analysis.cpu_fixed_grid_mapped_inputs import SCHEMA
        from scar.ir.execution_recipe import artifact_reference

        mapping = artifact_reference(Path(mapped_input_proof).resolve(strict=True), SCHEMA)
        verify_mapped_input_emission(mapping, manifest_sha256, publication)
    generated = render_recurrence(lowered, rhs_mode=placement['mode'], cpus=placement['cpus'],
        lanes=placement['lanes'], operation_order=placement['operation_order'], source_entry_abi=abi,
        source_file_publication=publication, map_source_inputs=mapping is not None)

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-source-aten-', dir=out.parent) as scratch:
        stage = Path(scratch)/'project'
        stage.mkdir()
        replaced = {*_ENTRY_SOURCES, manifest['executable'], 'validation/build.log', 'validation/ldd.txt'}
        for name, expected in manifest['files'].items():
            if name not in replaced:
                path = stage/name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root/name, path)
                if _content(path) != expected:
                    raise ValueError('native input member changed during source-entry build: '+name)
        for name, source_text in generated.items():
            path = stage/name
            if name in _ENTRY_SOURCES:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source_text)
            elif path.read_text() != source_text:
                raise ValueError('source-entry lowering would change the planned recurrence body')
        _write(stage/ABI_FILE, abi)
        _write(stage/BINDING_FILE, binding)
        launcher = stage/LAUNCHER
        launcher.write_text(render_source_launcher(placement['cpus']))
        launcher.chmod(0o755)
        dependencies = manifest['dependencies']
        libraries = tuple(Path(dependencies[name]) for name in ('libtorch_cpu.so', 'libc10.so'))
        _, emitted = build_native_project(stage, Path(manifest['executable']).name, shlex.split(compiler),
            expected_libraries=libraries, require_cpu_only=True)
        if dependencies != emitted:
            raise ValueError('source-entry ELF changed the original qualified CPU dependency closure')
        manifest['source_entry_cli'] = source_entry_description(manifest_sha256, source_file_publication=publication,
                                                                mapped_input_proof=mapping)
        manifest['files'] = {path.relative_to(stage).as_posix(): _content(path)
                             for path in sorted(stage.rglob('*')) if path.is_file()}
        _write(stage/'artifact.json', manifest)
        digest = _content(stage/'artifact.json')['sha256']
        verify_native_project(stage, digest)
        stage.rename(out)
    return dict(project=str(out), manifest_sha256=digest, entry_launcher=str(out/LAUNCHER),
        reused_placement=placement, native_schedule_replayed=True, source_cli_performance_measured=False)


def source_argv(abi, input_root, steps, output):
    if type(steps) is not int or not 1 <= steps <= 2**31-1:
        raise ValueError('positive signed 32-bit step count required for the normal entry contract')
    args = [''] * abi['argv']['count']
    for name, value in (('input_root', str(input_root)), ('steps', str(steps)), ('output', str(output))):
        args[abi['argv'][name]-1] = value
    return args


def _original_environment(manifest):
    libraries = Path(manifest['dependencies']['libtorch_cpu.so']).parent
    odeint = Path(manifest['method_provider']['sources']['odeint']['path'])
    pythonpath = [str(libraries.parents[1]), str(odeint.parents[2])]
    env = dict(LANG='C.UTF-8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=':'.join(pythonpath))
    check = subprocess.run([sys.executable, '-B', '-c',
        'from pathlib import Path; import json, torch, importlib; '
        'm = importlib.import_module("torchdiffeq._impl.odeint"); '
        'print(json.dumps([str(Path(torch.__file__).resolve().parent / "lib"), '
        'str(Path(m.__file__).resolve()), torch.version.cuda]))'], env=env, check=True, capture_output=True)
    if json.loads(check.stdout) != [str(libraries), str(odeint), None]:
        raise ValueError('original Python resolves a different Torch or fixed-grid provider')
    return env


def _inputs(abi, input_root):
    from math import prod

    receipts = []
    for row in abi['inputs']:
        path = input_root/row['path']
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 4*prod(row['shape'])):
            raise ValueError('original Tensor input must be a regular file of its exact admitted size: '+str(path))
        receipts.append(dict(port=row['port'], path=str(path), **_content(path)))
    return receipts


def validate_source_entry(project, manifest_sha256, input_root, out, *, steps, measure=False,
                          argv=None, original_cpu_ids=None, expected_environment_sha256=None):
    """Compare the unchanged Python entry and one selected C++ under the same argv."""
    from scar.optimization.paired_entry_selection import compare_entry_pairs
    from scar.runtime.validate_native_project import verify_native_project

    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh validation output directory required')
    root, _ = verify_native_project(project, manifest_sha256)
    manifest = _read(root/'artifact.json')
    if 'source_entry_cli' not in manifest:
        raise ValueError('replayed native source-entry project required')
    abi = _read(root/ABI_FILE)
    explicit = argv is not None or original_cpu_ids is not None
    if explicit and (type(argv) is not list or len(argv) != abi['argv']['count']
            or any(type(arg) is not str or '\x00' in arg for arg in argv)):
        raise ValueError('exact source-entry argv required with an explicit CPU cohort')
    input_root = Path(input_root).resolve(strict=True)
    publication = out/'publication.f32'
    if explicit:
        input_argument = Path(argv[abi['argv']['input_root'] - 1])
        publication = Path(argv[abi['argv']['output'] - 1])
        step_argument = argv[abi['argv']['steps'] - 1]
        significant = step_argument.lstrip('0')
        if (not input_argument.is_absolute() or input_argument.resolve(strict=True) != input_root
                or re.fullmatch('[0-9]+', step_argument) is None or not significant or len(significant) > 10
                or type(steps) is not int or int(significant) != steps or not 1 <= steps < 2**31
                or not publication.is_absolute() or not publication.parent.is_dir()
                or publication.exists() or publication.is_symlink()):
            raise ValueError('explicit argv differs from the admitted input, legal count or fresh publication scope')
        input_root = input_argument
    else:
        argv = source_argv(abi, input_root, steps, publication)
    if original_cpu_ids is None:
        original_cpu_ids = manifest['cpus']
    if (type(original_cpu_ids) is not list or not original_cpu_ids
            or any(type(cpu) is not int or cpu < 0 for cpu in original_cpu_ids)
            or len(set(original_cpu_ids)) != len(original_cpu_ids)
            or not set(manifest['cpus']) <= set(original_cpu_ids)
            or explicit and not set(original_cpu_ids) <= os.sched_getaffinity(0)):
        raise ValueError('original and native CPU cohorts differ from the permitted validation scope')
    inputs = _inputs(abi, input_root)
    source = Path(abi['solver']['source']['path'])
    expected_source = {key: abi['solver']['source'][key] for key in ('sha256', 'size_bytes')}
    source_files = {source: expected_source, **{
        Path(row['path']): {key: row[key] for key in ('sha256', 'size_bytes')}
        for row in manifest['method_provider']['sources'].values()}}
    if any(_content(path) != expected for path, expected in source_files.items()):
        raise ValueError('original Python entry or fixed-grid provider differs from its admitted bytes')
    env = _original_environment(manifest)
    if (expected_environment_sha256 is not None and expected_environment_sha256 !=
            sha256(json.dumps(sorted(env.items()), separators=(',', ':')).encode()).hexdigest()):
        raise ValueError('validation provider environment differs from the selected entry scope')
    out.mkdir(parents=True)
    cpus = ','.join(map(str, original_cpu_ids))
    commands = dict(ORIGINAL_PYTHON=['/usr/bin/taskset', '-c', cpus, sys.executable, '-B', str(source), *argv],
                    NATIVE_CPP=[str(root/LAUNCHER), *argv])
    runs, observations = [], []
    count = 4 if measure else 1
    original_bytes = None
    for index in range(count):
        outputs = {}
        order = tuple(commands) if index % 2 == 0 else tuple(reversed(commands))
        for mode in order:
            if publication.exists() or publication.is_symlink():
                raise ValueError('source-entry comparison requires an absent publication before every launch')
            if measure:
                start = perf_counter_ns()
            result = subprocess.run(commands[mode], env=env, check=True, capture_output=True)
            elapsed = perf_counter_ns()-start if measure else None
            if not stat.S_ISREG(publication.lstat().st_mode):
                raise ValueError('source-entry comparison produced a non-file publication')
            outputs[mode] = (result.stdout, result.stderr, publication.read_bytes())
            retained = out/f'pair-{index}-{mode}.f32'
            shutil.move(publication, retained)
            runs.append(dict(pair=index, mode=mode, wall_ns=elapsed, returncode=result.returncode,
                stdout_sha256=sha256(result.stdout).hexdigest(), stderr_sha256=sha256(result.stderr).hexdigest()))
            observations.append(dict(pair=index, mode=mode, stdout=result.stdout.decode(), stderr=result.stderr.decode(),
                output=dict(path=str(retained), **_content(retained))))
        baseline = outputs['ORIGINAL_PYTHON']
        if baseline != outputs['NATIVE_CPP'] or baseline[1] or original_bytes is not None and original_bytes != baseline[2]:
            raise ValueError('source-entry stdout, stderr or exact final publication differs')
        original_bytes = baseline[2]
    if _inputs(abi, input_root) != inputs or any(_content(path) != expected for path, expected in source_files.items()):
        raise ValueError('original inputs or source authority changed while comparing the entry')
    decision = compare_entry_pairs(runs, incumbent='ORIGINAL_PYTHON', challenger='NATIVE_CPP',
        warmup_pairs=1, measurement_pairs=3, minimum_gain=.05) if measure else None
    report = dict(schema='scar.aten-source-entry-validation', schema_version=1,
        project=str(root), manifest_sha256=manifest_sha256, steps=steps, inputs=inputs, argv=argv,
        commands=commands, cpu_ids=manifest['cpus'], python_provider_environment=env,
        observations=observations, runs=runs if measure or explicit else [],
        exact_publication_stdout_stderr_exit_verified=True,
        full_process_measured=measure, timing_scope='SOURCE_CLI_PROCESS_LAUNCH_TO_EXIT_INCLUDING_FILE_IO',
        decision=decision, deployment_authorized=False)
    if explicit:
        report.update(original_cpu_ids=original_cpu_ids, working_directory=str(Path.cwd().resolve()))
    _write(out/'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    bind = subs.add_parser('bind')
    validate = subs.add_parser('validate')
    for command in (bind, validate):
        command.add_argument('--project', required=True)
        command.add_argument('--manifest-sha256', required=True)
        command.add_argument('--out', required=True)
    bind.add_argument('--compiler', default='g++')
    bind.add_argument('--preserve-source-file-mode', action='store_true',
                      help='create the fresh source output with the normal process and directory permissions')
    bind.add_argument('--mapped-input-proof',
                      help='emit a candidate using the proved provider-backed source input mapping')
    validate.add_argument('--input-root', required=True)
    validate.add_argument('--steps', required=True, type=int)
    validate.add_argument('--measure', action='store_true')
    args = vars(parser.parse_args(argv))
    action = bind_source_entry if args.pop('command') == 'bind' else validate_source_entry
    result = action(**args)
    print(json.dumps({key: result[key] for key in ('project', 'manifest_sha256')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
