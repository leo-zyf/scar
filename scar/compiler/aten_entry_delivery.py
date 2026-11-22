"""Aten entry delivery."""

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from scar.codegen.native_provider_bundle import _content
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


# Bind a source-proved CLI to one replayed standalone ATen recurrence.
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
        warmup_pairs=1, measurement_pairs=3) if measure else None
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


# Enumerate a source-qualified fixed-grid CPU entry domain before CLI lowering.
SCHEMA = 'scar.aten-entry-variants'
BODY_SCHEMA = 'scar.aten-recurrence-delivery'
FAMILY = 'ATEN_FIXED_GRID_CPU_FILE_ENTRY'
_MODES = ('READY_DAG', 'SERIAL')


class _ScopeFrontier(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _require(condition, reason, detail):
    if not condition:
        raise _ScopeFrontier(reason, detail)


def _bind_scope(scope, emission):
    from scar.compiler.aten_entry_delivery import _inputs

    abi = emission['abi']
    origin = abi['solver']['source']
    source = artifact_reference(origin['path'], 'scar.python-source')
    _require((source['uri'], source['sha256'], source['size_bytes'])
             == (scope['entry'], scope['entry_sha256'], origin['size_bytes'])
             and source['sha256'] == origin['sha256'],
             'ENTRY_SOURCE_MISMATCH', 'qualified recurrence source differs from the F4 entry')
    _require(emission['target']['sha256'] == scope['target_sha256'],
             'TARGET_MISMATCH', 'qualified recurrence target differs from the F4 hardware snapshot')
    _require(emission['storage']['budget_bytes'] == scope['memory_budget_bytes'],
             'TENSOR_BUDGET_MISMATCH', 'qualified recurrence requires its exact modeled Tensor budget')
    argv, slots = scope['argv'], abi['argv']
    _require(len(argv) == slots['count'], 'ABI_ARGUMENT_MISMATCH', 'entry argument count differs from the source ABI')
    value = argv[slots['steps'] - 1]
    digits = value.lstrip('0') or '0'
    _require(re.fullmatch('[0-9]+', value) is not None and len(digits) <= 10
             and abi['domain']['step_count_min'] <= int(digits) <= abi['domain']['step_count_max'],
             'ABI_STEP_DOMAIN_MISMATCH', 'step count must satisfy the original signed-int32 decimal domain')
    input_root, output = (Path(argv[slots[name] - 1]) for name in ('input_root', 'output'))
    _require(input_root.is_absolute() and output.is_absolute() and output.parent.is_dir()
             and not output.is_symlink() and not output.exists(),
             'FILE_BOUNDARY_MISMATCH', 'absolute Tensor inputs and a fresh output with an existing parent are required')
    common_shape = scope.get('input_shapes', {}).get('common_shape')
    _require(common_shape is None or all(row['shape'] == common_shape for row in abi['inputs']),
             'INPUT_SHAPE_MISMATCH', 'declared input shapes differ from the qualified source ports')
    return source, int(digits), _inputs(abi, input_root)


def _cost_demand(body, emission, observation, alternative, *, steps, file_inputs):
    placement = emission['placement']
    input_bytes = sum(row['size_bytes'] for row in file_inputs)
    output_bytes = next(row['size_bytes'] for row in file_inputs
                        if row['port'] == emission['abi']['solver']['state_input_index'])
    inputs_match = [{key: val for key, val in row.items() if key != 'port'} for row in file_inputs]
    inputs_match = inputs_match == observation['inputs']
    step_ns = alternative['body_estimate_ns']
    return dict(boundary='PROCESS_LAUNCH_THROUGH_EXIT', model='SCOPED_COMPONENT_DEMANDS',
        trip_count=steps, body=dict(estimated_step_ns=step_ns,
            estimated_total_ns=step_ns * steps if inputs_match else None,
            calibration_matches_exact_files=inputs_match, physical_schedule=emission['schedule'],
            calibration_repeats=observation['repeats'], claimed_process_bound=False),
        input_logical_bytes=input_bytes, output_logical_bytes=output_bytes,
        graph_held_peak_bytes=alternative['peak_tensor_storage_bytes'], unmodeled_memory=emission['storage']['excluded'],
        placement=placement, source_evidence=body,
        phases=dict(startup='ELF_LOADER_AND_ATEN_THREAD_CONFIGURATION', imports='INAPPLICABLE',
            guards='SOURCE_ARGV_DOMAIN_AND_INPUT_FILE_CHECKS', loading='INPUT_FILE_BACKING_AND_TENSOR_VIEWS',
            worker_lifecycle='FIXED_NATIVE_TEAM' if placement['mode'] == 'READY_DAG' else 'INAPPLICABLE',
            allocation='ATEN_TENSOR_AND_PROVIDER_WORKSPACE', file_io='READ_INPUT_AND_WRITE_OUTPUT',
            transfers='INAPPLICABLE', synchronization='CPU_TEAM_ORDER' if placement['mode'] == 'READY_DAG' else 'INAPPLICABLE',
            contention='ATEN_AND_HOST_RESOURCE_CONTENTION', execution='REPEATED_REPLAYED_CPU_STEP',
            publication='OUTPUT_FILE_THEN_STDOUT', fallback='INAPPLICABLE', teardown='RESOURCE_CLEANUP_AND_PROCESS_EXIT'),
        complete_process_ns=None)


def derive_aten_entry_variants(scope, body_ref):
    """Use original recurrence evidence to enumerate a finite source-entry family.

    The implementation IDs bind per-mode execution and the declared entry
    domain. Replay proves qualification on that domain. Estimates from the
    calibrated step are not full-entry bounds or deployment permission.
    """
    from scar.compiler.aten_source_entry_plan import plan_source_entry_emission

    scope_sha = semantic_sha256(scope)
    result = dict(schema=SCHEMA, schema_version=1, family=FAMILY, scope_sha256=scope_sha,
        body=body_ref, options=[], coverage_frontier=[], baseline_id='original',
        compiled_source_entry=False, target_executed=False)
    frontier = result['coverage_frontier']
    if body_ref is None:
        frontier.append(dict(reason='FIXED_GRID_BODY_EVIDENCE_REQUIRED'))
        return result
    try:
        _require(scope['boundary'] == 'PROCESS_LAUNCH_THROUGH_EXIT'
                 and scope['native_output_contract'] == 'STANDALONE_CPP',
                 'BOUNDARY_MISMATCH', 'fixed-grid generation requires a standalone process contract')
        _require(body_ref['schema'] == BODY_SCHEMA, 'BODY_SCHEMA_MISMATCH', 'CPU recurrence body evidence required')
        body = read_artifact(body_ref)
        _require(body.get('source_entry_replayable') is True and body.get('step_lifetime_planned') is True
                 and 'source_entry_cli' not in body and 'cpu_linkage' in body,
                 'BODY_PROOF_REQUIRED', 'source-linked CPU-only recurrence with replayable lifetime plans required')
        original_mode = body['rhs_mode']
        selected = plan_source_entry_emission(body_ref, original_mode)
        source, steps, file_inputs = _bind_scope(scope, selected)
    except _ScopeFrontier as error:
        frontier.append(dict(reason=error.reason, detail=str(error)))
        return result
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        frontier.append(dict(reason='BODY_EVIDENCE_REPLAY_REQUIRED', detail=str(error)))
        return result

    allowed_cpus = set(scope['process_affinity_cpu_ids'])
    alternatives = body['alternatives']
    for mode in _MODES:
        if mode not in alternatives:
            frontier.append(dict(mode=mode, reason='MODE_NOT_COVERED_BY_RETAINED_BODY_PROOF'))
            continue
        alternative = alternatives[mode]
        proof = dict(body=body_ref, mode=mode, source=source, source_contract_sha256=selected['source_contract_sha256'],
                     target=selected['target'], file_inputs=file_inputs, steps=steps,
                     memory_budget_bytes=scope['memory_budget_bytes'], peak_bytes=alternative['peak_tensor_storage_bytes'])
        demand = None
        status, reason = 'INELIGIBLE', 'GRAPH_HELD_STORAGE_EXCEEDS_BUDGET'
        if alternative['within_budget']:
            try:
                emission = selected if mode == original_mode else plan_source_entry_emission(body_ref, mode)
                if emission['source_contract_sha256'] != proof['source_contract_sha256'] or emission['target'] != proof['target']:
                    raise ValueError('alternate emission must preserve entry ABI and target')
                proof['emission_plan'] = emission
                if set(emission['placement']['cpus']) <= allowed_cpus:
                    demand = _cost_demand(body_ref, emission, read_artifact(emission['calibration']), alternative,
                                          steps=steps, file_inputs=file_inputs)
                    status, reason = 'LEGAL', 'REPLAYED_SOURCE_ABI_AND_CPU_BODY_PLAN'
                else:
                    reason = 'EXACT_CPU_PLACEMENT_OUTSIDE_ENTRY_COHORT'
            except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
                status, reason = 'UNRESOLVED', 'MODE_PROOF_REPLAY_REQUIRED'
                proof['detail'] = str(error)
        identity = dict(scope_sha256=scope_sha, body_sha256=body_ref['sha256'], mode=mode)
        result['options'].append(dict(id='planned:' + semantic_sha256(identity), legal_status=status,
            reason=reason, proof=proof, cost_demand=demand))
    read_artifact(body_ref)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
