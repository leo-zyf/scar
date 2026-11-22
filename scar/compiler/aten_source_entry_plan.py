"""Replay and emit one exact source-entry placement from a qualified ATen body."""

from pathlib import Path
import shlex
import shutil
from tempfile import TemporaryDirectory

from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_entry_delivery import ABI_FILE, BINDING_FILE, LAUNCHER, _read, _write, render_source_launcher, source_entry_description
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.aten-source-entry-emission-plan'
BODY = 'scar.aten-recurrence-delivery'
_EXECUTION = 'plan/execution.json'
_RECORDS = ('plan/problem.json', 'plan/schedule.json', 'validation/calibration.json')
_EMITTED = (_EXECUTION, 'src/main.cpp', 'src/regions/step.hpp', 'src/runtime/file_io.hpp',
            'launch.sh', 'validation/build.log', 'validation/ldd.txt')


def _receipt(root, files, name, schema):
    return dict(uri=str(root / name), schema=schema, **files[name])


def _execution(root, manifest, mode, recipe):
    from scar.optimization.aten_recurrence import plan_recurrence

    original = _read(root / _EXECUTION)
    if mode == manifest['rhs_mode']:
        return original
    inputs = _read(root / 'plan/inputs.json')
    rhs, ready = inputs['rhs_candidate_storage'], inputs['rhs_candidate_placement']
    peak = rhs['serial_peak_bytes' if mode == 'SERIAL' else 'peak_tensor_storage_bytes']
    placement = ready if mode == 'READY_DAG' else dict(mode='SERIAL', cpus=ready['cpus'][:1],
                                                        lanes=None, operation_order=None)
    execution = plan_recurrence(inputs['lowered_step'], recipe, dict(rhs, peak_tensor_storage_bytes=peak),
        state_input_index=original['state_input_index'], step_size=original['step_size'], optimize_lifetime=True)
    execution['storage']['budget_bytes'] = manifest['tensor_budget_bytes']
    execution.update(placement=placement, timing_basis=original['timing_basis'],
        program_graph=original['program_graph'], target=original['target'],
        rhs_schedule=inputs['rhs']['execution_plan'] if mode == 'READY_DAG' else None)
    return execution


def _selected_members(root, manifest, mode):
    files = manifest['files']
    prefix = 'validation/alternative/' if mode != manifest['rhs_mode'] else None
    schemas = ('scar.implementation-planning-problem', 'scar.implementation-execution-plan',
               'scar.tensor-body-observation')
    members = {name: _receipt(root, files, prefix + Path(name).name if prefix else name, schema)
               for name, schema in zip(_RECORDS, schemas, strict=True)}
    if mode == 'READY_DAG':
        members.update({name: _receipt(root, files, prefix + Path(name).name if prefix else name, schema)
                        for name, schema in zip(('plan/rhs-problem.json', 'plan/rhs-schedule.json'), schemas)})
    return members


def _material(body, mode):
    from scar.analysis.fixed_grid_entry_abi import derive_fixed_grid_file_entry
    from scar.compiler.aten_source_entry_binding import verify_source_entry_binding
    from scar.compiler.tensor_planning_inputs import scheduled_tensor_body
    from scar.runtime.validate_native_project import verify_native_project

    manifest = read_artifact(body)
    origin = Path(body['uri'])
    if (manifest['schema'] != BODY or origin.name != 'artifact.json'
            or 'source_entry_cli' in manifest or 'source_entry_plan' in manifest
            or manifest.get('step_lifetime_planned') is not True
            or manifest.get('source_entry_replayable') is not True
            or mode not in ('SERIAL', 'READY_DAG')
            or not manifest.get('alternatives', {}).get(mode, {}).get('within_budget')):
        raise ValueError('emission requires a budget-eligible placement of an original source-replayable ATen body')
    root, receipt = verify_native_project(origin.parent, body['sha256'])
    if 'cpu_linkage' not in receipt:
        raise ValueError('standalone emission requires an admitted CPU-only operator dependency closure')
    members = _selected_members(root, manifest, mode)
    problem, schedule = (read_artifact(members[name]) for name in _RECORDS[:2])
    recipe = _read(root / 'plan/rhs-recipe.json')
    execution = _execution(root, manifest, mode, recipe)
    placement = execution['placement']
    selected = dict(manifest, rhs_mode=mode, cpus=placement['cpus'], lanes=placement['lanes'],
                    operation_order=placement['operation_order'])
    boundary = _read(root / 'plan/source-entry.json')
    abi = derive_fixed_grid_file_entry(_read(root / 'plan/source-entry-binding.json'), boundary['entry'],
                                      source_bytes=(root / 'plan/source-entry.py').read_bytes())
    source = abi['solver']['source']
    source_files = (source, *selected['method_provider']['sources'].values())
    if any(_content(Path(row['path'])) != {key: row[key] for key in ('sha256', 'size_bytes')}
           for row in source_files):
        raise ValueError('live entry or fixed-grid provider source differs from the admitted recurrence')
    binding = verify_source_entry_binding(abi, execution, recipe, manifest=selected)
    read_artifact(binding['target'])
    lowered = scheduled_tensor_body(problem, schedule, execution)
    plan = dict(schema=SCHEMA, schema_version=1, body=body, mode=mode, source=source,
        source_contract_sha256=binding['source_contract_sha256'], execution_plan_sha256=binding['execution_plan_sha256'],
        problem=members[_RECORDS[0]], schedule=members[_RECORDS[1]], calibration=members[_RECORDS[2]],
        target=binding['target'],
        placement=placement, storage=binding['storage'], abi=abi)
    read_artifact(body, json_document=False)
    return plan, root, selected, members, execution, binding, lowered


def plan_source_entry_emission(body, mode):
    """Pin one source/buffer/schedule plan without emitting or running an entry."""
    plan, *_ = _material(body, mode)
    return plan


def _retained_members(manifest):
    replaced = {*_EMITTED, *_RECORDS, 'plan/rhs-problem.json', 'plan/rhs-schedule.json',
                'validation/comparison.json', manifest['executable']}
    return {name: receipt for name, receipt in manifest['files'].items()
            if name not in replaced and not name.startswith('validation/alternative/')}


def verify_source_entry_emission(manifest, member):
    """Rebuild the selected plan and check the emitted project's preserved members."""
    plan = manifest['source_entry_plan']
    expected, _, original, selected_members, execution, binding, _ = _material(plan['body'], plan['mode'])
    keys = ('alternatives', 'provider', 'cpu_linkage', 'dependencies', 'source_references', 'method_provider',
            'rhs_mode', 'cpus', 'lanes', 'operation_order', 'tensor_budget_bytes', 'executable', 'calibration_steps',
            'source_entry_inferred', 'source_entry_replayable', 'step_lifetime_planned', 'cxx_standard', 'abi_version')
    if (plan != expected or manifest.get('source_entry_cli') != source_entry_description(plan['body']['sha256'])
            or manifest.get('selection_reason') != 'SELECTED_ENTRY_PLAN'
            or manifest.get('full_entry_performance_measured') is not False
            or any(manifest[key] != original[key] for key in keys)
            or _read(member(_EXECUTION)) != execution or _read(member(BINDING_FILE)) != binding):
        raise ValueError('generated source entry does not implement the selected emission plan')
    expected_members = _retained_members(original)
    expected_members.update({name: {key: ref[key] for key in ('size_bytes', 'sha256')}
                             for name, ref in selected_members.items()})
    if any(_content(member(name)) != receipt for name, receipt in expected_members.items()):
        raise ValueError('generated source entry changed a checked parent plan or provider member')


def emit_planned_source_entry(plan, out, *, compiler='g++'):
    """Emit precisely the selected placement, without consulting body A/B to choose again."""
    from scar.codegen.aten_recurrence_source import render_recurrence
    from scar.codegen.native_project import build_native_project
    from scar.compiler.compile_aten_recurrence import _render_recurrence_launch
    from scar.runtime.validate_native_project import verify_native_project

    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh source-entry project required')
    expected, root, manifest, members, execution, binding, lowered = _material(plan['body'], plan['mode'])
    if plan != expected:
        raise ValueError('source-entry emission plan differs from the original schedule replay')
    placement = execution['placement']
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-source-aten-plan-', dir=out.parent) as scratch:
        stage = Path(scratch) / 'project'
        for name, receipt in _retained_members(manifest).items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
            if _content(target) != receipt:
                raise ValueError('native parent member changed during emission: ' + name)
        for name, reference in members.items():
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            (stage / name).write_bytes(read_artifact(reference, json_document=False))
        _write(stage / _EXECUTION, execution)
        _write(stage / ABI_FILE, plan['abi'])
        _write(stage / BINDING_FILE, binding)
        for name, source in render_recurrence(lowered, rhs_mode=placement['mode'], cpus=placement['cpus'],
                lanes=placement['lanes'], operation_order=placement['operation_order'], source_entry_abi=plan['abi']).items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        for name, source in (('launch.sh', _render_recurrence_launch(placement['cpus'])),
                             (LAUNCHER, render_source_launcher(placement['cpus']))):
            launcher = stage / name
            launcher.write_text(source)
            launcher.chmod(0o755)
        dependencies = manifest['dependencies']
        libraries = tuple(Path(dependencies[name]) for name in ('libtorch_cpu.so', 'libc10.so'))
        executable, actual = build_native_project(stage, Path(manifest['executable']).name, shlex.split(compiler),
            expected_libraries=libraries, require_cpu_only=True)
        if actual != dependencies or executable.relative_to(stage).as_posix() != manifest['executable']:
            raise ValueError('source-entry ELF changed the qualified dependency or executable contract')
        manifest.update(source_entry_plan=plan, selection_reason='SELECTED_ENTRY_PLAN',
            full_entry_performance_measured=False, source_entry_cli=source_entry_description(plan['body']['sha256']),
            files={path.relative_to(stage).as_posix(): _content(path)
                   for path in sorted(stage.rglob('*')) if path.is_file()})
        _write(stage / 'artifact.json', manifest)
        digest = _content(stage / 'artifact.json')['sha256']
        verify_native_project(stage, digest)
        stage.rename(out)
    return dict(project=str(out), manifest_sha256=digest, entry_launcher=str(out / LAUNCHER),
        artifact=artifact_reference(out / 'artifact.json', BODY),
        executable=artifact_reference(out / manifest['executable'], 'scar.native-elf'),
        reused_placement=placement, native_schedule_replayed=True, source_cli_performance_measured=False)
