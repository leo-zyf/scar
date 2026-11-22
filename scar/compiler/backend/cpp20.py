"""Verify selected execution products and deliver C++20 targets."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess

from scar.artifacts import file_reference
from scar.codegen.native_cpu_elf import _dependencies, cpu_executable_dynamic
from scar.codegen.native_project import (
    build_native_project, publish_cpp20_inputs, replay_cpp20_inputs, write_native_project,
)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA, loads
from scar.optimization.resource_solver import semantic_sha256

from ..contracts import ExecutionPlanRef, ScheduledLoop, verify_source_lock
from ..middle.plan import copy_source_edits, publish_plan, read_native_binding
from ..operator_stack import (
    load_operator_stack, nvidia_stack_build, verify_operator_stack_linkage, verify_stack_components_once,
)
from ..qualification import executable_receipt


_NATIVE_ENTRY = 'CPP20_NATIVE_ENTRY'
_SERIAL_LOOP = 'CPP_SERIAL_TEMPORAL'
_OPENMP_LOOP = 'CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL'
_ABI1_PROFILES = frozenset((_NATIVE_ENTRY, _SERIAL_LOOP, _OPENMP_LOOP))
_ENTRY_TARGET = 'scar_entry'
_ENTRY_BINARY = Path('bin') / _ENTRY_TARGET
_REVISION_SCHEMAS = {
    'request': 'scar.native-entry', 'program': PROGRAM_SCHEMA,
    'graph_evidence': 'scar.native-graph-evidence', 'calls': 'scar.native-call-links',
    'hardware': 'scar.target-hardware-graph', 'simplified': PROGRAM_SCHEMA,
    'rewrites': 'scar.native-rewrites',
}


def cpp20_capability(profile: str, *, operator_abi_version: int = 1) -> dict:
    """Describe the target ABI used by an admitted delivery profile."""
    if profile not in _ABI1_PROFILES or type(operator_abi_version) is not int or operator_abi_version != 1:
        raise ValueError('unqualified C++20 operator ABI or delivery profile')
    report = dict(schema='scar.cpp20-backend-capability', schema_version=1, profile=profile,
                  cxx_standard=20, operator_abi_version=operator_abi_version)
    if profile == _NATIVE_ENTRY:
        report['target_kind'] = 'EXECUTABLE'
    else:
        report.update(target_kind='SHARED_OPERATOR', temporal_control_abi_version=1)
    return report


@dataclass(frozen=True, slots=True)
class Cpp20Toolchain:
    compiler: dict
    stack: dict | None
    route: str
    inputs: dict
    stack_document: dict | None = None

    def receipt(self) -> dict:
        return dict(schema='scar.cpp20-toolchain', schema_version=1, compiler=self.compiler,
                    operator_stack=self.stack, route=self.route)


def bind_cpp20_toolchain(stack=None, *, route: str = 'CPU') -> Cpp20Toolchain:
    """Resolve the declared toolchain and verify its locked library stack."""
    if route not in ('CPU', 'CUDA', 'HYBRID'):
        raise ValueError(f'unsupported operator-stack route: {route}')
    if stack is not None:
        reference, document = load_operator_stack(stack)
        build = nvidia_stack_build(document, route)
        compiler = file_reference(build.pop('compiler')[0], 'scar.cpp20-compiler')
        inputs = {key: tuple(map(str, values)) for key, values in build.items()}
        return Cpp20Toolchain(compiler, reference, route, inputs, document)
    if route != 'CPU':
        raise ValueError('accelerator builds require a locked operator stack')
    path = shutil.which('g++')
    if path is None:
        raise ValueError('g++ is required for the default CPU toolchain')
    compiler = file_reference(Path(path).resolve(strict=True), 'scar.cpp20-compiler')
    return Cpp20Toolchain(compiler, None, route, {})


def _entry_toolchain(manifest):
    stack = manifest.get('operator_stack')
    if stack is not None and not isinstance(stack, dict):
        raise ValueError('native operator stack must be a locked artifact reference')
    return bind_cpp20_toolchain(stack)


@dataclass(frozen=True, slots=True)
class NativeProject:
    executable: Path
    receipt: dict
    observation: dict
    delivery: dict
    dynamic: dict
    artifacts: dict


def deliver(plan: ExecutionPlanRef | ScheduledLoop, project: Path | None = None, *,
            catalog=None, plan_only: bool = False) -> NativeProject | dict:
    """Lower a verified entry or region plan through the C++20 backend."""
    if isinstance(plan, ExecutionPlanRef) and project is not None and not plan_only:
        return _emit_native(plan, Path(project).resolve())
    if isinstance(plan, ScheduledLoop) and project is None and catalog is not None:
        return _lower_loop(plan, catalog=catalog, plan_only=plan_only)
    raise ValueError('C++20 delivery requires a project path for an entry or a source catalog for a loop')


def _emit_native(schedule: ExecutionPlanRef, project: Path) -> NativeProject:
    """Replay the schedule, emit its C++20 project and seal the linked output."""
    from ..frontend.clang import verify_mixed_linkage

    artifacts = schedule.artifacts.copy()
    manifest, observation, delivery = admit_entry(schedule)
    with verify_stack_components_once():
        toolchain = _entry_toolchain(manifest)
        delivery['toolchain'] = toolchain.receipt()
        if 'trial_variant' in delivery:
            executable, inputs, dynamic = promote_cpp20_variant(project, artifacts, delivery, toolchain=toolchain)
        else:
            executable, inputs, dynamic = _realize_entry(
                project, manifest, delivery['source_edits'], delivery['compiler_types'],
                plan=delivery['plan'], toolchain=toolchain,
            )
    delivery['cpp20_inputs'] = artifacts['cpp20_inputs'] = inputs
    delivery['mixed_linkage'] = verify_mixed_linkage(manifest, artifacts['calls'], executable)
    fortran = project / 'fortran-emission.json'
    if fortran.is_file():
        delivery['fortran_emission'] = artifacts['fortran_emission'] = file_reference(fortran, 'scar.fortran-emission')
    receipt = executable_receipt(executable)
    artifacts['delivery'] = publish_plan(project / 'delivery.json', dict(delivery, executable=receipt))
    return NativeProject(executable, receipt, observation, delivery, dynamic, artifacts)


def replay_native_revision(artifacts):
    """Verify the original graph, source lock and exact proposed rewrites."""
    from scar.analysis.replay_transaction import detached_replay

    identity = tuple((role, tuple(sorted(artifacts[role].items()))) for role in _REVISION_SCHEMAS)
    material = detached_replay('scar.cpp20-native-revision.v1', identity, lambda: _read_native_revision(artifacts))
    verify_source_lock(material[0])
    return material


def _read_native_revision(artifacts):
    for role, schema in _REVISION_SCHEMAS.items():
        ref = artifacts[role]
        if ref['schema'] != schema:
            raise ValueError(f'invalid {role} artifact schema')
        if role in {'graph_evidence', 'simplified'} and str(Path(ref['uri']).absolute()) != ref['uri']:
            raise ValueError(f'{role} artifact differs from the plan input')
    manifest = read_artifact(artifacts['request'])
    rewrite = read_artifact(artifacts['rewrites'])
    read_artifact(artifacts['hardware'])
    if (rewrite['schema_version'] != 1 or rewrite['program'] != artifacts['program']
            or rewrite['result'] != artifacts['simplified']):
        raise ValueError('rewrite ledger does not bind the admitted input and output graphs')
    edits = rewrite.get('source_edits', [])
    typed_units = {unit['path'] for unit in manifest['units']
                   if unit.get('language', manifest['language']) in ('c', 'fortran')}
    inspect_evidence = bool(edits or typed_units)
    hash_only = ['program', 'calls']
    if not inspect_evidence:
        hash_only.append('graph_evidence')
    if not edits:
        hash_only.append('simplified')
    for role in hash_only:
        ref = artifacts[role]
        if file_reference(ref['uri'], ref['schema']) != ref:
            raise ValueError(f'{role} artifact differs from the plan input')
    evidence = read_artifact(artifacts['graph_evidence']) if inspect_evidence else None
    if evidence and any(evidence[key] != artifacts[key] for key in ('request', 'program', 'hardware', 'calls')):
        raise ValueError('compiler evidence differs from the admitted frontend outputs')
    compiler_types = {row['unit']: row['artifacts']['types'] for row in (evidence or {}).get('units', ())
                      if row['unit'] in typed_units and 'types' in row['artifacts']}
    if compiler_types.keys() != typed_units:
        raise ValueError('C and Fortran translation require original type evidence for every unit')
    if edits:
        if copy_source_edits(manifest, evidence, rewrite['rules']) != edits:
            raise ValueError('source copy edits differ from frontend and IR evidence')
        simplified = loads(read_artifact(artifacts['simplified'], json_document=False).decode('utf-8'))
        remaining = {op.id.wire for op in simplified.operations}
        regions = {region.id.wire for region in simplified.regions}
        if any(edit['operation'] in remaining or edit['region'] not in regions for edit in edits):
            raise ValueError('source edits do not match the simplified IR')
    return manifest, edits, compiler_types


def admit_entry(schedule: ExecutionPlanRef):
    """Replay the selected implementation against the frozen problem and source."""
    artifacts = schedule.artifacts
    manifest, edits, compiler_types = replay_native_revision(artifacts)
    observation, binding = read_native_binding(schedule)
    operations = binding.pop('rewrite_operations')
    selected_edits = edits if operations else []
    if operations != [edit['operation'] for edit in selected_edits]:
        raise ValueError('scheduled source edits differ from the verified graph revision')
    capability = cpp20_capability(_NATIVE_ENTRY, operator_abi_version=manifest.get('operator_abi_version', 1))
    delivery = dict(schema='scar.native-delivery', schema_version=1,
                    plan=artifacts['execution_plan'], problem=artifacts['planning_problem'],
                    request=artifacts['request'], source_edits=selected_edits, compiler_types=compiler_types,
                    capability=capability, **binding)
    return manifest, observation, delivery


def _verify_linkage(executable, toolchain, dependencies=None):
    dynamic = cpu_executable_dynamic(executable)
    for name in dynamic['needed']:
        if any(provider in name for provider in ('libpython', 'libtorch', 'libgfortran')):
            raise ValueError(f'unqualified target dependency: {name}')
    if toolchain.stack_document is not None:
        if dependencies is None:
            env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
            dependencies = _dependencies(subprocess.check_output(['ldd', str(executable)], env=env, text=True))
        dynamic['operator_stack'] = verify_operator_stack_linkage(toolchain.stack_document, dependencies, toolchain.route)
    return dynamic


def build_cpp20_project(project, inputs, *, plan, toolchain=None):
    """Build source-adapter output with one target language and verify its ABI."""
    configuration = replay_cpp20_inputs(project, inputs, plan=plan)
    toolchain = toolchain or bind_cpp20_toolchain()
    _verify_toolchain_inputs(toolchain, configuration)
    _verify_compiler(toolchain)
    write_native_project(project, _ENTRY_TARGET, **configuration)
    executable, dependencies = build_native_project(project, _ENTRY_TARGET, (toolchain.compiler['uri'],))
    replay_cpp20_inputs(project, inputs, plan=plan)
    _verify_compiler(toolchain)
    return executable, _verify_linkage(executable, toolchain, dependencies)


def _realize_entry(project, manifest, edits, compiler_types, *, plan, toolchain):
    from ..frontend.clang import stage_source_revision

    inputs = stage_source_revision(manifest, project, edits, captures=compiler_types, plan=plan,
                                   target_inputs=toolchain.inputs)
    executable, dynamic = build_cpp20_project(project, inputs, plan=plan, toolchain=toolchain)
    return executable, inputs, dynamic


def _verify_compiler(toolchain):
    compiler = toolchain.compiler
    if file_reference(compiler['uri'], compiler['schema']) != compiler:
        raise ValueError('selected C++20 compiler changed during delivery')


def _verify_toolchain_inputs(toolchain, configuration):
    for key, members in toolchain.inputs.items():
        if not set(members) <= set(configuration.get(key, ())):
            raise ValueError(f'selected toolchain inputs are absent from the C++20 project: {key}')


def _trial_source(artifacts, edits, mode):
    return dict(schema='scar.native-source-variant', schema_version=1,
                request=artifacts['request'], rewrites=artifacts['rewrites'], mode=mode, source_edits=edits)


def build_trial_variant(directory, artifacts, manifest, edits, compiler_types, *, mode):
    directory.mkdir(parents=True)
    plan = publish_plan(directory / 'source-variant.json', _trial_source(artifacts, edits, mode))
    with verify_stack_components_once():
        toolchain = _entry_toolchain(manifest)
        executable, inputs, _ = _realize_entry(directory / 'project', manifest, edits, compiler_types,
                                              plan=plan, toolchain=toolchain)
    binary = file_reference(executable, 'scar.cpp20-executable')
    variant = publish_plan(directory / 'result.json', dict(
        schema='scar.native-compiled-variant', schema_version=1, plan=plan, inputs=inputs, binary=binary,
        toolchain=toolchain.receipt(),
    ))
    return executable, variant


def promote_cpp20_variant(project, artifacts, delivery, *, toolchain):
    """Deliver the scheduled executable byte for byte, without rebuilding it."""
    reference = delivery['trial_variant']
    if reference['schema'] != 'scar.native-compiled-variant':
        raise ValueError('selected native trial receipt required')
    variant = read_artifact(reference)
    recipe = read_artifact(variant['plan'])
    mode = delivery['selected'][0]['option']
    if (recipe != _trial_source(artifacts, delivery['source_edits'], mode)
            or variant['schema_version'] != 1 or variant.get('toolchain') != toolchain.receipt()):
        raise ValueError('selected trial belongs to a different source revision')
    staged = Path(reference['uri']).parent / 'project'
    original = staged / _ENTRY_BINARY
    binary = variant['binary']
    if file_reference(original, 'scar.cpp20-executable') != binary:
        raise ValueError('selected executable differs from the calibrated binary')
    build = replay_cpp20_inputs(staged, variant['inputs'], plan=variant['plan'])
    _verify_toolchain_inputs(toolchain, build)
    project.mkdir()
    for name in ('src', 'include', 'validation'):
        if (staged / name).is_dir():
            shutil.copytree(staged / name, project / name)
    for name in ('translation.json', 'fortran-emission.json'):
        if (staged / name).is_file():
            shutil.copy2(staged / name, project / name)
    # CMakeLists is derived from the sealed configuration, not part of a
    # calibrated binary's identity. Generate it from the verified inputs.
    write_native_project(project, _ENTRY_TARGET, **build)
    executable = project / _ENTRY_BINARY
    executable.parent.mkdir()
    shutil.copy2(original, executable)
    inputs = publish_cpp20_inputs(project, build, plan=delivery['plan'])
    before = read_artifact(variant['inputs'])
    after = read_artifact(inputs)
    if ({name: dict(row, uri=str(project / name)) for name, row in before['files'].items()} != after['files']
            or before['libraries'] != after['libraries']):
        raise ValueError('delivered project inputs differ from the calibrated variant')
    delivered = file_reference(executable, binary['schema'])
    if delivered != dict(binary, uri=str(executable)):
        raise ValueError('published executable differs from the calibrated binary')
    _verify_compiler(toolchain)
    return executable, inputs, _verify_linkage(executable, toolchain)


def _lower_loop(work: ScheduledLoop, *, catalog, plan_only: bool) -> dict:
    from scar.compiler.invariant_export import export_temporal_invariant, export_composite_temporal_invariant
    from scar.compiler.loop_admission import read_invariant_loop_motion
    from scar.compiler.pipeline_loop_frontend import read_composite_invariant_loop_motion
    from scar.compiler.temporal_invariant_loop import emit_temporal_invariant_loop, uses_openmp

    folder, composite, parallel = work.folder, work.composite, work.parallel
    reader = read_composite_invariant_loop_motion if composite else read_invariant_loop_motion
    expected = dict(schema='scar.cpp20-loop-lowering', schema_version=1,
                    loop_id=work.loop_id, proof=work.proof, source_motion=work.source_motion,
                    schedule=work.schedule, parallel=parallel)
    prepared = read_artifact(work.lowering)
    if {key: value for key, value in prepared.items() if key != 'plan'} != expected:
        raise ValueError('loop lowering does not bind the scheduled source and resources')
    schedule = read_artifact(work.schedule)
    read_artifact(work.proof)
    proof, motion = reader(work.proof['uri'], catalog=catalog)
    if (work.loop_id != proof.id
            or semantic_sha256(motion) != semantic_sha256(read_artifact(work.source_motion))):
        raise ValueError('loop movement changed between original F4 admission and lowering')
    plan = prepared['plan']
    if uses_openmp(plan) != parallel:
        raise ValueError('C++20 loop lowering uses a different worker profile than the schedule')
    execution = plan['execution_plan']
    interface = plan['loop_interface']
    specialization = dict(input_shapes=schedule['input_shapes'], trip_count=schedule['trip_count'],
                          input_layout='C_CONTIGUOUS')
    proof_key = 'original_composite_f4_proof' if composite else 'original_f4_proof'
    if (schedule['selected_kind'] != 'REGISTER_TEMPORAL_FUSION'
            or semantic_sha256(interface.get(proof_key)) != semantic_sha256(asdict(proof))
            or interface.get('source_motion') != work.source_motion
            or interface.get('execution_specialization') != specialization
            or plan['body_sha256'] != schedule['body_sha256']
            or plan['memory_budget_bytes'] != schedule['effective_incremental_budget_bytes']
            or semantic_sha256(execution.get('resource_binding')) != semantic_sha256(schedule['resource_binding'])):
        raise ValueError('C++20 loop lowering differs from the selected proof, specialization or placement')
    consumed = plan['execution_movement_binding']
    if (consumed is None or consumed['source_motion'] != work.source_motion
            or consumed['source_motion_sha256'] != semantic_sha256(motion)):
        raise ValueError('C++20 loop lowering requires the original source movement evidence')
    profile = _OPENMP_LOOP if parallel else _SERIAL_LOOP
    capability = cpp20_capability(profile, operator_abi_version=execution['operator_abi_version'])
    movement = publish_plan(folder / 'movement-binding.json', consumed)
    family = 'COMPOSITE_LOOP' if composite else 'LOOP'
    if plan_only:
        return dict(status=f'MODEL_SELECTED_{family}_PLAN_ONLY', movement_binding=movement,
                    capability=capability)

    native = folder / 'native'
    artifact = emit_temporal_invariant_loop(plan, native)
    if (semantic_sha256(artifact.plan) != semantic_sha256(plan)
            or artifact.report['cxx_standard'] != capability['cxx_standard']
            or artifact.report['operator_descriptor_abi_version'] != capability['operator_abi_version']
            or artifact.report['temporal_control_abi_version'] != capability['temporal_control_abi_version']):
        raise ValueError('compiled loop differs from the admitted lowering or ABI')
    export = export_composite_temporal_invariant if composite else export_temporal_invariant
    exported = export(native, folder / 'bundle', schedule_path=work.schedule['uri'],
                      proof_reference=work.proof, catalog=catalog)
    return dict(
        status=f'MODEL_SELECTED_{family}_COMPILED', native_directory=str(native),
        movement_binding=movement, capability=capability,
        native_artifact=artifact_reference(native / 'artifact.json', artifact.report['schema']),
        bundle=exported['bundle'],
        bundle_export=artifact_reference(folder / 'bundle/export-report.json', exported['schema']),
        actual_target_executed=False,
    )
