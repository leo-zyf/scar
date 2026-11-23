"""Verify selected execution products and deliver C++20 targets."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess

from scar.analysis.replay_transaction import pinned_file_reference, replay_transaction
from scar.artifacts import file_reference
from scar.codegen.native_provider_bundle import _dependencies, cpu_executable_dynamic
from scar.codegen.native_project import (
    build_native_project, publish_cpp20_inputs, render_cpp_template, replay_cpp20_inputs, write_native_project,
)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

from ..contracts import (
    ExecutionPlanRef, ScheduledArrayLoop, ScheduledLoop, cpp20_capability, entry_baseline_only,
    native_compiler_types, publish_plan, read_native_metadata, verify_native_binding, verify_source_lock,
)
from ..operator_stack import (
    load_operator_stack, nvidia_stack_build, verify_operator_stack_linkage, verify_stack_components_once,
)
from ..qualification import executable_receipt


_NATIVE_ENTRY = 'CPP20_NATIVE_ENTRY'
_SERIAL_LOOP = 'CPP_SERIAL_TEMPORAL'
_OPENMP_LOOP = 'CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL'
_ENTRY_TARGET = 'scar_entry'
_ENTRY_BINARY = Path('bin') / _ENTRY_TARGET
_NATIVE_LOWERING = 'scar.native-cpp20-lowering'
_PYTHON_LOWERING = 'scar.python-cpp20-lowering'
_PYTHON_PROFILES = {'CPP20_INVARIANT_LOOP_SERIAL_V1': (_SERIAL_LOOP, 'INVARIANT_LOOP'),
                    'CPP20_INVARIANT_LOOP_OPENMP_V1': (_OPENMP_LOOP, 'INVARIANT_LOOP'),
                    'CPP20_COMPOSITE_INVARIANT_LOOP_SERIAL_V1': (_SERIAL_LOOP, 'COMPOSITE_LOOP'),
                    'CPP20_COMPOSITE_INVARIANT_LOOP_OPENMP_V1': (_OPENMP_LOOP, 'COMPOSITE_LOOP'),
                    'CPP20_ARRAY_LOOP_TEMPORAL_FUSION_V1': ('CPP_SYNCHRONOUS_CPU_TEMPORAL_LOOP', 'ARRAY_LOOP')}


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
        compiler = pinned_file_reference(build.pop('compiler')[0], 'scar.cpp20-compiler')
        inputs = {key: tuple(map(str, values)) for key, values in build.items()}
        return Cpp20Toolchain(compiler, reference, route, inputs, document)
    if route != 'CPU':
        raise ValueError('accelerator builds require a locked operator stack')
    path = shutil.which('g++')
    if path is None:
        raise ValueError('g++ is required for the default CPU toolchain')
    compiler = pinned_file_reference(Path(path).resolve(strict=True), 'scar.cpp20-compiler')
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
    manifest: dict
    observation: dict
    delivery: dict
    dynamic: dict
    artifacts: dict


def deliver(plan: ExecutionPlanRef | ScheduledLoop | ScheduledArrayLoop, project: Path | None = None, *,
            catalog=None, plan_only: bool = False) -> NativeProject | dict:
    """Lower a verified entry or region plan through the C++20 backend."""
    if isinstance(plan, ExecutionPlanRef) and project is not None and not plan_only:
        project = Path(project).resolve()
        if plan.lowering is None:
            raise ValueError('delivery requires a version 2 compiler plan; replay older plans with their producer')
        with replay_transaction():
            plan.read_graph_binding()
            lowering = read_artifact(plan.lowering)
            if (type(lowering.get('schema_version')) is not int or lowering['schema_version'] != 1
                    or lowering.get('graph') != plan.graph):
                raise ValueError('C++20 lowering differs from its sealed graph')
            if lowering['schema'] == _NATIVE_LOWERING:
                return _emit_native(plan, project, admission=_admit_native_lowering(plan, lowering))
            if lowering['schema'] == _PYTHON_LOWERING:
                return _emit_python(plan, project, lowering)
            raise ValueError('unsupported sealed C++20 lowering profile')
    if isinstance(plan, ScheduledLoop) and project is None and catalog is not None:
        return _lower_loop(plan, catalog=catalog, plan_only=plan_only)
    if isinstance(plan, ScheduledArrayLoop) and project is None and catalog is not None:
        return _lower_array_loop(plan, catalog=catalog, plan_only=plan_only)
    raise ValueError('C++20 delivery requires a project path for an entry or a source catalog for a loop')


def _emit_native(schedule: ExecutionPlanRef, project: Path, *, admission) -> NativeProject:
    """Emit the admitted C++20 entry and seal its linked output."""
    from ..frontend.clang import verify_mixed_linkage

    artifacts = schedule.artifacts.copy()
    manifest, observation, delivery = admission
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
    return NativeProject(executable, receipt, manifest, observation, delivery, dynamic, artifacts)


def _read_selection(reference):
    from scar.ir.implementation_plan import problem_from_dict

    problem, schedule = reference.read_pair()
    problem = problem_from_dict(problem)
    if asdict(problem.target) != reference.artifacts['hardware']:
        raise ValueError('selected execution plan differs from the original hardware graph')
    return problem, schedule


def _admit_native_lowering(reference, lowering):
    from scar.optimization.joint_planner import verify_execution_plan

    artifacts = reference.artifacts
    if (set(lowering) != {'schema', 'schema_version', 'graph', 'plan', 'problem', 'delivery'}
            or lowering['plan'] != artifacts['execution_plan']
            or lowering['problem'] != artifacts['planning_problem']):
        raise ValueError('native lowering requires the original planning and graph receipts')
    problem, schedule = _read_selection(reference)
    if type(schedule.get('baseline_only')) is not bool or entry_baseline_only(problem) and not schedule['baseline_only']:
        raise ValueError('execution plan does not honor the planning problem baseline scope')
    verification = verify_execution_plan(problem, schedule)
    observation, binding = verify_native_binding(reference, problem, schedule, verification)

    manifest = read_native_metadata(artifacts['request'])
    verify_source_lock(manifest)
    evidence = read_native_metadata(artifacts['graph_evidence'])
    if (evidence['entry'] != reference.entry
            or any(evidence[name] != artifacts[name] for name in ('request', 'program', 'hardware', 'calls'))):
        raise ValueError('native lowering graph evidence differs from the selected entry')
    ledger = read_artifact(artifacts['rewrites'])
    if (ledger['schema_version'] != 1 or ledger['program'] != artifacts['program']
            or ledger['result'] != artifacts['simplified']):
        raise ValueError('native lowering is not bound to the original rewrite ledger')
    operations = binding.pop('rewrite_operations')
    edits = ledger.get('source_edits', []) if operations else []
    if operations != [edit['operation'] for edit in edits]:
        raise ValueError('selected source edits differ from the sealed rewrite ledger')
    effects = ledger.get('entry_effects')
    if binding.get('entry_effect_scope') != (effects['status'] if effects else None):
        raise ValueError('native lowering differs from its sealed entry effect scope')
    expected = dict(schema='scar.native-delivery', schema_version=1, plan=artifacts['execution_plan'],
                    problem=artifacts['planning_problem'], request=artifacts['request'], source_edits=edits,
                    compiler_types=native_compiler_types(manifest, evidence),
                    capability=cpp20_capability(_NATIVE_ENTRY,
                        operator_abi_version=manifest.get('operator_abi_version', 1)), **binding)
    if lowering['delivery'] != expected:
        raise ValueError('sealed native lowering differs from the selected executable contract')
    return manifest, observation, expected


def _prepared_python_loop(reference, row, option):
    from scar.compiler.temporal_invariant_loop import uses_openmp

    if (set(row) != {'operation', 'option', 'profile', 'loop_id', 'proof', 'source_motion', 'schedule', 'lowering'}
            or row['option'] != 'REGISTER_TEMPORAL_FUSION' or row['profile'] not in _PYTHON_PROFILES):
        raise ValueError('unsupported selected Python C++20 region profile')
    family, kind = _PYTHON_PROFILES[row['profile']]
    composite = kind == 'COMPOSITE_LOOP'
    prepared = read_artifact(row['lowering'])
    expected = dict(schema='scar.cpp20-loop-lowering', schema_version=1, loop_id=row['loop_id'],
        proof=row['proof'], source_motion=row['source_motion'], schedule=row['schedule'],
        parallel=family == _OPENMP_LOOP)
    if {key: value for key, value in prepared.items() if key != 'plan'} != expected:
        raise ValueError('sealed Python lowering differs from the selected loop descriptor')
    plan = prepared['plan']
    schedule = read_artifact(row['schedule'])
    motion = read_artifact(row['source_motion'])
    interface = plan['loop_interface']
    proof = read_artifact(row['proof'])
    proof_schema = 'scar.composite-invariant-loop-proof' if composite else 'scar.invariant-loop-proof'
    if proof.pop('schema') != proof_schema or proof.pop('schema_version') != 1:
        raise ValueError('sealed Python lowering requires its original loop proof profile')
    compositions = tuple(proof['compositions']) if composite else ()
    if composite:
        retained = interface['original_composite_f4_proof']
        if (not compositions or proof['body_identity'] != 'DERIVED_FUSED_DAG_NOT_A_NEW_PYTHON_CALLABLE'
                or proof['transformation_authorized'] is not False
                or motion['proof_sha256'] != semantic_sha256(retained)):
            raise ValueError('composite proof must retain its original derived body and routes')
        for shard, material in zip(compositions, retained['compositions'], strict=True):
            item = read_artifact(shard)
            if (item['schema'] != 'scar.composite-loop-composition' or item['schema_version'] != 1
                    or item['fusion_certificate'] != material['fusion_certificate']
                    or item['fusion_certificate']['pipeline_sha256'] != semantic_sha256(item['pipeline'])):
                raise ValueError('sealed composite lowering differs from the original composition receipts')
        proof['compositions'] = retained['compositions']
    binding = schedule['resource_binding']
    consumed = plan['execution_movement_binding']
    specialization = dict(input_shapes=schedule['input_shapes'], trip_count=schedule['trip_count'],
                          input_layout='C_CONTIGUOUS')
    hardware = reference.artifacts['hardware']
    proof_key = 'original_composite_f4_proof' if composite else 'original_f4_proof'
    if (uses_openmp(plan) != expected['parallel'] or proof['id'] != row['loop_id']
            or schedule['selected_kind'] != 'REGISTER_TEMPORAL_FUSION'
            or binding['target_sha256'] != hardware['sha256']
            or Path(binding['target_path']) != Path(hardware['uri']).parent
            or option.demands != {f'cpu:{cpu}': 1 for cpu in binding['cpu_ids']}
            or semantic_sha256(binding) != semantic_sha256(plan['execution_plan']['resource_binding'])
            or plan['body_sha256'] != schedule['body_sha256']
            or plan['memory_budget_bytes'] != schedule['effective_incremental_budget_bytes']
            or semantic_sha256(interface.get(proof_key)) != semantic_sha256(proof)
            or interface.get('execution_specialization') != specialization
            or interface.get('source_motion') != row['source_motion'] or motion['proof'] != row['proof']
            or consumed is None or consumed['source_motion'] != row['source_motion']
            or consumed['source_motion_sha256'] != semantic_sha256(motion)):
        raise ValueError('prepared Python loop differs from its original evidence or selected resource scope')
    rows, columns = schedule['input_shapes'][plan['carry_parameter']]
    if list(option.workspace_bytes.values()) != [rows * columns * 8]:
        raise ValueError('selected Python loop workspace does not cover the emitted final array')
    return plan, schedule, proof, compositions


def _emit_python(reference, project, lowering):
    from scar.compiler.invariant_export import publish_prepared_temporal_export
    from scar.compiler.temporal_invariant_loop import emit_prepared_temporal_invariant_loop
    from scar.optimization.joint_planner import verify_structural_trial_plan

    if (reference.scope != 'PYTHON_STRUCTURAL_TRIAL_ONLY'
            or reference.decision.get('admitted') is not False
            or reference.decision.get('default_backend') != 'NO_OP'
            or reference.decision.get('timing_qualified') is not False
            or reference.decision.get('deployment_authorized') is not False
            or set(lowering) != {'schema', 'schema_version', 'graph', 'operations'}
            or type(lowering['operations']) is not dict):
        raise ValueError('Python C++20 delivery requires an unpriced structural-trial plan')
    problem, schedule = _read_selection(reference)
    verification = verify_structural_trial_plan(problem, schedule)
    selections = schedule['selected']
    if (set(lowering['operations']) != {item['operation'] for item in selections}
            or reference.decision['trial_only'] is not bool(selections)):
        raise ValueError('sealed Python lowering does not cover exactly the selected region')
    if not selections:
        return dict(delivery='KEEP_ORIGINAL', artifacts={}, validation=[], verification=verification,
                    deployment_authorized=False)

    selected = selections[0]
    region = problem.operations[0]
    option = next(item for item in region.options if item.id == selected['option'])
    row = lowering['operations'][selected['operation']]
    if type(row) is not dict or row.get('profile') not in _PYTHON_PROFILES:
        raise ValueError('selected Python lowering requires a typed descriptor')
    family, kind = _PYTHON_PROFILES[row['profile']]
    lowering_kind = 'array_loop_temporal_fusion' if kind == 'ARRAY_LOOP' else 'register_temporal_fusion'
    if (option.lowering != lowering_kind or row != option.parameters.get('delivery')
            or (row['operation'], row['option']) != (selected['operation'], selected['option'])):
        raise ValueError('sealed Python delivery differs from the selected implementation option')
    graph = reference.artifacts
    evidence = read_artifact(graph['graph_evidence'])
    if (evidence['schema'] != 'scar.python-graph-evidence' or evidence['target_executed'] is not False
            or any(evidence[key] != graph[key] for key in ('program', 'hardware', 'request', 'calls'))
            or not any(item['region'] == selected['operation'] and item['kind'] == kind
                and item['proof'] == row['proof'] and item.get('source_motion') == row['source_motion']
                for item in evidence['regions'])):
        raise ValueError('selected Python descriptor is absent from the original graph evidence')
    if kind == 'ARRAY_LOOP':
        return _emit_array_trial(reference, project, row, option, verification)
    plan, local_schedule, proof, compositions = _prepared_python_loop(reference, row, option)
    capability = cpp20_capability(family,
                                 operator_abi_version=plan['execution_plan']['operator_abi_version'])
    artifact = emit_prepared_temporal_invariant_loop(plan, project / 'native')
    if (artifact.report['cxx_standard'] != capability['cxx_standard']
            or artifact.report['operator_descriptor_abi_version'] != capability['operator_abi_version']
            or artifact.report['temporal_control_abi_version'] != capability['temporal_control_abi_version']):
        raise ValueError('emitted temporal operator has a different C++20 ABI')
    bundle = project / 'bundle'
    exported = publish_prepared_temporal_export(artifact, bundle, schedule=local_schedule,
        schedule_reference=row['schedule'], proof=proof, proof_reference=row['proof'], compositions=compositions)
    invocation = dict(input_order=proof['body']['parameters'], input_shapes=local_schedule['input_shapes'],
                      trip_count=local_schedule['trip_count'])
    return _publish_python_delivery(reference, project, row, artifact, exported, capability, invocation,
                                    verification, library_name='temporal.so')


def _emit_array_trial(reference, project, row, option, verification):
    from scar.codegen.loop import build, describe_capability
    from scar.compiler.invariant_loop_runtime import export_loop
    from scar.compiler.loop_admission import loop_from_dict
    from scar.compiler.model import plan_from_dict

    prepared = read_artifact(row['lowering'])
    if (set(row) != {'operation', 'option', 'profile', 'loop_id', 'proof', 'source_motion', 'schedule', 'lowering'}
            or row['option'] != 'ARRAY_LOOP_TEMPORAL_FUSION' or row['source_motion'] is not None
            or set(prepared) != {'schema', 'schema_version', 'loop_id', 'proof', 'strategy', 'schedule', 'hardware'}
            or prepared['schema'] != 'scar.cpp20-array-loop-lowering' or prepared['schema_version'] != 1
            or any(row[key] != prepared[key] for key in ('loop_id', 'proof', 'schedule'))
            or prepared['hardware'] != reference.artifacts['hardware']):
        raise ValueError('sealed array descriptor differs from the selected source and target')
    loop = loop_from_dict(read_artifact(row['proof']))
    execution = plan_from_dict(read_artifact(row['schedule']))
    strategy = read_artifact(prepared['strategy'])
    binding = execution.resource_binding
    hardware = prepared['hardware']
    if (loop.id != row['loop_id'] or binding is None or binding.target_sha256 != hardware['sha256']
            or Path(binding.target_path) != Path(hardware['uri']).parent
            or option.demands != {f'cpu:{cpu}': 1 for cpu in binding.cpu_ids}
            or list(option.workspace_bytes.values()) != [strategy['final_bytes']]):
        raise ValueError('selected array lowering has different source, CPU or final-array authority')
    artifact = build(loop, project / 'native', strategy=strategy, execution_plan=execution)
    exported = export_loop(artifact.directory, project / 'bundle', source_proof=row['proof'],
                           execution_plan=row['schedule'])
    invocation = dict(input_order=[loop.state_parameter],
        input_shapes={loop.state_parameter: artifact.report['state_shape']}, trip_count=artifact.report['trip_count'])
    return _publish_python_delivery(reference, project, row, artifact, exported, describe_capability(execution),
        invocation, verification, library_name=artifact.report['files']['library'],
        array_contract=artifact.report['contract_sha256'])


def _publish_python_delivery(reference, project, row, artifact, exported, capability, invocation,
                             verification, *, library_name, array_contract=None):
    bundle = project / 'bundle'
    library = pinned_file_reference(bundle / library_name, 'scar.native-temporal-library')
    if library['sha256'] != exported['library_sha256']:
        raise ValueError('prepared temporal bundle changed before native validation')
    executable, inputs, validation = _build_temporal_entry(project, row, invocation, library_name, array_contract)
    artifacts = dict(native_artifact=artifact_reference(artifact.directory / 'artifact.json', artifact.report['schema']),
        bundle_export=artifact_reference(bundle / 'export-report.json', exported['schema']),
        native_library=library, cpp20_inputs=inputs)
    result = dict(schema='scar.python-cpp20-delivery', schema_version=1, graph=reference.graph,
        lowering=reference.lowering, operation=row['operation'], capability=capability,
        artifacts=artifacts.copy(), bundle=str(bundle), executable=executable_receipt(executable),
        project=str(project), verification=verification, validation=[validation],
        invocation=dict(invocation, arguments=['INPUT.bin', 'OUTPUT.bin'], representation='LITTLE_ENDIAN_FLOAT64_C_CONTIGUOUS'),
        delivery='TRIAL_COMPILED', deployment_authorized=False)
    artifacts['delivery'] = publish_plan(project / 'delivery.json', result)
    return dict(result, artifacts=artifacts)


def _build_temporal_entry(project, row, invocation, library_name, array_contract):
    from scar.codegen import __file__ as codegen_path

    shape = invocation['input_shapes'][invocation['input_order'][0]]
    inputs = len(invocation['input_order'])
    template = Path(codegen_path).with_name('temporal_entry.cpp.in')
    source = render_cpp_template(template, dict(zip(
        ('@SCAR_ROWS@', '@SCAR_COLUMNS@', '@SCAR_INPUT_COUNT@', '@SCAR_STEPS@',
         '@SCAR_ARRAY_LOOP@', '@SCAR_LIBRARY@', '@SCAR_CONTRACT_SHA256@'),
        (*shape, inputs, invocation['trip_count'], int(array_contract is not None), library_name, array_contract or ''))))
    (project / 'src').mkdir()
    (project / 'src/main.cpp').write_bytes(source)
    shutil.copy2(project / 'native/scar_abi_v1.h', project / 'src/scar_abi_v1.h')
    if array_contract is not None:
        shutil.copy2(project / 'native/loop_abi_v1.h', project / 'src/loop_abi_v1.h')
    configuration = dict(sources=['src/main.cpp'], libraries=['dl'], install_dirs=[['bundle', '.']])
    receipt = publish_cpp20_inputs(project, configuration, plan=row['lowering'])
    executable, _ = build_cpp20_project(project, receipt, plan=row['lowering'])
    output = project / 'validation/native-smoke.bin'
    try:
        report = subprocess.run([str(executable), '--smoke', str(output)], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        validation = dict(status='TIMED_OUT', native_abi_passed=False)
    else:
        ok = report.returncode == 0 and output.is_file() and output.stat().st_size == shape[0] * shape[1] * 8
        status = 'PASSED' if ok else 'NATIVE_ABI_VALIDATION_FAILED'
        if 1 <= report.returncode <= (6 if array_contract is not None else 4):
            status = 'NATIVE_ABI_GUARD_REFUSED'
        validation = dict(status=status, native_abi_passed=ok,
            returncode=report.returncode, stderr=report.stderr.decode(errors='replace'))
        if ok:
            validation['output'] = pinned_file_reference(output, 'scar.temporal-validation-output')
    validation.update(profile='CXX20_TEMPORAL_ABI_SMOKE', original_program_executed=False,
                      numerical_equivalence_qualified=False)
    return executable, receipt, validation


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
    configuration = replay_cpp20_inputs(project, inputs, plan=plan)['configuration']
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
    if pinned_file_reference(compiler['uri'], compiler['schema']) != compiler:
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
    binary = pinned_file_reference(executable, 'scar.cpp20-executable')
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
    if pinned_file_reference(original, 'scar.cpp20-executable') != binary:
        raise ValueError('selected executable differs from the calibrated binary')
    before = replay_cpp20_inputs(staged, variant['inputs'], plan=variant['plan'])
    build = before['configuration']
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
    after = read_artifact(inputs)
    if ({name: dict(row, uri=str(project / name)) for name, row in before['files'].items()} != after['files']
            or before['libraries'] != after['libraries']):
        raise ValueError('delivered project inputs differ from the calibrated variant')
    delivered = pinned_file_reference(executable, binary['schema'])
    if delivered != dict(binary, uri=str(executable)):
        raise ValueError('published executable differs from the calibrated binary')
    _verify_compiler(toolchain)
    return executable, inputs, _verify_linkage(executable, toolchain)


def _lower_loop(work: ScheduledLoop, *, catalog, plan_only: bool) -> dict:
    from scar.compiler.invariant_export import publish_temporal_export, read_temporal_export_source
    from scar.compiler.temporal_invariant_loop import emit_temporal_invariant_loop, uses_openmp

    folder, composite, parallel = work.folder, work.composite, work.parallel
    expected = dict(schema='scar.cpp20-loop-lowering', schema_version=1,
                    loop_id=work.loop_id, proof=work.proof, source_motion=work.source_motion,
                    schedule=work.schedule, parallel=parallel)
    prepared = read_artifact(work.lowering)
    if {key: value for key, value in prepared.items() if key != 'plan'} != expected:
        raise ValueError('loop lowering does not bind the scheduled source and resources')
    schedule = read_artifact(work.schedule)
    source = read_temporal_export_source(work.proof, catalog=catalog, composite=composite)
    proof, motion, _ = source
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
    schedule_receipt = dict(path=str(Path(work.schedule['uri']).resolve()), sha256=work.schedule['sha256'],
                           bytes=work.schedule['size_bytes'])
    exported = publish_temporal_export(artifact, folder / 'bundle', schedule=schedule,
        schedule_receipt=schedule_receipt, proof_reference=work.proof, source=source,
        catalog=catalog, composite=composite)
    return dict(
        status=f'MODEL_SELECTED_{family}_COMPILED', native_directory=str(native),
        movement_binding=movement, capability=capability,
        native_artifact=artifact_reference(native / 'artifact.json', artifact.report['schema']),
        bundle=exported['bundle'],
        bundle_export=artifact_reference(folder / 'bundle/export-report.json', exported['schema']),
        actual_target_executed=False,
    )


def _lower_array_loop(work: ScheduledArrayLoop, *, catalog, plan_only: bool) -> dict:
    from scar.codegen.loop import build, describe_capability
    from scar.compiler.invariant_loop_runtime import export_loop
    from scar.compiler.loop_admission import admit_loop, read_array_loop_proof
    from scar.compiler.model import plan_from_dict

    loop = read_array_loop_proof(work.proof, catalog=catalog)
    strategy = read_artifact(work.strategy)
    execution = plan_from_dict(read_artifact(work.schedule))
    binding = execution.resource_binding
    target = Path(work.hardware['path']).parent
    if loop.id != work.loop_id or binding is None or (binding.target_path, binding.target_sha256) != (
            str(target), work.hardware['sha256']):
        raise ValueError('array loop lowering differs from the admitted source or hardware placement')
    if plan_only:
        admit_loop(loop, strategy, execution)
        return dict(status='TEMPORAL_FUSION_PLAN_ONLY', capability=describe_capability(execution),
                    actual_target_executed=False)

    native = work.folder / 'native'
    artifact = build(loop, native, strategy=strategy, execution_plan=execution)
    report = artifact.report
    bundle = work.folder / 'bundle'
    exported = export_loop(native, bundle, source_proof=work.proof, execution_plan=work.schedule)
    return dict(status='TEMPORAL_FUSION_COMPILED', native_directory=str(native),
                native_artifact=artifact_reference(native / 'artifact.json', report['schema']),
                bundle=exported['bundle'],
                bundle_export=artifact_reference(bundle / 'export-report.json', exported['schema']),
                actual_target_executed=False)
