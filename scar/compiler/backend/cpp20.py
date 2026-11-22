"""Emit one C++20 project from a locked native source revision."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import shutil

from scar.artifacts import file_reference
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import (
    build_native_project, publish_cpp20_inputs, replay_cpp20_inputs, write_native_project,
)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, problem_document, problem_from_dict
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA, loads
from scar.optimization.joint_planner import verify_execution_plan

from scar.compiler.model import CpuPlacement
from scar.optimization.resource_solver import semantic_sha256

from ..middle.plan import NativeSchedule, copy_source_edits, entry_baseline_only, entry_cpus, entry_problem, publish_plan
from ..middle.python import ScheduledLoop
from ..qualification import executable_receipt


@dataclass(frozen=True, slots=True)
class NativeProject:
    executable: Path
    receipt: dict
    observation: dict
    delivery: dict
    dynamic: dict
    artifacts: dict


def emit_native(schedule: NativeSchedule, project: Path) -> NativeProject:
    """Replay the schedule, emit its C++20 project and seal the linked output."""
    from ..frontend.clang import stage_source_revision, verify_mixed_linkage

    artifacts = schedule.artifacts.copy()
    manifest, observation, delivery = admit_entry(artifacts, schedule.entry)
    if 'trial_variant' in delivery:
        executable, inputs, dynamic = promote_cpp20_variant(project, artifacts, delivery)
    else:
        inputs = stage_source_revision(manifest, project, delivery['source_edits'],
                                       captures=delivery['compiler_types'], plan=delivery['plan'])
        executable, dynamic = build_cpp20_project(project, inputs, plan=delivery['plan'])
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
    schemas = {
        'request': 'scar.native-entry', 'program': PROGRAM_SCHEMA,
        'graph_evidence': 'scar.native-graph-evidence', 'calls': 'scar.native-call-links',
        'hardware': 'scar.target-hardware-graph', 'simplified': PROGRAM_SCHEMA,
        'rewrites': 'scar.native-rewrites',
    }
    documents = {}
    for role, schema in schemas.items():
        ref = artifacts[role]
        if ref['schema'] != schema:
            raise ValueError(f'invalid {role} artifact schema')
        if role in {'program', 'graph_evidence', 'calls', 'simplified'}:
            if file_reference(ref['uri'], schema) != ref:
                raise ValueError(f'{role} artifact differs from the plan input')
        else:
            documents[role] = read_artifact(ref)
    manifest = documents['request']
    rewrite = documents['rewrites']
    if (rewrite['schema_version'] != 1 or rewrite['program'] != artifacts['program']
            or rewrite['result'] != artifacts['simplified']):
        raise ValueError('rewrite ledger does not bind the admitted input and output graphs')
    edits = rewrite.get('source_edits', [])
    typed_units = {unit['path'] for unit in manifest['units']
                   if unit.get('language', manifest['language']) in ('c', 'fortran')}
    evidence = read_artifact(artifacts['graph_evidence']) if edits or typed_units else None
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
    root = Path(manifest['source_root'])
    for name, lock in manifest['source_lock']['files'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f'locked source outside the source root: {name}')
        ref = file_reference(path, 'scar.native-source')
        if any(ref[key] != lock[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source changed: {name}')
    return manifest, edits, compiler_types


def admit_entry(artifacts, entry):
    """Reconstruct the cost domain and verify the selected implementation."""
    manifest, edits, compiler_types = replay_native_revision(artifacts)
    schemas = {'cost': 'scar.native-entry-cost', 'planning_problem': PROBLEM_SCHEMA, 'execution_plan': PLAN_SCHEMA}
    documents = {}
    for role, schema in schemas.items():
        ref = artifacts[role]
        if ref['schema'] != schema:
            raise ValueError(f'invalid {role} artifact schema')
        documents[role] = read_artifact(ref)
    observation = documents['cost']
    problem = entry_problem(entry, artifacts, observation, edits)
    if (observation['request'] != artifacts['request'] or observation['exit'] != 0
            or problem_document(problem) != documents['planning_problem']):
        raise ValueError('native problem differs from the admitted inputs')
    plan = documents['execution_plan']
    verification = verify_execution_plan(problem, plan)
    operation = problem.operations[0]
    baseline_only = entry_baseline_only(problem)
    eligible = (option for option in operation.options
                if not baseline_only or option.id == operation.baseline_option)
    option = min(eligible, key=lambda item: (item.cost.estimate_ns, item.id != operation.baseline_option))
    if (plan['baseline_only'] != baseline_only or plan['transfers'] or plan.get('activities')
            or len(plan['selected']) != 1 or plan['selected'][0]['option'] != option.id):
        raise ValueError('backend cannot lower the selected native schedule')
    cpu = int(next(iter(option.demands)).removeprefix('cpu:'))
    if cpu not in entry_cpus(artifacts['hardware']):
        raise ValueError('scheduled CPU is unavailable')
    selected_edits = edits if option.id == 'canonical' else []
    delivery = dict(schema='scar.native-delivery', schema_version=1, cpus=[cpu],
                    plan=artifacts['execution_plan'], problem=artifacts['planning_problem'],
                    request=artifacts['request'], selected=plan['selected'],
                    source_edits=selected_edits, compiler_types=compiler_types,
                    lowering=option.lowering, verification=verification)
    if observation['schema_version'] == 2:
        delivery['trial_variant'] = observation['variants'][option.id]
    return manifest, observation, delivery


def build_cpp20_project(project, inputs, *, plan):
    """Build source-adapter output with one target language and verify its ABI."""
    configuration = replay_cpp20_inputs(project, inputs, plan=plan)
    write_native_project(project, 'scar_entry', **configuration)
    executable, _ = build_native_project(project, 'scar_entry', ('g++',))
    replay_cpp20_inputs(project, inputs, plan=plan)
    dynamic = cpu_executable_dynamic(executable)
    for name in dynamic['needed']:
        if any(provider in name for provider in ('libpython', 'libtorch', 'libgfortran')):
            raise ValueError(f'unqualified target dependency: {name}')
    return executable, dynamic


def build_trial_variant(directory, artifacts, manifest, edits, compiler_types, *, mode):
    from ..frontend.clang import stage_source_revision

    directory.mkdir(parents=True)
    plan = publish_plan(directory / 'source-variant.json', dict(
        schema='scar.native-source-variant', schema_version=1,
        request=artifacts['request'], rewrites=artifacts['rewrites'], mode=mode, source_edits=edits,
    ))
    project = directory / 'project'
    inputs = stage_source_revision(manifest, project, edits, captures=compiler_types, plan=plan)
    executable, _ = build_cpp20_project(project, inputs, plan=plan)
    binary = file_reference(executable, 'scar.cpp20-executable')
    variant = publish_plan(directory / 'result.json', dict(
        schema='scar.native-compiled-variant', schema_version=1, plan=plan, inputs=inputs, binary=binary,
    ))
    return executable, variant


def promote_cpp20_variant(project, artifacts, delivery):
    """Deliver the scheduled executable byte for byte, without rebuilding it."""
    reference = delivery['trial_variant']
    if reference['schema'] != 'scar.native-compiled-variant':
        raise ValueError('selected native trial receipt required')
    variant = read_artifact(reference)
    recipe = read_artifact(variant['plan'])
    mode = delivery['selected'][0]['option']
    if (recipe != dict(schema='scar.native-source-variant', schema_version=1,
                       request=artifacts['request'], rewrites=artifacts['rewrites'],
                       mode=mode, source_edits=delivery['source_edits'])
            or variant['schema_version'] != 1):
        raise ValueError('selected trial belongs to a different source revision')
    staged = Path(reference['uri']).parent / 'project'
    original = staged / 'bin/scar_entry'
    binary = variant['binary']
    if file_reference(original, 'scar.cpp20-executable') != binary:
        raise ValueError('selected executable differs from the calibrated binary')
    build = replay_cpp20_inputs(staged, variant['inputs'], plan=variant['plan'])
    shutil.copytree(staged, project, ignore=shutil.ignore_patterns('cpp20-inputs.json'))
    inputs = publish_cpp20_inputs(project, build, plan=delivery['plan'])
    before = read_artifact(variant['inputs'])['files']
    after = read_artifact(inputs)['files']
    if ({name: (row['sha256'], row['size_bytes']) for name, row in before.items()}
            != {name: (row['sha256'], row['size_bytes']) for name, row in after.items()}):
        raise ValueError('delivered project sources differ from the calibrated variant')
    executable = project / 'bin/scar_entry'
    delivered = file_reference(executable, binary['schema'])
    if delivered != dict(binary, uri=str(executable)):
        raise ValueError('published executable differs from the calibrated binary')
    return executable, inputs, cpu_executable_dynamic(executable)


def lower_loop(work: ScheduledLoop, *, catalog, plan_only) -> dict:
    from scar.compiler.invariant_export import export_temporal_invariant, export_composite_temporal_invariant
    from scar.compiler.loop_admission import read_invariant_loop_motion
    from scar.compiler.pipeline_loop_frontend import read_composite_invariant_loop_motion

    folder, composite, parallel = work.folder, work.composite, work.parallel
    reader = read_composite_invariant_loop_motion if composite else read_invariant_loop_motion
    # Reopen the exact published schedule and source proof before codegen.
    schedule = read_artifact(work.schedule)
    read_artifact(work.proof)
    proof, motion = reader(work.proof['uri'], catalog=catalog)
    if (work.loop_id != proof.id
            or semantic_sha256(motion) != semantic_sha256(read_artifact(work.source_motion))):
        raise ValueError('loop movement changed between original F4 admission and lowering')
    binding = dict(schedule['resource_binding'])
    binding['cpu_ids'] = tuple(binding['cpu_ids'])
    binding['evidence'] = tuple(binding.get('evidence', ()))
    placement = CpuPlacement(**binding)
    budget = schedule['effective_incremental_budget_bytes']
    specialization = dict(input_shapes=schedule['input_shapes'], trip_count=schedule['trip_count'],
                          input_layout='C_CONTIGUOUS')
    if composite:
        from scar.compiler.composite_invariant_loop import _loop_interface
        interface = _loop_interface(proof, execution_specialization=specialization,
                                    source_motion=work.source_motion)
    else:
        interface = dict(
            original_f4_proof=asdict(proof), carry_parameter=proof.carry_parameter,
            readonly_parameters=[name for name in proof.body.parameters if name != proof.carry_parameter],
            authority='ORIGINAL_F4_REPLAYED_WITH_RESOURCE_SCHEDULE',
            execution_specialization=specialization, source_motion=work.source_motion,
        )
    if parallel:
        from scar.compiler.parallel_temporal_invariant import (
            _plan as preview, build_parallel_temporal_invariant_loop as build,
        )
    else:
        from scar.compiler.temporal_invariant_loop import _plan as preview, build_temporal_invariant_loop as build
    planned, _ = preview(proof.body, proof.carry_parameter, interface, budget, placement)
    consumed = planned['execution_movement_binding']
    if consumed is None:
        raise ValueError('loop delivery requires consumed source movement legality')
    movement = publish_plan(folder / 'movement-binding.json', consumed)
    family = 'COMPOSITE_LOOP' if composite else 'LOOP'
    if plan_only:
        return dict(status=f'MODEL_SELECTED_{family}_PLAN_ONLY', movement_binding=movement)

    native = folder / 'native'
    if composite:
        from scar.compiler.composite_invariant_loop import build_composite_invariant_loop
        artifact = build_composite_invariant_loop(
            proof, native, catalog=catalog, memory_budget_bytes=budget, resource_binding=placement,
            execution_specialization=specialization, parallel=True if parallel else None,
            source_motion=work.source_motion,
        )
        export = export_composite_temporal_invariant
    else:
        artifact = build(proof.body, native, carry_parameter=proof.carry_parameter,
                         loop_interface=interface, memory_budget_bytes=budget, resource_binding=placement)
        export = export_temporal_invariant
    emitted = artifact.plan
    if (emitted['execution_movement_binding'] != consumed
            or emitted['loop_interface']['execution_specialization'] != specialization
            or emitted['memory_budget_bytes'] != budget
            or semantic_sha256(emitted['execution_plan']['resource_binding']) !=
               semantic_sha256(schedule['resource_binding'])):
        raise ValueError('compiled loop differs from the admitted movement, specialization or placement')
    exported = export(native, folder / 'bundle', schedule_path=work.schedule['uri'],
                      proof_reference=work.proof, catalog=catalog)
    return dict(
        status=f'MODEL_SELECTED_{family}_COMPILED', native_directory=str(native),
        movement_binding=movement,
        native_artifact=artifact_reference(native / 'artifact.json', artifact.report['schema']),
        bundle=exported['bundle'],
        bundle_export=artifact_reference(folder / 'bundle/export-report.json', exported['schema']),
        actual_target_executed=False,
    )
