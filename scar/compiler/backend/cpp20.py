"""Emit one C++20 project from a locked native source revision."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from scar.artifacts import file_reference
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, problem_document, problem_from_dict
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA, loads
from scar.optimization.joint_planner import verify_execution_plan

from scar.compiler.model import CpuPlacement
from scar.optimization.resource_solver import semantic_sha256

from ..middle.plan import copy_source_edits, entry_cpus, entry_problem, publish_plan


def admit_entry(artifacts, entry):
    """Replay the artifact and execution contracts before emitting source."""
    schemas = {
        'request': 'scar.native-entry', 'program': PROGRAM_SCHEMA,
        'graph_evidence': 'scar.native-graph-evidence', 'calls': 'scar.native-call-links',
        'hardware': 'scar.target-hardware-graph', 'simplified': PROGRAM_SCHEMA,
        'rewrites': 'scar.native-rewrites', 'cost': 'scar.native-entry-cost',
        'planning_problem': PROBLEM_SCHEMA, 'execution_plan': PLAN_SCHEMA,
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
    observation = documents['cost']
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
    problem = problem_from_dict(documents['planning_problem'])
    if (observation['schema_version'] != 1 or observation['request'] != artifacts['request']
            or observation['exit'] != 0 or problem_document(entry_problem(entry, artifacts, observation, edits))
            != documents['planning_problem']):
        raise ValueError('native problem differs from the admitted inputs')
    plan = documents['execution_plan']
    verification = verify_execution_plan(problem, plan)
    option = problem.operations[0].options[0]
    if (not plan['baseline_only'] or plan['transfers'] or plan.get('activities')
            or len(plan['selected']) != 1 or plan['selected'][0]['option'] != option.id):
        raise ValueError('backend cannot lower the selected native schedule')
    cpu = int(next(iter(option.demands)).removeprefix('cpu:'))
    if cpu not in entry_cpus(artifacts['hardware']):
        raise ValueError('scheduled CPU is unavailable')
    root = Path(manifest['source_root'])
    for name, lock in manifest['source_lock']['files'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f'locked source outside the source root: {name}')
        ref = file_reference(path, 'scar.native-source')
        if any(ref[key] != lock[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source changed: {name}')
    delivery = dict(schema='scar.native-delivery', schema_version=1, cpus=[cpu],
                    plan=artifacts['execution_plan'], problem=artifacts['planning_problem'],
                    request=artifacts['request'], selected=plan['selected'],
                    source_edits=edits, compiler_types=compiler_types,
                    lowering=option.lowering, verification=verification)
    return manifest, observation, delivery


def build_cpp20_project(project, build):
    """Build source-adapter output with one target language and verify its ABI."""
    if not build['sources'] or any(Path(path).suffix not in {'.cpp', '.cc', '.cxx'}
                                   for path in build['sources']):
        raise ValueError('the backend requires staged C++ translation units')
    write_native_project(project, 'scar_entry', **build)
    executable, _ = build_native_project(project, 'scar_entry', ('g++',))
    dynamic = cpu_executable_dynamic(executable)
    for name in dynamic['needed']:
        if any(provider in name for provider in ('libpython', 'libtorch', 'libgfortran')):
            raise ValueError(f'unqualified target dependency: {name}')
    return executable, dynamic


def lower_loop(proof, spec, row, folder, *, catalog, reader, composite, parallel, plan_only):
    from scar.compiler.invariant_export import export_temporal_invariant, export_composite_temporal_invariant

    # Reopen the exact published schedule and source proof before codegen.
    schedule = read_artifact(row['schedule'])
    proof_again, motion = reader(spec['proof']['uri'], catalog=catalog)
    if (semantic_sha256(asdict(proof_again)) != semantic_sha256(asdict(proof))
            or semantic_sha256(motion) != semantic_sha256(read_artifact(row['source_motion']))):
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
                                    source_motion=row['source_motion'])
    else:
        interface = dict(
            original_f4_proof=asdict(proof), carry_parameter=proof.carry_parameter,
            readonly_parameters=[name for name in proof.body.parameters if name != proof.carry_parameter],
            authority='ORIGINAL_F4_REPLAYED_WITH_RESOURCE_SCHEDULE',
            execution_specialization=specialization, source_motion=row['source_motion'],
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
    row['movement_binding'] = publish_plan(folder / 'movement-binding.json', consumed)
    family = 'COMPOSITE_LOOP' if composite else 'LOOP'
    if plan_only:
        row['status'] = f'MODEL_SELECTED_{family}_PLAN_ONLY'
        return

    native = folder / 'native'
    if composite:
        from scar.compiler.composite_invariant_loop import build_composite_invariant_loop
        artifact = build_composite_invariant_loop(
            proof, native, catalog=catalog, memory_budget_bytes=budget, resource_binding=placement,
            execution_specialization=specialization, parallel=True if parallel else None,
            source_motion=row['source_motion'],
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
    exported = export(native, folder / 'bundle', schedule_path=row['schedule']['uri'],
                      proof_reference=spec['proof'], catalog=catalog)
    row.update(
        status=f'MODEL_SELECTED_{family}_COMPILED', native_directory=str(native),
        native_artifact=artifact_reference(native / 'artifact.json', artifact.report['schema']),
        bundle=exported['bundle'],
        bundle_export=artifact_reference(folder / 'bundle/export-report.json', exported['schema']),
        actual_target_executed=False,
    )
