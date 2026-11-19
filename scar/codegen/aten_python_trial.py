"""Build an opt-in native call bridge for an exact observed Python function."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sysconfig
from tempfile import TemporaryDirectory

from scar.analysis.aten_python_trial_source import qualify_python_trial_source
from scar.analysis.aten_trial_geometry import qualify_trial_geometry
from scar.codegen.aten_native_graph import _read_declared
from scar.codegen.aten_provider import _build_interface, _dependencies
from scar.codegen.native_provider_bundle import _content
from scar.compiler.plan_aten_native_graph import _bundle, _segment_plan, _statistics, _window_model
from scar.compiler.plan_aten_native_team import team_segments, team_statistics
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, problem_document
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.aten_native_team import model_cpu_team
from scar.runtime.aten_provider_identity import file_stat_identity, local_file_identity, verify_library_file


_MODULE = '_scar_aten_python_trial'
_TEMPLATE = Path(__file__).with_name('aten_python_trial_v1.cpp.in')
_PAIR_HEADER = Path(__file__).with_name('aten_native_pair_v1.h')
_TEAM_HEADER = 'aten_native_team_v1.h'


def _file_ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _read_plan(path, digest):
    path, selection, _ = _read_declared(path, digest)
    mode = selection['selected']
    if ((selection['schema'], selection['schema_version'], selection['boundary']) != (
            'scar.aten-native-graph-schedule', 1, 'GENERATED_WARM_ATEN_GRAPH_CALL')
            or mode not in ('TWO_CPU', 'THREE_CPU')):
        raise ValueError('a validated parallel generated graph selection is required')
    bundle = selection['bundle']
    artifact, origin, recipe, hardware, executable, key = _bundle(bundle['uri'], bundle['sha256'])
    if (origin != bundle or key not in ('cpu_windows', 'cpu_team') or 'tensor_lifetime' not in artifact
            or selection['executable'] != dict(path=str(executable), **_content(executable))):
        raise ValueError('selected program does not match the retained graph delivery')
    schedule = artifact[key]
    evidence = selection['observations']
    calibrated = read_artifact(evidence)
    validation = read_artifact(selection['paired_validation'])
    warmup, count = selection['warmup_pairs'], selection['measured_pairs']
    budget = selection['model']['tensor_storage']['budget_bytes']
    if key == 'cpu_team':
        model = model_cpu_team(calibrated['calibration'], recipe, schedule, warmup, tensor_budget_bytes=budget)
        starts = selection['selected_windows']
        stats = team_statistics(validation, schedule, warmup, count, starts, mode)
        profits = stats['profitable_parallel'] and mode == model['selected']
        expected_starts = model['selected_windows']
        problem, _, timeline = team_segments(hardware, schedule, origin, evidence, model, mode)
    else:
        model = _window_model(calibrated['calibration'], recipe, schedule, warmup, observe_storage=True,
                              tensor_budget_bytes=budget)
        stats = _statistics(validation, warmup, count, schedule['cpus'])
        starts = selection['selected_window_starts']
        profits = stats['profitable_two_cpu'] and mode == 'TWO_CPU'
        expected_starts = model['selected_starts']
        problem, _, timeline = _segment_plan(hardware, schedule, origin, evidence, model, starts)
    if (model != selection['model'] or stats != selection['validation'] or not profits
            or not starts or starts != expected_starts or starts != validation['windows']
            or selection['cpu_ids'] != schedule['cpus'] or selection['original_target'] != schedule['target']
            or validation['result_storage_bytes'] != calibrated['calibration']['result_storage_bytes']):
        raise ValueError('retained ATen time, storage or window choice differs from its evidence')
    problem_ref = _file_ref(path.with_name('problem.json'), PROBLEM_SCHEMA)
    execution_ref = _file_ref(path.with_name('plan.json'), PLAN_SCHEMA)
    if problem_document(problem) != read_artifact(problem_ref) or timeline != selection['execution_timeline']:
        raise ValueError('retained segment timeline differs from the verified graph model')
    verify_execution_plan(problem, read_artifact(execution_ref))
    return selection, artifact, recipe, dict(selection=_file_ref(path, selection['schema']),
        planning_problem=problem_ref, execution_plan=execution_ref, calibration=evidence,
        paired_validation=selection['paired_validation'])


def _linked_source(artifact, recipe, projection, proof):
    if ((projection['schema'], projection['schema_version']) != ('scar.aten-capture-graph-projection', 1)
            or projection['recipe_sha256'] != artifact['recipe']['sha256']
            or projection['original_source_decision'] != 'KEEP_ORIGINAL'):
        raise ValueError('exact original source projection of the compiled recipe required')
    occurrences = projection['linkage']['occurrences']
    observed = [(f"call_{event['call_attempt']}", event['schema'], event['call_site']['code'])
                for event in occurrences]
    if observed != [(op['id'], op['schema'], proof['code_key']) for op in recipe['operations']]:
        raise ValueError('original Python CodeID or call order differs from its scoped capture')
    linkage = projection['linkage']['linkage_report']
    if (linkage['uri'], linkage['sha256'], linkage['size_bytes']) != tuple(
            artifact['linkage'][key] for key in ('path', 'sha256', 'size_bytes')):
        raise ValueError('original operator linkage differs from the compiled bundle')


def _library_receipt(path, expected=None):
    with path.open('rb') as member:
        before = file_stat_identity(os.fstat(member.fileno()))
        identity = local_file_identity(member)
        content = verify_library_file(member, expected)
        after_identity = local_file_identity(member)
        after = file_stat_identity(os.fstat(member.fileno()))
        if before != after or before != file_stat_identity(path.stat()):
            raise ValueError('LibTorch library changed while issuing its receipt: ' + str(path))
    receipt = dict(path=str(path), **content)
    if identity is not None and identity == after_identity:
        receipt['local_file_identity'] = identity
    return receipt


def _geometry_header(geometry):
    lines = ['#pragma once', '#include <array>', '#include <cstddef>', '#include <cstdint>', '#include <span>',
             'namespace scar_aten_trial_geometry {', 'struct TensorGeometry {',
             '    std::span<const std::int64_t> sizes;', '    std::span<const std::int64_t> strides;',
             '    std::size_t storage_bytes;', '};']
    for name, count in (('inputs', 'input_count'), ('operations', 'operation_count')):
        records = []
        for index, row in enumerate(geometry[name]):
            fields = []
            for key in ('shape', 'strides'):
                field = f'{name}_{key}_{index}'
                values = ', '.join(map(str, row[key]))
                lines.append(f'inline constexpr std::array<std::int64_t, {len(row[key])}> {field}{{{values}}};')
                fields.append(field)
            records.append('    {' + ', '.join((*fields, str(row['storage_bytes']))) + '},')
        lines.extend((f'inline constexpr std::array<TensorGeometry, scar_aten_graph::{count}> {name}{{{{',
                      *records, '}};'))
    return '\n'.join((*lines, '}  // namespace scar_aten_trial_geometry', ''))


def _build_extension(stage, artifact, torch, *, value_geometry=False):
    _, libraries, cxx, compile_flags, link_flags = _build_interface(torch)
    python_torch = libraries[0].with_name('libtorch_python.so').resolve(strict=True)
    filename = _MODULE + sysconfig.get_config_var('EXT_SUFFIX')
    # CPython API symbols are resolved by the importing interpreter, including static CPython builds.
    extension_links = [flag for flag in link_flags if flag != '-Wl,-z,defs']
    options = ['-DSCAR_ATEN_VALUE_GEOMETRY'] if value_geometry else []
    command = [*cxx, *compile_flags, *options, '-I', sysconfig.get_path('include'), '-O2', '-fPIC', '-shared',
               '-fvisibility=hidden', '-pthread', 'trial.cpp', *extension_links, str(python_torch), '-o', filename]
    env = {name: value for name, value in os.environ.items() if not name.startswith('LD_')}
    subprocess.run(command, cwd=stage, check=True, env=env)
    listing = subprocess.run(['ldd', './' + filename], cwd=stage, check=True, capture_output=True, text=True, env=env).stdout
    dependencies = _dependencies(listing)
    if any(dependencies.get(path.name) != str(path.resolve()) for path in [*libraries, python_torch]):
        raise ValueError('Python trial extension resolves to a different LibTorch installation')
    (stage / 'ldd.txt').write_text(listing)
    original = artifact['provider']['libraries']
    expected = {Path(row['path']): row for row in original}
    if len(expected) != len(original) or expected.keys() != set(libraries):
        raise ValueError('Python trial libraries differ from the verified graph bundle')
    pinned = [_library_receipt(path, expected[path]) for path in libraries]
    pinned.append(_library_receipt(python_torch))
    provider = dict(artifact['provider'], libraries=pinned)
    return dict(name=filename, **_content(stage / filename)), provider, command


def build_aten_python_trial(source, function, *, projection, projection_sha256, selection, selection_sha256,
                           contract, contract_sha256, out, geometry_contract=None, geometry_contract_sha256=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    selected, artifact, recipe, replay = _read_plan(selection, selection_sha256)
    projection_path, captured, _ = _read_declared(projection, projection_sha256)
    contract_path, contract, _ = _read_declared(contract, contract_sha256)
    proof = qualify_python_trial_source(source, function, recipe, contract)
    _linked_source(artifact, recipe, captured, proof)
    import torch

    if (torch.__version__, torch.version.git_version) != (artifact['provider']['version'], artifact['provider']['revision']):
        raise ValueError('Python trial requires the same installed LibTorch provider revision')
    guard = None
    if geometry_contract is not None or geometry_contract_sha256 is not None:
        if geometry_contract is None or geometry_contract_sha256 is None:
            raise ValueError('the value-independent geometry contract requires its exact digest')
        geometry_path, declaration, _ = _read_declared(geometry_contract, geometry_contract_sha256)
        calibration = read_artifact(replay['calibration'])['calibration']
        geometry = qualify_trial_geometry(recipe, declaration, calibration, torch)
        guard = dict(kind='VALUE_INDEPENDENT_GEOMETRY', geometry=geometry,
                     contract=_file_ref(geometry_path, declaration['schema']))
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-python-trial-', dir=out.parent) as tmp:
        stage = Path(tmp)
        bundle = Path(selected['bundle']['uri']).parent
        team = 'cpu_team' in artifact
        for name in ('graph.hpp', 'operators.hpp', *((_TEAM_HEADER,) if team else ())):
            dest = stage / name
            dest.write_bytes((bundle / name).read_bytes())
            if _content(dest) != artifact['files'][name]:
                raise ValueError('generated graph header differs from the exact verified bundle: ' + name)
        if not team:
            (stage / _PAIR_HEADER.name).write_bytes(_PAIR_HEADER.read_bytes())
        (stage / 'trial.cpp').write_bytes(_TEMPLATE.read_bytes())
        if guard is not None:
            (stage / 'trial_geometry.hpp').write_text(_geometry_header(guard['geometry']))
        extension, provider, command = _build_extension(stage, artifact, torch, value_geometry=guard is not None)
        manifest = dict(schema='scar.aten-python-trial', schema_version=1,
            scope='EXPLICIT_PYTHON_CPU_DIFFERENTIAL_TRIAL', source_proof=proof, recipe=recipe, contract=contract,
            cpus=selected['cpu_ids'], inputs=selected['inputs'],
            provider=provider, extension=extension, references=dict(replay,
                source_projection=_file_ref(projection_path, captured['schema']),
                wrapper_contract=_file_ref(contract_path, contract['schema'])),
            compiler_command=command, cxx_standard=20, source_replacement_authorized=False,
            python_boundary_performance_measured=False)
        if team:
            manifest['window_groups'] = selected['selected_windows']
        else:
            manifest['window_starts'] = selected['selected_window_starts']
        if guard is not None:
            manifest['input_guard'] = guard
        (stage / 'trial.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stage.rename(out)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar build-aten-python-trial', description=__doc__)
    for name in ('source', 'function', 'projection', 'projection-sha256', 'selection', 'selection-sha256',
                 'contract', 'contract-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--geometry-contract', help='declared value-independent operator effect and storage contract')
    parser.add_argument('--geometry-contract-sha256')
    args = parser.parse_args(argv)
    result = build_aten_python_trial(**vars(args))
    print(json.dumps(dict(trial=str(Path(args.out) / 'trial.json'), code_key=result['source_proof']['code_key'],
                          source_replacement_authorized=False)))
    return 0
