"""Build an opt-in native call bridge for an exact observed Python function."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sysconfig
from tempfile import TemporaryDirectory

from scar.analysis.aten_python_trial_source import qualify_python_trial_source
from scar.codegen.aten_native_graph import _read_declared
from scar.codegen.aten_provider import _build_interface, _dependencies
from scar.codegen.native_provider_bundle import _content
from scar.compiler.plan_aten_native_graph import _bundle, _segment_plan, _statistics, _window_model
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, problem_document
from scar.optimization.joint_planner import verify_execution_plan
from scar.runtime.aten_provider_identity import file_stat_identity, local_file_identity, verify_library_file


_MODULE = '_scar_aten_python_trial'
_TEMPLATE = Path(__file__).with_name('aten_python_trial_v1.cpp.in')
_PAIR_HEADER = Path(__file__).with_name('aten_native_pair_v1.h')


def _file_ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _read_plan(path, digest):
    path, selection, _ = _read_declared(path, digest)
    if (selection['schema'], selection['schema_version'], selection['selected'], selection['boundary']) != (
            'scar.aten-native-graph-schedule', 1, 'TWO_CPU', 'GENERATED_WARM_ATEN_GRAPH_CALL'):
        raise ValueError('a validated two-CPU generated graph selection is required')
    bundle = selection['bundle']
    artifact, origin, recipe, hardware, executable, key = _bundle(bundle['uri'], bundle['sha256'])
    if (origin != bundle or key != 'cpu_windows' or 'tensor_lifetime' not in artifact
            or selection['executable'] != dict(path=str(executable), **_content(executable))):
        raise ValueError('selected program does not match the retained graph delivery')
    schedule = artifact[key]
    evidence = selection['observations']
    calibrated = read_artifact(evidence)
    validation = read_artifact(selection['paired_validation'])
    warmup, count = selection['warmup_pairs'], selection['measured_pairs']
    model = _window_model(calibrated['calibration'], recipe, schedule, warmup, observe_storage=True,
                          tensor_budget_bytes=selection['model']['tensor_storage']['budget_bytes'])
    stats = _statistics(validation, warmup, count, schedule['cpus'])
    starts = selection['selected_window_starts']
    if (model != selection['model'] or stats != selection['validation'] or not stats['profitable_two_cpu']
            or not starts or starts != model['selected_starts'] or starts != validation['windows']
            or selection['cpu_ids'] != schedule['cpus'] or selection['original_target'] != schedule['target']
            or validation['result_storage_bytes'] != calibrated['calibration']['result_storage_bytes']):
        raise ValueError('retained ATen time, storage or window choice differs from its evidence')
    problem, _, timeline = _segment_plan(hardware, schedule, origin, evidence, model, starts)
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


def _build_extension(stage, artifact, torch):
    _, libraries, cxx, compile_flags, link_flags = _build_interface(torch)
    python_torch = libraries[0].with_name('libtorch_python.so').resolve(strict=True)
    filename = _MODULE + sysconfig.get_config_var('EXT_SUFFIX')
    # CPython API symbols are resolved by the importing interpreter, including static CPython builds.
    extension_links = [flag for flag in link_flags if flag != '-Wl,-z,defs']
    command = [*cxx, *compile_flags, '-I', sysconfig.get_path('include'), '-O2', '-fPIC', '-shared',
               '-pthread', 'trial.cpp', *extension_links, str(python_torch), '-o', filename]
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
                           contract, contract_sha256, out):
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
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-python-trial-', dir=out.parent) as tmp:
        stage = Path(tmp)
        bundle = Path(selected['bundle']['uri']).parent
        for name in ('graph.hpp', 'operators.hpp'):
            dest = stage / name
            dest.write_bytes((bundle / name).read_bytes())
            if _content(dest) != artifact['files'][name]:
                raise ValueError('generated graph header differs from the exact verified bundle: ' + name)
        (stage / _PAIR_HEADER.name).write_bytes(_PAIR_HEADER.read_bytes())
        (stage / 'trial.cpp').write_bytes(_TEMPLATE.read_bytes())
        extension, provider, command = _build_extension(stage, artifact, torch)
        manifest = dict(schema='scar.aten-python-trial', schema_version=1,
            scope='EXPLICIT_PYTHON_CPU_DIFFERENTIAL_TRIAL', source_proof=proof, recipe=recipe, contract=contract,
            cpus=selected['cpu_ids'], window_starts=selected['selected_window_starts'], inputs=selected['inputs'],
            provider=provider, extension=extension, references=dict(replay,
                source_projection=_file_ref(projection_path, captured['schema']),
                wrapper_contract=_file_ref(contract_path, contract['schema'])),
            compiler_command=command, cxx_standard=20, source_replacement_authorized=False,
            python_boundary_performance_measured=False)
        (stage / 'trial.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stage.rename(out)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar build-aten-python-trial', description=__doc__)
    for name in ('source', 'function', 'projection', 'projection-sha256', 'selection', 'selection-sha256',
                 'contract', 'contract-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = build_aten_python_trial(**vars(args))
    print(json.dumps(dict(trial=str(Path(args.out) / 'trial.json'), code_key=result['source_proof']['code_key'],
                          source_replacement_authorized=False)))
    return 0
