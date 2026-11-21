"""Build and compare a standalone NVHPC CPU path for a functional matrix export."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _dependencies, _resolve
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_export_cuda_source import qualify_cuda_export
from scar.compiler.aten_native_recipe import normalize_recipe
from scar.compiler.compile_exported_aten import _calibration_paths, _project, _shape_profiles, _write_record
from scar.compiler.tensor_physical_problem import tensor_physical_problem
from scar.compiler.tensor_program_graph import straight_line_graph, tensor_type
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import codec
from scar.optimization.aten_tensor_lifetime import tensor_footprint


SCHEMA = 'scar.native-exported-nvhpc-cpu'
VALIDATION = 'scar.native-exported-nvhpc-cpu-validation'
_NAME = 'nvhpc-matrix-entry'
_EXECUTABLE = 'bin/' + _NAME
_PROGRAM = 'plan/program.json'
_TEMPLATE = Path(__file__).resolve().parents[1] / 'codegen/cpu_blas_export.cpp.in'


def _source(path, digest):
    path = Path(path).resolve(strict=True)
    receipt = dict(path=str(path), **_content(path))
    if receipt['sha256'] != digest:
        raise ValueError('functional source export differs from its pinned digest')
    return receipt


def _functional(exported, profiles, torch):
    recipe, shapes, weights, symbols = _project(exported, profiles, torch)
    include = Path(torch.__file__).resolve().parent / 'include'
    resolved = _resolve([row['schema'] for row in recipe['operations']], torch, include)
    recipe = normalize_recipe(recipe, resolved, torch)
    graph = straight_line_graph(recipe, {name: tensor_type(shape) for name, shape in shapes.items()})
    return recipe, shapes, weights, symbols, graph


def _contract(atol, rtol):
    if atol is None and rtol is None:
        return dict(kind='EXACT_BYTES')
    if any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError('declare both finite nonnegative FP32 output tolerances')
    return dict(kind='DECLARED_FLOAT32_OUTPUT_TOLERANCE', atol=atol, rtol=rtol)


def _plan(recipe, projection, program, hardware, budget, cpus, threads, *, observations=(), stack=None):
    if (type(budget) is not int or budget <= 0 or type(threads) is not int or threads != len(cpus)
            or not cpus or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or list(cpus) != sorted(set(cpus))):
        raise ValueError('NVHPC CPU matrix entry requires a positive memory budget and one thread per selected hardware core')
    inputs, ops = projection['inputs'], projection['operations']
    sizes = {row['id']: row['nbytes'] for row in (*inputs, *ops)}
    footprint = tensor_footprint(recipe, input_bytes=[sizes[row['id']] for row in inputs],
        output_bytes=[sizes[row['id']] for row in ops], candidates=(), budget_bytes=budget)
    costs = None
    if observations:
        from scar.ir.program.providers import ProviderArtifact
        from scar.optimization.nvhpc_sgemm_cost import sgemm_body_costs

        target_ref = artifact_reference(Path(hardware).resolve()/'result.json', 'scar.target-hardware-graph')
        costs = sgemm_body_costs(projection, observations, stack=stack, target=ProviderArtifact(**target_ref), cpus=cpus)
    problem = tensor_physical_problem(hardware, cpus, budget=budget, operations=recipe['operations'],
        inputs={row['id'] for row in inputs}, outputs=projection['outputs'], sizes=sizes,
        reserved=footprint['persistent_input_bytes'], source=program, region='entry',
        dispatch=lambda _: ('NVHPC_CBLAS_SGEMM_LP64', cpus, 0), costs=costs)
    return problem, footprint


def _estimate_body(problem):
    from scar.ir.implementation_plan import problem_from_dict
    from scar.optimization.constructive_planner import construct_implementations
    from scar.optimization.joint_planner import verify_execution_plan

    typed = problem_from_dict(problem)
    candidate = construct_implementations(typed)
    candidate.pop('elapsed_seconds', None)
    if candidate['makespan_ns'] is not None:
        verify_execution_plan(typed, candidate)
    return dict(schema='scar.nvhpc-matrix-body-forecast', schema_version=1,
        scope='WARM_OPERATOR_BODY_ONLY', full_process_duration_ns=None, schedule=candidate)


def _generate(projection, footprint, threads):
    inputs, ops = projection['inputs'], projection['operations']
    indices = {row['id']: index for index, row in enumerate((*inputs, *ops))}
    sizes = {row['id']: row['nbytes'] for row in (*inputs, *ops)}

    def array(name, rows):
        return f'inline constexpr std::array<std::size_t, {len(rows)}> {name}{{' + ', '.join(map(str, rows)) + '};'

    products = ['{' + ', '.join(map(str, (indices[row['left']], indices[row['right']], indices[row['id']],
        *row['shape'], row['reduction']))) + '}' for row in ops]
    offsets, release = [0], []
    for row in footprint['release_after']:
        release.extend(len(inputs) + index for index in row)
        offsets.append(len(release))
    outputs = projection['outputs']
    return '\n'.join(('#pragma once', '#include <array>', '#include <cstddef>', 'namespace matrix_plan {',
        'struct Product { std::size_t left, right, result; int m, n, k; };',
        f'inline constexpr int threads = {threads};', f'inline constexpr std::size_t buffers = {len(sizes)};',
        array('input_bytes', [row['nbytes'] for row in inputs]),
        f'inline constexpr std::array<Product, {len(ops)}> products{{{{{", ".join(products)}}}}};',
        array('release_offsets', offsets), array('releases', release), array('outputs', [indices[row] for row in outputs]),
        array('output_bytes', [sizes[row] for row in outputs]), '}', ''))


def _launcher(inputs, outputs, cpus, threads):
    checks, arguments = [], []
    index = 1
    for row in inputs:
        if row['kind'] == 'WEIGHT':
            argument = f'"$root/{row["path"]}"'
            checks.append(f'test "$(sha256sum < {argument} | cut -d " " -f 1)" = {row["sha256"]}')
        else:
            argument = '"${' + str(index) + '}"'
            index += 1
        arguments.append(argument)
    arguments.extend('"${' + str(value) + '}"' for value in range(index, index + len(outputs)))
    command = shlex.join(['env', '-i', 'PATH=/usr/bin:/bin', f'OMP_NUM_THREADS={threads}',
                          f'OPENBLAS_NUM_THREADS={threads}', 'OMP_DYNAMIC=FALSE', 'taskset', '-c',
                          ','.join(map(str, cpus))])
    command += f' "$root/{_EXECUTABLE}" ' + ' '.join(arguments)
    return '\n'.join(('#!/bin/sh', 'set -eu', 'root=$(dirname -- "$0")',
        f'test "$#" -eq {index + len(outputs) - 1}', *checks, 'exec ' + command, ''))


def _actual_linkage(stack, dependencies):
    from scar.compiler.operator_stack import verify_operator_stack_linkage

    if any(name.startswith(('libtorch', 'libc10', 'libpython', 'libcuda', 'libcublas')) for name in dependencies):
        raise ValueError('NVHPC CPU artifact depends on Python, LibTorch or a GPU library')
    return verify_operator_stack_linkage(stack, dependencies, 'CPU')


def compile_exported_nvhpc(model=None, *, model_sha256=None, stack, stack_sha256, target, cpus, threads, tensor_budget_bytes,
                           out, input_shapes=(), atol=None, rtol=None, sgemm_observations=(), source_entry=None):
    from scar.compiler.operator_stack import load_operator_stack, nvidia_stack_build

    direct_source = model is None
    if (direct_source and (source_entry is None or model_sha256 is not None or input_shapes)
            or not direct_source and model_sha256 is None):
        raise ValueError('request a static Python entry or a pinned functional export and its shape profile')
    stack_ref, stack_doc = load_operator_stack(stack)
    if stack_ref['sha256'] != stack_sha256:
        raise ValueError('operator stack differs from its pinned digest')
    build = nvidia_stack_build(stack_doc, 'CPU')
    profiles = _shape_profiles(input_shapes)
    if direct_source:
        from scar.analysis.nvhpc_source_entry import build_nvhpc_source_recipe

        path = Path(source_entry).resolve(strict=True)
        source = dict(path=str(path), **_content(path))
        recipe = build_nvhpc_source_recipe(path)
        shapes = {row['id']: row['shape'] for row in recipe['inputs']}
        for row in recipe['operations']:
            left, right = (shapes[arg['value']] for arg in row['arguments'])
            shapes[row['id']] = [left[0], right[1]]
        graph = straight_line_graph(recipe, {key: tensor_type(shape) for key, shape in shapes.items()})
        weights, symbols = {}, {}
    else:
        import torch

        source = _source(model, model_sha256)
        recipe, shapes, weights, symbols, graph = _functional(torch.export.load(source['path']), profiles, torch)
    graph_content = (codec.dumps(graph) + '\n').encode()
    program = dict(path=_PROGRAM, sha256=sha256(graph_content).hexdigest(), size_bytes=len(graph_content))
    kinds = {row['id']: 'WEIGHT' if row['id'] in weights else 'INPUT' for row in recipe['inputs']}
    projection = qualify_cuda_export(recipe, shapes, graph=graph, program_ref=program, input_kinds=kinds)
    source_contract = None
    if source_entry is not None:
        from scar.analysis.nvhpc_source_entry import derive_nvhpc_source_entry

        if weights:
            raise ValueError('source file entry proof requires every original matrix as an input port')
        source_contract = derive_nvhpc_source_entry(source_entry, recipe)
        if direct_source and any(source[key] != source_contract['source'][key] for key in source):
            raise ValueError('original Python entry changed while the numerical graph was built')
    cpus = list(cpus)
    from scar.optimization.nvhpc_sgemm_cost import SCHEMA as OBSERVATION

    observations = [artifact_reference(Path(path).resolve(), OBSERVATION) for path in sgemm_observations]
    problem, footprint = _plan(recipe, projection, program, target, tensor_budget_bytes, cpus, threads,
                               observations=observations, stack=stack_doc)
    numerics = _contract(atol, rtol)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-nvhpc-entry-', dir=out.parent) as scratch:
        root = Path(scratch) / 'project'
        for part in ('plan', 'data', 'include/scar_generated', 'src'):
            (root / part).mkdir(parents=True, exist_ok=True)
        (root / _PROGRAM).write_bytes(graph_content)
        inputs = []
        for port in projection['inputs']:
            row = dict(port)
            if port['kind'] == 'WEIGHT':
                name = f'data/{row["id"]}.f32'
                weights[row['id']].numpy().tofile(root / name)
                row.update(path=name, **_content(root / name))
            inputs.append(row)
        records = dict(recipe=recipe, projection=projection, problem=problem, execution=dict(source=source,
            program=program, inputs=inputs, input_shapes=profiles, symbolic_profile=symbols, cpu_ids=cpus,
            threads=threads, storage=footprint, numerical_contract=numerics, time_status='UNMEASURED'))
        if observations:
            records['execution'].update(sgemm_observations=observations, time_status='COMPONENT_MODEL')
            records['body-forecast'] = _estimate_body(problem)
        if source_contract is not None:
            records['source-entry'] = source_contract
        if direct_source:
            records['execution']['source_kind'] = 'ORIGINAL_PYTHON_FILE_ENTRY'
        for name, data in records.items():
            _write_record(root / f'plan/{name}.json', data)
        (root / 'include/scar_generated/matrix_plan.hpp').write_text(_generate(projection, footprint, threads))
        (root / 'src/main.cpp').write_bytes(_TEMPLATE.read_bytes())
        launch = root / 'launch.sh'
        launch.write_text(_launcher(inputs, projection['outputs'], cpus, threads))
        launch.chmod(0o755)
        libraries = tuple(build['libraries'])
        write_native_project(root, _NAME, include_dirs=build['include_dirs'], compile_options=('-O3', '-Kieee'),
            libraries=tuple(map(str, libraries)), runtime_dirs=build['runtime_dirs'], install_dirs=(('data', '.'),),
            install_launcher=True)
        _, dependencies = build_native_project(root, _NAME, build['compiler'], expected_libraries=libraries)
        linkage = _actual_linkage(stack_doc, dependencies)
        if _source(source['path'], source['sha256']) != source or load_operator_stack(stack_ref) != (stack_ref, stack_doc):
            raise ValueError('functional source export or toolchain changed during native compilation')
        manifest = dict(schema=SCHEMA, schema_version=1, status='STANDALONE_CPP20_NVHPC_CPU_CANDIDATE',
            cxx_standard=20, executable=_EXECUTABLE, stack=stack_ref, linkage=linkage, dependencies=dependencies,
            deployment_authorized=False, files={path.relative_to(root).as_posix(): _content(path)
                for path in sorted(root.rglob('*')) if path.is_file()})
        _write_record(root / 'artifact.json', manifest)
        check_nvhpc_project(root, _content(root/'artifact.json')['sha256'])
        root.rename(out)
    return manifest


def check_nvhpc_project(project, manifest_sha256):
    from scar.compiler.operator_stack import load_operator_stack

    root = Path(project).resolve(strict=True)
    reference = artifact_reference(root / 'artifact.json', SCHEMA)
    if reference['sha256'] != manifest_sha256:
        raise ValueError('NVHPC project differs from its pinned manifest')
    manifest = read_artifact(reference)
    if (manifest['schema_version'], manifest['status'], manifest['cxx_standard'], manifest['executable'],
            manifest['deployment_authorized']) != (1, 'STANDALONE_CPP20_NVHPC_CPU_CANDIDATE', 20, _EXECUTABLE, False):
        raise ValueError('NVHPC CPU native project has a different execution contract')

    def member(name):
        path = root / name
        if not path.is_relative_to(root) or path.resolve(strict=True) != path or _content(path) != manifest['files'][name]:
            raise ValueError('NVHPC project contains a mismatched member: ' + name)
        return path

    def read(name):
        return json.loads(member(f'plan/{name}.json').read_bytes())

    for name in manifest['files']:
        member(name)
    _, stack_doc = load_operator_stack(manifest['stack'])
    dependencies = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(member(_EXECUTABLE))], text=True, env={}))
    if dependencies != manifest['dependencies'] or _actual_linkage(stack_doc, dependencies) != manifest['linkage']:
        raise ValueError('NVHPC CPU executable resolves a different operator stack')
    execution, recipe = read('execution'), read('recipe')
    source = _source(execution['source']['path'], execution['source']['sha256'])
    program = dict(path=_PROGRAM, **_content(member(_PROGRAM)))
    if source != execution['source'] or program != execution['program']:
        raise ValueError('NVHPC original source graph or export changed')
    graph = codec.loads(member(_PROGRAM).read_text())
    inputs = execution['inputs']
    projection = qualify_cuda_export(recipe, graph=graph, program_ref=program,
        input_kinds={row['id']: row['kind'] for row in inputs})
    if projection != read('projection'):
        raise ValueError('NVHPC saved projection differs from the original typed graph')
    if 'plan/source-entry.json' in manifest['files']:
        from scar.analysis.nvhpc_source_entry import build_nvhpc_source_recipe, derive_nvhpc_source_entry

        contract = read('source-entry')
        if (any(row['kind'] != 'INPUT' for row in inputs)
                or derive_nvhpc_source_entry(contract['source']['path'], recipe) != contract):
            raise ValueError('original Python file ABI differs from its structural proof')
        if execution.get('source_kind') == 'ORIGINAL_PYTHON_FILE_ENTRY' and (
                contract['source']['path'] != source['path'] or recipe != build_nvhpc_source_recipe(source['path'])):
            raise ValueError('NVHPC graph differs from direct original source normalization')
    elif execution.get('source_kind') == 'ORIGINAL_PYTHON_FILE_ENTRY':
        raise ValueError('direct original Python entry requires its source effect record')
    for row, port in zip(inputs, projection['inputs'], strict=True):
        expected = dict(port)
        if row['kind'] == 'WEIGHT':
            name = f'data/{row["id"]}.f32'
            weights = _content(member(name))
            if weights['size_bytes'] != port['nbytes']:
                raise ValueError('NVHPC weight extent differs from its graph port')
            expected.update(path=name, **weights)
        if expected != row:
            raise ValueError('NVHPC file arguments differ from the original graph ports')
    original_problem = read('problem')
    target = Path(original_problem['target']['uri']).parent
    cpus, threads = execution['cpu_ids'], execution['threads']
    observations = execution.get('sgemm_observations', ())
    problem, footprint = _plan(recipe, projection, program, target, execution['storage']['budget_bytes'], cpus, threads,
                               observations=observations, stack=stack_doc)
    if observations and read('body-forecast') != _estimate_body(problem):
        raise ValueError('NVHPC body forecast differs from matched component and schedule replay')
    contract = execution['numerical_contract']
    if (problem != original_problem or footprint != execution['storage']
            or contract != _contract(contract.get('atol'), contract.get('rtol'))
            or member('src/main.cpp').read_bytes() != _TEMPLATE.read_bytes()
            or member('include/scar_generated/matrix_plan.hpp').read_text() != _generate(projection, footprint, threads)
            or member('launch.sh').read_text() != _launcher(inputs, projection['outputs'], cpus, threads)):
        raise ValueError('NVHPC CPU source or storage plan differs from independent replay')
    return root, reference, execution, recipe, projection


def _compare(left, right, contract, torch):
    exact = left == right
    if contract['kind'] == 'EXACT_BYTES' or exact:
        return dict(accepted=exact, exact_bytes=exact, max_abs_error=0.0 if exact else None)
    a, b = (torch.frombuffer(bytearray(row), dtype=torch.float32).to(torch.float64) for row in (left, right))
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return dict(accepted=False, exact_bytes=False, max_abs_error=None)
    delta = (a - b).abs()
    passed = bool((delta <= contract['atol'] + contract['rtol'] * a.abs()).all())
    return dict(accepted=passed, exact_bytes=False, max_abs_error=float(delta.max()))


def validate_nvhpc_project(project, manifest_sha256, arguments, out):
    import torch

    root, artifact, execution, recipe, projection = check_nvhpc_project(project, manifest_sha256)
    if execution.get('source_kind') == 'ORIGINAL_PYTHON_FILE_ENTRY':
        raise ValueError('use validate-source to compare the direct original Python process')
    inputs = execution['inputs']
    paths = _calibration_paths(arguments, inputs)
    receipts = [dict(path=str(path), **_content(path)) for path in paths]
    exported = torch.export.load(execution['source']['path'])
    expected_recipe, _, weights, symbols, _ = _functional(exported, execution['input_shapes'], torch)
    if recipe != expected_recipe or symbols != execution['symbolic_profile']:
        raise ValueError('live functional export differs from its archived projection')
    user = iter(paths)
    ordered = []
    for row in inputs:
        if row['kind'] == 'WEIGHT':
            value = weights[row['id']]
            if sha256(value.numpy().tobytes()).hexdigest() != row['sha256']:
                raise ValueError('live functional export weight differs from its native bytes')
        else:
            raw = bytearray(next(user).read_bytes())
            if len(raw) != row['nbytes']:
                raise ValueError('input file differs from its admitted FP32 shape')
            value = torch.frombuffer(raw, dtype=torch.float32).reshape(row['shape'])
        ordered.append(value)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    native = [out / f'native-{index}.f32' for index in range(len(projection['outputs']))]
    with torch.no_grad():
        originals = exported.graph_module(*ordered)
        expected = tuple(value.detach().numpy().tobytes() for value in originals)
    child = subprocess.run([root / 'launch.sh', *paths, *native], env={}, capture_output=True)
    (out / 'native.stdout').write_bytes(child.stdout)
    (out / 'native.stderr').write_bytes(child.stderr)
    contract = execution['numerical_contract']
    outputs = []
    for index, (raw, path) in enumerate(zip(expected, native, strict=True)):
        reference = out / f'original-{index}.f32'
        reference.write_bytes(raw)
        row = dict(index=index, reference=dict(path=str(reference), **_content(reference)), native=None, accepted=False,
                   exact_bytes=False, max_abs_error=None)
        if path.is_file() and path.stat().st_size == len(raw):
            row.update(native=dict(path=str(path), **_content(path)), **_compare(raw, path.read_bytes(), contract, torch))
        outputs.append(row)
    if receipts != [dict(path=str(path), **_content(path)) for path in paths]:
        raise ValueError('functional validation inputs changed during execution')
    _source(execution['source']['path'], execution['source']['sha256'])
    accepted = not child.returncode and not child.stdout and not child.stderr and all(row['accepted'] for row in outputs)
    result = dict(schema=VALIDATION, schema_version=1, artifact=artifact, inputs=receipts, outputs=outputs,
        returncode=child.returncode, accepted=accepted, numerical_contract=contract,
        deployment_authorized=False, validation_scope='ORDERED_RAW_MATRIX_OUTPUTS_ON_THE_GIVEN_INPUT_FILES')
    _write_record(out / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-exported-nvhpc', description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    build = actions.add_parser('compile')
    build.add_argument('--model')
    build.add_argument('--model-sha256')
    for name in ('stack', 'stack-sha256', 'target', 'out'):
        build.add_argument('--' + name, required=True)
    build.add_argument('--tensor-budget-bytes', type=int, required=True)
    build.add_argument('--cpu-id', dest='cpus', type=int, action='append', required=True)
    build.add_argument('--threads', type=int, default=1)
    build.add_argument('--input-shape', dest='input_shapes', action='append', default=[])
    build.add_argument('--atol', type=float)
    build.add_argument('--rtol', type=float)
    build.add_argument('--sgemm-observation', dest='sgemm_observations', action='append', default=[])
    build.add_argument('--source-entry', help='original Python file entry for structural source-to-export replay')
    check = actions.add_parser('check')
    validate = actions.add_parser('validate')
    for command in (check, validate):
        command.add_argument('--project', required=True)
        command.add_argument('--manifest-sha256', required=True)
    validate.add_argument('--input', action='append', default=[])
    validate.add_argument('--out', required=True)
    source = actions.add_parser('validate-source')
    source.add_argument('--project', required=True)
    source.add_argument('--manifest-sha256', required=True)
    source.add_argument('--input', action='append', default=[])
    source.add_argument('--out', required=True)
    source.add_argument('--warmup-pairs', type=int, default=1)
    source.add_argument('--measurement-pairs', type=int, default=3)
    parsed = parser.parse_args(argv)
    action, args = parsed.action, vars(parsed).copy()
    del args['action']
    if action == 'compile':
        result = compile_exported_nvhpc(**args)
        print(json.dumps(dict(status=result['status'], executable=str(Path(parsed.out).resolve() / _EXECUTABLE))))
    elif action == 'check':
        _, ref, *_ = check_nvhpc_project(**args)
        print(json.dumps(dict(project=ref, checked=True)))
    elif action == 'validate':
        result = validate_nvhpc_project(parsed.project, parsed.manifest_sha256, parsed.input, parsed.out)
        print(json.dumps(dict(accepted=result['accepted'], output=str(Path(parsed.out).resolve() / 'result.json'))))
        if not result['accepted']:
            return 2
    else:
        from scar.compiler.nvhpc_source_validation import validate_source_process

        result = validate_source_process(parsed.project, parsed.manifest_sha256, parsed.input, parsed.out,
            warmup_pairs=parsed.warmup_pairs, measurement_pairs=parsed.measurement_pairs)
        print(json.dumps(dict(accepted=result['accepted'], timing=result['timing'],
                              output=str(Path(parsed.out).resolve() / 'result.json'))))
        if not result['accepted']:
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
