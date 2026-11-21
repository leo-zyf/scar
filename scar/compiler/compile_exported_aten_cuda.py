"""Compile an offline CUDA or hybrid candidate from a functional torch.export."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _build_interface, _dependencies, _resolve
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_native_recipe import normalize_recipe
from scar.compiler.compile_exported_aten import _project, _shape_profiles, _write_record
from scar.compiler.tensor_program_graph import straight_line_graph, tensor_type
from scar.ir.implementation_plan import problem_from_dict
from scar.ir.program import codec


SCHEMA = 'scar.native-exported-aten-cuda'
EXECUTABLE = 'bin/aten-inference-cuda'
CUDA_ENVIRONMENT = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MPS_PIPE_DIRECTORY')
_MODEL_PATH = 'plan/cuda-model.json'
_PROGRAM_PATH = 'plan/program.json'
_HYBRID_MODEL = 'scar.aten-export-hybrid-offline-profile'
_PLACEMENT_MODEL = 'scar.aten-export-placement-offline-profile'
_PLACEMENT_CANDIDATE = 'OFFLINE_CPP20_PLACEMENT_CANDIDATE'
_PLACEMENT_TREATMENT = 'OFFLINE_CPP20_PLACEMENT_CALIBRATION_TREATMENT'
PLACEMENT_RECORDS = {_PLACEMENT_CANDIDATE: 'placement-search', _PLACEMENT_TREATMENT: 'placement-treatment'}
ACTIVITY_BINDING = 'plan/activity-binding.json'
_ACTIVITY_MEMBERS = dict(program=_PROGRAM_PATH, problem='plan/problem.json', schedule='plan/schedule.json',
                         bindings='plan/bindings.json', executable=EXECUTABLE, launcher='launch.sh')


def _pipeline(model, *, treatment=False):
    schema = model.get('schema') if isinstance(model, dict) else None
    if treatment and schema != _PLACEMENT_MODEL:
        raise ValueError('an exact calibration placement requires the typed placement profile')
    if schema == _PLACEMENT_MODEL:
        from scar.compiler.aten_export_device_placement import qualify_device_export

        if treatment:
            from scar.compiler.native_export_placement_treatment import plan_device_treatment, replay_device_treatment

            return qualify_device_export, plan_device_treatment, replay_device_treatment, _PLACEMENT_TREATMENT
        from scar.compiler.aten_export_device_placement import plan_device_export, replay_device_export

        return qualify_device_export, plan_device_export, replay_device_export, _PLACEMENT_CANDIDATE
    if schema == _HYBRID_MODEL:
        from scar.compiler.aten_export_hybrid_plan import plan_hybrid_export, replay_hybrid_export
        from scar.compiler.aten_export_hybrid_source import qualify_hybrid_export

        return qualify_hybrid_export, plan_hybrid_export, replay_hybrid_export, 'OFFLINE_CPP20_HYBRID_CANDIDATE'
    from scar.compiler.aten_export_cuda_plan import plan_cuda_export, replay_cuda_export
    from scar.compiler.aten_export_cuda_source import qualify_cuda_export

    return qualify_cuda_export, plan_cuda_export, replay_cuda_export, 'OFFLINE_CPP20_CUDA_CANDIDATE'


def _provider_libraries(toolkit):
    lib = toolkit / 'lib64'
    libraries = []
    for name in ('libcudart', 'libcublas'):
        candidate = (lib / (name + '.so')).resolve(strict=True)
        listing = subprocess.check_output(['readelf', '-d', str(candidate)], text=True)
        match = re.search(r'\(SONAME\).*\[(' + re.escape(name) + r'\.so\.[0-9]+)\]', listing)
        if match is None:
            raise ValueError(f'CUDA provider {name} has no qualified runtime soname')
        soname = lib / match[1]
        if soname.resolve(strict=True) != candidate:
            raise ValueError(f'CUDA provider {name} resolves to different link and runtime files')
        libraries.append(soname)
    return libraries


def _library_locks(libraries, dependencies, runtime_dirs):
    locks = set(libraries)
    for directory in map(Path, runtime_dirs):
        for name, resolved in dependencies.items():
            if Path(name).name == name and resolved is not None and (directory / name).resolve() == Path(resolved):
                locks.add(directory / name)
    return [dict(path=str(path), **_content(path)) for path in sorted(locks)]


def verify_cuda_operator_stack(root, manifest):
    """Recheck stack identity and the clean ELF loader's resolution before execution."""
    if 'operator_stack' not in manifest:
        return None
    from scar.compiler.operator_stack import load_operator_stack, verify_operator_stack_linkage

    retained = manifest['operator_stack']
    if type(retained) is not dict or set(retained) != {'reference', 'linkage'}:
        raise ValueError('CUDA project requires its original operator stack and linkage receipt')
    reference, document = load_operator_stack(retained['reference'])
    executable = root / EXECUTABLE
    listing = subprocess.check_output(['/usr/bin/ldd', str(executable)], text=True, env={})
    dependencies = _dependencies(listing)
    linkage = verify_operator_stack_linkage(document, dependencies, 'CUDA')
    if dependencies != manifest['dependencies'] or retained != dict(reference=reference, linkage=linkage):
        raise ValueError('CUDA operator stack or loaded NVIDIA dependencies differ from the retained receipt')
    return retained


def _render_launch(inputs, output_count, *, host_lane=False, calibrate=False, stack_sha256=None):
    weight_checks = '\n'.join(
        f'test "$(sha256sum < "$root/{row["path"]}" | cut -d " " -f 1)" = {row["payload"]["sha256"]}'
        for row in inputs if row['kind'] == 'WEIGHT')
    cuda_scope = '\n'.join(f'  if [ "${{{name}+x}}" ]; then set -- "{name}=${name}" "$@"; fi'
                           for name in CUDA_ENVIRONMENT)
    ports = iter(range(1, len(inputs) + output_count + 1))
    arguments = [f'"$root/{row["path"]}"' if row['kind'] == 'WEIGHT' else '"${' + str(next(ports)) + '}"'
                 for row in inputs]
    arguments.extend('"${' + str(next(ports)) + '}"' for _ in range(output_count))
    user_count = sum(row['kind'] == 'INPUT' for row in inputs)
    host_limits = 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ' if host_lane else ''
    file_mode = '--trial|--calibrate) mode=$1; shift' if calibrate else '--trial) shift'
    mode = '"$mode"' if calibrate else '--trial'
    mode_help = '--trial|--calibrate' if calibrate else '--trial'
    stack_marker = '' if stack_sha256 is None else f'# Operator stack SHA256: {stack_sha256}\n'
    return f'''#!/bin/sh
{stack_marker}set -eu
root=$(dirname -- "$0")
run_cuda() {{
  set -- "$root/{EXECUTABLE}" "$@"
{cuda_scope}
  exec env -i {host_limits}"$@"
}}
case "${{1:-}}" in
  --describe|--probe) test "$#" -eq 1
    run_cuda "$1" ;;
  {file_mode} ;;
  *) echo "usage: launch.sh --describe | --probe | {mode_help} INPUT... NEW_OUTPUT..." >&2
    exit 2 ;;
esac
test "$#" -eq {user_count + output_count}
{weight_checks}
run_cuda {mode} {' '.join(arguments)}
'''


def _native_sources(projection, problem, schedule, bindings, *, calibrate=False):
    from scar.codegen.aten_export_cuda_source import render_exported_cuda_entry

    shapes = {row['id']: row['shape'] for row in (*projection['inputs'], *projection['operations'])}
    input_ports = [dict(value=value, shape=shapes[row['id']])
                   for row, value in zip(projection['inputs'], bindings['input_values'], strict=True)]
    output_ports = [dict(value=value, shape=shapes[name])
                    for name, value in zip(projection['outputs'], bindings['output_values'], strict=True)]
    keywords = {key: bindings[key] for key in ('device', 'memory_kinds', 'stream_channels', 'compute_resource')}
    if 'host_resource' in bindings:
        keywords['host_compute_resource'] = bindings['host_resource']
    if calibrate:
        keywords['calibrate'] = True
    return render_exported_cuda_entry(problem_from_dict(problem), schedule, **keywords,
        input_ports=input_ports, output_ports=output_ports)


def _describe(root):
    observation = subprocess.check_output([str(root / EXECUTABLE), '--describe'], text=True, env={})
    return json.loads(observation)


def compile_exported_aten_cuda(model, *, model_sha256, cuda_model, cuda_model_sha256,
                               out, input_shapes=(), cuda_root=None, with_activity_calibration=False,
                               placement_ordinal=None, operator_stack=None):
    import torch

    if operator_stack is not None:
        from scar.compiler.operator_stack import (
            load_operator_stack, nvidia_stack_build, verify_operator_stack_linkage,
        )

        if cuda_root is not None or with_activity_calibration or placement_ordinal is not None:
            raise ValueError('the NVIDIA operator stack pins CUDA paths and requires direct CUDA execution')
        stack_reference, stack_document = load_operator_stack(operator_stack)
        stack_build = nvidia_stack_build(stack_document, 'CUDA')

    model = Path(model).resolve(strict=True)
    source_lock = _content(model)
    if source_lock['sha256'] != model_sha256:
        raise ValueError('exported model differs from its declared digest')
    model_file = Path(cuda_model).resolve(strict=True)
    model_bytes = model_file.read_bytes()
    model_lock = _content(model_file)
    if model_lock['sha256'] != cuda_model_sha256:
        raise ValueError('declared CUDA model differs from its pinned digest')
    declared = json.loads(model_bytes)
    treatment = placement_ordinal is not None
    qualify, plan, _, status = _pipeline(declared, treatment=treatment)
    plan_options = {'ordinal': placement_ordinal} if treatment else {}
    with_activity_calibration = with_activity_calibration or treatment
    exported = torch.export.load(model)
    recipe, shapes, weights, symbols = _project(exported, _shape_profiles(input_shapes), torch)
    include = Path(torch.__file__).resolve().parent / 'include'
    recipe = normalize_recipe(recipe, _resolve([op['schema'] for op in recipe['operations']], torch, include), torch)
    graph = straight_line_graph(recipe, {name: tensor_type(shape) for name, shape in shapes.items()})
    kinds = {row['id']: 'WEIGHT' if row['id'] in weights else 'INPUT' for row in recipe['inputs']}
    compiler = shlex.split(os.environ.get('CXX', 'c++'))

    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-export-cuda-', dir=out.parent) as temporary:
        root = Path(temporary)
        (root / 'plan').mkdir()
        (root / 'data').mkdir()
        (root / _PROGRAM_PATH).write_text(codec.dumps(graph) + '\n')
        (root / _MODEL_PATH).write_bytes(model_bytes)
        program_ref = dict(path=_PROGRAM_PATH, **_content(root / _PROGRAM_PATH))
        model_ref = dict(path=_MODEL_PATH, **_content(root / _MODEL_PATH))
        if model_ref != dict(path=_MODEL_PATH, **model_lock):
            raise ValueError('declared CUDA model changed when it was read')
        projection = qualify(recipe, shapes, graph=graph, program_ref=program_ref, input_kinds=kinds)
        problem, schedule, bindings, *placement = plan(projection, declared, model_ref=model_ref, **plan_options)
        host_lane = 'host_resource' in bindings
        if operator_stack is not None and host_lane:
            raise ValueError('the NVIDIA operator stack requires direct CUDA lowering without the LibTorch host lane')
        if with_activity_calibration and not host_lane:
            raise ValueError('activity calibration requires a hybrid CPU and CUDA plan')
        generated = _native_sources(projection, problem, schedule, bindings, calibrate=with_activity_calibration)
        options = ['-O2']
        if operator_stack is None:
            toolkit = Path(cuda_root or '/usr/local/cuda').resolve(strict=True)
            libraries = _provider_libraries(toolkit)
            include_dirs = [str(toolkit / 'include')]
            runtime_dirs = [str(toolkit / 'lib64')]
        else:
            compiler = stack_build['compiler']
            libraries = stack_build['libraries']
            include_dirs = stack_build['include_dirs']
            runtime_dirs = stack_build['runtime_dirs']
        if host_lane:
            _, cpu_libraries, compiler, flags, _ = _build_interface(torch)
            libraries.extend(cpu_libraries)
            options.extend(flags[1:])
            runtime_dirs.append(str(cpu_libraries[0].parent))
        inputs = []
        for port in projection['inputs']:
            row = dict(port)
            if row['kind'] == 'WEIGHT':
                relative = f'data/{row["id"]}.f32'
                path = root / relative
                weights[row['id']].numpy().tofile(path)
                row.update(path=relative, payload=_content(path))
            inputs.append(row)
        source = dict(path=str(model), **source_lock)
        records = {'recipe': recipe, 'projection': projection, 'problem': problem, 'schedule': schedule,
                   'bindings': bindings, 'execution': dict(source=source, program=program_ref, model=model_ref,
                       inputs=inputs, symbolic_profile=symbols,
                       actual_device_timing='UNKNOWN', deployment_authorized=False, default_backend='NO_OP')}
        if with_activity_calibration:
            records['execution']['activity_calibration_abi'] = 1
        if placement:
            records[PLACEMENT_RECORDS[status]], = placement
        launch = root / 'launch.sh'
        launch.write_text(_render_launch(inputs, len(projection['outputs']), host_lane=host_lane,
            calibrate=with_activity_calibration,
            stack_sha256=None if operator_stack is None else stack_reference['sha256']))
        launch.chmod(0o755)
        for name, content in generated.sources.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        write_native_project(root, Path(EXECUTABLE).name, include_dirs=include_dirs,
            compile_options=options, libraries=tuple(map(str, libraries)),
            runtime_dirs=runtime_dirs, install_dirs=(('data', '.'),), install_launcher=True)
        _, dependencies = build_native_project(root, Path(EXECUTABLE).name, compiler, expected_libraries=libraries)
        stack = None
        if operator_stack is not None:
            stack = dict(reference=stack_reference,
                linkage=verify_operator_stack_linkage(stack_document, dependencies, 'CUDA'))
            if load_operator_stack(stack_reference) != (stack_reference, stack_document):
                raise ValueError('CUDA operator stack changed during native compilation')
            records['execution']['operator_stack'] = stack
        for name, value in records.items():
            _write_record(root / f'plan/{name}.json', value)
        # Include the transitive libraries from the same qualified provider directories.
        lock_dirs = runtime_dirs if operator_stack is not None else [str(toolkit / 'lib64')]
        _write_record(root / 'plan/dependencies.json', _library_locks(libraries, dependencies, lock_dirs))
        _write_record(root / 'validation/description.json', _describe(root))
        if with_activity_calibration:
            from scar.compiler.native_device_activity_binding import derive_activity_binding

            members = {role: dict(path=path, **_content(root / path)) for role, path in _ACTIVITY_MEMBERS.items()}
            _write_record(root / ACTIVITY_BINDING,
                          derive_activity_binding(graph, problem, schedule, bindings, references=members))
        if _content(model) != source_lock or _content(model_file) != model_lock:
            raise ValueError('exported source or CUDA profile changed during native compilation')
        report = dict(schema=SCHEMA, schema_version=1, status=status,
            cxx_standard=20, executable=EXECUTABLE, source=source, dependencies=dependencies,
            default_backend='NO_OP', deployment_authorized=False, actual_device_timing='UNKNOWN',
            files={path.relative_to(root).as_posix(): _content(path)
                   for path in sorted(root.rglob('*')) if path.is_file()})
        if with_activity_calibration:
            report['activity_calibration_abi'] = 1
        if stack is not None:
            report['operator_stack'] = stack
        _write_record(root / 'artifact.json', report)
        if stack is not None:
            from scar.runtime.validate_native_project import verify_native_project

            verify_native_project(root, _content(root / 'artifact.json')['sha256'])
        root.rename(out)
    return report


def replay_cuda_project(root, manifest, member):
    """Replay source geometry, modeled route timing and native lowering offline."""
    def read(name):
        return json.loads(member(f'plan/{name}.json').read_bytes())

    execution = read('execution')
    if (('operator_stack' in manifest) != ('operator_stack' in execution)
            or execution.get('operator_stack') != manifest.get('operator_stack')):
        raise ValueError('CUDA execution disagrees with the retained operator stack receipt')
    declared = json.loads(member(_MODEL_PATH).read_bytes())
    treatment = manifest['status'] == _PLACEMENT_TREATMENT
    qualify, _, replay, status = _pipeline(declared, treatment=treatment)
    policy = dict(actual_device_timing='UNKNOWN', default_backend='NO_OP', deployment_authorized=False)
    if (any(record.get(key) != expected for record in (manifest, execution) for key, expected in policy.items())
            or execution['source'] != manifest['source'] or manifest['status'] != status
            or manifest['executable'] != EXECUTABLE):
        raise ValueError('CUDA candidate has a different source or deployment policy')
    files = manifest['files']
    if (execution['program'] != dict(path=_PROGRAM_PATH, **files[_PROGRAM_PATH])
            or execution['model'] != dict(path=_MODEL_PATH, **files[_MODEL_PATH])):
        raise ValueError('CUDA candidate has a different source graph or declared profile')
    graph = codec.loads(member(_PROGRAM_PATH).read_text())
    inputs = execution['inputs']
    kinds = {row['id']: row['kind'] for row in inputs}
    projection = qualify(read('recipe'), graph=graph, program_ref=execution['program'], input_kinds=kinds)
    if projection != read('projection'):
        raise ValueError('CUDA projection differs from the replayed typed source graph')
    for observed, port in zip(inputs, projection['inputs'], strict=True):
        expected = dict(port)
        if port['kind'] == 'WEIGHT':
            path = f'data/{port["id"]}.f32'
            payload = _content(member(path))
            if payload['size_bytes'] != port['nbytes']:
                raise ValueError('CUDA weight file extent differs from its input port')
            expected.update(path=path, payload=payload)
        if expected != observed:
            raise ValueError('CUDA file input differs from the qualified ordered port')
    problem, schedule, bindings = (read(name) for name in ('problem', 'schedule', 'bindings'))
    if 'operator_stack' in manifest and 'host_resource' in bindings:
        raise ValueError('the NVIDIA operator stack excludes the LibTorch host lane')
    record_name = PLACEMENT_RECORDS.get(status)
    expected_records = set() if record_name is None else {record_name}
    declared_records = {name for name in PLACEMENT_RECORDS.values() if f'plan/{name}.json' in files}
    if declared_records != expected_records:
        raise ValueError('CUDA project placement record differs from its artifact kind')
    placement = None
    replay_options = {}
    if record_name is not None:
        placement = read(record_name)
        replay_options[record_name.replace('-', '_')] = placement
    replay(projection, declared, model_ref=execution['model'], problem=problem, schedule=schedule, bindings=bindings,
           **replay_options)
    abi = execution.get('activity_calibration_abi')
    if (abi != manifest.get('activity_calibration_abi') or (abi is None) != (ACTIVITY_BINDING not in files)
            or (treatment and abi is None)
            or (abi is not None and (type(abi) is not int or abi != 1 or 'host_resource' not in bindings))):
        raise ValueError('CUDA activity calibration ABI differs from its archived binding')
    generated = _native_sources(projection, problem, schedule, bindings, calibrate=abi is not None)
    stack = verify_cuda_operator_stack(root, manifest)
    sources = dict(generated.sources)
    sources['launch.sh'] = _render_launch(inputs, len(projection['outputs']), host_lane='host_resource' in bindings,
        calibrate=abi is not None, stack_sha256=None if stack is None else stack['reference']['sha256'])
    if any(member(name).read_text() != content for name, content in sources.items()):
        raise ValueError('CUDA source differs from its replayed copy and cuBLAS lowering')
    description = json.loads(member('validation/description.json').read_bytes())
    if _describe(root) != description or description['execution_requires_trial'] is not True:
        raise ValueError('CUDA executable description differs from its archived trial contract')
    result = dict(graph_regions=len(graph.regions), graph_operations=len(graph.operations),
                physical_activities=len(problem['operations']), offline_schedule_replayed=True,
                executed_schedule_replayed=False, device_result_verified=False, **policy)
    if stack is not None:
        result['operator_stack'] = stack
    if 'host_resource' in bindings:
        regions = projection['cuda_regions'] if placement is None else placement['selected_cuda_regions']
        result.update(cuda_regions=len(regions), host_provider_calls=description['host_provider_calls'])
    if placement is not None:
        result['device_placements_replayed'] = True
    if treatment:
        result['calibration_placement_ordinal'] = placement['ordinal']
    if abi is not None:
        from scar.codegen.aten_export_cuda_source import ACTIVITY_CALIBRATION_SCHEMA
        from scar.compiler.native_device_activity_binding import replay_activity_binding

        row = json.loads(member(ACTIVITY_BINDING).read_bytes())
        members = {role: dict(path=path, **files[path]) for role, path in _ACTIVITY_MEMBERS.items()}
        replay_activity_binding(row, graph, problem, schedule, bindings, references=members)
        if description.get('activity_calibration') != dict(schema=ACTIVITY_CALIBRATION_SCHEMA, schema_version=1,
                                                            steps=len(row['activities'])):
            raise ValueError('instrumented executable does not describe the bound activity steps')
        result['activity_calibration_bound'] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-exported-aten-cuda', description=__doc__)
    for name in ('model', 'model-sha256', 'cuda-model', 'cuda-model-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--input-shape', action='append', default=[], dest='input_shapes')
    cuda_provider = parser.add_mutually_exclusive_group()
    cuda_provider.add_argument('--cuda-root', help='unlocked CUDA toolkit root, defaults to /usr/local/cuda')
    cuda_provider.add_argument('--operator-stack', help='pinned NVIDIA compiler and CUDA operator stack')
    parser.add_argument('--with-activity-calibration', action='store_true',
                        help='include explicit activity timing alongside the clean hybrid entry')
    parser.add_argument('--placement-ordinal', type=int,
                        help='build a capacity-feasible GPU placement for calibration in CPU-first source order')
    args = parser.parse_args(argv)
    report = compile_exported_aten_cuda(**vars(args))
    print(json.dumps(dict(status=report['status'], executable=str(Path(args.out) / report['executable']),
                          deployment_authorized=report['deployment_authorized'])))
    return 0
