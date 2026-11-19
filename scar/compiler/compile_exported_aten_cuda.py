"""Compile an offline CUDA or hybrid candidate from a functional torch.export."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _build_interface, _resolve
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
ACTIVITY_BINDING = 'plan/activity-binding.json'
_ACTIVITY_MEMBERS = dict(program=_PROGRAM_PATH, problem='plan/problem.json', schedule='plan/schedule.json',
                         bindings='plan/bindings.json', executable=EXECUTABLE, launcher='launch.sh')


def _pipeline(model):
    if isinstance(model, dict) and model.get('schema') == _HYBRID_MODEL:
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


def _render_launch(inputs, output_count, *, host_lane=False, calibrate=False):
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
    return f'''#!/bin/sh
set -eu
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
                               out, input_shapes=(), cuda_root='/usr/local/cuda', with_activity_calibration=False):
    import torch

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
    qualify, plan, _, status = _pipeline(declared)
    if with_activity_calibration and declared['schema'] != _HYBRID_MODEL:
        raise ValueError('activity calibration requires a hybrid CPU and CUDA plan')
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
        problem, schedule, bindings = plan(projection, declared, model_ref=model_ref)
        generated = _native_sources(projection, problem, schedule, bindings, calibrate=with_activity_calibration)
        toolkit = Path(cuda_root).resolve(strict=True)
        libraries = _provider_libraries(toolkit)
        host_lane = 'host_resource' in bindings
        options = ['-O2']
        runtime_dirs = [str(toolkit / 'lib64')]
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
        for name, value in records.items():
            _write_record(root / f'plan/{name}.json', value)
        launch = root / 'launch.sh'
        launch.write_text(_render_launch(inputs, len(projection['outputs']), host_lane=host_lane,
                                         calibrate=with_activity_calibration))
        launch.chmod(0o755)
        for name, content in generated.sources.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        write_native_project(root, Path(EXECUTABLE).name, include_dirs=(str(toolkit / 'include'),),
            compile_options=options, libraries=tuple(map(str, libraries)),
            runtime_dirs=runtime_dirs, install_dirs=(('data', '.'),), install_launcher=True)
        _, dependencies = build_native_project(root, Path(EXECUTABLE).name, compiler, expected_libraries=libraries)
        # Include any CUDA toolkit libraries introduced transitively by the provider.
        lib = toolkit / 'lib64'
        locks = [lib / name for name, path in dependencies.items() if path is not None
                 and Path(path).is_relative_to(toolkit) and (lib / name).resolve() == Path(path)]
        _write_record(root / 'plan/dependencies.json',
                      [dict(path=str(path), **_content(path)) for path in sorted(set(libraries) | set(locks))])
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
        _write_record(root / 'artifact.json', report)
        root.rename(out)
    return report


def replay_cuda_project(root, manifest, member):
    """Replay source geometry, modeled route timing and native lowering offline."""
    def read(name):
        return json.loads(member(f'plan/{name}.json').read_bytes())

    execution = read('execution')
    declared = json.loads(member(_MODEL_PATH).read_bytes())
    qualify, _, replay, status = _pipeline(declared)
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
    replay(projection, declared, model_ref=execution['model'], problem=problem, schedule=schedule, bindings=bindings)
    abi = execution.get('activity_calibration_abi')
    if (abi != manifest.get('activity_calibration_abi') or (abi is None) != (ACTIVITY_BINDING not in files)
            or (abi is not None and (type(abi) is not int or abi != 1 or 'host_resource' not in bindings))):
        raise ValueError('CUDA activity calibration ABI differs from its archived binding')
    generated = _native_sources(projection, problem, schedule, bindings, calibrate=abi is not None)
    sources = dict(generated.sources)
    sources['launch.sh'] = _render_launch(inputs, len(projection['outputs']), host_lane='host_resource' in bindings,
                                          calibrate=abi is not None)
    if any(member(name).read_text() != content for name, content in sources.items()):
        raise ValueError('CUDA source differs from its replayed copy and cuBLAS lowering')
    description = json.loads(member('validation/description.json').read_bytes())
    if _describe(root) != description or description['execution_requires_trial'] is not True:
        raise ValueError('CUDA executable description differs from its archived trial contract')
    result = dict(graph_regions=len(graph.regions), graph_operations=len(graph.operations),
                physical_activities=len(problem['operations']), offline_schedule_replayed=True,
                executed_schedule_replayed=False, device_result_verified=False, **policy)
    if 'host_resource' in bindings:
        result.update(cuda_regions=len(projection['cuda_regions']), host_provider_calls=description['host_provider_calls'])
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
    parser.add_argument('--cuda-root', default='/usr/local/cuda')
    parser.add_argument('--with-activity-calibration', action='store_true',
                        help='include explicit activity timing alongside the clean hybrid entry')
    args = parser.parse_args(argv)
    report = compile_exported_aten_cuda(**vars(args))
    print(json.dumps(dict(status=report['status'], executable=str(Path(args.out) / report['executable']),
                          deployment_authorized=report['deployment_authorized'])))
    return 0
