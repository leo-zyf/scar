"""Import a functional torch.export graph and deliver specialized native CPU inference."""

import argparse
import json
from math import prod
import os
from pathlib import Path
import shlex
from shutil import copytree
from tempfile import TemporaryDirectory

from scar.analysis.target_graph import TargetHardwareGraph
from scar.codegen.aten_export_source import render_exported_entry, render_profile_dispatch, valid_profile_name
from scar.codegen.aten_provider import _build_interface, _resolve, _write_binding
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_native_recipe import SCHEMA as RECIPE, normalize_recipe
from scar.compiler.plan_aten_native_graph import _host_tensor_budget
from scar.compiler.tensor_planning_inputs import calibrate_tensor_body, compare_native_projects, operation_costs, replay_native_comparison, scheduled_tensor_body
from scar.compiler.tensor_planning_inputs import tensor_physical_problem
from scar.compiler.tensor_planning_inputs import straight_line_graph, tensor_type
from scar.ir.implementation_plan import problem_from_dict
from scar.ir.program import codec
from scar.ir.record_codec import encode
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS, eliminate_dead_export_ops
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.aten_native_schedule import recorded_cpu_domain
from scar.optimization.aten_live_values import tensor_footprint


SCHEMA = 'scar.native-exported-aten'
CPU_READY_SCOPE = 'NORMAL_COMPLETION_STANDALONE_CPU_FLOAT32_FILE_ABI'
_EXECUTABLE = 'aten-inference'


def _shape_profiles(text):
    profiles = {}
    for item in text:
        name, found, shape = item.partition('=')
        if not found or not name or name in profiles:
            raise ValueError('each shape profile must be unique and use NAME=DIM,DIM')
        extents = [int(value) for value in shape.split(',')] if shape else []
        if any(extent <= 0 for extent in extents):
            raise ValueError('a shape profile requires positive dimension sizes')
        profiles[name] = extents
    return profiles


def _tensor_geometry(tensor, location, torch):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.dtype != torch.float32
            or tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.storage_offset() != 0):
        raise ValueError(f'{location} must be a contiguous CPU float32 Tensor')
    return list(tensor.shape)


def _bind_shape(node, requested, symbols, torch):
    meta = _tensor_geometry(node.meta['val'], node.name, torch)
    if requested is None:
        if any(type(dim) is not int for dim in meta):
            raise ValueError(f'a concrete --input-shape is required for {node.name}')
        return meta
    if len(requested) != len(meta):
        raise ValueError(f'shape rank differs from the export for {node.name}')
    for dim, size in zip(meta, requested, strict=True):
        if isinstance(dim, torch.SymInt):
            key = dim.node.expr
            if not key.is_Symbol or symbols.setdefault(key, size) != size:
                raise ValueError(f'unsupported or conflicting symbolic dimension on {node.name}')
        elif type(dim) is not int or dim != size:
            raise ValueError(f'static shape extent differs on {node.name}')
    return requested


def _project(exported, profiles, torch):
    signatures = iter(exported.graph_signature.input_specs)
    inputs, operations, weights, results, symbols = [], [], {}, {}, {}
    outputs = None
    users = set()
    with torch.no_grad():
        for node in exported.graph.nodes:
            if node.op == 'placeholder':
                spec = next(signatures)
                if getattr(spec.arg, 'name', None) != node.name:
                    raise ValueError('exported input signature differs from its graph')
                kind = spec.kind.name
                if kind == 'USER_INPUT':
                    users.add(node.name)
                elif kind in ('PARAMETER', 'BUFFER', 'CONSTANT_TENSOR'):
                    table = exported.constants if kind == 'CONSTANT_TENSOR' or spec.persistent is False else exported.state_dict
                    weights[node.name] = table[spec.target].detach()
                else:
                    raise ValueError(f'unsupported export input {kind} at {node.name}')
                shape = _bind_shape(node, profiles.get(node.name), symbols, torch)
                if node.name in weights and _tensor_geometry(weights[node.name], node.name, torch) != shape:
                    raise ValueError('exported parameter geometry differs from its payload')
                inputs.append(dict(id=node.name, shape=shape))
                results[node] = torch.empty(shape, dtype=torch.float32, device='meta')
            elif node.op == 'call_function':
                if str(node.target) not in FUNCTIONAL_EXPORT_SCHEMAS:
                    raise ValueError(f'unsupported functional export operator at {node.name}: {node.target}')
                schema = node.target._schema
                args, positional, keywords = [], [], {}
                for index, param in enumerate(schema.arguments):
                    operand = node.args[index] if index < len(node.args) else node.kwargs.get(param.name, param.default_value)
                    if isinstance(operand, torch.fx.Node) and operand in results:
                        args.append(dict(value=operand.name))
                        value = results[operand]
                    elif type(operand) is int:
                        args.append(dict(scalar=operand))
                        value = operand
                    else:
                        raise ValueError(f'unsupported argument to {node.name}: {param.name}')
                    if param.kwarg_only:
                        keywords[param.name] = value
                    else:
                        positional.append(value)
                if len(node.args) > len(schema.arguments) or set(node.kwargs) - {p.name for p in schema.arguments}:
                    raise ValueError(f'export argument list differs from dispatcher schema at {node.name}')
                result = node.target(*positional, **keywords)
                if (result.device.type != 'meta' or result.dtype != torch.float32 or not result.is_contiguous()
                        or result.storage_offset() != 0 or result.untyped_storage().nbytes() != 4 * result.numel()):
                    raise ValueError(f'native output geometry is outside the file ABI at {node.name}')
                results[node] = result
                operations.append(dict(id=node.name, schema=str(schema), arguments=args))
            elif node.op == 'output':
                values, = node.args
                output_specs = exported.graph_signature.output_specs
                if (type(values) is not tuple or len(values) != len(output_specs)
                        or any(not isinstance(value, torch.fx.Node) or value not in results
                               or spec.kind.name != 'USER_OUTPUT' or spec.arg.name != value.name
                               for value, spec in zip(values, output_specs, strict=True))):
                    raise ValueError('the native export requires only ordinary Tensor results and no state mutations')
                outputs = [value.name for value in values]
            else:
                raise ValueError(f'unsupported export node {node.op} at {node.name}')
    if next(signatures, None) is not None or not outputs or set(profiles) - users:
        raise ValueError('export inputs, shape profiles or outputs differ from the functional graph')
    for symbol, bounds in exported.range_constraints.items():
        if symbol not in symbols or not bounds.lower <= symbols[symbol] <= bounds.upper:
            raise ValueError(f'exported symbolic constraint does not hold for {symbol}')
    recipe = dict(schema=RECIPE, schema_version=1, inputs=inputs, operations=operations, outputs=outputs)
    shapes = {node.name: list(tensor.shape) for node, tensor in results.items()}
    return recipe, shapes, weights, {str(key): value for key, value in symbols.items()}


def _calibration_paths(arguments, inputs):
    paths = {}
    for argument in arguments:
        name, separator, path = argument.partition('=')
        if not separator or not name or not path or name in paths:
            raise ValueError('use --calibration-input NAME=FILE once per user Tensor')
        paths[name] = Path(path).resolve(strict=True)
    user = [row for row in inputs if row['kind'] == 'INPUT']
    if set(paths) != {row['id'] for row in user}:
        raise ValueError('native calibration files must match all original user inputs')
    return [paths[row['id']] for row in user]


def _profile_arguments(arguments):
    groups = {}
    for argument in arguments:
        name, separator, item = argument.partition(':')
        if not separator or not item or not valid_profile_name(name):
            raise ValueError('use PROFILE:NAME=VALUE for native inference profile arguments')
        groups.setdefault(name, []).append(item)
    return groups


def _write_record(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def _render_launch(inputs, output_count, placements):
    command = shlex.join(['env', '-i', 'PATH=/usr/bin:/bin', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                          'taskset', '-c'])
    weights = [item for item in inputs if item['kind'] == 'WEIGHT']
    checks = '\n'.join(f'test "$(sha256sum < "$root/{item["path"]}" | cut -d " " -f 1)" = {item["sha256"]}'
                       for item in weights)
    ordinal = iter(range(1, len(inputs) + output_count + 1))
    arguments = [f'"$root/{item["path"]}"' if item['kind'] == 'WEIGHT' else '"${' + str(next(ordinal)) + '}"'
                 for item in inputs]
    arguments.extend('"${' + str(next(ordinal)) + '}"' for _ in range(output_count))
    count = len(inputs) - len(weights) + output_count
    setup = f'test "$#" -eq {count}\n'
    profile_argument = ''
    if None not in placements:
        setup = f'test "$#" -eq {count + 1}\nprofile=$1\nshift\n'
        profile_argument = '"$profile" '
    masks = {name: ','.join(map(str, cpus)) for name, cpus in placements.items()}
    if len(set(masks.values())) == 1:
        mask = next(iter(masks.values()))
    else:
        setup += 'case "$profile" in\n'
        setup += ''.join(f'  {name}) mask={mask} ;;\n' for name, mask in masks.items())
        setup += '  *) echo "unknown native inference profile" >&2; exit 2 ;;\nesac\n'
        mask = '"$mask"'
    return ('#!/bin/sh\nset -eu\nroot=$(dirname -- "$0")\nprobe=\n'
        'if [ "${1:-}" = --calibrate ]; then probe=--calibrate; shift; fi\n'
        f'{setup}{checks}\nexec {command} {mask} "$root/bin/{_EXECUTABLE}" {profile_argument}$probe '
        + ' '.join(arguments) + '\n')


def _prepare_profile(root, label, recipe, shapes, symbols, weight_records, bindings, source, cpu,
                     target_reference, capacity, simplify_normal_completion):
    relative = Path('plan') if label is None else Path('plan/profiles') / label
    plan = root / relative
    plan.mkdir(parents=True)
    graph = straight_line_graph(recipe, {name: tensor_type(shape) for name, shape in shapes.items()})
    program = plan / 'program.json'
    program.write_text(codec.dumps(graph) + '\n')
    if simplify_normal_completion:
        _write_record(plan / 'source-recipe.json', recipe)
        recipe, receipt = eliminate_dead_export_ops(recipe)
        _write_record(plan / 'rewrite.json', receipt)
    storage = tensor_footprint(recipe, input_bytes=[4 * prod(row['shape']) for row in recipe['inputs']],
        output_bytes=[4 * prod(shapes[row['id']]) for row in recipe['operations']], candidates=(),
        budget_bytes=capacity)
    inputs = [dict(item, kind='WEIGHT', **weight_records[item['id']]) if item['id'] in weight_records
              else dict(item, kind='INPUT') for item in recipe['inputs']]
    execution = dict(schema='scar.native-tensor-entry-plan', schema_version=1, source=source,
        target=target_reference, cpu_ids=[cpu], bindings=bindings, inputs=inputs,
        output_shapes=[shapes[name] for name in recipe['outputs']], symbolic_profile=symbols, storage=storage,
        schedule='SOURCE_DATA_ORDER', program_graph=dict(path=(relative / 'program.json').as_posix(), **_content(program)),
        timing_basis='UNMEASURED_MODEL_ENTRY')
    sources = render_exported_entry(recipe, bindings, storage['release_after'], cpu, profile=label)
    for name, content in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _write_record(plan / 'recipe.json', recipe)
    return plan, execution, recipe, shapes, sources


def _plan_profile(root, label, plan, execution, recipe, shapes, sources, target, capacity, calibration_input):
    cpu, = execution['cpu_ids']
    inputs = execution['inputs']
    costs = None
    if calibration_input:
        samples = _calibration_paths(calibration_input, inputs)
        receipts = [dict(path=str(path), **_content(path)) for path in samples]
        arguments = [str(path) for path in samples]
        if label is not None:
            arguments.insert(0, label)
        with TemporaryDirectory(prefix='.scar-model-samples-') as scratch:
            observed = calibrate_tensor_body(root, arguments, scratch, repeats=1, outputs=len(recipe['outputs']))
        if receipts != [dict(path=str(path), **_content(path)) for path in samples]:
            raise ValueError('model calibration samples changed during execution')
        observed.update(inputs=receipts, cpu_ids=[cpu])
        calibration = root / 'validation/calibration.json' if label is None else plan / 'calibration.json'
        _write_record(calibration, observed)
        evidence = dict(path=calibration.relative_to(root).as_posix(), **_content(calibration))
        costs = operation_costs(observed, recipe['operations'], evidence)
        execution['timing_basis'] = 'MEASURED_CPU_BODY_ONE_FIXED_SHAPE'
    problem = tensor_physical_problem(target, [cpu], budget=capacity, operations=recipe['operations'],
        inputs={row['id'] for row in inputs}, outputs=recipe['outputs'], sizes={k: 4 * prod(s) for k, s in shapes.items()},
        reserved=execution['storage']['persistent_input_bytes'], source=execution['program_graph'], region='entry', costs=costs)
    _write_record(plan / 'problem.json', problem)
    _write_record(plan / 'execution.json', execution)
    if costs is not None:
        schedule = construct_implementations(problem_from_dict(problem))
        scheduled = scheduled_tensor_body(problem, schedule, recipe)
        if render_exported_entry(scheduled, execution['bindings'], scheduled['release_after'], cpu, profile=label) != sources:
            raise ValueError('calibrated model source differs from its verified schedule lowering')
        _write_record(plan / 'schedule.json', schedule)


def _build_profiles(root, prepared, cpp_sources, toolchain, placements, *, require_cpu_only=False):
    _, libraries, cxx, flags, links = toolchain
    first = next(iter(prepared.values()))
    launch = root / 'launch.sh'
    launch.write_text(_render_launch(first[1]['inputs'], len(first[2]['outputs']), placements))
    launch.chmod(0o755)
    threads = ('-pthread',) if any(len(cpus) > 1 for cpus in placements.values()) else ()
    write_native_project(root, _EXECUTABLE, sources=cpp_sources,
        include_dirs=('src/providers', 'src/runtime') if threads else ('src/providers',),
        compile_options=(*flags[1:], '-O2', *threads), link_options=(links[0], *threads),
        libraries=tuple(map(str, libraries)), runtime_dirs=(str(libraries[0].parent),),
        install_dirs=(('data', '.'),), install_launcher=True)
    _, dependencies = build_native_project(root, _EXECUTABLE, cxx, expected_libraries=libraries,
                                            require_cpu_only=require_cpu_only)
    return dependencies


def _emit_ready(root, label, prepared, placement):
    _, execution, recipe, _, _ = prepared[label]
    files = render_exported_entry(recipe, execution['bindings'], execution['storage']['release_after'],
                                  placement['cpu_ids'][0], profile=label, placement=placement)
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(source)


def _select_ready_profiles(root, prepared, cpp_sources, toolchain, target, capacity, calibration,
                           serial_dependencies, *, require_cpu_only=False):
    from scar.compiler.aten_export_placement import plan_export_cpu

    with TemporaryDirectory(prefix='.scar-export-placement-', dir=root.parent) as temporary:
        scratch = Path(temporary)
        placements, candidate_args = {}, {}
        for index, (label, (plan, execution, recipe, shapes, _)) in enumerate(prepared.items()):
            sample_paths = _calibration_paths(calibration[label], execution['inputs'])
            calibration_name = root / 'validation/calibration.json' if label is None else plan / 'calibration.json'
            observation = json.loads(calibration_name.read_bytes())
            if observation['inputs'] != [dict(path=str(path), **_content(path)) for path in sample_paths]:
                raise ValueError('CPU ready planning requires the exact serial calibration inputs')
            by_port = iter(sample_paths)
            inputs = [root / row['path'] if row['kind'] == 'WEIGHT' else next(by_port)
                      for row in execution['inputs']]
            serial_cpu, = execution['cpu_ids']
            ready = plan / 'ready'
            placement, evidence, problem, schedule = plan_export_cpu(recipe, target=target,
                input_paths=inputs, tensor_budget_bytes=capacity, bindings=execution['bindings'],
                toolchain=toolchain, build_root=scratch / f'probe-{index}', origin=execution['program_graph'],
                serial_cpu=serial_cpu, result_bytes=[4 * prod(shapes[row['id']]) for row in recipe['operations']],
                evidence_path=(ready / 'proposal.json').relative_to(root).as_posix())
            ready.mkdir()
            _write_record(ready / 'proposal.json', dict(placement=placement, evidence=evidence))
            if problem is not None:
                _write_record(ready / 'problem.json', problem)
                _write_record(ready / 'schedule.json', schedule)
            if placement['mode'] == 'READY_DAG':
                placements[label] = placement
                candidate_args[label] = sample_paths, observation

        serial_masks = {label: items[1]['cpu_ids'] for label, items in prepared.items()}
        # When a profile drops out, re-measure the reduced package before
        # publishing it. Accepted evidence then binds the exact shipped ELF.
        while placements:
            with TemporaryDirectory(prefix='candidate-', dir=scratch) as trial_directory:
                candidate = Path(trial_directory)
                for directory in ('src', 'include'):
                    copytree(root / directory, candidate / directory)
                copytree(root / 'data', candidate / 'data', copy_function=os.link)
                for label, placement in placements.items():
                    _emit_ready(candidate, label, prepared, placement)
                masks = dict(serial_masks)
                masks.update({label: placement['cpu_ids'] for label, placement in placements.items()})
                dependencies = _build_profiles(candidate, prepared, cpp_sources, toolchain, masks,
                                               require_cpu_only=require_cpu_only)
                projects = {'SERIAL': root, 'READY_DAG': candidate}
                tested_files = {mode: {name: _content(project / name) for name in ('bin/' + _EXECUTABLE, 'launch.sh')}
                                for mode, project in projects.items()}
                choices = {}
                for label, placement in placements.items():
                    plan, _, recipe, _, _ = prepared[label]
                    paths, observation = candidate_args[label]
                    arguments = ([label] if label is not None else []) + list(map(str, paths))
                    comparison = compare_native_projects(projects, arguments, scratch, observation['outputs'],
                                                         output_count=len(recipe['outputs']))
                    if observation['inputs'] != [dict(path=str(path), **_content(path)) for path in paths]:
                        raise ValueError('CPU ready comparison inputs changed during benchmarking')
                    comparison['tested_files'] = tested_files
                    _write_record(plan / 'ready/entry-comparison.json', comparison)
                    if replay_native_comparison(comparison, observation['outputs']) == 'READY_DAG':
                        choices[label] = placement
                if choices.keys() != placements.keys():
                    placements = choices
                    continue
                for label, placement in choices.items():
                    _emit_ready(root, label, prepared, placement)
                    plan, execution, _, _, _ = prepared[label]
                    ready = plan / 'ready'
                    for name in ('problem.json', 'schedule.json'):
                        (plan / name).rename(ready / ('serial-' + name))
                        (plan / name).write_bytes((ready / name).read_bytes())
                    execution.update(cpu_ids=placement['cpu_ids'], schedule='DEPENDENCY_READY_CPU_TEAM',
                        timing_basis='MEASURED_CPU_LANE_SERVICES_AND_ENTRY_COMPARISON', placement=placement)
                    _write_record(plan / 'execution.json', execution)
                for name in ('CMakeLists.txt', 'launch.sh', 'bin/' + _EXECUTABLE, 'validation/build.log',
                             'validation/ldd.txt'):
                    (candidate / name).replace(root / name)
                return dependencies, choices
        return serial_dependencies, {}


def compile_exported_aten(model, *, model_sha256, out, target, tensor_budget_bytes, input_shapes=(),
                          calibration_input=(), shape_profiles=(), simplify_normal_completion=False,
                          plan_cpu_ready=False, require_cpu_only=False):
    import torch

    model = Path(model).resolve(strict=True)
    original = _content(model)
    if original['sha256'] != model_sha256:
        raise ValueError('exported model differs from its declared digest')
    if type(tensor_budget_bytes) is not int or not 0 < tensor_budget_bytes < 2**63:
        raise ValueError('a positive bounded Tensor storage budget is required')
    if shape_profiles:
        groups = _profile_arguments(shape_profiles)
        samples = _profile_arguments(calibration_input)
        if input_shapes or len(groups) < 2 or set(samples) - groups.keys():
            raise ValueError('use two or more named profiles with their own calibration inputs')
    else:
        groups = {None: input_shapes}
        samples = {None: calibration_input}
    if plan_cpu_ready and not all(samples.get(label) for label in groups):
        raise ValueError('CPU ready planning requires calibration files for each inference profile')
    exported = torch.export.load(model)
    projected = {label: _project(exported, _shape_profiles(group), torch) for label, group in groups.items()}
    first_recipe, _, weights, _ = next(iter(projected.values()))
    if len({tuple(sorted(item[3].items())) for item in projected.values()}) != len(projected):
        raise ValueError('native profiles must select distinct symbolic assignments')
    toolchain = _build_interface(torch)
    include, libraries = toolchain[:2]
    bindings = _resolve([row['schema'] for row in first_recipe['operations']], torch, include)
    inventory = TargetHardwareGraph(target)
    _, domain = recorded_cpu_domain(inventory)
    cpu = next((cpu for cpu in domain if cpu in os.sched_getaffinity(0)), None)
    if cpu is None:
        raise ValueError('no recorded CPU is available for the native export')
    target_reference = encode(inventory._snapshot)
    memory = _host_tensor_budget(target, tensor_budget_bytes, (dict(target=target_reference),))
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-export-native-', dir=out.parent) as temporary:
        pending = Path(temporary)
        provider = pending / 'src/providers'
        provider.mkdir(parents=True)
        _write_binding(provider, bindings)
        (provider / 'link-check.cpp').unlink()
        (pending / 'data').mkdir()
        weight_records = {}
        for name, tensor in weights.items():
            file = pending / 'data' / (name + '.f32')
            tensor.numpy().tofile(file)
            weight_records[name] = dict(path=file.relative_to(pending).as_posix(), **_content(file))
        source = dict(path=str(model), **original)
        prepared = {}
        for label, (recipe, shapes, _, symbols) in projected.items():
            recipe = normalize_recipe(recipe, bindings, torch)
            prepared[label] = _prepare_profile(pending, label, recipe, shapes, symbols, weight_records,
                bindings, source, cpu, target_reference, memory.capacity, simplify_normal_completion)
        sources = ['src/main.cpp']
        if shape_profiles:
            (pending / sources[0]).write_text(render_profile_dispatch(list(groups)))
            sources.extend(f'src/profiles/{name}.cpp' for name in groups)
        providers = [dict(path=str(path), **_content(path)) for path in libraries]
        _write_record(pending / 'plan/dependencies.json', providers)
        dependencies = _build_profiles(pending, prepared, sources, toolchain, dict.fromkeys(groups, [cpu]),
                                       require_cpu_only=require_cpu_only)
        for label, artifacts in prepared.items():
            _plan_profile(pending, label, *artifacts, target, memory.capacity, samples.get(label))
        ready = {}
        if plan_cpu_ready:
            dependencies, ready = _select_ready_profiles(pending, prepared, sources, toolchain, target,
                memory.capacity, samples, dependencies, require_cpu_only=require_cpu_only)
        if _content(model) != original:
            raise ValueError('exported model changed during its native build')
        report = dict(schema=SCHEMA, schema_version=1, executable='bin/' + _EXECUTABLE,
            selected='SERIAL_CPU_LIBTORCH', status='STANDALONE_CPP20_SHAPE_PROFILE', dependencies=dependencies,
            source=source, program_graph=codec.SCHEMA, performance_measured=False,
            body_calibrated=all(bool(samples.get(label)) for label in groups),
            simplified_normal_completion=simplify_normal_completion,
            files={path.relative_to(pending).as_posix(): _content(path)
                   for path in sorted(pending.rglob('*')) if path.is_file()})
        if shape_profiles:
            report.update(status='STANDALONE_CPP20_SHAPE_FAMILY', profiles=list(groups))
        if plan_cpu_ready:
            report.update(cpu_ready_planned=True, cpu_ready_scope=CPU_READY_SCOPE,
                selected='CPU_PROFILE_PLACEMENTS' if shape_profiles else
                    ('READY_DAG_CPU_LIBTORCH' if ready else report['selected']),
                performance_measured=any((items[0] / 'ready/entry-comparison.json').is_file()
                                         for items in prepared.values()))
        if require_cpu_only:
            from scar.runtime.aten_provider_identity import qualify_cpu_dependencies

            report['cpu_linkage'] = qualify_cpu_dependencies(dependencies)
        _write_record(pending / 'artifact.json', report)
        pending.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-exported-aten', description=__doc__)
    for name in ('model', 'model-sha256', 'out', 'target'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--tensor-budget-bytes', type=int, required=True)
    parser.add_argument('--input-shape', action='append', default=[])
    parser.add_argument('--calibration-input', action='append', default=[])
    parser.add_argument('--shape-profile', action='append', default=[], metavar='PROFILE:NAME=DIM,DIM',
                        help='register an explicit shape under a native dispatch name')
    parser.add_argument('--simplify-normal-completion', action='store_true',
                        help='apply structural rules under the native normal-completion contract')
    parser.add_argument('--plan-cpu-ready', action='store_true',
                        help='choose measured native parallel placement under the normal-completion contract')
    parser.add_argument('--require-cpu-only', action='store_true',
                        help='require a qualified CPU-only ELF dependency closure')
    args = parser.parse_args(argv)
    kwargs = vars(args)
    kwargs['input_shapes'] = kwargs.pop('input_shape')
    kwargs['shape_profiles'] = kwargs.pop('shape_profile')
    report = compile_exported_aten(**kwargs)
    print(json.dumps(dict(status=report['status'], executable=str(Path(args.out) / report['executable']))))
    return 0
