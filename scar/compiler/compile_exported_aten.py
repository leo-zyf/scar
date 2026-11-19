"""Import a functional torch.export graph and deliver a native CPU shape profile."""

import argparse
import hashlib
import json
from math import prod
import os
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory

from scar.analysis.target_graph import TargetHardwareGraph
from scar.codegen.aten_native_graph import _DRIVER, _render
from scar.codegen.aten_provider import _build_interface, _resolve, _write_binding
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_native_recipe import SCHEMA as RECIPE, normalize_recipe
from scar.compiler.plan_aten_native_graph import _host_tensor_budget
from scar.compiler.tensor_physical_problem import tensor_physical_problem
from scar.compiler.tensor_program_graph import straight_line_graph, tensor_type
from scar.ir.program import codec
from scar.ir.record_codec import encode
from scar.optimization.aten_native_schedule import recorded_cpu_domain
from scar.optimization.aten_tensor_lifetime import tensor_footprint


SCHEMA = 'scar.native-exported-aten'
_EXECUTABLE = 'aten-inference'
# This profile binds functional deterministic math. Dispatcher metadata alone
# cannot certify that an arbitrary ATen overload lacks mutable or RNG effects.
_OPERATOR_NAMES = {'aten.mm.default', 'aten.matmul.default', 'aten.add.Tensor', 'aten.sub.Tensor',
                   'aten.mul.Tensor', 'aten.relu.default', 'aten.sigmoid.default', 'aten.tanh.default'}


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
                if str(node.target) not in _OPERATOR_NAMES:
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


def compile_exported_aten(model, *, model_sha256, out, target, tensor_budget_bytes, input_shapes=()):
    import torch

    model = Path(model).resolve(strict=True)
    original = _content(model)
    if original['sha256'] != model_sha256:
        raise ValueError('exported model differs from its declared digest')
    if type(tensor_budget_bytes) is not int or not 0 < tensor_budget_bytes < 2**63:
        raise ValueError('a positive bounded Tensor storage budget is required')
    exported = torch.export.load(model)
    recipe, shapes, weights, symbols = _project(exported, _shape_profiles(input_shapes), torch)
    include, libraries, cxx, flags, links = _build_interface(torch)
    bindings = _resolve([row['schema'] for row in recipe['operations']], torch, include)
    recipe = normalize_recipe(recipe, bindings, torch)
    graph = straight_line_graph(recipe, {name: tensor_type(shape) for name, shape in shapes.items()})
    inventory = TargetHardwareGraph(target)
    _, domain = recorded_cpu_domain(inventory)
    cpu = next((cpu for cpu in domain if cpu in os.sched_getaffinity(0)), None)
    if cpu is None:
        raise ValueError('no recorded CPU is available for the native export')
    target_reference = encode(inventory._snapshot)
    memory = _host_tensor_budget(target, tensor_budget_bytes, (dict(target=target_reference),))
    storage = tensor_footprint(recipe, input_bytes=[4 * prod(row['shape']) for row in recipe['inputs']],
        output_bytes=[4 * prod(shapes[row['id']]) for row in recipe['operations']], candidates=(),
        budget_bytes=memory.capacity)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-export-native-', dir=out.parent) as temporary:
        pending = Path(temporary)
        header = pending / 'include/scar_generated'
        provider = pending / 'src/providers'
        header.mkdir(parents=True)
        provider.mkdir(parents=True)
        _write_binding(provider, bindings)
        (provider / 'link-check.cpp').unlink()
        (header / 'graph.hpp').write_text(_render(recipe, bindings, release_after=storage['release_after']))
        (pending / 'src/main.cpp').write_bytes(_DRIVER.read_bytes())
        plan = pending / 'plan'
        plan.mkdir()
        program = plan / 'program.json'
        program.write_text(codec.dumps(graph) + '\n')
        program_ref = dict(path='plan/program.json', **_content(program))
        (pending / 'data').mkdir()
        inputs = []
        for item in recipe['inputs']:
            name = item['id']
            record = dict(item, kind='WEIGHT' if name in weights else 'INPUT')
            if name in weights:
                file = pending / 'data' / (name + '.f32')
                weights[name].numpy().tofile(file)
                record.update(path=file.relative_to(pending).as_posix(), **_content(file))
            inputs.append(record)
        source = dict(path=str(model), **original)
        execution = dict(schema='scar.native-tensor-entry-plan', schema_version=1, source=source,
            target=target_reference, cpu_ids=[cpu], bindings=bindings, inputs=inputs,
            output_shapes=[shapes[name] for name in recipe['outputs']], symbolic_profile=symbols, storage=storage,
            schedule='SOURCE_DATA_ORDER', program_graph=program_ref, timing_basis='UNMEASURED_MODEL_ENTRY')
        problem = tensor_physical_problem(target, [cpu], budget=memory.capacity, operations=recipe['operations'],
            inputs={row['id'] for row in inputs}, outputs=recipe['outputs'], sizes={k: 4 * prod(s) for k, s in shapes.items()},
            reserved=storage['persistent_input_bytes'], source=program_ref, region='entry')
        (plan / 'problem.json').write_text(json.dumps(problem, indent=2, sort_keys=True) + '\n')
        (plan / 'execution.json').write_text(json.dumps(execution, indent=2, sort_keys=True) + '\n')
        (plan / 'recipe.json').write_text(json.dumps(recipe, indent=2, sort_keys=True) + '\n')
        providers = [dict(path=str(path), **_content(path)) for path in libraries]
        (plan / 'dependencies.json').write_text(json.dumps(providers, indent=2, sort_keys=True) + '\n')
        launch = pending / 'launch.sh'
        command = shlex.join(['env', '-i', 'PATH=/usr/bin:/bin', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                              'taskset', '-c', str(cpu)])
        checks = '\n'.join(f'test "$(sha256sum < "$root/{item["path"]}" | cut -d " " -f 1)" = {item["sha256"]}'
                           for item in inputs if item['kind'] == 'WEIGHT')
        ordinal = iter(range(1, len(inputs) + len(recipe['outputs']) + 1))
        arguments = [f'"$root/{item["path"]}"' if item['kind'] == 'WEIGHT' else '"${' + str(next(ordinal)) + '}"'
                     for item in inputs]
        arguments.extend('"${' + str(next(ordinal)) + '}"' for _ in recipe['outputs'])
        count = len(inputs) - len(weights) + len(recipe['outputs'])
        launch.write_text('#!/bin/sh\nset -eu\nroot=$(dirname -- "$0")\n'
            f'test "$#" -eq {count}\n{checks}\nexec {command} "$root/bin/{_EXECUTABLE}" ' + ' '.join(arguments) + '\n')
        launch.chmod(0o755)
        write_native_project(pending, _EXECUTABLE, include_dirs=('src/providers',), compile_options=(*flags[1:], '-O2'),
            link_options=(links[0],), libraries=tuple(map(str, libraries)), runtime_dirs=(str(libraries[0].parent),),
            install_dirs=(('data', '.'),), install_launcher=True)
        _, dependencies = build_native_project(pending, _EXECUTABLE, cxx, expected_libraries=libraries)
        if _content(model) != original:
            raise ValueError('exported model changed during its native build')
        report = dict(schema=SCHEMA, schema_version=1, executable='bin/' + _EXECUTABLE,
            selected='SERIAL_CPU_LIBTORCH', status='STANDALONE_CPP20_SHAPE_PROFILE', dependencies=dependencies,
            source=source, program_graph=codec.SCHEMA, performance_measured=False,
            files={path.relative_to(pending).as_posix(): _content(path)
                   for path in sorted(pending.rglob('*')) if path.is_file()})
        (pending / 'artifact.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        pending.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-exported-aten', description=__doc__)
    for name in ('model', 'model-sha256', 'out', 'target'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--tensor-budget-bytes', type=int, required=True)
    parser.add_argument('--input-shape', action='append', default=[])
    args = parser.parse_args(argv)
    kwargs = vars(args)
    kwargs['input_shapes'] = kwargs.pop('input_shape')
    report = compile_exported_aten(**kwargs)
    print(json.dumps(dict(status=report['status'], executable=str(Path(args.out) / report['executable']))))
    return 0
