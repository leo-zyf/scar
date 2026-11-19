"""Lower an explicit native tensor DAG to ordered C++20 LibTorch calls."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _build_interface, _dependencies, _resolve
from scar.codegen.aten_team_graph_source import render_team_calls
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_native_recipe import normalize_recipe
from scar.ir.record_codec import loads
from scar.optimization.aten_tensor_lifetime import tensor_releases


_DRIVER = Path(__file__).with_name('aten_native_cli_v1.cpp.in')
_PAIR_HEADER = Path(__file__).with_name('aten_native_pair_v1.h')
_PAIR_DRIVER = Path(__file__).with_name('aten_native_graph_ab_v1.cpp.in')
_TEAM_HEADER = Path(__file__).with_name('aten_native_team_v1.h')
_TEAM_DRIVER = Path(__file__).with_name('aten_native_team_ab_v1.cpp.in')
_READY_HEADER = Path(__file__).with_name('aten_ready_dag_v1.h')
_NUMA_HEADER = Path(__file__).with_name('aten_numa_pages_v1.h')
_EXECUTABLE = 'aten-native-graph'


def _read_declared(path, digest):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        contents = stream.read(1_000_001)
    if len(contents) > 1_000_000 or sha256(contents).hexdigest() != digest:
        raise ValueError('native graph input exceeds its bound or differs from its declared digest: ' + str(path))
    return path, loads(contents), dict(path=str(path), sha256=digest, size_bytes=len(contents))


def _installed_linkage(linkage, torch):
    include, libraries, cxx, compile_flags, link_flags = _build_interface(torch)
    if (linkage['schema'], linkage['schema_version'], linkage['status'], linkage['cxx_standard']) != (
            'scar.aten-provider-linkage', 1, 'EXTERNAL_CPP_CALL_LINKED', 20):
        raise ValueError('linked C++20 ATen provider required')
    provider = linkage['provider']
    if (provider['name'], provider['version'], provider['revision']) != (
            'LibTorch', torch.__version__, torch.version.git_version):
        raise ValueError('installed LibTorch version and revision differ from declared linkage')
    if provider['libraries'] != [dict(path=str(path), **_content(path)) for path in libraries]:
        raise ValueError('installed LibTorch library images differ from declared linkage')
    operators = _resolve([row['schema'] for row in linkage['operators']], torch, include)
    if linkage['operators'] != operators:
        raise ValueError('installed LibTorch overloads differ from declared linkage')
    return operators, libraries, cxx, compile_flags, link_flags


def _argument(expression, values):
    if 'value' in expression:
        return values[expression['value']]
    scalar = expression['scalar']
    literal = '(-9223372036854775807LL - 1)' if scalar == -(1 << 63) else str(scalar)
    return 'c10::Scalar(std::int64_t{' + literal + '})'


def _render(recipe, operators, schedule=None, *, release_after, ready_dag=False):
    entries = {row['schema']: row['entry'] for row in operators}
    values = {}
    shapes, calls = [], []
    for index, item in enumerate(recipe['inputs']):
        dims = ', '.join(map(str, item['shape']))
        shapes.append(f"inline constexpr std::array<std::int64_t, {len(item['shape'])}> shape_{index}{{{dims}}};")
        values[item['id']] = f'inputs[{index}]'
    for index, op in enumerate(recipe['operations']):
        target = f'value_{index}'
        arguments = ', '.join(_argument(arg, values) for arg in op['arguments'])
        calls.append((target, f"{entries[op['schema']]}({arguments})"))
        values[op['id']] = target
    input_shapes = ', '.join(f'shape_{i}' for i in range(len(shapes)))
    outputs = ', '.join(values[output] for output in recipe['outputs'])
    shape_definitions = '\n'.join(shapes)
    releases = [[f'{calls[dying][0]}.reset();' for dying in row] for row in release_after]
    ordered = []
    for i, (value, expr) in enumerate(calls):
        ordered.append(f'    at::Tensor {value} = {expr};')
        ordered.extend(f'    {statement}' for statement in releases[i])
    operator_calls = '\n'.join(ordered)
    pair_include = '' if schedule is None else f'#include "{_PAIR_HEADER.name}"\n'
    pair_support = ''
    if schedule is not None and 'legal_window_widths' in schedule:
        pair_include = f'#define SCAR_ATEN_TEAM 1\n#include "{_TEAM_HEADER.name}"\n'
        producers = {op['id']: i for i, op in enumerate(recipe['operations'])}
        dependencies = sorted({(producers[arg['value']], consumer)
                               for consumer, op in enumerate(recipe['operations']) for arg in op['arguments']
                               if arg.get('value') in producers})
        pair_support = render_team_calls(calls, releases, outputs, schedule['legal_window_widths'],
                                         dependencies=dependencies)
        if ready_dag:
            from scar.codegen.aten_team_graph_source import render_ready_support

            pair_include += f'#define SCAR_ATEN_READY_DAG 2\n#include "{_READY_HEADER.name}"\n'
            exported = tuple(producers[value] for value in recipe['outputs'])
            pair_support += render_ready_support(calls, outputs, dependencies, exported_indices=exported)
    elif schedule is not None:
        starts = {row['pair_indices'][0] for row in schedule['windows']}
        legal = ', '.join('true' if i in starts else 'false' for i in range(len(calls)))
        parallel = [f'    at::Tensor {value};' for value, _ in calls]
        for i, (value, expr) in enumerate(calls):
            observed = f'observe({i}, [&] {{ return {expr}; }})'
            indent = '    '
            after_window = i - 1 in starts
            if after_window:
                parallel.append(f'    if (!windows[{i - 1}]) {{')
                indent += '    '
            if i in starts:
                right, right_expr = calls[i + 1]
                right_observed = f'observe({i + 1}, [&] {{ return {right_expr}; }})'
                parallel.extend((f'{indent}if (windows[{i}]) {{',
                    f'{indent}    pair.run([&] {{ {value} = {observed}; }},',
                    f'{indent}             [&] {{ {right} = {right_observed}; }});'))
                # Both readers have completed before either release row applies.
                parallel.extend(f'{indent}    {statement}' for row in releases[i:i + 2] for statement in row)
                parallel.extend((f'{indent}}} else {{', f'{indent}    {value} = {observed};'))
                parallel.extend(f'{indent}    {statement}' for statement in releases[i])
                parallel.append(f'{indent}}}')
            else:
                parallel.append(f'{indent}{value} = {observed};')
                parallel.extend(f'{indent}{statement}' for statement in releases[i])
            if after_window:
                parallel.append('    }')
        parallel_calls = '\n'.join(parallel)
        timed = []
        for i, (value, expr) in enumerate(calls):
            timed.append(f'    at::Tensor {value} = observe({i}, [&] {{ return {expr}; }});')
            timed.extend(f'    {statement}' for statement in releases[i])
        timed_calls = '\n'.join(timed)
        pair_support = f'''
inline constexpr std::size_t operation_count = {len(calls)};
using WindowSelection = std::array<bool, operation_count>;
inline constexpr WindowSelection legal_window_starts{{{legal}}};

template <class Observer>
inline std::array<at::Tensor, output_count> run_observed(std::span<const at::Tensor> inputs, Observer&& observe) {{
    validate_inputs(inputs);
{timed_calls}
    return {{{outputs}}};
}}

template <class Observer>
inline std::array<at::Tensor, output_count> run_parallel_observed(
        std::span<const at::Tensor> inputs, CpuPair& pair, const WindowSelection& windows, Observer&& observe) {{
    validate_inputs(inputs);
    for (std::size_t i = 0; i < operation_count; ++i) {{
        if (windows[i] && (!legal_window_starts[i] || (i > 0 && windows[i - 1]))) {{
            throw std::invalid_argument("native graph has an unqualified or overlapping window");
        }}
    }}
{parallel_calls}
    return {{{outputs}}};
}}

inline std::array<at::Tensor, output_count> run_parallel(
        std::span<const at::Tensor> inputs, CpuPair& pair, const WindowSelection& windows) {{
    return run_parallel_observed(inputs, pair, windows, [](std::size_t, auto&& call) {{ return call(); }});
}}
'''
    # SSA temporaries below carry Tensor handles; all numerical work stays in the provider.
    return f'''#pragma once
#include "operators.hpp"
{pair_include}#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace scar_aten_graph {{
inline constexpr int abi_version = 1;
inline constexpr std::size_t input_count = {len(shapes)};
inline constexpr std::size_t output_count = {len(recipe['outputs'])};
{shape_definitions}
inline constexpr std::array<std::span<const std::int64_t>, input_count> input_shapes{{{input_shapes}}};

inline void validate_inputs(std::span<const at::Tensor> inputs) {{
    if (inputs.size() != input_count) {{
        throw std::invalid_argument("native graph input count differs");
    }}
    for (std::size_t index = 0; index < input_count; ++index) {{
        const auto& tensor = inputs[index];
        const auto shape = input_shapes[index];
        if (!tensor.defined() || !tensor.device().is_cpu() || tensor.layout() != c10::kStrided
            || tensor.scalar_type() != at::kFloat || !tensor.is_contiguous() || tensor.requires_grad()
            || tensor.is_conj() || tensor.is_neg()) {{
            throw std::invalid_argument("native graph input tensor differs from its CPU float32 contract");
        }}
        const auto actual = tensor.sizes();
        if (actual.size() != shape.size() || (!shape.empty() && !std::equal(shape.begin(), shape.end(), actual.begin()))) {{
            throw std::invalid_argument("native graph input shape differs");
        }}
    }}
}}

inline std::array<at::Tensor, output_count> run(std::span<const at::Tensor> inputs) {{
    validate_inputs(inputs);
{operator_calls}
    return {{{outputs}}};
}}
{pair_support}
}}  // namespace scar_aten_graph
'''


def compile_aten_native_graph(recipe, *, recipe_sha256, linkage, linkage_sha256, out,
                              concurrency_contract=None, concurrency_sha256=None, target=None, cpu_ids=None,
                              allow_cross_numa=False, ready_dag=False):
    import torch

    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    _, recipe, recipe_receipt = _read_declared(recipe, recipe_sha256)
    linkage_path, linked, linkage_receipt = _read_declared(linkage, linkage_sha256)
    operators, libraries, cxx, compile_flags, link_flags = _installed_linkage(linked, torch)
    recipe = normalize_recipe(recipe, operators, torch)
    release_after = tensor_releases(recipe)
    schedule = None
    contract_receipt = None
    requested = (concurrency_contract, concurrency_sha256, target, cpu_ids)
    if allow_cross_numa and (cpu_ids is None or len(cpu_ids) != 3):
        raise ValueError('cross-NUMA team placement requires three CPU IDs')
    if any(item is not None for item in requested):
        if any(item is None for item in requested):
            raise ValueError('CPU scheduling requires its contract, digest, target graph and CPU IDs')
        from scar.optimization.aten_native_schedule import propose_cpu_windows
        from scar.optimization.aten_native_team import propose_cpu_team

        _, contract, contract_receipt = _read_declared(concurrency_contract, concurrency_sha256)
        if len(cpu_ids) == 3:
            schedule = propose_cpu_team(recipe, contract=contract, target=target, cpu_ids=cpu_ids,
                                        allow_cross_numa=allow_cross_numa)
        else:
            schedule = propose_cpu_windows(recipe, contract=contract, target=target, cpu_ids=cpu_ids)
    team = schedule is not None and 'legal_window_widths' in schedule
    if ready_dag:
        from scar.optimization.aten_native_schedule import _declared_operators

        if not team or len({lane['numa_node'] for lane in schedule['lanes']}) != 1:
            raise ValueError('ready DAG requires a qualified CPU team on one NUMA node')
        declared = _declared_operators(contract)
        if any(op['schema'] not in declared for op in recipe['operations']):
            raise ValueError('ready DAG requires exact concurrency qualification of every reordered operator')
    operator_path = linkage_path.with_name('operators.hpp')
    command_env = {key: val for key, val in os.environ.items() if not key.startswith('LD_')}
    command = [*cxx, *compile_flags, '-O2']
    if team:
        command.extend(('-pthread', 'team.cpp'))
    elif schedule is not None:
        command.extend(('-pthread', 'pair.cpp'))
    else:
        command.append('entry.cpp')
    command.extend((*link_flags, '-o', _EXECUTABLE))
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-native-graph-', dir=out.parent) as temporary:
        stage = Path(temporary)
        header = stage / 'operators.hpp'
        header.write_bytes(operator_path.read_bytes())
        if _content(header) != linked['artifacts']['operators.hpp']:
            raise ValueError('retained ATen C++ binding differs from its linkage receipt')
        (stage / 'graph.hpp').write_text(_render(recipe, operators, schedule, release_after=release_after,
                                                ready_dag=ready_dag))
        (stage / 'entry.cpp').write_bytes(_DRIVER.read_bytes())
        if team:
            (stage / _TEAM_HEADER.name).write_bytes(_TEAM_HEADER.read_bytes())
            (stage / _NUMA_HEADER.name).write_bytes(_NUMA_HEADER.read_bytes())
            (stage / 'team.cpp').write_bytes(_TEAM_DRIVER.read_bytes())
            if ready_dag:
                (stage / _READY_HEADER.name).write_bytes(_READY_HEADER.read_bytes())
        elif schedule is not None:
            (stage / _PAIR_HEADER.name).write_bytes(_PAIR_HEADER.read_bytes())
            (stage / _NUMA_HEADER.name).write_bytes(_NUMA_HEADER.read_bytes())
            (stage / 'pair.cpp').write_bytes(_PAIR_DRIVER.read_bytes())
        (stage / 'recipe.json').write_text(json.dumps(recipe, indent=2, sort_keys=True) + '\n')
        subprocess.run(command, cwd=stage, check=True, env=command_env)
        listing = subprocess.run(['ldd', './' + _EXECUTABLE], cwd=stage, check=True,
                                 env=command_env, capture_output=True, text=True).stdout
        dependencies = _dependencies(listing)
        if (any(dependencies.get(path.name) != str(path.resolve()) for path in libraries)
                or any('libpython' in name or 'libtorch_python' in name for name in dependencies)):
            raise ValueError('linked standalone native graph requires the declared provider without Python')
        script = stage / 'launch.sh'
        launcher = 'env -i PATH=/usr/bin:/bin'
        suffix = ''
        if schedule is not None:
            caller = str(schedule['cpus'][0])
            launcher += ' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c ' + caller
            suffix = ' ' + shlex.join(['--run-serial', caller])
        script.write_text('#!/bin/sh\nset -eu\nexec ' + launcher + ' "$(dirname -- "$0")/'
                          + _EXECUTABLE + '"' + suffix + ' "$@"\n')
        script.chmod(0o755)
        (stage / 'ldd.txt').write_text(listing)
        receipt = dict(schema='scar.aten-native-graph-delivery', schema_version=1, cxx_standard=20,
            native_abi_version=1, graph_scope='DECLARED_NATIVE_CPU_GRAPH', recipe=recipe_receipt,
            linkage=linkage_receipt, provider=linked['provider'], dependencies=dependencies,
            operation_order=[op['id'] for op in recipe['operations']], implementation='ORDERED_PROVIDER_CALLS',
            compiler_command=command, source_replacement_authorized=False, performance_measured=False,
            tensor_lifetime=dict(profile='LAST_CONSUMER_OR_EXPORT_V1', release_after=release_after),
            files={path.name: _content(path) for path in sorted(stage.iterdir())})
        if team:
            receipt.update(cpu_team=schedule, concurrency_contract=contract_receipt,
                           implementation='ORDERED_DEFAULT_WITH_DECLARED_CPU_TEAM_TRIAL')
        elif schedule is not None:
            receipt.update(cpu_windows=schedule, concurrency_contract=contract_receipt,
                           implementation='ORDERED_DEFAULT_WITH_DECLARED_CPU_WINDOWS_TRIAL')
        if ready_dag:
            receipt['ready_dag_abi'] = 2
        (stage / 'artifact.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        stage.rename(out)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-aten-graph', description=__doc__)
    parser.add_argument('--recipe', required=True, help='declared CPU ATen dataflow recipe JSON')
    parser.add_argument('--recipe-sha256', required=True)
    parser.add_argument('--linkage', required=True, help='retained link-aten-provider result.json')
    parser.add_argument('--linkage-sha256', required=True)
    parser.add_argument('--concurrency-contract')
    parser.add_argument('--concurrency-sha256')
    parser.add_argument('--target', help='original target hardware graph for a CPU-window trial')
    parser.add_argument('--cpu', type=int, action='append', help='caller and worker IDs, in order')
    parser.add_argument('--allow-cross-numa', action='store_true', help='admit NUMA-aware three-core team placement')
    parser.add_argument('--ready-dag', action='store_true', help='enable dependency-ready execution on a local CPU team')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    compile_aten_native_graph(args.recipe, recipe_sha256=args.recipe_sha256, linkage=args.linkage,
                              linkage_sha256=args.linkage_sha256, out=args.out,
                              concurrency_contract=args.concurrency_contract,
                              concurrency_sha256=args.concurrency_sha256, target=args.target, cpu_ids=args.cpu,
                              allow_cross_numa=args.allow_cross_numa, ready_dag=args.ready_dag)
    print(json.dumps(dict(executable=str(Path(args.out) / _EXECUTABLE),
                          launcher=str(Path(args.out) / 'launch.sh'))))
    return 0
