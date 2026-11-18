"""Lower an explicit native tensor DAG to ordered C++20 LibTorch calls."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _build_interface, _dependencies, _resolve
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_native_recipe import normalize_recipe
from scar.ir.record_codec import loads


_DRIVER = Path(__file__).with_name('aten_native_cli_v1.cpp.in')
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


def _render(recipe, operators):
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
        calls.append(f"    const at::Tensor {target} = {entries[op['schema']]}({arguments});")
        values[op['id']] = target
    input_shapes = ', '.join(f'shape_{i}' for i in range(len(shapes)))
    outputs = ', '.join(values[output] for output in recipe['outputs'])
    shape_definitions, operator_calls = '\n'.join(shapes), '\n'.join(calls)
    # SSA temporaries below carry Tensor handles; all numerical work stays in the provider.
    return f'''#pragma once
#include "operators.hpp"
#include <algorithm>
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

inline std::array<at::Tensor, output_count> run(std::span<const at::Tensor> inputs) {{
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
{operator_calls}
    return {{{outputs}}};
}}
}}  // namespace scar_aten_graph
'''


def compile_aten_native_graph(recipe, *, recipe_sha256, linkage, linkage_sha256, out):
    import torch

    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    _, recipe, recipe_receipt = _read_declared(recipe, recipe_sha256)
    linkage_path, linked, linkage_receipt = _read_declared(linkage, linkage_sha256)
    operators, libraries, cxx, compile_flags, link_flags = _installed_linkage(linked, torch)
    recipe = normalize_recipe(recipe, operators, torch)
    operator_path = linkage_path.with_name('operators.hpp')
    command_env = {key: val for key, val in os.environ.items() if not key.startswith('LD_')}
    command = [*cxx, *compile_flags, '-O2', 'entry.cpp', *link_flags, '-o', _EXECUTABLE]
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-native-graph-', dir=out.parent) as temporary:
        stage = Path(temporary)
        header = stage / 'operators.hpp'
        header.write_bytes(operator_path.read_bytes())
        if _content(header) != linked['artifacts']['operators.hpp']:
            raise ValueError('retained ATen C++ binding differs from its linkage receipt')
        (stage / 'graph.hpp').write_text(_render(recipe, operators))
        (stage / 'entry.cpp').write_bytes(_DRIVER.read_bytes())
        (stage / 'recipe.json').write_text(json.dumps(recipe, indent=2, sort_keys=True) + '\n')
        subprocess.run(command, cwd=stage, check=True, env=command_env)
        listing = subprocess.run(['ldd', './' + _EXECUTABLE], cwd=stage, check=True,
                                 env=command_env, capture_output=True, text=True).stdout
        dependencies = _dependencies(listing)
        if (any(dependencies.get(path.name) != str(path.resolve()) for path in libraries)
                or any('libpython' in name or 'libtorch_python' in name for name in dependencies)):
            raise ValueError('linked standalone native graph requires the declared provider without Python')
        script = stage / 'launch.sh'
        script.write_text('#!/bin/sh\nset -eu\nexec env -i PATH=/usr/bin:/bin "$(dirname -- "$0")/'
                          + _EXECUTABLE + '" "$@"\n')
        script.chmod(0o755)
        (stage / 'ldd.txt').write_text(listing)
        receipt = dict(schema='scar.aten-native-graph-delivery', schema_version=1, cxx_standard=20,
            native_abi_version=1, graph_scope='DECLARED_NATIVE_CPU_GRAPH', recipe=recipe_receipt,
            linkage=linkage_receipt, provider=linked['provider'], dependencies=dependencies,
            operation_order=[op['id'] for op in recipe['operations']], implementation='ORDERED_PROVIDER_CALLS',
            compiler_command=command, source_replacement_authorized=False, performance_measured=False,
            files={path.name: _content(path) for path in sorted(stage.iterdir())})
        (stage / 'artifact.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        stage.rename(out)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-aten-graph', description=__doc__)
    parser.add_argument('--recipe', required=True, help='declared CPU ATen dataflow recipe JSON')
    parser.add_argument('--recipe-sha256', required=True)
    parser.add_argument('--linkage', required=True, help='retained link-aten-provider result.json')
    parser.add_argument('--linkage-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    compile_aten_native_graph(args.recipe, recipe_sha256=args.recipe_sha256, linkage=args.linkage,
                              linkage_sha256=args.linkage_sha256, out=args.out)
    print(json.dumps(dict(executable=str(Path(args.out) / _EXECUTABLE),
                          launcher=str(Path(args.out) / 'launch.sh'))))
    return 0
