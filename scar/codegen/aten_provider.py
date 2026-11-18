"""Bind declared ATen schemas to the installed provider's C++20 call interface."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_provider_bundle import _content


_SCHEMA = 'scar.aten-provider-linkage'
_OPERATOR = re.compile(r'aten::([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\(')


def _resolve(schemas, torch, include):
    operators = []
    for schema in dict.fromkeys(schemas):
        match = _OPERATOR.match(schema)
        if match is None:
            raise ValueError('a full aten:: operator schema is required: ' + schema)
        name, overload = match.group(1), match.group(2) or ''
        try:
            registered = torch._C._dispatch_find_schema_or_throw('aten::' + name, overload).schema()
        except RuntimeError as error:
            raise ValueError('operator schema not found in the installed dispatcher: ' + schema) from error
        if schema != str(registered):
            raise ValueError('operator schema differs from the installed dispatcher: ' + schema)
        # Torchgen groups overloads and their in-place variants in a common header.
        header = next((f'ATen/ops/{base}_ops.h' for base in (name, name.removesuffix('_'))
                       if (include / f'ATen/ops/{base}_ops.h').is_file()), None)
        if header is None:
            raise ValueError('no installed generated C++ header for ' + schema)
        identifier = name + ('_' + overload if overload else '')
        operators.append(dict(schema=schema, header=header,
            provider_type='at::_ops::' + identifier, entry='scar_aten::call_' + str(len(operators))))
    if not operators:
        raise ValueError('at least one installed operator schema is required')
    return operators


def _write_binding(directory, operators):
    headers = sorted({row['header'] for row in operators})
    lines = ['#pragma once', '#include <ATen/core/Tensor.h>', '#include <string_view>',
             *(f'#include <{header}>' for header in headers), '', 'namespace scar_aten {']
    probe = ['#include "operators.hpp"', '', 'int main() {']
    for index, row in enumerate(operators):
        identifier = f'op_{index}'
        lines.extend((f"using {identifier} = {row['provider_type']};",
                      f'static_assert(std::string_view({identifier}::schema_str) == '
                      f'{json.dumps(row["schema"].removeprefix("aten::"))});',
                      f'inline constexpr {identifier}::ptr_schema call_{index} = &{identifier}::call;'))
        # A volatile read retains the relocation, even when no operation is run.
        probe.extend((f'    const auto volatile call_{index} = {row["entry"]};',
                      f'    if (call_{index} == nullptr) return 1;'))
    lines.append('}  // namespace scar_aten')
    probe.extend(('    return 0;', '}'))
    (directory / 'operators.hpp').write_text('\n'.join(lines) + '\n')
    (directory / 'link-check.cpp').write_text('\n'.join(probe) + '\n')


def link_aten_provider(schemas, out, *, provider_revision):
    """Generate typed provider references and link them without running the target.

    This checks exposure of the exact dispatcher overload under the installed
    C++ ABI. The caller still owns tensor bridging, dispatch, effects and costs.
    """
    import torch

    if not provider_revision or provider_revision != torch.version.git_version:
        raise ValueError('operator declaration and installed PyTorch revisions differ')
    root = Path(torch.__file__).resolve().parent
    include, lib = root / 'include', root / 'lib'
    operators = _resolve(schemas, torch, include)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    libraries = [lib / name for name in ('libtorch_cpu.so', 'libc10.so')]
    flags = ['-std=c++20', f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
             '-I', str(include)]
    link = ['-Wl,-z,defs', f'-Wl,-rpath,{lib}', *map(str, libraries)]
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-', dir=out.parent) as temporary:
        directory = Path(temporary)
        _write_binding(directory, operators)
        subprocess.run([*compiler, *flags, '-O2', 'link-check.cpp', *link, '-o', 'link-check'],
                       cwd=directory, check=True)
        report = dict(schema=_SCHEMA, schema_version=1,
            provider=dict(name='LibTorch', version=torch.__version__, revision=provider_revision,
                libraries=[dict(path=str(path), **_content(path)) for path in libraries]),
            operators=operators, compiler=compiler, compile_flags=flags, link_flags=link,
            artifacts={name: _content(directory / name)
                       for name in ('operators.hpp', 'link-check.cpp', 'link-check')},
            status='EXTERNAL_CPP_CALL_LINKED', cxx_standard=20,
            scope='INSTALLED_ATEN_DISPATCHER_LINKAGE', target_executed=False,
            dispatch_device_proven=False, replacement_authorized=False, performance_measured=False)
        (directory / 'result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        directory.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar link-aten-provider', description=__doc__)
    parser.add_argument('--schema', action='append', required=True)
    parser.add_argument('--provider-revision', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = link_aten_provider(args.schema, args.out, provider_revision=args.provider_revision)
    print(json.dumps(dict(status=report['status'], operators=len(report['operators']),
                          result=str(Path(args.out) / 'result.json'))))
    return 0
