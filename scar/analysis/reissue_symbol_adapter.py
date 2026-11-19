"""Reissue only the normalizer pins of an installed official symbol adapter."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import sysconfig
from tempfile import TemporaryDirectory

from scar.analysis import source_symbols
from scar.analysis.source_symbol_context_adapter import COMMIT, SymbolContextAdapter, _file
from scar.ir import record_codec
from scar.ir.program import source_symbols as symbol_types


_NORMALIZERS = (source_symbols, symbol_types, record_codec)
_PROBE_NAME = '<scar-symbol-adapter-reissue>'
_PROBE = '''from __future__ import annotations
import scar_symbol_fixture_missing as imported

def outer(argument):
    local = argument
    def inner():
        nonlocal local
        return imported, local
    return inner

raise RuntimeError("symbol capture must not evaluate this source")
'''


def _installed_paths(directory, receipt):
    runtime = receipt['runtime']
    soabi = sysconfig.get_config_var('SOABI')
    native = directory / ('_scar_symtable_context' + sysconfig.get_config_var('EXT_SUFFIX'))
    if (receipt['native']['path'] != str(native) or native.resolve(strict=True) != native
            or runtime['version'] != sys.version or runtime['soabi'] != soabi):
        raise ValueError('installed symbol adapter path or runtime ABI differs')
    backend, helper = receipt['backend'], receipt['helper']
    if backend == 'CURRENT_RUNTIME' and helper is None:
        return
    executable = directory / ('cpython-' + COMMIT) / 'python'
    if (backend != 'ISOLATED_OFFICIAL_RUNTIME' or type(helper) is not dict
            or helper['executable']['path'] != str(executable) or executable.resolve(strict=True) != executable
            or helper['source']['commit'] != COMMIT):
        raise ValueError('installed official symbol helper path or source revision differs')


def _smoke(adapter):
    import _symtable

    capture = adapter.capture(_PROBE, _PROBE_NAME, max_scopes=8, max_symbols=32)
    official = source_symbols.capture_raw_view(_symtable.symtable(_PROBE, _PROBE_NAME, 'exec'),
                                              max_scopes=8, max_symbols=32)
    if capture.scopes != official or capture.warnings or len(capture.scopes) != 3:
        raise ValueError('reissued symbol capture differs from the official compiler view')
    top, outer, inner = capture.scopes
    flags = capture.profile['constants']
    symbol_flags = [{item.name: item.flags for item in row.symbols} for row in capture.scopes]
    globals_, outer_symbols, inner_symbols = symbol_flags

    def lexical(value, expected):
        return (value >> flags['SCOPE_OFF']) & flags['SCOPE_MASK'] == flags[expected]

    if ((top.name, outer.name, inner.name) != ('top', 'outer', 'inner')
            or (top.children, outer.children, inner.children) != ((1,), (2,), ())
            or outer.varnames != ('argument',) or not inner.nested
            or not globals_.get('imported', 0) & flags['DEF_IMPORT']
            or not outer_symbols.get('argument', 0) & flags['DEF_PARAM']
            or not lexical(outer_symbols.get('local', 0), 'CELL')
            or not lexical(inner_symbols.get('local', 0), 'FREE')
            or not inner_symbols.get('local', 0) & flags['DEF_NONLOCAL']
            or not lexical(inner_symbols.get('imported', 0), 'GLOBAL_IMPLICIT')):
        raise ValueError('reissued symbol capture lacks the expected lexical and import relations')
    return dict(source_sha256=capture.request['text_utf8_sha256'], scope_count=len(capture.scopes),
                official_view_equal=True, lexical_and_import_relations_checked=True, target_executed=False)


def reissue_symbol_adapter(directory, out) -> dict:
    """Publish a fresh receipt and lineage without replacing any installed artifact."""
    directory = Path(directory).resolve(strict=True)
    parent = directory / 'result.json'
    if parent.resolve(strict=True) != parent:
        raise ValueError('symbol adapter receipt must be local to the installed directory')
    raw = parent.read_bytes()
    receipt = json.loads(raw)
    _installed_paths(directory, receipt)
    installed = [_file(module.__file__) for module in _NORMALIZERS]
    by_path = {pin['path']: pin for pin in installed}
    old = receipt['normalizer_pins']
    if (type(old) is not list or len(old) != len(installed)
            or any(type(pin) is not dict or set(pin) != {'path', 'sha256', 'bytes'}
                   or type(pin['path']) is not str or type(pin['bytes']) is not int or pin['bytes'] < 0
                   or type(pin['sha256']) is not str or not re.fullmatch(r'[0-9a-f]{64}', pin['sha256'])
                   for pin in old)
            or {pin['path'] for pin in old} != by_path.keys()):
        raise ValueError('symbol adapter receipt must pin exactly the installed normalizer modules')
    updated = [by_path[pin['path']] for pin in old]
    reissued = (json.dumps(dict(receipt, normalizer_pins=updated), sort_keys=True, indent=2) + '\n').encode()

    out = Path(out)
    out = out.parent.resolve() / out.name
    if out.is_relative_to(directory):
        raise ValueError('reissued receipt must be separate from the installed adapter')
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-symbol-reissue-', dir=out.parent) as temporary:
        stage = Path(temporary)
        (stage / 'result.json').write_bytes(reissued)
        with SymbolContextAdapter(stage) as adapter:
            smoke = _smoke(adapter)
        metadata = dict(schema='scar.cpython-symtable-adapter-reissue', schema_version=1,
            parent=dict(path=str(parent), bytes=len(raw), sha256=sha256(raw).hexdigest()),
            receipt=dict(path=str(out / 'result.json'), bytes=len(reissued), sha256=sha256(reissued).hexdigest()),
            changed_normalizer_paths=[previous['path'] for previous, current in zip(old, updated, strict=True)
                                      if previous != current], smoke=smoke)
        (stage / 'reissue.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
        if parent.read_bytes() != raw or [_file(module.__file__) for module in _NORMALIZERS] != installed:
            raise ValueError('installed symbol adapter receipt or normalizers changed during reissue')
        if out.exists() or out.is_symlink():
            raise FileExistsError(out)
        stage.rename(out)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-directory', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(reissue_symbol_adapter(args.adapter_directory, args.out), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
