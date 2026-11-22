"""Derive the typed Python source layers in one owned publication."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from scar.analysis.compiler_symbols import _project_symbols as _project_compiler_symbols
from scar.analysis.compiler_topology import _project_topology
from scar.analysis.source_bindings import _project_bindings
from scar.analysis.source_contexts import _project_contexts
from scar.analysis.source_declarations import _project_declarations
from scar.analysis.source_materials import _material
from scar.analysis.source_role_materials import _project_roles
from scar.analysis.source_symbols import compiler_hash_context, _project_symbols
from scar.analysis.source_syntax import decoded_source_bytes, _project_tokens, _project_syntax, _restore_syntax
from scar.analysis.static.query import _Fragments
from scar.analysis.static_program import (
    _artifacts, _json, _write, build_static_program, read_static_program, refine_static_program,
)
from scar.artifacts import _durable_bytes, _durable_mkdir, _sync_directory, file_reference
from scar.ir.record_codec import encode


_RAW = ('source_bytes',)
_SYNTAX = (*_RAW, 'source_tokens', 'source_syntax')
_ROLES = (*_SYNTAX, 'source_roles')
_CONTEXT = (*_ROLES, 'source_contexts')
_DECLARATIONS = (*_CONTEXT, 'source_declarations')
_LAYERS = {
    'source_tokens': ('F1', _RAW),
    'source_syntax': ('F2', _SYNTAX[:2]),
    'source_symbols': ('F3-A', _SYNTAX),
    'source_roles': ('F3-B', _SYNTAX),
    'source_contexts': ('F3-C1', _ROLES),
    'source_declarations': ('F3-C2a', _CONTEXT),
    'source_bindings': ('F3-C2b', _DECLARATIONS),
    'compiler_topology': ('F3-C3T', (*_CONTEXT, 'source_symbols')),
    'compiler_symbols': ('F3-C3S', (*_DECLARATIONS, 'source_symbols', 'compiler_topology')),
}
_BLOCKED = frozenset({'BUDGET_STOPPED', 'PARSER_RESOURCE_ERROR', 'RESOURCE_ERROR', 'PROJECTION_CONTRADICTION'})


def _derive_layers(f0):
    """Publish a source unit's original layers from one owned typed derivation."""
    original = decoded_source_bytes(f0)
    docs = {'source_bytes': f0}
    records = {}

    def produce(name, project, *arguments):
        start = time.monotonic()
        material = project(*arguments)
        status = material.status
        if status in _BLOCKED:
            raise ValueError(f'{_LAYERS[name][0]} {status} for {f0["path"]}')
        if name == 'source_syntax':
            _restore_syntax(material)
        doc = encode(material)
        docs[name] = doc
        records[name] = dict(status=status, seconds=time.monotonic() - start)
        return material, doc

    _, f1 = produce('source_tokens', _project_tokens, f0, original)
    syntax, f2 = produce('source_syntax', _project_syntax, f0, f1, original)
    native, f3a = produce('source_symbols', _project_symbols, f0, f1, f2, original)
    roles, f3b = produce('source_roles', _project_roles, f0, f1, f2, syntax)
    contexts, c1 = produce('source_contexts', _project_contexts, f0, f1, f2, f3b, syntax, roles)
    declarations, c2a = produce('source_declarations', _project_declarations,
        f0, f1, f2, f3b, c1, syntax, roles, contexts)
    produce('source_bindings', _project_bindings, f0, f1, f2, f3b, c1, c2a, syntax, roles, contexts, declarations)
    topology, c3t = produce('compiler_topology', _project_topology, f0, f1, f2, f3b, c1, f3a, syntax, roles, contexts, native)
    produce('compiler_symbols', _project_compiler_symbols,
        f0, f1, f2, f3b, c1, c2a, f3a, c3t, syntax, roles, contexts, declarations, native, topology)
    return docs, records


def _derive_unit(task):
    root, unit, hash_context = task
    if compiler_hash_context() != hash_context:
        raise ValueError('source workers require the same Python hash context')
    key = unit['key']
    if not key or Path(key).name != key or key in {'.', '..'}:
        raise ValueError('invalid source unit key')
    start = time.monotonic()
    f0, origin, detail = _material(unit, _Fragments(root).read)
    if f0 is None:
        raise ValueError(f'F0 unavailable for {unit["path"]}: {detail}')
    acquisition = time.monotonic() - start
    docs, records = _derive_layers(f0)
    records['source_bytes'] = dict(status=origin, seconds=acquisition)
    refs = {name: _write(root, f'units/{key}/{name}.json.gz', doc) for name, doc in docs.items()}
    return key, refs, records, os.getpid()


def build_source_layers(roots, model, report, *, jobs, source_files=(), max_files=4096,
                        max_bytes=64_000_000, max_directories=16384):
    """Publish a fresh model only after all declared units pass their layer contracts."""
    model, report = Path(model).absolute(), Path(report).absolute()
    if (model.exists() or report.exists() or model.resolve() != model or report.resolve() != report
            or report.is_relative_to(model)):
        raise ValueError('fresh canonical source model and separate report required')
    _durable_mkdir(model.parent)
    started = time.monotonic()
    with TemporaryDirectory(prefix='.source-layers-', dir=model.parent) as directory:
        staged = Path(directory) / 'model'
        build_static_program(roots[0], source_roots=roots[1:], source_files=source_files, out=staged,
                             jobs=jobs, phase='sources', max_files=max_files, max_bytes=max_bytes,
                             max_directories=max_directories)
        manifest = refine_static_program(staged)
        if (manifest['source_catalog_status'] != 'COMPLETE' or manifest['worker_errors']
                or manifest['source_changed'] or any(unit['status'] in {'ERROR', 'SYNTAX_ERROR'}
                                                     for unit in manifest['units'])):
            raise ValueError('source unit inventory is incomplete')
        units = manifest['units']
        hash_context = compiler_hash_context()
        tasks = [(staged, unit, hash_context) for unit in units]
        count = min(jobs, len(tasks))
        if count <= 1:
            results = list(map(_derive_unit, tasks))
        else:
            with ProcessPoolExecutor(max_workers=count, mp_context=get_context('spawn')) as pool:
                results = list(pool.map(_derive_unit, tasks, chunksize=1))
        for unit, (key, refs, _, _) in zip(units, results, strict=True):
            if unit['key'] != key or unit['artifacts'].keys() & refs.keys():
                raise ValueError('derived materials do not belong to a fresh source unit')
            unit['artifacts'].update(refs)
        layers = []
        for name, stage, requires in (('source_bytes', 'F0', ('source',)), *(
                (artifact, stage, requires) for artifact, (stage, requires) in _LAYERS.items())):
            rows = [record[name] for _, _, record, _ in results]
            layers.append(dict(stage=stage, artifact=name, requires=list(requires),
                               statuses=dict(Counter(row['status'] for row in rows)),
                               worker_seconds=sum(row['seconds'] for row in rows)))
        manifest['source_layers'] = [dict((key, value) for key, value in layer.items() if key != 'worker_seconds')
                                     for layer in layers]
        descriptors = {ref['path']: ref for ref in _artifacts(manifest)}
        manifest['storage'] = dict(files=len(descriptors), compressed_bytes=sum(ref['bytes'] for ref in descriptors.values()))
        _durable_bytes(staged / 'manifest.json', _json(manifest), replace=True)
        read_static_program(staged, verify=False)
        directories = [staged]
        for path in staged.rglob('*'):
            if path.is_dir():
                directories.append(path)
            else:
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _sync_directory(path)
        if model.exists():
            raise ValueError('source model appeared during preparation')
        staged.rename(model)
        _sync_directory(model.parent)
    result = dict(schema='scar.source-layer-batch', schema_version=1, model=str(model),
                  manifest=file_reference(model / 'manifest.json', 'scar.static-program'),
                  units=len(units), layer_workers=count, active_workers=len({pid for _, _, _, pid in results}),
                  layers=layers, complete_for_declared_units=True,
                  target_executed=False, seconds=time.monotonic() - started)
    _durable_mkdir(report.parent)
    _durable_bytes(report, _json(result), replace=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--jobs', type=int, choices=range(1, 33), default=4)
    parser.add_argument('--source-file', action='append', default=[], type=Path)
    parser.add_argument('--max-files', type=int, default=4096)
    parser.add_argument('--max-bytes', type=int, default=64_000_000)
    parser.add_argument('--max-directories', type=int, default=16384)
    args = parser.parse_args(argv)
    build_source_layers(args.roots, args.model, args.report, jobs=args.jobs, source_files=args.source_file,
                        max_files=args.max_files, max_bytes=args.max_bytes, max_directories=args.max_directories)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
