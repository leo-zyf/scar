"""Derive the typed Python source layers in one owned publication."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Callable

from scar.analysis.compiler_symbols import compiler_symbols_document
from scar.analysis.compiler_topology import compiler_topology_document
from scar.analysis.source_bindings import source_bindings_document
from scar.analysis.source_contexts import source_contexts_document
from scar.analysis.source_declarations import source_declarations_document
from scar.analysis.source_materials import _material
from scar.analysis.source_role_materials import source_roles_document
from scar.analysis.source_symbols import compiler_hash_context, source_symbols_document
from scar.analysis.source_syntax import source_syntax_document
from scar.analysis.source_tokens import source_tokens_document
from scar.analysis.static.query import _Fragments
from scar.analysis.static_program import (
    _artifacts, _json, _write, build_static_program, read_static_program, refine_static_program,
)
from scar.artifacts import _durable_bytes, _durable_mkdir, _sync_directory, file_reference


@dataclass(frozen=True, slots=True)
class Layer:
    artifact: str
    stage: str
    requires: tuple[str, ...]
    derive: Callable


_RAW = ('source_bytes',)
_SYNTAX = (*_RAW, 'source_tokens', 'source_syntax')
_ROLES = (*_SYNTAX, 'source_roles')
_CONTEXT = (*_ROLES, 'source_contexts')
_DECLARATIONS = (*_CONTEXT, 'source_declarations')
_LAYERS = (
    Layer('source_tokens', 'F1', _RAW, source_tokens_document),
    Layer('source_syntax', 'F2', _SYNTAX[:2], source_syntax_document),
    Layer('source_symbols', 'F3-A', _SYNTAX, source_symbols_document),
    Layer('source_roles', 'F3-B', _SYNTAX, source_roles_document),
    Layer('source_contexts', 'F3-C1', _ROLES, source_contexts_document),
    Layer('source_declarations', 'F3-C2a', _CONTEXT, source_declarations_document),
    Layer('source_bindings', 'F3-C2b', _DECLARATIONS, source_bindings_document),
    Layer('compiler_topology', 'F3-C3T', (*_CONTEXT, 'source_symbols'), compiler_topology_document),
    Layer('compiler_symbols', 'F3-C3S', (*_DECLARATIONS, 'source_symbols', 'compiler_topology'),
          compiler_symbols_document),
)
_BLOCKED = frozenset({'BUDGET_STOPPED', 'PARSER_RESOURCE_ERROR', 'RESOURCE_ERROR', 'PROJECTION_CONTRADICTION'})


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
    docs = {'source_bytes': f0}
    records = {'source_bytes': dict(status=origin, seconds=time.monotonic() - start)}
    for spec in _LAYERS:
        start = time.monotonic()
        doc = spec.derive(*(docs[name] for name in spec.requires))
        status = doc['status']
        if status in _BLOCKED:
            raise ValueError(f'{spec.stage} {status} for {unit["path"]}')
        docs[spec.artifact] = doc
        records[spec.artifact] = dict(status=status, seconds=time.monotonic() - start)
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
                (spec.artifact, spec.stage, spec.requires) for spec in _LAYERS)):
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
