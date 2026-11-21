"""Compile the admitted boundary of a Python entry through a native CPU provider."""

import argparse
from importlib import import_module
import json
from pathlib import Path
from typing import NamedTuple

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-entry-compilation'
CATALOG_SCHEMA = 'scar.cpu-entry-provider-catalog'
_SOURCE = 'scar.python-source'


class _Provider(NamedTuple):
    qualifier: str
    compiler: str
    result_schema: str
    authority_field: str


# Dispatch on checked compiler contracts. A provider's document supplies data
# and artifact references; it cannot specify code to import or execute.
_PROVIDERS = {
    'scar.triangle-profile-family': _Provider(
        'scar.compiler.triangle_profile_family',
        'scar.compiler.compile_triangle_profile',
        'scar.source-triangle-entry-compilation',
        'family',
    ),
    'scar.cpu-fixed-grid-source-provider': _Provider(
        'scar.compiler.fixed_grid_source_provider',
        'scar.compiler.fixed_grid_source_provider',
        'scar.cpu-fixed-grid-entry-compilation',
        'provider',
    ),
    'scar.cpu-recursive-stream-source-provider': _Provider(
        'scar.compiler.recursive_stream_source_provider',
        'scar.compiler.recursive_stream_source_provider',
        'scar.cpu-recursive-stream-entry-compilation',
        'provider',
    ),
    'scar.cpu-array-task-source-provider': _Provider(
        'scar.compiler.array_task_source_provider',
        'scar.compiler.array_task_source_provider',
        'scar.cpu-array-task-entry-compilation',
        'provider',
    ),
}


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _candidate(path):
    reference = _reference(path, 'scar.cpu-provider-document')
    value = json.loads(read_artifact(reference, json_document=False))
    schema = value.get('schema') if type(value) is dict else None
    if type(schema) is not str or schema not in _PROVIDERS:
        raise ValueError('unsupported CPU provider contract: ' + reference['uri'])
    reference['schema'] = schema
    return reference, read_artifact(reference)


def _unique(candidates):
    references = {ref['uri'] for ref, _ in candidates}
    if len(references) != len(candidates):
        raise ValueError('CPU provider authority was supplied more than once')


def write_catalog(paths, out):
    candidates = [_candidate(path) for path in paths]
    _unique(candidates)
    if not candidates:
        raise ValueError('CPU provider catalog requires at least one authority')
    authorities = sorted((ref for ref, _ in candidates), key=lambda ref: ref['uri'])
    document = dict(schema=CATALOG_SCHEMA, schema_version=1, authorities=authorities)
    out = Path(out).absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return _reference(out, CATALOG_SCHEMA)


def _read_catalog(reference):
    document = read_artifact(reference)
    authorities = document.get('authorities')
    if (reference['schema'] != CATALOG_SCHEMA or type(document.get('schema_version')) is not int
            or document['schema_version'] != 1 or type(authorities) is not list or not authorities):
        raise ValueError('unsupported CPU provider catalog')
    candidates = []
    for authority in authorities:
        schema = authority.get('schema') if type(authority) is dict else None
        if type(schema) is not str or schema not in _PROVIDERS:
            raise ValueError('catalog contains an unsupported CPU provider contract')
        if _reference(authority['uri'], schema) != authority:
            raise ValueError('CPU catalog authority changed after indexing')
        candidates.append((authority, read_artifact(authority)))
    return candidates


def _match(candidates, source):
    _unique(candidates)
    matching = [(ref, doc) for ref, doc in candidates if doc.get('entry') == source]
    if len(matching) != 1:
        raise ValueError('entry requires exactly one CPU provider qualified for its path and bytes')
    (reference, declared), = matching
    return reference, declared


def _resolve(paths, source, catalog=None):
    candidates = _read_catalog(catalog) if catalog else [_candidate(path) for path in paths]
    reference, declared = _match(candidates, source)
    provider = _PROVIDERS[reference['schema']]
    # These provider modules revalidate the exact authority in compile_entry.
    # The separate family qualifier is still required before delegating.
    if provider.qualifier != provider.compiler and import_module(provider.qualifier).check(reference['uri']) != declared:
        raise ValueError('CPU provider authority changed during qualification')
    return reference, provider


def _check_scope(result, provider, *, authority, source, argv):
    if (result.get(provider.authority_field) != authority
            or result.get('entry') != source or result.get('argv') != argv):
        raise ValueError('native provider result differs from the requested exact entry scope')
    read_artifact(source, json_document=False)
    read_artifact(authority)


def compile_cpu_entry(providers, entry, argv, out, *, catalog=None):
    source = _reference(entry, _SOURCE)
    argv = list(argv)
    if providers and catalog is not None:
        raise ValueError('supply direct CPU providers or a catalog')
    catalog_ref = _reference(catalog, CATALOG_SCHEMA) if catalog is not None else None
    authority, provider = _resolve(providers, source, catalog_ref)
    out = Path(out).resolve()
    inputs = (source, authority, catalog_ref) if catalog_ref else (source, authority)
    if any(Path(ref['uri']).is_relative_to(out) for ref in inputs):
        raise ValueError('compilation output overlaps an entry or provider input')
    out.mkdir(parents=True, exist_ok=False)
    backend = import_module(provider.compiler)
    result = backend.compile_entry(authority['uri'], source['uri'], argv, out/'native')
    if (result['schema'] != provider.result_schema
            or not Path(result['uri']).resolve(strict=True).is_relative_to(out/'native')):
        raise ValueError('CPU provider did not return its contracted compilation receipt')
    _check_scope(read_artifact(result), provider, authority=authority, source=source, argv=argv)
    receipt = out/'result.json'
    document = dict(schema=SCHEMA, schema_version=1, provider=authority,
                    entry=source, argv=argv, compilation=result)
    if catalog_ref:
        document['catalog'] = catalog_ref
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return _reference(receipt, SCHEMA)


def check(path):
    """Cold-read the provider-specific result through the same scope contract."""
    record = read_artifact(_reference(path, SCHEMA))
    authority = record['provider']
    source = record['entry']
    result = record['compilation']
    if record['schema_version'] != 1 or authority['schema'] not in _PROVIDERS:
        raise ValueError('unsupported CPU entry compilation receipt')
    provider = _PROVIDERS[authority['schema']]
    if result['schema'] != provider.result_schema or source['schema'] != _SOURCE:
        raise ValueError('CPU entry receipt carries incompatible artifact schemas')
    if catalog := record.get('catalog'):
        matched, _ = _match(_read_catalog(catalog), source)
        if matched != authority:
            raise ValueError('CPU entry provider differs from its catalog selection')
    declared = read_artifact(result)
    if import_module(provider.compiler).check_entry(result['uri']) != declared:
        raise ValueError('CPU provider compilation changed during replay')
    _check_scope(declared, provider, authority=authority, source=source, argv=record['argv'])
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', action='append', type=Path,
                        help='explicit provider authority (repeatable)')
    parser.add_argument('--catalog', type=Path, help='select the provider from a sealed local catalog')
    parser.add_argument('--write-catalog', type=Path, help='seal the supplied provider authorities')
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--argv', nargs=argparse.REMAINDER,
                        help='entry arguments (must come last)')
    args = parser.parse_args(argv)
    if args.write_catalog and args.provider and not any((args.check, args.catalog, args.entry, args.out, args.argv is not None)):
        result = write_catalog(args.provider, args.write_catalog)
    elif args.check is not None and not any((args.provider, args.catalog, args.write_catalog, args.entry, args.out, args.argv is not None)):
        result = check(args.check)
    elif args.check is None and not args.write_catalog and (args.provider or args.catalog) and args.entry and args.out:
        result = compile_cpu_entry(args.provider or (), args.entry, args.argv or (), args.out, catalog=args.catalog)
    else:
        parser.error('use --provider or --catalog with --entry and --out, --provider with --write-catalog, or --check RESULT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
