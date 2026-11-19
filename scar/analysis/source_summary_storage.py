"""Store large source summaries in their original typed record collections."""

import json
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

from scar.analysis.interprocedural_summary import SCHEMA
from scar.analysis.normal_output import write_table
from scar.analysis.source_stage import _durable_bytes
from scar.analysis.source_symbol_variants import _table
from scar.ir.execution_recipe import _LIMIT, artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


_ARRAYS = ('functions', 'calls', 'tokens', 'sccs')
_MAPS = ('dependency_sets', 'unknown_facts')


def write_source_summary(path, report):
    """Retain the single document for small summaries and shard larger ones."""
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent, prefix='.source-summary-') as folder:
        work = Path(folder)
        single = work / 'summary.json'
        with single.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        if single.stat().st_size <= _LIMIT:
            _durable_bytes(path, single.read_bytes(), replace=False)
        else:
            data = work / 'tables'
            data.mkdir()
            tables = {
                'functions': [write_table(data / f'functions-{index:06d}.jsonl.gz', (row,))
                              for index, row in enumerate(report['functions'])]
            }
            for key in _ARRAYS[1:]:
                tables[key] = [write_table(data / f'{key}.jsonl.gz', report[key])]
            for key in _MAPS:
                tables[key] = [write_table(data / f'{key}.jsonl.gz', sorted(report[key].items()))]
            header = {key: value for key, value in report.items() if key not in tables}
            manifest = dict(schema=SCHEMA, schema_version=2, header=header, tables=tables)
            # Reserve this destination before moving records. Publish the
            # manifest only after every table can be read.
            destination = path.with_name(path.name + '.tables')
            destination.mkdir()
            try:
                for member in data.iterdir():
                    member.rename(destination / member.name)
                _durable_bytes(path, json.dumps(manifest, sort_keys=True, allow_nan=False).encode() + b'\n',
                    replace=False)
            except BaseException:
                if not path.exists():
                    rmtree(destination)
                raise
    return artifact_reference(path, SCHEMA)


def read_source_summary(path):
    """Restore the same semantic report after checking every shard receipt."""
    path = Path(path).absolute()
    reference = artifact_reference(path, SCHEMA)
    result = read_artifact(reference)
    if result['schema_version'] == 1:
        return result
    if result['schema_version'] != 2 or set(result) != {'schema', 'schema_version', 'header', 'tables'}:
        raise ValueError('unsupported source summary storage envelope')
    header, tables = result['header'], result['tables']
    if (header.get('schema') != SCHEMA or header.get('schema_version') != 1
            or set(tables) != set((*_ARRAYS, *_MAPS)) or set(header) & set(tables)):
        raise ValueError('source summary storage fields differ from its logical document')
    folder = path.with_name(path.name + '.tables')
    if folder.resolve() != folder:
        raise ValueError('source summary shard directory must retain its original path')
    report = dict(header)
    for key, descriptors in tables.items():
        files = ([f'functions-{index:06d}.jsonl.gz' for index in range(len(descriptors))]
                 if key == 'functions' else [f'{key}.jsonl.gz'])
        rows = []
        for filename, receipt in zip(files, descriptors, strict=True):
            if receipt['path'] != filename or (folder / filename).is_symlink():
                raise ValueError('source summary shard name differs from its typed table')
            shard = _table(folder, receipt)
            if key == 'functions' and len(shard) != 1:
                raise ValueError('one original function summary required per shard')
            rows.extend(shard)
        if key in _MAPS:
            report[key] = dict(rows)
            if len(report[key]) != len(rows):
                raise ValueError('source summary shard contains duplicate record identities')
        else:
            report[key] = rows
    identity = report['id']
    if identity != 'interprocedural-summary:' + semantic_sha256({k: v for k, v in report.items() if k != 'id'}):
        raise ValueError('sharded summary differs from its original semantic identity')
    read_artifact(reference, json_document=False)
    return report
