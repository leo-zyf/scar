"""Explicitly reissue a known schema-3 catalog under the current reader rules.

Only the catalog directory is rebuilt. Original F4 leaves, interface shards,
sources, observations and the previous report stay immutable and are retained
by their actual receipts. The private legacy-reader entry point is used only
after the caller supplied the exact prior producer hash.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from scar.analysis.source_declared_catalog import _check, _receipt
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.artifacts import _json


REISSUE_SCHEMA = 'scar.observed-source-catalog-reissue'


def reissue_observed_catalog(prior, out, *, expected_prior_implementation_sha256):
    """Validate a pinned legacy catalog, rebuild its directory, admit version 4.

    The explicit prior hash is not an alternative normal-reader policy. It
    authorizes this one migration input; the published result must pass the
    ordinary current ``ObservedSourceCatalog`` constructor without an override.
    """
    from scar.analysis import observed_source_catalog as model

    started = time.monotonic()
    expected = expected_prior_implementation_sha256
    if type(expected) is not str or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
        raise ValueError('explicit lowercase prior implementation SHA256 required')
    prior, out = Path(prior).absolute(), Path(out).absolute()
    if prior.is_file():
        if prior.name != 'result.json':
            raise ValueError('prior observed catalog must name its original result.json')
        prior = prior.parent
    if (prior.resolve() != prior or not prior.is_dir()
            or out.resolve() != out or out.exists() or out.is_relative_to(prior)):
        raise ValueError('canonical original catalog and fresh external reissue directory required')
    original_receipt = _receipt(prior/'result.json')
    original = json.loads((prior/'result.json').read_bytes())
    if (original.get('schema') != model.SCHEMA or original.get('schema_version') != 3
            or original.get('implementation_sha256') != expected):
        raise ValueError('prior observed catalog schema or explicitly pinned producer differs')
    _check(original_receipt)

    # All prior evidence, source/code matches, leaves and actual interface
    # metadata are checked by the existing reader, not by a second migrator IR.
    old = model.ObservedSourceCatalog(prior, _expected_implementation=expected)
    if old.report != original or old._root_receipt != original_receipt:
        raise ValueError('original observed catalog changed during migration admission')

    # Reconstruct this routing directory from its validated primitive parents.
    # Do not copy the old directory or normalize any source function again.
    rows = dict(old.base.fact_store.rows)
    for identity, material in sorted(old.materials.items()):
        if identity in rows:
            raise ValueError('reissued compiled unit collides with an original file unit')
        facts = material.fact_store
        rows[identity] = dict(facts.row, facts=str(material.path/facts.row['facts']), module=None)
    if (type(original['source_units']) is not list
            or len(original['source_units']) != len(rows) or set(original['source_units']) != set(rows)):
        raise ValueError('prior source-unit inventory differs from its primitive directory')
    directory = dict(schema='scar.observed-source-facts', schema_version=1, units=rows)
    current_implementation = sha256(Path(model.__file__).read_bytes()).hexdigest()
    reissuer_implementation = sha256(Path(__file__).read_bytes()).hexdigest()
    contract = model.reader_contract()
    provenance = dict(schema=REISSUE_SCHEMA, schema_version=1,
        prior_catalog=original_receipt, prior_schema_version=3,
        prior_implementation_sha256=expected,
        current_catalog_implementation_sha256=current_implementation,
        reissuer_implementation_sha256=reissuer_implementation,
        rules_sha256=sha256(_json(contract)).hexdigest(),
        operation='REISSUE_ROUTING_DIRECTORY_REUSE_VALIDATED_PRIMITIVE_LEAVES',
        original_materials_modified=False, target_executed=False)
    _check(original_receipt)
    old.validate_parents()
    _durable_mkdir(out)
    _durable_bytes(out/'facts.json', _json(directory), replace=False)
    # Preserve all actual base/leaf/interface/input_catalog/external evidence
    # receipts and selection declarations. Only this routing publication and
    # its explicitly versioned reader contract receive new identities.
    result = dict(original, schema_version=model.SCHEMA_VERSION,
        implementation_sha256=current_implementation, reader_contract=contract,
        facts=_receipt(out/'facts.json'), reissue=provenance,
        seconds=time.monotonic()-started)
    if (sha256(Path(model.__file__).read_bytes()).hexdigest() != current_implementation
            or sha256(Path(__file__).read_bytes()).hexdigest() != reissuer_implementation
            or model.reader_contract() != contract):
        raise ValueError('catalog reissue rules changed during publication')
    _durable_bytes(out/'result.json', _json(result), replace=False)
    # No legacy override is permitted for the delivered artifact.
    current = model.ObservedSourceCatalog(out)
    if current.report != result:
        raise ValueError('reissued catalog differs from its normal reader admission')
    _check(original_receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expected-prior-implementation-sha256', required=True)
    args = parser.parse_args()
    result = reissue_observed_catalog(args.prior, args.out,
        expected_prior_implementation_sha256=args.expected_prior_implementation_sha256)
    print(json.dumps(dict(schema=result['schema'], schema_version=result['schema_version'],
        output=str(Path(args.out).absolute()), source_units=len(result['source_units']),
        compiled_units=len(result['compiled']), reissue=result['reissue'],
        seconds=result['seconds'], first_stage_complete=False)), flush=True)


if __name__ == '__main__':
    main()
