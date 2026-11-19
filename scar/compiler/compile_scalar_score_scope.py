"""Lower an observed typed scalar function to a detached native program."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_score_ir import plan


SCHEMA = 'scar.scalar-score-scope-compilation'


def _write(path, record):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, record['schema'])


def _match(native, planned):
    if native['plan'] != planned:
        raise ValueError('native scalar function is detached from its checked execution plan')


def compile_scope(evidence, out):
    from scar.codegen.scalar_score_standalone import export

    lowered = plan(evidence)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    planned = _write(out/'plan.json', lowered)
    native = export(planned['uri'], out/'native')
    _match(read_artifact(native), planned)
    return _write(out/'result.json', dict(schema=SCHEMA, schema_version=1, evidence=lowered['evidence'],
        plan=planned, native=native, deployment_authorized=False))


def check(path):
    from scar.codegen.scalar_score_standalone import check as check_native

    published = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    planned, native = published['plan'], published['native']
    restored = check_native(native['uri'])
    _match(restored, planned)
    if (restored != read_artifact(native) or read_artifact(planned)['evidence'] != published['evidence']
            or published['schema_version'] != 1 or published['deployment_authorized'] is not False):
        raise ValueError('scalar scope compilation differs from cold source and native replay')
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.check is not None and args.evidence is None and args.out is None:
        result = check(args.check)
    elif args.check is None and args.evidence is not None and args.out is not None:
        result = compile_scope(args.evidence, args.out)
    else:
        parser.error('provide --evidence and --out, or --check RESULT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
