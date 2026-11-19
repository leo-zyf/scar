"""Compile an observed immutable-record transition and its scoring function."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_transition_ir import plan


SCHEMA = 'scar.scalar-transition-scope-compilation'


def _write(path, document):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _match(native, planned, score):
    if native['plan'] != planned or native['score_plan'] != score:
        raise ValueError('native transition differs from its exact transition and score plans')


def compile_scope(evidence, score, out):
    from scar.codegen.scalar_transition_standalone import _export_verified

    lowered = plan(evidence, score)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    planned = _write(out/'plan.json', lowered)
    native = _export_verified(planned, lowered, read_artifact(lowered['score_plan']), out/'native')
    _match(read_artifact(native), planned, lowered['score_plan'])
    return _write(out/'result.json', dict(schema=SCHEMA, schema_version=1,
        evidence=lowered['evidence'], score_plan=lowered['score_plan'], plan=planned,
        native=native, deployment_authorized=False))


def check(path):
    from scar.codegen.scalar_transition_standalone import check as check_native

    result = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    native = check_native(result['native']['uri'])
    _match(native, result['plan'], result['score_plan'])
    lowered = read_artifact(result['plan'])
    if (native != read_artifact(result['native']) or lowered['evidence'] != result['evidence']
            or result['schema_version'] != 1 or result['deployment_authorized'] is not False):
        raise ValueError('transition compilation differs from cold source and native replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--score-plan', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    inputs = (args.evidence, args.score_plan, args.out)
    if args.check is not None and not any(inputs):
        result = check(args.check)
    elif args.check is None and all(inputs):
        result = compile_scope(*inputs)
    else:
        parser.error('provide --evidence, --score-plan and --out, or --check RESULT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
