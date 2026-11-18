"""Compile a cold-checked partition successor for explicit validation."""

import argparse
import json
from pathlib import Path

from scar.codegen.partition_multiply import build_partition_multiply
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.partition_copy import SCHEMA as PLAN_SCHEMA, check


SCHEMA = 'scar.partition-copy-trial'


def compile_trial(plan, out):
    reference = artifact_reference(plan, PLAN_SCHEMA + '-index')
    if check(reference['uri']) != 1:
        raise ValueError('explicit trial requires one discovered partition region')
    candidate, = read_artifact(reference)['candidates']
    directory = Path(out).resolve()
    native = build_partition_multiply(directory / 'native')
    receipt = dict(schema=SCHEMA, schema_version=1, plan=reference,
        source=candidate['source'], successor=candidate['graphs']['after'], native=native,
        deployment_authorized=False, validation_scope='exclusive normal-completion trial')
    path = directory / 'trial.json'
    with path.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return artifact_reference(path, SCHEMA)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-partition-trial', description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compile_trial(args.plan, args.out)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
