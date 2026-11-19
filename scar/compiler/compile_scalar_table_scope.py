"""Plan and deliver a detached evaluator for a source-proven scalar expression."""

import argparse
import json
from pathlib import Path

from scar.compiler.scalar_table_region import prepare
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.scalar_table_contract import SCHEMA as REGION


SCHEMA = 'scar.scalar-table-scope-compilation'
PLAN = 'scar.scalar-table-scope-plan'
_HANDOFF = 'scar.scalar-table-handoff'
_READS = ('FIRST_ROW', 'LEFT_ELEMENT', 'REPEATED_ROW', 'RIGHT_ELEMENT')


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, record):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return _ref(path, record['schema'])


def _source(path):
    from scar.analysis.scalar_table_handoff import check

    ref = _ref(path, _HANDOFF)
    graph = check(ref['uri'])
    if graph != read_artifact(ref):
        raise ValueError('original scalar table evidence changed during replay')
    return ref, graph


def _plan(handoff, graph, region):
    prepared = read_artifact(region)
    index = graph['proof_case']
    proof = read_artifact(graph['proof'])
    case = proof['cases'][index]
    reads = case['reads']
    ports = [row['result'] for row in reads]
    reuse = case['rewrite']
    if (prepared['proof'] != graph['proof'] or prepared['proof_case'] != index
            or prepared['region']['source'] != case['source']
            or prepared['region']['original_code_key'] != case['code_key']
            or tuple(row['role'] for row in reads) != _READS
            or case['original_normal_read_order'] != ports
            or case['operation'] != 'Sub' or reuse['replace'] != ports[2] or reuse['reuse'] != ports[0]
            or reuse['preserve_first_lookup_order'] is not True
            or reuse['preserve_both_index_reads'] is not True):
        raise ValueError('prepared scalar boundary differs from the original F4 read and result ports')
    return dict(schema=PLAN, schema_version=1, handoff=handoff, region=region,
        capture=graph['capture'], operation=case['operation'], original_result=case['result'],
        original_reads=ports, simplified_reads=[ports[0], ports[1], ports[3]],
        second_index_row_input=ports[0], rewrite=reuse,
        admission=dict(table_values='EXACT_CAPTURED_FROZEN_CONTENT_VERSION', keys='EXACT_STRING',
            indices='SIGNED_64_BIT_WITH_TUPLE_BOUNDS', result='EXACT_REPRESENTABLE_SIGNED_64_BIT',
            unsupported_input='DECLINE_OUTSIDE_DETACHED_CONTRACT'),
        disposition='DETACHED_EVALUATOR', original_call_replacement_proven=False,
        profitability='UNMEASURED', deployment_authorized=False)


def _match_native(artifact, plan):
    if artifact['region'] != plan['region'] or artifact['observation'] != plan['capture']:
        raise ValueError('native evaluator differs from the exact source plan and runtime evidence')


def compile_scope(handoff, out):
    from scar.codegen.scalar_table_standalone import export

    source_ref, graph = _source(handoff)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    region = prepare(graph['proof']['uri'], out/'region', case_index=graph['proof_case'])
    plan = _plan(source_ref, graph, region)
    decision = _write(out/'plan.json', plan)
    artifact = export(region['uri'], graph['capture']['uri'], out/'native')
    _match_native(read_artifact(artifact), plan)
    return _write(out/'result.json', dict(schema=SCHEMA, schema_version=1, handoff=source_ref,
        plan=decision, native=artifact, deployment_authorized=False))


def check(path):
    from scar.codegen.scalar_table_standalone import check as check_native

    value = read_artifact(_ref(path, SCHEMA))
    source_ref, graph = _source(value['handoff']['uri'])
    plan_ref, artifact_ref = value['plan'], value['native']
    plan = read_artifact(plan_ref)
    region = _ref(plan['region']['uri'], REGION)
    artifact = check_native(artifact_ref['uri'])
    if (value['schema_version'] != 1 or value['deployment_authorized'] is not False
            or source_ref != value['handoff'] or plan != _plan(source_ref, graph, region)
            or artifact != read_artifact(artifact_ref)):
        raise ValueError('scalar table compilation differs from cold source and native replay')
    _match_native(artifact, plan)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.check is not None and args.handoff is None and args.out is None:
        result = check(args.check)
    elif args.check is None and args.handoff is not None and args.out is not None:
        result = compile_scope(args.handoff, args.out)
    else:
        parser.error('provide --handoff and --out, or --check RESULT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
