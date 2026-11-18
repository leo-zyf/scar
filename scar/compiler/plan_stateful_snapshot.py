"""Plan a source-proved change to snapshot and payload read edges."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem,
    PlanningResource, problem_document, validate_problem,
)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode
from scar.optimization.constructive_planner import construct_implementations


SCHEMA = 'scar.stateful-snapshot-rewrite-preparation'
DEMAND = 'scar.stateful-snapshot-rewrite-demand'
_BASE = 'FULL_SNAPSHOT'
_ELIDE = 'BORROW_ENTRY_ROOTS_UNTIL_COMMIT'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _fact(artifact, locator='#'):
    return dict(artifact=artifact, locator=locator)


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _snapshot(rows):
    bytes_ = sum(row['snapshot_bytes'] for row in rows)
    return dict(copy_count=len(rows), payload_bytes=bytes_, logical_read_bytes=bytes_, logical_write_bytes=bytes_)


def _replay(native, evidence):
    from scar.analysis.workspace_step_snapshot import SCHEMA as PROOF, replay_step_snapshot_elision

    proof = replay_step_snapshot_elision(native['uri'])
    _need(evidence['schema'] == PROOF and proof == read_artifact(evidence),
        'snapshot rewrite requires the same cold-replayed original source proof')
    parents = {row['role']: row['artifact'] for row in proof['parents']['source_refs']}
    _need(parents['original_native'] == native, 'snapshot proof names another native recipe')
    step_ref = parents['step_plan']
    step = read_artifact(step_ref)
    roots = proof['roots']
    retained = [row for row in roots if row['decision'] == 'KEEP']
    elided = [row for row in roots if row['decision'] == 'ELIDE']
    _need(len(retained) + len(elided) == len(roots), 'typed root snapshot decisions required')
    before, after = _snapshot(roots), _snapshot(retained)
    removed = _snapshot(elided)
    count = proof['work_counts']
    _need((before['payload_bytes'], after['payload_bytes'], removed['payload_bytes']) == (
        count['original_snapshot_bytes'], count['remaining_snapshot_bytes'], count['elided_snapshot_bytes']),
        'snapshot demand differs from original per-root evidence')
    stages = step['stages']
    before_order = [row['id'] for row in stages]
    excluded = {row['snapshot_stage'] for row in elided}
    _need(len(excluded) == len(elided) and all(stages[row['snapshot_stage_index']]['id'] == row['snapshot_stage']
        and stages[row['snapshot_stage_index']]['kind'] == 'SNAPSHOT_ROOT' for row in roots),
        'removed stage is not the original root snapshot')
    overlay = [dict(input_id=row['input_id'], shape=row['shape'],
        removed_stage=_fact(step_ref, '#/stages/'+str(row['snapshot_stage_index'])),
        rewritten_reads=_fact(evidence, '#/roots/'+str(index)+'/borrowed_reads'),
        borrowed_reads=row['borrowed_reads'], initialized_rectangle=row['write_rectangle'],
        initialized_by=_fact(step_ref, '#/stages/'+str(row['write_stage_index'])),
        preserved_commit=_fact(step_ref, '#/stages/'+str(row['commit_stage_index'])))
        for index, row in enumerate(roots) if row['decision'] == 'ELIDE']
    hardware = step['resource_context']['target_snapshot']
    resources = dict(target=hardware, cpu_ids=step['resource_binding']['cpu_ids'],
        budget_bytes=step['memory_budget_bytes'], borrowed_payload_bytes=step['external_array_bytes'],
        private_payload_bytes=step['workspace_payload_bytes'], physical_residency='UNKNOWN')
    preserved = dict(non_snapshot_stage_order=[row['id'] for row in stages if row['kind'] != 'SNAPSHOT_ROOT'],
        input_finite_checks=True, original_commit_order=True, workspace_slots_unchanged=True,
        provider_borrowing_widened=False, runtime_guards_discharged=False)
    demand = dict(schema=DEMAND, schema_version=1, proof=evidence, original_context=proof['parents'],
        per_step=True, scope='ONE_GUARDED_TRANSACTIONAL_NATIVE_STEP', resources=resources,
        before=dict(snapshot=before, stage_order=before_order),
        after=dict(snapshot=after, stage_order=[name for name in before_order if name not in excluded],
            borrowed_root_overlay=overlay), removed_snapshot=removed, preserved=preserved,
        scattered_rectangle_contents_defined_outside_written_roi=False,
        physical_dram_traffic_bytes=None, complete_entry_cost_ns=None)
    return proof, demand


def _problem(demand, reference):
    resources = demand['resources']
    target = decode(ProviderArtifact, resources['target'])
    ref = _fact(reference)
    lane, pool = 'scoped.normal-call', 'scoped.original-payload-budget'
    options = []
    changes = bool(demand['after']['borrowed_root_overlay'])
    for mode, snapshot in ((_BASE, 'before'), (_ELIDE, 'after')) if changes else ((_BASE, 'before'),):
        source = (_fact(reference, '#/'+snapshot),)
        unknown = CostEstimate(None, None, None, 'UNKNOWN', 'snapshot-work:'+reference['sha256'], source,
            ('MISSING_WHOLE_GUARDED_STEP_COST', 'CHANGED_CACHE_AND_COPY_SERVICE_COST_UNQUALIFIED'))
        options.append(ImplementationOption(mode, mode, source, {lane:1},
            {pool:resources['private_payload_bytes']}, {}, {}, unknown,
            dict(source=source[0], native_cpu_ids=resources['cpu_ids'],
                conditions=_fact(demand['proof'], '#/original_guard_contract'), explicit_trial_only=True)))
    problem = PlanningProblem(target, (ref,), (PlanningResource(lane, 'EXECUTION', 1, (ref,)),
        PlanningResource(pool, 'MEMORY', resources['budget_bytes'], (ref,))),
        (PlanningOperation('snapshot-step:'+reference['sha256'], (ref,), (), tuple(options), _BASE),), (),
        assumptions=('NORMAL_SUCCESS_WITH_ORIGINAL_TRANSACTION_AND_GUARDS',
            'ARRAY_PAYLOAD_BUDGET_ONLY', 'NATIVE_CPU_MASK_RETAINED_WITHOUT_RESERVATION'),
        reserved_memory={pool:resources['borrowed_payload_bytes']})
    return validate_problem(problem)


def _report(native, proof_ref, demand_ref, files, demand):
    inputs = [row['input_id'] for row in demand['after']['borrowed_root_overlay']]
    return dict(schema=SCHEMA, schema_version=1, original_native=native, proof=proof_ref, demand=demand_ref,
        **files, decision='KEEP', expected_cost_available=False, default_backend='NO_OP',
        candidate=None if not inputs else dict(option=_ELIDE, elided_roots=inputs, explicit_trial_only=True),
        deployment_authorized=False)


def prepare(native, proof, out):
    from scar.analysis.workspace_step_snapshot import SCHEMA as PROOF

    path = Path(native).resolve()
    native = artifact_reference(path if path.is_file() else path/'artifact.json', 'scar.native-stateful-workspace-repeat')
    proof = artifact_reference(Path(proof).resolve(), PROOF)
    _, demand = _replay(native, proof)
    output = Path(out).resolve()
    output.mkdir(parents=True)
    demand_ref = _write(output/'demand.json', demand)
    problem = _problem(demand, demand_ref)
    files = dict(problem=_write(output/'problem.json', problem_document(problem)),
        execution_plan=_write(output/'plan.json', construct_implementations(problem)))
    report = _report(native, proof, demand_ref, files, demand)
    _write(output/'result.json', report)
    return report


def check(result):
    """Return the preparation and the independently source-replayed proof."""
    reference = result if type(result) is dict else artifact_reference(Path(result).resolve(), SCHEMA)
    report = read_artifact(reference)
    proof, demand = _replay(report['original_native'], report['proof'])
    demand_ref = report['demand']
    _need(read_artifact(demand_ref) == demand, 'snapshot demand differs from source replay')
    files = {name:report[name] for name in ('problem', 'execution_plan')}
    problem = _problem(demand, demand_ref)
    _need(read_artifact(files['problem']) == problem_document(problem), 'snapshot planning problem differs from replay')
    actual = read_artifact(files['execution_plan'])
    expected = construct_implementations(problem)
    _need({k:v for k,v in actual.items() if k != 'elapsed_seconds'} ==
        {k:v for k,v in expected.items() if k != 'elapsed_seconds'}
        and report == _report(report['original_native'], report['proof'], demand_ref, files, demand),
        'snapshot planning selection differs from replay')
    return report, proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native')
    parser.add_argument('--proof')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        report, _ = check(args.check)
    elif all((args.native, args.proof, args.out)):
        report = prepare(args.native, args.proof, args.out)
    else:
        parser.error('--native, --proof and --out are required')
    print(json.dumps({key:report[key] for key in ('decision', 'demand', 'candidate', 'deployment_authorized')}))


if __name__ == '__main__':
    main()
