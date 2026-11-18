"""Project source-bound snapshot phase evidence into a conservative cost receipt."""

import argparse
from copy import deepcopy
import json
from math import prod
from pathlib import Path
from statistics import median

from scar.compiler import observe_stateful_source_phases as observer
from scar.compiler import stateful_holdout as documents
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.stateful-snapshot-phase-cost'
DEMAND = 'scar.stateful-snapshot-rewrite-demand'
PROOF = 'scar.workspace-step-snapshot-elision-proof'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _reference(value, schema):
    return value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)


def _read(value, schema):
    document = read_artifact(value)
    _need(value['schema'] == schema and document['schema_version'] == 1, 'unexpected cost/source receipt version')
    return document


def _range(samples):
    return dict(count=len(samples), min_ns=min(samples), median_ns=median(samples), max_ns=max(samples))


def _observations(reference):
    saved = _read(reference, observer.OBSERVATIONS)
    _, design, campaign, template = observer._load(saved['design'], saved['design']['sha256'])
    _need(saved['summary'] == observer._statistics(saved['episodes'], design, campaign),
        'original measured source phase summary differs from its exact clock cuts')
    view = design['phases']['observer']['parent_view']
    plan = read_artifact(view['plan'])
    stages = [row for row in plan['stages'] if row['kind'] == 'SNAPSHOT_ROOT']
    group, _ = design['phases']['observer']['groups']
    _need([row['index'] for row in stages] == group['stage_indices'] and len(stages) > 0,
        'original time window must cover exactly the source SNAPSHOT_ROOT stages')
    roots = [{'input_id':row['target'], 'stage_id':row['id'], 'shape':row['shape'],
        'payload_bytes':prod(row['shape'])*8} for row in stages]
    _need(len({root['input_id'] for root in roots}) == len(roots), 'one copy per original root required')
    per_episode = [[observer._components(step)['snapshot'] for step in row['raw_cuts_ns']]
        for row in saved['episodes'][design['warmups']:]]
    samples = dict(first=_range([row[0] for row in per_episode]),
        continuation=_range([elapsed for row in per_episode for elapsed in row[1:]]),
        whole_episode_snapshot_sum=_range([sum(row) for row in per_episode]))
    context = dict(original_manifest=view['native_manifest'], original_source=view['source'],
        original_step_plan=view['plan'], original_region=view['region'], original_native_protocol=campaign['context']['protocol'],
        exact_shape_and_array_roles=roots, actual_abi_cpu_ids=plan['resource_binding']['cpu_ids'],
        hardware_snapshot_sha256=plan['resource_binding']['target_sha256'],
        caller_cpu_ids=campaign['context']['caller_cpu_ids'],
        requested_memory_policy=plan['resource_binding']['memory_policy'],
        actual_numa_pages=None, physical_bytes=None, exact_entry=campaign['context']['entry'],
        snapshot_allocation_scope='ORIGINAL_STANDALONE_ABI_PER_INVOCATION_WORKSPACE',
        argv=campaign['context']['argv'], original_phase_contract=deepcopy(campaign['phases']),
        measurement_basis='INSTRUMENTED_RECOMPILED_ORIGINAL_ELF_SOURCE_SNAPSHOT_WINDOW',
        training_or_holdout_clocks_used=False, empirical_ranges_are_guaranteed=False,
        original_snapshot_samples_are_correlated_within_episode=True)
    _need(view['native_manifest'] == template['native_artifact'], 'observed original implementation differs')
    return roots, samples, context


def _work(copy_count, payload):
    return dict(copy_count=copy_count, payload_bytes=payload,
        logical_read_bytes=payload, logical_write_bytes=payload)


def _validate_source_revision(demand, proof, context, original_roots):
    from scar.compiler.plan_stateful_snapshot import _replay

    parents = proof['parents']
    refs = parents['source_refs']
    mandatory_roles = ('original_native', 'region', 'step_plan', 'workspace_plan')
    _need([ref.get('role') for ref in refs[:len(mandatory_roles)]] == list(mandatory_roles),
        'snapshot proof must name its original native and graph inputs')
    native_ref = refs[0]['artifact']
    native = read_artifact(native_ref)
    original = read_artifact(context['original_manifest'])
    checked_proof, checked_demand = _replay(native_ref, demand['proof'])
    _need(proof == checked_proof and demand == checked_demand and demand['original_context'] == parents
        and proof['status'] == 'PROVEN_CONDITIONAL'
        and parents['region_sha256'] == original['region_sha256']
        and parents['step_plan_sha256'] == original['plan_sha256']
        and native['cpu_ids'] == context['actual_abi_cpu_ids'],
        'snapshot change cannot use clock evidence from a different source/CPU plan')
    expected = [dict(input_id=row['input_id'], stage_id=row['snapshot_stage'], shape=row['shape'],
        payload_bytes=row['snapshot_bytes']) for row in proof['roots']]
    _need(expected == original_roots, 'snapshot source roots do not match measured geometry')
    retained = [row for row in proof['roots'] if row['decision'] == 'KEEP']
    _need(all(row['decision'] in ('KEEP', 'ELIDE') for row in proof['roots']), 'unsupported snapshot scope change')
    return retained


def derive_snapshot_phase_cost(phase_observations, demand):
    observations_ref = _reference(phase_observations, observer.OBSERVATIONS)
    demand_ref = _reference(demand, DEMAND)
    before, measured, context = _observations(observations_ref)
    requested = _read(demand_ref, DEMAND)
    proof_ref = requested['proof']
    proof = _read(proof_ref, PROOF)
    retained = _validate_source_revision(requested, proof, context, before)
    original, candidate = requested['before']['snapshot'], requested['after']['snapshot']
    total = sum(row['payload_bytes'] for row in before)
    _need(original == _work(len(before), total)
        and candidate == _work(len(retained), sum(row['snapshot_bytes'] for row in retained)),
        'snapshot geometries must match the observed original and ordered subset')
    return dict(schema=SCHEMA, schema_version=1, phase_observations=observations_ref,
        demand=demand_ref, source_proof=proof_ref, original_context=context,
        geometry=dict(original=original, candidate=candidate,
            avoided={key:original[key]-candidate[key] for key in original},
            coverage='STATIC_LOGICAL_PAYLOAD_PER_STEP'),
        original_snapshot_reference_ns=measured, candidate_snapshot_ns=None,
        other_services_cost_change_ns=None, predicted_whole_entry_gain_ns=None,
        break_even=dict(contract='SAME_PHASE_EQUAL_RETURN_STATE_AND_RESOURCE_SCENARIO',
            inequality='NEW_SNAPSHOT_SERVICE_NS + OTHER_CHANGED_SERVICES_DELTA_NS < ORIGINAL_SNAPSHOT_SERVICE_NS',
            observed_reference_thresholds_ns=measured,
            thresholds_validity='EMPIRICAL_ORIGINAL_REFERENCE_FOR_FUTURE_MATCHED_COMPONENT_EVIDENCE',
            learned_seconds_per_byte=None, physical_bound_available=False),
        timing_transfer=dict(original_evidence_available_for='EXACT_ORIGINAL_DIAGNOSTIC_GEOMETRY_AND_PHASE_ONLY',
            candidate_exact_shape_alone_is_sufficient=False,
            independently_required=['candidate snapshot service for the exact retained roots in the new frame',
                'matching first/continuation slot lifetime, first-write placement, CPU and predecessor context',
                'independent qualification of other changed service times, including moved first writes and rereads',
                'clean matching whole-entry validation before deployment'],
            unavailable=['isolated cost of a removed root from a joint two-root interval',
                'candidate working-set residency / page-touch state',
                'uncalibrated data-layout, alias, compiler and scheduling effects']),
        estimated_cost_status='UNRESOLVED', automatic_selection_authorized=False,
        default_backend='NO_OP', deployment_authorized=False,
        implementation=artifact_reference(Path(__file__), 'scar.implementation-source'))


def check_snapshot_phase_cost(report):
    reference = _reference(report, SCHEMA)
    saved = _read(reference, SCHEMA)
    replayed = derive_snapshot_phase_cost(saved['phase_observations'], saved['demand'])
    _need(replayed == saved, 'snapshot phase cost differs from source and original raw clock replay')
    return replayed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-observations')
    parser.add_argument('--demand')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        result = check_snapshot_phase_cost(args.check)
        print(json.dumps(result, sort_keys=True))
    elif args.phase_observations and args.demand and args.out:
        result = derive_snapshot_phase_cost(args.phase_observations, args.demand)
        path = Path(args.out).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(documents._write(path, result), sort_keys=True))
    else:
        parser.error('--phase-observations, --demand and --out are required to publish a new receipt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
