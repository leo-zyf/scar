"""Plan and verify dependency-ready LibTorch execution on a recorded CPU team."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.codegen.aten_native_graph import _read_declared
from scar.compiler.plan_aten_boundary import _target, _write
from scar.compiler.plan_aten_native_graph import (
    _bundle, _compare_measured_pairs, _host_tensor_budget, _invoke, _launcher, _receipt,
)
from scar.compiler.plan_aten_native_team import team_window_spec
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (
    PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, CostEstimate, ImplementationOption,
    PlanningDataEdge, PlanningOperation, PlanningProblem, problem_document,
)
from scar.optimization.aten_native_schedule import _declared_operators
from scar.optimization.aten_native_team import model_cpu_team
from scar.optimization.aten_ready_dag import plan_ready_dag
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan


SCHEMA = 'scar.aten-ready-selection'
_CALIBRATION = 'scar.aten-generated-team-observations'
_PAIRED = 'scar.aten-ready-paired-validation'
_ENTRY_TRIAL_POLICY = 'NATIVE_SERIAL_GAIN_FOR_ENTRY_COMPARISON'


def _ready_bundle(path, digest):
    artifact, origin, recipe, hardware, executable, key = _bundle(path, digest)
    abi = artifact.get('ready_dag_abi')
    if key != 'cpu_team' or type(abi) is not int or abi not in (1, 2):
        raise ValueError('compile the original three-core ATen graph with --ready-dag')
    contract = artifact['concurrency_contract']
    _, requirements, _ = _read_declared(contract['path'], contract['sha256'])
    declared = _declared_operators(requirements)
    if any(op['schema'] not in declared for op in recipe['operations']):
        raise ValueError('every ready DAG operation requires exact concurrency qualification')
    return artifact, origin, recipe, hardware, executable, artifact[key]


def _resource_plan(hardware, schedule, origin, evidence, recipe, calibration, model):
    target, domains = _target(hardware, tuple(schedule['availability_mask']), allow_cpu_subset=True)
    refs = (dict(artifact=origin, locator='#/cpu_team'), dict(artifact=evidence, locator='#/calibration'))
    storage = model['tensor_storage']
    memory = _host_tensor_budget(hardware, storage['budget_bytes'], refs)
    resources = (*[item for domain in domains.values() for item in domain], memory)
    inputs = {spec['id']: nbytes for spec, nbytes in
              zip(recipe['inputs'], calibration['input_storage_bytes'], strict=True)}
    outputs = {op['id']: nbytes for op, nbytes in
               zip(recipe['operations'], calibration['result_storage_bytes'], strict=True)}
    join = '$ready-return'
    last_consumer = storage['accounting'] == 'RELEASE_ON_LAST_COMPLETED_CONSUMER'
    kept = set(recipe['outputs']) if last_consumer else outputs.keys()
    read = {model['timeline'][parent]['operation'] for parents in model['data_dependencies'] for parent in parents}
    locations = dict.fromkeys((value for value in outputs if value in kept), memory.id)
    operations, edges = [], []
    lane_ends = {}
    for row, native in zip(model['timeline'], recipe['operations'], strict=True):
        identifier = row['operation']
        prior = row['lane_predecessor']
        predecessors = () if prior is None else (model['timeline'][prior]['operation'],)
        operands = sorted({arg['value'] for arg in native['arguments'] if 'value' in arg})
        for value in operands:
            producer = value if value in outputs else None
            edges.append(PlanningDataEdge(f'{value}->{identifier}', value, producer, identifier,
                outputs[value] if producer else inputs[value], None if producer else memory.id, source_refs=refs))
        if identifier in kept:
            edges.append(PlanningDataEdge(f'{identifier}->{join}', identifier, identifier, join,
                                          outputs[identifier], None, source_refs=refs))
        has_consumer = identifier in kept or identifier in read
        demands = {resource.id: 1 for resource in domains[row['cpu_id']]}
        duration = row['end_ns'] - row['start_ns']
        option = ImplementationOption('CPU_LANE', 'EXISTING_LIBTORCH_READY_DAG', refs, demands,
            {} if has_consumer else {memory.id: outputs[identifier]}, dict.fromkeys(operands, memory.id),
            {identifier: memory.id} if has_consumer else {},
            CostEstimate(duration, None, None, 'EMPIRICAL_MODEL', 'ISOLATED_ATEN_LANE', refs),
            dict(cpu_id=row['cpu_id'], lane=row['lane']))
        operations.append(PlanningOperation(identifier, refs, predecessors, (option,), 'CPU_LANE'))
        lane_ends[row['lane']] = identifier
    return_uses = ImplementationOption('RETURN', 'LOGICAL_TENSOR_RETENTION_ENDPOINT', refs,
        {}, {}, locations, {}, CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT', refs),
        dict(exported_tensors=recipe['outputs']))
    join_predecessors = tuple(lane_ends[lane] for lane in sorted(lane_ends)) if last_consumer else ()
    operations.append(PlanningOperation(join, refs, join_predecessors, (return_uses,), 'RETURN'))
    lifetime_assumption = ('LAST_CONSUMER_OR_EXPORT_EXCLUDING_PROVIDER_WORKSPACE' if last_consumer
                           else 'RETAINS_ALL_TENSOR_HANDLES_EXCLUDING_PROVIDER_WORKSPACE')
    problem = PlanningProblem(target, refs, resources, tuple(operations), tuple(edges),
        assumptions=('DATA_AND_LOCAL_LANE_ORDER', 'DECLARED_CONCURRENT_OPERATOR_EFFECTS',
                     lifetime_assumption, 'SERVICES_EXCLUDE_DISPATCH_AND_JOIN'),
        reserved_memory={memory.id: storage['persistent_input_bytes']})
    execution = construct_implementations(problem)
    verified = verify_execution_plan(problem, execution)
    actual_peak = verified['resource_peaks'][memory.id]
    envelope = storage['peak_tensor_storage_bytes']
    if actual_peak > envelope or (not last_consumer and actual_peak != envelope):
        raise ValueError('ready DAG material lifetimes exceed the admitted storage envelope')
    observed = {row['operation']: (row['start_ns'], row['end_ns']) for row in execution['selected']}
    expected_join = max(row['end_ns'] for row in model['timeline'])
    if observed.pop(join) != (expected_join, expected_join):
        raise ValueError('ready DAG return is not synchronized with all of its operations')
    if observed != {row['operation']: (row['start_ns'], row['end_ns']) for row in model['timeline']}:
        raise ValueError('ready DAG timeline differs from its fixed-lane resource certificate')
    return problem_document(problem), execution


def _case(raw, model, baseline, warmup, count, storage, *, recipe):
    if (raw['all_pairs_validated'] is not True or
            (raw['lanes'], raw['baseline'], raw['cpu_ids'], raw['warmup_pairs'], raw['measured_pairs'],
             raw['result_storage_bytes']) != (model['lanes'], baseline, model['cpu_ids'], warmup, count, storage)):
        raise ValueError('ready DAG validation differs from its planned boundary or storage geometry')
    residency = model['tensor_storage']
    lifetime = None
    if residency['accounting'] == 'RELEASE_ON_LAST_COMPLETED_CONSUMER':
        outputs = set(recipe['outputs'])
        releases = [i for i, op in enumerate(recipe['operations']) if op['id'] not in outputs]
        retained = residency['persistent_input_bytes'] + sum(
            nbytes for op, nbytes in zip(recipe['operations'], storage, strict=True) if op['id'] in outputs)
        lifetime = raw['ready_lifetime']
        peak = lifetime['graph_held_peak_bytes']
        if (lifetime['released_indices'] != releases or lifetime['retained_bytes_at_return'] != retained
                or type(peak) is not int or not retained <= peak <= residency['peak_tensor_storage_bytes']):
            raise ValueError('ready DAG allocation/release observation does not satisfy its storage proof')
    rows = []
    for i, row in enumerate(raw['samples']):
        order = 'AB' if (i + warmup) % 2 == 0 else 'BA'
        if row.keys() != {'order', 'baseline_ns', 'ready_ns'} or row['order'] != order:
            raise ValueError('ready DAG validation requires balanced alternating cases')
        modes = ['SERIAL', 'TWO_CPU'] if order == 'AB' else ['TWO_CPU', 'SERIAL']
        rows.append(dict(pair_index=i, order=modes, serial_ns=row['baseline_ns'], dual_ns=row['ready_ns'],
                         bitwise_identical=True, input_values_unchanged=True, fresh_owned_outputs=True))
    metrics, wins, directional, ratio, faster = _compare_measured_pairs(rows, count)
    results = dict(baseline=baseline, metrics={'BASELINE': metrics['SERIAL'], 'READY_DAG': metrics['TWO_CPU']},
                wins=wins, paired_order_differences={'AB': directional['SERIAL'], 'BA': directional['TWO_CPU']},
                speedup=ratio, profitable=faster)
    if lifetime is not None:
        results['untimed_lifetime_observation'] = lifetime
    return results


def _derive(recipe, schedule, raw, budget, warmup, *, ready_dag_abi):
    if raw['cpu_ids'] != schedule['cpus'] or raw['warmup_pairs'] != warmup:
        raise ValueError('ready DAG calibration belongs to a different CPU execution context')
    calibration = raw['calibration']
    model = plan_ready_dag(recipe, schedule, calibration, warmup, tensor_budget_bytes=budget, ready_dag_abi=ready_dag_abi)
    baselines = ['serial']
    if model['status'] == 'PLAN_READY_DAG' and model['tensor_storage']['serial_peak_bytes'] <= budget:
        windows = model_cpu_team(calibration, recipe, schedule, warmup, tensor_budget_bytes=budget)['selected_windows']
        if windows:
            baselines.append(team_window_spec(windows))
    return model, baselines


def plan_aten_ready(bundle, *, bundle_sha256, input_paths, out, tensor_budget_bytes, warmup_pairs=5,
                    measured_pairs=12, primitive_observation=None, entry_comparison=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 3 <= warmup_pairs <= 100 or not 4 <= measured_pairs <= 500 or measured_pairs % 2:
        raise ValueError('ready DAG requires 3–100 warmups and an even number of 4–500 measurement pairs')
    artifact, origin, recipe, hardware, executable, schedule = _ready_bundle(bundle, bundle_sha256)
    inputs = [_receipt(path) for path in input_paths]
    if len(inputs) != len(recipe['inputs']):
        raise ValueError('supply inputs in original recipe order')
    guarded = [_receipt(executable), *artifact['provider']['libraries'], *inputs]
    if primitive_observation is None:
        raw = _invoke(executable, schedule, '--calibrate-team', [warmup_pairs], inputs)
    else:
        if primitive_observation['inputs'] != inputs:
            raise ValueError('ready DAG calibration used different input bytes')
        raw = primitive_observation['raw']
    abi = artifact['ready_dag_abi']
    model, baselines = _derive(recipe, schedule, raw, tensor_budget_bytes, warmup_pairs, ready_dag_abi=abi)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-ready-', dir=out.parent) as temporary:
        stage = Path(temporary)

        def publish(name, schema, contents):
            path = stage / name
            _write(path, dict(contents, schema=schema, schema_version=1))
            return dict(artifact_reference(path, schema), uri=str(out / name))

        calibration = publish('observations.json', _CALIBRATION, raw)
        validations, measurements = [], []
        reason = model['reason']
        if model['status'] == 'PLAN_READY_DAG':
            problem, execution = _resource_plan(hardware, schedule, origin, calibration, recipe, raw['calibration'], model)
            _write(stage / 'problem.json', problem)
            _write(stage / 'plan.json', execution)
            lanes = ''.join(map(str, model['lanes']))
            for i, baseline in enumerate(baselines):
                measured = _invoke(executable, schedule, '--benchmark-ready',
                                   [lanes, baseline, warmup_pairs, measured_pairs], inputs)
                validations.append(_case(measured, model, baseline, warmup_pairs, measured_pairs,
                                         raw['calibration']['result_storage_bytes'], recipe=recipe))
                measurements.append(publish(f'paired-{i}.json', _PAIRED, measured))
            if not all(result['profitable'] for result in (validations[:1] if entry_comparison else validations)):
                reason = 'READY_DAG_DID_NOT_BEAT_MATCHED_BASELINES'
        if any(_receipt(row['path']) != row for row in guarded):
            raise ValueError('native ready DAG members changed during measurement')
        if reason is None and all(result['profitable'] for result in validations):
            _launcher(stage, executable, 'READY_DAG', schedule, inputs, guarded, model['lanes'])
        report = dict(schema=SCHEMA, schema_version=abi, bundle=origin, executable=guarded[0], inputs=inputs,
            cpu_ids=model['cpu_ids'], model=model, observations=calibration, paired_validation=measurements,
            validation=validations, warmup_pairs=warmup_pairs, measured_pairs=measured_pairs,
            selected='READY_DAG' if reason is None else 'KEEP_ORIGINAL', reason=reason,
            source_replacement_authorized=False)
        if entry_comparison:
            report['validation_policy'] = _ENTRY_TRIAL_POLICY
        _write(stage / 'selection.json', report)
        stage.rename(out)
    return report


def read_ready_selection(path, selection):
    if selection['schema'] != SCHEMA or selection['selected'] != 'READY_DAG':
        raise ValueError('selected and independently validated ready DAG required')
    bundle = selection['bundle']
    artifact, origin, recipe, hardware, executable, schedule = _ready_bundle(bundle['uri'], bundle['sha256'])
    abi = artifact['ready_dag_abi']
    if (origin != bundle or selection['schema_version'] != abi
            or selection['executable'] != _receipt(executable)):
        raise ValueError('ready DAG selection does not identify the original executable')
    calibration = selection['observations']
    raw = read_artifact(calibration)
    warmup, count = selection['warmup_pairs'], selection['measured_pairs']
    if not 3 <= warmup <= 100 or not 4 <= count <= 500 or count % 2:
        raise ValueError('ready DAG selection requires its balanced matched trial')
    budget = selection['model']['tensor_storage']['budget_bytes']
    model, baselines = _derive(recipe, schedule, raw, budget, warmup, ready_dag_abi=abi)
    results = [_case(read_artifact(ref), model, baseline, warmup, count, raw['calibration']['result_storage_bytes'], recipe=recipe)
               for ref, baseline in zip(selection['paired_validation'], baselines, strict=True)]
    policy = selection.get('validation_policy')
    if policy not in (None, _ENTRY_TRIAL_POLICY):
        raise ValueError('unknown ready DAG validation policy')
    required = results[:1] if policy == _ENTRY_TRIAL_POLICY else results
    if (model != selection['model'] or model['status'] != 'PLAN_READY_DAG' or model['cpu_ids'] != selection['cpu_ids']
            or not required or results != selection['validation'] or not all(row['profitable'] for row in required)):
        raise ValueError('retained ready DAG choice differs from its calibration or matched validation')
    problem, execution = _resource_plan(hardware, schedule, origin, calibration, recipe, raw['calibration'], model)
    problem_ref = artifact_reference(path.with_name('problem.json'), PROBLEM_SCHEMA)
    plan_ref = artifact_reference(path.with_name('plan.json'), PLAN_SCHEMA)
    stored = read_artifact(plan_ref)
    for plan in (stored, execution):
        plan.pop('elapsed_seconds', None)
    if read_artifact(problem_ref) != problem or stored != execution:
        raise ValueError('retained ready DAG execution certificate differs from its original evidence')
    return selection, artifact, recipe, dict(selection=artifact_reference(path, SCHEMA),
        planning_problem=problem_ref, execution_plan=plan_ref, calibration=calibration,
        paired_validation=selection['paired_validation'])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-aten-ready', description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--tensor-budget-bytes', required=True, type=int)
    parser.add_argument('--warmup-pairs', type=int, default=5)
    parser.add_argument('--measured-pairs', type=int, default=12)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    selection = plan_aten_ready(args.bundle, bundle_sha256=args.bundle_sha256, input_paths=args.input,
        tensor_budget_bytes=args.tensor_budget_bytes, warmup_pairs=args.warmup_pairs,
        measured_pairs=args.measured_pairs, out=args.out)
    print(json.dumps(dict(selected=selection['selected'], reason=selection['reason'],
        selection=str(Path(args.out) / 'selection.json'))))
    return 0
