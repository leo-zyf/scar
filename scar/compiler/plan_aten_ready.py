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
from scar.optimization.aten_boundary_trials import ATEN_GAIN_POLICY
from scar.optimization.aten_native_schedule import _declared_operators
from scar.optimization.aten_native_team import model_cpu_team
from scar.optimization.aten_ready_dag import plan_ready_dag
from scar.optimization.aten_ready_replica import nominate_ready_replica, price_ready_replica
from scar.optimization.aten_ready_retained import price_ready_retained
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan


SCHEMA = 'scar.aten-ready-selection'
_CALIBRATION = 'scar.aten-generated-team-observations'
_PAIRED = 'scar.aten-ready-paired-validation'
_ENTRY_TRIAL_POLICY = 'NATIVE_SERIAL_GAIN_FOR_ENTRY_COMPARISON'
_NUMA_REQUIRED = 'READY_DAG_NUMA_CALIBRATION_REQUIRED'
_REPLICA_CALIBRATION = 'scar.aten-ready-replica-calibration'
_REPLICA_PAIRED = 'scar.aten-ready-replica-paired'
_NATIVE_ONLY = 'STANDALONE_CPP20_ONLY'


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


def _resource_plan(hardware, schedule, origin, evidence, recipe, calibration, model, *, replica=None,
                   program_graph=None, recorded_resources=None):
    storage = model['tensor_storage']
    if program_graph is None:
        refs = (dict(artifact=origin, locator='#/cpu_team'), dict(artifact=evidence, locator='#/calibration'))
    else:
        refs = (dict(artifact=evidence, locator='#/evidence/team'),
                dict(artifact=evidence, locator='#/evidence/calibration/calibration'))
    if recorded_resources is None:
        target, domains = _target(hardware, tuple(schedule['availability_mask']), allow_cpu_subset=True)
        memory = _host_tensor_budget(hardware, storage['budget_bytes'], refs)
    else:
        target, domains, memory = recorded_resources
    resources = (*[item for domain in domains.values() for item in domain], memory)
    inputs = {spec['id']: nbytes for spec, nbytes in
              zip(recipe['inputs'], calibration['input_storage_bytes'], strict=True)}
    outputs = {op['id']: nbytes for op, nbytes in
               zip(recipe['operations'], calibration['result_storage_bytes'], strict=True)}
    join = '$ready-return' if program_graph is None else '$return'
    last_consumer = storage['accounting'] == 'RELEASE_ON_LAST_COMPLETED_CONSUMER'
    kept = set(recipe['outputs']) if last_consumer else outputs.keys()
    read = {model['timeline'][parent]['operation'] for parents in model['data_dependencies'] for parent in parents}
    locations = dict.fromkeys((value for value in outputs if value in kept), memory.id)
    locality = model.get('ready_numa')
    page_index = {value: f'input_pages/{i}' for i, value in enumerate(inputs)}
    page_index.update({value: f'output_pages/{i}' for i, value in enumerate(outputs)})
    cost_context = 'ATEN_READY_NUMA_PROVIDER_REPLAY' if locality else 'ISOLATED_ATEN_LANE'
    cost_refs = (*refs, dict(artifact=evidence, locator='#/calibration/ready_numa/operation_samples_ns')) if locality else refs
    operations, edges = [], []
    timeline = model['timeline']
    copy_id, replica_value = '$ready-input-copy', '$ready-input-replica'
    replica_witness = None
    retained = False
    if replica is not None:
        retained = replica.get('retained', False)
        transform = replica['model']
        candidate = replica['candidate']
        timeline = transform['timeline']
        copy = transform['copy_prologue']
        replica_witness = dict(artifact=replica['calibration'], locator='#/raw/replica_pages')
        price_field = 'check_samples_ns' if retained else 'copy_samples_ns'
        copy_refs = (*refs, replica_witness, dict(artifact=replica['calibration'], locator=f'#/raw/{price_field}'),
                     dict(artifact=origin, locator=f"#/cpu_team/lanes/{copy['lane']}"))
        value, size = candidate['value'], candidate['allocated_bytes']
        cpu = copy['cpu_id']
        demand = {resource.id: 1 for resource in domains[cpu]}
        if retained:
            copy_id = '$ready-input-check'
        action = 'CHECK' if retained else 'COPY'
        replica_producer = None if retained else copy_id
        replica_source = memory.id if retained else None
        prologue_inputs = {value: memory.id}
        if retained:
            prologue_inputs[replica_value] = memory.id
            edges.append(PlanningDataEdge(f'{replica_value}->{copy_id}', replica_value, None, copy_id,
                                          size, memory.id, source_refs=copy_refs))
        implementation = ImplementationOption(action,
            'EXACT_RETAINED_NUMA_REPLICA_CHECK' if retained else 'LIBTORCH_COPY_INTO_FIRST_TOUCHED_HOST_STORAGE', copy_refs,
            demand, {}, prologue_inputs, {} if retained else {replica_value: memory.id},
            CostEstimate(copy['end_ns'] - copy['start_ns'], None, None, 'EMPIRICAL_MODEL',
                         'ATEN_READY_RETAINED_CHECK' if retained else 'ATEN_READY_REPLICA_COPY', copy_refs),
            dict(logical_value=value, replica_value=replica_value, allocation_bytes=size, target_numa_node=candidate['dest_node'],
                 cpu_id=cpu, lane=copy['lane']))
        operations.append(PlanningOperation(copy_id, copy_refs, (), (implementation,), action))
        edges.append(PlanningDataEdge(f'{value}->{copy_id}', value, None, copy_id, inputs[value], memory.id,
                                      source_refs=copy_refs))
        edges.append(PlanningDataEdge(f'{replica_value}->{join}', replica_value, replica_producer, join, size, replica_source,
                                      source_refs=copy_refs))
        locations[replica_value] = memory.id
        cost_context = 'ATEN_READY_RETAINED_PROVIDER_REPLAY' if retained else 'ATEN_READY_REPLICATED_PROVIDER_REPLAY'
        operation_samples = 'retained_samples_ns' if retained else 'replicated_samples_ns'
        cost_refs = (*refs, dict(artifact=replica['calibration'], locator=f'#/raw/{operation_samples}'))
    lane_ends = {}
    order = model.get('operation_order', range(len(timeline)))
    for index in order:
        row, native = timeline[index], recipe['operations'][index]
        identifier = row['operation']
        source_refs = refs if program_graph is None else (
            dict(program_graph, operation=f'program_operation:entry/operation/{identifier}'),)
        prior = row['lane_predecessor']
        predecessors = () if prior is None else (timeline[prior]['operation'],)
        if replica is not None and prior is None and row['lane'] == copy['lane']:
            predecessors = (copy_id,)
        operands = sorted({arg['value'] for arg in native['arguments'] if 'value' in arg})
        input_values = []
        for value in operands:
            if replica is not None and value == candidate['value'] and row['index'] in candidate['consumers']:
                edge_refs = (*refs, replica_witness, dict(artifact=origin, locator=f"#/cpu_team/lanes/{row['lane']}"))
                edges.append(PlanningDataEdge(f'{replica_value}->{identifier}', replica_value, replica_producer, identifier,
                                              size, replica_source, source_refs=edge_refs))
                input_values.append(replica_value)
                continue
            producer = value if value in outputs else None
            edge_refs = source_refs
            if locality is not None:
                edge_refs = (*refs, dict(artifact=evidence, locator='#/calibration/ready_numa/' + page_index[value]),
                             dict(artifact=origin, locator=f"#/cpu_team/lanes/{row['lane']}"))
            edges.append(PlanningDataEdge(f'{value}->{identifier}', value, producer, identifier,
                outputs[value] if producer else inputs[value], None if producer else memory.id, source_refs=edge_refs))
            input_values.append(value)
        if identifier in kept:
            edges.append(PlanningDataEdge(f'{identifier}->{join}', identifier, identifier, join,
                                          outputs[identifier], None, source_refs=refs))
        has_consumer = identifier in kept or identifier in read
        demands = {resource.id: 1 for resource in domains[row['cpu_id']]}
        duration = row['end_ns'] - row['start_ns']
        option = ImplementationOption('CPU_LANE', 'EXISTING_LIBTORCH_READY_DAG', refs, demands,
            {} if has_consumer else {memory.id: outputs[identifier]}, dict.fromkeys(input_values, memory.id),
            {identifier: memory.id} if has_consumer else {},
            CostEstimate(duration, None, None, 'EMPIRICAL_MODEL', cost_context, cost_refs),
            dict(cpu_id=row['cpu_id'], lane=row['lane']))
        operations.append(PlanningOperation(identifier, source_refs, predecessors, (option,), 'CPU_LANE'))
        lane_ends[row['lane']] = identifier
    return_uses = ImplementationOption('RETURN', 'LOGICAL_TENSOR_RETENTION_ENDPOINT', refs,
        {}, {}, locations, {}, CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT', refs),
        dict(exported_tensors=recipe['outputs']))
    join_predecessors = tuple(lane_ends[lane] for lane in sorted(lane_ends)) if last_consumer else ()
    operations.append(PlanningOperation(join, refs, join_predecessors, (return_uses,), 'RETURN'))
    lifetime_assumption = ('LAST_CONSUMER_OR_EXPORT_EXCLUDING_PROVIDER_WORKSPACE' if last_consumer
                           else 'RETAINS_ALL_TENSOR_HANDLES_EXCLUDING_PROVIDER_WORKSPACE')
    assumptions = ('DATA_AND_LOCAL_LANE_ORDER', 'DECLARED_CONCURRENT_OPERATOR_EFFECTS',
                   lifetime_assumption, 'SERVICES_EXCLUDE_DISPATCH_AND_JOIN')
    if 'operation_order' in model:
        assumptions = (*assumptions, 'TOPOLOGICAL_OPERATION_ORDER_BOUND_TO_NATIVE_EXECUTOR')
    if locality is not None:
        assumptions = (*assumptions, 'SAMPLED_NUMA_RESIDENCY_IN_SHARED_HOST_MEMORY')
    if replica is not None:
        assumptions = (*assumptions, 'ONE_PRIVATE_INPUT_REPLICA_HELD_TO_RETURN')
        if retained:
            assumptions = (*assumptions, 'WARM_HIT_ON_ONE_PRIOR_PRIVATE_MAPPING_AFTER_EXACT_CONTENT_CHECK')
    problem = PlanningProblem(target, refs if program_graph is None else (program_graph,),
        resources, tuple(operations), tuple(edges),
        assumptions=assumptions,
        reserved_memory={memory.id: storage['persistent_input_bytes'] + (size if retained else 0)})
    execution = construct_implementations(problem)
    verified = verify_execution_plan(problem, execution)
    actual_peak = verified['resource_peaks'][memory.id]
    envelope = storage['peak_tensor_storage_bytes']
    if replica is not None:
        envelope += size
    if actual_peak > envelope or (not last_consumer and actual_peak != envelope):
        raise ValueError('ready DAG material lifetimes exceed the admitted storage envelope')
    observed = {row['operation']: (row['start_ns'], row['end_ns']) for row in execution['selected']}
    expected_join = max(row['end_ns'] for row in timeline)
    if observed.pop(join) != (expected_join, expected_join):
        raise ValueError('ready DAG return is not synchronized with all of its operations')
    expected = {row['operation']: (row['start_ns'], row['end_ns']) for row in timeline}
    if replica is not None:
        expected[copy_id] = (copy['start_ns'], copy['end_ns'])
    if observed != expected:
        raise ValueError('ready DAG timeline differs from its fixed-lane resource certificate')
    return problem_document(problem), execution


def _case(raw, model, baseline, warmup, count, storage, *, recipe):
    if (raw['all_pairs_validated'] is not True or
            (raw['lanes'], raw['baseline'], raw['cpu_ids'], raw['warmup_pairs'], raw['measured_pairs'],
             raw['result_storage_bytes']) != (model['lanes'], baseline, model['cpu_ids'], warmup, count, storage)
            or raw.get('operation_order') != model.get('operation_order')):
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
    metrics, wins, directional, ratio, faster = _compare_measured_pairs(rows, count,
                                                                      gain_policy=model.get('gain_policy'))
    results = dict(baseline=baseline, metrics={'BASELINE': metrics['SERIAL'], 'READY_DAG': metrics['TWO_CPU']},
                wins=wins, paired_order_differences={'AB': directional['SERIAL'], 'BA': directional['TWO_CPU']},
                speedup=ratio, profitable=faster)
    if lifetime is not None:
        results['untimed_lifetime_observation'] = lifetime
    return results


def _ready_model(recipe, schedule, raw, budget, warmup, *, ready_dag_abi, joint_order=False,
                 gain_policy=ATEN_GAIN_POLICY):
    if raw['cpu_ids'] != schedule['cpus'] or raw['warmup_pairs'] != warmup:
        raise ValueError('ready DAG calibration belongs to a different CPU execution context')
    return plan_ready_dag(recipe, schedule, raw['calibration'], warmup, tensor_budget_bytes=budget,
                          ready_dag_abi=ready_dag_abi, joint_order=joint_order, gain_policy=gain_policy)


def _baselines(recipe, schedule, calibration, model, budget, warmup):
    baselines = ['serial']
    if model['status'] == 'PLAN_READY_DAG' and model['tensor_storage']['serial_peak_bytes'] <= budget:
        windows = model_cpu_team(calibration, recipe, schedule, warmup, tensor_budget_bytes=budget)['selected_windows']
        if windows:
            baselines.append(team_window_spec(windows))
    return baselines


def _source_order_baseline(recipe, schedule, raw, model, budget, warmup, abi):
    if 'operation_order' in model and model['status'] == 'PLAN_READY_DAG':
        source_order = _ready_model(recipe, schedule, raw, budget, warmup, ready_dag_abi=abi,
                                    gain_policy=model.get('gain_policy'))
        if source_order['status'] == 'PLAN_READY_DAG':
            return source_order, 'ready:' + ''.join(map(str, source_order['lanes']))
    return None, None


def calibrate_ready_placement(executable, recipe, schedule, inputs, raw, *, budget, warmup, ready_dag_abi,
                              joint_order=False):
    model = _ready_model(recipe, schedule, raw, budget, warmup, ready_dag_abi=ready_dag_abi, joint_order=joint_order)
    if model['status'] != _NUMA_REQUIRED:
        return raw, model
    lanes = ''.join(map(str, model['lanes']))
    measured = _invoke(executable, schedule, '--calibrate-ready-locality', [lanes, warmup], inputs)
    if measured['cpu_ids'] != schedule['cpus'] or measured['warmup_pairs'] != warmup:
        raise ValueError('ready NUMA calibration ran outside its nominated placement')
    raw = dict(raw, calibration=dict(raw['calibration'], ready_numa=measured['ready_numa']))
    model = _ready_model(recipe, schedule, raw, budget, warmup, ready_dag_abi=ready_dag_abi)
    if model['status'] == _NUMA_REQUIRED:
        raise ValueError('ready NUMA calibration did not resolve the requested locality evidence')
    return raw, model


def _replica_validation(raw, model, candidate, warmup, count, *, retained=False):
    if (raw['all_pairs_validated'] is not True or
            (raw['cpu_ids'], raw['lanes'], raw['input_index'], raw['lane'], raw['warmup_pairs'], raw['measured_pairs']) !=
            (model['cpu_ids'], model['lanes'], candidate['input_index'], candidate['lane'], warmup, count)):
        raise ValueError('ready input replica validation ran outside its nominated configuration')
    if retained and (type(raw['retained_hits']) is not int or type(raw['retained_misses']) is not int
                     or raw['retained_hits'] != count or raw['retained_misses'] != 0):
        raise ValueError('ready retained comparison requires only hits in the measured warm pairs')
    rows = []
    for i, row in enumerate(raw['samples']):
        order = 'AB' if (i + warmup) % 2 == 0 else 'BA'
        if row.keys() != {'order', 'baseline_ns', 'ready_ns'} or row['order'] != order:
            raise ValueError('ready input replica validation requires balanced alternating samples')
        modes = ['SERIAL', 'TWO_CPU'] if order == 'AB' else ['TWO_CPU', 'SERIAL']
        rows.append(dict(pair_index=i, order=modes, serial_ns=row['baseline_ns'], dual_ns=row['ready_ns'],
                         bitwise_identical=True, input_values_unchanged=True, fresh_owned_outputs=True))
    metrics, wins, directional, ratio, faster = _compare_measured_pairs(rows, count,
                                                                      gain_policy=model.get('gain_policy'))
    label = 'RETAINED' if retained else 'REPLICATED'
    result = dict(metrics={'DIRECT': metrics['SERIAL'], label: metrics['TWO_CPU']}, wins=wins,
        paired_order_differences={'AB': directional['SERIAL'], 'BA': directional['TWO_CPU']}, speedup=ratio, profitable=faster)
    if retained:
        result.update(retained_hits=raw['retained_hits'], retained_misses=raw['retained_misses'])
    return result


def _try_replica(executable, recipe, schedule, model, inputs, warmup, count, publish, *, retained=False):
    candidate = nominate_ready_replica(recipe, schedule, model)
    record = dict(candidate=candidate)
    if retained:
        record['retained'] = True
    if candidate is None:
        return dict(record, decision='NO_ADMISSIBLE_INPUT_REPLICA')
    spec = [''.join(map(str, model['lanes'])), candidate['input_index'], candidate['lane'], warmup]
    mode = 'retained' if retained else 'replica'
    stage = 'RETAINED' if retained else 'REPLICA'
    raw = _invoke(executable, schedule, f'--calibrate-ready-{mode}', spec, inputs)
    price = price_ready_retained if retained else price_ready_replica
    priced = price(recipe, schedule, model, candidate, raw, warmup)
    record.update(model=priced, decision=priced['status'],
                  calibration=publish('input-replica.json', _REPLICA_CALIBRATION, dict(raw=raw)))
    if priced['status'] == f'PLAN_READY_{stage}':
        paired = _invoke(executable, schedule, f'--benchmark-ready-{mode}', [*spec, count], inputs)
        validation = _replica_validation(paired, model, candidate, warmup, count, retained=retained)
        record.update(paired=publish('input-replica-paired.json', _REPLICA_PAIRED, dict(raw=paired)),
                      validation=validation, decision=f'READY_DAG_{stage}' if validation['profitable'] else 'KEEP_DIRECT_READY')
    return record


def _replay_replica(recipe, schedule, model, replica, warmup, count):
    candidate = nominate_ready_replica(recipe, schedule, model)
    if replica['candidate'] != candidate:
        raise ValueError('ready input replica differs from the qualified DATA edges')
    retained = 'retained' in replica
    if retained and replica['retained'] is not True:
        raise ValueError('invalid retained replica profile')
    if candidate is None:
        expected = dict(candidate=None, decision='NO_ADMISSIBLE_INPUT_REPLICA')
        if retained:
            expected['retained'] = True
        if replica != expected:
            raise ValueError('ready selection has evidence for an unqualified input replica')
        return 'READY_DAG'
    raw = read_artifact(replica['calibration'])['raw']
    pricer = price_ready_retained if retained else price_ready_replica
    stage = 'RETAINED' if retained else 'REPLICA'
    price = pricer(recipe, schedule, model, candidate, raw, warmup)
    if replica['model'] != price:
        raise ValueError('ready input replica price differs from its service measurements')
    decision = price['status']
    if decision == f'PLAN_READY_{stage}':
        validation = _replica_validation(read_artifact(replica['paired'])['raw'], model, candidate, warmup, count,
                                         retained=retained)
        if validation != replica['validation']:
            raise ValueError('ready input replica validation differs from its paired measurements')
        decision = f'READY_DAG_{stage}' if validation['profitable'] else 'KEEP_DIRECT_READY'
    if replica['decision'] != decision:
        raise ValueError('ready input replica choice differs from its replay')
    return decision if decision == f'READY_DAG_{stage}' else 'READY_DAG'


def plan_aten_ready(bundle, *, bundle_sha256, input_paths, out, tensor_budget_bytes, warmup_pairs=5,
                    measured_pairs=12, primitive_observation=None, entry_comparison=False, try_input_replica=False,
                    try_retained_input_replica=False, joint_order=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 3 <= warmup_pairs <= 100 or not 4 <= measured_pairs <= 500 or measured_pairs % 2:
        raise ValueError('ready DAG requires 3–100 warmups and an even number of 4–500 measurement pairs')
    artifact, origin, recipe, hardware, executable, schedule = _ready_bundle(bundle, bundle_sha256)
    if joint_order and (artifact.get('ready_order_abi') != 1 or entry_comparison or try_input_replica
                        or try_retained_input_replica or schedule.get('cross_numa', False)):
        raise ValueError('joint ordering requires a current same-NUMA native-only graph')
    if try_input_replica and artifact.get('ready_replica_abi') != 1:
        raise ValueError('input replica planning requires a bundle built with --ready-dag --allow-cross-numa')
    if try_retained_input_replica and (try_input_replica or artifact.get('ready_retained_abi') != 1):
        raise ValueError('select one replica trial and use a current retained-capable cross-NUMA bundle')
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
    raw, model = calibrate_ready_placement(executable, recipe, schedule, inputs, raw,
        budget=tensor_budget_bytes, warmup=warmup_pairs, ready_dag_abi=abi, joint_order=joint_order)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-ready-', dir=out.parent) as temporary:
        stage = Path(temporary)

        def publish(name, schema, contents):
            path = stage / name
            _write(path, dict(contents, schema=schema, schema_version=1))
            return dict(artifact_reference(path, schema), uri=str(out / name))

        calibration = publish('observations.json', _CALIBRATION, raw)
        validations, measurements = [], []
        replica = None
        order_trial = None
        if joint_order:
            original_model, baseline = _source_order_baseline(recipe, schedule, raw, model,
                                                               tensor_budget_bytes, warmup_pairs, abi)
            if original_model is not None:
                spec = [''.join(map(str, model['lanes'])), ','.join(map(str, model['operation_order']))]
                paired = _invoke(executable, schedule, '--benchmark-ready-ordered',
                                 [*spec, baseline, warmup_pairs, measured_pairs], inputs)
                validation = _case(paired, model, baseline, warmup_pairs, measured_pairs,
                                   raw['calibration']['result_storage_bytes'], recipe=recipe)
                order_trial = dict(prediction=model['joint_cpu_plan'], validation=validation,
                                   observations=publish('joint-order-paired.json', _PAIRED, paired))
                if not validation['profitable']:
                    model = original_model
        baselines = _baselines(recipe, schedule, raw['calibration'], model, tensor_budget_bytes, warmup_pairs)
        selected = 'READY_DAG'
        reason = model['reason']
        if model['status'] == 'PLAN_READY_DAG':
            lanes = ''.join(map(str, model['lanes']))
            order = model.get('operation_order')
            mode, prefix = '--benchmark-ready', [lanes]
            if order is not None:
                mode, prefix = '--benchmark-ready-ordered', [lanes, ','.join(map(str, order))]
            for i, baseline in enumerate(baselines):
                measured = _invoke(executable, schedule, mode, [*prefix, baseline, warmup_pairs, measured_pairs], inputs)
                validations.append(_case(measured, model, baseline, warmup_pairs, measured_pairs,
                                         raw['calibration']['result_storage_bytes'], recipe=recipe))
                measurements.append(publish(f'paired-{i}.json', _PAIRED, measured))
            if not all(result['profitable'] for result in (validations[:1] if entry_comparison else validations)):
                reason = 'READY_DAG_DID_NOT_BEAT_MATCHED_BASELINES'
            if try_input_replica or try_retained_input_replica:
                replica = _try_replica(executable, recipe, schedule, model, inputs, warmup_pairs, measured_pairs, publish,
                                       retained=try_retained_input_replica)
                if reason is None and replica['decision'] in ('READY_DAG_REPLICA', 'READY_DAG_RETAINED'):
                    selected = replica['decision']
            problem, execution = _resource_plan(hardware, schedule, origin, calibration, recipe, raw['calibration'], model,
                                               replica=replica if selected != 'READY_DAG' else None)
            _write(stage / 'problem.json', problem)
            _write(stage / 'plan.json', execution)
        if any(_receipt(row['path']) != row for row in guarded):
            raise ValueError('native ready DAG members changed during measurement')
        delivered = 'KEEP_ORIGINAL'
        if reason is None and selected != 'READY_DAG_RETAINED' and all(result['profitable'] for result in validations):
            placement = replica['candidate'] if selected == 'READY_DAG_REPLICA' else None
            _launcher(stage, executable, selected, schedule, inputs, guarded, model['lanes'], replica=placement,
                      operation_order=model.get('operation_order'))
        if reason is None:
            delivered = selected
        elif joint_order:
            if model['tensor_storage']['serial_peak_bytes'] <= tensor_budget_bytes:
                delivered = 'SERIAL'
                _launcher(stage, executable, delivered, schedule, inputs, guarded, ())
            else:
                delivered = 'NO_NATIVE_DELIVERY'
        report = dict(schema=SCHEMA, schema_version=abi, bundle=origin, executable=guarded[0], inputs=inputs,
            cpu_ids=model['cpu_ids'], model=model, observations=calibration, paired_validation=measurements,
            validation=validations, warmup_pairs=warmup_pairs, measured_pairs=measured_pairs,
            selected=delivered, reason=reason,
            source_replacement_authorized=False)
        if joint_order:
            report['delivery_scope'] = _NATIVE_ONLY
            if order_trial is not None:
                report['joint_order_comparison'] = order_trial
        if entry_comparison:
            report['validation_policy'] = _ENTRY_TRIAL_POLICY
        if replica is not None:
            report['input_replica'] = replica
        _write(stage / 'selection.json', report)
        stage.rename(out)
    return report


def read_ready_selection(path, selection):
    if selection['schema'] != SCHEMA or selection['selected'] not in ('READY_DAG', 'READY_DAG_REPLICA', 'READY_DAG_RETAINED'):
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
    delivery_scope = selection.get('delivery_scope')
    if (delivery_scope not in (None, _NATIVE_ONLY) or delivery_scope == _NATIVE_ONLY
            and (artifact.get('ready_order_abi') != 1 or selection.get('validation_policy') is not None
                 or selection.get('input_replica') is not None)):
        raise ValueError('ready selection differs from the standalone native delivery scope')
    model = _ready_model(recipe, schedule, raw, budget, warmup, ready_dag_abi=abi,
                         joint_order=delivery_scope == _NATIVE_ONLY, gain_policy=selection['model'].get('gain_policy'))
    if delivery_scope == _NATIVE_ONLY:
        original_model, baseline = _source_order_baseline(recipe, schedule, raw, model, budget, warmup, abi)
        trial = selection.get('joint_order_comparison')
        if original_model is None:
            if trial is not None:
                raise ValueError('unexpected source-order comparison for this ready plan')
        else:
            if trial is None:
                raise ValueError('standalone joint plan requires its source-order comparison')
            paired = read_artifact(trial['observations'])
            expected = _case(paired, model, baseline, warmup, count, raw['calibration']['result_storage_bytes'], recipe=recipe)
            if trial['validation'] != expected or trial['prediction'] != model['joint_cpu_plan']:
                raise ValueError('joint ready comparison differs from the replayed evidence')
            if not expected['profitable']:
                model = original_model
    baselines = _baselines(recipe, schedule, raw['calibration'], model, budget, warmup)
    results = [_case(read_artifact(ref), model, baseline, warmup, count, raw['calibration']['result_storage_bytes'], recipe=recipe)
               for ref, baseline in zip(selection['paired_validation'], baselines, strict=True)]
    policy = selection.get('validation_policy')
    if policy not in (None, _ENTRY_TRIAL_POLICY):
        raise ValueError('unknown ready DAG validation policy')
    required = results[:1] if policy == _ENTRY_TRIAL_POLICY else results
    if (model != selection['model'] or model['status'] != 'PLAN_READY_DAG' or model['cpu_ids'] != selection['cpu_ids']
            or not required or results != selection['validation'] or not all(row['profitable'] for row in required)):
        raise ValueError('retained ready DAG choice differs from its calibration or matched validation')
    replica = selection.get('input_replica')
    chosen = 'READY_DAG'
    if replica is not None:
        abi_key = 'ready_retained_abi' if 'retained' in replica else 'ready_replica_abi'
        if artifact.get(abi_key) != 1:
            raise ValueError('ready selection has input replica evidence for an incompatible native bundle')
        chosen = _replay_replica(recipe, schedule, model, replica, warmup, count)
    if chosen != selection['selected']:
        raise ValueError('ready selection does not agree with its replica comparison')
    problem, execution = _resource_plan(hardware, schedule, origin, calibration, recipe, raw['calibration'], model,
                                       replica=replica if chosen != 'READY_DAG' else None)
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
    replicas = parser.add_mutually_exclusive_group()
    replicas.add_argument('--try-input-replica', action='store_true', help='price one NUMA-local input copy per call')
    replicas.add_argument('--try-retained-input-replica', action='store_true',
                          help='price one byte-checked NUMA-local input copy retained across calls')
    replicas.add_argument('--joint-order', action='store_true',
                          help='optimize CPU placement and topological order for standalone C++20 delivery')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    selection = plan_aten_ready(args.bundle, bundle_sha256=args.bundle_sha256, input_paths=args.input,
        tensor_budget_bytes=args.tensor_budget_bytes, warmup_pairs=args.warmup_pairs,
        measured_pairs=args.measured_pairs, out=args.out, try_input_replica=args.try_input_replica,
        try_retained_input_replica=args.try_retained_input_replica, joint_order=args.joint_order)
    print(json.dumps(dict(selected=selection['selected'], reason=selection['reason'],
        selection=str(Path(args.out) / 'selection.json'))))
    return 0
