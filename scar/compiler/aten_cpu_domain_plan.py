"""Price topology representatives from native ATen primitives before graph validation."""

from scar.codegen.aten_native_graph import _read_declared
from scar.compiler.plan_aten_native_graph import _bundle, _invoke, _receipt, _window_model
from scar.optimization.aten_boundary_trials import ATEN_GAIN_POLICY, _passes_gain
from scar.optimization.aten_native_schedule import propose_cpu_windows
from scar.optimization.aten_native_team import TEAM_PLACEMENT_POLICY, model_cpu_team, propose_cpu_team


def _placement_cost(option):
    _, _, _, mode, row = option
    model = row['ready_model'] if mode == 'READY_DAG' else row['model']
    return model['predictions'][mode]


def choose_aten_cpu_domain(candidates, *, team, gain_policy=ATEN_GAIN_POLICY):
    options = [(cost['estimate_ns'], mode != 'SERIAL', tuple(row['cpus']), mode, row)
               for row in candidates for mode, cost in row['model']['predictions'].items()
               if not team or mode in ('SERIAL', row['model']['selected'])]
    options.extend((row['ready_model']['predictions']['READY_DAG']['estimate_ns'], True,
                    tuple(row['cpus']), 'READY_DAG', row)
                   for row in candidates if row.get('ready_model', {}).get('status') == 'PLAN_READY_DAG')
    by_point = lambda option: (*option[:3], option[3] == 'READY_DAG')
    serial = min((option for option in options if option[3] == 'SERIAL'), key=by_point)
    chosen = min(options, key=by_point)
    ranking = None
    if team and all('cross_numa' in row for row in candidates):
        ranking = dict(policy='SAME_NODE_WHEN_CROSS_NODE_CALIBRATION_SPANS_OVERLAP',
                       point_preferred_cpu_ids=list(chosen[2]), decision='MINIMUM_MODELED_TIME')
        if chosen[3] != 'SERIAL' and chosen[4]['cross_numa']:
            local = min((option for option in options if option[3] != 'SERIAL' and not option[4]['cross_numa']),
                        key=by_point, default=None)
            if local is not None:
                cross_cost = _placement_cost(chosen)
                local_cost = _placement_cost(local)
                ranking.update(same_node_cpu_ids=list(local[2]), same_node_prediction=local_cost,
                               cross_node_prediction=cross_cost)
                if cross_cost['sample_max_ns'] >= local_cost['sample_min_ns']:
                    ranking['decision'] = 'CROSS_NODE_RANK_UNRESOLVED_WITHIN_CALIBRATION_SPANS'
                    chosen = local
    if not _passes_gain(serial[0], chosen[0], gain_policy):
        chosen = serial
    return chosen, serial, ranking


def plan_aten_cpu_domain(probe, *, domain, input_paths, warmup_pairs, tensor_budget_bytes, price_ready=False):
    artifact, origin, recipe, hardware, executable, key = _bundle(probe['uri'], probe['sha256'])
    if origin != probe or key not in ('cpu_windows', 'cpu_team') or not 3 <= warmup_pairs <= 100:
        raise ValueError('the graph must support bounded CPU window calibration')
    declarations = artifact['concurrency_contract']
    _, contract, _ = _read_declared(declarations['path'], declarations['sha256'])
    inputs = [_receipt(path) for path in input_paths]
    candidates = []
    team = key == 'cpu_team'
    for representative in domain['representatives']:
        if team:
            schedule = propose_cpu_team(recipe, contract=contract, target=hardware, cpu_ids=representative['cpus'],
                 allow_cross_numa=artifact[key].get('placement_policy') == TEAM_PLACEMENT_POLICY,
                 ready_dag=artifact[key]['schema_version'] == 2)
        else:
            schedule = propose_cpu_windows(recipe, contract=contract, target=hardware, cpu_ids=representative['cpus'])
        if schedule['target'] != domain['target'] or 'target_cpu_domain' not in schedule:
            raise ValueError('the graph probe must replay the nominated broader CPU target')
        raw = _invoke(executable, schedule, '--calibrate-team' if team else '--calibrate-windows', [warmup_pairs], inputs)
        cpus = raw['cpu_ids'] if team else [raw['caller_cpu'], raw['worker_cpu']]
        if cpus != schedule['cpus'] or raw['warmup_pairs'] != warmup_pairs:
            raise ValueError('primitive calibration ran outside its nominated CPU placement')
        if team:
            model = model_cpu_team(raw['calibration'], recipe, schedule, warmup_pairs,
                                   tensor_budget_bytes=tensor_budget_bytes)
        else:
            model = _window_model(raw['calibration'], recipe, schedule, warmup_pairs, observe_storage=True,
                                  tensor_budget_bytes=tensor_budget_bytes)
        candidate = dict(cpus=schedule['cpus'], primitive_observations=raw, model=model)
        if 'cross_numa' in schedule:
            candidate['cross_numa'] = schedule['cross_numa']
        if price_ready and team:
            from scar.compiler.plan_aten_ready import calibrate_ready_placement

            raw, ready_model = calibrate_ready_placement(executable, recipe, schedule, inputs, raw,
                budget=tensor_budget_bytes, warmup=warmup_pairs, ready_dag_abi=artifact['ready_dag_abi'])
            candidate.update(ready_model=ready_model, primitive_observations=raw)
        candidates.append(candidate)
    if any(_receipt(row['path']) != row for row in inputs):
        raise ValueError('a calibration input changed while pricing the CPU domain')

    chosen, best_serial, ranking = choose_aten_cpu_domain(candidates, team=team)
    estimate, _, cpus, mode, samples = chosen
    residency = any(row['model'].get('numa_locality', {}).get('sampled_node_residency_observed') for row in candidates)
    report = dict(schema='scar.aten-cpu-domain-plan', schema_version=1, topology=domain, probe=origin,
                  inputs=inputs, comparisons=candidates, selected_cpu_ids=list(cpus), selected_mode=mode,
                  selected_estimate_ns=estimate, best_serial_estimate_ns=best_serial[0],
                  best_serial_cpu_ids=list(best_serial[2]), workload_variant_runs=0,
                  memory_page_placement_observed=residency, source_replacement_authorized=False,
                  gain_policy=dict(ATEN_GAIN_POLICY))
    if ranking is not None:
        report['topology_ranking'] = ranking
    return report, samples
