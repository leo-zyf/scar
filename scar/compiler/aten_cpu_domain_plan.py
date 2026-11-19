"""Price topology representatives from native ATen primitives before graph validation."""

from scar.codegen.aten_native_graph import _read_declared
from scar.compiler.plan_aten_native_graph import _GAIN, _bundle, _invoke, _receipt, _window_model
from scar.optimization.aten_native_schedule import propose_cpu_windows
from scar.optimization.aten_native_team import model_cpu_team, propose_cpu_team


def plan_aten_cpu_domain(probe, *, domain, input_paths, warmup_pairs, tensor_budget_bytes):
    artifact, origin, recipe, hardware, executable, key = _bundle(probe['uri'], probe['sha256'])
    if origin != probe or key not in ('cpu_windows', 'cpu_team') or not 3 <= warmup_pairs <= 100:
        raise ValueError('the graph must support bounded CPU window calibration')
    declarations = artifact['concurrency_contract']
    _, contract, _ = _read_declared(declarations['path'], declarations['sha256'])
    inputs = [_receipt(path) for path in input_paths]
    candidates = []
    team = key == 'cpu_team'
    propose = propose_cpu_team if team else propose_cpu_windows
    for representative in domain['representatives']:
        schedule = propose(recipe, contract=contract, target=hardware, cpu_ids=representative['cpus'])
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
        candidates.append(dict(cpus=schedule['cpus'], primitive_observations=raw, model=model))
    if any(_receipt(row['path']) != row for row in inputs):
        raise ValueError('a calibration input changed while pricing the CPU domain')

    alternatives = [(cost['estimate_ns'], mode != 'SERIAL', tuple(row['cpus']), mode, row)
                    for row in candidates for mode, cost in row['model']['predictions'].items()
                    if not team or mode in ('SERIAL', row['model']['selected'])]
    best_serial = min((option for option in alternatives if option[3] == 'SERIAL'), key=lambda option: option[:3])
    chosen = min(alternatives, key=lambda option: option[:3])
    if (best_serial[0] - chosen[0]) / best_serial[0] < _GAIN:
        chosen = best_serial
    estimate, _, cpus, mode, samples = chosen
    residency = any(row['model'].get('numa_locality', {}).get('sampled_node_residency_observed') for row in candidates)
    report = dict(schema='scar.aten-cpu-domain-plan', schema_version=1, topology=domain, probe=origin,
                  inputs=inputs, comparisons=candidates, selected_cpu_ids=list(cpus), selected_mode=mode,
                  selected_estimate_ns=estimate, best_serial_estimate_ns=best_serial[0],
                  best_serial_cpu_ids=list(best_serial[2]), workload_variant_runs=0,
                  memory_page_placement_observed=residency, source_replacement_authorized=False)
    return report, samples
