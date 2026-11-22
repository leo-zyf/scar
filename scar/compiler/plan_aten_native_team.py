"""Verify the model-selected team plan against its matched native execution."""

from pathlib import Path
from tempfile import TemporaryDirectory

from scar.compiler.plan_aten_boundary import _write
from scar.compiler.plan_aten_native_graph import (
    _ENV, _compare_measured_pairs, _invoke, _launcher, _receipt, _verify_resource_segments,
)
from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import problem_document
from scar.optimization.aten_boundary_trials import ATEN_GAIN_POLICY, _passes_gain
from scar.optimization.aten_native_team import model_cpu_team


def team_window_spec(windows):
    return ','.join(f'{start}:{width}' for start, width in windows)


def team_statistics(raw, schedule, warmup, repetitions, windows, mode, *, gain_policy=ATEN_GAIN_POLICY):
    if (raw['cpu_ids'] != schedule['cpus'] or raw['windows'] != windows or raw['warmup_pairs'] != warmup
            or raw['measured_pairs'] != repetitions or raw['all_pairs_validated'] is not True):
        raise ValueError('native team validation does not match its placement and configuration')
    rows = []
    for i, row in enumerate(raw['samples']):
        if row.keys() != {'order', 'serial_ns', 'parallel_ns'} or row['order'] not in ('AB', 'BA'):
            raise ValueError('invalid native team paired observation')
        order = ['SERIAL', 'TWO_CPU']
        rows.append(dict(pair_index=i, order=order if row['order'] == 'AB' else order[::-1],
                         serial_ns=row['serial_ns'], dual_ns=row['parallel_ns'], bitwise_identical=True,
                         input_values_unchanged=True, fresh_owned_outputs=True))
    metrics, wins, directional, ratio, passes = _compare_measured_pairs(rows, repetitions, gain_policy=gain_policy)
    return dict(metrics={'SERIAL': metrics['SERIAL'], mode: metrics['TWO_CPU']}, wins_parallel=wins,
                paired_order_differences={'AB': directional['SERIAL'], 'BA': directional['TWO_CPU']},
                speedup_parallel=ratio, profitable_parallel=passes)


def team_segments(hardware, schedule, origin, evidence, model, mode):
    timeline = model['timelines'][mode]
    sampled = model.get('numa_locality', {}).get('sampled_node_residency_observed', False)
    return _verify_resource_segments(hardware, schedule, origin, evidence, timeline, model['tensor_storage'],
                                     key='cpu_team', numa_sampled=sampled)


def plan_aten_native_team(verified, *, input_paths, out, warmup_pairs, measured_pairs,
                          tensor_budget_bytes, primitive_observation):
    artifact, origin, recipe, hardware, executable, key = verified
    schedule = artifact[key]
    inputs = [_receipt(p) for p in input_paths]
    if len(inputs) != len(recipe['inputs']):
        raise ValueError('provide the native input files in their recipe order')
    binary = _receipt(executable)
    guarded = [binary, *artifact['provider']['libraries'], *inputs]
    if primitive_observation is None:
        raw = _invoke(executable, schedule, '--calibrate-team', [warmup_pairs], inputs)
    else:
        if primitive_observation['inputs'] != inputs:
            raise ValueError('team primitive observations were measured with different input bytes')
        raw = primitive_observation['raw']
    if raw['cpu_ids'] != schedule['cpus'] or raw['warmup_pairs'] != warmup_pairs:
        raise ValueError('team calibration ran outside its proposed CPU context')
    model = model_cpu_team(raw['calibration'], recipe, schedule, warmup_pairs, tensor_budget_bytes=tensor_budget_bytes)
    predicted = model['selected']
    chosen_windows = model['selected_windows']
    serial = model['predictions']['SERIAL']['estimate_ns']
    preferred = model['predictions'][predicted]['estimate_ns']
    gain = (serial - preferred) / serial
    reasons, validation, matched = [], None, None
    if predicted != 'SERIAL':
        if not _passes_gain(serial, preferred, ATEN_GAIN_POLICY):
            reasons.append('PREDICTED_GAIN_BELOW_THRESHOLD')
        else:
            matched = _invoke(executable, schedule, '--benchmark-team',
                              [team_window_spec(chosen_windows), warmup_pairs, measured_pairs], inputs)
            validation = team_statistics(matched, schedule, warmup_pairs, measured_pairs, chosen_windows, predicted)
            if not validation['profitable_parallel']:
                reasons.append('TEAM_FAILED_MATCHED_VALIDATION')
            if matched['result_storage_bytes'] != raw['calibration']['result_storage_bytes']:
                reasons.append('TEAM_STORAGE_CHANGED_DURING_VALIDATION')
    selected = 'SERIAL' if reasons else predicted
    windows = model['mode_windows'][selected]
    if any(_receipt(record['path']) != record for record in guarded):
        raise ValueError('native team members changed during measurement')

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-team-plan-', dir=out.parent) as temporary:
        stage = Path(temporary)

        def publish_evidence(name, schema, document):
            path = stage / name
            _write(path, dict(document, schema=schema, schema_version=1))
            return dict(artifact_reference(path, schema), uri=str(out / name))

        evidence = publish_evidence('observations.json', 'scar.aten-generated-team-observations', raw)
        matched_ref = (None if matched is None else publish_evidence(
            'paired-validation.json', 'scar.aten-generated-team-paired-validation', matched))
        problem, plan, timeline = team_segments(hardware, schedule, origin, evidence, model, selected)
        report = dict(schema='scar.aten-native-graph-schedule', schema_version=1,
            boundary='GENERATED_WARM_ATEN_GRAPH_CALL', bundle=origin, executable=binary, inputs=inputs,
            observations=evidence, paired_validation=matched_ref, model=model, original_target=schedule['target'],
            cpu_ids=schedule['cpus'], model_preferred=predicted, selected=selected, reasons=reasons,
            selected_windows=windows, predicted_gain=gain, validation=validation, warmup_pairs=warmup_pairs,
            measured_pairs=0 if validation is None else measured_pairs, execution_timeline=timeline,
            runtime_environment=_ENV, source_replacement_authorized=False, cold_launcher_performance_measured=False,
            gain_policy=dict(ATEN_GAIN_POLICY),
            planning_domain='ADJACENT_INDEPENDENT_GROUPS_ON_THREE_RECORDED_LOCAL_CORES')
        _write(stage / 'selection.json', report)
        _write(stage / 'problem.json', problem_document(problem))
        _write(stage / 'plan.json', plan)
        _launcher(stage, executable, selected, schedule, inputs, guarded, windows)
        stage.rename(out)
    return report
