"""Bind native render diagnostics to one existing synchronous source effect."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.source_controller_plot import check as check_plot_source
from scar.analysis.target_graph import TargetHardwareGraph
from scar.compiler.observe_native_plot_entry import SCHEMA as DIAGNOSTIC, check as check_diagnostics
from scar.compiler.plan_native_controller import _at, _port_key, read_native_controller_plan
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem, PlanningResource,
    ResourceObligation, problem_document, validate_problem,
)
from scar.ir.program.target_hardware import CpuCore, HardwareThread, StorageEndpoint, TargetResourceRef
from scar.ir.record_codec import encode
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.native-plot-boundary-plan'


def _need(ok, explanation):
    if not ok:
        raise ValueError('native plot boundary: ' + explanation)


def _one(rows, explanation):
    rows = list(rows)
    _need(len(rows) == 1, explanation)
    return rows[0]


def _binding(application_refs, diagnostic):
    controller_ref = plot_ref = None
    for app_ref in application_refs:
        app = read_artifact(app_ref)
        controller = app['controller']
        plot = controller['plot']
        _need(plot['synchronous'] is True and app['cpu_ids'] == diagnostic['cpu_ids'],
              'same synchronous source callback and CPU mask required')
        if controller_ref is not None:
            _need((controller['planning'], plot['planning']) == (controller_ref, plot_ref),
                  'plot implementations do not share source contracts')
        controller_ref, plot_ref = controller['planning'], plot['planning']
    source = check_plot_source(plot_ref['uri'])
    controller, problem = read_native_controller_plan(controller_ref['uri'])
    _need(controller == read_artifact(controller_ref) and source == read_artifact(plot_ref),
          'source bindings changed during replay')
    return controller, problem, source, plot_ref


def _covered_effect(problem, controller, source):
    calls = controller['replay']['original_opaque_calls']
    callback = _one((row for row in calls if row['site'] == source['effect']['site']
                     and row['id'] == source['effect']['id'] and row['inside_loop']),
                    'need one original callback within a shared loop')
    publication = _one((row for row in controller['replay']['observed_publications']
                        if row['original_controller_call'] == callback['source']
                        and row['publication_before_first_whole_update']),
                       'need one known original publication before the following update')
    order_ref = publication['write_order']
    order = read_artifact(order_ref['artifact'])['observed_effects']['order']['related_writes']
    fact = _one((row for index, row in enumerate(order) if order_ref['locator'] ==
                 f'#/observed_effects/order/related_writes/{index}'), 'invalid publication order locator')
    _need(_port_key(fact['original_source_port']) == _port_key(source['output']['save']['source_result']),
          'observed file write and selected save call differ')
    effect = _one((op for op in problem.operations if order_ref in op.source_refs),
                  'need one existing planning operation for the source publication')
    successors = [op.id for op in problem.operations if effect.id in op.predecessors]
    _need(bool(successors), 'original source publication completion order was lost')
    return callback['source'], effect, successors


def _observed_cpu(problem, cpu_ids, hardware):
    _need(hardware._snapshot == problem.target, 'hardware artifact differs')
    cpu, = cpu_ids
    resources = hardware.resources()
    thread = _one((row for row in resources if type(row) is HardwareThread
                   and row.online and row.cpu_id == cpu), 'observed CPU has no target thread')
    core = _one((row for row in resources if type(row) is CpuCore and cpu in row.cpu_ids),
                'observed CPU has no target core')
    return thread, core


def _calibrated_publication(calibration, observed, problem, hardware, core):
    from scar.compiler.calibrate_native_png import SCHEMA as CALIBRATION, check

    reference = artifact_reference(Path(calibration).resolve(strict=True), CALIBRATION)
    service = check(reference['uri'])
    _need(service == read_artifact(reference), 'clean calibration changed during replay')
    app = read_artifact(observed['arms']['v1']['application'])
    _need(service['provider'] == app['controller']['plot']['provider']
          and service['cpu_id'] == observed['cpu_ids'][0],
          'measured PNG operator and CPU must match the pinned native baseline')
    storage = _measured_resources(service['environment'], problem, hardware, core)
    witness = read_artifact(service['witnesses'])
    expected = {(out['sha256'], out['size_bytes']) for out in observed['runs'][0]['v1']['outputs'].values()}
    quotes = {}
    for row in witness['samples']:
        case = row['id']
        png = row['expected_png']
        if (png['sha256'], png['size_bytes']) not in expected:
            continue
        cost_fact = _at(reference, f'#/per_frame/{case}')
        point = service['per_frame'][case]['median_ns']
        context = service['scope'] + ':' + reference['sha256'] + ':' + case
        quotes[case] = encode(CostEstimate(point, None, None, 'EMPIRICAL_MODEL', context, (cost_fact,)))
    _need(len(quotes) >= 2, 'calibration and controlled entry must share distinct output witnesses')
    width, height = service['canvas']
    return dict(artifact=reference, measured_scope=service['scope'],
                measured_storage_endpoint=storage, measured_context=_at(reference, '#/environment'),
                borrowed_rgba_payload_bytes=4 * width * height, sample_bound_costs=quotes,
                transferable_to_parent=False, private_workspace_bound_bytes=None)


def _measured_resources(environment, problem, hardware, core):
    cpu = environment['cpu']
    _need(cpu['package_id'] == core.package_id and cpu['die_id'] == core.die_id and cpu['core_id'] == core.core_id
          and tuple(cpu['thread_siblings']) == core.cpu_ids,
          'measured CPU topology differs from the target inventory')
    mount = environment['filesystem']['mount']
    endpoints = [node for node in hardware.resources() if type(node) is StorageEndpoint
                 and (node.mount_point, node.source, node.filesystem) == (mount['path'], mount['source'], mount['fstype'])]
    _need(len(endpoints) <= 1, 'calibration mount names multiple target endpoints')
    return encode(TargetResourceRef(problem.target, endpoints[0].id)) if endpoints else None


def _isolated_render_plan(count, comparison, receipt, callback, target, thread, storage):
    report_root = f'#/cases/{count}'
    quote = _at(receipt, report_root + '/selection/metrics')
    cap = _at(receipt, '#/address_space_limit_bytes')
    context = _at(receipt, '#/environment')
    refs = (callback, quote, cap, context)
    lane = thread.id.wire
    modes = []
    for version, metrics in comparison['metrics'].items():
        cost = CostEstimate(metrics['median_ns_twice'] // 2, None, None, 'EMPIRICAL_MODEL',
                            f'ISOLATED_RENDER_EPISODE:{receipt["sha256"]}:{count}:{version}', (quote,))
        modes.append(ImplementationOption(version, 'PRESERVE_RELEASED_SYNCHRONOUS_RENDER_ABI', refs,
                     {lane: 1}, {}, {}, {}, cost, dict(provider_version=version, prefix_frames=int(count),
                     measured_storage_endpoint=storage, observed_address_space_cap=cap)))
    episode = PlanningOperation('render-episode', refs, (), tuple(modes), 'v1')
    problem = PlanningProblem(target, refs, (PlanningResource(lane, 'EXECUTION', 1, (context,)),), (episode,), (),
        assumptions=('KNOWN_CPU_AFFINITY_NOT_EXCLUSIVE_PHYSICAL_CORE', 'WHOLE_PROCESS_VIRTUAL_CAP_NOT_PRIVATE_WORKSPACE',
                     'ONE_SCOPE_WITH_ALL_SYNCHRONOUS_PNG_EFFECTS_IN_ORIGINAL_ORDER'))
    plan = construct_implementations(problem, baseline_only=comparison['verdict'] == 'KEEP_INCUMBENT')
    plan.pop('elapsed_seconds')
    return dict(comparison=_at(receipt, report_root + '/selection'), problem=problem_document(problem), plan=plan)


def _entry_reviews(sample, observed):
    from scar.compiler.observe_native_plot_entry import BENCHMARK, ENTRY, _benchmark_ref

    sequence = read_artifact(sample['sequence'])
    references = (dict(sequence['clean_controller_witness'], schema=BENCHMARK), observed['benchmark'])
    result = []
    context = None
    for reference in dict((row['sha256'], row) for row in references).values():
        benchmark = read_artifact(reference)
        case = _one((row for row in benchmark['evidence'] if row['id'] == observed['case']),
                    'need a matching whole entry cohort')
        _need(benchmark['qualified'] and not benchmark['instrumented'] and not benchmark['baseline_only']
              and benchmark['original_and_candidate_share_environment_and_affinity']
              and benchmark['cpu_ids'] == observed['cpu_ids'] and case['arguments'] == sequence['arguments']
              and case['outputs_equal_for_required_comparisons'], 'whole entry output or benchmark context differs')
        executions = tuple(benchmark['pinned_inputs'][f'{case["id"]}:{side}_executable']
                           for side in ('original', 'candidate'))
        settings = (benchmark['working_directory'], benchmark['environment'], executions)
        _need(context is None or context == settings, 'whole entry cohorts use different binaries or launch context')
        context = settings
        warmups, repetitions = benchmark['warmups'], benchmark['paired_repetitions']
        rows = []
        expected_outputs = {name: (row['sha256'], row['size_bytes']) for name, row in
                            case['samples']['original'][0]['outputs'].items()}
        for arm, side in (('v1', 'original'), ('v2', 'candidate')):
            pinned = _benchmark_ref(benchmark['pinned_inputs'][f'{arm}_package_manifest'], ENTRY)
            _need(pinned == sample['source_applications'][arm], 'whole entry cohort used another source application')
            samples = case['warmup'][side] + case['samples'][side]
            _need(len(case['warmup'][side]) == warmups and len(samples) == warmups + repetitions,
                  'whole entry cohort has incomplete paired observations')
            for pair, run in enumerate(samples):
                _need(not run['timed_out'] and run['status'] == 0 and {
                    name: (row['sha256'], row['size_bytes']) for name, row in run['outputs'].items()
                } == expected_outputs, 'whole entry run changed release output identity')
                rows.append(dict(pair=pair, mode=arm, wall_ns=run['elapsed_ns'], returncode=run['status'],
                                 stdout_sha256=run['stdout']['sha256'], stderr_sha256=run['stderr']['sha256']))
        result.append(dict(benchmark=reference, selection=compare_entry_pairs(rows, incumbent='v1', challenger='v2',
                           warmup_pairs=warmups, measurement_pairs=repetitions)))
    return result


def _calibrated_render(path, observed, problem, hardware, thread, core, callback):
    from scar.compiler.calibrate_native_render import SCHEMA as RENDER, check

    receipt = artifact_reference(Path(path).resolve(strict=True), RENDER)
    sample = check(receipt['uri'])
    expected = {arm: row['application'] for arm, row in observed['arms'].items()}
    _need(sample == read_artifact(receipt) and sample['source_applications'] == expected
          and sample['cpu'] == observed['cpu_ids'][0], 'complete native render calibration used another entry or CPU')
    storage = _measured_resources(sample['environment'], problem, hardware, core)
    _need(storage is not None, 'render episode needs an identified target publication endpoint')
    cases = {count: _isolated_render_plan(count, row['selection'], receipt, callback, problem.target, thread, storage)
             for count, row in sample['cases'].items()}
    reviews = _entry_reviews(sample, observed)
    proposed = all(row['selection']['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW' for row in reviews)
    return dict(artifact=receipt, measured_scope=sample['scope'], measured_storage_endpoint=storage,
                address_space_limit_bytes=sample['address_space_limit_bytes'], cases=cases,
                whole_entry_reviews=reviews, whole_entry_verdict='REVIEW_REQUIRED' if proposed else 'KEEP_INCUMBENT',
                transferable_to_parent=False, deployment_authorized=False)


@replay_transaction()
def project(diagnostic, calibration=None, render_calibration=None):
    reference = artifact_reference(Path(diagnostic).resolve(strict=True), DIAGNOSTIC)
    observed = check_diagnostics(reference['uri'])
    _need(observed == read_artifact(reference) and observed['clean_performance_claim'] is False,
          'unchanged instrumented observation required')
    controller, problem, plot, plot_ref = _binding(
        [row['application'] for row in observed['arms'].values()], observed)
    callback, effect, successors = _covered_effect(problem, controller, plot)
    hardware = TargetHardwareGraph(Path(problem.target.uri).parent)
    thread, core = _observed_cpu(problem, observed['cpu_ids'], hardware)
    cpu = [encode(TargetResourceRef(problem.target, row.id)) for row in (thread, core)]
    detail = _at(reference, '#/summary')
    refs = (callback, _at(plot_ref, '#/output/save'), _at(plot_ref, '#/counter'), detail)
    baseline, = effect.options
    measured = {}
    if calibration is not None:
        measured = {'calibrated_child': _calibrated_publication(calibration, observed, problem, hardware, core)}
        refs = (*refs, _at(measured['calibrated_child']['artifact']))
    missing_time = 'CLEAN_RENDER_AND_PUBLICATION_SERVICE_TIME'
    if render_calibration is not None:
        measured['calibrated_render'] = _calibrated_render(render_calibration, observed, problem, hardware, thread, core, callback)
        refs = (*refs, _at(measured['calibrated_render']['artifact']))
        missing_time = 'MATCHED_FULL_CALLER_SERVICE_WITH_QUALIFIED_RENDER_MODE'
    cost = replace(baseline.cost, source_refs=baseline.cost.source_refs + (detail,),
                   missing_terms=(*baseline.cost.missing_terms, missing_time))
    option = replace(baseline, cost=cost, proof_refs=baseline.proof_refs + refs,
                     parameters={**baseline.parameters, **measured},
                     unresolved_resources=baseline.unresolved_resources + (
                         ResourceObligation('private_live_set', 'per-frame peak workspace and placement unknown', refs),
                         ResourceObligation('storage_endpoint', 'no same-run filesystem resource binding', refs),
                     ))
    operation = replace(effect, source_refs=effect.source_refs + refs, options=(option,))
    problem = validate_problem(replace(problem, source_refs=problem.source_refs + refs,
        operations=tuple(operation if row.id == effect.id else row for row in problem.operations)))
    plan = construct_implementations(problem, baseline_only=True)
    plan.pop('elapsed_seconds')
    report = dict(schema=SCHEMA, schema_version=1, diagnostic=reference, plot=plot_ref,
                  controller=controller['source'], source_callback=callback, operation=effect.id,
                  required_successors=successors, observed_cpu_target_inventory_refs=cpu,
                  same_run_hardware_topology_proven=False, complete_effect_cost='UNKNOWN',
                  effect_motion='BLOCKED_BY_ORIGINAL_COMPLETION_ORDER',
                  child_phases_reorderable=False, deployment_authorized=False)
    return dict(report, **measured, problem=problem_document(problem), plan=plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--diagnostic', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--render-calibration', type=Path)
    args = parser.parse_args(argv)
    if args.check:
        saved = json.loads(args.check.read_bytes())
        calibration = saved.get('calibrated_child', {}).get('artifact', {}).get('uri')
        render = saved.get('calibrated_render', {}).get('artifact', {}).get('uri')
        report = project(saved['diagnostic']['uri'], calibration, render)
        _need(saved == report, 'saved publication planning result differs at cold replay')
    elif args.out:
        report = project(args.diagnostic, args.calibration, args.render_calibration)
        with args.out.open('x') as output:
            output.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        parser.error('--diagnostic requires --out')
    print(json.dumps({key: report[key] for key in ('complete_effect_cost', 'effect_motion')}, sort_keys=True))


if __name__ == '__main__':
    main()
