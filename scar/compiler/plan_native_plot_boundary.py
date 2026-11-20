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
from scar.ir.implementation_plan import CostEstimate, ResourceObligation, problem_document, validate_problem
from scar.ir.program.target_hardware import CpuCore, HardwareThread, StorageEndpoint, TargetResourceRef
from scar.ir.record_codec import encode
from scar.optimization.constructive_planner import construct_implementations


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
    cpu = service['environment']['cpu']
    _need(service['provider'] == app['controller']['plot']['provider']
          and service['cpu_id'] == cpu['id'] == observed['cpu_ids'][0],
          'measured PNG operator and CPU must match the pinned native baseline')
    _need(cpu['package_id'] == core.package_id and cpu['die_id'] == core.die_id and cpu['core_id'] == core.core_id
          and tuple(cpu['thread_siblings']) == core.cpu_ids,
          'measured CPU topology differs from the target inventory')
    mount = service['environment']['filesystem']['mount']
    endpoints = [node for node in hardware.resources() if type(node) is StorageEndpoint
                 and (node.mount_point, node.source, node.filesystem) == (mount['path'], mount['source'], mount['fstype'])]
    _need(len(endpoints) <= 1, 'calibration mount names multiple target endpoints')
    storage = encode(TargetResourceRef(problem.target, endpoints[0].id)) if endpoints else None
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


@replay_transaction()
def project(diagnostic, calibration=None):
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
    cost = replace(baseline.cost, source_refs=baseline.cost.source_refs + (detail,),
                   missing_terms=(*baseline.cost.missing_terms, 'CLEAN_RENDER_AND_PUBLICATION_SERVICE_TIME'))
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
    args = parser.parse_args(argv)
    if args.check:
        saved = json.loads(args.check.read_bytes())
        calibration = saved.get('calibrated_child', {}).get('artifact', {}).get('uri')
        report = project(saved['diagnostic']['uri'], calibration)
        _need(saved == report, 'saved publication planning result differs at cold replay')
    elif args.out:
        report = project(args.diagnostic, args.calibration)
        with args.out.open('x') as output:
            output.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        parser.error('--diagnostic requires --out')
    print(json.dumps({key: report[key] for key in ('complete_effect_cost', 'effect_motion')}, sort_keys=True))


if __name__ == '__main__':
    main()
