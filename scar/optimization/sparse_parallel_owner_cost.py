"""Project ordered sparse owner work and its measurable costs from pinned evidence."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.sparse_parallel_scaling_scope import _release_lineage
from scar.ir.execution_recipe import SCHEMA as RECIPE, artifact_reference, read_artifact
from scar.optimization.sparse_parallel_cost import SCHEMA as CALIBRATION, check as check_calibration
from scar.optimization.sparse_parallel_recipe import verify_sparse_parallel_recipe
from scar.runtime.sparse_parallel_entry import SCHEMA as RELEASE

SCHEMA = 'scar.sparse-parallel-owner-cost'
EXECUTION = 'scar.sparse-parallel-entry-execution'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _boundary(recipe, scope, pipeline):
    source = recipe['semantic_boundary']
    normalization = pipeline['normalization']
    detached = recipe['artifacts']['detached']
    status = source['original_python_status_boundary']
    contract = scope['speculation_contract']
    _require(pipeline['source_scope'] == source['source_scope']
        and pipeline['solve_scope'] == scope['solve_scope']
        and pipeline['regions']['sparse_solve']['call_point'] == source['original_call']
        and normalization['original_normalization_required'] is True
        and normalization['order'][-1] == 'NATIVE_CALL'
        and contract['original_effects_preserved'] == source['original_effects']
        and contract['raw_inputs_always_revalidated'] is True
        and contract['acceptance_key'] == source['Q']['acceptance_key']
        and status['preserve_in_python'] is True,
        'owner work does not match the ordered recipe source and visible effects')

    tasks = {row['id']: row for row in recipe['task_graph']['tasks']}
    stages = [
        ('capture-live-ins', 'ONCE', dict(detached=detached, locator='#/roles')),
        ('prepare-original', 'PER_CALL', dict(entry=source['original_entry'],
            source_spans=source['original_preparation_spans'], row_edit=pipeline['regions']['row_edit'],
            operand_routes='#/value_routes', expands='#/graph/original_normalization')),
        ('bootstrap-team', 'ONCE_AFTER_FIRST_ORIGINAL_REQUEST', dict(original_call=source['original_call'])),
        ('prepare-private', 'PER_CALL', dict(detached=detached, locator='#/scale')),
        ('compare-request', 'PER_CALL', dict(original_call=source['original_call'],
            normalized_operands='#/normalization/terminal_call', acceptance_key=contract['acceptance_key'])),
        ('publish-original', 'PER_CALL', dict(sparse_status=status,
            preserved_effects=source['original_effects'])),
    ]
    overlay = []
    for task, cadence, source_binding in stages:
        row = tasks[task]
        _require(any(lease['cpu_id'] == recipe['resources']['selected_cpu']['owner_cpu_id']
            for lease in row['resource_refs'] if lease.get('kind') == 'OWNER_CPU'),
            'expanded preparation task lost its ordered CPU owner')
        overlay.append(dict(task=task, cadence=cadence, source=source_binding,
            reads=row['reads'], writes=row['writes'], depends_on=row['depends_on'],
            resource_refs=row['resource_refs']))
    steps = normalization['order'][:-1]
    return dict(stages=overlay, indexed_control=recipe['control']['cross_instance_dependencies'],
        original_normalization=dict(parent='prepare-original', axis='i',
            operations=[dict(id=step, duration_ns=None) for step in steps],
            normal_path_order=[dict(before=left, after=right) for left, right in zip(steps, steps[1:])],
            retained_f4_targets='#/normalization/retained_original_call_targets',
            native_operand_ports=normalization['terminal_call']['input_ports'],
            native_world_in=normalization['terminal_call']['world_before'],
            native_world_out=normalization['terminal_call']['world_after']),
        outside_recipe=dict(original_entry=source['original_entry'], before='capture-live-ins',
            setup_source=pipeline['entry']['setup_source'],
            exact_setup_prefix_bytes=pipeline['entry']['exact_setup_prefix_bytes']),
        logical_values_distinct=['original-arguments', 'private-arguments'],
        cross_region_storage_alias=normalization['actual_cross_region_storage_alias'],
        raw_keys_interchangeable=pipeline['composition_contract']['raw_keys_interchangeable'],
        original_normalization_required=True, exact_match_at_source_carrier=True)


def _execution(execution, release_ref, release, recipe):
    count = release['count']
    offers = execution['parallel'] or {}
    team = offers.get('workers') or {}
    workers = recipe['resources']['selected_cpu']['worker_cpu_ids']
    _require(execution['schema_version'] == 1 and execution['selection_sha256'] == release_ref['sha256']
        and execution['entry_path'] == release['entry']['uri'] and execution['argv'] == release['context']['argv']
        and execution['applied'] == 'PARALLEL' and execution['exit_code'] == 0
        and execution['statistics']['fallback_calls'] == 0 and execution['statistics']['native_calls'] == count
        and offers['inputs_captured'] is True and offers['matched'] == count
        and offers['serial_calls'] == offers['rejected'] == 0 and offers['peak_window'] == release['workers']['window']
        and team['closed'] is True and team['retained_after_close_bytes'] == team['pending_snapshot_bytes'] == 0
        and team['native_calls'] == team['submissions'] == count and team['fallback_calls'] == 0
        and sorted(row['cpu'] for row in team['workers']) == workers
        and [(row['issue'], row['native']) for row in team['publications']]
            == [(index, True) for index in range(count)],
        'live clocks are not from the exact released ordered execution')
    return team


def _costs(calibration, team):
    live = []
    for key, boundary in (
            ('owner_validation_ns', 'First serial provider validation and close, after raw request capture'),
            ('startup_ns', 'Create courier and workers, exchange structure, await symbolic plans')):
        value = team[key]
        _require(type(value) is int and value > 0, 'one positive clock per released startup interval required')
        live.append(dict(task='bootstrap-team', interval=boundary, duration_ns=value, evidence=key))
    ordered = calibration['ordered']
    phase = calibration['phase_cost_ledger']['measured']
    diagnostic = {
        'prepare-private': dict(evidence='ordered/prepare_ns', sample=ordered['prepare_ns']),
        'prepare-original': dict(evidence='ordered/replay_before_wait_ns', sample=ordered['replay_before_wait_ns'],
            coverage='Harness row edit and partial normalization on non-first calls'),
        'publish-original': dict(evidence='phase_cost_ledger/measured/publish_digest_proxy_ns',
            sample=phase['publish_digest_proxy_ns'], coverage='Harness output digest'),
    }
    return dict(actual_released_bootstrap_subphases=live,
        actual_released_bootstrap_observed_disjoint_elapsed_ns=sum(row['duration_ns'] for row in live),
        sample_diagnostics_by_task=diagnostic,
        unresolved_actual_duration_ns={
            'entry-assembly-and-exit': None, 'capture-live-ins': None, 'prepare-original': None,
            'bootstrap-team': None, 'prepare-private': None, 'compare-request': None, 'publish-original': None},
        section_projection_is_not_full_entry=True, full_entry_cost_ns=None,
        charge_policy='LIVE_SUBPHASES_AND_CALIBRATION_DIAGNOSTICS_ARE_SEPARATE')


@replay_transaction()
def project(recipe_path, calibration_path, release_path, execution_path):
    """Bind one released execution and same-recipe calibration to original F4 work."""
    refs = {key: artifact_reference(path, schema) for key, path, schema in (
        ('recipe', recipe_path, RECIPE), ('calibration', calibration_path, CALIBRATION),
        ('release', release_path, RELEASE), ('execution', execution_path, EXECUTION))}
    release = read_artifact(refs['release'])
    _require(release['execution_recipe'] == refs['recipe'] and release['cost'] == refs['calibration'],
        'owner evidence must bind the recipe and calibration of the exact release')
    recipe = verify_sparse_parallel_recipe(read_artifact(refs['recipe']))
    calibration = check_calibration(refs['calibration']['uri'])
    scope_ref = recipe['artifacts']['parallel_scope']
    scope = read_artifact(scope_ref)
    _release_lineage(scope_ref, scope, release)
    _require(calibration['execution_recipe'] == refs['recipe']
        and calibration['calibration_is_section_only'] is True and calibration['full_entry_cost_ns'] is None,
        'owner diagnostics require the original section-scoped calibration')
    team = _execution(read_artifact(refs['execution']), refs['release'], release, recipe)
    memory = recipe['resources']['memory']
    graph = _boundary(recipe, scope, read_artifact(scope['pipeline_scope']))
    return dict(schema=SCHEMA, schema_version=1, inputs=refs, recipe_id=recipe['recipe_id'],
        source_pipeline=scope['pipeline_scope'], owner=recipe['resources']['selected_cpu'],
        iteration=recipe['control']['iteration'], graph=graph,
        existing_storage=dict(detached_snapshot_bound_bytes=memory['owner_detached_snapshot_bound_bytes'],
            pending_worker_input_peak_bytes=team['peak_snapshot_bytes'], physical_reservation=False),
        costs=_costs(calibration, team), legality=dict(
            delete_original_normalization=False, replace_original_publication=False,
            independent_private_compute_only=True, standalone_cxx_scope='PREPARED_INPUT_TO_ORDERED_NUMERIC_OUTPUT',
            automatic_rewrite=False))


def check(path):
    """Recreate the complete Stage II overlay from its original evidence."""
    record = read_artifact(artifact_reference(path, SCHEMA))
    refs = record['inputs']
    rebuilt = project(*(refs[key]['uri'] for key in ('recipe', 'calibration', 'release', 'execution')))
    _require(record == rebuilt, 'owner cost and legality projection differs from cold replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    for name in ('recipe', 'calibration', 'release', 'execution', 'out'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(check(args.check)['recipe_id'])
        return
    if not all((args.recipe, args.calibration, args.release, args.execution, args.out)):
        parser.error('provide recipe, calibration, release, execution and output')
    result = project(args.recipe, args.calibration, args.release, args.execution)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
