"""Joint conditional costs over already cold-admitted native group records.

This adapter checks identities, scope and original clocks.  It neither repeats
source admission nor imports a whole-method estimate or a qualified profile.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.native-group-costs'
_ROUNDING = 'NEAREST_INTEGER_NANOSECOND_TIES_TO_EVEN'
_TRANSFER = (
    'OBSERVER_TO_ORIGINAL_NATIVE_TIME_TRANSFER_NOT_VALIDATED',
    'TARGET_DATA_SCALAR_TAG_AND_CALLER_PREDECESSOR_CONTEXT_NOT_VALIDATED',
    'CALLER_ALLOCATOR_FIRST_TOUCH_CACHE_AND_NUMA_TRANSFER_NOT_VALIDATED',
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    return semantic_sha256(left) == semantic_sha256(right)


def _material(value, schema):
    if type(value) is dict and value.get('schema') == schema and 'schema_version' in value:
        return deepcopy(value), None
    reference = value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)
    document = read_artifact(reference)
    _need(reference['schema'] == schema, 'native group material schema differs')
    return document, deepcopy(reference)


def derive_native_group_costs(observations, design):
    """Return joint cost_groups; original controller cold admission is required.

    Inputs may be checked documents or their ArtifactRefs/paths.  A document
    alone retains semantic provenance and never invents a file receipt.
    Warmups are not samples. Each observed contiguous group is charged once.
    """
    from scar.compiler.calibrate_native_groups import (DESIGN, OBSERVATIONS,
        native_group_statistics, _timer)

    observed, observed_ref = _material(observations, OBSERVATIONS)
    frozen, supplied_design_ref = _material(design, DESIGN)
    _need((observed.get('schema'), observed.get('schema_version')) == (OBSERVATIONS, 1)
        and type(observed['schema_version']) is int
        and (frozen.get('schema'), frozen.get('schema_version')) == (DESIGN, 1)
        and type(frozen['schema_version']) is int, 'known native group observation/design versions required')
    design_ref = observed['design']
    _need((supplied_design_ref is None or supplied_design_ref == design_ref)
        and _same(read_artifact(design_ref), frozen)
        and observed['frozen_design_sha256'] == design_ref['sha256'], 'original frozen group design differs')
    _need(frozen['timer'] == _timer() and observed['clock_overhead_subtracted'] is False,
        'original stage-body timer with retained observer clocks required')
    _need(type(observed['warmup']) is int and observed['warmup'] == frozen['request']['warmup']
        and observed['warmup'] >= 1 and type(observed['repetitions']) is int
        and observed['repetitions'] == frozen['request']['repetitions']
        and observed['repetitions'] >= 3, 'frozen warmups and distinct measurement sample count required')
    observer = frozen['observer']; context = frozen['stage_context']; view = observer['parent_view']
    _need(_same(observed['group_layout'], observer['groups'])
        and observer['cut_count'] == 2*len(observer['groups'])
        and observed['full_native_prefix_suffix_and_commits_executed'] is True,
        'observed groups differ from the complete original native frame')
    _need(view['native_manifest'] == frozen['request']['native_manifest']
        == observer['specification']['native_manifest'] == context['source_refs']['native_manifest']
        and view['plan'] == context['source_refs']['plan']
        and _same(view['identity'], context['native_identity']), 'native/context/plan identities differ')
    # Read only the two authoritative metadata records, not source rediscovery,
    # compiler queries, ELF loading, or the observer's generation machinery.
    native = read_artifact(view['native_manifest']); plan = read_artifact(view['plan'])
    _need(Path(view['plan']['uri']).resolve() ==
        (Path(view['native_manifest']['uri']).parent/native['files']['plan']).resolve()
        and view['plan']['sha256'] == native['sha256']['plan']
        and semantic_sha256(plan) == native['plan_sha256'] == view['identity']['plan_sha256']
        and plan['region_sha256'] == native['region_sha256'] == view['identity']['region_sha256']
        and native['build_id'] == view['identity']['build_id']
        and native['compiler'] == view['compiler'], 'original native plan/compiler binding differs')
    stages = context['stages']; tasks = context['versioned_schedule']['tasks']
    _need([row['index'] for row in stages] == list(range(len(plan['stages'])))
        and len({row['id'] for row in tasks}) == len(tasks), 'unique complete native stage/task map required')
    task_ids = {row['id'] for row in tasks}
    for index, stage in enumerate(stages):
        _need(stage['id'] == plan['stages'][index]['id'] and stage['kind'] == plan['stages'][index]['kind']
            and stage['id'] in task_ids and stage['source_stage_ref'] ==
                dict(artifact=view['plan'], locator='#/stages/'+str(index)),
            'source stage does not map to its exact original task/plan record')
    statistics = native_group_statistics(observed)
    _need(_same(statistics, observed['statistics']), 'statistics differ from original joint clock samples')
    _need(len(context['groups']) == len(observer['groups']), 'native stage context lost an observed group')
    residuals = []
    for rows in (observed['residuals'], observer['retained_residuals'], _TRANSFER):
        _need(type(rows) in (list, tuple) and all(type(row) is str and row for row in rows),
            'explicit native group transfer residual terms required')
        residuals.extend(rows)
    residuals = tuple(dict.fromkeys(residuals))
    observation_identity = (dict(artifact=observed_ref) if observed_ref is not None else
        dict(document_schema=OBSERVATIONS, document_semantic_sha256=semantic_sha256(observed)))
    groups = []; occupied = set(); previous = -1
    for index, (layout, values, group) in enumerate(zip(observer['groups'], statistics['groups'], context['groups'])):
        indices = layout['stage_indices']
        _need(type(indices) is list and indices and all(type(i) is int for i in indices)
            and indices == list(range(indices[0], indices[-1]+1))
            and 0 <= indices[0] <= indices[-1] < len(stages), 'nonempty contiguous original stage group required')
        _need(indices[0] > previous and not occupied.intersection(indices)
            and layout['index'] == index and type(layout['index']) is int
            and layout['begin_cut'] == 2*index and layout['end_cut'] == 2*index+1,
            'ordered disjoint stage groups and exact original clock cuts required')
        _need(group['index'] == index and group['stage_indices'] == indices
            and group['begin_cut'] == layout['begin_cut'] and group['end_cut'] == layout['end_cut']
            and _same(group['original_frame_entry'], layout)
            and not any(stages[i]['kind'] == 'COMMIT' for i in indices),
            'group source context differs or imports unobserved commit work')
        occupied.update(indices); previous = indices[-1]
        members = [stages[i]['id'] for i in indices]
        rational = values['exact_mean_ns']; mean = Fraction(rational['numerator'], rational['denominator'])
        point = round(mean)
        refs = [dict(**observation_identity, locator='#/statistics/groups/'+str(index)),
            dict(**observation_identity, locator='#/samples'),
            dict(artifact=design_ref, locator='#/timer'),
            dict(artifact=design_ref, locator='#/observer/groups/'+str(index)),
            dict(artifact=design_ref, locator='#/stage_context/groups/'+str(index)),
            *[deepcopy(stages[i]['source_stage_ref']) for i in indices]]
        applicability = dict(design=design_ref, native_manifest=view['native_manifest'], plan=view['plan'],
            stage_context_sha256=semantic_sha256(context), observer_identity=observer['observer_identity'],
            group_index=index, task_ids=members, timer=frozen['timer'], canonical_request=frozen['request'],
            runtime_context=frozen['runtime_context'], scope='JOINT_CONTIGUOUS_NATIVE_STAGE_BODY_GROUP_ONLY')
        cost = CostEstimate(point, None, None, 'DECLARED_MODEL',
            'native-stage-group-cost:'+semantic_sha256(applicability), tuple(refs), residuals)
        groups.append(dict(id='native-group-cost:'+semantic_sha256(dict(applicability=applicability,
                observations=observation_identity, exact_mean_ns=rational, integer_rounding=_ROUNDING)),
            task_ids=members, cost=encode(cost), source_refs=refs,
            statistics=dict(exact_mean_ns=deepcopy(rational), integer_rounding=_ROUNDING, estimate_ns=point,
                warmup_calls_excluded=observed['warmup'], measurement_samples=len(values['samples_ns'])),
            jointly_charged_once=True, per_task_duration_inferred=False, transfer_qualified=False))
    return dict(schema=SCHEMA, schema_version=1, observations=observation_identity, design=design_ref,
        native_manifest=view['native_manifest'], plan=view['plan'], plan_semantic_sha256=semantic_sha256(plan),
        stage_context_sha256=semantic_sha256(context), timer=deepcopy(frozen['timer']), cost_groups=groups,
        entry_path_protocol=dict(status='UNKNOWN', observed_boundary=frozen['timer']['outer'],
            whole_method_dispatch_guard_trial_and_packing_cost_ns=None,
            compiler_artifact_and_compact_entry_cost_equivalence='UNKNOWN',
            target_data_predecessor_allocator_and_cpu_context_transfer='UNKNOWN'),
        whole_method_estimate_ns=None, ffi_window_imported_as_cost=False,
        outside_group_complement_imported_as_cost=False, component_medians_summed=False,
        original_controller_cold_admission_required=True, timing_qualified=False,
        target_samples_fitted=False, target_executed=False, compiled=False, native_loaded=False,
        deployment_authorized=False)


__all__ = ['derive_native_group_costs']
