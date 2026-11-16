"""Resource and conditional-cost lineage for a certified raw-copy fusion.

Fusion preserves one selected calling thread, not the sum of sequential thread
demands. Original unfused timings do not establish the new fused duration.
"""
from dataclasses import replace
from pathlib import Path
import re

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.activity_costs import (ActivityCost, CostBasis, CostCoverage,
    activity_cost_document, validate_activity_cost_table)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import (ResourceProjection, projection_document,
    validate_resource_projection)
from scar.ir.versioned_schedule import (TemporalDurationBasis, TemporalOccupancyKind as O,
    VersionedTaskKind as K, schedule_document, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _distinct(records):
    found = {}
    for row in records:
        found.setdefault(semantic_sha256(row), row)
    return tuple(found.values())


def _admit(original, resources, rewritten, certificate):
    from scar.optimization.fused_copy_graph import verify_fused_copy_graph
    validate_versioned_schedule(original); validate_versioned_schedule(rewritten)
    replay = verify_fused_copy_graph(original, rewritten, certificate)
    _need(semantic_sha256(replay) == semantic_sha256(certificate),
          'copy fusion certificate differs from complete graph rewrite replay')
    parent_sha = semantic_sha256(schedule_document(original))
    rewritten_sha = semantic_sha256(schedule_document(rewritten))
    _need(type(resources) is ResourceProjection
          and resources.source_schedule_sha256 == parent_sha
          and certificate['parent_schedule_sha256'] == parent_sha
          and certificate['rewritten_schedule_sha256'] == rewritten_sha,
          'copy fusion resource projection and parent/rewritten graph identities differ')
    target_ref = encode(resources.target)
    graph = TargetHardwareGraph(Path(resources.target.uri).parent)
    _need(artifact_reference(graph.path/'result.json', 'scar.target-hardware-graph') == target_ref,
          'copy fusion must retain the exact original hardware snapshot')
    inventory = graph.resources()
    validate_resource_projection(resources, original, resources=inventory)
    ids = certificate['original_task_ids']; replacement = certificate['replacement_task_id']
    _need(type(ids) is list and len(ids) >= 2 and len(set(ids)) == len(ids),
          'multiple distinct original copy task occurrences required')
    tasks = {t.id: t for t in original.tasks}
    _need(all(tasks[name].kind is K.SNAPSHOT for name in ids), 'original copies must remain raw snapshot tasks')
    _need(original.storage_slots == rewritten.storage_slots
          and (original.budget_bytes, original.budget_scope, original.budget_refs)
              == (rewritten.budget_bytes, rewritten.budget_scope, rewritten.budget_refs),
          'copy fusion cannot alter storage allocation or budget obligations')
    old_activities = {name: [a for a in original.temporal.activities if a.task == name] for name in ids}
    new_activities = [a for a in rewritten.temporal.activities if a.task == replacement]
    _need(all(len(rows) == 1 for rows in old_activities.values()) and len(new_activities) == 1,
          'copy fusion requires one synchronous activity per original/replacement task')
    old_activity_ids = tuple(old_activities[name][0].id for name in ids)
    new_activity = new_activities[0]
    _need(new_activity.duration_basis is TemporalDurationBasis.UNKNOWN
          and new_activity.duration_lower_ticks is None and new_activity.duration_upper_ticks is None,
          'fused copy duration is UNKNOWN, not inherited from original components')
    _need(all(certificate['activity_map'][key] == new_activity.id for key in old_activity_ids),
          'original copy activities must map to their one replacement')
    old_exec = [o for o in original.temporal.occupancies
                if o.kind is O.EXECUTION_DEMAND and o.activity in old_activity_ids]
    _need(len(old_exec) == len(ids)
          and all(sum(o.activity == key for o in old_exec) == 1 for key in old_activity_ids),
          'one original execution occupancy per selected copy required')
    occupancy_map = certificate['occupancy_map']
    mapped = {occupancy_map[o.id] for o in old_exec}
    _need(len(mapped) == 1, 'selected copy occupancies must map to one fused occupancy')
    fused_id = next(iter(mapped))
    new_exec = [o for o in rewritten.temporal.occupancies if o.kind is O.EXECUTION_DEMAND]
    fused = [o for o in new_exec if o.id == fused_id]
    bindings = {row.occupancy: row for row in resources.executions}
    selected = [bindings[row.id] for row in old_exec]
    first = selected[0]
    _need(first.exact_thread is not None and type(first.units) is int and first.units == 1
          and first.candidate_threads == (first.exact_thread,)
          and all((row.resource, row.exact_thread, row.core_refs, row.candidate_threads, row.units)
              == (first.resource, first.exact_thread, first.core_refs, first.candidate_threads, 1)
              for row in selected), 'copy fusion requires the same original exact hardware thread and unit demand')
    _need(len(fused) == 1 and fused[0].activity == new_activity.id
          and fused[0].resource == first.resource and fused[0].units == 1,
          'fused copy must retain one original calling-thread demand, never sum sequential demands')
    _need(first.exact_thread.snapshot == resources.target
          and rewritten.temporal.resource_capacities.get(first.resource) == 1,
          'fused calling-thread identity/capacity differs from the original snapshot')
    read_artifact(target_ref)
    return dict(parent_sha=parent_sha, rewritten_sha=rewritten_sha, inventory=inventory,
        parent_resource_sha=semantic_sha256(projection_document(resources)),
        certificate_sha=semantic_sha256(certificate), old_activity_ids=old_activity_ids,
        new_activity=new_activity.id, old_occupancies=[o.id for o in old_exec], fused_id=fused_id,
        selected=selected, new_execution=new_exec)


def _project_resources(original, resources, rewritten, certificate, context):
    old_by_id = {row.occupancy: row for row in resources.executions}
    selected = set(context['old_occupancies']); first = context['selected'][0]
    inverse = {}
    for key in old_by_id:
        inverse.setdefault(certificate['occupancy_map'][key], []).append(key)
    executions = []
    for occupancy in context['new_execution']:
        parents = inverse.get(occupancy.id, ())
        if occupancy.id == context['fused_id']:
            _need(set(parents) == selected, 'fused occupancy has foreign or missing execution parents')
            refs = _distinct(ref for row in context['selected'] for ref in row.source_refs)
            lineage = dict(kind='CERTIFIED_FUSED_COPY_EXECUTION_OCCUPANCY',
                parent_resource_projection_sha256=context['parent_resource_sha'],
                certificate_sha256=context['certificate_sha'], original_occupancies=list(parents),
                same_exact_calling_thread=True, actual_execution_observed=False,
                physical_reservation_proven=False)
            executions.append(replace(first, occupancy=occupancy.id, source_refs=(*refs, lineage)))
        else:
            _need(len(parents) == 1 and parents[0] == occupancy.id and parents[0] not in selected,
                  'unmodified execution occupancy identity cannot be silently rebound')
            row = old_by_id[parents[0]]
            _need((row.resource, row.units) == (occupancy.resource, occupancy.units),
                  'unmodified execution resource/unknown demand changed')
            executions.append(row)
    _need(len(executions) == len(resources.executions) - len(selected) + 1,
          'copy fusion execution coverage differs')
    new_index = {row.occupancy: i for i, row in enumerate(executions)}
    index_map = {i: new_index[certificate['occupancy_map'][row.occupancy]]
                 for i, row in enumerate(resources.executions)}
    unresolved = []
    for row in resources.unresolved:
        copy = dict(row); field = copy.get('field')
        match = re.fullmatch(r'executions/(\d+)(/.*)?', field) if type(field) is str else None
        if match:
            index = int(match.group(1))
            _need(index in index_map, 'original unresolved execution index is out of range')
            updated = 'executions/' + str(index_map[index]) + (match.group(2) or '')
            if updated != field:
                copy.update(field=updated, parent_field=field,
                    parent_resource_projection_sha256=context['parent_resource_sha'])
        unresolved.append(copy)
    source = dict(schema='scar.fused-copy-resource-lineage', schema_version=1,
        parent_schedule_sha256=context['parent_sha'], rewritten_schedule_sha256=context['rewritten_sha'],
        parent_resource_projection_sha256=context['parent_resource_sha'],
        certificate_sha256=context['certificate_sha'],
        occupancy_map=certificate['occupancy_map'],
        unchanged_storage_and_budget=True, unknown_demands_preserved=True,
        actual_execution_observed=False, physical_reservation_proven=False)
    result = replace(resources, source_schedule_sha256=context['rewritten_sha'], executions=tuple(executions),
        unresolved=tuple(unresolved), source_refs=(*resources.source_refs, source))
    validate_resource_projection(result, rewritten, resources=context['inventory'])
    return result


def project_fused_copy_resources(original_schedule, original_resources, rewritten_schedule, certificate):
    """Merge selected sequential same-thread occupancies; preserve all holes."""
    context = _admit(original_schedule, original_resources, rewritten_schedule, certificate)
    return _project_resources(original_schedule, original_resources, rewritten_schedule, certificate, context)


def project_fused_copy_costs(original_schedule, original_resources, original_table,
                           rewritten_schedule, rewritten_resources, certificate):
    """Retain unchanged canonical evidence and mark the fused duration UNKNOWN."""
    context = _admit(original_schedule, original_resources, rewritten_schedule, certificate)
    validate_activity_cost_table(original_table, original_schedule, original_resources)
    expected_resources = _project_resources(original_schedule, original_resources,
        rewritten_schedule, certificate, context)
    _need(projection_document(expected_resources) == projection_document(rewritten_resources),
          'rewritten copy resources differ from exact parent/certificate projection')
    selected = set(context['old_activity_ids'])
    removed = [row for row in original_table.costs if selected.intersection(row.activities)]
    _need(len(removed) == len(selected)
          and all(len(row.activities) == 1 and set(row.activities) <= selected for row in removed),
          'copy fusion requires singleton cost subjects; collective/pool/shared costs cannot be split')
    retained = [row for row in original_table.costs if row not in removed]
    for row in retained:
        _need(all(certificate['activity_map'][key] == key for key in row.activities),
              'unmodified cost activity IDs cannot be silently rebound')
        _need(row.basis is not CostBasis.DECLARED_COMPONENT_MODEL
              or row.coverage is CostCoverage.COMPONENT_ONLY and row.missing_terms,
              'unchanged numeric costs require conditional component scope and retained residuals')
    subject = 'cost:fused-copy:' + semantic_sha256(dict(certificate=context['certificate_sha'],
        activity=context['new_activity']))[:32]
    conditions = tuple(dict.fromkeys(condition for row in removed for condition in row.conditions))
    missing = tuple(dict.fromkeys(('FUSED_COPY_COMPONENT_COST_NOT_CALIBRATED',
        'FUSED_COPY_COMPILATION_CACHE_AND_MEMORY_CONTEXT_DELTA_UNVALIDATED',
        *(term for row in removed for term in row.missing_terms))))
    fused = ActivityCost(subject, (context['new_activity'],), CostBasis.UNKNOWN,
        conditions=conditions, missing_terms=missing)
    costs = []; inserted = False
    for row in original_table.costs:
        if row in removed:
            if not inserted:
                costs.append(fused); inserted = True
        else:
            costs.append(row)
    parent_table_sha = semantic_sha256(activity_cost_document(original_table))
    source = dict(schema='scar.fused-copy-cost-source', schema_version=1,
        parent_cost_table_sha256=parent_table_sha,
        parent_schedule_sha256=context['parent_sha'], certificate_sha256=context['certificate_sha'],
        original_subjects=[row.subject for row in removed], replacement_subject=subject,
        old_component_sum_used_as_fused_cost=False, remaining_component_context_unvalidated=True)
    residual = dict(kind='FUSED_COPY_IMPLEMENTATION_CONTEXT_RESIDUAL',
        certificate_sha256=context['certificate_sha'], missing_terms=list(missing),
        signed_unknown=True, additional_work=False, cost_charged=False)
    result = replace(original_table, source_schedule_sha256=context['rewritten_sha'],
        resource_projection_sha256=semantic_sha256(projection_document(rewritten_resources)),
        costs=tuple(costs), source_refs=(*original_table.source_refs, source),
        unmapped_terms=(*original_table.unmapped_terms, residual))
    summary = validate_activity_cost_table(result, rewritten_schedule, rewritten_resources)
    lineage = dict(schema='scar.fused-copy-cost-lineage', schema_version=1,
        parent_schedule_sha256=context['parent_sha'], rewritten_schedule_sha256=context['rewritten_sha'],
        parent_resource_projection_sha256=context['parent_resource_sha'],
        rewritten_resource_projection_sha256=result.resource_projection_sha256,
        parent_cost_table_sha256=parent_table_sha,
        rewritten_cost_table_sha256=semantic_sha256(activity_cost_document(result)),
        certificate_sha256=context['certificate_sha'], original_source_refs=encode(original_table.source_refs),
        retired_subjects=[encode(row) for row in removed], replacement_subject=encode(fused),
        original_activities=list(context['old_activity_ids']), replacement_activity=context['new_activity'],
        unmodified_subjects=[dict(subject=row.subject, activities=list(row.activities),
            record_sha256=semantic_sha256(encode(row))) for row in retained],
        original_table_publication_reference_required_from_caller=True,
        unmodified_component_records_preserved=True, component_context_residuals_preserved=True,
        old_component_sum_used_as_fused_cost=False, fused_duration_known=False,
        physical_scope_latency=None, summary=summary, physical_optimality_proven=False,
        measurement_performed=False, execution_authorized=False, deployment_authorized=False)
    return result, lineage


__all__ = ['project_fused_copy_resources', 'project_fused_copy_costs']
