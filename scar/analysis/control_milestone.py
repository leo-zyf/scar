"""Represent a source-proven semantic gate without inventing CPU work."""
from dataclasses import replace

from scar.analysis.implementation_scope import derive_stateful_scope_envelope
from scar.ir.versioned_schedule import (TemporalDurationBasis as Basis,
    TemporalOccupancyKind as OccupancyKind, validate_versioned_schedule)
from scar.optimization.schedule import schedule_from_dict


def derive_scope_control_milestone(schedule, recipe):
    """Recheck the source skeleton, then remove only its phantom execution demand."""
    validate_versioned_schedule(schedule)
    original = derive_stateful_scope_envelope(schedule_from_dict(recipe['task_graph']), recipe)
    if replace(schedule, temporal=None) != original or schedule.temporal is None:
        raise ValueError('complete original scope envelope and semantic time are required')
    scope = original.proof['implementation_scope_cleanup']
    marker = scope['milestone']; identity = marker['task_id']
    activities = [row for row in schedule.temporal.activities if row.task == identity]
    if len(activities) != 1:
        raise ValueError('one original source control milestone lifecycle required')
    activity = activities[0]
    if activity.duration_basis not in (Basis.UNKNOWN, Basis.SOURCE_CONTROL_MILESTONE):
        raise ValueError('cannot replace an observed or already modeled service interval')
    removed = [row for row in schedule.temporal.occupancies
               if row.kind is OccupancyKind.EXECUTION_DEMAND and row.activity == activity.id]
    if any(row.units is not None or schedule.temporal.resource_capacities[row.resource] is not None
           for row in removed):
        raise ValueError('cannot discard an existing qualified execution demand')
    certificate = dict(schema='scar.source-control-milestone', schema_version=1,
        task_id=identity, scope_proof_key='implementation_scope_cleanup',
        source=scope['source'], source_adapter=scope['source_adapter'],
        milestone=marker, measured_zero_latency_claimed=False,
        additional_emitted_work=False)
    refs = tuple(ref for ref in activity.source_refs if ref.get('schema') != certificate['schema'])
    marked = replace(activity, duration_basis=Basis.SOURCE_CONTROL_MILESTONE,
        duration_lower_ticks=0, duration_upper_ticks=0, source_refs=(*refs, certificate))
    occupancies = tuple(row for row in schedule.temporal.occupancies if row not in removed)
    capacities = dict(schedule.temporal.resource_capacities)
    used = {row.resource for row in occupancies}
    for row in removed:
        if row.resource not in used:
            capacities.pop(row.resource)
    result = replace(schedule, temporal=replace(schedule.temporal,
        activities=tuple(marked if row.id == marked.id else row for row in schedule.temporal.activities),
        occupancies=occupancies, resource_capacities=capacities))
    validate_versioned_schedule(result)
    return result


__all__ = ['derive_scope_control_milestone']
