"""Conditional component costs for original activity groups, never a new graph.

This IR checks identity, ownership, units and evidence references. The producer
must independently replay component facts and their source/resource context;
structural validation does not qualify a measurement, mathematical guarantee,
complete call latency, or a deployment decision.
"""
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode
from scar.ir.temporal_resources import ResourceProjection, projection_document
from scar.ir.versioned_schedule import schedule_document, validate_versioned_schedule

SCHEMA = 'scar.activity-cost-table'
EFFECT_SCOPE = 'MODELED_EXISTING_CPU_SCHEDULE_ACTIVITIES_NOT_COMPLETE_CALL'


class CostBasis(str, Enum):
    UNKNOWN = 'UNKNOWN'
    DECLARED_COMPONENT_MODEL = 'DECLARED_COMPONENT_MODEL'
    OBSERVED_SCOPE = 'OBSERVED_SCOPE'


class IntervalKind(str, Enum):
    EMPIRICAL_PREDICTION = 'EMPIRICAL_PREDICTION'
    MATHEMATICAL_BOUND = 'MATHEMATICAL_BOUND'


class CostCoverage(str, Enum):
    COMPLETE_SUBJECT = 'COMPLETE_SUBJECT'
    COMPONENT_ONLY = 'COMPONENT_ONLY'


@dataclass(frozen=True, slots=True)
class CostInterval:
    lower_ticks: int | None
    upper_ticks: int | None
    kind: IntervalKind = IntervalKind.EMPIRICAL_PREDICTION


@dataclass(frozen=True, slots=True)
class ActivityCost:
    subject: str
    activities: tuple[str, ...]
    basis: CostBasis
    estimate_ticks: int | None = None
    interval: CostInterval | None = None
    component_refs: tuple[ProviderFactReference, ...] = ()
    charge_keys: tuple[str, ...] = ()
    coverage: CostCoverage = CostCoverage.COMPONENT_ONLY
    conditions: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActivityCostTable:
    source_schedule_sha256: str
    resource_projection_sha256: str
    target: ProviderArtifact
    costs: tuple[ActivityCost, ...]
    effect_scope: str = EFFECT_SCOPE
    source_refs: tuple[dict, ...] = ()
    tick_numerator: int = 1
    tick_denominator: int = 10**9
    unmapped_terms: tuple[dict, ...] = ()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _strings(value, label, *, nonempty=False):
    _need(type(value) is tuple and (bool(value) or not nonempty)
        and all(type(item) is str and bool(item) for item in value)
        and len(set(value)) == len(value), 'unique original ' + label + ' tuple required')


def _typed(table):
    _need(type(table) is ActivityCostTable and type(table.target) is ProviderArtifact,
          'typed activity cost table and original target required')
    _need(type(table.costs) is tuple and all(type(row) is ActivityCost for row in table.costs),
          'immutable activity cost records required')
    detached = decode(ActivityCostTable, encode(table))
    _need(detached.target.schema == 'scar.target-hardware-graph' and Path(detached.target.uri).is_absolute(),
          'original absolute hardware snapshot required')
    _need(table.effect_scope == EFFECT_SCOPE, 'activity component scope cannot imply complete call cost')
    _need(all(type(value) is int and value > 0 for value in (table.tick_numerator, table.tick_denominator)),
          'one exact positive rational table tick required')
    for records in (table.source_refs, table.unmapped_terms):
        _need(type(records) is tuple and all(type(row) is dict for row in records), 'original reference records required')
    return detached


def activity_cost_document(table):
    return dict(schema=SCHEMA, schema_version=1, **encode(_typed(table)))


def activity_cost_from_dict(document):
    value = dict(document)
    _need(value.pop('schema', None) == SCHEMA and type(value.get('schema_version')) is int
          and value.pop('schema_version') == 1, 'versioned activity cost table required')
    return _typed(decode(ActivityCostTable, value))


def validate_activity_cost_table(table, schedule, resource_projection):
    """Validate original group coverage; the producer owns source-fact replay."""
    table = _typed(table)
    validate_versioned_schedule(schedule)
    _need(schedule.temporal is not None and type(resource_projection) is ResourceProjection,
          'original temporal schedule and typed resource projection required')
    digest = _digest(schedule_document(schedule))
    _need(table.source_schedule_sha256 == digest == resource_projection.source_schedule_sha256
          and table.resource_projection_sha256 == _digest(projection_document(resource_projection))
          and table.target == resource_projection.target, 'cost table source/resource/target identity differs')
    activities = {row.id: row for row in schedule.temporal.activities}
    subjects, covered, charged = set(), set(), set()
    known = complete = empirical = 0
    for row in table.costs:
        _need(type(row.subject) is str and row.subject and row.subject not in subjects, 'unique original cost subject required')
        subjects.add(row.subject)
        _strings(row.activities, 'subject activity IDs', nonempty=True)
        _need(set(row.activities) <= set(activities) and not covered.intersection(row.activities),
              'parent/group/leaf must not double-charge original activities')
        _need(len({activities[key].scope for key in row.activities}) == 1, 'cost group must retain one original body scope')
        covered.update(row.activities)
        for field in ('charge_keys', 'conditions', 'missing_terms'):
            _strings(getattr(row, field), field)
        _need(not charged.intersection(row.charge_keys), 'original demand term occurrence charged more than once')
        charged.update(row.charge_keys)
        _need(type(row.basis) is CostBasis and type(row.coverage) is CostCoverage, 'typed basis and subject coverage required')
        _need(type(row.component_refs) is tuple and all(type(ref) is ProviderFactReference
            and Path(ref.artifact.uri).is_absolute() and ref.locator.startswith('#') for ref in row.component_refs),
            'original component FactRefs required, not unqualified numeric profiles')
        if row.basis is CostBasis.UNKNOWN:
            _need(row.estimate_ticks is None and row.interval is None and bool(row.missing_terms)
                  and row.coverage is CostCoverage.COMPONENT_ONLY, 'UNKNOWN must retain missing terms and no numeric cost')
            continue
        _need(row.basis is CostBasis.DECLARED_COMPONENT_MODEL,
              'OBSERVED_SCOPE requires an independent whole-scope reader; canonical components are not scope observations')
        _need(type(row.estimate_ticks) is int and row.estimate_ticks >= 0 and row.component_refs
              and row.charge_keys and row.conditions, 'numeric component cost needs source facts, occurrence keys and conditions')
        if row.coverage is CostCoverage.COMPLETE_SUBJECT:
            _need(not row.missing_terms, 'complete subject cannot omit required terms')
            complete += 1
        known += 1
        if row.interval is not None:
            interval = row.interval
            _need(type(interval) is CostInterval and interval.kind is IntervalKind.EMPIRICAL_PREDICTION,
                  'mathematical guaranteed bounds have no supported proof reader; sample ranges remain empirical')
            _need(any(value is not None for value in (interval.lower_ticks, interval.upper_ticks))
                  and all(value is None or type(value) is int and value >= 0
                          for value in (interval.lower_ticks, interval.upper_ticks))
                  and (interval.lower_ticks is None or interval.upper_ticks is None
                       or interval.lower_ticks <= interval.upper_ticks), 'ordered empirical tick range required')
            _need((interval.lower_ticks is None or interval.lower_ticks <= row.estimate_ticks)
                  and (interval.upper_ticks is None or row.estimate_ticks <= interval.upper_ticks),
                  'component estimate must lie inside its empirical interval')
            empirical += 1
    _need(covered == set(activities), 'every original activity must have exactly one cost subject, including UNKNOWN')
    return dict(subjects=len(subjects), activities=len(activities), known_component_subjects=known,
        complete_subjects=complete, unknown_subjects=len(subjects)-known, empirical_intervals=empirical,
        unmapped_terms=len(table.unmapped_terms), original_activity_coverage_complete=True,
        whole_scope_cost_closed=False, source_component_semantics_replayed_by_this_validator=False,
        mathematical_bounds_verified=False, physical_optimality_proven=False,
        solver_authorized=False, execution_authorized=False)


__all__ = ['SCHEMA', 'EFFECT_SCOPE', 'CostBasis', 'IntervalKind', 'CostCoverage', 'CostInterval',
           'ActivityCost', 'ActivityCostTable', 'activity_cost_document', 'activity_cost_from_dict',
           'validate_activity_cost_table']
