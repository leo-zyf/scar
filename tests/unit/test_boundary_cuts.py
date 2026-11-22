"""Exclusive cut algebra and UNKNOWN-group projection; no native execution."""
from copy import deepcopy
from dataclasses import replace
import json
import statistics

import pytest

from test_implementation_envelope import original_and_envelope
from test_temporal_resources import inputs
from scar.analysis.schedule_time import attach_task_time
from scar.analysis.software_resources import derive_temporal_resources
from scar.compiler.calibrate_boundary_cuts import (
    CONDITIONS, CUTS, MISSING, PHASES, SCHEMA, boundary_cut_statistics,
)
from scar.ir.activity_costs import (ActivityCost, ActivityCostTable, CostBasis as B,
    CostCoverage as C, CostInterval, IntervalKind, validate_activity_cost_table)
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.temporal_resources import projection_document
from scar.ir.versioned_schedule import schedule_document
from scar.optimization.boundary_activity_costs import project_boundary_activity_costs
from scar.optimization.resource_solver import semantic_sha256


def _cuts():
    # Median of differences deliberately differs from difference of medians.
    vectors = ([100, 200, 202, 205, 209, 214, 220, 227],
               [200, 201, 203, 206, 210, 215, 221, 228],
               [300, 400, 402, 405, 409, 414, 420, 427])
    return dict(schema=SCHEMA, schema_version=1, cuts=list(CUTS),
        phases=deepcopy(list(PHASES)), conditions=list(CONDITIONS), missing_terms=list(MISSING),
        clock=dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
        repetitions=3, warmup=1,
        samples=[dict(id='same-call:'+str(i), timestamps_ns=list(vector))
                 for i, vector in enumerate(vectors)])


def test_exclusive_shared_cut_samples_partition_calls_and_reject_invalid_partitions():
    document = _cuts()
    rows = boundary_cut_statistics(document)
    assert len(rows) == 7
    for i, sample in enumerate(document['samples']):
        assert sum(row['samples_ns'][i] for row in rows) == (
            sample['timestamps_ns'][-1] - sample['timestamps_ns'][0])
    assert rows[0]['samples_ns'] == [100, 1, 100] and rows[0]['median_ns'] == 100
    median_subtraction = statistics.median(s['timestamps_ns'][1] for s in document['samples']) - (
        statistics.median(s['timestamps_ns'][0] for s in document['samples']))
    assert median_subtraction == 1 != rows[0]['median_ns']
    invalid = []
    bad = deepcopy(document); bad['samples'][0]['timestamps_ns'][4] = 204; invalid.append(bad)
    bad = deepcopy(document); bad['samples'][0]['timestamps_ns'][0] = -1; invalid.append(bad)
    bad = deepcopy(document); bad['samples'][1]['id'] = bad['samples'][0]['id']; invalid.append(bad)
    bad = deepcopy(document); bad['samples'][0]['timestamps_ns'].pop(); invalid.append(bad)
    bad = deepcopy(document); bad['cuts'].pop(); invalid.append(bad)
    bad = deepcopy(document); bad['phases'][1]['start'] = CUTS[0]; invalid.append(bad)
    bad = deepcopy(document); bad['missing_terms'] = []; invalid.append(bad)
    bad = deepcopy(document); bad['conditions'] = []; invalid.append(bad)
    for bad in invalid:
        with pytest.raises(ValueError):
            boundary_cut_statistics(bad)


def test_boundary_projection_only_replaces_complete_unknown_groups_and_preserves_known_pool(
        original_and_envelope, inputs, tmp_path, monkeypatch):
    from scar.compiler import calibrate_boundary_cuts as calibration
    from scar.optimization import stateful_cpu_model as stateful_cpu_model
    _, envelope = original_and_envelope
    schedule = attach_task_time(envelope)
    _, target, _, _, context = inputs
    resources = derive_temporal_resources(schedule, target, source_context=context)
    document = _cuts(); document['resource_context'] = {'fixture': 'SOURCE_READER_ISOLATED'}
    path = tmp_path / 'synthetic-cuts.json'; path.write_text(json.dumps(document))
    reference = artifact_reference(path, SCHEMA)
    fact = ProviderFactReference(ProviderArtifact(**reference), '#/samples')
    activities = {row.task: row.id for row in schedule.temporal.activities}
    unknown = tuple(ActivityCost('unknown:'+key, (key,), B.UNKNOWN,
        missing_terms=('NO_QUALIFIED_COMPONENT',)) for key in activities.values())
    # One allocate task owns both original workspace slots. Its pool cost is
    # retained once, rather than split/repeated for the source cut intervals.
    pools = tuple(ActivityCost('known-pool:'+task, (activities[task],),
        B.DECLARED_COMPONENT_MODEL, 12, CostInterval(9, 15), component_refs=(fact,),
        charge_keys=('fixture:original-pool:'+task,), coverage=C.COMPLETE_SUBJECT,
        conditions=('STRUCTURAL_FIXTURE_NOT_SOURCE_MEASUREMENT_ADMISSION',))
        for task in ('allocate', 'release'))
    pool_activities = {a for row in pools for a in row.activities}
    table = ActivityCostTable(semantic_sha256(schedule_document(schedule)),
        semantic_sha256(projection_document(resources)), resources.target,
        (*pools, *(r for r in unknown if not pool_activities.intersection(r.activities))),
        unmapped_terms=({'term': 'original-unmapped', 'reason': 'NOT_AN_ACTIVITY'},))
    # Only the heavy source reader/context admission boundary is isolated.
    # The real table, group coverage and numeric cut validators still run.
    monkeypatch.setattr(calibration, 'read_boundary_cuts', lambda *a, **kw: deepcopy(document))
    monkeypatch.setattr(stateful_cpu_model, 'stateful_resource_context',
        lambda *a, **kw: document['resource_context'])
    domain = {'candidates': [{'id': 'serial', 'worker_count': 1}]}
    def project(value=table, refs=(reference,)):
        return project_boundary_activity_costs(value, schedule, resources,
            domain=domain, candidate_id='serial', calibrations=refs)
    result, derivation = project()
    assert tuple(row for row in result.costs if row.subject.startswith('known-pool:')) == pools
    mapped = [row for row in result.costs if row.subject.startswith('cost:boundary:')]
    assert len(mapped) == 7
    assert all(row.basis is B.DECLARED_COMPONENT_MODEL and row.coverage is C.COMPONENT_ONLY
        and row.interval.kind is IntervalKind.EMPIRICAL_PREDICTION
        and row.conditions == CONDITIONS and row.missing_terms == MISSING for row in mapped)
    assert {a for row in result.costs for a in row.activities} == set(activities.values())
    assert sum(len(row.activities) for row in result.costs) == len(activities)
    assert result.unmapped_terms[0] == table.unmapped_terms[0]
    assert result.unmapped_terms[-1]['cost_charged'] is False
    assert derivation['complete_call_cost'] is False and derivation['native_execution_authorized'] is False
    mapping = derivation['mapping'][0]
    assert mapping['samplewise_cut_differences'] and not mapping['median_subtraction_used']
    assert mapping['phases'][0]['statistics']['median_ns'] == 100
    assert not validate_activity_cost_table(result, schedule, resources)['whole_scope_cost_closed']
    entry_task = next(t.id for t in schedule.tasks
        if t.attrs.get('phase_kind') == PHASES[0]['phase_kinds'][0])
    entry = next(r for r in table.costs if activities[entry_task] in r.activities)
    known_entry = replace(pools[0], subject='known-boundary', activities=entry.activities,
        charge_keys=('fixture:known-boundary',))
    with pytest.raises(ValueError, match='complete UNKNOWN'):
        project(replace(table, costs=tuple(known_entry if r == entry else r for r in table.costs)))
    other = next(r for r in table.costs if activities['snapshot'] in r.activities)
    crossing = replace(entry, subject='unknown-crossing', activities=(*entry.activities, *other.activities))
    with pytest.raises(ValueError, match='complete UNKNOWN'):
        project(replace(table, costs=(crossing, *(r for r in table.costs if r not in (entry, other)))))
    with pytest.raises(ValueError, match='charged twice'):
        project(refs=(reference, reference))
