"""Cost-table structure and ownership, not canonical measurement admission."""
from dataclasses import replace
import json

import pytest

from test_temporal_resources import inputs
from scar.analysis.software_resources import derive_temporal_resources
from scar.ir.activity_costs import (
    ActivityCost, ActivityCostTable, CostBasis as B, CostCoverage as C,
    CostInterval, IntervalKind, activity_cost_document, activity_cost_from_dict,
    validate_activity_cost_table,
)
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.temporal_resources import projection_document
from scar.ir.versioned_schedule import schedule_document
from scar.optimization.resource_solver import semantic_sha256


def _tables(inputs, tmp_path):
    schedule,target,_,_,context=inputs
    projection=derive_temporal_resources(schedule,target,source_context=context)
    path=tmp_path/'declared-component.json'
    path.write_text(json.dumps(dict(schema='scar.test-declared-component',sample_range=[9,12],median=10)))
    ref=ProviderArtifact(**artifact_reference(path,'scar.test-declared-component'))
    fact=ProviderFactReference(ref,'#/median')
    ids=[row.id for row in schedule.temporal.activities]
    unknown=tuple(ActivityCost('subject:'+key,(key,),B.UNKNOWN,missing_terms=('NO_ADMITTED_COMPONENT',)) for key in ids)
    table=ActivityCostTable(semantic_sha256(schedule_document(schedule)),
        semantic_sha256(projection_document(projection)),projection.target,unknown,
        source_refs=({'fixture':'STRUCTURAL_MODEL_ONLY_NO_MEASUREMENT_QUALIFICATION'},),
        unmapped_terms=({'term':'guard-boundary','reason':'NOT_A_MODELED_SOURCE_ACTIVITY'},))
    group=ActivityCost('subject:original-group',tuple(ids[:2]),B.DECLARED_COMPONENT_MODEL,10,
        CostInterval(9,12),component_refs=(fact,),charge_keys=('domain:candidate:term:0',),
        coverage=C.COMPLETE_SUBJECT,conditions=('EXACT_ORIGINAL_GROUP_AND_RESOURCE_CONTEXT',))
    repeated=ActivityCost('subject:separate-occurrence',(ids[2],),B.DECLARED_COMPONENT_MODEL,10,
        CostInterval(9,12),component_refs=(fact,),charge_keys=('domain:candidate:term:1',),
        conditions=('INDEPENDENT_OCCURRENCE_SAME_CANONICAL_PRIMITIVE',),missing_terms=('UNMODELED_SUBJECT_OVERHEAD',))
    known=replace(table,costs=(group,repeated,*unknown[3:]))
    return schedule,projection,table,known


def test_unknown_stays_none_and_grouped_costs_reuse_fact_but_not_occurrence(inputs,tmp_path):
    schedule,projection,unknown,known=_tables(inputs,tmp_path)
    status=validate_activity_cost_table(unknown,schedule,projection)
    assert status['unknown_subjects']==len(schedule.temporal.activities)
    assert all(row.estimate_ticks is None and row.interval is None for row in unknown.costs)
    result=validate_activity_cost_table(known,schedule,projection)
    assert result['known_component_subjects']==2 and result['complete_subjects']==1
    assert result['unmapped_terms']==1 and result['whole_scope_cost_closed'] is False
    assert result['mathematical_bounds_verified'] is False and result['solver_authorized'] is False
    assert result['source_component_semantics_replayed_by_this_validator'] is False
    assert activity_cost_from_dict(json.loads(json.dumps(activity_cost_document(known))))==known
    # The same canonical sample is legitimate evidence for two occurrences.
    assert known.costs[0].component_refs==known.costs[1].component_refs
    assert known.costs[0].charge_keys!=known.costs[1].charge_keys


def test_bare_numbers_sample_bounds_observed_upgrade_and_duplicate_charges_are_rejected(inputs,tmp_path):
    schedule,projection,unknown,known=_tables(inputs,tmp_path)
    with pytest.raises(ValueError,match='UNKNOWN'):
        validate_activity_cost_table(replace(unknown,costs=(
            replace(unknown.costs[0],estimate_ticks=0),*unknown.costs[1:])),schedule,projection)
    first=known.costs[0]
    for bad in (replace(first,component_refs=()),
                replace(first,interval=CostInterval(9,12,IntervalKind.MATHEMATICAL_BOUND)),
                replace(first,estimate_ticks=20),
                replace(first,basis=B.OBSERVED_SCOPE)):
        with pytest.raises(ValueError):
            validate_activity_cost_table(replace(known,costs=(bad,*known.costs[1:])),schedule,projection)
    with pytest.raises(ValueError,match='occurrence'):
        validate_activity_cost_table(replace(known,costs=(first,
            replace(known.costs[1],charge_keys=first.charge_keys),*known.costs[2:])),schedule,projection)
    with pytest.raises(ValueError,match='double-charge'):
        validate_activity_cost_table(replace(known,costs=(*known.costs,
            replace(first,subject='parent-again',activities=(first.activities[0],)))),schedule,projection)
