"""Aggregation only; actual source/cost admission is checked by the L3 reader."""
from scar.compiler.solve_pipeline_temporal import _decision


def test_finite_model_tie_policy_keeps_unknown_and_exhausted_cases_in_domain():
    def case(identifier, duration, *, complete=True):
        return dict(id=identifier, makespan_ticks=duration, search_complete=complete,
                    proved_infeasible=False, lower_bound_ticks=duration if complete else None)
    domain = [case('b', 13), case('a', 13), case('c', 21)]
    closed = _decision(domain)
    assert closed['selected_case_id']=='a' and closed['tied_best_case_ids']==['a','b']
    assert closed['model_optimum_proven'] and closed['unique_best_case_proven'] is False
    unknown = _decision([*domain, case('unknown', None, complete=False)])
    assert unknown['status']=='UNRESOLVED' and unknown['selected_case_id'] is None
    assert unknown['provisional_best_case_id']=='a' and unknown['lower_bound_ticks'] is None
    exhausted = _decision([case('full', 21), case('limited', 13, complete=False)])
    assert exhausted['selection_decision']=='NO_OP' and exhausted['unresolved_case_ids']==['limited']
