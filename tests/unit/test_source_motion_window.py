"""Balanced source-control queries distinguish MAY membership and all paths."""
from collections import defaultdict
from types import SimpleNamespace

from scar.analysis.source_motion_window import _control_walk, _transport_slice
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.program.model import ValueKind, ValueType


def test_balanced_entered_body_avoiding_repeat_and_budget_paths():
    key = SourceTemplateKey(0, P.ProductionPurpose.STATEMENT)
    region = P.ProductionRegionID
    step = P.ProductionStepID
    body = region('body')
    start = G.Port(key, body, G.PortKind.ENTER)
    done = G.Port(key, body, G.PortKind.EXIT, completion=C.FALLTHROUGH)
    returned = G.Port(key, body, G.PortKind.EXIT, completion=C.RETURN)
    selected = G.Port(key, step('selected'), G.PortKind.ENTER)
    after = G.Port(key, step('selected'), G.PortKind.EXIT, completion=C.FALLTHROUGH)
    header = G.Port(key, step('loop'), G.PortKind.ENTER)
    child = G.Port(key, region('shared'), G.PortKind.ENTER)
    child_done = G.Port(key, region('shared'), G.PortKind.EXIT, completion=C.FALLTHROUGH)
    first = G.CallSite(key, step('first'))
    unrelated = G.CallSite(key, step('other'))
    edge = lambda source,target,**kwargs:G.Edge(source,target,G.EdgeKind.CONTROL,**kwargs)
    edges = [edge(start,child,transition=G.Transition.CALL,site=first),
        edge(child,child_done),
        edge(child_done,selected,transition=G.Transition.RETURN,site=first),
        edge(child_done,returned,transition=G.Transition.RETURN,site=unrelated),
        edge(selected,after),edge(after,done)]
    def query(rows, **kwargs):
        adjacency = defaultdict(list)
        for row in rows:adjacency[row.source].append(row)
        return _control_walk(start, adjacency,
            boundary=lambda p:p.owner==body and p.kind is G.PortKind.EXIT,
            header=header, selected=selected, **kwargs)
    normal = query(edges)
    assert normal['complete'] and normal['selected_seen']
    assert not normal['repeated'] and not normal['avoiding_cycle']
    assert normal['terminals'] and all(row['hits']==1 for row in normal['terminals'])
    assert returned not in normal['ports']  # wrong shared-recipe return was not spliced
    avoid = query([*edges,edge(start,returned)])
    assert any(row['hits']==0 for row in avoid['terminals'])
    repeat = query([*edges,edge(after,selected,transition=G.Transition.BACKEDGE)])
    assert repeat['repeated']
    cycle = query([*edges,edge(start,start)])
    assert cycle['avoiding_cycle']
    resumed = query([*edges,edge(start,selected,transition=G.Transition.RESUME,site=first)])
    assert not resumed['complete']
    assert any(row['reason']=='SUSPENDED_ACTIVATION_NOT_SELECTED' for row in resumed['frontier'])
    bounded = query(edges,max_states=1)
    assert not bounded['complete']
    assert any(row['reason']=='CONTROL_STATE_BUDGET_EXHAUSTED' for row in bounded['frontier'])
    reference = ValueType(ValueKind.REFERENCE)
    produced = G.Port(key, step('read'), G.PortKind.RESULT, completion=C.FALLTHROUGH)
    child_value = G.Port(key, region('shared'), G.PortKind.RESULT, completion=C.FALLTHROUGH)
    actual = G.Port(key, step('caller'), G.PortKind.RESULT, completion=C.FALLTHROUGH)
    parameter = G.Port(key, region('other'), G.PortKind.PARAMETER)
    unrelated_value = G.Port(key, region('other'), G.PortKind.RESULT, completion=C.FALLTHROUGH)
    data = (G.Edge(produced,child_value,G.EdgeKind.DATA,reference),
        G.Edge(child_value,actual,G.EdgeKind.DATA,reference,site=first,transition=G.Transition.RETURN),
        G.Edge(parameter,unrelated_value,G.EdgeKind.DATA,reference))
    original = SimpleNamespace(edges=data)
    sliced, count, complete = _transport_slice(original,(actual,),max_states=20)
    assert sliced == data[:2] and complete and count == 3
    assert original.edges == data  # query projection never modifies the shared flow
    assert not _transport_slice(original,(actual,),max_states=1)[2]
    assert normal['complete'] and not normal['frontier']  # DATA budget cannot erase control facts
