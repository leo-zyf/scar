"""Opposite branches cannot share an activation; common suffixes still may."""
from dataclasses import replace

from scar.analysis.source_selected_call_window import _normal_window
from scar.ir.program import normal_graph as G,normal_targets as T,source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.source_value_routes import SourceBodyValueFlow, SourceOperation


def test_selected_call_excludes_opposite_branch_but_not_common_suffix_or_other_iterations():
    key=SourceTemplateKey(0,P.ProductionPurpose.STATEMENT);rid,sid=P.ProductionRegionID,P.ProductionStepID
    port=lambda owner,kind:G.Port(key,owner,kind,completion=C.RETURN
        if owner==rid('body')and kind is G.PortKind.EXIT else C.FALLTHROUGH
        if kind in(G.PortKind.CALL_RETURN,G.PortKind.EXIT)else None)
    root=port(rid('body'),G.PortKind.ENTER);branch=port(sid('branch'),G.PortKind.ENTER)
    selected=port(sid('selected'),G.PortKind.ENTER);opposite=port(sid('opposite'),G.PortKind.ENTER)
    common=port(sid('common'),G.PortKind.ENTER);end=port(rid('body'),G.PortKind.EXIT)
    condition=G.Port(key,sid('branch'),G.PortKind.ARGUMENT)
    yes,no=G.CallSite(key,sid('yes-site')),G.CallSite(key,sid('no-site'))
    edge=lambda a,b,transition=G.Transition.DIRECT,site=None,when=None:G.Edge(a,b,G.EdgeKind.CONTROL,
        site=site,transition=transition,condition=None if when is None else condition,when=when)
    yes_enter=port(rid('yes'),G.PortKind.ENTER);no_enter=port(rid('no'),G.PortKind.ENTER)
    yes_exit=port(rid('yes'),G.PortKind.EXIT);no_exit=port(rid('no'),G.PortKind.EXIT)
    yes_return=port(sid('yes-return'),G.PortKind.CALL_RETURN)
    no_return=port(sid('no-return'),G.PortKind.CALL_RETURN)
    edges=(edge(root,branch),edge(branch,yes_enter,G.Transition.CALL,yes,True),
        edge(branch,no_enter,G.Transition.CALL,no,False),edge(yes_enter,opposite),edge(opposite,yes_exit),
        edge(no_enter,selected),edge(selected,no_exit),
        # This wrong caller return must not splice the selected branch into
        # the opposite body, despite sharing the same original return port.
        edge(no_exit,yes_return,G.Transition.RETURN,yes),
        edge(no_exit,no_return,G.Transition.RETURN,no),edge(yes_exit,yes_return,G.Transition.RETURN,yes),
        edge(yes_return,common),edge(no_return,common),edge(common,end))
    operation=lambda p:SourceOperation(p,P.ApplyLanguageAction(P.LanguageActionKind.CALL),(),())
    flow=SourceBodyValueFlow(T.GlobalPort('unit',G.Port(key,rid('body'),G.PortKind.PARAMETER)),
        T.Body('unit',key,rid('body')),ProviderArtifact('/fixture/source.json','0'*64,1,'fixture'),
        edges,tuple(operation(p)for p in(selected,opposite,common)),(),(),())
    prefix,suffix,excluded,boundaries=_normal_window(flow,T.GlobalPort('unit',selected))
    assert not boundaries and len(excluded)==1
    assert excluded[0].operation==T.GlobalPort('unit',opposite)
    assert excluded[0].selected_guard.when is False and excluded[0].excluded_guard.when is True
    assert excluded[0].selected_guard.condition==excluded[0].excluded_guard.condition==condition
    assert T.GlobalPort('unit',common)in suffix
    assert T.GlobalPort('unit',selected)in prefix and T.GlobalPort('unit',selected)in suffix
    looped=replace(flow,edges=(*edges,edge(common,branch,G.Transition.BACKEDGE)))
    _,_,excluded,boundaries=_normal_window(looped,T.GlobalPort('unit',selected))
    assert not excluded and boundaries[0].reason=='WRAPPER_NORMAL_CFG_HAS_BACKEDGE'
