"""Carry roles follow original write bindings, independently of argument order."""
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace as NS

import pytest

from test_loop_frontend import _original_loop
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.compiler.graph_frontend import GraphRegionRejected
from scar.compiler.invariant_loop_frontend import _InvariantLoopLowerer
from scar.compiler import invariant_loop_frontend as frontend
from scar.compiler.model import ArrayRegion, Expr
from scar.compiler.pipeline_frontend import _Callee
from scar.ir.program import source_productions as P, normal_graph as G, normal_links as F
from scar.ir.program.control import LanguageCompletionKind as C


def test_multi_input_leaf_carry_write_infers_roles_and_rejects_duplicate_carry(tmp_path,monkeypatch):
    original,function,plan,_,pipeline = _original_loop()
    ref=plan.regions[0].parameters[0]; world=plan.regions[0].parameters[-1]
    sid=P.ProductionStepID; rid=P.ProductionRegionID
    result=lambda name,i=0:P.ProductionResult(sid(name),i)
    normal=P.ProductionCompletion(C.FALLTHROUGH,(ref,world))
    extra=[]
    for name,previous in (('extra-a','carry'),('extra-b','extra-a')):
        extra.append(P.ProductionStep(sid(name),rid('body'),P.ProductionOccurrence(4),
            P.AccessBinding(P.BindingActionKind.READ,0),(result(previous,1),),(normal,)))
    steps=[]
    for row in plan.steps:
        if row.id.value=='range-name':row=replace(row,arguments=(P.ProductionParameter(rid('entry'),5),))
        if row.id.value=='body-call':
            steps.extend(extra)
            row=replace(row,arguments=(result('callee'),result('extra-b'),result('carry'),result('extra-a'),result('extra-b',1)))
        steps.append(row)
    regions=[]
    for region in plan.regions:
        if region.id.value=='entry':region=replace(region,parameters=(ref,ref,ref,ref,ref,world))
        if region.id.value=='body':
            names=list(region.steps);at=names.index(sid('body-call'));names[at:at]=[row.id for row in extra]
            region=replace(region,steps=tuple(names))
        regions.append(region)
    plan=replace(plan,regions=tuple(regions),steps=tuple(steps))
    context=NamespacePlanView(original.unit,plan)
    ports=[replace(port,ordinal=5) if port.owner==rid('entry') and port.kind is G.PortKind.PARAMETER
        and port.ordinal==3 else port for port in original.facts.ports]
    def value(raw):
        port=context.value(raw).port
        if port not in ports:ports.append(port)
        return ports.index(port)
    new_formals=[value(P.ProductionParameter(rid('entry'),i)) for i in (3,4)]
    source_names=list(original.facts.names)
    for namespace,formal,name in ((F.Namespace(F.NamespaceKind.LEXICAL,4,'independent_a'),new_formals[0],'extra-a'),
            (F.Namespace(F.NamespaceKind.LEXICAL,5,'independent_b'),new_formals[1],'extra-b')):
        source_names.extend((F.NameAccess(namespace,formal,True),F.NameAccess(namespace,value(result(name)),False)))
    body_call=F.Call(value(result('callee')),(value(result('extra-b')),value(result('carry')),value(result('extra-a'))),(),
        value(result('body-call')),value(result('extra-b',1)),value(result('body-call',1)))
    environment=NS(kind=P.DeferredBodyKind.FUNCTION,captures=(),owned_cells=(),
        parameters=tuple(NS(kind=P.DeferredParameterKind.POSITIONAL_OR_KEYWORD,default_source=None) for _ in range(4)))
    function=NS(**{**vars(function),'parameter_names':('clock','payload','external_one','external_two'),
        'formals':(*function.formals[:3],*new_formals,function.formals[-1]),'environment':environment})
    facts=NS(**{**vars(original.facts),'ports':tuple(ports),'names':tuple(source_names),
        'calls':(original.facts.calls[0],body_call)})
    original=NS(**{**vars(original),'facts':facts})
    leaf=ArrayRegion('original:leaf',pipeline.source,('external_b','state','external_a'),
        (Expr('b','input',attrs={'name':'external_b'}),Expr('carry','input',attrs={'name':'state'}),
         Expr('a','input',attrs={'name':'external_a'}),Expr('invariant','add',('a','b')),
         Expr('output','add',('carry','invariant'))),'output')
    leaf_function=NS(template=function.template,environment=NS(parameters=environment.parameters[:3]))
    globals={'arbitrary_body':(_Callee('arbitrary_body',leaf_function,leaf),)}
    resolver=NS(plan=lambda *_:context)
    lowered=_InvariantLoopLowerer(original,function,resolver,globals).lower()
    body,carry,trip,mapping,name,origins=lowered
    assert body is leaf and carry=='state' and trip=='clock' and name=='arbitrary_body'
    assert mapping=={'external_b':'external_two','state':'payload','external_a':'external_one'}
    witness=next(row for row in origins if row.get('role')=='lexical_carry_write')
    assert witness['outer_parameter_index']==1 and witness['leaf_formal_index']==1
    assert any(row.get('role')=='single_shared_invariant_loop' and row['loop_unrolled'] is False for row in origins)
    # Same carry supplied twice is outside the first profile even when both
    # reads could be mathematically legal; the dependency map must be unique.
    bad_steps=tuple(replace(row,arguments=(result('callee'),result('carry'),result('carry'),result('extra-a'),result('extra-b',1)))
        if row.id.value=='body-call' else row for row in plan.steps)
    changed=replace(plan,steps=bad_steps);changed_context=NamespacePlanView(original.unit,changed)
    bad_call=replace(body_call,positional=(value(result('carry')),value(result('carry')),value(result('extra-a'))))
    bad_original=NS(**{**vars(original),'facts':NS(**{**vars(facts),'calls':(facts.calls[0],bad_call)})})
    with pytest.raises(GraphRegionRejected) as error:
        _InvariantLoopLowerer(bad_original,function,NS(plan=lambda *_:changed_context),globals).lower()
    assert error.value.code=='INVARIANT_LOOP_CARRY_MULTIPLICITY'
    # Persisted provenance is only a replay request: matching source bytes and
    # F4 marker strings do not permit replacing the discovered leaf's DAG.
    source_path=tmp_path/'original.py';source_path.write_text('def original():\n    pass\n')
    digest=sha256(source_path.read_bytes()).hexdigest()
    def identity(key):
        return dict(kind='source_identity',unit='unit',method='COMPILE_ONLY_RETAINED_F0',
            semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS',code_key=key)
    outer=replace(body.source,path=str(source_path),sha256=digest,qualname='outer',
        origins=(identity(body.source.code_key),))
    retained_body=replace(body,source=replace(body.source,path=str(source_path),sha256=digest,
        origins=(identity(body.source.code_key),)))
    proof=frontend.InvariantLoopProof('invariant-array-loop:fixture',outer,function.parameter_names,
        retained_body,carry,trip,mapping,name,origins)
    path=tmp_path/'proof.json';frontend.write_invariant_loop_proof(proof,path)
    catalog=object();replays=[]
    def rediscover(candidate,source):
        assert candidate is catalog and source==str(source_path)
        replays.append(source);return [proof],[]
    monkeypatch.setattr(frontend,'discover_invariant_loops',rediscover)
    assert frontend.read_invariant_loop_proof(path,catalog=catalog)==proof
    changed=json.loads(path.read_text());changed['body']['nodes'][-1]['op']='mul'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='original F4 rediscovery'):
        frontend.read_invariant_loop_proof(path,catalog=catalog)
    assert len(replays)==2
