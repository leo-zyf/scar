"""Source-window facts remain conditional, with exact operation blockers."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from scar.analysis.array_motion_legality import derive_array_motion_legality, verify_array_motion_legality
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.ir.program import normal_graph as G, normal_graph as F, normal_graph as T, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.structured import PythonProtocolKind
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256


def _strict_window_case():
    template = G.SourceTemplateKey(0,P.ProductionPurpose.STATEMENT)
    region_id, step_id = P.ProductionRegionID, P.ProductionStepID
    loop_region, body_region, exhausted, otherwise = map(region_id,('loop_region','body','exhausted','otherwise'))

    def gp(owner, kind, ordinal=0):
        return encode(T.GlobalPort('unit',G.Port(template,owner,kind,ordinal,
            C.FALLTHROUGH if kind in (G.PortKind.RESULT,G.PortKind.EXIT,G.PortKind.CALL_RESULT) else None)))

    def result(step, ordinal=0):
        return gp(step_id(step),G.PortKind.RESULT,ordinal)

    def origin(step, **fields):
        return dict(kind='original_f4',unit='unit',template=encode(template),step=step_id(step).wire,**fields)

    source = SourceRef('/source/original.py','0'*64,'loop',1,0,'pycode-v2:'+'1'*64,
        origins=({'kind':'source_identity','unit':'unit','occurrence':0},))
    body = ArrayRegion('leaf',source,('state','fixed','also'),(
        Expr('carry','input',attrs={'name':'state'}), Expr('x','input',attrs={'name':'fixed'}),
        Expr('y','input',attrs={'name':'also'}), Expr('inv','mul',('x','y')),
        Expr('next','add',('carry','inv'))),'next')
    condition = P.IterationCondition(P.ProductionRegionCall(region_id('advance'),
        (P.ProductionParameter(loop_region,0),P.ProductionParameter(loop_region,1))),exhausted,1)
    header_instruction = P.LoopControl(condition,body_region,otherwise,1)
    header = gp(step_id('header'),G.PortKind.ENTER)
    selected = dict(entry=gp(step_id('call'),G.PortKind.ENTER),callee=result('callee'),result=result('call'),
        positional=[result('forward_state'),result('forward_fixed'),result('forward_also')],keywords=[],
        positional_pack=None,keyword_pack=None,world_before=gp(body_region,G.PortKind.PARAMETER,1),
        world_after=result('call',1),query=0,fact={'locator':'#/calls/0'})
    carry_namespace = encode(F.Namespace(F.NamespaceKind.LEXICAL,0,'seed'))
    fixed_namespace = encode(F.Namespace(F.NamespaceKind.LEXICAL,0,'input'))
    origins = [origin('header',role='single_shared_invariant_loop',body_region=body_region.wire,
        iteration_condition=encode(condition),lexical_carry_binding=carry_namespace),
        origin('call',role='invariant_array_leaf_call',leaf_region_id=body.id,
            call_callee=selected['callee'],call_result=selected['result'],call_inputs=selected['positional'],
            outer_parameter_indices=[0,1,1],arguments=[selected['callee'],*selected['positional'],selected['world_before']],
            results=[selected['result'],selected['world_after']]),
        origin('store',role='lexical_carry_write',binding=carry_namespace,
            leaf_formal_index=0,outer_parameter_index=0,binding_port=7),
        *(origin(name,binding_port=index) for index,name in enumerate(('read_state','read_fixed','read_also'))),
        *(origin(name) for name in ('forward_state','forward_fixed','forward_also'))]
    proof = dict(body=asdict(body),source=asdict(source),carry_parameter='state',parameters=['seed','input','count'],
        parameter_map={'state':'seed','fixed':'input','also':'input'},origins=origins)
    interface = dict(carry_parameter='state',readonly_parameters=['fixed','also'],original_f4_proof=proof)
    operations = []

    def operation(name, instruction, region, values=(), outputs=(), membership='BODY'):
        sid = step_id(name)
        operations.append(dict(index=len(operations),entry=gp(sid,G.PortKind.ENTER),region=encode(region),
            instruction=encode(instruction,record_tags=P.PRODUCTION_RECORD_TAGS),
            inputs=[{'port':gp(sid,G.PortKind.ARGUMENT,index),'value':value} for index,value in enumerate(values)],
            outputs=[{'port':value} for value in outputs],membership=[membership]))

    operation('header',header_instruction,loop_region,membership='CONDITION_OR_EXIT')
    for index,name in enumerate(('read_state','read_fixed','read_also')):
        operation(name,P.AccessBinding(P.BindingActionKind.READ,index),body_region,outputs=[result(name)])
    for name in ('forward_state','forward_fixed','forward_also'):
        operation(name,P.EvaluateChild(P.SlotRef(0,'FIELD',0),P.ProductionPurpose.VALUE),body_region,outputs=[result(name)])
    operation('call',P.ApplyLanguageAction(P.LanguageActionKind.CALL),body_region,
        [selected['callee'],*selected['positional'],selected['world_before']],
        [selected['result'],selected['world_after']])
    operation('store',P.AccessBinding(P.BindingActionKind.WRITE,7),body_region,[selected['result']])
    operation('stop',P.ApplyPythonProtocol(PythonProtocolKind.MATCH_STOP_ITERATION,()),exhausted,
        [gp(exhausted,G.PortKind.PARAMETER,0),gp(exhausted,G.PortKind.PARAMETER,1)],
        [result('stop')],membership='CONDITION_OR_EXIT')
    operation('exhausted_done',P.TransferCompletion(C.FALLTHROUGH),exhausted,
        [result('stop')],membership='CONDITION_OR_EXIT')
    operation('otherwise_done',P.TransferCompletion(C.FALLTHROUGH),otherwise,
        [gp(otherwise,G.PortKind.PARAMETER,1)],membership='CONDITION_OR_EXIT')
    window = dict(schema='scar.source-motion-window',schema_version=1,unit='unit',source_version={
        'path':source.path,'sha256':source.sha256},loop_header=header,
        function=dict(template=encode(template),body=encode(T.Body('unit',template,region_id('function')))),
        loop=dict(header=header,body=encode(T.Body('unit',template,body_region)),
            body_entry=gp(body_region,G.PortKind.ENTER),condition=encode(condition,record_tags=P.PRODUCTION_RECORD_TAGS),
            header_instruction=encode(header_instruction,record_tags=P.PRODUCTION_RECORD_TAGS),otherwise=encode(otherwise),
            control_backedges=[encode(G.Edge(G.Port(template,body_region,G.PortKind.EXIT,completion=C.FALLTHROUGH),
                G.Port(template,step_id('header'),G.PortKind.ENTER),G.EdgeKind.CONTROL,transition=G.Transition.BACKEDGE))]),
        selected_call=selected,operations=operations,call_sites=[selected],accesses=[],frontier=[],
        request=dict(max_states=20000,max_stack=128),
        reference_transports=[edge for source_name,target_name in zip(('read_state','read_fixed','read_also'),
            ('forward_state','forward_fixed','forward_also')) for edge in (
                dict(source=result(source_name),target=gp(step_id(target_name),G.PortKind.CALL_RESULT),kind='DATA',
                    transition='RETURN',site=encode(G.CallSite(template,step_id(target_name))),condition=None,when=None),
                dict(source=gp(step_id(target_name),G.PortKind.CALL_RESULT),target=result(target_name),kind='DATA',
                    transition='DIRECT',site=None,condition=None,when=None))],
        reference_transport_scope='SELECTED_CALL_VALUES_BACKWARD_ORIGINAL_DATA_SLICE_NOT_PRIMITIVE_OR_CONTENT_EQUIVALENCE',
        reference_transport_slice=dict(complete=True,seeds=[selected['callee'],*selected['positional']],
            visited_ports=10,edges=6,max_states=20000,balanced_path_selection_required=True,frontier=[]),
        lexical_bindings=[dict(namespace=carry_namespace,reads=[{'value':result('read_state')}],
            writes=[{'value':selected['result']}]),dict(namespace=fixed_namespace,
            reads=[{'value':result(name)} for name in ('read_fixed','read_also')],writes=[])],
        coverage=dict(window_complete=True,body_complete=True,
            control_scope='ORIGINAL_NORMAL_CONTROL_BALANCED_RECIPE_CALL_RETURN'),
        must_execute=dict(status='CONDITIONAL',fact='ONE_PER_ENTERED_BODY',missing=[],
            normal_paths_avoid_selected=False,selected_repeat_reachable=False,avoiding_cycle_reachable=False))
    window['id']=semantic_sha256(window)
    return body,interface,window


def test_strict_window_alias_reads_exhaustion_abi_and_precise_unresolved_clobbers():
    body,interface,window = _strict_window_case()
    selected = window['selected_call']

    def identity(document):
        document['id']=semantic_sha256({key:value for key,value in document.items() if key!='id'})
        return document

    def derive(document):
        return derive_array_motion_legality(body,carry_parameter='state',loop_interface=interface,source_window=document)

    identity(window)
    report = derive(window)
    assert report['status']=='CONDITIONAL' and report['stable_formal_inputs']==['fixed','also']
    root = report['roots'][0]
    assert root['expression']=='inv' and root['read_parameters']==['also','fixed']
    assert root['materialized_exprs']==['inv'] and not root['materialized_value_provenance_is_repeat_input_load']
    assert root['alias_requirements']['read_read_aliasing_allowed']
    assert not root['alias_requirements']['pairwise_read_input_disjointness_required']
    assert not report['source_window_effects']['blockers']
    assert all(binding['reference_trace']['origins'][0]['context'] for binding in report['input_bindings']
        if binding['reference_trace'] is not None)
    assert any(row['role']=='CARRY_LEXICAL_REBIND_NOT_INPUT_MUTATION' for row in report['source_window_effects']['operations'])
    assert not report['source_f4_replayed'] and report['source_window_replay_required']
    assert not report['runtime_guards_discharged'] and not report['source_replacement_authorized']
    assert verify_array_motion_legality(report,body,carry_parameter='state',loop_interface=interface,source_window=window)==report
    forged = deepcopy(report); forged['roots'][0]['no_clobber']['obligations']=[]
    with pytest.raises(ValueError,match='source-window replay'):
        verify_array_motion_legality(forged,body,carry_parameter='state',loop_interface=interface,source_window=window)
    wrong_boundary = deepcopy(window)
    protocol = next(row for row in wrong_boundary['operations'] if row['instruction']['record_kind']=='ApplyPythonProtocol')
    protocol['inputs'][0]['value']=selected['result']
    assert derive(identity(wrong_boundary))['status']=='UNRESOLVED'  # protocol name alone is insufficient
    opaque = deepcopy(window)
    extra = deepcopy(next(row for row in opaque['operations'] if row['entry']==selected['entry']))
    extra['index']=len(opaque['operations'])
    extra['entry']['port']['owner']=encode(P.ProductionStepID('opaque')); opaque['operations'].append(extra)
    blocked = derive(identity(opaque))
    assert blocked['status']=='UNRESOLVED'
    assert any(row['reason']=='OPAQUE_CALL_OR_LANGUAGE_PROTOCOL_IN_MOVEMENT_WINDOW'
        for row in blocked['source_window_effects']['blockers'])
    incomplete = deepcopy(window); incomplete['coverage']['window_complete']=False
    assert derive(identity(incomplete))['status']=='UNRESOLVED'
    incomplete_data = deepcopy(window)
    incomplete_data['reference_transport_slice'].update(complete=False,frontier=[{'reason':'REFERENCE_TRANSPORT_SLICE_BUDGET_EXHAUSTED'}])
    partial = derive(identity(incomplete_data))['roots'][0]
    assert partial['data_available']['status']=='UNRESOLVED'
    assert all(partial[field]['status']=='CONDITIONAL' for field in ('no_clobber','execution','effects'))
    bounded = deepcopy(window); bounded['request']['max_states']=1
    blocked = derive(identity(bounded))
    assert any(row['reason']=='REFERENCE_STATE_BUDGET_EXHAUSTED'
        for binding in blocked['input_bindings'] if binding['reference_trace'] is not None
        for row in binding['reference_trace']['frontier'])
    spliced = deepcopy(window)
    edge = spliced['reference_transports'][2]
    original_read = edge['source']; shared_result = deepcopy(original_read)
    shared_result['port']['owner']=encode(P.ProductionStepID('shared_return'))
    shared_parameter = deepcopy(original_read)
    shared_parameter['port'].update(owner=encode(P.ProductionRegionID('shared')),kind='PARAMETER',completion=None)
    edge['source']=shared_result
    spliced['reference_transports'].extend([
        dict(source=shared_parameter,target=shared_result,kind='DATA',transition='DIRECT',site=None,condition=None,when=None),
        dict(source=original_read,target=shared_parameter,kind='DATA',transition='CALL',condition=None,when=None,
            site=encode(G.CallSite(G.SourceTemplateKey(0,P.ProductionPurpose.STATEMENT),P.ProductionStepID('another_caller'))))])
    spliced['reference_transport_slice']['edges']=len(spliced['reference_transports'])
    assert derive(identity(spliced))['roots'][0]['data_available']['status']=='UNRESOLVED'
