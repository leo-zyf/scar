"""Bind conditional composite effects to original, shared F4 call interfaces.

This query authenticates source coordinates and reuses the original signature
binder and namespace resolver.  The enclosing composite reader still owns full
F4 array/pipeline rediscovery and its compile-only CodeID admission.  Neither
this query nor a serialized effect summary establishes live runtime dispatch,
immutable array contents, exception behavior or source replacement authority.
"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_field_interfaces import _InterfaceOnly
from scar.analysis.source_function_connections import Invocation, connect
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import _origin
from scar.compiler.pipeline_frontend import SourceFunctionComponents
from scar.ir.program import normal_graph as T, source_productions as P
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256


SCHEMA='scar.composite-call-effect-binding'


def _source_key(source):
    return source.path,source.sha256,source.code_key


def _identity(source):
    rows=[row for row in source.origins if row.get('kind')=='source_identity']
    if len(rows)!=1 or rows[0].get('code_key')!=source.code_key:
        raise ValueError('one original source identity with the exact CodeID required')
    return rows[0]


def _source_label(source):
    return dict(path=source.path,sha256=source.sha256,code_key=source.code_key,
        source_ref_sha256=semantic_sha256(asdict(source)))


def derive_composite_call_effect_binding(view,proof,effects):
    """Return source-bound conditional interfaces, without duplicating bodies."""
    from scar.analysis.array_call_effects import verify_composite_call_effects
    from scar.compiler.loop_admission import _motion_origin_entry
    from scar.compiler.pipeline_loop_frontend import CompositeInvariantLoopProof

    if type(proof) is not CompositeInvariantLoopProof:
        raise TypeError('the separate original composite proof is required')
    verify_composite_call_effects(effects,proof)
    routes=SourceFunctionComponents(view);slicer=routes.resolver.slicer
    resolver=BodyInterfaceResolver(view);functions={};callees=[];frontier=[]

    def function(source):
        key=_source_key(source);identity=_identity(source)
        if key in functions:
            original,record,_=functions[key]
            if original.facts.unit!=identity['unit'] or record.template.occurrence!=identity['occurrence']:
                raise ValueError('one CodeID cannot acquire a different original F4 identity')
            return functions[key]
        original=view.unit(identity['unit']);facts=original.facts
        model=view.catalog.fact_store.model_units[identity['unit']]
        if (str(Path(model['path']).absolute())!=str(Path(source.path).absolute())
                or sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256):
            raise ValueError('callee source identity differs from original source bytes/unit')
        matches=[(i,row) for i,row in enumerate(facts.functions)
            if row.template.occurrence==identity['occurrence']]
        if len(matches)!=1:raise ValueError('callee occurrence is not one original function')
        ordinal,record=matches[0]
        if record.body_scope is None:raise ValueError('original function body scope required')
        body=T.Body(facts.unit,record.template,record.body)
        resolver.prepare(body)
        index=len(callees)
        callees.append(dict(index=index,source=_source_label(source),
            identity=identity,definition=encode(slicer.fact(facts,'functions',ordinal)),
            source_function=encode(source_port(facts,record.value)),body=encode(body),
            formals=[encode(source_port(facts,value)) for value in record.formals],
            normal_return=None if record.returns is None else encode(source_port(facts,record.returns)),
            return_world=None if record.return_world is None else encode(source_port(facts,record.return_world)),
            code_identity_basis='ENCLOSING_FULL_COMPOSITE_F4_REDISCOVERY_OF_RETAINED_F0_CODE_ID',
            body_shared=True,body_copied=False))
        value=original,record,index
        functions[key]=value
        return value

    function(proof.source);function(proof.root_callee_source)
    calls=[];call_keys={};namespace_routes=[];namespace_indexes={}

    def namespace_route(owner,name,selected):
        if selected is None:return None
        key=owner.facts.unit,name,selected.unit,selected.function.template
        if key not in namespace_indexes:
            namespace_indexes[key]=len(namespace_routes)
            namespace_routes.append(selected.evidence)
        return namespace_indexes[key]

    def bind_call(owner_source,target_source,name,origin,*,pipeline_id,call_id,
                  inputs=None,output=None,parameters=None):
        owner,owner_function,owner_index=function(owner_source)
        target,target_function,target_index=function(target_source)
        entry=_motion_origin_entry(origin)
        if entry.unit!=owner.facts.unit:
            raise ValueError('CALL origin belongs to a different source unit')
        if (entry.unit,entry.port.template) not in resolver.prepared:
            raise ValueError('CALL template is outside its original source function closure')
        context=resolver.plan(entry.unit,entry.port.template)
        steps=[row for row in context.plan.steps if row.id==entry.port.owner]
        if len(steps)!=1:raise ValueError('CALL origin is not one original step')
        step=steps[0]
        if (type(step.instruction) is not P.ApplyLanguageAction
                or step.instruction.action is not P.LanguageActionKind.CALL
                or context.owner(step.region).owner!=owner_function.body_scope.owner):
            raise ValueError('original CALL must be inside its claimed function code scope')
        operands=[encode(context.value(value)) for value in step.arguments]
        expected=_origin(context,step.region,step)
        if any(origin.get(key)!=value for key,value in expected.items()):
            raise ValueError('CALL original production/source/result coordinates changed')
        if origin.get('arguments')!=operands:
            raise ValueError('CALL origin operands differ from the original production')
        records=[(i,row) for i,row in enumerate(owner.facts.calls)
            if source_port(owner.facts,row.result)==context.result(step)]
        if len(records)!=1:raise ValueError('one exact original CALL fact required')
        ordinal,record=records[0]
        port=lambda value:encode(source_port(owner.facts,value))
        call_fact=dict(result=port(record.result),callee=port(record.callee),
            positional=[port(value) for value in record.positional],
            keywords=[dict(name=key,value=port(value)) for key,value in record.keywords])
        if 'call_fact' in origin and origin['call_fact']!=call_fact:
            raise ValueError('pipeline CALL fact differs from original F4 operands')
        if 'call_callee' in origin and (origin['call_callee']!=call_fact['callee']
                or origin['call_inputs']!=call_fact['positional']
                or origin['call_result']!=call_fact['result']):
            raise ValueError('outer composite CALL differs from original F4 operands')
        expected_operands=[port(record.callee),*call_fact['positional'],
            *(row['value'] for row in call_fact['keywords']),port(record.world)]
        if (operands!=expected_operands or context.worlds(step)!=(
                source_port(owner.facts,record.world),source_port(owner.facts,record.result_world))):
            raise ValueError('original CALL operand/world ABI differs')
        if parameters is not None and tuple(parameters)!=target_function.parameter_names:
            raise ValueError('typed leaf formal order differs from its original source function')
        selected=routes.targets(owner.facts.unit).get(name)
        target_matches=(selected is not None and selected.unit==target.facts.unit
            and selected.function==target_function)
        if not target_matches:
            frontier.append(dict(reason='ORIGINAL_FUNCTION_GLOBAL_TARGET_ROUTE_OPEN',
                pipeline_id=pipeline_id,call_id=call_id,owner=owner_index,lookup_name=name))
        mini=_InterfaceOnly(slicer,target.facts,target_function)
        offset=slicer.store.offsets[owner.facts.unit]
        invocation=Invocation(offset+record.callee,tuple(offset+value for value in record.positional),
            tuple((key,offset+value) for key,value in record.keywords),offset+record.result,
            offset+record.world,offset+record.result_world,
            None if record.positional_pack is None else offset+record.positional_pack,
            None if record.keyword_pack is None else offset+record.keyword_pack)
        connection=connect(mini,mini.offsets[target.facts.unit]+target_function.value,invocation,
            conditions=('ORIGINAL_COMPOSITE_NAMESPACE_ROUTE_AND_CURRENT_FUNCTION_GLOBAL_CODE_IDS_MATCH',))
        if (connection is None or connection.activation is not T.FunctionActivation.NORMAL
                or connection.binding is not T.ArgumentBinding.DIRECT):
            frontier.append(dict(reason='ORIGINAL_CALL_SIGNATURE_NOT_DIRECT',
                pipeline_id=pipeline_id,call_id=call_id))
        row=dict(index=len(calls),pipeline_id=pipeline_id,call_id=call_id,
            owner=owner_index,callee=target_index,lookup_name=name,entry=encode(entry),
            fact=encode(slicer.fact(owner.facts,'calls',ordinal)),
            source_origin=dict(unit=origin['unit'],template=origin['template'],step=origin['step']),
            callee_port=call_fact['callee'],positional=call_fact['positional'],keywords=call_fact['keywords'],
            result=call_fact['result'],world_before=port(record.world),world_after=port(record.result_world),
            positional_pack=None if record.positional_pack is None else port(record.positional_pack),
            keyword_pack=None if record.keyword_pack is None else port(record.keyword_pack),
            connection=None if connection is None else encode(connection),
            logical_inputs=None if inputs is None else list(inputs),logical_output=output,
            conditional_target=target_matches,
            namespace_route=namespace_route(owner,name,selected),runtime_dispatch_proven=False)
        key=(_source_key(owner_source),_source_key(target_source),semantic_sha256(row['source_origin']))
        if key in call_keys:raise ValueError('one source CALL appeared twice in shared composition definitions')
        call_keys[key]=row['index'];calls.append(row)
        return row

    outer=[row for row in proof.origins if row.get('kind')=='original_f4'
        and row.get('role')=='invariant_array_leaf_call']
    if len(outer)!=1:raise ValueError('one outer original composite CALL required')
    outer_call=bind_call(proof.source,proof.root_callee_source,proof.root_global_name,outer[0],
        pipeline_id=None,call_id='OUTER_LOOP_BODY_CALL',parameters=proof.body.parameters)
    for composition in proof.compositions:
        pipeline=composition.pipeline
        for index,call in enumerate(pipeline.calls):
            origins=[row for row in call.origins if row.get('kind')=='original_f4' and row.get('call_index')==index]
            if len(origins)!=1:raise ValueError('one original static CALL occurrence required')
            bind_call(pipeline.source,call.callee.source,call.callee.lookup_path[0],origins[0],
                pipeline_id=pipeline.id,call_id=call.id,inputs=call.inputs,output=call.output,
                parameters=call.plan.region.parameters)
    guards=[];covered=set()
    for index,guard in enumerate(proof.nested_callee_guards):
        current=proof.source;path=[]
        for name in guard.global_path:
            original,_,owner_index=function(current)
            selected=routes.targets(original.facts.unit).get(name)
            if selected is None:
                frontier.append(dict(reason='FUNCTION_GLOBAL_GUARD_PATH_OPEN',guard=index,path=list(guard.global_path)))
                break
            choices=[key for key,(unit,record,_) in functions.items()
                if unit.facts.unit==selected.unit and record==selected.function]
            # Every path hop is an existing source function referenced by the proof.
            if not choices:
                candidates=[guard.owner_source,guard.target_source,
                    *(row.pipeline.source for row in proof.compositions),
                    *(call.callee.source for row in proof.compositions for call in row.pipeline.calls)]
                for candidate in candidates:function(candidate)
                choices=[key for key,(unit,record,_) in functions.items()
                    if unit.facts.unit==selected.unit and record==selected.function]
            if len(choices)!=1:raise ValueError('guard path has no unique retained source function')
            target_key=choices[0]
            target_source=next(candidate for candidate in (proof.source,proof.root_callee_source,
                guard.owner_source,guard.target_source,
                *(row.pipeline.source for row in proof.compositions),
                *(call.callee.source for row in proof.compositions for call in row.pipeline.calls))
                if _source_key(candidate)==target_key)
            path.append(dict(owner=owner_index,lookup_name=name,callee=function(target_source)[2],
                namespace_route=namespace_route(original,name,selected)))
            if len(path)==len(guard.global_path) and (_source_key(current)!=_source_key(guard.owner_source)
                    or target_key!=_source_key(guard.target_source) or name!=guard.lookup_name):
                raise ValueError('function-global path differs from its exact owner/target source identity')
            current=target_source
        if not guard.global_path or guard.lookup_semantics!='FUNCTION_GLOBALS_AT_EACH_HOP':
            raise ValueError('explicit original function-global lookup path required')
        occurrences=[]
        for origin in guard.origins:
            if origin.get('kind')!='original_f4':continue
            coordinate=dict(unit=origin['unit'],template=origin['template'],step=origin['step'])
            key=(_source_key(guard.owner_source),_source_key(guard.target_source),semantic_sha256(coordinate))
            if key not in call_keys:raise ValueError('global guard origin is not one bound source CALL')
            occurrences.append(call_keys[key]);covered.add(call_keys[key])
        if not occurrences:raise ValueError('global guard has no original CALL provenance')
        guards.append(dict(index=index,owner_source=_source_label(guard.owner_source),
            target_source=_source_label(guard.target_source),lookup_name=guard.lookup_name,
            global_path=list(guard.global_path),lookup_semantics=guard.lookup_semantics,
            source_call_indexes=occurrences,path=path,
            status='CONDITIONAL' if len(path)==len(guard.global_path) else 'UNRESOLVED'))
    if covered!=set(range(len(calls))):
        raise ValueError('every original source CALL needs a recorded function-global guard')
    for original,_,_ in functions.values():original.check()
    slicer.store.validate_parents()
    report=dict(schema=SCHEMA,schema_version=1,
        status='CONDITIONAL' if not frontier else 'UNRESOLVED',
        proof_sha256=semantic_sha256(asdict(proof)),body_sha256=semantic_sha256(asdict(proof.body)),
        callee_effects_sha256=semantic_sha256(effects),root_source=asdict(proof.root_callee_source),
        source=asdict(proof.source),outer_call=outer_call,callees=callees,calls=calls,guards=guards,
        namespace_routes=namespace_routes,
        source_refs=[encode(original.material.result_artifact) for original in
            {original.facts.unit:original for original,_,_ in functions.values()}.values()],
        frontier=frontier,source_f4_authority='ENCLOSING_FULL_COMPOSITE_F4_REDISCOVERY_REQUIRED',
        obligations=['ACTUAL_FUNCTION_GLOBAL_CODE_IDS_MATCH_AT_EVERY_RECORDED_HOP',
            'EXACT_ARRAY_PROFILE_AND_STABLE_REFERENT_CONTENTS',
            'EXCEPTION_FAILURE_AND_EXTERNAL_EFFECT_CONDITIONS_REMAIN_OPEN'],
        source_replacement_authorized=False,runtime_guards_discharged=False,
        effects_unconditionally_closed=False,source_bodies_copied=False,iterations_unrolled=False)
    report['id']='composite-call-effect-binding:'+semantic_sha256(report)
    return report


def verify_composite_call_effect_binding(report,view,proof,effects):
    expected=derive_composite_call_effect_binding(view,proof,effects)
    if semantic_sha256(report)!=semantic_sha256(expected):
        raise ValueError('composite effect binding differs from original F4 interfaces')
    return expected


__all__=['derive_composite_call_effect_binding','verify_composite_call_effect_binding']
