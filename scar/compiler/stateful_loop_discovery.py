"""Locate a stateful step through one original, caller-owned CALL interface.

This is source discovery, not a loop/effect proof. The retained result is
consumed by stateful_loop_frontend, which replays the step and shared loop.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

from scar.analysis.program_view import ProgramView
from scar.analysis.source_call_view import SourceFunctionCallProvider,function_interface_fact,source_call_site
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import ReferenceSlicer,source_port
from scar.compiler.caller_pipeline import _abi,_reference,_signature
from scar.compiler.model import SourceRef
from scar.compiler.stateful_frontend import discover_stateful_regions
from scar.compiler.stateful_model import (StatefulRegion,SCHEMA,document,stateful_from_dict,
    stateful_digest,validate_stateful_region)
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.record_codec import encode


@dataclass(frozen=True)
class DiscoveredStatefulLoopStep:
    step: StatefulRegion
    target_source: SourceRef
    evidence: dict


def _source_key(source):
    return source.path,source.sha256,source.code_key,source.qualname,source.line,source.column


def derive_stateful_loop_step(view,provider,*,caller_unit,function_ordinal,interface_ordinal):
    """Find one step by original body/CodeID, without selecting by its name."""
    if type(view)is not ProgramView or type(provider)is not SourceFunctionCallProvider or provider.view is not view:
        raise TypeError('one shared ProgramView and its original source-call provider required')
    if caller_unit not in provider.available_units:
        raise ValueError('selected CALL unit is outside its original interface store')
    source=view.unit(caller_unit)
    if type(function_ordinal)is not int or not 0<=function_ordinal<len(source.facts.functions):
        raise ValueError('original caller function ordinal required')
    function=source.facts.functions[function_ordinal];_signature(function)
    graph=provider.store.read_unit(caller_unit)
    if type(interface_ordinal)is not int or not 0<=interface_ordinal<len(graph.interfaces):
        raise ValueError('original source CALL interface ordinal required')
    interface=graph.interfaces[interface_ordinal];link=interface.link
    queries=[row for row in graph.queries if row.query==interface.query]
    if len(queries)!=1 or function.body_scope.owner not in queries[0].owners:
        raise ValueError('selected source CALL is not owned by the original caller function')
    if link.body is None or link.source_callable is None:
        raise ValueError('selected source CALL has no retained source body/callable')
    operation,_,_=source_call_site(view,link)
    original=source.records[operation.node]
    context=NamespacePlanView(caller_unit,source.plans[link.result.port.template])
    if context.owner(original.region).owner!=function.body_scope.owner:
        raise ValueError('selected CALL operation and original caller scope differ')
    target=view.unit(link.body.unit)
    targets=[(ordinal,row)for ordinal,row in enumerate(target.facts.functions)
        if row.template==link.body.template and row.body==link.body.region
        and source_port(target.facts,row.value)==link.source_callable]
    if len(targets)!=1:
        raise ValueError('selected CALL body/callable lacks one original function definition')
    target_ordinal,target_function=targets[0];_signature(target_function)
    # Reuse the existing explicit argument/environment/world/return binder;
    # this adapter does not add receiver/default/pack profile alternatives.
    _abi(source,function,interface,target_function)
    identity=_reference(view,target.unit,target_function,(),{})
    regions,rejections=discover_stateful_regions(view,source=identity.path)
    matches=[]
    for region in regions:
        origins=[row for row in region.source.origins if row.get('kind')=='source_identity']
        if (_source_key(region.source)==_source_key(identity) and len(origins)==1
                and origins[0].get('unit')==target.unit
                and origins[0].get('occurrence')==target_function.template.occurrence
                and region.parameter_names==target_function.parameter_names):
            matches.append(region)
    if len(matches)!=1:
        rejected=[row for row in rejections if row.get('unit')==target.unit
            and row.get('function')==target_ordinal]
        raise ValueError('selected source CALL needs exactly one closed stateful step; '
            f'matches={len(matches)}, original_target_rejections={rejected!r}')
    step=validate_stateful_region(matches[0])
    slicer=ReferenceSlicer(NS(fact_store=view.catalog.fact_store))
    provider.attach(caller_unit,queries=(interface.query,))
    frame=provider.frames[caller_unit,interface_ordinal]
    evidence=dict(schema='scar.stateful-loop-step-source-selection',schema_version=1,
        catalog=artifact_reference(view.catalog.path/'result.json',view.catalog.report['schema']),
        caller_unit=caller_unit,function_ordinal=function_ordinal,interface_ordinal=interface_ordinal,
        query=interface.query,original_call=encode(interface.call),
        interface=encode(function_interface_fact(provider.store,caller_unit,interface_ordinal)),
        caller_function=encode(slicer.fact(source.facts,'functions',function_ordinal)),
        callee_function=encode(slicer.fact(target.facts,'functions',target_ordinal)),
        original_operation=encode(operation),source_callable=encode(link.source_callable),
        shared_body=encode(link.body),frame=encode(frame),conditions=list(link.conditions),
        step_digest=stateful_digest(step),profile_guards=list(step.guards),
        discovery_scope='EXACT_SELECTED_CALLEE_SOURCE_FROM_ORIGINAL_BODY_AND_COMPLETE_CODE_ID',
        target_source_executed=False,runtime_dispatch_proven=False,
        stateful_loop_proven=False,transformation_authorized=False)
    provider.store.validate();source.check();target.check()
    return DiscoveredStatefulLoopStep(step,identity,evidence)


def publish_discovered_step(discovery,path):
    """Retain the full ordinary step schema; no discovery flag authorizes it."""
    if type(discovery)is not DiscoveredStatefulLoopStep:
        raise TypeError('source-selected stateful step derivation required')
    step=validate_stateful_region(discovery.step)
    if (_source_key(step.source)!=_source_key(discovery.target_source)
            or discovery.evidence.get('step_digest')!=stateful_digest(step)):
        raise ValueError('source-selected step changed before publication')
    import json
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as stream:
        stream.write(json.dumps(document(step),sort_keys=True,indent=2,allow_nan=False)+'\n')
    reference=artifact_reference(path,SCHEMA)
    if stateful_from_dict(read_artifact(reference))!=step:
        raise ValueError('retained source-selected stateful step differs on typed readback')
    return reference


__all__=['DiscoveredStatefulLoopStep','derive_stateful_loop_step','publish_discovered_step']
