"""Connect observed original construction to demanded controller state ports."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_controller_effects import _Observed, _original_call, _shared_body
from scar.analysis.source_native_controller import SCHEMA as CONTROLLER, check as check_controller
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_links as F, normal_targets as T
from scar.ir.program.call_edges import FrameCompletion
from scar.ir.program.python_call_targets import PythonTargetKind
from scar.ir.program.source_address_paths import AddressRootKind, FieldSelector
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.source-controller-initialization'


def _only(rows, detail):
    result = tuple(rows)
    if len(result) != 1:
        raise ValueError(detail)
    return result[0]


def _function(view, body):
    unit = view.unit(body.unit)
    fn = _only((row for row in unit.facts.functions if (row.template,row.body)==(body.template,body.region)),
               'one original function body required at the initialization boundary')
    return unit,fn


def _resolver(view, unit, scope):
    return AddressResolver(view.catalog.fact_store,unit.facts,scope)


def _field_path(path):
    if (path.frontier is None and path.root_kind is AddressRootKind.FUNCTION_PARAMETER
            and all(type(part) is FieldSelector for part in path.selectors)):
        return path.root,tuple(part.name for part in path.selectors)
    return None


def _walk(program):
    for row in program:
        yield row
        for name in ('condition','body','then','otherwise'):
            yield from _walk(row.get(name,()))


def _demands(controller, update):
    operations = tuple(_walk(controller['program']))
    native = _only((row for row in operations if row['kind']=='CALL' and row['role']=='NATIVE_UPDATE'),
                   'one original controller update site required')
    route = _only((row for row in native['formal_routes'] if row['binding']['role']=='ATTRIBUTE_RECEIVER'),
                  'one original controller receiver route required')
    receiver_path = tuple(route['binding']['source_path'])
    receiver = _only((p for p in controller['parameters'] if p['name']==receiver_path[0]),
                     'controller receiver path requires its original formal')
    demands = {}

    def add(path, kind, use):
        if len(path) > 1 and path[0] == receiver['name']:
            identity = use.get('input','field:'+'.'.join(path))
            row = demands.setdefault(tuple(path),dict(id=identity,controller_path=list(path),kind=kind,uses=[]))
            row['uses'].append(use)

    for port in (*update['array_inputs'],*update['scalar_inputs']):
        if port['parent_path'][0] != route['parameter']:
            continue
        path = (*receiver_path,*port['parent_path'][1:])
        use = dict(role='NUMERICAL_INPUT',input=port['id'],original_path=port['input_path'])
        if port['kind'] == 'array':
            use['later_geometry'] = {key:port[key] for key in ('shape','strides','offset','dtype','device')}
        add(path,port['kind'],use)
    for guard in update['entry_guards']['guards']:
        if guard['path'][0] == route['parameter']:
            add((*receiver_path,*guard['path'][1:]),'bool',dict(role='UPDATE_BRANCH_GUARD',original=guard))

    producers = {row['id']:row['expression'] for row in operations if row['kind']=='EVAL'}
    callee_values = {row['callee'].get('id') for row in operations if row['kind']=='CALL'}

    def path(value, seen=()):
        identity = value.get('id')
        if identity in seen or identity not in producers:
            return None
        operation = producers[identity]
        if operation['kind']=='SLOT' and operation['slot']==receiver['slot']:
            return (receiver['name'],)
        if operation['kind']=='FIELD':
            prefix = path(operation['receiver'],(*seen,identity))
            if prefix is not None:
                return (*prefix,operation['field'])
        return None

    for row in operations:
        if row['kind']=='EVAL' and row['expression']['kind']=='FIELD' and row['id'] not in callee_values:
            p = path(dict(id=row['id']))
            if p is not None:
                add(p,'field',dict(role='CONTROLLER_READ',origin=row['origin']))
        if row['kind']=='STORE_FIELD':
            p = path(row['receiver'])
            if p is not None:
                add((*p,row['field']),'field',dict(role='CONTROLLER_WRITE',origin=row['origin']))
    return receiver,demands


def _frame_formal(frame, name):
    return _only((row for row in frame.parameters if row.name == name),
                 'exact observed frame formal required')


def _bind_formals(unit, fn, frame, caller_facts, call, *, receiver=None):
    """Preserve admitted direct arguments and declaration defaults separately."""
    if call.positional_pack is not None or call.keyword_pack is not None or not frame.parameters_complete:
        raise ValueError('initialization needs original direct arguments and complete formal names')
    args = tuple(source_port(caller_facts,i) for i in call.positional)
    keywords = {name:source_port(caller_facts,i) for name,i in call.keywords}
    if len(keywords) != len(call.keywords):
        raise ValueError('initialization original CALL has conflicting keywords')
    records = []
    asserted = 0
    for i,(name,ordinal,parameter) in enumerate(zip(
            fn.parameter_names,fn.formals[1:-1],fn.environment.parameters,strict=True)):
        kind = parameter.kind.value
        if i == 0 and receiver is not None:
            role,actual = 'OBSERVED_IMPLICIT_RECEIVER',receiver
        elif kind in ('POSITIONAL_ONLY','POSITIONAL_OR_KEYWORD') and asserted < len(args):
            role,actual = 'POSITIONAL',args[asserted]
            asserted += 1
        elif kind in ('POSITIONAL_OR_KEYWORD','KEYWORD_ONLY') and name in keywords:
            role,actual = 'KEYWORD',keywords.pop(name)
        elif parameter.default_source is not None and kind in ('POSITIONAL_ONLY','POSITIONAL_OR_KEYWORD','KEYWORD_ONLY'):
            role,actual = 'DECLARED_DEFAULT',None
        else:
            raise ValueError('initializer call formal route remains outside the direct profile')
        records.append(dict(name=name,formal=source_port(unit.facts,ordinal),role=role,actual=actual,
            parameter_kind=parameter.kind,declared_default=parameter.default_source,
            observed_reference=_frame_formal(frame,name).ref,default_value_verified=False))
    if asserted != len(args) or keywords:
        raise ValueError('initialization original actual denominator differs from formal routes')
    return records


def _call_at(unit, site):
    return _only(((i,row) for i,row in enumerate(unit.facts.calls) if source_port(unit.facts,row.result)==site),
                 'one original CALL fact required at the initialization boundary')


def _child_writes(view, observed, unit, fn, init_frame, init_receiver, profiles):
    slicer = ReferenceSlicer(view.catalog)
    addresses = _resolver(view,unit,fn.body_scope.owner)
    rows = []
    field_writes = defaultdict(list)
    item_writes = defaultdict(list)
    for ordinal,call in enumerate(unit.facts.calls):
        site = source_port(unit.facts,call.result)
        for evidence,candidates,frame in observed.at(site,caller=init_frame.invocation):
            original = _original_call(unit,slicer,call,ordinal)
            row = dict(original_call=original,observation=evidence,body=None,arguments=[],
                       frontier='EXTERNAL_OR_UNADMITTED_INITIALIZATION_CALL')
            if len(candidates) == 1 and evidence['normal_completion_observed'] and frame.parameters_complete:
                body, = candidates
                child_unit,child = _function(view,body)
                if body not in profiles:
                    profiles[body] = _shared_body(view,body)
                profile = profiles[body]
                receiver_ref = init_frame.parameters[0].ref
                implicit = (bool(frame.parameters) and frame.parameters[0].ref==receiver_ref
                    and any(p is not None and p[0]==init_receiver and bool(p[1])
                            for p in map(_field_path,addresses.paths(call.callee))))
                row.update(body=body,frontier='NESTED_INITIALIZATION_PROVIDER_EFFECTS_OPEN')
                try:
                    arguments = _bind_formals(child_unit,child,frame,unit.facts,call,
                                              receiver=init_receiver if implicit else None)
                except ValueError:
                    row['frontier'] = 'ORIGINAL_INITIALIZATION_CHILD_FORMAL_ROUTES_OPEN'
                else:
                    row['arguments'] = arguments
                    roots = {}
                    for argument in arguments:
                        actual = argument['actual']
                        if actual is None:
                            continue
                        _, local = slicer.local(actual)
                        alternatives = {_field_path(path) for path in addresses.paths(local)}
                        if len(alternatives)==1 and None not in alternatives:
                            roots[argument['formal']] = next(iter(alternatives))
                    for collection, action, output in ((profile['field_accesses'],'SET',field_writes),
                                                       (profile['item_accesses'],'WRITE',item_writes)):
                        for access in collection:
                            if access['kind'] != action:
                                continue
                            for path in access['receiver_paths']:
                                address = _field_path(path)
                                if address is None or address[0] not in roots:
                                    continue
                                root,selectors = roots[address[0]]
                                selectors = (*selectors,*address[1])
                                if action=='SET':
                                    selectors = (*selectors,access['selector'])
                                if root!=init_receiver or not selectors:
                                    continue
                                output[selectors].append(dict(body=body,child_call=site,attempt=evidence['attempt'],
                                    operation=access['operation'],source_fact=access['source_fact'],receiver_path=path,
                                    argument_route=next(arg for arg in arguments if arg['formal']==address[0])))
            rows.append(row)
    rows.sort(key=lambda row:row['observation']['attempt'].index)
    return rows,field_writes,item_writes


def _array_argument_receipts(update, directory):
    """Keep exact later argument observations in their original trace actor."""
    receipts = defaultdict(list)
    for call in update['ordered_calls']:
        if call['kind'] != 'ARRAY_COPY_CALL':
            continue
        for route in call['formal_to_root']:
            for row in call['observations']:
                if str(Path(row['entry']['path']).parent) != directory:
                    continue
                ordinal = route['formal_index']
                actual = row['formal_references'][ordinal]
                reference = actual['ref']
                if reference.startswith('weak:'):
                    receipts[route['root_id'],reference].append(dict(source_result=call['source_result'],
                        formal_index=ordinal, invocation=row['invocation'], entry=row['entry'], attempt=row['attempt']))
    return receipts


def _same_argument_objects(port_id, item_writers, observed_array_args):
    receipts = {}
    for writer in item_writers:
        if 'argument_route' not in writer or writer['receiver_path'].selectors:
            continue  # An object carrying an array field is a different observation.
        reference = writer['argument_route']['observed_reference']
        later = observed_array_args.get((port_id,reference))
        if later:
            attempt = writer['attempt']
            receipts[attempt,reference] = dict(reference=reference,startup_source=writer['child_call'],
                startup_attempt=attempt,later_observations=later,
                scope='OBSERVED_WEAK_OBJECT_IDENTITY_AT_ORIGINAL_ARRAY_ARGUMENTS')
    return list(receipts.values())


def _constructor(view, entry, controller, update, observed):
    receiver,demands = _demands(controller,update)
    receiver_formal = decode(T.GlobalPort,receiver['formal'])
    controller_body = decode(T.Body,controller['body'])
    ordinal,row = _only(((i,row) for i,row in enumerate(entry['interfaces'])
                         if row['interface']['link']['body']==controller['body']),
                        'controller needs one admitted outer source call')
    link = decode(T.CallLink,row['interface']['link'])
    scope = decode(F.CodeScopeRef,row['caller_scope'])
    source = view.unit(scope.unit)
    slicer = ReferenceSlicer(view.catalog)
    route = _only((item for item in link.parameters if item.formal==receiver_formal and item.kind is T.RouteKind.ARGUMENT),
                  'controller receiver needs its original outer argument route')
    actual, = route.actuals
    _, local = slicer.local(actual)
    origin = _only(_resolver(view,source,scope).paths(local),
                   'controller receiver needs one original constructor result route')
    if origin.root_kind is not AddressRootKind.CALL_RESULT or origin.selectors or origin.frontier is not None:
        raise ValueError('controller receiver source does not resolve to an original call result')
    activation,_,frame = _only(((e,b,f) for e,b,f in observed.at(link.result)
                                if controller_body in b and e['normal_completion_observed']),
                               'one original observed controller activation required')
    use = _frame_formal(frame,receiver['name']).ref
    if not use.startswith('weak:'):
        raise ValueError('original controller receiver lacks a shared weak identity')
    candidates = [(e,b,f) for e,b,f in observed.at(origin.root,caller=activation['caller'])
                  if e['target'].kind is PythonTargetKind.TYPE_OBJECT and e['attempt'].index < activation['attempt'].index]
    constructor,_,_ = _only(candidates,'one observed type call must supply the controller receiver')
    attempt = _only((a for a in observed.trace.attempts if a.anchor==constructor['attempt']),
                    'type call must retain its original operand attempt')
    bodies = []
    for observed_frame in observed.trace.entries:
        if (observed_frame.caller_invocation!=attempt.invocation or observed_frame.caller!=attempt.site
                or not attempt.frame_index < observed_frame.anchor.index < frame.anchor.index
                or not observed_frame.parameters or observed_frame.parameters[0].ref!=use
                or observed_frame.suspended):
            continue
        exits = observed.exits[observed_frame.invocation]
        if len(exits)==1 and exits[0].completion is FrameCompletion.NORMAL and exits[0].anchor.index < frame.anchor.index:
            bodies.extend((body,observed_frame,exits[0]) for body in observed.source_bodies[
                observed_frame.invocation.process_id,observed_frame.code])
    body,init,exit_ = _only(bodies,'one original ordinary initializer must match type-call site and actual controller receiver')
    owner,fn = _function(view,body)
    original_ordinal,call = _call_at(source,origin.root)
    owner_formal = source_port(owner.facts,fn.formals[1])
    args = _bind_formals(owner,fn,init,source.facts,call,receiver=origin.root)
    profiles = {body:_shared_body(view,body)}
    children,nested_fields,nested_items = _child_writes(view,observed,owner,fn,init,owner_formal,profiles)
    fields = defaultdict(list,nested_fields)
    addresses = _resolver(view,owner,fn.body_scope.owner)
    for access in profiles[body]['field_accesses']:
        if access['kind'] is not F.AttributeAction.SET:
            continue
        for path in access['receiver_paths']:
            selected = _field_path(path)
            if selected is None or selected[0]!=owner_formal:
                continue
            _,payload = slicer.local(access['value'])
            fields[(*selected[1],access['selector'])].insert(0,dict(body=body,
                operation=access['operation'],source_fact=access['source_fact'],
                receiver_path=path,payload=access['value'],payload_paths=addresses.paths(payload)))
    for access in profiles[body]['item_accesses']:
        if access['kind']!='WRITE':
            continue
        for path in access['receiver_paths']:
            selected = _field_path(path)
            if selected is not None and selected[0]==owner_formal and selected[1]:
                nested_items[selected[1]].insert(0,dict(body=body,operation=access['operation'],
                    source_fact=access['source_fact'],receiver_path=path))
    ports = []
    observed_array_args = _array_argument_receipts(update,observed.trace.directory)
    for path,demand in demands.items():
        suffix = path[1:]
        writers = fields[suffix]
        content_writers = nested_items[suffix]
        ports.append(dict(**demand,field_writers=writers,content_writers=content_writers,
            same_observed_array_arguments=_same_argument_objects(demand['id'],content_writers,observed_array_args),
            status='ORIGINAL_FIELD_WRITES_CONNECTED' if writers else 'ORIGINAL_FIELD_PRODUCER_OPEN',
            exact_initial_value_materialized=False))
    return dict(controller_call=dict(interface='#/interfaces/'+str(ordinal),actual=activation,
                    receiver_formal=receiver_formal,receiver_source=origin,receiver_route=route,
                    constructor_instance_reference=use),
                constructor=dict(original_call=_original_call(source,slicer,call,original_ordinal),observation=constructor,
                    initializer=dict(body=body,invocation=init.invocation,entry=init.anchor,exit=exit_.anchor,
                        actual_caller=init.caller,code=init.code,normal_completion_observed=True,
                        receiver_reference=use,arguments=args),
                    class_allocation_and_descriptor_protocol_closed=False),
                initializer_calls=children,
                bodies=[profiles[key] for key in sorted(profiles,key=repr)],
                initial_receiver_fields=[dict(path=list(path),field_writers=rows,content_writers=nested_items[path])
                    for path,rows in sorted(fields.items()) if rows],
                ports=ports)


@replay_transaction()
def project(controller):
    reference = artifact_reference(Path(controller).resolve(strict=True),CONTROLLER)
    control = check_controller(reference['uri'])
    if read_artifact(reference) != control:
        raise ValueError('controller changed during original startup replay')
    update = read_artifact(control['update_source'])
    ordered = read_artifact(update['ordered_graph'])
    entry = read_artifact(update['entry_calls'])
    scope = ordered['observed_interval']
    with BuildSession(Path(entry['program']['uri']).parent) as session:
        with _admit_entry_calls(session,entry):
            observed = _Observed(session.view,scope['trace_directory'],scope['project_root'])
            graph = _constructor(session.view,entry,control,update,observed)
            observed.index.check()
            trace = observed.receipt()
    return encode(dict(schema=SCHEMA,schema_version=1,controller=reference,
        entry_calls=update['entry_calls'],trace_input=trace,**graph,
        frontiers=['CONSTRUCTOR_BRANCHES_AND_FIELD_WRITE_OCCURRENCES_NOT_EXHAUSTIVE',
                   'CURRENT_DEFAULT_VALUES_AND_ORDINARY_RECEIVER_SLOT_GUARDS',
                   'INITIAL_NUMERICAL_PROVIDERS_VALUES_AND_EARLY_ARRAY_GEOMETRY',
                   'UNMODELED_EXTERNAL_EFFECTS_ALLOCATION_AND_EXCEPTION_PROTOCOLS'],
        original_constructor_execution_observed=True,exact_startup_input_materialization_qualified=False,
        numerical_provider_equivalence_proven=False,transformation_authorized=False))


def check(path):
    result = read_artifact(artifact_reference(Path(path).resolve(strict=True),SCHEMA))
    if result != project(result['controller']['uri']):
        raise ValueError('startup connection differs from original source and observed construction')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args = parser.parse_args(argv)
    if args.check and not (args.controller or args.out):
        result = check(args.check)
    elif args.controller and args.out and not args.check:
        result = project(args.controller)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or --controller with --out')
    print(json.dumps(dict(schema=SCHEMA,ports=len(result['ports']),
        connected=sum(row['status']=='ORIGINAL_FIELD_WRITES_CONNECTED' for row in result['ports']),
        initializer_calls=len(result['initializer_calls']))))


if __name__ == '__main__':
    main()
