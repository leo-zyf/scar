"""Project shared source effect bodies behind observed controller CALL ports."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.session_caller_effects import derive_caller_effects
from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_native_controller import SCHEMA as CONTROLLER, check as check_controller
from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex
from scar.analysis.source_observed_codes import ObservedSourceCodeProvider
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T
from scar.ir.program.call_edges import FrameCompletion
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.python_call_targets import PythonTargetKind
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.source-controller-effects'
_NATIVE = frozenset((PythonTargetKind.NATIVE_BUILTIN, PythonTargetKind.NATIVE_WRAPPER))


def _only(rows, detail):
    result = tuple(rows)
    if len(result) != 1:
        raise ValueError(detail)
    return result[0]


class _Observed:
    def __init__(self, view, directory, project_root):
        self.codes = ObservedSourceCodeProvider(view, directory, project_root)
        self.index = ObservedSourceCallsiteIndex(self.codes)
        self.trace = self.codes.trace
        self.frames = {frame.invocation:frame for frame in self.trace.entries}
        self.exits = defaultdict(list)
        self.calls = defaultdict(list)
        self.source_bodies = defaultdict(set)
        for event in self.trace.exits:
            self.exits[event.invocation].append(event)
        for row in self.index.bindings:
            self.calls[row.result].append(row)
        for link in self.codes.body_links():
            for capture in link.captures:
                self.source_bodies[capture.process_id, capture.code.key].add(link.body)
        self.entries = {(row.caller,row.attempt_index):(row,anchor) for row,anchor in zip(
            self.codes.targets.entries,self.codes.targets.entry_anchors,strict=True)}

    def receipt(self):
        return dict(directory=self.trace.directory, project_root=self.trace.project_root,
                    frame_files=self.trace.frame_scan.files, operand_files=self.trace.files,
                    python_targets_sha256=self.codes.targets.source_sha256)

    def at(self, site, *, caller=None):
        for row in self.calls.get(site, ()):
            owner = row.attempt.invocation
            if caller is not None and owner != caller:
                continue
            pair = self.entries.get((owner,row.attempt.index))
            frame = self.frames[pair[0].callee] if pair is not None else None
            exits = () if frame is None else self.exits[frame.invocation]
            end = exits[0] if len(exits) == 1 else None
            normal = end is not None and end.completion is FrameCompletion.NORMAL
            bodies = (() if frame is None else self.source_bodies[frame.invocation.process_id,frame.code])
            evidence = dict(trace_directory=self.trace.directory, caller=owner,
                callsite=row.id, source_result=row.result, source_fact=row.call,
                attempt=row.attempt.anchor, monitored_arg0=row.attempt.arg0,
                target=row.target, target_anchor=row.target_anchor,
                entry_binding=None if pair is None else pair[1],
                callee=None if frame is None else frame.invocation,
                entry=None if frame is None else frame.anchor,
                exit=None if end is None else end.anchor,
                returned_reference=None if end is None else end.return_ref,
                completion=None if end is None else end.completion,
                parameters=() if frame is None else frame.parameters,
                parameters_complete=frame is not None and frame.parameters_complete,
                normal_completion_observed=normal)
            yield evidence, tuple(sorted(bodies,key=repr)), frame


def _original_call(unit, slicer, call, ordinal):
    point = lambda i: source_port(unit.facts,i)
    return dict(source_result=point(call.result), source_fact=slicer.fact(unit.facts,'calls',ordinal),
                callee=point(call.callee), positional=[point(i) for i in call.positional],
                keywords=[dict(name=name,value=point(i)) for name,i in call.keywords],
                positional_pack=None if call.positional_pack is None else point(call.positional_pack),
                keyword_pack=None if call.keyword_pack is None else point(call.keyword_pack),
                world_before=point(call.world), normal_world_after=point(call.result_world))


def _paths(resolver, index):
    return () if index is None else resolver.paths(index)


def _shared_body(view, body):
    unit = view.unit(body.unit)
    ordinal, fn = _only(((i,row) for i,row in enumerate(unit.facts.functions)
                         if row.template == body.template and row.body == body.region),
                        'observed effect body needs one original function interface')
    slicer = ReferenceSlicer(view.catalog)
    addresses = AddressResolver(view.catalog.fact_store,unit.facts,fn.body_scope.owner)
    effects = derive_caller_effects(view,unit=unit.unit,function_ordinal=ordinal)
    params = [dict(name=name,formal=source_port(unit.facts,value),
                   kind=parameter.kind,default_source=parameter.default_source)
              for name,value,parameter in zip(fn.parameter_names,fn.formals[1:-1],fn.environment.parameters,strict=True)]
    fields, items = [], []
    for operation in effects['operations']:
        for record in operation['source_facts']:
            table = record['table']
            if table not in ('attributes','items'):
                continue
            fact = getattr(unit.facts,table)[record['ordinal']]
            access = dict(operation=operation['original']['entry'], source_fact=record['reference'],
                receiver=source_port(unit.facts,fact.receiver), receiver_paths=_paths(addresses,fact.receiver),
                value=None if fact.value is None else source_port(unit.facts,fact.value),
                world_before=source_port(unit.facts,fact.world), normal_world_after=source_port(unit.facts,fact.result_world),
                protocol_and_actual_aliases_required=True)
            if table == 'attributes':
                fields.append(dict(**access,kind=fact.action,selector=fact.name))
            else:
                key = source_port(unit.facts,fact.key) if type(fact.key) is int else fact.key
                items.append(dict(**access,kind='WRITE' if fact.write else 'READ',selector=key))
    calls = []
    for site in effects['call_sites']:
        query = site['query_ordinal']; fact = unit.facts.calls[query]
        row = _original_call(unit,slicer,fact,query)
        row.update(operation=site['original_operation'],
            callee_paths=_paths(addresses,fact.callee),
            positional_paths=[_paths(addresses,i) for i in fact.positional],
            keyword_paths=[dict(name=name,paths=_paths(addresses,i)) for name,i in fact.keywords],
            opaque_callable_may_escape_or_mutate_arguments=True, observations=[])
        calls.append(row)
    # F4 language completions remain explicit references. Raw scoped
    # callbacks and exception handlers are never treated as closed effects.
    return dict(body=body, source=effects['source'], source_function=effects['source_function'],
        function_fact=effects['function_fact'], original_f4=effects['original_f4'],
        source_templates=effects['source_templates'], parameters=params, field_accesses=fields, item_accesses=items,
        calls=calls,
        normal_return=T.GlobalPort(body.unit,G.Port(body.template,body.region,G.PortKind.RESULT,completion=C.RETURN)),
        order_semantics='SHARED_SOURCE_WORLD_PORTS_WITH_OBSERVED_OCCURRENCE_ANCHORS',
        frontiers=['UNOBSERVED_BRANCH_AND_EXCEPTION_EFFECTS',
                   'EXTERNAL_CALL_AND_ARGUMENT_ESCAPE_OR_MUTATION',
                   'ORDINARY_DESCRIPTOR_STORAGE_AND_TEMPORARY_RELEASE_PROTOCOL'],
        observed_path_exhaustive=False, effects_closed=False)


def _formal_links(link, profile, frame):
    if link is None:
        return []
    assert frame is not None
    by_name = {row.name:row for row in frame.parameters}
    return [dict(name=param['name'],formal=param['formal'],routes=[route for route in link.parameters
                if route.formal==param['formal']],declared_default=param['default_source'],
                actual_frame_reference=None if param['name'] not in by_name else by_name[param['name']].ref,
                selected_function_defaults_proven=False) for param in profile['parameters']]


def _classify(evidence, candidates):
    if candidates and evidence['normal_completion_observed']:
        return 'OBSERVED_SOURCE_BODY_NORMAL'
    if evidence['target'].kind in _NATIVE:
        return 'OBSERVED_NATIVE_CALL_BOUNDARY'
    if evidence['callee'] is not None:
        return 'OBSERVED_PYTHON_FRAME_EFFECTS_OPEN'
    return 'OBSERVED_TARGET_WITHOUT_ADMITTED_BODY'


@replay_transaction()
def project(controller, *, additional_traces=()):
    reference = artifact_reference(Path(controller).resolve(strict=True),CONTROLLER)
    control = check_controller(reference['uri'])
    if read_artifact(reference) != control:
        raise ValueError('controller artifact changed during effect projection')
    update = read_artifact(control['update_source'])
    ordered = read_artifact(update['ordered_graph'])
    entry = read_artifact(update['entry_calls'])
    source_trace = ordered['observed_interval']
    directories = tuple(dict.fromkeys(str(Path(path).resolve(strict=True)) for path in
        (source_trace['trace_directory'],*additional_traces)))
    body_rows = {}
    body_occurrences = defaultdict(set)
    effects = []
    links = defaultdict(list)
    for index,row in enumerate(entry['interfaces']):
        link = decode(T.CallLink,row['interface']['link'])
        links[link.result].append((index,link))
    with BuildSession(Path(entry['program']['uri']).parent) as session:
        with _admit_entry_calls(session,entry):
            view = session.view
            observed = tuple(_Observed(view,path,source_trace['project_root']) for path in directories)
            slicer = ReferenceSlicer(view.catalog)
            for effect in control['effects']:
                site = decode(T.GlobalPort,effect['site'])
                unit = view.unit(site.unit)
                query = effect['query_ordinal']
                original = _original_call(unit,slicer,unit.facts.calls[query],query)
                if (original['source_result'] != site or
                        original['source_fact'] != decode(ProviderFactReference,effect['source_fact'])):
                    raise ValueError('controller effect selector differs from its exact original CALL')
                rows, bodies, frontiers = [], set(), set()
                admits = tuple((i,link) for i,link in links[site] if link.body is not None)
                for agent in observed:
                    for row,candidates,frame in agent.at(site):
                        if row['source_fact'] != original['source_fact']:
                            raise ValueError('observed effect CALL matches a different original fact')
                        matching = tuple((i,link) for i,link in admits if link.body in candidates)
                        if matching:
                            candidates = tuple(dict.fromkeys(link.body for _,link in matching))
                        status = _classify(row,candidates)
                        row.update(status=status,body=None,source_interface=None,formal_routes=[])
                        if status == 'OBSERVED_SOURCE_BODY_NORMAL' and len(candidates) == 1:
                            body, = candidates
                            if body not in body_rows:
                                body_rows[body] = _shared_body(view,body)
                            bodies.add(body)
                            row['body'] = body
                            body_occurrences[body].add((agent.trace.directory,frame.invocation))
                            if len(matching) == 1:
                                i,link = matching[0]
                                row['source_interface'] = '#/interfaces/'+str(i)
                                row['formal_routes'] = _formal_links(link,body_rows[body],frame)
                            else:
                                frontiers.add('ORIGINAL_CURRENT_FORMAL_ROUTE_UNADMITTED')
                        elif status == 'OBSERVED_SOURCE_BODY_NORMAL':
                            row['status'] = 'AMBIGUOUS_SOURCE_BODY'
                            frontiers.add(row['status'])
                        else:
                            frontiers.add(status)
                        rows.append(row)
                if not rows:
                    frontiers.add('NO_OBSERVED_ACTIVATION_AT_ORIGINAL_CALL')
                effects.append(dict(id=effect['id'],site=site,source_fact=original['source_fact'],
                    original_call=original, result_policy=effect['result_policy'],
                    observations=rows,bodies=tuple(sorted(bodies,key=repr)),frontiers=sorted(frontiers),
                    effects_closed=False))
            for body,profile in body_rows.items():
                by_trace = defaultdict(set)
                for directory,invocation in body_occurrences[body]:
                    by_trace[directory].add(invocation)
                for call in profile['calls']:
                    for agent in observed:
                        for caller in sorted(by_trace[agent.trace.directory],key=repr):
                            for episode,candidates,_ in agent.at(call['source_result'],caller=caller):
                                episode['status'] = _classify(episode,candidates)
                                episode['source_body_candidates'] = candidates
                                call['observations'].append(episode)
                    call['observations'].sort(key=lambda row:(row['trace_directory'],row['attempt'].index))
                    if not call['observations']:
                        call['frontier'] = 'ORIGINAL_CALL_NOT_RECORDED_IN_SELECTED_ACTIVATIONS'
                    else:
                        call['frontier'] = 'NESTED_TARGET_EFFECTS_AND_ESCAPES_NOT_COMPOSED'
            for agent in observed:
                agent.index.check()
            receipts = [agent.receipt() for agent in observed]
    return encode(dict(schema=SCHEMA,schema_version=1,controller=reference,trace_inputs=receipts,
        effects=effects,bodies=[body_rows[key] for key in sorted(body_rows,key=repr)],
        shared_bodies_expanded_for_every_iteration=False, dynamic_control_path_exhaustive=False,
        external_provider_or_file_semantics_verified=False, whole_effects_closed=False,
        transformation_authorized=False))


def check(path):
    saved = read_artifact(artifact_reference(Path(path).resolve(strict=True),SCHEMA))
    result = project(saved['controller']['uri'],additional_traces=[row['directory'] for row in saved['trace_inputs'][1:]])
    if result != saved:
        raise ValueError('controller effect bodies differ from original source and observed activation replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',type=Path)
    parser.add_argument('--additional-trace',type=Path,action='append',default=[])
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args = parser.parse_args(argv)
    if args.check and not (args.controller or args.out or args.additional_trace):
        result = check(args.check)
    elif args.controller and args.out and not args.check:
        result = project(args.controller,additional_traces=args.additional_trace)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or --controller with --out')
    print(json.dumps(dict(schema=SCHEMA,bodies=len(result['bodies']),
        effect_status={row['id']:[observed['status'] for observed in row['observations']] for row in result['effects']})))


if __name__=='__main__':
    main()
