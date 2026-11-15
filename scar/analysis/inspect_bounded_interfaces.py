"""Bounded inspection of conditional CALL interfaces in the shared main graph.

This reads an admitted build and its original metadata publications. Static
caller frames preserve argument/world/return routing; they are not recorded
invocations or observations of callee entry/completion. Bodies are referenced,
never copied, unfolded, executed or reconstructed by this query.
"""
import argparse
from collections import Counter
from dataclasses import fields,is_dataclass,replace
from enum import Enum
import json
from pathlib import Path
import time


class _ReportReferences:
    """Report-local aliases only; original IDs and evidence remain in tables.

    This is a compact query result, not a new graph serialization or identity
    scheme. In particular, a frame reference still denotes a conditional
    static source interface, never a recorded runtime invocation.
    """
    def __init__(self):
        from scar.ir.program.call_edges import CapturedCode
        from scar.ir.program.dependencies import (DependencyCondition,
            DependencyEvidence,DependencyFrame,DependencyFrontier,DependencyPoint,GraphNamespace)
        from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
        from scar.ir.program.source_call_bindings import SourceBodyNamespace
        from scar.ir.program.source_references import SourceArtifactLocator
        self.types={ProviderArtifact:'material',ProviderFactReference:'fact',
            SourceArtifactLocator:'locator',GraphNamespace:'namespace',
            DependencyFrame:'frame',DependencyPoint:'point',
            DependencyCondition:'condition',DependencyEvidence:'evidence',
            DependencyFrontier:'frontier',SourceBodyNamespace:'body',CapturedCode:'code'}
        self.tables={name:[] for name in (*self.types.values(),'predicate','evidence_rule','frontier_rule')}
        self.indexes={name:{} for name in self.tables}

    def _intern(self,name,key,build):
        index=self.indexes[name].get(key)
        if index is None:
            index=len(self.tables[name]);self.indexes[name][key]=index
            self.tables[name].append(None);self.tables[name][index]=build()
        return {'$'+name:index}

    def _key(self,value):
        # GraphNamespace equality intentionally ignores alternative declaration
        # receipts. Report deduplication must retain those original fields too.
        if is_dataclass(value):
            return (type(value),tuple(self._key(getattr(value,field.name)) for field in fields(value)))
        if type(value) in (tuple,list):return (type(value),tuple(self._key(item) for item in value))
        if type(value) is dict:return tuple((key,self._key(item)) for key,item in sorted(value.items()))
        return value

    def encode(self,value):
        from scar.ir.record_codec import encode
        from scar.ir.v2.ids import Identifier,ValueVersionID
        from scar.ir.program.dependencies import OpenDependencyContext
        if value is None or type(value) in (str,int,float,bool):return value
        if isinstance(value,Enum):return value.value
        if isinstance(value,(Identifier,ValueVersionID)):return value.wire
        if type(value) is OpenDependencyContext:
            return dict(outer_prefix='UNKNOWN',frames=self.encode(value.frames))
        name=self.types.get(type(value))
        if name is not None:
            def build():
                row={field.name:self.encode(getattr(value,field.name)) for field in fields(value)}
                # Preserve None versus closed () versus open-prefix context in
                # a form that cannot be mistaken for a dynamic stack.
                if name=='point':
                    row['context']=(dict(outer_prefix='UNKNOWN',frames=[]) if value.context is None
                        else dict(outer_prefix='CLOSED',frames=self.encode(value.context))
                        if type(value.context) is tuple else self.encode(value.context))
                elif name=='condition':
                    row.pop('domain');row.pop('name')
                    row['predicate']=self._intern('predicate',(value.domain,value.name),
                        lambda:dict(domain=value.domain,name=value.name))
                elif name=='evidence':
                    row['rule']=self._intern('evidence_rule',value.rule,lambda:value.rule)
                elif name=='frontier':
                    row.pop('kind');row.pop('detail')
                    row['reason']=self._intern('frontier_rule',(value.kind,value.detail),
                        lambda:dict(kind=value.kind,detail=value.detail))
                return row
            return self._intern(name,self._key(value),build)
        if is_dataclass(value):
            return {field.name:self.encode(getattr(value,field.name)) for field in fields(value)}
        if type(value) in (tuple,list):return [self.encode(item) for item in value]
        if type(value) is dict:return {key:self.encode(item) for key,item in value.items()}
        return encode(value)


def inspect_bounded_interfaces(program,out,*,max_calls=64,max_edges=4096):
    from scar.analysis.build import BuildSession
    from scar.analysis.source_call_view import SourceFunctionCallProvider,Flow
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.ir.program.dependencies import context_push
    if any(type(n) is not int or n<1 for n in (max_calls,max_edges)):
        raise ValueError('positive explicit interface and edge report budgets required')
    program=Path(program).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('fresh bounded interface inspection output required')
    parent=artifact_reference(program/'manifest.json','scar.program-build');manifest=read_artifact(parent)
    if manifest.get('backend')!='F4':raise ValueError('original F4 unified program build required')
    started=time.monotonic();edges=[];edge_ids={};frontiers=[];frontier_ids={}
    providers=[];calls=[];metadata=[];truncated=False;missing_relations=0;interface_kinds=Counter()
    phase_seconds=dict(build_session_constructor=0.0,explicit_provider_unit_attach=0.0,
        main_graph_outgoing_calls=0.0,report_record_encoding=0.0)
    phase_calls=dict(build_session_constructor=0,explicit_provider_unit_attach=0,main_graph_outgoing_calls=0)
    references=_ReportReferences()
    def encode(value):
        before=time.monotonic()
        try:return references.encode(value)
        finally:phase_seconds['report_record_encoding']+=time.monotonic()-before
    def note_frontiers(neighborhood):
        for item in neighborhood.frontiers:
            if item not in frontier_ids:
                frontier_ids[item]=len(frontiers);frontiers.append(item)
        return [frontier_ids[item] for item in neighborhood.frontiers]
    before=time.monotonic();admitted_session=BuildSession(program)
    phase_seconds['build_session_constructor']=time.monotonic()-before
    phase_calls['build_session_constructor']=1
    with admitted_session as session:
        graph=session.view.dependencies;inner=session.session
        for name,provider in inner.providers.items():
            if name.startswith('bounded_call_bindings:'):
                provider.check()
                for point,operation,record in provider.nodes.values():
                    metadata.append(dict(call=encode(operation),attempt_evidence=encode(record.evidence),
                        target_kind=record.target_kind,target_code=encode(record.target_code),
                        source_body=encode(provider.body_targets.get((point.namespace,point.node))),
                        source_body_frontier=provider.body_frontiers.get((point.namespace,point.node)),
                        remaining_fields=list(record.remaining_fields),runtime_invocation_created=False))
        for name,provider in inner.providers.items():
            if not name.startswith('bounded_call_interfaces:'):continue
            if type(provider) is not SourceFunctionCallProvider:
                raise TypeError('bounded interfaces must use the existing shared source-call provider')
            for unit in sorted(provider.available_units):
                if len(calls)>=max_calls:truncated=True;break
                before=time.monotonic();provider.attach(unit)
                phase_seconds['explicit_provider_unit_attach']+=time.monotonic()-before
                phase_calls['explicit_provider_unit_attach']+=1
                for key,frame in tuple(provider.frames.items()):
                    if key[0]!=unit:continue
                    if len(calls)>=max_calls:truncated=True;break
                    interface=provider.links[key];link=interface.link
                    # None means an unknown static caller prefix. A closed ()
                    # would assert module activation and misidentify a CALL
                    # whose containing function has not been selected.
                    caller_context=None;callee_context=context_push(caller_context,frame)
                    relations=[];query_frontiers=[]
                    for relation in provider.relations:
                        if relation.interface!=key:continue
                        if len(edges)>=max_edges:truncated=True;break
                        enter=relation.flow is Flow.ENTER
                        source=replace(relation.source,context=caller_context if enter else callee_context)
                        target=replace(relation.target,context=callee_context if enter else caller_context)
                        before=time.monotonic();neighborhood=graph.outgoing(source,kinds=provider.relation_kinds)
                        phase_seconds['main_graph_outgoing_calls']+=time.monotonic()-before
                        phase_calls['main_graph_outgoing_calls']+=1
                        found=[edge for edge in neighborhood.edges
                            if edge.source==source and edge.target==target and edge.kind is relation.kind]
                        query_frontiers.extend(note_frontiers(neighborhood))
                        indexes=[]
                        for edge in found:
                            if edge not in edge_ids:
                                if len(edges)>=max_edges:truncated=True;break
                                edge_ids[edge]=len(edges);edges.append(edge)
                            indexes.append(edge_ids[edge])
                        missing_relations+=int(not indexes)
                        relations.append(dict(flow=relation.flow.value,kind=relation.kind.value,
                            route_ordinal=relation.route,source=encode(source),target=encode(target),
                            main_graph_edge_indexes=indexes,matched_in_main_graph=bool(indexes)))
                    attempts=getattr(provider.store,'interface_attempts',None)
                    refs=(frame.selection,) if attempts is None else attempts(*key)
                    evidence=getattr(provider.store,'interface_evidence',None)
                    binding_refs=(frame.selection,) if evidence is None else evidence(*key)
                    returns=[i for i,row in enumerate(relations) if row['flow']=='RETURN']
                    calls.append(dict(provider=name,source_unit=unit,interface_ordinal=key[1],
                        source_query_ordinal=interface.query,source_call=encode(link.result),
                        source_callable=encode(link.source_callable),shared_body=encode(link.body),
                        static_caller_frame=encode(frame),caller_context='UNKNOWN_STATIC_OUTER_PREFIX',
                        source_call_evidence=encode(interface.call),original_attempt_refs=[encode(ref) for ref in refs],
                        original_binding_evidence=[encode(ref) for ref in binding_refs],
                        relations=relations,paired_return_relation_indexes=returns,
                        query_frontier_indexes=sorted(set(query_frontiers)),
                        runtime_invocation=None,callee_entry_observed=False,callee_completion_observed=False,
                        body_value_equivalence_proven=False,transformation_authorized=False))
            # These include unsupported/non-observed targets that cannot create
            # an ordinary frame. Omitting them would disguise interface gaps.
            unresolved=[encode(item) for material in provider.graphs.values() for item in material.frontiers]
            coverage=getattr(provider.store,'coverage_summary',None)
            coverage=None if coverage is None else coverage()
            if coverage is not None:interface_kinds.update(coverage.get('callable_binding_alternatives',{}))
            providers.append(dict(name=name,summary=provider.summary(),coverage=coverage,interface_frontiers=unresolved))
        read_artifact(parent)
        gaps=list(inner.gaps)
    encoded_edges=[encode(edge) for edge in edges]
    report=dict(schema='scar.bounded-interface-inspection',schema_version=2,program=parent,
        original_bounded_publications=manifest.get('bounded_call_bindings',[]),providers=providers,
        metadata_attempts=metadata,interfaces=calls,edges=encoded_edges,
        query_frontiers=[encode(item) for item in frontiers],build_gaps=gaps,
        summary=dict(conditional_static_frames=len(calls),raw_attempt_records=len(metadata),
            relations=sum(len(row['relations']) for row in calls),main_graph_edges=len(edges),
            missing_main_graph_relations=missing_relations,
            enter_relations=sum(r['flow']=='ENTER' for row in calls for r in row['relations']),
            return_relations=sum(r['flow']=='RETURN' for row in calls for r in row['relations']),
            kinds=dict(Counter(edge.kind.value for edge in edges)),
            conditions=dict(Counter(condition.name for edge in edges for condition in edge.conditions)),
            frontier_reasons=dict(Counter(item.kind for item in frontiers)),
            interface_kinds=dict(interface_kinds),
            report_budget_exhausted=truncated,interface_providers=len(providers)),
        limits=dict(max_calls=max_calls,max_edges=max_edges),
        semantics='CONDITIONAL_SOURCE_INTERFACES_WITH_CALLER_MATCHED_STATIC_FRAMES',
        return_scope='SOURCE_COMPLETION_PORT_ROUTES_NOT_RECORDED_RETURN_EVENTS',
        runtime_invocation_ids_created=False,callee_entry_or_completion_proven=False,
        data_content_versions_proven=False,effects_closed=False,unobserved_paths_covered=False,
        source_bodies_copied=False,iterations_unrolled=False,target_executed=False,
        first_stage_complete=False,transformation_authorized=False,seconds=time.monotonic()-started)
    report['reference_tables']=references.tables
    report['reference_scope']='REPORT_LOCAL_ALIASES_OF_ORIGINAL_RECORDS_NOT_NEW_PROGRAM_IDS'
    report['reference_syntax']='A single-key {$TYPE: ordinal} indexes reference_tables[TYPE]; typed IDs use their original canonical wire strings; source-qualified endpoints and evidence remain in shared tables.'
    report['summary']['reference_records']={name:len(rows) for name,rows in references.tables.items()}
    report['seconds']=time.monotonic()-started
    phase_seconds['other_before_publication']=max(0.0,report['seconds']-sum(phase_seconds.values()))
    report['phase_seconds']=phase_seconds
    report['phase_calls']=phase_calls
    report['timing_scope']='NONOVERLAPPING_WALL_INTERVALS_BEFORE_FINAL_JSON_SERIALIZATION_AND_FILE_WRITE; outgoing_calls_include_any_resolution_triggered_inside_that_API; constructor_is_not_graph_query_time'
    out.mkdir(parents=True);(out/'result.json').write_text(json.dumps(report,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--max-calls',type=int,default=64);parser.add_argument('--max-edges',type=int,default=4096)
    args=parser.parse_args(argv)
    report=inspect_bounded_interfaces(args.build,args.out,max_calls=args.max_calls,max_edges=args.max_edges)
    print(json.dumps(dict(result=str(Path(args.out)/'result.json'),summary=report['summary'],
        seconds=report['seconds'],phase_seconds=report['phase_seconds'],phase_calls=report['phase_calls']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
