"""Cold-consume local lifetime facts by replaying their original source graph."""
import argparse
from dataclasses import replace
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import time

from scar.analysis.build import BuildSession
from scar.analysis.inspect_ordered_activations import validate_public_interfaces
from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
from scar.analysis.ordered_region_lifetime import OrderedRegionLifetimeProvider,_pins,SCHEMA as PUBLICATION_SCHEMA
from scar.analysis.read_ordered_activations import _header as _activation_header,_read_tables as _activation_tables
from scar.analysis.source_call_view import SourceFunctionCallProvider
from scar.analysis.source_symbol_variants import _table
from scar.analysis.stateful_dependency_view import StatefulDependencyProvider,read_stateful_graph
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.program import ordered_region_lifetime as L
from scar.ir.program.dependencies import DependencyEdge,context_push
from scar.ir.record_codec import decode


SCHEMA='scar.ordered-input-lifetime-independent-replay'
_FIELDS={'schema','schema_version','activation_publication','original_region','pins','tables','summary',
    'graph_validation','original_body_bytes_copied','iterations_unrolled','target_executed','transformation_authorized'}


def _header(report):
    if (set(report)!=_FIELDS or report.get('schema')!=PUBLICATION_SCHEMA
            or type(report.get('schema_version'))is not int or report['schema_version']!=1):
        raise ValueError('unsupported ordered lifetime publication fields/schema')
    if report['pins']!=_pins():raise ValueError('ordered lifetime producing rules changed')
    if any(report[name]is not False for name in('iterations_unrolled','target_executed','transformation_authorized')):
        raise ValueError('ordered lifetime authority flags differ')
    if type(report['original_body_bytes_copied'])is not int or report['original_body_bytes_copied']!=0:
        raise ValueError('ordered lifetime publication copied original bodies')
    for name,schema in(('activation_publication','scar.ordered-activation-interfaces'),
            ('original_region','scar.stateful-array-region')):
        if type(report[name])is not dict or report[name].get('schema')!=schema:
            raise ValueError('ordered lifetime original reference differs: '+name)
        read_artifact(report[name])


def _tables(path,report,provider):
    edges=tuple(dict.fromkeys(edge for source,_,_,_ in provider._edges
        for edge in provider.outgoing(source).edges))
    expected=dict(inputs=(L.OrderedInputLifetime,provider.inputs),epochs=(L.OrderedEpochLifetime,provider.epochs),
        uses=(L.OrderedInputUse,provider.uses),edges=(DependencyEdge,edges))
    if set(report['tables'])!=set(expected):raise ValueError('ordered lifetime table domain differs')
    missing=object();counts={}
    for name,(cls,rows)in expected.items():
        receipt=report['tables'][name]
        if receipt.get('path')!=name+'.jsonl.gz':raise ValueError('ordered lifetime table local path differs: '+name)
        count=0
        for wire,original in zip_longest(_table(path,receipt),rows,fillvalue=missing):
            if wire is missing or original is missing or decode(cls,wire)!=original:
                raise ValueError('ordered lifetime '+name+' differs from independently replayed original dataflow')
            count+=1
        counts[name]=count
    return counts,edges


def _graph(provider,edges):
    def check(rows):
        for edge in rows:
            graph=provider.view.dependencies
            if edge not in graph.outgoing(edge.source,kinds=(edge.kind,)).edges:
                raise ValueError('cold lifetime public outgoing edge differs')
            if edge not in graph.incoming(edge.target,kinds=(edge.kind,)).edges:
                raise ValueError('cold lifetime public incoming edge differs')
    check(edges);contextual=set()
    for binding in provider.activation.bindings:
        context=context_push(None,binding.ordered_frame)
        for source,_,_,_ in provider._edges:
            contextual.update(provider.outgoing(replace(source,context=context)).edges)
    check(contextual)
    return dict(conditional_edges=len(edges),bidirectional_missing=0),len(contextual)


def read_ordered_lifetimes(path):
    """Return an independent receipt without modifying the source publication.

    The fresh original BuildSession, ordered body, caller interfaces, input
    routes and lifetime rules are replayed once. Parent activation validation
    precedes registering the extension, preserving its original edge domain.
    """
    started=time.perf_counter();path=Path(path).resolve()
    result_path=path if path.is_file()else path/'result.json';path=result_path.parent
    publication=artifact_reference(result_path,PUBLICATION_SCHEMA)
    report=read_artifact(publication);_header(report);pins=report['pins']
    replayer=sha256(Path(__file__).read_bytes()).hexdigest()
    parent=read_artifact(report['activation_publication']);_activation_header(parent)
    if parent['original_region']!=report['original_region']:
        raise ValueError('ordered lifetime original region differs from its activation parent')
    program_path=Path(parent['program']['uri'])
    if program_path.name!='manifest.json':raise ValueError('original program reference is not its manifest')
    with BuildSession(program_path.parent)as session:
        if artifact_reference(session.path/'manifest.json','scar.program-build')!=parent['program']:
            raise ValueError('ordered lifetime cold session selected another original program')
        ordered=StatefulDependencyProvider(session.view,report['original_region']['uri'])
        if ordered.reference!=report['original_region']:
            raise ValueError('ordered lifetime original source replay differs')
        body,_=read_stateful_graph(Path(parent['shared_body']['uri']).parent,session.view,_provider=ordered)
        if body['original_region']!=report['original_region']:
            raise ValueError('ordered lifetime shared body differs from its source region')
        calls={name:p for name,p in session.session.providers.items()if type(p)is SourceFunctionCallProvider}
        activation=OrderedRegionActivationProvider(ordered,calls)
        if activation.summary()!=parent['summary']:
            raise ValueError('ordered lifetime original activation summary differs')
        activation_counts=_activation_tables(Path(report['activation_publication']['uri']).parent,parent,activation)
        activation_validation=validate_public_interfaces(activation)
        if activation_validation!=parent['validation']:
            raise ValueError('ordered lifetime original activation graph differs')
        provider=OrderedRegionLifetimeProvider(activation,register=True)
        if provider.summary()!=report['summary']:
            raise ValueError('ordered lifetime summary differs from original dataflow')
        counts,edges=_tables(path,report,provider)
        validation,context_edges=_graph(provider,edges)
        if validation!=report['graph_validation']:
            raise ValueError('ordered lifetime public graph validation differs')
        provider.check()
        if pins!=_pins()or replayer!=sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('ordered lifetime rules changed during independent replay')
        for reference in(publication,report['activation_publication'],report['original_region'],
                parent['program'],parent['shared_body']):read_artifact(reference,json_document=False)
    return dict(schema=SCHEMA,schema_version=1,publication=publication,
        activation_publication=report['activation_publication'],original_region=report['original_region'],
        program=parent['program'],producer_pins=pins,replayer_sha256=replayer,
        table_counts=counts,activation_table_counts=activation_counts,summary=report['summary'],
        graph_validation=validation,matched_context_edges_checked=context_edges,
        activation_validation=activation_validation,independent_cold_semantic_rederivation=True,
        original_body_bytes_copied=0,iterations_unrolled=False,runtime_invocations_created=0,
        global_logical_versions_created=0,cross_call_content_identity_proven=False,
        external_storage_release_authorized=False,target_executed=False,transformation_authorized=False,
        seconds=time.perf_counter()-started)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication',help='lifetime publication directory or result.json')
    parser.add_argument('--out',help='fresh independent replay receipt JSON')
    args=parser.parse_args(argv);receipt=read_ordered_lifetimes(args.publication)
    if args.out:
        output=Path(args.out).absolute();output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x')as stream:stream.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(schema=receipt['schema'],table_counts=receipt['table_counts'],
        graph_validation=receipt['graph_validation'],matched_context_edges_checked=receipt['matched_context_edges_checked'],
        seconds=receipt['seconds'],target_executed=False),sort_keys=True))
    return 0


__all__=['read_ordered_lifetimes']


if __name__=='__main__':raise SystemExit(main())
