"""Retain and independently replay original shared-loop state transfers.

Live publication checks the already-published activation/lifetime tables in
the same source view. Independent admission reconstructs all three layers in
one fresh session, in order, before attaching the next relation extension.
"""
from contextlib import contextmanager
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import json
import time

from scar.analysis.ordered_loop_transfer import OrderedLoopTransferProvider
from scar.analysis.read_ordered_lifetimes import (_header as _life_header,
    _tables as _life_tables,_graph as _life_graph)
from scar.analysis.read_ordered_activations import (_header as _activation_header,
    _read_tables as _activation_tables)
from scar.analysis.source_symbol_variants import _table
from scar.ir.program import ordered_regions as R
from scar.ir.program import call_returns as W
from scar.ir.program.dependencies import DependencyEdge
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.record_codec import encode,decode


SCHEMA='scar.ordered-loop-transfer-publication'
REPLAY_SCHEMA='scar.ordered-loop-transfer-independent-replay'
_FIELDS={'schema','schema_version','lifetime_publication','activation_publication','original_region','program',
    'pins','tables','summary','graph_validation','original_body_bytes_copied','iterations_unrolled',
    'runtime_invocations_created','global_logical_versions_created','unconditional_content_recurrence_proven',
    'target_executed','transformation_authorized'}


def _pins():
    from scar.analysis import ordered_loop_transfer, source_control as source_loop_dependencies, source_value_flow, source_references as source_reference_slice, source_namespace_interfaces as source_namespace_interfaces, source_call_view, normal_graph, normal_output, read_ordered_lifetimes, ordered_region_lifetime, source_selected_call_window
    from scar.ir.program import call_returns as W
    from scar.ir import record_codec
    modules=(('storage',__import__(__name__,fromlist=['__file__'])),('transfer',ordered_loop_transfer),
        ('transfer_types',R),('loop_interfaces',source_loop_dependencies),('value_flow',source_value_flow),
        ('reference_paths',source_reference_slice),('namespace_interfaces',source_namespace_interfaces),
        ('original_call_interfaces',source_call_view),('normal_edges',normal_graph),('table_writer',normal_output),
        ('codec',record_codec),('lifetime_replayer',read_ordered_lifetimes),
        ('selected_call_window',source_selected_call_window),('selected_call_window_types',W))
    return dict(rules={name:sha256(Path(module.__file__).read_bytes()).hexdigest()for name,module in modules},
        parent_lifetime=ordered_region_lifetime._pins())


def _reference(path,schema):
    path=Path(path).resolve()
    return artifact_reference(path if path.is_file()else path/'result.json',schema)


def _edges(provider):
    result=set()
    for row in provider._edges:
        # Include the unselected definition and its open-prefix matched CALL
        # family. At source depth zero the two share a context=None endpoint.
        for point in(row.source,row.target,*provider._project(row,None)):
            result.update(provider.incoming(point).edges)
            result.update(provider.outgoing(point).edges)
    return tuple(sorted(result,key=lambda edge:json.dumps(encode(edge),sort_keys=True,separators=(',',':'))))


def _rows(provider):
    return dict(loops=(R.OrderedLoopInterface,provider.loops),roots=(R.OrderedLoopRootRoute,provider.root_routes),
        transfers=(R.OrderedLoopInputTransfer,provider.transfers),writes=(R.OrderedLoopCallerWrite,provider.caller_writes),
        selected_windows=(W.OrderedWrapperCallWindow,provider.selected_windows),
        frontiers=(R.OrderedLoopEffectFrontier,provider.frontiers),edges=(DependencyEdge,_edges(provider)))


def _read_tables(path,report,provider):
    expected=_rows(provider)
    if set(report['tables'])!=set(expected):raise ValueError('ordered loop transfer table domain differs')
    missing=object();counts={}
    for name,(cls,rows)in expected.items():
        receipt=report['tables'][name]
        if receipt.get('path')!=name+'.jsonl.gz':raise ValueError('ordered loop transfer table path differs: '+name)
        count=0
        for wire,original in zip_longest(_table(path,receipt),rows,fillvalue=missing):
            if wire is missing or original is missing or decode(cls,wire)!=original:
                raise ValueError('ordered loop '+name+' differs from original loop/CALL/state-path derivation')
            count+=1
        counts[name]=count
    return counts


def _graph(provider):
    edges=_edges(provider);graph=provider.view.dependencies
    for edge in edges:
        if edge not in graph.outgoing(edge.source,kinds=(edge.kind,)).edges:
            raise ValueError('ordered loop transfer public outgoing edge differs')
        if edge not in graph.incoming(edge.target,kinds=(edge.kind,)).edges:
            raise ValueError('ordered loop transfer public incoming edge differs')
    definitions=sum(e.source.context is None and e.target.context is None for e in edges)
    return dict(conditional_edges=len(edges),definition_edges=definitions,
        matched_call_path_edges=len(edges)-definitions,bidirectional_missing=0)


def _parents(reference):
    life=read_artifact(reference);_life_header(life)
    activation=read_artifact(life['activation_publication']);_activation_header(activation)
    if activation['original_region']!=life['original_region']:
        raise ValueError('loop transfer lifetime/activation source regions differ')
    program=read_artifact(activation['program'])
    return life,activation,program


def _live_parent(provider,reference):
    life,activation,program=_parents(reference);lifetime=provider.lifetime
    if (type(provider)is not OrderedLoopTransferProvider or provider.view is not lifetime.view
            or provider.activation is not lifetime.activation or provider.ordered is not lifetime.ordered
            or life['original_region']!=provider.ordered.reference or lifetime.summary()!=life['summary']
            or provider.activation.summary()!=activation['summary']):
        raise ValueError('loop transfer parent is not the same source-replayed live lifetime/activation')
    # The parent publication already validated the base 360-edge domain before
    # registering the lifetime layer. Recheck exact original typed semantics;
    # do not count the lifetime extension against that older base domain.
    from scar.analysis.build import _admit_rules,_receipt
    catalog=program.get('f4',{}).get('catalog')
    if program.get('backend')!='F4' or catalog!=_receipt(provider.view.catalog.path/'result.json'):
        raise ValueError('loop transfer parent program uses another original source catalog/rules')
    _admit_rules(program.get('rules'))
    activation_counts=_activation_tables(Path(life['activation_publication']['uri']).parent,
        activation,provider.activation)
    life_counts,edges=_life_tables(Path(reference['uri']).parent,life,lifetime)
    validation,contexts=_life_graph(lifetime,edges)
    if validation!=life['graph_validation']:raise ValueError('loop transfer parent lifetime graph differs')
    return life,activation,activation_counts,life_counts,contexts


def publish_ordered_loop_transfers(provider,out,*,lifetime_publication):
    """Publish an unregistered extension after checking its existing parents."""
    from scar.analysis.normal_output import write_table
    if type(provider)is not OrderedLoopTransferProvider:raise TypeError('original ordered loop transfer provider required')
    if any(p is provider for p in provider.view.dependencies.providers):
        raise ValueError('publish parent lifetime before registering its loop extension; construct with register=False')
    reference=_reference(lifetime_publication,'scar.ordered-input-lifetime')
    life,activation,_,_,_=_live_parent(provider,reference);pins=_pins();provider.check()
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh ordered loop transfer publication required')
    out.mkdir(parents=True);provider.view.register(provider)
    rows=_rows(provider)
    tables={name:write_table(out/(name+'.jsonl.gz'),values)for name,(_,values)in rows.items()}
    result=dict(schema=SCHEMA,schema_version=1,lifetime_publication=reference,
        activation_publication=life['activation_publication'],original_region=provider.ordered.reference,
        program=activation['program'],pins=pins,tables=tables,summary=provider.summary(),
        graph_validation=_graph(provider),original_body_bytes_copied=0,iterations_unrolled=False,
        runtime_invocations_created=0,global_logical_versions_created=0,
        unconditional_content_recurrence_proven=False,target_executed=False,transformation_authorized=False)
    _read_tables(out,result,provider);provider.check()
    for ref in(reference,life['activation_publication'],life['original_region'],activation['program'],activation['shared_body']):
        read_artifact(ref,json_document=False)
    if pins!=_pins():raise ValueError('ordered loop transfer producing rules changed during publication')
    with (out/'result.json').open('x')as stream:stream.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


def _header(report):
    if (set(report)!=_FIELDS or report.get('schema')!=SCHEMA
            or type(report.get('schema_version'))is not int or report['schema_version']!=1):
        raise ValueError('unsupported ordered loop transfer publication')
    from scar.analysis.build_reader_versions import LOOP_TRANSFER_TRANSITION, compatible_pins
    active=_pins();recorded=report['pins']
    if ({**recorded,'rules':active['rules']}!=active
            or not compatible_pins(recorded['rules'],active['rules'],LOOP_TRANSFER_TRANSITION)):
        raise ValueError('ordered loop transfer producing rules changed')
    for name in('iterations_unrolled','unconditional_content_recurrence_proven','target_executed','transformation_authorized'):
        if report[name]is not False:raise ValueError('ordered loop transfer authority flag differs: '+name)
    for name in('original_body_bytes_copied','runtime_invocations_created','global_logical_versions_created'):
        if type(report[name])is not int or report[name]!=0:raise ValueError('ordered loop transfer authority count differs: '+name)
    for name,schema in(('lifetime_publication','scar.ordered-input-lifetime'),
            ('activation_publication','scar.ordered-activation-interfaces'),
            ('original_region','scar.stateful-array-region'),('program','scar.program-build')):
        if type(report[name])is not dict or report[name].get('schema')!=schema:
            raise ValueError('ordered loop original reference differs: '+name)
        read_artifact(report[name])
    return active


@contextmanager
def _open_lifetime(reference):
    """One cold original session for activation, lifetime and loop layers."""
    from scar.analysis.build import BuildSession
    from scar.analysis.stateful_dependency_view import StatefulDependencyProvider,read_stateful_graph
    from scar.analysis.source_call_view import SourceFunctionCallProvider
    from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
    from scar.analysis.inspect_ordered_activations import validate_public_interfaces
    from scar.analysis.ordered_region_lifetime import OrderedRegionLifetimeProvider
    life,activation,_=_parents(reference);path=Path(activation['program']['uri'])
    if path.name!='manifest.json':raise ValueError('loop transfer original program is not its manifest')
    with BuildSession(path.parent)as session:
        if artifact_reference(session.path/'manifest.json','scar.program-build')!=activation['program']:
            raise ValueError('cold loop transfer session opened another original program')
        ordered=StatefulDependencyProvider(session.view,life['original_region']['uri'])
        if ordered.reference!=life['original_region']:raise ValueError('cold loop transfer original source replay differs')
        read_stateful_graph(Path(activation['shared_body']['uri']).parent,session.view,_provider=ordered)
        calls={name:p for name,p in session.session.providers.items()if type(p)is SourceFunctionCallProvider}
        original=OrderedRegionActivationProvider(ordered,calls)
        if original.summary()!=activation['summary']:raise ValueError('cold loop transfer original activation differs')
        acounts=_activation_tables(Path(life['activation_publication']['uri']).parent,activation,original)
        aval=validate_public_interfaces(original)
        if aval!=activation['validation']:raise ValueError('cold loop transfer original activation graph differs')
        lifetime=OrderedRegionLifetimeProvider(original,register=True)
        if lifetime.summary()!=life['summary']:raise ValueError('cold loop transfer original lifetime differs')
        lcounts,edges=_life_tables(Path(reference['uri']).parent,life,lifetime)
        lval,lcontexts=_life_graph(lifetime,edges)
        if lval!=life['graph_validation']:raise ValueError('cold loop transfer lifetime graph differs')
        yield NS(session=session,lifetime=lifetime,life=life,activation=activation,
            activation_counts=acounts,activation_validation=aval,lifetime_counts=lcounts,
            lifetime_validation=lval,lifetime_context_edges=lcontexts)
        lifetime.check()
        for ref in(reference,life['activation_publication'],life['original_region'],activation['program'],activation['shared_body']):
            read_artifact(ref,json_document=False)


def read_ordered_loop_transfers(path):
    """Independently rederive all typed transfer rows, guards and public edges."""
    started=time.perf_counter();publication=_reference(path,SCHEMA)
    report=read_artifact(publication);active_pins=_header(report);pins=report['pins']
    with _open_lifetime(report['lifetime_publication'])as parent:
        if (report['activation_publication']!=parent.life['activation_publication']
                or report['original_region']!=parent.life['original_region']
                or report['program']!=parent.activation['program']):
            raise ValueError('ordered loop transfer has disconnected parent source/program references')
        provider=OrderedLoopTransferProvider(parent.lifetime)
        if provider.summary()!=report['summary']:raise ValueError('cold ordered loop transfer summary differs')
        counts=_read_tables(Path(publication['uri']).parent,report,provider)
        validation=_graph(provider)
        if validation!=report['graph_validation']:raise ValueError('cold ordered loop transfer graph differs')
        provider.check();read_artifact(publication,json_document=False)
        if active_pins!=_pins():raise ValueError('ordered loop transfer rules changed during independent replay')
        receipt=dict(schema=REPLAY_SCHEMA,schema_version=1,publication=publication,
            lifetime_publication=report['lifetime_publication'],activation_publication=report['activation_publication'],
            original_region=report['original_region'],program=report['program'],producer_pins=pins,
            table_counts=counts,summary=provider.summary(),graph_validation=validation,
            activation_table_counts=parent.activation_counts,activation_validation=parent.activation_validation,
            lifetime_table_counts=parent.lifetime_counts,lifetime_graph_validation=parent.lifetime_validation,
            lifetime_matched_context_edges_checked=parent.lifetime_context_edges,
            independent_cold_semantic_rederivation=True,original_body_bytes_copied=0,iterations_unrolled=False,
            runtime_invocations_created=0,global_logical_versions_created=0,
            unconditional_content_recurrence_proven=False,target_executed=False,transformation_authorized=False)
    receipt['seconds']=time.perf_counter()-started
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication',help='loop transfer directory or result.json')
    parser.add_argument('--out',help='fresh independent replay receipt JSON')
    args=parser.parse_args(argv);receipt=read_ordered_loop_transfers(args.publication)
    if args.out:
        output=Path(args.out).absolute();output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x')as stream:stream.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(schema=receipt['schema'],table_counts=receipt['table_counts'],
        graph_validation=receipt['graph_validation'],seconds=receipt['seconds'],target_executed=False),sort_keys=True))
    return 0


__all__=['publish_ordered_loop_transfers','read_ordered_loop_transfers']


if __name__=='__main__':raise SystemExit(main())
