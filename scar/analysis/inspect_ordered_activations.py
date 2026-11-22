"""Build shared ordered bodies and parametric matched caller interfaces."""
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import time

from scar.analysis.build import BuildSession
from scar.analysis.normal_output import write_table
from scar.analysis.source_symbol_variants import _table
from scar.analysis.stateful_dependency_view import StatefulDependencyProvider,publish_stateful_graph,read_stateful_graph
from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
from scar.analysis.source_call_view import SourceFunctionCallProvider
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program import ordered_regions as A, ordered_regions as I, ordered_regions as J, view as V
from scar.ir.program.dependencies import context_push
from scar.ir.record_codec import decode


def _pins():
    from scar.analysis import ordered_region_activation,ordered_region_inputs,ordered_region_inline,stateful_dependency_view,source_call_view
    return {name:sha256(Path(module.__file__).read_bytes()).hexdigest()for name,module in (
        ('publisher',__import__(__name__,fromlist=['__file__'])),('activation',ordered_region_activation),
        ('inputs',ordered_region_inputs),('activation_types',A),('input_types',I),
        ('inline',ordered_region_inline),('inline_types',J),
        ('ordered_body',stateful_dependency_view),('original_calls',source_call_view))}


def _rows(provider):
    witnesses={}
    for route in provider.inputs:
        for read in route.read_chain:
            content=provider.view.state(read.content_access.unit).content(complete=False)
            witness=A.OrderedContentAccessWitness(read.content_access,content.accesses[read.content_access.ordinal])
            witnesses[witness.reference]=witness
    return dict(bindings=(A.OrderedActivationBinding,provider.bindings),
        routes=(A.OrderedActivationRoute,provider.routes),inputs=(I.OrderedInputRoute,provider.inputs),
        input_boundaries=(I.OrderedInputBoundary,provider.input_boundaries),
        boundary_nodes=(V.ProgramNode,provider.nodes.values()),
        boundary_records=(A.OrderedBoundaryPort,provider.records.values()),
        inline_routes=(J.OrderedInlineRoute,provider.inline.routes),
        inline_boundaries=(J.OrderedInlineBoundary,provider.inline.boundaries),
        original_content_accesses=(A.OrderedContentAccessWitness,witnesses.values()))


def validate_public_interfaces(provider):
    """Query one abstract family per original selection; never unroll calls."""
    expected=set();observed=set();frontiers=Counter()
    for binding_index,binding in enumerate(provider.bindings):
        contexts=provider._contexts(binding,None)
        for route in provider.routes:
            if route.binding==binding_index:expected.add(provider._edge(route,contexts))
        context=context_push(None,binding.ordered_frame)
        points=(provider.ordered.entry,*provider.ordered.points.values(),*provider.ports.values())
        for original in points:
            point=replace(original,context=context)
            for query in (provider.view.dependencies.incoming,provider.view.dependencies.outgoing):
                result=query(point)
                observed.update(edge for edge in result.edges
                    if edge.source.namespace==provider.ordered.namespace or edge.target.namespace==provider.ordered.namespace)
                frontiers.update(f.kind for f in result.frontiers if 'INLINED_SOURCE' in f.kind)
    observed.update(expected)
    missing=[]
    for edge in observed:
        if edge not in provider.view.dependencies.outgoing(edge.source,kinds=(edge.kind,)).edges:missing.append('OUTGOING')
        if edge not in provider.view.dependencies.incoming(edge.target,kinds=(edge.kind,)).edges:missing.append('INCOMING')
    if missing:raise ValueError('ordered interface is not bidirectional: '+repr(Counter(missing)))
    return dict(projected_edges=len(observed),parametric_routes=len(expected),bidirectional_missing=0,
        frontier_queries=dict(frontiers),abstract_context_prefix='OPEN_NOT_A_RUNTIME_STACK',
        iterations_unrolled=False,runtime_invocation_proven=False)


def publish_activation_interfaces(provider,out,*,program,body_graph):
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh ordered activation report required')
    out.mkdir(parents=True);pins=_pins();provider.check();tables={}
    for name,(_,values)in _rows(provider).items():
        tables[name]=write_table(out/(name+'.jsonl.gz'),values)
    # A same-session typed readback preserves sparse parent-table citations.
    # Independent consumers must reconstruct the original source ABI/provider;
    # these rows alone are not a source or runtime-dispatch certificate.
    sentinel=object()
    for name,(cls,expected)in _rows(provider).items():
        actual=(decode(cls,row)for row in _table(out,tables[name]))
        if any(a!=b for a,b in zip_longest(actual,expected,fillvalue=sentinel)):
            raise ValueError('ordered activation table differs: '+name)
    validation=validate_public_interfaces(provider)
    provider.check()
    if pins!=_pins():raise ValueError('ordered activation producer changed during publication')
    report=dict(schema='scar.ordered-activation-interfaces',schema_version=1,
        program=artifact_reference(Path(program)/'manifest.json','scar.program-build'),
        shared_body=artifact_reference(Path(body_graph)/'result.json','scar.stateful-public-graph'),
        original_region=provider.ordered.reference,pins=pins,summary=provider.summary(),tables=tables,
        validation=validation,typed_table_readback=True,independent_cold_semantic_rederivation=False,
        source_bodies_copied=0,iterations_unrolled=False,runtime_invocations_created=0,
        global_logical_versions_proven=False,transformation_authorized=False,target_executed=False)
    (out/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    from scar.compiler.stateful_frontend import discover_stateful_regions
    from scar.compiler.stateful_model import write_stateful
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True);parser.add_argument('--region');parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);out=Path(args.out).resolve()
    if out.exists():raise ValueError('fresh ordered activation graph catalog required')
    started=time.perf_counter();out.mkdir(parents=True)
    with BuildSession(args.build)as session:
        if args.region:paths=[Path(args.region).resolve()];rejected=[]
        else:
            models,rejected=discover_stateful_regions(session.view);paths=[]
            for index,model in enumerate(models):
                folder=out/('region-'+str(index));folder.mkdir()
                path=folder/'original-region.json';write_stateful(model,path);paths.append(path)
        providers={name:p for name,p in session.session.providers.items()if type(p)is SourceFunctionCallProvider}
        rows=[]
        for index,path in enumerate(paths):
            folder=out/('region-'+str(index));folder.mkdir(exist_ok=True)
            ordered=StatefulDependencyProvider(session.view,path)
            publish_stateful_graph(ordered,folder/'body')
            read_stateful_graph(folder/'body',session.view,_provider=ordered)
            provider=OrderedRegionActivationProvider(ordered,providers)
            report=publish_activation_interfaces(provider,folder/'interfaces',program=session.path,body_graph=folder/'body')
            rows.append(dict(artifact=artifact_reference(folder/'interfaces/result.json',report['schema']),
                summary=report['summary'],validation=report['validation'],
                table_bytes=sum(row['bytes']for row in report['tables'].values())))
        result=dict(schema='scar.ordered-activation-graph-catalog',schema_version=1,
            program=artifact_reference(session.path/'manifest.json','scar.program-build'),regions=rows,
            source_profile_boundaries=rejected,automatic_region_discovery=args.region is None,
            seconds=time.perf_counter()-started,target_executed=False,first_stage_complete=False)
        (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(regions=len(rows),summaries=[r['summary']for r in rows],
            seconds=result['seconds'],target_executed=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
