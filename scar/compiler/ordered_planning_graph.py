"""Retain one source-bound public graph through the planning boundary.

This runs inside the original BuildSession. Caller interfaces, local content
epochs and lifetimes reference the same source region that creates the CPU
domain; they are not a detached graph reconstructed from event names.
"""
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,read_artifact


def retain_ordered_planning_graph(session,region,out):
    from scar.analysis.stateful_dependency_view import StatefulDependencyProvider,publish_stateful_graph,read_stateful_graph
    from scar.analysis.source_call_view import SourceFunctionCallProvider
    from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
    from scar.analysis.inspect_ordered_activations import publish_activation_interfaces
    from scar.analysis.ordered_region_lifetime import OrderedRegionLifetimeProvider,publish_ordered_lifetimes
    from scar.analysis.ordered_loop_transfer import OrderedLoopTransferProvider
    from scar.analysis.ordered_loop_transfer_storage import publish_ordered_loop_transfers

    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh original planning graph directory required')
    out.mkdir(parents=True)
    ordered=StatefulDependencyProvider(session.view,region)
    body=publish_stateful_graph(ordered,out/'body')
    read_stateful_graph(out/'body',session.view,_provider=ordered)
    providers={name:p for name,p in session.session.providers.items() if type(p)is SourceFunctionCallProvider}
    activation=OrderedRegionActivationProvider(ordered,providers)
    interface=publish_activation_interfaces(activation,out/'interfaces',program=session.path,body_graph=out/'body')
    lifetime=OrderedRegionLifetimeProvider(activation)
    life=publish_ordered_lifetimes(lifetime,out/'lifetimes',activation_publication=out/'interfaces/result.json')
    loop_provider=OrderedLoopTransferProvider(lifetime,register=False)
    loops=publish_ordered_loop_transfers(loop_provider,out/'loop-transfers',
        lifetime_publication=out/'lifetimes/result.json')
    # The historical dependency projection is a view of the same retained
    # region. Reuse its records rather than deriving another disconnected DAG.
    dependencies=ordered.projection;dependencies['original_region']=ordered.reference
    result=dict(schema='scar.ordered-program-planning-graph',schema_version=1,
        program=artifact_reference(session.path/'manifest.json','scar.program-build'),
        original_region=ordered.reference,
        body=artifact_reference(out/'body/result.json',body['schema']),
        interfaces=artifact_reference(out/'interfaces/result.json',interface['schema']),
        lifetimes=artifact_reference(out/'lifetimes/result.json',life['schema']),
        loop_transfers=artifact_reference(out/'loop-transfers/result.json',loops['schema']),
        summary=dict(body=body['summary'],interfaces=interface['summary'],lifetimes=life['summary'],loop_transfers=loops['summary']),
        graph_validation=dict(interfaces=interface['validation'],lifetimes=life['graph_validation'],
            loop_transfers=loops['graph_validation']),
        source_bodies_copied=0,iterations_unrolled=False,target_executed=False,
        transformation_authorized=False,global_logical_versions_proven=False)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return artifact_reference(out/'result.json',result['schema']),dependencies


def admit_ordered_planning_graph(reference,*,program,region,out):
    """Consume the graph through independent original-source replay."""
    from scar.analysis.read_ordered_lifetimes import read_ordered_lifetimes
    graph=read_artifact(reference)
    expected_region=artifact_reference(region,'scar.stateful-array-region')
    if (graph.get('schema'),graph.get('schema_version'))!=('scar.ordered-program-planning-graph',1):
        raise ValueError('source-bound ordered planning graph required')
    if (graph['program']!=program or graph['original_region']!=expected_region
            or any(graph[key]is not False for key in ('iterations_unrolled','target_executed',
                'transformation_authorized','global_logical_versions_proven'))
            or type(graph['source_bodies_copied'])is not int or graph['source_bodies_copied']!=0):
        raise ValueError('planning graph source scope or authority differs')
    body=read_artifact(graph['body']);interface=read_artifact(graph['interfaces']);life=read_artifact(graph['lifetimes'])
    if (body['original_region']!=expected_region or interface['original_region']!=expected_region
            or interface['shared_body']!=graph['body'] or life['original_region']!=expected_region
            or life['activation_publication']!=graph['interfaces']):
        raise ValueError('planning graph has disconnected source/body/interface/lifetime parents')
    summary=dict(body=body['summary'],interfaces=interface['summary'],lifetimes=life['summary'])
    if 'loop_transfers'in graph:
        from scar.analysis.ordered_loop_transfer_storage import read_ordered_loop_transfers
        loop=read_artifact(graph['loop_transfers']);replay=read_ordered_loop_transfers(graph['loop_transfers']['uri'])
        if (loop['lifetime_publication']!=graph['lifetimes'] or replay['publication']!=graph['loop_transfers']
                or replay['lifetime_publication']!=graph['lifetimes']):
            raise ValueError('planning graph loop transfer has a different source lifetime')
        summary['loop_transfers']=loop['summary']
        validation=dict(interfaces=replay['activation_validation'],lifetimes=replay['lifetime_graph_validation'],
            loop_transfers=replay['graph_validation'])
    else:
        replay=read_ordered_lifetimes(graph['lifetimes']['uri'])
        if replay['publication']!=graph['lifetimes']:raise ValueError('planning graph lifetime publication differs')
        validation=dict(interfaces=replay['activation_validation'],lifetimes=replay['graph_validation'])
    if (replay['program']!=program or graph['summary']!=summary or graph['graph_validation']!=validation):
        raise ValueError('planning graph summaries or public relations differ from source replay')
    read_artifact(reference)
    result=dict(schema='scar.ordered-program-planning-graph-admission',schema_version=1,
        graph=reference,program=program,original_region=expected_region,replay=replay,
        target_executed=False,transformation_authorized=False)
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh graph admission receipt required')
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return artifact_reference(out,result['schema'])


__all__=['retain_ordered_planning_graph','admit_ordered_planning_graph']
