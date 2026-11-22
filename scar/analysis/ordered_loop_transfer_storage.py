"""Publish loop transfers and independently replay ordered source relations.

Activation, lifetime and transfer checks retain their own typed tables and
source obligations. Cold readers open the original F4 session once and attach
each relation only after the preceding layer has passed its checks.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import time
from types import SimpleNamespace

from scar.analysis.inspect_ordered_activations import (
    _pins as _activation_pins, _rows as _activation_rows, validate_public_interfaces,
)
from scar.analysis.ordered_loop_transfer import OrderedLoopTransferProvider
from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
from scar.analysis.ordered_region_lifetime import (
    OrderedRegionLifetimeProvider, _pins as _lifetime_pins, SCHEMA as LIFETIME_SCHEMA,
)
from scar.analysis.source_call_view import SourceFunctionCallProvider
from scar.analysis.stateful_dependency_view import StatefulDependencyProvider, read_stateful_graph
from scar.artifacts import read_record_table, write_table
from scar.ir.execution_recipe import _reference as _checked_reference
from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact
from scar.ir.program import call_returns as W, ordered_regions as R
from scar.ir.program.dependencies import DependencyEdge, context_push
from scar.ir.record_codec import decode, encode


ACTIVATION_SCHEMA = 'scar.ordered-activation-interfaces'
SCHEMA = 'scar.ordered-loop-transfer-publication'
REPLAY_SCHEMA = 'scar.ordered-loop-transfer-independent-replay'
_LIFETIME_FIELDS = {'schema', 'schema_version', 'activation_publication', 'original_region', 'pins',
    'tables', 'summary', 'graph_validation', 'original_body_bytes_copied', 'iterations_unrolled',
    'target_executed', 'transformation_authorized'}
_FIELDS = {'schema', 'schema_version', 'lifetime_publication', 'activation_publication', 'original_region',
    'program', 'pins', 'tables', 'summary', 'graph_validation', 'original_body_bytes_copied',
    'iterations_unrolled', 'runtime_invocations_created', 'global_logical_versions_created',
    'unconditional_content_recurrence_proven', 'target_executed', 'transformation_authorized'}

_ACTIVATION_ERRORS = {
    'activation': (
        'original build reference is not its manifest',
        'independent session opened a different original program build',
        'independent ordered source replay uses another original region',
        'shared ordered body cites another original region',
        'ordered activation summary differs from independently replayed original interfaces',
        'ordered activation endpoint validation differs from independent public queries',
    ),
    'lifetime': (
        'original program reference is not its manifest',
        'ordered lifetime cold session selected another original program',
        'ordered lifetime original source replay differs',
        'ordered lifetime shared body differs from its source region',
        'ordered lifetime original activation summary differs',
        'ordered lifetime original activation graph differs',
    ),
    'transfer': (
        'loop transfer original program is not its manifest',
        'cold loop transfer session opened another original program',
        'cold loop transfer original source replay differs',
        'cold loop transfer shared body differs from its source region',
        'cold loop transfer original activation differs',
        'cold loop transfer original activation graph differs',
    ),
}


def _source_digest():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _pins():
    from scar import artifacts
    from scar.analysis import (
        normal_graph, ordered_loop_transfer, source_call_view, source_control, source_namespace_contents,
        source_observed_calls, source_references, source_value_flow,
    )
    from scar.ir import record_codec

    modules = (('transfer', ordered_loop_transfer), ('transfer_types', R),
        ('loop_interfaces', source_control), ('value_flow', source_value_flow),
        ('reference_paths', source_references), ('namespace_interfaces', source_namespace_contents),
        ('original_call_interfaces', source_call_view), ('normal_edges', normal_graph),
        ('table_writer', artifacts), ('codec', record_codec),
        ('selected_call_window', source_observed_calls), ('selected_call_window_types', W))
    digest = _source_digest()
    rules = {'storage': digest, 'lifetime_replayer': digest}
    rules.update((name, sha256(Path(module.__file__).read_bytes()).hexdigest()) for name, module in modules)
    return dict(rules=rules, parent_lifetime=_lifetime_pins())


def _report_path(path):
    path = Path(path).resolve()
    return path if path.is_file() else path / 'result.json'


def _document_reader(publication=None, report=None):
    """Reuse a decoded parent only inside this transaction and by its full receipt."""
    documents = {}

    def key(reference):
        return (reference['uri'], reference['sha256'], reference['size_bytes'], reference['schema'])

    if publication is not None:
        documents[key(publication)] = report

    def load(reference):
        reference = _checked_reference(reference)
        identity = key(reference)
        if identity not in documents:
            documents[identity] = read_artifact(reference)
        return documents[identity]

    return load


def _require_references(report, expected, load, message):
    for name, schema in expected:
        if type(report.get(name)) is not dict or report[name].get('schema') != schema:
            raise ValueError(message.format(name))
        load(report[name])


def _activation_header(report, load=read_artifact):
    if (report.get('schema'), report.get('schema_version')) != (ACTIVATION_SCHEMA, 1):
        raise ValueError('unsupported ordered activation publication')
    if report.get('pins') != _activation_pins():
        raise ValueError('ordered activation producer changed')
    if report.get('typed_table_readback') is not True:
        raise ValueError('ordered activation typed publication readback is absent')
    if any(report.get(name) is not False for name in ('independent_cold_semantic_rederivation',
            'iterations_unrolled', 'global_logical_versions_proven', 'transformation_authorized', 'target_executed')):
        raise ValueError('ordered activation publication authority flags differ')
    if any(type(report.get(name)) is not int or report[name] != 0
           for name in ('source_bodies_copied', 'runtime_invocations_created')):
        raise ValueError('ordered activation publication copied bodies or runtime invocations')
    _require_references(report, (('program', 'scar.program-build'), ('original_region', 'scar.stateful-array-region'),
        ('shared_body', 'scar.stateful-public-graph')), load, 'ordered activation lacks original {} reference')


def _lifetime_header(report, load):
    from scar.analysis.ordered_publication_versions import compatible_lifetime_pins

    if (set(report) != _LIFETIME_FIELDS or report.get('schema') != LIFETIME_SCHEMA
            or type(report.get('schema_version')) is not int or report['schema_version'] != 1):
        raise ValueError('unsupported ordered lifetime publication fields/schema')
    active = _lifetime_pins()
    if not compatible_lifetime_pins(report['pins'], active):
        raise ValueError('ordered lifetime producing rules changed')
    if any(report[name] is not False for name in ('iterations_unrolled', 'target_executed', 'transformation_authorized')):
        raise ValueError('ordered lifetime authority flags differ')
    if type(report['original_body_bytes_copied']) is not int or report['original_body_bytes_copied'] != 0:
        raise ValueError('ordered lifetime publication copied original bodies')
    _require_references(report, (('activation_publication', ACTIVATION_SCHEMA),
        ('original_region', 'scar.stateful-array-region')), load, 'ordered lifetime original reference differs: {}')
    return active


def _header(report, load):
    from scar.analysis.ordered_publication_versions import compatible_transfer_pins

    if (set(report) != _FIELDS or report.get('schema') != SCHEMA
            or type(report.get('schema_version')) is not int or report['schema_version'] != 1):
        raise ValueError('unsupported ordered loop transfer publication')
    active = _pins()
    if not compatible_transfer_pins(report['pins'], active):
        raise ValueError('ordered loop transfer producing rules changed')
    for name in ('iterations_unrolled', 'unconditional_content_recurrence_proven',
                 'target_executed', 'transformation_authorized'):
        if report[name] is not False:
            raise ValueError('ordered loop transfer authority flag differs: ' + name)
    for name in ('original_body_bytes_copied', 'runtime_invocations_created', 'global_logical_versions_created'):
        if type(report[name]) is not int or report[name] != 0:
            raise ValueError('ordered loop transfer authority count differs: ' + name)
    _require_references(report, (('lifetime_publication', LIFETIME_SCHEMA),
        ('activation_publication', ACTIVATION_SCHEMA), ('original_region', 'scar.stateful-array-region'),
        ('program', 'scar.program-build')), load, 'ordered loop original reference differs: {}')
    return active


def _compare_tables(path, tables, expected, domain_error, path_error, row_error):
    if set(tables) != set(expected):
        raise ValueError(domain_error)
    missing = object()
    counts = {}
    for name, (cls, rows) in expected.items():
        receipt = tables[name]
        if receipt.get('path') != name + '.jsonl.gz':
            raise ValueError(path_error + name)
        count = 0
        for wire, original in zip_longest(read_record_table(path, receipt), rows, fillvalue=missing):
            if wire is missing or original is missing or decode(cls, wire) != original:
                raise ValueError(row_error.format(name))
            count += 1
        counts[name] = count
    return counts


def _activation_tables(path, report, provider):
    return _compare_tables(path, report.get('tables', ()), _activation_rows(provider),
        'ordered activation table domain differs from its original interfaces',
        'ordered activation table is outside its declared local table: ',
        'ordered activation {} differs from its original source derivation')


def _lifetime_rows(provider):
    sources = dict.fromkeys(source for source, _, _, _ in provider._edges)
    edges = tuple(dict.fromkeys(edge for source in sources for edge in provider.outgoing(source).edges))
    return dict(inputs=(R.OrderedInputLifetime, provider.inputs), epochs=(R.OrderedEpochLifetime, provider.epochs),
                uses=(R.OrderedInputUse, provider.uses), edges=(DependencyEdge, edges))


def _lifetime_tables(path, report, rows):
    return _compare_tables(path, report['tables'], rows, 'ordered lifetime table domain differs',
        'ordered lifetime table local path differs: ',
        'ordered lifetime {} differs from independently replayed original dataflow')


def _edges(provider):
    points = dict.fromkeys(point for row in provider._edges
                           for point in (row.source, row.target, *provider._project(row, None)))
    edges = set()
    for point in points:
        edges.update(provider.incoming(point).edges)
        edges.update(provider.outgoing(point).edges)
    return tuple(sorted(edges, key=lambda edge: json.dumps(encode(edge), sort_keys=True, separators=(',', ':'))))


def _rows(provider):
    return dict(loops=(R.OrderedLoopInterface, provider.loops), roots=(R.OrderedLoopRootRoute, provider.root_routes),
        transfers=(R.OrderedLoopInputTransfer, provider.transfers), writes=(R.OrderedLoopCallerWrite, provider.caller_writes),
        selected_windows=(W.OrderedWrapperCallWindow, provider.selected_windows),
        frontiers=(R.OrderedLoopEffectFrontier, provider.frontiers), edges=(DependencyEdge, _edges(provider)))


def _read_tables(path, report, rows):
    return _compare_tables(path, report['tables'], rows, 'ordered loop transfer table domain differs',
        'ordered loop transfer table path differs: ',
        'ordered loop {} differs from original loop/CALL/state-path derivation')


def _edge_checker(provider, label):
    """Check both public directions, resolving each original point/kind once."""
    graph = provider.view.dependencies
    incoming, outgoing = {}, {}

    def check(edges):
        for edge in edges:
            source = edge.source, edge.kind
            if source not in outgoing:
                outgoing[source] = frozenset(graph.outgoing(edge.source, kinds=(edge.kind,)).edges)
            if edge not in outgoing[source]:
                raise ValueError(label + ' public outgoing edge differs')
            target = edge.target, edge.kind
            if target not in incoming:
                incoming[target] = frozenset(graph.incoming(edge.target, kinds=(edge.kind,)).edges)
            if edge not in incoming[target]:
                raise ValueError(label + ' public incoming edge differs')

    return check


def _lifetime_graph(provider, edges):
    check = _edge_checker(provider, 'cold lifetime')
    check(edges)
    sources = dict.fromkeys(source for source, _, _, _ in provider._edges)
    contextual = set()
    for binding in provider.activation.bindings:
        context = context_push(None, binding.ordered_frame)
        for source in sources:
            contextual.update(provider.outgoing(replace(source, context=context)).edges)
    check(contextual)
    return dict(conditional_edges=len(edges), bidirectional_missing=0), len(contextual)


def _graph(provider, edges):
    _edge_checker(provider, 'ordered loop transfer')(edges)
    definitions = sum(edge.source.context is None and edge.target.context is None for edge in edges)
    return dict(conditional_edges=len(edges), definition_edges=definitions,
                matched_call_path_edges=len(edges) - definitions, bidirectional_missing=0)


def _check_bytes(*references):
    for reference in references:
        read_artifact(reference, json_document=False)


def _activation_references(report):
    return report['program'], report['original_region'], report['shared_body']


@contextmanager
def _open_activation(reference, report, layer):
    from scar.analysis.build import BuildSession

    manifest_error, session_error, region_error, body_error, summary_error, graph_error = _ACTIVATION_ERRORS[layer]
    program_path = Path(report['program']['uri'])
    if program_path.name != 'manifest.json':
        raise ValueError(manifest_error)
    with BuildSession(program_path.parent) as session:
        if artifact_reference(session.path / 'manifest.json', 'scar.program-build') != report['program']:
            raise ValueError(session_error)
        ordered = StatefulDependencyProvider(session.view, report['original_region']['uri'])
        if ordered.reference != report['original_region']:
            raise ValueError(region_error)
        body, _ = read_stateful_graph(Path(report['shared_body']['uri']).parent, session.view, _provider=ordered)
        if body['original_region'] != report['original_region']:
            raise ValueError(body_error)
        calls = {name: provider for name, provider in session.session.providers.items()
                 if type(provider) is SourceFunctionCallProvider}
        provider = OrderedRegionActivationProvider(ordered, calls)
        if provider.summary() != report['summary']:
            raise ValueError(summary_error)
        counts = _activation_tables(Path(reference['uri']).parent, report, provider)
        validation = validate_public_interfaces(provider)
        if validation != report['validation']:
            raise ValueError(graph_error)
        yield provider, counts, validation


def _parents(reference, load, *, source_error):
    life = load(reference)
    active = _lifetime_header(life, load)
    activation = load(life['activation_publication'])
    _activation_header(activation, load)
    if activation['original_region'] != life['original_region']:
        raise ValueError(source_error)
    return life, activation, active


@contextmanager
def _open_lifetime(reference, load, layer):
    source_error = ('ordered lifetime original region differs from its activation parent' if layer == 'lifetime'
                    else 'loop transfer lifetime/activation source regions differ')
    life, activation, active = _parents(reference, load, source_error=source_error)
    with _open_activation(life['activation_publication'], activation, layer) as (parent, counts, validation):
        lifetime = OrderedRegionLifetimeProvider(parent, register=True)
        if lifetime.summary() != life['summary']:
            raise ValueError('ordered lifetime summary differs from original dataflow' if layer == 'lifetime'
                             else 'cold loop transfer original lifetime differs')
        rows = _lifetime_rows(lifetime)
        life_counts = _lifetime_tables(Path(reference['uri']).parent, life, rows)
        life_validation, context_edges = _lifetime_graph(lifetime, rows['edges'][1])
        if life_validation != life['graph_validation']:
            raise ValueError('ordered lifetime public graph validation differs' if layer == 'lifetime'
                             else 'cold loop transfer lifetime graph differs')
        yield SimpleNamespace(lifetime=lifetime, life=life, activation=activation, active_pins=active,
            activation_counts=counts, activation_validation=validation, lifetime_counts=life_counts,
            lifetime_validation=life_validation, lifetime_context_edges=context_edges)
        lifetime.check()
        _check_bytes(reference, life['activation_publication'], *_activation_references(activation))


def read_ordered_activations(path):
    """Independently rederive the typed activation interfaces and their public edges."""
    started = time.perf_counter()
    publication, report = open_artifact(_report_path(path), ACTIVATION_SCHEMA)
    _activation_header(report, _document_reader(publication, report))
    digest = _source_digest()
    pins = report['pins']
    with _open_activation(publication, report, 'activation') as (provider, counts, validation):
        provider.check()
        if pins != _activation_pins() or _source_digest() != digest:
            raise ValueError('ordered activation/replay rules changed during independent read')
        _check_bytes(publication, *_activation_references(report))
        summary = provider.summary()
    return dict(schema='scar.ordered-activation-independent-replay', schema_version=1, publication=publication,
        program=report['program'], original_region=report['original_region'], shared_body=report['shared_body'],
        producer_pins=pins, replayer_sha256=digest, table_counts=counts, summary=summary, validation=validation,
        independent_cold_semantic_rederivation=True, publication_declared_independent_cold_semantic_rederivation=False,
        original_body_bytes_copied=0, iterations_unrolled=False, runtime_invocations_created=0,
        global_logical_versions_proven=False, transformation_authorized=False, target_executed=False,
        seconds=time.perf_counter() - started)


def read_ordered_lifetimes(path):
    """Independently rederive lifetimes after replaying their activation parent."""
    started = time.perf_counter()
    publication, report = open_artifact(_report_path(path), LIFETIME_SCHEMA)
    digest = _source_digest()
    with _open_lifetime(publication, _document_reader(publication, report), 'lifetime') as parent:
        pass
    if parent.active_pins != _lifetime_pins() or _source_digest() != digest:
        raise ValueError('ordered lifetime rules changed during independent replay')
    return dict(schema='scar.ordered-input-lifetime-independent-replay', schema_version=1, publication=publication,
        activation_publication=report['activation_publication'], original_region=report['original_region'],
        program=parent.activation['program'], producer_pins=report['pins'], replayer_sha256=digest,
        table_counts=parent.lifetime_counts, activation_table_counts=parent.activation_counts, summary=report['summary'],
        graph_validation=parent.lifetime_validation, matched_context_edges_checked=parent.lifetime_context_edges,
        activation_validation=parent.activation_validation, independent_cold_semantic_rederivation=True,
        original_body_bytes_copied=0, iterations_unrolled=False, runtime_invocations_created=0,
        global_logical_versions_created=0, cross_call_content_identity_proven=False,
        external_storage_release_authorized=False, target_executed=False, transformation_authorized=False,
        seconds=time.perf_counter() - started)


def _live_parent(provider, reference):
    load = _document_reader()
    life, activation, _ = _parents(reference, load, source_error='loop transfer lifetime/activation source regions differ')
    program = load(activation['program'])
    lifetime = provider.lifetime
    if (type(provider) is not OrderedLoopTransferProvider or provider.view is not lifetime.view
            or provider.activation is not lifetime.activation or provider.ordered is not lifetime.ordered
            or life['original_region'] != provider.ordered.reference or lifetime.summary() != life['summary']
            or provider.activation.summary() != activation['summary']):
        raise ValueError('loop transfer parent is not the same source-replayed live lifetime/activation')
    # The activation was checked against the original base graph before attaching
    # the lifetime extension. Replay its tables in the same original source view.
    from scar.analysis.build import _admit_rules, _receipt
    catalog = program.get('f4', {}).get('catalog')
    if program.get('backend') != 'F4' or catalog != _receipt(provider.view.catalog.path / 'result.json'):
        raise ValueError('loop transfer parent program uses another original source catalog/rules')
    _admit_rules(program.get('rules'))
    _activation_tables(Path(life['activation_publication']['uri']).parent, activation, provider.activation)
    rows = _lifetime_rows(lifetime)
    _lifetime_tables(Path(reference['uri']).parent, life, rows)
    validation, _ = _lifetime_graph(lifetime, rows['edges'][1])
    if validation != life['graph_validation']:
        raise ValueError('loop transfer parent lifetime graph differs')
    return life, activation


def publish_ordered_loop_transfers(provider, out, *, lifetime_publication):
    """Publish an unregistered extension after checking its existing parents."""
    if type(provider) is not OrderedLoopTransferProvider:
        raise TypeError('original ordered loop transfer provider required')
    if any(candidate is provider for candidate in provider.view.dependencies.providers):
        raise ValueError('publish parent lifetime before registering its loop extension; construct with register=False')
    reference = artifact_reference(_report_path(lifetime_publication), LIFETIME_SCHEMA)
    life, activation = _live_parent(provider, reference)
    pins = _pins()
    provider.check()
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh ordered loop transfer publication required')
    out.mkdir(parents=True)
    provider.view.register(provider)
    rows = _rows(provider)
    tables = {name: write_table(out / (name + '.jsonl.gz'), values) for name, (_, values) in rows.items()}
    result = dict(schema=SCHEMA, schema_version=1, lifetime_publication=reference,
        activation_publication=life['activation_publication'], original_region=provider.ordered.reference,
        program=activation['program'], pins=pins, tables=tables, summary=provider.summary(),
        graph_validation=_graph(provider, rows['edges'][1]), original_body_bytes_copied=0, iterations_unrolled=False,
        runtime_invocations_created=0, global_logical_versions_created=0,
        unconditional_content_recurrence_proven=False, target_executed=False, transformation_authorized=False)
    _read_tables(out, result, rows)
    provider.check()
    _check_bytes(reference, life['activation_publication'], *_activation_references(activation))
    if pins != _pins():
        raise ValueError('ordered loop transfer producing rules changed during publication')
    with (out / 'result.json').open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return result


def read_ordered_loop_transfers(path):
    """Independently rederive typed transfer rows, guards and public edges."""
    started = time.perf_counter()
    publication, report = open_artifact(_report_path(path), SCHEMA)
    load = _document_reader(publication, report)
    active = _header(report, load)
    with _open_lifetime(report['lifetime_publication'], load, 'transfer') as parent:
        if (report['activation_publication'] != parent.life['activation_publication']
                or report['original_region'] != parent.life['original_region']
                or report['program'] != parent.activation['program']):
            raise ValueError('ordered loop transfer has disconnected parent source/program references')
        provider = OrderedLoopTransferProvider(parent.lifetime)
        if provider.summary() != report['summary']:
            raise ValueError('cold ordered loop transfer summary differs')
        rows = _rows(provider)
        counts = _read_tables(Path(publication['uri']).parent, report, rows)
        validation = _graph(provider, rows['edges'][1])
        if validation != report['graph_validation']:
            raise ValueError('cold ordered loop transfer graph differs')
        provider.check()
        _check_bytes(publication)
        if active != _pins():
            raise ValueError('ordered loop transfer rules changed during independent replay')
        receipt = dict(schema=REPLAY_SCHEMA, schema_version=1, publication=publication,
            lifetime_publication=report['lifetime_publication'], activation_publication=report['activation_publication'],
            original_region=report['original_region'], program=report['program'], producer_pins=report['pins'],
            table_counts=counts, summary=provider.summary(), graph_validation=validation,
            activation_table_counts=parent.activation_counts, activation_validation=parent.activation_validation,
            lifetime_table_counts=parent.lifetime_counts, lifetime_graph_validation=parent.lifetime_validation,
            lifetime_matched_context_edges_checked=parent.lifetime_context_edges,
            independent_cold_semantic_rederivation=True, original_body_bytes_copied=0, iterations_unrolled=False,
            runtime_invocations_created=0, global_logical_versions_created=0,
            unconditional_content_recurrence_proven=False, target_executed=False, transformation_authorized=False)
    receipt['seconds'] = time.perf_counter() - started
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication', help='loop transfer directory or result.json')
    parser.add_argument('--out', help='fresh independent replay receipt JSON')
    args = parser.parse_args(argv)
    receipt = read_ordered_loop_transfers(args.publication)
    if args.out:
        output = Path(args.out).absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(schema=receipt['schema'], table_counts=receipt['table_counts'],
        graph_validation=receipt['graph_validation'], seconds=receipt['seconds'], target_executed=False), sort_keys=True))
    return 0


__all__ = ['publish_ordered_loop_transfers', 'read_ordered_activations', 'read_ordered_lifetimes',
           'read_ordered_loop_transfers']


if __name__ == '__main__':
    raise SystemExit(main())
