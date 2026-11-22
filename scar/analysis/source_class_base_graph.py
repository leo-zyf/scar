"""Source class base graph."""

import argparse
import json
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from scar.analysis.class_bindings import project_class_bases
from scar.analysis.source_context_query import ContextTargetGraph
from scar.analysis.source_references import source_port, ReferenceSlicer
from scar.artifacts import (
    read_record_table as _table,
    write_table,
    _durable_bytes,
    _durable_mkdir,
    _json,
)
from scar.analysis.source_value_flow import _artifact
from scar.ir.program import (
    class_ports as B,
    class_ports as C,
    source_productions as P,
    normal_graph as T,
)
from scar.ir.record_codec import encode, decode
from scar.ir.program.source_value_routes import SourceBodyCall


# Store ordered source base operands and conditional namespace provenance.
#
# Class roots are analysis queries into a retained source catalog. Selecting a
# query never establishes program entry, a live class object or runtime MRO.
def _catalog(path):
    catalog = ContextTargetGraph(path)
    def validate():
        if _artifact(catalog.path/'result.json', catalog.report['schema']) != artifact:
            raise ValueError('class base catalog changed')
        catalog.fact_store.validate_parents(); catalog.interface_store.validate_parents()
    artifact = _artifact(catalog.path/'result.json', catalog.report['schema'])
    return catalog, SimpleNamespace(fact_store=catalog.fact_store, validate_parents=validate), artifact


def _pins():
    from scar.analysis import class_bindings as source_class_bases, source_namespace_contents as source_namespace_references, source_address_paths
    from scar.ir.program import name_flow as N
    return {name: sha256(Path(path).read_bytes()).hexdigest() for name, path in (
        ('producer', __file__), ('bases', source_class_bases.__file__), ('types', B.__file__),
        ('namespaces', source_namespace_references.__file__), ('namespace_types', N.__file__),
        ('addresses', source_address_paths.__file__))}


def _counts(graph):
    namespaces = [slot.namespace for slot in graph.slots if slot.namespace is not None]
    return dict(declarations=len(graph.declarations), slots=len(graph.slots),
        packed_declarations=sum(d.bases_pack is not None for d in graph.declarations),
        no_explicit_bases=sum(d.no_explicit_bases for d in graph.declarations),
        slots_with_class_candidates=sum(bool(s.source_classes) for s in graph.slots),
        class_candidates=sum(len(s.source_classes) for s in graph.slots),
        namespace_states=sum(len(g.states) for g in namespaces), namespace_edges=sum(len(g.edges) for g in namespaces),
        namespace_origins=dict(Counter(o.kind.value for g in namespaces for o in g.origins)),
        frontiers=len(graph.frontiers), frontier_reasons=dict(Counter(f.reason for f in graph.frontiers)))


def build_class_base_graph(catalog, source_classes, out):
    started=time.monotonic(); pins=_pins(); out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh class base output required')
    catalog, snapshot, parent=_catalog(catalog)
    graph=project_class_bases(snapshot, source_classes); snapshot.validate_parents()
    if pins!=_pins(): raise ValueError('class base rules changed')
    _durable_mkdir(out)
    tables={name:write_table(out/(name+'.jsonl.gz'),getattr(graph,name)) for name in ('declarations','slots','frontiers')}
    report=dict(schema='scar.source-class-base-graph',schema_version=1,status='ORDERED_BASE_OPERANDS_AND_NAMESPACE_SOURCES',
        catalog=str(catalog.path),parent=encode(parent),pins=pins,tables=tables,
        source_classes=[encode(d.source_class) for d in graph.declarations],counts=_counts(graph),
        roots_are_query_selection_not_entry_admission=True,current_namespace_versions_proven=False,
        runtime_bases_proven=False,mro_proven=False,target_executed=False,first_stage_complete=False,
        seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json',_json(report),replace=False);return report


def read_class_base_graph(path):
    path=Path(path).absolute();raw=(path/'result.json').read_bytes();report=json.loads(raw)
    if (report['schema'],report['schema_version'])!=('scar.source-class-base-graph',1) or report['pins']!=_pins():
        raise ValueError('class base producer differs')
    _,snapshot,parent=_catalog(report['catalog'])
    if report['parent']!=encode(parent):raise ValueError('class base original catalog differs')
    graph=B.ClassBaseGraph(**{name:tuple(decode(cls,row,record_tags=P.PRODUCTION_RECORD_TAGS)
        for row in _table(path,report['tables'][name])) for name,cls in
        (('declarations',B.ClassBaseDeclaration),('slots',B.ClassBaseSlot),('frontiers',B.ClassBaseFrontier))})
    roots=tuple(decode(T.GlobalPort,row) for row in report['source_classes'])
    if graph!=project_class_bases(snapshot,roots) or _counts(graph)!=report['counts']:
        raise ValueError('class operands or namespace reference sources differ from original facts')
    if (report['roots_are_query_selection_not_entry_admission'] is not True or any(report[k] is not False for k in (
            'current_namespace_versions_proven','runtime_bases_proven','mro_proven','target_executed','first_stage_complete'))):
        raise ValueError('class base source queries cannot prove runtime identity or inheritance')
    snapshot.validate_parents()
    if raw!=(path/'result.json').read_bytes():raise ValueError('class base graph changed during verification')
    return report,graph


def query_classes_from_inputs(path):
    """Read query roots, without treating their selection as entry evidence."""
    path=Path(path).absolute();raw=(path/'result.json').read_bytes();report=json.loads(raw)
    if report['schema']=='scar.constructor-class-input-projection':name='roots'
    elif report['schema']=='scar.constructor-member-contexts':name='class_roots'
    else:raise ValueError('expected constructor class input query selection')
    roots=tuple(decode(C.ConstructorClassInputRoot,row,record_tags=P.PRODUCTION_RECORD_TAGS)
        for row in _table(path,report['tables'][name]))
    if raw!=(path/'result.json').read_bytes():raise ValueError('class query selection changed')
    return tuple(dict.fromkeys(value.source_class for root in roots for value in root.context.class_inputs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog');parser.add_argument('--inputs');parser.add_argument('--out');parser.add_argument('--read')
    parser.add_argument('--class-value',nargs=2,action='append',metavar=('UNIT','LOCAL_PORT'),default=[])
    args=parser.parse_args()
    if args.read:report,_=read_class_base_graph(args.read)
    else:
        if not args.catalog or not args.out:parser.error('--catalog and --out required')
        roots=list(query_classes_from_inputs(args.inputs)) if args.inputs else []
        if args.class_value:
            catalog=ContextTargetGraph(args.catalog)
            roots.extend(source_port(catalog.fact_store.load(unit),int(local)) for unit,local in args.class_value)
        if not roots:parser.error('--inputs or --class-value required')
        report=build_class_base_graph(args.catalog,roots,args.out)
    print(json.dumps(dict(status=report['status'],counts=report['counts'],seconds=report['seconds'])),flush=True)


# Project original constructor interfaces without rebinding calls or bodies.
#
# Construction ports retain their original call/class anchors and roles. Source
# new results need not be fresh allocations, and initializer data is discarded.
# Only the exact recorded source connection may provide a guarded body entry.
_PENDING = frozenset((T.ConstructionStage.NEW_LOOKUP_PENDING,
    T.ConstructionStage.INIT_LOOKUP_PENDING, T.ConstructionStage.RESULT_TYPE_OPEN))
_SOURCE = frozenset((T.ConstructionStage.SOURCE_NEW, T.ConstructionStage.SOURCE_INIT))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _site(slicer, identity, record):
    _require(type(identity) is str and bool(identity) and type(record) is SourceBodyCall,
        'constructor site needs its original source unit and typed call record')
    unit, *_ = slicer.unit(identity)
    prefix = slicer.fact(unit, 'calls', 0).locator.rsplit('/', 1)[0] + '/'
    suffix = record.fact.locator.removeprefix(prefix)
    _require(record.fact.locator.startswith(prefix) and suffix.isascii() and suffix.isdecimal(),
        'constructor site must cite its original unit calls table')
    ordinal = int(suffix)
    _require(str(ordinal) == suffix and 0 <= ordinal < len(unit.calls),
        'constructor call locator differs from its canonical original ordinal')
    call = unit.calls[ordinal]
    _require(record.fact == slicer.fact(unit, 'calls', ordinal) and record.call == call,
        'constructor site changed its original source fact or artifact receipt')
    return unit, call


def _endpoint(slicer, point, result, source_class):
    if type(point) is T.GlobalPort:
        slicer.local(point)
    else:
        _require(type(point) is T.ConstructionPort and type(point.role) is T.ConstructionPortKind
            and point.call == result and point.source_class == source_class,
            'constructor endpoint changed its original call/class anchor or role')


def _stage(slicer, unit, call, link):
    result, callee = source_port(unit, call.result), source_port(unit, call.callee)
    before, after = source_port(unit, call.world), source_port(unit, call.result_world)
    _require(link.result == result and link.callee == callee and type(link.stage) is T.ConstructionStage,
        'construction stage differs from its original source call or callee')
    owner, local = slicer.local(link.source_class)
    _require(any(cls.value == local for cls in owner.classes),
        'construction source class is not an original source class definition')
    new_result = T.ConstructionPort(result, link.source_class, T.ConstructionPortKind.NEW_RESULT)
    new_world = T.ConstructionPort(result, link.source_class, T.ConstructionPortKind.NEW_WORLD)
    for point in (*link.inputs, *link.outputs):
        _endpoint(slicer, point, result, link.source_class)
    if link.stage in (T.ConstructionStage.NEW_LOOKUP_PENDING, T.ConstructionStage.OBJECT_NEW,
                      T.ConstructionStage.SOURCE_NEW):
        inputs, outputs = (link.source_class, before), (new_result, new_world)
    elif link.stage in (T.ConstructionStage.INIT_LOOKUP_PENDING, T.ConstructionStage.RESULT_TYPE_OPEN):
        inputs, outputs = (new_result, new_world), ()
    elif link.stage is T.ConstructionStage.EXPLICIT_OBJECT_NEW:
        _require(len(call.positional) == 1 and not call.keywords
            and call.positional_pack is None and call.keyword_pack is None,
            'explicit object allocation lost its original single-class argument layout')
        inputs, outputs = (source_port(unit, call.positional[0]), before), (result, after)
    else:
        inputs, outputs = (new_result, new_world), (result, after)
    _require(link.inputs == inputs and link.outputs == outputs,
        'constructor stage input/output endpoints differ from the original protocol ABI')
    _require(link.stage in _SOURCE or link.connection is None,
        'non-source constructor stage cannot supply a source function connection')
    return new_result, new_world


def _source_entry(slicer, unit, call, link, new_result, new_world):
    connection = link.connection
    _require(type(connection) is T.FunctionConnection,
        'source constructor stage requires its original function connection')
    provider, local = slicer.local(connection.source_callable)
    functions = [fn for fn in provider.functions if fn.value == local]
    _require(len(functions) == 1, 'constructor source callable is absent or ambiguous in original facts')
    fn = functions[0]
    _require(connection.body == T.Body(provider.unit, fn.template, fn.body)
        and len(fn.formals) == len(fn.environment.parameters)+2
        and len(fn.parameter_names) == len(fn.defaults) == len(fn.environment.parameters),
        'constructor source body or original formal/default interface differs')
    formals = tuple(source_port(provider, point) for point in fn.formals)
    expected_activation = (T.FunctionActivation.NORMAL if fn.environment.kind is P.DeferredBodyKind.FUNCTION
        else T.FunctionActivation.CREATE_DEFERRED_BODY)
    _require(connection.activation is expected_activation,
        'constructor connection confused ordinary body entry and deferred body creation')
    new = link.stage is T.ConstructionStage.SOURCE_NEW
    required = ('NEW_RECEIVES_SOURCE_CLASS',) if new else (
        'NEW_RESULT_IS_REQUESTED_CLASS_INSTANCE', 'INIT_DATA_RETURN_DISCARDED')
    _require(set(required) <= set(link.conditions) <= set(connection.conditions),
        'constructor source connection lost its original stage/type-call guards')
    for route in connection.parameters:
        allowed = (formals[0],) if route.kind is T.RouteKind.FUNCTION_ENVIRONMENT else (
            (formals[-1],) if route.kind is T.RouteKind.STATE else formals[1:-1])
        _require(route.formal in allowed, 'constructor route is not an original function formal of this kind')
        for point in route.actuals:
            _endpoint(slicer, point, link.result, link.source_class)
    environment = [route for route in connection.parameters if route.kind is T.RouteKind.FUNCTION_ENVIRONMENT]
    state = [route for route in connection.parameters if route.kind is T.RouteKind.STATE]
    _require(len(environment) == 1 and environment[0].actuals == (connection.source_callable,),
        'constructor environment must use the selected original function, not the class-call callee')
    world = source_port(unit, call.world) if new else new_world
    _require(len(state) == 1 and state[0].actuals == (world,),
        'constructor source body receives a different original protocol world')
    if new:
        _require(any(link.source_class in route.actuals and route.kind in (
            T.RouteKind.ARGUMENT, T.RouteKind.CLASS_RECEIVER, T.RouteKind.PACK_POSITIONALS)
            for route in connection.parameters), 'source new does not receive this original source class')
    else:
        _require(any(new_result in route.actuals and route.kind in (
            T.RouteKind.ARGUMENT, T.RouteKind.CLASS_RECEIVER, T.RouteKind.PACK_POSITIONALS)
            for route in connection.parameters), 'initializer receiver does not come from this constructor NEW_RESULT')
    packs = connection.argument_packs
    positional = None if call.positional_pack is None else source_port(unit, call.positional_pack)
    keywords = None if call.keyword_pack is None else source_port(unit, call.keyword_pack)
    _require((packs is not None) == (positional is not None or keywords is not None),
        'constructor connection lost its original evaluated argument packs')
    if packs is not None:
        _require(packs.positional == positional and packs.keywords == keywords,
            'constructor argument packs differ from the original source call')
        for point in (packs.receiver, packs.class_receiver):
            if point is not None: _endpoint(slicer, point, link.result, link.source_class)
    for projection in connection.projections:
        _require(projection.formal in formals[1:-1] and projection.pack in (positional, keywords),
            'constructor projection changed its original parameter or evaluated source pack')
        _endpoint(slicer, projection.pack, link.result, link.source_class)
    for source, target in connection.returns:
        _endpoint(slicer, source, link.result, link.source_class)
        _endpoint(slicer, target, link.result, link.source_class)
    if connection.activation is not T.FunctionActivation.NORMAL:
        _require(not connection.returns, 'deferred constructor creation cannot claim ordinary body returns')
        return 'DEFERRED_BODY_CREATION_IS_NOT_BODY_ENTRY'
    expected = ((source_port(provider, fn.returns), new_result),) if new and fn.returns is not None else ()
    if fn.return_world is not None:
        expected = (*expected, (source_port(provider, fn.return_world),
            new_world if new else source_port(unit, call.result_world)))
    _require(connection.returns == expected,
        'constructor source return crossed NEW_RESULT/NEW_WORLD or included discarded initializer data')
    if fn.return_world is None or new and fn.returns is None:
        return 'CONSTRUCTOR_SOURCE_NORMAL_RETURN_INTERFACE_UNRESOLVED'
    return None


def project_constructor_interfaces(snapshot, sites):
    """Project all recorded class/stage alternatives in one constructions scan.

    Site ordinals retain the supplied order, including repeated source sites.
    Every step keeps its original historical evidence and connection. No call
    is rebound, no completion is added, and no source body is copied or run.
    The caller owns the snapshot's final parent-guard validation.
    """
    slicer = ReferenceSlicer(snapshot); records = []; selected_sites = defaultdict(list)
    wanted = set(); wanted_units = set()
    for identity, record in sites:
        unit, call = _site(slicer, identity, record)
        index = len(records); records.append(C.ConstructorInterfaceSite(identity, record))
        result = source_port(unit, call.result)
        selected_sites[result].append((index, unit, call))
        wanted.add(_json(encode(result))); wanted_units.add(identity)
    steps = []; entries = []; frontiers = []
    for _, link, evidence in snapshot.rows('constructions', lambda wire:
            wire['result']['unit'] in wanted_units and _json(wire['result']) in wanted, T.ConstructionLink):
        for site, unit, call in selected_sites[link.result]:
            new_result, new_world = _stage(slicer, unit, call, link)
            step = len(steps); steps.append(C.ConstructorInterfaceStep(site, evidence, link))
            if link.stage in _PENDING:
                frontiers.append(C.ConstructorInterfaceFrontier(step, link.stage.value+'_UNRESOLVED'))
            elif link.stage in _SOURCE:
                reason = _source_entry(slicer, unit, call, link, new_result, new_world)
                if reason is None:
                    entries.append(C.ConstructorBodyEntry(step, link.connection))
                else:
                    frontiers.append(C.ConstructorInterfaceFrontier(step, reason))
    return C.ConstructorInterfaceGraph(tuple(records), tuple(steps), tuple(entries), tuple(frontiers))


if __name__=='__main__':main()
