"""Bind source-function origins to original parameters and returns.

Ordinary declarations may select a normal-flow branch. Explicit compiled
units retain their exec-context namespace and remain outside the import
provider index. Both scopes share the same signature and table rules.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import sys
import time

from scar.analysis import source_declared_calls as declared
from scar.analysis.source_declared_calls import DeclaredCallStore, _NoTargets, _check_targets, _folder, _flat_tables
from scar.analysis.source_control import ControlNamespaceStore
from scar.analysis.source_field_interfaces import _InterfaceOnly
from scar.analysis.source_function_connections import Invocation, connect
from scar.analysis.source_references import source_port
from scar.analysis.source_world_returns import _table_ref
from scar.analysis.source_value_flow import _artifact
from scar.artifacts import read_record_table as _table
from scar.artifacts import write_table
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.artifacts import _json
from scar.ir.program import function_sites as R, name_flow as N
from scar.ir.program import normal_graph as T, source_productions as P
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.source-function-interfaces'
NO_PROOF = ('source_scopes_activated', 'caller_contexts_selected', 'runtime_dispatch_proven',
    'current_function_environment_versions_proven', 'effects_proven', 'source_relations_rederived',
    'namespace_semantics_rederived', 'loops_unrolled', 'target_executed', 'first_stage_complete')
COVERAGE = 'ALL_FUNCTION_DEFINITION_ORIGINS_IN_SELECTED_RETAINED_CALLEE_QUERY_BRANCHES'
TABLES = ('queries', 'interfaces', 'frontiers')
_WORKER = None


class FunctionInterfaceParents:
    def __init__(self, calls, *, control=None):
        _check_targets()
        self.calls = calls if isinstance(calls, DeclaredCallStore) else DeclaredCallStore(calls)
        self.control = None; self.control_parent = None
        if control is not None:
            self.control = control if isinstance(control, ControlNamespaceStore) else ControlNamespaceStore(control)
            if (self.control.parent.path != self.calls.path or self.control.report['parent'] !=
                    encode(_artifact(self.calls.path/'result.json', self.calls.report['schema']))):
                raise ValueError('source function interfaces use a different control/declaration parent')
            # Both strict readers established the exact same parent receipt.
            # Share its catalog and caches instead of retaining a second copy.
            self.control.parent = self.calls
            self.control_parent = _artifact(self.control.path/'result.json', self.control.report['schema'])
        self.parent = _artifact(self.calls.path/'result.json', self.calls.report['schema'])

    def validate(self):
        self.calls.validate()
        if self.control is not None:
            if (self.control.raw != (self.control.path/'result.json').read_bytes()
                    or self.control_parent != _artifact(self.control.path/'result.json', self.control.report['schema'])):
                raise ValueError('source function normal-control parent changed')
            from scar.analysis.source_control import _pins
            if self.control.report['pins'] != _pins(): raise ValueError('source function normal-control rules changed')

    def project(self, unit):
        """Rebind retained origins; the parent's namespace proof is not replayed.

        Optional control queries are separately rederived and select only their
        recorded normal-view branch. The original declaration and all its
        frontiers remain available through each output query's parent index.
        """
        controlled = self.control is not None and unit in self.control.report['shards']
        original = self.calls.load_unit(unit); references = {}
        if controlled:
            graph = self.control.read_unit(unit)
            shard = self.control.report['shards'][unit]; folder = self.control.path/shard['path']
            for ordinal, query in enumerate(graph.queries):
                if (not 0 <= query.query < len(original.queries) or query.query in references
                        or query.callee != original.queries[query.query].callee):
                    raise ValueError('controlled callee differs from original source call')
                references[query.query] = (query.reference, _table_ref(folder, shard['tables']['queries'],
                    'scar.control-qualified-source-query', ordinal), query.conditions)
        source_shard = self.calls.report['shards'][unit]
        table = (self.calls.path / source_shard['path'], source_shard['tables']['queries'],
                 'scar.declared-source-call-query')
        result = _project_interfaces(original, self.calls.runtime.resolver.slicer, table, references)
        self.validate(); _check_targets()
        return result


def _project_interfaces(original, slicer, declaration_table, references, *, context_guards=()):
    """Connect an admitted declaration table to the original function definitions."""
    unit = original.unit; caller = slicer.store.load(unit); offset = slicer.store.offsets[unit]
    interfaces = []; frontiers = []; queries = []; functions = {}; origins = 0
    for qi, query in enumerate(original.queries):
        declared = _table_ref(*declaration_table, qi)
        reference, evidence, branch = references.get(qi, (query.reference,
            declared, ()))
        call = caller.calls[qi]
        invocation = Invocation(offset+call.callee, tuple(offset+p for p in call.positional),
            tuple((name, offset+p) for name,p in call.keywords), offset+call.result,
            offset+call.world, offset+call.result_world,
            None if call.positional_pack is None else offset+call.positional_pack,
            None if call.keyword_pack is None else offset+call.keyword_pack)
        origin_indices = []; interface_indices = []
        for oi, origin in enumerate(reference.origins):
            if origin.kind is not N.NamespaceOriginKind.FUNCTION_DEFINITION: continue
            origins += 1; origin_indices.append(oi); terminal = reference.states[origin.state]
            if terminal.selectors: raise ValueError('source function origin retains pending attributes')
            if terminal.port not in functions:
                provider, local = slicer.local(terminal.port)
                matches = [(i, f) for i, f in enumerate(provider.functions) if f.value == local]
                if len(matches) != 1: raise ValueError('source function origin is not one original function definition')
                index, function = matches[0]
                mini = _InterfaceOnly(slicer, provider, function)
                functions[terminal.port] = (mini, mini.offsets[provider.unit] + function.value,
                                            slicer.fact(provider, 'functions', index))
            mini, function_point, definition = functions[terminal.port]
            if origin.evidence != (definition,): raise ValueError('source function definition evidence differs')
            connection = connect(mini, function_point, invocation, conditions=(
                'RECORDED_SOURCE_CALLEE_REFERENCE_PATH_AND_ITS_PRECONDITIONS_SELECTED',
                'CURRENT_CALLABLE_IS_RECORDED_SOURCE_FUNCTION_WITH_COMPATIBLE_CALL_PROTOCOL',
                'FUNCTION_CODE_CAPTURE_DEFAULT_AND_NAMESPACE_VERSIONS_MATCH_DECLARATION',
                *context_guards,
                *origin.conditions,
                *branch))
            if connection is None:
                frontiers.append(R.SourceFunctionInterfaceFrontier(qi, oi, evidence,
                    'SOURCE_SIGNATURE_REQUIRES_DIFFERENT_ARGUMENT_OR_DESCRIPTOR_BINDING')); continue
            mode = ('SOURCE_FUNCTION' if connection.activation is T.FunctionActivation.NORMAL
                else 'CREATE_DEFERRED_BODY')
            link = T.CallLink(source_port(caller, call.result), source_port(caller, call.callee),
                connection.source_callable, connection.body, mode, None, connection.parameters,
                connection.returns, connection.conditions, argument_packs=connection.argument_packs,
                projections=connection.projections, binding=connection.binding)
            interface_indices.append(len(interfaces))
            interfaces.append(R.SourceFunctionInterface(qi, oi, query.call, definition, evidence, connection.activation, link))
            if connection.activation is not T.FunctionActivation.NORMAL:
                frontiers.append(R.SourceFunctionInterfaceFrontier(qi, oi, evidence,
                    'DEFERRED_CREATION_HAS_NO_ORDINARY_BODY_DATA_OR_WORLD_RETURN'))
        if not origin_indices:
            frontiers.append(R.SourceFunctionInterfaceFrontier(qi, None, evidence,
                'NO_RECORDED_SOURCE_FUNCTION_ORIGIN_OTHER_CALLEE_ALTERNATIVES_REMAIN_OPEN'))
        if reference.frontiers:
            frontiers.append(R.SourceFunctionInterfaceFrontier(qi, None, evidence,
                'PARENT_CALLEE_REFERENCE_FRONTIERS_REMAIN_OPEN'))
        queries.append(R.SourceFunctionInterfaceQuery(qi, query.call, declared, evidence, query.owners,
            tuple(origin_indices), tuple(interface_indices), tuple(range(len(query.reference.frontiers))),
            tuple(range(len(reference.frontiers)))))
    return R.SourceFunctionInterfaceUnit(unit, len(original.queries), origins, tuple(references),
        tuple(interfaces), tuple(frontiers), tuple(queries))


def _counts(graph):
    return dict(calls=graph.calls, function_origins=graph.function_origins, controlled_queries=len(graph.controlled_queries),
        query_indexes=len(graph.queries),
        declared_namespace_frontiers=sum(len(q.declared_frontiers) for q in graph.queries),
        retained_namespace_frontiers=sum(len(q.reference_frontiers) for q in graph.queries),
        interfaces=len(graph.interfaces), source_functions=len({r.link.source_callable for r in graph.interfaces}),
        connected_calls=len({r.query for r in graph.interfaces}),
        unconnected_calls=graph.calls-len({r.query for r in graph.interfaces}),
        binding_kinds=dict(Counter(r.link.binding.value for r in graph.interfaces)),
        activation_kinds=dict(Counter(r.activation.value for r in graph.interfaces)),
        parameter_routes=sum(len(r.link.parameters) for r in graph.interfaces),
        projection_routes=sum(len(r.link.projections) for r in graph.interfaces),
        data_returns=sum(sum(target == r.link.result for _, target in r.link.returns) for r in graph.interfaces),
        state_returns=sum(sum(target != r.link.result for _, target in r.link.returns) for r in graph.interfaces),
        frontiers=len(graph.frontiers), frontier_reasons=dict(Counter(f.reason for f in graph.frontiers)))


def _totals(shards):
    scalar = Counter(); dictionaries = {k:Counter() for k in ('binding_kinds', 'activation_kinds', 'frontier_reasons')}
    for shard in shards.values():
        for key,value in shard['counts'].items():
            if key in dictionaries: dictionaries[key].update(value)
            else: scalar[key] += value
    # source_functions counts per-unit unique targets, not globally unique functions.
    scalar['source_function_unit_pairs'] = scalar.pop('source_functions', 0)
    return dict(units=len(shards), **scalar, **{k:dict(v) for k,v in dictionaries.items()})


def _pins():
    from scar.analysis import source_function_connections, source_field_interfaces
    from scar.analysis import source_references as source_reference_slice
    from scar import artifacts as artifact_io
    from scar.analysis import source_world_returns, source_value_flow
    from scar.ir import record_codec
    return dict(producer=sha256(Path(__file__).read_bytes()).hexdigest(), **{
        name:sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in (
            ('types', R), ('binder', source_function_connections), ('interface_adapter', source_field_interfaces),
            ('target_types', T), ('production_types', P), ('namespace_types', N),
            ('references', source_reference_slice), ('table_references', source_world_returns),
            ('material_receipts', source_value_flow), ('output', artifact_io),
            ('tables', artifact_io), ('codec', record_codec))})


def _selected(parent, units=None):
    if isinstance(units, str): units = (units,)
    requested = tuple(units or ())
    if any(type(u) is not str or u not in parent['shards'] for u in requested):
        raise ValueError('source function interface unit is absent from declared calls')
    return tuple(sorted(set(requested))) if requested else tuple(parent['shards'])


def _denominator(parent, shards):
    parent_calls = sum(s['counts']['calls'] for s in parent['shards'].values())
    selected_calls = sum(s['calls'] for s in shards.values())
    return dict(parent_units=len(parent['shards']), selected_units=len(shards),
        unselected_units=len(parent['shards'])-len(shards), parent_calls=parent_calls,
        selected_calls=selected_calls, unselected_calls=parent_calls-selected_calls,
        selected_declared_function_origins=sum(parent['shards'][u]['counts']['origin_kinds'].get(
            'FUNCTION_DEFINITION', 0) for u in shards))


def _write_interface_tables(folder, graph):
    return {name: write_table(folder / (name + '.jsonl.gz'), getattr(graph, name)) for name in TABLES}


def _write(parents, unit, out, pins):
    started = time.monotonic(); graph = parents.project(unit); folder = Path(out)/_folder(unit)
    _durable_mkdir(folder)
    tables = _write_interface_tables(folder, graph)
    if pins != _pins(): raise ValueError('source function interface rules changed')
    parents.calls.runtime.clear()
    return dict(path=_folder(unit), calls=graph.calls, function_origins=graph.function_origins,
        controlled_queries=list(graph.controlled_queries), tables=tables, counts=_counts(graph), seconds=time.monotonic()-started)


def _initialize(calls, control, out, pins):
    global _WORKER
    sys.meta_path.insert(0, _NoTargets()); _check_targets()
    _WORKER = FunctionInterfaceParents(calls, control=control), out, pins


def _worker(unit):
    parents,out,pins = _WORKER
    return unit,_write(parents,unit,out,pins)


def build_function_interfaces(calls, out, *, control=None, jobs=4, units=None):
    started = time.monotonic(); pins = _pins(); out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh source function interface output required')
    if type(jobs) is not int or jobs < 1: raise ValueError('positive source function interface jobs required')
    parents = FunctionInterfaceParents(calls, control=control)
    selected = _selected(parents.calls.report, units)
    selection = dict(mode='EXPLICIT_UNITS' if units else 'ALL_PARENT_UNITS', units=list(selected))
    _durable_mkdir(out)
    if jobs == 1:
        shards = {u:_write(parents,u,out,pins) for u in selected}
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(selected)), mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
                initargs=(str(parents.calls.path), None if parents.control is None else str(parents.control.path), str(out), pins)) as pool:
            shards = dict(pool.map(_worker, selected, chunksize=1))
    parents.validate()
    if pins != _pins(): raise ValueError('source function interface rules changed during build')
    report = dict(schema=SCHEMA, schema_version=1, calls=str(parents.calls.path), parent=encode(parents.parent),
        control=None if parents.control is None else str(parents.control.path),
        control_parent=None if parents.control_parent is None else encode(parents.control_parent), pins=pins,
        shards=shards,tables=_flat_tables(shards),counts=_totals(shards),jobs=jobs,seconds=time.monotonic()-started,
        selection=selection, denominator=_denominator(parents.calls.report, shards), coverage=COVERAGE,
        scope='CONDITIONAL_SIGNATURE_INTERFACES_NOT_SELECTED_CALLERS_OR_RUNTIME_DISPATCH',
        parent_query_indexes_include_unconnected_calls_and_frontiers=True,
        **{name:False for name in NO_PROOF})
    _durable_bytes(out/'result.json',_json(report),replace=False)
    return report


class SourceFunctionInterfaceStore:
    """Material-only load_unit; read_unit replays only the interface projection.

    The declared namespace graph is loaded with its original receipts/ABI;
    read_unit does not certify that parent's namespace relations anew.
    """
    def __init__(self,path):
        self.path=Path(path).absolute();self.raw=(self.path/'result.json').read_bytes();self.report=r=json.loads(self.raw)
        if (r['schema'],r['schema_version'])!=(SCHEMA,1) or r['pins']!=_pins():
            raise ValueError('source function interface schema or rules differ')
        self.parents=FunctionInterfaceParents(r['calls'],control=r['control'])
        if (r['parent']!=encode(self.parents.parent) or r['control_parent']!=(None if self.parents.control_parent is None
                else encode(self.parents.control_parent))): raise ValueError('source function interface parents differ')
        selection=r['selection']
        if selection['mode'] not in ('ALL_PARENT_UNITS', 'EXPLICIT_UNITS'):
            raise ValueError('source function interface selection mode differs')
        selected=_selected(self.parents.calls.report,
            None if selection['mode']=='ALL_PARENT_UNITS' else selection['units'])
        if (selection['units']!=list(selected) or tuple(r['shards'])!=selected or r['tables']!=_flat_tables(r['shards'])
                or r['counts']!=_totals(r['shards']) or any(r[n] is not False for n in NO_PROOF)
                or r['coverage']!=COVERAGE or r['denominator']!=_denominator(self.parents.calls.report,r['shards'])
                or r['scope']!='CONDITIONAL_SIGNATURE_INTERFACES_NOT_SELECTED_CALLERS_OR_RUNTIME_DISPATCH'
                or r['parent_query_indexes_include_unconnected_calls_and_frontiers'] is not True):
            raise ValueError('source function interface coverage, receipts or proof status differ')
        for unit,shard in r['shards'].items():
            source=self.parents.calls.report['shards'][unit]
            if (shard['path']!=_folder(unit) or shard['calls']!=source['counts']['calls']
                    or set(shard['tables'])!=set(TABLES)
                    or any(v['path']!=name+'.jsonl.gz' for name,v in shard['tables'].items())):
                raise ValueError('source function interface original unit or tables differ')
        self.catalog=self.parents.calls.catalog; self.snapshot=self.parents.calls.snapshot
        self.validate()

    def validate(self):
        if self.raw!=(self.path/'result.json').read_bytes() or self.report['pins']!=_pins():
            raise ValueError('source function interface report/rules changed')
        self.parents.validate()

    def load_unit(self,unit):
        """Check stored bytes/types/counts/indexes; no namespace or ABI rebinding."""
        self.validate();shard=self.report['shards'][unit];folder=self.path/shard['path']
        parent = self.parents.calls.report['shards'][unit]
        table = (self.parents.calls.path / parent['path'], parent['tables']['queries'],
                 'scar.declared-source-call-query')
        graph = _read_interface_tables(unit, folder, shard['tables'], shard['counts'],
                                       shard['controlled_queries'], table)
        self.validate()
        return graph

    def read_unit(self,unit):
        """Rebind this selected unit against retained parent queries/original facts."""
        graph=self.load_unit(unit)
        if graph!=self.parents.project(unit): raise ValueError('source function interfaces differ from original callee candidates/ABI')
        self.validate(); _check_targets(); return graph


def _read_interface_tables(unit, folder, tables, counts, controlled_queries, declaration_table):
    def rows(name, kind):
        return tuple(decode(kind, wire, record_tags=P.PRODUCTION_RECORD_TAGS)
                     for wire in _table(folder, tables[name]))

    graph = R.SourceFunctionInterfaceUnit(unit, counts['calls'], counts['function_origins'],
        tuple(controlled_queries), rows('interfaces', R.SourceFunctionInterface),
        rows('frontiers', R.SourceFunctionInterfaceFrontier), rows('queries', R.SourceFunctionInterfaceQuery))
    if _counts(graph) != counts: raise ValueError('source function interface unit counts differ')
    controlled = set(graph.controlled_queries)
    if (tuple(q.query for q in graph.queries) != tuple(range(graph.calls))
            or sum(len(q.function_origins) for q in graph.queries) != graph.function_origins
            or len(controlled) != len(graph.controlled_queries)
            or any(not 0 <= q < graph.calls for q in controlled)):
        raise ValueError('source function interface query coverage differs')
    indexed=defaultdict(list)
    for i,interface in enumerate(graph.interfaces): indexed[interface.query].append(i)
    for query in graph.queries:
        if (query.declared != _table_ref(*declaration_table, query.query)
                or (query.query not in controlled and query.reference != query.declared)
                or not query.owners or any(o.unit!=unit for o in query.owners)
                or query.interfaces!=tuple(indexed[query.query])):
            raise ValueError('source function interface parent query/index differs')
    for interface in graph.interfaces:
        if not 0<=interface.query<graph.calls:
            raise ValueError('source function interface lies outside its original calls')
        query=graph.queries[interface.query]
        if (interface.origin not in query.function_origins or interface.call!=query.call
                or interface.reference!=query.reference or interface.link.runtime_dispatch_verified is not False
                or interface.link.source_callable is None or interface.link.body is None
                or (interface.activation is T.FunctionActivation.NORMAL and interface.link.mode!='SOURCE_FUNCTION')
                or (interface.activation is T.FunctionActivation.CREATE_DEFERRED_BODY
                    and (interface.link.mode!='CREATE_DEFERRED_BODY' or interface.link.returns))
                or interface.activation not in (T.FunctionActivation.NORMAL,T.FunctionActivation.CREATE_DEFERRED_BODY)):
            raise ValueError('source function interface candidate/activation differs')
    for frontier in graph.frontiers:
        if (not 0<=frontier.query<graph.calls or frontier.reference!=graph.queries[frontier.query].reference
                or (frontier.origin is not None and frontier.origin not in graph.queries[frontier.query].function_origins)):
            raise ValueError('source function interface frontier index differs')
    return graph


def read_function_interfaces(path, *, unit):
    store=SourceFunctionInterfaceStore(path)
    return store.report,store.read_unit(unit)


COMPILED_SCHEMA = 'scar.source-compiled-function-interfaces'
COMPILED_UNIT_SCHEMA = COMPILED_SCHEMA+'-unit'
COMPILED_SCOPE = 'COMPILED_EXEC_CONTEXT'
COMPILED_TABLES = ('declared_queries', 'symbols', 'queries', 'interfaces', 'frontiers')
COMPILED_NO_PROOF = ('source_scopes_activated', 'caller_contexts_selected', 'runtime_dispatch_proven',
    'current_exec_globals_proven', 'current_function_environment_versions_proven',
    'historical_target_relations_rederived', 'effects_proven', 'source_bodies_copied',
    'loops_unrolled', 'target_executed', 'first_stage_complete')


def _compiled_pins():
    from scar.analysis import source_graph_material, source_world_flow as source_world_material, source_relations as source_relation_selection
    signature_rules = _pins()
    return dict(producer=signature_rules['producer'], declaration_rules=declared._pins(),
                signature_rules=signature_rules, **{
            name:sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in (
                ('compiled_material',source_graph_material),('material_parents',source_world_material),
                ('material_selection',source_relation_selection))})


def _selected_compiled(catalog, units):
    if isinstance(units,str):units=(units,)
    units=tuple(units or ())
    if not units or any(type(u) is not str or u not in catalog.fact_store.original_source_units
            or catalog.fact_store.rows[u].get('facts_format')!='COMPILE_CONTEXT_F4_JSONL' for u in units):
        raise ValueError('explicit original compiled exec-context unit identities required')
    return tuple(sorted(set(units)))


def _compiled_declarations(runtime, unit, limits):
    _selected_compiled(runtime.catalog, (unit,))
    return declared._project_admitted_calls(runtime.snapshot, unit, runtime.resolver, limits)


def _compiled_interfaces(runtime, declarations, folder, receipt):
    table = (folder, receipt, 'scar.compiled-declared-source-call-query')
    graph = _project_interfaces(declarations, runtime.resolver.slicer, table, {}, context_guards=(
        'CURRENT_FRAME_GLOBALS_SELECT_RECORDED_COMPILE_CONTEXT_DICT',
        'COMPILED_EXEC_CONTEXT_DECLARATION_DOES_NOT_ESTABLISH_SCOPE_ENTRY',
    ))
    runtime.snapshot.validate_parents(); _check_targets()
    return graph


def _compiled_counts(graph, declarations):
    return dict(_counts(graph), namespace=declared._counts(declarations))


def _compiled_totals(shards):
    result=_totals({u:dict(counts={k:v for k,v in s['counts'].items() if k!='namespace'})
        for u,s in shards.items()})
    result['namespace']=declared._totals({u:dict(counts=s['counts']['namespace']) for u,s in shards.items()})
    return result


def _compiled_unit_report(runtime, unit, source_templates, tables, counts, pins, limits):
    row=runtime.catalog.fact_store.rows[unit]
    return dict(schema=COMPILED_UNIT_SCHEMA,schema_version=1,scope=COMPILED_SCOPE,unit=unit,
        catalog_parent=encode(runtime.parent),source_templates=encode(source_templates),
        original_source_unit=row['original_source_unit'],source=row['path'],module=row['module'],
        limits=limits,pins=pins,tables=tables,counts=counts,
        namespace_reference_rules_rederived=True,source_signature_rules_rederived=True,
        **{name:False for name in COMPILED_NO_PROOF})


def build_compiled_interfaces(catalog, out, *, units, max_states=4000, max_selectors=32):
    """Publish only explicit compiled units; never widen the import provider index."""
    declared._check_targets();started=time.monotonic();out=Path(out).absolute()
    if out.exists():raise ValueError('fresh compiled function interface output required')
    limits=declared._limits(max_states,max_selectors);pins=_compiled_pins();runtime=declared._Runtime(catalog)
    from scar.analysis.source_graph_material import CompiledGraphMaterial

    selected=_selected_compiled(runtime.catalog,units);_durable_mkdir(out);shards={}
    for unit in selected:
        begin=time.monotonic();material=CompiledGraphMaterial(runtime.catalog,unit)
        declarations=_compiled_declarations(runtime,unit,limits);folder=out/declared._folder(unit);_durable_mkdir(folder)
        tables={name:write_table(folder/(name+'.jsonl.gz'),rows) for name,rows in (
            ('declared_queries',declarations.queries),('symbols',declarations.symbols))}
        graph=_compiled_interfaces(runtime,declarations,folder,tables['declared_queries'])
        tables.update(_write_interface_tables(folder, graph))
        counts=_compiled_counts(graph,declarations);report=_compiled_unit_report(runtime,unit,material.result_artifact,tables,counts,pins,limits)
        material.validate_parents()
        if _compiled_pins()!=pins:raise ValueError('compiled interface producer changed')
        _durable_bytes(folder/'result.json',_json(report),replace=False)
        shards[unit]=dict(path=declared._folder(unit),tables=tables,counts=counts,
            source_templates=report['source_templates'],result=encode(_artifact(folder/'result.json',COMPILED_UNIT_SCHEMA)),
            seconds=time.monotonic()-begin)
        runtime.clear()
    runtime.snapshot.validate_parents()
    if _compiled_pins()!=pins:raise ValueError('compiled interface producer changed')
    report=dict(schema=COMPILED_SCHEMA,schema_version=1,scope=COMPILED_SCOPE,catalog=str(runtime.catalog.path),
        catalog_parent=encode(runtime.parent),selection=dict(mode='EXPLICIT_COMPILED_UNITS',units=list(selected)),
        limits=limits,pins=pins,shards=shards,tables=declared._flat_tables(shards),counts=_compiled_totals(shards),
        denominator=dict(parent_catalog_units=len(runtime.catalog.fact_store.rows),
            parent_compiled_units=len(runtime.catalog.fact_store.original_source_units),selected_units=len(selected),
            selected_calls=sum(runtime.catalog.fact_store.rows[u]['counts']['calls'] for u in selected)),
        namespace_reference_rules_rederived=True,source_signature_rules_rederived=True,
        import_provider_index_unchanged=True,parent_query_indexes_include_unconnected_calls_and_frontiers=True,
        seconds=time.monotonic()-started,**{name:False for name in COMPILED_NO_PROOF})
    _durable_bytes(out/'result.json',_json(report),replace=False);return report


class CompiledFunctionInterfaceStore:
    """Native compiled schema; strict unit reads replay namespace and ABI rules."""
    def __init__(self,path,catalog=None):
        self.path=Path(path).absolute();self.raw=(self.path/'result.json').read_bytes();self.report=r=json.loads(self.raw)
        if (r['schema'],r['schema_version'],r['scope'])!=(COMPILED_SCHEMA,1,COMPILED_SCOPE) or r['pins']!=_compiled_pins():
            raise ValueError('compiled function interface schema or producer differs')
        self.runtime=declared._Runtime(r['catalog'] if catalog is None else catalog)
        self.catalog=self.runtime.catalog;self.snapshot=self.runtime.snapshot;self.materials={}
        self._cached_unit = None
        if str(self.catalog.path)!=r['catalog'] or encode(self.runtime.parent)!=r['catalog_parent']:
            raise ValueError('compiled function interface original catalog differs')
        self.units=_selected_compiled(self.catalog,r['selection']['units']);declared._limits(**r['limits'])
        denominator=dict(parent_catalog_units=len(self.catalog.fact_store.rows),
            parent_compiled_units=len(self.catalog.fact_store.original_source_units),selected_units=len(self.units),
            selected_calls=sum(self.catalog.fact_store.rows[u]['counts']['calls'] for u in self.units))
        if (r['selection']!=dict(mode='EXPLICIT_COMPILED_UNITS',units=list(self.units))
                or tuple(r['shards'])!=self.units or r['tables']!=declared._flat_tables(r['shards'])
                or r['counts']!=_compiled_totals(r['shards']) or r['denominator']!=denominator
                or any(r[n] is not False for n in COMPILED_NO_PROOF)
                or any(r[n] is not True for n in ('namespace_reference_rules_rederived','source_signature_rules_rederived',
                    'import_provider_index_unchanged','parent_query_indexes_include_unconnected_calls_and_frontiers'))):
            raise ValueError('compiled interface selection, coverage or proof status differs')
        for unit,shard in r['shards'].items():
            if (shard['path']!=declared._folder(unit) or set(shard['tables'])!=set(COMPILED_TABLES)
                    or any(v['path']!=name+'.jsonl.gz' for name,v in shard['tables'].items())
                    or shard['counts']['calls']!=self.catalog.fact_store.rows[unit]['counts']['calls']):
                raise ValueError('compiled interface original unit or table metadata differs')
        self.validate()

    def validate(self):
        if self.raw!=(self.path/'result.json').read_bytes() or self.report['pins']!=_compiled_pins():
            raise ValueError('compiled interface report or rules changed')
        self.snapshot.validate_parents()
        for material in self.materials.values():material.validate_parents()

    def _load(self, unit):
        from scar.analysis.source_consumer_parents import _stamp
        from scar.analysis.source_graph_material import CompiledGraphMaterial

        self.validate()
        if unit not in self.report['shards']:raise ValueError('compiled interface unit was not selected')
        shard=self.report['shards'][unit];folder=self.path/shard['path']
        if encode(_artifact(folder/'result.json',COMPILED_UNIT_SCHEMA))!=shard['result']:
            raise ValueError('compiled interface unit report receipt differs')
        if unit not in self.materials:self.materials[unit]=CompiledGraphMaterial(self.catalog,unit)
        material=self.materials[unit]
        expected=_compiled_unit_report(self.runtime,unit,material.result_artifact,shard['tables'],
            shard['counts'],self.report['pins'],self.report['limits'])
        raw=(folder/'result.json').read_bytes()
        if json.loads(raw)!=expected or shard['source_templates']!=encode(material.result_artifact):
            raise ValueError('compiled interface unit provenance differs')

        def stamps():
            result = []
            for name in COMPILED_TABLES:
                path = folder / shard['tables'][name]['path']
                if path.resolve() != path or not path.is_file():
                    raise ValueError('compiled interface table is absent or traverses a symlink')
                result.append(_stamp(path))
            return tuple(result)

        before = stamps()
        key = unit, raw, before
        # Calls and their namespace declarations share one admitted shard.
        # Keep one decoded unit so adjacent readers avoid reopening all tables.
        cached = self._cached_unit
        if cached is not None and cached[0] == key:
            return cached[1]

        declarations = declared._read_declaration_tables(self.runtime, unit, folder, shard['tables'],
                                                         query_table='declared_queries')
        counts = {name: count for name, count in shard['counts'].items() if name != 'namespace'}
        table = (folder, shard['tables']['declared_queries'], 'scar.compiled-declared-source-call-query')
        graph = _read_interface_tables(unit, folder, shard['tables'], counts, (), table)
        if declared._counts(declarations) != shard['counts']['namespace'] or graph.calls != len(declarations.queries):
            raise ValueError('compiled interface declaration count differs')
        for indexed, declaration in zip(graph.queries, declarations.queries, strict=True):
            origin_indices = tuple(i for i, origin in enumerate(declaration.reference.origins)
                                   if origin.kind is N.NamespaceOriginKind.FUNCTION_DEFINITION)
            frontier_indices = tuple(range(len(declaration.reference.frontiers)))
            if (indexed.call != declaration.call or indexed.owners != declaration.owners
                    or indexed.function_origins != origin_indices or indexed.declared_frontiers != frontier_indices
                    or indexed.reference_frontiers != frontier_indices):
                raise ValueError('compiled interface original declaration indexes differ')
        if before != stamps(): raise ValueError('compiled interface tables changed during reading')
        if raw!=(folder/'result.json').read_bytes():raise ValueError('compiled unit report changed during reading')
        self.validate()
        value = graph, declarations
        self._cached_unit = key, value
        return value

    def load_unit(self,unit):
        """Receipts/types/counts/original call checks; namespace/connection rules not replayed."""
        return self._load(unit)[0]

    def load_declarations(self,unit):
        """Retain complete original namespace candidates and frontier reasons."""
        return self._load(unit)[1]

    def read_unit(self,unit):
        graph,recorded=self._load(unit);shard=self.report['shards'][unit];folder=self.path/shard['path']
        expected=_compiled_declarations(self.runtime,unit,self.report['limits'])
        if recorded!=expected:raise ValueError('compiled namespace references differ from original source rules')
        if graph!=_compiled_interfaces(self.runtime,expected,folder,shard['tables']['declared_queries']):
            raise ValueError('compiled function interfaces differ from original callee candidates and ABI')
        self.validate();declared._check_targets();return graph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sources=parser.add_mutually_exclusive_group()
    sources.add_argument('--calls');sources.add_argument('--catalog')
    parser.add_argument('--control');parser.add_argument('--out')
    parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--read')
    parser.add_argument('--unit',action='append',default=[])
    parser.add_argument('--max-states',type=int,default=4000)
    parser.add_argument('--max-selectors',type=int,default=32)
    args=parser.parse_args()
    sys.meta_path.insert(0,_NoTargets()); _check_targets()
    if args.read:
        if not args.unit:parser.error('--read requires --unit for explicit source interface rederivation')
        schema=json.loads((Path(args.read)/'result.json').read_bytes())['schema']
        if schema == COMPILED_SCHEMA:
            store=CompiledFunctionInterfaceStore(args.read,catalog=args.catalog)
            units=_selected_compiled(store.catalog,args.unit)
            result=dict(status='COMPILED_FUNCTION_INTERFACES_REDERIVED',target_executed=False)
        else:
            store=SourceFunctionInterfaceStore(args.read)
            units=_selected(store.report,args.unit)
            result=dict(status='SOURCE_FUNCTION_INTERFACES_REDERIVED',
                namespace_semantics_rederived=False,source_relations_rederived=False)
        result['units']={unit:_counts(store.read_unit(unit)) for unit in units}
    elif args.catalog:
        if not args.out or not args.unit:parser.error('--catalog requires --unit and --out')
        if args.control:parser.error('--control requires --calls')
        report=build_compiled_interfaces(args.catalog,args.out,units=args.unit,
            max_states=args.max_states,max_selectors=args.max_selectors)
        result=dict(status='COMPILED_FUNCTION_INTERFACES_BUILT',counts=report['counts'],
            seconds=report['seconds'],target_executed=False)
    else:
        if not args.calls or not args.out:parser.error('--calls and --out required')
        report=build_function_interfaces(args.calls,args.out,control=args.control,jobs=args.jobs,units=args.unit)
        result=dict(status='SOURCE_FUNCTION_INTERFACES_BUILT',counts=report['counts'],
            denominator=report['denominator'],seconds=report['seconds'])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
