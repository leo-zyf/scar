"""Publish F4 graph/state shards and solve the observed binding requests."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import gc
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

from scar.analysis.build_reader_versions import STATE_OWNER_PREDECESSOR, STATE_OWNER_READER
from scar.analysis.program_view import ProgramView
from scar.analysis.source_declared_catalog import read_source_catalog
from scar.analysis.source_declared_calls import _NoTargets,_check_targets
from scar.analysis.source_state import local_read_tracks
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.artifacts import _json
from scar.artifacts import write_table
from scar.analysis.program_state_view import (
    ENCODING, _counts, _pins as state_pins,
    _previous_state_pins, _rows, BindingInventory, compatible_state_pins, compatible_storage_pins, storage_pins,
)
from scar.analysis.source_consumer_parents import _stamp
from scar.analysis.source_declared_catalog import _check, _receipt
from scar.ir.program.dependencies import DependencyNeighborhood as Neighborhood
from scar.analysis.program_view_storage import pack_nodes,pack_edges
from scar.analysis.source_value_flow import _artifact
from scar.ir.record_codec import encode


_WORKER = None
OBSERVED_SCHEMA = 'scar.observed-source-binding-states'


def _pins():
    from scar.analysis import source_state as source_local_state, program_view_storage, program_view
    from scar import artifacts as artifact_io
    from scar.ir import record_codec
    return dict(inventory=sha256(Path(__file__).read_bytes()).hexdigest(),state=state_pins(),
        graph_storage=sha256(Path(program_view_storage.__file__).read_bytes()).hexdigest(),
        graph_mapping=program_view._pins(),
        wire_codec=sha256(Path(record_codec.__file__).read_bytes()).hexdigest(),
        table_output=sha256(Path(artifact_io.__file__).read_bytes()).hexdigest(),
        track_selection=sha256(Path(source_local_state.__file__).read_bytes()).hexdigest())


def _graph_contract():
    """Use the actual graph mapper's grammar, not a second version registry."""
    from scar.analysis.program_view import SOURCE_REPRESENTATION,mapping_contract
    return dict(representation=SOURCE_REPRESENTATION,mapping=mapping_contract())


def _release(catalog,view):
    if view is not None:
        view.dependencies.providers.clear();view.metadata_providers.clear()
        view.units.clear();view.namespaces.clear();view.state_providers.clear()
    keep=set(catalog.fact_store.original_source_units);seen=set()
    def release(parent):
        if id(parent) in seen:return
        seen.add(id(parent))
        for unit in tuple(parent.loaded):
            if unit not in keep:del parent.loaded[unit]
        for child in getattr(parent,'parents',()):release(child)
    release(catalog.fact_store)
    for parent in catalog.fact_store._source_parents.values():release(parent)


def _initialize_worker(catalog, out, memory_mib, context):
    global _WORKER
    sys.meta_path.insert(0, _NoTargets())
    _check_targets()
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = memory_mib * 1024**2
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    _WORKER = read_source_catalog(catalog), Path(out), context


def _collect_units(worker, work, *, jobs, catalog, out, memory_mib, context):
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker, initargs=(catalog, out, memory_mib, context)) as pool:
        for task in as_completed([pool.submit(worker, *arguments) for arguments in work]):
            yield task.result()


def _unit(unit):
    catalog,out,pins=_WORKER;started=time.monotonic();view=None;folder=out/'units'/unit
    _durable_mkdir(folder);row=catalog.fact_store.rows[unit]
    try:
        view=ProgramView(catalog);state=view.state(unit);state.content();source=state.source
        steps=sum(len(plan.steps) for plan in source.plans.values())
        if steps!=len(state.requests):raise ValueError('original operation denominator differs from state requests')
        counts=_counts(state)
        counts.update(templates=len(source.plans),original_steps=steps,
            regions=sum(len(plan.regions) for plan in source.plans.values()),
            normalized_endpoints=len(source.nodes),normalized_relations=len(source.edge_sources),
            local_read_tracks_available=len(local_read_tracks(state)),local_read_tracks_solved=len(state.tracks))
        instructions=Counter(request.rule.instruction for request in state.requests.values())
        families=Counter(request.rule.family.value for request in state.requests.values())
        pending=Counter(reason for request in state.requests.values() for reason in request.pending)
        gaps=tuple(request for request in state.requests.values() if request.missing_rules)
        table=write_table(folder/'gaps.jsonl.gz',gaps)
        if set(source.expanded)!=set(source.material.interfaces['nodes']):
            raise ValueError('local graph did not cover every declared shared template')
        graph_folder=folder/'graph';_durable_mkdir(graph_folder)
        graph_tables=dict(nodes=write_table(graph_folder/'nodes.jsonl.gz',pack_nodes(source)),
            edges=write_table(graph_folder/'edges.jsonl.gz',pack_edges(source)))
        projections=sum(len(rows) for rows in source.edge_sources.values())
        if (graph_tables['nodes']['records'],graph_tables['edges']['records'])!=(len(source.nodes),projections):
            raise ValueError('stored graph node or edge projection denominator differs')
        state_folder=folder/'state';_durable_mkdir(state_folder)
        state_tables={name:write_table(state_folder/(name+'.jsonl.gz'),rows)
            for name,(_,rows) in _rows(state).items()}
        result=dict(schema='scar.source-state-unit-coverage',schema_version=1,status='INVENTORIED',
            unit=unit,path=row['path'],namespace=encode(state.namespace),source_material=encode(state.material.result_artifact),
            counts=counts,instructions=dict(instructions),families=dict(families),open_effect_fields=dict(pending),
            gaps=table,gap_operations=len(gaps),binding_fixed_points_complete=False,
            graph=dict(contract=_graph_contract(),tables=graph_tables,templates=len(source.expanded),nodes=len(source.nodes),
                relations=len(source.edge_sources),edge_projections=projections),state_tables=state_tables,
            original_body_bytes_copied=0,loops_unrolled=False,
            target_executed=False,first_stage_complete=False)
        state.check();_check_targets()
    except Exception as error:
        result=dict(schema='scar.source-state-unit-coverage',schema_version=1,status='FAILED',unit=unit,path=row['path'],
            error_type=type(error).__name__,error=str(error),target_executed=False,first_stage_complete=False)
    if pins!=_pins():raise ValueError('coverage producer changed while a unit was running')
    result.update(seconds=time.monotonic()-started,worker_pid=os.getpid(),
        worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _durable_bytes(folder/'result.json',_json(result),replace=False)
    if view is not None:
        _release(catalog,view)
    else:_release(catalog,None)
    # Break state/content/view cycles before the next independent unit.
    if 'state' in locals():del state
    if 'source' in locals():del source
    del view;gc.collect()
    return result


def inventory(catalog,out,*,units=(),jobs=16,memory_mib=8192):
    started=time.monotonic();pins=_pins();out=Path(out).absolute()
    if out.exists():raise ValueError('fresh state-coverage output required')
    if type(jobs) is not int or jobs<1 or type(memory_mib) is not int or memory_mib<256:
        raise ValueError('positive worker count and at least 256 MiB per worker required')
    source=read_source_catalog(catalog);all_units=set(source.fact_store.rows)
    selected=all_units if not units else set(units)
    if not selected<=all_units:raise ValueError('coverage unit is outside the original catalog')
    ordered=sorted(selected,key=lambda u:(-source.fact_store.rows[u]['counts']['ports'],u))
    _durable_mkdir(out);parent=_artifact(source.path/'result.json',source.report['schema'])
    plan=dict(schema='scar.source-state-coverage-plan',schema_version=1,catalog=str(source.path),parent=encode(parent),
        scope='CATALOG' if not units else 'EXPLICIT_ORIGINAL_UNITS',catalog_units=len(all_units),
        selected_units=ordered,entry_sources=source.report.get('entry_sources'),pins=pins,
        catalog_schema=source.report['schema'],local_graph_shards=True,
        graph_contract=_graph_contract(),
        jobs=min(jobs,len(ordered) or 1),worker_memory_mib=memory_mib,
        binding_fixed_points_run=False,target_executed=False,first_stage_complete=False)
    _durable_bytes(out/'plan.json',_json(plan),replace=False);results={}
    for row in _collect_units(_unit, ((unit,) for unit in ordered), jobs=plan['jobs'],
            catalog=source.path, out=out, memory_mib=memory_mib, context=pins):
        results[row['unit']]=row
        print(json.dumps(dict(done=len(results),total=len(ordered),unit=row['unit'],status=row['status'],
            seconds=round(row['seconds'],3),gap_operations=row.get('gap_operations'),
            error_type=row.get('error_type'))),flush=True)
    if pins!=_pins() or encode(_artifact(source.path/'result.json',parent.schema))!=encode(parent):
        raise ValueError('coverage source or producer changed')
    scalar=Counter();gaps=Counter();instructions=Counter();families=Counter();failures=[]
    for unit,row in results.items():
        if row['status']=='FAILED':failures.append(dict(unit=unit,error_type=row['error_type'],error=row['error']));continue
        for key,value in row['counts'].items():
            if type(value) is int:scalar[key]+=value
        gaps.update(row['counts']['missing_address_rules']);instructions.update(row['instructions']);families.update(row['families'])
    report=dict(schema='scar.source-state-coverage',schema_version=1,status='COMPLETE_INVENTORY' if not failures else 'INCOMPLETE_INVENTORY',
        plan=plan,completed_units=len(results),inventoried_units=len(results)-len(failures),failures=failures,
        counts=dict(scalar),missing_address_rules=dict(gaps),instructions=dict(instructions),families=dict(families),
        local_model_closed=False,call_reachability_proven=False,binding_fixed_points_complete=False,
        target_executed=False,first_stage_complete=False,seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json',_json(report),replace=False)
    index_inventory(out)
    return report


def _inventory_index(path):
    """Authenticate completed shards without rebuilding any original graph.

    This also seals inventories made before the shard index was introduced.
    Their original plan, producer pins and result bytes remain untouched.
    """
    path=Path(path).absolute();raw=(path/'result.json').read_bytes();report=json.loads(raw)
    if (report['schema'],report['schema_version'])!=('scar.source-state-coverage',1):
        raise ValueError('unsupported coverage inventory')
    plan=json.loads((path/'plan.json').read_bytes())
    if plan!=report['plan'] or len(set(plan['selected_units']))!=len(plan['selected_units']):
        raise ValueError('inventory plan or original unit domain differs')
    if plan.get('local_graph_shards') and plan.get('graph_contract')!=_graph_contract():
        raise ValueError('inventory graph wire/mapping contract differs')
    actual={};counts=Counter();gaps=Counter();instructions=Counter();families=Counter()
    failures=[];peaks={};stored=0
    for unit in plan['selected_units']:
        folder=path/'units'/unit;data=(folder/'result.json').read_bytes();row=json.loads(data)
        if (row['unit']!=unit or row['schema']!='scar.source-state-unit-coverage'
                or row['schema_version']!=1 or row['target_executed'] is not False
                or row['first_stage_complete'] is not False):
            raise ValueError('inventory shard identity or scope differs')
        actual[unit]=encode(_artifact(folder/'result.json',row['schema']));stored+=len(data)
        pid=str(row['worker_pid']);peaks[pid]=max(peaks.get(pid,0),row['worker_peak_rss_kib'])
        if row['status']=='FAILED':
            failures.append(dict(unit=unit,error_type=row['error_type'],error=row['error']));continue
        if row['status']!='INVENTORIED' or row['binding_fixed_points_complete'] is not False:
            raise ValueError('inventory shard claims an unperformed analysis')
        table=row['gaps']
        if table['path']!='gaps.jsonl.gz':raise ValueError('inventory gap table path differs')
        payload=(folder/table['path']).read_bytes();stored+=len(payload)
        if (len(payload)!=table['bytes'] or sha256(payload).hexdigest()!=table['sha256']
                or table['records']!=row['gap_operations']
                or table['records']!=sum(row['counts']['missing_address_rules'].values())):
            raise ValueError('inventory gap receipt/count differs')
        if plan.get('local_graph_shards'):
            graph=row['graph']
            if (graph.get('contract')!=plan['graph_contract']
                    or row['namespace']['representation']!=plan['graph_contract']['representation']
                    or set(graph['tables'])!={'nodes','edges'} or row['loops_unrolled']is not False
                    or row['original_body_bytes_copied']!=0
                    or graph['templates']!=row['counts']['templates']
                    or graph['nodes']!=graph['tables']['nodes']['records']
                    or graph['edge_projections']!=graph['tables']['edges']['records']):
                raise ValueError('inventory local graph coverage or storage domain differs')
            for subfolder,tables in (('graph',graph['tables']),('state',row['state_tables'])):
                for name,receipt in tables.items():
                    if receipt['path']!=name+'.jsonl.gz':raise ValueError('inventory table path differs')
                    payload=(folder/subfolder/receipt['path']).read_bytes();stored+=len(payload)
                    if len(payload)!=receipt['bytes'] or sha256(payload).hexdigest()!=receipt['sha256']:
                        raise ValueError('inventory local graph/state table receipt differs')
        counts.update({k:v for k,v in row['counts'].items() if type(v)is int})
        gaps.update(row['counts']['missing_address_rules']);instructions.update(row['instructions']);families.update(row['families'])
    if (report['completed_units']!=len(actual) or report['inventoried_units']!=len(actual)-len(failures)
            or report['counts']!=dict(counts) or report['missing_address_rules']!=dict(gaps)
            or report['instructions']!=dict(instructions) or report['families']!=dict(families)
            or sorted(report['failures'],key=lambda r:r['unit'])!=sorted(failures,key=lambda r:r['unit'])
            or report['status']!=('INCOMPLETE_INVENTORY' if failures else 'COMPLETE_INVENTORY')):
        raise ValueError('inventory aggregate differs from its unit shards')
    if raw!=(path/'result.json').read_bytes():raise ValueError('inventory changed while indexing')
    result=dict(schema='scar.source-state-coverage-index',schema_version=1,
        inventory=encode(_artifact(path/'result.json',report['schema'])),
        plan=encode(_artifact(path/'plan.json',plan['schema'])),units=actual,
        graph_contract=plan.get('graph_contract'),
        stored_shard_bytes=stored,worker_peak_rss_kib=peaks,
        catalog_local_graph_materialized=bool(plan.get('local_graph_shards') and not failures
            and plan['scope']=='CATALOG' and len(actual)==plan['catalog_units']),
        producer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        semantic_rederivation=False,target_executed=False,first_stage_complete=False)
    return result


def index_inventory(path):
    """Seal all finished graph/state shards without repeating their analysis."""
    result=_inventory_index(path)
    _durable_bytes(Path(path)/'index.json',_json(result),replace=False)
    return result


def compatible_inventory_pins(recorded, active):
    if recorded == active:
        return True
    if (active.get('inventory') != STATE_OWNER_READER['program_state_coverage.py']
            or active.get('track_selection') != STATE_OWNER_READER['source_state.py']):
        return False
    previous = dict(active, inventory=STATE_OWNER_PREDECESSOR['program_state_coverage.py'],
                    track_selection=STATE_OWNER_PREDECESSOR['source_state.py'],
                    state=_previous_state_pins(active['state']))
    return recorded == previous and compatible_state_pins(previous['state'], active['state'])


def read_inventory_index(path, raw):
    actual = json.loads(raw)
    expected = _inventory_index(path)
    if actual != expected and (expected['producer_sha256'] != STATE_OWNER_READER['program_state_coverage.py']
            or actual != dict(expected, producer_sha256=STATE_OWNER_PREDECESSOR['program_state_coverage.py'])):
        raise ValueError('stored inventory index differs from its original shards')
    return actual


def read_inventory(path,*,catalog=None):
    """Admit the complete stored unit domain and receipts, not a graph walk."""
    path=Path(path).resolve();raw=(path/'index.json').read_bytes();actual=read_inventory_index(path,raw)
    plan=json.loads((path/'plan.json').read_bytes())
    if not compatible_inventory_pins(plan['pins'],_pins()):raise ValueError('inventory implementation changed')
    source=read_source_catalog(catalog if catalog is not None else plan['catalog'])
    if (str(source.path)!=plan['catalog']
            or encode(_artifact(source.path/'result.json',source.report['schema']))!=plan['parent']
            or plan['scope']=='CATALOG' and set(plan['selected_units'])!=set(source.fact_store.rows)):
        raise ValueError('inventory original catalog unit denominator differs')
    if raw!=(path/'index.json').read_bytes():raise ValueError('inventory index changed during read')
    return actual


def _observed_pins():
    from scar.analysis import source_graph_material, source_observed_interfaces, source_state
    return dict(producer=sha256(Path(__file__).read_bytes()).hexdigest(), state=state_pins(),
        selection=sha256(Path(source_observed_interfaces.__file__).read_bytes()).hexdigest(),
        materials=sha256(Path(source_graph_material.__file__).read_bytes()).hexdigest(),
        local_slots=sha256(Path(source_state.__file__).read_bytes()).hexdigest(), storage=storage_pins())


def compatible_observed_pins(actual):
    """Check used analysis rules while retaining the known historical wire formats."""
    if type(actual) is not dict:
        return False
    expected = _observed_pins()
    producer = actual.get('producer')
    if type(producer) is not str or len(producer) != 64 or any(c not in '0123456789abcdef' for c in producer):
        return False
    inputs = ('state', 'selection', 'materials', 'local_slots')
    if (any(actual.get(key) != expected[key] for key in ('selection', 'materials'))
            or not compatible_state_pins(actual.get('state'), expected['state'])
            or actual.get('local_slots') != actual['state']['state']):
        return False
    keys = {'producer', *inputs}
    return (set(actual) == keys or
            (set(actual) == keys | {'storage'} and compatible_storage_pins(actual['storage'], expected['storage'])))


def _observed_unit(unit, functions):
    catalog, out, context = _WORKER
    inventory, index_sha256, limits = context
    started = time.monotonic(); view = ProgramView(catalog); state = view.state(unit)
    static = BindingInventory(state, inventory, index_sha256)
    tracks = static.tracks(functions)
    for function, namespace in tracks:
        state.track(function, namespace, **limits)
    folder = out / 'units' / unit
    published = static.publish(folder)
    complete = sum(graph.traversal_complete and graph.dataflow_complete for graph in state.tracks)
    result = dict(unit=unit, state=_receipt(folder / 'result.json'), selected_functions=len(functions),
        selected_local_read_slots=len(tracks), completed_fixed_points=complete,
        budget_or_control_incomplete=len(tracks)-complete, counts=published['counts'],
        frontier_counts=dict(Counter(f.reason for graph in state.tracks for f in graph.frontiers)),
        shared_function_value_flows=len(state.flows), static_inventory=published['static_inventory'],
        loop_iterations_unrolled=False, runtime_value_identity_proven=False, seconds=time.monotonic()-started,
        worker_pid=os.getpid(), worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _check_targets()
    _release(catalog, view); del state, view, static; gc.collect()
    return result


def _observed_selection(index):
    actual = {(entry.invocation.process_id, entry.code) for entry in index.codes.trace.entries}
    selected, links = defaultdict(set), []
    for row in index.codes.records.values():
        captures = tuple(c for c in row.captures if (c.process_id, c.code.key) in actual)
        if captures:
            selected[row.function.unit].add(row.function)
            links.append(dict(function=encode(row.function), code_link=encode(row.id),
                function_fact=encode(row.function_fact), captured_codes=encode(captures)))
    return selected, links


def build_observed(interfaces, inventory, out, *, jobs=8, memory_mib=8192):
    from scar.analysis.program_inventory import admit_static_inventory
    from scar.analysis.source_observed_interfaces import ObservedFunctionInterfaceStore
    from scar.analysis.program_view import _digest

    started = time.monotonic(); out = Path(out).absolute(); inventory = Path(inventory).resolve(strict=True)
    if out.exists() or out.resolve() != out or not 1 <= jobs <= 32 or memory_mib < 256:
        raise ValueError('fresh canonical output, 1..32 workers and >=256 MiB required')
    store = ObservedFunctionInterfaceStore(interfaces); index = store.index
    index_receipt = _receipt(inventory / 'index.json')
    admitted = admit_static_inventory(inventory, catalog=store.catalog)
    if not admitted['catalog_local_graph_materialized']:
        raise ValueError('observed bindings require complete static source shard receipts')
    _check(index_receipt)
    selected, links = _observed_selection(index)
    if not selected.keys() <= admitted['units'].keys():
        raise ValueError('actual body selection has no static shard')
    parent = _receipt(store.path / 'result.json'); catalog = store.catalog
    limits = dict(max_states=20000, max_stack=128, max_updates=1000000)
    pins = _observed_pins()
    plan = dict(schema=OBSERVED_SCHEMA+'.plan', schema_version=1, parent=parent,
        catalog=encode(_artifact(catalog.path / 'result.json', catalog.report['schema'])),
        selection='RECORDED_FUNCTION_BODY_ALTERNATIVES_ALL_LOCAL_READ_SLOTS', functions=links,
        units=sorted(selected), run_scope=index.codes.targets.source_sha256,
        non_function_body_correspondences=sum(len(getattr(index.codes, name)) for name in
            ('scope_records', 'request_records', 'compiler_records')),
        non_function_body_state='ORIGINAL_SOURCE_WORLD_INTERFACES_NOT_LOCAL_FUNCTION_BINDING_TRACKS',
        static_inventory=index_receipt, jobs=min(jobs, len(selected) or 1), worker_memory_mib=memory_mib,
        pins=pins, limits=limits)
    _durable_mkdir(out); _durable_bytes(out / 'plan.json', _json(plan), replace=False)
    weights = {unit: json.loads(Path(admitted['units'][unit]['uri']).read_bytes())['counts']['original_steps']
               for unit in selected}
    rows = []
    work = ((unit, tuple(sorted(selected[unit], key=_digest)))
            for unit in sorted(selected, key=lambda unit: (-weights[unit], unit)))
    for row in _collect_units(_observed_unit, work, jobs=plan['jobs'], catalog=catalog.path, out=out,
            memory_mib=memory_mib, context=(inventory, index_receipt['sha256'], limits)):
        rows.append(row)
        print(json.dumps(dict(done=len(rows), total=len(selected), unit=row['unit'],
            slots=row['selected_local_read_slots'], versions=row['counts']['versions'], seconds=row['seconds'])), flush=True)
    store.validate(); _check(parent); _check(index_receipt)
    if pins != _observed_pins():
        raise ValueError('binding inventory producer dependencies changed')
    rows.sort(key=lambda row: row['unit'])
    counts, frontiers = Counter(), Counter()
    for row in rows:
        counts.update({key: value for key, value in row['counts'].items() if type(value) is int})
        frontiers.update(row['frontier_counts'])
    report = dict(schema=OBSERVED_SCHEMA, schema_version=1, plan=plan, units=rows, counts=dict(counts),
        selected_local_read_slots=sum(row['selected_local_read_slots'] for row in rows),
        completed_fixed_points=sum(row['completed_fixed_points'] for row in rows),
        budget_or_control_incomplete=sum(row['budget_or_control_incomplete'] for row in rows),
        frontier_counts=dict(frontiers), source_bodies_copied=False, loops_unrolled=False,
        actual_frame_or_tensor_contents_proven=False, opaque_writers_excluded=False,
        target_executed=False, first_stage_complete=False, seconds=time.monotonic()-started)
    _durable_bytes(out / 'result.json', _json(report), replace=False)
    return report


class SourceObservedStateProvider:
    """Admit saved state receipts and open typed unit views on demand."""

    def __init__(self,view,path,*,index=None):
        from scar.analysis.source_observed_interfaces import ObservedFunctionInterfaceStore

        self.view=view;self.path=Path(path).resolve();self.loaded={};self.headers={};self.guards={}
        self.report=r=json.loads((self.path/'result.json').read_bytes());plan=r['plan']
        if ((r['schema'],r['schema_version'])!=(OBSERVED_SCHEMA,1) or not compatible_observed_pins(plan['pins'])
                or any(r[k]is not False for k in ('source_bodies_copied','loops_unrolled',
                    'actual_frame_or_tensor_contents_proven','opaque_writers_excluded','target_executed','first_stage_complete'))):
            raise ValueError('observed state schema, implementation or authority differs')
        _check(plan['parent'])
        if json.loads((self.path/'plan.json').read_bytes())!=plan:
            raise ValueError('observed state plan differs from publication')
        artifact=_artifact(view.catalog.path/'result.json',view.catalog.report['schema'])
        if encode(artifact)!=plan['catalog']:raise ValueError('observed state uses another original catalog')
        if index is None:
            index=ObservedFunctionInterfaceStore(Path(plan['parent']['path']).parent).index
        if (index.codes.targets.source_sha256!=plan['run_scope']
                or index.view.catalog.path!=view.catalog.path):
            raise ValueError('observed state index uses another catalog or actual capture')
        selected, links = _observed_selection(index)
        units = set(selected)
        key=lambda row:json.dumps(row,sort_keys=True)
        if sorted(links,key=key)!=sorted(plan['functions'],key=key) or sorted(units)!=plan['units']:
            raise ValueError('saved binding selection differs from actual code body alternatives')
        if {row['unit'] for row in r['units']}!=units or len(r['units'])!=len(units):
            raise ValueError('observed binding unit denominator differs')
        totals=Counter()
        for row in r['units']:
            unit=row['unit'];path=self.path/'units'/unit/'result.json'
            if row['state']['path']!=str(path):raise ValueError('observed state shard path differs')
            _check(row['state']);header=json.loads(path.read_bytes())
            if ((header['schema'], header['unit']) != ('scar.source-state-view', unit)
                    or not compatible_state_pins(header['pins']) or header['schema_version'] not in (1, 2)
                    or header['parent']!=plan['catalog'] or header['counts']!=row['counts']):
                raise ValueError('observed state shard domain or implementation differs')
            if header['schema_version']==2:
                if (header.get('track_encoding') != ENCODING
                        or not compatible_storage_pins(header.get('storage_pins'))):
                    raise ValueError('observed state storage contract differs')
            self.headers[unit]=header;self.guards[path]=_stamp(path)
            totals.update({k:v for k,v in row['counts'].items() if type(v)is int})
        if dict(totals)!=r['counts']:raise ValueError('observed state aggregate differs from unit records')
        for name in ('result.json','plan.json'):self.guards[self.path/name]=_stamp(self.path/name)
        self.guards[Path(plan['parent']['path'])]=_stamp(Path(plan['parent']['path']))
        self.index=index;self.check();view.register(self)

    def check(self):
        if any(_stamp(path)!=stamp for path,stamp in self.guards.items()):
            raise ValueError('admitted observed state publication changed')
        self.index.codes._check_trace()

    def ensure(self,unit):
        self.check()
        if unit in self.loaded:return self.loaded[unit]
        from scar.analysis.source_state import StoredBindingVersions
        header=self.headers[unit];folder=self.path/'units'/unit
        summary=next(row for row in self.report['units'] if row['unit']==unit)
        state=StoredBindingVersions(self.view,folder,header,summary)
        self.loaded[unit]=state;self.guards.update(state.guards);self.check();return state

    def _state(self,point):
        source=self.view.namespaces.get(point.namespace)
        if source is None or source.unit not in self.headers:return None
        return self.ensure(source.unit)

    def incoming(self,point):
        state=self._state(point);return Neighborhood() if state is None else state.incoming(point)
    def outgoing(self,point):
        state=self._state(point);return Neighborhood() if state is None else state.outgoing(point)
    def describe(self,point):
        state=self._state(point)
        if state is None:return None
        return state.describe(point)
    def _owner(self,point):
        state=self._state(point)
        if state is None:raise ValueError('point is outside observed binding state units')
        return state
    def original(self,point):return self._owner(point).original(point)
    def expand(self,point):return self._owner(point).expand(point)
    def explain(self,point):return self._owner(point).explain(point)


def _static_main(argv):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog');parser.add_argument('--out');parser.add_argument('--index')
    parser.add_argument('--unit',action='append',default=[]);parser.add_argument('--jobs',type=int,default=16)
    parser.add_argument('--memory-mib',type=int,default=8192)
    args=parser.parse_args(argv)
    if args.index:
        if args.catalog or args.out or args.unit:parser.error('--index stands alone')
        report=index_inventory(args.index)
        print(json.dumps(dict(status='INVENTORY_SHARDS_INDEXED',units=len(report['units']),stored_shard_bytes=report['stored_shard_bytes'])))
        return
    if not args.catalog or not args.out:parser.error('--catalog and --out are required')
    report=inventory(args.catalog,args.out,units=args.unit,jobs=args.jobs,memory_mib=args.memory_mib)
    print(json.dumps({key:report[key] for key in ('status','completed_units','inventoried_units','missing_address_rules','seconds')}))


def _observed_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interfaces', required=True)
    parser.add_argument('--static-inventory', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args(argv)
    result = build_observed(args.interfaces, args.static_inventory, args.out, jobs=args.jobs)
    print(json.dumps({key: result[key] for key in ('completed_fixed_points', 'counts', 'seconds')}))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'observed':
        return _observed_main(argv[1:])
    return _static_main(argv)


if __name__ == '__main__':
    main()
