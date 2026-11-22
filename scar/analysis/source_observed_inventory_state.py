"""Publish and admit observed reaching definitions over the sealed static graph."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import resource
from time import monotonic

from scar.analysis.binding_track_storage import ENCODING, pins as storage_pins
from scar.analysis.program_inventory import admit_static_inventory
from scar.analysis.program_state_view import _pins as state_pins
from scar.analysis.program_state_coverage import _release
from scar.analysis.program_view import ProgramView, _digest
from scar.analysis.source_binding_inventory import BindingInventory
from scar.analysis.source_consumer_parents import _stamp
from scar.analysis.source_declared_catalog import _check, _receipt, read_source_catalog
from scar.analysis.source_observed_interfaces import ObservedFunctionInterfaceStore
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.artifacts import _json
from scar.analysis.source_value_flow import _artifact
from scar.ir.program.dependencies import DependencyNeighborhood as Neighborhood
from scar.ir.record_codec import encode


SCHEMA = 'scar.observed-source-binding-states'
_WORKER = None


def _pins():
    from scar.analysis import source_graph_material, source_observed_interfaces, source_state
    return dict(producer=sha256(Path(__file__).read_bytes()).hexdigest(), state=state_pins(),
        selection=sha256(Path(source_observed_interfaces.__file__).read_bytes()).hexdigest(),
        materials=sha256(Path(source_graph_material.__file__).read_bytes()).hexdigest(),
        local_slots=sha256(Path(source_state.__file__).read_bytes()).hexdigest(), storage=storage_pins())


def compatible_pins(actual):
    """Check used analysis rules while retaining the known historical wire formats."""
    if type(actual) is not dict:
        return False
    expected = _pins()
    producer = actual.get('producer')
    if type(producer) is not str or len(producer) != 64 or any(c not in '0123456789abcdef' for c in producer):
        return False
    inputs = ('state', 'selection', 'materials', 'local_slots')
    if any(actual.get(key) != expected[key] for key in inputs):
        return False
    keys = {'producer', *inputs}
    return (set(actual) == keys or
            (set(actual) == keys | {'storage'} and actual['storage'] == expected['storage']))


def _initialize(catalog, inventory, index_sha256, out, limits, memory_mib):
    global _WORKER
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = memory_mib * 1024**2
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    _WORKER = read_source_catalog(catalog), inventory, index_sha256, out, limits


def _unit(unit, functions):
    catalog, inventory, index_sha256, out, limits = _WORKER
    started = monotonic(); view = ProgramView(catalog); state = view.state(unit)
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
        loop_iterations_unrolled=False, runtime_value_identity_proven=False, seconds=monotonic()-started,
        worker_pid=os.getpid(), worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _release(catalog, view); del state, view, static; gc.collect()
    return result


def _selected(index):
    actual = {(entry.invocation.process_id, entry.code) for entry in index.codes.trace.entries}
    selected, links = defaultdict(set), []
    for row in index.codes.records.values():
        captures = tuple(c for c in row.captures if (c.process_id, c.code.key) in actual)
        if captures:
            selected[row.function.unit].add(row.function)
            links.append(dict(function=encode(row.function), code_link=encode(row.id),
                function_fact=encode(row.function_fact), captured_codes=encode(captures)))
    return selected, links


def build(interfaces, inventory, out, *, jobs=8, memory_mib=8192):
    started = monotonic(); out = Path(out).absolute(); inventory = Path(inventory).resolve(strict=True)
    if out.exists() or out.resolve() != out or not 1 <= jobs <= 32 or memory_mib < 256:
        raise ValueError('fresh canonical output, 1..32 workers and >=256 MiB required')
    store = ObservedFunctionInterfaceStore(interfaces); index = store.index
    index_receipt = _receipt(inventory / 'index.json')
    admitted = admit_static_inventory(inventory, catalog=store.catalog)
    if not admitted['catalog_local_graph_materialized']:
        raise ValueError('observed bindings require complete static source shard receipts')
    _check(index_receipt)
    selected, links = _selected(index)
    if not selected.keys() <= admitted['units'].keys():
        raise ValueError('actual body selection has no static shard')
    parent = _receipt(store.path / 'result.json'); catalog = store.catalog
    limits = dict(max_states=20000, max_stack=128, max_updates=1000000)
    pins = _pins()
    plan = dict(schema=SCHEMA+'.plan', schema_version=1, parent=parent,
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
    with ProcessPoolExecutor(max_workers=plan['jobs'], mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize, initargs=(catalog.path, inventory, index_receipt['sha256'], out, limits, memory_mib)) as pool:
        pending = [pool.submit(_unit, unit, tuple(sorted(selected[unit], key=_digest)))
                   for unit in sorted(selected, key=lambda unit: (-weights[unit], unit))]
        for future in as_completed(pending):
            row = future.result(); rows.append(row)
            print(json.dumps(dict(done=len(rows), total=len(pending), unit=row['unit'],
                slots=row['selected_local_read_slots'], versions=row['counts']['versions'], seconds=row['seconds'])), flush=True)
    store.validate(); _check(parent); _check(index_receipt)
    if pins != _pins():
        raise ValueError('binding inventory producer dependencies changed')
    rows.sort(key=lambda row: row['unit'])
    counts, frontiers = Counter(), Counter()
    for row in rows:
        counts.update({key: value for key, value in row['counts'].items() if type(value) is int})
        frontiers.update(row['frontier_counts'])
    report = dict(schema=SCHEMA, schema_version=1, plan=plan, units=rows, counts=dict(counts),
        selected_local_read_slots=sum(row['selected_local_read_slots'] for row in rows),
        completed_fixed_points=sum(row['completed_fixed_points'] for row in rows),
        budget_or_control_incomplete=sum(row['budget_or_control_incomplete'] for row in rows),
        frontier_counts=dict(frontiers), source_bodies_copied=False, loops_unrolled=False,
        actual_frame_or_tensor_contents_proven=False, opaque_writers_excluded=False,
        target_executed=False, first_stage_complete=False, seconds=monotonic()-started)
    _durable_bytes(out / 'result.json', _json(report), replace=False)
    return report


class SourceObservedStateProvider:
    """Admit saved state receipts and open typed unit views on demand."""

    def __init__(self,view,path,*,index=None):
        self.view=view;self.path=Path(path).resolve();self.loaded={};self.headers={};self.guards={}
        self.report=r=json.loads((self.path/'result.json').read_bytes());plan=r['plan']
        if ((r['schema'],r['schema_version'])!=(SCHEMA,1) or not compatible_pins(plan['pins'])
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
        selected, links = _selected(index)
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
            if ((header['schema'],header['unit'],header['pins'])!=
                    ('scar.source-state-view',unit,state_pins()) or header['schema_version'] not in (1,2)
                    or header['parent']!=plan['catalog'] or header['counts']!=row['counts']):
                raise ValueError('observed state shard domain or implementation differs')
            if header['schema_version']==2:
                if (header.get('track_encoding'),header.get('storage_pins'))!=(ENCODING,storage_pins()):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interfaces', required=True)
    parser.add_argument('--static-inventory', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    result = build(args.interfaces, args.static_inventory, args.out, jobs=args.jobs)
    print(json.dumps({key: result[key] for key in ('completed_fixed_points', 'counts', 'seconds')}))


if __name__ == '__main__':
    main()
