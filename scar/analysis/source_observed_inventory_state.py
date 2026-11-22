"""Publish observed reaching definitions over the same frontend's static inventory."""

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

from scar.analysis.program_inventory import admit_static_inventory
from scar.analysis.program_state_coverage import _release
from scar.analysis.program_view import ProgramView, _digest
from scar.analysis.source_binding_inventory import BindingInventory
from scar.analysis.source_declared_catalog import _check, _receipt, read_source_catalog
from scar.analysis.source_observed_interfaces import ObservedFunctionInterfaceStore
from scar.analysis.source_observed_state import SCHEMA, _pins
from scar.analysis.source_stage import _durable_bytes, _durable_mkdir
from scar.analysis.source_template_links import _json
from scar.analysis.source_value_flow import _artifact
from scar.ir.record_codec import encode


_WORKER = None


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
    producer = Path(__file__)
    pins = dict(_pins(), producer=sha256(producer.read_bytes()).hexdigest())
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
    if pins != dict(_pins(), producer=sha256(producer.read_bytes()).hexdigest()):
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
