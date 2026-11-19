"""Observe the admitted sparse loop with bounded local CPython events."""
from __future__ import annotations

import builtins
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from threading import get_ident
from types import CodeType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.call_binding import target_metadata
from scar.trace.code_snapshot import code_key, describe_code


SCHEMA = 'scar.study-aggregate-source-scope'
DESIGN = 'scar.aggregate-ordered-source-scope-design'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _marked(report, pointer):
    _require(re.fullmatch(r'#/observed_targets/[0-9]+', pointer) is not None, 'invalid F4 observation pointer')
    return report['observed_targets'][int(pointer.rsplit('/', 1)[1])]['observation']


def scope_design(anchor_digest_package, argv):
    """Decode the established call sites, parameter roles and aggregate contract."""
    from scar.codegen.native_ordered_sparse_export import SCHEMA as PACKAGE, check_native_ordered_export

    _require(type(argv) is list and len(argv) == 1 and type(argv[0]) is str
        and re.fullmatch('[1-9][0-9]{0,8}', argv[0]) is not None and int(argv[0]) > 1,
        'expected one positive bounded repetition argument for the ordered source scope')
    package = Path(anchor_digest_package).resolve(strict=True)
    manifest = check_native_ordered_export(package)
    _require(manifest['schema_version'] == 5, 'expected the admitted standalone digest family')
    detached = read_artifact({**manifest['detached_source'], 'uri': str(package/'detached-source.json')})
    source = read_artifact(detached['source'])
    source_scope = read_artifact(source['source_scope'])
    reference = source_scope['observed']['exact_count_evidence']
    original = read_artifact(reference)
    report = read_artifact(source['source_graph'])
    observations = [row['observation'] for row in report['observed_targets'] if row['entry_scope_reachable']]
    targets = {}
    for name, key in original['code_keys'].items():
        matches = [row for row in observations if (row['target_code'] or {}).get('key') == key]
        _require(len(matches) == 1, 'expected one original observed frame per scoped counter')
        targets[name] = dict(code=matches[0]['target_code'], caller=matches[0]['caller_site'])
    body, = (name for name, row in targets.items() if row['code']['key'] == detached['body_code_key'])
    children = sorted((name for name in targets if name != body), key=lambda name: targets[name]['caller']['offset'])
    _require(children and all(targets[name]['caller']['code'] == detached['body_code_key'] for name in children),
        'ordered row preparation and solve must be original same-iteration calls')
    iterator, = (row for row in observations if row['caller_site']['code'] == detached['body_code_key']
        and json.loads(row['metadata_json']).get('constructed_type') == {'module': 'builtins', 'qualname': 'range'})
    publication = _marked(report, source['dependences']['ordered_publication']['original_update'])
    _require(publication['caller_site']['code'] == detached['body_code_key']
        and targets[children[-1]]['caller']['offset'] < publication['caller_site']['offset'],
        'expected ordered publication downstream of the original row preparation and solve')
    previous, count = source_scope['observed']['consumer_completions'], int(argv[0])
    counts = {event: count if total == previous else total for event, total in original['local_event_counts'].items()}
    return dict(schema=DESIGN, schema_version=1, entry=original['entry'], argv=argv, count=count,
        archive=artifact_reference(package/'artifact.json', PACKAGE), provenance=reference,
        code_keys=original['code_keys'], targets=targets, body=body, ordered_children=children,
        formals=detached['roles'], expected_counts=counts, loop_inputs=original['loop_inputs'],
        range_site=iterator['caller_site'], publication_site=publication['caller_site'],
        publication_callable=json.loads(publication['metadata_json']))


def _geometry(frame, formals):
    values = frame.f_locals
    matrix, rows, load = (values[formals[key]] for key in ('matrix', 'boundary', 'load'))
    return dict(matrix_shape=list(matrix.shape), data_dtype=matrix.dtype.str,
        nnz=matrix.nnz, row_count=len(rows), index_width=matrix.indices.itemsize,
        pointer_width=matrix.indptr.itemsize, rows_dtype=rows.dtype.str, load_dtype=load.dtype.str)


@contextmanager
def observe_scope(design, out_path):
    """Capture the actual loop operand, original completions, and ordered digest."""
    _require(design['schema'] == DESIGN and design['schema_version'] == 1,
        'expected checked aggregate observation design')
    monitor = sys.monitoring
    events = monitor.events
    tool = next((slot for slot in reversed(range(6)) if monitor.get_tool(slot) is None), None)
    _require(tool is not None, 'no independent CPython monitoring slot available')
    specs = design['targets']
    selected = {(row['code']['filename'], row['code']['qualname'], row['code']['first_line']): name
        for name, row in specs.items()}
    paths = {key[0] for key in selected}
    body, children = design['body'], design['ordered_children']
    count = design['count']
    counters = Counter()
    active = {}
    watched = {}
    owner_thread = get_ident()
    closed = False
    body_frame = None
    stage = 0
    loop_inputs = None
    returned = None
    publisher = None
    publication_target = None
    range_site = design['range_site']['offset']
    publication_site = design['publication_site']['offset']

    def arm(event, args):
        if closed or event != 'exec' or not args:
            return
        module = args[0]
        if type(module) is not CodeType or module.co_filename not in paths:
            return
        pending = [module]
        while pending:
            code = pending.pop()
            pending.extend(child for child in code.co_consts if type(child) is CodeType)
            name = selected.get((code.co_filename, code.co_qualname, code.co_firstlineno))
            if name is None:
                continue
            _require(describe_code(code) == specs[name]['code'], 'observed source code differs from its exact F4 frame')
            watched[code] = name
            mask = events.PY_START | events.PY_RETURN
            if name == body:
                mask |= events.CALL
            monitor.set_local_events(tool, code, mask)

    def start(code, _offset):
        nonlocal body_frame, loop_inputs
        frame = sys._getframe(1)
        caller = frame.f_back
        name = watched[code]
        if name != body and (body_frame is None or caller is not body_frame):
            return
        site = specs[name]['caller']
        _require(get_ident() == owner_thread and frame.f_code is code and caller is not None
            and caller.f_lasti == site['offset'] and code_key(caller.f_code) == site['code'],
            'original scoped invocation differs from its observed F4 caller and thread')
        if name == body:
            _require(body_frame is None and not counters[name+'_started'],
                'expected exactly one original loop body activation')
            value = frame.f_locals[design['formals']['count']]
            _require(type(value) is int and value == count, 'bound original loop count differs from argv')
            body_frame = frame
            loop_inputs = _geometry(frame, design['formals'])
            _require(loop_inputs == design['loop_inputs'], 'runtime sparse operand geometry differs from the family')
        else:
            _require(stage < len(children) and name == children[stage],
                'original per-iteration call order differs from the admitted route')
        active[frame] = name
        counters[name+'_started'] += 1

    def finish(_code, _offset, value):
        nonlocal returned, body_frame, stage
        frame = sys._getframe(1)
        name = active.pop(frame, None)
        if name is None:
            return
        counters[name+'_completed'] += 1
        if name == body:
            _require(not stage and type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
                'original ordered publication did not complete as a digest')
            returned, body_frame = value, None
        else:
            stage += 1

    def call(_code, offset, target, arg0):
        nonlocal publisher, publication_target
        if offset == range_site and sys._getframe(1) is body_frame:
            _require(target is builtins.range and type(arg0) is int and arg0 == count,
                'original bounded iterator changed provider or count')
            counters['builtin_range_calls'] += 1
        if offset != publication_site or sys._getframe(1) is not body_frame:
            return
        if target is not publication_target:
            _require(target_metadata(target) == design['publication_callable'],
                'original ordered publication carrier changed')
            publication_target = target
        _require(stage == len(children), 'original publication must follow successful ordered solve')
        if publisher is None:
            publisher = arg0
        _require(publisher is arg0, 'ordered publication switched receivers')

    def c_return(_code, offset, target, _arg0):
        nonlocal stage
        if offset == publication_site and sys._getframe(1) is body_frame:
            _require(stage == len(children) and target is publication_target,
                'original publication did not return from the identified carrier')
            counters['digest_update_sites_reached'] += 1
            stage = 0

    callbacks = ((events.PY_START, start), (events.PY_RETURN, finish),
                 (events.CALL, call), (events.C_RETURN, c_return))
    monitor.use_tool_id(tool, 'scar-aggregate-ordered-scope')
    try:
        for event, callback in callbacks:
            monitor.register_callback(tool, event, callback)
        sys.addaudithook(arm)
        yield
    finally:
        closed = True
        for code in watched:
            monitor.set_local_events(tool, code, 0)
        for event, _ in callbacks:
            monitor.register_callback(tool, event, None)
        monitor.free_tool_id(tool)
    _require(returned is not None and not active and dict(counters) == design['expected_counts'],
        'original loop did not complete the exact admitted iteration cardinality')
    stdout_hash = sha256((returned+'\n').encode('ascii')).hexdigest()
    result = dict(schema=SCHEMA, schema_version=1, entry=design['entry'], argv=design['argv'],
        installed_native_providers=0, local_event_counts=dict(counters), affinity=sorted(os.sched_getaffinity(0)),
        code_keys=design['code_keys'], normal_completion=True, per_iteration_records=False, array_contents_saved=False,
        loop_inputs=loop_inputs, return_digest=returned, process_id=os.getpid(), thread_id=owner_thread,
        expected_stdout_sha256=stdout_hash, local_monitoring_only=True, performance_measurement=False)
    with Path(out_path).open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
