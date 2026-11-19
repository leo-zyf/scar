"""Time the released sparse owner through the original guarded entry."""

import argparse
import ast
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from functools import wraps
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns as clock
from types import BuiltinFunctionType, FunctionType
from unittest.mock import patch

from scar.analysis.sparse_pipeline_scope import _row
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime import sparse_parallel_entry as entry
from scar.runtime import sparse_parallel_lu as pool
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.sparse_solve_entry import SparseSolveScope
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.sparse-parallel-owner-observation'
SOURCE = 'scar.python-source'
SAMPLE_PHASES = ('row_edit', 'normalize', 'capture_request', 'future_wait',
                 'finish_match', 'status_return', 'publication')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


@contextmanager
def _replace(namespace, name, value):
    original = namespace[name]
    namespace[name] = value
    try:
        yield
    finally:
        if namespace.get(name) is value:
            namespace[name] = original


def _call_binding(source, observation):
    line, end_line, column, end_column = observation['caller_site']['position']
    calls = [node for node in ast.walk(source) if type(node) is ast.Call and
             (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
             == (line, end_line, column, end_column)]
    _require(len(calls) == 1 and type(calls[0].func) is ast.Name,
             'timed source call must have one original direct binding')
    return calls[0].func.id


def probe_plan(release):
    recipe = read_artifact(release['execution_recipe'])
    scope = read_artifact(recipe['artifacts']['parallel_scope'])
    pipeline = read_artifact(scope['pipeline_scope'])
    graph = read_artifact(scope['source_graph'])
    source = ast.parse(read_artifact(release['entry'], json_document=False))
    publish = scope['dependences']['ordered_publication']
    constructor = _row(graph, publish['original_constructor'], 'observed_targets')['observation']
    edit = [row['observation'] for row in graph['observed_targets']
            if row['entry_scope_reachable'] and row['operation'] == pipeline['regions']['row_edit']['call_point']
            and row['observation']['caller_site']['code'] == scope['source_body']['code_key']]
    _require(len(edit) == 1 and publish['kind'] == 'ORDERED_HASH_ACCUMULATION'
        and constructor['caller_site']['code'] == scope['source_body']['code_key']
        and publish['requires_original_provider_and_normal_completion'] is True,
        'owner probe requires the original row edit and lexical hash accumulation')
    original = recipe['semantic_boundary']
    solver = read_artifact(release['serial_study'])['callsite']
    _require(solver['code_key'] == original['original_code_key']
        and original['original_python_status_boundary']['preserve_in_python'] is True,
        'owner probe cannot bind the original solver and status')
    return dict(body=scope['source_body']['code_key'], solver=solver['code_key'],
        row_binding=_call_binding(source, edit[0]), row_code=edit[0]['target_code']['key'],
        constructor_binding=_call_binding(source, constructor),
        constructor_native=json.loads(constructor['metadata_json'])['native_boundary'],
        source_scope=recipe['artifacts']['parallel_scope'], source_graph=scope['source_graph'],
        pipeline_scope=scope['pipeline_scope'])


class _HashMeter:
    def __init__(self, original, probe):
        self.original, self.probe = original, probe

    def update(self, *args, **kwargs):
        begin = clock()
        result = self.original.update(*args, **kwargs)
        end = clock()
        self.probe.rows[-1]['publication'] = [begin, end]
        return result

    def hexdigest(self, *args, **kwargs):
        return self.original.hexdigest(*args, **kwargs)


class _Meter:
    def __init__(self, plan, stack):
        self.plan, self.stack = plan, stack
        self.rows, self.predictions = [], []
        self.cuts = {}
        self.active = self.live_capture = None

    def capture(self, original, offers, values):
        begin = clock()
        original(offers, values)
        end = clock()
        self.cuts['snapshot'] = [begin, end]
        frame = sys._getframe(3)
        namespace = frame.f_globals
        _require(code_key(frame.f_code) == self.plan['body'], 'live-ins were provided by another entry code')
        edit = namespace[self.plan['row_binding']]
        make_hash = namespace[self.plan['constructor_binding']]
        identity = self.plan['constructor_native']
        _require(type(edit) is FunctionType and code_key(edit.__code__) == self.plan['row_code']
            and type(make_hash) is BuiltinFunctionType and make_hash.__module__ == identity['module']
            and make_hash.__qualname__ == identity['qualname'],
            'live original row edit or hash provider differs')

        @wraps(edit)
        def row_edit(*args, **kwargs):
            begin = clock()
            result = edit(*args, **kwargs)
            end = clock()
            self.rows.append({'row_edit': [begin, end]})
            return result

        @wraps(make_hash)
        def constructor(*args, **kwargs):
            return _HashMeter(make_hash(*args, **kwargs), self)

        self.stack.enter_context(_replace(namespace, self.plan['row_binding'], row_edit))
        self.stack.enter_context(_replace(namespace, self.plan['constructor_binding'], constructor))

    def attach(self, original, scope, module, name, *, natural):
        prior = len(scope.bindings)
        original(scope, module, name, natural=natural)
        if len(scope.bindings) == prior:
            return
        _require(scope.original_key == self.plan['solver'] and len(scope.bindings) == 1,
                 'owner probe observed an unrelated solver')
        function = scope.bindings[-1]['function']

        @wraps(function)
        def measured_solver(*args, **kwargs):
            self.active = self.rows[-1]
            self.active['normalize'] = [clock()]
            try:
                result = function(*args, **kwargs)
                self.active['status_return'].append(clock())
                return result
            finally:
                self.active = None

        self.stack.enter_context(patch.object(module, scope.name, measured_solver))

    def offer(self, original, offers, args, options):
        self.active['normalize'].append(clock())
        result = original(offers, args, options)
        self.active['status_return'] = [clock()]
        return result

    def prepare_private(self, original, snapshot, index):
        begin = clock()
        result = original(snapshot, index)
        end = clock()
        self.predictions.append(dict(index=index, interval=[begin, end]))
        return result

    def initial_match(self, original, *args):
        begin = clock()
        result = original(*args)
        end = clock()
        self.cuts['initial_match'] = [begin, end]
        return result

    def capture_request(self, original, solver, args, options):
        if self.live_capture is None:
            return original(solver, args, options)
        begin = clock()
        result = original(solver, args, options)
        end = clock()
        self.live_capture['capture_request'] = [begin, end]
        return result

    def consume(self, original, solver, future, args, options):
        row = self.active
        real_result = future.result

        @wraps(real_result)
        def measured_result(*args, **kwargs):
            begin = clock()
            result = real_result(*args, **kwargs)
            end = clock()
            row['future_wait'] = [begin, end]
            return result

        self.live_capture = row
        try:
            with patch.object(future, 'result', measured_result):
                result = original(solver, future, args, options)
                end = clock()
                row['finish_match'] = [row['future_wait'][1], end]
                return result
        finally:
            self.live_capture = None

    def install(self):
        for owner, name, hook in (
            (entry._OrderedOffers, 'capture', self.capture),
            (entry._OrderedOffers, '__call__', self.offer),
            (entry.DetachedCSR, 'args', self.prepare_private),
            (entry, 'match_request', self.initial_match),
            (pool._ParallelSolver, '_capture', self.capture_request),
            (pool._ParallelSolver, 'consume', self.consume),
            (SparseSolveScope, 'attach', self.attach),
        ):
            self.install_one(owner, name, hook)

    def install_one(self, owner, name, hook):
        original = getattr(owner, name)

        @wraps(original)
        def timed(*args, **kwargs):
            return hook(original, *args, **kwargs)

        self.stack.enter_context(patch.object(owner, name, timed))


def observe(release_ref):
    release = read_artifact(release_ref)
    argv = release['context']['argv']
    context = launch_context(argv)
    _require(release['decision'] == 'PARALLEL'
        and context == {**release['context'], 'cpu_ids': release['launch_cpu_ids']},
        'owner probe requires the exact released launch context')
    plan = probe_plan(release)
    stdout, stderr = StringIO(), StringIO()
    with ExitStack() as stack, redirect_stdout(stdout), redirect_stderr(stderr):
        meter = _Meter(plan, stack)
        meter.install()
        begin = clock()
        run = entry.run(release['entry']['uri'], argv, selection=release_ref['uri'],
                        sha256=release_ref['sha256'], exclusive=True)
        end = clock()
    _require(launch_context(argv) == context, 'owner probe changed the launch context')
    return dict(schema=SCHEMA, schema_version=1, release=release_ref, context=context, plan=plan,
        instrument=artifact_reference(__file__, SOURCE), pid=os.getpid(), context_elapsed=[begin, end],
        execution=run, instrumented=True,
        stdout_sha256=sha256(stdout.getvalue().encode()).hexdigest(),
        stderr_sha256=sha256(stderr.getvalue().encode()).hexdigest(),
        cuts=meter.cuts, rows=meter.rows, predictions=meter.predictions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    reference = artifact_reference(args.release, entry.SCHEMA)
    _require(reference['sha256'] == args.sha256, 'released selection SHA-256 differs')
    _require(not args.out.exists(), 'fresh owner observation path required')
    result = observe(reference)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
