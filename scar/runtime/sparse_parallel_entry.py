"""Run isolated numeric jobs and publish them at original ordered solve calls."""

import argparse
from collections import deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from types import CodeType

from scar.ir.execution_recipe import read_artifact
from scar.runtime.csr_region_entry import _observer_active, launch_context
from scar.runtime.entry_runner import _arguments, _execute
from scar.runtime.sparse_solve_entry import admit as admit_serial
from scar.runtime.sparse_solve_prediction import DetachedCSR, match_request
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.sparse-parallel-entry-selection'


@contextmanager
def _affinity(cpus):
    original = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, cpus)
        yield
    finally:
        os.sched_setaffinity(0, original)


@contextmanager
def _observe_inputs(entry, recipe, prepare):
    """Capture exactly one original function's live-in objects."""
    monitor = getattr(sys, 'monitoring', None)
    if monitor is None or _observer_active():
        yield
        return
    token = 4
    event = monitor.events.PY_START
    target, active = None, True
    monitor.use_tool_id(token, 'SCAR sparse live-ins')

    def detach():
        nonlocal active
        if active:
            active = False
            monitor.set_events(token, 0)
            if target is not None:
                monitor.set_local_events(token, target, 0)
            monitor.register_callback(token, event, None)
            monitor.free_tool_id(token)

    def observe(code, offset):
        nonlocal target
        if target is None:
            if code.co_name == '<module>' and code.co_filename == str(entry):
                matches = [body for body in code.co_consts if type(body) is CodeType
                           and code_key(body) == recipe['body_code_key']]
                if len(matches) == 1:
                    target, = matches
                    monitor.set_events(token, 0)
                    monitor.set_local_events(token, target, event)
                else:
                    detach()
        elif code is target:
            locals_ = sys._getframe(1).f_locals
            detach()
            prepare(locals_)

    try:
        monitor.register_callback(token, event, observe)
        monitor.set_events(token, event)
        yield
    finally:
        detach()


class _OrderedOffers:
    def __init__(self, recipe, selection, serial_factory, allowed):
        self.recipe, self.selection, self.serial_factory, self.allowed = recipe, selection, serial_factory, allowed
        self.original = None
        self.serial = self.pool = self.snapshot = None
        self.serial_unavailable = False
        self.pending = deque()
        self.next_index = 0
        self.started = False
        self.stats = dict(inputs_captured=False, matched=0, serial_calls=0, rejected=0, peak_window=0)

    def activate(self, original):
        self.original = original
        return self

    def capture(self, values):
        try:
            inputs = {role:values[name] for role,name in self.recipe['roles'].items()}
            if type(inputs['count']) is not int or inputs['count'] != self.selection['count']:
                return
            self.snapshot = DetachedCSR(**inputs, expression=self.recipe['scale'], **self.selection['matrix'])
            self.stats['inputs_captured'] = True
        except (ArithmeticError, ValueError, TypeError, KeyError, MemoryError):
            self.stats['rejected'] += 1

    def _start(self, args, options):
        from scar.runtime.sparse_parallel_lu import load_parallel_symbolic_solve

        self.started = True
        if self.snapshot is None or not match_request(self.snapshot.args(0), args, options):
            return
        policy = self.selection['workers']
        with _affinity(self.allowed):
            self.pool = load_parallel_symbolic_solve(self.selection['native'], original=self.original,
                worker_cpu_ids=policy['cpu_ids'], address_space_limit=policy['address_space_limit_bytes'],
                budget_bytes=self.selection['symbolic_budget_bytes'])
        if self.pool.prepare(args, options):
            self._fill()
        else:
            self._stop_speculation()

    def _fill(self):
        window = self.selection['workers']['window']
        while len(self.pending) < window and self.next_index < self.snapshot.count:
            future = self.pool.submit(self.snapshot.args(self.next_index), {'ColPerm': None})
            if future is None:
                break
            self.pending.append(future)
            self.next_index += 1
        self.stats['peak_window'] = max(self.stats['peak_window'], len(self.pending))

    def __call__(self, args, options):
        result = None
        try:
            if not self.started:
                self._start(args, options)
            if self.pending:
                result = self.pool.consume(self.pending.popleft(), args, options)
                if result is not None:
                    self.stats['matched'] += 1
                    self._fill()
                else:
                    self.stats['rejected'] += 1
                    self._stop_speculation()
        except (ArithmeticError, EOFError, OSError, ValueError, MemoryError, RuntimeError):
            self.stats['rejected'] += 1
            self._stop_speculation()
        if result is not None:
            return result
        if self.serial is None and not self.serial_unavailable:
            try:
                self.serial = self.serial_factory(self.original)
            except (OSError, ImportError, TypeError, KeyError, ValueError, AttributeError):
                self.serial_unavailable = True
        if self.serial is None:
            return None
        self.stats['serial_calls'] += 1
        return self.serial(args, options)

    def _stop_speculation(self):
        if self.pool is not None:
            self.pool.close()
        self.pending.clear()
        self.snapshot = None

    def close(self):
        self._stop_speculation()
        if self.serial is not None:
            self.serial.close()

    def receipt(self):
        return dict(self.stats, workers=None if self.pool is None else dict(self.pool.stats))


def _admit(entry, argv, path, fingerprint, exclusive):
    if not path or not fingerprint or exclusive is not True or _observer_active():
        return None
    try:
        path = Path(path).resolve()
        selected = read_artifact(dict(uri=str(path), sha256=fingerprint, schema=SCHEMA, size_bytes=path.stat().st_size))
        allowed = sorted(os.sched_getaffinity(0))
        if (selected['schema_version'] != 1 or selected['entry']['uri'] != str(entry)
                or selected['launch_cpu_ids'] != allowed
                or not {selected['owner_cpu'], *selected['workers']['cpu_ids']} <= set(allowed)
                or {**launch_context(argv), 'cpu_ids':[selected['owner_cpu']]} != selected['context']):
            return None
        for reference in selected['runtime_members']:
            read_artifact(reference, json_document=False)
        return selected
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _resources_fit(plan):
    from scar.optimization.sparse_parallel_cost import collect_hardware

    try:
        hardware = collect_hardware(plan['launch_cpu_ids'])
    except (OSError, ValueError):
        return False
    processors = hardware['processors']
    cores = {(row['package_id'], row['die_id'], row['core_id']) for row in processors}
    quota = hardware['cpu_quota_equivalents']
    need = plan['memory_floor_bytes']
    node = plan['numa_node']
    free = hardware['node_free_snapshot'].get(str(node), {})
    cgroup = hardware['finite_memory_headroom_snapshot_bytes']
    return (hardware['hierarchy_observation_complete'] and len(cores) == len(processors)
            and all(row['permitted'] and row['numa_nodes'] == [node] for row in processors)
            and (quota is None or quota[0] >= len(processors)*quota[1])
            and (cgroup is None or cgroup >= need)
            and free.get('allowed') is True and (free.get('bytes') or 0) >= need)


def run(entry, argv, *, selection=None, sha256=None, exclusive=False, study_parallel=False):
    entry, argv = _arguments(entry, argv)
    plan = _admit(entry, argv, selection, sha256, exclusive)
    offers = None
    if plan is None:
        result = _execute(entry, argv, None, 'SPARSE_PARALLEL_CONTEXT_CHANGED')
    else:
        broad = os.sched_getaffinity(0)
        study = plan['serial_study']
        requested = plan['decision'] == 'PARALLEL' or study_parallel is True and plan['study_parallel']
        try_parallel = requested and _resources_fit(plan)
        with _affinity({plan['owner_cpu']}):
            scope, reason = admit_serial(entry, argv, study['uri'], study['sha256'], study=True, exclusive=True)
            if scope is not None and try_parallel:
                recipe = read_artifact(plan['recipe'])
                offers = _OrderedOffers(recipe, plan, scope.loader, broad)
                scope.loader = offers.activate
                try:
                    with _observe_inputs(entry, recipe, offers.capture):
                        result = _execute(entry, argv, scope, reason)
                finally:
                    offers.close()
            else:
                result = _execute(entry, argv, scope, reason)
    result['schema'] = 'scar.sparse-parallel-entry-execution'
    result['selection_sha256'] = sha256
    result['parallel'] = None if offers is None else offers.receipt()
    result['applied'] = ('PARALLEL' if offers is not None and offers.stats['matched'] else
                        'SERIAL' if result['statistics']['native_calls'] else 'KEEP')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection')
    parser.add_argument('--selection-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--study-parallel', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    report = run(command[0], command[1:], selection=args.selection, sha256=args.selection_sha256,
        exclusive=args.exclusive_inputs, study_parallel=args.study_parallel)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
