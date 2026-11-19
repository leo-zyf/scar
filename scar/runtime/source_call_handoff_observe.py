"""Observe same-reference handoffs between two selected local Python call sites."""

from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import os
import sys
from threading import get_ident
from types import CodeType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site

DESIGN = 'scar.source-call-handoff-selection'
SCHEMA = 'scar.source-call-handoff-capture'


def _require(ok, detail):
    if not ok:
        raise ValueError(detail)


class _Observer:
    def __init__(self, design):
        self.design = design
        self.wanted = {_target_coordinate(code): code for code in
                       (design['caller'], design['producer']['code'], design['consumer']['code'])}
        self.paths = {coordinate[0] for coordinate in self.wanted}
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.slot = next(i for i in range(5, -1, -1) if self.monitor.get_tool(i) is None)
        self.active = False
        self.selected = {}
        self.names = {}
        self.producers = {}
        self.pending = {}
        self.rows = []
        self.frontiers = []
        self.thread = None
        self.process = os.getpid()
        self.ordinal = 0

    def attach(self, event, args):
        if not self.active or event != 'exec' or not args or type(args[0]) is not CodeType:
            return
        root = args[0]
        if root.co_filename not in self.paths:
            return
        for code in _codes(root):
            descriptor = self.wanted.get(_coordinate(code))
            if descriptor is not None:
                _require(describe_code(code) == descriptor, 'selected body changed after source preparation')
                self.selected[code] = descriptor['key']
                self.monitor.set_local_events(self.slot, code,
                    self.events.PY_START | self.events.PY_RETURN)

    def identify(self, frame):
        if frame not in self.names:
            self.ordinal += 1
            self.names[frame] = self.ordinal
        return self.names[frame]

    def begin(self, code, _offset):
        thread = get_ident()
        if self.thread not in (None, thread) or os.getpid() != self.process:
            return
        frame = sys._getframe(1)
        parent = frame.f_back
        if parent is None or self.selected.get(parent.f_code) != self.design['caller']['key']:
            return
        for role in ('producer', 'consumer'):
            expected = self.design[role]
            if self.selected[code] != expected['code']['key'] or parent.f_lasti != expected['caller_site']['offset']:
                continue
            _require(site(parent.f_code, parent.f_lasti) == expected['caller_site'],
                     'selected producer/consumer call instruction changed')
            if self.thread is None:
                self.thread = thread
            if len(self.names) >= 4 * self.design['max_pairs']:
                self.frontiers.append('LIVE_SCOPE_BUDGET_EXCEEDED')
                self.close_window()
                return
            parent_id, child_id = self.identify(parent), self.identify(frame)
            if role == 'producer':
                self.pending.pop(parent, None)
                self.producers[frame] = parent, parent_id, child_id
                return
            output = self.pending.pop(parent, None)
            if output is None:
                self.frontiers.append('NORMAL_PRODUCER_RETURN_NOT_OBSERVED')
            else:
                value, provider, returned = output
                # Retention ends at the immediate consumer entry.
                local = frame.f_locals
                matches = [i for i, name in enumerate(expected['code']['parameters'])
                           if name in local and local[name] is value]
                if len(matches) == 1:
                    self.rows.append(dict(parent=parent_id, producer=provider, consumer=child_id,
                                          producer_return=returned, formal_ordinal=matches[0]))
                else:
                    self.frontiers.append('UNIQUE_SAME_REFERENCE_CONSUMER_FORMAL_NOT_OBSERVED')
            if len(self.rows) + len(self.frontiers) >= self.design['max_pairs']:
                self.close_window()
            return

    def finish(self, code, offset, value):
        frame = sys._getframe(1)
        pending = self.producers.pop(frame, None)
        if pending is not None:
            parent, _, provider = pending
            returned = site(code, offset)
            _require(returned['opname'] in ('RETURN_VALUE', 'RETURN_CONST'),
                     'selected producer has no ordinary return instruction')
            self.pending[parent] = value, provider, returned
        self.names.pop(frame, None)
        self.pending.pop(frame, None)

    def close_window(self):
        self.active = False
        for code in self.selected:
            self.monitor.set_local_events(self.slot, code, 0)
        self.names.clear()
        self.producers.clear()
        self.pending.clear()


@contextmanager
def _observe(design):
    recorder = _Observer(design)
    monitor, slot, events = recorder.monitor, recorder.slot, recorder.events
    callbacks = ((events.PY_START, recorder.begin), (events.PY_RETURN, recorder.finish))
    monitor.use_tool_id(slot, SCHEMA)
    _require(monitor.get_events(slot) == 0, 'handoff monitor requires local events only')
    recorder.active = True
    try:
        for event, callback in callbacks:
            monitor.register_callback(slot, event, callback)
        sys.addaudithook(recorder.attach)
        yield recorder
    finally:
        recorder.close_window()
        for event, _ in callbacks:
            monitor.register_callback(slot, event, None)
        monitor.free_tool_id(slot)


def capture(selection, out):
    design_ref = artifact_reference(Path(selection).resolve(strict=True), DESIGN)
    design = read_artifact(design_ref)
    source = design['source']
    read_artifact(source, json_document=False)
    _require(design['schema_version'] == 1 and type(design['max_pairs']) is int
             and 1 <= design['max_pairs'] <= 32, 'expected a small selected handoff window')
    for row in design['producer'], design['consumer']:
        _require(row['caller_site']['code'] == design['caller']['key'],
                 'selected sites require one original caller code')
        read_artifact(row['source'], json_document=False)
    with _observe(design) as recorder:
        launch = run_original_script(source['uri'], design['argv'])
    read_artifact(design_ref)
    record = dict(schema=SCHEMA, schema_version=1, selection=design_ref,
        process_id=recorder.process, thread_id=recorder.thread, launch=launch,
        matched=recorder.rows, observation_frontiers=recorder.frontiers,
        reference_test='LIVE_RETURN_OBJECT_IS_SELECTED_ENTRY_FORMAL',
        values_saved=False, original_execution=True, performance_measurement=False)
    with Path(out).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    capture(args.selection, args.out)


if __name__ == '__main__':
    main()
