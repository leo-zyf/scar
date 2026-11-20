"""Bounded CPython local control transfers in exact observed activations."""
from hashlib import file_digest
import json
import os
import sys
from threading import get_ident


SCHEMA = 'scar.observed-control-transfers'
MAX_TRANSFERS = 8192


class ControlTransfers:
    def __init__(self, recorder, selected):
        self.recorder = recorder
        self.selected = frozenset(selected)
        self.monitor = sys.monitoring
        events = self.monitor.events
        self.mask = events.JUMP | events.BRANCH_LEFT | events.BRANCH_RIGHT
        self.name = f'scar.control-transfers.{recorder.pid}'
        self.tool = next((slot for slot in range(5, -1, -1) if self.monitor.get_tool(slot) is None), None)
        if self.tool is None:
            raise RuntimeError('no monitoring tool for local control transfers')
        self.monitor.use_tool_id(self.tool, self.name)
        self.monitor.register_callback(self.tool, events.JUMP, self.jump)
        self.monitor.register_callback(self.tool, events.BRANCH_LEFT, self.branch)
        self.monitor.register_callback(self.tool, events.BRANCH_RIGHT, self.branch)
        self.codes = set()
        self.open = {}
        self.frames = []
        self.records = []
        self.failure = None

    def intact(self, code):
        return (self.monitor.get_tool(self.tool) == self.name and self.monitor.get_events(self.tool) == 0
                and self.monitor.get_local_events(self.tool, code) == self.mask)

    def enter(self, frame, thread, invocation, code, suspended):
        if self.failure or code not in self.selected or suspended:
            return
        if frame.f_code not in self.codes:
            self.monitor.set_local_events(self.tool, frame.f_code, self.mask)
            self.codes.add(frame.f_code)
        if not self.intact(frame.f_code):
            self.failure = 'CONTROL_MONITORING_CHANGED'
            return
        self.open[thread, invocation] = self.recorder.index

    def exit(self, frame, thread, invocation):
        start = self.open.pop((thread, invocation), None)
        if start is not None:
            if not self.intact(frame.f_code):
                self.failure = 'CONTROL_MONITORING_CHANGED'
            self.frames.append([thread, invocation, start, self.recorder.index])

    def jump(self, code, source, destination):
        self.record(sys._getframe(1), code, source, destination, 'JUMP')

    def branch(self, code, source, destination):
        self.record(sys._getframe(1), code, source, destination, 'BRANCH')

    def record(self, frame, code, source, destination, event):
        recorder = self.recorder
        if recorder._closing or self.failure or os.getpid() != recorder.pid:
            return
        with recorder.lock:
            thread = get_ident()
            state = recorder.active.get((thread, id(frame)))
            if (state is None or frame.f_code is not code or (thread, state[0]) not in self.open
                    or not self.intact(code)):
                self.failure = 'CONTROL_TRANSFER_WITHOUT_EXACT_ACTIVE_FRAME'
            elif len(self.records) >= MAX_TRANSFERS:
                self.failure = 'CONTROL_TRANSFER_LIMIT'
            else:
                self.records.append([thread, state[0], recorder.index, source, destination, event])

    def close(self):
        recorder = self.recorder
        if self.monitor.get_tool(self.tool) == self.name:
            for code in self.codes:
                self.monitor.set_local_events(self.tool, code, 0)
            self.monitor.free_tool_id(self.tool)
        receipt = recorder.out / f'control-transfers.{recorder.pid}.json'
        events = recorder.out / f'call-events.{recorder.pid}.jsonl'
        if recorder.io_failed or not events.is_file():
            return
        with events.open('rb') as stream:
            frame_sha = file_digest(stream, 'sha256').hexdigest()
        report = dict(schema=SCHEMA, schema_version=1, process_id=recorder.pid, selected=sorted(self.selected),
                      frame_events_sha256=frame_sha, failure=self.failure, open_frames=len(self.open),
                      frames=self.frames, transfers=self.records)
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
