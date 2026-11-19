"""Observe bounded normal returns of a selected complete scalar function."""

import argparse
import builtins
from collections import Counter, defaultdict
from pathlib import Path
import os
import sys
from threading import get_ident
from types import CodeType

from scar.analysis.scalar_score_evidence import CAPTURE, DESIGN, _need
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_table_handoff_observe import _Scores, _int, _position
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site


def _bindings(frame, names):
    constants, carriers = {}, {}
    for name in names:
        value = frame.f_globals[name] if name in frame.f_globals else frame.f_builtins[name]
        if value is builtins.abs:
            carriers[name] = 'CPYTHON_EXACT_BUILTIN_ABS'
        else:
            constants[name] = _int(value)
    return constants, carriers


def _argument(value):
    kind = type(value)
    _need(tuple in type.__getattribute__(kind, '__mro__')
          and type.__getattribute__(kind, '__iter__') is tuple.__iter__
          and tuple.__len__(value) == 3, 'original argument requires a transparent three-slot tuple')
    first, second, third = (tuple.__getitem__(value, index) for index in range(3))
    _need(type(third) is str and third.isascii(), 'original tuple string field outside the scalar domain')
    return [_int(first), _int(second), third]


class _Probe:
    def __init__(self, design):
        self.design = design
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.slot = next(index for index in range(5, -1, -1) if self.monitor.get_tool(index) is None)
        self.code = None
        self.active = True
        self.window = {}
        self.roots = []
        self.versions = {}
        self.recent = None
        self.attempted = Counter()
        self.frontiers = Counter()
        self.samples = defaultdict(list)
        self.globals = self.builtins = None
        self.ordinal = 0
        self.offsets = {row[key] for row in design['branches'] for key in
                        ('test_offset', 'then_offset', 'else_offset') if row[key] is not None}

    def _mask(self, instruction=False):
        mask = self.events.PY_START | self.events.PY_RETURN
        if instruction:
            mask |= self.events.INSTRUCTION
        self.monitor.set_local_events(self.slot, self.code, mask)

    def attach(self, event, args):
        if not self.active or event != 'exec' or not args or type(args[0]) is not CodeType:
            return
        root = args[0]
        if root.co_filename != self.design['source']['uri']:
            return
        wanted = _target_coordinate(self.design['code'])
        for code in _codes(root):
            if _coordinate(code) == wanted:
                _need(self.code is None and describe_code(code) == self.design['code'], 'original score code changed on load')
                self.code = code
                self._mask()

    def _table(self, root):
        if self.recent is not None:
            token, known, members, digest = self.recent
            if (root is known and len(root) == len(members) and all(type(key) is str for key in root)
                    and all(root.get(key) is row for key, row in members)):
                return self.recent
        # Use the same exact dict/tuple/i64 snapshot contract as the retained
        # source handoff. Content versions are never reconstructed from id().
        token, members, digest = _Scores._table(self, root)
        _need(len(self.versions) <= self.design['limits']['max_versions'], 'table version capture budget exceeded')
        self.recent = token, root, members, digest
        return self.recent

    def begin(self, _code, _offset):
        frame = sys._getframe(1)
        parent = frame.f_back
        if parent is None or self.window:
            return
        spec = self.design
        try:
            selected = self._table(frame.f_globals[spec['table']])
            token, _, _, version = selected
            if self.attempted[version] >= spec['limits']['attempts_per_version']:
                return
            entry = site(parent.f_code, parent.f_lasti)
            _need(entry['opname'] in ('CALL', 'CALL_KW'), 'original score caller protocol not selected')
            locals_ = frame.f_locals
            fields = _position(locals_[spec['receiver']], spec['receiver_fields'])
            _need(all(type(value) is int or value.isascii() for value in fields.values()),
                  'exact ASCII source field required')
            arg = _argument(locals_[spec['argument']])
            constants, carriers = _bindings(frame, spec['global_loads'])
            if self.globals is None:
                self.globals, self.builtins = constants, carriers
            _need(self.globals == constants and self.builtins == carriers,
                  'global scalar or builtin changed between sampled frames')
            self.attempted[version] += 1
            self.ordinal += 1
            self.window[frame] = (selected, set(), dict(table_version=version, root_token=token,
                frame_ordinal=self.ordinal, thread_id=get_ident(), entry_site=entry,
                receiver_fields=fields, argument=arg, move=arg))
            self._mask(instruction=True)
        except (ValueError, KeyError):
            self.frontiers['SCORE_INPUT_BINDINGS_OPEN'] += 1

    def instruction(self, _code, offset):
        if offset not in self.offsets:
            return
        row = self.window.get(sys._getframe(1))
        if row is not None:
            row[1].add(offset)

    @staticmethod
    def _signature(row):
        return tuple(branch['arm'] for branch in row['branch_arms'])

    def _retain(self, sample):
        rows = self.samples[sample['table_version']]
        if len(rows) < self.design['limits']['per_version']:
            rows.append(sample)
            return
        signatures = Counter(map(self._signature,rows))
        current = self._signature(sample)
        if current in signatures:
            return
        slot = next((i for i,row in enumerate(rows) if signatures[self._signature(row)] > 1), None)
        if slot is not None:
            rows[slot] = sample

    def finish(self, code, offset, value):
        frame = sys._getframe(1)
        sample = self.window.pop(frame, None)
        if sample is None:
            return
        self._mask()
        (token, root, members, version), visited, row = sample
        try:
            current = frame.f_globals[self.design['table']]
            _need(current is root and len(current) == len(members)
                  and all(type(key) is str for key in current)
                  and all(current.get(name) is member for name, member in members),
                  'table changed across selected score invocation boundaries')
            _need(_bindings(frame, self.design['global_loads']) == (self.globals, self.builtins) and
                  get_ident() == row['thread_id'], 'sampled score namespace or thread changed')
            returned = site(code, offset)
            _need(returned['opname'] in ('RETURN_VALUE', 'RETURN_CONST'), 'normal scalar return did not occur')
            _need(row['receiver_fields'] == _position(frame.f_locals[self.design['receiver']],
                  self.design['receiver_fields']) and row['argument'] == _argument(frame.f_locals[self.design['argument']]),
                  'receiver/argument fields changed within selected score')
            arms = []
            for branch in self.design['branches']:
                if branch['test_offset'] not in visited:
                    arm = 'NOT_REACHED'
                elif branch['then_offset'] in visited:
                    arm = 'THEN'
                elif branch['else_offset'] is None:
                    arm = 'FALSE'
                else:
                    _need(branch['else_offset'] in visited, 'selected ordinary branch did not enter either arm')
                    arm = 'ELSE'
                arms.append(dict(id=branch['id'], arm=arm))
            row.update(result=_int(value), return_site=returned, branch_arms=arms)
            self._retain(row)
            if (len(self.attempted) == self.design['limits']['max_versions'] and
                    all(count >= self.design['limits']['attempts_per_version'] for count in self.attempted.values())):
                self.active = False
                self.monitor.set_local_events(self.slot, code, 0)
        except (ValueError, KeyError):
            self.frontiers['NORMAL_SCALAR_RESULT_OR_BOUNDARY_OPEN'] += 1


def capture(selection, out):
    reference = _ref(selection, DESIGN)
    design = read_artifact(reference)
    read_artifact(design['source'], json_document=False)
    observer = _Probe(design)
    monitor, events, slot = observer.monitor, observer.events, observer.slot
    callbacks = ((events.PY_START, observer.begin), (events.PY_RETURN, observer.finish),
                 (events.INSTRUCTION, observer.instruction))
    monitor.use_tool_id(slot, CAPTURE)
    try:
        _need(monitor.get_events(slot) == 0, 'global monitoring mask unavailable for scoped capture')
        for event, callback in callbacks:
            monitor.register_callback(slot, event, callback)
        sys.addaudithook(observer.attach)
        launch = run_original_script(design['source']['uri'], [])
    finally:
        observer.active = False
        if observer.code is not None:
            monitor.set_local_events(slot, observer.code, 0)
        for event, _ in callbacks:
            monitor.register_callback(slot, event, None)
        monitor.free_tool_id(slot)
    kept = {version: observer.versions[version] for version in observer.samples}
    samples = sorted((row for group in observer.samples.values() for row in group), key=lambda row: row['frame_ordinal'])
    return _write(out, dict(schema=CAPTURE, schema_version=1, selection=reference, source=design['source'],
        case=design['case'], case_index=design['case_index'], launch=launch, process_id=os.getpid(),
        table_versions=kept, globals=observer.globals, builtins=observer.builtins,
        builtin_abs_verified=bool(observer.builtins),
        attempts_per_version=dict(observer.attempted), samples=samples, observation_frontiers=dict(observer.frontiers),
        original_execution=True, performance_measurement=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    capture(args.selection, args.out)


if __name__ == '__main__':
    main()
