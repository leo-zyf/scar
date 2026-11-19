"""Sample exact scalar operands and selected position fields during one source run."""

import argparse
from collections import Counter
from _collections import _tuplegetter
from contextlib import contextmanager
from pathlib import Path
import os
import sys
from threading import get_ident
from types import CodeType

from scar.analysis.scalar_table_handoff import CAPTURE, DESIGN, _need, _ref, _write, table_version
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.source_call_handoff_observe import SCHEMA as HANDOFF, _observe as observe_handoff
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site


def _int(word):
    _need(type(word) is int and -(1 << 63) <= word < (1 << 63), 'exact i64 value required')
    return word


def _field(value, name):
    kind = type(value)
    _need(tuple in type.__getattribute__(kind, '__mro__') and
          type.__getattribute__(kind, '__getattribute__') is tuple.__getattribute__,
          'position requires an original tuple descriptor')
    mro = type.__getattribute__(kind, '__mro__')
    descriptor = next((type.__getattribute__(base, '__dict__')[name] for base in mro
                      if name in type.__getattribute__(base, '__dict__')), None)
    _need(type(descriptor) is _tuplegetter, 'selected original position field is not a tuple slot')
    return _tuplegetter.__get__(descriptor, value, kind)


def _position(value, names):
    result = {name: _field(value, name) for name in names}
    _need(all(type(cell) is int or type(cell) is str for cell in result.values()),
          'selected position fields must be immutable exact primitive values')
    return result


def _operand(frame, selector):
    return _int(selector['integer'] if 'integer' in selector else frame.f_locals[selector['local']])


class _Scores:
    def __init__(self, design, handoff):
        self.design = design
        self.handoff = handoff
        self.wanted = {_target_coordinate(design[name]): role
                       for name, role in (('value_code', 'value'), ('producer_code', 'move'))}
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.slot = next(index for index in range(5, -1, -1) if self.monitor.get_tool(index) is None)
        self.active = False
        self.codes = {}
        self.samples = {}
        self.values = {}
        self.moves = {}
        self.roots = []
        self.versions = {}
        self.frontiers = Counter()

    def _mask(self, code, *, instruction=False):
        mask = self.events.PY_START | self.events.PY_RETURN
        if instruction:
            mask |= self.events.INSTRUCTION
        self.monitor.set_local_events(self.slot, code, mask)

    def attach(self, event, args):
        if not self.active or event != 'exec' or not args or type(args[0]) is not CodeType:
            return
        root = args[0]
        if root.co_filename != self.design['source']['uri']:
            return
        for code in _codes(root):
            selected = self.wanted.get(_coordinate(code))
            if selected is None:
                continue
            name = 'value_code' if selected == 'value' else 'producer_code'
            _need(describe_code(code) == self.design[name], 'original scalar body changed on load')
            self.codes[code] = selected
            self._mask(code)

    def enter(self, code, _offset):
        if self.codes[code] != 'value' or not self.handoff.active or get_ident() != self.handoff.thread:
            return
        frame = sys._getframe(1)
        parent = frame.f_back
        if parent not in self.handoff.producers:
            return
        design = self.design
        if site(parent.f_code, parent.f_lasti) != design['value_call']:
            return
        try:
            parent_local, value_local = parent.f_locals, frame.f_locals
            receiver = value_local[design['value_receiver']]
            _need(receiver is parent_local[design['producer_receiver']]
                  and value_local[design['value_formal']] is parent_local[design['producer_formal']],
                  'value receiver and argument must be the same original producer inputs')
            origin = self.handoff.producers[parent][2]
            _need(frame not in self.values and parent not in self.moves, 'one selected accumulator call expected')
            self.values[frame] = (parent, origin, _position(receiver, design['position_fields']))
            self._mask(code, instruction=True)
        except (ValueError, KeyError):
            self.frontiers['ORIGINAL_SCALAR_CALL_INPUTS_OPEN'] += 1

    def _table(self, table):
        _need(type(table) is dict and 0 < len(table) <= 64 and all(type(key) is str for key in table),
              'exact dictionary with exact string keys required')
        anchors = tuple(sorted(table.items()))
        for token, root, old, version in self.roots:
            if root is table and len(old) == len(anchors) and all(k == k0 and row is row0
                    for (k, row), (k0, row0) in zip(anchors, old, strict=True)):
                return token, anchors, version
        _need(len(self.roots) < 32 and all(type(row) is tuple and row for _, row in anchors)
              and sum(len(row) for _, row in anchors) <= 4096, 'selected table exceeds scalar capture contract')
        rows = {key: [_int(item) for item in row] for key, row in anchors}
        version = table_version(rows)
        token = next((token for token, root, _, _ in self.roots if root is table), len(self.roots)+1)
        if version in self.versions:
            _need(self.versions[version]['root_token'] == token, 'same content selected under different table roots')
        else:
            self.versions[version] = dict(rows=rows, root_token=token)
        self.roots.append((token, table, anchors, version))
        return token, anchors, version

    def _before(self, frame, parent, origin, position):
        design = self.design
        table = frame.f_globals[design['table']]
        token, anchors, version = self._table(table)
        key = frame.f_locals[design['key']]
        _need(type(key) is str and key in table, 'exact selected string key required')
        row = table[key]
        _need(type(row) is tuple, 'exact selected tuple row required')
        left, right = (_operand(frame, design[operand]) for operand in ('left', 'right'))
        _need(-len(row) <= left < len(row) and -len(row) <= right < len(row), 'selected index is outside the row')
        record = dict(producer=origin, table_version=version, root_token=token, key=key,
            left=left, right=right, left_value=_int(row[left]), right_value=_int(row[right]),
            row_size=len(row), input_position=position)
        self.moves[parent] = (record, table, anchors)

    def _after(self, frame, parent):
        design = self.design
        record, table, anchors = self.moves[parent]
        original = frame.f_globals[design['table']]
        _need(original is table and len(table) == len(anchors)
              and all(type(key) is str for key in table)
              and all(table.get(key) is row for key, row in anchors), 'table root or rows changed at window boundaries')
        result = _int(frame.f_locals[design['base_local']])
        _need(result == record['left_value'] - record['right_value'], 'original scalar subtraction differs')
        record['result'] = result

    def instruction(self, code, offset):
        role = self.codes[code]
        frame = sys._getframe(1)
        design = self.design
        try:
            if role == 'value' and frame in self.values:
                parent, origin, position = self.values[frame]
                if offset == design['value_before']['offset']:
                    self._before(frame, parent, origin, position)
                elif offset == design['value_after']['offset'] and parent in self.moves:
                    self._after(frame, parent)
                    self._mask(code)
            elif role == 'move' and frame in self.moves and offset == design['producer_store_after']['offset']:
                record, _, _ = self.moves[frame]
                total = _int(frame.f_locals[design['producer_total']])
                start = _int(record['input_position'][design['score_field']])
                _need(total == start + record['value_delta'], 'original caller accumulation differs')
                record['producer_total'] = total
                self._mask(code)
        except (ValueError, KeyError):
            self.frontiers['ORIGINAL_SCALAR_WINDOW_OR_ACCUMULATOR_OPEN'] += 1
            selected_parent = self.values[frame][0] if role == 'value' else frame
            self.moves.pop(selected_parent, None)
            self._mask(code)

    def finish(self, code, offset, value):
        frame = sys._getframe(1)
        role = self.codes[code]
        if role == 'value':
            selected = self.values.pop(frame, None)
            if selected is None:
                return
            self._mask(code)
            parent, _, _ = selected
            found = self.moves.get(parent)
            if found is None or 'result' not in found[0]:
                self.moves.pop(parent, None)
                return
            try:
                found[0]['value_delta'] = _int(value)
                found[0]['value_return'] = site(code, offset)
            except ValueError:
                self.frontiers['NORMAL_SCALAR_VALUE_RETURN_OPEN'] += 1
                self.moves.pop(parent, None)
                return
            self._mask(parent.f_code, instruction=True)
            return
        selected = self.moves.pop(frame, None)
        if selected is None or 'producer_total' not in selected[0]:
            return
        record, _, _ = selected
        try:
            record['returned_position'] = _position(value, self.design['position_fields'])
        except ValueError:
            self.frontiers['NORMAL_PRODUCER_POSITION_FIELDS_OPEN'] += 1
            return
        record['producer_return'] = site(code, offset)
        self.samples[record['producer']] = record


@contextmanager
def _observe_scores(design, handoff):
    recorder = _Scores(design, handoff)
    monitor, events, slot = recorder.monitor, recorder.events, recorder.slot
    callbacks = ((events.PY_START, recorder.enter), (events.PY_RETURN, recorder.finish),
                 (events.INSTRUCTION, recorder.instruction))
    monitor.use_tool_id(slot, CAPTURE)
    recorder.active = True
    try:
        _need(monitor.get_events(slot) == 0, 'local monitoring scope required')
        for event, callback in callbacks:
            monitor.register_callback(slot, event, callback)
        sys.addaudithook(recorder.attach)
        yield recorder
    finally:
        recorder.active = False
        for code in recorder.codes:
            monitor.set_local_events(slot, code, 0)
        for event, _ in callbacks:
            monitor.register_callback(slot, event, None)
        monitor.free_tool_id(slot)


def capture(selection, out):
    design_ref = _ref(selection, DESIGN)
    design = read_artifact(design_ref)
    handoff_ref = design['handoff_selection']
    handoff = read_artifact(handoff_ref)
    _need(design['schema_version'] == 1 and handoff['source'] == design['source']
          and handoff['producer']['code'] == design['producer_code'], 'score and handoff launches differ')
    read_artifact(design['source'], json_document=False)
    with observe_handoff(handoff) as flow, _observe_scores(design, flow) as scalars:
        launch = run_original_script(design['source']['uri'], handoff['argv'])
    out = Path(out)
    raw_flow = _write(out/'handoff-capture.json', dict(schema=HANDOFF, schema_version=1, selection=handoff_ref,
        process_id=flow.process, thread_id=flow.thread, launch=launch, matched=flow.rows,
        observation_frontiers=flow.frontiers, reference_test='LIVE_RETURN_OBJECT_IS_SELECTED_ENTRY_FORMAL',
        values_saved=False, original_execution=True, performance_measurement=False))
    samples = [dict(scalars.samples[row['producer']], handoff_index=index)
        for index, row in enumerate(flow.rows) if row['producer'] in scalars.samples]
    _write(out/'capture.json', dict(schema=CAPTURE, schema_version=1, selection=design_ref,
        source=design['source'], case=design['case'], case_index=design['case_index'], handoff_capture=raw_flow,
        process_id=os.getpid(), thread_id=flow.thread, table_versions=scalars.versions,
        samples=samples, observation_frontiers=dict(scalars.frontiers),
        table_scope='EXACT_BUILTIN_ROW_SNAPSHOTS_AT_SELECTED_EXPRESSION_BOUNDARIES',
        original_execution=True, performance_measurement=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    capture(args.selection, args.out)


if __name__ == '__main__':
    main()
