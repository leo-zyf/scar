"""Observe two actual score returns in an original, selected loop activation."""

import argparse
import builtins
from collections import Counter
import os
from pathlib import Path
import sys
from threading import get_ident
from types import BuiltinMethodType, CodeType

from scar.analysis.scalar_score_reuse_evidence import CAPTURE, DESIGN, _need
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_table_handoff_observe import _int
from scar.runtime.scalar_transition_observe import _Transitions, _unchanged
from scar.runtime.source_value_observe import _codes, _coordinate, _instance_field, _target_coordinate
from scar.trace.code_snapshot import describe_code, site

_SORTED = builtins.sorted


class _Origins:
    def __init__(self, design, transitions):
        self.design, self.transitions = design, transitions
        self.monitor, self.events = sys.monitoring, sys.monitoring.events
        self.slot = next(i for i in range(5,-1,-1) if self.monitor.get_tool(i) is None)
        roles = dict(parent=design['caller'], generator=design['generator'], score=transitions.design['score'])
        arms = {row['arm']:row for row in design['arms']}
        roles['comprehension'] = arms['SORTED_DELEGATION']['code']
        self.wanted = {_target_coordinate(value):(role,value) for role,value in roles.items()}
        self.arms, self.codes, self.running = arms, {}, True
        self.live_scores, self.batches, self.sorted_calls, self.yielded = {}, {}, {}, {}
        self.generators = {}
        self.pending_reads, self.pending_stores = {}, {}
        self.state_events, self.state_roots = [], []
        self.frontiers = Counter()
        self.sequence = 0

    def tick(self):
        self.sequence += 1
        return self.sequence

    def attach(self, event, args):
        if (not self.running or event != 'exec' or not args or type(args[0]) is not CodeType
                or args[0].co_filename != self.design['source']['uri']):
            return
        masks = dict(parent=self.events.CALL|self.events.PY_RETURN|self.events.INSTRUCTION,
            score=self.events.PY_START|self.events.PY_RETURN,
            generator=self.events.CALL|self.events.PY_YIELD|self.events.PY_RETURN,
            comprehension=self.events.PY_YIELD)
        for code in _codes(args[0]):
            selected = self.wanted.get(_coordinate(code))
            if selected is None:
                continue
            role, descriptor = selected
            _need(describe_code(code) == descriptor,'original score predecessor code differs')
            self.codes[code] = role
            self.monitor.set_local_events(self.slot,code,masks[role])

    def _owner(self, frame):
        if frame and frame.f_back and self.codes.get(frame.f_back.f_code) == 'parent':
            return frame.f_back
        return None

    def start(self, code, _offset):
        frame = sys._getframe(1)
        caller = frame.f_back
        if not self.running or caller is None:
            return
        role = self.codes.get(caller.f_code)
        if role == 'generator':
            generator, arm = caller, self.arms['DIRECT_YIELD']
        elif role == 'comprehension':
            generator, arm = caller.f_back, self.arms['SORTED_DELEGATION']
            if generator is None or self.codes.get(generator.f_code) != 'generator':
                self.frontiers['NO_ORIGINAL_COMPREHENSION_OWNER'] += 1
                return
        else:
            return
        owner = self._owner(generator)
        if (owner is None or caller.f_lasti != arm['score_call']['offset']
                or site(caller.f_code,caller.f_lasti) != arm['score_call']):
            self.frontiers['NO_ORIGINAL_SCORE_OWNER_OR_SITE'] += 1
            return
        ts = self.transitions.design
        local = frame.f_locals
        try:
            receiver, argument = local[ts['score_receiver']],local[ts['score_argument']]
            _need(receiver is owner.f_locals[self.design['loop_binding']['receiver']]
                  and argument is caller.f_locals[arm['move_name']], 'score actuals differ from original source locals')
            table = self.transitions._table(frame.f_globals[ts['table']])
            if self.transitions.counts[table[3]] >= ts['per_version']:
                return
            prior = self.generators.setdefault(owner,generator)
            _need(prior is generator,'original bound frame has another live generator')
            row = dict(arm=arm['arm'], first_score_site=arm['score_call'], first_enter_sequence=self.tick(),
                       thread_id=get_ident())
            self.live_scores[frame] = (generator,owner,receiver,argument,table,row)
            self.monitor.set_events(self.slot,self.events.PY_UNWIND)
        except (ValueError,TypeError,KeyError):
            self.frontiers['ORIGINAL_FIRST_SCORE_INPUTS_OPEN'] += 1

    def _score_end(self, frame):
        match = self.live_scores.pop(frame,None)
        if match is not None and not self.live_scores:
            self.monitor.set_events(self.slot,0)
        return match

    def unwind(self, _code, _offset, _exception):
        if self._score_end(sys._getframe(1)) is not None:
            self.frontiers['FIRST_SCORE_EXCEPTION_UNWIND'] += 1

    def finish(self, code, offset, result):
        frame = sys._getframe(1)
        kind = self.codes[code]
        if kind == 'generator':
            self.batches.pop(frame,None)
            self.sorted_calls.pop(frame,None)
        elif kind == 'parent':
            self.pending_reads.pop(frame,None)
            self.pending_stores.pop(frame,None)
            self.yielded.pop(frame,None)
            generator = self.generators.pop(frame,None)
            self.batches.pop(generator,None)
            self.sorted_calls.pop(generator,None)
        else:
            match = self._score_end(frame)
            if match is None:
                return
            generator,owner,receiver,argument,table,row = match
            token,root,anchors,version = table
            try:
                returned = site(code,offset)
                _need(row['thread_id'] == get_ident() and returned['opname'] in ('RETURN_VALUE','RETURN_CONST')
                      and frame.f_globals[self.transitions.design['table']] is root and _unchanged(root,anchors),
                      'score did not return normally with original table root/rows')
                row.update(first_result=_int(result), first_return_site=returned, first_return_sequence=self.tick(),
                           table_version=version, root_token=token)
                records = self.batches.setdefault(generator,[])
                _need(len(records) < 256 and len(self.batches) < 96,'selected generator sample exceeds bounded record budget')
                records.append((owner,receiver,argument,table,row))
            except (TypeError,ValueError,KeyError):
                self.frontiers['FIRST_NORMAL_SCORE_OR_OBSERVATION_BUDGET_OPEN'] += 1

    def _table_state(self, frame):
        contract = self.design['state']
        local = frame.f_locals
        root = _instance_field(local[contract['owner']],contract['field'])
        _need(type(root) is dict,'sampled table must be original builtin dict')
        return root,local[contract['key']]

    def call(self, code, offset, target, arg0):
        frame = sys._getframe(1)
        role = self.codes[code]
        sorted_site = self.arms['SORTED_DELEGATION']['sorted_call']
        if role == 'generator' and offset == sorted_site['offset']:
            if target is _SORTED and type(target) is BuiltinMethodType and site(code,offset) == sorted_site:
                self.sorted_calls[frame] = self.tick()
            else:
                self.frontiers['SORTED_CARRIER_OPEN'] += 1
        state = self.design['state']
        if role != 'parent' or len(self.state_events) >= 4 or offset != state['lookup']['offset']:
            return
        try:
            root,key = self._table_state(frame)
            target_selected = target is dict.get and arg0 is root
            if type(target) is BuiltinMethodType:
                target_selected = target.__self__ is root and target.__name__ == 'get'
            _need(target_selected and site(code,offset) == state['lookup'],'actual lookup is not the declared builtin dict method')
            self.pending_reads[frame] = (root,key,self.tick(),len(root))
        except (ValueError,TypeError,KeyError):
            self.frontiers['ORIGINAL_DICT_READ_OPEN'] += 1

    def instruction(self, _code, offset):
        if len(self.state_events) >= 4:
            return
        state = self.design['state']
        if offset not in (state['write']['offset'],state['after']['offset']):
            return
        frame = sys._getframe(1)
        if offset == state['write']['offset']:
            seen = self.pending_reads.get(frame)
            if seen is None:
                return
            try:
                root,key = self._table_state(frame)
                _need(seen[0] is root and seen[1] is key,'actual dict root/key differ from original read')
                self.pending_stores[frame] = (seen,self.tick())
            except (ValueError,TypeError,KeyError):
                self.frontiers['ORIGINAL_DICT_READ_WRITE_IDENTITY_OPEN'] += 1
            return
        sealed = self.pending_stores.pop(frame,None)
        if sealed is None:
            return
        (root,key,read,size),write = sealed
        if all(root is not prior for prior in self.state_roots):
            self.state_roots.append(root)
        token = next(i for i,prior in enumerate(self.state_roots,1) if root is prior)
        self.state_events.append(dict(table_root_token=token, caller_frame=self.transitions._identity(frame),
            thread_id=get_ident(), read_site=state['lookup'], write_site=state['write'], normal_next=state['after'],
            read_sequence=read,write_sequence=write,normal_sequence=self.tick(),same_key_object=True,
            exact_builtin_dict=True,size_at_read=size,size_after_write=len(root)))
        if len(self.state_events) == 4:
            self.monitor.set_local_events(self.slot,frame.f_code,self.events.PY_RETURN)
            self.pending_reads.clear()
            self.pending_stores.clear()

    def yield_value(self, code, offset, value):
        frame = sys._getframe(1)
        role = self.codes[code]
        arm = self.arms['SORTED_DELEGATION']
        if role == 'comprehension':
            generator = frame.f_back
            if generator is not None and offset == arm['inner_yield']['offset']:
                found = self._same_tuple(self.batches.get(generator,[]),arm['arm'],value)
                if found is not None:
                    found[-1]['inner_yield_site'] = arm['inner_yield']
                    found[-1]['inner_yield_sequence'] = self.tick()
            return
        if offset != arm['yield_site']['offset']:
            arm = self.arms['DIRECT_YIELD']
            if offset != arm['yield_site']['offset']:
                return
        records = self.batches.get(frame)
        if records is None:
            return
        packet = self._same_tuple(records,arm['arm'],value)
        order = self.sorted_calls.get(frame)
        if (packet is None or (arm['arm'] == 'SORTED_DELEGATION' and
                ('inner_yield_site' not in packet[-1] or order is None))):
            self.frontiers['UNMATCHED_ORIGINAL_SCORE_YIELD'] += 1
            return
        owner,receiver,argument,table,row = packet
        records.pop(next(i for i,value in enumerate(records) if value is packet))
        row.update(yield_site=arm['yield_site'],yield_sequence=self.tick())
        if arm['arm'] == 'SORTED_DELEGATION':
            row['sorted_call_sequence'] = order
        self.yielded[owner] = (receiver,argument,table,row)

    @staticmethod
    def _same_tuple(records, arm, value):
        if type(value) is not tuple or tuple.__len__(value) != 2 or type(value[0]) is not int:
            return None
        candidates = [row for row in records if row[2] is value[1] and row[-1]['arm'] == arm
                      and row[-1]['first_result'] == value[0]]
        return candidates[0] if len(candidates) == 1 else None

    def close(self):
        self.running = False
        self.monitor.set_events(self.slot,0)
        for code in self.codes:
            self.monitor.set_local_events(self.slot,code,0)
        for retained in (self.live_scores,self.batches,self.sorted_calls,self.yielded,
                         self.pending_reads,self.pending_stores,self.generators):
            retained.clear()


class _ReuseTransitions(_Transitions):
    origins = None

    def _begin_move(self, frame, parent, selected):
        super()._begin_move(frame,parent,selected)
        handoff = self.origins.yielded.pop(parent,None)
        state = self.active_moves.get(frame)
        if state is None:
            return
        if handoff is None:
            self.origins.frontiers['TRANSITION_WITHOUT_SELECTED_SCORE_YIELD'] += 1
            return
        receiver,argument,table,row = handoff
        loc = parent.f_locals
        bindings = self.origins.design['loop_binding']
        try:
            _need(get_ident() == row['thread_id'] == state['data']['thread_id']
                  and loc[bindings['receiver']] is receiver is state['receiver']
                  and loc[bindings['move']] is argument is state['argument']
                  and _int(loc[bindings['score']]) == row['first_result']
                  and table[0] == state['table'][0] and table[1] is state['table'][1]
                  and table[3] == state['table'][3] and _unchanged(table[1],table[2]),
                  'real loop val and actual receiver/move/table identity do not meet at the transition')
            row.update(transition_enter_sequence=self.origins.tick(),same_original_loop_receiver=True,
                       same_original_move_object=True,same_table_root_and_content=True)
            state['data']['score_reuse'] = row
        except (TypeError,KeyError,ValueError):
            self.origins.frontiers['ORIGINAL_LOOP_SCORE_TRANSITION_IDENTITY_OPEN'] += 1

    def close(self):
        super().close()
        if self.origins is not None and self.origins.running:
            self.origins.close()


def capture(selection,out):
    reference = _ref(selection,DESIGN)
    design = read_artifact(reference)
    old = read_artifact(design['transition_selection'])
    read_artifact(design['source'],json_document=False)
    transition = _ReuseTransitions(old)
    monitor,events = transition.monitor,transition.events
    monitor.use_tool_id(transition.slot,CAPTURE + '.transition')
    origins = _Origins(design,transition)
    transition.origins = origins
    monitor.use_tool_id(origins.slot,CAPTURE)
    tools = ((transition,((events.PY_START,transition.begin),(events.PY_RETURN,transition.finish),
                         (events.CALL,transition.call))),
             (origins,((events.PY_START,origins.start),(events.PY_RETURN,origins.finish),
                      (events.CALL,origins.call),(events.PY_YIELD,origins.yield_value),
                      (events.PY_UNWIND,origins.unwind),(events.INSTRUCTION,origins.instruction))))
    try:
        for selected,callbacks in tools:
            for event,callback in callbacks:
                monitor.register_callback(selected.slot,event,callback)
            sys.addaudithook(selected.attach)
        launch = run_original_script(design['source']['uri'],[])
    finally:
        transition.close()
        for selected,callbacks in tools:
            for event,_ in callbacks:
                monitor.register_callback(selected.slot,event,None)
            monitor.free_tool_id(selected.slot)
    return _write(out,dict(schema=CAPTURE,schema_version=1,selection=reference,source=design['source'],
        score_evidence=old['score_evidence'],launch=launch,process_id=os.getpid(),state_fields=transition.state_fields,
        globals=transition.globals,builtins=transition.builtins,builtin_abs_verified=bool(transition.builtins),
        table_versions={version:transition.versions[version] for version in transition.counts},
        samples=transition.samples,observation_frontiers=dict(transition.frontiers),
        scoring_frontiers=dict(origins.frontiers),state_events=origins.state_events,
        original_execution=True,performance_measurement=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    capture(args.selection,args.out)


if __name__ == '__main__':
    main()
