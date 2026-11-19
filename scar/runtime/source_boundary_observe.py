"""Observe once-only CALL boundaries without changing the program code or operands."""

import argparse
from dis import get_instructions
import json
import os
from pathlib import Path
import sys
from threading import get_native_id
from time import perf_counter_ns, thread_time_ns
from types import CodeType

from scar.analysis.target_inventory import _cpu_list
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.entry_runner import run_original_script
from scar.trace.call_binding import target_metadata
from scar.trace.code_snapshot import code_key

SOURCE = 'scar.python-source'
DESIGN = 'scar.source-boundary-observation-design'
OBSERVATION = 'scar.source-boundary-observation'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def lane_snapshot(cpu):
    _require(type(cpu) is int and cpu >= 0, 'one physical CPU index required')
    root = Path(f'/sys/devices/system/cpu/cpu{cpu}')
    fields = {}
    for name in ('physical_package_id', 'die_id', 'core_id', 'thread_siblings_list'):
        file = root / 'topology' / name
        fields[name] = dict(path=str(file), value=file.read_text().strip())
    numa_nodes = sorted(int(entry.name[4:]) for entry in root.glob('node[0-9]*'))
    siblings = _cpu_list(fields['thread_siblings_list']['value'])
    _require(cpu in siblings and len(numa_nodes) == 1, 'selected owner lacks an exact Linux CPU/NUMA identity')
    return dict(cpu_id=cpu, numa_node=numa_nodes[0], thread_siblings=siblings,
        physical_core=[int(fields[key]['value']) for key in ('physical_package_id', 'die_id', 'core_id')],
        topology=fields, required_affinity=[cpu])


def source_cuts(source, cuts, optimize):
    raw = read_artifact(source, json_document=False)
    module = compile(raw, source['uri'], 'exec', optimize=optimize, dont_inherit=True)
    bodies = {}
    pending = [module]
    while pending:
        code = pending.pop()
        bodies[code_key(code)] = code
        pending.extend(member for member in code.co_consts if type(member) is CodeType)
    offsets = {}
    for key, row in cuts.items():
        site = row['caller_site']
        instructions = list(get_instructions(bodies[site['code']]))
        index, = (index for index, op in enumerate(instructions) if op.offset == site['offset'])
        op = instructions[index]
        _require(op.opname == site['opname'] and op.opname.startswith('CALL')
            and list(op.positions) == site['position'] and index+1 < len(instructions),
            'source CALL no longer has the recorded CodeID, BytecodeOffset and original span')
        offsets[key] = instructions[index+1].offset
    return code_key(module), offsets


class _Meter:
    def __init__(self, design, root_code):
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.design, self.root_code = design, root_code
        self.codes, self.samples, self.pending = [], {}, {}
        self.calls = {}
        self.off_target_calls = 0
        for key, cut in design['cuts'].items():
            site = cut['caller_site']
            by_code = self.calls.setdefault(site['code'], {})
            _require(site['offset'] not in by_code, 'each once-only source location requires a unique case')
            by_code[site['offset']] = key, cut
        self.by_code = {}
        self.tool = next(index for index in range(5, -1, -1) if self.monitor.get_tool(index) is None)

    def __enter__(self):
        self.monitor.use_tool_id(self.tool, 'scar.source-boundary-observation')
        self.monitor.register_callback(self.tool, self.events.PY_START, self.attach_at_entry)
        self.monitor.register_callback(self.tool, self.events.CALL, self.start_call)
        self.monitor.register_callback(self.tool, self.events.INSTRUCTION, self.end_call)
        self.monitor.set_events(self.tool, self.events.PY_START)
        return self

    def __exit__(self, *args):
        self.monitor.set_events(self.tool, 0)
        for code in self.codes:
            self.monitor.set_local_events(self.tool, code, 0)
        for event in (self.events.PY_START, self.events.CALL, self.events.INSTRUCTION):
            self.monitor.register_callback(self.tool, event, None)
        self.monitor.free_tool_id(self.tool)

    def attach_at_entry(self, code, offset):
        if code.co_name != '<module>' or code.co_filename != self.design['source']['uri']:
            return
        _require(code_key(code) == self.root_code, 'executed entry differs from the admitted code')
        pending = [code]
        while pending:
            child = pending.pop()
            identity = code_key(child)
            if identity in self.calls:
                self.codes.append(child)
                self.by_code[child] = identity, self.calls[identity]
                self.monitor.set_local_events(self.tool, child, self.events.CALL)
            pending.extend(member for member in child.co_consts if type(member) is CodeType)
        _require(len(self.by_code) == len(self.calls), 'selected call bodies did not enter the original module')
        self.monitor.set_events(self.tool, 0)

    def start_call(self, code, offset, target, _argument):
        site = self.by_code[code][1].get(offset)
        if site is None:
            self.off_target_calls += 1
            return self.monitor.DISABLE
        key, cut = site
        observed = target_metadata(target)
        _require(observed == cut['target'] and key not in self.samples and code not in self.pending,
                 'runtime call target differs from the planned CodeID/ABI or occurs more than once')
        row = dict(target=observed, entry_code=self.by_code[code][0],
                   call_offset=offset, after_call_offset=cut['after_call_offset'], thread_id=get_native_id())
        self.pending[code] = key, row
        self.monitor.set_local_events(self.tool, code, self.events.CALL | self.events.INSTRUCTION)
        row['wall'] = [perf_counter_ns()]
        row['thread_cpu'] = [thread_time_ns()]
        return self.monitor.DISABLE

    def end_call(self, code, offset):
        key, row = self.pending[code]
        if offset != row['after_call_offset']:
            return
        row['thread_cpu'].append(thread_time_ns())
        row['wall'].append(perf_counter_ns())
        self.samples[key] = row
        del self.pending[code]
        self.monitor.set_local_events(self.tool, code, self.events.CALL)


def observe(reference):
    design = read_artifact(reference)
    context = launch_context(design['context']['argv'])
    lane = lane_snapshot(design['lane']['cpu_id'])
    _require(context == design['context'] and lane == design['lane']
        and len(context['cpu_ids']) == 1 and sys.gettrace() is None and sys.getprofile() is None,
        'owner timing requires the exact single-core launch and no other trace hook')
    root, offsets = source_cuts(design['source'], design['cuts'], context['interpreter']['optimize'])
    _require(offsets == {name:cut['after_call_offset'] for name,cut in design['cuts'].items()},
        'execution site end offsets differ from the frozen design')
    meter = _Meter(design, root)
    with meter:
        begin = perf_counter_ns()
        execution = run_original_script(design['source']['uri'], context['argv'])
        end = perf_counter_ns()
    _require(set(meter.samples) == set(design['cuts']) and not meter.pending and launch_context(context['argv']) == context,
        'original subprocess did not observe all selected prep and publication sites')
    return dict(schema=OBSERVATION, schema_version=1, design=reference, context=context, lane=lane,
        diagnostic=True, instrument=artifact_reference(__file__, SOURCE), launch=execution,
        pid=os.getpid(), owner_thread_id=get_native_id(), run=[begin, end], samples=meter.samples,
        discarded_call_site_callbacks=meter.off_target_calls, probe_scope='TWO_SOURCE_CODE_OBJECTS_NO_LOOP_BODY_TIMING')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args(argv)
    reference = artifact_reference(args.design.resolve(strict=True), DESIGN)
    _require(reference['sha256'] == args.sha256 and not args.report.exists(), 'exact design and fresh report required')
    row = observe(reference)
    with args.report.open('x') as stream:
        json.dump(row, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
