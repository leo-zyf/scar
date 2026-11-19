"""Observe bounded producer returns and consumer operands in one original run."""

import argparse
from dis import get_instructions
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from threading import get_native_id
from time import perf_counter_ns, thread_time_ns
from types import CodeType, GetSetDescriptorType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.entry_runner import run_original_script
from scar.runtime.source_boundary_observe import lane_snapshot
from scar.trace.code_snapshot import code_key

DESIGN = 'scar.source-value-route-design'
OBSERVATION = 'scar.source-value-route-observation'
CLOCK = 'MONOTONIC_PERF_COUNTER_NS_AND_THREAD_CPU_NS'
_ENTRY_GRAPH = 'scar.bounded-entry-calls'
_MAX_FIELDS_BYTES = 16 * 1024 * 1024


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _codes(root):
    pending = [root]
    while pending:
        code = pending.pop()
        yield code
        pending.extend(child for child in code.co_consts if type(child) is CodeType)


def _coordinate(code):
    return code.co_filename, code.co_qualname, code.co_firstlineno


def _target_coordinate(target):
    return target['filename'], target['qualname'], target['first_line']


def _instance_field(value, field):
    # Bypass custom attribute access, and admit only the native instance-dict
    # descriptor. A property or user descriptor is never invoked here.
    cls = type(value)
    for base in type.__getattribute__(cls, '__mro__'):
        descriptor = type.__getattribute__(base, '__dict__').get('__dict__')
        if descriptor is not None:
            _require(type(descriptor) is GetSetDescriptorType, 'operand instance has a custom dictionary')
            storage = descriptor.__get__(value)
            _require(type(storage) is dict, 'ordinary operand instance dictionary required')
            return storage[field]
    raise ValueError('operand instance has no native dictionary')


def _field(frame, returned, selector):
    origin = selector['origin']
    value = returned if origin == 'return' else frame.f_locals[origin['local']]
    array_type = sys.modules['numpy'].ndarray
    for operation, operand in selector['steps']:
        if operation == 'tuple':
            _require(type(value) is tuple and type(operand) is int, 'exact built-in tuple operand required')
            value = value[operand]
        elif operation == 'array':
            _require(type(value) is array_type and type(operand) is int and value.ndim >= 2,
                'ndarray projection requires an integer view index')
            value = value[operand]
        elif operation == 'instance':
            _require(type(operand) is str, 'operand field requires an exact instance-dict key')
            value = _instance_field(value, operand)
        else:
            raise ValueError('unsupported operand selector')
    _require(type(value) is array_type and value.flags.c_contiguous and not value.dtype.hasobject,
        'a contiguous plain ndarray is required for exact operand evidence')
    return value


def prepare(reference):
    design = read_artifact(reference)
    graph_ref = design['source_graph']
    _require(design['schema_version'] == 1 and graph_ref['schema'] == _ENTRY_GRAPH,
        'source value design requires an exact bounded call graph')
    graph = read_artifact(graph_ref)
    _require(read_artifact(graph['program'])['entry'] == design['entry']['uri']
        and design['entry']['schema'] == 'scar.python-source', 'value design belongs to another entry')
    read_artifact(design['entry'], json_document=False)
    endpoints = design['endpoints']
    _require(0 < len(endpoints) <= 32 and 0 < len(design['routes']) <= 32,
        'source value observation requires a finite selected route')
    seen, compiled, return_sites = set(), {}, {}
    for name, endpoint in endpoints.items():
        pointer, index = endpoint['observation'].rsplit('/', 1)
        _require(pointer == '#/observed_targets' and index.isdecimal() and index not in seen,
            'one original CALL observation is required per endpoint')
        seen.add(index)
        row = graph['observed_targets'][int(index)]
        observed = row['observation']
        target, caller = endpoint['target_code'], endpoint['caller_site']
        source = endpoint['source']
        _require(row['entry_scope_reachable'] and observed['target_code'] == target
            and observed['caller_site'] == caller and source['uri'] == target['filename']
            and source['schema'] == 'scar.python-source' and endpoint['phase'] in ('ENTRY', 'RETURN')
            and len(endpoint['fields']) <= 32, 'value endpoint differs from the selected observed call')
        if source['uri'] not in compiled:
            raw = read_artifact(source, json_document=False)
            root = compile(raw, source['uri'], 'exec', dont_inherit=True, optimize=sys.flags.optimize)
            compiled[source['uri']] = source, {_coordinate(code): code for code in _codes(root)}
        original, codes = compiled[source['uri']]
        code = codes[_target_coordinate(target)]
        _require(original == source and code_key(code) == target['key'], 'value endpoint source identity changed')
        return_sites[name] = {op.offset for op in get_instructions(code)
                              if op.opname in ('RETURN_VALUE', 'RETURN_CONST')}
    for endpoint in endpoints.values():
        parent = endpoint.get('parent')
        _require(parent is None or (endpoints[parent]['phase'] == 'RETURN'
            and endpoints[parent]['target_code']['key'] == endpoint['caller_site']['code']),
            'observed parent must be the exact selected direct caller')
    for route in design['routes']:
        producer, consumer = (endpoints[route[key]] for key in ('producer', 'consumer'))
        _require(producer['phase'] == 'RETURN' and consumer['phase'] == 'ENTRY' and route['fields'],
            'route needs a producer normal return and a consumer entry')
        for field in route['fields']:
            _require(field['producer_field'] in producer['fields'] and field['consumer_field'] in consumer['fields'],
                'route selects an uncaptured operand')
    return design, return_sites


class _Recorder:
    def __init__(self, design, return_sites):
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.endpoints = design['endpoints']
        self.definitions = design['routes']
        self.return_sites = return_sites
        self.sources = {}
        for name, endpoint in self.endpoints.items():
            coordinate = _target_coordinate(endpoint['target_code'])
            self.sources.setdefault(coordinate[0], {}).setdefault(coordinate, []).append(name)
        self.attached, self.frames, self.pending, self.activations = {}, {}, {}, {}
        self.arrays = {}
        self.captured_bytes = self.ordinal = 0
        self.result = None
        self.owner = get_native_id()
        self.tool = next(index for index in range(5, -1, -1) if self.monitor.get_tool(index) is None)
        self.closed = False

    def __enter__(self):
        self.monitor.use_tool_id(self.tool, OBSERVATION)
        self.monitor.register_callback(self.tool, self.events.PY_START, self.start)
        self.monitor.register_callback(self.tool, self.events.PY_RETURN, self.finish)
        # The process is dedicated to this one capture. The source-load hook
        # remains inert once capture closes, and there is no global PY_* mask.
        sys.addaudithook(self.attach)
        return self

    def __exit__(self, *exc):
        self.disarm()
        for event in (self.events.PY_START, self.events.PY_RETURN):
            self.monitor.register_callback(self.tool, event, None)
        self.monitor.free_tool_id(self.tool)

    def disarm(self):
        self.closed = True
        for code in self.attached:
            self.monitor.set_local_events(self.tool, code, 0)
        self.attached.clear()
        self.arrays.clear()

    def attach(self, event, arguments):
        if self.closed or event != 'exec' or not arguments or type(arguments[0]) is not CodeType:
            return
        root = arguments[0]
        selected = self.sources.get(root.co_filename)
        if selected is None:
            return
        for code in _codes(root):
            names = selected.get(_coordinate(code))
            if names is None:
                continue
            key = code_key(code)
            _require(all(self.endpoints[name]['target_code']['key'] == key for name in names),
                'executed operand source differs from its selected CodeID')
            self.attached[code] = names
            events = self.events.PY_START
            if any(self.endpoints[name]['phase'] == 'RETURN' for name in names):
                events |= self.events.PY_RETURN
            self.monitor.set_local_events(self.tool, code, events)

    def start(self, code, _offset):
        frame = sys._getframe(1)
        caller = frame.f_back
        if caller is None:
            return
        for name in self.attached[code]:
            endpoint = self.endpoints[name]
            expected = endpoint['caller_site']
            if caller.f_lasti != expected['offset'] or code_key(caller.f_code) != expected['code']:
                continue
            _require(name not in self.activations and get_native_id() == self.owner,
                'selected endpoint activated repeatedly or on an unexpected thread')
            parent = self.frames.get(caller)
            if endpoint.get('parent') is not None:
                _require(parent is not None and parent[0] == endpoint['parent'],
                    'selected caller activation did not match its planned parent')
            self.ordinal += 1
            row = dict(target_code=endpoint['target_code'], caller_site=expected,
                activation=self.ordinal, parent_activation=None if parent is None else parent[1],
                thread_id=self.owner, return_offset=None, wall_ns=[perf_counter_ns()],
                thread_cpu_ns=[thread_time_ns()], fields={})
            self.activations[name] = row
            if endpoint['phase'] == 'RETURN':
                self.pending[frame] = name
                self.frames[frame] = name, self.ordinal
            else:
                self.capture(name, frame, None)
                self.complete()
            break

    def finish(self, _code, offset, returned):
        end_cpu, end_wall = thread_time_ns(), perf_counter_ns()
        frame = sys._getframe(1)
        name = self.pending.pop(frame, None)
        if name is None:
            return
        del self.frames[frame]
        _require(offset in self.return_sites[name], 'observed return is outside the admitted code site')
        row = self.activations[name]
        row['return_offset'] = offset
        row['thread_cpu_ns'].append(end_cpu)
        row['wall_ns'].append(end_wall)
        self.capture(name, frame, returned)
        self.complete()

    def capture(self, name, frame, returned):
        for field, selector in self.endpoints[name]['fields'].items():
            array = _field(frame, returned, selector)
            self.captured_bytes += array.nbytes
            _require(self.captured_bytes <= _MAX_FIELDS_BYTES, 'value observation byte budget exceeded')
            row = dict(dtype=array.dtype.str, shape=list(array.shape), strides=list(array.strides),
                nbytes=array.nbytes, sha256=sha256(memoryview(array).cast('B')).hexdigest())
            self.activations[name]['fields'][field] = row
            # Keep the allocation live through both samples. Only equality
            # classes leave the process, never object IDs or virtual addresses.
            self.arrays[name, field] = array, array.__array_interface__['data'][0]

    def complete(self):
        if len(self.activations) != len(self.endpoints) or self.pending:
            return
        result = []
        for route in self.definitions:
            producer, consumer = route['producer'], route['consumer']
            before, after = self.activations[producer], self.activations[consumer]
            _require(before['wall_ns'][1] <= after['wall_ns'][0], 'observed operands are out of producer order')
            fields = []
            for field in route['fields']:
                left, right = field['producer_field'], field['consumer_field']
                a, b = before['fields'][left], after['fields'][right]
                _require(all(a[key] == b[key] for key in ('dtype', 'shape', 'nbytes', 'sha256')),
                    'consumer contents differ from the selected producer return: ' + field['name'])
                x, addr_x = self.arrays[producer, left]
                y, addr_y = self.arrays[consumer, right]
                relation = 'EQUAL_OBSERVED_BYTES'
                if x is y:
                    relation = 'SAME_OBJECT_SAME_BYTES'
                elif addr_x == addr_y and a['nbytes'] > 0 and a['strides'] == b['strides']:
                    relation = 'SAME_REGION_SAME_BYTES'
                fields.append(dict(field, relation=relation))
            result.append(dict(route, fields=fields))
        self.result = result
        self.disarm()


def observe(reference):
    design, sites = prepare(reference)
    context = launch_context(design['argv'])
    lane = lane_snapshot(design['lane']['cpu_id'])
    _require(context['cpu_ids'] == lane['required_affinity'] and design['lane'] == lane
        and context == design['context'] and not sys.gettrace() and not sys.getprofile(),
        'value observation requires the original launch and one physical CPU lane')
    recorder = _Recorder(design, sites)
    with recorder:
        begin = perf_counter_ns()
        launch = run_original_script(design['entry']['uri'], design['argv'])
        end = perf_counter_ns()
    _require(recorder.result is not None,
        'original execution did not produce every selected value endpoint')
    return dict(schema=OBSERVATION, schema_version=1, design=reference, pid=os.getpid(),
        owner_thread_id=recorder.owner, lane=lane, launch=launch, context=context, run_ns=[begin, end],
        clock=CLOCK, diagnostic=True, activations=recorder.activations, routes=recorder.result,
        instrument=artifact_reference(__file__, 'scar.python-source'), captured_field_bytes=recorder.captured_bytes,
        value_scope='SAME_PROCESS_SELECTED_ARRAY_ENDPOINTS', proves_future_value=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    _require(not args.out.exists(), 'fresh value observation path required')
    row = observe(artifact_reference(args.design.resolve(strict=True), DESIGN))
    with args.out.open('x') as stream:
        json.dump(row, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
