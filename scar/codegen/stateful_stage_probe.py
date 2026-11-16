"""Render serial canonical probes around admitted original stage bodies.

This is harness/control code for the existing emitter, not a mathematical
backend. The caller admits source, plan, capture and the full DAG signature.
"""
from hashlib import sha256
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _pin(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(_ROOT)), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _fragment(path, source, start, end):
    _need(source.count(start) == source.count(end) == 1, 'unique original helper markers required')
    a, b = source.index(start), source.index(end)
    _need(a < b, 'helper source markers are reversed')
    text = source[a:b]
    return text, dict(source=_pin(path), char_start=a, char_end_exclusive=b,
        line_start=source.count('\n', 0, a)+1, line_end=source.count('\n', 0, b-1)+1,
        sha256=sha256(text.encode()).hexdigest())


def render_stage_dag_probe(region, plan, capture, stage_indices, cpus, *, max_payload_bytes=256*2**20):
    """Return source/provenance only; never compile, sample or call a provider."""
    _need(type(cpus) in (tuple, list) and len(cpus) == 1 and type(cpus[0]) is int
          and 0 <= cpus[0] < 2**20 and type(plan.get('cpu_threads')) is int and plan['cpu_threads'] == 1,
          'the original serial one-CPU stateful profile is required')
    _need(type(max_payload_bytes) is int and 0 <= max_payload_bytes <= 2**63-1,
          'bounded nonnegative canonical payload budget required')
    _need(type(stage_indices) in (tuple, list) and stage_indices
          and all(type(index) is int and index >= 0 for index in stage_indices)
          and len(set(stage_indices)) == len(stage_indices), 'distinct original stage indices required')
    _need(capture.get('schema') == 'scar.stateful-stage-source-capture' and capture.get('schema_version') == 1,
          'original stage source capture required')
    stage_map = {stage['index']: stage for stage in plan['stages']}
    captured = {stage['index']: stage for stage in capture['stages']}
    _need(len(stage_map) == len(plan['stages']) and len(captured) == len(capture['stages']),
          'duplicate original stage or capture indices')
    nodes = list(region.nodes); order = {node.id: index for index, node in enumerate(nodes)}
    arrays = [row.id for row in region.inputs if row.kind == 'array']
    scalars = [row.id for row in region.inputs if row.kind == 'scalar']
    shapes = [plan['input_shapes'][name] for name in arrays]
    _need(all(type(shape) in (tuple, list) and len(shape) == 2
              and all(type(n) is int and 0 <= n <= (2**63-1)//8 for n in shape) for shape in shapes),
          'representable original rank-two array shapes required')
    slots = list(plan['workspace_slots']); slot_indexes = {slot['id']: index for index, slot in enumerate(slots)}
    capacities = [slot['capacity_bytes'] for slot in slots]
    _need(len(slot_indexes) == len(slots) and all(type(n) is int and 0 <= n <= 2**63-1 and n % 8 == 0 for n in capacities),
          'original independent physical-slot capacities required')
    external = sum(rows*cols*8 for rows, cols in shapes)
    payload = external + sum(capacities)
    _need(payload == plan['total_region_array_payload_bytes'] and external == plan['external_array_bytes']
          and sum(capacities) == plan['workspace_payload_bytes']
          and payload <= max_payload_bytes and payload <= plan['memory_budget_bytes'],
          'full original array plus physical-slot payload exceeds canonical budget')
    loop_path = _HERE/'loop_v1.cpp.in'; loop = loop_path.read_text()
    backend_path = _HERE/'stateful_v1.cpp.in'; backend = backend_path.read_text()
    fragments = {}

    def helper(name, path, text, start, end):
        body, fragments[name] = _fragment(path, text, start, end)
        return body

    affinity = helper('affinity', loop_path, loop, 'class ThreadAffinity {', '\n#endif\n\nconstexpr int64_t modeled_rows')
    floating = helper('floating', loop_path, loop, 'class FloatingScope {', '\nbool valid_array(')
    finite = helper('finite', backend_path, backend, 'bool finite(double value) {', 'bool valid_array(')
    scalar_helpers = helper('scalar_helpers', backend_path, backend, 'struct Scalar {', '\nint32_t run(')
    functions, receipts = [], []
    for component_index, index in enumerate(stage_indices):
        _need(index in stage_map and index in captured, 'requested original stage was not captured')
        stage, record = stage_map[index], captured[index]
        _need(stage['kind'] in ('COMPUTE', 'SCALAR') and type(record['body']) is str
              and sha256(record['body'].encode()).hexdigest() == record['body_sha256'],
              'only intact captured COMPUTE/SCALAR stage bodies are supported')
        declarations = '\n'.join(f'    Scalar scalar{i} = saved[{i}];' for i, node in enumerate(nodes)
            if record['shapes'][node.id] is None and node.op not in ('input', 'const'))
        if stage['kind'] == 'COMPUTE':
            assignment = plan['assignment'].get(stage['output_payload'])
            _need(assignment is not None and assignment['slot_id'] in slot_indexes
                  and type(stage['shape']) in (tuple, list) and len(stage['shape']) == 2,
                  'original stage output slot/shape required')
            count = stage['shape'][0]*stage['shape'][1]
            slot_index = slot_indexes[assignment['slot_id']]
            _need(type(count) is int and 0 <= count*8 <= capacities[slot_index], 'stage output exceeds its actual slot')
            output = f'slots[{slot_index}].get()'
            barrier = f'    asm volatile("" : : "g"({output}), "g"(numeric_guard) : "memory");'
            checksum = f'    for (uint64_t i = 0; i < {count}ULL; ++i) sink ^= std::bit_cast<uint64_t>({output}[i]);'
        else:
            root = stage['root_expr']
            _need(root in order and record['shapes'][root] is None and nodes[order[root]].op not in ('input', 'const'),
                  'scalar stage requires its original emitted scalar result variable')
            output = f'scalar{order[root]}'
            barrier = f'    asm volatile("" : : "g"({output}.value), "g"({output}.integer), "g"(numeric_guard) : "memory");'
            checksum = f'    sink ^= std::bit_cast<uint64_t>({output}.value) ^ uint64_t({output}.integer);'
        # The original captured body is inserted verbatim; only surrounding
        # port loads and liveness/clock harness code are new.
        function = '''__attribute__((noinline, noipa))
int32_t component_@INDEX@(const scar_array_v1 *arrays, const scar_scalar_v1 *scalars,
    const Scalar *saved, std::vector<std::unique_ptr<double[]>> &slots,
    uint64_t *elapsed, uint64_t *checksum) {
    uint32_t numeric_guard = 0;
    uint64_t start = 0, end = 0;
    if (!clock_ns(start)) return 3;
    asm volatile("" : : : "memory");
@DECLARATIONS@
@BODY@
@BARRIER@
    if (!clock_ns(end) || end < start) return 3;
    *elapsed = end - start;
    if (numeric_guard || std::fetestexcept(floating_flags)) return 3;
    uint64_t sink = 0;
@CHECKSUM@
    *checksum = sink;
    return 0;
}
'''
        for name, value in dict(INDEX=str(component_index), DECLARATIONS=declarations,
                BODY=record['body'], BARRIER=barrier, CHECKSUM=checksum).items():
            function = function.replace('@'+name+'@', value)
        functions.append(function)
        receipts.append(dict(component_index=component_index, stage_index=index, stage_kind=stage['kind'],
            body_sha256=record['body_sha256'], emitted_function_sha256=sha256(function.encode()).hexdigest(),
            stage_ref=f'#/stages/{index}', output_port=output,
            entry_available=record['entry_available'], entry_scalar_ready=record['entry_scalar_ready'],
            body_copied_verbatim=True))
    template_path = _HERE/'stateful_stage_probe_v1.cpp.in'
    replacements = dict(CPU_ID=str(cpus[0]), CPU_SLOTS=str(cpus[0]+1), AFFINITY_CLASS=affinity,
        FLOATING_CLASS=floating, FINITE_HELPER=finite, SCALAR_HELPERS=scalar_helpers,
        ARRAY_COUNT=str(len(arrays)), SCALAR_COUNT=str(len(scalars)), NODE_COUNT=str(len(nodes)),
        SLOT_COUNT=str(len(capacities)), ARRAY_SHAPES=', '.join('{{'+str(r)+','+str(c)+'}}' for r,c in shapes),
        SLOT_CAPACITIES=', '.join(str(n)+'ULL' for n in capacities), PAYLOAD_BYTES=str(payload),
        MAX_PAYLOAD_BYTES=str(max_payload_bytes), COMPONENT_FUNCTIONS='\n'.join(functions),
        COMPONENT_COUNT=str(len(functions)), COMPONENT_POINTERS=', '.join('&component_'+str(i) for i in range(len(functions))))
    source = template_path.read_text()
    for name, value in replacements.items():
        source = source.replace('@'+name+'@', value)
    _need('@' not in source, 'unresolved original-stage probe token')
    provenance = dict(schema='scar.stateful-stage-dag-probe-source', schema_version=1,
        source_pins=[_pin(path) for path in (Path(__file__).resolve(), template_path, backend_path, loop_path,
            _HERE/'stateful.py', _HERE/'stateful_stage_capture.py', _HERE/'stateful_abi_v1.h', _HERE/'scar_abi_v1.h')],
        helper_fragments=fragments, source_sha256=sha256(source.encode()).hexdigest(), stages=receipts,
        capture_full_source_sha256=capture['full_source_sha256'], capture_provenance=capture['provenance'],
        nativecomponent_topology_specialized=True, capture_and_signature_admission_required=True,
        clock='CLOCK_MONOTONIC', clock_unit='INTEGER_NANOSECONDS',
        compiler='GNU_GXX_ORIGINAL_SERIAL_CPP20_OPTIONS_NOINLINE_NOIPA_REQUIRED', cpus=list(cpus),
        full_original_payload_bytes=payload, max_payload_bytes=max_payload_bytes,
        original_slot_aliases_preserved=True, independent_external_inputs=True,
        canonical_array_value=1.25, canonical_scalar_value=0.75, canonical_scalar_integer_tag=False,
        entry_ready_scalar_scenario='INDEPENDENT_CANONICAL_PORTS_NOT_EXECUTED_PREDECESSOR_SCALAR_DAG',
        initialization='ONCE_BEFORE_BIND_AND_CLOCKS_REPEATED_COMPONENT_BUFFERS', numa_residency_proven=False,
        retained_residuals=['SAVED_SCALAR_LOCAL_LOADS_AFTER_START', 'CLOCK_CALLS_AND_COMPILER_FENCES'],
        output_checksum_after_end=True, setup_allocation_affinity_fenv_inside_timing=False,
        instrumentation_subtracted=False, provider_called=False, original_entry_executed=False,
        target_program_executed=False, native_compiled=False, performance_measured=False,
        target_fragment_compilation_performed_by='CALLER_IF_COMPILED_NOT_BY_RENDERER')
    return source, provenance


__all__ = ['render_stage_dag_probe']
