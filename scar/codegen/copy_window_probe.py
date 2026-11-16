"""Render paired canonical probes of exact original and fused copy windows.

The caller admits capture/certificate/source before rendering. This module
checks the emitted ports and control skeleton; it never compiles or samples.
"""
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.codegen.stateful_stage_probe import _fragment, _need, _pin

_HERE = Path(__file__).resolve().parent


def _hash(raw):
    return sha256(raw).hexdigest()


def _digest(value):
    return _hash(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def render_copy_window_probe(plan, cpus, original_bodies, fused_body, *, max_payload_bytes=256*2**20):
    """Return C++20 source/provenance for two fixed canonical components only."""
    _need(type(plan) is dict and plan.get('cpu_threads') == 1 and type(plan['cpu_threads']) is int
          and type(cpus) in (tuple, list) and len(cpus) == 1
          and type(cpus[0]) is int and 0 <= cpus[0] < 2**20,
          'one explicit original serial CPU required')
    _need(type(max_payload_bytes) is int and 0 <= max_payload_bytes <= 2**63-1,
          'bounded canonical payload budget required')
    array_order = plan.get('copy_window_array_order')
    input_shapes = plan.get('input_shapes')
    _need(type(input_shapes) is dict and type(array_order) in (tuple, list)
          and 0 < len(array_order) <= 4096 and all(type(key) is str for key in array_order)
          and len(set(array_order)) == len(array_order) and set(array_order) == set(input_shapes),
          'explicit original region array order must cover all plan input shapes')
    shapes = [input_shapes[key] for key in array_order]
    _need(all(type(shape) in (tuple, list) and len(shape) == 2
              and all(type(axis) is int and 0 <= axis <= (2**63-1)//8 for axis in shape)
              for shape in shapes), 'representable original rank-two array dimensions required')
    slots = plan.get('workspace_slots')
    _need(type(slots) is list and len(slots) <= 4096
          and all(type(row) is dict and type(row.get('id')) is str
                  and type(row.get('capacity_bytes')) is int and 0 <= row['capacity_bytes'] <= 2**63-1
                  and row['capacity_bytes'] % 8 == 0 for row in slots)
          and len({row['id'] for row in slots}) == len(slots), 'original distinct physical-slot capacities required')
    capacities = [row['capacity_bytes'] for row in slots]
    slot_indexes = {row['id']: index for index, row in enumerate(slots)}
    external = sum(rows*cols*8 for rows, cols in shapes)
    workspace = sum(capacities); payload = external+workspace
    _need(all(type(plan.get(key)) is int and 0 <= plan[key] <= 2**63-1 for key in
              ('external_array_bytes', 'workspace_payload_bytes', 'total_region_array_payload_bytes', 'memory_budget_bytes'))
          and plan['external_array_bytes'] == external and plan['workspace_payload_bytes'] == workspace
          and plan['total_region_array_payload_bytes'] == payload
          and payload <= plan['memory_budget_bytes'] and payload <= max_payload_bytes,
          'full original input/physical-slot payload exceeds or differs from the original budget')
    _need(type(original_bodies) in (tuple, list) and 2 <= len(original_bodies) <= 4096
          and all(type(row) is dict and set(row) == {'task', 'body'}
                  and type(row['task']) is str and type(row['body']) is str for row in original_bodies)
          and len({row['task'] for row in original_bodies}) == len(original_bodies)
          and type(fused_body) is str, 'multiple original task/body records and one fused body required')
    stages = {row['id']: row for row in plan['stages']}
    _need(len(stages) == len(plan['stages']), 'unique original stage identities required')
    skeleton = None; inners = []; copies = []; receipts = []; target_slots = set(); indexes = []
    for row in original_bodies:
        _need(row['task'] in stages, 'captured copy task is absent from the original plan')
        stage = stages[row['task']]; lines = row['body'].split('\n')
        _need(stage['kind'] == 'SNAPSHOT_ROOT' and len(lines) == 5 and lines[-2:] == ['    }', '}'],
              'exact five-line original snapshot loop body required')
        r = re.fullmatch(r'for \(int64_t row = 0; row < ([0-9]+); \+\+row\) \{', lines[0])
        c = re.fullmatch(r'    for \(int64_t col = 0; col < ([0-9]+); \+\+col\) \{', lines[1])
        assignment = re.fullmatch(r'        slots\[([0-9]+)\]\.get\(\)\[row \* ([0-9]+) \+ col\] = '
            r'static_cast<const double \*>\(arrays\[([0-9]+)\]\.data\)\[row \* ([0-9]+) \+ col\];', lines[2])
        _need(r is not None and c is not None and assignment is not None,
              'only original concrete row/column loops and one raw load-store statement are accepted')
        rows, cols = int(r[1]), int(c[1]); target, target_stride, source, source_stride = map(int, assignment.groups())
        _need(list(stage['shape']) == [rows, cols] and source < len(array_order)
              and array_order[source] == stage['target'] and list(shapes[source]) == [rows, cols]
              and target_stride == source_stride == cols
              and target < len(slots) and target not in target_slots
              and plan['assignment'][stage['output_payload']]['slot_id'] == slots[target]['id']
              and rows*cols*8 <= capacities[target],
              'copy body indices/shape/stride must match original array order and private output assignment')
        candidate = (lines[:2], lines[-2:])
        _need(skeleton is None or candidate == skeleton, 'all original loops must have the identical exact domain/skeleton')
        skeleton = candidate; inners.append(lines[2]); target_slots.add(target); indexes.append(stage['index'])
        copies.append(dict(source_array=source, target_slot=target, elements=rows*cols))
        receipts.append(dict(task=row['task'], source_array=source, input_id=stage['target'],
            target_slot=target, slot_id=slots[target]['id'], shape=[rows, cols],
            body_sha256=_hash(row['body'].encode()), inner_sha256=_hash(lines[2].encode())))
    _need(indexes == list(range(indexes[0], indexes[0]+len(indexes))),
          'one contiguous original source window in original task order required')
    expected_fused = '\n'.join([*skeleton[0], *inners, *skeleton[1]])
    _need(fused_body == expected_fused, 'fused body must preserve every original inner statement and its order verbatim')
    loop_path = _HERE/'loop_v1.cpp.in'; loop = loop_path.read_text()
    affinity, affinity_ref = _fragment(loop_path, loop, 'class ThreadAffinity {',
        '\n#endif\n\nconstexpr int64_t modeled_rows')
    floating, floating_ref = _fragment(loop_path, loop, 'class FloatingScope {', '\nbool valid_array(')
    template_path = _HERE/'copy_window_probe_v1.cpp.in'
    body_indent = lambda text: '\n'.join('    '+line for line in text.split('\n'))
    # Expose the unchanged vector of private output pointers with one operand;
    # memory clobbers and the post-end reads preserve every written buffer.
    # This avoids GCC's finite asm-operand limit for a large source window.
    liveness = '"g"(slots.data())'
    replacements = dict(CPU_ID=str(cpus[0]), CPU_SLOTS=str(cpus[0]+1),
        AFFINITY_CLASS=affinity, FLOATING_CLASS=floating,
        ARRAY_COUNT=str(len(array_order)), SLOT_COUNT=str(len(slots)), COPY_COUNT=str(len(copies)),
        ARRAY_SHAPES=', '.join('{{'+str(rows)+','+str(cols)+'}}' for rows, cols in shapes),
        SLOT_CAPACITIES=', '.join(str(value)+'ULL' for value in capacities),
        COPY_PORTS=', '.join('{'+str(row['source_array'])+','+str(row['target_slot'])+','+str(row['elements'])+'ULL}' for row in copies),
        TARGET_SLOTS=', '.join(str(index) for index in sorted(target_slots)), TARGET_COUNT=str(len(target_slots)),
        PAYLOAD_BYTES=str(payload), MEMORY_BUDGET_BYTES=str(plan['memory_budget_bytes']),
        MAX_PAYLOAD_BYTES=str(max_payload_bytes),
        ORIGINAL_BODY=body_indent('\n'.join(row['body'] for row in original_bodies)),
        FUSED_BODY=body_indent(fused_body), LIVENESS_OPERANDS=liveness)
    source = template_path.read_text()
    for name, value in replacements.items():
        source = source.replace('@'+name+'@', value)
    _need('@' not in source, 'unresolved copy-window canonical probe token')
    original_plan = {key: value for key, value in plan.items() if key != 'copy_window_array_order'}
    provenance = dict(schema='scar.copy-window-probe-source', schema_version=1,
        source_pins=[_pin(path) for path in (Path(__file__), template_path, loop_path,
            _HERE/'stateful_stage_probe.py', _HERE/'stateful_abi_v1.h', _HERE/'scar_abi_v1.h')],
        helper_fragments=dict(affinity=affinity_ref, floating=floating_ref),
        generated_source_sha256=_hash(source.encode()), base_plan_sha256=_digest(original_plan),
        harness_array_order=list(array_order), harness_array_order_authority='CALLER_ADMITTED_ORIGINAL_REGION_INPUT_ORDER',
        original_bodies=receipts, fused_body_sha256=_hash(fused_body.encode()),
        original_window_sha256=_hash('\n'.join(row['body'] for row in original_bodies).encode()),
        cpus=list(cpus), original_array_shapes=[list(shape) for shape in shapes],
        original_slot_capacities=capacities, target_slots=sorted(target_slots),
        original_full_payload_bytes=payload, max_payload_bytes=max_payload_bytes,
        memory_budget_bytes=plan['memory_budget_bytes'],
        clock=dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
        abi=dict(symbol='scar_copy_window_probe_v1', result='int32',
            arguments=['uint64 repetitions', 'uint64 warmup', 'uint64* elapsed_pairs',
                       'uint64* checksums', 'int32* orders'],
            elapsed_pairs_elements='2*repetitions', checksums_elements='2*repetitions', orders_elements='repetitions',
            column_order=['ORIGINAL_COMPLETE_WINDOW', 'FUSED_WINDOW'],
            order_codes={'0': 'ORIGINAL_THEN_FUSED', '1': 'FUSED_THEN_ORIGINAL'},
            statuses={'0': 'SUCCESS', '1': 'ARGUMENTS', '2': 'BITWISE_COPY_MISMATCH',
                      '3': 'CLOCK_OR_FLOATING_ENVIRONMENT', '4': 'CPU_AFFINITY', '5': 'ALLOCATION', '7': 'PAYLOAD_BUDGET'}),
        pair_order='GLOBAL_PAIR_INDEX_MODULO_2_INCLUDING_WARMUP',
        canonical_arrays='EACH_ARRAY_FILLED_WITH_1.25_PLUS_ORIGINAL_ARRAY_ORDINAL_TIMES_0.125',
        slot_reset='EVERY_TARGET_SLOT_FULL_CAPACITY_ZEROED_BEFORE_EACH_VARIANT_OUTSIDE_CLOCK',
        identical_input_and_slot_objects_for_both_variants=True, entire_original_window_measured_as_one_interval=True,
        original_unfused_component_times_summed=False, output_bitwise_checked_after_end=True,
        full_output_checksum_after_end=True, allocation_initialization_affinity_restore_inside_timing=False,
        retained_residuals=['CLOCK_READS_AND_COMPILER_MEMORY_FENCES',
            'ALTERNATING_PAIR_ORDER_AND_ZERO_RESET_CANONICAL_CACHE_REGIME'],
        instrumentation_subtracted=False, numa_residency_proven=False,
        original_inner_statements_preserved=True, mathematical_expression_rewritten=False,
        canonical_variant_count=2, black_box_target_variant_search=False,
        provider_called=False, target_program_executed=False, native_compiled=False, performance_measured=False,
        target_fragment_compilation_performed_by='CALLER_IF_COMPILED_NOT_RENDERER',
        source_capture_and_certificate_admission_required=True, deployment_authorized=False)
    return source, provenance


__all__ = ['render_copy_window_probe']
