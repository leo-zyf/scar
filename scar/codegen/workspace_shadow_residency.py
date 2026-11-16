"""Render conditional retention of committed private-shadow ROIs across steps.

The upstream producer owns complete source/ROI/provider/property admission.
This module checks exact original stage bytes and transforms only workspace
snapshot copies. It neither authorizes caller replacement nor builds a kernel.
"""
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


PLAN_SCHEMA = 'scar.workspace-shadow-residency-plan'
EXECUTION_SCHEMA = 'scar.workspace-shadow-residency-execution-plan'
MAPPING_SCHEMA = 'scar.workspace-shadow-residency-source-mapping'
_HERE = Path(__file__).resolve().parent
_ATTRIBUTES = '__attribute__((noinline, noipa))\n'
_CLONE = _ATTRIBUTES+'int32_t run_with_workspace('
_GETTER = 'extern "C" SCAR_EXPORT_V1 const char *scar_workspace_plan_sha256_v1()'
_CALL = '''        const int32_t status = run_with_workspace(arrays, array_count, scalars,
            scalar_count, providers, provider_count, slots);'''


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(raw):
    return sha256(raw).hexdigest()


def _same(left, right):
    return semantic_sha256(left) == semantic_sha256(right)


def _complement(shape, roi):
    _need(type(roi) in (list, tuple) and len(roi) == 2
          and all(type(axis) in (list, tuple) and len(axis) == 2
                  and all(type(bound) is int for bound in axis) for axis in roi),
          'one exact normalized committed rectangle required')
    rows, cols = shape
    (r0, r1), (c0, c1) = roi
    _need(0 <= r0 <= r1 <= rows and 0 <= c0 <= c1 <= cols,
          'committed rectangle is outside its original root')
    candidates = (((0, r0), (0, cols)), ((r1, rows), (0, cols)),
                  ((r0, r1), (0, c0)), ((r0, r1), (c1, cols)))
    return [[list(r), list(c)] for r, c in candidates if r[0] < r[1] and c[0] < c[1]]


def _source_plan(original_source, metadata, plan):
    """Check source interfaces; conditional status itself is not semantic proof."""
    _need(type(original_source) is bytes and type(metadata) is dict and type(plan) is dict
          and plan.get('schema') == PLAN_SCHEMA and type(plan.get('schema_version')) is int
          and plan['schema_version'] == 1 and plan.get('status') == 'PROVEN_CONDITIONAL',
          'an upstream-qualified conditional shadow-residency plan is required')
    blueprint = plan['native_blueprint']
    _need(set(blueprint) == {'manifest', 'cpp_source', 'workspace_plan_sha256',
                           'execution_plan_sha256', 'base_ids', 'source_mapping'}
          and metadata.get('schema') == 'scar.native-stateful-workspace-repeat'
          and _same(read_artifact(blueprint['manifest']), metadata),
          'exact original workspace native blueprint required')
    directory = Path(blueprint['manifest']['uri']).parent
    _need(artifact_reference(directory/metadata['files']['source'], blueprint['cpp_source']['schema'])
          == blueprint['cpp_source']
          and read_artifact(blueprint['cpp_source'], json_document=False) == original_source
          and _hash(original_source) == metadata['sha256']['source'],
          'original C++ source does not match the residency blueprint')
    for key in ('workspace_plan_sha256', 'execution_plan_sha256', 'base_ids', 'source_mapping'):
        _need(_same(blueprint[key], metadata[key]), 'residency blueprint differs: '+key)
    _need(type(plan.get('trip_count')) is int and plan['trip_count'] == metadata['trip_count']
          and metadata['cpu_threads'] == 1 and len(metadata['cpu_ids']) == 1,
          'one exact fixed-count serial workspace source required')
    for key in ('region', 'step_plan', 'workspace_plan'):
        path = directory/metadata['files'][key]
        _need(_hash(path.read_bytes()) == metadata['sha256'][key], 'original source-plan member changed: '+key)
    from scar.compiler.stateful_model import read_stateful, stateful_digest
    from scar.compiler.model import CpuPlacement
    from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
    region = read_stateful(directory/metadata['files']['region'])
    step = json.loads((directory/metadata['files']['step_plan']).read_bytes())
    workspace = json.loads((directory/metadata['files']['workspace_plan']).read_bytes())
    _need(stateful_digest(region) == metadata['base_ids']['region_sha256']
          and semantic_sha256(step) == metadata['base_ids']['plan_sha256']
          and semantic_sha256(workspace) == metadata['workspace_plan_sha256'],
          'residency source region, memory layout or step plan differs')
    base = read_artifact(metadata['base_native_reference'])
    _need(metadata['base_ids'] == {key: base[key] for key in metadata['base_ids']},
          'original native base identities differ')
    binding = dict(step['resource_binding'])
    binding['cpu_ids'] = tuple(binding['cpu_ids']); binding['evidence'] = tuple(binding.get('evidence', ()))
    capture = capture_stateful_stage_sources(region, step, CpuPlacement(**binding),
        dict(region_sha256=metadata['base_ids']['region_sha256'], plan_sha256=metadata['base_ids']['plan_sha256'],
             provider_runtime=base['provider_runtime']))
    _need(capture['full_source_sha256'] == metadata['source_mapping']['original_full_source_sha256'],
          'captured mathematical host differs from the unmodified original base')
    return region, step, capture


def render_shadow_residency(original_source, original_metadata, qualified_plan):
    """Condition only exact snapshots; preserve all other per-step source bytes."""
    region, step, capture = _source_plan(original_source, original_metadata, qualified_plan)
    text = original_source.decode('utf-8')
    _need(text.count(_CLONE) == text.count(_GETTER) == text.count(_CALL) == 1,
          'one original workspace clone, controller getter and step call are required')
    start, getter_start = text.index(_CLONE), text.index(_GETTER)
    segment = text[start:getter_start]
    _need(segment.endswith('\n}\n\n'), 'original clone namespace boundary differs')
    clone = segment[:-len('\n}\n\n')]
    _need(_hash(clone.encode()) == original_metadata['source_mapping']['cloned_run_sha256'],
          'original private-workspace clone source differs')
    signature_end = clone.index(') {')+3
    signature = clone[len(_ATTRIBUTES):signature_end]
    _need(_hash(signature.encode()) == original_metadata['source_mapping']['clone_signature_sha256']
          and signature.endswith('> &slots) {'), 'original workspace clone signature differs')
    new_signature = signature[:-3]+', bool first_iteration) {'
    arrays = [row.id for row in region.inputs if row.kind == 'array']
    stages = {row['id']: row for row in step['stages'] if row['kind'] == 'SNAPSHOT_ROOT'}
    records = qualified_plan['snapshots']
    _need(type(records) is list and len(records) == len(stages)
          and len({row['snapshot_stage_id'] for row in records}) == len(records)
          and {row['snapshot_stage_id'] for row in records} == set(stages),
          'residency plan must cover every original snapshot exactly once')
    captured = {row['index']: row for row in capture['stages']}
    slots = {row['id']: i for i, row in enumerate(step['workspace_slots'])}
    new_clone = clone.replace(signature, new_signature, 1)
    replacements, proofs = [], []
    for record in records:
        stage = stages[record['snapshot_stage_id']]
        target, shape, payload = stage['target'], list(stage['shape']), stage['output_payload']
        slot = step['assignment'][payload]['slot_id']; index = slots[slot]
        array_index = arrays.index(target)
        _need(record['input_id'] == target and record['shape'] == shape
              and type(record['array_index']) is int and record['array_index'] == array_index
              and record['output_payload'] == payload and record['slot_id'] == slot
              and type(record['slot_index']) is int and record['slot_index'] == index,
              'snapshot residency ports differ from the original ABI or private layout')
        _need(record['mode'] in ('KEEP_FULL_SNAPSHOT', 'REFRESH_COMPLEMENT', 'SKIP_AFTER_FIRST'),
              'unsupported snapshot residency policy')
        if record['mode'] == 'KEEP_FULL_SNAPSHOT':
            continue
        rows, cols = shape
        _need(type(rows) is int and type(cols) is int and rows >= 0 and cols >= 0
              and list(step['input_shapes'][target]) == shape
              and rows*cols*8 <= step['workspace_slots'][index]['capacity_bytes'],
              'snapshot residency root dimensions or slot capacity differ')
        commits = [row for row in step['commit_rows'] if row['target'] == target]
        _need(commits and record['commit_stage_ids'] == [row['stage_id'] for row in commits]
              and all(_same(row['normalized_bounds'], record['committed_roi']) for row in commits),
              'reused private content requires one exact committed rectangle, including all original commits')
        rectangles = _complement(shape, record['committed_roi'])
        _need(record['complement_rectangles'] == rectangles
              and (record['mode'] == 'SKIP_AFTER_FIRST') == (not rectangles),
              'residency complement must be the complete disjoint top/bottom/left/right rectangle set')
        body = captured[stage['index']]['body']; lines = body.split('\n')
        inner = (f'        slots[{index}].get()[row * {cols} + col] = '
                 f'static_cast<const double *>(arrays[{array_index}].data)[row * {cols} + col];')
        expected = [f'for (int64_t row = 0; row < {rows}; ++row) {{',
                    f'    for (int64_t col = 0; col < {cols}; ++col) {{', inner, '    }', '}']
        _need(lines == expected, 'snapshot body is not the exact original five-line raw-copy stage')
        before = '\n'.join(' '*12+line for line in lines)
        _need(clone.count(before) == 1 and new_clone.count(before) == 1,
              'snapshot must have one unique exact span inside the workspace clone')
        later = []
        for (r0, r1), (c0, c1) in rectangles:
            later.extend([f'for (int64_t row = {r0}; row < {r1}; ++row) {{',
                f'    for (int64_t col = {c0}; col < {c1}; ++col) {{', inner, '    }', '}'])
        after = ' '*12+'if (first_iteration) {\n'+before+'\n'+' '*12+'}'
        if later:
            after += ' else {\n'+'\n'.join(' '*12+line for line in later)+'\n'+' '*12+'}'
        new_clone = new_clone.replace(before, after, 1)
        replacements.append((before, after))
        proofs.append(dict(snapshot_stage_id=stage['id'], stage_index=stage['index'], input_id=target,
            array_index=array_index, slot_id=slot, slot_index=index, output_payload=payload,
            shape=shape, committed_roi=record['committed_roi'], complement_rectangles=rectangles,
            source_char_start=start+clone.index(before), source_char_end_exclusive=start+clone.index(before)+len(before),
            original_body_sha256=_hash(body.encode()), original_span_sha256=_hash(before.encode()),
            replacement_span_sha256=_hash(after.encode()), original_inner_sha256=_hash(inner.encode()),
            first_iteration_full_copy_verbatim=True, later_inner_statement_verbatim=True,
            later_iteration_elements=sum((b-a)*(d-c) for (a,b),(c,d) in rectangles),
            full_root_committed_later_no_copy=not rectangles))
    _need(proofs, 'at least one qualified snapshot must actually change')
    restored = new_clone
    for before, after in reversed(replacements):
        _need(restored.count(after) == 1, 'residency replacement source must be uniquely reversible')
        restored = restored.replace(after, before, 1)
    _need(restored.replace(new_signature, signature, 1) == clone,
          'residency inverse must preserve all original checks, arithmetic, providers, restoration and commits')
    parent_execution = original_metadata['execution_plan_sha256']
    _need(re.fullmatch('[0-9a-f]{64}', parent_execution) is not None, 'original controller identity required')
    execution_plan = dict(schema=EXECUTION_SCHEMA, schema_version=1,
        parent_execution_plan_sha256=parent_execution, residency_plan_sha256=semantic_sha256(qualified_plan))
    execution_sha = semantic_sha256(execution_plan)
    old_getter = _GETTER+' {\n    return "'+parent_execution+'";\n}'
    new_getter = _GETTER+' {\n    return "'+execution_sha+'";\n}'
    loop = '''    for (uint64_t step = 0; step < repetitions; ++step) {
'''+_CALL+'''
        if (status) return status;
        // The failed iteration never commits; successful prefixes stay visible.
        *completed = step + 1;
    }'''
    _need(text.count(loop) == text.count(old_getter) == 1, 'original transaction cursor and controller differ')
    new_call = _CALL.replace('provider_count, slots);', 'provider_count, slots, step == 0);')
    source = text.replace(clone, new_clone, 1).replace(_CALL, new_call, 1).replace(old_getter, new_getter, 1)
    inverse = source.replace(new_getter, old_getter, 1).replace(new_call, _CALL, 1).replace(new_clone, clone, 1)
    _need(inverse.encode() == original_source and source[:start] == text[:start],
          'residency source inverse must restore every original byte and retain the original run/table/getter')
    controller = text[text.index('extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_repeat_stateful_v1('):]
    zero_prefix = controller[:controller.index('    if (repetitions != ')]
    _need('    if (repetitions == 0) return 0;\n' in zero_prefix
          and 'arrays[' not in zero_prefix and 'scalars[' not in zero_prefix and 'new ' not in zero_prefix,
          'zero-trip must precede unused input fields and allocation')
    mapping = dict(schema=MAPPING_SCHEMA, schema_version=1,
        original_source_sha256=_hash(original_source), source_sha256=_hash(source.encode()),
        residency_plan_sha256=execution_plan['residency_plan_sha256'], execution_plan=execution_plan,
        execution_plan_sha256=execution_sha, parent_execution_plan_sha256=parent_execution,
        original_clone_sha256=_hash(clone.encode()), cloned_run_sha256=_hash(new_clone.encode()),
        original_prefix_sha256=_hash(text[:start].encode()), zero_trip_prefix_sha256=_hash(zero_prefix.encode()),
        snapshots=proofs, capture_provenance=capture['provenance'],
        source_pins={name:_hash((_HERE/name).read_bytes()) for name in
            ('workspace_shadow_residency.py', 'stateful_stage_capture.py', 'stateful.py', 'stateful_v1.cpp.in',
             'stateful_workspace.py', 'stateful_workspace_v1.cpp.in')},
        source_inverse_byte_equal=True, original_run_table_and_stateful_getter_unchanged=True,
        per_step_finite_scalar_affinity_fp_math_provider_restore_commit_retained_verbatim=True,
        noinline_noipa_retained=True, allocations_and_layout_unchanged=True,
        success_prefix_cursor_and_failed_suffix_policy_unchanged=True, zero_trip_no_input_fields_or_allocations=True,
        later_copy_domain='DISJOINT_COMPLEMENT_RECTANGLES_NOT_FULL_DOMAIN_WITH_PER_ELEMENT_IF',
        semantic_residency_admission_required_by_caller=True, complete_residency_plan_replayed_here=False,
        trial_only=True, default_backend='NO_OP', deployment_authorized=False,
        numerical_backend_added=False, target_program_executed=False, native_compiled=False, performance_measured=False)
    return source.encode(), mapping


__all__ = ['render_shadow_residency', 'PLAN_SCHEMA', 'EXECUTION_SCHEMA', 'MAPPING_SCHEMA']
