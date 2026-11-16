"""Source-faithful conditional finite-input guard reuse for one native loop.

This renderer is not semantic, caller-loop or deployment authority. Its
producer must independently replay the original source/execution certificate
and the complete finite-content property plan before using these bytes.
"""
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


PROPERTY_SCHEMA = 'scar.workspace-finite-content-property-plan'
EXECUTION_SCHEMA = 'scar.workspace-guard-reuse-execution-plan'
MAPPING_SCHEMA = 'scar.workspace-guard-reuse-source-mapping'
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


def _source_plan(original_source, original_metadata, property_plan):
    """Check the conditional plan's exact source/scan interface, not its proof."""
    _need(type(original_source) is bytes and type(original_metadata) is dict
          and type(property_plan) is dict
          and property_plan.get('schema') == PROPERTY_SCHEMA
          and type(property_plan.get('schema_version')) is int
          and property_plan['schema_version'] == 1,
          'a typed original workspace source and finite-content property plan are required')
    blueprint = property_plan['native_blueprint']
    _need(set(blueprint) == {'manifest', 'cpp_source', 'workspace_plan_sha256',
                           'execution_plan_sha256', 'base_ids', 'source_mapping'},
          'exact original native source blueprint required')
    _need(_same(read_artifact(blueprint['manifest']), original_metadata)
          and original_metadata.get('schema') == 'scar.native-stateful-workspace-repeat'
          and original_metadata.get('stateful_abi_version') == 1
          and original_metadata.get('repeat_abi_version') == 1
          and _hash(original_source) == original_metadata['sha256']['source']
          and read_artifact(blueprint['cpp_source'], json_document=False) == original_source,
          'finite-content property plan does not bind these exact original native bytes')
    directory = Path(blueprint['manifest']['uri']).parent
    _need(artifact_reference(directory/original_metadata['files']['source'], blueprint['cpp_source']['schema'])
          == blueprint['cpp_source'], 'property source must be the original manifest C++ member')
    for key in ('workspace_plan_sha256', 'execution_plan_sha256', 'base_ids', 'source_mapping'):
        _need(_same(blueprint[key], original_metadata[key]), 'property native identity differs: '+key)
    _need(type(property_plan.get('trip_count')) is int
          and property_plan['trip_count'] == original_metadata['trip_count'],
          'property induction must cover the exact original repetition count')
    from scar.compiler.stateful_model import read_stateful
    region = read_stateful(directory/original_metadata['files']['region'])
    step_plan = json.loads((directory/original_metadata['files']['step_plan']).read_bytes())
    workspace_plan = json.loads((directory/original_metadata['files']['workspace_plan']).read_bytes())
    _need(not region.providers and step_plan['cpu_threads'] == original_metadata['cpu_threads']
          and semantic_sha256(workspace_plan) == original_metadata['workspace_plan_sha256'],
          'opaque providers cannot preserve the finite-input induction')
    # This status is a required input interface, never proof by itself. The
    # producer replays the full property plan with its execution certificate;
    # this source-only API has neither that context nor transformation authority.
    _need(property_plan.get('status') == 'PROVEN_CONDITIONAL',
          'a caller-qualified conditional finite-content property plan is required')
    arrays = [row for row in region.inputs if row.kind == 'array']
    scans = property_plan['input_scans']
    _need(type(scans) is list and len(scans) == len(arrays) and bool(scans),
          'every original ABI array requires its initial finite scan')
    for ordinal, (row, array) in enumerate(zip(scans, arrays)):
        shape = step_plan['input_shapes'][array.id]
        _need(row['input_id'] == array.id and type(row['array_index']) is int and row['array_index'] == ordinal
              and row['shape'] == list(shape) and len(shape) == 2
              and type(row['element_count']) is int and row['element_count'] == shape[0]*shape[1]
              and row['nbytes'] == row['element_count']*8,
              'property finite-scan coverage differs from the exact original ABI shapes')
    return scans


def render_guard_reuse(original_source, original_metadata, property_plan):
    """Return new C++ bytes and a reversible, explicitly conditional source mapping.

    Only the workspace clone's array finite guards become conditional. Its
    scalar, descriptor, FP, affinity, arithmetic, restoration and commit code
    remain verbatim. The unchanged original ABI step/table stays available.
    """
    scans = _source_plan(original_source, original_metadata, property_plan)
    text = original_source.decode('utf-8')
    _need(text.count(_CLONE) == text.count(_GETTER) == text.count(_CALL) == 1,
          'one original workspace clone, getter and loop call are required')
    start, getter_start = text.index(_CLONE), text.index(_GETTER)
    original_prefix = text[:start]
    clone_and_namespace = text[start:getter_start]
    _need(clone_and_namespace.endswith('\n}\n\n'), 'original workspace clone namespace boundary differs')
    clone = clone_and_namespace[:-len('\n}\n\n')]
    old_mapping = original_metadata['source_mapping']
    _need(_hash(clone.encode()) == old_mapping['cloned_run_sha256'],
          'original workspace clone differs from its exact source mapping')
    signature_end = clone.index(') {')+3
    signature = clone[len(_ATTRIBUTES):signature_end]
    _need(_hash(signature.encode()) == old_mapping['clone_signature_sha256']
          and signature.endswith('> &slots) {'), 'original private-workspace signature differs')
    new_signature = signature[:-3]+', bool need_input_scan) {'
    finite = '\n'.join(f'    for (int64_t i = 0; i < {row["element_count"]}; ++i) '
        f'if (!finite(static_cast<const double *>(arrays[{row["array_index"]}].data)[i])) return 3;'
        for row in scans)
    guard_prefix = '    uint32_t numeric_guard = 0, placement_guard = 0;\n'
    _need(clone.count(guard_prefix+finite+'\n') == 1,
          'all original input finite guards must occur in their exact source position')
    conditional = '    if (need_input_scan) {\n'+'\n'.join('    '+line for line in finite.splitlines())+'\n    }'
    new_clone = clone.replace(signature, new_signature, 1).replace(guard_prefix+finite+'\n',
        guard_prefix+conditional+'\n', 1)
    restored_clone = new_clone.replace(new_signature, signature, 1).replace(
        guard_prefix+conditional+'\n', guard_prefix+finite+'\n', 1)
    _need(restored_clone == clone and new_clone.startswith(_ATTRIBUTES),
          'array-scan conditioning must preserve the entire original mathematical step')
    old_execution = original_metadata['execution_plan_sha256']
    _need(re.fullmatch('[0-9a-f]{64}', old_execution) is not None, 'original execution identity required')
    execution_plan = dict(schema=EXECUTION_SCHEMA, schema_version=1,
        parent_execution_plan_sha256=old_execution, property_plan_sha256=semantic_sha256(property_plan))
    execution_sha = semantic_sha256(execution_plan)
    old_getter = _GETTER+' {\n    return "'+old_execution+'";\n}'
    new_getter = _GETTER+' {\n    return "'+execution_sha+'";\n}'
    _need(text.count(old_getter) == 1, 'unique original workspace execution getter required')
    loop = '''    for (uint64_t step = 0; step < repetitions; ++step) {
'''+_CALL+'''
        if (status) return status;
        // The failed iteration never commits; successful prefixes stay visible.
        *completed = step + 1;
    }'''
    _need(text.count(loop) == 1, 'original success-prefix and early-failure loop must be unchanged')
    new_call = _CALL.replace('provider_count, slots);', 'provider_count, slots, step == 0);')
    source = text.replace(clone, new_clone, 1).replace(_CALL, new_call, 1).replace(old_getter, new_getter, 1)
    inverse = source.replace(new_getter, old_getter, 1).replace(new_call, _CALL, 1).replace(new_clone, clone, 1)
    _need(inverse.encode() == original_source and source[:start] == original_prefix,
          'reversing only the guard, call argument and controller identity must restore every original byte')
    controller = text[text.index('extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_repeat_stateful_v1('):]
    zero_prefix = controller[:controller.index('    if (repetitions != ')]
    _need('    if (repetitions == 0) return 0;\n' in zero_prefix
          and 'arrays[' not in zero_prefix and 'scalars[' not in zero_prefix and 'new ' not in zero_prefix,
          'zero-trip must return before unused input fields or workspace allocation')
    pins = {name: _hash((_HERE/name).read_bytes()) for name in
        ('workspace_guard_reuse.py', 'stateful_workspace.py', 'stateful_workspace_v1.cpp.in',
         'stateful.py', 'stateful_v1.cpp.in')}
    mapping = dict(schema=MAPPING_SCHEMA, schema_version=1,
        original_source_sha256=_hash(original_source), source_sha256=_hash(source.encode()),
        property_plan_sha256=execution_plan['property_plan_sha256'], execution_plan=execution_plan,
        execution_plan_sha256=execution_sha, parent_execution_plan_sha256=old_execution,
        original_clone_sha256=_hash(clone.encode()), cloned_run_sha256=_hash(new_clone.encode()),
        input_finite_guards_sha256=_hash(finite.encode()), conditional_finite_guards_sha256=_hash(conditional.encode()),
        input_scans=scans, source_pins=pins, original_stateful_prefix_sha256=_hash(original_prefix.encode()),
        zero_trip_prefix_sha256=_hash(zero_prefix.encode()),
        source_inverse_byte_equal=True, original_run_table_and_stateful_getter_unchanged=True,
        step_compilation_boundary_noinline_noipa_retained=True,
        descriptor_scalar_fp_affinity_math_restore_commit_retained_verbatim=True,
        first_iteration_input_scan=True, later_input_scans='SKIPPED_ONLY_AFTER_EACH_PRIOR_STEP_SUCCESS',
        failed_iteration_exits_before_next_step=True, completed_cursor_and_suffix_policy_unchanged=True,
        zero_trip_no_input_fields_or_allocations=True, numerical_backend_added=False,
        source_authority='CONDITIONAL_PROPERTY_PLAN_AND_ORIGINAL_PRODUCER_REPLAY_REQUIRED',
        semantic_property_admission_required_by_caller=True, complete_property_plan_replayed_here=False,
        trial_only=True, default_backend='NO_OP', deployment_authorized=False,
        target_program_executed=False, native_compiled=False, performance_measured=False)
    return source.encode(), mapping


__all__ = ['render_guard_reuse', 'PROPERTY_SCHEMA', 'EXECUTION_SCHEMA', 'MAPPING_SCHEMA']
