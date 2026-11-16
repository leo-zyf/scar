"""Render a serial canonical boundary observer from the current backend text.

No compiler, target program, sampler or native library is invoked here. Cuts
retain their instrumentation work and do not measure the skipped target body.
"""
import ast
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.runtime.stateful_contract import canonical_sha256

_HERE = Path(__file__).resolve().parent
_REPOSITORY = _HERE.parent.parent
_CUT_NAMES = ('entry', 'metadataready', 'captureready', 'bindingready', 'workerchecked', 'restoreready')


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _pin(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(_REPOSITORY)), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _fragment(path, text, start_marker, end_marker):
    _need(text.count(start_marker) == 1 and text.count(end_marker) == 1,
          'backend fragment markers must be unique: ' + path.name)
    start = text.index(start_marker)
    end = text.index(end_marker)
    _need(end > start, 'backend fragment markers are reversed')
    fragment = text[start:end]
    return fragment, dict(source=_pin(path), start_marker=start_marker, end_marker=end_marker,
        char_start=start, char_end_exclusive=end,
        line_start=text.count('\n', 0, start) + 1,
        line_end=text.count('\n', 0, max(start, end - 1)) + 1,
        extracted_sha256=sha256(fragment.encode()).hexdigest())


def _serial_lines(path, text):
    tree = ast.parse(text, filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_source')
    candidates = []
    for node in ast.walk(function):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'execution'):
            continue
        value = node.value
        if (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
                and isinstance(value.right, ast.Constant) and type(value.right.value) is str
                and isinstance(value.left, ast.BinOp) and isinstance(value.left.op, ast.Add)
                and isinstance(value.left.left, ast.Constant) and type(value.left.left.value) is str
                and isinstance(value.left.right, ast.Name) and value.left.right.id == 'program'):
            candidates.append((node, value.left.left.value, value.right.value))
    _need(len(candidates) == 1, 'one original serial renderer concatenation required')
    node, bind, check = candidates[0]
    _need(bind.splitlines() == ['    if (!caller_affinity.bind(planned_cpu_ids[0])) return 4;']
          and check.splitlines() == ['', '    placement_guard |= static_cast<uint32_t>(!caller_affinity.bound());'],
          'original serial bind/bound-check statement changed')
    provenance = dict(source=_pin(path), line_start=node.lineno, line_end=node.end_lineno,
        extraction='STATIC_AST_STRING_CONSTANTS_FROM_ORIGINAL_EXECUTION_ASSIGNMENT',
        bind_sha256=sha256(bind.encode()).hexdigest(), boundcheck_sha256=sha256(check.encode()).hexdigest())
    return bind, check, provenance


def render_boundary_cut_probe(parameters, memory, cpus, core):
    """Return (C++20 source, provenance) without compiling or executing it."""
    parameters = json.loads(json.dumps(parameters, sort_keys=True, allow_nan=False))
    memory = json.loads(json.dumps(memory, sort_keys=True, allow_nan=False))
    core = json.loads(json.dumps(core, sort_keys=True, allow_nan=False))
    _need(type(cpus) in (tuple, list) and len(cpus) == 1
          and type(cpus[0]) is int and 0 <= cpus[0] < 2**20, 'one explicit serial CPU required')
    for key in ('arrays', 'scalars', 'providers'):
        _need(type(parameters.get(key)) is int and 0 <= parameters[key] <= 4096,
              'bounded exact array/scalar/provider counts required')
    shapes = parameters['array_shapes']
    _need(type(shapes) is list and len(shapes) == parameters['arrays']
          and all(type(shape) is list and len(shape) == 2
                  and all(type(n) is int and 0 <= n <= 2**63 - 1 for n in shape) for shape in shapes),
          'one concrete rank-two shape per array required')
    capacities = parameters['workspace_slot_capacities']
    _need(type(capacities) is list and len(capacities) <= 4096
          and all(type(n) is int and 0 <= n <= 2**63 - 1 and n % 8 == 0 for n in capacities),
          'original fixed-pool capacities required even though this probe does not allocate it')
    memory_keys = ('workspace_payload_bytes', 'external_array_bytes', 'total_region_array_payload_bytes', 'memory_budget_bytes')
    _need(all(type(memory.get(key)) is int and 0 <= memory[key] <= 2**63 - 1 for key in memory_keys)
          and memory['workspace_payload_bytes'] == sum(capacities)
          and memory['external_array_bytes'] == sum(rows * cols * 8 for rows, cols in shapes)
          and memory['total_region_array_payload_bytes'] == memory['workspace_payload_bytes'] + memory['external_array_bytes']
          and memory['total_region_array_payload_bytes'] <= memory['memory_budget_bytes'], 'canonical payload metadata differs')
    _need(all(type(core.get(key)) is str and re.fullmatch('[0-9a-f]{64}', core[key])
              for key in ('region_sha256', 'plan_sha256')) and type(core.get('provider_runtime')) is dict,
          'original ABI core identities and provider runtime required')
    backend_path = _HERE/'stateful_v1.cpp.in'; backend = backend_path.read_text()
    loop_path = _HERE/'loop_v1.cpp.in'; loop = loop_path.read_text()
    renderer_path = _HERE/'stateful.py'; renderer = renderer_path.read_text()
    fragments = {}

    def extract(name, path, source, start, end):
        value, fragments[name] = _fragment(path, source, start, end)
        return value

    replacements = dict(
        METADATA_PREFIX=extract('metadata', backend_path, backend,
            '    if (array_count != @ARRAY_COUNT@ || scalar_count != @SCALAR_COUNT@ ||',
            '    std::array<std::unique_ptr<double[]>, @SLOT_COUNT@> slots;'),
        CAPTURE_AND_SCALAR=extract('capture_and_scalar', backend_path, backend,
            '    ThreadAffinity caller_affinity;', '    uint32_t numeric_guard = 0, placement_guard = 0;'),
        RESTORE_AND_CHECKS=extract('restore_and_checks', backend_path, backend,
            '    numeric_guard |= static_cast<uint32_t>(std::fetestexcept(floating_flags) != 0);',
            '    // All failing work is complete.'),
        ARRAY_HELPERS=extract('array_helpers', backend_path, backend, 'bool finite(double value) {', 'struct Scalar {'),
        AFFINITY_CLASS=extract('affinity', loop_path, loop, 'class ThreadAffinity {',
            '\n#endif\n\nconstexpr int64_t modeled_rows'),
        FLOATING_CLASS=extract('floating', loop_path, loop, 'class FloatingScope {', '\nbool valid_array('),
        NUMERIC_DECLARATIONS=extract('numeric_declarations', backend_path, backend,
            '    uint32_t numeric_guard = 0, placement_guard = 0;', '@INPUT_FINITE_GUARDS@'),
        FLOATING_FLAGS=extract('floating_flags', backend_path, backend,
            'constexpr int floating_flags =', 'constexpr int32_t affinity_guard_status ='))
    bind, check, serial_provenance = _serial_lines(renderer_path, renderer)
    replacements.update(SERIAL_BIND=bind, SERIAL_BOUNDCHECK=check,
        CPU_IDS=str(cpus[0]), CPU_SLOTS=str(cpus[0]+1), ARRAY_COUNT=str(parameters['arrays']),
        SCALAR_COUNT=str(parameters['scalars']), PROVIDER_COUNT=str(parameters['providers']),
        SLOT_COUNT=str(len(capacities)), SLOT_CAPACITIES=', '.join(str(n)+'ULL' for n in capacities),
        PAYLOAD_BYTES=str(memory['workspace_payload_bytes']), EXTERNAL_BYTES=str(memory['external_array_bytes']),
        TOTAL_BYTES=str(memory['total_region_array_payload_bytes']), BUDGET=str(memory['memory_budget_bytes']),
        ARRAY_GUARDS='\n'.join(f'    if (!valid_array(arrays[{i}], {shape[0]}, {shape[1]})) return 2;'
                               for i, shape in enumerate(shapes)),
        REGION_SHA256=core['region_sha256'], PLAN_SHA256=core['plan_sha256'], CONTRACT_SHA256=canonical_sha256(core))
    template_path = _HERE/'stateful_boundary_cuts_v1.cpp.in'; source = template_path.read_text()
    # Fragment substitution precedes scalar specialization: copied backend text
    # still contains the original template's count/shape/payload tokens.
    for name, value in replacements.items():
        source = source.replace('@'+name+'@', value)
    _need('@' not in source, 'unresolved canonical boundary-cut source token')
    provenance = dict(schema='scar.stateful-boundary-cut-source', schema_version=1,
        profile='SERIAL_COMPILER_STATEFUL_ARTIFACT_CANONICAL_BOUNDARY_CUTS_V1',
        source_pins=[_pin(path) for path in (Path(__file__), template_path, backend_path, loop_path, renderer_path,
            _HERE/'stateful_abi_v1.h', _HERE/'scar_abi_v1.h', _HERE.parent/'runtime/stateful_contract.py',
            _HERE.parent/'runtime/loop_contract.py')], fragments=fragments, serial_renderer=serial_provenance,
        generated_source_sha256=sha256(source.encode()).hexdigest(),
        parameters=parameters, memory=memory, cpus=list(cpus), abi_core=core,
        contract_sha256=canonical_sha256(core), clock='CLOCK_MONOTONIC', clock_unit='INTEGER_NANOSECONDS',
        cut_names=list(_CUT_NAMES), read_abi='scar_boundary_cut_read_v1(uint64_t*,uint64_t)',
        original_operator_abi=1, original_compiler_options_required=True,
        skipped_work=['POOL_ALLOCATION_AND_RELEASE', 'INPUT_FINITE_SCANS', 'SCALAR_DAG', 'ARRAY_DAG', 'COMMIT'],
        retained_residual=['CLOCK_GETTIME_AND_THREAD_LOCAL_STORES', 'CPP_POST_CUT_RAII_METADATA_CLEANUP',
            'FFI_AND_PYTHON_RETURN'], instrumentation_subtracted=False,
        source_fragments_copied_verbatim_before_parameter_specialization=True,
        compiler_wrapper_measured_by_renderer=False, compact_runtime_equivalence_claimed=False,
        target_program_executed=False, native_compiled=False, performance_measured=False)
    return source, provenance


__all__ = ['render_boundary_cut_probe']
