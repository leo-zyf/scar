"""Source-proven normal-success RAII scope and a serial canonical observer.

Rendering only: no target execution, compiler, sampling or cost authorization.
The old boundary observer and its receipts are unchanged.
"""
import ast
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re

from scar.codegen.boundary_cuts import _fragment, _need, _pin, render_boundary_cut_probe

_HERE = Path(__file__).resolve().parent
_CUT_NAMES = ('entry', 'metadataready', 'captureready', 'bindingready',
              'workerchecked', 'restoreready', 'helper_cleanup_ready')


def stateful_scope_source_evidence():
    """Replay the current source skeleton; this proves no measured duration."""
    backend_path = _HERE/'stateful_v1.cpp.in'; backend = backend_path.read_text()
    renderer_path = _HERE/'stateful.py'; renderer = renderer_path.read_text()
    loop_path = _HERE/'loop_v1.cpp.in'; loop = loop_path.read_text()
    recipe_path = _HERE.parent/'optimization/stateful_recipe.py'; recipe = recipe_path.read_text()
    fragments = {}

    def extract(name, path, text, start, end):
        fragment, fragments[name] = _fragment(path, text, start, end)
        return fragment

    declarations = extract('allocation_and_helper_declarations', backend_path, backend,
        '    std::array<std::unique_ptr<double[]>, @SLOT_COUNT@> slots;',
        '    fenv_t execution_environment;')
    _need(declarations.count('    ThreadAffinity caller_affinity;') == 1
          and declarations.count('    FloatingScope caller_environment;') == 1
          and declarations.index('    ThreadAffinity caller_affinity;')
              < declarations.index('    FloatingScope caller_environment;'),
          'original pool/affinity/floating declaration order changed')
    # These original declarations are all at the same run-body brace depth.
    # The only inner scope in this fragment is the completed pool-allocation loop.
    _need(declarations.count('{') == 1 and declarations.count('}') == 1
          and declarations.index('}') < declarations.index('    ThreadAffinity caller_affinity;'),
          'original helper/pool objects no longer share the normal run scope')
    restore = extract('restore_checks_and_commit_boundary', backend_path, backend,
        '    numeric_guard |= static_cast<uint32_t>(std::fetestexcept(floating_flags) != 0);',
        '@COMMITS@')
    expected_checks = (
        '    numeric_guard |= static_cast<uint32_t>(std::fetestexcept(floating_flags) != 0);\n'
        '    const bool environment_restored = caller_environment.restore();\n'
        '    const bool affinity_restored = caller_affinity.restore_original();\n'
        '    if (numeric_guard || !environment_restored) return 3;\n'
        '    if (placement_guard || !affinity_restored) return 4;')
    code = re.sub(r'//[^\n]*', '', restore).strip()
    _need(code == expected_checks.strip(), 'restore checks or the precommit statement-free boundary changed')
    tail = extract('commit_and_native_return', backend_path, backend,
        '@COMMITS@', '\nconst scar_stateful_table_v1 table')
    _need(tail == '@COMMITS@\n    return 0;\n}', 'normal commit/return skeleton changed')
    floating_destructor = extract('floating_destructor', loop_path, loop,
        '    ~FloatingScope()', '\n};\n\nbool valid_array(')
    _need(floating_destructor.strip() ==
          '~FloatingScope() { if (saved_ok_ && !restored_) std::fesetenv(&saved_); }',
          'successful floating helper cleanup changed')
    affinity_destructor = extract('affinity_destructor', loop_path, loop,
        '    ~ThreadAffinity()', '\n};\n#endif')
    _need(all(token in affinity_destructor for token in
          ('restore();', 'CPU_FREE(saved_);', 'CPU_FREE(target_);', 'CPU_FREE(observed_);')),
          'original affinity helper cleanup changed')
    extract('floating_restore', loop_path, loop, '    bool restore() noexcept {\n        if (!saved_ok_)',
        '    ~FloatingScope()')
    extract('affinity_restore', loop_path, loop, '    bool restore() noexcept {\n        if (!changed_)',
        '    ~ThreadAffinity()')
    extract('renderer_commit_generation', renderer_path, renderer, '    commits=[]',
        "    original=(_HERE/'loop_v1.cpp.in').read_text()")
    fence = extract('recipe_semantic_success_gate', recipe_path, recipe,
        "        if kind=='COMMIT' and not fence_added:", '        dependencies=list(stage[\'depends_on\'])')
    _need('Kind.SUCCESS_FENCE' in fence and 'ALL_COMPUTE_FINITE_AND_FENV_SUCCESS' in fence
          and 'CPU_AND_FENV_RESTORATION_SUCCESS' in fence, 'original recipe success gate changed')
    release_rule = extract('recipe_reverse_pool_release_rule', recipe_path, recipe,
        "    release_slots = reversed(plan['workspace_slots']) if physical_release_order else plan['workspace_slots']",
        '    return VersionedResourceSchedule(')
    _need(release_rule.startswith("    release_slots = reversed(plan['workspace_slots']) if physical_release_order else plan['workspace_slots']\n    for row in release_slots:\n")
          and "releases=(row['id'],)" in release_rule,
          'source recipe no longer emits each slot in C++ reverse destruction order')
    _need(recipe.count("    if physical_release_order: control['pool_release_order']=CPP_POOL_RELEASE_ORDER") == 1
          and recipe.count('recipe=_assemble(region,plan,artifacts,physical_release_order=True)') == 1,
          'new recipes must explicitly declare the physical C++ reverse pool release rule')
    tree = ast.parse(renderer, filename=str(renderer_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_source')
    _need(not any(isinstance(node, ast.Constant) and type(node.value) is str
                  and ('SUCCESS_FENCE' in node.value or 'success:precommit' in node.value)
                  for node in ast.walk(function)), 'renderer now emits an explicit semantic success fence')
    _need(renderer.count("        if kind=='COMMIT': continue") == 1
          and renderer.count("COMMITS='\\n'.join(commits)") == 1,
          'original stage/commit source generation changed')
    return dict(schema='scar.stateful-implementation-scope-source', schema_version=1,
        source_pins=[_pin(path) for path in (Path(__file__), backend_path, renderer_path, loop_path, recipe_path)],
        fragments=fragments, normal_success_only=True,
        reverse_raii_order=['FloatingScope', 'ThreadAffinity', 'WORKSPACE_POOL_UNIQUE_PTR_ARRAY'],
        pool_release_rule=dict(control_flag='pool_release_order',
            required_value='CPP_STD_ARRAY_REVERSE_DESTRUCTION',
            order='REVERSED_ORIGINAL_WORKSPACE_SLOT_DECLARATION_ORDER',
            legacy_missing_flag='REPLAY_LEGACY_GRAPH_ONLY_NOT_MODERN_MULTI_RELEASE_SCOPE_AUTHORITY'),
        helper_cleanup_before_pool_release=True,
        successful_explicit_restoration_before_commit=True,
        no_runtime_statement_between_restore_checks_and_commit=True,
        recipe_fence_is_semantic_gate_not_additional_emitted_work=True,
        source_proven_zero_additional_service_not_measured_zero=True,
        costs_inferred=False, target_program_executed=False)


def render_scope_boundary_probe(parameters, memory, cpus, core):
    """Return (C++20 source, provenance); native cuts=7, original table ABI=1."""
    evidence = stateful_scope_source_evidence()
    original, base = render_boundary_cut_probe(parameters, memory, cpus, core)
    header_marker = 'int32_t run(const scar_array_v1 *arrays, uint64_t array_count,'
    body_marker = '    boundary_cuts.fill(0);'
    first_cut_end = '    if (!record_cut(0)) return 3;\n'
    helper_marker = '    ThreadAffinity caller_affinity;'
    last_cut_end = '    if (!record_cut(5)) return 3;\n'
    suffix_marker = 'const scar_stateful_table_v1 table ='
    for marker in (header_marker, body_marker, first_cut_end, helper_marker, last_cut_end, suffix_marker):
        _need(original.count(marker) == 1, 'original canonical boundary skeleton changed: '+marker)
    prefix, after_header = original.split(header_marker, 1)
    header_rest, after_body = after_header.split(body_marker, 1)
    _, after_first = after_body.split(first_cut_end, 1)
    metadata, after_helper = after_first.split(helper_marker, 1)
    helper_rest, tail = after_helper.split(last_cut_end, 1)
    _need(tail == '    // Affinity/FENV RAII destruction and FFI return follow this last cut.\n'
          '    return 0;\n}\n'+suffix_marker+original.split(suffix_marker, 1)[1],
          'original canonical helper/return boundary changed')
    _need(prefix.count('std::array<uint64_t, 6> boundary_cuts') == 1,
          'original canonical cut storage changed')
    prefix = prefix.replace('std::array<uint64_t, 6> boundary_cuts', 'std::array<uint64_t, 7> boundary_cuts')
    replacements = dict(ORIGINAL_PREFIX=prefix.rstrip(), RUN_HEADER=header_marker+header_rest.rstrip(),
        METADATA_TO_CAPTURE=metadata.rstrip(),
        CAPTURE_THROUGH_RESTORE=helper_marker+helper_rest+last_cut_end.rstrip(),
        ORIGINAL_SUFFIX=suffix_marker+original.split(suffix_marker, 1)[1])
    template = _HERE/'stateful_scope_boundary_v1.cpp.in'
    source = template.read_text()
    for name, value in replacements.items():
        _need(source.count('@'+name+'@') == 1, 'one explicit scope template slot required')
        source = source.replace('@'+name+'@', value)
    _need('@' not in source, 'unresolved canonical scope boundary token')
    provenance = deepcopy(base)
    provenance.update(schema='scar.stateful-scope-boundary-source',
        profile='SERIAL_COMPILER_STATEFUL_ARTIFACT_CANONICAL_SCOPE_BOUNDARY_V1',
        source_pins=[*base['source_pins'], _pin(Path(__file__)), _pin(template)],
        generated_source_sha256=sha256(source.encode()).hexdigest(), cut_names=list(_CUT_NAMES),
        original_boundary_source_sha256=sha256(original.encode()).hexdigest(),
        original_boundary_provenance=base, implementation_scope_source=evidence,
        scope_transformation=dict(method='MECHANICAL_ORIGINAL_RENDERED_FRAGMENT_REASSEMBLY',
            original_native_cuts_preserved=6, added_native_cut='helper_cleanup_ready',
            helper_scope_closed_before_last_cut=True, original_source_fragments_unchanged=True,
            original_table_abi_unchanged=True, pool_release_measured=False),
        retained_residual=['CLOCK_GETTIME_AND_THREAD_LOCAL_STORES', 'HELPER_SCOPE_REASSEMBLY_AND_EXTRA_CUT',
            'FFI_AND_PYTHON_RETURN'], helper_cleanup_inside_cut_interval=True)
    return source, provenance


__all__ = ['render_scope_boundary_probe', 'stateful_scope_source_evidence']
