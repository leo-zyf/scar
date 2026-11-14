"""SIMD layout uses sourced proxies and a replayable two-address live model."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_cpu_hardware_placement import hardware
from test_loop_strategy import make_loop
from scar.codegen.loop import layout_compiler_options
from scar.optimization import bind_cpu_placement, propose_cpu_placements
from scar.optimization.loop_layout import plan_loop_layout, verify_loop_layout
from scar.optimization.loop_strategy import loop_execution_plan, plan_loop_strategy


def setup_layout(hardware, tmp_path, shape=(2, 9)):
    """Pure declared numbers for model tests; these are not hardware timings."""
    loop = make_loop()
    budget = shape[0] * shape[1] * 8
    strategy = plan_loop_strategy(loop, input_shapes={'state': list(shape)},
                                  trip_count=3, memory_budget_bytes=budget)
    plan = loop_execution_plan(loop, strategy)
    placement = next(item for item in propose_cpu_placements(plan, hardware,
        max_threads=2, memory_budget_bytes=budget) if len(item.cpu_ids) == 2)
    bound = bind_cpu_placement(plan, placement)
    context = dict(target_path=placement.target_path, target_sha256=placement.target_sha256,
        executor_id=placement.executor_id, measured_cpu_id=placement.cpu_ids[0],
        cpu_ids=list(placement.cpu_ids), cpu_model='fixture CPU',
        compiler=dict(argv=['g++'], identity='declared model-test compiler',
                      options=list(layout_compiler_options(bound))))
    isa = dict(architecture='x86_64', name='SSE2_X86_64', lanes=2,
               vector_registers=16, supported=True)
    registers = dict(reserved_vector_registers=2, max_vector_chains=8)
    facts = dict(schema='scar.test-loop-layout-facts', context=context, isa=isa,
        registers=registers, values=dict(dependent=4, independent=1, finite=1),
        finite_mask_implementation='SSE2_INTEGER_EXPONENT_AND_CMPEQ32_MOVEMASK_PD')
    path = tmp_path / ('layout-facts-' + str(shape[1]) + '.json')
    path.write_text(json.dumps(facts))
    raw = path.read_bytes()
    def reference(pointer):
        return dict(artifact=dict(uri=str(path), sha256=sha256(raw).hexdigest(),
            size_bytes=len(raw), schema=facts['schema']), locator=pointer)
    def metric(name):
        return dict(value=facts['values'][name], unit='ns/vector',
                    evidence=[reference('#/values/' + name)])
    profile = dict(schema='scar.cpu-simd-primitives', schema_version=1,
        context={**context, 'evidence': [reference('#/context')]},
        isa={**isa, 'evidence': [reference('#/isa')]},
        register_model={**registers, 'evidence': [reference('#/registers')]},
        measurements=dict(ops=dict(neg=dict(dependent_ns_per_vector=metric('dependent'),
            independent_ns_per_vector=metric('independent'))),
            finite_mask=dict(ns_per_vector=metric('finite'))))
    return loop, strategy, bound, profile


def test_two_address_liveness_proxy_and_row_geometry_are_purely_replayable(hardware, tmp_path):
    loop, strategy, bound, profile = setup_layout(hardware, tmp_path)
    layout = plan_loop_layout(loop, strategy, bound, hardware_profile=profile)
    assert layout['status'] == 'PLANNED_UNDER_MODEL' and not layout['cost_frontier']
    assert layout['selected']['vector_chains'] == 4 and layout['selected']['tile_elements'] == 8
    assert layout['geometry']['full_tiles_per_row'] == layout['geometry']['scalar_tail_elements_per_row'] == 1
    registers = layout['register_model']
    assert registers['per_chain_live_peak'] == 1 and registers['predicted_vector_registers'] == 6
    assert all(row['reuse_first_operand'] for row in registers['live_ranges'])
    assert not registers['compiler_spills_verified'] and not layout['time_optimality_proven']
    assert not layout['model']['whole_loop_duration_predicted'] and not layout['target_variants_executed']
    assert layout['scratch_budget_scope'] == 'ABSTRACT_SIMD_REGISTER_STORAGE_ONLY_NOT_NATIVE_STACK_CAP'
    assert verify_loop_layout(loop, strategy, bound, layout) == layout
    changed = deepcopy(layout); changed['selected']['vector_chains'] = 5
    with pytest.raises(ValueError, match='cost replay'):
        verify_loop_layout(loop, strategy, bound, changed)


def test_unknown_cost_wrong_guard_and_infeasible_row_never_claim_time_optimality(hardware, tmp_path):
    loop, strategy, bound, profile = setup_layout(hardware, tmp_path)
    missing = deepcopy(profile); del missing['measurements']['ops']['neg']['dependent_ns_per_vector']
    unknown = plan_loop_layout(loop, strategy, bound, hardware_profile=missing)
    assert unknown['cost_frontier'] and unknown['selected']['vector_chains'] == 4
    assert unknown['model']['effective_dependency_proxy_ns'] is None
    assert unknown['model']['selection_policy'].endswith('NO_TIME_MODEL')
    # Repoint a valid numeric receipt to a source carrying the superseded FP
    # comparison guard: equal numeric value alone does not establish its cost.
    bad_guard = deepcopy(profile)
    source = tmp_path / 'layout-facts-9.json'
    old = json.loads(source.read_bytes()); old.pop('finite_mask_implementation')
    old_path = tmp_path / 'old-guard-facts.json'; old_path.write_text(json.dumps(old))
    raw = old_path.read_bytes()
    for section in (bad_guard['context'], bad_guard['isa'], bad_guard['register_model'],
                    *bad_guard['measurements']['ops']['neg'].values(),
                    bad_guard['measurements']['finite_mask']['ns_per_vector']):
        artifact = section['evidence'][0]['artifact']
        artifact.update(uri=str(old_path), sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    guarded = plan_loop_layout(loop, strategy, bound, hardware_profile=bad_guard)
    assert any('integer SSE2 finite-mask' in row['reason'] for row in guarded['cost_frontier'])
    assert guarded['model']['effective_dependency_proxy_ns'] is None
    narrow_loop, narrow_strategy, narrow_plan, narrow_profile = setup_layout(hardware, tmp_path, (2, 1))
    narrow = plan_loop_layout(narrow_loop, narrow_strategy, narrow_plan, hardware_profile=narrow_profile)
    assert narrow['status'] == 'UNRESOLVED' and narrow['selected'] is None
    assert any(row['field'] == 'state_shape.columns' for row in narrow['frontier'])
    assert not narrow['physical_optimality_proven'] and not guarded['time_optimality_proven']
