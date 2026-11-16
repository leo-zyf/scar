"""Original renderer metadata and stage signatures, without native execution."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import subprocess

from test_cpu_hardware_placement import hardware
from scar.codegen.stateful import _source
from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
from scar.compiler.model import Expr, SourceRef
from scar.compiler.stateful_model import (REQUIRED_GUARDS, StatefulAction,
    StatefulInput, StatefulRegion)
from scar.optimization.hardware import propose_cpu_allocations
from scar.optimization.stateful_schedule import plan_stateful_region
from scar.optimization.stateful_stage_signature import project_stage_signature


def _region(nodes, values, *, second_write=False):
    inputs = (StatefulInput('left', ('state', 'left'), 'array'),
        StatefulInput('right', ('state', 'right'), 'array'),
        StatefulInput('scale', ('state', 'scale'), 'scalar'))
    actions = []
    for i, value in enumerate(values):
        if value == 'WRITE':
            actions.append(StatefulAction('write:0', 'write', values[i-1], 'left',
                None, 0, 1, 'assign', {'port': 'world0'}, {'port': 'world1'}))
        else:
            actions.append(StatefulAction('evaluate:'+value, 'evaluate', value))
    if second_write:
        actions.append(StatefulAction('write:1', 'write', values[-1], 'left',
            None, 1, 2, 'assign', {'port': 'world1'}, {'port': 'world2'}))
    return StatefulRegion('stateful:stage-signature-fixture',
        SourceRef('/tmp/stage_signature_fixture.py', '0'*64, 'step', 1, 0, 'pycode-v2:'+'1'*64),
        ('state',), inputs, tuple(nodes), tuple(actions), guards=REQUIRED_GUARDS)


def _inputs():
    return (Expr('x', 'input', attrs=dict(name='left', kind='array', epoch=0)),
        Expr('y', 'input', attrs=dict(name='right', kind='array', epoch=0)),
        Expr('s', 'input', attrs=dict(name='scale', kind='scalar', epoch=None)))


def _render(region, hardware):
    binding = next(row for row in propose_cpu_allocations(hardware, max_threads=1,
        memory_budget_bytes=8192) if len(row.cpu_ids) == 1)
    plan = plan_stateful_region(region, input_shapes={'left': [3, 5], 'right': [3, 5]},
        cpu_threads=1, memory_budget_bytes=8192, resource_binding=binding)
    assert plan['status'] == 'READY'
    contract = {'fixture': 'HOST_RENDER_ONLY_NOT_ADMITTED_NATIVE_EXECUTION'}
    capture = capture_stateful_stage_sources(region, plan, binding, contract)
    return plan, capture, binding, contract


def test_stage_key_retains_order_sharing_constant_tags_but_not_expression_spelling(hardware, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('stage-signature test must not compile or execute a target')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    def make(*, swap=False, constant=1):
        nodes = (*_inputs(), Expr('c', 'const', attrs={'value': constant}),
            Expr('q', 'sub', ('x', 'y')), Expr('r', 'mul', ('q', 'q')),
            Expr('z', 'add', ('c', 'r') if swap else ('r', 'c')))
        return _region(nodes, ('q', 'r', 'z', 'WRITE'))
    def signature(region):
        plan, capture, _, _ = _render(region, hardware)
        stage = next(row for row in plan['stages'] if row['kind'] == 'COMPUTE' and row['root_expr'] == region.actions[-1].value)
        return project_stage_signature(region, plan, capture, stage['index'])
    original = make()
    first = signature(original)
    swapped = signature(make(swap=True))
    floating = signature(make(constant=1.0))
    assert len({first['key'], swapped['key'], floating['key']}) == 3
    dag = first['parameters']['dag']
    multiply = next(row for row in dag['nodes'] if row.get('op') == 'mul')
    assert multiply['inputs'][0] == multiply['inputs'][1]
    assert len([row for row in dag['nodes'] if row.get('op') == 'sub']) == 1
    assert dag['nodes'][dag['root']]['inputs'] != swapped['parameters']['dag']['nodes'][
        swapped['parameters']['dag']['root']]['inputs']
    constant = next(row for row in dag['nodes'] if row['kind'] == 'CONSTANT')
    float_constant = next(row for row in floating['parameters']['dag']['nodes'] if row['kind'] == 'CONSTANT')
    assert constant['float64_hex'] == float_constant['float64_hex']
    assert constant['integer_tag'] is True and float_constant['integer_tag'] is False
    names = {node.id: 'renamed_'+str(i) for i, node in enumerate(original.nodes)}
    renamed = replace(original,
        nodes=tuple(replace(node, id=names[node.id], inputs=tuple(names[key] for key in node.inputs))
                    for node in original.nodes),
        actions=tuple(replace(action, value=names[action.value]) for action in original.actions))
    renamed_signature = signature(renamed)
    assert renamed_signature['key'] == first['key']
    assert renamed_signature['parameters'] == first['parameters']
    assert renamed_signature['source_mapping'] != first['source_mapping']
    assert first['runtime_logical_identity_claimed'] is False and first['measurement_claimed'] is False


def test_capture_replays_original_history_ready_scalars_and_physical_backing_relationships(hardware, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('stage capture must remain host-source-only')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    nodes = (*_inputs(), Expr('factor', 'neg', ('s',)),
        Expr('held', 'add', ('x', 'x')), Expr('rhs', 'mul', ('held', 'factor')),
        Expr('x1', 'input', attrs=dict(name='left', kind='array', epoch=1)),
        Expr('later', 'add', ('x1', 'held')), Expr('last', 'mul', ('later', 'factor')))
    region = _region(nodes, ('factor', 'held', 'rhs', 'WRITE', 'later', 'last'), second_write=True)
    plan, capture, binding, contract = _render(region, hardware)
    full_source = _source(region, plan, binding, contract)
    assert capture['full_source_sha256'] == sha256(full_source.encode()).hexdigest()
    assert capture['provenance']['full_source_equal_to_original_renderer']
    assert capture['provenance']['ast_transform']['removing_capture_hooks_restores_original_ast']
    assert not capture['provenance']['native_compiled'] and not capture['provenance']['target_program_executed']
    original_nodes = {node.id: node for node in region.nodes}
    stages = {stage['index']: stage for stage in plan['stages']}
    available = {}; ready = set(); releases_seen = False; held_after_write = False
    signatures = []
    for item in capture['stages']:
        stage = stages[item['index']]
        assert item['entry_available'] == available
        assert item['entry_scalar_ready'] == sorted(ready)
        assert item['body_sha256'] == sha256(item['body'].encode()).hexdigest()
        if stage['kind'] == 'SCALAR':
            ready.update(key for key in stage['expr_ids'] if original_nodes[key].op not in ('input', 'const', 'slice'))
        else:
            if stage['output_payload'] and stage['root_expr']:
                available[stage['root_expr']] = stage['output_payload']
        assert item['exit_scalar_ready'] == sorted(ready)
        released = set(stage['release_values_after'])
        releases_seen |= bool(released)
        available = {key: value for key, value in available.items() if value not in released}
        if stage['kind'] in ('SCALAR', 'COMPUTE'):
            signature = project_stage_signature(region, plan, capture, item['index'])
            assert signature['entry']['available'] == item['entry_available']
            assert signature['entry']['scalar_ready'] == item['entry_scalar_ready']
            signatures.append((stage, signature))
            if stage['kind'] == 'COMPUTE' and stage['root_expr'] == 'last':
                held_after_write = 'held' in item['entry_available']
                kinds = [row['kind'] for row in signature['parameters']['dag']['nodes']]
                assert 'READY_SCALAR' in kinds
                assert not any(row.get('op') == 'neg' for row in signature['parameters']['dag']['nodes'])
    assert releases_seen and held_after_write
    assert _source(region, plan, binding, contract) == full_source
    stage, signature = next((s, g) for s, g in signatures if s['kind'] == 'COMPUTE' and s['root_expr'] == 'rhs')
    groups = signature['source_mapping']['backing_groups']
    assert len({(row['kind'], row['original']) for row in groups}) == len(groups)
    # A deliberately unadmitted layout mutant checks encoding only. It is not
    # a legal plan or a transformation, and no generated code is executed.
    aliased = deepcopy(plan)
    input_slot = next(row['original'] for row in groups if row['kind'] == 'WORKSPACE_SLOT')
    aliased['assignment'][stage['output_payload']]['slot_id'] = input_slot
    alias_capture = capture_stateful_stage_sources(region, aliased, binding, contract)
    alias_signature = project_stage_signature(region, aliased, alias_capture, stage['index'])
    alias_dag = alias_signature['parameters']['dag']
    assert alias_dag['output']['group'] in {row['group'] for row in alias_dag['nodes'] if row['kind'] == 'ARRAY_LOAD'}
    assert alias_signature['key'] != signature['key']
