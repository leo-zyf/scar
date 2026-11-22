"""A declared unit model tests mechanical delivery, never hardware performance."""
from copy import deepcopy
from hashlib import sha256
import json
import os
import sys

import numpy as np
import pytest

from scar.analysis.target_graph import build_target_graph
from scar.compiler.selected_stateful_lowering import lower_stateful_choices
from scar.compiler.model import Expr, SourceRef
from scar.compiler.stateful_model import (StatefulRegion, StatefulInput,
    StatefulAction, REQUIRED_GUARDS)
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.stateful_cpu_model import build_stateful_cpu_domain, stateful_cpu_cost_context, plan_stateful_cpu_choices


def test_selected_model_lowers_exactly_one_real_kernel_and_unknown_keeps_original(tmp_path):
    if sys.platform != 'linux' or np.__version__ != '2.5.2':
        pytest.skip('existing stateful backend requires Linux and NumPy 2.5.2')
    cpu = min(os.sched_getaffinity(0))
    inventory = dict(schema='scar.target-inventory', schema_version=1, sources=[], diagnostics=[],
        platform=dict(system='linux', machine='x86_64'), dram=dict(memory_total_bytes=2**20),
        cpu=dict(processors=[dict(cpu_id=cpu, package_id=0, core_id=0)],
            online_cpu_ids=[cpu], model_names=['retained unit-test CPU'],
            numa_nodes=[dict(node_id=0, cpu_ids=[cpu], memory_total_bytes=2**20)]),
        gpus=[], paths=[], storage=dict(disks=[], block_devices=[]))
    inventory_path = tmp_path / 'inventory.json'; inventory_path.write_text(json.dumps(inventory))
    target = tmp_path / 'target'; build_target_graph(target, inventory=inventory_path)
    source = tmp_path / 'step.py'
    source.write_text('def step(state, gain):\n    state[:] += state * gain\n')
    region = StatefulRegion('test:choice-delivery',
        SourceRef(str(source), sha256(source.read_bytes()).hexdigest(), 'step', 1, 0, 'pycode-v2:'+'1'*64),
        ('state', 'gain'), (StatefulInput('a', ('state',), 'array'), StatefulInput('g', ('gain',), 'scalar')),
        (Expr('a0', 'input', attrs=dict(name='a', kind='array', epoch=0)),
         Expr('g0', 'input', attrs=dict(name='g', kind='scalar', epoch=None)),
         Expr('rhs', 'mul', ('a0', 'g0'))),
        (StatefulAction('eval', 'evaluate', 'rhs'),
         StatefulAction('write', 'write', 'rhs', 'a', None, 0, 1, 'add',
             dict(port='before'), dict(port='after'))), guards=REQUIRED_GUARDS)
    domain = build_stateful_cpu_domain(region, target, input_shapes={'a': [3, 5]},
        memory_budget_bytes=4096, affinity=[cpu], max_threads=1, out=tmp_path / 'domain')
    unknown = plan_stateful_cpu_choices(domain, out=tmp_path / 'unknown')
    preserved = lower_stateful_choices(tmp_path / 'unknown', tmp_path / 'preserved')
    assert unknown['selected_id'] is None and preserved['status'] == 'ORIGINAL_EXECUTION_RETAINED'
    assert preserved['compiled_variant_count'] == 0 and preserved['recipe'] is None

    facts = dict(schema='test.declared-stateful-model', contexts={}, estimates={}, derivations={})
    for identifier in [row['id'] for row in domain['candidates']] + ['NO_OP']:
        facts['contexts'][identifier] = stateful_cpu_cost_context(domain, identifier)
        facts['estimates'][identifier] = dict(duration_ticks=200 if identifier == 'NO_OP' else 100, tick_seconds=1e-9)
        facts['derivations'][identifier] = dict(origin='DECLARED_FORWARD_MODEL', target_variant_trials=False,
            assumptions=['SYNTHETIC_UNIT_MODEL_NOT_MEASURED_HARDWARE_COST'])
    facts_path = tmp_path / 'facts.json'; facts_path.write_text(json.dumps(facts))
    artifact = artifact_reference(facts_path, facts['schema'])
    def ref(pointer): return dict(artifact=artifact, locator=pointer)
    profiles = {identifier: dict(schema='scar.stateful-cpu-cost-profile', schema_version=1,
        context=facts['contexts'][identifier], context_evidence=[ref('#/contexts/'+identifier)],
        estimate={**facts['estimates'][identifier], 'evidence': [ref('#/estimates/'+identifier)]},
        derivation={**facts['derivations'][identifier], 'evidence': [ref('#/derivations/'+identifier)]})
        for identifier in facts['contexts']}
    baseline = profiles.pop('NO_OP')
    chosen = plan_stateful_cpu_choices(domain, cost_profiles=profiles, no_op_profile=baseline,
        out=tmp_path / 'chosen')
    preview = lower_stateful_choices(tmp_path / 'chosen', tmp_path / 'preview', plan_only=True)
    assert preview['status'] == 'MODEL_SELECTED_PLAN_ONLY' and preview['compiled_variant_count'] == 0
    compiled = lower_stateful_choices(tmp_path / 'chosen', tmp_path / 'compiled')
    assert compiled['status'] == 'COMPILED_UNMEASURED' and compiled['compiled_variant_count'] == 1
    assert not compiled['deployment_authorized'] and not compiled['selected_for_use']
    assert compiled['selected_id'] == chosen['selected_id']
    assert compiled['recipe_id'] == compiled['artifact']['recipe_id']
    assert compiled['stateful_abi_version'] == compiled['operator_abi_version'] == 1
    from scar.codegen.stateful import load
    native = load(compiled['native_directory'])
    state = np.arange(15, dtype=np.float64).reshape(3, 5) / 8
    expected = state.copy(); expected += expected * 0.25
    native.run({'a': state}, {'g': 0.25})
    assert np.array_equal(state.view(np.uint64), expected.view(np.uint64))
    bad = deepcopy(chosen); bad['selected_id'] = 'NO_OP'
    bad_path = tmp_path / 'bad-choice.json'; bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='complete replay'):
        lower_stateful_choices(bad_path, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
