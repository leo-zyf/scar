"""One recipe binds semantic loop, hardware, readiness, cost and lowering inputs."""
from copy import deepcopy
from hashlib import sha256
import json
from types import CodeType

import pytest

from test_cpu_hardware_placement import hardware
from test_loop_layout import setup_layout
from scar.codegen import loop as backend
from scar.compiler.loop_model import ArrayLoopPlan
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.pipeline_frontend import CalleeGuard, RegionCall, compose_pipeline
from scar.ir.execution_recipe import artifact_reference, recipe_id
from scar.optimization import optimize_region, bind_cpu_placement, propose_cpu_placements
from scar.optimization.loop_layout import plan_loop_layout
from scar.optimization.loop_recipe import (make_loop_recipe, verify_loop_recipe,
    unpack_loop_recipe, estimate_loop_recipe)
from scar.optimization.loop_strategy import plan_loop_strategy, loop_execution_plan
from scar.optimization.schedule import schedule_from_dict
from scar.trace.code_snapshot import code_key


def setup_recipe(hardware, tmp_path, layout=True):
    """Source-backed metadata fixture; no source function or native kernel runs."""
    def source(name, text):
        path = tmp_path/(name + '.py'); path.write_text(text)
        module = compile(text, str(path), 'exec')
        code = next(value for value in module.co_consts if isinstance(value, CodeType) and value.co_name == name)
        return SourceRef(str(path), sha256(path.read_bytes()).hexdigest(), name, code.co_firstlineno, 0, code_key(code))
    leaf_source = source('leaf', 'def leaf(a):\n    return -(-a)\n')
    body_source = source('advance', 'def advance(x):\n    return leaf(leaf(x))\n')
    loop_source = source('evolve', 'def evolve(steps, state):\n    for _ in range(steps):\n        state = advance(state)\n    return state\n')
    leaf = optimize_region(ArrayRegion('leaf', leaf_source, ('a',), (
        Expr('a','input',attrs={'name':'a'}), Expr('negative','neg',('a',)), Expr('result','neg',('negative',))), 'result'))
    pipeline = compose_pipeline(body_source, ('x',), (
        RegionCall('first',leaf,('arg:0',),'middle',CalleeGuard(leaf_source,('leaf',))),
        RegionCall('second',leaf,('middle',),'final',CalleeGuard(leaf_source,('leaf',)))), 'final')
    loop = ArrayLoopPlan('loop:recipe',loop_source,('steps','state'),'state','steps',pipeline,CalleeGuard(body_source,('advance',)))
    strategy = plan_loop_strategy(loop,input_shapes={'state':[2,9]},trip_count=400,memory_budget_bytes=144)
    plan = loop_execution_plan(loop,strategy)
    binding = next(item for item in propose_cpu_placements(plan,hardware,max_threads=2,memory_budget_bytes=144)
                   if len(item.cpu_ids)==2)
    plan = bind_cpu_placement(plan,binding)
    layout_doc = None
    if layout:
        _,_,_,primitive = setup_layout(hardware,tmp_path)
        # Real compiler identity is required for a buildable recipe. Timing
        # numbers remain explicit test declarations; this does not compile/run.
        compiler = backend.describe_capability(plan)['compiler']
        primitive['context']['compiler'].update(argv=compiler['argv'], identity=compiler['identity'])
        facts_path = tmp_path/'layout-facts-9.json'
        facts = json.loads(facts_path.read_bytes()); facts['context']['compiler'] = primitive['context']['compiler']
        facts_path.write_text(json.dumps(facts)); raw = facts_path.read_bytes()
        def refresh(value):
            if isinstance(value, dict):
                if 'artifact' in value:
                    value['artifact'].update(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
                for item in value.values(): refresh(item)
            elif isinstance(value, list):
                for item in value: refresh(item)
        refresh(primitive)
        layout_doc = plan_loop_layout(loop,strategy,plan,hardware_profile=primitive)
    recipe = make_loop_recipe(loop,strategy,plan,layout=layout_doc,out=tmp_path/'recipe',
        source_artifacts=[artifact_reference(leaf_source.path,'scar.python-source')])
    return recipe,loop,strategy,plan,layout_doc


def test_common_recipe_keeps_ordered_time_one_body_readiness_and_unknown_cost(hardware,tmp_path,monkeypatch):
    recipe,loop,strategy,plan,layout = setup_recipe(hardware,tmp_path)
    path = tmp_path/'recipe/recipe.json'
    assert verify_loop_recipe(path) == recipe
    assert unpack_loop_recipe(recipe) == (loop,strategy,plan,layout)
    schedule = schedule_from_dict(recipe['task_graph'])
    assert len(schedule.tasks)==3 and recipe['control']['temporal']['count']==400
    assert recipe['control']['temporal']['shared_body_instances']==1
    assert recipe['control']['temporal']['unrolled'] is False
    allocation,compute,export = schedule.tasks
    assert allocation.attrs['establishes_value_ready'] is False
    output = schedule.materializations[-1]
    assert output.allocated_by==allocation.id and output.ready_task==compute.id
    assert compute.depends_on==(allocation.id,) and export.depends_on==(compute.id,)
    assert recipe['semantic_boundary']['Q']['implementation_scope']=='CURRENT_STRICT_CPU_CONTRACT_NOT_A_CROSS_DEVICE_EQUIVALENCE_PROOF'
    resources = recipe['resources']
    assert len(resources['inventory']['all_resources'])==5
    assert resources['selected_cpu']['cpu_threads']==2 and resources['unselected_resources_retained']
    assert resources['host_allocation']['memory_policy']=='UNBOUND'
    assert not resources['host_allocation']['physical_numa_residency_proven']
    assert resources['cache_hierarchy']['allocation_budget_bytes'] is None
    before = deepcopy(recipe)
    estimate = estimate_loop_recipe(path)
    assert estimate['recipe_id']==recipe['recipe_id'] and recipe==before
    assert estimate['payload']['incremental_payload_bytes']==144
    assert estimate['timing']['status']=='UNKNOWN' and estimate['timing']['duration_ticks'] is None
    assert all(row['duration_ticks'] is None for row in estimate['timing']['per_task'].values())
    changed = deepcopy(recipe); changed['control']['temporal']['count']=401
    changed['recipe_id']=recipe_id(changed)
    with pytest.raises(ValueError,match='complete source/control/resource replay'): verify_loop_recipe(changed)
    capability = recipe['implementation']['backend_capability_record']
    assert capability['schema']=='scar.loop-backend-capability'
    changed_capability = deepcopy(capability); changed_capability['compiler']['identity'] += ' changed'
    with monkeypatch.context() as patched:
        patched.setattr(backend, 'describe_capability', lambda *args, **kwargs: changed_capability)
        with pytest.raises(ValueError,match='complete source/control/resource replay'): verify_loop_recipe(recipe)
