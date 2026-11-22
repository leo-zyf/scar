"""A macro recipe pins its components and preserves task readiness."""
from copy import deepcopy

import pytest

from scar.ir import execution_recipe
from scar.ir.execution_recipe import (artifact_reference, make_recipe, open_artifact, recipe_id,
                                      read_recipe, validate_recipe, write_recipe)
from scar.optimization.schedule import LogicalValue, Materialization, ResourceSchedule, Task, TaskKind


def test_recipe_identity_causality_source_pins_and_detached_inputs(tmp_path):
    component = tmp_path/'operation.json'; component.write_text('{"schema":"test.operation","kind":"increment"}')
    source, content = open_artifact(component, 'test.operation')
    assert source == artifact_reference(component, 'test.operation')
    assert content == dict(schema='test.operation', kind='increment')
    boundary = dict(id='original-boundary', parameters=['x'])
    graph = ResourceSchedule('region',
        (LogicalValue('input'), LogicalValue('output', 'compute', True)),
        (Materialization('input', 'input', 'host', external_ready=True, ownership='EXTERNAL'),
         Materialization('output', 'output', 'host', ready_task='compute', allocated_by='allocate')),
        (Task('allocate', TaskKind.ALLOCATION, writes=('output',)),
         Task('compute', TaskKind.COMPUTE, reads=('input',), writes=('output',), depends_on=('allocate',)),
         Task('export', TaskKind.EXPORT, reads=('output',), depends_on=('compute',))))
    recipe = make_recipe('test-profile', semantic_boundary=boundary,
        control=dict(kind='SHARED_REGION'), task_graph=graph, resources=dict(cpu_ids=[0, 1]),
        implementation=dict(family='TEST_DECLARATION_NOT_EXECUTABLE', backend_inputs=dict(operation='operation')),
        artifacts=dict(operation=source), obligations=[dict(id='input', status='RUNTIME_GUARD', detail='exact input')])
    boundary['parameters'].append('caller-mutation')
    assert recipe['semantic_boundary']['parameters'] == ['x']
    ref = write_recipe(recipe, tmp_path/'recipe.json')
    assert ref['schema'] == 'scar.execution-recipe' and read_recipe(ref['uri']) == recipe
    changed = deepcopy(recipe); changed['resources']['cpu_ids'] = [0]
    with pytest.raises(ValueError, match='identity'): validate_recipe(changed)
    changed['recipe_id'] = recipe_id(changed)
    assert validate_recipe(changed)['semantic_boundary'] == recipe['semantic_boundary']
    assert changed['recipe_id'] != recipe['recipe_id']
    early = deepcopy(recipe)
    early['task_graph']['materializations'][1]['ready_task'] = 'allocate'
    early['recipe_id'] = recipe_id(early)
    with pytest.raises(ValueError, match='allocation does not establish'): validate_recipe(early)
    component.write_text('{"schema":"test.operation","kind":"different"}')
    with pytest.raises(ValueError, match='byte identity'): validate_recipe(recipe)


def test_open_artifact_rejects_replacement_during_the_read(tmp_path, monkeypatch):
    original = tmp_path / 'operation.json'
    replacement = tmp_path / 'replacement.json'
    original.write_text('{"schema":"test.operation","value":1}')
    replacement.write_text('{"schema":"test.operation","value":2}')
    read = execution_recipe._bounded_bytes

    def replace_after_read(stream):
        raw = read(stream)
        replacement.replace(original)
        return raw

    monkeypatch.setattr(execution_recipe, '_bounded_bytes', replace_after_read)
    with pytest.raises(ValueError, match='changed while opening'):
        open_artifact(original, 'test.operation')
