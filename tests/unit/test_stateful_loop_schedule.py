"""The loop model preserves mutable content and shares one step at any count."""
from scar.compiler.model import Expr, SourceRef
from scar.compiler.stateful_model import (
    REQUIRED_GUARDS, StatefulAction, StatefulInput, StatefulRegion,
)
from scar.optimization.stateful_loop_schedule import _loop_model


def _step():
    return StatefulRegion('test:stateful-step',
        SourceRef('/tmp/stateful_loop_schedule.py', '0' * 64, 'step', 1, 0,
                  'pycode-v2:' + '1' * 64),
        ('state', 'fixed', 'scale'),
        (StatefulInput('written', ('state',), 'array'),
         StatefulInput('readonly', ('fixed',), 'array'),
         StatefulInput('factor', ('scale',), 'scalar')),
        (Expr('a', 'input', attrs=dict(name='written', kind='array', epoch=0)),
         Expr('b', 'input', attrs=dict(name='readonly', kind='array', epoch=0)),
         Expr('s', 'input', attrs=dict(name='factor', kind='scalar', epoch=None)),
         Expr('sum', 'add', ('a', 'b')),
         Expr('rhs', 'mul', ('sum', 's'))),
        (StatefulAction('eval:sum', 'evaluate', 'sum'),
         StatefulAction('eval:rhs', 'evaluate', 'rhs'),
         StatefulAction('write', 'write', 'rhs', 'written', None, 0, 1, 'assign')),
        guards=REQUIRED_GUARDS)


def test_written_content_is_carried_not_loop_invariant():
    model = _loop_model(_step(), dict(state='initial', fixed='fixed', scale='scale'), 7)
    rows = {row['input_id']: row for row in model['inputs']}
    assert rows['written']['role'] == 'ORDERED_ARRAY_CONTENT_CARRY'
    assert rows['written']['invariant_content_under_guards'] is False
    assert rows['written']['backedge']['previous_exit']['local_epoch'] == 1
    assert rows['written']['backedge']['next_entry']['local_epoch'] == 0
    assert rows['readonly']['role'] == 'GUARDED_LOOP_LIVE_IN_ARRAY'
    assert rows['factor']['role'] == 'GUARDED_LOOP_LIVE_IN_SCALAR'
    assert all(row['computation_hoisted'] is False for row in model['inputs'])
    assert model['runtime_logical_value_or_storage_identity_inferred'] is False


def test_large_trip_count_has_one_shared_step_and_constant_size_structure():
    mapping = dict(state='initial', fixed='fixed', scale='scale')
    small = _loop_model(_step(), mapping, 1)
    large = _loop_model(_step(), mapping, 10**12)
    assert large['step_instances_stored'] == 1
    assert large['iterations_unrolled'] is False
    assert len(small['inputs']) == len(large['inputs']) == 3
    assert sum(len(row.get('ordered_writes', ())) for row in small['inputs']) == 1
    assert sum(len(row.get('ordered_writes', ())) for row in large['inputs']) == 1
    assert large['trip_count'] == 10**12
