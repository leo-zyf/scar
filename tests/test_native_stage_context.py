"""One projection regression for real dependency ports and physical slot reuse."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from scar.ir.versioned_schedule import (
    VersionedLogicalValue as Value, VersionedMaterialization as Material,
    VersionedResourceSchedule as Schedule, VersionedTask as Task,
    VersionedTaskKind as K, StorageRegion, StorageSlot,
)
from scar.optimization.native_stage_context import _project


class NativeStageContextTest(unittest.TestCase):
    def test_reused_slot_is_not_a_live_value_or_source_epoch_and_ready_scalar_is_real(self):
        names = ['alpha', 'beta', 'gamma', 'delta']
        tasks = [Task('allocate:reuse', K.ALLOCATE), Task('allocate:keep', K.ALLOCATE),
            Task('alpha', K.COMPUTE, writes=('mat:A',)),
            Task('beta', K.COMPUTE, reads=('mat:A',), writes=('mat:B',), depends_on=('alpha',)),
            Task('gamma', K.COMPUTE, depends_on=('beta',)),
            Task('delta', K.COMPUTE, reads=('mat:B',), writes=('mat:C',), depends_on=('gamma', 'beta')),
            Task('export', K.EXPORT, reads=('mat:C',), depends_on=('delta',))]
        schedule = Schedule('fixture-region',
            tuple(Value('value:'+name, producer=producer, attrs=dict(immutable=True))
                for name,producer in [('A','alpha'), ('B','beta'), ('C','delta')]),
            (Material('mat:A', 'value:A', StorageRegion('reuse', 0, 32), 0, ready_task='alpha'),
             Material('mat:B', 'value:B', StorageRegion('keep', 0, 32), 0, ready_task='beta'),
             Material('mat:C', 'value:C', StorageRegion('reuse', 0, 32), 1, ready_task='delta')),
            (StorageSlot('reuse', 'host', 32), StorageSlot('keep', 'host', 32)), tuple(tasks))
        stages = [dict(id=name, index=index, kind='SCALAR' if name=='gamma' else 'COMPUTE', epoch=0,
            target=None, shape=None if name=='gamma' else [2,2], normalized_bounds=None,
            root_expr='ready-scalar' if name=='gamma' else name,
            scalar_inputs=['ready-scalar'] if name=='delta' else [],
            expr_ids=['runtime-input', 'ready-scalar'] if name=='gamma' else [],
            array_reads=[], source_action_ids=[], depends_on=list(tasks[index+2].depends_on))
            for index,name in enumerate(names)]
        plan = dict(stages=stages)
        region = SimpleNamespace(inputs=[SimpleNamespace(id='scale', path=('self','scale'))],
            nodes=[SimpleNamespace(id='runtime-input', op='input', attrs=dict(name='scale', kind='scalar')),
                SimpleNamespace(id='ready-scalar', op='add', attrs={})])
        view = dict(region={'fixture':'region'}, plan={'fixture':'plan'}, identity={'fixture':'native'},
            stages=[dict(index=i, body_sha256='fixture', source_line_spans=[], mapping_status='FIXTURE')
                for i in range(4)])
        observed = dict(groups=[dict(index=0, stage_indices=[2,3], begin_cut=0, end_cut=1,
            entry_scalar_ready=[], exit_scalar_ready=['ready-scalar'])],
            observer_identity='fixture-observer', timed_scope='FIXTURE_GROUP', excluded_from_group_clock=[])
        report = _project(schedule, plan, region, observed, view)
        group = report['groups'][0]
        self.assertEqual(group['live_in_payloads'], ['mat:B'])
        self.assertEqual(group['required_ready_scalar_live_in'], [])
        previous = next(row for row in group['entry_slot_contents'] if row['slot_id']=='reuse')
        self.assertEqual(previous['current_materialization'], 'mat:A')
        self.assertFalse(previous['live_content_required'])
        self.assertEqual(report['materializations']['mat:C']['storage_version'], 1)
        self.assertIsNone(report['materializations']['mat:C']['local_content_epoch'])
        self.assertNotEqual(report['materializations']['mat:A']['logical_value'],
            report['materializations']['mat:C']['logical_value'])
        hazards = [row for row in report['dependency_edges']
            if row['kind']=='PHYSICAL_SLOT_OVERWRITE_ORDER_NOT_VALUE_FLOW']
        self.assertEqual(len(hazards), 1)
        self.assertFalse(hazards[0]['logical_values_equal'])
        self.assertTrue(any(row['kind']=='READY_SCALAR_DEPENDENCY' and row['before']=='gamma'
            and row['after']=='delta' for row in report['dependency_edges']))
        self.assertFalse(group['potential_first_accesses'])  # reused bytes were already written by the prefix
        self.assertEqual(group['physical_cache_state'], 'UNKNOWN')

        selected = deepcopy(observed)
        selected['groups'][0].update(stage_indices=[3], entry_scalar_ready=['ready-scalar'])
        later = _project(schedule, plan, region, selected, view)['groups'][0]
        self.assertEqual(later['required_ready_scalar_live_in'], ['ready-scalar'])
        self.assertEqual(later['live_in_payloads'], ['mat:B'])


if __name__ == '__main__':
    unittest.main()
