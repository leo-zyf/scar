"""Dynamic context substitution keeps canonical stages and serial terms."""
from copy import deepcopy
from scar.codegen.cpu_batch_boundary_resource import _sources, _specialized_demand, _HERE


def test_dynamic_context_only_and_serial_copy_is_not_worker_scaled():
    parameters=dict(arrays=1,scalars=1,providers=0,array_shapes=[[2,3]],workspace_slot_capacities=[48])
    memory=dict(workspace_payload_bytes=48,external_array_bytes=48,total_region_array_payload_bytes=96,memory_budget_bytes=4096)
    resource,boundary,_=_sources([14,15],56,True,parameters,memory)
    assert 'std::vector<int> planned_cpu_ids = {14, 15};' in resource
    assert 'size_t minimum_cpu_slots = 56;' in resource and 'constexpr size_t minimum_cpu_slots' not in resource
    assert 'minimum_cpu_slots=mask_slots;' in resource and 'minimum_cpu_slots=mask_slots;' in boundary
    assert 'scar_batch_set_cpu_ids_v1' in resource and 'scar_batch_set_context_v1' in boundary
    assert 'const int workers=static_cast<int>(planned_cpu_ids.size());' in boundary
    original=(_HERE/'cpu_resource_probe_v1.cpp.in').read_text()
    original_stage=original[original.index('template<int Op> double arithmetic'):original.index('\n}\nextern "C"')]
    assert original_stage in resource
    original_boundary=(_HERE/'stateful_boundary_probe_v1.cpp.in').read_text()
    scalar_body=original_boundary[original_boundary.index('    if (!operations'):]
    assert scalar_body in boundary
    demand=dict(context=dict(candidate_id='original',cpu_binding={}),resource_context=dict(worker_cpu_ids=[0],executor_id='old'),
        terms=[dict(component='ROW_COPY_SERIAL',parameters=dict(workers=1),key='before'),
            dict(component='ROW_COPY_STAGE',parameters=dict(workers=1),key='before')])
    snapshot=deepcopy(demand)
    row=dict(id='candidate',worker_count=2,binding=dict(cpu_ids=[14,15],executor_id='executor',evidence=[]))
    actual=_specialized_demand(demand,row)
    assert actual['terms'][0]['parameters']['workers']==1
    assert actual['terms'][1]['parameters']['workers']==2
    assert actual['context']['cpu_binding']==dict(cpu_ids=[14,15],executor_id='executor')
    assert actual['resource_context']['worker_cpu_ids']==[14,15]
    assert demand==snapshot
