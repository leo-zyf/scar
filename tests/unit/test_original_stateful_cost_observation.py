"""Failed/mismatched original timing samples cannot become NO_OP costs."""
from copy import deepcopy

import pytest

from scar.compiler.observe_original_stateful_cost import _validate_sample,_SAMPLE,_BASIS


def test_original_sample_requires_exact_frozen_scope_and_positive_original_return():
    context=dict(process_affinity_cpu_ids=[0,1],native_thread_environment={'OMP_NUM_THREADS':None,'GOMP_CPU_AFFINITY':'0,1'})
    spec=dict(phase='MEASUREMENT',index=0,worker_nonce='owned-child',
        frozen_design={'sha256':'a'*64},entry_context=context)
    sample=dict(schema=_SAMPLE,schema_version=1,**deepcopy(spec),basis=_BASIS,
        status='OBSERVED_ORIGINAL_RETURN',recorded_calls=1,process_id=123,duration_ns=456,
        function_internals_instrumented=False,global_trace_or_profile_installed=False,
        tensor_values_recorded=False,target_variants_executed=False,target_variants_ranked=False,
        native_calls=0,target_workload_executed=True,observed_caller_affinity_cpu_ids=[0,1],
        observed_native_thread_environment=context['native_thread_environment'],
        actual_full_thread_environment=context['native_thread_environment'])
    assert _validate_sample(sample,spec)==456
    for changed in (dict(duration_ns=0),dict(duration_ns=None),dict(duration_ns=True),
            dict(status='ORIGINAL_RAISED'),dict(recorded_calls=0),dict(native_calls=1),dict(native_calls=False),
            dict(target_variants_ranked=None),dict(function_internals_instrumented=True),
            dict(worker_nonce='another-child'),dict(schema_version=True),
            dict(actual_full_thread_environment={'OMP_NUM_THREADS':None})):
        with pytest.raises(ValueError):_validate_sample({**sample,**changed},spec)
    del sample['target_variants_executed']
    with pytest.raises(ValueError):_validate_sample(sample,spec)
