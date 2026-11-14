"""Forward estimates use static demand and explicit retained model evidence."""

from copy import deepcopy
from hashlib import sha256
import json

from test_cpu_hardware_placement import hardware, plan
from scar.compiler.model import PROFILE
from scar.codegen.cpp import runtime_implementation_sha256
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.optimization import bind_cpu_placement, propose_cpu_placements
from scar.optimization.cost_model import cost_feature_units, features_for_demand, predict_region_costs
from scar.optimization.resource_solver import solve_resource_schedule
from scar.optimization.schedule import build_region_schedule


def setup_model(hardware, tmp_path):
    original = plan()
    binding = propose_cpu_placements(original, hardware, max_threads=2, memory_budget_bytes=4096)[1]
    bound = bind_cpu_placement(original, binding)
    schedule = build_region_schedule(bound, hardware)
    target = {'target_sha256': binding.target_sha256, 'executor_id': binding.executor_id,
              'cpu_ids': list(binding.cpu_ids)}
    compiler = {'identity': 'fixture compiler', 'options': ['-fixture'], 'argv': ['fixture-cxx', '-extra-prefix-flag']}
    implementation = runtime_implementation_sha256()
    source = tmp_path / 'calibration.json'
    fits = {
        'ALLOCATION': {'intercept': {'value': 2, 'unit': 'ticks'},
                       'coefficients': {'output_bytes': {'value': 0.125, 'unit': 'ticks/byte'}}},
        'COMPUTE': {'intercept': {'value': 3, 'unit': 'ticks'}, 'coefficients': {
            'output_elements': {'value': 0.1, 'unit': 'ticks/output_element'},
            'logical_payload_bytes': {'value': 0.01, 'unit': 'ticks/logical_byte'},
            'arithmetic.add': {'value': 0.25, 'unit': 'ticks/operation'},
            'arithmetic.mul': {'value': 0.5, 'unit': 'ticks/operation'},
            'setup.input_bindings': {'value': 1, 'unit': 'ticks/binding'}}},
        'EXPORT': {'intercept': {'value': 1, 'unit': 'ticks'}, 'coefficients': {}}}
    source.write_text(json.dumps({
        'schema': 'scar.primitive-calibration-test', 'fits': fits,
        'tick_seconds': {'value': 1e-9, 'unit': 'seconds/tick'},
        'zero_fit': {'value': 0, 'unit': 'ticks/operation'}, 'observations': [10, 20, 30],
        'target': target, 'runtime_implementation_sha256': implementation,
        'execution_boundary': 'TIMED_SYNCHRONOUS_CPU_PROFILE', 'compiler': compiler,
        'legacy_context': {'target': target, 'compiler': compiler},
        'regime': {'basis': 'SYNTHETIC_UNIT_TEST_COEFFICIENTS_NOT_HARDWARE_MEASUREMENTS'}}))
    raw = source.read_bytes()
    artifact = ProviderArtifact(str(source), sha256(raw).hexdigest(), len(raw), 'scar.primitive-calibration-test')

    def fact(locator):
        return encode(ProviderFactReference(artifact, locator))

    def parameter(record, locator):
        return {**record, 'evidence': [fact(locator), fact('#/observations')]}

    models = {name: {'intercept': parameter(row['intercept'], f'#/fits/{name}/intercept/value'),
                     'coefficients': {key: parameter(record, f'#/fits/{name}/coefficients/{key}/value')
                                      for key, record in row['coefficients'].items()}}
              for name, row in fits.items()}

    profile = {
        'schema': 'scar.region-cost-profile', 'schema_version': 1,
        'target': deepcopy(target),
        'backend': {'name': 'cpp', 'execution_profile': PROFILE, 'operator_abi_version': 1,
                    'runtime_implementation_sha256': implementation,
                    'guard_mode': 'TIMED_SYNCHRONOUS_CPU_PROFILE', 'context_evidence': [fact('#')],
                    'compiler': {**deepcopy(compiler), 'evidence': [fact('#/compiler')]}},
        'applicability': {'layout': 'C_CONTIGUOUS_FLOAT64_2D', 'working_set': {
            'regime': 'DECLARED_TEST_ONLY', 'output_elements_min': 1, 'output_elements_max': 100,
            'logical_payload_bytes_min': 1, 'logical_payload_bytes_max': 10000, 'evidence': [fact('#/regime')]}},
        'model': {'kind': 'EFFECTIVE_AFFINE', 'composition': 'SUM_EFFECTIVE_TERMS',
                  'tick_seconds': parameter({'value': 1e-9, 'unit': 'seconds/tick'}, '#/tick_seconds/value'),
                  'task_models': models}}
    return bound, schedule, profile, source


def test_static_work_becomes_conditional_ticks_without_worker_scaling(hardware, tmp_path):
    bound, schedule, profile, _ = setup_model(hardware, tmp_path)
    result = predict_region_costs(bound, schedule, profile, input_shapes=(4, 5))
    assert result['prediction_available'] and not result['unresolved']
    assert [result['task_costs'][task.id]['duration_ticks'] for task in schedule.tasks] == [22, 25, 1]
    terms = {term['feature']: term for term in result['task_costs']['compute:region']['terms']}
    assert terms['logical_payload_bytes']['count'] == 320
    assert terms['arithmetic.mul']['count'] == 20
    assert result['context']['compiler_match'] == 'MANDATORY_GUARD_REQUIRED'
    assert result['context']['calibration_context_match'] == 'MATCHED_RETAINED_RECORD'
    assert result['context']['input_layout']['required'] == 'C_CONTIGUOUS'
    assert result['context']['cachewarm_regime']['status'] == 'EXTERNAL_ASSUMPTION'
    assert result['context']['cachewarm_regime']['actual_cache_state_verified'] is False
    assert result['unknown']['actual_dram_bytes'] is None
    assert not result['target_program_executed'] and result['source_artifacts']
    assert features_for_demand(result['work_demand'], 20)['COMPUTE']['arithmetic.mul'][0] == 20
    units = cost_feature_units()
    units['COMPUTE']['arithmetic.mul'] = 'changed'
    assert cost_feature_units()['COMPUTE']['arithmetic.mul'] == 'ticks/operation'
    solved = solve_resource_schedule(schedule, **result['solver_model'])
    assert solved['solution']['makespan_ticks'] == 48


def test_missing_parameters_and_unsourced_zero_are_not_guessed(hardware, tmp_path):
    bound, schedule, profile, _ = setup_model(hardware, tmp_path)
    coefficients = profile['model']['task_models']['COMPUTE']['coefficients']
    original = coefficients.pop('arithmetic.mul')
    missing = predict_region_costs(bound, schedule, profile, input_shapes=(4, 5))
    assert not missing['prediction_available'] and 'compute:region' not in missing['task_costs']
    assert any(issue['field'].endswith('coefficients.arithmetic.mul') for issue in missing['cost_unresolved'])
    coefficients['arithmetic.mul'] = {**original, 'value': 0}
    coefficients['arithmetic.mul']['evidence'][0]['locator'] = '#/zero_fit/value'
    zero = predict_region_costs(bound, schedule, profile, input_shapes=(4, 5))
    assert any(issue['field'].endswith('.zero_basis') for issue in zero['cost_unresolved'])
    coefficients['arithmetic.mul']['zero_basis'] = 'NONNEGATIVE_FIT_ACTIVE_CONSTRAINT_TEST_ONLY'
    assert predict_region_costs(bound, schedule, profile, input_shapes=(4, 5))['prediction_available']


def test_context_range_and_evidence_receipts_fail_closed(hardware, tmp_path):
    bound, schedule, profile, source = setup_model(hardware, tmp_path)
    guard_wrong = deepcopy(profile)
    guard_wrong['backend']['guard_mode'] = 'UNTIMED'
    wrong = predict_region_costs(bound, schedule, guard_wrong, input_shapes=(4, 5))
    assert not wrong['prediction_available']
    assert any(issue['field'] == 'profile.backend.guard_mode' for issue in wrong['cost_unresolved'])
    context_missing = deepcopy(profile)
    context_missing['backend'].pop('context_evidence')
    missing_context = predict_region_costs(bound, schedule, context_missing, input_shapes=(4, 5))
    assert not missing_context['prediction_available']
    assert any(issue['field'] == 'profile.backend.context_evidence' for issue in missing_context['cost_unresolved'])
    legacy = deepcopy(profile)
    legacy['backend']['context_evidence'][0]['locator'] = '#/legacy_context'
    assert not predict_region_costs(bound, schedule, legacy, input_shapes=(4, 5))['prediction_available']
    argv_mismatch = predict_region_costs(bound, schedule, profile, input_shapes=(4, 5), backend_context={
        'backend': 'cpp', 'compiler_version': 'fixture compiler', 'compiler_options': ['-fixture'],
        'compiler': ['fixture-cxx', '-other-extra-prefix-flag'], 'runtime_implementation_sha256': runtime_implementation_sha256()})
    assert not argv_mismatch['prediction_available']
    assert any(issue['field'] == 'backend_context.compiler' for issue in argv_mismatch['cost_unresolved'])
    coefficient = profile['model']['task_models']['COMPUTE']['coefficients']['arithmetic.mul']
    original_locator = coefficient['evidence'][0]['locator']
    coefficient['evidence'][0]['locator'] = '#/fits/COMPUTE/coefficients/arithmetic.add/value'
    wrong_value = predict_region_costs(bound, schedule, profile, input_shapes=(4, 5))
    assert not wrong_value['prediction_available']
    assert any('fit result value differs' in issue['reason'] for issue in wrong_value['cost_unresolved'])
    coefficient['evidence'][0]['locator'] = original_locator
    profile['target']['cpu_ids'] = [7]
    profile['applicability']['working_set']['output_elements_max'] = 10
    result = predict_region_costs(bound, schedule, profile, input_shapes=(4, 5),
                                  backend_context={'backend': 'cpp', 'compiler_version': 'other', 'compiler_options': []})
    fields = {issue['field'] for issue in result['cost_unresolved']}
    assert {'profile.target.cpu_ids', 'profile.applicability.working_set.output_elements',
            'backend_context.compiler_version', 'backend_context.runtime_implementation_sha256'}.issubset(fields)
    assert not result['task_costs']
    runtime_missing = deepcopy(profile)
    runtime_missing['backend'].pop('runtime_implementation_sha256')
    missing = predict_region_costs(bound, schedule, runtime_missing, input_shapes=(4, 5))
    assert any(issue['field'] == 'profile.backend.runtime_implementation_sha256' for issue in missing['cost_unresolved'])
    source.write_text('{}')
    tampered = predict_region_costs(bound, schedule, profile, input_shapes=(4, 5))
    assert any('SHA/size' in issue['reason'] for issue in tampered['cost_unresolved'])
