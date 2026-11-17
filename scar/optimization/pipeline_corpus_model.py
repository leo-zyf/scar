"""Fit declared effective costs to clean, complete canonical caller timings.

This module only consumes retained metadata.  Canonical model transfer is an
explicit assumption, not source admission or observation of target components.
"""
from copy import deepcopy
import math
from pathlib import Path
import statistics

from scar.ir.execution_recipe import artifact_reference
from scar.ir.record_codec import encode
from scar.optimization.deployment_cost import _digest, _fact, _number
from scar.optimization.macro_pass import COST_SCOPE
from scar.optimization.pipeline_forward_cost import (
    _document, _materials, coefficient_profile_skeleton,
)


SCHEMA = 'scar.pipeline-corpus-cost-fit'
OBSERVATIONS_SCHEMA = 'scar.pipeline-corpus-observations'
_SIDES = ('original', 'fused')
_OPS = ('add', 'sub', 'mul', 'div', 'neg')
_ROLES = ('TRAIN', 'CANONICAL_HOLDOUT', 'TARGET_HOLDOUT')
_WARM = ('original_one_time', 'native_load', 'native_setup')
_MEASUREMENT = dict(scope=COST_SCOPE, boundary='WARM_SESSION', reuse_count=1,
    original_entry='ORIGINAL_PYTHON_NO_OP', fused_entry='FULL_PROTECTED_FUSION_CALL',
    compilation_included=False, native_load_included=False,
    internal_instrumentation=False, final_output_release_included=False)
_QUALIFICATION = dict(topology_fixed=True, input_arity_fixed=True,
    unique_callee_count_fixed=True, clean_protected_call=True)
_FIXED = dict(original=('python_region_calls', 'python_leaf_calls', 'output_elements'),
    fused=('native_region_calls', 'output_elements', 'fused_array_allocations',
           'fused_output_bytes', 'fused_input_bindings'))
_GROUPED = 'EFFECTIVE_BASIS_GROUPING_NO_SEPARATE_CHARGE_NOT_A_PHYSICAL_ZERO'
_CONSTRAINED = 'NONNEGATIVE_FIT_ACTIVE_CONSTRAINT_NOT_A_PHYSICAL_ZERO'
_ASSUMPTIONS = [
    'COEFFICIENTS_ARE_DECLARED_EFFECTIVE_MODEL_PARAMETERS_NOT_OBSERVED_COMPONENTS',
    'ONLY_TRAIN_CASES_FIT_PARAMETERS_ALL_HOLDOUTS_CHECK_FROZEN_PARAMETERS',
    'CANONICAL_TO_TARGET_TRANSFER_ASSUMES_FIXED_SHAPE_GUARD_TOPOLOGY_INPUT_ARITY_AND_CALLEE_COUNT',
    'FIXED_STRUCTURE_IS_DRIVER_METADATA_NOT_INDEPENDENT_SOURCE_F4_ADMISSION',
    'ENTRY_AND_ARITHMETIC_TERMS_JOINTLY_MODEL_THE_COMPLETE_CALL_INCLUDING_GUARDS_ALLOCATION_AND_RETURN',
    'NONBASIS_FEATURE_SERVICES_ARE_ABSORBED_BY_EFFECTIVE_TERMS_NOT_PHYSICALLY_FREE',
    'CLEAN_COMPLETE_CALL_MEDIANS_ARE_FIT_DIRECTLY_WITHOUT_SCOPE_DIFFERENCES',
    'WARM_REPEATED_ARRAYS_DO_NOT_PROVE_CACHE_NUMA_RESIDENCY_OR_LATENCY_BOUNDS',
    'HOLDOUT_ERROR_IS_SAMPLE_VALIDATION_NOT_A_GENERALIZATION_OR_PHYSICAL_BOUND',
]


def _parameter(value, unit, zero_basis=_CONSTRAINED):
    result = dict(value=value, unit=unit, basis='DECLARED_MODEL')
    if value == 0:
        result['zero_basis'] = zero_basis
    return result


def fit_pipeline_corpus(region_document, blueprint, demand, observations, *,
                        maximum_holdout_relative_error=.30):
    """Jointly fit each side's raw full-call medians; never execute a case."""
    _, _, demand, context = _materials(region_document, blueprint, demand)
    observations = _document(observations)
    limit = _number(maximum_holdout_relative_error)
    if limit is None or limit >= 1:
        raise ValueError('holdout relative error threshold must be in [0, 1)')
    frontier = []

    def issue(field, reason):
        frontier.append(dict(field=field, reason=reason))

    if (observations.get('schema'), observations.get('schema_version')) != (OBSERVATIONS_SCHEMA, 1) or type(observations.get('schema_version')) is not int:
        issue('observations.schema', 'versioned clean corpus observations required')
    if _digest(observations.get('context')) != _digest(context):
        issue('observations.context', 'complete target protected-call context differs')
    if observations.get('demand_sha256') != _digest(demand):
        issue('observations.demand_sha256', 'exact target demand identity differs')
    for group, required in (('measurement', _MEASUREMENT), ('qualification', _QUALIFICATION)):
        record = observations.get(group)
        if not isinstance(record, dict) or any(
                type(record.get(key)) is not type(value) or record.get(key) != value
                for key, value in required.items()):
            issue('observations.' + group, 'exact clean warm full-entry contract required')

    active = []
    for op in _OPS:
        name = 'arithmetic.' + op
        counts = [demand[side]['features'].get(name) for side in _SIDES]
        if any(type(value) is not int or value < 0 for value in counts):
            issue('demand.' + name, 'target arithmetic demand must be exact and nonnegative')
        elif any(counts):
            active.append(name)
    basis = ['intercept', *active]
    cases = observations.get('cases')
    if not isinstance(cases, list):
        issue('observations.cases', 'explicit corpus cases required'); cases = []
    valid, ids = [], set()
    for index, case in enumerate(cases):
        field = 'observations.cases.' + str(index)
        start = len(frontier)
        if not isinstance(case, dict):
            issue(field, 'case dictionary required'); continue
        identity = case.get('id')
        if not isinstance(identity, str) or not identity or identity in ids:
            issue(field + '.id', 'unique nonempty case id required')
        else:
            ids.add(identity)
        if case.get('role') not in _ROLES:
            issue(field + '.role', 'explicit training or frozen holdout role required')
        for side in _SIDES:
            row = case.get(side)
            if not isinstance(row, dict):
                issue(field + '.' + side, 'features and complete-call timing required'); continue
            features, timing = row.get('features'), row.get('timing')
            target = demand[side]['features']
            if not isinstance(features, dict) or set(features) != set(target) or any(
                    type(value) is not int or value < 0 for value in features.values()):
                issue(field + '.' + side + '.features', 'one exact count for every target feature required')
            elif case.get('role') == 'TARGET_HOLDOUT' and features != target:
                issue(field + '.' + side + '.features', 'target holdout must use exact target demands')
            elif any(features.get(name) != target.get(name) for name in _FIXED[side] if name in target):
                issue(field + '.' + side + '.features', 'fixed shape, caller count or input arity differs')
            elif any(features['arithmetic.' + op] for op in _OPS if 'arithmetic.' + op not in active):
                issue(field + '.' + side + '.features', 'corpus uses an arithmetic operator outside the selected target basis')
            if not isinstance(timing, dict):
                issue(field + '.' + side + '.timing', 'raw complete-call samples required'); continue
            samples = timing.get('samples_ns')
            if not isinstance(samples, list) or len(samples) < 3 or any(
                    _number(value) is None or value <= 0 for value in samples):
                issue(field + '.' + side + '.timing', 'at least three positive finite raw nanosecond samples required')
                continue
            expected = dict(median_ns=statistics.median(samples), min_ns=min(samples), max_ns=max(samples))
            if any(_number(timing.get(key)) != _number(value) for key, value in expected.items()):
                issue(field + '.' + side + '.timing', 'median/min/max must equal retained raw samples')
        if len(frontier) == start:
            valid.append(case)
    role_counts = {role:sum(case['role'] == role for case in valid) for role in _ROLES}
    for role in _ROLES:
        if not role_counts[role]:
            issue('observations.cases.' + role, 'at least one valid ' + role + ' case required')

    fits, holdouts, parameters = {}, [], {}
    # Malformed metadata never supplies a partial fitted model.
    if not frontier:
        from scar.compiler.calibrate_cpu import _fit
        import numpy as np

        training = [case for case in valid if case['role'] == 'TRAIN']
        for side in _SIDES:
            rows = [dict(intercept=1, **{name:case[side]['features'][name] for name in active}) for case in training]
            medians = [case[side]['timing']['median_ns'] for case in training]
            try:
                fitted = _fit(basis, rows, medians)
                matrix = np.asarray([[row[name] for name in basis] for row in rows], dtype=float)
                scales = np.max(np.abs(matrix), axis=0)
                singular = np.linalg.svd(matrix / np.where(scales == 0, 1, scales), compute_uv=False)
                full_rank = fitted['rank'] == len(basis) and fitted['parameters'] == len(basis)
                condition = float(singular[0]/singular[-1]) if full_rank else None
                fits[side] = dict(nnls=fitted, fit_target_unit='nanoseconds', basis=basis,
                    singular_values=[float(value) for value in singular], column_scales=[float(value) for value in scales],
                    condition_number=condition if condition is not None and math.isfinite(condition) else None,
                    full_basis_identified=full_rank, train_case_ids=[case['id'] for case in training])
                if not full_rank:
                    issue(side + '.rank', 'full selected basis is not identifiable from TRAIN cases')
                if any(_number(value) is None for value in fitted['coefficients'].values()):
                    issue(side + '.coefficients', 'fit did not produce finite nonnegative coefficients')
                    continue
                coefficients = {}
                for name, count in demand[side]['features'].items():
                    if count == 0:
                        continue
                    value = fitted['coefficients'][name]/1e9 if name in active else 0
                    coefficients[name] = _parameter(value, 'seconds/' + demand[side]['feature_units'][name],
                        _CONSTRAINED if name in active else _GROUPED)
                parameters[side] = dict(protected_entry_overhead=_parameter(
                    fitted['coefficients']['intercept']/1e9, 'seconds/call'), coefficients=coefficients)
                for case in valid:
                    if case['role'] == 'TRAIN':
                        continue
                    predicted = sum(fitted['coefficients'][name] * (1 if name == 'intercept' else case[side]['features'][name]) for name in basis)
                    actual = case[side]['timing']['median_ns']
                    if _number(predicted) is None:
                        issue(side + '.holdout.' + case['id'], 'frozen prediction is not finite nonnegative nanoseconds')
                        continue
                    error = abs(predicted - actual)/actual
                    if not math.isfinite(error):
                        issue(side + '.holdout.' + case['id'], 'frozen holdout relative error is not finite')
                        continue
                    passed = error <= float(limit)
                    holdouts.append(dict(case_id=case['id'], role=case['role'], side=side,
                        predicted_ns=predicted, observed_complete_call_median_ns=actual,
                        absolute_error_ns=abs(predicted-actual), relative_error=error, passed=passed,
                        parameters_refitted=False))
                    if not passed:
                        issue(side + '.holdout.' + case['id'], 'frozen full-call holdout exceeds declared relative error threshold')
            except (ValueError, RuntimeError, OverflowError, np.linalg.LinAlgError) as error:
                issue(side + '.fit', str(error))
    warm = {name:_parameter(0, 'seconds/call',
        'OUTSIDE_EXACT_WARM_PROTECTED_CALL_BOUNDARY_BUILD_LOAD_SETUP_COMPLETE_BEFORE_ENTRY') for name in _WARM}
    complete = not frontier
    root = Path(__file__).resolve().parents[1]
    return dict(schema=SCHEMA, schema_version=1,
        status='ESTIMATED_UNDER_PROFILE_CONTRACT' if complete else 'UNRESOLVED',
        profile_available=complete, context=context, demand_sha256=_digest(demand),
        observations_sha256=_digest(observations), basis=basis,
        maximum_holdout_relative_error=float(limit), measurement=deepcopy(observations.get('measurement')),
        qualification=deepcopy(observations.get('qualification')), role_counts=role_counts,
        fits=fits, holdouts=holdouts, parameters=dict(**parameters, warm_exclusions=warm) if complete else {},
        frontier=frontier, model_transfer_assumptions=list(_ASSUMPTIONS),
        sourcepins={name:artifact_reference(root / name, 'scar.backend-implementation-source') for name in (
            'optimization/pipeline_corpus_model.py', 'compiler/calibrate_cpu.py',
            'optimization/pipeline_forward_cost.py')},
        source_admission='QUALIFIED_PARENT_F4_REPLAY_REQUIRED', source_replay_performed=False,
        whole_scope_observed=False, coefficients_promoted_to_observed_components=False,
        scope_difference_used_as_component=False, physical_bounds_proven=False,
        workload_executed=False, native_loaded=False, deployment_authorized=False)


def profile_for_corpus_fit(fit_artifact_ref, region, bp, demand, observations):
    """Replay a published fit and source its coefficients into the existing profile."""
    reference = (artifact_reference(Path(fit_artifact_ref).resolve(), SCHEMA)
        if isinstance(fit_artifact_ref, (str, Path)) else encode(fit_artifact_ref))
    report, _ = _fact(dict(artifact=reference, locator='#'))
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('versioned retained corpus fit required')
    expected = fit_pipeline_corpus(region, bp, demand, observations,
        maximum_holdout_relative_error=report.get('maximum_holdout_relative_error'))
    if _digest(report) != _digest(expected):
        raise ValueError('retained corpus fit differs from exact TRAIN/holdout replay')
    profile = coefficient_profile_skeleton(region, bp, demand)
    profile['model_assumptions'].extend(report['model_transfer_assumptions'])
    profile['provenance'] = [dict(corpus_fit=deepcopy(reference),
        observations_sha256=report['observations_sha256'], fit_status=report['status'],
        canonical_to_target_transfer='DECLARED_MODEL_NOT_TARGET_COMPONENT_OBSERVATION')]
    if not report['profile_available']:
        return profile

    def sourced(record, pointer):
        return dict(**deepcopy(record), evidence=[dict(artifact=deepcopy(reference), locator=pointer)])

    profile['context_evidence'] = [dict(artifact=deepcopy(reference), locator='#/context')]
    for side in _SIDES:
        source = report['parameters'][side]
        model = profile['model']['implementations'][side]
        model['protected_entry_overhead'] = sourced(source['protected_entry_overhead'],
            '#/parameters/' + side + '/protected_entry_overhead')
        model['coefficients'] = {name:sourced(record,
            '#/parameters/' + side + '/coefficients/' + name.replace('~', '~0').replace('/', '~1'))
            for name, record in source['coefficients'].items()}
    profile['warm_exclusions'] = {name:sourced(record, '#/parameters/warm_exclusions/' + name)
        for name, record in report['parameters']['warm_exclusions'].items()}
    return profile


__all__ = ['SCHEMA', 'OBSERVATIONS_SCHEMA', 'fit_pipeline_corpus', 'profile_for_corpus_fit']
