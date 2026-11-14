"""Forward effective-affine estimates; no target execution or hardware guessing."""

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import math
from pathlib import Path

from scar.ir.program.providers import ProviderFactReference
from scar.ir.record_codec import decode, loads
from .demand import analyze_region_demand
from .resource_solver import hardware_model_skeleton, semantic_sha256


_OPS = ('add', 'sub', 'mul', 'div', 'neg')
_UNITS = {
    'ALLOCATION': {'output_bytes': 'ticks/byte'}, 'EXPORT': {},
    'COMPUTE': {
        'output_elements': 'ticks/output_element', 'logical_payload_bytes': 'ticks/logical_byte',
        'setup.input_bindings': 'ticks/binding', 'setup.view_descriptors': 'ticks/descriptor',
        'setup.view_axis_normalizations': 'ticks/normalization', 'setup.scalar_constants': 'ticks/constant',
        **{'arithmetic.' + op: 'ticks/operation' for op in _OPS},
        **{'setup.scalar_arithmetic.' + op: 'ticks/operation' for op in _OPS}}}
_ASSUMPTIONS = [
    'EFFECTIVE_AFFINE_SUM_IS_AN_EMPIRICAL_MODEL_NOT_AN_ADDITIVITY_THEOREM',
    'LOGICAL_PAYLOAD_BYTES_ARE_NOT_DRAM_TRAFFIC_OR_CACHE_MISSES',
    'EXACT_CPU_SET_COEFFICIENTS_WITHOUT_LINEAR_WORKER_SCALING',
    'COMPILER_LAYOUT_AND_WORKING_SET_REGIME_ARE_APPLICABILITY_CONTRACTS',
    'FIXED_RESOURCE_DOMAIN_AND_PLACEMENT_NOT_JOINTLY_OPTIMIZED',
    'ESTIMATES_ARE_NOT_PHYSICAL_OPTIMALITY_OR_DEPLOYMENT_AUTHORIZATION']


def _fraction(value, positive=False):
    if type(value) not in (int, float):
        return None
    try:
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            return None
        return Fraction(str(value))
    except (ValueError, OverflowError):
        return None


def _ratio(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def _same(left, right):
    try:
        return semantic_sha256(left) == semantic_sha256(right)
    except (TypeError, ValueError):
        return False


def _pointer(document, locator):
    if locator == '#':
        return document
    if not locator.startswith('#/'):
        raise ValueError('JSON pointer required')
    for part in locator[2:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(document, list):
            if not part.isdigit() or (part.startswith('0') and part != '0'):
                raise ValueError('canonical array index required')
            document = document[int(part)]
        else:
            document = document[part]
    return document


def _features(demand, elements):
    setup = demand['per_call_setup']
    compute = {
        'output_elements': (elements, 'output_elements'),
        'logical_payload_bytes': (
            demand['fused']['logical_payload_transfer_bytes']['per_output_element'] * elements,
            demand['fused']['logical_payload_transfer_bytes']['total']),
        **{'arithmetic.' + op: (demand['per_output_element']['arithmetic'][op] * elements,
                                demand['total_work']['arithmetic'][op]['total']) for op in _OPS},
        **{'setup.' + key: (setup[key], str(setup[key])) for key in
           ('input_bindings', 'view_descriptors', 'view_axis_normalizations', 'scalar_constants')},
        **{'setup.scalar_arithmetic.' + op: (setup['scalar_arithmetic'][op],
                                            str(setup['scalar_arithmetic'][op])) for op in _OPS}}
    return {'ALLOCATION': {'output_bytes': (8 * elements, '8 * output_elements')},
            'COMPUTE': compute, 'EXPORT': {}}


def features_for_demand(demand, elements):
    """Expose the same feature counts/expressions used by forward prediction."""
    if type(elements) is not int or elements < 0:
        raise ValueError('elements must be a nonnegative integer')
    return _features(demand, elements)


def cost_feature_units():
    """Return an isolated copy of the profile's feature/unit authority."""
    return deepcopy(_UNITS)


def predict_region_costs(plan, schedule, profile, *, input_shapes, backend_context=None):
    """Estimate ticks from sourced coefficients under an exact profile contract.

    Parameters are {value, unit, evidence:[ProviderFactReference wire]}; fitted
    zeros also need zero_basis. Evidence SHA/size/schema/JSON pointers are
    checked locally. Every nonzero demand feature needs a coefficient. Compiler
    identity/options are matched when artifact context is supplied, otherwise
    retained as a mandatory applicability guard. Available budgets stay unknown.
    """
    from scar.codegen.cpp import runtime_implementation_sha256

    skeleton = hardware_model_skeleton(plan, schedule, input_shapes=input_shapes)
    demand = analyze_region_demand(plan)
    issues, cache, artifacts, context_sources = [], {}, {}, set()

    def issue(field, reason):
        issues.append(dict(field=field, reason=reason))

    def evidence(records, field, *, parameter_record=None, compiler_record=None, context_record=None):
        if not isinstance(records, (list, tuple)) or not records:
            issue(field, 'nonempty original ProviderFactReference evidence required')
            return False
        valid, anchors = True, 0
        for index, record in enumerate(records):
            try:
                reference = decode(ProviderFactReference, record)
                artifact = reference.artifact
                location = Path(artifact.uri)
                if not location.is_absolute():
                    raise ValueError('absolute local artifact path required')
                key = (artifact.uri, artifact.sha256, artifact.size_bytes, artifact.schema)
                if key not in cache:
                    raw = location.read_bytes()
                    if len(raw) != artifact.size_bytes or sha256(raw).hexdigest() != artifact.sha256:
                        raise ValueError('artifact SHA/size receipt differs')
                    document = loads(raw)
                    if isinstance(document, dict) and document.get('schema') != artifact.schema:
                        raise ValueError('artifact schema differs')
                    cache[key] = document
                referent = _pointer(cache[key], reference.locator)
                if parameter_record is not None:
                    measured = referent.get('value') if isinstance(referent, Mapping) else referent
                    if type(measured) in (int, float):
                        if _fraction(measured) != _fraction(parameter_record.get('value')):
                            raise ValueError('fit result value differs from the coefficient value')
                        if isinstance(referent, Mapping) and referent.get('unit') != parameter_record.get('unit'):
                            raise ValueError('fit result unit differs from the coefficient unit')
                        if context_sources and key not in context_sources:
                            raise ValueError('fit result artifact is not bound to the retained calibration context')
                        anchors += 1
                if compiler_record is not None and isinstance(referent, Mapping):
                    identity = referent.get('identity', referent.get('compiler_version'))
                    options = referent.get('options', referent.get('compiler_options'))
                    argv = referent.get('argv', referent.get('compiler'))
                    if identity is not None or options is not None or argv is not None:
                        if (not _same(identity, compiler_record.get('identity'))
                                or not _same(options, compiler_record.get('options'))
                                or not _same(argv, compiler_record.get('argv'))):
                            raise ValueError('compiler evidence identity/options/argv differ from profile')
                        anchors += 1
                if context_record is not None and isinstance(referent, Mapping):
                    if any(key in referent for key in context_record):
                        target_record, compiler_context = referent.get('target'), referent.get('compiler')
                        if not isinstance(target_record, Mapping) or not isinstance(compiler_context, Mapping):
                            raise ValueError('complete original target/compiler/runtime/execution-boundary context required')
                        normalized = {
                            'target': {key: target_record.get(key) for key in context_record['target']},
                            'compiler': {key: compiler_context.get(key) for key in context_record['compiler']},
                            'runtime_implementation_sha256': referent.get('runtime_implementation_sha256'),
                            'execution_boundary': referent.get('execution_boundary')}
                        if not _same(normalized, context_record):
                            raise ValueError('original calibration target/compiler/runtime/execution-boundary context differs from profile')
                        context_sources.add(key)
                        anchors += 1
                artifacts[semantic_sha256(record['artifact'])] = deepcopy(record['artifact'])
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                issue(f'{field}.{index}', 'invalid retained calibration evidence: ' + str(error))
                valid = False
        if (parameter_record is not None or compiler_record is not None or context_record is not None) and anchors == 0:
            issue(field, 'at least one complete matching fit-result/compiler/calibration-context evidence anchor required')
            valid = False
        return valid

    def parameter(record, field, unit, positive=False):
        if not isinstance(record, Mapping):
            issue(field, 'explicit coefficient required; unknown is not zero')
            return None
        start = len(issues)
        if record.get('status') in ('UNKNOWN', 'UNRESOLVED'):
            issue(field + '.status', 'parameter remains unknown')
        value = _fraction(record.get('value'), positive)
        if value is None:
            issue(field + '.value', 'finite ' + ('positive' if positive else 'nonnegative') + ' number required')
        if record.get('unit') != unit:
            issue(field + '.unit', 'expected ' + unit)
        if value == 0 and not (isinstance(record.get('zero_basis'), str) and record['zero_basis'].strip()):
            issue(field + '.zero_basis', 'zero needs an explicit calibration/fit justification')
        evidence(record.get('evidence'), field + '.evidence', parameter_record=record)
        return value if len(issues) == start else None

    def mapping(value):
        return value if isinstance(value, Mapping) else {}

    profile = deepcopy(dict(profile)) if isinstance(profile, Mapping) else {}
    try:
        profile_hash = semantic_sha256(profile)
    except (TypeError, ValueError):
        profile_hash = None
        issue('profile', 'finite semantic JSON required')
    if profile.get('schema') != 'scar.region-cost-profile' or type(profile.get('schema_version')) is not int or profile['schema_version'] != 1:
        issue('profile.schema', 'scar.region-cost-profile version 1 required')
    binding, target = plan.resource_binding, mapping(profile.get('target'))
    for key, expected in (
        ('target_sha256', binding.target_sha256 if binding else None),
        ('executor_id', binding.executor_id if binding else None),
        ('cpu_ids', list(binding.cpu_ids) if binding else None)):
        if expected is None or not _same(target.get(key), expected):
            issue('profile.target.' + key, 'must match exact recorded target/executor/CPU allocation')
    backend = mapping(profile.get('backend'))
    if backend.get('guard_mode') != 'TIMED_SYNCHRONOUS_CPU_PROFILE':
        issue('profile.backend.guard_mode', 'exact TIMED_SYNCHRONOUS_CPU_PROFILE boundary required')
    current_implementation = runtime_implementation_sha256()
    runtime_match = _same(backend.get('runtime_implementation_sha256'), current_implementation)
    if not runtime_match:
        issue('profile.backend.runtime_implementation_sha256', 'must match the current C++ boundary/template/header/timed implementation')
    for key, expected in (('name', 'cpp'), ('execution_profile', plan.profile),
                          ('operator_abi_version', plan.operator_abi_version)):
        if not _same(backend.get(key), expected):
            issue('profile.backend.' + key, 'backend contract differs from region profile')
    compiler = mapping(backend.get('compiler'))
    if not isinstance(compiler.get('identity'), str) or not compiler['identity'].strip():
        issue('profile.backend.compiler.identity', 'explicit compiler identity required')
    options = compiler.get('options')
    if not isinstance(options, list) or any(not isinstance(v, str) or not v for v in options):
        issue('profile.backend.compiler.options', 'explicit compiler option list required')
    argv = compiler.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or not v for v in argv):
        issue('profile.backend.compiler.argv', 'explicit complete nonempty CXX command prefix required')
    evidence(compiler.get('evidence'), 'profile.backend.compiler.evidence', compiler_record=compiler)
    calibration_context = {
        'target': {key: target.get(key) for key in ('target_sha256', 'executor_id', 'cpu_ids')},
        'runtime_implementation_sha256': backend.get('runtime_implementation_sha256'),
        'execution_boundary': backend.get('guard_mode'),
        'compiler': {key: compiler.get(key) for key in ('identity', 'options', 'argv')}}
    calibration_match = evidence(backend.get('context_evidence'), 'profile.backend.context_evidence',
                                 context_record=calibration_context)
    compiler_match = 'MANDATORY_GUARD_REQUIRED'
    if backend_context is not None:
        compiler_match = 'MATCHED_DECLARED_ARTIFACT_CONTEXT'
        context = mapping(backend_context)
        for key, expected in (('compiler_version', compiler.get('identity')),
                              ('compiler_options', options), ('backend', 'cpp'),
                              ('compiler', argv),
                              ('runtime_implementation_sha256', current_implementation)):
            if not _same(context.get(key), expected):
                issue('backend_context.' + key, 'native artifact context differs from profile')
                compiler_match = 'MISMATCHED_ARTIFACT_CONTEXT'
    applicability = mapping(profile.get('applicability'))
    if applicability.get('layout') != 'C_CONTIGUOUS_FLOAT64_2D':
        issue('profile.applicability.layout', 'C_CONTIGUOUS_FLOAT64_2D contract required')
    working = mapping(applicability.get('working_set'))
    if not isinstance(working.get('regime'), str) or not working['regime'].strip():
        issue('profile.applicability.working_set.regime', 'explicit regime required; residency is not inferred')
    evidence(working.get('evidence'), 'profile.applicability.working_set.evidence')
    instantiation = skeleton['model_instantiation']
    elements = instantiation['output_elements']
    features = _features(demand, elements) if elements is not None else None
    logical_bytes = features['COMPUTE']['logical_payload_bytes'][0] if features else None
    for key, value in (('output_elements', elements), ('logical_payload_bytes', logical_bytes)):
        minimum, maximum = working.get(key + '_min'), working.get(key + '_max')
        if type(minimum) is not int or type(maximum) is not int or not 0 <= minimum <= maximum:
            issue('profile.applicability.working_set.' + key, 'explicit inclusive min/max calibration range required')
        elif value is not None and not minimum <= value <= maximum:
            issue('profile.applicability.working_set.' + key, 'demand is outside declared estimate range')
    model = mapping(profile.get('model'))
    if model.get('kind') != 'EFFECTIVE_AFFINE':
        issue('profile.model.kind', 'only EFFECTIVE_AFFINE is supported')
    if model.get('composition') != 'SUM_EFFECTIVE_TERMS':
        issue('profile.model.composition', 'explicit SUM_EFFECTIVE_TERMS assumption required')
    tick_seconds = parameter(model.get('tick_seconds'), 'profile.model.tick_seconds', 'seconds/tick', True)
    models = mapping(model.get('task_models'))
    for name in models:
        if name not in _UNITS:
            issue('profile.model.task_models.' + str(name), 'transport/other task models unsupported')
    context_valid = not issues
    costs = {}
    for task in schedule.tasks:
        name, field = task.kind.value, 'profile.model.task_models.' + task.kind.value
        if name not in _UNITS:
            issue('task_costs.' + task.id, 'COPY/WAIT/RELEASE cost and execution unresolved')
            continue
        before = len(issues)
        task_model = mapping(models.get(name))
        intercept = parameter(task_model.get('intercept'), field + '.intercept', 'ticks')
        coefficients = task_model.get('coefficients')
        if not isinstance(coefficients, Mapping):
            issue(field + '.coefficients', 'explicit mapping required, including {} for EXPORT')
            coefficients = {}
        for feature in coefficients:
            if feature not in _UNITS[name]:
                issue(field + '.coefficients.' + str(feature), 'unknown static demand feature')
        terms, estimate = [], intercept if intercept is not None else Fraction(0)
        if intercept is not None:
            terms.append(dict(feature='intercept', count=1, count_expression='1',
                              coefficient=deepcopy(task_model['intercept']), contribution_ticks=_ratio(intercept)))
        for feature, (count, expression) in (features[name].items() if features else ()):
            if count == 0 and feature not in coefficients:
                continue  # Zero work does not assert a zero hardware coefficient.
            coefficient = parameter(coefficients.get(feature), field + '.coefficients.' + feature, _UNITS[name][feature])
            if coefficient is not None:
                contribution = coefficient * count
                estimate += contribution
                terms.append(dict(feature=feature, count=count, count_expression=expression,
                                  coefficient=deepcopy(coefficients[feature]), contribution_ticks=_ratio(contribution)))
        if not context_valid or len(issues) != before or features is None or tick_seconds is None:
            continue
        duration = (estimate.numerator + estimate.denominator - 1) // estimate.denominator
        costs[task.id] = dict(duration_ticks=duration, status='ESTIMATED', unit='ticks',
                              profile_sha256=profile_hash, model='EFFECTIVE_AFFINE',
                              unrounded_ticks=_ratio(estimate), terms=terms,
                              duration_seconds_from_ticks=_ratio(duration * tick_seconds))
    resource_issues = [item for item in skeleton['unresolved'] if not item['field'].startswith('task_costs.')]
    if not instantiation['complete']:
        costs = {}
    complete = len(costs) == len(schedule.tasks) and not issues
    fields = ('task_placements', 'resource_capacities', 'memory_capacities',
              'materialization_sizes', 'memory_locations', 'model_instantiation')
    return dict(
        schema='scar.region-cost-prediction', schema_version=1,
        status='ESTIMATED_UNDER_PROFILE_CONTRACT' if complete else 'UNRESOLVED',
        prediction_available=complete, target_program_executed=False, physical_optimality_proven=False,
        deployment_authorized=False, placement_optimization='FIXED_INPUT_PLACEMENTS_NOT_JOINT',
        profile_sha256=profile_hash, schedule_sha256=skeleton['schedule_sha256'], profile=profile,
        source_artifacts=list(artifacts.values()), task_costs=costs,
        solver_model={**{field: skeleton[field] for field in fields}, 'task_costs': costs},
        unresolved=issues + resource_issues, cost_unresolved=issues, resource_unresolved=resource_issues,
        model_assumptions=list(_ASSUMPTIONS), work_demand=demand, instantiation=instantiation,
        context=dict(compiler_match=compiler_match, applicability=deepcopy(applicability),
                     calibration_context_match='MATCHED_RETAINED_RECORD' if calibration_match else 'MISSING_OR_MISMATCHED',
                     target=deepcopy(target), backend=deepcopy(backend),
                     runtime_implementation_match='MATCHED_CURRENT_IMPLEMENTATION' if runtime_match else 'MISSING_OR_MISMATCHED',
                     input_layout=dict(required='C_CONTIGUOUS', status='MANDATORY_RUNTIME_GUARD', actual_layout_verified=False),
                     cachewarm_regime=dict(required=working.get('regime'), status='EXTERNAL_ASSUMPTION', actual_cache_state_verified=False),
                     runtime_guards=['EXACT_TARGET_SNAPSHOT_EXECUTOR_CPU_IDS_MATCH_PROFILE',
                                     'RUNTIME_IMPLEMENTATION_SHA256_MATCHES_PROFILE',
                                     'ALL_INPUTS_C_CONTIGUOUS_ALIGNED_NATIVE_FLOAT64_2D',
                                     'INPUT_OUTPUT_SHAPES_MATCH_MODEL_INSTANTIATION',
                                     'WORKING_SET_REGIME_MATCHES_PROFILE_EXTERNAL_CONTRACT',
                                     'COMPILER_IDENTITY_AND_OPTIONS_MATCH_PROFILE']),
        unknown=dict(actual_dram_bytes=None, cache_misses=None, actual_duration_seconds=None))


__all__ = ['predict_region_costs', 'features_for_demand', 'cost_feature_units']
