"""Bridge exact pipeline demands to the existing protected-call cost gate.

Only metadata is replayed.  A declared effective model is not an observed total,
physical traffic estimate, source admission or permission to execute native code.
"""
from copy import deepcopy
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference
from scar.ir.record_codec import encode
from scar.optimization.cost_model import evaluate_sourced_affine
from scar.optimization.scope_accounting import _digest, _fact, _number, _profile
from scar.optimization.macro_pass import COST_SCOPE, _admit, _cost_context
from scar.optimization.semantic_region_demand import (
    SUCCESSOR_SCHEMA, _verify_admitted_pipeline_demand, verify_successor_pipeline_cost_demand,
)


SCHEMA = 'scar.pipeline-forward-cost-prediction'
PROFILE_SCHEMA = 'scar.pipeline-forward-coefficient-profile'
_SIDES = ('original', 'fused')
_WARM = ('original_one_time', 'native_load', 'native_setup')
_ENTRY_COMPONENT = 'ALL_PROTECTED_CALL_WORK_EXCEPT_LISTED_FEATURE_SERVICES'
_OVERLAP = 'DISJOINT_EFFECTIVE_MODEL_ASSUMPTION'
_ASSUMPTIONS = [
    'DECLARED_EFFECTIVE_AFFINE_MODEL_NOT_AN_OBSERVED_PROTECTED_CALL_TOTAL',
    'LOGICAL_DEMAND_FEATURES_ARE_NOT_DRAM_TRAFFIC_OR_COMPILER_INSTRUCTION_COUNTS',
    'PROTECTED_ENTRY_OVERHEAD_EXCLUDES_THE_LISTED_FEATURE_SERVICES',
    'EFFECTIVE_COEFFICIENT_SUM_IS_A_MODEL_ASSUMPTION_NOT_AN_ADDITIVITY_THEOREM',
    'WARM_ONE_TIME_EXCLUSIONS_REQUIRE_SOURCED_ZERO_JUSTIFICATION',
]


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _document(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_bytes())
    _need(type(value) is dict, 'versioned metadata dictionary required')
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _materials(region_document, blueprint, demand):
    admitted = _admit(region_document, blueprint)
    region, blueprint = admitted.region, admitted.blueprint
    context = _cost_context(region, blueprint, admitted.graph)
    demand = _document(demand)
    if demand.get('schema') == SUCCESSOR_SCHEMA:
        demand = verify_successor_pipeline_cost_demand(region, blueprint, demand)
        _need(context['Q'] == demand['Q'], 'published successor and complete entry cost require the same Q')
    else:
        demand = _verify_admitted_pipeline_demand(admitted, demand)
    return region, blueprint, demand, context


def coefficient_profile_skeleton(region_document, blueprint, demand):
    """Return an entirely UNKNOWN template; no missing cost is filled with zero."""
    _, _, demand, context = _materials(region_document, blueprint, demand)

    def unknown(unit):
        return dict(value=None, unit=unit, basis='DECLARED_MODEL', evidence=[])

    models = {}
    for side in _SIDES:
        coefficients = {name: unknown('seconds/' + demand[side]['feature_units'][name])
            for name, count in demand[side]['features'].items() if count != 0}
        models[side] = dict(protected_entry_overhead=unknown('seconds/call'),
            coefficients=coefficients,
            coverage=dict(scope=COST_SCOPE, entry_component=_ENTRY_COMPONENT,
                entry_excludes_features=sorted(coefficients), overlap=_OVERLAP,
                runtime_boundary_fields=sorted(demand['runtime_boundary'])))
    return dict(schema=PROFILE_SCHEMA, schema_version=1, context=context,
        demand_sha256=_digest(demand), context_evidence=[],
        model=dict(kind='EFFECTIVE_AFFINE', composition='SUM_EFFECTIVE_TERMS',
            implementations=models),
        warm_exclusions={name:unknown('seconds/call') for name in _WARM},
        model_assumptions=list(_ASSUMPTIONS), provenance=[])


def predict_pipeline_forward_cost(region_document, blueprint, demand, coefficient_profile):
    """Evaluate the two exact implementation demands with the shared affine helper."""
    region, blueprint, demand, context = _materials(region_document, blueprint, demand)
    profile, profile_source = _profile({} if coefficient_profile is None else coefficient_profile)
    profile = _document(profile)
    frontier = []

    def issue(field, reason):
        frontier.append(dict(field=field, reason=reason))

    def mapping(value, field):
        if type(value) is not dict:
            issue(field, 'explicit model dictionary required')
            return {}
        return value

    allowed = {'schema', 'schema_version', 'context', 'demand_sha256', 'context_evidence',
        'model', 'warm_exclusions', 'model_assumptions', 'provenance'}
    if set(profile) - allowed:
        issue('profile', 'unsupported profile fields cannot be silently ignored')
    if (profile.get('schema'), profile.get('schema_version')) != (PROFILE_SCHEMA, 1) or type(profile.get('schema_version')) is not int:
        issue('profile.schema', 'versioned pipeline forward coefficient profile required')
    if _digest(profile.get('context')) != _digest(context):
        issue('profile.context', 'complete protected-call macro cost context must match exactly')
    if profile.get('demand_sha256') != _digest(demand):
        issue('profile.demand_sha256', 'exact original/fused source-demand identity differs')
    model = mapping(profile.get('model'), 'profile.model')
    if set(model) - {'kind', 'composition', 'implementations'}:
        issue('profile.model', 'unsupported model fields')
    if model.get('kind') != 'EFFECTIVE_AFFINE' or model.get('composition') != 'SUM_EFFECTIVE_TERMS':
        issue('profile.model', 'explicit effective affine sum model required')
    implementations = mapping(model.get('implementations'), 'profile.model.implementations')
    if set(implementations) != set(_SIDES):
        issue('profile.model.implementations', 'exactly original and fused models required')
    predictions, coverages = {}, {}
    for side in _SIDES:
        field = 'profile.model.implementations.' + side
        row = mapping(implementations.get(side), field)
        if set(row) - {'protected_entry_overhead', 'coefficients', 'coverage'}:
            issue(field, 'unsupported implementation fields')
        coefficients = mapping(row.get('coefficients'), field + '.coefficients')
        coverage = mapping(row.get('coverage'), field + '.coverage')
        coverages[side] = deepcopy(coverage)
        expected_coverage = dict(scope=COST_SCOPE, entry_component=_ENTRY_COMPONENT,
            entry_excludes_features=sorted(coefficients), overlap=_OVERLAP,
            runtime_boundary_fields=sorted(demand['runtime_boundary']))
        if coverage != expected_coverage:
            issue(field + '.coverage',
                'complete protected entry coverage and disjoint feature-service exclusions required')
        prediction = evaluate_sourced_affine(context=context,
            context_evidence=profile.get('context_evidence'),
            features=demand[side]['features'], feature_units=demand[side]['feature_units'],
            intercept=row.get('protected_entry_overhead'), coefficients=coefficients)
        predictions[side] = prediction
        frontier.extend(dict(field=side + '.' + item['field'], reason=item['reason'])
            for item in prediction['frontier'])
    exclusions = mapping(profile.get('warm_exclusions'), 'profile.warm_exclusions')
    if set(exclusions) != set(_WARM):
        issue('profile.warm_exclusions', 'all three warm one-time exclusions need sourced zero records')
    warm = {}
    for name in _WARM:
        record = exclusions.get(name)
        prediction = evaluate_sourced_affine(context=context,
            context_evidence=profile.get('context_evidence'), features={}, feature_units={},
            intercept=record, coefficients={})
        warm[name] = prediction
        frontier.extend(dict(field='warm_exclusions.' + name + '.' + item['field'],
            reason=item['reason']) for item in prediction['frontier'])
        if type(record) is not dict or _number(record.get('value')) != 0:
            issue('warm_exclusions.' + name, 'warm protected-call scope excludes this one-time term; a sourced zero is required')
    assumptions = profile.get('model_assumptions', [])
    if type(assumptions) is not list or any(type(item) is not str or not item for item in assumptions):
        issue('profile.model_assumptions', 'model assumptions must be explicit nonempty strings')
        assumptions = []
    complete = not frontier
    components = {}
    if complete:
        estimates = {**warm, 'original_per_call':predictions['original'],
                     'native_per_call':predictions['fused']}
        components = {name:dict(value=value['value_seconds'], unit='seconds',
            basis='DECLARED_MODEL') for name, value in estimates.items()}
    root = Path(__file__).resolve().parents[1]
    implementations = {name:artifact_reference(root / name, 'scar.backend-implementation-source')
        for name in ('optimization/pipeline_forward_cost.py', 'optimization/semantic_region_demand.py',
                     'optimization/cost_model.py', 'optimization/scope_accounting.py',
                     'optimization/macro_pass.py')}
    successor = ({'macro_preparation': deepcopy(demand['macro_preparation']),
        'candidate_binding': deepcopy(demand['candidate_binding'])} if 'candidate_binding' in demand else {})
    return dict(schema=SCHEMA, schema_version=1,
        status='ESTIMATED_UNDER_PROFILE_CONTRACT' if complete else 'UNRESOLVED',
        prediction_available=complete, context=context, demand_sha256=_digest(demand),
        region_document_sha256=_digest(region), blueprint_sha256=_digest(blueprint),
        coefficient_profile=profile, coefficient_profile_source=profile_source,
        coefficient_profile_sha256=_digest(profile),
        features={side:deepcopy(demand[side]) for side in _SIDES},
        implementation_predictions=predictions, warm_exclusion_predictions=warm,
        coverage=coverages,
        components=components, frontier=frontier,
        source_demand_frontier=deepcopy(demand['frontier']),
        runtime_boundary=deepcopy(demand['runtime_boundary']),
        physical_unknowns=deepcopy(demand['physical']),
        required_hardware_cost_fields=deepcopy(demand['required_hardware_cost_fields']),
        model_assumptions=list(dict.fromkeys([*_ASSUMPTIONS, *assumptions])),
        sourcepins=dict(source_refs=deepcopy(demand['source_refs']),
            demand_accounting_authority=demand['accounting_authority'], implementations=implementations),
        source_admission='QUALIFIED_PARENT_F4_REPLAY_REQUIRED', source_replay_performed=False,
        whole_scope_observed=False, coefficients_promoted_to_observed_components=False,
        scope_difference_used_as_component=False, physical_bounds_proven=False,
        target_program_executed=False, native_loaded=False, deployment_authorized=False, **successor)


def verify_pipeline_forward_cost(report, region_document, blueprint, demand, coefficient_profile):
    """Replay source demands, coefficient evidence and the complete bridge output."""
    report = _document(report)
    _need((report.get('schema'), report.get('schema_version')) == (SCHEMA, 1)
        and type(report.get('schema_version')) is int,
        'versioned pipeline forward prediction required')
    expected = predict_pipeline_forward_cost(region_document, blueprint, demand, coefficient_profile)
    _need(_digest(expected) == _digest(report),
        'pipeline forward prediction differs from exact source-demand/coefficient replay')
    return expected


def deployment_profile_for_prediction(prediction_ref, *, region_document, blueprint,
                                      demand, coefficient_profile):
    """Bind derived DECLARED_MODEL seconds to a published evidence artifact."""
    reference = (artifact_reference(Path(prediction_ref).resolve(), SCHEMA)
        if isinstance(prediction_ref, (str, Path)) else encode(prediction_ref))
    report, _ = _fact(dict(artifact=reference, locator='#'))
    report = verify_pipeline_forward_cost(report, region_document, blueprint, demand, coefficient_profile)

    def fact(locator):
        return dict(artifact=deepcopy(reference), locator=locator)

    costs = {name:dict(value=record['value'], unit='seconds', basis='DECLARED_MODEL',
        evidence=[fact('#/components/' + name)]) for name, record in report['components'].items()}
    successor = ({'macro_preparation': deepcopy(report['macro_preparation']),
        'candidate_binding': deepcopy(report['candidate_binding'])} if 'candidate_binding' in report else {})
    return dict(schema='scar.deployment-cost-profile', schema_version=1, mode='COMPONENTS',
        context=deepcopy(report['context']), context_evidence=[fact('#/context')], costs=costs,
        model_assumptions=deepcopy(report['model_assumptions']),
        provenance=dict(prediction=deepcopy(reference), demand_sha256=report['demand_sha256'],
            coefficient_profile_sha256=report['coefficient_profile_sha256'],
            derived_component_basis='DECLARED_MODEL', whole_scope_observed=False,
            source_demand_replayed=True, coefficient_evidence_replayed=True, **successor))


__all__ = ['SCHEMA', 'PROFILE_SCHEMA', 'coefficient_profile_skeleton',
    'predict_pipeline_forward_cost', 'verify_pipeline_forward_cost',
    'deployment_profile_for_prediction']
