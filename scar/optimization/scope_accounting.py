"""Scope accounting."""

from dataclasses import replace, asdict, is_dataclass
from scar.ir.activity_costs import (
    ActivityCost,
    CostBasis,
    CostCoverage,
    validate_activity_cost_table,
)
from scar.ir.execution_recipe import read_artifact
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.versioned_schedule import (
    VersionedTaskKind as K,
    schedule_document,
    schedule_from_dict,
)
from scar.optimization.resource_solver import semantic_sha256
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import math
import re
from pathlib import Path
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN


# Source-backed work ownership and nonadditive control milestone accounting.
#
# A ledger relates old descriptive terms to final subjects. It neither adds the
# old coarse estimates nor turns implementation/model residuals into new work.
SCHEMA = 'scar.stateful-scope-accounting'
_COARSE = {
    'GUARDED_METHOD_AND_NATIVE_ARTIFACT_BOUNDARY': ('PYTHON_ENTRY_GUARD_PACKING',
        'NATIVE_METADATA_GUARD', 'NATIVE_RETURN', 'PYTHON_STATUS_RETURN'),
    'CALLER_WORKER_AFFINITY_FENV_TEAM': ('CALLER_CAPTURE_SCALAR_GUARD', 'TEAM_ENTER'),
    'SUCCESS_CHECK_AND_CALLER_RESTORATION': ('WORKER_EXIT', 'CALLER_RESTORE', 'NATIVE_HELPER_CLEANUP'),
}
_CONDITIONS = ('SOURCE_REPLAYED_STANDALONE_SYNCHRONOUS_NORMAL_SUCCESS_PROFILE',
    'PRECOMMIT_CHECK_WORK_OWNED_BY_ORIGINAL_CALLER_RESTORE',
    'NO_ADDITIONAL_NATIVE_FENCE_STATEMENT_OR_SERVICE',
    'CONTROL_MILESTONE_IS_NOT_A_MEASUREMENT_OR_PHYSICAL_ZERO_LATENCY_BOUND')


def project_scope_accounting(table, schedule, resources, recipe, demand, *, mapping,
                             boundary_derivation=None, dag_derivation=None, scope_schedule_ref=None):
    """Replay the exact modern source graph, then qualify its semantic fence."""
    from scar.analysis.implementation_scope import derive_stateful_scope_envelope
    validate_activity_cost_table(table, schedule, resources)
    replay = derive_stateful_scope_envelope(schedule_from_dict(recipe['task_graph']), recipe)
    if schedule_document(replace(schedule, temporal=None)) != schedule_document(replay):
        raise ValueError('scope accounting differs from complete original cleanup/source milestone replay')
    if (type(scope_schedule_ref) is not dict or scope_schedule_ref.get('schema') != 'scar.resource-schedule'
            or read_artifact(scope_schedule_ref) != schedule_document(schedule)):
        raise ValueError('original complete saved scope schedule and source certificate reference required')
    proof = replay.proof['implementation_scope_cleanup']; certificate = proof['milestone']
    source = proof['source']
    if (certificate['basis'] != 'SOURCE_PROVEN_ZERO_ADDITIONAL_SERVICE'
            or source.get('no_runtime_statement_between_restore_checks_and_commit') is not True
            or source.get('recipe_fence_is_semantic_gate_not_additional_emitted_work') is not True):
        raise ValueError('source replay has no nonadditive precommit control milestone')
    tasks = {row.id: row for row in schedule.tasks}
    fence = tasks[certificate['task_id']]; restore = tasks[certificate['restore_task_id']]
    if (fence.kind is not K.SUCCESS_FENCE or restore.attrs.get('phase_kind') != 'CALLER_RESTORE'
            or schedule.tasks.index(fence) != schedule.tasks.index(restore) + 1):
        raise ValueError('original restore/fence normal-success boundary differs')
    activities = [row for row in schedule.temporal.activities if row.task == fence.id]
    if len(activities) != 1:
        raise ValueError('one original semantic fence lifecycle required')
    activity = activities[0].id
    old = [row for row in table.costs if activity in row.activities]
    if (len(old) != 1 or old[0].basis is not CostBasis.UNKNOWN or old[0].activities != (activity,)):
        raise ValueError('source milestone may only replace its complete UNKNOWN subject')
    pointer = '#/proof/implementation_scope_cleanup/milestone'
    marker = ActivityCost('cost:source-milestone:' + semantic_sha256(dict(
        schedule=table.source_schedule_sha256, task=fence.id))[:32], (activity,),
        CostBasis.SOURCE_CONTROL_MILESTONE, estimate_ticks=0, coverage=CostCoverage.COMPLETE_SUBJECT,
        component_refs=(ProviderFactReference(ProviderArtifact(**scope_schedule_ref), pointer),),
        conditions=_CONDITIONS)
    result = replace(table, costs=tuple(marker if row is old[0] else row for row in table.costs),
        source_refs=(*table.source_refs, dict(artifact=scope_schedule_ref, locator=pointer)))
    summary = validate_activity_cost_table(result, schedule, resources)
    by_subject = {row.subject: row for row in result.costs}
    by_activity = {name: row.subject for row in result.costs for name in row.activities}
    activity_by_task = {row.task: row.id for row in schedule.temporal.activities}
    replacements = {}
    for derivation, field in ((boundary_derivation, 'phases'), (dag_derivation, 'stages')):
        if derivation is None:
            continue
        for group in derivation['mapping']:
            for row in group[field]:
                for original in row['replaced_unknown_subjects']:
                    if original in replacements:
                        raise ValueError('one original subject has ambiguous source-model replacements')
                    replacements[original] = row['subject']
    def resolve(subject):
        chain = [subject]
        while subject not in by_subject and subject in replacements:
            subject = replacements[subject]
            if subject in chain:
                raise ValueError('source cost replacement graph contains a cycle')
            chain.append(subject)
        return subject if subject in by_subject else None, chain
    original_mapping = {row['source_term']: row for row in mapping}
    records = []
    for term in demand['terms']:
        row = dict(term=term['id'], component=term['component'], original_source=term['source_ref'],
            cost_charged_separately=False, legacy_numeric_estimate_added=False)
        if term['component'] in _COARSE:
            kinds = _COARSE[term['component']]
            selected = [task for task in schedule.tasks if task.kind is K.IMPLEMENTATION_PHASE
                        and task.attrs.get('phase_kind') in kinds]
            if {task.attrs['phase_kind'] for task in selected} != set(kinds):
                row.update(disposition='SOURCE_FRONTIER', reason='ORIGINAL_COARSE_SCOPE_HAS_NO_COMPLETE_SOURCE_PHASE_OWNERS')
            else:
                row.update(disposition='SUPERSEDED_COARSE_DESCRIPTION', phase_kinds=list(kinds),
                    task_ids=[task.id for task in selected],
                    active_subjects=list(dict.fromkeys(by_activity[activity_by_task[task.id]] for task in selected)),
                    relation='SOURCE_WORK_DESCRIPTION_REPLACED_NOT_NUMERIC_SCOPE_EQUALITY')
        elif term['id'] in original_mapping:
            original = original_mapping[term['id']]
            subject, chain = resolve(original['subject'])
            if subject is None:
                row.update(disposition='SOURCE_FRONTIER', reason='ORIGINAL_COMPONENT_HAS_NO_FINAL_SUBJECT', replacement_chain=chain)
            else:
                row.update(disposition='ADDITIVE_COMPONENT', active_subjects=[subject],
                    replacement_chain=chain, activities=list(by_subject[subject].activities),
                    legacy_component_signature_has_canonical_sample=(len(chain) == 1 and by_subject[subject].basis is CostBasis.DECLARED_COMPONENT_MODEL))
        else:
            row.update(disposition='SOURCE_FRONTIER', reason='ORIGINAL_TERM_HAS_NO_SOURCE_COMPONENT_OR_PHASE_MAPPING')
        records.append(row)
    residuals = [dict(source=row.get('source'), kind=row['kind'], missing_terms=row['missing_terms'],
                      additional_work=False, cost_charged=False)
                 for row in result.unmapped_terms if 'missing_terms' in row and 'kind' in row]
    subject_residuals = {}
    for row in result.costs:
        for term in row.missing_terms:
            subject_residuals.setdefault(term, []).append(row.subject)
    ledger = dict(schema=SCHEMA, schema_version=1, scope_schedule=scope_schedule_ref,
        source_certificate=dict(artifact=scope_schedule_ref, locator='#/proof/implementation_scope_cleanup'),
        milestone=dict(subject=marker.subject, activity=activity, task_id=fence.id,
            restore_task_id=restore.id, additional_service=0, measured_zero=False, physical_zero_latency_claimed=False),
        terms=records, active_subjects=list(by_subject), residuals=residuals,
        subject_residuals=[dict(term=term, subjects=subjects, additional_work=False, cost_charged=False)
                           for term, subjects in subject_residuals.items()],
        numeric_unknown_subjects=[row.subject for row in result.costs if row.basis is CostBasis.UNKNOWN],
        source_frontiers=[row['term'] for row in records if row['disposition'] == 'SOURCE_FRONTIER'],
        summary=summary, original_demand_terms_accounted=len(records) == len(demand['terms']),
        original_activity_ownership_exactly_once=True, coarse_terms_are_nonadditive_descriptions=True,
        coarse_scope_numeric_equality_claimed=False, residuals_are_additional_work=False,
        complete_target_implementation_cost=False, physical_cost_closed=False,
        physical_optimality_proven=False, solver_authorized=False, execution_authorized=False)
    read_artifact(scope_schedule_ref)
    return result, ledger


# Sourced deployment cost arithmetic; never import or load a native artifact.
#
# Component sums are an affine deployment model, including when each component
# was observed. Complete-scope observations are replayed only at their own count
# and boundary; their difference is never used to manufacture setup costs.
_COMPONENTS = ('original_one_time', 'native_load', 'native_setup',
               'original_per_call', 'native_per_call')
_TOTALS = ('original_total', 'native_total')


def _wire(value):
    return asdict(value) if is_dataclass(value) else deepcopy(value)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def _pointer(value, locator):
    if locator == '#':
        return value
    if not isinstance(locator, str) or not locator.startswith('#/'):
        raise ValueError('JSON pointer required')
    for key in locator[2:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not key.isdigit() or str(int(key)) != key:
                raise ValueError('canonical array index required')
            value = value[int(key)]
        else:
            value = value[key]
    return value


def _fact(reference):
    reference = _wire(reference)
    if set(reference) != {'artifact', 'locator'}:
        raise ValueError('ProviderFactReference fields required')
    artifact = reference['artifact']
    if (set(artifact) != {'uri', 'sha256', 'size_bytes', 'schema'} or
            type(artifact['size_bytes']) is not int or artifact['size_bytes'] < 0 or
            not re.fullmatch('[0-9a-f]{64}', artifact['sha256']) or
            not isinstance(artifact['schema'], str) or not artifact['schema']):
        raise ValueError('valid original ProviderArtifact fields required')
    path = Path(artifact['uri'])
    if not path.is_absolute():
        raise ValueError('absolute retained local artifact required')
    raw = path.read_bytes()
    if len(raw) != artifact['size_bytes'] or sha256(raw).hexdigest() != artifact['sha256']:
        raise ValueError('retained artifact SHA/size differs')
    data = json.loads(raw)
    if data.get('schema') != artifact['schema']:
        raise ValueError('retained artifact schema differs')
    return _pointer(data, reference['locator']), _digest(artifact)


def _profile(value):
    value = _wire(value)
    if isinstance(value, (str, Path)):
        path = Path(value).absolute(); raw = path.read_bytes()
        return json.loads(raw), dict(kind='PATH', path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))
    if isinstance(value, dict) and set(value) == {'artifact', 'locator'}:
        document, _ = _fact(value)
        return document, dict(kind='FACT_REFERENCE', reference=value)
    return value, dict(kind='INLINE')


def _number(value):
    if type(value) not in (int, float) or value < 0:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return Fraction(str(value))


def _quantity(value):
    try:
        approximate = float(value)
        if not math.isfinite(approximate): approximate = None
    except OverflowError:
        approximate = None
    return dict(seconds=approximate, exact_seconds=dict(numerator=value.numerator, denominator=value.denominator))


def _range(original_setup, native_setup, original_call, native_call, gain, strict_minimum_gain):
    """Positive integer counts satisfying the exact gain inequality."""
    intercept = (1 - gain) * original_setup - native_setup
    slope = (1 - gain) * original_call - native_call
    strict = strict_minimum_gain or gain == 0
    if slope == 0:
        return dict(minimum=1, maximum=None) if (intercept > 0 if strict else intercept >= 0) else None
    crossing = -intercept / slope
    floor = crossing.numerator // crossing.denominator
    ceil = -(-crossing.numerator // crossing.denominator)
    if slope > 0:
        return dict(minimum=max(1, floor + 1 if strict else ceil), maximum=None)
    maximum = ceil - 1 if strict else floor
    return dict(minimum=1, maximum=maximum) if maximum >= 1 else None


def plan_deployment(profile, *, context, reuse_count, available_cpu_ids=None,
                    memory_budget_bytes=None, minimum_gain=DEFAULT_MINIMUM_GAIN, strict_minimum_gain=True):
    """Return a metadata-only NO_OP or permission to consider loading one candidate.

    This cost gate cannot authorize deployment or replace source, correctness,
    CPU affinity, memory, input, and external-contract admission at execution.
    """
    if type(strict_minimum_gain) is not bool:
        raise ValueError('minimum gain comparator must be a boolean')
    frontier, infeasible, cache = [], [], {}
    def issue(field, reason): frontier.append(dict(field=field, reason=reason))
    try:
        document, source = _profile(profile)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        document, source = None, dict(kind='INVALID')
        issue('profile', str(error))
    try:
        available = None if available_cpu_ids is None else sorted(available_cpu_ids)
    except TypeError:
        available = None
    request = dict(context=_wire(context), reuse_count=reuse_count,
        available_cpu_ids=available,
        memory_budget_bytes=memory_budget_bytes, minimum_gain=minimum_gain)
    if strict_minimum_gain:
        request['strict_minimum_gain'] = True
    if not isinstance(document, dict):
        document = {}; issue('profile', 'canonical deployment cost profile required')
    allowed = {'schema', 'schema_version', 'mode', 'context', 'context_evidence', 'costs',
               'observed_reuse_count', 'model_assumptions', 'provenance'}
    if set(document) - allowed: issue('profile', 'unsupported fields cannot be silently ignored')
    if document.get('schema') != 'scar.deployment-cost-profile' or type(document.get('schema_version')) is not int or document.get('schema_version') != 1:
        issue('profile.schema', 'scar.deployment-cost-profile v1 required')
    expected = document.get('context', {})
    if not isinstance(expected, dict): expected = {}; issue('context', 'explicit context required')
    if _digest(expected) != _digest(_wire(context)):
        issue('context', 'complete original context must match exactly; boundaries cannot be mixed')
    if expected.get('boundary') not in ('FRESH_PROCESS', 'WARM_SESSION'):
        issue('context.boundary', 'FRESH_PROCESS or WARM_SESSION required')
    for key in ('workload_id', 'artifact_id', 'memory_scope'):
        if not isinstance(expected.get(key), str) or not expected[key]: issue('context.' + key, 'explicit nonempty identity/scope required')
    for key in ('external_assumptions', 'runtime_guards'):
        if not isinstance(expected.get(key), list) or any(not isinstance(item, str) or not item for item in expected[key]):
            issue('context.' + key, 'explicit complete contract list required')
    cpus = expected.get('cpu_ids')
    if (not isinstance(cpus, list) or not cpus or any(type(v) is not int or v < 0 for v in cpus)
            or cpus != sorted(set(cpus))):
        issue('context.cpu_ids', 'sorted unique nonnegative CPU allocation required'); cpus = []
    available = request['available_cpu_ids']
    if available is None or any(type(v) is not int or v < 0 for v in available):
        issue('available_cpu_ids', 'explicit available CPU mask required')
    elif not set(cpus) <= set(available): infeasible.append('required_cpu_allocation_unavailable')
    required = expected.get('required_memory_bytes')
    if type(required) is not int or required < 0: issue('context.required_memory_bytes', 'explicit nonnegative byte requirement required')
    if type(memory_budget_bytes) is not int or memory_budget_bytes < 0:
        issue('memory_budget_bytes', 'explicit budget in the declared memory scope required; capacity is not available budget')
    elif type(required) is int and required > memory_budget_bytes: infeasible.append('declared_memory_budget_exceeded')
    if type(reuse_count) is not int or reuse_count < 1: issue('reuse_count', 'exact positive invocation count required')
    gain = _number(minimum_gain)
    if gain is None or gain >= 1: issue('minimum_gain', 'finite fraction in [0, 1) required'); gain = None

    def read(reference):
        key = _digest(_wire(reference))
        if key not in cache: cache[key] = _fact(reference)
        return cache[key]
    context_sources = set()
    context_references = document.get('context_evidence', [])
    if not isinstance(context_references, list):
        context_references = []; issue('context_evidence', 'explicit FactReference list required')
    for index, reference in enumerate(context_references):
        try:
            value, artifact = read(reference)
            if _digest(value) != _digest(expected): raise ValueError('original context value differs')
            context_sources.add(artifact)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            issue(f'context_evidence.{index}', str(error))
    if not context_sources: issue('context_evidence', 'matching original artifact context required')
    mode = document.get('mode')
    names = _COMPONENTS if mode == 'COMPONENTS' else _TOTALS if mode == 'COMPLETE_SCOPE' else ()
    if not names: issue('mode', 'COMPONENTS or COMPLETE_SCOPE required')
    costs = document.get('costs', {})
    if not isinstance(costs, dict): costs = {}; issue('costs', 'explicit cost records required')
    if set(costs) - set(names): issue('costs', 'unsupported cost terms cannot be ignored')
    values, bases = {}, {}
    for name in names:
        field = 'costs.' + name; record = costs.get(name)
        if not isinstance(record, dict): issue(field, 'unknown cost is not zero'); continue
        if set(record) - {'value', 'unit', 'basis', 'evidence'}: issue(field, 'unsupported cost constraints')
        number = _number(record.get('value')); basis = record.get('basis')
        if number is None or record.get('unit') != 'seconds' or basis not in ('OBSERVED', 'DECLARED_MODEL'):
            issue(field, 'finite nonnegative seconds and explicit evidence basis required'); continue
        references = record.get('evidence')
        if not isinstance(references, list) or not references: issue(field + '.evidence', 'original value/basis evidence required'); continue
        valid = True
        for index, reference in enumerate(references):
            try:
                source_value, artifact = read(reference)
                if artifact not in context_sources: raise ValueError('cost source lacks matching original boundary context')
                if (not isinstance(source_value, dict) or type(source_value.get('value')) not in (int, float)
                        or _number(source_value['value']) != number or source_value.get('unit') != 'seconds'
                        or source_value.get('basis') != basis):
                    raise ValueError('source value/unit/basis differs')
                direct = 'DIRECT_COMPONENT_TIMING' if mode == 'COMPONENTS' else 'DIRECT_COMPLETE_SCOPE_TIMING'
                if basis == 'OBSERVED' and source_value.get('measurement_scope') != direct:
                    raise ValueError('direct matching-scope timing required; differences of scope medians are not components')
                if mode == 'COMPLETE_SCOPE' and (type(source_value.get('reuse_count')) is not int or
                        source_value['reuse_count'] != document.get('observed_reuse_count')):
                    raise ValueError('complete-scope source reuse count differs')
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                valid = False; issue(f'{field}.evidence.{index}', str(error))
        if valid: values[name] = number; bases[name] = basis
    totals, qualifying, break_even = None, None, None
    if mode == 'COMPLETE_SCOPE' and (type(document.get('observed_reuse_count')) is not int or document['observed_reuse_count'] != reuse_count):
        issue('observed_reuse_count', 'complete-scope total is valid only for its original exact reuse count')
    count_matches = mode != 'COMPLETE_SCOPE' or document.get('observed_reuse_count') == reuse_count
    if names and len(values) == len(names) and count_matches and type(reuse_count) is int and reuse_count > 0 and gain is not None:
        if mode == 'COMPONENTS':
            original = values['original_one_time'] + reuse_count * values['original_per_call']
            native_setup = values['native_load'] + values['native_setup']
            native = native_setup + reuse_count * values['native_per_call']
            qualifying = _range(values['original_one_time'], native_setup, values['original_per_call'],
                                values['native_per_call'], gain, strict_minimum_gain)
            break_even = qualifying['minimum'] if qualifying and qualifying['maximum'] is None else None
        else: original, native = values['original_total'], values['native_total']
        totals = dict(original=_quantity(original), native=_quantity(native))
        threshold = (1 - gain) * original
        if original <= 0: issue('costs.original_total', 'positive original total required for a gain comparison')
        elif native >= original or native > threshold or (strict_minimum_gain and native == threshold):
            infeasible.append('cost_gain_below_required_threshold')
    assumptions = document.get('model_assumptions', [])
    if not isinstance(assumptions, list) or any(not isinstance(item, str) or not item for item in assumptions):
        issue('model_assumptions', 'explicit model assumptions must be a list of nonempty strings'); assumptions = []
    complete = not frontier
    candidate = complete and not infeasible and totals is not None
    return dict(schema='scar.deployment-cost-plan', schema_version=1,
        status=('UNRESOLVED' if not complete else 'COST_MODEL_EVALUATED' if mode == 'COMPONENTS' else 'COMPLETE_SCOPE_REPLAYED'),
        decision='LOAD_NATIVE_CANDIDATE' if candidate else 'NO_OP', selected=candidate,
        profile=document, profile_source=source, profile_sha256=_digest(document), request=request,
        context=deepcopy(expected), mode=mode, total_costs=totals, cost_bases=bases,
        qualifying_reuse_range=qualifying, break_even_reuse_count=break_even,
        whole_scope_cost_observed=mode == 'COMPLETE_SCOPE' and len(bases) == 2 and set(bases.values()) == {'OBSERVED'},
        setup_inferred_from_scope_difference=False,
        frontier=frontier, reasons=infeasible if infeasible else ([] if candidate else ['unresolved_cost_or_contract']),
        external_assumptions=deepcopy(expected.get('external_assumptions', [])),
        runtime_guards=deepcopy(expected.get('runtime_guards', [])),
        model_assumptions=deepcopy(assumptions) +
            (['AFFINE_REUSE_SUM_ASSUMES_STATIONARY_COMPONENT_COSTS_NOT_AN_OBSERVED_TOTAL'] if mode == 'COMPONENTS' else []),
        selection_before_heavy_load=dict(metadata_only=True, requires_heavy_load=False,
            candidate_may_be_loaded=candidate, execution_contract_must_be_revalidated=True),
        choice_scope='ORIGINAL_NO_OP_VS_ONE_DECLARED_NATIVE_ARTIFACT',
        deployment_authorized=False, target_variants_executed=False, physical_optimality_proven=False)


def verify_deployment_plan(report):
    """Replay retained evidence and cost arithmetic; this is not execution admission."""
    value = json.loads(Path(report).read_bytes()) if isinstance(report, (str, Path)) else deepcopy(report)
    source = value['profile_source']; profile = value['profile']
    if source['kind'] == 'PATH':
        raw = Path(source['path']).read_bytes()
        if len(raw) != source['bytes'] or sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('deployment profile receipt differs')
        profile = source['path']
    elif source['kind'] == 'FACT_REFERENCE': profile = source['reference']
    request = dict(value['request'])
    request.setdefault('strict_minimum_gain', False)
    expected = plan_deployment(profile, **request)
    if _digest(expected) != _digest(value): raise ValueError('deployment cost plan differs from original scope/evidence replay')
    return value


__all__ = [
    'SCHEMA',
    'project_scope_accounting',
    'plan_deployment',
    'verify_deployment_plan',
]
