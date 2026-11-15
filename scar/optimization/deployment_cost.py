"""Sourced deployment cost arithmetic; never import or load a native artifact.

Component sums are an affine deployment model, including when each component
was observed. Complete-scope observations are replayed only at their own count
and boundary; their difference is never used to manufacture setup costs.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re


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


def _range(original_setup, native_setup, original_call, native_call, gain):
    """Positive integer counts satisfying the exact gain inequality."""
    intercept = (1 - gain) * original_setup - native_setup
    slope = (1 - gain) * original_call - native_call
    strict = gain == 0  # A zero gain threshold still requires an improvement.
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
                    memory_budget_bytes=None, minimum_gain=0.05):
    """Return a metadata-only NO_OP or permission to consider loading one candidate.

    This cost gate cannot authorize deployment or replace source, correctness,
    CPU affinity, memory, input, and external-contract admission at execution.
    """
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
            qualifying = _range(values['original_one_time'], native_setup, values['original_per_call'], values['native_per_call'], gain)
            break_even = qualifying['minimum'] if qualifying and qualifying['maximum'] is None else None
        else: original, native = values['original_total'], values['native_total']
        totals = dict(original=_quantity(original), native=_quantity(native))
        if original <= 0: issue('costs.original_total', 'positive original total required for a gain comparison')
        elif native >= original or native > (1 - gain) * original: infeasible.append('cost_gain_below_required_threshold')
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
    expected = plan_deployment(profile, **value['request'])
    if _digest(expected) != _digest(value): raise ValueError('deployment cost plan differs from original scope/evidence replay')
    return value


__all__ = ['plan_deployment', 'verify_deployment_plan']
