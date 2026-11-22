"""Select between two matched standalone ATen CPU call-boundary trials."""

from fractions import Fraction
import json
from statistics import variance

from scar.ir.execution_recipe import read_artifact
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.optimization.implementation_costs import SCHEMA as COST_PROFILE, CostQuery, query_key
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN, _metrics


EVIDENCE = 'scar.aten-boundary-trial-evidence'
SCOPE = 'DECLARED_NATIVE_ATEN_CPU_CALL_BOUNDARY'
_MODES = ('SERIAL', 'TWO_CPU')
_CONTEXT_FIELDS = {'boundary', 'provider', 'target', 'inputs', 'output', 'placements'}
_EVIDENCE_FIELDS = {'schema', 'schema_version', 'boundary', 'context', 'warmup_pairs', 'measured_pairs', 'samples'}
_OBSERVATIONS = ('bitwise_identical', 'input_values_unchanged', 'fresh_owned_outputs')
_SAMPLE_FIELDS = {'pair_index', 'order', 'serial_ns', 'dual_ns', *_OBSERVATIONS}
ATEN_GAIN_POLICY = dict(minimum_gain=DEFAULT_MINIMUM_GAIN, strict_minimum_gain=True)


def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _passes_gain(incumbent_ns, candidate_ns, policy):
    if policy is None:
        return (incumbent_ns - candidate_ns) / incumbent_ns >= .05
    _require(policy == ATEN_GAIN_POLICY and policy.get('strict_minimum_gain') is True,
             'unsupported ATen gain policy')
    return Fraction(incumbent_ns - candidate_ns, incumbent_ns) > Fraction(str(policy['minimum_gain']))


def _canonical(document):
    return json.dumps(encode(document), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _measurements(rows, repetitions):
    _require(type(rows) is list and len(rows) == repetitions,
             'exact post-warmup measured pair rows required')
    samples = {mode: [] for mode in _MODES}
    directional = {tuple(_MODES): [], tuple(reversed(_MODES)): []}
    differences = []
    previous = None
    for index, row in enumerate(rows):
        _require(type(row) is dict and set(row) == _SAMPLE_FIELDS
                 and type(row['pair_index']) is int and row['pair_index'] == index
                 and type(row['order']) is list and row['order'] in (list(_MODES), list(reversed(_MODES)))
                 and all(row[key] is True for key in _OBSERVATIONS),
                 'paired trial changed its order, validation or output contract')
        serial, dual = row['serial_ns'], row['dual_ns']
        _require(all(type(ns) is int and 0 < ns < 2**60 for ns in (serial, dual)),
                 'positive bounded integer nanosecond observations required')
        order = tuple(row['order'])
        _require(order != previous, 'measured native trial order must alternate AB and BA')
        previous = order
        samples['SERIAL'].append(serial)
        samples['TWO_CPU'].append(dual)
        difference = serial - dual
        differences.append(difference)
        directional[order].append(difference)
    return samples, differences, directional


def project_aten_boundary_trial(evidence_ref, *, expected_context, minimum_gain=DEFAULT_MINIMUM_GAIN,
                                strict_minimum_gain=True):
    """Replay pinned evidence as exact-context costs for the two trial modes.

    The producer establishes the native runtime receipt and validation checks
    before publication. This reducer checks the promised boundary, context,
    and sample inventory against that receipt. Its costs stay in SCOPE.
    """
    _require(type(evidence_ref) in (dict, ProviderArtifact), 'pinned ATen boundary trial evidence required')
    reference = encode(evidence_ref)
    _require(reference.get('schema') == EVIDENCE,
             'pinned ATen boundary trial evidence required')
    evidence = read_artifact(reference)
    _require(set(evidence) == _EVIDENCE_FIELDS and type(evidence['schema_version']) is int
             and evidence['schema_version'] == 1, 'supported ATen boundary trial evidence required')
    context, boundary = evidence['context'], evidence['boundary']
    _require(type(context) is dict and _CONTEXT_FIELDS <= context.keys()
             and type(boundary) is str and bool(boundary) and context['boundary'] == boundary
             and all(type(context[k]) is dict and context[k] for k in ('target', 'provider', 'output'))
             and type(context['inputs']) in (list, dict) and bool(context['inputs']),
             'pinned boundary, provider, target and input/output geometry required')
    placements = context['placements']
    _require(type(placements) is dict and set(placements) == set(_MODES)
             and all(type(row) is dict and row for row in placements.values())
             and placements['SERIAL'] != placements['TWO_CPU'] and 'trial_mode' not in context,
             'distinct explicit placements for the two native trial modes required')
    _require(type(expected_context) is dict and _canonical(context) == _canonical(expected_context),
             'original timing context or boundary differs from the requested native trial')
    warmup, repetitions = evidence['warmup_pairs'], evidence['measured_pairs']
    _require(type(warmup) is int and warmup >= 1 and type(repetitions) is int
             and repetitions >= 3 and repetitions % 2 == 1,
             'at least one warmup pair and an odd number of at least three measured pairs required')
    _require(type(minimum_gain) in (float, int) and 0 <= minimum_gain < 1 and type(strict_minimum_gain) is bool,
             'minimum gain must be a finite fraction in [0,1) with an explicit comparison')
    threshold = Fraction(str(minimum_gain))
    samples, differences, directional = _measurements(evidence['samples'], repetitions)
    metrics = {mode: dict(_metrics(s), variance_ns2=variance(s)) for mode, s in samples.items()}
    serial, dual = (metrics[m]['median_ns'] for m in _MODES)
    gain = Fraction(serial - dual, serial)
    wins = sum(n > 0 for n in differences)
    by_order = {
        ('AB' if order[0] == 'SERIAL' else 'BA'): _metrics(values)
        for order, values in directional.items()
    }
    reasons = []
    if gain <= 0 or gain < threshold or (strict_minimum_gain and gain == threshold):
        reasons.append('MEASURED_MEDIAN_GAIN_BELOW_THRESHOLD')
    if 4 * wins < 3 * repetitions:
        reasons.append('TWO_CPU_NOT_FASTER_IN_THREE_QUARTERS_OF_PAIRS')
    if any(row['median_ns_twice'] <= 0 for row in by_order.values()):
        reasons.append('NONPOSITIVE_PAIRED_MEDIAN_BENEFIT_IN_AT_LEAST_ONE_ORDER')

    facts = (dict(artifact=reference, locator='#/samples'),)
    entries, queries = {}, {}
    for mode in _MODES:
        query = CostQuery(SCOPE, dict(context, trial_mode=mode), facts)
        queries[mode] = query
        measured = metrics[mode]
        entries[query_key(query)] = dict(query=dict(scope=query.scope, context=query.context), basis='EMPIRICAL_MODEL',
            estimate_ns=measured['median_ns'], lower_ns=measured['min_ns'], upper_ns=measured['max_ns'],
            observations=dict(warmup=warmup, repetitions=repetitions, samples_ns=measured['samples_ns']),
            missing_terms=[], conditions=['EXACT_STANDALONE_ATEN_TRIAL_CONTEXT', 'POST_WARMUP_PAIRED_SAMPLES'])
    profile = dict(schema=COST_PROFILE, schema_version=1, entries=entries)
    selection = dict(schema='scar.aten-boundary-trial-selection', schema_version=1,
        scope=SCOPE, boundary=boundary, original_evidence=reference,
        selected_mode='SERIAL' if reasons else 'TWO_CPU', reasons=reasons,
        metrics=metrics, paired_difference_ns=differences, paired_wins=wins, differences_by_order=by_order,
        median_gain=float(gain), median_speedup=serial / dual, warmup_pairs=warmup, measured_pairs=repetitions,
        policy=dict(minimum_gain=minimum_gain, minimum_paired_win_fraction=0.75,
                    positive_order_group_medians_required=True),
        sample_ranges_are_physical_bounds=False, python_replacement_authorized=False, deployment_authorized=False)
    if strict_minimum_gain:
        selection['policy']['strict_minimum_gain'] = True
    return selection, profile, queries


__all__ = ['EVIDENCE', 'SCOPE', 'project_aten_boundary_trial']
