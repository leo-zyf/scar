"""Compare repeated CSR updates inside one source-qualified installation scope."""

import argparse
from fractions import Fraction
import json
from math import ceil, floor
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_row_cost import SCHEMA as ROW_COST, _statistics

SCHEMA = 'scar.sparse-csr-scope-cost'
TIMING = 'scar.sparse-csr-scope-timing'
SECTION = 'scar.csr-enforce-section-calibration'
LIFETIME = 'scar.study-verified-source-loop-scope'
_MODES = ('original', 'native')
_TERMS = ('load_once', 'cold_first', 'warm')
_BOUND_FIELDS = ('estimate_ns', 'lower_ns', 'upper_ns')
_SETUP_ZERO = dict(estimate_ns=0, lower_ns=0, upper_ns=0,
    basis='CANDIDATE_ONLY_INSTALL_ACTIVITY_ABSENT_FROM_ORIGINAL')
_UNKNOWN = dict(estimate_ns=None, lower_ns=None, upper_ns=None, basis='UNKNOWN')


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _reference(value, schema=None):
    if type(value) is dict:
        return value
    path = Path(value).resolve()
    return artifact_reference(path, schema or json.loads(path.read_bytes())['schema'])


def _bound_term(samples):
    return dict(**_statistics(samples), basis='EMPIRICAL_SCOPE_SAMPLE_RANGE')


def _empty_terms():
    return {name:{mode:dict(_UNKNOWN) for mode in _MODES} for name in _TERMS}


def _section_timing(scope, calibration, context):
    raw = read_artifact(calibration)
    _need(calibration['schema'] == SECTION and raw['schema_version'] == 1, 'versioned section samples required')
    design = read_artifact(raw['design'])
    count = context['count']['value']
    launch = design['context']
    entry = context['entry']
    _need(count is not None and count > 1 and design['count'] == count
        and design['entry'] == entry['source'] and launch['argv'] == entry['argv']
        and launch['cpu_ids'] == entry['affinity']
        and all(launch['environment'][key] == value for key, value in entry['thread_environment'].items()),
        'the staged measurements must bind the exact installed entry/call/CPU scenario')
    harness = design['measurement_harness']
    read_artifact(harness, json_document=False)
    selection = read_artifact(design['selection'])
    target = context['f4_call']['target']
    proof = read_artifact(selection['proof'])
    _need(selection['source']['uri'] == target['path']
        and selection['source']['sha256'] == target['sha256']
        and proof['region']['code_key'] == target['code_key']
        and selection['entry_code']['proof'] == selection['proof'],
        'staged calibration must use the exact same source and sealed runtime code')
    read_artifact(selection['entry_code']['artifact'], json_document=False)
    for ref in selection['native_members']:
        read_artifact(ref, json_document=False)
    native = read_artifact(selection['native'])
    provider = native['provider']
    for ref in (*provider['sources'], provider['sparse_binary']):
        read_artifact(ref, json_document=False)
    adapter = Path(__file__).resolve().parents[1]/'runtime/sparse_csr.py'
    adapter_ref = artifact_reference(adapter, 'scar.python-source')
    _need(adapter_ref in selection['native_members'], 'the final guarded CSR adapter is not bound to samples')
    samples = dict(load_once=[], **{name:{mode:[] for mode in _MODES} for name in ('cold_first','warm')})
    _need(len(raw['samples']) == design['pairs'] >= 3, 'three or more fresh matched process pairs required')
    inputs = context['inputs']
    contents = None
    previous = 0
    for index, pair in enumerate(raw['samples']):
        order = list(_MODES if not index % 2 else _MODES[::-1])
        _need(pair['index'] == index and pair['order'] == order, 'frozen pair order differs')
        for mode in order:
            record = pair['workers'][mode]
            worker = read_artifact(record['artifact'])
            command = [launch['interpreter']['executable'], '-B', harness['uri'], 'worker',
                entry['source']['uri'], design['selection']['uri'], str(count), mode, record['artifact']['uri']]
            _need(worker['mode'] == mode and worker['count'] == count and worker['context'] == launch
                and record['command'] == command and worker['observed_rhs_and_diag_from_original_ast'] is True,
                'section worker differs from frozen argv, runtime, or context')
            row = worker['contents']
            if contents is None:
                contents = row
            _need(row == contents and row['shape'] == inputs['matrix_shape'] and row['nnz'] == inputs['nnz']
                and row['selected_rows'] == inputs['row_count'],
                'section loses original CSR geometry or paired outputs differ')
            clocks = worker['timings_ns']
            _need(len(clocks) == count, 'all source-ordered section intervals must be retained')
            for start, stop in clocks:
                _need(type(start) is int and type(stop) is int and previous <= start < stop,
                    'every direct section interval must be monotone and serial')
                previous = stop
            first = clocks[0][1]-clocks[0][0]
            if mode == 'native':
                activation = worker['activation']
                _need(activation['attempts'] == 1 and activation['available'] is True
                    and worker['stats'] == dict(native_calls=count, fallback_calls=0, reasons={})
                    and 0 < activation['load_ns'] < first, 'invalid nested loader or native count')
                samples['load_once'].append(activation['load_ns'])
                first -= activation['load_ns']
            else:
                _need(worker['activation'] is None and worker['stats'] is None, 'original scope unexpectedly attached')
            samples['cold_first'][mode].append(first)
            samples['warm'][mode].append(round(Fraction(sum(stop-start for start, stop in clocks[1:]), count-1)))
    return dict(schema=TIMING, schema_version=1, scope=scope, context=context,
        section_calibration=calibration, native_artifact=selection['native'], adapter_source=adapter_ref,
        measurement_harness=harness, samples_ns=samples,
        measured_scope='ORIGINAL_CALL_ARGS_AND_SEALED_ENFORCE_SECTION_WITH_ONE_NESTED_PROVIDER_LOAD',
        qualification=dict(one_installation_per_activation=True, native_fallback_calls=0,
            active_scope_violations=0, native_updates_per_activation=count, complete_covered_scope=False,
            section_excludes=['CALLER_IMPORT_HOOK_INSTALLATION', 'LINEAR_SOLVE', 'OUTPUT_DIGEST',
                'COMPLETE_ENTRY_LAUNCH_AND_SHUTDOWN']),
        cold_first='SAME_CALL_CLOCK_SPAN_EXCLUDING_ITS_NESTED_MEASURED_LOAD_ONCE',
        warm='PER_ACTIVATION_MEAN_OF_EVERY_POST_FIRST_SECTION_INTERVAL')


def derive_section_timing(scope, calibration):
    scope_ref = _reference(scope, LIFETIME)
    return _section_timing(scope_ref, _reference(calibration, SECTION), _scope_context(scope_ref))


def _read_timing(reference, scope, context):
    data = read_artifact(reference)
    _need(reference['schema'] == TIMING and data['schema_version'] == 1
        and data['scope'] == scope and data['context'] == context,
        'retained timing belongs to another source/installation/device scope')
    for key in ('native_artifact', 'adapter_source', 'measurement_harness'):
        read_artifact(data[key], json_document=False)
    if data.get('section_calibration') is not None:
        _need(data == _section_timing(scope, data['section_calibration'], context),
            'retained section timings differ from the original worker clocks')
    _need(data['native_artifact']['schema'] == 'scar.native-csr-enforce'
        and data['adapter_source'] == artifact_reference(
            Path(__file__).resolve().parents[1]/'runtime/sparse_csr.py', 'scar.python-source'),
        'timings must pin the guarded native use of the current CSR backend')
    observed = data['qualification']
    exact_count = context['count']['value']
    _need(type(exact_count) is int and exact_count > 0
        and observed['one_installation_per_activation'] is True
        and observed['native_fallback_calls'] == observed['active_scope_violations'] == 0
        and observed['native_updates_per_activation'] == exact_count
        and type(observed['complete_covered_scope']) is bool,
        'a matched original/native section with exact active-scope call counts is needed')
    recorded = data['samples_ns']
    terms = _empty_terms()
    terms['load_once']['original'] = dict(_SETUP_ZERO)
    if recorded.get('load_once') is not None:
        terms['load_once']['native'] = _bound_term(recorded['load_once'])
    for name in ('cold_first', 'warm'):
        if name == 'warm' and exact_count == 1:
            continue
        for mode in _MODES:
            values = recorded.get(name, {}).get(mode)
            if values is not None:
                terms[name][mode] = _bound_term(values)
    return terms, dict(native_artifact=data['native_artifact'], adapter_source=data['adapter_source'],
        measurement_harness=data['measurement_harness'], qualification=observed,
        measured_scope=data['measured_scope'])


def _total(terms, mode, count, field):
    values = [terms['load_once'][mode][field], terms['cold_first'][mode][field]]
    if count > 1:
        warm = terms['warm'][mode][field]
        values.append((count-1)*warm if warm is not None else None)
    return sum(values) if None not in values else None


def _call_domain(terms, original_metric, native_metric, factor, count):
    """Solve the inequality on positive call counts and clip to the source case."""
    orig = [terms[name]['original'][original_metric] for name in _TERMS]
    native = [terms[name]['native'][native_metric] for name in _TERMS]
    if None in (*orig, *native):
        return None
    # factor*(original fixed + k*original warm) >= native fixed + k*native warm.
    deficit = sum(native[:2]) - factor*sum(orig[:2])
    slope = factor*orig[2] - native[2]
    low, high = 1, None
    strict = factor == 1
    if slope > 0:
        threshold = floor(deficit/slope)+1 if strict else ceil(deficit/slope)
        low = max(low, threshold+1)
    elif slope < 0:
        high = ceil(deficit/slope) if strict else floor(deficit/slope)+1
    elif deficit > 0 or (strict and deficit == 0):
        return dict(positive_count_interval=None, source_case_intersection=None)
    if high is not None and high < low:
        return dict(positive_count_interval=None, source_case_intersection=None)
    applies = count is not None and low <= count and (high is None or count <= high)
    return dict(positive_count_interval=[low, high],
        source_case_intersection=[count, count] if applies else None)


def _comparison(count, terms, gain):
    factor = 1 - gain
    totals = {mode:{field:_total(terms, mode, count, field) if count is not None else None
        for field in _BOUND_FIELDS} for mode in _MODES}
    original, native = (totals[mode] for mode in _MODES)
    lower, upper = original['lower_ns'], native['upper_ns']
    qualifies = (lower is not None and upper is not None and lower > 0
        and upper < lower and upper <= factor*lower)
    first, second = original['estimate_ns'], native['estimate_ns']
    expected_gain = None if first is None or second is None or first == 0 else float(Fraction(first-second, first))
    envelope_gain = None if lower is None or upper is None or lower == 0 else float(Fraction(lower-upper, lower))
    break_even = dict(
        point=_call_domain(terms, 'estimate_ns', 'estimate_ns', factor, count),
        sample_envelope=_call_domain(terms, 'lower_ns', 'upper_ns', factor, count),
        extrapolation_authorized=False)
    return totals, break_even, expected_gain, envelope_gain, qualifies


def _scope_context(reference):
    report = read_artifact(reference)
    _need(reference['schema'] == LIFETIME and report['schema_version'] == 1,
        'versioned source lifetime receipt required')
    entry, loop, call = (report[key] for key in ('entry', 'f4_loop', 'f4_call'))
    read_artifact(entry['source'], json_document=False)
    observed, lifetime = (report[key] for key in ('observed','lifetime'))
    requested = entry['argv_repetitions']
    value = None
    if (type(requested) is int and requested > 0
        and observed['completed_body_count'] == observed['region_attempts'] == observed['consumer_completions'] == requested
        and observed['scope_entry_count'] == observed['scope_exit_count'] == 1
        and loop['static_exact_condition']['repetitions'] == requested
        and loop['static_exact_condition']['executions_per_normal_iteration'] == 1
        and entry['argv_positive_guard'] and observed['exact_count_evidence']
        and observed['normal_completion'] is True
        and lifetime['admissible_installations_per_entry'] == 1
        and lifetime['loads_per_scope_is_required_runner_guarantee'] is True):
        evidence = read_artifact(observed['exact_count_evidence'])
        _need(evidence['entry'] == entry['source'] and evidence['argv'] == entry['argv']
            and evidence['affinity'] == entry['affinity'] and evidence['loop_inputs'] == observed['loop_inputs'],
            'observed counts belong to another source or runtime case')
        read_artifact(call['source_witness'])
        read_artifact(call['source_graph'])
        value = requested
    count = dict(value=value, requested_count=requested,
        observed_count=observed['region_attempts'], source_case_only=True,
        lower=value, upper=value, conditional_normal_completion=True)
    return dict(scope=reference, entry=entry, count=count, f4_loop=loop, f4_call=call,
        installation=lifetime, inputs=observed['loop_inputs'])


def derive_sparse_scope_cost(scope, warm_region=None, *, terms=None, minimum_gain=0):
    """Build a sourced setup + first + recurrent cost decision without timing work."""
    gain = Fraction(str(minimum_gain))
    _need(0 <= gain < 1, 'minimum gain must fall in [0, 1)')
    scope_ref = _reference(scope)
    context = _scope_context(scope_ref)
    count = context['count']['value']
    warm_ref = None if warm_region is None else _reference(warm_region, ROW_COST)
    reference = None
    if warm_ref is not None:
        prior = read_artifact(warm_ref)
        _need(warm_ref['schema'] == ROW_COST, 'versioned warm-region receipt required')
        reference = dict(artifact=warm_ref, original_scope=prior['measurement_scope'],
            data_shape=prior['observations']['matrix_shape'],
            timings={mode:{field:prior['timings'][mode][field] for field in _BOUND_FIELDS}
                for mode in _MODES}, admitted_as_repeated_scope_cost=False)
    terms_ref = None if terms is None else _reference(terms, TIMING)
    estimates, admitted = _empty_terms(), None
    missing = []
    if count is None:
        missing.append('EXACT_SOURCE_COUNT_FOR_CURRENT_ACTIVATION')
    if terms_ref is None:
        missing.extend(('FINAL_ONCE_LOADING_AND_INSTALLATION_COST_RANGE',
            'MATCHED_ORIGINAL_AND_NATIVE_COLD_FIRST_COSTS',
            'REPEATED_INVOCATION_COSTS_WITH_SAME_SOURCE_GUARD_SCOPE'))
    elif count is not None:
        estimates, admitted = _read_timing(terms_ref, scope_ref, context)
        required = _TERMS if count > 1 else _TERMS[:-1]
        missing.extend('MISSING_SCOPE_COST:'+name+':'+mode for name in required for mode in _MODES
            if estimates[name][mode]['estimate_ns'] is None)
        if not admitted['qualification']['complete_covered_scope']:
            missing.append('MEASURED_SECTION_TRANSFER_TO_COMPLETE_ACTIVE_ENTRY_SCOPED_WORK')
    totals, break_even, gain_point, gain_range, profitable = _comparison(count, estimates, gain)
    profitable = profitable and not missing
    return dict(schema=SCHEMA, schema_version=1, scope=scope_ref, warm_region=warm_ref, timing=terms_ref,
        context=context, count=context['count'], terms=estimates, original_region_reference=reference,
        admitted_timing_implementations=admitted, minimum_gain=float(gain),
        totals=totals, break_even=break_even,
        decision='LOAD_NATIVE_CANDIDATE' if profitable else 'KEEP',
        qualification=dict(status='SCOPED_SAMPLE_ENVELOPE_GAIN' if profitable else 'KEEP',
            point_gain=gain_point, conservative_sample_envelope_gain=gain_range,
            sample_envelope_is_physical_bound=False, missing_terms=missing,
            other_entry_services_inferred_zero=False, whole_entry_timing_qualified=False,
            bound_to_one_completed_installation_scope=True),
        resource_demands=dict(native_region_cpu_threads=1,
            loader_and_first_call_must_follow_entry_installation=True,
            physical_cpu_availability=None, numa_pages=None, concurrent_payload_and_rss=None),
        deployment_authorized=False)


def check_sparse_scope_cost(report):
    reference = _reference(report, SCHEMA)
    saved = read_artifact(reference)
    expected = derive_sparse_scope_cost(saved['scope'], saved['warm_region'],
        terms=saved['timing'], minimum_gain=saved['minimum_gain'])
    _need(saved == expected, 'CSR scope cost differs from source lifetime and original timing replay')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope')
    parser.add_argument('--warm-region')
    parser.add_argument('--timing')
    parser.add_argument('--minimum-gain', default='0')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        result = check_sparse_scope_cost(args.check)
    elif args.scope and args.out:
        result = derive_sparse_scope_cost(args.scope, args.warm_region, terms=args.timing,
            minimum_gain=args.minimum_gain)
        path = Path(args.out).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write('\n')
    else:
        parser.error('--scope and --out required')
    print(json.dumps(dict(decision=result['decision'], count=result['count'], qualification=result['qualification']),
        sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
