"""Replay three source-matched CSR section alternatives and their scoped costs."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_scope_cost import (
    _BOUND_FIELDS, _SETUP_ZERO, _bound_term, _comparison, _need, _reference, _scope_context,
)

SCHEMA = 'scar.csr-topology-scope-cost'
CALIBRATION = 'scar.csr-topology-section-calibration'
TOPOLOGY_SCOPE = 'scar.csr-topology-scope'
MODES = ('original', 'native', 'topology')
TERMS = ('load_once', 'cold_first', 'plan_build_once', 'warm', 'release_once')
_ORDER = (MODES, MODES[1:] + MODES[:1], MODES[2:] + MODES[:2])


def _phases(worker, mode, count):
    clocks = worker['timings_ns']
    _need(len(clocks) == count and all(type(start) is int and type(stop) is int and 0 < start < stop
        for start, stop in clocks) and all(left[1] <= right[0] for left, right in zip(clocks, clocks[1:])),
        'complete monotone intervals required in each section worker')
    first = clocks[0][1]-clocks[0][0]
    warm = round(Fraction(sum(stop-start for start, stop in clocks[1:]), count-1))
    activation = worker['activation']
    if mode == 'original':
        _need(activation is None and worker['stats'] is None and worker['topology'] is None
            and worker['owned_metadata_released'] is False,
            'the original path must execute without a native installation')
        return dict(load_once=0, cold_first=first, plan_build_once=0, warm=warm, release_once=0)
    _need(activation['attempts'] == 1 and activation['available'] is True
        and worker['stats'] == dict(native_calls=count, fallback_calls=0, reasons={})
        and 0 < activation['load_ns'] < first, 'exactly one load and no native fallbacks required')
    load = activation['load_ns']
    if mode == 'native':
        _need(worker['topology'] is None and worker['release_ns'] is None
            and worker['owned_metadata_released'] is False, 'stepwise validation must own no topology plan')
        return dict(load_once=load, cold_first=first-load, plan_build_once=0, warm=warm, release_once=0)
    topology = worker['topology']
    _need(topology['build_attempts'] == topology['builds'] == 1 and topology['hits'] == count-1
        and topology['misses'] == topology['numeric_rejections'] == 0
        and worker['owned_metadata_released'] is True
        and 0 < topology['build_ns'] < first-load, 'one measured build before exact hot hits is required')
    start, stop = worker['release_ns']
    _need(type(start) is int and type(stop) is int and clocks[-1][1] <= start < stop,
        'the measured metadata release must follow its final consumer')
    return dict(load_once=load, cold_first=first-load-topology['build_ns'], plan_build_once=topology['build_ns'],
        warm=warm, release_once=stop-start)


def _documents(scope_ref, calibration_ref):
    topology = read_artifact(scope_ref)
    _need(scope_ref['schema'] == TOPOLOGY_SCOPE and topology['schema_version'] == 1,
        'versioned source topology scope required')
    scope = topology['source_scope']
    context = _scope_context(scope)
    entry, count = context['entry'], context['count']['value']
    _need(type(count) is int and count > 1 and topology['entry']['source'] == entry['source']
        and topology['entry']['argv'] == entry['argv'], 'topology cost requires a finite exact source case')
    calibration = read_artifact(calibration_ref)
    _need(calibration_ref['schema'] == CALIBRATION and calibration['schema_version'] == 1,
        'versioned independent calibration required')
    design = read_artifact(calibration['design'])
    launch = design['context']
    _need(design['topology_scope'] == scope_ref and design['entry'] == entry['source']
        and launch['argv'] == entry['argv'] and launch['cpu_ids'] == entry['affinity']
        and design['count'] == count and design['trials'] == len(calibration['batches']) == 3
        and all(launch['environment'][key] == value for key, value in entry['thread_environment'].items()),
        'scope, CPU placement or calibration differs from real source lifetime evidence')
    for key in ('harness', 'original_section_harness'):
        read_artifact(design[key], json_document=False)
    selection = read_artifact(design['selection'])
    budget = design['budget_bytes']
    _need(selection['proof']['sha256'] == topology['region_proof']['sha256']
        and selection['topology'] == dict(proof=scope_ref, budget_bytes=budget)
        and selection['entry_code']['proof'] == selection['proof']
        and type(budget) is int and budget > 0,
        'one sealed and memory-budgeted topology execution required')
    read_artifact(topology['region_proof'])
    read_artifact(selection['entry_code']['artifact'], json_document=False)
    native = read_artifact(selection['native'])
    runner = Path(__file__).resolve().parents[1]/'runtime/csr_region_entry.py'
    adapter = runner.with_name('sparse_csr.py')
    runner_ref, adapter_ref = (artifact_reference(path, 'scar.python-source') for path in (runner,adapter))
    _need(design['runner_source'] == runner_ref and adapter_ref in selection['native_members'],
        'measurement must bind the exact Python installation and guard bytes')
    for ref in (*selection['native_members'], *native['provider']['sources'], native['provider']['sparse_binary']):
        read_artifact(ref, json_document=False)
    return topology, context, calibration, design, selection, runner_ref, adapter_ref


def _read_observations(calibration, design, context):
    count = design['count']
    launch = design['context']
    stats = {mode:{term:[] for term in TERMS} for mode in MODES}
    footprint = []
    oracle = None
    previous = 0
    for index, batch in enumerate(calibration['batches']):
        _need(batch['order'] == list(_ORDER[index]), 'three-way rotation order differs from precommit')
        for mode in batch['order']:
            record = batch['workers'][mode]
            reference = record['artifact']
            worker = read_artifact(reference)
            command = [launch['interpreter']['executable'], '-B', design['harness']['uri'], 'worker',
                calibration['design']['uri'], mode, reference['uri']]
            _need(worker['mode'] == mode and worker['count'] == count and worker['context'] == launch
                and record['command'] == command, 'worker differs from frozen command and runtime')
            contents = worker['contents']
            if oracle is None:
                oracle = contents
            inputs = context['inputs']
            _need(contents == oracle and contents['shape'] == inputs['matrix_shape']
                and contents['nnz'] == inputs['nnz'] and contents['selected_rows'] == inputs['row_count'],
                'the three workers must share exact original inputs and matching output hashes')
            clocks = worker['timings_ns']
            _need(previous <= clocks[0][0], 'section workers must run sequentially')
            phases = _phases(worker, mode, count)
            for name, elapsed in phases.items():
                stats[mode][name].append(elapsed)
            if mode == 'topology':
                footprint.append(worker['topology']['retained_bytes'])
                previous = worker['release_ns'][1]
            else:
                previous = clocks[-1][1]
    _need(all(type(value) is int and 0 < value <= design['budget_bytes'] for value in footprint)
        and len(set(footprint)) == 1, 'every topology activation must retain the same bounded payload')
    terms = {mode:{term:_bound_term(values) if any(values) else dict(_SETUP_ZERO)
        for term, values in phases.items()} for mode, phases in stats.items()}
    return terms, footprint[0], oracle


def _compare(count, terms, left, right, gain):
    pair = {name:{m:{} for m in ('original','native')} for name in ('load_once','cold_first','warm')}
    for kind, mode in (('original',left), ('native',right)):
        row = terms[mode]
        for field in _BOUND_FIELDS:
            # All one-time phases feed the existing setup + cold + recurrent model.
            pair['load_once'][kind][field] = sum(row[term][field]
                for term in ('load_once','plan_build_once','release_once'))
            for phase in ('cold_first','warm'):
                pair[phase][kind][field] = row[phase][field]
    totals, break_even, point, envelope, passes = _comparison(count, pair, gain)
    spending = {left:totals['original'], right:totals['native']}
    left_upper = spending[left]['upper_ns']
    right_lower = spending[right]['lower_ns']
    reverse_passes = left_upper < right_lower and left_upper <= (1-gain)*right_lower
    return spending, dict(point_gain=point, conservative_sample_envelope_gain=envelope,
        break_even=break_even,
        preferred_by_sample_envelope=right if passes else left if reverse_passes else None)


def derive_csr_topology_cost(scope, calibration, *, minimum_gain=0):
    gain = Fraction(str(minimum_gain))
    _need(0 <= gain < 1, 'minimum gain must be in [0, 1)')
    top_ref, cal_ref = _reference(scope, TOPOLOGY_SCOPE), _reference(calibration, CALIBRATION)
    topology, context, observed, design, selection, runner, adapter = _documents(top_ref, cal_ref)
    count = context['count']['value']
    terms, owned, identity = _read_observations(observed, design, context)
    totals = {}
    comparisons = {}
    for left,right in (('original','native'),('original','topology'),('native','topology')):
        spending, comparison = _compare(count, terms, left, right, gain)
        totals.update(spending)
        comparisons[left+'_vs_'+right] = comparison
    wins = [row['preferred_by_sample_envelope'] for row in comparisons.values()]
    preferred = next((mode for mode in MODES if wins.count(mode) == 2), None)
    inputs = context['inputs']
    rows = inputs['matrix_shape'][0]
    key_bytes = (inputs['nnz']*inputs['index_width'] + (rows+1)*inputs['pointer_width']
        + inputs['row_count']*int(inputs['rows_dtype'][2:]))
    _need(owned >= key_bytes, 'the retained payload must cover all exact guard key copies')
    return dict(schema=SCHEMA, schema_version=1, topology_scope=top_ref, source_scope=topology['source_scope'],
        entry=context['entry'], count=count, count_evidence=context['count'],
        launch_context=design['context'], minimum_gain=float(gain),
        calibration=cal_ref, study_selection=design['selection'], native_artifact=selection['native'],
        runner_source=runner, adapter_source=adapter, terms=terms, totals=totals, comparisons=comparisons,
        resource_demands=dict(cpu_ids=context['entry']['affinity'], execution_lanes=1,
            memory_domain='HOST_NATIVE_PRIVATE_METADATA', topology_budget_bytes=design['budget_bytes'],
            topology_peak_owned_payload_bytes=owned, exact_guard_key_payload_bytes=key_bytes,
            topology_other_metadata_payload_bytes=owned-key_bytes,
            exact_guard_logical_compare_operand_bytes=key_bytes*2,
            retained_from='FIRST_PLAN_BUILD_UNTIL_SCOPE_FINISH',
            owned_payload_bytes_by_mode=dict(original=0,native=0,topology=owned),
            setup_and_warm_phases_must_serialize=True,
            physical_rss_bytes=None, numa_residency=None, dram_bytes=None),
        comparison_fixture=identity,
        qualification=dict(status='MATCHED_GUARDED_SECTION_EMPIRICAL_MODEL', preferred_section_option=preferred,
            options={mode:dict(measured_in_exact_section=True, complete_entry_timing_qualified=False)
                for mode in MODES},
            measured_scope='SOURCE_ENFORCE_CALL_WITH_FINAL_ADAPTER',
            exclusions=['CALLER_INSTALLATION_AND_IMPORT_CONTROL', 'LINEAR_SOLVES', 'PROGRAM_STARTUP_AND_EXIT'],
            sample_envelope_is_physical_bound=False, warm_includes_exact_structure_alias_and_numeric_guards=True,
            complete_entry_timing_qualified=False),
        decision='KEEP', deployment_authorized=False)


def check_csr_topology_cost(report):
    saved = read_artifact(_reference(report, SCHEMA))
    expected = derive_csr_topology_cost(saved['topology_scope'], saved['calibration'],
        minimum_gain=saved['minimum_gain'])
    _need(saved == expected, 'CSR topology costs differ from complete source and three-way clock replay')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope')
    parser.add_argument('--calibration')
    parser.add_argument('--out')
    parser.add_argument('--minimum-gain', default='0')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        result = check_csr_topology_cost(args.check)
    elif args.scope and args.calibration and args.out:
        result = derive_csr_topology_cost(args.scope, args.calibration, minimum_gain=args.minimum_gain)
        with Path(args.out).open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write('\n')
    else:
        parser.error('--scope, --calibration and --out required')
    print(json.dumps({key:result[key] for key in ('totals','comparisons','qualification','decision')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
