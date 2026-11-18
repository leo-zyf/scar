"""Matched complete canonical entry costs for one retained implementation.

The existing ELF is a macro cost primitive.  Its real guard/adapter/packing,
native body, allocation and cleanup execute together, with no internal clocks
or sums of isolated medians.  Canonical transfer remains a hypothesis until a
separate, subsequently frozen real-entry holdout validates it.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import CodeType, SimpleNamespace
import uuid

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

DESIGN = 'scar.stateful-canonical-entry-design'
OBSERVATIONS = 'scar.stateful-canonical-entry-calibration'
SCHEMA = 'scar.stateful-canonical-entry-model'
FACTS = 'scar.stateful-canonical-entry-facts'
BASIS = 'MATCHED_CANONICAL_GUARDED_ENTRY_POINT_MODEL'
_ROUNDING = 'NEAREST_INTEGER_NANOSECOND_TIES_TO_EVEN'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(value, schema=None):
    if type(value) is dict:
        return deepcopy(value)
    path = Path(value).absolute()
    return artifact_reference(path, schema or json.loads(path.read_bytes())['schema'])


def _same(left, right):
    return semantic_sha256(left) == semantic_sha256(right)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _pins():
    root = Path(__file__).resolve().parents[1]
    names = ('compiler/calibrate_stateful_entry.py', 'compiler/canonical_stateful_inputs.py',
        'analysis/stateful_entry_protocol.py', 'compiler/stateful_runtime.py',
        'compiler/export_stateful.py', 'compiler/native_lowering_view.py',
        'compiler/stateful_model.py', 'optimization/stateful_cpu_choices.py',
        'runtime/stateful_guard.py', 'runtime/stateful_contract.py',
        'trace/code_snapshot.py', 'codegen/stateful.py', 'codegen/cpp.py')
    return {name:artifact_reference(root/name, 'scar.implementation-source') for name in names}


def _materials(specialization):
    from scar.compiler.native_lowering_view import derive_stateful_lowering_view
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.compiler.export_stateful import project_stateful_contract
    from scar.optimization.stateful_cpu_choices import _domain, stateful_cpu_cost_context

    spec = read_artifact(specialization)
    _need((spec.get('schema'), spec.get('schema_version')) ==
        ('scar.observed-stateful-specialization', 1), 'original observed specialization required')
    candidate = read_artifact(spec['candidate'])
    domain = _domain(read_artifact(spec['domain']))
    row = next((row for row in domain['candidates'] if row['id'] == candidate['candidate_id']), None)
    _need(row is not None and row['legal_under_guards'] and row['worker_count'] == 1
        and candidate['program'] == spec['program'], 'one admitted serial source candidate required')
    native = candidate['native_artifact']
    view = derive_stateful_lowering_view(native)
    region = stateful_from_dict(read_artifact(view['region']))
    plan = read_artifact(view['plan']); metadata = read_artifact(native)
    _need(plan['region_sha256'] == domain['region_sha256']
        and view['actual_resource_binding'] == row['binding'],
        'native frame and original domain candidate differ')
    artifact = SimpleNamespace(region=region, plan=plan, report=metadata,
        directory=Path(native['uri']).parent)
    contract = project_stateful_contract(artifact, None, require_selection=False)
    return spec, candidate, domain, view, contract, stateful_cpu_cost_context(domain, row['id'])


def _derive(request):
    from scar.analysis.stateful_entry_protocol import derive_stateful_entry_protocol
    from scar.compiler.canonical_stateful_inputs import bind_canonical_stateful_inputs
    from scar.compiler.stateful_holdout import TIMER
    from scar.runtime.compiled_deployment import thread_environment

    _need(set(request) == {'specialization', 'warmup', 'repetitions', 'array_value',
        'scalar_value', 'max_external_bytes', 'capture_epoch'}, 'complete canonical request required')
    _need(type(request['warmup']) is int and request['warmup'] == 2
        and type(request['repetitions']) is int and request['repetitions'] == 7,
        'two warmups and seven fixed measurement calls required')
    _need(type(request['max_external_bytes']) is int and request['max_external_bytes'] > 0,
        'positive canonical input byte budget required')
    _need(all(type(request[name]) is float and math.isfinite(request[name])
        for name in ('array_value', 'scalar_value')), 'finite explicit canonical floats required')
    _need(type(request['capture_epoch']) is str and len(request['capture_epoch']) == 32
        and all(value in '0123456789abcdef' for value in request['capture_epoch']),
        'prepared canonical capture epoch required')
    spec, candidate, domain, view, contract, context = _materials(request['specialization'])
    protocol = derive_stateful_entry_protocol(candidate['native_artifact'],
        trial=True, exclusive_inputs=True, selection=None)
    bound = bind_canonical_stateful_inputs(contract, **{name:request[name] for name in
        ('array_value', 'scalar_value', 'max_external_bytes')})
    return dict(context=context, domain=spec['domain'], candidate_id=candidate['candidate_id'],
        native_manifest=candidate['native_artifact'], source_native_view=view,
        contract=contract, entry_protocol=protocol, canonical_binding=bound['metadata'],
        implementation_sources=_pins(), timer=deepcopy(TIMER),
        runtime=dict(python_executable=sys.executable, python_version=sys.version,
            caller_affinity_cpu_ids=sorted(os.sched_getaffinity(0)),
            native_thread_environment=thread_environment()),
        conditions=['CANONICAL_FINITE_FLOAT_INPUTS_TRANSFER_TO_ACTUAL_SOURCE_INPUT_REGIME',
            'ONE_PERSISTENT_GUARD_AFTER_TWO_WARMUP_CALLS',
            'SAME_ENTRY_PROTOCOL_ELF_CPU_SHAPE_LAYOUT_SCALAR_TAG_AND_SOURCE',
            'RESET_CANONICAL_ARRAYS_OUTSIDE_TIMER_NO_PER_STEP_CONTENT_RECORDING',
            'CACHE_NUMA_ALLOCATOR_PREDECESSOR_DATA_AND_CONTENTION_NOT_PHYSICAL_GUARANTEES',
            'INDEPENDENT_REAL_ENTRY_HOLDOUT_REQUIRED_BEFORE_TIMING_USE'],
        original_f4_source_admission_replayed=False,
        original_f4_replay_required_by_enclosing_source_controller=True,
        preparation_target_executed=False, compiled=False, native_loaded=False,
        physical_feasibility_proven=False, deployment_authorized=False)


def prepare_canonical_entry_calibration(specialization, out, *, array_value=1.25,
        scalar_value=0.75, max_external_bytes=64*2**20):
    request = dict(specialization=_ref(specialization), warmup=2, repetitions=7,
        array_value=array_value, scalar_value=scalar_value,
        max_external_bytes=max_external_bytes, capture_epoch=uuid.uuid4().hex)
    derived = _derive(request)
    directory = Path(out).absolute(); directory.mkdir(parents=True, exist_ok=False)
    document = dict(schema=DESIGN, schema_version=1, created_at=_now(),
        request=request, **derived)
    return _write(directory/'design.json', document)


def read_canonical_entry_design(path, *, expected_sha256=None):
    reference = _ref(path, DESIGN); document = read_artifact(reference)
    _need(expected_sha256 is None or reference['sha256'] == expected_sha256,
        'frozen canonical entry design identity differs')
    _need((document.get('schema'), document.get('schema_version')) == (DESIGN, 1),
        'versioned canonical entry design required')
    expected = dict(schema=DESIGN, schema_version=1, created_at=document['created_at'],
        request=document['request'], **_derive(document['request']))
    _need(_same(document, expected), 'canonical entry source/context/protocol differs from cold replay')
    return document


def _summary(samples):
    _need(type(samples) is list and len(samples) == 7, 'seven canonical measurement calls required')
    previous = -1; durations = []
    for index, sample in enumerate(samples):
        _need(sample['ordinal'] == index and all(type(sample[name]) is int
            for name in ('ordinal', 'start_ns', 'end_ns', 'elapsed_ns', 'native_calls_after', 'fallback_calls'))
            and previous <= sample['start_ns'] < sample['end_ns']
            and sample['elapsed_ns'] == sample['end_ns']-sample['start_ns']
            and sample['native_calls_after'] == index+3 and sample['fallback_calls'] == 0,
            'original monotone clocks and exact successful-call counters required')
        previous = sample['end_ns']; durations.append(sample['elapsed_ns'])
    rational = dict(numerator=sum(durations), denominator=len(durations))
    return dict(exact_mean_ns=rational, estimate_ns=round(Fraction(**rational)),
        integer_rounding=_ROUNDING, min_ns=min(durations), max_ns=max(durations),
        samples_ns=durations, sample_range_is_guaranteed_bound=False)


def run_canonical_entry_calibration(path, *, expected_sha256):
    import numpy as np
    from scar.codegen.stateful import load
    from scar.compiler.canonical_stateful_inputs import bind_canonical_stateful_inputs
    from scar.compiler.stateful_runtime import _ArtifactAdapter
    from scar.runtime.stateful_guard import guarded_stateful_function
    from scar.runtime.compiled_deployment import thread_environment
    from scar.trace.code_snapshot import code_key

    reference = _ref(path, DESIGN)
    design = read_canonical_entry_design(reference, expected_sha256=expected_sha256)
    directory = Path(reference['uri']).parent/'run'; directory.mkdir(exist_ok=False)
    request = design['request']; bound = bind_canonical_stateful_inputs(design['contract'],
        **{name:request[name] for name in ('array_value', 'scalar_value', 'max_external_bytes')})
    _need(_same(bound['metadata'], design['canonical_binding']), 'canonical binding changed')
    artifact = load(design['native_manifest']['uri'])
    adapter = _ArtifactAdapter(artifact, None)
    _need(_same(adapter.contract, design['contract']), 'loaded complete entry contract differs')

    def fatal(reason):
        raise RuntimeError('canonical source fallback is forbidden: '+reason)

    wrapper = guarded_stateful_function(bound['original'], adapter, owner=bound['owner'],
        exclusive_inputs=True, selection=None, trial=True, on_fallback=fatal)
    closure = inspect.getclosurevars(wrapper).nonlocals
    _need(closure['artifact'] is adapter and closure['original'] is bound['original']
        and closure['owner'] is bound['owner'] and closure['trial'] is True
        and closure['exclusive_inputs'] is True and closure['selection'] is None,
        'actual persistent guard closure differs from the frozen successful entry protocol')
    shared_call = next(code for code in guarded_stateful_function.__code__.co_consts
        if type(code) is CodeType and code.co_name == 'call')
    _need(code_key(wrapper.__code__) == code_key(shared_call), 'actual successful guard code differs')
    before = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    oracle_arrays = {name:value.copy() for name,value in bound['arrays'].items()}
    artifact.run(oracle_arrays, dict(bound['scalars']))
    samples = []; observed_from = _now()
    for ordinal in range(request['warmup']+request['repetitions']):
        bound['reset']()
        start = time.perf_counter_ns()
        result = wrapper(*bound['args'], **bound['kwargs'])
        end = time.perf_counter_ns()
        _need(result is None and wrapper.scar_stats['native_calls'] == ordinal+1
            and wrapper.scar_stats['fallback_calls'] == 0, 'complete canonical native call failed')
        _need(sorted(os.sched_getaffinity(0)) == before and thread_environment() == environment,
            'native call changed caller affinity or native thread environment')
        if ordinal >= request['warmup']:
            samples.append(dict(ordinal=ordinal-request['warmup'], start_ns=start, end_ns=end,
                elapsed_ns=end-start, native_calls_after=wrapper.scar_stats['native_calls'], fallback_calls=0))
    _need(all(np.array_equal(value, oracle_arrays[name]) for name,value in bound['arrays'].items()),
        'canonical full guard path differs from direct ABI oracle with independent buffers')
    read_canonical_entry_design(reference, expected_sha256=expected_sha256)
    document = dict(schema=OBSERVATIONS, schema_version=1, design=reference,
        capture_epoch=request['capture_epoch'], observation_origin='CANONICAL_NATIVE_EXECUTION',
        observed_from=observed_from, observed_until=_now(), samples=samples, statistics=_summary(samples),
        entry_protocol=design['entry_protocol'], canonical_binding=design['canonical_binding'],
        timer=design['timer'], runtime=design['runtime'], native_manifest=design['native_manifest'],
        guard_code_key=code_key(wrapper.__code__), native_calls=wrapper.scar_stats['native_calls'],
        fallback_calls=0, independent_direct_native_oracle_calls=1, oracle_outputs_bitwise_equal=True,
        caller_affinity_before=before, caller_affinity_after=sorted(os.sched_getaffinity(0)),
        native_thread_environment=environment, original_python_body_executed=False,
        target_entry_program_executed=False, target_region_dag_executed=True,
        compiled=False, original_elf_reused=True, internal_observer_clocks_inserted=False,
        per_step_array_contents_recorded=False, clock_overhead_subtracted=False,
        variants_ranked=False, holdout_samples_fitted=False, qualified_profile=None,
        deployment_authorized=False)
    return _write(directory/'observations.json', document)


def _calibration(reference):
    from scar.runtime.stateful_guard import guarded_stateful_function
    from scar.trace.code_snapshot import code_key
    observed = read_artifact(reference)
    _need((observed.get('schema'), observed.get('schema_version')) == (OBSERVATIONS, 1),
        'complete canonical entry observations required')
    design = read_canonical_entry_design(observed['design'])
    _need(observed['observation_origin'] == 'CANONICAL_NATIVE_EXECUTION'
        and observed['capture_epoch'] == design['request']['capture_epoch']
        and _same(observed['statistics'], _summary(observed['samples']))
        and type(observed['native_calls']) is int and observed['native_calls'] == 9
        and type(observed['fallback_calls']) is int and observed['fallback_calls'] == 0
        and observed['oracle_outputs_bitwise_equal'] is True
        and type(observed['independent_direct_native_oracle_calls']) is int
        and observed['independent_direct_native_oracle_calls'] == 1,
        'canonical clocks, frozen epoch or complete native-call validation differs')
    shared_call = next(code for code in guarded_stateful_function.__code__.co_consts
        if type(code) is CodeType and code.co_name == 'call')
    _need(observed['guard_code_key'] == code_key(shared_call),
        'observed persistent successful guard code differs from the implementation')
    for key in ('entry_protocol', 'canonical_binding', 'timer', 'runtime', 'native_manifest'):
        _need(_same(observed[key], design[key]), 'canonical observation differs from design: '+key)
    _need(observed['caller_affinity_before'] == observed['caller_affinity_after'] ==
        design['runtime']['caller_affinity_cpu_ids']
        and observed['native_thread_environment'] == design['runtime']['native_thread_environment'],
        'canonical observed caller context differs')
    for key in ('compiled', 'original_python_body_executed', 'target_entry_program_executed',
        'internal_observer_clocks_inserted', 'per_step_array_contents_recorded',
        'clock_overhead_subtracted', 'variants_ranked', 'holdout_samples_fitted', 'deployment_authorized'):
        _need(observed[key] is False, 'canonical entry must not claim '+key)
    _need(observed['target_region_dag_executed'] is True and observed['original_elf_reused'] is True
        and observed['qualified_profile'] is None, 'explicit native calibration without qualification required')
    return observed, design


def _facts(reference):
    observed, design = _calibration(reference)
    return dict(schema=FACTS, schema_version=1, context=deepcopy(design['context']),
        estimate=dict(duration_ticks=observed['statistics']['estimate_ns'], tick_seconds=1e-9),
        derivation=dict(origin='DECLARED_FORWARD_MODEL', basis=BASIS, target_variant_trials=False,
            entry_protocol=deepcopy(design['entry_protocol']), calibration=reference,
            assumptions=deepcopy(design['conditions']), canonical_complete_entry_measured=True,
            one_existing_implementation_measured_on_canonical_inputs=True,
            target_region_dag_executed=True, target_entry_program_executed=False,
            implementation_search_performed=False,
            target_entry_holdout_used_to_fit=False, independent_holdout_required=True,
            component_medians_summed=False, whole_target_latency_guaranteed=False))


def _profile(facts, reference):
    fact = lambda locator:dict(artifact=reference, locator=locator)
    return dict(schema='scar.stateful-cpu-cost-profile', schema_version=1,
        context=deepcopy(facts['context']), context_evidence=[fact('#/context')],
        estimate={**deepcopy(facts['estimate']), 'evidence':[fact('#/estimate')]},
        derivation={**deepcopy(facts['derivation']), 'evidence':[fact('#/derivation')]})


def verify_canonical_entry_profile(profile):
    _need(profile.get('derivation', {}).get('basis') == BASIS, 'matched canonical entry basis required')
    reference = profile['derivation']['calibration']; facts = _facts(reference)
    fact_ref = profile['context_evidence'][0]['artifact']
    _need(_same(read_artifact(fact_ref), facts) and _same(profile, _profile(facts, fact_ref)),
        'canonical entry profile differs from complete design and raw clock replay')
    return facts


def _model(calibration, facts_ref, profile_ref, facts):
    observed, design = _calibration(calibration)
    return dict(schema=SCHEMA, schema_version=1, specialization=design['request']['specialization'],
        domain=design['domain'], candidate_id=design['candidate_id'], calibration=calibration,
        facts=facts_ref, profile=profile_ref, duration_ns=facts['estimate']['duration_ticks'],
        entry_protocol=deepcopy(design['entry_protocol']),
        estimate_origin='MATCHED_CANONICAL_COMPLETE_ENTRY_COMPONENT',
        independent_real_entry_qualified=False, qualified_profile=None,
        automatic_deployment_authorized=False, default_backend='NO_OP')


def prepare_canonical_entry_model(observations, out):
    calibration = _ref(observations, OBSERVATIONS); facts = _facts(calibration)
    directory = Path(out).absolute(); directory.mkdir(parents=True, exist_ok=False)
    facts_ref = _write(directory/'facts.json', facts)
    profile_ref = _write(directory/'profile.json', _profile(facts, facts_ref))
    result = _model(calibration, facts_ref, profile_ref, facts)
    _write(directory/'result.json', result)
    return result


def read_canonical_entry_model(path):
    reference = _ref(path, SCHEMA); model = read_artifact(reference)
    _need((model.get('schema'), model.get('schema_version')) == (SCHEMA, 1), 'canonical entry model required')
    facts = _facts(model['calibration'])
    _need(_same(read_artifact(model['facts']), facts)
        and _same(read_artifact(model['profile']), _profile(facts, model['facts']))
        and _same(model, _model(model['calibration'], model['facts'], model['profile'], facts)),
        'complete canonical model differs from source/protocol/raw clock replay')
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--specialization'); source.add_argument('--run-design')
    source.add_argument('--model-observations'); source.add_argument('--check')
    parser.add_argument('--sha256'); parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.specialization:
        if not args.out:parser.error('--out required')
        result = prepare_canonical_entry_calibration(args.specialization, args.out)
    elif args.run_design:
        if not args.sha256 or args.out:parser.error('--sha256 required; design owns its output directory')
        result = run_canonical_entry_calibration(args.run_design, expected_sha256=args.sha256)
    elif args.model_observations:
        if not args.out:parser.error('--out required')
        result = prepare_canonical_entry_model(args.model_observations, args.out)
        result = dict(model=_ref(Path(args.out)/'result.json', SCHEMA),
            duration_ns=result['duration_ns'], qualified_profile=None)
    else:
        if args.out:parser.error('--check is read-only')
        model = read_canonical_entry_model(args.check)
        result = dict(model=_ref(args.check, SCHEMA), duration_ns=model['duration_ns'],
            cold_replayed=True, qualified_profile=None)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
