"""Partition normal compiler boundaries with an explicit helper-destructor cut.

The pool is absent from the canonical probe. Helper cleanup and ABI/Python
handoff are separate samples so their original intervening pool releases stay
in the source graph. No target DAG or whole target call is executed.
"""
import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import sys
import time

from scar.compiler import calibrate_boundary_cuts as base
from scar.compiler.calibrate_stateful_boundary import _canonical_binding, _json, _now, _receipt
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.stateful_cpu_model import _domain
from scar.runtime.stateful_contract import validate_contract

SCHEMA = 'scar.stateful-scope-boundary-calibration'
CUTS = (*base.CUTS[:-1], 'helper_cleanup_ready', 'outer_return')
PHASES = (*deepcopy(base.PHASES[:-1]),
    dict(id='helper_cleanup', start='restore_ready', end='helper_cleanup_ready',
        phase_kinds=['NATIVE_HELPER_CLEANUP']),
    dict(id='return_handoff', start='helper_cleanup_ready', end='outer_return',
        phase_kinds=['NATIVE_RETURN', 'PYTHON_STATUS_RETURN']))
CONDITIONS = (*base.CONDITIONS,
    'ORIGINAL_HELPER_DESTRUCTION_PRECEDES_WORKSPACE_POOL_RELEASE',
    'EXPLICIT_HELPER_CLEANUP_CUT_SEPARATES_DESTRUCTION_FROM_ABI_PYTHON_HANDOFF')
MISSING = (*base.MISSING, 'HELPER_SCOPE_REASSEMBLY_AND_ADDITIONAL_CLOCK_CUT_DELTA',
    'ABSENT_WORKSPACE_DESTRUCTION_CAN_CHANGE_CANONICAL_HANDOFF_REGIME')
_EXTRA_FILES = ('compiler/calibrate_scope_boundary.py', 'analysis/implementation_scope.py',
    'codegen/scope_boundary_probe.py', 'codegen/stateful_scope_boundary_v1.cpp.in')


def _pins():
    return {**base._pins(), **{name: base.sha256((base._ROOT/name).read_bytes()).hexdigest()
                             for name in _EXTRA_FILES}}


def _spec(domain, candidate_id):
    from scar.codegen.scope_boundary_probe import render_scope_boundary_probe
    spec = base._spec(domain, candidate_id)
    demand = base.stateful_cpu_demand(domain, candidate_id)
    parameters = next(row['parameters'] for row in demand['terms']
        if row['component'] == 'GUARDED_METHOD_AND_NATIVE_ARTIFACT_BOUNDARY')
    source, provenance = render_scope_boundary_probe(parameters, spec['memory'], spec['cpus'], spec['core'])
    return {**spec, 'source': source, 'provenance': provenance}


def scope_boundary_statistics(document):
    """Validate nine cuts, then reuse the same exclusive interval algebra."""
    base._need((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1)
        and document.get('cuts') == list(CUTS) and document.get('phases') == list(PHASES)
        and document.get('conditions') == list(CONDITIONS) and document.get('missing_terms') == list(MISSING),
        'source-scoped helper cleanup cuts and applicability contract required')
    repetitions, warmup = document.get('repetitions'), document.get('warmup')
    base._need(type(repetitions) is int and 3 <= repetitions <= 10000
        and type(warmup) is int and 1 <= warmup <= 10000, 'warm repeated scope samples required')
    samples = document.get('samples')
    base._need(type(samples) is list and len(samples) == repetitions,
        'complete original scoped boundary sample vectors required')
    base._need(document.get('clock') == dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
        'one native/Python integer monotonic clock required')
    identities = set(); columns = [[] for _ in PHASES]
    for row in samples:
        identity, cuts = row.get('id'), row.get('timestamps_ns')
        base._need(type(identity) is str and identity and identity not in identities,
            'unique scope boundary sample identity required')
        identities.add(identity)
        base._need(type(cuts) is list and len(cuts) == len(CUTS)
            and all(type(value) is int and value > 0 for value in cuts)
            and all(left <= right for left, right in zip(cuts, cuts[1:])),
            'complete monotone integer scope cut vector required')
        deltas = [right-left for left, right in zip(cuts, cuts[1:])]
        base._need(sum(deltas) == cuts[-1]-cuts[0], 'scope cuts must exactly partition each sample')
        for values, delta in zip(columns, deltas):
            values.append(delta)
    return [dict(phase=phase['id'], samples_ns=values, median_ns=base.statistics.median(values),
        min_ns=min(values), max_ns=max(values), stdev_ns=base.statistics.stdev(values))
        for phase, values in zip(PHASES, columns)]


def calibrate_scope_boundary(domain, candidate_id, out, *, repetitions=7, warmup=2):
    from scar.codegen.stateful import StatefulArtifact, _providers
    from scar.compiler.stateful_runtime import _ArtifactAdapter
    from scar.runtime.stateful_guard import guarded_stateful_function
    from scar.runtime.compiled_deployment import thread_environment
    base._need(sys.gettrace() is None and sys.getprofile() is None, 'scope sampling requires no trace/profile hook')
    base._need(type(repetitions) is int and 3 <= repetitions <= 10000
        and type(warmup) is int and 1 <= warmup <= 10000, 'warm repeated scoped samples required')
    started = time.monotonic(); observed_from = _now()
    domain = _domain(domain); spec = _spec(domain, candidate_id)
    actual = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    base._need(actual == domain['live_constraints']['process_affinity_cpu_ids']
        and set(spec['cpus']) <= set(actual)
        and environment == domain['live_constraints']['native_thread_environment'],
        'canonical scope caller context differs from original domain')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    original, args, inputs, callees, providers, structure = _canonical_binding(spec['region'], spec['plan'], out)
    library, table, probe = base._build(spec, out/'native')
    probe['implementation_files'] = _pins()
    contract = base._contract(original, inputs, callees, providers, spec['region'], spec, probe, out)
    (out/'canonical-contract.json').write_bytes(_json(contract))
    runtime, native_providers, owners, power = _providers(spec['region'])
    base._need(runtime == spec['core']['provider_runtime'], 'original provider binary context differs')
    artifact = StatefulArtifact(spec['region'], spec['plan'], out, dict(contract['native']['ids']),
        library, table, native_providers, owners, power)
    adapter = object.__new__(_ArtifactAdapter)
    adapter.original, adapter.contract, adapter.report = artifact, contract, artifact.report
    wrapper = guarded_stateful_function(original, adapter, owner=type(args[0]),
        exclusive_inputs=True, selection=contract['selection'])
    reader = library.scar_boundary_cut_read_v1
    reader.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]; reader.restype = ctypes.c_int32
    native = (ctypes.c_uint64*7)(); samples = []; load_before = list(os.getloadavg())
    try:
        for rep in range(repetitions+warmup):
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            result = wrapper(*args)
            end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            base._need(result is None and reader(native, 7) == 0, 'canonical scope call or native cut collection failed')
            if rep >= warmup:
                samples.append(dict(id='canonical-scope-call:'+str(rep-warmup), timestamps_ns=[start, *native, end]))
        base._need(wrapper.scar_stats['native_calls'] == repetitions+warmup
            and wrapper.scar_stats['fallback_calls'] == 0, 'actual compiler scope path was not traversed')
    finally:
        after = sorted(os.sched_getaffinity(0))
        if after != actual:
            os.sched_setaffinity(0, actual)
    base._need(after == actual and thread_environment() == environment, 'scope caller context was not restored')
    document = dict(schema=SCHEMA, schema_version=1, domain_id=domain['domain_id'], candidate_id=candidate_id,
        domain_source=dict(region=domain['region'], stage_plan=domain['base_stage_plan'], target=domain['target_snapshot']),
        resource_context=spec['resource_context'], cuts=list(CUTS), phases=deepcopy(list(PHASES)),
        samples=samples, repetitions=repetitions, warmup=warmup,
        conditions=list(CONDITIONS), missing_terms=list(MISSING),
        clock=dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
        probe=probe, canonical_structure=structure, canonical_runtime_contract=_receipt(out/'canonical-contract.json'),
        runtime_path=['scar.runtime.stateful_guard.guarded_stateful_function.call',
            'scar.compiler.stateful_runtime._ArtifactAdapter.run', 'scar.codegen.stateful.StatefulArtifact.run'],
        runtime_stats=deepcopy(wrapper.scar_stats), observed_from=observed_from, observed_until=_now(),
        seconds=time.monotonic()-started, host_load=dict(before=load_before, after=list(os.getloadavg()), isolated=False),
        observed_caller_affinity_before=actual, observed_caller_affinity_after=after,
        observed_native_thread_environment=environment, caller_affinity_restored=True,
        exclusive_partition_verified=True, helper_cleanup_separated=True, pool_release_measured=False,
        target_workload_executed=False, target_variants_ranked=False, target_variants_compiled=False,
        whole_target_call_measured=False, clock_overhead_subtracted=False, physical_optimality_proven=False,
        deployment_authorized=False)
    document['phase_statistics'] = scope_boundary_statistics(document)
    path = out/'observations.json'; path.write_bytes(_json(document))
    result = dict(schema='scar.stateful-scope-boundary-result', schema_version=1,
        observations=artifact_reference(path, SCHEMA), phases=document['phase_statistics'],
        helper_cleanup_separated=True, target_workload_executed=False, deployment_authorized=False)
    (out/'result.json').write_bytes(_json(result)); return result


def read_scope_boundary_calibration(reference, *, domain, candidate_id):
    """Replay source fragments, cleanup scope, actual wrapper and raw cut algebra."""
    if isinstance(reference, (str, Path)):
        reference = artifact_reference(Path(reference).resolve(), SCHEMA)
    document = read_artifact(reference)
    domain = _domain(domain) if isinstance(domain, (str, Path)) else domain
    spec = _spec(domain, candidate_id)
    base._need(document.get('phase_statistics') == scope_boundary_statistics(document),
        'scope statistics differ from the original exclusive cuts')
    base._need(document.get('domain_id') == domain['domain_id'] and document.get('candidate_id') == candidate_id
        and document.get('domain_source') == dict(region=domain['region'], stage_plan=domain['base_stage_plan'], target=domain['target_snapshot'])
        and document.get('resource_context') == spec['resource_context'], 'scope source/candidate/resource context differs')
    probe = document['probe']
    base._need(probe.get('implementation_files') == _pins() and probe.get('source_fragments') == spec['provenance']
        and probe.get('compiler') == spec['compiler'] and probe.get('target_dag_compiled') is False,
        'scope observer implementation provenance differs')
    base._need(set(probe['headers']) == {'stateful_abi_v1.h', 'scar_abi_v1.h'}, 'original ABI headers required')
    for receipt in (probe['source'], probe['library'], *probe['headers'].values(),
                    document['canonical_runtime_contract'], document['canonical_structure']['canonical_source']):
        base._need(_receipt(receipt['path']) == receipt, 'retained scope artifact changed')
    base._need(Path(probe['source']['path']).read_text() == spec['source']
        and all(Path(value['path']).read_bytes() == (base._ROOT/'codegen'/name).read_bytes()
                for name, value in probe['headers'].items()), 'scope probe source/headers differ from replay')
    base._need(probe.get('compiler_command') == [*spec['compiler']['argv'], *spec['compiler']['options'],
        probe['source']['path'], '-o', probe['library']['path'], '-lm'], 'scope compiler invocation differs')
    contract = validate_contract(json.loads(Path(document['canonical_runtime_contract']['path']).read_bytes()))
    structure = document['canonical_structure']
    base._need(contract['native']['contract'] == spec['core'] and contract['cpu']['ids'] == spec['cpus']
        and contract['native']['library_sha256'] == probe['library']['sha256']
        and contract['parameters'] == list(spec['region'].parameter_names) == structure['parameters']
        and contract['inputs'] == structure['inputs']
        and [row['path'] for row in contract['inputs']] == [list(row.path) for row in spec['region'].inputs]
        and structure['source_body'] == 'RAISE_IF_FALLBACK_NO_TARGET_DAG'
        and contract['source']['sha256'] == structure['canonical_source']['sha256'],
        'canonical scope runtime/source structure differs')
    base._need(document.get('runtime_path') == ['scar.runtime.stateful_guard.guarded_stateful_function.call',
        'scar.compiler.stateful_runtime._ArtifactAdapter.run', 'scar.codegen.stateful.StatefulArtifact.run']
        and document.get('runtime_stats', {}).get('native_calls') == document['repetitions']+document['warmup']
        and document['runtime_stats'].get('fallback_calls') == 0, 'scope observations must use the actual compiler path')
    base._need(document.get('observed_caller_affinity_before') == spec['resource_context']['caller_affinity_cpu_ids']
        and document.get('observed_caller_affinity_after') == document['observed_caller_affinity_before']
        and document.get('observed_native_thread_environment') == spec['resource_context']['native_thread_environment']
        and document.get('caller_affinity_restored') is True, 'scope placement/runtime observations differ')
    base._need(document.get('exclusive_partition_verified') is True and document.get('helper_cleanup_separated') is True,
        'explicit exclusive helper cleanup partition required')
    for name in ('pool_release_measured', 'target_workload_executed', 'target_variants_ranked',
                 'target_variants_compiled', 'whole_target_call_measured', 'clock_overhead_subtracted',
                 'physical_optimality_proven', 'deployment_authorized'):
        base._need(document.get(name) is False, 'scoped canonical model cannot claim '+name)
    read_artifact(reference); return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True); parser.add_argument('--candidate', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--out'); source.add_argument('--check')
    parser.add_argument('--repetitions', type=int, default=7); parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args(argv)
    if args.check:
        document = read_scope_boundary_calibration(args.check, domain=args.domain, candidate_id=args.candidate)
        result = dict(phases=document['phase_statistics'], source_replayed=True,
            helper_cleanup_separated=True, target_workload_executed=False)
    else:
        result = calibrate_scope_boundary(args.domain, args.candidate, args.out,
            repetitions=args.repetitions, warmup=args.warmup)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
