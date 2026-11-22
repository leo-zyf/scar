"""Compare two native implementations of the same ordered SHA-256 output."""

import argparse
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from statistics import variance
from struct import pack
import subprocess
from tempfile import TemporaryFile
from time import perf_counter_ns

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_solve_cost import _duration, _summary

SCHEMA = 'scar.native-ordered-publication-choice'
DESIGN = SCHEMA + '-design'
OBSERVATION = SCHEMA + '-observation'
SOURCE_INPUT = 'scar.native-source-csr-input-observation'
BASE, FUSED = 'BINARY_NATIVE_CONSUMER', 'IN_PROCESS_SHA256'
ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1'}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _member(root, reference):
    path = root / reference['uri']
    _require(path.resolve(strict=True) == path and path.is_relative_to(root),
             'native publication must use a regular package member')
    return {**reference, 'uri': str(path)}


def _packages(base, fused):
    from scar.codegen.native_ordered_sparse_export import SCHEMA as EXPORT, check_native_ordered_export

    roots = [Path(path).resolve(strict=True) for path in (base, fused)]
    references = [_reference(root / 'artifact.json', EXPORT) for root in roots]
    original, candidate = [check_native_ordered_export(root) for root in roots]
    for key in ('recipe', 'recipe_id', 'numeric', 'plan', 'qualified_release', 'detached_source'):
        _require(original[key] == candidate[key], 'publication variants differ in ' + key)
    _require((original['schema_version'], candidate['schema_version']) == (4, 5),
             'publication comparison requires the native source CSR and digest packages')
    old_stream, new_stream = [read_artifact(_member(root, manifest['stream']['protocol']))
                              for root, manifest in zip(roots, (original, candidate), strict=True)]
    for key in ('recipe_id', 'resource_plan_sha256', 'source', 'input', 'header'):
        _require(old_stream[key] == new_stream[key], 'publication variants differ in input ' + key)
    plan = read_artifact(_member(roots[0], original['plan']))
    release = read_artifact(_member(roots[0], original['qualified_release']))
    executables = [_member(root, manifest['stream']['executable'])
                   for root, manifest in zip(roots, (original, candidate), strict=True)]
    return references, executables, original, candidate, old_stream, new_stream, plan, release


def _source(input_ref, manifest, stream, release):
    source = read_artifact(input_ref)
    binding = source['stream']
    reference = _reference(binding['uri'], 'scar.native-source-csr-input-file')
    header = stream['header']
    _require(source['schema_version'] == 1 and source['recipe'] == manifest['recipe']
        and source['entry'] == release['entry'] and source['matches_original_carrier'] == header['count']
        and all(reference[key] == binding[key] for key in ('uri', 'size_bytes', 'sha256')),
        'canonical input does not identify the common released recipe and all native requests')
    binary = read_artifact(reference, json_document=False)
    expected_header = pack(header['format'], stream['input']['magic'].encode(),
                           bytes.fromhex(header['recipe_digest_hex']), header['size'], header['nnz'],
                           header['count'], header['flags'])
    _require(binary.startswith(expected_header)
        and reference['size_bytes'] == stream['bytes']['input_fixed']
            + stream['bytes']['selected_row'] * source['selected_rows'],
        'canonical input header or exact row-packed size differs')
    signature = source['original_stdout_sha256_payload']
    _require(type(signature) is str and re.fullmatch('[0-9a-f]{64}', signature) is not None,
        'released source must provide one original lowercase SHA-256 digest')
    validation = read_artifact(release['validation'])
    stdout_sha = sha256((signature + '\n').encode()).hexdigest()
    _require({row['stdout_sha256'] for row in validation['runs']} == {stdout_sha},
        'the original input oracle does not reproduce released output')
    return reference, signature + '\n'


def _check_output_contract(package, manifest, digest_stream, binary_stream, release):
    from scar.codegen.native_sparse_source import PROFILE, SCHEMA as PUBLICATION, STREAM

    root = Path(package['uri']).parent
    local = manifest['publication']
    contract_ref = _member(root, local['contract'])
    provider_ref = _member(root, local['provider'])
    contract = read_artifact(contract_ref)
    read_artifact(provider_ref)
    _require(contract_ref['schema'] == PUBLICATION and contract['schema_version'] == 1
        and contract['execution_recipe'] == manifest['recipe']
        and {**contract['release'], 'uri': manifest['qualified_release']['uri']} == manifest['qualified_release']
        and contract['entry_source'] == release['entry'] and contract['profile'] == PROFILE
        and digest_stream['schema'] == STREAM and binary_stream['output']['magic'] == 'SCAROUT1'
        and digest_stream['output'] == dict(profile=PROFILE, success_bytes=65, failure_stdout_bytes=0,
                                          provider_sha256=contract['provider']['libcrypto']['sha256'])
        and digest_stream['binary_consumer'] == dict(argument='--consume-records', input=binary_stream['output'],
                                                     completion='EXACT_TERMINAL_AND_EOF')
        and digest_stream['bytes']['binary_consumer_input_complete'] == binary_stream['bytes']['output_complete']
        and digest_stream['bytes']['output_complete'] == 65,
        'both publication variants require the same explicit ordered SHA-256 scope')
    return dict(contract=contract_ref, native_provider=provider_ref, profile=contract['profile'])


def _graph(stream, plan, pipe_capacity):
    count, n = (stream['header'][field] for field in ('count', 'size'))
    wire_bytes = stream['bytes']['output_complete']
    payload = 8*n
    owner = plan['placement']['owner_cpu_id']
    window = plan['dispatch']['window']
    case = dict(requests=count, payload_bytes_per_result=payload, numeric_payload_bytes=count*payload,
                stdout_bytes=65, issue_order='ASCENDING_ZERO_BASED')
    worker = dict(cpus=plan['placement']['worker_cpu_ids'], capacity=window,
                  in_flight_per_worker=1, pending_result_bytes=window*payload)
    return dict(case=case, worker=worker, owner=dict(cpu=owner, dispatch_execution_capacity=1,
        startup_and_finalization_allowed_cpus=plan['placement']['launch_cpu_ids']),
        shared_edges=[
            ['canonical-ingress', 'worker-solve[i]', 'SOURCE_OPERANDS'],
            ['worker-solve[i]', 'ordered-retire[i]', 'ZERO_STATUS_RESULT'],
            ['ordered-retire[i-1]', 'ordered-retire[i]', 'ORDERED_PREFIX'],
        ],
        options={
            BASE: dict(
                edges=[
                    ['producer-write-wire-header', 'consumer-read-wire-header', 'PIPE_BYTES_READY'],
                    ['consumer-read-wire-header', 'consumer-initialize-digest', 'VALID_RECIPE_AND_EXTENTS'],
                    ['consumer-initialize-digest', 'consumer-validate-and-update[0]', 'INITIAL_SHA_STATE'],
                    ['ordered-retire[i]', 'owner-write-record[i]', 'RESULT'],
                    ['owner-write-chunk[i,k]', 'consumer-read-chunk[i,k]', 'PIPE_BYTES_READY'],
                    ['consumer-read-complete-record[i]', 'consumer-validate-and-update[i]',
                     'PAYLOAD_IF_EXPECTED_SEQUENCE'],
                    ['consumer-validate-and-update[i-1]', 'consumer-validate-and-update[i]', 'SHA_STATE'],
                    ['owner-write-record[i]', 'worker-solve[i+W]', 'RETIRED_SLOT_CREDIT'],
                    ['consumer-validate-and-update[N-1]', 'terminal-eof-and-complete-hash', 'FINAL_SHA_STATE'],
                    ['producer-terminal-and-close', 'terminal-eof-and-complete-hash', 'EXACT_COUNT_AND_EOF'],
                    ['terminal-eof-and-complete-hash', 'publish-digest', 'BOTH_PROCESSES_SUCCEED'],
                ],
                producer_and_consumer_share_owner=True, pipe_capacity_bytes=pipe_capacity,
                pipe_can_backpressure_owner=True, cross_process_publication_bytes=wire_bytes,
                consumer_payload_buffer_bytes=payload,
                pipe_constraint='0 <= cumulative_written_bytes - cumulative_read_bytes <= pipe_capacity_bytes',
                cpu_service_terms=['owner-wire-write', 'external-read-validate', 'external-hash-update']),
            FUSED: dict(
                edges=[
                    ['owner-initialize-digest', 'owner-update-digest[0]', 'INITIAL_SHA_STATE'],
                    ['ordered-retire[i]', 'owner-update-digest[i]', 'RESULT'],
                    ['owner-update-digest[i-1]', 'owner-update-digest[i]', 'SHA_STATE'],
                    ['owner-update-digest[i]', 'worker-solve[i+W]', 'RETIRED_SLOT_CREDIT'],
                    ['owner-update-digest[N-1]', 'publish-digest', 'FINAL_SHA_STATE'],
                    ['all-successful-retirements', 'publish-digest', 'COMPLETE_HASH'],
                ],
                producer_and_consumer_share_owner=True, pipe_capacity_bytes=0,
                pipe_can_backpressure_owner=False, cross_process_publication_bytes=0,
                consumer_payload_buffer_bytes=0,
                cpu_service_terms=['owner-hash-update']),
        },
        source_scope='NATIVE_CANONICAL_INGRESS_TO_EXACT_DIGEST',
        models_full_python_effects=False, within_run_only=True,
        forward_cost=dict(status='PARTIAL', native_publish_service_ns=None, pipe_service_ns=None,
            native_process_startup_and_crypto_initialization_ns=None,
            actual_pipe_occupancy_bytes=None, worker_duration_change_from_backpressure_ns=None,
            binary_total_ns=None, fused_total_ns=None,
            conclusion='FUSED_REDUCES_FRAMED_TRAFFIC; TIMING_REQUIRES_MATCHED_MEASUREMENT'))


def _pipe_capacity():
    reader, writer = os.pipe()
    try:
        return fcntl.fcntl(writer, fcntl.F_GETPIPE_SZ)
    finally:
        os.close(reader)
        os.close(writer)


def design(base, fused, source, *, pairs, warmups, threshold_bps, pipe_capacity, owner_cost=None):
    _require(type(pairs) is int and pairs > 0 and type(warmups) is int and warmups >= 0
        and type(threshold_bps) is int and 0 <= threshold_bps < 10000
        and type(pipe_capacity) is int and pipe_capacity > 0,
        'positive pair count, nonnegative warmup, percentage and pipe capacity required')
    packages, executable, original, candidate, legacy, digest, plan, release = _packages(base, fused)
    source_ref = _reference(source, SOURCE_INPUT)
    input_ref, output = _source(source_ref, original, legacy, release)
    publication = _check_output_contract(packages[1], candidate, digest, legacy, release)
    historical = None
    if owner_cost is not None:
        from scar.optimization.sparse_parallel_owner_cost import SCHEMA as OWNER_COST

        history_ref = _reference(owner_cost, OWNER_COST)
        history = read_artifact(history_ref)
        _require(history['recipe_id'] == original['recipe_id'] and history['inputs']['recipe'] == original['recipe']
            and history['inputs']['release']['sha256'] == original['qualified_release']['sha256']
            and history['owner'] == dict(owner_cpu_id=plan['placement']['owner_cpu_id'],
                worker_cpu_ids=plan['placement']['worker_cpu_ids'],
                launch_cpu_ids=plan['placement']['launch_cpu_ids'], cpu_numa_node=plan['placement']['cpu_numa_node']),
            'historical owner clocks refer to another release, recipe or CPU placement')
        historical = dict(artifact=history_ref, scope='INSTRUMENTED_PYTHON_OWNER', price_native_tasks=False,
                          calls_per_process=plan['case']['call_count'])
        matches = history['costs'].get('instrumented_original_entry')
        if matches is not None:
            _require(matches['calls_per_process'] == plan['case']['call_count'],
                'historical owner call count differs from the native source case')
            historical['original_publication_sample'] = matches['subphases']['publication']
    plan_cpus = plan['placement']['launch_cpu_ids']
    owner_cpu = plan['placement']['owner_cpu_id']
    return dict(schema=DESIGN, schema_version=1, packages=packages, source=source_ref,
        recipe=original['recipe'], release_sha256=original['qualified_release']['sha256'],
        expected_stdout=output, input=input_ref, publication=publication, historical_owner=historical,
        executables=dict(zip((BASE, FUSED), executable, strict=True)),
        launch=dict(cpu_ids=plan_cpus, consumer_cpu_id=owner_cpu, environment=ENV, cwd='/',
            consumer_uses_additional_core=False),
        study=dict(pairs=pairs, warmup_pairs=warmups, minimum_reduction_basis_points=threshold_bps,
            orders=[[BASE, FUSED] if i % 2 == 0 else [FUSED, BASE] for i in range(pairs + warmups)]),
        graph=_graph(legacy, plan, pipe_capacity))


def _rebuild_design(value):
    study = value['study']
    historical = value['historical_owner']
    rebuilt = design(*(Path(ref['uri']).parent for ref in value['packages']), value['source']['uri'],
        pairs=study['pairs'], warmups=study['warmup_pairs'], threshold_bps=study['minimum_reduction_basis_points'],
        pipe_capacity=value['graph']['options'][BASE]['pipe_capacity_bytes'],
        owner_cost=historical['artifact']['uri'] if historical else None)
    _require(value == rebuilt, 'publication design differs from its exact native packages and input')
    return rebuilt


@contextmanager
def _affinity(cpus):
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, set(cpus))
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def _commands(plan, mode):
    fused = plan['executables'][FUSED]['uri']
    return [[fused]] if mode == FUSED else [
        [plan['executables'][BASE]['uri']], [fused, '--consume-records']]


def _read_output(process):
    try:
        stdout, _ = process.communicate(timeout=120)
        return stdout
    except subprocess.TimeoutExpired:
        process.kill()
        raise


def _run(plan, mode, pair):
    commands = _commands(plan, mode)
    cpus, consumer = (plan['launch'][name] for name in ('cpu_ids', 'consumer_cpu_id'))
    process_context = dict(env=plan['launch']['environment'], cwd=plan['launch']['cwd'])
    with open(plan['input']['uri'], 'rb') as stdin, TemporaryFile() as first_err, TemporaryFile() as last_err:
        with _affinity(cpus):
            begin = perf_counter_ns()
            if mode == BASE:
                producer = subprocess.Popen(commands[0], stdin=stdin, stdout=subprocess.PIPE, stderr=first_err,
                                            **process_context)
                try:
                    pipe_capacity = fcntl.fcntl(producer.stdout.fileno(), fcntl.F_GETPIPE_SZ)
                    _require(pipe_capacity == plan['graph']['options'][BASE]['pipe_capacity_bytes'],
                        'live publication pipe capacity differs from its plan')
                    with _affinity([consumer]):
                        with subprocess.Popen(commands[1], stdin=producer.stdout, stdout=subprocess.PIPE,
                                              stderr=last_err, **process_context) as process:
                            producer.stdout.close()
                            stdout = _read_output(process)
                    producer.wait(timeout=120)
                    codes = [producer.returncode, process.returncode]
                    pids = [producer.pid, process.pid]
                finally:
                    if producer.poll() is None:
                        producer.kill()
                        producer.wait()
                    producer.stdout.close()
            else:
                pipe_capacity = 0
                with subprocess.Popen(commands[0], stdin=stdin, stdout=subprocess.PIPE,
                                      stderr=last_err, **process_context) as process:
                    stdout = _read_output(process)
                codes, pids = [process.returncode], [process.pid]
            end = perf_counter_ns()
        first_err.seek(0)
        last_err.seek(0)
        errors = first_err.read() + last_err.read()
    return dict(pair=pair, mode=mode, commands=commands, perf_clock_ns=[begin, end],
        process_exit_codes=codes, child_pids=pids, stdout=stdout.decode('ascii'), stderr=errors.decode('utf-8'),
        observed_pipe_capacity_bytes=pipe_capacity)


def collect(design_path):
    """Run both native arms without profiling, with equal reserved CPU lanes."""
    ref = _reference(design_path, DESIGN)
    plan = _rebuild_design(read_artifact(ref))
    _require(set(plan['launch']['cpu_ids']) <= os.sched_getaffinity(0),
        'all fixed release CPUs must be available to the clean benchmark')
    rows = [_run(plan, mode, i) for i, pair in enumerate(plan['study']['orders']) for mode in pair]
    for reference in (plan['input'], *plan['executables'].values()):
        read_artifact(reference, json_document=False)
    return dict(schema=OBSERVATION, schema_version=1, design=ref, launch=plan['launch'],
        scope='FRESH_NATIVE_PROCESSES_TO_COMPLETE_DIGEST', instrumentation=False, rows=rows)


def choose(design_path, observations_path=None):
    """Cold-replay output qualification and conservative selection over sample ranges."""
    design_ref = _reference(design_path, DESIGN)
    plan = _rebuild_design(read_artifact(design_ref))
    result = dict(schema=SCHEMA, schema_version=1, design=design_ref, observations=None,
        selected=BASE, decision='NATIVE_TIMING_UNRESOLVED', timing=None,
        forward_cost=plan['graph']['forward_cost'], deployment_authorized=False)
    if observations_path is None:
        return result
    reference = _reference(observations_path, OBSERVATION)
    observed = read_artifact(reference)
    _require(observed['schema_version'] == 1 and observed['design'] == design_ref
        and observed['scope'] == 'FRESH_NATIVE_PROCESSES_TO_COMPLETE_DIGEST'
        and observed['instrumentation'] is False and observed['launch'] == plan['launch'],
        'native observation is outside the exact clean output comparison')
    rows = observed['rows']
    study = plan['study']
    expected = [(i, mode) for i, pair in enumerate(study['orders']) for mode in pair]
    _require([(row['pair'], row['mode']) for row in rows] == expected,
        'observation must preserve the frozen alternating pair schedule')
    values = {BASE: [], FUSED: []}
    previous, children = 0, set()
    for row in rows:
        mode = row['mode']
        commands = _commands(plan, mode)
        pids = row['child_pids']
        _require(row['stdout'] == plan['expected_stdout'] and row['stderr'] == ''
            and row['commands'] == commands and row['process_exit_codes'] == [0] * len(commands)
            and all(type(pid) is int and pid > 0 for pid in pids)
            and len(pids) == len(commands) == len(set(pids)) and children.isdisjoint(pids)
            and row['observed_pipe_capacity_bytes'] == plan['graph']['options'][mode]['pipe_capacity_bytes']
            and previous < row['perf_clock_ns'][0],
            'native timings require ordered non-overlapping success under the same output contract')
        duration = _duration(row['perf_clock_ns'])
        children.update(pids)
        previous = row['perf_clock_ns'][1]
        if row['pair'] >= study['warmup_pairs']:
            values[mode].append(duration)
    output = {mode: dict(_summary(series), variance_ns2=variance(series) if len(series) > 1 else None)
              for mode, series in values.items()}
    best_base, worst_fused = output[BASE]['lower_ns'], output[FUSED]['upper_ns']
    enough = study['pairs'] >= 3
    qualifies = enough and worst_fused < best_base and worst_fused*10000 <= best_base * (
        10000-study['minimum_reduction_basis_points'])
    reason = 'PROFITABLE_WITHIN_OBSERVED_ENVELOPE' if qualifies else 'KEEP_BASELINE_NO_ROBUST_GAIN'
    if not enough:
        reason = 'KEEP_BASELINE_INSUFFICIENT_MATCHED_PAIRS'
    result.update(observations=reference, selected=FUSED if qualifies else BASE,
        decision=reason,
        timing=dict(arms=output,
            paired_reduction_ns=_summary([a-b for a,b in zip(values[BASE], values[FUSED], strict=True)]),
            minimum_reduction_basis_points=study['minimum_reduction_basis_points'],
            observed_worst_reduction_basis_points=((best_base-worst_fused)*10000)//best_base,
            measured_output_bytes=65, same_exact_stdout=True, same_no_stderr=True,
            bound_scope='OBSERVED_SAMPLES_ONLY', full_python_entry=False))
    return result


def check(path):
    row = read_artifact(_reference(path, SCHEMA))
    observed = row['observations']
    _require(row == choose(row['design']['uri'], observed['uri'] if observed else None),
        'publication decision does not replay')
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    freeze = sub.add_parser('design')
    for argument in ('base', 'fused', 'source', 'out', 'owner-cost'):
        freeze.add_argument('--' + argument, type=Path, required=argument != 'owner-cost')
    freeze.add_argument('--pairs', type=int, default=5)
    freeze.add_argument('--warmups', type=int, default=1)
    freeze.add_argument('--minimum-reduction-bps', type=int, default=200)
    record = sub.add_parser('measure')
    rank = sub.add_parser('choose')
    for command in (record, rank):
        command.add_argument('--design', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
    rank.add_argument('--observations', type=Path)
    replay = sub.add_parser('check')
    replay.add_argument('path', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'check':
        print(check(args.path)['decision'])
        return
    _require(not args.out.exists(), 'fresh result path required')
    if args.mode == 'design':
        result = design(args.base, args.fused, args.source, pairs=args.pairs, warmups=args.warmups,
            threshold_bps=args.minimum_reduction_bps, pipe_capacity=_pipe_capacity(), owner_cost=args.owner_cost)
    elif args.mode == 'measure':
        result = collect(args.design)
    else:
        result = choose(args.design, args.observations)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
