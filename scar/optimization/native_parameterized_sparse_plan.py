"""Plan a count-parameterized sparse loop on a checked ordered CPU cohort."""

import argparse
import json
from pathlib import Path
import re

from scar.codegen import native_source_triangle_profile as fixed_entry
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_profile as fixed_profile
from scar.optimization import scalar_sorted_transition_cost as entry_cost


SCHEMA = 'scar.native-parameterized-sparse-plan'
SOURCE_COUNT = 'scar.source-sparse-count'
MODE = 'RUNTIME_COUNT_ORDERED_CPU_SPARSE_DIGEST'
BENCHMARK = 'scar.native-parameterized-sparse-paired-run'
COMPARISON = 'scar.native-parameterized-sparse-paired-selection'
EXECUTABLE = 'scar.native-parameterized-sparse-executable'
_ARMS = ('original','entry_native')
_DIGEST_LINE = re.compile(rb'[0-9a-f]{64}\n')
_WIRES = ('seed','mesh','basis','coo','canonical_csr')


def _require(condition, detail):
    if not condition:
        raise ValueError('parameterized sparse placement: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _package_root(path):
    path = Path(path).resolve(strict=True)
    manifest = path / 'artifact.json' if path.is_dir() else path
    _require(manifest.name == 'artifact.json', 'expected the original standalone package root or manifest')
    return manifest.parent


def _member(root, ref):
    relative = Path(ref['uri'])
    path = root / relative
    _require(not relative.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True) == path,
        'fixed package member escaped its admitted root')
    return dict(ref, uri=str(path))


def _fixed(root):
    manifest = fixed_entry.check(root)
    reference = _ref(root / 'artifact.json',fixed_entry.SCHEMA)
    profile_ref = manifest['profile_plan']
    profile = read_artifact(profile_ref)
    seed = read_artifact(manifest['source_seed'])
    solver_ref = _member(root,manifest['solver'])
    solver = read_artifact(solver_ref)
    protocol_ref = _member(Path(solver_ref['uri']).parent,solver['stream']['protocol'])
    protocol = read_artifact(protocol_ref)
    resources = profile['resources']
    case = resources['case']
    _require(profile_ref['schema'] == fixed_profile.SCHEMA and solver['profile_plan'] == profile_ref
        and seed['entry']['schema'] == 'scar.python-source' and seed['iterations'] == case['call_count'] == solver['count']
        and protocol['header']['count'] == case['call_count'] and protocol['header']['size'] == case['size']
        and protocol['header']['nnz'] == case['nnz'] and protocol['schema'] == 'scar.native-ordered-source-digest-stream'
        and profile['deployment_authorized'] is manifest['original_effects_closed'] is False,
        'fixed solver geometry or numerical output scope is open')
    return dict(package=reference,profile=profile_ref,seed=manifest['source_seed'],solver=solver_ref,protocol=protocol_ref), \
        seed,profile,protocol


def _resources(profile, protocol):
    before = profile['resources']
    where, dispatch = (before[key] for key in ('placement','dispatch'))
    width = profile['width']
    slots = where['workers']
    _require(type(width) is int and width > 1 and len(slots) == width == dispatch['window']
        and [row['slot'] for row in slots] == list(range(width))
        and [row['cpu_id'] for row in slots] == where['worker_cpu_ids']
        and set(where['launch_cpu_ids']) == {where['owner_cpu_id'],*where['worker_cpu_ids']}
        and len(where['launch_cpu_ids']) == width + 1
        and dispatch['one_inflight_per_worker'] is True and dispatch['worker_threads'] == 1
        and before['launch']['matching_live_topology_required'] is True
        and before['launch']['live_cpu_quota_and_memory_headroom_required'] is True,
        'expected the same fixed isolated owner/worker window with live admission guards')
    case = before['case']
    _require(protocol['input']['original_values']['dtype'] == '<f8'
        and protocol['input']['original_rhs']['dtype'] == '<f8'
        and protocol['output']['profile']['input'] == 'ORDERED_CONTIGUOUS_LITTLE_ENDIAN_BINARY64_SOLUTION_BYTES'
        and protocol['output']['profile']['algorithm'] == 'SHA256'
        and protocol['output']['success_bytes'] == 65 and protocol['output']['failure_stdout_bytes'] == 0
        and protocol['bytes']['compact_request_bytes_per_worker'] == 8,
        'work byte accounting must bind the exact binary64 ordered-stream ABI')
    memory = before['memory']
    _require(memory['worker_address_space_sum_bytes'] == width * memory['worker_address_space_limit_bytes']
        and memory['minimum_launch_headroom_bytes'] >= memory['worker_address_space_sum_bytes']
        and memory['physical_reservation'] is memory['physical_numa_residency_proven'] is False,
        'memory gate must retain the conservative worker capacity and unknown physical residence')
    return dict(case=dict(size=case['size'],nnz=case['nnz'],call_count=dict(symbol='count',binding='RUNTIME_ARGUMENT')),
        placement=where,dispatch=dispatch,launch=before['launch'],memory=memory,width=width,
        placement_status='REUSE_TESTED_CPU_COHORT_SUBJECT_TO_LIVE_GUARDS',globally_optimal=False,
        historical_service_costs_used=False,
        cost=dict(basis='UNKNOWN',startup_ns=None,per_call_ns=None,close_ns=None,whole_scope_ns=None))


def _control(resources):
    stages = ('count_guard','source_topology','source_mesh','source_basis','source_coo_csr',
        'prepare_ordered_owner','ordered_solve_loop','commit_digest')
    return dict(stages=list(stages),dependencies=[dict(before=a,after=b) for a,b in zip(stages,stages[1:])],
        loop=dict(axis='i',iterations='count',unrolled=False,loop_body_instances=1,
            initial_fill=dict(operation='min', operands=['count',resources['dispatch']['window']]),
            issue=dict(order='ASCENDING',ready=['CANONICAL_CSR_READY','WORKER_SLOT_IDLE','INITIAL_CREDIT_OR_PRIOR_RETIREMENT']),
            worker_slot=dict(operation='modulo',operands=['i',resources['width']]),
            refill=dict(after='ORDERED_COMMIT_OF_I',issue='i+window',condition='i+window<count'),
            retirement='OLDEST_OUTSTANDING_SUCCESS_ONLY',output='SHA256_APPEND_COMPLETE_BINARY64_SOLUTION_BYTES',
            completion='DRAIN_ALL_COUNT_SUCCESS_THEN_EMIT_ONE_DIGEST_LINE'),
        failed_request='TERMINATE_WITHOUT_STDOUT_DIGEST',
        count_wire='REBIND_LAYOUT_AND_RUNTIME_COUNT_BEFORE_DISPATCH',
        setup_repeat='ONE_SOURCE_GEOMETRY_SETUP_PER_ENTRY')


def _work(resources, protocol):
    return dict(owner_setup_instances=1,worker_slots=resources['width'],
        successful_request_issues='count',successful_solves='count',ordered_digest_appends='count',
        logical_scale_bytes_per_issue=protocol['bytes']['compact_request_bytes_per_worker'],
        logical_solution_bytes_per_issue=resources['case']['size'] * 8,
        successful_stdout_bytes=protocol['output']['success_bytes'],
        prior_solution_prefix_reusable=False,task_service_ns=None,transport_time_ns=None,
        byte_scope='LOGICAL_NUMERIC_PAYLOAD_ONLY_EXCLUDES_PROTOCOL_AND_KERNEL_COPY_OVERHEAD')


def work_for_count(plan,count):
    """Evaluate exact successful-path cardinalities without estimating time."""
    scope = plan['scope']
    _require(plan['decision']['native_experiment_authorized'] is True and type(count) is int
        and scope['minimum_count'] <= count <= scope['maximum_count'], 'unadmitted positive runtime count')
    work, resources = plan['work'],plan['resources']
    width, window = resources['width'],resources['dispatch']['window']
    return dict(count=count,issues=count,solves=count,digest_appends=count,initial_fill=min(count,window),
        refills=max(0,count-window),issues_per_worker=[(count+width-1-slot)//width for slot in range(width)],
        scale_bytes=count*work['logical_scale_bytes_per_issue'],solution_bytes=count*work['logical_solution_bytes_per_issue'],
        stdout_bytes=work['successful_stdout_bytes'],elapsed_ns=None)


def _admit_count(scope,proof,fixed,protocol,resources):
    port, loop, geometry = (proof[key] for key in ('count_port','loop','invariant_geometry'))
    observed = [row['count'] for row in proof['observations']]
    _require(proof['entry'] == scope['source'] and proof['target_executed'] is False
        and len(set(observed)) == len(observed) == len(proof['source_seeds']) and len(observed) >= 2
        and observed == port['observed_counts'] and scope['fixed_profile_count'] in observed,
        'runtime count evidence must join independent original runs and the fixed provider source')
    ordinal = observed.index(scope['fixed_profile_count'])
    baseline, witness = (read_artifact(ref) for ref in (fixed['seed'],proof['source_seeds'][ordinal]))
    _require(all(baseline[key] == witness[key] for key in (
        'entry','argv','iterations','fields','depth','sources','topology_conditions','scope')),
        'independently captured seeds have different source or typed geometry constants')
    minimum, maximum = port['minimum'],port['maximum']
    _require(port['kind'] == 'ARGV_INTEGER' and port['argv_index'] == port['argv_arity'] == 1
        and port['dtype'] == 'SIGNED_INT32' and port['encoding'] == 'CANONICAL_ASCII_DECIMAL'
        and type(minimum) is type(maximum) is int and 1 == minimum <= maximum <= 512,
        'expected the fixed bounded runtime integer count port')
    iterator, scale, digest = (loop[key] for key in ('iterator','scale','digest'))
    _require(loop['code']['key'] == protocol['source']['body_code_key']
        and (iterator['start'],iterator['stop'],iterator['step'],iterator['unrolled']) == (0,'count',1,False)
        and scale['expression'] == '1.0 + (i + 1) / (count + 1)' and scale['dependencies'] == ['i','count']
        and scale['arithmetic'] == 'INTEGER_ADD_THEN_BINARY64_DIVIDE_AND_ADD_NO_FMA'
        and protocol['source']['expression'] == ['Add',['constant',1.0],
            ['Div',['Add',['input',0],['constant',1]],['Add',['input',1],['constant',1]]]]
        and digest['algorithm'] == 'SHA256' and digest['update_input'] == 'SAME_ITERATION_SOLVE_RESULT'
        and digest['order'] == 'ASCENDING_LOOP_INDEX' and digest['prior_prefix_reusable'] is False,
        'runtime scale or ordered digest is not the qualified numerical provider relation')

    shape = {key:resources['case'][key] for key in ('size','nnz')}
    _require(geometry['count_independent'] is True and geometry['metadata_is_count_dependent'] is True
        and geometry['whole_wire_reusable'] is False and geometry['upstream_shape'] == shape
        and isinstance(geometry['numeric_payload_sha256'],dict)
        and len(geometry['historical_wires']) == len(observed),
        'count-independent canonical geometry needs attested source numerical bodies')
    actual_package = read_artifact(fixed['package'])
    root = Path(fixed['package']['uri']).parent
    upstream = _member(root,actual_package['upstream'])
    evidence = geometry['historical_wires'][ordinal]['wires']
    actual_wires = read_artifact(upstream)['wires']
    _require(len(actual_wires) == len(_WIRES) and set(evidence) == set(_WIRES),
        'missing exact typed preparation wires')
    for role, member in zip(_WIRES,actual_wires,strict=True):
        linked = _member(Path(upstream['uri']).parent,member)
        original = evidence[role]
        offset = original['metadata_bytes']
        raw = read_artifact(linked,json_document=False)
        witnessed = read_artifact(original['artifact'],json_document=False)
        _require(linked['schema'] == original['artifact']['schema'] and 0 < offset < len(raw)
            and raw[offset:] == witnessed[offset:],
            'fixed compiled numerical payload differs from the original geometry witness for ' + role)
    scope.update(count_port=port,minimum_count=minimum,maximum_count=maximum,archived_source_counts=observed,
        exact_integer_numerator=dict(minimum=1,maximum=maximum),
        exact_integer_denominator=dict(minimum=2,maximum=maximum+1),
        binary64_scale_domain=dict(minimum=1.0+1.0/(maximum+1),maximum=1.0+maximum/(maximum+1)))
    return dict(source_loop=loop,invariant_geometry=geometry,source_frontiers=proof['proof_frontiers'])


def propose(fixed_package,source_count_path=None):
    root = _package_root(fixed_package)
    fixed,seed,profile,protocol = _fixed(root)
    resources = _resources(profile,protocol)
    proof_ref = _ref(source_count_path,SOURCE_COUNT) if source_count_path is not None else None
    scope = dict(source=seed['entry'],count_port=None,minimum_count=None,maximum_count=None,
        fixed_profile_count=seed['iterations'],mode=MODE,full_original_effect_equivalence=False)
    fields = dict(source_loop=None,invariant_geometry=None,source_frontiers=None)
    blockers = ['MISSING_SOURCE_COUNT_AND_GEOMETRY_PROOF']
    if proof_ref is not None:
        from scar.analysis import source_sparse_count
        proof = source_sparse_count.check(proof_ref['uri'])
        fields = _admit_count(scope,proof,fixed,protocol,resources)
        blockers = []
    return dict(schema=SCHEMA,schema_version=1,artifacts=dict(fixed_package=fixed['package'],source_count=proof_ref,
            fixed_profile_plan=fixed['profile'],fixed_seed=fixed['seed'],fixed_solver=fixed['solver'],
            fixed_protocol=fixed['protocol']), scope=scope,**fields,resources=resources,
        control=_control(resources),work=_work(resources,protocol),
        numeric_contract=dict(precision='BINARY64',fp_contraction=False,rounding='ROUND_TO_NEAREST',
            count_changes_scale_denominator=True,require_fresh_source_scale_each_issue=True,
            whole_fixed_csr_wire_reusable=False,provider_costs_transferred=False),
        decision=dict(selected='RETAIN_ORIGINAL_SOURCE',native_experiment_authorized=not blockers,
            deployment_authorized=False,blockers=blockers))


def _load_plan(path):
    stored = read_artifact(_ref(path,SCHEMA))
    refs = stored['artifacts']
    rebuilt = propose(refs['fixed_package']['uri'],refs['source_count']['uri'] if refs['source_count'] else None)
    _require(_canonical(stored) == _canonical(rebuilt),'parameterized sparse plan differs from cold replay')
    return rebuilt


def _native(arm,plan,reference):
    ref = arm['artifact']
    _require(ref['schema'] == EXECUTABLE,'fresh native process must use the exact parameterized package')
    artifact = read_artifact(ref)
    scope, resources = plan['scope'],plan['resources']
    protocol = artifact['protocol']
    header = protocol['header']
    _require(artifact['schema_version'] == 1 and artifact['parameterized_plan'] == reference
        and artifact['source'] == scope['source'] and artifact['cxx_standard'] == 20
        and artifact['python_runtime_required'] is artifact['deployment_authorized'] is False
        and artifact['source_count_read_at_runtime'] is True
        and protocol['placement'] == resources['placement'] and protocol['memory'] == resources['memory']
        and protocol['width'] == resources['width'] and protocol['window'] == resources['dispatch']['window']
        and header['size'] == resources['case']['size'] and header['nnz'] == resources['case']['nnz']
        and header['count'] == dict(minimum=scope['minimum_count'],maximum=scope['maximum_count']),
        'compiled count protocol differs from the planned geometry and CPU resources')
    binary = artifact['binary']
    _require(binary['uri'] == 'scar_sparse_parameterized' and binary['schema'] == 'scar.elf'
        and arm['executable'] == _member(Path(ref['uri']).parent,binary)
        and read_artifact(arm['executable'],json_document=False).startswith(b'\x7fELF'),
        'fresh native process must bind the package-local ELF bytes')


def _mask(text):
    cpus = set()
    for item in text.split(','):
        match = re.fullmatch(r'([0-9]+)(?:-([0-9]+))?',item)
        _require(match is not None,'invalid paired CPU affinity')
        low, high = int(match[1]),int(match[2] or match[1])
        _require(0 <= low <= high < 1024,'paired CPU affinity outside the admitted mask representation')
        cpus.update(range(low,high+1))
    return sorted(cpus)


def _launch(row, run, argv):
    mode = row['arm']
    command = row['command']
    _require(type(command) is list and len(command) >= 5 and all(type(x) is str for x in command)
        and Path(command[0]).name == 'taskset' and command[1] == '-c' and _mask(command[2]) == run['cpu_ids'],
        'original and native launches must preserve the same exact CPU placement')
    prefix = command[:3]
    tail = command[3:]
    if Path(tail[0]).name == 'timeout':
        _require(len(tail) >= 6 and tail[1:3] == ['-k','2'] and tail[3].isdigit(),
            'unexpected timed process supervision')
        prefix.extend(tail[:4])
        tail = tail[4:]
    arm = run['arms'][mode]
    if mode == 'entry_native':
        _require(tail == [arm['executable']['uri'],*argv], 'native launch did not use the exact tested ELF and arguments')
    else:
        _require(len(tail) >= 3 and Path(tail[0]).is_absolute()
            and str(Path(tail[0]).resolve(strict=True)) == arm['python_executable']
            and tail[-2:] == [run['source']['uri'],*argv]
            and all(flag in ('-B','-u') for flag in tail[1:-2]),
            'baseline did not launch the original source directly with the same arguments')
    return tuple(prefix)


def _case(case,run,plan):
    count, argv = case['count'],case['argv']
    work = work_for_count(plan,count)
    _require(argv == [str(count)],
        'paired trials must use the same admitted runtime count argument')
    rows = case['rows']
    _require(type(rows) is list and len(rows) % 2 == 0 and len(rows) >= 6,
        'at least two clean paired samples and one warmup pair are required for each count')
    samples = {mode:[] for mode in _ARMS}
    outputs, launchers, orders = set(),set(),[]
    for index,row in enumerate(rows):
        number, slot = divmod(index,2)
        order = _ARMS if number % 2 == 0 else _ARMS[::-1]
        if slot == 0:
            orders.append(list(order))
        expected = ('OBSERVED_WARMUP','WARMUP') if number == 0 else ('CLEAN_MEASURED',)
        _require(type(row['round']) is int and row['round'] == number and row['arm'] == order[slot]
            and row['kind'] in expected and type(row['returncode']) is int and row['returncode'] == 0
            and type(row['wall_ns']) is int and row['wall_ns'] > 0,
            'trial does not match the fresh paired scope or completed exit')
        launchers.add(_launch(row,run,argv))
        stdout, stderr = (read_artifact(row[key],json_document=False) for key in ('stdout','stderr'))
        _require(_DIGEST_LINE.fullmatch(stdout) and not stderr,
            'each successful invocation must publish only one SHA-256 line with empty stderr')
        outputs.add((stdout,stderr))
        if number:
            samples[row['arm']].append(row['wall_ns'])
    _require(len(outputs) == len(launchers) == 1,'paired invocations changed their supervised launch or exact output')
    measurements, gains = entry_cost._envelopes(samples,minimum_gain_bps=0)
    gain = gains['entry_native']
    return dict(count=count,argv=argv,work=work,orders=orders,clean_pairs=len(samples['original']),warmup_pairs=1,
        exact_stdout_hex=outputs.pop()[0].decode('ascii').rstrip('\n'),measurements=measurements,
        original_minus_native_evidence=gain,selected='entry_native' if gain['verified_observed_gain'] else 'original')


def compare(plan_path,benchmark_path):
    """Replay fresh direct original and native samples by exact held-out count."""
    refs = dict(plan=_ref(plan_path,SCHEMA),benchmark=_ref(benchmark_path,BENCHMARK))
    plan = _load_plan(refs['plan']['uri'])
    run = read_artifact(refs['benchmark'])
    _require(plan['decision']['native_experiment_authorized'] is True and run['schema_version'] == 1
        and run['parameterized_plan'] == refs['plan'] and run['source'] == plan['scope']['source']
        and run['cpu_ids'] == plan['resources']['placement']['launch_cpu_ids']
        and type(run['cwd']) is str and Path(run['cwd']).is_absolute()
        and type(run['environment']) is dict and all(type(key) is type(value) is str
            for key,value in run['environment'].items()), 'comparison changed the admitted source or CPU cohort')
    read_artifact(run['launcher'],json_document=False)
    original, native = (run['arms'][mode] for mode in _ARMS)
    _require(set(run['arms']) == set(_ARMS) and set(original) == {'python_executable','python_version'}
        and set(native) == {'artifact','executable'} and type(original['python_executable']) is str
        and type(original['python_version']) is str and original['python_version'].strip(),
        'comparison needs one direct original and one direct C++ runtime arm')
    python = Path(original['python_executable'])
    _require(python.is_absolute() and python.is_file() and python.resolve(strict=True) == python,
        'recorded original interpreter no longer resolves to its measured real path')
    from scar.compiler.compile_triangle_profile import _launch_environment

    profile = read_artifact(plan['artifacts']['fixed_profile_plan'])
    context = read_artifact(profile['anchor']['release'])['context']
    _require(run['cwd'] == context['cwd'] and run['environment'] == _launch_environment(context)
        and Path(context['interpreter']['executable']).resolve(strict=True) == python,
        'fresh direct launch must share the qualified original interpreter and provider environment')
    _native(native,plan,refs['plan'])
    cases = [_case(case,run,plan) for case in run['cases']]
    counts = [case['count'] for case in cases]
    _require(len(cases) >= 2 and len(set(counts)) == len(cases)
        and not set(counts).intersection(plan['scope']['archived_source_counts']),
        'comparison must use distinct counts outside the original qualification samples')
    return dict(schema=COMPARISON,schema_version=1,artifacts=refs,scope='DIRECT_HELD_OUT_SPARSE_COUNT_PROCESSES',
        matched_launch={key:run[key] for key in ('source','cpu_ids','cwd','environment','launcher')},
        original_interpreter=original,executable=native,cases=cases,
        selected='entry_native' if all(row['selected'] == 'entry_native' for row in cases) else 'original',
        deployment_authorized=False,historical_costs_used_for_selection=False)


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    if schema == SCHEMA:
        return _load_plan(path)
    _require(schema == COMPARISON,'unsupported sparse runtime plan or comparison')
    stored = read_artifact(_ref(path,COMPARISON))
    refs = stored['artifacts']
    result = compare(refs['plan']['uri'],refs['benchmark']['uri'])
    _require(_canonical(stored) == _canonical(result),'sparse runtime decision differs from cold replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixed-package','source-count','check','out','plan','benchmark'):
        parser.add_argument('--' + name,type=Path)
    args = parser.parse_args(argv)
    inputs = (args.fixed_package,args.source_count)
    pair = (args.plan,args.benchmark)
    if args.check and not any((*inputs,*pair,args.out)):
        result = check(args.check)
    elif args.check is None and args.fixed_package and args.out and not any(pair):
        result = propose(args.fixed_package,args.source_count)
    elif args.check is None and all(pair) and args.out and not any(inputs):
        result = compare(*pair)
    else:
        parser.error('provide --check, plan inputs, or --plan and --benchmark with --out')
    if args.out:
        with args.out.open('x') as stream:
            stream.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False) + '\n')
    summary = result['decision'] if result['schema'] == SCHEMA else dict(selected=result['selected'],
        counts=[row['count'] for row in result['cases']])
    print(_canonical(summary))


if __name__ == '__main__':
    main()
