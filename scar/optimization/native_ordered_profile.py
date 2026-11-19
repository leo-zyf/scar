"""Specialize a checked standalone ordered CPU route for a fresh count."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import re
from statistics import median, stdev

from scar.analysis.source_triangle_seed import SCHEMA as SEED, check as check_seed
from scar.analysis.sparse_parallel_scaling_scope import _lanes
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.sparse_parallel_cost import HARDWARE
from scar.optimization.sparse_parallel_recipe import _storage_requirements


SCHEMA = 'scar.native-ordered-count-profile'
ORIGINAL_OBSERVATION = 'scar.study-aggregate-source-scope'
ORACLE = 'scar.native-ordered-source-stdout'
BENCHMARK = 'scar.source-triangle-profile-benchmark'
SELECTION = 'scar.native-ordered-count-selection'
FULL_PACKAGE = 'scar.native-source-triangle-profile'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _local(package, ref):
    return {**ref, 'uri': str(package / ref['uri'])}


def _original(seed, observation, original, before, detached, oracle):
    count = seed['iterations']
    prior_count = before['case']['call_count']
    _require(type(count) is int and 1 < count < 2**31 - 1 and type(prior_count) is int and prior_count > 1,
        'an exact finite repeated loop count is required')
    classes = original['local_event_counts']
    expected = {event: count if n == prior_count else n for event, n in classes.items()}
    _require(len(classes) > 1 and sum(n == prior_count for n in classes.values()) >= 2
        and observation['local_event_counts'] == expected
        and observation['entry'] == original['entry'] == seed['entry']
        and observation['normal_completion'] is True and observation['installed_native_providers'] == 0
        and observation['code_keys'] == original['code_keys']
        and detached['body_code_key'] in observation['code_keys'].values()
        and observation['loop_inputs'] == original['loop_inputs']
        and observation['argv'] == seed['argv'] == [str(count)],
        'fresh original activation, loop completion counts, code or input geometry differ')
    proof = seed['provenance']
    fields = proof.get('observed_loop_count')
    _require(isinstance(fields, str) and re.fullmatch(r'#/records/[0-9]+/fields/selected_count', fields) is not None,
        'the new count must be observed at its original F4 call')
    index = int(fields.split('/')[2])
    row = read_artifact(proof['frames'])['records'][index]
    _require(row['code']['key'] == detached['body_code_key']
        and row['fields']['selected_count']['formal'] == detached['roles']['count']
        and row['fields']['selected_count']['value'] == count,
        'the observed original operand does not bind the detached count port')
    digest = observation['return_digest']
    _require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None
        and oracle == (digest + '\n').encode('ascii'), 'oracle differs from the original ordered publication')


def _resources(before, hardware_ref, hardware, count, width, identity):
    old = before['placement']
    owner, node = old['owner_cpu_id'], old['cpu_numa_node']
    _require(type(width) is int and 1 < width <= min(count, len(old['worker_cpu_ids'])),
        'requested width is outside the admitted physical cohort')
    workers = old['worker_cpu_ids'][:width]
    evidence = _lanes(hardware, [owner, *workers], node)
    lanes = [old['owner'], *old['workers'][:width]]
    _require(all(row['cpu_id'] == lane['cpu_id'] and row['observed_smt_siblings'] == lane['siblings']
        and row['physical_core'] == [lane[key] for key in ('package_id', 'die_id', 'core_id')]
        for row, lane in zip(evidence, lanes, strict=True)), 'original CPU lanes have changed')
    _require(hardware['schema_version'] == 1 and hardware['hierarchy_observation_complete']
        and hardware['memory_reservation'] is False and hardware['memory_placement_enforced'] is False
        and all(limit['parsed']['memory_max']['bytes'] is None or type(limit['memory_current_bytes']) is int
            for limit in hardware['limits']), 'complete, current CPU and memory constraints are required')
    quota = hardware['cpu_quota_equivalents']
    _require(quota is None or quota[0] >= (width + 1)*quota[1], 'CPU quota cannot supply the pinned cohort')
    case, old_memory = before['case'], before['memory']
    arithmetic = _storage_requirements(case['size'], case['nnz'], width,
        old_memory['worker_address_space_limit_bytes'], old_memory['per_worker_symbolic_budget_bytes'])
    memory = {key: arithmetic.get(key, value) for key, value in old_memory.items()}
    node_free = hardware['node_free_snapshot'].get(str(node), {})
    limit = hardware['finite_memory_headroom_snapshot_bytes']
    headroom = memory['minimum_launch_headroom_bytes']
    _require(node_free.get('allowed') is True and type(node_free.get('bytes')) is int
        and node_free['bytes'] >= headroom and (limit is None or limit >= headroom),
        'current NUMA or cgroup headroom does not admit the new placement')
    control = dict(iteration=dict(axis='i', count=count, unrolled=False, shared_body_instances=1),
        window=width, initial_fill='FIRST_W_ISSUES', refill='ORDERED_PUBLICATION_I_RETURNS_SLOT_TO_I_PLUS_W',
        retirement='ASCENDING_ISSUE_ONLY', completion='DRAIN_ALL_SUCCESS_THEN_CLOSE_AND_PUBLISH')
    return dict(before, recipe_id=identity, case={**case, 'call_count': count},
        source_control_sha256=semantic_sha256(control),
        placement={**old, 'hardware': hardware_ref, 'choice': None, 'worker_cpu_ids': workers,
                   'launch_cpu_ids': sorted([owner, *workers]), 'workers': lanes[1:]},
        dispatch={**before['dispatch'], 'window': width},
        memory={**memory, 'headroom_basis': 'CHECKED_PROFILE_OWNED_BYTES_AND_LIVE_RECHECK'},
        cost=dict(basis='UNKNOWN', startup_ns=None, per_call_ns=None, close_ns=None, whole_scope_ns=None)), control


def plan_profile(anchor_package, source_seed_path, hardware_path, oracle_path, width, *, original_observation_path):
    """Bind source, observed count, invariant loop and provider to a new fixed CPU plan."""
    from scar.codegen.native_ordered_sparse_export import SCHEMA as PACKAGE, check_native_ordered_export
    from scar.codegen.native_sparse_source import project_source_csr_stream

    package = Path(anchor_package).resolve(strict=True)
    manifest = check_native_ordered_export(package)
    _require(manifest['schema_version'] == 5, 'profile requires an admitted canonical native digest provider')
    anchor = dict(package=_ref(package/'artifact.json', PACKAGE),
        plan=_local(package, manifest['plan']), detached=_local(package, manifest['detached_source']),
        publication=_local(package, manifest['publication']['contract']),
        release=_local(package, manifest['qualified_release']))
    before, detached, contract, release = (read_artifact(anchor[key])
        for key in ('plan', 'detached', 'publication', 'release'))
    scope = read_artifact(detached['source'])
    original_scope = read_artifact(scope['source_scope'])['observed']
    original = read_artifact(original_scope['exact_count_evidence'])
    seed_ref = _ref(source_seed_path, SEED)
    seed = check_seed(seed_ref['uri'])
    observation_ref = _ref(original_observation_path, ORIGINAL_OBSERVATION)
    observation = read_artifact(observation_ref)
    oracle_ref = _ref(oracle_path, ORACLE)
    count = seed['iterations']
    _require(seed['entry'] == contract['entry_source'] == release['entry'] == scope['entry']['source']
        and scope['source_body']['code_key'] == detached['body_code_key']
        and count != release['count'] and release['count'] == before['case']['call_count']
        and original_scope['consumer_completions'] == release['count'],
        'source, detached code or original fixed profile differs')
    _original(seed, observation, original, before, detached, read_artifact(oracle_ref, json_document=False))
    # Width and hardware are intentionally absent from the data ABI identity.
    logical = dict(source=seed['entry']['sha256'], seed=seed_ref['sha256'], argv=seed['argv'],
        detached=anchor['detached']['sha256'], provider=manifest['numeric']['sha256'],
        digest_provider=contract['provider']['libcrypto']['sha256'],
        size=before['case']['size'], nnz=before['case']['nnz'])
    identity = 'recipe-v1:' + semantic_sha256(logical)
    hardware_ref = _ref(hardware_path, HARDWARE)
    plan, control = _resources(before, hardware_ref, read_artifact(hardware_ref), count, width, identity)
    wire = project_source_csr_stream(plan, detached)
    return dict(schema=SCHEMA, schema_version=1, anchor=anchor, source_seed=seed_ref, argv=seed['argv'],
        original_observation=observation_ref, oracle=oracle_ref, hardware=hardware_ref,
        width=width, resources=plan, control=control, wire_identity=logical,
        source_stream_input=wire['input'], source_stream_header=wire['header'],
        decision='CALIBRATION_ELIGIBLE_STANDALONE', historical_costs_transferred=False,
        full_original_effect_equivalence=False, deployment_authorized=False)


def verify_profile_plan(path):
    stored = read_artifact(_ref(path, SCHEMA))
    replay = plan_profile(Path(stored['anchor']['package']['uri']).parent,
        stored['source_seed']['uri'], stored['hardware']['uri'], stored['oracle']['uri'], stored['width'],
        original_observation_path=stored['original_observation']['uri'])
    _require(replay == stored, 'count profile differs from its exact cold replay')
    return stored


def _full_package(reference, profile_ref, profile):
    package = read_artifact(reference)
    root = Path(reference['uri']).parent
    resources = profile['resources']
    _require(reference['schema'] == FULL_PACKAGE
        and package['profile_plan'] == profile_ref and package['source_seed'] == profile['source_seed']
        and package['fixed_argv'] == profile['argv'] and package['recipe_id'] == resources['recipe_id']
        and package['cpu_ids'] == resources['placement']['launch_cpu_ids']
        and package['owner_cpu'] == resources['placement']['owner_cpu_id'],
        'measured native package has another placement or count scope')
    for part in ('solver', 'upstream', 'plan', 'executable'):
        member = _local(root, package[part])
        path = Path(member['uri'])
        _require(not Path(package[part]['uri']).is_absolute()
            and path.is_relative_to(root) and path.resolve(strict=True) == path,
            'candidate member must be within its native package')
        read_artifact(member, json_document=False)
    solver = read_artifact(_local(root, package['solver']))
    _require(solver['profile_plan'] == profile_ref and solver['recipe_id'] == package['recipe_id']
        and solver['width'] == profile['width'] and solver['count'] == resources['case']['call_count'],
        'measured native solver differs from its exact plan')
    return read_artifact(_local(root, package['plan'])), (package['upstream'], solver['numeric'],
        solver['sha256_provider'])


def select_profile(benchmark_path):
    """Use only matched samples from the new count and keep the narrower proven width on overlap."""
    from scar.optimization.native_ordered_publication import ENV

    benchmark = _ref(benchmark_path, BENCHMARK)
    observed = read_artifact(benchmark)
    plans, packages = observed['plans'], observed['packages']
    _require(observed['schema_version'] == 1 and plans.keys() == packages.keys()
        and len(plans) >= 2 and all(re.fullmatch('[1-9][0-9]*', name) for name in plans),
        'a matched count study needs two or more qualified widths')
    profiles, native_launches, native_providers = {}, [], []
    for name, profile_ref in plans.items():
        _require(profile_ref == _ref(profile_ref['uri'], SCHEMA), 'profile changed before the measured comparison')
        profile = verify_profile_plan(profile_ref['uri'])
        _require(name == str(profile['width']), 'benchmark width does not name its exact physical plan')
        native_launch, providers = _full_package(packages[name], profile_ref, profile)
        native_launches.append(native_launch)
        native_providers.append(providers)
        profiles[name] = profile
    widths = sorted(profiles, key=int)
    first = profiles[widths[0]]
    oracle = read_artifact(first['oracle'], json_document=False)
    scope = ('wire_identity', 'source_seed', 'argv', 'oracle', 'original_observation', 'hardware')
    _require(all(all(row[key] == first[key] for key in scope) for row in profiles.values())
        and all(providers == native_providers[0] for providers in native_providers),
        'benchmark arms do not share one exact original, upstream route and numeric/digest provider')
    launch = observed['launch']
    seed = read_artifact(first['source_seed'])
    context = read_artifact(first['anchor']['release'])['context']
    environment = {**ENV, **{key:value for key,value in context['environment'].items() if value is not None}}
    cpus = sorted({cpu for row in profiles.values() for cpu in row['resources']['placement']['launch_cpu_ids']})
    _require(launch['argv'] == first['argv'] and launch['entry'] == seed['entry']
        and launch['cpu_ids'] == cpus and launch['cwd'] == context['cwd']
        and launch['environment'] == environment and launch['interpreter'] == context['interpreter']['executable']
        and launch['stdin'] == 'DEVNULL' and
        all(row['cwd'] == launch['internal_native_cwd'] and row['environment'] == ENV for row in native_launches),
        'benchmark launch differs from the common source scope')

    warmup, repeats = observed['warmup_rounds'], observed['measured_rounds']
    _require(type(warmup) is int and warmup >= 1 and type(repeats) is int and repeats >= 2,
        'expected recorded warmups and independent measured repeats')
    arms = {'original': [], **{name: [] for name in widths}}
    raw = observed['raw']
    keys = Counter((row['round'], row['case']) for row in raw)
    _require(keys == Counter((index, case) for index in range(warmup+repeats) for case in arms),
        'some measurement rounds omitted or repeated a width')
    digest = sha256(oracle).hexdigest()
    for row in raw:
        elapsed = row['elapsed_ns']
        _require(row['stdout_sha256'] == digest and row['stderr_size'] == 0
            and type(elapsed) is int and elapsed > 0, 'fresh-process output or duration differs from fixed oracle')
        if row['round'] >= warmup:
            arms[row['case']].append((row['round'], elapsed))
    times = {case: [elapsed for _, elapsed in sorted(items)] for case, items in arms.items()}
    summary = {case: dict(samples_ns=values, median_ns=median(values), min_ns=min(values), max_ns=max(values),
        sample_stdev_ns=stdev(values)) for case, values in times.items()}
    _require(observed['arms'] == {case: {key: val for key,val in row.items() if key != 'sample_stdev_ns'}
        for case,row in summary.items()}, 'archived arm summary differs from the recorded samples')

    # Keep the same conservative relative margin as the publication comparison.
    minimum = 200
    def robustly_faster(case, other):
        upper, lower = summary[case]['max_ns'], summary[other]['min_ns']
        return repeats >= 3 and upper < lower and upper*10000 <= lower*(10000-minimum)

    robust = [case for case in widths if all(robustly_faster(case, other) for other in widths if other != case)]
    selected = robust[0] if robust else widths[0]
    recommended = selected if robustly_faster(selected, 'original') else 'original'
    return dict(schema=SELECTION, schema_version=1, benchmark=benchmark, selected_width=int(selected),
        selected_plan=plans[selected], selected_package=packages[selected], measurements=summary,
        policy=dict(minimum_reduction_basis_points=minimum, insufficient_rounds=repeats < 3,
            bound_scope='OBSERVED_SAMPLE_ENVELOPES_ONLY', fallback='RETAIN_NARROWEST_WITHOUT_ROBUST_GAIN'),
        decision='ROBUST_GAIN_WITHIN_MATCHED_WIDTHS' if robust else 'NARROWEST_WITHOUT_ROBUST_GAIN',
        recommended_arm=recommended,
        recommendation='MEASURED_NATIVE_GAIN' if recommended != 'original' else 'KEEP_ORIGINAL_NO_ROBUST_GAIN',
        exact_stdout_sha256=digest, original_to_selected_median_speedup=
        summary['original']['median_ns']/summary[selected]['median_ns'],
        historical_costs_transferred=False, global_optimum_proven=False,
        full_original_effect_equivalence=False, deployment_authorized=False)


def verify_profile_selection(path):
    stored = read_artifact(_ref(path, SELECTION))
    _require(stored == select_profile(stored['benchmark']['uri']), 'count choice differs from cold replay')
    return stored
