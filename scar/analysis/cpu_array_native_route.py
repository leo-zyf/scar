"""Bind two qualified native execution routes to one retained array-task scope."""

from pathlib import Path

from scar.analysis.replay_transaction import detached_replay
from scar.codegen.native_cpu_arena_reuse import _numerical_links
from scar.codegen.native_cpu_fp_guard_groups import verify_native_cpu_fp_guard_groups
from scar.codegen.cpu_catalog_runtime_package import _loader
from scar.compiler.array_task_source_provider import _profile
from scar.compiler.cpu_entry_replay_adapters import _match_json_report
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-array-native-route-proof'
_PROCESS = 'ORDERED_LOCAL_PROCESSES'
_OPENMP = 'ORDERED_OPENMP'
_ROLE_FIELDS = ('mode', 'owner_cpu_id', 'worker_cpu_ids', 'worker_count', 'launch_cpu_ids', 'memory', 'window')
_FORBIDDEN_ENV = {'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'GOMP_STACKSIZE',
                  'SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT'}


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU array native route: ' + reason)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _identity(reference):
    return tuple(reference[key] for key in ('schema', 'uri', 'sha256', 'size_bytes'))


def _content(reference):
    return {key: reference[key] for key in ('sha256', 'size_bytes')}


def _machine_identity(links):
    def relocatable(row):
        return {name: value for name, value in row.items() if name != 'address'}

    return dict(stock={role: relocatable(row) for role, row in links['stock']['functions'].items()},
                normalization=relocatable(links['normalization']), random=links['random'])


def _binary_loader(reference, package, executable, cwd, environment):
    root = Path(reference['uri']).parent
    files = {'project/' + name: _content(ref) for name, ref in package['files'].items()
             if name.startswith('provider/objects/')}
    result = _loader(Path(executable['uri']), root, files, cwd, environment)
    _need(result['dynamic'] == package['linkage']['dynamic']
          and {row['path'] for row in result['dependencies'].values() if row['kind'] == 'PACKAGE_MEMBER'} == files.keys(),
          'actual loader route differs from the qualified native package')
    return result


def _providers(local_ref, local, grouped_ref, grouped, local_executable, cwd, local_env, grouped_env):
    openmp_executable = grouped['executable']
    a = _binary_loader(local_ref, local, local_executable, cwd, local_env)
    b = _binary_loader(grouped_ref, grouped, openmp_executable, cwd, grouped_env)
    shared, candidate_only = a['dependencies'], b['dependencies'].keys() - a['dependencies'].keys()
    locks = grouped['linkage']['qualified_runtime_dependencies']
    _need(all(b['dependencies'].get(name) == value for name, value in shared.items())
          and set(candidate_only) == set(locks) == {'libgomp.so.1'}, 'unqualified loader differences between task routes')
    for name, reference in locks.items():
        row = b['dependencies'][name]
        _need(row == dict(kind='HOST_FILE', path=reference['uri'], **_content(reference)),
              'grouped OpenMP runtime does not bind its exact host library')
        _raw(reference)

    local_links = _numerical_links(local, local_executable['uri'])
    _need(local_links == {key: local['linkage'][key] for key in local_links},
          'retained local-process numerical closure changed')
    numerical = _machine_identity(local_links)
    _need(numerical == _machine_identity(grouped['linkage']['numerical']),
          'grouped candidate changes a linked stock numerical function')
    return dict(numerical=numerical, common_libraries=shared, openmp_runtime=locks,
                local_dynamic=a['dynamic'], openmp_dynamic=b['dynamic'])


def _placement(local, omp, profile, schedule):
    cpus = profile['invocation']['cpu_ids']
    hardware = read_artifact(schedule['hardware'])
    processors = {row['cpu_id']: row for row in hardware['processors'] if row['permitted']}
    left, right = local['decision'], omp['decision']
    _need(local['cpu_ids'] == omp['cpu_ids'] == cpus and bool(cpus)
          and len(set(cpus)) == len(cpus) and set(cpus) <= processors.keys()
          and left['owner_cpu_id'] not in left['worker_cpu_ids'] and right['worker_cpu_ids'][0] == right['owner_cpu_id'],
          'original launch affinity or process/OpenMP ownership differs')
    for choice in (left, right):
        _need(choice['launch_cpu_ids'] == cpus and choice['owner_cpu_id'] in cpus
              and choice['worker_count'] == len(choice['worker_cpu_ids']) == len(set(choice['worker_cpu_ids']))
              and set(choice['worker_cpu_ids']) <= set(cpus) and not choice['capacity_reasons'],
              'worker placement is outside the original qualified topology')
    return dict(hardware=schedule['hardware'], launch_cpu_ids=cpus,
                local_processes={name:left[name] for name in _ROLE_FIELDS},
                openmp_grouped={name:right[name] for name in _ROLE_FIELDS},
                physical_reservation_proven=False)


def derive_cpu_array_native_route(retention_ref, baseline_grouped_ref):
    """Replay exact same-workload and provider identities without running an entry."""
    identity = _identity(retention_ref), _identity(baseline_grouped_ref)
    return detached_replay('scar.cpu-array-native-route.v1', identity,
        lambda: _derive(retention_ref, baseline_grouped_ref))


def _derive(retention_ref, baseline_grouped_ref):
    source, profile = _profile(retention_ref)
    retained = read_artifact(retention_ref)
    accepted = read_artifact(retained['existing_acceptance'])
    local_ref = profile['runtime']
    local = read_artifact(local_ref)
    grouped = verify_native_cpu_fp_guard_groups(baseline_grouped_ref)
    omp_ref = grouped['runtime']
    omp = read_artifact(omp_ref)
    schedule = read_artifact(local['schedule'])
    recipe = read_artifact(profile['recipe'])
    holdout = read_artifact(profile['holdout'])
    prior = read_artifact(local['native'])
    scope = local['scope']
    roles = local['input_options']
    expected_scope = {key:holdout['parameters'][roles[role]] for key, role in (
        ('task_count', 'task_count'), ('round_count', 'round_count'), ('requested_workers', 'worker_limit'))}
    _need(retained['retained_release'] == local_ref and accepted['entry'] == omp_ref
          and accepted['holdout'] == profile['holdout'] and accepted['parameters'] == holdout['parameters']
          and local['decision']['mode'] == _PROCESS and omp['decision']['mode'] == _OPENMP
          and grouped['decision'] == omp['decision'] and grouped['cpu_ids'] == omp['cpu_ids']
          and scope == expected_scope == grouped['scope'] == omp['scope'] == schedule['scope']
          and local['schedule'] == grouped['schedule'] == omp['schedule'] == accepted['schedule']
          and local['native'] == omp['native'] == schedule['native']
          and local['family'] == omp['family'] == schedule['family']
          and local['numerical_body_reused'] == omp['numerical_body_reused'] == prior['files']['src/profile_0.cpp']
          and local['recipe'] == grouped['recipe'] == omp['recipe'] == profile['recipe'] == prior['recipe']
          and roles == omp['input_options'] and local['output_fields'] == omp['output_fields'],
          'native routes do not share the same original schedule, parameters and private numerical body')
    _need(local['no_worker_visible_output'] is True and omp['no_worker_visible_output'] is True
          and local['preserves_report_schema'] is True and omp['preserves_report_schema'] is True
          and local['explicit_checkpoint_diagnostics'] is True and omp['explicit_checkpoint_diagnostics'] is True,
          'native ordered reports or diagnostic scope differ')
    effects = read_artifact(grouped['proof'])['proof']
    _need(effects['normal_publication_unchanged'] is True and effects['fp_operation_order_unchanged'] is True
          and effects['guard_provider_exact'] is True, 'grouped guards do not preserve the scoped publication')
    invocation = profile['invocation']
    original_env = invocation['original_environment']
    local_env = invocation['environment']
    grouped_env = dict(original_env, **grouped['launch_environment'])
    _need(local_env == dict(original_env, **local['launch_environment'])
          and grouped['launch_environment'] == omp['launch_environment']
          and not ((_FORBIDDEN_ENV & local_env.keys()) | (_FORBIDDEN_ENV & grouped_env.keys()))
          and all(env.get(key) == '1' for env in (local_env, grouped_env)
                  for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')),
          'native route requires its original clean single-provider-thread launch scope')
    oracle = holdout['normalized']
    publication = holdout['files'][recipe['output_option']]
    published = holdout['verified_publication']
    _need(_match_json_report(_raw(holdout['stdout']), _raw(publication), recipe, _raw(oracle))
          and published['stdout_matches_file'] is True and published['original_stderr_empty'] is True
          and published['state_vectors_saved'] is False and published['order'] == 'INCREASING_SOURCE_ORDINAL'
          and published['number_of_reported_results'] == scope['task_count']
          and published['source_argv_output'] == profile['argv'][invocation['output_index']]
          and holdout['independent_original_run'] is True
          and holdout['normalized_runtime_fields'] == profile['normalization'],
          'retained holdout does not establish an exact normalized publication oracle')
    providers = _providers(local_ref, local, baseline_grouped_ref, grouped, invocation['executable'],
                           invocation['cwd'], local_env, grouped_env)
    resources = _placement(local, omp, profile, schedule)
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_SCOPED_CPU_ARRAY_NATIVE_ROUTE',
        retention=retention_ref, baseline_grouped=baseline_grouped_ref, retained_runtime=local_ref,
        grouped_runtime=omp_ref, scope=scope,
        workload=dict(source=source, argv=profile['argv'], holdout=profile['holdout'], recipe=profile['recipe'],
            cwd=invocation['cwd'], output_index=invocation['output_index'], normalization=profile['normalization'],
            normalized_oracle=oracle, original_environment=original_env, local_process_environment=local_env,
            openmp_grouped_environment=grouped_env, original_interpreter=invocation['original_executable']),
        resources=resources, effects=dict(scope='SCOPED_NORMAL_SUCCESS_NUMERICAL_AND_ORDERED_PUBLICATION',
            same_task_seed_and_round_routes=True, same_numeric_provider_bytes=True, normalized_report_oracle_pinned=True,
            ordered_publication_contract_preserved=True, requires_fresh_paired_validation=True,
            original_python_equivalence_proven=False, failure_and_import_effects_closed=False),
        evidence=dict(source_family=local['family'], original_numerical_body=local['native'],
            periodic_memory=prior['periodic_memory'], storage=prior['memory'], schedule=local['schedule'],
            existing_acceptance=retained['existing_acceptance'], sticky_guard_proof=grouped['proof'],
            retained_executable=invocation['executable'], openmp_executable=grouped['executable'], providers=providers),
        target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_array_native_route(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved == derive_cpu_array_native_route(
        saved['retention'], saved['baseline_grouped']), 'saved native route differs on cold replay')
    return saved
