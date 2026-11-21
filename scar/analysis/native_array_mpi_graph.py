"""Describe the single-host MPI realization of a proved array task family."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import independent_array_task_schedule as process
from scar.optimization import independent_array_task_team_schedule as team


SCHEMA = 'scar.native-array-mpi-graph'
MODE = 'ORDERED_LOCAL_MPI'
_DEFAULT_RANK_LIMIT = 384 << 20
_MPI_INT_MAX = 2**31 - 1
_STATUS_BYTES = 4
_TASKSET = '/usr/bin/taskset'
_PRLIMIT = '/usr/bin/prlimit'
_LAUNCHER = '/usr/bin/mpirun'
_COMPILER = '/usr/bin/mpicxx'
_INFO = '/usr/bin/ompi_info'


def _need(ok, reason):
    if not ok:
        raise ValueError('native array MPI graph: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _runtime(environment):
    from scar.codegen.aten_provider import _dependencies

    # Explicit launcher options fix placement. MPI/loader overrides in the
    # recipe would change the provider and resource domain of that launch.
    _need(not any(key.startswith(('LD_', 'OMPI_', 'OPAL_', 'ORTE_', 'PMIX_', 'PRTE_', 'PMI_'))
                  for key in environment), 'unqualified MPI or dynamic-loader launch settings')

    def inspect(*argv):
        result = subprocess.run(argv, env=dict(environment, LC_ALL='C'), capture_output=True, text=True,
                                timeout=10, check=True)
        _need(not result.stderr, 'runtime inspection produced stderr')
        return result.stdout.strip()

    launcher_version = inspect(_LAUNCHER, '--version').splitlines()[0]
    match = re.fullmatch(r'mpirun \(Open MPI\) (4\.1\.\d+)', launcher_version)
    _need(match is not None, 'Open MPI 4.1 command-line contract required')
    version = match.group(1)
    wrapper_version = inspect(_COMPILER, '--showme:version')
    _need(wrapper_version == f'{_COMPILER}: Open MPI {version} (Language: C++)', 'compiler and launcher versions differ')
    compiler_driver, = shlex.split(inspect(_COMPILER, '--showme:command'))
    _need(compiler_driver == 'g++', 'qualified GNU MPI C++ wrapper required')
    include_dirs = shlex.split(inspect(_COMPILER, '--showme:incdirs'))
    library_dirs = shlex.split(inspect(_COMPILER, '--showme:libdirs'))
    compile_flags = shlex.split(inspect(_COMPILER, '--showme:compile'))
    link_flags = shlex.split(inspect(_COMPILER, '--showme:link'))
    libraries = shlex.split(inspect(_COMPILER, '--showme:libs'))
    _need(include_dirs and library_dirs and all(Path(path).is_absolute() and Path(path).is_dir()
                                               for path in (*include_dirs, *library_dirs))
          and compile_flags == ['-I' + path for path in include_dirs]
          and libraries == ['mpi_cxx', 'mpi']
          and link_flags == [*('-L' + path for path in library_dirs), *('-l' + name for name in libraries)],
          'MPI wrapper flags fall outside the qualified include/link contract')
    mpi_headers = {Path(path, 'mpi.h').resolve() for path in include_dirs if Path(path, 'mpi.h').is_file()}
    mpi_libraries = {Path(path, 'libmpi.so').resolve() for path in library_dirs if Path(path, 'libmpi.so').is_file()}
    _need(len(mpi_headers) == len(mpi_libraries) == 1, 'a single MPI header and provider ELF are required')
    header = _ref(mpi_headers.pop(), 'scar.native-provider-header')
    library = _ref(mpi_libraries.pop(), 'scar.native-provider-library')
    tags = inspect('/usr/bin/readelf', '-Wd', library['uri'])
    sonames = re.findall(r'\(SONAME\).*?\[([^]]+)\]', tags)
    _need(len(sonames) == 1 and re.fullmatch(r'libmpi\.so\.\d+', sonames[0]) is not None,
          'MPI C library SONAME is unresolved')
    dependencies = {}
    for name, path in (('launcher', _LAUNCHER), ('libmpi', library['uri'])):
        discovered = _dependencies(inspect('/usr/bin/ldd', path))
        _need(discovered and all(soname == 'linux-vdso.so.1' if item is None else Path(item).is_absolute()
                                for soname, item in discovered.items()),
              'MPI ELF dependency paths are unresolved')
        dependencies[name] = {soname: _ref(item, 'scar.native-provider-library')
                              for soname, item in discovered.items() if item is not None}
    return dict(implementation='Open MPI', version=version, launcher=_ref(_LAUNCHER, 'scar.native-mpi-launcher'),
                launcher_path=_LAUNCHER, compiler=_ref(_COMPILER, 'scar.native-mpi-compiler-wrapper'),
                compiler_path=_COMPILER, compiler_driver=_ref('/usr/bin/g++', 'scar.native-compiler-binary'),
                info=_ref(_INFO, 'scar.native-mpi-info'), mpicxx_version=wrapper_version, header=header, libmpi=library,
                libmpi_soname=sonames[0], dependencies=dependencies, compile_flags=compile_flags, link_flags=link_flags,
                include_dirs=include_dirs, library_dirs=library_dirs, mpi_libraries=libraries,
                mpi_rank_env='OMPI_COMM_WORLD_RANK', taskset=_ref(_TASKSET, 'scar.native-affinity-tool'),
                prlimit=_ref(_PRLIMIT, 'scar.native-resource-limit-tool'))


def _rank_graph(scope, lanes):
    count = scope['task_count']
    width = len(lanes)
    ranks = []
    offset = 0
    for rank, lane in enumerate(lanes):
        tasks = (count - rank - 1) // width + 1
        size = tasks * process.RESULT_BYTES
        ranks.append(dict(rank=rank, cpu_id=lane['cpu_id'], physical_core=lane['physical_core'],
                          numa_node=lane['numa_node'], first_ordinal=rank, ordinal_stride=width, task_count=tasks,
                          local_result_bytes=size, gather_displacement_bytes=offset,
                          off_rank_result_payload_bytes=0 if rank == 0 else size))
        offset += size
    collective = dict(status=dict(operation='MPI_Allreduce', datatype='MPI_INT', elements=1, reduction='MPI_MAX',
                                  application_bytes_per_rank=_STATUS_BYTES, proceed_when=0),
                      results=dict(operation='MPI_Gatherv', datatype='MPI_BYTE', root_rank=0,
                                   recvcounts_bytes=[row['local_result_bytes'] for row in ranks],
                                   displacements_bytes=[row['gather_displacement_bytes'] for row in ranks],
                                   application_receive_extent_bytes=offset,
                                   off_rank_payload_bytes=offset - ranks[0]['local_result_bytes'],
                                   transport_framing_bytes=None))
    return dict(placement='ONE_NUMA_NODE_DISTINCT_PHYSICAL_CORES', rank_count=width, ranks=ranks,
                task_owner='ORDINAL_MOD_RANK_COUNT', input=dict(source='REPLICATED_LAUNCH_ARGV',
                    task_request_bytes=process.REQUEST_BYTES, task_request_wire_bytes=0,
                    effective_seed='BASE_SEED_PLUS_ORDINAL', round_count='SHARED_ORIGINAL_ROUND_COUNT'),
                collective=collective, rank0_computes=True, task_state='PRIVATE_PER_RANK',
                sequence=['INITIALIZE_MPI', 'COMPUTE_PRIVATE_ORDINALS', 'REDUCE_STATUS', 'GATHER_IF_ALL_SUCCEEDED',
                          'RESTORE_ORIGINAL_ORDINAL_ORDER', 'FINALIZE_MPI', 'PUBLISH_FROM_RANK_ZERO'],
                publication=dict(rank=0, result_order='ASCENDING_ORDINAL', after_all_workers=True))


def _capacity_reasons(hardware, node, budget, width, memory):
    reasons = []
    if not hardware['hierarchy_observation_complete']:
        reasons.append('UNRESOLVED_CGROUP_HIERARCHY')
    quota = hardware['cpu_quota_equivalents']
    if quota is not None and quota[0] < width * quota[1]:
        reasons.append('CPU_QUOTA_BELOW_CONFIGURED_LANES')
    headroom = memory['required_headroom_bytes']
    if headroom > budget:
        reasons.append('EXCEEDS_DECLARED_PROCESS_MEMORY_BUDGET')
    locality = hardware['node_free_snapshot'].get(str(node), {})
    available = locality.get('bytes')
    if locality.get('allowed') is not True or type(available) is not int:
        reasons.append('UNRESOLVED_NUMA_HEADROOM')
    elif available < headroom:
        reasons.append('EXCEEDS_OBSERVED_NUMA_FREE_MEMORY')
    cgroup = hardware['finite_memory_headroom_snapshot_bytes']
    if cgroup is not None and headroom > cgroup:
        reasons.append('EXCEEDS_OBSERVED_CGROUP_HEADROOM')
    if memory['retained_result_payload_bytes'] > _MPI_INT_MAX:
        reasons.append('MPI_GATHERV_BYTE_COUNT_EXCEEDS_INT_RANGE')
    if memory['explicit_owner_payload_bytes'] >= memory['coordinator_address_space_limit_bytes']:
        reasons.append('EXPLICIT_ARRAY_AND_RESULT_BUFFERS_EXCEED_ADDRESS_SPACE_CAP')
    return reasons


def _alternative(scope, lanes, launch_cpus, rank_limit, budget, arena, hardware):
    width = len(lanes)
    rank_graph = _rank_graph(scope, lanes)
    collection = rank_graph['collective']['results']
    retained = collection['application_receive_extent_bytes']
    root_local = collection['recvcounts_bytes'][0]
    owner_bytes = arena + process.REQUEST_BYTES + root_local + 2 * retained + 2 * width * _STATUS_BYTES
    worker_bytes = arena + process.REQUEST_BYTES + max(collection['recvcounts_bytes'][1:])
    memory = dict(private_arena_bytes_per_numerical_worker=arena, request_bytes=process.REQUEST_BYTES,
                  result_bytes=process.RESULT_BYTES, request_frame_bytes=0, result_frame_bytes=None,
                  payload_budget_bytes=collection['off_rank_payload_bytes'], retained_result_payload_bytes=retained,
                  explicit_owner_payload_bytes=owner_bytes, explicit_worker_payload_bytes=worker_bytes,
                  worker_address_space_limit_bytes=rank_limit, coordinator_address_space_limit_bytes=rank_limit,
                  launcher_address_space_limit_bytes=rank_limit, required_headroom_bytes=(width + 1) * rank_limit,
                  bound_scope='HARD_VAS_CAP_PER_RANK_AND_LAUNCHER_PLUS_EXPLICIT_RESULT_BUFFERS',
                  physical_reservation=False, mpi_transport_buffers_in_process_vas_cap=True)
    prefix = [_TASKSET, '-c', ','.join(map(str, launch_cpus)), _PRLIMIT, f'--as={rank_limit}:{rank_limit}',
              _LAUNCHER, '--bind-to', 'none', '--map-by', 'slot', '--host', f'localhost:{width}', '-np', str(width)]
    return dict(mode=MODE, owner_cpu_id=lanes[0]['cpu_id'], worker_count=width,
                worker_cpu_ids=[lane['cpu_id'] for lane in lanes], window=width, launch_cpu_ids=launch_cpus,
                launch_prefix=prefix, rank_graph=rank_graph, memory=memory,
                capacity_reasons=_capacity_reasons(hardware, lanes[0]['numa_node'], budget, width, memory),
                scheduled_pod_result_lifetime='RETAIN_IN_OWNER_UNTIL_ALL_TASKS_HAVE_SUCCEEDED')


def project(team_plan_path, *, rank_limit_bytes=_DEFAULT_RANK_LIMIT):
    """Derive rank and communication layouts without generating a workload."""
    team_ref = _ref(team_plan_path, team.SCHEMA)
    plan = team.check(team_ref['uri'])
    scope = process._scope(**plan['scope'])
    family, native, hardware = map(read_artifact, (plan['family'], plan['native'], plan['hardware']))
    _need(family['state']['kind'] == 'PER_TASK_PRIVATE' and family['publication']['after_all_workers'] is True
          and family['publication']['result_order'] == 'ASCENDING_ORDINAL'
          and family['ordinal_range'] == {'start': 0, 'step': 1} and family['effective_seed']['operation'] == 'ADD',
          'qualified source task ownership and ordered output are required')
    _need(type(rank_limit_bytes) is int and 0 < rank_limit_bytes <= process._MAX_I64,
          'a finite positive rank/launcher address-space cap is required')
    serial = plan['alternatives'][0]
    cpus = serial['launch_cpu_ids']
    _, lanes = process._cohort(hardware, serial['owner_cpu_id'])
    _need(lanes == plan['allowed_cpu_lanes'] and set(lane['cpu_id'] for lane in lanes) <= set(cpus),
          'MPI rank placement differs from the qualified physical cohort')
    maximum = min(len(lanes), scope['task_count'], scope['requested_workers'])
    alternatives = [_alternative(scope, lanes[:width], cpus, rank_limit_bytes, plan['budget_bytes'],
                                 native['arena_bytes'], hardware) for width in range(2, maximum + 1)]
    recipe = read_artifact(native['recipe'])
    runtime = _runtime(recipe['environment'])
    return dict(schema=SCHEMA, schema_version=1, team_plan=team_ref, family=plan['family'], native=plan['native'],
                hardware=plan['hardware'], scope=scope, runtime=runtime, rank_limit_bytes=rank_limit_bytes,
                alternatives=alternatives, rank_placement=dict(domain='LOCALHOST_ONLY', rank0_is_worker=True,
                    affinity_guard='INHERITED_LAUNCH_MASK_THEN_PIN_RANK_TO_PLANNED_CPU',
                    verify_environment_rank_against_communicator=True),
                common_processes=dict(launcher_has_one_address_space_cap=True, launcher_shared_cpu_cohort=cpus,
                                      transport_service_cost='UNMEASURED'), deployment_authorized=False)


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    actual = project(stored['team_plan']['uri'], rank_limit_bytes=stored['rank_limit_bytes'])
    _need(stored == actual, 'stored MPI graph differs during replay')
    return actual


def command(row, binary, argv):
    """Supply the exact qualified launcher prefix for one feasible alternative."""
    binary = Path(binary)
    args = tuple(argv)
    _need(row['mode'] == MODE and not row['capacity_reasons'] and binary.is_absolute(),
          'a feasible MPI alternative and absolute executable path are required')
    _need(all(type(arg) is str for arg in args), 'explicit argv strings required')
    return [*row['launch_prefix'], str(binary), *args]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--team-plan')
    action.add_argument('--check')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--rank-limit-bytes', type=int, default=_DEFAULT_RANK_LIMIT)
    args = parser.parse_args(argv)
    if args.check and args.out is None and args.rank_limit_bytes == _DEFAULT_RANK_LIMIT:
        result = check(args.check)
    elif args.team_plan and args.out is not None:
        result = project(args.team_plan, rank_limit_bytes=args.rank_limit_bytes)
        process._write(args.out, result)
    else:
        parser.error('provide --check RECEIPT or --team-plan PLAN --out RECEIPT')
    print(json.dumps(dict(runtime=result['runtime']['version'], feasible_ranks=[row['worker_count']
                          for row in result['alternatives'] if not row['capacity_reasons']])))


if __name__ == '__main__':
    main()
