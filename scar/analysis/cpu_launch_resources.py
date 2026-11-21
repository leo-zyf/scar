"""Replay CPU launch resources without inferring descendant constraints."""

from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.cpu_provider_graph import SCHEMA as CPU_PROVIDER
from scar.analysis.cpu_stream_graph import SCHEMA as CPU_STREAM, _retained
from scar.analysis.entry_effect_frontier import verify_entry_effect_frontier
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.target_graph import TargetHardwareGraph
from scar.compiler import cpu_entry_replay
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode

SCHEMA = 'scar.f4-cpu-launch-resources'
_LIMITS = {'RLIMIT_AS': 'bytes', 'RLIMIT_DATA': 'bytes', 'RLIMIT_STACK': 'bytes',
           'RLIMIT_CPU': 'cpu_seconds', 'RLIMIT_NPROC': 'os_count'}


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU launch resources: ' + detail)


def _inventory(graph):
    receipt = graph.report['inventory']
    constraints = graph.constraints()
    _need(len(constraints) <= 1, 'expected one querying-task constraints snapshot')
    if not constraints:
        return dict(inventory=receipt, scope='QUERYING_PROCESS_ONLY', constraints=None, rlimits={}, source=None)
    original = graph.inventory['execution_constraints']
    source_ids = set(original['source_ids'])
    sources = [(index, source) for index, source in enumerate(graph.inventory['sources'])
               if source['id'] in source_ids and source['kind'] == 'self_process_limits']
    _need(len(sources) <= 1, 'querying-task rlimit source is ambiguous')
    limits, source = {}, None
    if sources:
        index, observation = sources[0]
        snapshot = json.loads(observation['text'])
        _need(observation['scope'] == 'QUERYING_PROCESS_ONLY'
              and snapshot['affinity_cpu_ids'] == original['affinity_cpu_ids']
              and snapshot['rlimits'] == original['rlimits'],
              'retained querying-task rlimits differ from their original source')
        source = f'#/sources/{index}'
        for name, unit in _LIMITS.items():
            if name in snapshot['rlimits']:
                soft, hard = snapshot['rlimits'][name]
                _need(all(type(item) is int and item >= -1 for item in (soft, hard)),
                      'unexpected retained Linux rlimit')
                limits[name] = dict(soft_unlimited=soft == -1, hard_unlimited=hard == -1,
                    soft=None if soft == -1 else soft, hard=None if hard == -1 else hard, unit=unit)
    return dict(inventory=receipt, scope='QUERYING_PROCESS_ONLY', constraints=encode(constraints[0]),
                rlimits=limits, source=source, cgroup_ancestors_complete=False)


def _launch(graph):
    if graph is None:
        return None
    if graph['schema'] == CPU_STREAM:
        owner = graph['original_observer']
        cpu = owner.get('taskset_cpu')
        cohort = None if cpu is None else [cpu]
        _need(cohort is None or cohort == [row['cpu_id'] for row in graph['requested_cpu_scope']],
              'original stream launcher and requested CPU scope differ')
        interpreter = owner['interpreter_at_capture_path']
        read_artifact(interpreter, json_document=False)
        capture, _, _, _, retained_cpu = _retained(graph['capture'], dict(cpu_stream_capture=graph['capture'],
            argv=owner['argv'], process_affinity_cpu_ids=cohort, working_directory=owner['cwd'],
            python_executable=interpreter['uri'], environment_sha256=owner['environment_sha256']))
        _need(capture['source'] == graph['source'] and capture['captured_launch'] == owner['launcher']
              and retained_cpu == cpu, 'original retained launcher differs from the stream graph')
        return dict(source=graph['source'], argv=owner['argv'], capture=graph['capture'], provider=graph['provider'],
            launch_evidence=owner['launcher'], cwd=owner['cwd'], environment_sha256=owner['environment_sha256'],
            scope='ORIGINAL_OBSERVER_TASKSET' if cohort is not None else 'ORIGINAL_OBSERVER_PLACEMENT_UNRECORDED',
            cpu_ids=cohort, interpreter=interpreter, bound_original_argv=None)
    _need(graph['schema'] == CPU_PROVIDER, 'unsupported CPU provider observation')
    captured = None
    if graph['observable_contract']['stdin_kind'] == 'ADMITTED_PACED_TRANSCRIPT':
        stream = graph.get('stream_capture')
        captured = read_artifact(graph['replay'])['capture'] if stream is None else stream['capture']
    return dict(source=graph['entry']['source'], argv=graph['argv'], provider=graph['provider'],
        launch_evidence=graph['replay'], cwd=graph['working_directory'],
        environment_sha256=graph['original_environment_sha256'], scope='PAIRED_LAUNCH_TASKSET',
        cpu_ids=[row['cpu_id'] for row in graph['observed_launch_cpu_ids']],
        capture=captured, interpreter=None, bound_original_argv=graph['source_launch'])


def _placement(hardware, launch):
    if launch is None or launch['cpu_ids'] is None:
        return []
    threads = {node.cpu_id: node for node in hardware.resources() if type(node) is H.HardwareThread}
    result = []
    for cpu in launch['cpu_ids']:
        thread = threads.get(cpu)
        result.append(dict(cpu_id=cpu, target_thread=None if thread is None else thread.id.wire,
            online_in_snapshot=None if thread is None else thread.online))
    return result


def _replay_resources(reference, launch, anchored):
    report = read_artifact(reference)
    _need(reference['schema'] == cpu_entry_replay.SCHEMA and report['schema_version'] == 1
          and report['whole_program_equivalence_proven'] is False and report['deployment_authorized'] is False,
          'expected bounded paired resource observations')
    spec = report['invocation']
    compilation = read_artifact(report['compilation'])
    source, original = spec['entry'], spec['arms']['original']
    _need(all(source[key] == launch['source'][key] for key in ('uri', 'sha256', 'size_bytes'))
          and compilation['entry'] == source and compilation['argv'] == launch['argv']
          and compilation['provider'] == launch['provider'] and spec['cwd'] == launch['cwd']
          and spec['cpu_ids'] == launch['cpu_ids']
          and report['capture'] == launch['capture'] and
          sha256(json.dumps(sorted(original['environment'].items()), separators=(',', ':')).encode()).hexdigest()
              == launch['environment_sha256'],
          'paired resource observation differs from original source, provider or launch')
    if launch['bound_original_argv'] is not None:
        _need(original['argv'] == launch['bound_original_argv'], 'paired original command differs')
    else:
        executable = original['executable']
        expected = launch['interpreter']
        _need(report['cpu_override'] == launch['cpu_ids'][0]
              and all(executable[key] == expected[key] for key in ('sha256', 'size_bytes'))
              and Path(executable['uri']).resolve() == Path(expected['uri']).resolve()
              and original['argv'] == [executable['uri'], '-B', '-u', source['uri'], *launch['argv']],
              'paired original interpreter or stream command differs')
    warmups, pairs = report['warmup_pairs'], report['measurement_pairs']
    root = Path(reference['uri']).parent
    _need(str(root) == report['output_root'] and type(warmups) is int and 0 <= warmups <= 256
          and type(pairs) is int and 1 <= pairs <= 256 and len(report['runs']) == 2*(warmups+pairs),
          'invalid paired resource sample extent')
    layout = cpu_entry_replay._layout(spec, root, warmups, pairs)
    _need(all(row['status'] == 0 and row['execution'] == execution
              for row, (execution, _) in zip(report['runs'], layout, strict=True)),
          'paired resource rows differ from their original launch layout')
    stored = report.get('resource_observation')
    runner = None if stored is None else stored.get('runner')
    if runner is not None:
        _need(cpu_entry_replay._runner_receipt(root, runner['executable'], runner['source'],
            native_capacity_guard=runner.get('native_capacity_guard')) == runner,
              'isolated resource accounting runner differs at replay')
    observation = cpu_entry_replay._resource_observation(report['runs'], warmups, runner=runner)
    _need(observation == stored, 'paired resource summary differs at replay')
    return dict(reference=reference, status='NO_RECORDED_RESOURCE_SAMPLES' if observation is None
                else 'RETAINED_PAIRED_RESOURCE_SAMPLES', linked_by_original_graph=anchored,
                observation_pointer=None if observation is None else '#/resource_observation',
                observation=observation)


def derive_cpu_launch_resources(entry_effect_frontier, *, replay=None):
    """Bind original F4, target and optional paired metadata using no live probes."""
    effects = verify_entry_effect_frontier(entry_effect_frontier)
    program = read_artifact(effects['program'])
    target = program['target']
    reference = dict(uri=target['path'], sha256=target['sha256'], size_bytes=target['bytes'],
                     schema='scar.target-hardware-graph')
    with replay_transaction():
        hardware = TargetHardwareGraph(Path(reference['uri']).parent)
        _need(encode(hardware._snapshot) == reference, 'target does not match the original F4 build')
        constraints = _inventory(hardware)
        graph = None if effects['cpu_graph'] is None else read_artifact(effects['cpu_graph'])
        launch = _launch(graph)
        placement = _placement(hardware, launch)
    linked = None
    if graph is not None:
        linked = graph.get('paired_replay') if graph['schema'] == CPU_STREAM else graph['replay']
    _need(linked is None or replay in (None, linked), 'previously linked paired evidence cannot be substituted')
    selected = replay or linked
    _need(selected is None or (launch is not None and launch['cpu_ids'] is not None),
          'paired resource replay needs an exact original launcher cohort')
    paired = None if selected is None else _replay_resources(selected, launch, selected == linked)
    gaps = [
        'ORIGINAL_LAUNCH_CHILD_EFFECTIVE_RLIMITS_UNOBSERVED',
        'ORIGINAL_LAUNCH_CHILD_CGROUP_MEMBERSHIP_AND_EFFECTIVE_LIMITS_UNOBSERVED',
        'PROCESS_AND_THREAD_COUNT_CEILINGS_UNBOUND',
        'DESCENDANT_PROCESS_THREAD_CPU_PLACEMENT_UNOBSERVED',
        'CONCURRENT_TREE_MEMORY_PEAK_UNOBSERVED',
        'ORIGINAL_PROVIDER_CAN_CHANGE_LAUNCH_RESOURCE_STATE',
    ]
    if not placement:
        gaps.append('ORIGINAL_LAUNCH_CPU_SCOPE_UNOBSERVED')
    elif any(row['target_thread'] is None for row in placement):
        gaps.append('ORIGINAL_LAUNCH_CPU_TARGET_ENDPOINT_OPEN')
    if paired is None or paired['observation'] is None:
        gaps.append('PAIRED_RESOURCE_SAMPLES_UNAVAILABLE')
    named = effects['observed']['named_effects']
    effect_context = dict(
        original_calls='#/static/calls', original_imports='#/static/imports',
        original_operation_obligations='#/static/operation_obligations',
        launch_observations=[index for index, row in enumerate(named)
                             if row['kind'] in ('PROCESS_EXIT', 'INVOCATION_CHAIN')])
    return dict(schema=SCHEMA, schema_version=1, entry_effect_frontier=entry_effect_frontier,
        program=effects['program'], cpu_graph=effects['cpu_graph'], target=reference, replay=replay,
        querying_task=constraints, launch=launch, launch_owner_cpu_endpoints=placement,
        paired_resources=paired, original_effect_scope=effect_context, coverage_frontier=gaps,
        inherited_target_limits_proven=False, complete_thread_placement_proven=False,
        physical_memory_reservation_proven=False, transformation_authorized=False)


def verify_cpu_launch_resources(reference):
    report = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and report == derive_cpu_launch_resources(
        report['entry_effect_frontier'], replay=report['replay']), 'saved launch resources differ at replay')
    return report
