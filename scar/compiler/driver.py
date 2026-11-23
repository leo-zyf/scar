"""SCAR compiler driver for locked native build targets."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys


def _publish_result(directory, result):
    from scar.artifacts import _durable_bytes

    payload = json.dumps(result, indent=2, ensure_ascii=False).encode('utf-8') + b'\n'
    _durable_bytes(directory / 'result.json', payload, replace=False)


def _graph_report(graph):
    facts = graph.report
    incomplete = any(facts[key] for key in ('frontiers', 'frontend_frontiers', 'excluded_control'))
    state = 'BLOCKED' if graph.program is None else 'PARTIAL' if incomplete else 'STRUCTURE_VALIDATED'
    counts = ('entry', 'regions', 'operations', 'values', 'locations', 'excluded_control',
              'call_frontiers', 'call_sites', 'call_links', 'reachable_regions')
    report = dict(schema='scar.native-entry-result', schema_version=1,
                graph={key: facts[key] for key in counts}, artifacts=graph.references(),
                frontend_frontiers=len(facts['frontend_frontiers']), cfg_frontiers=len(facts['frontiers']),
                status=f'GRAPH_{state}', selection='KEEP_ORIGINAL',
                layers=dict(source='LOCKED', compiler='CAPTURED', graph=state, hardware='OBSERVED',
                            calls='PARTIAL' if facts['call_frontiers'] or graph.program is None else 'BOUND'))
    if graph.publication is not None:
        report['graph_bundle'] = asdict(graph.publication)
    return report


def compile_native(manifest, out, *, jobs, repetitions, case_seconds, graph_only=False,
                   adapters=None):
    from scar.analysis.replay_transaction import replay_transaction
    from .contracts import lock_native_inputs
    from .frontend.graph import BUILTIN_ADAPTERS, build_graph

    out = out.resolve()
    adapters = BUILTIN_ADAPTERS if adapters is None else tuple(adapters)
    languages = tuple(language for adapter in adapters for language in adapter.languages)
    with replay_transaction():
        inputs = lock_native_inputs(manifest, out, languages=languages)
        graph = build_graph(inputs, out, jobs=jobs, adapters=adapters)
        result = _graph_report(graph)
        if graph_only and graph.program is None:
            result.update(delivery='INCOMPLETE', delivery_error='no validated entry region')
        if not graph_only:
            _compile_graph(graph, out, result, repetitions=repetitions, case_seconds=case_seconds)
    _publish_result(out, result)
    return result


def _plan_graph(graph, out, result, *, repetitions, case_seconds):
    from .backend.cpp20 import build_trial_variant
    from .contracts import verify_source_lock
    from .middle.plan import (
        entry_cpus, publish_plan, replay_native_revision, rewrite_native, schedule_native, seal_native_lowering,
    )
    from .qualification import NativeCases

    layers = result['layers']
    layer = 'rewrite'
    try:
        revision, request = rewrite_native(graph, out)
        result.update(artifacts=revision.artifacts, graph_rewrites=len(revision.rules),
                      source_rewrites=len(revision.edits))
        layers[layer] = 'IR_VALIDATED'
        layers['entry_effects'] = revision.entry_effect_scope
        layer = 'cost'
        if not request.get('baseline') or not request.get('cases'):
            raise ValueError('baseline executable and cases are required to qualify native delivery')
        cpus = entry_cpus(revision.artifacts['hardware'])
        if not cpus:
            raise ValueError('no online CPU is available in the inventoried execution scope')
        cases = NativeCases(request, min(cpus), case_seconds)
        observation = cases.observe(revision.artifacts['request'])
        if observation['exit'] != 0:
            raise ValueError(f'baseline exited with status {observation["exit"]}')
        if revision.edits:
            manifest, edits, types, _ = replay_native_revision(revision.artifacts)
            variants = out / 'variants'

            def build(mode):
                return build_trial_variant(variants / mode, revision.artifacts, manifest,
                                           edits if mode == 'canonical' else [], types, mode=mode)

            cases.calibrate(observation, variants / 'measurements', repetitions, build)
        layers[layer] = 'OBSERVED'
        layer = 'schedule'
        schedule = schedule_native(revision, observation, request['cases'], out)
        verify_source_lock(request)
        schedule = seal_native_lowering(graph, schedule, out)
        retained_plan = publish_plan(out / 'compiler-plan.json', schedule.document())
        result.update(artifacts=dict(schedule.artifacts, compiler_plan=retained_plan), cost_decision=schedule.decision,
                      planning_scope=schedule.scope)
        layers[layer] = 'SCHEDULED'
        return schedule, cases
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        layers[layer] = 'BLOCKED'
        result.update(delivery='INCOMPLETE', delivery_error=str(error))
        return None, None


def _compile_graph(graph, out, result, *, repetitions, case_seconds):
    from .backend.cpp20 import deliver

    schedule, cases = _plan_graph(graph, out, result, repetitions=repetitions, case_seconds=case_seconds)
    if schedule is None:
        return
    layers = result['layers']
    layer = 'build'
    try:
        retained_plan = result['artifacts']['compiler_plan']
        project = deliver(schedule, out / 'project')
        result.update(artifacts=dict(project.artifacts, compiler_plan=retained_plan),
                      cpus=project.delivery['cpus'], executable=project.receipt,
                      elf=project.dynamic, applied_source_rewrites=len(project.delivery['source_edits']))
        layers.update(admission='REPLAYED', source_revision='CPP20_STAGED', build='BUILT')
        layer = 'validation'
        rows = _validate_delivery(project, cases, out, repetitions)
        matched = all(row['matches'] for row in rows)
        result.update(validation=rows, delivery='COMPARED' if matched else 'MISMATCH')
        layers[layer] = 'MATCHED' if matched else 'MISMATCH'
        if matched and project.delivery['source_edits']:
            result['selection'] = 'CANONICALIZED'
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        layers[layer] = 'BLOCKED'
        result.update(delivery='INCOMPLETE', delivery_error=str(error))


def plan_graph(directory, out, *, repetitions, case_seconds, shapes=None, trip_count=None, threads=None,
               region=None, memory_budget_bytes=None):
    """Schedule a retained graph without repeating frontend capture."""
    from scar.analysis.replay_transaction import replay_transaction
    from .contracts import GraphBundle

    with replay_transaction():
        graph = GraphBundle.open(directory)
        out = out.resolve()
        if graph.request.schema == 'scar.native-entry':
            if any(value is not None for value in (shapes, trip_count, threads, region, memory_budget_bytes)):
                raise ValueError('region inputs cannot specialize a native entry')
            out.mkdir(parents=True, exist_ok=False)
            result = _graph_report(graph)
            schedule, _ = _plan_graph(graph, out, result, repetitions=repetitions, case_seconds=case_seconds)
        else:
            from .middle.plan import publish_plan
            from .middle.python import schedule_python_graph

            constraints = dict(shapes=shapes, trip_count=trip_count, threads=threads,
                               memory_budget_bytes=memory_budget_bytes)
            if region is not None:
                constraints['entry_region'] = region
            schedule = schedule_python_graph(graph, out, **constraints)
            retained = publish_plan(out / 'compiler-plan.json', schedule.document())
            result = dict(schema='scar.compiler-planning-result', schema_version=2,
                          graph_bundle=asdict(graph.publication), planning_scope=schedule.scope,
                          artifacts=dict(schedule.artifacts, compiler_plan=retained), cost_decision=schedule.decision,
                          status='PLAN_TRIAL_READY' if schedule.decision['trial_only'] else 'PLAN_BASELINE_RETAINED')
    if schedule is not None and graph.request.schema == 'scar.native-entry':
        result['status'] = 'PLAN_SCHEDULED'
    _publish_result(out, result)
    return result


def plan_main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan', description='Plan a retained program graph.')
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--case-seconds', type=int, default=3600)
    parser.add_argument('--input-shapes', type=Path, help='JSON rank-two shape or map of inputs to shapes')
    parser.add_argument('--trip-count', type=int)
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--region', help='exact Python region identifier from the graph')
    parser.add_argument('--memory-budget-bytes', type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.case_seconds <= 3600:
        parser.error('--case-seconds must be between 1 and 3600')
    try:
        shapes = None if args.input_shapes is None else json.loads(args.input_shapes.read_bytes())
        result = plan_graph(args.graph, args.out, repetitions=args.repetitions, case_seconds=args.case_seconds,
                            shapes=shapes, trip_count=args.trip_count, threads=args.cpu_threads,
                            region=args.region, memory_budget_bytes=args.memory_budget_bytes)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] in ('PLAN_SCHEDULED', 'PLAN_TRIAL_READY', 'PLAN_BASELINE_RETAINED') else 1


def build_python_main(argv=None):
    """Project an admitted Python source build into the compiler's typed graph."""
    from scar.analysis.replay_transaction import replay_transaction
    from .frontend.python_graph import build_python_graph

    parser = argparse.ArgumentParser(prog='scar build')
    parser.add_argument('--python-build', required=True, type=Path)
    parser.add_argument('--source', type=Path, help='source file in the admitted Python build')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with replay_transaction():
            graph = build_python_graph(args.python_build, args.out, source=args.source)
        report = dict(schema='scar.compiler-graph-result', schema_version=2, status='GRAPH_STRUCTURE_VALIDATED',
                      graph_bundle=asdict(graph.publication), artifacts=graph.references(),
                      entries=list(graph.entry_regions))
        _publish_result(args.out, report)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(report, indent=2))
    return 0


def _validate_delivery(project, cases, out, repetitions):
    from .qualification import executable_receipt

    if project.delivery['cpus'] != [cases.cpu]:
        raise ValueError('delivery uses a different CPU from the cost observation')
    rows = cases.paired(project.executable, out / 'validation', project.observation, repetitions)
    if executable_receipt(project.executable) != project.receipt:
        raise ValueError('C++20 executable changed during validation')
    return rows


def compile_saved_plan(path, out, *, repetitions, case_seconds):
    """Replay a delivered entry plan in a separate process without replanning."""
    from scar.analysis.replay_transaction import replay_transaction
    from .backend.cpp20 import deliver
    from .contracts import ExecutionPlanRef
    from .qualification import NativeCases

    out = out.resolve()
    with replay_transaction() as transaction:
        reference, plan = ExecutionPlanRef.open(path)
        transaction.watch(reference['uri'], reference['sha256'], reference['size_bytes'])
        out.mkdir(parents=True, exist_ok=False)
        project = deliver(plan, out / 'project')
        if isinstance(project, dict):
            result = project.copy()
        else:
            cases = NativeCases(project.manifest, project.observation['cpu'], case_seconds)
            rows = _validate_delivery(project, cases, out, repetitions)
            result = dict(artifacts=project.artifacts, executable=project.receipt,
                cpus=project.delivery['cpus'], validation=rows,
                delivery='COMPARED' if all(row['matches'] for row in rows) else 'MISMATCH')
    result.update(schema='scar.compiler-replay', schema_version=2, graph_bundle=plan.graph,
                  project=str(out / 'project'), artifacts=dict(result['artifacts'], compiler_plan=reference))
    _publish_result(out, result)
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest', type=Path)
    inputs.add_argument('--plan', type=Path, help='replay a sealed compiler-plan.json for an entry or region')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--jobs', type=int, choices=range(1, 13), default=4)
    parser.add_argument('--repetitions', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--case-seconds', type=int, default=3600)
    parser.add_argument('--graph-only', action='store_true')
    args = parser.parse_args(arguments)
    if not 1 <= args.case_seconds <= 3600:
        parser.error('--case-seconds must be between 1 and 3600')
    if args.plan and args.graph_only:
        parser.error('--graph-only requires --manifest')
    try:
        if args.plan:
            result = compile_saved_plan(args.plan, args.out, repetitions=args.repetitions, case_seconds=args.case_seconds)
        else:
            result = compile_native(args.manifest, args.out, jobs=args.jobs, repetitions=args.repetitions,
                                    case_seconds=args.case_seconds, graph_only=args.graph_only)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2))
    return 1 if result.get('delivery') in ('INCOMPLETE', 'MISMATCH') else 0
