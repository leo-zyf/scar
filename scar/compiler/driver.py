"""SCAR compiler driver for locked native build targets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _graph_report(graph):
    facts = graph.report
    incomplete = any(facts[key] for key in ('frontiers', 'frontend_frontiers', 'excluded_control'))
    state = 'BLOCKED' if graph.program is None else 'PARTIAL' if incomplete else 'STRUCTURE_VALIDATED'
    counts = ('entry', 'regions', 'operations', 'values', 'locations', 'excluded_control',
              'call_frontiers', 'call_sites', 'call_links', 'reachable_regions')
    return dict(schema='scar.native-entry-result', schema_version=1,
                graph={key: facts[key] for key in counts}, artifacts=graph.references(),
                frontend_frontiers=len(facts['frontend_frontiers']), cfg_frontiers=len(facts['frontiers']),
                status=f'GRAPH_{state}', selection='KEEP_ORIGINAL',
                layers=dict(source='LOCKED', compiler='CAPTURED', graph=state, hardware='OBSERVED',
                            calls='PARTIAL' if facts['call_frontiers'] or graph.program is None else 'BOUND'))


def compile_native(manifest, out, *, jobs, repetitions, case_seconds, graph_only=False,
                   adapters=None):
    from .frontend.graph import BUILTIN_ADAPTERS, build_graph
    from .frontend.units import lock_units

    out = out.resolve()
    adapters = BUILTIN_ADAPTERS if adapters is None else tuple(adapters)
    languages = tuple(language for adapter in adapters for language in adapter.languages)
    inputs = lock_units(manifest, out, languages=languages)
    request = inputs.manifest
    graph = build_graph(request, out, jobs=jobs, request=inputs.request, adapters=adapters)
    result = _graph_report(graph)
    if graph_only and graph.program is None:
        result.update(delivery='INCOMPLETE', delivery_error='no validated entry region')
    if not graph_only:
        _compile_graph(graph, out, result, repetitions=repetitions, case_seconds=case_seconds)
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def _plan_graph(graph, out, result, *, repetitions, case_seconds):
    from scar.ir.execution_recipe import read_artifact
    from .backend.cpp20 import build_trial_variant, replay_native_revision
    from .contracts import verify_source_lock
    from .middle.plan import entry_cpus, publish_plan, rewrite_native, schedule_native
    from .qualification import NativeCases

    layers = result['layers']
    layer = 'rewrite'
    try:
        request = read_artifact(graph.request)
        verify_source_lock(request)
        revision = rewrite_native(graph, out)
        result.update(artifacts=revision.artifacts, graph_rewrites=len(revision.rules),
                      source_rewrites=len(revision.edits))
        layers[layer] = 'IR_VALIDATED'
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
            manifest, edits, types = replay_native_revision(revision.artifacts)
            variants = out / 'variants'

            def build(mode):
                return build_trial_variant(variants / mode, revision.artifacts, manifest,
                                           edits if mode == 'canonical' else [], types, mode=mode)

            cases.calibrate(observation, variants / 'measurements', repetitions, build)
        layers[layer] = 'OBSERVED'
        layer = 'schedule'
        schedule = schedule_native(revision, observation, request['cases'], out)
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


def plan_native_graph(directory, out, *, repetitions, case_seconds):
    """Schedule a retained native graph without repeating frontend capture."""
    from .contracts import GraphBundle

    graph = GraphBundle.read_native(directory)
    result = _graph_report(graph)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    schedule, _ = _plan_graph(graph, out, result, repetitions=repetitions, case_seconds=case_seconds)
    if schedule is not None:
        result['status'] = 'PLAN_SCHEDULED'
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def plan_main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan', description='Plan a retained native program graph.')
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--case-seconds', type=int, default=3600)
    args = parser.parse_args(argv)
    if not 1 <= args.case_seconds <= 3600:
        parser.error('--case-seconds must be between 1 and 3600')
    try:
        result = plan_native_graph(args.graph, args.out, repetitions=args.repetitions,
                                   case_seconds=args.case_seconds)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'PLAN_SCHEDULED' else 1


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
    from scar.artifacts import file_reference
    from scar.ir.execution_recipe import read_artifact
    from .backend.cpp20 import deliver
    from .contracts import ExecutionPlanRef
    from .qualification import NativeCases

    plan = ExecutionPlanRef.read(path)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    project = deliver(plan, out / 'project')
    manifest = read_artifact(project.delivery['request'])
    cases = NativeCases(manifest, project.observation['cpu'], case_seconds)
    rows = _validate_delivery(project, cases, out, repetitions)
    artifacts = dict(project.artifacts, compiler_plan=file_reference(path, 'scar.compiler-plan'))
    result = dict(schema='scar.compiler-replay', schema_version=1, artifacts=artifacts,
                  executable=project.receipt, cpus=project.delivery['cpus'], validation=rows,
                  delivery='COMPARED' if all(row['matches'] for row in rows) else 'MISMATCH')
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def compile_loops(document, out, *, loop_trip_count=None, loop_inputs=None, plan_only=False,
                  loop_cpu_threads=None, loop_cpu_model=False, loop_calibrations=(),
                  loop_no_op_profiles=None, loop_max_threads=None):
    from scar.analysis.build import BuildSession, _check
    from scar.ir.execution_recipe import read_artifact
    from .backend.cpp20 import deliver
    from .middle.python import schedule_loops

    if not (document.get('invariant_loops') or document.get('composite_invariant_loops')
            or loop_inputs or loop_no_op_profiles):
        return [], []
    program, hardware = document['program'], document['hardware']
    read_artifact(program)
    _check(hardware)
    with BuildSession(Path(program['uri']).parent) as session:
        direct, composite, work = schedule_loops(
            document, out, session.view, trip_count=loop_trip_count, input_specs=loop_inputs,
            cpu_threads=loop_cpu_threads, cpu_model=loop_cpu_model, calibrations=loop_calibrations,
            original_costs=loop_no_op_profiles, max_threads=loop_max_threads,
        )
        for row, loop in work:
            row.update(deliver(loop, catalog=session.view, plan_only=plan_only))
    read_artifact(program)
    _check(hardware)
    return direct, composite


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if any(argument == '--domains' or argument.startswith('--domains=') for argument in arguments):
        from .pipeline import compile_domains_main

        return compile_domains_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest', type=Path)
    inputs.add_argument('--plan', type=Path, help='replay a sealed compiler-plan.json')
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
