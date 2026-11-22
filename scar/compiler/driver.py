"""SCAR compiler driver for locked native build targets."""

import argparse
import json
from pathlib import Path
import subprocess

from .backend.cpp20 import build_trial_variant, emit_native, replay_native_revision
from .frontend.graph import build_graph
from .frontend.units import lock_units
from .middle.plan import entry_cpus, rewrite_native, schedule_native
from .qualification import NativeCases, executable_receipt


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


def compile_native(manifest, out, *, jobs, repetitions, case_seconds, graph_only=False):
    out = out.resolve()
    inputs = lock_units(manifest, out)
    request = inputs.manifest
    graph = build_graph(request, out, jobs=jobs, request=inputs.request)
    result = _graph_report(graph)
    if graph_only and graph.program is None:
        result.update(delivery='INCOMPLETE', delivery_error='no validated entry region')
    if not graph_only:
        _compile_graph(inputs, graph, out, result, repetitions=repetitions, case_seconds=case_seconds)
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def _compile_graph(inputs, graph, out, result, *, repetitions, case_seconds):
    request = inputs.manifest
    layers = result['layers']
    layer = 'rewrite'
    try:
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
        result.update(artifacts=schedule.artifacts, cost_decision=schedule.decision,
                      planning_scope=schedule.scope)
        layers[layer] = 'SCHEDULED'
        layer = 'build'
        project = emit_native(schedule, out / 'project')
        result.update(artifacts=project.artifacts, cpus=project.delivery['cpus'], executable=project.receipt,
                      elf=project.dynamic, applied_source_rewrites=len(project.delivery['source_edits']))
        layers.update(admission='REPLAYED', source_revision='CPP20_STAGED', build='BUILT')
        layer = 'validation'
        if project.delivery['cpus'] != [cases.cpu]:
            raise ValueError('delivery uses a different CPU from the cost observation')
        rows = cases.paired(project.executable, out / 'validation', project.observation, repetitions)
        if executable_receipt(project.executable) != project.receipt:
            raise ValueError('C++20 executable changed during validation')
        matched = all(row['matches'] for row in rows)
        result.update(validation=rows, delivery='COMPARED' if matched else 'MISMATCH')
        layers[layer] = 'MATCHED' if matched else 'MISMATCH'
        if matched and project.delivery['source_edits']:
            result['selection'] = 'CANONICALIZED'
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        layers[layer] = 'BLOCKED'
        result.update(delivery='INCOMPLETE', delivery_error=str(error))


def compile_loops(document, out, *, loop_trip_count=None, loop_inputs=None, plan_only=False,
                  loop_cpu_threads=None, loop_cpu_model=False, loop_calibrations=(),
                  loop_no_op_profiles=None, loop_max_threads=None):
    from scar.analysis.build import BuildSession, _check
    from scar.ir.execution_recipe import read_artifact
    from .backend.cpp20 import lower_loop
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
            row.update(lower_loop(loop, catalog=session.view, plan_only=plan_only))
    read_artifact(program)
    _check(hardware)
    return direct, composite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--jobs', type=int, choices=range(1, 13), default=4)
    parser.add_argument('--repetitions', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--case-seconds', type=int, default=3600)
    parser.add_argument('--graph-only', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.case_seconds <= 3600:
        parser.error('--case-seconds must be between 1 and 3600')
    try:
        result = compile_native(args.manifest, args.out, jobs=args.jobs, repetitions=args.repetitions,
                                case_seconds=args.case_seconds, graph_only=args.graph_only)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2))
    return 1 if result.get('delivery') in ('INCOMPLETE', 'MISMATCH') else 0
