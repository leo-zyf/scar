"""Compile admitted source graphs and resource plans through the C++20 backend."""

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies

from scar.artifacts import file_reference
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PLANNING_SCHEMA
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA
from .frontend.graph import build_graph
from .middle.plan import (copy_source_edits, entry_cpus, plan_entry, plan_loop_region,
                          publish_plan, simplify_copies)


_LANGUAGES = {'c', 'c++', 'fortran'}
_TIMED_FIELD = re.compile(rb'(time:) +[0-9]+\.[0-9]+')


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')


def _receipt(path):
    reference = file_reference(path, 'binary')
    return dict(path=str(path), sha256=reference['sha256'], size_bytes=reference['size_bytes'])


def _manifest(path):
    manifest = json.loads(path.read_bytes())
    if (manifest.get('schema'), manifest.get('schema_version')) != ('scar.native-entry', 1):
        raise ValueError('expected scar.native-entry schema v1')
    if manifest.get('language') not in _LANGUAGES | {'mixed'}:
        raise ValueError('unsupported native target language')
    root = Path(manifest['source_root']).resolve(strict=True)
    manifest['source_root'] = str(root)
    locks = manifest['source_lock']['files']
    for name, expected in locks.items():
        path = (root / name).resolve(strict=True)
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f'source path outside of root: {name}')
        current = _receipt(path)
        if any(current[key] != expected[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source differs: {name}')
    units = manifest['units']
    if not units or len({unit['path'] for unit in units}) != len(units):
        raise ValueError('target requires uniquely named translation units')
    for unit in units:
        if unit.get('language', manifest['language']) not in _LANGUAGES:
            raise ValueError(f'translation unit has no language profile: {unit["path"]}')
        if unit['path'] not in locks:
            raise ValueError(f'translation unit missing source lock: {unit["path"]}')
    if baseline := manifest.get('baseline'):
        cwd = Path(baseline['cwd']).resolve(strict=True)
        argv = baseline['argv']
        if not argv:
            raise ValueError('baseline requires an executable')
        argv[0] = str((cwd / argv[0]).resolve() if '/' in argv[0]
                      else Path(shutil.which(argv[0]) or cwd / argv[0]).resolve())
        baseline['cwd'] = str(cwd)
    case_ids = [case['id'] for case in manifest.get('cases', ())]
    if (len(set(case_ids)) != len(case_ids)
            or any(not re.fullmatch(r'[\w.-]+', name) or name in ('.', '..') for name in case_ids)):
        raise ValueError('case IDs must be unique directory names')
    return manifest


def _run(command, cwd, stdin, cpus, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('native case exceeded its time budget')
    argv = ['taskset', '-c', ','.join(map(str, cpus)), *command]
    start = time.perf_counter_ns()
    result = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, timeout=remaining)
    return result, time.perf_counter_ns() - start


def _compare(case, original, generated):
    if original.returncode or generated.returncode or original.stderr != generated.stderr:
        return False
    if original.stdout == generated.stdout:
        return True
    if case.get('comparison') == 'timed-fields':
        return (_TIMED_FIELD.sub(rb'\1 <timing>', original.stdout)
                == _TIMED_FIELD.sub(rb'\1 <timing>', generated.stdout))
    if case.get('comparison') == 'scientific-text':
        # The contract compares exact formatted decimal values, accepting the
        # exponent-letter/padding representational differences of format runtimes.
        from decimal import Decimal
        notation = rb'[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)[DEde][+-]?[0-9]+'
        token = re.compile(notation)
        def normalize(raw):
            labels = token.sub(b'<decimal>', raw).split()
            values = [Decimal(value.upper().replace(b'D', b'E').decode()) for value in token.findall(raw)]
            return labels, values
        return normalize(original.stdout) == normalize(generated.stdout)
    return False


def _case_input(case):
    return b'' if not case.get('stdin') else Path(case['stdin']).read_bytes()


def _paired(manifest, executable, out, cpus, observation, repetitions, case_seconds):
    baseline = manifest['baseline']
    if _receipt(baseline['argv'][0]) != observation['baseline']:
        raise ValueError('baseline executable differs from calibration')
    rows = []
    for case in manifest['cases']:
        stdin = _case_input(case)
        input_digest = sha256(stdin).hexdigest()
        if input_digest != observation['case_inputs'][case['id']]:
            raise ValueError(f'case input differs from calibration: {case["id"]}')
        case_dir = out / case['id']
        case_dir.mkdir()
        deadline = time.monotonic() + case_seconds
        arguments = case.get('argv', [])
        arms = {'original': (baseline['argv'] + arguments, baseline['cwd']),
                'generated': ([str(executable), *arguments], str(case_dir))}
        matched, records = True, []
        for index in range(repetitions + 1):
            pair, record = {}, {}
            order = tuple(arms) if index % 2 == 0 else tuple(reversed(arms))
            for arm in order:
                command, working_dir = arms[arm]
                result, elapsed = _run(command, working_dir, stdin, cpus, deadline)
                pair[arm] = result
                (case_dir / f'{index}-{arm}.stdout').write_bytes(result.stdout)
                (case_dir / f'{index}-{arm}.stderr').write_bytes(result.stderr)
                record[arm] = dict(ns=elapsed, exit=result.returncode)
            records.append(dict(warmup=index == 0, order=order, arms=record,
                                matches=_compare(case, pair['original'], pair['generated'])))
            matched &= records[-1]['matches']
            if not matched:
                break
        samples = {arm: [record['arms'][arm]['ns'] for record in records if not record['warmup']]
                   for arm in arms}
        timing = {arm: dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values))
                  for arm, values in samples.items() if values}
        rows.append(dict(case=case['id'], matches=matched, stdin_sha256=input_digest,
                         comparison=case.get('comparison', 'exact'), pairs=records, timing=timing))
    return rows


def _observe_entry(manifest, artifacts, case_seconds):
    baseline = manifest['baseline']
    cases = manifest['cases']
    available = entry_cpus(artifacts['hardware'])
    if not available:
        raise ValueError('no online CPU is available in the inventoried execution scope')
    cpu = min(available)
    executable = _receipt(baseline['argv'][0])
    case = cases[0]
    fixture = _case_input(case)
    inputs = {row['id']: sha256(fixture if row is case else _case_input(row)).hexdigest()
              for row in cases}
    result, elapsed = _run(baseline['argv'] + case.get('argv', []), baseline['cwd'], fixture,
                           [cpu], time.monotonic() + case_seconds)
    return dict(schema='scar.native-entry-cost', schema_version=1, request=artifacts['request'],
                scope='ONE_BASELINE_LAUNCH', cpu=cpu, elapsed_ns=elapsed, exit=result.returncode,
                baseline=executable, case=case['id'], case_inputs=inputs)


def compile_native(manifest_path, out, *, jobs, repetitions, case_seconds, graph_only=False):
    manifest = _manifest(manifest_path)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    request_path = out / 'request.json'
    _write(request_path, manifest)
    request = file_reference(request_path, 'scar.native-entry')
    frontend = build_graph(manifest, out, jobs=jobs, request=request)
    graph = frontend.report
    artifacts = dict(request=request, graph_evidence=asdict(frontend.evidence),
                     calls=asdict(frontend.calls), hardware=asdict(frontend.hardware))
    if frontend.program is not None:
        artifacts['program'] = asdict(frontend.program)
    partial_graph = any(graph[key] for key in ('frontiers', 'frontend_frontiers', 'excluded_control'))
    graph_status = ('BLOCKED' if frontend.program is None else
                    'PARTIAL' if partial_graph else 'STRUCTURE_VALIDATED')
    layers = dict(source='LOCKED', compiler='CAPTURED',
                  graph=graph_status,
                  calls='PARTIAL' if graph['call_frontiers'] or not frontend.program else 'BOUND',
                  hardware='OBSERVED')
    report = dict(schema='scar.native-entry-result', schema_version=1, graph={
        key: graph[key] for key in ('entry', 'regions', 'operations', 'values', 'locations',
                                    'excluded_control', 'call_frontiers', 'call_sites', 'call_links',
                                    'reachable_regions')},
        frontend_frontiers=len(graph['frontend_frontiers']), cfg_frontiers=len(graph['frontiers']),
        status=f'GRAPH_{graph_status}', selection='KEEP_ORIGINAL', artifacts=artifacts, layers=layers)
    if graph_only:
        if frontend.program is None:
            report.update(delivery='INCOMPLETE', delivery_error='no validated entry region')
        _write(out / 'result.json', report)
        return report

    layer = 'rewrite'
    try:
        if frontend.program is None:
            raise ValueError('no validated entry region')
        simplified = out / 'simplified.json'
        rewrites = simplify_copies(frontend.program, graph['anchors'], simplified)
        source_edits = copy_source_edits(manifest, graph, rewrites)
        artifacts['simplified'] = file_reference(simplified, PROGRAM_SCHEMA)
        rewrite_path = out / 'rewrites.json'
        _write(rewrite_path, dict(schema='scar.native-rewrites', schema_version=1, rules=rewrites,
                                  source_edits=source_edits,
                                  program=artifacts['program'], result=artifacts['simplified']))
        artifacts['rewrites'] = file_reference(rewrite_path, 'scar.native-rewrites')
        report['graph_rewrites'] = len(rewrites)
        report['source_rewrites'] = len(source_edits)
        layers[layer] = 'IR_VALIDATED'
        layer = 'cost'
        if not manifest.get('baseline') or not manifest.get('cases'):
            raise ValueError('baseline executable and cases are required to qualify native delivery')
        observation = _observe_entry(manifest, artifacts, case_seconds)
        cost_path = out / 'cost-observation.json'
        _write(cost_path, observation)
        artifacts['cost'] = file_reference(cost_path, observation['schema'])
        if observation['exit'] != 0:
            raise ValueError(f'baseline exited with status {observation["exit"]}')
        layers[layer] = 'OBSERVED'
        layer = 'schedule'
        problem, schedule = plan_entry(graph['entry'], artifacts, observation, source_edits)
        for role, path, document, schema in (
            ('planning_problem', out / 'planning-problem.json', problem, PLANNING_SCHEMA),
            ('execution_plan', out / 'plan.json', schedule, PLAN_SCHEMA),
        ):
            _write(path, document)
            artifacts[role] = file_reference(path, schema)
        if not schedule.get('selected'):
            raise ValueError('no feasible native execution schedule')
        report['planning_scope'] = 'BASELINE_ONLY'
        layers[layer] = 'SCHEDULED'
        layer = 'admission'
        from .backend.cpp20 import admit_entry, build_cpp20_project
        from .frontend.clang import stage_source_revision, verify_mixed_linkage

        manifest, observation, delivery = admit_entry(artifacts, graph['entry'])
        report['cpus'] = delivery['cpus']
        layers[layer] = 'REPLAYED'
        layer = 'source_revision'
        build = stage_source_revision(
            manifest, out / 'project', delivery['source_edits'], captures=delivery['compiler_types'])
        layers[layer] = 'CPP20_STAGED'
        layer = 'build'
        executable, dynamic = build_cpp20_project(out / 'project', build)
        report['executable'] = _receipt(executable)
        report['elf'] = dynamic
        delivery['mixed_linkage'] = verify_mixed_linkage(manifest, artifacts['calls'], executable)
        fortran_emission = out / 'project' / 'fortran-emission.json'
        if fortran_emission.is_file():
            artifacts['fortran_emission'] = file_reference(fortran_emission, 'scar.fortran-emission')
            delivery['fortran_emission'] = artifacts['fortran_emission']
        delivery_path = out / 'project' / 'delivery.json'
        _write(delivery_path, dict(delivery, executable=report['executable']))
        artifacts['delivery'] = file_reference(delivery_path, delivery['schema'])
        layers[layer] = 'BUILT'
        layer = 'validation'
        comparison = out / 'validation'
        comparison.mkdir()
        rows = _paired(manifest, executable, comparison, delivery['cpus'], observation,
                       repetitions, case_seconds)
        report['validation'] = rows
        report['delivery'] = 'COMPARED' if all(case['matches'] for case in rows) else 'MISMATCH'
        if report['delivery'] == 'COMPARED' and source_edits:
            report['selection'] = 'CANONICALIZED'
        layers[layer] = 'MATCHED' if report['delivery'] == 'COMPARED' else 'MISMATCH'
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        layers[layer] = 'BLOCKED'
        report['delivery'] = 'INCOMPLETE'
        report['delivery_error'] = str(error)
    _write(out / 'result.json', report)
    return report


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
        report = compile_native(args.manifest, args.out, jobs=args.jobs, repetitions=args.repetitions,
                                case_seconds=args.case_seconds, graph_only=args.graph_only)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(report, indent=2))
    return 1 if report.get('delivery') in ('INCOMPLETE', 'MISMATCH') else 0


def _compile_loop_regions(document, out, *, loop_trip_count=None, loop_inputs=None, plan_only=False,
                         loop_cpu_threads=None, loop_cpu_model=False, loop_calibrations=(),
                         loop_no_op_profiles=None, loop_max_threads=None):
    """Use the same proof, schedule and lowering sequence for both loop profiles."""
    from scar.analysis.build import BuildSession, _check
    from scar.ir.execution_recipe import read_artifact
    from .backend.cpp20 import lower_loop
    from scar.compiler.loop_admission import read_invariant_loop_motion, validate_invariant_loop_links
    from scar.compiler.pipeline_loop_frontend import (
        read_composite_invariant_loop_motion, validate_composite_invariant_loop_links,
    )

    simple = document.get('invariant_loops', ())
    composed = document.get('composite_invariant_loops', ())
    supplied = {} if loop_inputs is None else loop_inputs
    original_costs = {} if loop_no_op_profiles is None else loop_no_op_profiles
    identifiers = {row['id'] for specs in (simple, composed) for row in specs}
    if set(supplied) - identifiers or set(original_costs) - identifiers:
        raise ValueError('loop inputs and original costs must name retained loop IDs')
    if not identifiers:
        return [], []
    if not loop_cpu_model and (loop_calibrations or original_costs or loop_max_threads is not None):
        raise ValueError('loop calibration and CPU model limits require --loop-cpu-model')
    if loop_cpu_model and loop_cpu_threads is not None:
        raise ValueError('automatic loop cost selection and an explicit CPU preview are separate modes')
    program, hardware = document['program'], document['hardware']
    read_artifact(program)
    _check(hardware)
    output = ([], [])
    profiles = (
        (simple, 'loops', read_invariant_loop_motion, validate_invariant_loop_links, False),
        (composed, 'composite-loops', read_composite_invariant_loop_motion,
         validate_composite_invariant_loop_links, True),
    )
    with BuildSession(Path(program['uri']).parent) as session:
        for (specs, subdir, reader, validate_links, composite), rows in zip(profiles, output):
            if not specs:
                continue
            directory = out / subdir
            directory.mkdir()
            for index, spec in enumerate(specs):
                folder = directory / f'loop-{index}'
                folder.mkdir()
                read_artifact(spec['proof'])
                proof, motion = reader(spec['proof']['uri'], catalog=session.view)
                if spec['id'] != proof.id or spec['hardware'] != hardware:
                    raise ValueError('loop proof or hardware differs from the original program domain')
                validate_links(proof, spec, document['loop_dependencies'])
                row = dict(id=proof.id, proof=spec['proof'], source_loop_authority='ORIGINAL_F4_REPLAYED',
                           native_directory=None, selected_for_use=False, whole_program_compiled=False,
                           source_motion=publish_plan(folder / 'source-motion.json', motion))
                if composite:
                    row.update(
                        loop_links=spec['loop_links'], root_callee_source=asdict(proof.root_callee_source),
                        body_identity=proof.body_identity, nested_global_guard_count=len(proof.nested_callee_guards),
                        source_loop_authority='ORIGINAL_F4_COMPOSITION_AND_GLOBAL_ROUTES_REPLAYED',
                    )
                rows.append(row)
                family = 'COMPOSITE_LOOP' if composite else 'LOOP'
                retained = f'ORIGINAL_{family}_RETAINED'
                legality = motion['movement_legality']
                if legality['status'] != 'CONDITIONAL':
                    row.update(status=retained, selected_kind='NO_OP', selection_scope='SOURCE_MOVEMENT_LEGALITY',
                               source_legality_status=legality['status'],
                               reason='original source window does not justify all required invariant placements')
                    if composite:
                        row['unresolved_roots'] = [item['expression'] for item in legality['roots']
                                                   if item['status'] != 'CONDITIONAL']
                        row['source_blockers'] = [
                            *legality['source_window_effects']['blockers'],
                            *(dict(blocker, expression=item['expression'], property=field)
                              for item in legality['roots'] for field in ('data_available', 'no_clobber', 'execution', 'effects')
                              for blocker in item[field]['blockers']),
                        ]
                    continue
                inputs = supplied.get(proof.id, {})
                shapes = inputs.get('input_shapes', spec.get('input_shapes'))
                count = inputs.get('trip_count', loop_trip_count)
                missing = [name for name, value in (('input_shapes', shapes), ('trip_count', count)) if value is None]
                if missing:
                    row.update(status=f'{family}_INPUT_CONSTRAINTS_REQUIRED', missing=missing)
                    continue
                budget = inputs.get('memory_budget_bytes', spec.get('memory_budget_bytes'))
                if budget is None:
                    raise ValueError('loop requires retained incremental memory budget')
                threads = inputs.get('cpu_threads', loop_cpu_threads)
                row.update(plan_loop_region(
                    proof, hardware, folder, shapes=shapes, count=count, budget=budget, threads=threads,
                    catalog=session.view if composite else None, model=loop_cpu_model,
                    max_threads=loop_max_threads, calibrations=loop_calibrations,
                    original_cost=original_costs.get(proof.id),
                ))
                if row['selected_kind'] != 'REGISTER_TEMPORAL_FUSION':
                    row['status'] = retained
                    continue
                lower_loop(proof, spec, row, folder, catalog=session.view, reader=reader,
                            composite=composite, parallel=loop_cpu_model or threads is not None,
                            plan_only=plan_only)
    read_artifact(program)
    _check(hardware)
    return output


@reuse_decoded_source_bodies()
def compile_build(domains, out, *, calibrations=(), no_op_profiles=None, plan_only=False,
                  loop_trip_count=None,loop_inputs=None,prepare_stateful_validation=False,stateful_cpu_threads=1,
                  loop_cpu_threads=None,loop_cpu_model=False,loop_calibrations=(),loop_no_op_profiles=None,
                  loop_max_threads=None,stateful_interval_plans=None,entry_costs=None,verify_entry=False,
                  enforce_cpu_capacity=False,cpu_entry_forward=False,cpu_launch_probes=(),cpu_team_probe=None,
                  cpu_native_phase_cost=None,cpu_python_import_cost=None,cpu_paired_evidence=None,
                  cpu_import_bindings=None,cpu_import_sources=None,cpu_initializer_observation=None,
                  cpu_native_array_route=None,cpu_native_array_route_cost=None,cpu_fixed_grid_cli_evidence=None,
                  cpu_fixed_grid_mapped_input_plan=None,cpu_fixed_grid_lifecycle_cost=None,
                  cpu_fixed_grid_input_route_review=None,cpu_fixed_grid_source_qualification=None,
                  cpu_fixed_grid_forward=None):
    from scar.analysis.build import _check
    from scar.analysis.stateful_dependencies import project_stateful_dependencies
    from scar.compiler.choose_stateful_cpu import choose_stateful_cpu
    from scar.compiler.selected_stateful_lowering import lower_stateful_choices
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import build_stateful_cpu_profiles
    domains=Path(domains).resolve();out=Path(out).resolve()
    if domains.is_dir():domains=domains/'result.json'
    if out.exists():raise ValueError('fresh compile-build output required')
    started=time.monotonic()
    parent=artifact_reference(domains,'scar.program-optimization-domains');document=read_artifact(parent)
    if enforce_cpu_capacity and (not verify_entry or plan_only or 'whole_entry' not in document):
        raise ValueError('CPU capacity enforcement requires a whole-entry process trial')
    if (cpu_entry_forward or cpu_launch_probes or cpu_team_probe is not None
            or cpu_native_phase_cost is not None or cpu_python_import_cost is not None
            or cpu_paired_evidence is not None or cpu_import_bindings is not None
            or cpu_import_sources is not None or cpu_initializer_observation is not None
            or cpu_native_array_route is not None or cpu_native_array_route_cost is not None
            or cpu_fixed_grid_cli_evidence is not None or cpu_fixed_grid_mapped_input_plan is not None
            or cpu_fixed_grid_lifecycle_cost is not None or cpu_fixed_grid_input_route_review is not None
            or cpu_fixed_grid_source_qualification is not None or cpu_fixed_grid_forward is not None
            ) and 'whole_entry' not in document:
        raise ValueError('CPU entry evidence requires a whole-entry plan')
    if cpu_native_array_route_cost is not None and cpu_native_array_route is None:
        raise ValueError('CPU native array route costs require the exact C++ route package')
    version=document.get('schema_version')
    if version not in (1,2):raise ValueError('supported program planning domain required')
    read_artifact(document['program']);_check(document['hardware'])
    source_admission=document.get('source_admission')
    if version==2 and source_admission is None:
        raise ValueError('program planning domain is missing original graph admission')
    if source_admission is not None and 'whole_entry' not in document:
        from scar.analysis.build_admission import verify_build_admission
        verify_build_admission(source_admission,program=document['program'],hardware=document['hardware'],
            source_filter=document.get('source_filter'))
    loop_ref=document.get('loop_dependencies')
    if loop_ref is not None:read_artifact(loop_ref)
    if 'whole_entry' in document:
        if source_admission is None:
            raise ValueError('whole-entry lowering requires a replayed original graph admission')
        if (calibrations or loop_calibrations or prepare_stateful_validation or loop_cpu_model
                or stateful_cpu_threads!=1 or any(value is not None for value in (no_op_profiles,
                loop_trip_count,loop_inputs,loop_cpu_threads,loop_no_op_profiles,loop_max_threads,stateful_interval_plans))):
            raise ValueError('whole-entry request already fixes the scope and declared implementation evidence')
        if plan_only and verify_entry:
            raise ValueError('process validation requires native emission')
        from scar.analysis.replay_transaction import replay_transaction
        from scar.compiler.whole_entry_pipeline import replay_whole_entry
        out.mkdir(parents=True)
        paired = (cpu_paired_evidence is not None or cpu_import_bindings is not None or cpu_import_sources is not None
                  or cpu_initializer_observation is not None or cpu_native_array_route is not None
                  or cpu_fixed_grid_cli_evidence is not None
                  or 'cpu_paired_observation' in document['whole_entry']['scope'])
        with replay_transaction() if paired else nullcontext():
            whole=replay_whole_entry(document,out,plan_only=plan_only,entry_costs=entry_costs,verify_entry=verify_entry,
                                 enforce_cpu_capacity=enforce_cpu_capacity,cpu_entry_forward=cpu_entry_forward,
                                 cpu_launch_probes=cpu_launch_probes,cpu_team_probe=cpu_team_probe,
                                 cpu_native_phase_cost=cpu_native_phase_cost,cpu_python_import_cost=cpu_python_import_cost,
                                 cpu_paired_evidence=cpu_paired_evidence,cpu_import_bindings=cpu_import_bindings,
                                 cpu_import_sources=cpu_import_sources,cpu_initializer_observation=cpu_initializer_observation,
                                 cpu_native_array_route=cpu_native_array_route,cpu_native_array_route_cost=cpu_native_array_route_cost,
                                 cpu_fixed_grid_cli_evidence=cpu_fixed_grid_cli_evidence,
                                 cpu_fixed_grid_mapped_input_plan=cpu_fixed_grid_mapped_input_plan,
                                 cpu_fixed_grid_lifecycle_cost=cpu_fixed_grid_lifecycle_cost,
                                 cpu_fixed_grid_input_route_review=cpu_fixed_grid_input_route_review,
                                 cpu_fixed_grid_source_qualification=cpu_fixed_grid_source_qualification,
                                 cpu_fixed_grid_forward=cpu_fixed_grid_forward)
        read_artifact(parent);read_artifact(document['program']);_check(document['hardware'])
        report=dict(schema='scar.program-compiler-pipeline',schema_version=1,parent=parent,
            program=document['program'],hardware=document['hardware'],source_admission=source_admission,
            whole_entry=whole['reference'],regions=[],invariant_loops=[],composite_invariant_loops=[],
            whole_entry_status=whole['selection']['status'],whole_entry_action=whole['delivery']['final_action'],
            scope='REPLAYED_WHOLE_ENTRY_OPTION_DOMAIN',selected_for_use=False,
            whole_program_compiled=whole['native_compiled'],target_program_executed=whole['target_executed'],
            three_stages_complete=False,seconds=time.monotonic()-started)
        if cpu_native_array_route is not None:
            report['cpu_native_array_route'] = whole['cpu_native_array_route']
        if cpu_native_array_route_cost is not None:
            report['cpu_native_array_route_cost'] = whole['cpu_native_array_route_cost']
        if cpu_fixed_grid_cli_evidence is not None:
            report['cpu_fixed_grid_cli_review'] = whole['cpu_fixed_grid_cli_review']
        if 'cpu_fixed_grid_mapped_input_plan' in whole:
            report['cpu_fixed_grid_mapped_input_plan'] = whole['cpu_fixed_grid_mapped_input_plan']
        if cpu_fixed_grid_lifecycle_cost is not None:
            report['cpu_fixed_grid_lifecycle_cost'] = whole['cpu_fixed_grid_lifecycle_cost']
        if 'cpu_fixed_grid_input_route_review' in whole:
            report['cpu_fixed_grid_input_route_review'] = whole['cpu_fixed_grid_input_route_review']
        if cpu_fixed_grid_source_qualification is not None:
            report['cpu_fixed_grid_source_qualification'] = whole['cpu_fixed_grid_source_qualification']
        if cpu_fixed_grid_forward is not None:
            report['cpu_fixed_grid_forward'] = whole['cpu_fixed_grid_forward']
        (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        return report
    if entry_costs is not None or verify_entry:
        raise ValueError('entry cost evidence and process validation require a whole-entry domain')
    no_op_profiles={} if no_op_profiles is None else no_op_profiles
    stateful_interval_plans={} if stateful_interval_plans is None else stateful_interval_plans
    if not isinstance(stateful_interval_plans,dict):raise ValueError('interval plans require a region-id to plan mapping')
    if any(not isinstance(value,(str,Path,dict)) for value in stateful_interval_plans.values()):
        raise ValueError('each region interval plan must be an original plan path or ArtifactRef')
    identifiers={row['region_id'] for row in document['regions']}
    if set(no_op_profiles)-identifiers:raise ValueError('NO_OP profiles must name retained original regions')
    if set(stateful_interval_plans)-identifiers:raise ValueError('interval plans must name retained original regions')
    if set(stateful_interval_plans)&set(no_op_profiles):
        raise ValueError('one region cannot consume both an interval plan and scalar NO_OP profiles')
    out.mkdir(parents=True);rows=[]
    for index,original in enumerate(document['regions']):
        directory=out/('region-'+str(index));directory.mkdir()
        row=dict(region_id=original['region_id'],selected_for_use=False,validation='NOT_EXECUTED')
        graph=original.get('program_graph')
        if graph is not None:
            from scar.compiler.ordered_planning_graph import admit_ordered_planning_graph
            row['program_graph']=graph
            row['program_graph_admission']=admit_ordered_planning_graph(graph,
                program=document['program'],region=original['region'],out=directory/'graph-admission.json')
        projection=project_stateful_dependencies(original['region'],out=directory/'dependencies')
        row['dependencies']=artifact_reference(directory/'dependencies/dependencies.json',projection['schema'])
        if original['status']!='HARDWARE_DOMAIN_BUILT':
            if original['region_id'] in stateful_interval_plans:
                raise ValueError('interval-plan region needs its retained original hardware domain')
            row.update(status=original['status'],reason='input constraints required before hardware planning')
            rows.append(row);continue
        domain=_domain(original['domain'])
        if domain['region']['uri']!=original['region'] or projection['region_id']!=original['region_id']:
            raise ValueError('planning domain differs from retained original region')
        target=domain['target_snapshot'];hardware=document['hardware']
        if (target['uri'],target['sha256'],target['size_bytes'])!=(hardware['path'],hardware['sha256'],hardware['bytes']):
            raise ValueError('planning domain differs from the program hardware snapshot')
        interval_value=stateful_interval_plans.get(original['region_id'])
        if interval_value is not None:
            from scar.compiler.selected_stateful_lowering import lower_stateful_bounds
            from scar.compiler.stateful_domain_equivalence import retain_stateful_domain_equivalence
            if isinstance(interval_value,(str,Path)):
                interval_path=Path(interval_value).resolve()
                if interval_path.is_dir():interval_path/='result.json'
                interval_ref=artifact_reference(interval_path,'scar.stateful-interval-choice-plan')
            elif isinstance(interval_value,dict):
                interval_ref=interval_value;interval_path=Path(interval_ref['uri'])
                if interval_ref.get('schema')!='scar.stateful-interval-choice-plan':
                    raise ValueError('retained interval-choice plan ArtifactRef required')
            else:raise ValueError('interval plans must contain original plan paths or ArtifactRefs')
            interval=read_artifact(interval_ref)
            main_domain=artifact_reference(original['domain'],'scar.stateful-cpu-domain')
            equivalence=retain_stateful_domain_equivalence(interval['original_domain'],main_domain,
                directory/'domain-equivalence.json')
            # This existing lowering owns complete source/fact/interval replay.
            # Do not run scalar calibration/choice or an unranked preview here.
            lowered=lower_stateful_bounds(interval_path,directory/'interval-lowering',plan_only=plan_only)
            if lowered['upstream_plan']!=interval_ref or lowered['domain']!=interval['original_domain']:
                raise ValueError('verified interval lowering differs from the retained source plan/domain')
            read_artifact(interval_ref);bounds=read_artifact(interval['source_bounds_artifact'])
            row.update(status=lowered['status'],planning_route='VERIFIED_INTERVAL_PLAN',
                domain=main_domain,interval_source_domain=interval['original_domain'],
                interval_plan=interval_ref,interval_choice=interval['interval_choice'],
                interval_lowering=artifact_reference(directory/'interval-lowering/lowering.json',lowered['schema']),
                domain_equivalence=equivalence,lowering=str(directory/'interval-lowering/lowering.json'),
                native_directory=lowered['native_directory'],candidate_count=len(domain['candidates']),
                complete_cost_profiles=bounds['counts']['previously_complete_options']-1,
                cost_domain_complete=False,interval_choice_complete=interval['status']=='FINITE_MODEL_OPTIMAL',
                interval_status=interval['status'],selected_candidate_id=interval['selected_candidate_id'],
                remaining_cost_frontier=interval['remaining_frontier'],
                partial_options_strictly_dominated=interval['prior_partial_options_strictly_dominated'],
                scalar_calibration_and_choice_consumed=False,
                caller_loop_proof='NOT_ESTABLISHED_BY_PER_CALL_TRANSFER',
                automatic_caller_replacement_authorized=False)
            rows.append(row);continue
        profiles=build_stateful_cpu_profiles(domain,calibrations,out=directory/'costs')
        choice=choose_stateful_cpu(directory/'planning',domain=original['domain'],
            cost_profiles=profiles['cost_profiles'],
            no_op_profile=no_op_profiles.get(original['region_id'],profiles['no_op_profile']))
        lowered=lower_stateful_choices(choice['choice'],directory/'lowering',plan_only=plan_only)
        gaps=Counter();rejected=Counter()
        for model in profiles['reports']:
            report=json.loads(Path(model['model']).read_bytes())
            for issue in report['unresolved']:
                gaps[issue['component']]+=1
                if issue.get('rejected_estimates'):rejected[issue['component']]+=1
        row.update(status=lowered['status'],choice=choice['choice'],lowering=str(directory/'lowering/lowering.json'),
            native_directory=lowered['native_directory'],candidate_count=choice['candidate_count'],
            complete_cost_profiles=len(profiles['cost_profiles']),cost_domain_complete=profiles['complete_domain'],
            unresolved_components=dict(gaps),rejected_components=dict(rejected),
            caller_loop_proof='NOT_ESTABLISHED_BY_PER_CALL_TRANSFER',
            automatic_caller_replacement_authorized=False)
        if prepare_stateful_validation and not plan_only and lowered['native_directory'] is None:
            from scar.compiler.stateful_candidate import prepare_stateful_candidate
            candidate=prepare_stateful_candidate(document['program']['uri'],domain,directory/'validation-candidate',
                cpu_threads=stateful_cpu_threads)
            row['validation_candidate']=artifact_reference(directory/'validation-candidate/candidate.json',candidate['schema'])
        rows.append(row)
    loops, composite_loops = _compile_loop_regions(
        document, out, loop_trip_count=loop_trip_count, loop_inputs=loop_inputs,
        plan_only=plan_only, loop_cpu_threads=loop_cpu_threads, loop_cpu_model=loop_cpu_model,
        loop_calibrations=loop_calibrations, loop_no_op_profiles=loop_no_op_profiles,
        loop_max_threads=loop_max_threads,
    )
    read_artifact(parent);read_artifact(document['program']);_check(document['hardware'])
    report=dict(schema='scar.program-compiler-pipeline',schema_version=1,parent=parent,
        program=document['program'],hardware=document['hardware'],loop_dependencies=loop_ref,
        regions=rows,invariant_loops=loops,source_rejections=document['rejections'],
        source_rejection_views=dict(
            invariant_loops=dict(artifact=parent,locator='#/invariant_loop_rejections',
                count=len(document.get('invariant_loop_rejections',[]))),
            composite_loops=dict(artifact=parent,locator='#/composite_rejections',
                count=len(document.get('composite_rejections',[])))),
        composite_invariant_loops=composite_loops,
        scope='ORIGINAL_F4_STATEFUL_AND_PURE_LOOP_REGIONS_WITH_SCOPED_CALLER_COST_AND_RUNTIME_CONTRACTS',
        source_graph_unrolled=False,target_variants_ranked_by_measurement=False,
        target_program_executed=False,whole_program_compiled=False,three_stages_complete=False,
        selected_for_use=False,seconds=time.monotonic()-started)
    if source_admission is not None:
        report['source_admission']=source_admission
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def compile_domains_main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domains',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--calibration',action='append',default=[])
    parser.add_argument('--no-op-profiles',help='JSON mapping region IDs to same-boundary original-call profiles')
    parser.add_argument('--stateful-interval-plans',help='JSON mapping exact region IDs to retained interval-choice plan paths or ArtifactRefs')
    loops=parser.add_mutually_exclusive_group()
    loops.add_argument('--loop-trip-count',type=int,help='declared common trip count for retained invariant loops')
    loops.add_argument('--loop-inputs',help='JSON mapping original loop IDs to input_shapes/trip_count/budget constraints')
    parser.add_argument('--prepare-stateful-validation',action='store_true',help='prepare one original-F4-proven stateful candidate without changing cost selection')
    parser.add_argument('--stateful-cpu-threads',type=int,default=1,help='explicit unranked worker count for stateful validation preview')
    cpu=parser.add_mutually_exclusive_group()
    cpu.add_argument('--loop-cpu-threads',type=int,help='explicit hardware-checked OpenMP team preview; not a claim of best worker count')
    cpu.add_argument('--loop-cpu-model',action='store_true',help='select only from a complete sourced forward CPU component cost domain')
    parser.add_argument('--loop-calibration',action='append',default=[],help='independent fixed-family temporal resource observations or projected component facts')
    parser.add_argument('--loop-no-op-profiles',help='JSON mapping original loop IDs to retained same-boundary original-call cost profiles')
    parser.add_argument('--loop-max-threads',type=int,help='explicit maximum team size for the cost model; all representative masks within that limit remain visible')
    parser.add_argument('--entry-costs',help='original whole-entry domain reference with late-bound model cost facts')
    parser.add_argument('--verify-entry',action='store_true',
        help='validate the selected entry or the eligible source-derived device and CPU nominations')
    parser.add_argument('--enforce-cpu-capacity',action='store_true',
        help='require a proved native CPU address-space contract for the entry trial')
    parser.add_argument('--cpu-entry-forward',action='store_true',
        help='replay hardware-aware component costs of an independently nominated CPU entry')
    parser.add_argument('--cpu-launch-probe',
        help='attach an independent CPU process proxy observation to the forward report')
    parser.add_argument('--cpu-team-probe',
        help='replay a second independent component campaign in the exact retained CPU team model')
    parser.add_argument('--cpu-native-phase-cost',
        help='attach same-provider native phase observations as diagnostics in the CPU forward report')
    parser.add_argument('--cpu-python-import-cost',
        help='attach original CPython import observations to the exact entry scope')
    parser.add_argument('--cpu-paired-evidence',
        help='attach checked original/native CPU observations to an existing exact entry scope')
    parser.add_argument('--cpu-import-bindings',
        help='bind exact observed import selectors to the original F4 initializer graph')
    parser.add_argument('--cpu-import-sources',
        help='attach exact source initializers recovered from the original CPU import observations')
    parser.add_argument('--cpu-initializer-observation',
        help='bind observed module code entry and exit to their admitted F4 initializer bodies')
    parser.add_argument('--cpu-native-array-route',
        help='bind a scoped C++20 array task route to its exact CPU catalog option')
    parser.add_argument('--cpu-native-array-route-cost',
        help='attach a replayed forward timeline to the exact native array route')
    parser.add_argument('--cpu-fixed-grid-cli-evidence',
        help='review a source-compatible fixed-grid C++ entry in its original catalog scope')
    parser.add_argument('--cpu-fixed-grid-mapped-input-plan',
        help='attach the scoped native copy versus mapping decision to the exact fixed-grid CLI review')
    parser.add_argument('--cpu-fixed-grid-lifecycle-cost',
        help='replay fixed-grid lifecycle evidence for the exact source CLI review')
    parser.add_argument('--cpu-fixed-grid-input-route-review',
        help='replay selected native input graph, hardware observations and C++20 route package')
    parser.add_argument('--cpu-fixed-grid-source-qualification',
        help='replay same-scope startup, external publication and clean full-process timing evidence')
    parser.add_argument('--cpu-fixed-grid-forward',
        help='attach a replayed count-calibrated forward model and clean holdout review')
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args(argv)
    result=compile_build(args.domains,args.out,calibrations=args.calibration,
        no_op_profiles=None if args.no_op_profiles is None else json.loads(Path(args.no_op_profiles).read_bytes()),
        plan_only=args.plan_only,loop_trip_count=args.loop_trip_count,
        loop_inputs=None if args.loop_inputs is None else json.loads(Path(args.loop_inputs).read_bytes()),
        prepare_stateful_validation=args.prepare_stateful_validation,stateful_cpu_threads=args.stateful_cpu_threads,
        loop_cpu_threads=args.loop_cpu_threads,loop_cpu_model=args.loop_cpu_model,
        loop_calibrations=args.loop_calibration,
        loop_no_op_profiles=None if args.loop_no_op_profiles is None else json.loads(Path(args.loop_no_op_profiles).read_bytes()),
        loop_max_threads=args.loop_max_threads,
        stateful_interval_plans=None if args.stateful_interval_plans is None else json.loads(Path(args.stateful_interval_plans).read_bytes()),
        entry_costs=args.entry_costs,verify_entry=args.verify_entry,enforce_cpu_capacity=args.enforce_cpu_capacity,
        cpu_entry_forward=args.cpu_entry_forward,
        cpu_launch_probes=() if args.cpu_launch_probe is None else (args.cpu_launch_probe,),
        cpu_team_probe=args.cpu_team_probe,cpu_native_phase_cost=args.cpu_native_phase_cost,
        cpu_python_import_cost=args.cpu_python_import_cost,cpu_paired_evidence=args.cpu_paired_evidence,
        cpu_import_bindings=args.cpu_import_bindings,cpu_import_sources=args.cpu_import_sources,
        cpu_initializer_observation=args.cpu_initializer_observation,cpu_native_array_route=args.cpu_native_array_route,
        cpu_native_array_route_cost=args.cpu_native_array_route_cost,
        cpu_fixed_grid_cli_evidence=args.cpu_fixed_grid_cli_evidence,
        cpu_fixed_grid_mapped_input_plan=args.cpu_fixed_grid_mapped_input_plan,
        cpu_fixed_grid_lifecycle_cost=args.cpu_fixed_grid_lifecycle_cost,
        cpu_fixed_grid_input_route_review=args.cpu_fixed_grid_input_route_review,
        cpu_fixed_grid_source_qualification=args.cpu_fixed_grid_source_qualification,
        cpu_fixed_grid_forward=args.cpu_fixed_grid_forward)
    print(json.dumps(dict(regions=len(result['regions']),invariant_loops=len(result['invariant_loops']),
        composite_invariant_loops=len(result['composite_invariant_loops']),
        statuses=[r['status'] for r in [*result['regions'],*result['invariant_loops'],*result['composite_invariant_loops']]],
        whole_entry_status=result.get('whole_entry_status'),whole_entry_action=result.get('whole_entry_action'),
        result=str(Path(args.out)/'result.json'),seconds=result['seconds']),indent=2));return 0


if __name__ == '__main__':
    raise SystemExit(main())
