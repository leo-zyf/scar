"""Verify a native project and compare its outputs with explicit reference files."""

import argparse
import json
from math import prod
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_elf import _dependencies
from scar.codegen.native_provider_bundle import _content
from scar.compiler.tensor_planning_inputs import replay_native_comparison, replay_operation_costs, replay_recurrence_placement, scheduled_tensor_body
from scar.ir.implementation_plan import problem_from_dict
from scar.ir.program import codec
from scar.ir.v2.schemas import canonical_json


def _replay_recurrence_rhs_storage(recipe, geometry, storage, placement):
    from scar.optimization.aten_native_team import data_dependencies
    from scar.optimization.aten_live_values import ready_lifetime_bound
    from scar.optimization.aten_live_values import _bytes, tensor_footprint

    inputs, results = geometry['inputs'], geometry['operations']
    cpus = placement['cpus']
    if (placement['mode'] != 'READY_DAG' or len(cpus) != 3 or len(set(cpus)) != len(cpus)
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or [row['shape'] for row in inputs] != [row['shape'] for row in recipe['inputs']]
            or [row['id'] for row in results] != [row['id'] for row in recipe['operations']]
            or any(row['storage_bytes'] != 4 * prod(row['shape']) for row in (*inputs, *results))):
        raise ValueError('recurrence RHS geometry or placement differs from the admitted dense Tensor ABI')
    input_bytes, result_bytes = ([row['storage_bytes'] for row in rows] for rows in (inputs, results))
    serial = tensor_footprint(recipe, input_bytes=input_bytes, output_bytes=result_bytes, candidates=())
    returned = set(recipe['outputs'])
    exports = [index for index, row in enumerate(results) if row['id'] in returned]
    ready = ready_lifetime_bound(data_dependencies(recipe), placement['lanes'], exports, result_bytes,
                                 serial['persistent_input_bytes'], operation_order=placement['operation_order'])
    budget = storage['budget_bytes']
    expected = {key: serial[key] for key in ('scope', 'persistent_input_bytes', 'serial_peak_bytes', 'excluded_bytes')}
    expected.update(accounting='RELEASE_ON_LAST_COMPLETED_CONSUMER', budget_bytes=budget,
                    full_retention_bytes=sum(input_bytes) + sum(result_bytes), lifetime_proof=ready,
                    peak_tensor_storage_bytes=ready['peak_tensor_storage_bytes'])
    if not _bytes(budget) or not 0 < expected['peak_tensor_storage_bytes'] <= budget or expected != storage:
        raise ValueError('recurrence RHS storage differs from replayed serial and ready Tensor lifetimes')
    return expected


def _replay_recurrence_lifetimes(manifest, member, graph, execution, recipe):
    from scar.compiler.tensor_planning_inputs import recurrence_graph
    from scar.optimization.aten_recurrence import plan_recurrence

    source = json.loads(member('plan/inputs.json').read_bytes())
    step, geometry = source['lowered_step'], source['source_geometry']
    port, step_size = execution['state_input_index'], execution['step_size']
    if (source['rhs'] != manifest['source_references'] or source['solver'] != manifest['method_provider']
            or not any(row['id'] == recipe['outputs'][0] and row['shape'] == step['inputs'][0]['shape']
                       for row in geometry['operations'])):
        raise ValueError('recurrence source and RHS result differ from the admitted step')
    ready = source['rhs_candidate_placement']
    storage = _replay_recurrence_rhs_storage(recipe, geometry, source['rhs_candidate_storage'], ready)
    plans = {}
    alternatives = manifest['alternatives']
    budget = manifest['tensor_budget_bytes']
    if set(alternatives) not in ({'SERIAL'}, {'READY_DAG', 'SERIAL'}) or type(budget) is not int or not 0 < budget < 2**63:
        raise ValueError('recurrence lifetime plan needs its original candidate domain and Tensor budget')
    for mode, row in alternatives.items():
        peak = storage['serial_peak_bytes'] if mode == 'SERIAL' else storage['peak_tensor_storage_bytes']
        plan = plan_recurrence(step, recipe, dict(storage, peak_tensor_storage_bytes=peak),
                               state_input_index=port, step_size=step_size, optimize_lifetime=True)
        needed = plan['peak_tensor_storage_bytes']
        if (row['peak_tensor_storage_bytes'] != needed or row['within_budget'] is not (needed <= budget)
                or row['source_order_peak_tensor_storage_bytes'] != plan['step_lifetime_order']['original_peak_bytes']):
            raise ValueError('recurrence alternative differs from its replayed lifetime and budget admission')
        placement = dict(mode='SERIAL', cpus=ready['cpus'][:1], lanes=None, operation_order=None)
        plan['storage']['budget_bytes'] = budget
        plan.update(placement=ready if mode == 'READY_DAG' else placement, timing_basis='MEASURED_CPU_BODY_PER_STEP')
        plans[mode] = plan
    selected = plans[execution['placement']['mode']]
    if canonical_json(selected) != canonical_json({key: execution.get(key) for key in selected}):
        raise ValueError('recurrence execution differs from the replayed step lifetime plan')
    original = recurrence_graph(step, recipe, geometry, selected['control'], selected['rhs_bindings'], step_size=step_size)
    if codec.dumps(graph) != codec.dumps(original):
        raise ValueError('recurrence program graph differs from the original lowered step')
    return plans, source['rhs']


def _replay_recurrence_body(problem, plan, lowered, *, source):
    named = plan['operations']
    operations = [row['id'] for row in named]
    storage = plan['storage']
    sizes = {row['id']: (8 if row['dtype'] == 'FLOAT64' else 4) * prod(row['shape'])
             for row in (*plan['inputs'], *named)}
    consumed = {edge.value for edge in problem.data_edges}
    memory, = [resource for resource in problem.resources if resource.kind == 'MEMORY']
    reserved = sum(storage[key] for key in ('raw_file_backing_bytes', 'previous_step_state_bytes',
                                         'fixed_step_size_bytes'))
    order = 'PROVEN_TENSOR_LIFETIME_ORDER' if plan['step_lifetime_order']['status'] == 'IMPROVED' else 'SOURCE_ORDER'
    if (problem.source_refs != (source,) or [task.id for task in problem.operations] != [*operations, '$return']
            or problem.assumptions != ('SYNCHRONOUS_CPU_BODY', order, 'PROVIDER_WORKSPACE_NOT_MEASURED')
            or any(task.predecessors != tuple(operations[max(0, index - 1):index])
                   for index, task in enumerate(problem.operations))
            or memory.capacity != storage['budget_bytes'] or problem.reserved_memory != {memory.id: reserved}
            or any(edge.nbytes != sizes[edge.value] for edge in problem.data_edges)
            or (lowered['operations'], lowered['release_after']) != (named, plan['release_after'])):
        raise ValueError('recurrence physical schedule differs from the verified step lifetime order or storage')
    for task, row in zip(problem.operations[:-1], named, strict=True):
        mode, = task.options
        extra = storage['rhs_non_input_peak_bytes'] - sizes[task.id] if row['op'] == 'RHS' else 0
        extra += 0 if task.id in consumed else sizes[task.id]
        lowering = 'NATIVE_SHARED_RHS' if row['op'] == 'RHS' else 'NATIVE_ATEN_CALL'
        origin = dict(source, operation=f'program_operation:step/operation/{task.id}')
        if (mode.lowering != lowering or mode.workspace_bytes != ({memory.id: extra} if extra else {})
                or task.source_refs != (origin,) or mode.proof_refs != (origin,)):
            raise ValueError('recurrence operation provenance or workspace differs from its step lifetime proof')


def _recurrence_rhs_archive(member, references, base):
    documents = []
    for name, role in (('rhs-problem.json', 'planning_problem'), ('rhs-schedule.json', 'execution_plan')):
        raw = member(base + name)
        if _content(raw) != {key: references[role][key] for key in ('sha256', 'size_bytes')}:
            raise ValueError('packaged recurrence RHS differs from its admitted ready schedule')
        documents.append(json.loads(raw.read_bytes()))
    return documents


def _replay_recurrence_selection(manifest, member, execution, observation, schedule, recipe, *, lifetimes=None):
    def read(relative):
        return json.loads(member(relative).read_bytes())

    alternatives = manifest['alternatives']
    admitted = {key for key, item in alternatives.items() if item['within_budget']}
    mode = execution['placement']['mode']
    if (mode not in admitted or alternatives[mode]['body_estimate_ns'] != schedule['makespan_ns']
            or alternatives[mode]['peak_tensor_storage_bytes'] != execution['peak_tensor_storage_bytes']
            or any(item['within_budget'] != (item['peak_tensor_storage_bytes'] <= manifest['tensor_budget_bytes'])
                   for item in alternatives.values())):
        raise ValueError('recurrence selection disagrees with the admitted storage or body cost')
    expected_mode = mode
    sampled_comparison = False
    if admitted == {'SERIAL', 'READY_DAG'}:
        alternate_mode, = admitted - {mode}
        archive = 'validation/alternative/'
        other_obs = read(archive + 'calibration.json')
        if ((other_obs['inputs'], other_obs['repeats'], other_obs['outputs']) !=
                (observation['inputs'], observation['repeats'], observation['outputs'])
                or other_obs['placement']['mode'] != alternate_mode):
            raise ValueError('alternative calibration has a different input or execution scope')
        other_document = read(archive + 'problem.json')
        other_problem = problem_from_dict(other_document)
        other_schedule = read(archive + 'schedule.json')
        alternate = execution if lifetimes is None else lifetimes[0][alternate_mode]
        replay_operation_costs(other_problem, alternate, other_obs,
            dict(path='validation/calibration.json', **_content(member(archive + 'calibration.json'))))
        other_lowered = scheduled_tensor_body(other_document, other_schedule, alternate)
        if lifetimes is not None:
            placement = alternate['placement']
            if other_obs['placement'] != placement or other_problem.target.sha256 != execution['target']['sha256']:
                raise ValueError('alternative calibration differs from its replayed recurrence placement or target')
            _replay_recurrence_body(other_problem, alternate, other_lowered, source=execution['program_graph'])
            rhs = (_recurrence_rhs_archive(member, lifetimes[1], archive)
                   if alternate_mode == 'READY_DAG' else (None, None))
            replay_recurrence_placement(other_problem, placement, recipe, *rhs)
        if alternatives[alternate_mode]['body_estimate_ns'] != other_schedule['makespan_ns']:
            raise ValueError('alternative estimate differs from its physical schedule')
        if alternatives['READY_DAG']['body_estimate_ns'] < alternatives['SERIAL']['body_estimate_ns']:
            comparison = read('validation/comparison.json')
            expected_mode = replay_native_comparison(comparison, observation['outputs'])
            sampled_comparison = True
        else:
            expected_mode = 'SERIAL'
    if expected_mode != mode or sampled_comparison != manifest['full_entry_performance_measured']:
        raise ValueError('published recurrence does not implement the replayed native selection')


def _read_profile(member, files, base):
    graph_file = base + '/program.json'
    graph = codec.loads(member(graph_file).read_text())
    execution = json.loads(member(base + '/execution.json').read_text())
    document = json.loads(member(base + '/problem.json').read_text())
    problem = problem_from_dict(document)
    reference = execution['program_graph']
    if reference != dict(path=graph_file, **files[graph_file]):
        raise ValueError('execution plan does not reference its original program graph')
    known = {operation.id.wire for operation in graph.operations}
    if any(ref.get('operation') not in known for row in problem.operations if row.id != '$return'
           for ref in row.source_refs):
        raise ValueError('physical operation lacks a source program graph node')
    if problem.source_refs != (reference,) or execution['target']['sha256'] != problem.target.sha256:
        raise ValueError('shared physical problem differs from its graph or target binding')
    return graph, execution, document, problem


def _require_generated(member, generated):
    if any(member(name).read_text() != source for name, source in generated.items()):
        raise ValueError('generated C++ differs from the verified native schedule lowering')


def _replay_exported_choice(member, files, base, execution, recipe, sizes, observation):
    from scar.compiler.aten_export_placement import replay_export_cpu

    ready = base + '/ready/'
    proposal = json.loads(member(ready + 'proposal.json').read_bytes())
    candidate = proposal['placement']
    if candidate['cpu_ids'][0:1] != observation['cpu_ids']:
        raise ValueError('inference ready candidate and serial baseline have different caller CPUs')
    sampled = proposal['evidence'].get('inputs')
    if sampled is not None:
        user_receipts = iter(observation['inputs'])
        expected = []
        for port in execution['inputs']:
            receipt = port if port['kind'] == 'WEIGHT' else next(user_receipts)
            expected.append(dict(id=port['id'], sha256=receipt['sha256'], size_bytes=receipt['size_bytes']))
        if sampled != expected:
            raise ValueError('inference ready and serial costs bind different file contents')
    physical_path, schedule_path = ready + 'problem.json', ready + 'schedule.json'
    has_candidate_plan = physical_path in files
    if has_candidate_plan != (schedule_path in files):
        raise ValueError('inference ready candidate is missing a physical certificate')
    candidate_problem = json.loads(member(physical_path).read_bytes()) if has_candidate_plan else None
    candidate_schedule = json.loads(member(schedule_path).read_bytes()) if has_candidate_plan else None
    replay_export_cpu(recipe, proposal['evidence'], origin=execution['program_graph'],
        tensor_budget_bytes=execution['storage']['budget_bytes'],
        result_bytes=[sizes[row['id']] for row in recipe['operations']],
        problem=candidate_problem, schedule=candidate_schedule, placement=candidate,
        evidence_path=ready + 'proposal.json')
    placement = execution.get('placement')
    comparison_path = ready + 'entry-comparison.json'
    choice = 'SERIAL'
    if candidate['mode'] == 'READY_DAG':
        comparison = json.loads(member(comparison_path).read_bytes())
        choice = replay_native_comparison(comparison, observation['outputs'])
    elif comparison_path in files:
        raise ValueError('serial inference has a comparison without an admitted ready candidate')
    if choice == 'READY_DAG':
        if (placement != candidate or execution['cpu_ids'] != candidate['cpu_ids']
                or comparison['tested_files']['READY_DAG'] != {name: files[name]
                    for name in ('bin/aten-inference', 'launch.sh')}
                or any(files[base + '/' + name] != files[ready + name] for name in ('problem.json', 'schedule.json'))):
            raise ValueError('native inference does not execute the verified ready placement')
    elif placement is not None or execution['cpu_ids'] != observation['cpu_ids']:
        raise ValueError('native inference differs from the replayed serial choice')
    return placement


def _replay_exported(member, manifest, base, label, graph, execution, document, problem):
    from scar.codegen.aten_export_source import render_exported_entry
    from scar.compiler.tensor_planning_inputs import straight_line_graph, tensor_type
    from scar.ir.program.model import DataValue
    from scar.optimization.aten_export_rewrite import eliminate_dead_export_ops
    from scar.optimization.aten_live_values import tensor_footprint

    files = manifest['files']
    if execution['schema'] != 'scar.native-tensor-entry-plan' or execution['source'] != manifest['source']:
        raise ValueError('inference profile is missing its qualified source binding')
    recipe = json.loads(member(base + '/recipe.json').read_bytes())
    original_path, rewrite_path = base + '/source-recipe.json', base + '/rewrite.json'
    original = recipe
    if manifest.get('simplified_normal_completion'):
        original = json.loads(member(original_path).read_bytes())
        reduced, evidence = eliminate_dead_export_ops(original)
        if reduced != recipe or evidence != json.loads(member(rewrite_path).read_bytes()):
            raise ValueError('inference structural reduction differs from its replayed recipe')
    elif original_path in files or rewrite_path in files:
        raise ValueError('inference profile has an undeclared structural reduction')
    definitions = {item.id.value: item.data_type for item in graph.values if isinstance(item, DataValue)}
    types = {row['id']: tensor_type(row['shape']) for row in original['inputs']}
    sizes = {row['id']: 4 * prod(row['shape']) for row in original['inputs']}
    shapes = {}
    for row in original['operations']:
        name = row['id']
        value_type = definitions.get(f'entry/operation/{name}/result', '')
        match = re.fullmatch(r'tensor\.FLOAT32\[((?:[0-9]+(?:,[0-9]+)*)?)\]', value_type)
        if match is None:
            raise ValueError('inference profile is missing a source Tensor type')
        extents = [int(size) for size in match[1].split(',')] if match[1] else []
        types[name] = tensor_type(extents)
        shapes[name] = extents
        sizes[name] = 4 * prod(extents)
    if codec.dumps(straight_line_graph(original, types)) != codec.dumps(graph):
        raise ValueError('inference recipe differs from its original typed program graph')
    storage = execution['storage']
    cpu = execution['cpu_ids'][0]
    schedule_path = base + '/schedule.json'
    calibration_path = 'validation/calibration.json' if label is None else base + '/calibration.json'
    observation = json.loads(member(calibration_path).read_bytes()) if schedule_path in files else None
    placement = None
    if manifest.get('cpu_ready_planned'):
        if not observation:
            raise ValueError('inference placement requires its measured serial baseline')
        placement = _replay_exported_choice(member, files, base, execution, recipe, sizes, observation)
    elif execution.get('placement') is not None:
        raise ValueError('inference has an undeclared ready placement')
    if placement is None and execution['cpu_ids'] != [cpu]:
        raise ValueError('serial inference requires exactly one CPU')

    serial_schedule_path = schedule_path
    if placement is not None:
        serial_problem_path = base + '/ready/serial-problem.json'
        serial_schedule_path = base + '/ready/serial-schedule.json'
        document = json.loads(member(serial_problem_path).read_bytes())
        problem = problem_from_dict(document)
        if (problem.source_refs != (execution['program_graph'],)
                or problem.target.sha256 != execution['target']['sha256']):
            raise ValueError('inference serial proof does not bind the same source graph and target')
    memory, = [resource for resource in problem.resources if resource.kind == 'MEMORY']
    operations = [row['id'] for row in recipe['operations']]
    produced = set(operations)
    operands = {(arg['value'], row['id']) for row in recipe['operations']
                for arg in row['arguments'] if 'value' in arg}
    operands.update((name, '$return') for name in recipe['outputs'])
    expected_edges = {(name, consumer, sizes[name], name if name in produced else None,
                       None if name in produced else memory.id) for name, consumer in operands}
    edges = {(edge.value, edge.consumer, edge.nbytes, edge.producer, edge.external_location)
             for edge in problem.data_edges}
    if (tensor_footprint(recipe, input_bytes=[sizes[row['id']] for row in recipe['inputs']],
            output_bytes=[sizes[row['id']] for row in recipe['operations']], candidates=(),
            budget_bytes=storage['budget_bytes']) != storage
            or execution['output_shapes'] != [shapes[name] for name in recipe['outputs']]
            or [dict(id=row['id'], shape=row['shape']) for row in execution['inputs']] != recipe['inputs']
            or [task.id for task in problem.operations] != [*operations, '$return']
            or len(problem.data_edges) != len(expected_edges) or edges != expected_edges
            or memory.capacity != storage['budget_bytes']
            or problem.reserved_memory != {memory.id: storage['persistent_input_bytes']}
            or any(option.parameters['cpu_ids'] != [cpu] for task in problem.operations if task.id != '$return'
                   for option in task.options)):
        raise ValueError('inference file geometry, storage or CPU placement differs from the executed recipe')
    for row in execution['inputs']:
        if row['kind'] == 'WEIGHT':
            if row['path'] not in files or _content(member(row['path'])) != {key: row[key] for key in ('sha256', 'size_bytes')}:
                raise ValueError('inference weight does not match its packaged payload')
        elif row['kind'] != 'INPUT':
            raise ValueError('inference profile input kind is not part of the native file ABI')
    if schedule_path in files:
        if observation['cpu_ids'] != [cpu]:
            raise ValueError('inference calibration and emitted placement differ')
        replay_operation_costs(problem, recipe, observation, dict(path=calibration_path, **files[calibration_path]))
        schedule = json.loads(member(serial_schedule_path).read_bytes())
        lowered = scheduled_tensor_body(document, schedule, recipe)
        releases = lowered['release_after']
    else:
        lowered = recipe
        releases = storage['release_after']
    _require_generated(member, render_exported_entry(lowered, execution['bindings'], releases, cpu,
                       profile=label, placement=placement))
    return schedule_path in files


def _profile_receipt(graph, problem, replayed):
    return dict(graph_regions=len(graph.regions), graph_operations=len(graph.operations),
                physical_activities=len(problem.operations), executed_schedule_replayed=replayed)


def _replay_family(manifest, member):
    from scar.codegen.aten_export_source import render_profile_dispatch

    profiles = manifest['profiles']
    _require_generated(member, {'src/main.cpp': render_profile_dispatch(profiles)})
    receipts, interface, assignments, masks = {}, None, set(), {}
    for label in profiles:
        base = 'plan/profiles/' + label
        graph, execution, document, problem = _read_profile(member, manifest['files'], base)
        signature = ([row if row['kind'] == 'WEIGHT' else row['id']
                     for row in execution['inputs']], len(execution['output_shapes']))
        assignment = tuple(sorted(execution['symbolic_profile'].items()))
        if (interface is not None and signature != interface) or assignment in assignments:
            raise ValueError('native inference profiles must share one file ABI and have distinct shape bindings')
        interface = signature
        assignments.add(assignment)
        replayed = _replay_exported(member, manifest, base, label, graph, execution, document, problem)
        masks[label] = execution['cpu_ids']
        receipts[label] = dict(_profile_receipt(graph, problem, replayed),
                               symbolic_profile=execution['symbolic_profile'])
        if manifest.get('cpu_ready_planned'):
            receipts[label]['cpu_ids'] = execution['cpu_ids']
    if manifest.get('cpu_ready_planned'):
        from scar.compiler.compile_exported_aten import _render_launch

        _require_generated(member, {'launch.sh': _render_launch(execution['inputs'], len(execution['output_shapes']), masks)})
    all_measured = all(row['executed_schedule_replayed'] for row in receipts.values())
    if all_measured != manifest['body_calibrated']:
        raise ValueError('profile family calibration receipt differs from its manifest')
    totals = {key: sum(row[key] for row in receipts.values())
              for key in ('graph_regions', 'graph_operations', 'physical_activities')}
    return dict(totals, profiles=receipts, executed_schedule_replayed=all_measured)


def _replay_single(manifest, member):
    files = manifest['files']
    graph, execution, document, problem = _read_profile(member, files, 'plan')
    lifetime_planned = manifest.get('step_lifetime_planned', False)
    if (type(lifetime_planned) is not bool or lifetime_planned != ('step_lifetime_order' in execution)
            or lifetime_planned and ('plan/schedule.json' not in files or execution['schema'] != 'scar.aten-recurrence-plan')):
        raise ValueError('recurrence lifetime proof differs from its manifest')
    replayed = False
    if execution['schema'] == 'scar.native-tensor-entry-plan' and (
            'simplified_normal_completion' in manifest or 'plan/schedule.json' in files):
        replayed = _replay_exported(member, manifest, 'plan', None, graph, execution, document, problem)
        if manifest.get('cpu_ready_planned'):
            from scar.compiler.compile_exported_aten import _render_launch

            _require_generated(member, {'launch.sh': _render_launch(execution['inputs'], len(execution['output_shapes']),
                               {None: execution['cpu_ids']})})
    elif 'plan/schedule.json' in files:
        schedule = json.loads(member('plan/schedule.json').read_bytes())
        calibration_path = 'validation/calibration.json'
        observation = json.loads(member(calibration_path).read_bytes())
        replay_operation_costs(problem, execution, observation, dict(path=calibration_path, **files[calibration_path]))
        lowered = scheduled_tensor_body(document, schedule, execution)
        if execution['schema'] == 'scar.aten-recurrence-plan':
            from scar.codegen.aten_recurrence_source import render_recurrence

            placement = execution['placement']
            if manifest.get('source_entry_inferred'):
                boundary = json.loads(member('plan/source-entry.json').read_bytes())
                if manifest.get('source_entry_replayable'):
                    from scar.analysis.fixed_grid_source import derive_fixed_grid_entry

                    source = json.loads(member('plan/source-entry-binding.json').read_bytes())
                    potential = derive_fixed_grid_entry(source, boundary['entry'],
                        source_bytes=member('plan/source-entry.py').read_bytes())
                    if potential != {key: item for key, item in boundary.items() if key != 'discovery'}:
                        raise ValueError('recurrence source entry differs from its replayed source binding')
                if (boundary['method'] != manifest['method_provider']['method']
                        or boundary['step_size'] != execution['step_size']
                        or boundary['state_input_index'] != execution['state_input_index']):
                    raise ValueError('recurrence differs from its admitted source-entry binding')
            rhs_recipe = json.loads(member('plan/rhs-recipe.json').read_bytes())
            if placement != observation['placement'] or (placement['mode'], placement['cpus']) != (
                    manifest['rhs_mode'], manifest['cpus']):
                raise ValueError('recurrence calibration and emitted placement differ')
            lifetimes = (_replay_recurrence_lifetimes(manifest, member, graph, execution, rhs_recipe)
                         if lifetime_planned else None)
            if lifetimes is not None:
                _replay_recurrence_body(problem, execution, lowered, source=execution['program_graph'])
            rhs_problem = rhs_schedule = None
            if placement['mode'] == 'READY_DAG':
                if lifetimes is None:
                    rhs_problem, rhs_schedule = (json.loads(member('plan/' + name).read_bytes())
                                                 for name in ('rhs-problem.json', 'rhs-schedule.json'))
                else:
                    if execution['rhs_schedule'] != lifetimes[1]['execution_plan']:
                        raise ValueError('recurrence execution does not bind the admitted ready RHS')
                    rhs_problem, rhs_schedule = _recurrence_rhs_archive(member, lifetimes[1], 'plan/')
            elif lifetimes is not None and execution['rhs_schedule'] is not None:
                raise ValueError('serial recurrence has no ready RHS schedule')
            if rhs_schedule is not None and execution['rhs_schedule']['sha256'] != files['plan/rhs-schedule.json']['sha256']:
                raise ValueError('parallel RHS schedule differs from the archived selection')
            replay_recurrence_placement(problem, placement, rhs_recipe, rhs_problem, rhs_schedule)
            if 'source_entry_plan' in manifest:
                from scar.compiler.aten_source_entry_plan import verify_source_entry_emission

                verify_source_entry_emission(manifest, member)
            else:
                _replay_recurrence_selection(manifest, member, execution, observation, schedule, rhs_recipe,
                                             lifetimes=lifetimes)
            source_abi = None
            source_file_publication = None
            mapped_input_proof = None
            if 'source_entry_cli' in manifest:
                from scar.analysis.fixed_grid_entry_abi import derive_fixed_grid_file_entry
                from scar.compiler.aten_entry_delivery import ABI_FILE, BINDING_FILE, LAUNCHER, render_source_launcher, source_entry_description, verify_mapped_input_emission
                from scar.compiler.aten_source_entry_binding import verify_source_entry_binding
                from scar.compiler.compile_aten_recurrence import _render_recurrence_launch

                descriptor = manifest['source_entry_cli']
                parent = descriptor['parent_manifest_sha256']
                source_file_publication = descriptor.get('source_file_publication')
                mapped_input_proof = descriptor.get('mapped_input_proof')
                if (not manifest.get('source_entry_inferred') or not manifest.get('source_entry_replayable')
                        or not isinstance(parent, str) or re.fullmatch('[a-f0-9]{64}', parent) is None
                        or descriptor != source_entry_description(parent, source_file_publication=source_file_publication,
                                                                 mapped_input_proof=mapped_input_proof)):
                    raise ValueError('invalid source-entry delivery descriptor')
                if mapped_input_proof is not None:
                    verify_mapped_input_emission(mapped_input_proof, parent, source_file_publication)
                source_abi = derive_fixed_grid_file_entry(source, boundary['entry'],
                    source_bytes=member('plan/source-entry.py').read_bytes())
                if (source_abi != json.loads(member(ABI_FILE).read_bytes())
                        or verify_source_entry_binding(source_abi, execution, rhs_recipe, manifest=manifest)
                        != json.loads(member(BINDING_FILE).read_bytes())):
                    raise ValueError('source-entry ABI or physical binding differs from original replay')
                _require_generated(member, {LAUNCHER: render_source_launcher(placement['cpus']),
                                            'launch.sh': _render_recurrence_launch(placement['cpus'])})
            generated = render_recurrence(lowered, rhs_mode=placement['mode'], cpus=placement['cpus'],
                lanes=placement['lanes'], operation_order=placement['operation_order'], source_entry_abi=source_abi,
                source_file_publication=source_file_publication, map_source_inputs=mapped_input_proof is not None)
        else:
            raise ValueError('native schedule is missing a source replay implementation')
        _require_generated(member, generated)
        replayed = True
    return _profile_receipt(graph, problem, replayed)


def verify_native_project(directory, manifest_sha256):
    root = Path(directory).resolve(strict=True)
    path = root / 'artifact.json'
    if _content(path)['sha256'] != manifest_sha256:
        raise ValueError('native project manifest differs from its declared digest')
    manifest = json.loads(path.read_bytes())
    files = manifest['files']
    if 'source_entry_cli' in manifest and (manifest['schema'] != 'scar.aten-recurrence-delivery'
            or 'profiles' in manifest or 'cpu_linkage' not in manifest or 'plan/schedule.json' not in files):
        raise ValueError('standalone source CLI requires the CPU-only recurrence schedule')

    def member(relative):
        path = root / relative
        if relative not in files or path.resolve(strict=True) != path or not path.is_relative_to(root):
            raise ValueError(f'not a local native project member: {relative}')
        return path

    for name, expected in files.items():
        if _content(member(name)) != expected:
            raise ValueError(f'native project member differs: {name}')
    if manifest.get('cpu_ready_planned'):
        from scar.compiler.compile_exported_aten import CPU_READY_SCOPE

        if manifest['cpu_ready_scope'] != CPU_READY_SCOPE:
            raise ValueError('inference CPU ready plan uses a different observable execution contract')
    if manifest['schema'] == 'scar.native-exported-aten-cuda':
        from scar.compiler.compile_exported_aten_cuda import replay_cuda_project

        receipt = replay_cuda_project(root, manifest, member)
    else:
        receipt = _replay_family(manifest, member) if 'profiles' in manifest else _replay_single(manifest, member)
    executable = member(manifest['executable'])
    current = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(executable)], text=True, env={}))
    if current != manifest['dependencies'] or any(name.startswith(('libpython', 'libtorch_python')) for name in current):
        raise ValueError('runtime dependency resolution differs from the checked native project')
    if 'cpu_linkage' in manifest:
        from scar.runtime.aten_provider_identity import qualify_cpu_dependencies

        if manifest['schema'] not in ('scar.aten-recurrence-delivery', 'scar.native-exported-aten'):
            raise ValueError('the CPU-only dependency contract requires a native CPU project')
        cpu_linkage = qualify_cpu_dependencies(current)
        if canonical_json(manifest['cpu_linkage']) != canonical_json(cpu_linkage):
            raise ValueError('CPU-only linkage differs from the replayed ELF dependency closure')
        receipt['cpu_linkage'] = cpu_linkage
    locks = json.loads(member('plan/dependencies.json').read_text())
    content = _content
    if 'operator_stack' in manifest:
        from scar.compiler.operator_stack import stack_component_content

        content = stack_component_content
    for lock in locks['libraries'] if type(locks) is dict else locks:
        library = Path(lock['path'])
        if current.get(library.name) != str(library.resolve()) or content(library) != {
                key: lock[key] for key in ('sha256', 'size_bytes')}:
            raise ValueError(f'native operator library differs from the build lock: {library.name}')
    if 'source_entry_cli' in manifest:
        receipt['source_entry_cli_replayed'] = True
    return root, dict(receipt, native_dependencies_checked=True)


def validate_native_outputs(root, arguments, references, *, environment=None):
    expected = [_content(Path(path).resolve(strict=True)) for path in references]
    with TemporaryDirectory(prefix='scar-native-validation-') as temporary:
        outputs = [Path(temporary) / str(i) for i in range(len(references))]
        subprocess.run([str(root / 'launch.sh'), *arguments, *map(str, outputs)], env=environment or {},
                       check=True, capture_output=True)
        observed = [_content(path) for path in outputs]
    if observed != expected:
        raise ValueError('native program output differs from its reference bytes')
    return observed


def validate_native_project(directory, manifest_sha256, *, arguments=(), references=()):
    root, receipt = verify_native_project(directory, manifest_sha256)
    report = dict(schema='scar.native-project-validation', schema_version=1,
                  manifest_sha256=manifest_sha256, **receipt)
    if references:
        cuda_trial = receipt.get('offline_schedule_replayed', False)
        if cuda_trial and list(arguments[:1]) != ['--trial']:
            raise ValueError('CUDA reference comparison requires an explicit --trial invocation')
        environment = {}
        if cuda_trial:
            from scar.compiler.compile_exported_aten_cuda import CUDA_ENVIRONMENT

            environment = {name: os.environ[name] for name in CUDA_ENVIRONMENT if name in os.environ}
        report['bitwise_output_receipts'] = validate_native_outputs(root, arguments, references, environment=environment)
        if cuda_trial:
            report['device_result_verified'] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar validate-native-project', description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    receipt = validate_native_project(args.project, args.manifest_sha256, arguments=arguments,
                                      references=args.reference)
    print(json.dumps(receipt, sort_keys=True))
    return 0
