"""Model selection within an explicit CPU implementation family, without trials."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from scar.compiler.model import ExecutionPlan, document, plan_from_dict
from scar.compiler.validation import validate_plan
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode, loads
from .cost_model import predict_region_costs
from .implementation_solver import solve_implementation_choices, task_implementation_contract
from .resource_solver import _shape_instantiation, semantic_sha256, solve_resource_schedule
from .schedule import ResourceSchedule, Task, TaskKind, schedule_document, schedule_from_dict


def _bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def _receipt(path):
    path = Path(path).absolute(); raw = path.read_bytes()
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw),
                semantic_sha256=semantic_sha256(loads(raw)))


def _emit(path, value, write):
    raw = _bytes(value)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    elif path.read_bytes() != raw:
        raise ValueError('CPU choice derived artifact differs: ' + str(path))
    return _receipt(path)


def _read(receipt, kind):
    if receipt is None:
        return None
    if _receipt(receipt['path']) != receipt:
        raise ValueError('CPU choice source receipt differs: ' + receipt['path'])
    value = loads(Path(receipt['path']).read_bytes())
    return plan_from_dict(value) if kind == 'plan' else schedule_from_dict(value) if kind == 'schedule' else value


def _source(value, kind, path):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return _receipt(value)
    value = document(value) if isinstance(value, ExecutionPlan) else schedule_document(value) if isinstance(value, ResourceSchedule) else value
    return _emit(path, value, True)


def _family(plan):
    value = document(plan)
    for field in ('schema_version', 'cpu_threads', 'resource_binding'):
        value.pop(field, None)
    value['cost'] = {k: v for k, v in value['cost'].items() if k not in
                     ('cpu_threads', 'available_cpu_affinity', 'thread_overhead', 'placement_selection', 'measurement')}
    value['proof'] = {k: v for k, v in value['proof'].items() if k not in
                      ('resource_allocation', 'runtime_guards', 'resource_binding_validation')}
    return value


def _common_profile(profile, threads):
    backend = profile['backend']; compiler = backend['compiler']
    options = compiler['options']
    if options.count('-fopenmp') != int(threads > 1):
        raise ValueError('only the worker-count-dependent single -fopenmp change is permitted')
    return dict(target_snapshot_sha256=profile['target']['target_sha256'],
                backend={k: backend[k] for k in ('name', 'execution_profile', 'operator_abi_version',
                                                'guard_mode', 'runtime_implementation_sha256')},
                compiler=dict(identity=compiler['identity'], argv=compiler['argv'],
                              options=[v for v in options if v != '-fopenmp']),
                layout=profile['applicability']['layout'], regime=profile['applicability']['working_set']['regime'],
                tick_seconds=profile['model']['tick_seconds']['value'])


def _semantic_contract(base, input_shapes):
    declared, shape, issues = _shape_instantiation(base, input_shapes)
    if issues or shape is None:
        raise ValueError('complete input/output shape instantiation required')
    return dict(base_family_sha256=semantic_sha256(_family(base)), input_shapes={k: list(v) for k, v in declared.items()},
                output_shape=list(shape), output_bytes=8 * shape[0] * shape[1],
                memory_policy='UNBOUND', invocation_scope='$CALL_INSTANCE')


def _evaluate(domain, out, write):
    from scar.codegen.schedule import admit_schedule

    fields = {'schema', 'schema_version', 'base_plan', 'variants', 'input_shapes',
              'semantic_contract', 'search_budget', 'completeness', 'scope'}
    if (set(domain) != fields or domain['schema'] != 'scar.cpu-choice-domain'
            or type(domain['schema_version']) is not int or domain['schema_version'] != 1
            or domain['completeness'] != 'ALL_EXPLICIT_INPUT_VARIANTS_RETAINED'
            or domain['scope'] != 'FINITE_DECLARED_CPP_CPU_FAMILY_NOT_ALL_PHYSICAL_IMPLEMENTATIONS'):
        raise ValueError('complete supported explicit CPU domain required')
    base = _read(domain['base_plan'], 'plan'); validate_plan(base)
    contract = domain['semantic_contract']; contract_hash = semantic_sha256(contract)
    if (base.resource_binding is not None
            or semantic_sha256(_semantic_contract(base, domain['input_shapes'])) != semantic_sha256(contract)):
        raise ValueError('unbound original CPU family contract differs')
    semantic = ResourceSchedule(base.region.id, (), (), (Task('cpu-call', TaskKind.COMPUTE,
        attrs=dict(semantic_composite='SYNCHRONOUS_ALLOCATION_COMPUTE_EXPORT',
                   semantic_contract_sha256=contract_hash, original_parameters=list(base.region.parameters),
                   original_output=base.region.output)),), loop_axes=base.schedule,
        proof=dict(external_assumptions=base.proof.get('legality', {}).get('external_assumptions', [])))
    task_hash = semantic_sha256(task_implementation_contract(semantic, 'cpu-call'))
    rows, options, capacities, context = [], [], {}, None
    for index, item in enumerate(domain['variants']):
        folder = out / ('variant-' + str(index))
        plan = _read(item['plan'], 'plan'); schedule = _read(item['schedule'], 'schedule')
        profile = _read(item['profile'], 'profile'); native_context = _read(item.get('backend_context'), 'context')
        problems = []
        try:
            validate_plan(plan)
            if plan.resource_binding is None or plan.resource_binding.memory_policy != 'UNBOUND':
                raise ValueError('recorded CPU binding with host UNBOUND memory required')
            if semantic_sha256(_family(plan)) != contract['base_family_sha256']:
                raise ValueError('region/source/numerics/IO/effects/shape or memory contract changed')
            guards = set(plan.proof.get('runtime_guards', ()))
            extra = {'CPU_IDS_WITHIN_CURRENT_PROCESS_AFFINITY', 'ORIGINAL_TARGET_SNAPSHOT_MATCHES',
                     'TARGET_MEMORY_ACCESS_CONDITIONS_HOLD'}
            original = set(base.proof.get('runtime_guards', ()))
            if not original <= guards or guards - original - extra:
                raise ValueError('original semantic guards changed')
            admit_schedule(plan, schedule)
        except (ValueError, TypeError, KeyError) as error:
            problems.append(str(error))
        prediction = None
        try:
            prediction = predict_region_costs(plan, schedule, profile or {}, input_shapes=domain['input_shapes'],
                                               backend_context=native_context)
            if prediction['prediction_available'] and not problems:
                current = _common_profile(profile, plan.cpu_threads)
                if context is None:
                    context = current
                elif semantic_sha256(current) != semantic_sha256(context):
                    raise ValueError('compiler/backend/layout/regime/tick context differs across CPU family')
            if prediction['resource_unresolved']:
                problems.append('resource/shape/budget constraints remain unresolved')
        except (ValueError, TypeError, KeyError) as error:
            problems.append(str(error))
        if item.get('frontier_reason'):
            problems.append(item['frontier_reason'])
        fixed = None
        if prediction and prediction['prediction_available'] and not problems:
            fixed = solve_resource_schedule(schedule, **prediction['solver_model'], search_budget=domain['search_budget'])
            if fixed['status'] != 'OPTIMAL_WITHIN_MODEL':
                problems.append('original three-task schedule has no complete feasible model proof')
        duration = (sum(v['duration_ticks'] for v in prediction['task_costs'].values())
                    if prediction and prediction['prediction_available'] else None)
        binding = plan.resource_binding
        demands = {binding.executor_id: plan.cpu_threads} if binding else {}
        if binding:
            capacities.setdefault(binding.executor_id, set()).update(binding.cpu_ids)
        conditions = ['EXACT_ORIGINAL_REGION_IO_EFFECTS_AND_HOST_UNBOUND_BUDGET',
                      'SINGLE_SYNCHRONOUS_CALL_COMPOSITE_COST_EQUALS_THREE_TASK_SUM',
                      'CALIBRATED_PROFILE_CONTEXT_AND_RUNTIME_GUARDS_REMAIN_OBLIGATIONS']
        evidence_doc = dict(schema='scar.cpu-choice-variant-evidence', schema_version=1, id=item['id'],
                            prediction=prediction, family_issues=problems,
                            composite_cost=dict(duration_ticks=duration, task_ticks=None if not prediction else
                                                {k: v['duration_ticks'] for k, v in prediction['task_costs'].items()}),
                            legality=dict(contract_sha256=task_hash, resource_demands=demands, conditions=conditions)
                            if not problems and fixed is not None else None)
        evidence = _emit(folder / 'evidence.json', evidence_doc, write)
        artifact = ProviderArtifact(evidence['path'], evidence['sha256'], evidence['bytes'], evidence_doc['schema'])
        fact = lambda pointer: [encode(ProviderFactReference(artifact, pointer))]
        options.append(dict(id=item['id'], duration_ticks=duration, resource_demands=demands,
                            contract_sha256=task_hash, cost_evidence=fact('#/composite_cost/duration_ticks'),
                            legality_conditions=conditions, legality_evidence=fact('#/legality')))
        prediction_ref = encode(ProviderFactReference(artifact, '#/prediction'))
        rows.append(dict(index=index, **item, prediction=prediction_ref, bound_solver_result=fixed,
                         family_issues=problems, evidence=evidence,
                         uncertainty=None if not profile else profile.get('validation')))
    solver = solve_implementation_choices(semantic, task_options={'cpu-call': options},
        resource_capacities={key: len(cpus) for key, cpus in capacities.items()}, search_budget=domain['search_budget'])
    chosen_id = (solver.get('chosen_option_ids') or {}).get('cpu-call')
    chosen = next((row for row in rows if row['id'] == chosen_id), None)
    return dict(schema='scar.cpu-implementation-choice', schema_version=1,
                choice_id='cpu-choice:' + semantic_sha256(domain), output_directory=str(out),
                base_plan=domain['base_plan'], domain=_receipt(out / 'domain.json'),
                semantic_contract=contract, semantic_contract_sha256=contract_hash,
                variants=rows, solver=solver,
                selected_variant=None if chosen is None else dict(id=chosen['id'], index=chosen['index']),
                selected_bound_solver_result=None if chosen is None else chosen['bound_solver_result'],
                complete_domain=(solver['status'] == 'OPTIMAL_WITHIN_FINITE_MODEL' and not solver['frontier']
                                 and not solver['unresolved'] and not solver['search_budget_exhausted']),
                selection_scope='CPP_IMPLEMENTATION_FOR_LOWERING', chosen_for_deployment=False,
                original_python_no_op=dict(available=True, cost_modeled=False, outperformed=False),
                composite_scope='ONE_SYNCHRONOUS_CALL_ONLY_NOT_AN_EXACT_MULTI_REGION_CONCURRENCY_MODEL',
                comparison_model='AUXILIARY_IMPLEMENTATION_OBJECTIVE_NOT_A_REPLACEMENT_PROGRAM_DATA_GRAPH',
                typed_program_io_authority='EACH_VARIANT_ORIGINAL_RESOURCE_SCHEDULE_AND_MATERIALIZATIONS',
                composite_resource_policy='WHOLE_CALL_CPU_RESERVATION_ONLY_FOR_SINGLE_CALL_COMPARISON',
                physical_optimality_proven=False, deployment_authorized=False,
                target_variants_compiled=False, target_variants_executed=False)


def plan_cpu_choices(base_plan, variants, out, *, input_shapes, search_budget=100000):
    """Persist and solve all explicitly declared CPU variants, including frontiers."""
    out = Path(out).absolute()
    if out.exists():
        raise ValueError('fresh CPU choice output required')
    if type(search_budget) is not int or search_budget < 1:
        raise ValueError('positive search budget required')
    out.mkdir(parents=True)
    base_receipt = _source(base_plan, 'plan', out / 'sources/base-plan.json')
    base = _read(base_receipt, 'plan'); validate_plan(base)
    if base.resource_binding is not None:
        raise ValueError('unbound semantic base plan required')
    contract = _semantic_contract(base, input_shapes)
    inputs, seen = [], set()
    for index, variant in enumerate(variants):
        if set(variant) - {'id', 'plan', 'schedule', 'profile', 'backend_context', 'frontier_reason'}:
            raise ValueError('unsupported CPU variant constraints')
        identifier = variant['id']
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('nonempty unique variant IDs required')
        seen.add(identifier)
        row = dict(id=identifier)
        for kind in ('plan', 'schedule', 'profile', 'backend_context'):
            row[kind] = _source(variant.get(kind), kind, out / f'sources/variant-{index}/{kind}.json')
        row['frontier_reason'] = variant.get('frontier_reason')
        if row['plan'] is None or row['schedule'] is None:
            raise ValueError('each CPU variant needs its retained plan and schedule')
        inputs.append(row)
    domain = dict(schema='scar.cpu-choice-domain', schema_version=1, base_plan=base_receipt,
                  variants=inputs, input_shapes=contract['input_shapes'], semantic_contract=contract,
                  search_budget=search_budget, completeness='ALL_EXPLICIT_INPUT_VARIANTS_RETAINED',
                  scope='FINITE_DECLARED_CPP_CPU_FAMILY_NOT_ALL_PHYSICAL_IMPLEMENTATIONS')
    _emit(out / 'domain.json', domain, True)
    report = _evaluate(domain, out, True)
    _emit(out / 'choice.json', report, True)
    return report


def verify_cpu_choice_report(value):
    """Read-only replay of sources, complete option domain and all model results."""
    report = loads(Path(value).read_bytes()) if isinstance(value, (str, Path)) else deepcopy(value)
    if (report.get('schema'), report.get('schema_version')) != ('scar.cpu-implementation-choice', 1):
        raise ValueError('versioned CPU implementation choice report required')
    out = Path(report['output_directory'])
    if isinstance(value, (str, Path)) and Path(value).absolute() != out / 'choice.json':
        raise ValueError('CPU choice report location differs')
    domain = _read(report['domain'], 'domain')
    _emit(out / 'domain.json', domain, False)
    expected = _evaluate(domain, out, False)
    if semantic_sha256(expected) != semantic_sha256(report):
        raise ValueError('CPU choice report differs from complete sourced replay')
    return report


def read_cpu_choice_prediction(row):
    """Read one shared prediction through its exact retained evidence receipt."""
    reference = decode(ProviderFactReference, row['prediction'])
    artifact = reference.artifact
    if reference.locator != '#/prediction' or artifact.schema != 'scar.cpu-choice-variant-evidence':
        raise ValueError('CPU prediction must reference its original variant evidence')
    receipt = _receipt(artifact.uri)
    if (receipt != row['evidence'] or receipt['sha256'] != artifact.sha256
            or receipt['bytes'] != artifact.size_bytes):
        raise ValueError('CPU prediction evidence receipt differs')
    evidence = loads(Path(artifact.uri).read_bytes())
    if evidence.get('schema') != artifact.schema or evidence.get('id') != row['id']:
        raise ValueError('CPU prediction evidence refers to a different variant')
    return deepcopy(evidence['prediction'])


__all__ = ['plan_cpu_choices', 'verify_cpu_choice_report', 'read_cpu_choice_prediction']
