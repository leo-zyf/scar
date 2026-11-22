"""Source-replayed canonical component evidence for existing CPU activities.

Canonical samples become conditional component predictions, never observations
of target activities or guaranteed latency bounds. Pool terms stay grouped.
"""
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from scar.ir.activity_costs import (ActivityCost, ActivityCostTable, CostBasis,
    CostCoverage, CostInterval, IntervalKind, _prepare_cost_table_inputs, _validate_cost_table)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_document
from scar.optimization.resource_solver import semantic_sha256


_BOUNDARY_SCHEMA = 'scar.stateful-boundary-cut-calibration'
_SCOPE_BOUNDARY_SCHEMA = 'scar.stateful-scope-boundary-calibration'
_WORKSPACE_BOUNDARY_SCHEMA = 'scar.workspace-boundary-calibration'
_STAGE_DAG_SCHEMA = 'scar.stateful-stage-dag-calibration'
_CUTS = ('outer_enter', 'native_enter', 'metadata_ready', 'capture_ready',
         'binding_ready', 'worker_checked', 'restore_ready', 'outer_return')
_SCOPE_CUTS = (*_CUTS[:-1], 'helper_cleanup_ready', _CUTS[-1])


def _activities_by_task(schedule, error):
    result = {}
    for activity in schedule.temporal.activities:
        if activity.task in result:
            raise ValueError(error)
        result[activity.task] = activity.id
    return result


def _source_stage_tasks(schedule):
    result = defaultdict(list)
    for task in schedule.tasks:
        pointer = task.attrs.get('source_stage_ref', {}).get('pointer')
        if type(pointer) is str:
            result[pointer].append(task)
    return result


def _sample_statistics(values):
    return dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
                stdev_ns=statistics.stdev(values))


def _sampled_cost(subject, activities, stats, references, charge, conditions, missing_terms):
    return ActivityCost(subject, activities, CostBasis.DECLARED_COMPONENT_MODEL,
        estimate_ticks=math.ceil(stats['median_ns']),
        interval=CostInterval(math.floor(stats['min_ns']), math.ceil(stats['max_ns']), IntervalKind.EMPIRICAL_PREDICTION),
        component_refs=references, charge_keys=(charge,), coverage=CostCoverage.COMPONENT_ONLY,
        conditions=conditions, missing_terms=missing_terms)


def _remove_unknown_costs(costs, activities, error):
    activity_ids = set(activities)
    old = [row for row in costs if activity_ids.intersection(row.activities)]
    if (any(row.basis is not CostBasis.UNKNOWN or not set(row.activities) <= activity_ids for row in old)
            or {key for row in old for key in row.activities} != activity_ids):
        raise ValueError(error)
    replaced = {row.subject for row in old}
    costs[:] = [row for row in costs if row.subject not in replaced]
    return sorted(replaced)


def _strings(value, label, accepted_types):
    if (type(value) not in accepted_types or not value
            or any(type(item) is not str or not item for item in value)
            or len(set(value)) != len(value)):
        raise ValueError('unique nonempty ' + label + ' required')
    return tuple(value)


def _samples(reference, term, session):
    """Recompute statistics from retained primitive samples and original scopes."""
    from scar.optimization.scope_accounting import _pointer
    source_key = semantic_sha256(reference['artifact'])
    key = (semantic_sha256(reference), term['key'])
    if key in session['statistics']:
        return session['statistics'][key]
    if source_key not in session['sources']:
        session['sources'][source_key] = read_artifact(reference['artifact'])
    calibration = session['sources'][source_key]
    record = _pointer(calibration, reference['locator'])
    if (record.get('basis') != 'DIRECT_COMPONENT_TIMING'
            or record.get('component') != term['component']
            or record.get('parameters') != term['parameters']):
        raise ValueError('direct canonical component signature required; declared numbers are not samples')
    measurement = record.get('measurement', {})
    values = record.get('samples_ns')
    if (type(values) is not list or len(values) < 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or values != measurement.get('samples_ns')
            or len(values) != calibration.get('repetitions')
            or measurement.get('repetitions') != calibration.get('repetitions')
            or measurement.get('warmup') != calibration.get('warmup')
            or type(calibration.get('warmup')) is not int or calibration['warmup'] < 1):
        raise ValueError('retained canonical samples/warmup/repetitions differ')
    expected = _sample_statistics(values)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('retained component statistics differ from original samples')
    if (measurement.get('component') != term['component']
            or measurement.get('parameters') != term['parameters']
            or measurement.get('basis') != 'DIRECT_COMPONENT_TIMING'
            or measurement.get('median_ns') != expected['median_ns']
            or measurement.get('unit') != 'ns'
            or measurement.get('whole_method_boundary_measured') is not False
            or measurement.get('target_program_executed') is not False
            or measurement.get('applicable_to_component') is not True
            or calibration.get('target_workload_executed') is not False
            or calibration.get('target_variants_ranked') is not False):
        raise ValueError('independent canonical measurement scope/flags differ')
    probe = calibration.get('probe', {})
    if (probe.get('context') != calibration.get('probe_context')
            or probe.get('measurement_boundary') != 'INDEPENDENT_NATIVE_RESOURCE_PRIMITIVES'
            or any(probe.get(key) is not False for key in (
                'target_dag_compiled', 'target_program_executed', 'whole_call_cost_measured'))):
        raise ValueError('retained canonical probe implementation boundary differs')
    if source_key not in session['validated_probes']:
        directory = Path(__file__).resolve().parents[1]/'codegen'
        implementation = probe.get('implementation_files', {})
        required = {'cpu_resource_probe.py', 'cpu_resource_probe_v1.cpp.in', 'loop_v1.cpp.in',
                    'stateful_abi_v1.h', 'scar_abi_v1.h', 'stateful.py'}
        if set(implementation) != required:
            raise ValueError('canonical probe implementation provenance incomplete')
        for name, digest in implementation.items():
            if sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError('canonical probe implementation changed')
        encoded = (json.dumps(implementation, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        if sha256(encoded).hexdigest() != probe.get('implementation_sha256'):
            raise ValueError('canonical probe implementation manifest differs')
        for receipt in (probe['source'], probe['library'], *probe['headers'].values()):
            raw = Path(receipt['path']).read_bytes()
            if len(raw) != receipt['bytes'] or sha256(raw).hexdigest() != receipt['sha256']:
                raise ValueError('retained canonical native artifact changed')
        session['validated_probes'].add(source_key)
    conditions = measurement.get('conditions')
    if type(conditions) is not list or not conditions or any(type(v) is not str or not v for v in conditions):
        raise ValueError('canonical component applicability conditions required')
    session['statistics'][key] = expected, tuple(conditions)
    return session['statistics'][key]


def _unknown(subject, activities, reason):
    return ActivityCost(subject, tuple(activities), CostBasis.UNKNOWN, missing_terms=(reason,))


def project_unknown_activity_costs(schedule, resources, *, source_refs=(), reason):
    inputs = _prepare_cost_table_inputs(schedule, resources)
    rows = tuple(_unknown('cost:'+activity.id, (activity.id,), reason)
                 for activity in schedule.temporal.activities)
    table = ActivityCostTable(inputs.schedule_sha256, inputs.resources_sha256, resources.target,
                              rows, source_refs=source_refs)
    _validate_cost_table(table, inputs)
    return table, dict(protocol='UNSUPPORTED', reason=reason)


def project_stateful_activity_costs(schedule, resources, recipe, domain, candidate_id,
                                    calibrations, *, source_refs=(), boundary_calibrations=(),
                                    dag_calibrations=(), scope_schedule_ref=None):
    """Map exact stage occurrences and whole-pool signatures, preserving holes.

    The caller replays the recipe and G2 resource publication. Domain/source,
    binding, backend and canonical facts are independently replayed here.
    """
    from scar.optimization.stateful_cpu_model import evaluate_stateful_cpu_components
    from scar.optimization.stateful_cpu_model import _family
    original = semantic_sha256(recipe['task_graph'])
    actual = semantic_sha256(schedule_document(replace(schedule, temporal=None)))
    if actual != original:
        if 'implementation_scope_cleanup' in schedule.proof:
            from scar.analysis.implementation_scope import derive_stateful_scope_envelope as refine
        else:
            from scar.analysis.implementation_envelope import derive_stateful_envelope as refine
        from scar.ir.versioned_schedule import schedule_from_dict
        refined = refine(schedule_from_dict(recipe['task_graph']), recipe)
        if actual != semantic_sha256(schedule_document(refined)):
            raise ValueError('cost activities must retain the original recipe graph or its source-replayed implementation refinement')
    if encode(resources.target) != recipe['artifacts']['target_snapshot']:
        raise ValueError('cost activities and resources must retain the complete original recipe task graph and target')
    evaluation = evaluate_stateful_cpu_components(domain, candidate_id, calibrations)
    demand = evaluation['demand']
    base_plan = read_artifact(domain['base_stage_plan'])
    actual_plan = read_artifact(recipe['artifacts']['stage_plan'])
    actual_region = read_artifact(recipe['artifacts']['region'])
    candidate = next(row for row in domain['candidates'] if row['id'] == candidate_id)
    binding = actual_plan['resource_binding']
    selected_backend = domain['backend_capabilities']['serial' if candidate['worker_count'] == 1 else 'parallel']
    if (not candidate['legal_under_guards']
            or semantic_sha256(actual_region) != semantic_sha256(read_artifact(domain['region']))
            or semantic_sha256(_family(base_plan)) != semantic_sha256(_family(actual_plan))
            or candidate['worker_count'] != actual_plan['cpu_threads']
            or selected_backend != recipe['implementation']['backend_capability_record']
            or recipe['artifacts']['target_snapshot'] != domain['target_snapshot']):
        raise ValueError('component domain differs from exact recipe/source/stages/backend/target')
    keys = ('cpu_ids', 'executor_id', 'memory_domain_id', 'memory_policy', 'numa_node',
            'target_path', 'target_sha256', 'memory_budget_bytes')
    if any(candidate['binding'][key] != binding[key] for key in keys):
        raise ValueError('component candidate differs from original selected compute resource context')
    inputs = _prepare_cost_table_inputs(schedule, resources)
    activities_by_task = _activities_by_task(schedule, 'canonical stage adapter needs one original CPU lifecycle per task')
    stage_tasks = _source_stage_tasks(schedule)
    pool_tasks = {kind: [task for task in schedule.tasks if task.kind is kind] for kind in (K.ALLOCATE, K.RELEASE)}
    scans = [task for task in schedule.tasks if task.kind is K.IMPLEMENTATION_PHASE
             and task.attrs.get('phase_kind') == 'INPUT_FINITE_SCAN_SERIAL']
    grouped, unmapped, represented = [], [], set()
    session = dict(sources={}, validated_probes=set(), statistics={})
    mapping = []
    for term in evaluation['components']:
        source = term['source_ref']
        selected = []; release_order_delta = False
        if source is not None:
            selected = stage_tasks.get(source['pointer'], ())
            if source['artifact_role'] != 'base_stage_plan' or len(selected) != 1:
                raise ValueError('stage cost must match one original source-stage occurrence')
        elif term['component'] in ('ALLOCATE_FIXED_POOL', 'RELEASE_FIXED_POOL'):
            kind = K.ALLOCATE if term['component'] == 'ALLOCATE_FIXED_POOL' else K.RELEASE
            selected = pool_tasks[kind]
            slots = [name for task in selected for name in (task.allocates if kind is K.ALLOCATE else task.releases)]
            expected = [row['id'] for row in actual_plan['workspace_slots']]
            if (kind is K.RELEASE and recipe['control'].get('pool_release_order')
                    == 'CPP_STD_ARRAY_REVERSE_DESTRUCTION'):
                expected.reverse()
                release_order_delta = len(expected) > 1
            if (slots != expected or term['parameters'] != dict(capacities=[
                    row['capacity_bytes'] for row in actual_plan['workspace_slots']])):
                raise ValueError('collective pool cost must match the exact original slot group')
        elif term['component'] == 'INPUT_FINITE_SCAN_SERIAL':
            selected = [task for task in scans
                        if activities_by_task[task.id] not in represented
                        and term['parameters'] == dict(
                            rows=actual_plan['input_shapes'][task.attrs['input_id']][0],
                            cols=actual_plan['input_shapes'][task.attrs['input_id']][1])][:1]
        if not selected:
            unmapped.append(dict(term=term['id'], component=term['component'],
                original_source=term['source_ref'], measurement=term['measurement'],
                reason='IMPLEMENTATION_ENVELOPE_HAS_NO_EXACT_ACTIVITY_ENDPOINTS',
                cost_charged=False))
            continue
        identities = tuple(activities_by_task[task.id] for task in selected)
        if represented.intersection(identities):
            raise ValueError('one implementation activity cannot be charged by multiple component subjects')
        represented.update(identities)
        subject = 'cost:'+semantic_sha256(dict(schedule=inputs.schedule_sha256,
            activities=identities, occurrence=term['id']))[:32]
        occurrence = 'component-occurrence:'+semantic_sha256(dict(domain=domain['domain_id'],
            candidate=candidate_id, term=term['id']))
        measurement = term['measurement']
        if measurement is None:
            row = _unknown(subject, identities, term['id']+':'+term['component']+':NO_ADMITTED_COMPONENT_EVIDENCE')
        elif measurement['basis'] != 'DIRECT_COMPONENT_TIMING':
            row = _unknown(subject, identities, term['id']+':DECLARED_ESTIMATE_NOT_A_CANONICAL_SAMPLE')
        else:
            stats, conditions = _samples(measurement['evidence'], term, session)
            fact = measurement['evidence']
            reference = ProviderFactReference(ProviderArtifact(**fact['artifact']), fact['locator'])
            row = _sampled_cost(subject, identities, stats, (reference,), occurrence,
                conditions=(*conditions, 'CANONICAL_SAMPLE_RANGE_IS_NOT_A_GUARANTEED_TARGET_BOUND',
                    'ACTUAL_IMPLEMENTATION_DISPATCH_AND_DATA_REGIME_HAVE_NOT_BEEN_VALIDATED',
                    *(('CANONICAL_FORWARD_POOL_RELEASE_IS_COLLECTIVE_NOT_SOURCE_REVERSE_ORDER_EQUIVALENCE',)
                      if release_order_delta else ())),
                missing_terms=('UNVALIDATED_IMPLEMENTATION_DELTA_AND_UNMAPPED_OUTER_ENVELOPE',
                    *(('SOURCE_CPP_REVERSE_POOL_RELEASE_ORDER_AND_ALLOCATOR_STATE_DELTA_NOT_VALIDATED',)
                      if release_order_delta else ())))
        grouped.append(row)
        mapping.append(dict(subject=subject, activities=identities, task_ids=[task.id for task in selected],
            source_term=term['id'], component=term['component'], count=term['count'],
            charged_as_one_collective_pool=term['component'] in ('ALLOCATE_FIXED_POOL', 'RELEASE_FIXED_POOL')))
    for task in schedule.tasks:
        identity = activities_by_task[task.id]
        if identity not in represented:
            grouped.append(_unknown('cost:'+identity, (identity,), 'NO_MATCHING_EXACT_ACTIVITY_COST_SCOPE'))
    table = ActivityCostTable(inputs.schedule_sha256, inputs.resources_sha256, resources.target, tuple(grouped),
        source_refs=source_refs, unmapped_terms=tuple(unmapped))
    table, boundary_derivation, dag_derivation = _project_timing_calibrations(
        table, inputs, domain=domain, candidate_id=candidate_id, by_task=activities_by_task, stage_tasks=stage_tasks,
        boundary_calibrations=boundary_calibrations, dag_calibrations=dag_calibrations)
    accounting = None
    if 'implementation_scope_cleanup' in schedule.proof:
        from scar.optimization.scope_accounting import project_scope_accounting
        table, accounting = project_scope_accounting(table, schedule, resources, recipe, demand,
            mapping=mapping, boundary_derivation=boundary_derivation, dag_derivation=dag_derivation,
            scope_schedule_ref=scope_schedule_ref)
    summary = _validate_cost_table(table, inputs)
    derivation = dict(protocol='EXACT_STATEFUL_CANONICAL_COMPONENT_SIGNATURES', candidate_id=candidate_id,
        demand_context=demand['context'], resource_context=demand['resource_context'], mapping=mapping,
        original_terms=len(demand['terms']), original_component_signatures=len({row['key'] for row in demand['terms']}),
        unknown_terms=evaluation['unresolved'], rejected_calibrations=evaluation['rejected_calibrations'],
        inapplicable_calibrations=evaluation['inapplicable_calibrations'],
        full_call_cost_closed=False, sample_ranges_are_guaranteed_bounds=False,
        pool_cost_split_or_multiplied_by_slot_count=False, summary=summary)
    if boundary_derivation is not None:
        derivation['boundary_cuts'] = boundary_derivation
    if dag_derivation is not None:
        derivation['stage_dag'] = dag_derivation
    if accounting is not None:
        derivation['scope_accounting'] = accounting
    return table, derivation


def boundary_calibration_reference(value):
    if type(value) is dict:
        reference = value
    else:
        path = Path(value).resolve()
        schema = json.loads(path.read_bytes()).get('schema')
        if schema not in (_BOUNDARY_SCHEMA, _SCOPE_BOUNDARY_SCHEMA, _WORKSPACE_BOUNDARY_SCHEMA):
            raise ValueError('original legacy or scope boundary calibration schema required')
        reference = artifact_reference(path, schema)
    if reference.get('schema') not in (_BOUNDARY_SCHEMA, _SCOPE_BOUNDARY_SCHEMA, _WORKSPACE_BOUNDARY_SCHEMA):
        raise ValueError('original legacy or scope boundary calibration reference required')
    return reference


def project_boundary_activity_costs(table, schedule, resources, *, domain, candidate_id,
                                    calibrations=()):
    """Replace complete UNKNOWN subjects by independently replayed cut models."""
    return _project_boundary_activity_costs(table, _prepare_cost_table_inputs(schedule, resources),
        domain=domain, candidate_id=candidate_id, calibrations=calibrations)


def _project_boundary_activity_costs(table, inputs, *, domain, candidate_id, calibrations, by_task=None):
    from scar.compiler.calibrate_boundary_cuts import read_boundary_cuts
    from scar.optimization.stateful_cpu_model import stateful_resource_context

    schedule = inputs.schedule
    _validate_cost_table(table, inputs)
    if table.tick_numerator != 1 or table.tick_denominator != 10**9:
        raise ValueError('boundary component model requires original nanosecond table ticks')
    if 'implementation_envelope' not in schedule.proof:
        raise ValueError('boundary cuts require the source-replayed standalone implementation envelope')
    candidate = next(row for row in domain['candidates'] if row['id'] == candidate_id)
    if candidate['worker_count'] != 1:
        raise ValueError('boundary cut contract currently requires the original serial candidate')
    if by_task is None:
        by_task = _activities_by_task(schedule, 'boundary cut adapter needs one synchronous lifecycle per original task')
    phase_tasks = defaultdict(list)
    for position, task in enumerate(schedule.tasks):
        kind = task.attrs.get('phase_kind')
        if task.kind is K.IMPLEMENTATION_PHASE and type(kind) is str:
            phase_tasks[kind].append((position, task))
    modern = 'implementation_scope_cleanup' in schedule.proof
    cuts = _SCOPE_CUTS if modern else _CUTS
    expected_schema = _SCOPE_BOUNDARY_SCHEMA if modern else _BOUNDARY_SCHEMA
    costs = list(table.costs); mappings = []; facts = []; residuals = []; seen = set()
    for value in calibrations:
        reference = boundary_calibration_reference(value)
        if reference['schema'] != expected_schema:
            raise ValueError('legacy mixed-helper boundary samples and helper-separated scope graphs cannot be interchanged')
        identity = semantic_sha256(reference)
        if identity in seen:
            raise ValueError('one boundary calibration publication cannot be charged twice')
        seen.add(identity)
        if modern:
            from scar.compiler.calibrate_scope_boundary import read_scope_boundary_calibration
            document = read_scope_boundary_calibration(reference, domain=domain, candidate_id=candidate_id)
        else:
            document = read_boundary_cuts(reference, domain=domain, candidate_id=candidate_id)
        if (document.get('schema'), document.get('schema_version')) != (expected_schema, 1):
            raise ValueError('source-replayed boundary cut calibration required')
        if document['resource_context'] != stateful_resource_context(domain, candidate_id):
            raise ValueError('boundary cuts differ from the original candidate resource context')
        if tuple(document['cuts']) != cuts:
            raise ValueError('original shared monotonic boundary cut order differs')
        conditions = _strings(document['conditions'], 'boundary applicability conditions', (tuple, list))
        missing = _strings(document['missing_terms'], 'boundary residual terms', (tuple, list))
        samples = document['samples']; sample_ids = set()
        if type(samples) is not list or len(samples) < 3:
            raise ValueError('retained repeated boundary cut vectors required')
        for sample in samples:
            sample_id, timestamps = sample['id'], sample['timestamps_ns']
            if type(sample_id) is not str or not sample_id or sample_id in sample_ids:
                raise ValueError('unique shared boundary sample identity required')
            sample_ids.add(sample_id)
            if (type(timestamps) is not list or len(timestamps) != len(cuts)
                    or any(type(t) is not int or t < 0 for t in timestamps)
                    or any(a > b for a, b in zip(timestamps, timestamps[1:]))):
                raise ValueError('one original monotonic integer cut vector required per sample')
        phases = document['phases']
        if type(phases) is not list or len(phases) != len(cuts) - 1:
            raise ValueError('one exclusive phase per adjacent original cut pair required')
        phase_ids = set(); covered_kinds = set(); phase_rows = []
        for index, phase in enumerate(phases):
            phase_id = phase['id']
            if type(phase_id) is not str or not phase_id or phase_id in phase_ids:
                raise ValueError('unique original boundary phase identity required')
            phase_ids.add(phase_id)
            if (phase['start'], phase['end']) != cuts[index:index+2]:
                raise ValueError('boundary samples must retain contiguous exclusive cut intervals')
            kinds = _strings(phase['phase_kinds'], 'boundary implementation phase kinds', (tuple, list))
            if covered_kinds.intersection(kinds):
                raise ValueError('one implementation phase cannot be charged by two cut intervals')
            covered_kinds.update(kinds)
            ordered = sorted(entry for kind in kinds for entry in phase_tasks[kind])
            positions = [position for position, _ in ordered]
            selected = [task for _, task in ordered]
            if tuple(task.attrs['phase_kind'] for task in selected) != kinds:
                raise ValueError('boundary kinds must match unique original implementation phases in order')
            if positions != list(range(positions[0], positions[0] + len(positions))):
                raise ValueError('one boundary subject must preserve a continuous original phase group')
            activities = tuple(by_task[task.id] for task in selected)
            replaced = _remove_unknown_costs(costs, activities,
                'boundary evidence may only replace complete UNKNOWN subjects; primitive facts remain charged once')
            values = [sample['timestamps_ns'][index+1] - sample['timestamps_ns'][index] for sample in samples]
            stats = dict(**_sample_statistics(values), repetitions=len(values))
            subject = 'cost:boundary:' + semantic_sha256(dict(source=reference, phase=phase_id,
                schedule=table.source_schedule_sha256, activities=activities))[:32]
            charge = 'boundary-occurrence:' + semantic_sha256(dict(source=reference, phase=phase_id, activities=activities))
            artifact = ProviderArtifact(**reference)
            row = _sampled_cost(subject, activities, stats,
                (ProviderFactReference(artifact, '#/phases/' + str(index)), ProviderFactReference(artifact, '#/samples')),
                charge, conditions, missing)
            costs.append(row)
            phase_rows.append(dict(subject=subject, phase=phase_id, activities=activities,
                task_ids=[task.id for task in selected], statistics=stats,
                replaced_unknown_subjects=replaced))
        mappings.append(dict(source=reference, shared_sample_ids=[sample['id'] for sample in samples],
            phases=phase_rows, exclusive_adjacent_intervals=True, whole_call_charged=False,
            target_activity_observed=False, samplewise_cut_differences=True,
            median_subtraction_used=False, residual_terms=missing))
        facts.append(dict(artifact=reference, locator='#/phases'))
        residuals.append(dict(source=reference, kind=('CANONICAL_SCOPE_BOUNDARY_MODEL_RESIDUAL' if modern
            else 'CANONICAL_BOUNDARY_MODEL_RESIDUAL'),
            missing_terms=list(missing), cost_charged=False))
    result = replace(table, costs=tuple(costs), source_refs=(*table.source_refs, *facts),
                     unmapped_terms=(*table.unmapped_terms, *residuals))
    summary = _validate_cost_table(result, inputs)
    return result, dict(protocol=('SOURCE_REPLAYED_EXCLUSIVE_SCOPE_BOUNDARY_CUTS' if modern
        else 'SOURCE_REPLAYED_EXCLUSIVE_CANONICAL_BOUNDARY_CUTS'),
        mapping=mappings, summary=summary, existing_known_subjects_preserved=True,
        complete_call_cost=False, native_execution_authorized=False)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def project_stage_dag_activity_costs(table, schedule, resources, *, domain, candidate_id,
                                    calibrations=()):
    """Bind reader-qualified full stage signatures to one original CPU task."""
    return _project_stage_dag_activity_costs(table, _prepare_cost_table_inputs(schedule, resources),
        domain=domain, candidate_id=candidate_id, calibrations=calibrations)


def _project_stage_dag_activity_costs(table, inputs, *, domain, candidate_id, calibrations,
                                      by_task=None, stage_tasks=None):
    from scar.compiler.calibrate_stage_dag import read_stage_dag_calibration
    from scar.optimization.stateful_cpu_model import stateful_resource_context

    schedule, resources = inputs.schedule, inputs.resources
    _validate_cost_table(table, inputs)
    _need((table.tick_numerator, table.tick_denominator) == (1, 10**9),
          'stage-DAG component models require original nanosecond table ticks')
    candidate = next((row for row in domain['candidates'] if row['id'] == candidate_id), None)
    _need(candidate is not None and candidate['worker_count'] == 1 and candidate['legal_under_guards'],
          'stage-DAG component model requires an admitted serial candidate')
    _need(encode(resources.target) == domain['target_snapshot'],
          'stage-DAG activities and original candidate target differ')
    plan = read_artifact(domain['base_stage_plan'])
    if by_task is None:
        by_task = _activities_by_task(schedule, 'stage-DAG adapter needs one synchronous lifecycle per original task')
    if stage_tasks is None:
        stage_tasks = _source_stage_tasks(schedule)
    costs = list(table.costs); seen_publications = set(); seen_stages = set()
    mappings = []; source_refs = []; residuals = []
    for value in calibrations:
        reference = value if type(value) is dict else artifact_reference(value, _STAGE_DAG_SCHEMA)
        publication = semantic_sha256(reference)
        _need(publication not in seen_publications,
              'one stage-DAG calibration publication cannot be charged twice')
        seen_publications.add(publication)
        document = read_stage_dag_calibration(reference, domain=domain, candidate_id=candidate_id)
        _need((document.get('schema'), document.get('schema_version')) == (_STAGE_DAG_SCHEMA, 1),
              'source-replayed stage-DAG calibration required')
        _need(document.get('resource_context') == stateful_resource_context(domain, candidate_id),
              'stage-DAG calibration differs from original candidate resource context')
        _need(document.get('component_topology_specialized') is True
              and document.get('target_fragment_compiled') is True
              and document.get('full_target_program_executed') is False
              and document.get('whole_target_call_measured') is False,
              'stage-DAG samples must preserve the specialized-fragment-only measurement scope')
        conditions = _strings(document.get('conditions'), 'stage-DAG applicability conditions', (list,))
        missing = _strings(document.get('missing_terms'), 'stage-DAG residual terms', (list,))
        components = document.get('components')
        _need(type(components) is list and components, 'retained stage-DAG components required')
        repetitions, warmup = document.get('repetitions'), document.get('warmup')
        _need(type(repetitions) is int and 3 <= repetitions <= 10000
              and type(warmup) is int and 1 <= warmup <= 10000,
              'warm repeated stage-DAG samples required')
        rows = []
        for component_index, component in enumerate(components):
            stage_index = component.get('stage_index')
            _need(type(stage_index) is int and 0 <= stage_index < len(plan['stages']),
                  'original stage-DAG index required')
            _need(stage_index not in seen_stages, 'one original stage cannot have duplicate stage-DAG publications')
            seen_stages.add(stage_index)
            stage = plan['stages'][stage_index]
            pointer = '#/stages/' + str(stage_index)
            stage_ref = dict(artifact=domain['base_stage_plan'], locator=pointer)
            _need(stage['index'] == stage_index and stage['kind'] in ('COMPUTE', 'SCALAR')
                  and component.get('stage_ref') == stage_ref,
                  'stage-DAG samples must retain the exact original COMPUTE/SCALAR stage')
            signature = component.get('signature')
            _need(type(signature) is dict and signature,
                  'reader-replayed complete stage-DAG signature required')
            selected = stage_tasks.get(pointer, ())
            _need(len(selected) == 1, 'stage-DAG sample must match exactly one source-stage task')
            task = selected[0]
            _need(task.kind is K.COMPUTE and task.id == stage['id']
                  and task.attrs.get('source_stage_ref') == dict(artifact_role='stage_plan', pointer=pointer)
                  and task.attrs.get('stage_kind') == stage['kind']
                  and task.attrs.get('original_root_expression') == stage['root_expr']
                  and task.attrs.get('shape') == stage['shape'],
                  'stage-DAG task differs from original source-stage kind/root/shape/identity')
            _need(task.id in by_task, 'original stage task has no activity lifecycle')
            activities = (by_task[task.id],)
            replaced = _remove_unknown_costs(costs, activities,
                'stage-DAG evidence may only replace complete UNKNOWN subjects; known costs remain charged once')
            samples = component.get('samples_ns')
            _need(type(samples) is list and len(samples) == repetitions
                  and all(type(sample) is int and sample >= 0 for sample in samples),
                  'retained nonnegative integer stage-DAG samples required')
            stats = _sample_statistics(samples)
            _need(component.get('statistics') == stats,
                  'stage-DAG statistics differ from original retained samples')
            occurrence = dict(domain=domain['domain_id'], candidate=candidate_id,
                task=task.id, activity=activities[0], source=reference, stage=stage_ref)
            subject = 'cost:stage-dag:' + semantic_sha256(dict(
                occurrence=occurrence, schedule=table.source_schedule_sha256))[:32]
            artifact = ProviderArtifact(**reference)
            row = _sampled_cost(subject, activities, stats,
                (ProviderFactReference(artifact, '#/components/' + str(component_index)),
                 ProviderFactReference(ProviderArtifact(**domain['base_stage_plan']), pointer)),
                'stage-dag-occurrence:' + semantic_sha256(occurrence), conditions, missing)
            costs.append(row)
            rows.append(dict(subject=subject, activities=activities, task_ids=[task.id],
                stage_index=stage_index, stage_ref=stage_ref, signature=signature,
                statistics=stats, replaced_unknown_subjects=replaced))
        mappings.append(dict(source=reference, stages=rows,
            full_source_stage_signatures_replayed_by_reader=True,
            component_topology_specialized=True, target_fragment_compiled=True,
            full_target_program_executed=False,
            target_activity_observed=False, whole_call_charged=False,
            residual_terms=missing))
        source_refs.append(dict(artifact=reference, locator='#/components'))
        residuals.append(dict(source=reference, kind='CANONICAL_STAGE_DAG_MODEL_RESIDUAL',
            missing_terms=list(missing), cost_charged=False))
    result = replace(table, costs=tuple(costs), source_refs=(*table.source_refs, *source_refs),
                     unmapped_terms=(*table.unmapped_terms, *residuals))
    summary = _validate_cost_table(result, inputs)
    return result, dict(protocol='SOURCE_REPLAYED_SERIAL_STAGE_DAG_SAMPLES', mapping=mappings,
        summary=summary, existing_known_subjects_preserved=True,
        primitive_unresolved_terms_are_not_final_activity_coverage=True,
        complete_call_cost=False, sample_ranges_are_guaranteed_bounds=False,
        native_execution_authorized=False)


def _project_timing_calibrations(table, inputs, *, domain, candidate_id, by_task, stage_tasks,
                                 boundary_calibrations, dag_calibrations):
    boundary = dag = None
    if boundary_calibrations:
        table, boundary = _project_boundary_activity_costs(table, inputs, domain=domain, candidate_id=candidate_id,
            calibrations=boundary_calibrations, by_task=by_task)
    if dag_calibrations:
        table, dag = _project_stage_dag_activity_costs(table, inputs, domain=domain, candidate_id=candidate_id,
            calibrations=dag_calibrations, by_task=by_task, stage_tasks=stage_tasks)
    return table, boundary, dag
