"""Source-replayed canonical component evidence for existing CPU activities.

Canonical samples become conditional component predictions, never observations
of target activities or guaranteed latency bounds. Pool terms stay grouped.
"""
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from scar.ir.activity_costs import (ActivityCost, ActivityCostTable, CostBasis,
    CostCoverage, CostInterval, IntervalKind, validate_activity_cost_table)
from scar.ir.execution_recipe import read_artifact
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import projection_document
from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_document
from scar.optimization.resource_solver import semantic_sha256


def _samples(reference, term, session):
    """Recompute statistics from retained primitive samples and original scopes."""
    from scar.optimization.deployment_cost import _pointer
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
    expected = dict(median_ns=statistics.median(values), min_ns=min(values),
        max_ns=max(values), stdev_ns=statistics.stdev(values))
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
    rows = tuple(_unknown('cost:'+activity.id, (activity.id,), reason)
                 for activity in schedule.temporal.activities)
    table = ActivityCostTable(semantic_sha256(schedule_document(schedule)),
        semantic_sha256(projection_document(resources)), resources.target, rows, source_refs=source_refs)
    validate_activity_cost_table(table, schedule, resources)
    return table, dict(protocol='UNSUPPORTED', reason=reason)


def project_stateful_activity_costs(schedule, resources, recipe, domain, candidate_id,
                                    calibrations, *, source_refs=()):
    """Map exact stage occurrences and whole-pool signatures, preserving holes.

    The caller replays the recipe and G2 resource publication. Domain/source,
    binding, backend and canonical facts are independently replayed here.
    """
    from scar.optimization.stateful_cpu_model import evaluate_stateful_cpu_components
    from scar.optimization.stateful_cpu_choices import _family
    original = semantic_sha256(recipe['task_graph'])
    actual = semantic_sha256(schedule_document(replace(schedule, temporal=None)))
    if actual != original:
        from scar.analysis.implementation_envelope import derive_stateful_envelope
        from scar.ir.versioned_schedule import schedule_from_dict
        refined = derive_stateful_envelope(schedule_from_dict(recipe['task_graph']), recipe)
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
    activities_by_task = {}
    for activity in schedule.temporal.activities:
        if activity.task in activities_by_task:
            raise ValueError('canonical stage adapter needs one original CPU lifecycle per task')
        activities_by_task[activity.task] = activity.id
    tasks = {task.id: task for task in schedule.tasks}
    grouped, unmapped, represented = [], [], set()
    session = dict(sources={}, validated_probes=set(), statistics={})
    mapping = []
    for term in evaluation['components']:
        source = term['source_ref']
        selected = []
        if source is not None:
            selected = [task for task in schedule.tasks
                        if task.attrs.get('source_stage_ref', {}).get('pointer') == source['pointer']]
            if source['artifact_role'] != 'base_stage_plan' or len(selected) != 1:
                raise ValueError('stage cost must match one original source-stage occurrence')
        elif term['component'] in ('ALLOCATE_FIXED_POOL', 'RELEASE_FIXED_POOL'):
            kind = K.ALLOCATE if term['component'] == 'ALLOCATE_FIXED_POOL' else K.RELEASE
            selected = [task for task in schedule.tasks if task.kind is kind]
            slots = [name for task in selected for name in (task.allocates if kind is K.ALLOCATE else task.releases)]
            expected = [row['id'] for row in actual_plan['workspace_slots']]
            if (slots != expected or term['parameters'] != dict(capacities=[
                    row['capacity_bytes'] for row in actual_plan['workspace_slots']])):
                raise ValueError('collective pool cost must match the exact original slot group')
        elif term['component'] == 'INPUT_FINITE_SCAN_SERIAL':
            selected = [task for task in schedule.tasks
                        if task.kind is K.IMPLEMENTATION_PHASE
                        and task.attrs.get('phase_kind') == 'INPUT_FINITE_SCAN_SERIAL'
                        and activities_by_task[task.id] not in represented
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
        subject = 'cost:'+semantic_sha256(dict(schedule=semantic_sha256(schedule_document(schedule)),
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
            row = ActivityCost(subject, identities, CostBasis.DECLARED_COMPONENT_MODEL,
                estimate_ticks=math.ceil(stats['median_ns']),
                interval=CostInterval(math.floor(stats['min_ns']), math.ceil(stats['max_ns']),
                    IntervalKind.EMPIRICAL_PREDICTION),
                component_refs=(reference,), charge_keys=(occurrence,),
                coverage=CostCoverage.COMPONENT_ONLY,
                conditions=(*conditions, 'CANONICAL_SAMPLE_RANGE_IS_NOT_A_GUARANTEED_TARGET_BOUND',
                    'ACTUAL_IMPLEMENTATION_DISPATCH_AND_DATA_REGIME_HAVE_NOT_BEEN_VALIDATED'),
                missing_terms=('UNVALIDATED_IMPLEMENTATION_DELTA_AND_UNMAPPED_OUTER_ENVELOPE',))
        grouped.append(row)
        mapping.append(dict(subject=subject, activities=identities, task_ids=[task.id for task in selected],
            source_term=term['id'], component=term['component'], count=term['count'],
            charged_as_one_collective_pool=term['component'] in ('ALLOCATE_FIXED_POOL', 'RELEASE_FIXED_POOL')))
    for task in schedule.tasks:
        identity = activities_by_task[task.id]
        if identity not in represented:
            grouped.append(_unknown('cost:'+identity, (identity,), 'NO_MATCHING_EXACT_ACTIVITY_COST_SCOPE'))
    table = ActivityCostTable(semantic_sha256(schedule_document(schedule)),
        semantic_sha256(projection_document(resources)), resources.target, tuple(grouped),
        source_refs=source_refs, unmapped_terms=tuple(unmapped))
    summary = validate_activity_cost_table(table, schedule, resources)
    derivation = dict(protocol='EXACT_STATEFUL_CANONICAL_COMPONENT_SIGNATURES', candidate_id=candidate_id,
        demand_context=demand['context'], resource_context=demand['resource_context'], mapping=mapping,
        original_terms=len(demand['terms']), original_component_signatures=len({row['key'] for row in demand['terms']}),
        unknown_terms=evaluation['unresolved'], rejected_calibrations=evaluation['rejected_calibrations'],
        inapplicable_calibrations=evaluation['inapplicable_calibrations'],
        full_call_cost_closed=False, sample_ranges_are_guaranteed_bounds=False,
        pool_cost_split_or_multiplied_by_slot_count=False, summary=summary)
    return table, derivation
