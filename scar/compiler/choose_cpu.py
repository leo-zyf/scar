"""Construct a hardware-derived CPU implementation domain and solve its model.

No target implementation is compiled or timed here. Primitive cost profiles
are matched to exact target/executor/CPU contexts; missing or ambiguous costs
remain entries in the finite domain, never silently removed alternatives.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .model import read_plan, write_plan
from scar.optimization.hardware import propose_cpu_placements, bind_cpu_placement
from scar.optimization.cost_model import predict_region_costs
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.schedule import build_region_schedule, schedule_document


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _receipt(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _context(target, executor, cpus):
    return target, executor, tuple(cpus)


def choose_cpu(plan_path, target, out, *, profile_paths=(), input_metadata_path=None,
               input_shapes=None, cpu_pool=None, max_threads=None, search_budget=100000,
               calibrate_missing=False):
    from scar.optimization.cpu_choices import plan_cpu_choices
    plan_path = Path(plan_path).absolute()
    base = read_plan(plan_path)
    if base.resource_binding is not None:
        raise ValueError('unbound semantic base plan required; an existing CPU binding is not an optimization domain')
    shape_evidence = None
    if input_metadata_path is not None:
        if input_shapes is not None:
            raise ValueError('observed metadata and declared shapes are mutually exclusive')
        from .input_metadata import shapes_from_metadata
        input_shapes, shape_evidence = shapes_from_metadata(
            input_metadata_path, base.region.source, base.region.parameters)
    if input_shapes is None:
        raise ValueError('observed or explicitly declared input shapes required')
    profiles, retained_profiles = {}, []
    for item in profile_paths:
        path = Path(item).absolute()
        profile = json.loads(path.read_bytes())
        if (profile.get('schema'), profile.get('schema_version')) != ('scar.region-cost-profile', 1):
            raise ValueError('versioned CPU cost profile required')
        row = profile['target']
        key = _context(row['target_sha256'], row['executor_id'], row['cpu_ids'])
        # Identical repeated arguments do not create an artificial ambiguity.
        profiles.setdefault(key, {})[semantic_sha256(profile)] = str(path)
        retained_profiles.append(dict(**_receipt(path), context=dict(row)))
    placements = propose_cpu_placements(base, target, affinity=cpu_pool, max_threads=max_threads)
    out = Path(out).absolute()
    if out.exists():
        raise ValueError('fresh CPU choice output required')
    out.mkdir(parents=True)
    variants, domain_rows = [], []
    for index, placement in enumerate(placements):
        folder = out / ('implementation-' + str(index))
        folder.mkdir()
        bound = bind_cpu_placement(base, placement)
        plan = folder / 'plan.json'
        resource_path = folder / 'resource-plan.json'
        write_plan(bound, plan)
        schedule = build_region_schedule(bound, placement.target_path)
        _write(resource_path, schedule_document(schedule))
        key = _context(placement.target_sha256, placement.executor_id, placement.cpu_ids)
        matching = profiles.get(key, {})
        applicability, applicable = [], []
        for profile_path in matching.values():
            prediction = predict_region_costs(bound, schedule,
                json.loads(Path(profile_path).read_bytes()), input_shapes=input_shapes)
            applicability.append(dict(profile=profile_path,
                prediction_available=prediction['prediction_available'], unresolved=prediction['unresolved']))
            if prediction['prediction_available']:
                applicable.append(profile_path)
        calibration = None
        if not applicable and calibrate_missing:
            # Calibrate the backend using canonical primitive programs only.
            # The target region is still neither compiled nor executed here.
            from .calibrate_cpu import calibrate_cpu
            calibration = folder / 'calibration'
            generated = calibrate_cpu(plan, calibration, input_shapes=input_shapes)
            profile_path = str(calibration / 'profile.json')
            checked = predict_region_costs(bound, schedule, generated, input_shapes=input_shapes)
            applicability.append(dict(profile=profile_path,
                prediction_available=checked['prediction_available'], unresolved=checked['unresolved']))
            if checked['prediction_available']:
                applicable.append(profile_path)
            retained_profiles.append(dict(**_receipt(profile_path), context=dict(generated['target']),
                origin='CANONICAL_PRIMITIVE_CALIBRATION_NOT_TARGET_TRIAL'))
        profile = applicable[0] if len(applicable) == 1 else None
        identifier = 'cpu-' + semantic_sha256(asdict(placement))[:16]
        variant = dict(id=identifier, plan=str(plan), schedule=str(resource_path), profile=profile)
        if profile is None:
            variant['frontier_reason'] = ('missing exact CPU-context cost profile' if not matching
                else 'no supplied CPU-context profile applies to this shape and backend' if not applicable
                else 'multiple applicable cost profiles for the same CPU context')
        variants.append(variant)
        domain_rows.append(dict(id=identifier, placement=asdict(placement),
            plan=_receipt(plan), resource_schedule=_receipt(resource_path), cost_profile=profile,
            calibration_directory=None if calibration is None else str(calibration),
            profile_applicability=applicability,
            cost_status=('UNIQUE_CONTEXT_PROFILE' if profile is not None else
                         'MISSING_CONTEXT_PROFILE' if not matching else
                         'NO_APPLICABLE_CONTEXT_PROFILE' if not applicable else 'AMBIGUOUS_CONTEXT_PROFILES')))
    domain = dict(schema='scar.cpu-implementation-domain', schema_version=1,
        base_plan=_receipt(plan_path), target=str(Path(target).absolute()),
        policy=dict(cpu_pool=None if cpu_pool is None else list(cpu_pool), max_threads=max_threads,
            enumeration='EXECUTOR_CPU_PREFIXES_AT_ONE_POWERS_OF_TWO_AND_CAP',
            scope='FINITE_DECLARED_CPU_ALLOCATION_FAMILY_NOT_ALL_PHYSICAL_IMPLEMENTATIONS'),
        input_shapes=input_shapes, input_shape_evidence=shape_evidence,
        profiles=retained_profiles, variants=domain_rows,
        primitive_calibration_enabled=calibrate_missing,
        target_variants_compiled=False, target_variants_executed=False,
        physical_optimality_proven=False, deployment_authorized=False)
    domain_path = out / 'domain.json'
    _write(domain_path, domain)
    choice = plan_cpu_choices(plan_path, variants, out / 'choice',
        input_shapes=input_shapes, search_budget=search_budget)
    report = dict(schema='scar.cpu-choice-pipeline', schema_version=1,
        domain=_receipt(domain_path), choice=_receipt(out / 'choice' / 'choice.json'),
        selection_scope='CPP_IMPLEMENTATION_FOR_LOWERING_ONLY_ORIGINAL_EXECUTION_REMAINS_LEGAL',
        finite_implementations=len(variants),
        missing_or_ambiguous_profiles=sum(row['cost_profile'] is None for row in domain_rows),
        model_status=choice['solver']['status'], complete_domain=choice['complete_domain'],
        selected_variant=choice['selected_variant'],
        selected_for_use=False, deployment_authorized=False,
        physical_optimality_proven=False, target_variants_compiled=False,
        target_variants_executed=False)
    _write(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--profile', action='append', default=[])
    shapes = parser.add_mutually_exclusive_group(required=True)
    shapes.add_argument('--inputs')
    shapes.add_argument('--shape', type=int, nargs=2)
    parser.add_argument('--cpu-pool', type=int, nargs='+')
    parser.add_argument('--max-threads', type=int)
    parser.add_argument('--search-budget', type=int, default=100000)
    parser.add_argument('--calibrate-missing', action='store_true',
        help='calibrate missing CPU/backend contexts using canonical primitives; never trial target implementations')
    args = parser.parse_args()
    result = choose_cpu(args.plan, args.target, args.out, profile_paths=args.profile,
        input_metadata_path=args.inputs, input_shapes=args.shape,
        cpu_pool=args.cpu_pool, max_threads=args.max_threads, search_budget=args.search_budget,
        calibrate_missing=args.calibrate_missing)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
