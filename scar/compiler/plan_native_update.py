"""Plan one source-ordered CPU update around an existing numeric recipe."""
import argparse
import json
from math import prod
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem, PlanningResource,
    ResourceObligation, SCHEMA as PROBLEM, digest, problem_document, problem_from_dict, validate_problem,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_recipe import verify_stateful_recipe

SCHEMA = 'scar.native-update-planning-preparation'
_SOURCE = 'scar.source-native-update'
_ORIGINAL = 'CURRENT_NUMERIC_RECIPE_AND_ORIGINAL_ORDERED_COPY_CALLS'
_COMPOSED = 'COMPOSED_NATIVE_CPU_UPDATE'


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _at(reference, locator='#'):
    return dict(artifact=reference, locator=locator)


def _extent(rectangle, shape):
    _require(len(rectangle) == len(shape) == 2 and all(
        type(lo) is int and type(hi) is int and 0 <= lo <= hi <= size
        for (lo, hi), size in zip(rectangle, shape, strict=True)
    ), 'copy geometry exceeds its original root')
    return prod(hi - lo for lo, hi in rectangle)


def _overlap(left, right):
    return all(max(a, c) < min(b, d) for (a, b), (c, d) in zip(left, right, strict=True))


def _copy_geometry(proof, reference):
    roots = {row['id']: row for row in proof['array_inputs']}
    _require(len(roots) == len(proof['array_inputs']), 'unique source array roots required')
    _require(all(row['dtype'] in ('float64', '<f8') and row['device'] == 'cpu'
                 and len(row['shape']) == 2 and all(type(n) is int and n >= 0 for n in row['shape'])
                 and row['strides'] == [row['shape'][1], 1]
                 for row in roots.values()), 'one contiguous rank-two CPU binary64 geometry per root required')
    _require(len({semantic_sha256(row['allocation']) for row in roots.values()}) == len(roots),
             'roots need distinct admitted allocations')
    calls = proof['ordered_calls']
    _require(len(calls) >= 2 and calls[0]['kind'] == 'STATEFUL_NUMERIC_REGION',
             'a leading numeric region and ordered boundary copies required')
    input_ids = set(roots) | {row['id'] for row in proof['scalar_inputs']}
    _require(set(calls[0]['input_ids']) == input_ids, 'copy roots must be the numeric region roots')
    rows, overlaps = [], []
    for ordinal, call in enumerate(calls[1:], 1):
        _require(call['kind'] == 'ARRAY_COPY_CALL', 'all following array calls need a copy-effect proof')
        summary = call['copy_effects']
        _require(summary['status'] == 'CLOSED_COPY_EFFECTS_CONDITIONAL',
                 'copy occurrence lacks its original conditional geometry')
        formals = {item['formal_index']: item['root_id'] for item in call['formal_to_root']}
        _require(len(formals) == len(call['formal_to_root']) and set(formals.values()) <= set(roots),
                 'copy call requires the original mapped numeric roots')
        copy_rows, read_bytes, write_bytes, views = [], 0, 0, 0
        for access in summary['accesses']:
            extent = _extent(access['rectangle'], roots[formals[access['formal_index']]]['shape'])
            if access['kind'] == 'CREATE_ALIAS_VIEW':
                views += 1
            elif access['kind'] in ('READ_COPY_SOURCE', 'READ_SCALAR_VALUE'):
                read_bytes += 8 * extent
            elif access['kind'] == 'WRITE_COPY_DESTINATION':
                write_bytes += 8 * extent
            else:
                raise ValueError('original copy content-access kind is unsupported')
        for index, copy in enumerate(summary['copies']):
            root = formals[copy['source_formal']]
            destination = formals[copy['destination_formal']]
            src, dst = copy['source_rectangle'], copy['destination_rectangle']
            nbytes = 8 * _extent(src, roots[root]['shape'])
            _require(nbytes == 8 * _extent(dst, roots[destination]['shape']), 'unequal copy payloads')
            ref = _at(reference, f'#/ordered_calls/{ordinal}/copy_effects/copies/{index}')
            overlap = root == destination and _overlap(src, dst)
            if overlap:
                overlaps.append(ref)
            copy_rows.append(dict(source=ref, source_root=root, destination_root=destination,
                payload_bytes=nbytes, same_root=root == destination, overlap=overlap))
        rows.append(dict(source=_at(reference, f'#/ordered_calls/{ordinal}'), copies=copy_rows,
            logical_read_bytes=read_bytes, logical_write_bytes=write_bytes, alias_descriptors=views))
    return roots, rows, overlaps


def _option(name, refs, lane, pool, private_bytes, obligations, order):
    cost = CostEstimate(None, None, None, 'UNKNOWN', name, refs,
                        ('MATCHED_COMPLETE_STEP_SERVICE_TIME_UNAVAILABLE',))
    return ImplementationOption(name, name, refs, {lane: 1}, {pool: private_bytes}, {}, {}, cost,
        dict(source_order=order, root_materialization='BORROW_ORIGINAL_HOST_ROOTS',
             copy_payload_read_write_preserved=True, lowering_authorized=False),
        tuple(ResourceObligation(field, reason, sources) for field, reason, sources in obligations))


def _derive(source_ref, recipe_ref):
    from scar.analysis.source_native_update import check

    proof = check(source_ref['uri'])
    _require(proof == read_artifact(source_ref), 'original update proof changed during replay')
    recipe = verify_stateful_recipe(recipe_ref['uri'])
    _require(recipe == read_artifact(recipe_ref), 'numeric recipe changed during replay')
    region_ref = recipe['artifacts']['region']
    _require(read_artifact(proof['numeric_region']) == read_artifact(region_ref),
             'update source and numeric recipe have different original regions')
    plan_ref = recipe['artifacts']['stage_plan']
    stage = read_artifact(plan_ref)
    roots, copies, overlaps = _copy_geometry(proof, source_ref)
    _require({key: row['shape'] for key, row in roots.items()} == stage['input_shapes'],
             'source and numeric recipe root extents differ')
    payload = sum(8 * prod(row['shape']) for row in roots.values())
    private_bytes, budget = stage['workspace_payload_bytes'], stage['memory_budget_bytes']
    _require(payload == stage['external_array_bytes']
             and payload + private_bytes == stage['total_region_array_payload_bytes']
             and 0 < payload + private_bytes <= budget,
             'the original counted payload must fit its unchanged numeric budget')
    target = ProviderArtifact(**recipe['artifacts']['target_snapshot'])
    lifetime_ref = _at(source_ref, '#/allocation_lifetime')
    refs = (_at(source_ref), _at(recipe_ref), _at(plan_ref), lifetime_ref)
    scope = semantic_sha256(dict(source=source_ref, recipe=recipe_ref))
    lane = 'synchronous-update-lane:' + scope
    pool = 'scoped-update-array-payload:' + scope
    resources = (
        PlanningResource(lane, 'EXECUTION', 1, (_at(source_ref, '#/parent'),)),
        PlanningResource(pool, 'MEMORY', budget, (_at(plan_ref, '#/memory_budget_bytes'),)),
    )
    order = [_at(source_ref, f'#/ordered_calls/{index}')
             for index in range(len(proof['ordered_calls']))]
    order.append(_at(source_ref, '#/scalar_tail'))
    open_resources = (
        ('physical_execution_and_memory_demand',
         'The scoped caller lane and original array budget do not qualify exclusive CPU service or physical residency.',
         refs),
    )
    composed_obligations = open_resources + (
        ('composed_runtime_admission',
         'Actual native lowering and complete guarded treatment have yet to be independently admitted.',
         refs),
    )
    if overlaps:
        composed_obligations += (('overlap_temporary_or_ordered_semantics',
            'An overlapping root copy needs its original semantics and additional allocation qualification.',
            tuple(overlaps)),)
    options = (
        _option(_ORIGINAL, refs, lane, pool, private_bytes, open_resources, order),
        _option(_COMPOSED, refs, lane, pool, private_bytes, composed_obligations, order),
    )
    operation = PlanningOperation('cpu-update:' + scope, refs, (), options, _ORIGINAL)
    problem = validate_problem(PlanningProblem(
        target, refs, resources, (operation,), (),
        assumptions=('ORIGINAL_NORMAL_ORDER_AND_SCALAR_TAIL_RETAINED',
            'SCOPED_ONE_ACTIVATION_LANE_NOT_A_PHYSICAL_CORE_RESERVATION',
            'BORROWED_ARRAY_ROOTS_COUNTED_ONCE_IN_ORIGINAL_PAYLOAD_BUDGET',
            'SOURCE_LOGICAL_TRANSFER_BYTES_NOT_CACHE_OR_DRAM_CHANNEL_DEMAND',
            'WHOLE_STEP_TIMING_MUST_MATCH_BOTH_COMPLETE_TREATMENTS'),
        reserved_memory={pool: payload},
    ))
    result = construct_implementations(problem)
    replay = dict(
        ordered_effects=order, copy_calls=copies, overlap_frontiers=overlaps,
        copy_moves=sum(len(row['copies']) for row in copies),
        copy_logical_read_bytes=sum(row['logical_read_bytes'] for row in copies),
        copy_logical_write_bytes=sum(row['logical_write_bytes'] for row in copies),
        materializations=dict(
            allocation_lifetime=lifetime_ref,
            two_modes_borrow_same_declared_roots=True,
            source_alias_descriptor_operations=sum(row['alias_descriptors'] for row in copies),
            composed_copy_access_requirement='DIRECT_SAME_ROOT_GEOMETRY_WITH_NO_NEW_ARRAY_ROOT',
            descriptor_allocation_cost_ns=None, content_reuse_across_updates=False,
        ),
        original_array_payload=dict(borrowed_bytes=payload, numeric_workspace_bytes=private_bytes,
                                    counted_budget_bytes=budget, budget_scope=stage['budget_scope']),
        numeric_cpu_ids=stage['resource_binding']['cpu_ids'],
        numeric_recipe_reused_without_graph_copy=True,
        composed_extra_array_payload_bytes=None, physical_transfer_bytes=None,
        whole_step_service_cost='UNKNOWN', planner_status=result['status'],
        default_backend=result['default_backend'], qualified_options=result['selected'],
        eligibility_frontiers=result['frontiers'], deployment_authorized=False,
    )
    return problem, replay


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return _reference(path, value['schema'])


def _report(source, recipe, problem_ref, problem, replay):
    return dict(schema=SCHEMA, schema_version=1, source=source, numeric_recipe=recipe,
                problem=problem_ref, problem_sha256=digest(problem), replay=replay,
                target_program_executed=False, performance_measured=False, deployment_authorized=False)


def prepare_native_update_plan(source, recipe, out):
    source_ref, recipe_ref = _reference(source, _SOURCE), _reference(recipe, 'scar.execution-recipe')
    problem, replay = _derive(source_ref, recipe_ref)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    problem_ref = _write(out / 'problem.json', problem_document(problem))
    report = _report(source_ref, recipe_ref, problem_ref, problem, replay)
    _write(out / 'result.json', report)
    return report


def read_native_update_plan(path):
    reference = _reference(path, SCHEMA)
    report = read_artifact(reference)
    problem, replay = _derive(report['source'], report['numeric_recipe'])
    problem_ref = _reference(Path(reference['uri']).with_name('problem.json'), PROBLEM)
    if (report != _report(report['source'], report['numeric_recipe'], problem_ref, problem, replay)
            or problem_from_dict(read_artifact(problem_ref)) != problem):
        raise ValueError('planning envelope differs from original update and numerical recipe replay')
    read_artifact(reference)
    return report, problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source')
    inputs.add_argument('--check')
    parser.add_argument('--recipe')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.recipe or args.out:
            parser.error('--check only consumes the retained receipt')
        report, _ = read_native_update_plan(args.check)
    else:
        if not args.recipe or not args.out:
            parser.error('preparation requires --recipe and --out')
        report = prepare_native_update_plan(args.source, args.recipe, args.out)
    fields = ('copy_moves', 'copy_logical_read_bytes', 'copy_logical_write_bytes',
              'whole_step_service_cost', 'planner_status', 'default_backend', 'deployment_authorized')
    print(json.dumps({key: report['replay'][key] for key in fields}, sort_keys=True))


if __name__ == '__main__':
    main()
