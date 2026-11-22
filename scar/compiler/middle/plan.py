"""Verified graph rewrites and resource plans."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.program.codec import dumps, loads
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem,
    PlanningResource, problem_document, problem_from_dict,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.constructive_planner import construct_implementations


def simplify_copies(program: ProviderArtifact, anchors, out):
    payload = Path(program.uri).read_bytes()
    if len(payload) != program.size_bytes or sha256(payload).hexdigest() != program.sha256:
        raise ValueError('program graph differs from the frontend receipt')
    graph = loads(payload.decode('utf-8'))
    source = {row['operation']: row['source'] for row in anchors}
    aliases, rewrites = {}, []
    retained = []
    for operation in graph.operations:
        # GCC's SSA copy is a value forwarding operation. A VUSE/VDEF
        # distinguishes memory-dependent loads, including volatile reads.
        if (operation.label == 'gimple_assign' and len(operation.results) == 1
                and len(operation.uses) == 1 and not operation.accesses
                and 'gimple_assign <ssa_name,' in source.get(operation.id.wire, '')):
            copied = operation.results[0]
            forwarded = operation.uses[0].value
            aliases[copied] = forwarded
            rewrites.append(dict(operation=operation.id.wire, result=copied.wire,
                                 forwarded=forwarded.wire, rule='SSA_COPY_PROPAGATION'))
        else:
            retained.append(operation)

    def resolve(value):
        while value in aliases:
            value = aliases[value]
        return value

    kept = {operation.id for operation in retained}
    regions = tuple(replace(region, blocks=tuple(
        replace(block, operations=tuple(op for op in block.operations if op in kept))
        for block in region.blocks)) for region in graph.regions)
    operations = tuple(replace(operation, uses=tuple(
        replace(use, value=resolve(use.value)) for use in operation.uses))
        for operation in retained)
    simplified = replace(graph, regions=regions, operations=operations,
                         values=tuple(value for value in graph.values if value.id not in aliases))
    out.write_text(dumps(simplified))
    return rewrites


def plan_entry(entry: str, program: ProviderArtifact, hardware: ProviderArtifact, core, calibration_ns):
    source_refs = (dict(path=program.uri, sha256=program.sha256, size_bytes=program.size_bytes,
                        region=entry),)
    cost = CostEstimate(calibration_ns, None, None, 'EMPIRICAL_MODEL', 'baseline_launch', source_refs)
    baseline = ImplementationOption('original', 'preserve_entry', source_refs, {f'cpu:{core}': 1},
                                    {}, {}, {}, cost, parameters={})
    problem = PlanningProblem(
        hardware, source_refs, (PlanningResource(f'cpu:{core}', 'EXECUTION', 1),),
        (PlanningOperation('entry', source_refs, (), (baseline,), baseline.id),), (),
        assumptions=('NORMAL_COMPLETION_ONLY', 'PHYSICAL_COST_BASELINE_ONLY'))
    return problem_document(problem), construct_implementations(problem, baseline_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan', description=__doc__,
                                     epilog='For a Python graph, use scar plan --build BUILD --help.')
    parser.add_argument('--problem', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exact', action='store_true', help='use the optional exact resource solver')
    args = parser.parse_args(argv)
    problem = problem_from_dict(json.loads(args.problem.read_bytes()))
    if args.exact:
        from scar.optimization.joint_planner import solve_implementations
        plan = solve_implementations(problem)
    else:
        plan = construct_implementations(problem)
    with args.out.open('x') as output:
        json.dump(plan, output, indent=2)
        output.write('\n')
    return 0
