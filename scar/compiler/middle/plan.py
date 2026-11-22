"""Verified graph rewrites and resource plans."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

from scar.ir.execution_recipe import read_artifact
from scar.ir.program.codec import dumps, loads
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem,
    PlanningResource, problem_document, problem_from_dict,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.constructive_planner import construct_implementations


def entry_sources(artifacts):
    roles = ('request', 'program', 'graph_evidence', 'calls', 'simplified', 'rewrites', 'cost')
    return tuple(dict(role=role, **artifacts[role]) for role in roles)


def entry_cpus(hardware):
    target = read_artifact(hardware)
    inventory = read_artifact(dict(target['inventory'], schema='scar.target-inventory'))
    available = os.sched_getaffinity(0).intersection(inventory['cpu']['online_cpu_ids'])
    constraints = inventory['execution_constraints']
    if constraints.get('allowed_cpu_ids') is not None:
        available.intersection_update(constraints['allowed_cpu_ids'])
    return available


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


def entry_problem(entry, artifacts, observation):
    source_refs = entry_sources(artifacts)
    core = f'cpu:{observation["cpu"]}'
    cost = CostEstimate(observation['elapsed_ns'], None, None, 'EMPIRICAL_MODEL',
                        'baseline_launch', (dict(role='cost', **artifacts['cost']),))
    baseline = ImplementationOption('original', 'preserve_entry', source_refs, {core: 1},
                                    {}, {}, {}, cost, parameters={'region': entry})
    return PlanningProblem(
        ProviderArtifact(**artifacts['hardware']), source_refs,
        (PlanningResource(core, 'EXECUTION', 1, source_refs=cost.source_refs),),
        (PlanningOperation('entry', source_refs, (), (baseline,), baseline.id),), (),
        assumptions=('NORMAL_COMPLETION_ONLY', 'PHYSICAL_COST_BASELINE_ONLY'))


def plan_entry(entry, artifacts, observation):
    problem = entry_problem(entry, artifacts, observation)
    return problem_document(problem), construct_implementations(problem, baseline_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan', description=__doc__,
                                     epilog='For a Python graph, use scar plan --build BUILD --help.')
    parser.add_argument('--problem', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exact', action='store_true', help='use the optional exact resource solver')
    args = parser.parse_args(argv)
    problem = problem_from_dict(json.loads(args.problem.read_bytes()))
    baseline_only = 'PHYSICAL_COST_BASELINE_ONLY' in problem.assumptions
    if args.exact:
        from scar.optimization.joint_planner import solve_implementations
        plan = solve_implementations(problem, baseline_only=baseline_only)
    else:
        plan = construct_implementations(problem, baseline_only=baseline_only)
    with args.out.open('x') as output:
        json.dump(plan, output, indent=2)
        output.write('\n')
    return 0
