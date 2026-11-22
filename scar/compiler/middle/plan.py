"""Verified graph rewrites and resource plans."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from scar.ir.execution_recipe import artifact_reference, read_artifact
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


_SIMPLE_COPY = re.compile(
    rb'\{\s*int\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*;\s*return\s+\1\s*;\s*\}')


def copy_source_edits(manifest, evidence, rewrites):
    """Join a removed SSA copy to a typed, source-replayable function body."""
    removed = {row['operation'] for row in rewrites}
    anchored = {}
    for row in evidence['anchors']:
        if row['operation'] in removed:
            anchored.setdefault(row['unit'], {}).setdefault(row['symbol'], []).append(row)
    root = Path(manifest['source_root'])
    edits = []
    for unit in evidence['units']:
        path = unit['unit']
        if unit['language'] not in ('c', 'c++') or 'types' not in unit['artifacts']:
            continue
        candidates = anchored.get(path)
        if not candidates:
            continue
        typed = json.loads(read_artifact(unit['artifacts']['types'], json_document=False))
        records = typed['records']
        children_by_parent = {}
        for index, node in enumerate(records):
            if node['path'] == path and node['parent'] is not None:
                children_by_parent.setdefault(node['parent'], []).append((index, node))

        def scalar_reference(index):
            children = children_by_parent.get(index, ())
            while len(children) == 1 and children[0][1]['kind'] == 'UnexposedExpr':
                children = children_by_parent.get(children[0][0], ())
            if len(children) == 1 and children[0][1]['kind'] == 'DeclRefExpr':
                return (children[0][1]['reference'] or {}).get('usr')
            return None

        source = (root / path).read_bytes()
        if sha256(source).hexdigest() != manifest['source_lock']['files'][path]['sha256']:
            raise ValueError(f'copy lowering source changed: {path}')
        for index, function in enumerate(records):
            symbol = function['symbol']
            if (function['path'] != path or function['kind'] != 'FunctionDecl'
                    or not function['definition'] or function['parent'] is not None
                    or function['canonical_type'] not in ('int (int)', 'int (int) noexcept')
                    or symbol not in candidates):
                continue
            children = children_by_parent.get(index, ())
            bodies = [(serial, node['span']) for serial, node in children if node['kind'] == 'CompoundStmt']
            params = [node for _, node in children if node['kind'] == 'ParmDecl']
            if len(bodies) != 1 or not bodies[0][1] or len(params) != 1 or params[0]['canonical_type'] != 'int':
                continue
            body_index, body = bodies[0]
            statements = children_by_parent.get(body_index, ())
            if [node['kind'] for _, node in statements] != ['DeclStmt', 'ReturnStmt']:
                continue
            locals_ = children_by_parent.get(statements[0][0], ())
            if len(locals_) != 1 or locals_[0][1]['kind'] != 'VarDecl' or locals_[0][1]['canonical_type'] != 'int':
                continue
            local_index, declaration = locals_[0]
            if (not params[0]['usr'] or not declaration['usr']
                    or scalar_reference(local_index) != params[0]['usr']
                    or scalar_reference(statements[1][0]) != declaration['usr']):
                continue
            start, stop = body
            original = source[start:stop]
            match = _SIMPLE_COPY.fullmatch(original)
            if not match or match.groups() != (declaration['name'].encode(), params[0]['name'].encode()):
                continue
            local, parameter = (name.decode('ascii') for name in match.groups())
            assignment = re.compile(
                rf'gimple_assign <ssa_name, {local}_\d+, {parameter}_\d+\(D\), NULL, NULL>$')
            origins = [row for row in candidates[symbol] if assignment.search(row['source'])]
            if len(origins) != 1:
                continue
            replacement = b'{ return ' + match[2] + b';' + b'\n' * original.count(b'\n') + b' }'
            edits.append(dict(unit=path, symbol=symbol, operation=origins[0]['operation'],
                              region=origins[0]['region'], span=body,
                              original_sha256=sha256(original).hexdigest(),
                              replacement=replacement.decode('ascii')))
    return sorted(edits, key=lambda row: (row['unit'], row['span']))


def entry_problem(entry, artifacts, observation, edits=()):
    source_refs = entry_sources(artifacts)
    core = f'cpu:{observation["cpu"]}'
    cost = CostEstimate(observation['elapsed_ns'], None, None, 'EMPIRICAL_MODEL',
                        'baseline_launch', (dict(role='cost', **artifacts['cost']),))
    lowering = 'canonicalize_scalar_copies' if edits else 'preserve_entry'
    parameters = {'region': entry}
    if edits:
        parameters['rewrites'] = [row['operation'] for row in edits]
    implementation = ImplementationOption('canonical' if edits else 'original', lowering, source_refs,
        {core: 1}, {}, {}, {}, cost, parameters=parameters)
    return PlanningProblem(
        ProviderArtifact(**artifacts['hardware']), source_refs,
        (PlanningResource(core, 'EXECUTION', 1, source_refs=cost.source_refs),),
        (PlanningOperation('entry', source_refs, (), (implementation,), implementation.id),), (),
        assumptions=('NORMAL_COMPLETION_ONLY', 'PHYSICAL_COST_BASELINE_ONLY'))


def plan_entry(entry, artifacts, observation, edits=()):
    problem = entry_problem(entry, artifacts, observation, edits)
    return problem_document(problem), construct_implementations(problem, baseline_only=True)


def project_loop_calibrations(domain,calibrations,out):
    from scar.compiler.calibrate_temporal_cpu import project_temporal_resource_fact_batch
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh loop cost input publication required')
    out.mkdir(parents=True);facts=[];sources=[];batches=[];skipped=[];materials=[]
    composite=domain['schema']=='scar.composite-temporal-cpu-domain'
    for value in calibrations:
        path=Path(value).resolve()
        if path.is_dir():path=path/'result.json'
        document=json.loads(path.read_bytes());schema=document.get('schema')
        reference=artifact_reference(path,schema);read_artifact(reference)
        sources.append(reference)
        materials.append((path,document,reference))
    base_calibrations=[path for path,document,_ in materials
        if document.get('schema')=='scar.temporal-cpu-resource-calibration']
    for index,(path,document,reference) in enumerate(materials):
        schema=document['schema']
        if schema=='scar.temporal-resource-calibration':
            facts.append(path);continue
        if schema=='scar.temporal-cpu-resource-calibration':
            batch=project_temporal_resource_fact_batch(domain,path,out/('calibration-'+str(index)))
            facts.extend(Path(row['observations']['uri']) for row in batch['candidates'])
            batches.append(artifact_reference(out/('calibration-'+str(index))/'result.json',batch['schema']))
            continue
        # Additional typed calibration protocols are handled below, never
        # classified by a program, source filename or mathematical variable.
        if schema=='scar.composite-temporal-boundary-calibration':
            if not composite:
                skipped.append(dict(calibration=reference,reason='COMPOSITE_BOUNDARY_INAPPLICABLE_TO_DIRECT_LOOP'))
                continue
            from scar.compiler.calibrate_temporal_cpu import project_composite_boundary_facts
            projected=project_composite_boundary_facts(domain,path,out/('calibration-'+str(index)),
                base_calibrations=base_calibrations)
            for row in projected['candidates']:facts.append(Path(row['observations']['uri']))
            batches.append(artifact_reference(out/('calibration-'+str(index))/'result.json',projected['schema']))
            continue
        raise ValueError('independent supported loop resource calibration required: '+str(schema))
    for reference in sources:read_artifact(reference,json_document=False)
    report=dict(schema='scar.loop-cost-inputs',schema_version=1,domain_id=domain['id'],
        calibrations=sources,batches=batches,skipped=skipped,
        facts=[artifact_reference(path,'scar.temporal-resource-calibration') for path in facts],
        source_independent_projections=True,target_program_executed=False)
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return facts,artifact_reference(out/'result.json',report['schema'])


def publish_plan(path, document):
    """Persist the versioned plan used by downstream replay."""
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    return artifact_reference(path, document['schema'])


def plan_loop_region(proof, hardware, directory, *, shapes, count, budget, catalog=None,
                     model=False, threads=None, max_threads=None, calibrations=(), original_cost=None):
    """Plan from an admitted loop proof and publish the selected resource contract."""
    target = Path(hardware['path']).parent
    composite = catalog is not None
    scope = {'catalog': catalog} if composite else {}
    arguments = dict(input_shapes=shapes, trip_count=count, memory_budget_bytes=budget)
    report = dict(physical_optimality_proven=False, performance_qualified=False)
    if model and threads is not None:
        raise ValueError('automatic loop cost selection and an explicit CPU preview are separate modes')

    if model or threads is not None:
        if composite:
            from scar.optimization.temporal_cpu_choices import (
                build_composite_temporal_cpu_domain as build_domain,
                preview_composite_temporal_cpu_choice as preview,
                verify_composite_temporal_cpu_preview as verify_preview,
                plan_selected_composite_temporal_cpu_choice as select,
                verify_selected_composite_temporal_cpu_choice as verify_selection,
            )
        else:
            from scar.optimization.temporal_cpu_choices import (
                build_temporal_cpu_domain as build_domain,
                preview_temporal_cpu_choice as preview,
                verify_temporal_cpu_preview as verify_preview,
                plan_selected_temporal_cpu_choice as select,
                verify_selected_temporal_cpu_choice as verify_selection,
            )
        domain = build_domain(proof, target, **scope, **arguments,
                              max_threads=max_threads if model else threads)
        report['cpu_domain'] = publish_plan(directory / 'cpu-domain.json', domain)
        if composite:
            report['guard_profile'] = publish_plan(directory / 'guard-profile.json', domain['guard_profile'])
        if model:
            facts, report['cost_inputs'] = project_loop_calibrations(domain, calibrations, directory / 'resource-facts')
            selection = select(domain, resource_facts=facts, no_op_profile=original_cost)
            choice, schedule = selection['choice'], selection['schedule']
            report.update(
                cpu_choice=publish_plan(directory / 'cpu-choice.json', choice),
                candidate_count=len(domain['candidates']),
                complete_cost_profiles=sum(row['model']['complete'] for row in choice['variants']),
                cost_domain_complete=choice['complete_cost_domain'],
                original_no_op_cost_known=choice['original_no_op']['complete'],
                selection_scope=('SOURCED_COMPOSITE_FORWARD_CPU_COMPONENT_MODEL' if composite
                                 else 'SOURCED_FORWARD_CPU_COMPONENT_MODEL'),
            )
            if schedule is None:
                report.update(selected_kind='NO_OP', cost_status=choice['status'],
                              reason='complete forward cost domain or profitable model winner unavailable')
                return report
            verify_selection(proof, target, schedule, **scope)
        else:
            schedule = preview(domain, cpu_threads=threads)
            verify_preview(proof, target, schedule, **scope)
            report['selection_scope'] = ('EXPLICIT_COMPOSITE_CPU_TEAM_PREVIEW_UNRANKED' if composite
                                         else 'EXPLICIT_CPU_TEAM_PREVIEW_UNRANKED')
    else:
        if composite:
            from scar.optimization.composite_invariant_schedule import (
                plan_composite_invariant_loop_schedule as plan_schedule,
                verify_composite_invariant_loop_schedule as verify_schedule,
            )
        else:
            from scar.optimization.invariant_loop_schedule import (
                plan_invariant_loop_schedule as plan_schedule,
                verify_invariant_loop_schedule as verify_schedule,
            )
        schedule = plan_schedule(proof, target, **scope, **arguments)
        verify_schedule(proof, target, schedule, **scope)
        report['selection_scope'] = (schedule['comparison_scope'] if composite
                                     else 'STRUCTURAL_LOGICAL_WORK_AND_RESOURCE_MODEL')

    snapshot = schedule['target_snapshot']
    if (snapshot['uri'], snapshot['sha256'], snapshot['size_bytes']) != (
            hardware['path'], hardware['sha256'], hardware['bytes']):
        raise ValueError('loop schedule differs from the original hardware snapshot')
    report.update(
        schedule=publish_plan(directory / 'schedule.json', schedule),
        selected_kind=schedule['selected_kind'],
        profitability_status=schedule['profitability_status'],
        input_shapes=shapes, trip_count=count,
        memory_budget_bytes=schedule['effective_incremental_budget_bytes'],
        requested_memory_budget_bytes=budget,
    )
    return report


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
