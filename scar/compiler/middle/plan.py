"""Verified graph rewrites and resource plans."""

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from subprocess import CompletedProcess

from scar.artifacts import _durable_bytes, file_reference
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA, dumps, loads
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem,
    PlanningResource, problem_document, problem_from_dict,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.joint_planner import _verify_admitted_plan

from ..contracts import ExecutionPlanRef, GraphBundle, verify_source_lock


_NATIVE_DELIVERY_PROFILE = 'CPP20_NATIVE_ENTRY'
_NATIVE_COST_SCHEMA = 'scar.native-entry-cost'

_REVISION_SCHEMAS = {
    'request': 'scar.native-entry', 'program': PROGRAM_SCHEMA,
    'graph_evidence': 'scar.native-graph-evidence', 'calls': 'scar.native-call-links',
    'hardware': 'scar.target-hardware-graph', 'simplified': PROGRAM_SCHEMA,
    'rewrites': 'scar.native-rewrites',
}


@dataclass(frozen=True, slots=True)
class NativeRevision:
    entry: str
    artifacts: dict
    rules: tuple[dict, ...]
    edits: tuple[dict, ...]


def _native_input(reference):
    """Share request and inventory documents within one pinned compiler replay."""
    from scar.analysis.replay_transaction import detached_replay

    identity = tuple(sorted(reference.items()))
    return detached_replay('scar.native-input-metadata.v1', identity, lambda: read_artifact(reference))


def rewrite_native(graph: GraphBundle, out):
    """Publish the source-locked revision and pass its parsed request to qualification."""
    if graph.program is None:
        raise ValueError('no validated entry region')
    path = out / 'simplified.json'
    artifacts = graph.references()
    manifest = _native_input(artifacts['request'])
    verify_source_lock(manifest)
    rules = simplify_copies(graph.program, graph.report['anchors'], path)
    edits = copy_source_edits(manifest, graph.report, rules)
    simplified = file_reference(path, PROGRAM_SCHEMA) if rules else artifacts['program']
    ledger = publish_plan(out / 'rewrites.json', dict(
        schema='scar.native-rewrites', schema_version=1, rules=rules, source_edits=edits,
        program=artifacts['program'], result=simplified,
    ))
    revision = NativeRevision(graph.report['entry'], dict(artifacts, simplified=simplified, rewrites=ledger),
                              tuple(rules), tuple(edits))
    return revision, manifest


def schedule_native(revision: NativeRevision, observation, cases, out) -> ExecutionPlanRef:
    from scar.optimization.constructive_planner import construct_implementations

    request = revision.artifacts['request']
    cpu = observation['cpu']
    if (observation.get('schema') != _NATIVE_COST_SCHEMA or observation['request'] != request
            or observation['exit'] != 0 or type(cpu) is not int
            or cases != _native_input(request)['cases']
            or cpu not in entry_cpus(revision.artifacts['hardware'])):
        raise ValueError('native observation differs from the locked request or target')
    artifacts = dict(revision.artifacts, cost=publish_plan(out / 'cost-observation.json', observation))
    decision = native_cost_decision(observation, cases)
    problem = entry_problem(revision.entry, artifacts, observation, revision.edits, decision=decision)
    schedule = construct_implementations(problem, baseline_only=entry_baseline_only(problem))
    artifacts['planning_problem'] = publish_plan(out / 'planning-problem.json', problem_document(problem))
    artifacts['execution_plan'] = publish_plan(out / 'plan.json', schedule)
    if not schedule.get('selected'):
        raise ValueError('no feasible native execution schedule')
    _require_plan_scope(problem, schedule)
    scope = 'BASELINE_ONLY' if observation['schema_version'] == 1 else decision['scope']
    return ExecutionPlanRef(revision.entry, artifacts, decision, scope)


def _require_plan_scope(problem, plan):
    baseline_only = plan.get('baseline_only')
    if type(baseline_only) is not bool or entry_baseline_only(problem) and not baseline_only:
        raise ValueError('execution plan does not honor the planning problem baseline scope')


def read_plan_contract(reference: ExecutionPlanRef):
    """Replay a published schedule against its typed problem without running a solver."""
    problem_document_, plan = reference.read_pair()
    problem = problem_from_dict(problem_document_)
    _require_plan_scope(problem, plan)
    return problem, plan, _verify_admitted_plan(problem, plan)


def read_native_binding(reference: ExecutionPlanRef):
    """Resolve a selected native lowering without reconstructing the choice domain."""
    problem, plan, verification = read_plan_contract(reference)
    artifacts = reference.artifacts
    if artifacts['cost']['schema'] != _NATIVE_COST_SCHEMA:
        raise ValueError('native delivery requires a native cost artifact')
    observation = read_artifact(artifacts['cost'])
    if (problem.target != ProviderArtifact(**artifacts['hardware'])
            or problem.source_refs != entry_sources(artifacts)
            or observation['request'] != artifacts['request'] or observation['exit'] != 0):
        raise ValueError('native planning problem does not bind the admitted inputs')
    if (len(problem.operations) != 1 or problem.operations[0].id != 'entry'
            or problem.data_edges or problem.activities or problem.timing_constraints
            or problem.reserved_memory):
        raise ValueError('native C++20 entry lowering requires one admitted execution region')

    selected = plan['selected'][0]
    mode = next(option for option in problem.operations[0].options if option.id == selected['option'])
    cpu = observation['cpu']
    if (type(cpu) is not int or cpu not in entry_cpus(artifacts['hardware'])
            or mode.parameters.get('region') != reference.entry
            or mode.demands != {f'cpu:{cpu}': 1}
            or mode.workspace_bytes or selected['start_ns'] != 0
            or mode.proof_refs != problem.source_refs):
        raise ValueError('selected native implementation has no admitted CPU binding')

    variant = observation.get('variants', {}).get(mode.id)
    if mode.parameters.get('delivery') != _native_delivery(cpu, variant):
        raise ValueError('selected native implementation has no admitted C++20 delivery binding')
    version = observation['schema_version']
    rewrites = mode.parameters.get('rewrites', [])
    if version == 1 and (variant is not None or not plan['baseline_only'] or mode.id != 'original'):
        raise ValueError('a baseline observation can only deliver the original implementation')
    if version not in (1, 2) or (version == 2 and variant is None):
        raise ValueError('selected native implementation has no calibrated project')
    if mode.id == 'canonical':
        if mode.lowering != 'canonicalize_scalar_copies' or not rewrites or observation.get('calibration_error'):
            raise ValueError('canonical delivery requires a qualified source revision')
    elif mode.id != 'original' or mode.lowering != 'preserve_entry' or rewrites:
        raise ValueError('unsupported native delivery implementation')

    binding = dict(cpus=[cpu], lowering=mode.lowering, selected=plan['selected'],
                   rewrite_operations=rewrites, verification=verification)
    if variant is not None:
        binding['trial_variant'] = variant
    return observation, binding


def entry_sources(artifacts):
    roles = ('request', 'program', 'graph_evidence', 'calls', 'simplified', 'rewrites', 'cost')
    return tuple(dict(role=role, **artifacts[role]) for role in roles)


def entry_cpus(hardware):
    target = _native_input(hardware)
    inventory = _native_input(dict(target['inventory'], schema='scar.target-inventory'))
    available = os.sched_getaffinity(0).intersection(inventory['cpu']['online_cpu_ids'])
    constraints = inventory['execution_constraints']
    if constraints.get('allowed_cpu_ids') is not None:
        available.intersection_update(constraints['allowed_cpu_ids'])
    return available


def simplify_copies(program: ProviderArtifact, anchors, out):
    """Publish the validated copy revision when the program graph changes."""
    payload = Path(program.uri).read_bytes()
    if len(payload) != program.size_bytes or sha256(payload).hexdigest() != program.sha256:
        raise ValueError('program graph differs from the frontend receipt')
    graph = loads(payload.decode('utf-8'))
    source = {row['operation']: row['source'] for row in anchors}
    aliases, rewrites, removed = {}, [], set()
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
            removed.add(operation.id)
            rewrites.append(dict(operation=operation.id.wire, result=copied.wire,
                                 forwarded=forwarded.wire, rule='SSA_COPY_PROPAGATION'))
        else:
            retained.append(operation)
    if not rewrites:
        return rewrites

    def resolve(value):
        while value in aliases:
            value = aliases[value]
        return value

    regions = []
    for region in graph.regions:
        blocks = []
        changed = False
        for block in region.blocks:
            if not removed.isdisjoint(block.operations):
                block = replace(block, operations=tuple(op for op in block.operations if op not in removed))
                changed = True
            blocks.append(block)
        regions.append(replace(region, blocks=tuple(blocks)) if changed else region)

    operations = []
    for operation in retained:
        if any(use.value in aliases for use in operation.uses):
            uses = tuple(replace(use, value=resolve(use.value)) if use.value in aliases else use
                         for use in operation.uses)
            operation = replace(operation, uses=uses)
        operations.append(operation)
    simplified = replace(graph, regions=tuple(regions), operations=tuple(operations),
                         values=tuple(value for value in graph.values if value.id not in aliases))
    out.write_text(dumps(simplified))
    return rewrites


_SIMPLE_COPY = re.compile(
    rb'\{\s*int\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*;\s*return\s+\1\s*;\s*\}')


def copy_source_edits(manifest, evidence, rewrites):
    """Join a removed SSA copy to a typed, source-replayable function body."""
    if not rewrites:
        return []
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


def replay_native_revision(artifacts):
    """Verify the original graph, source lock and exact proposed rewrites."""
    from scar.analysis.replay_transaction import detached_replay

    identity = tuple((role, tuple(sorted(artifacts[role].items()))) for role in _REVISION_SCHEMAS)
    material = detached_replay('scar.cpp20-native-revision.v1', identity, lambda: _read_native_revision(artifacts))
    verify_source_lock(material[0])
    return material


def _read_native_revision(artifacts):
    for role, schema in _REVISION_SCHEMAS.items():
        ref = artifacts[role]
        if ref['schema'] != schema:
            raise ValueError(f'invalid {role} artifact schema')
        if role in {'graph_evidence', 'simplified'} and str(Path(ref['uri']).absolute()) != ref['uri']:
            raise ValueError(f'{role} artifact differs from the plan input')
    manifest = _native_input(artifacts['request'])
    rewrite = read_artifact(artifacts['rewrites'])
    _native_input(artifacts['hardware'])
    if (rewrite['schema_version'] != 1 or rewrite['program'] != artifacts['program']
            or rewrite['result'] != artifacts['simplified']):
        raise ValueError('rewrite ledger does not bind the admitted input and output graphs')
    edits = rewrite.get('source_edits', [])
    typed_units = {unit['path'] for unit in manifest['units']
                   if unit.get('language', manifest['language']) in ('c', 'fortran')}
    inspect_evidence = bool(edits or typed_units)
    hash_only = ['program', 'calls']
    if not inspect_evidence:
        hash_only.append('graph_evidence')
    if not edits and artifacts['simplified'] != artifacts['program']:
        hash_only.append('simplified')
    for role in hash_only:
        ref = artifacts[role]
        if file_reference(ref['uri'], ref['schema']) != ref:
            raise ValueError(f'{role} artifact differs from the plan input')
    evidence = read_artifact(artifacts['graph_evidence']) if inspect_evidence else None
    if evidence and any(evidence[key] != artifacts[key] for key in ('request', 'program', 'hardware', 'calls')):
        raise ValueError('compiler evidence differs from the admitted frontend outputs')
    compiler_types = {row['unit']: row['artifacts']['types'] for row in (evidence or {}).get('units', ())
                      if row['unit'] in typed_units and 'types' in row['artifacts']}
    if compiler_types.keys() != typed_units:
        raise ValueError('C and Fortran translation require original type evidence for every unit')
    if edits:
        if copy_source_edits(manifest, evidence, rewrite['rules']) != edits:
            raise ValueError('source copy edits differ from frontend and IR evidence')
        simplified = loads(read_artifact(artifacts['simplified'], json_document=False).decode('utf-8'))
        remaining = {op.id.wire for op in simplified.operations}
        regions = {region.id.wire for region in simplified.regions}
        if any(edit['operation'] in remaining or edit['region'] not in regions for edit in edits):
            raise ValueError('source edits do not match the simplified IR')
    return manifest, edits, compiler_types


def _trial_output(run):
    return CompletedProcess([], run['exit'],
                            read_artifact(run['stdout'], json_document=False),
                            read_artifact(run['stderr'], json_document=False))


def native_cost_decision(observation, locked_cases):
    """Qualify a challenger across every locked case using fresh-process pairs."""
    if observation['schema_version'] == 1:
        return dict(scope='ONE_BASELINE_LAUNCH', admitted=False, cases={})
    if observation['schema_version'] != 2:
        raise ValueError('unsupported native cost evidence')
    from scar.optimization.paired_entry_selection import compare_entry_pairs
    from scar.runtime.cpu_paired_runner import compare_native_outputs

    variants, cases = observation['variants'], observation['cases']
    if set(variants) - {'original', 'canonical'} or 'original' not in variants:
        raise ValueError('native timing requires the original implementation')
    if error := observation.get('calibration_error'):
        return dict(scope='PAIRED_CPP20_VARIANTS', admitted=False, cases={}, reason=error)
    count = observation['measurement_pairs']
    case_by_id = {case['id']: case for case in locked_cases}
    if (set(variants) != {'original', 'canonical'} or not cases
            or set(cases) != set(observation['case_inputs']) or set(cases) != set(case_by_id)):
        raise ValueError('paired calibration does not cover the locked cases and implementations')
    decisions = {}
    for name, case in cases.items():
        if type(count) is not int or count < 3 or len(case['pairs']) != count + 1:
            raise ValueError(f'incomplete fresh-process pairs: {name}')
        reference = _trial_output(case['reference'])
        runs, mismatches = [], set()
        for index, pair in enumerate(case['pairs']):
            if set(pair) != set(variants):
                raise ValueError(f'unpaired implementation trial: {name}')
            for mode, run in pair.items():
                output = _trial_output(run)
                if not compare_native_outputs(case_by_id[name], reference, output):
                    mismatches.add(mode)
                runs.append(dict(pair=index, mode=mode, wall_ns=run['ns'], returncode=run['exit'],
                                 stdout_sha256=case['reference']['stdout']['sha256'],
                                 stderr_sha256=case['reference']['stderr']['sha256']))
        decisions[name] = (dict(verdict='VALIDATION_MISMATCH', modes=sorted(mismatches))
                           if mismatches else compare_entry_pairs(
                               runs, incumbent='original', challenger='canonical',
                               warmup_pairs=1, measurement_pairs=count, minimum_gain=0.0))
    return dict(scope='PAIRED_CPP20_VARIANTS', cases=decisions, admitted=all(
        row['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW' for row in decisions.values()))


def entry_baseline_only(problem):
    return 'PHYSICAL_COST_BASELINE_ONLY' in problem.assumptions


def _native_delivery(cpu, variant):
    binding = dict(profile=_NATIVE_DELIVERY_PROFILE, version=1, device='CPU', cpu_ids=[cpu])
    if variant is not None:
        binding['trial_variant'] = variant
    return binding


def entry_problem(entry, artifacts, observation, edits=(), *, decision=None):
    source_refs = entry_sources(artifacts)
    cpu = observation['cpu']
    core = f'cpu:{cpu}'
    cost = CostEstimate(observation['elapsed_ns'], None, None, 'EMPIRICAL_MODEL',
                        'baseline_launch', (dict(role='cost', **artifacts['cost']),))

    def implementation(mode, estimate):
        lowering = 'canonicalize_scalar_copies' if mode == 'canonical' else 'preserve_entry'
        variant = observation.get('variants', {}).get(mode)
        parameters = dict(region=entry, delivery=_native_delivery(cpu, variant))
        if mode == 'canonical':
            parameters['rewrites'] = [row['operation'] for row in edits]
        return ImplementationOption(mode, lowering, source_refs, {core: 1}, {}, {}, {}, estimate,
                                    parameters=parameters)

    baseline_only = True
    options = (implementation('original', cost),)
    if observation['schema_version'] == 2:
        if not edits:
            raise ValueError('paired native variants require a certified source rewrite')
        if decision is None:
            decision = native_cost_decision(observation, read_artifact(artifacts['request'])['cases'])
        observed = decision['cases'].values()
        if observed and all('metrics' in row for row in observed):
            modes = ('original', 'canonical')
            options = []
            for mode in modes:
                metrics = [row['metrics'][mode] for row in observed]
                estimate = CostEstimate((1 + sum(row['median_ns_twice'] for row in metrics)) // 2,
                    sum(row['min_ns'] for row in metrics), sum(row['max_ns'] for row in metrics),
                    'EMPIRICAL_MODEL', 'sum_of_locked_case_pair_medians', cost.source_refs)
                options.append(implementation(mode, estimate))
            options = tuple(options)
            baseline_only = not decision['admitted']
    elif observation['schema_version'] != 1 or 'variants' in observation:
        raise ValueError('unsupported native baseline cost evidence')
    assumptions = ('NORMAL_COMPLETION_ONLY', 'PHYSICAL_COST_BASELINE_ONLY' if baseline_only
                   else 'PAIRED_CPU_WALL_TIME_DOMINANCE')
    return PlanningProblem(
        ProviderArtifact(**artifacts['hardware']), source_refs,
        (PlanningResource(core, 'EXECUTION', 1, source_refs=cost.source_refs),),
        (PlanningOperation('entry', source_refs, (), options, options[0].id),), (),
        assumptions=assumptions)


def publish_plan(path, document):
    """Persist the versioned plan used by downstream replay."""
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _durable_bytes(path, encoded, replace=False)
    return artifact_reference(path, document['schema'])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan', description=__doc__,
                                     epilog='For a Python graph, use scar plan --build BUILD --help.')
    parser.add_argument('--problem', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exact', action='store_true', help='use the optional exact resource solver')
    args = parser.parse_args(argv)
    problem = problem_from_dict(json.loads(args.problem.read_bytes()))
    baseline_only = entry_baseline_only(problem)
    if args.exact:
        from scar.optimization.joint_planner import solve_implementations
        plan = solve_implementations(problem, baseline_only=baseline_only)
    else:
        from scar.optimization.constructive_planner import construct_implementations

        plan = construct_implementations(problem, baseline_only=baseline_only)
    if plan.get('selected'):
        _require_plan_scope(problem, plan)
    publish_plan(args.out, plan)
    return 0
