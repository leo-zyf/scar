"""Bind an admitted CSR subregion to a resource plan and C++ trial."""

import argparse
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import (
    SCHEMA as CALLS, _admit_entry_calls, _derive_entry_source_witness_admitted,
)
from scar.analysis.build import BuildSession
from scar.analysis.csr_row_region import FAMILY, recognize
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_namespace_interfaces import _state
from scar.codegen.sparse_csr import SCHEMA as NATIVE
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationChoice, ImplementationOption, PlanningActivity,
    PlanningOperation, PlanningProblem, PlanningResource, PlanningTimingConstraint,
    problem_document,
)
from scar.ir.program import source_productions as P
from scar.ir.program.normal_targets import Body
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode, encode
from scar.optimization.constructive_planner import construct_implementations
from scar.runtime.csr_region_entry import SELECTION

PROOF = 'scar.csr-row-region-proof'
COST = 'scar.sparse-csr-row-cost'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _write(directory, name, document):
    path = directory/name
    with path.open('x') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _original_suite(session, witness, region):
    bodies = {decode(Body, match['original_body']) for match in witness['matches']}
    _need(len(bodies) == 1, 'CSR rewrite requires one admitted source body')
    body, = bodies
    catalog, view = session._f4.catalog, session.view
    _, syntax, _, _, fingerprint = _source_identity(catalog, body.unit)
    _need(fingerprint == witness['source']['sha256'], 'CSR source syntax digest differs')
    context = BodyInterfaceResolver(view).plan(body.unit, body.template)
    source_links = view.unit(body.unit).material.interfaces['links']
    wanted = [region['copy'], *region['statements']]
    segments = []
    for step_id in context.regions[body.region].steps:
        instruction = context.steps[step_id].instruction
        if type(instruction) is not P.ApplyRegion:
            continue
        suite = context.regions[instruction.invocation.region]
        suite_point = view.source_region(body.unit, body.template, suite.id)
        points = view.expand(suite_point)
        statements = []
        for identity, point in zip(suite.steps, points, strict=True):
            step = context.steps[identity]
            link = source_links.get((context.key, identity))
            if type(step.instruction) is not P.EvaluateChild or link is None:
                statements.append(None)
                continue
            occurrence = link.callee.occurrence
            _, span, _ = _metadata(syntax, occurrence)
            coordinates = [span[key] for key in ('line', 'column', 'end_line', 'end_column')]
            statements.append(dict(span=coordinates, point=encode(point),
                original_step=encode(identity), child_template=encode(link.callee)))
        for start in range(len(statements) - len(wanted) + 1):
            segment = statements[start:start+len(wanted)]
            if all(item is not None and item['span'] == span for item, span in zip(segment, wanted)):
                next_step = start+len(wanted)
                selected_steps = [context.steps[step_id] for step_id in suite.steps[start:next_step]]
                worlds = [context.worlds(step) for step in selected_steps]
                _need(all(a[1] == b[0] for a, b in zip(worlds, worlds[1:])),
                      'CSR source steps do not form one consecutive original world chain')
                successor = encode(points[next_step]) if next_step < len(points) else None
                if successor is not None:
                    following = context.steps[suite.steps[next_step]]
                    live_worlds = [context.value(value) for value in following.arguments
                                   if _state(context.emitter.value(value)[1])]
                    _need(live_worlds == [worlds[-1][1]], 'CSR successor does not consume the original world exit')
                segments.append(dict(original_suite=encode(suite_point), copy=segment[0],
                    original_steps=segment[1:], next_point=successor,
                    input_world=encode(view.source_port(worlds[1][0])),
                    normal_output_world=encode(view.source_port(worlds[-1][1]))))
    _need(len(segments) == 1, 'CSR interval lacks one contiguous original F4 suite')
    return segments[0]


def discover(source_graph):
    """Admit a matching body from the original dispatch evidence, with no workload selector."""
    calls = read_artifact(source_graph)
    program = read_artifact(calls['program'])
    entries = dict()
    for row in calls['observed_targets']:
        code = row['observation']['target_code']
        if row['entry_scope_reachable'] and code is not None:
            entries[code['filename'], code['key']] = code
    candidates = []
    for (filename, key), code in entries.items():
        if not Path(filename).is_file() or '.' in code['qualname']:
            continue
        reference = artifact_reference(filename, 'scar.python-source')
        try:
            region, _ = recognize(read_artifact(reference, json_document=False), filename, key)
        except (ValueError, SyntaxError):
            continue
        candidates.append((reference, region))
    with BuildSession(Path(calls['program']['uri']).parent) as session, _admit_entry_calls(session, calls) as admission:
        proofs = []
        for source, region in candidates:
            witness = _derive_entry_source_witness_admitted(session, admission,
                path=source['uri'], sha256=source['sha256'], code_key=region['code_key'])
            if witness['witnessed']:
                proofs.append(dict(schema=PROOF, schema_version=1, source=source,
                    entry=artifact_reference(program['entry'], 'scar.python-source'),
                    region=region, source_graph=source_graph, original_body=witness,
                    source_binding=_original_suite(session, witness, region),
                    applicability='CONDITIONAL_PRIVATE_HOST_CSR_NORMAL_COMPLETION'))
    target = program['target']
    hardware = artifact_reference(target['path'], 'scar.target-hardware-graph')
    _need((hardware['sha256'], hardware['size_bytes']) == (target['sha256'], target['bytes']),
          'entry target graph differs')
    return proofs, hardware


def _planning_problem(proof, target, cost, measurements):
    proof_fact = dict(artifact=proof, locator='#/source_binding')
    source = (proof_fact,)
    lane = 'calling-cpu-thread'
    operation = 'csr-row-update:'+proof['sha256']
    options = []
    activities, constraints = (), ()
    for mode, name in (('KEEP', 'original'), (FAMILY, 'native')):
        if cost is None:
            bounds = CostEstimate(None, None, None, 'UNKNOWN', operation, source,
                ('MISSING_SAME_CONTEXT_REGION_AND_LOAD_COST',))
        else:
            measured = measurements['timings'][name]
            cost_source = (dict(artifact=cost, locator='#/timings/'+name),)
            bounds = CostEstimate(measured['estimate_ns'], measured['lower_ns'], measured['upper_ns'],
                'EMPIRICAL_MODEL', cost['sha256'], cost_source)
        options.append(ImplementationOption(mode, mode, source, {lane: 1}, {}, {}, {}, bounds,
            dict(memory_domain='HOST', serial_thread=True, guarded=mode != 'KEEP',
                 observed_caller_cpu_ids=None if cost is None else measurements['observations']['cpu_ids'],
                 resource_evidence=None if cost is None else dict(artifact=cost, locator='#/resource_demands'))))
    if cost is not None:
        load_ns = measurements['setup']['load_ns']
        load_ref = (dict(artifact=cost, locator='#/setup/load_ns'),)
        condition = (ImplementationChoice(operation, FAMILY),)
        activation = 'load-prebuilt-csr-provider'
        activities = (PlanningActivity(activation, CostEstimate(load_ns, None, None, 'EMPIRICAL_MODEL',
            cost['sha256'], load_ref), {lane: 1}, load_ref, condition),)
        constraints = (PlanningTimingConstraint('before-csr-call', activation, 'END', operation, 'START',
            when=condition, source_refs=load_ref),)
    return PlanningProblem(decode(ProviderArtifact, target), source,
        (PlanningResource(lane, 'EXECUTION', 1, source),),
        (PlanningOperation(operation, source, (), tuple(options), 'KEEP'),), (),
        assumptions=('ONE_COLD_CALL_WITH_PREBUILT_NATIVE', 'MEASURED_SAME_HOST_REGION_WITH_COPY',
                     'HOST_PAYLOAD_OBSERVED_BUT_PROCESS_MEMORY_CAPACITY_UNQUALIFIED',
                     'REMAINING_ENTRY_COST_AND_NUMA_PLACEMENT_UNQUALIFIED'),
        activities=activities, timing_constraints=constraints)


def prepare(calls, native, out, *, cost=None):
    source_graph = artifact_reference(Path(calls).resolve(), CALLS)
    proofs, target = discover(source_graph)
    _need(len(proofs) == 1, 'one admitted CSR subregion required for this single-region plan')
    proof, = proofs
    output = Path(out).resolve()
    if native is None:
        _need(cost is None, 'prebuilt CSR artifact required for retained cost evidence')
        from scar.codegen.sparse_csr import build_csr_enforce

        native = output/'native'
        build_csr_enforce(native)
    native_ref = artifact_reference(Path(native).resolve()/'artifact.json', NATIVE)
    build = read_artifact(native_ref)
    adapter = Path(__file__).resolve().parents[1]/'runtime/sparse_csr.py'
    members = [build['source'], build['library'], artifact_reference(adapter, 'scar.python-source')]
    for ref in members:
        read_artifact(ref, json_document=False)
    measurements = None
    if cost is not None:
        from scar.optimization.sparse_row_cost import check_sparse_row_cost

        cost = artifact_reference(Path(cost).resolve(), COST)
        measurements = check_sparse_row_cost(cost)
        span = measurements['source_span']
        _need(measurements['schema_version'] == 2
              and measurements.get('adapter_source') == members[-1]
              and measurements['source_sha256'] == proof['source']['sha256']
              and measurements['original_observation']['source']['sha256'] == proof['entry']['sha256']
              and (span['first_line'], span['last_line']) == (proof['region']['statements'][0][0],
                                                            proof['region']['statements'][-1][2])
              and measurements['native_artifact'] == native_ref
              and measurements['family'] == FAMILY and measurements['bitwise_output_equal'],
              'measured CSR region differs from the admitted source or native build')
    output.mkdir(parents=True, exist_ok=True)
    proof_ref = _write(output, 'proof.json', proof)
    problem = _planning_problem(proof_ref, target, cost, measurements)
    problem_ref = _write(output, 'problem.json', problem_document(problem))
    plan = construct_implementations(problem)
    plan_ref = _write(output, 'plan.json', plan)
    reason = ('PREBUILT_COLD_SCOPE_COST_PREFERS_ORIGINAL'
              if cost is not None and plan['selected'] and plan['selected'][0]['option'] == 'KEEP'
              else 'WHOLE_ENTRY_DEPLOYMENT_NOT_QUALIFIED')
    return _write(output, 'selection.json', dict(schema=SELECTION, schema_version=1,
        proof=proof_ref, source=proof['source'], entry=proof['entry'], native=native_ref,
        native_members=members, cost=cost, problem=problem_ref, plan=plan_ref,
        decision='KEEP', reason=reason, explicit_study_only=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('calls', 'out'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--native')
    parser.add_argument('--cost')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.calls, args.native, args.out, cost=args.cost)))


if __name__ == '__main__':
    main()
