"""Replay original caller evidence, jointly select CPU leaves, and optionally compile.

Declared models are examples, not measurements. This profile delivers a
guarded regional C++ artifact; automatic caller installation remains disabled.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import problem_document, problem_from_dict, digest
from scar.compiler.pipeline_model import pipeline_digest
from scar.optimization.pipeline_implementation import (build_pipeline_problem,
    cost_contexts, refine_pipeline_implementation, COST_SCHEMA, CERT_SCHEMA)
from scar.optimization.joint_planner import solve_implementations, verify_execution_plan
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.pipeline-implementation-preparation'


def _write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return artifact_reference(path, document['schema'])


def _reference(value, schema):
    return value if isinstance(value, dict) else artifact_reference(Path(value).resolve(), schema)


def _sources(reference):
    raw = read_artifact(reference)
    return (dict(role='ORIGINAL_F4_CALLER_PIPELINE_INPUTS', artifact=reference),
        dict(role='ORIGINAL_CALL_AND_PURITY_PROOF', artifact=raw['proof']),
        dict(role='ORIGINAL_PUBLIC_VALUE_GRAPH', artifact=raw['source_values']),
        dict(role='ORIGINAL_PHYSICAL_POOL_PLAN', artifact=raw['memory']))


def _producer():
    from scar.optimization import pipeline_implementation, joint_planner, implementation_costs
    from scar.ir import implementation_plan
    return {str(Path(module.__file__).resolve()): sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (pipeline_implementation, joint_planner, implementation_plan, implementation_costs)}


def prepare_pipeline_implementation(inputs, out, *, cpu_ids, cost_model=None, cost_profile_ref=None,
        max_seconds=30, workers=4, compile_native=False):
    from scar.compiler.project_pipeline_temporal import open_temporal_inputs, INPUTS
    reference = _reference(inputs, INPUTS)
    model_ref = None if cost_model is None else _reference(cost_model, COST_SCHEMA)
    from scar.optimization.implementation_costs import SCHEMA as PROFILE_SCHEMA
    profile_ref = None if cost_profile_ref is None else _reference(cost_profile_ref, PROFILE_SCHEMA)
    if model_ref is not None and profile_ref is not None:
        raise ValueError('cost model and cost profile are mutually exclusive')
    model = None if model_ref is None else read_artifact(model_ref)
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh joint pipeline output required')
    with open_temporal_inputs(reference) as (pipeline, memory, _, target):
        problem = build_pipeline_problem(pipeline, memory, target, cpu_ids=cpu_ids,
            cost_model=model, cost_profile_ref=profile_ref, source_refs=_sources(reference))
        execution = solve_implementations(problem, max_seconds=max_seconds, workers=workers)
        certificate = None
        if execution['selected'] and execution['domain_complete']:
            certificate = refine_pipeline_implementation(pipeline, memory, target, problem, execution)
        directory.mkdir(parents=True)
        contexts = dict(schema='scar.pipeline-implementation-cost-contexts', schema_version=1,
            pipeline_sha256=pipeline_digest(pipeline),
            memory_plan_sha256=semantic_sha256(memory), target=problem_document(problem)['target'],
            contexts=cost_contexts(problem), cost_scope='ABI1_LEAF_SERVICE_ONLY_HOST_ENVELOPE_UNCALIBRATED',
            cost_model_template=dict(schema=COST_SCHEMA, schema_version=1,
                pipeline_sha256=pipeline_digest(pipeline),
                memory_plan_sha256=semantic_sha256(memory), target=problem_document(problem)['target'],
                costs={key:dict(context=context, estimate_ns=None, lower_ns=None, upper_ns=None,
                    basis='DECLARED_MODEL', missing_terms=['DECLARE_EXACT_CONTEXT_POINT_ESTIMATE'])
                    for key,context in cost_contexts(problem).items()}))
        # Every file is an original reference; no embedded tensor data is saved.
        report = dict(schema=SCHEMA, schema_version=1, inputs=reference, cpu_ids=list(cpu_ids),
            cost_model=model_ref, cost_profile=profile_ref,
            problem=_write(directory/'problem.json', problem_document(problem)),
            cost_contexts=_write(directory/'cost-contexts.json', contexts),
            execution_plan=_write(directory/'execution-plan.json', execution),
            certificate=None if certificate is None else _write(directory/'certificate.json', certificate),
            producer_pins=_producer(), solver_status=execution['status'],
            implementation_scope='SELECTED_CPU_PLACEMENT_AND_COMPUTE_LANE_ORDER_ONLY',
            cost_scope='ABI1_LEAF_SERVICE_ONLY_HOST_ENVELOPE_UNCALIBRATED',
            model_makespan_is_native_runtime_prediction=False, physical_optimality_proven=False,
            target_program_executed=False, performance_measured=False, deployment_authorized=False,
            default_backend='NO_OP')
        _write(directory/'preparation.json', report)
    result = dict(schema='scar.pipeline-implementation-compiler-build', schema_version=1,
        preparation=artifact_reference(directory/'preparation.json', SCHEMA),
        native_artifact=None, native_compiled=False, default_backend='NO_OP',
        deployment_authorized=False, target_program_executed=False,
        performance_measured=False, whole_program_cpp=False,
        solver_status=execution['status'])
    if compile_native and certificate is not None:
        from scar.codegen.pipeline_implementation import build
        artifact = build(result['preparation'], directory/'native')
        result.update(native_compiled=True,
            native_artifact=artifact_reference(artifact.directory/'artifact.json', artifact.report['schema']),
            execution_family=certificate['execution_family'], call_cpu_ids=certificate['call_cpu_ids'],
            lane_call_order=certificate['lane_call_order'])
    _write(directory/'result.json', result)
    return result


@contextmanager
def open_pipeline_implementation(preparation):
    """Cold source/cost/domain replay, without solving again or loading native code."""
    from scar.compiler.project_pipeline_temporal import open_temporal_inputs
    reference = _reference(preparation, SCHEMA)
    report = read_artifact(reference)
    if report.get('schema_version') != 1 or report['producer_pins'] != _producer():
        raise ValueError('joint pipeline producer implementation differs')
    model = None if report['cost_model'] is None else read_artifact(report['cost_model'])
    saved_problem = problem_from_dict(read_artifact(report['problem']))
    execution = read_artifact(report['execution_plan'])
    with open_temporal_inputs(report['inputs']) as (pipeline, memory, _, target):
        problem = build_pipeline_problem(pipeline, memory, target, cpu_ids=report['cpu_ids'],
            cost_model=model, cost_profile_ref=report['cost_profile'], source_refs=_sources(report['inputs']))
        if digest(problem) != digest(saved_problem):
            raise ValueError('problem differs from original source, cost contexts, target or physical pool')
        contexts = read_artifact(report['cost_contexts'])
        if semantic_sha256(contexts['contexts']) != semantic_sha256(cost_contexts(problem)):
            raise ValueError('cost context publication differs')
        if execution['selected']:
            verify_execution_plan(problem, execution)
        certificate = None
        if report['certificate'] is not None:
            certificate = refine_pipeline_implementation(pipeline, memory, target, problem, execution)
            if semantic_sha256(certificate) != semantic_sha256(read_artifact(report['certificate'])):
                raise ValueError('selected implementation certificate differs from original source/model')
        elif execution['selected'] and execution['domain_complete']:
            raise ValueError('complete feasible selection lost its certificate')
        yield report, pipeline, memory, target, problem, execution, certificate
        read_artifact(reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs'); parser.add_argument('--out')
    parser.add_argument('--cpu-ids', help='explicit comma separated original CPU IDs')
    costs = parser.add_mutually_exclusive_group()
    costs.add_argument('--cost-model'); costs.add_argument('--cost-profile')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check', help='preparation.json or compiler result.json')
    args = parser.parse_args(argv)
    if args.check:
        raw = json.loads(Path(args.check).read_bytes())
        reference = raw['preparation'] if raw.get('schema') == 'scar.pipeline-implementation-compiler-build' else args.check
        with open_pipeline_implementation(reference) as rows:
            result = dict(source_model_replayed=True, solver_status=rows[5]['status'],
                certificate_present=rows[6] is not None, deployment_authorized=False)
        if raw.get('native_artifact') is not None:
            from scar.codegen.pipeline_implementation import load
            load(Path(raw['native_artifact']['uri']).parent)
            result['native_artifact_replayed'] = True
    else:
        if not args.inputs or not args.out or not args.cpu_ids:
            parser.error('--inputs, --out and --cpu-ids required')
        result = prepare_pipeline_implementation(args.inputs, args.out,
            cpu_ids=[int(value) for value in args.cpu_ids.split(',')], cost_model=args.cost_model,
            cost_profile_ref=args.cost_profile,
            max_seconds=args.max_seconds, workers=args.workers, compile_native=args.compile)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
