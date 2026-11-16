"""L2 source-to-resource/cost projection, without solver or native execution."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact

INPUTS = 'scar.pipeline-temporal-inputs'
SCHEMA = 'scar.pipeline-temporal-preparation'


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _provider(session, name, unit=None):
    from scar.analysis.source_call_view import SourceFunctionCallProvider
    matches = [(key, value) for key, value in session.session.providers.items()
        if type(value) is SourceFunctionCallProvider
        and (name is None or key == name)
        and (unit is None or unit in value.available_units)]
    if session.report.get('backend') != 'F4' or len(matches) != 1:
        raise ValueError('one original F4 source-call provider required')
    return matches[0]


def _hardware(session, ready=None):
    target = session.report.get('target')
    if target is None:
        raise ValueError('original program hardware snapshot required')
    if ready is not None and (str(Path(target['path']).resolve()), target['sha256'], target['bytes']) != (
            ready['target']['uri'], ready['target']['sha256'], ready['target']['size_bytes']):
        raise ValueError('ready plan differs from original program hardware')
    return Path(target['path']).parent


@contextmanager
def open_temporal_inputs(reference):
    """Replay original source/value routes, pool and CPU plan before L2 use."""
    from scar.analysis.build import BuildSession
    from scar.analysis.caller_pipeline_values import read_caller_values
    from scar.compiler.caller_pipeline import read_caller_pipeline_proof
    from scar.optimization.pipeline_memory import verify_pipeline_memory
    from scar.optimization.pipeline_ready_schedule import verify_pipeline_ready_schedule
    if isinstance(reference, (str, Path)):
        path = Path(reference).resolve()
        reference = artifact_reference(path/'inputs.json' if path.is_dir() else path, INPUTS)
    raw = read_artifact(reference)
    if (raw.get('schema'), raw.get('schema_version')) != (INPUTS, 1):
        raise ValueError('original temporal input publication required')
    for field in ('program', 'proof', 'source_values', 'memory', 'ready'):
        read_artifact(raw[field])
    manifest = Path(raw['program']['uri']).parent/'manifest.json'
    if raw['program'] != artifact_reference(manifest, 'scar.program-build'):
        raise ValueError('program reference must identify the original build manifest')
    with BuildSession(Path(raw['program']['uri']).parent) as session:
        _, provider = _provider(session, raw['provider'])
        proof = read_caller_pipeline_proof(raw['proof']['uri'], view=session.view, provider=provider)
        source_values = read_artifact(raw['source_values'])
        if source_values['original_proof'] != raw['proof']:
            raise ValueError('source value publication differs from selected proof')
        read_caller_values(raw['source_values']['uri'], view=session.view, provider=provider)
        memory = verify_pipeline_memory(proof.pipeline, read_artifact(raw['memory']))
        if source_values['input_shapes'] != memory['input_shapes']:
            raise ValueError('source value and memory specialization shapes differ')
        ready = read_artifact(raw['ready'])
        if ready['input_graph'] != raw['source_values']:
            raise ValueError('ready input graph differs from original source values')
        target = _hardware(session, ready)
        verify_pipeline_ready_schedule(proof.pipeline, memory, target, ready)
        yield proof.pipeline, memory, ready, target
        read_artifact(reference)


def _project(reference, out, cost_model):
    from scar.optimization.pipeline_temporal_model import project_pipeline_temporal_model
    with open_temporal_inputs(reference) as (pipeline, memory, ready, target):
        projection = project_pipeline_temporal_model(pipeline, memory, ready, target, cost_model=cost_model)
        projected = _write(out/'projection.json', projection)
    report = dict(schema=SCHEMA, schema_version=1, inputs=reference, projection=projected,
        layer='L2_COST_AND_CONSTRAINT_PROJECTION', solver_invoked=False,
        native_compiled=False, target_program_executed=False, performance_measured=False,
        deployment_authorized=False)
    _write(out/'result.json', report)
    return report


def prepare_pipeline_temporal(program, out, *, caller_unit, function_ordinal,
        input_shapes, memory_budget_bytes, pipeline_workers, provider_name=None,
        interface_ordinals=None, memory_order='dependencies', cost_model=None):
    from scar.analysis.build import BuildSession
    from scar.analysis.caller_pipeline_values import publish_caller_values
    from scar.compiler.caller_pipeline import build_caller_pipeline, write_caller_pipeline_proof
    from scar.optimization.hardware import propose_cpu_placements
    from scar.optimization.pipeline_ready_schedule import plan_pipeline_ready_schedule
    if type(pipeline_workers) is not int or pipeline_workers < 1:
        raise ValueError('explicit positive CPU lane count required')
    if memory_order not in ('source', 'dependencies'):
        raise ValueError('source or dependencies memory order required')
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh L2 output directory required')
    program = Path(program).resolve()
    program = program.parent if program.is_file() else program
    with BuildSession(program) as session:
        name, provider = _provider(session, provider_name, caller_unit)
        target = _hardware(session)
        proof = build_caller_pipeline(session.view, provider, caller_unit=caller_unit,
            function_ordinal=function_ordinal, interface_ordinals=interface_ordinals)
        out.mkdir(parents=True)
        proof_ref = write_caller_pipeline_proof(proof, out/'proof.json')
        values, proof = publish_caller_values(proof_ref, out/'source-values',
            view=session.view, provider=provider, proof=proof, input_shapes=input_shapes)
        if memory_order == 'dependencies':
            from scar.optimization.pipeline_partial_order_memory import plan_pipeline_partial_order_memory as planner
        else:
            from scar.optimization.pipeline_memory import plan_pipeline_memory as planner
        memory = planner(proof.pipeline, input_shapes=input_shapes, memory_budget_bytes=memory_budget_bytes)
        placements = propose_cpu_placements(proof.pipeline.calls[0].plan, target,
            memory_budget_bytes=memory_budget_bytes, max_threads=pipeline_workers)
        placement = next((row for row in placements if len(row.cpu_ids) == pipeline_workers), None)
        if placement is None:
            raise ValueError('requested CPU lanes unavailable in original hardware snapshot')
        values_ref = artifact_reference(out/'source-values/result.json', values['schema'])
        ready = plan_pipeline_ready_schedule(proof.pipeline, memory, target,
            cpu_ids=placement.cpu_ids, input_graph=values_ref)
        inputs = dict(schema=INPUTS, schema_version=1,
            program=artifact_reference(program/'manifest.json', 'scar.program-build'),
            provider=name, proof=proof_ref, source_values=values_ref,
            memory=_write(out/'memory.json', memory), ready=_write(out/'ready.json', ready))
        reference = _write(out/'inputs.json', inputs)
    return _project(reference, out, cost_model)


def reproject_pipeline_temporal(inputs, out, *, cost_model=None):
    path = Path(inputs).resolve()
    reference = artifact_reference(path/'inputs.json' if path.is_dir() else path, INPUTS)
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh L2 output directory required')
    out.mkdir(parents=True)
    return _project(reference, out, cost_model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--build'); origin.add_argument('--from-inputs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--unit'); parser.add_argument('--function-ordinal', type=int)
    parser.add_argument('--provider'); parser.add_argument('--interface-ordinal', type=int, action='append')
    parser.add_argument('--input-shapes', type=json.loads)
    parser.add_argument('--memory-budget-bytes', type=int); parser.add_argument('--pipeline-workers', type=int)
    parser.add_argument('--pipeline-memory-order', choices=('source', 'dependencies'))
    parser.add_argument('--cost-model')
    args = parser.parse_args(argv)
    if args.build:
        if any(value is None for value in (args.unit, args.function_ordinal, args.input_shapes,
                args.memory_budget_bytes, args.pipeline_workers)):
            parser.error('new source preparation requires unit, function, shapes, budget and CPU lanes')
        result = prepare_pipeline_temporal(args.build, args.out, caller_unit=args.unit,
            function_ordinal=args.function_ordinal, provider_name=args.provider,
            interface_ordinals=args.interface_ordinal, input_shapes=args.input_shapes,
            memory_budget_bytes=args.memory_budget_bytes, pipeline_workers=args.pipeline_workers,
            memory_order=args.pipeline_memory_order or 'dependencies', cost_model=args.cost_model)
    else:
        if any(value is not None for value in (args.unit, args.function_ordinal, args.provider,
                args.interface_ordinal, args.input_shapes, args.memory_budget_bytes, args.pipeline_workers,
                args.pipeline_memory_order)):
            parser.error('retained inputs already fix the original source and resource context')
        result = reproject_pipeline_temporal(args.from_inputs, args.out, cost_model=args.cost_model)
    print(json.dumps(dict(layer=result['layer'], result=str(Path(args.out)/'result.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
