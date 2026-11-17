"""Source demand to a forward full-entry model, without workload execution."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.pipeline-forward-model-build'


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _reference(value, schema):
    return value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)


def _pins():
    root = Path(__file__).resolve().parents[1]
    files = ('compiler/model_pipeline_cost.py', 'compiler/macro_pipeline.py',
        'compiler/project_pipeline_temporal.py', 'analysis/pipeline_region_pass.py',
        'compiler/pipeline_fusion_delivery.py', 'optimization/macro_pass.py',
        'optimization/pipeline_demand.py', 'optimization/demand.py',
        'optimization/pipeline_forward_cost.py', 'optimization/cost_model.py',
        'optimization/deployment_cost.py')
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in files}


def _derive(inputs, coefficients, cpus, budget, gain):
    from scar.compiler.macro_pipeline import _derive as derive_source
    from scar.optimization.pipeline_demand import derive_pipeline_demand
    from scar.optimization.pipeline_forward_cost import (
        coefficient_profile_skeleton, predict_pipeline_forward_cost,
    )
    pipeline, region, blueprint, _, budget = derive_source(inputs, None, cpus, budget, gain)
    demand = derive_pipeline_demand(pipeline, blueprint)
    template = coefficient_profile_skeleton(region, blueprint, demand)
    profile = template if coefficients is None else read_artifact(coefficients)
    prediction = predict_pipeline_forward_cost(region, blueprint, demand, profile)
    return _json(dict(region=region, blueprint=blueprint, demand=demand,
        coefficient_template=template, prediction=prediction)), budget


def _metadata(inputs, coefficients, cpus, budget, gain, files, decision):
    return dict(schema=SCHEMA, schema_version=1, inputs=inputs, coefficient_profile=coefficients,
        available_cpu_ids=list(cpus), memory_budget_bytes=budget, minimum_gain=gain,
        files=files, decision=decision, producer_pins=_pins(),
        original_F4_source_and_value_routes_replayed=True,
        original_python_preserved_when_unknown=True,
        complete_scope='PROTECTED_SOURCE_REGION_CALL_ENTRY_TO_RETURN_WARM',
        source_demand_is_not_physical_DRAM_traffic=True,
        predicted_cost_basis='DECLARED_MODEL_NOT_OBSERVED_TOTAL',
        target_program_executed=False, target_versions_benchmarked=False,
        solver_invoked=False, native_compiled=False, native_loaded=False,
        deployment_authorized=False, all_three_stages_complete=False)


def prepare(inputs, out, *, coefficient_profile=None, available_cpu_ids=None,
            memory_budget_bytes=None, minimum_gain=.05):
    from scar.compiler.project_pipeline_temporal import INPUTS
    from scar.optimization.pipeline_forward_cost import deployment_profile_for_prediction
    from scar.optimization.macro_pass import run_fusion_pass
    reference = _reference(inputs, INPUTS)
    coefficients = (None if coefficient_profile is None else
        _reference(coefficient_profile, 'scar.pipeline-forward-coefficient-profile'))
    cpus = sorted(os.sched_getaffinity(0)) if available_cpu_ids is None else list(available_cpu_ids)
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh forward model output required')
    documents, budget = _derive(reference, coefficients, cpus, memory_budget_bytes, minimum_gain)
    output.mkdir(parents=True)
    files = {name: _write(output/(name+'.json'), value) for name, value in documents.items()}
    profile = deployment_profile_for_prediction(files['prediction'],
        region_document=documents['region'], blueprint=documents['blueprint'], demand=documents['demand'],
        coefficient_profile=documents['coefficient_template'] if coefficients is None else read_artifact(coefficients))
    files['deployment_profile'] = _write(output/'profile.json', profile)
    report = run_fusion_pass(documents['region'], documents['blueprint'], cost_profile=files['deployment_profile']['uri'],
        available_cpu_ids=cpus, memory_budget_bytes=budget, minimum_gain=minimum_gain)
    files['pass_report'] = _write(output/'pass.json', report)
    result = _metadata(reference, coefficients, cpus, budget, minimum_gain, files, report['decision'])
    _write(output/'result.json', result)
    return result


def check(result):
    from scar.optimization.pipeline_forward_cost import deployment_profile_for_prediction
    from scar.optimization.macro_pass import run_fusion_pass
    reference = _reference(result, SCHEMA)
    saved = read_artifact(reference)
    if saved.get('schema_version') != 1 or saved.get('producer_pins') != _pins():
        raise ValueError('forward model producer differs')
    documents, budget = _derive(saved['inputs'], saved['coefficient_profile'],
        saved['available_cpu_ids'], saved['memory_budget_bytes'], saved['minimum_gain'])
    if any(read_artifact(saved['files'][name]) != value for name, value in documents.items()):
        raise ValueError('forward model differs from original source/demand/coefficient replay')
    coefficients = (documents['coefficient_template'] if saved['coefficient_profile'] is None else
        read_artifact(saved['coefficient_profile']))
    profile = deployment_profile_for_prediction(saved['files']['prediction'],
        region_document=documents['region'], blueprint=documents['blueprint'],
        demand=documents['demand'], coefficient_profile=coefficients)
    if read_artifact(saved['files']['deployment_profile']) != _json(profile):
        raise ValueError('derived deployment profile differs')
    report = run_fusion_pass(documents['region'], documents['blueprint'],
        cost_profile=saved['files']['deployment_profile']['uri'],
        available_cpu_ids=saved['available_cpu_ids'], memory_budget_bytes=budget,
        minimum_gain=saved['minimum_gain'])
    if read_artifact(saved['files']['pass_report']) != _json(report):
        raise ValueError('forward pass differs from original cost and OIR replay')
    expected = _metadata(saved['inputs'], saved['coefficient_profile'], saved['available_cpu_ids'],
        budget, saved['minimum_gain'], saved['files'], report['decision'])
    if saved != expected:
        raise ValueError('forward model summary differs')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs'); parser.add_argument('--out')
    parser.add_argument('--coefficient-profile')
    parser.add_argument('--cpu-ids', help='cost-applicable caller affinity domain, not CPU pinning')
    parser.add_argument('--memory-budget-bytes', type=int)
    parser.add_argument('--minimum-gain', type=float, default=.05)
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        saved = check(args.check)
        result = dict(source_demand_and_model_replayed=True, decision=saved['decision'])
    elif not args.inputs or not args.out:
        parser.error('--inputs and --out required')
    else:
        result = prepare(args.inputs, args.out, coefficient_profile=args.coefficient_profile,
            available_cpu_ids=None if args.cpu_ids is None else [int(cpu) for cpu in args.cpu_ids.split(',')],
            memory_budget_bytes=args.memory_budget_bytes, minimum_gain=args.minimum_gain)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
