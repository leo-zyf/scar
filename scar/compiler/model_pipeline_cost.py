"""Cost a guarded source region or its published successor without executing it."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN

SCHEMA = 'scar.pipeline-forward-model-build'
_MACRO_PARENTS = ('region', 'blueprint')


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
        'optimization/deployment_cost.py', 'compiler/semantic_revision.py',
        'optimization/semantic_region_demand.py')
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in files}


def _derive(inputs, preparation, coefficients, cpus, budget, gain, *, strict_minimum_gain=True):
    from scar.optimization.pipeline_demand import (
        derive_pipeline_demand, derive_successor_pipeline_cost_demand,
    )
    from scar.optimization.pipeline_forward_cost import (
        coefficient_profile_skeleton, predict_pipeline_forward_cost,
    )
    macro_context = None
    if preparation is None:
        from scar.compiler.macro_pipeline import _derive as derive_source

        cpus = sorted(os.sched_getaffinity(0)) if cpus is None else cpus
        gain = DEFAULT_MINIMUM_GAIN if gain is None else gain
        pipeline, region, blueprint, _, budget, _ = derive_source(inputs, None, cpus, budget, gain,
            strict_minimum_gain=strict_minimum_gain)
        demand = derive_pipeline_demand(pipeline, blueprint)
    else:
        from scar.compiler.macro_pipeline import check as check_macro

        macro_context = check_macro(preparation)
        parent, _, region, blueprint, _ = macro_context
        if any(requested is not None and requested != parent[field] for field, requested in (
                ('available_cpu_ids', cpus), ('memory_budget_bytes', budget), ('minimum_gain', gain))):
            raise ValueError('successor costing must retain the macro preparation CPU domain, budget and gain')
        inputs = parent['inputs']
        cpus, budget, gain = (parent[field] for field in ('available_cpu_ids', 'memory_budget_bytes', 'minimum_gain'))
        demand = derive_successor_pipeline_cost_demand(region, blueprint, preparation)
    template = coefficient_profile_skeleton(region, blueprint, demand)
    profile = template if coefficients is None else read_artifact(coefficients)
    prediction = predict_pipeline_forward_cost(region, blueprint, demand, profile)
    documents = _json(dict(region=region, blueprint=blueprint, demand=demand,
        coefficient_template=template, prediction=prediction))
    request = dict(available_cpu_ids=list(cpus), memory_budget_bytes=budget, minimum_gain=gain)
    if strict_minimum_gain:
        request['strict_minimum_gain'] = True
    return documents, inputs, request, macro_context


def _metadata(inputs, preparation, coefficients, request, files, decision, demand):
    successor = (dict(macro_preparation=preparation, candidate_binding=demand['candidate_binding'])
        if preparation is not None else {})
    return dict(schema=SCHEMA, schema_version=2 if successor else 1,
        inputs=inputs, coefficient_profile=coefficients, **request,
        files=files, decision=decision, producer_pins=_pins(),
        original_F4_source_and_value_routes_replayed=True,
        original_python_preserved_when_unknown=True,
        complete_scope='PROTECTED_SOURCE_REGION_CALL_ENTRY_TO_RETURN_WARM',
        source_demand_is_not_physical_DRAM_traffic=True,
        predicted_cost_basis='DECLARED_MODEL_NOT_OBSERVED_TOTAL',
        target_program_executed=False, target_versions_benchmarked=False,
        solver_invoked=False, native_compiled=False, native_loaded=False,
        deployment_authorized=False, all_three_stages_complete=False, **successor)


def _deployment_profile(documents, files, coefficients):
    from scar.optimization.pipeline_forward_cost import deployment_profile_for_prediction

    return deployment_profile_for_prediction(files['prediction'],
        region_document=documents['region'], blueprint=documents['blueprint'], demand=documents['demand'],
        coefficient_profile=documents['coefficient_template'] if coefficients is None else read_artifact(coefficients))


def _pass_report(documents, files, request):
    from scar.optimization.macro_pass import run_fusion_pass

    settings = dict(request)
    settings.setdefault('strict_minimum_gain', False)
    report = _json(run_fusion_pass(documents['region'], documents['blueprint'],
        cost_profile=files['deployment_profile']['uri'], **settings))
    binding = documents['demand'].get('candidate_binding')
    if binding is not None and report['decision'] == 'REWRITE':
        revision = read_artifact(binding['candidate_revision'])
        selection, = report['optimization_selection']['optimization']['plans']
        if ({key:value for key, value in selection.items() if key != 'reason'}
                != {key:value for key, value in revision['selection'].items() if key != 'reason'}
                or report['revision']['delta_summary'] != revision['delta']):
            raise ValueError('cost decision selects a different rewrite from the published candidate')
    return report


def prepare(inputs=None, out=None, *, macro_preparation=None, coefficient_profile=None,
            available_cpu_ids=None, memory_budget_bytes=None, minimum_gain=None):
    from scar.compiler.macro_pipeline import SCHEMA as PREPARATION_SCHEMA
    from scar.compiler.project_pipeline_temporal import INPUTS

    if (inputs is None) == (macro_preparation is None) or out is None:
        raise ValueError('one source input or macro preparation and an output directory required')
    reference = None if inputs is None else _reference(inputs, INPUTS)
    preparation = None if macro_preparation is None else _reference(macro_preparation, PREPARATION_SCHEMA)
    coefficients = (None if coefficient_profile is None else
        _reference(coefficient_profile, 'scar.pipeline-forward-coefficient-profile'))
    cpus = None if available_cpu_ids is None else list(available_cpu_ids)
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh forward model output required')
    documents, reference, request, macro_context = _derive(reference, preparation, coefficients, cpus,
        memory_budget_bytes, minimum_gain)
    output.mkdir(parents=True)
    parents = {} if macro_context is None else {name: macro_context[0]['files'][name] for name in _MACRO_PARENTS}
    files = {name: parents[name] if name in parents else _write(output/(name+'.json'), value)
        for name, value in documents.items()}
    profile = _deployment_profile(documents, files, coefficients)
    files['deployment_profile'] = _write(output/'profile.json', profile)
    report = _pass_report(documents, files, request)
    files['pass_report'] = _write(output/'pass.json', report)
    result = _metadata(reference, preparation, coefficients, request, files, report['decision'], documents['demand'])
    _write(output/'result.json', result)
    return result


def check(result, *, include_macro_context=False):
    """Cold-replay the selected input mode and optionally return its macro context."""
    reference = _reference(result, SCHEMA)
    saved = read_artifact(reference)
    preparation = saved.get('macro_preparation')
    if (saved.get('schema_version') != (2 if preparation is not None else 1)
            or saved.get('producer_pins') != _pins()):
        raise ValueError('forward model producer differs')
    documents, inputs, request, macro_context = _derive(saved['inputs'], preparation, saved['coefficient_profile'],
        saved['available_cpu_ids'], saved['memory_budget_bytes'], saved['minimum_gain'],
        strict_minimum_gain=saved.get('strict_minimum_gain', False))
    if macro_context is not None and any(saved['files'][name] != macro_context[0]['files'][name]
            for name in _MACRO_PARENTS):
        raise ValueError('forward model parents differ from the published macro region and blueprint')
    if any(read_artifact(saved['files'][name]) != value for name, value in documents.items()):
        raise ValueError('forward model differs from original source/demand/coefficient replay')
    profile = _deployment_profile(documents, saved['files'], saved['coefficient_profile'])
    if read_artifact(saved['files']['deployment_profile']) != _json(profile):
        raise ValueError('derived deployment profile differs')
    report = _pass_report(documents, saved['files'], request)
    if read_artifact(saved['files']['pass_report']) != _json(report):
        raise ValueError('forward pass differs from original cost and OIR replay')
    expected = _metadata(inputs, preparation, saved['coefficient_profile'], request,
        saved['files'], report['decision'], documents['demand'])
    if saved != expected:
        raise ValueError('forward model summary differs')
    return (saved, macro_context) if include_macro_context else saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--inputs')
    source.add_argument('--macro-preparation', help='cost the exact published semantic candidate and its lowering')
    parser.add_argument('--out')
    parser.add_argument('--coefficient-profile')
    parser.add_argument('--cpu-ids', help='cost-applicable caller affinity domain, not CPU pinning')
    parser.add_argument('--memory-budget-bytes', type=int)
    parser.add_argument('--minimum-gain', type=float)
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        saved = check(args.check)
        result = dict(source_demand_and_model_replayed=True, decision=saved['decision'])
    elif not (args.inputs or args.macro_preparation) or not args.out:
        parser.error('--inputs or --macro-preparation, and --out required')
    else:
        result = prepare(args.inputs, args.out, macro_preparation=args.macro_preparation,
            coefficient_profile=args.coefficient_profile,
            available_cpu_ids=None if args.cpu_ids is None else [int(cpu) for cpu in args.cpu_ids.split(',')],
            memory_budget_bytes=args.memory_budget_bytes, minimum_gain=args.minimum_gain)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
