"""Connect a discovered ATen region to hardware planning and guarded application validation."""

import argparse
import json
from math import prod
import os
from pathlib import Path

from scar.codegen.aten_native_graph import compile_aten_native_graph
from scar.codegen.aten_python_trial import build_aten_python_trial
from scar.compiler.aten_cpu_domain_plan import plan_aten_cpu_domain
from scar.compiler.aten_region_inputs import read_aten_region_inputs
from scar.compiler.plan_aten_native_graph import plan_aten_native_graph
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.aten_cpu_placement_domain import aten_cpu_placement_domain
from scar.optimization.aten_native_schedule import (
    CONTRACT_SCHEMA, AtenCpuWindowFrontier, propose_cpu_windows, resolve_aten_cpu_windows,
)
from scar.runtime.validate_aten_python_entry import validate_aten_python_entry


def _calibration_inputs(named_inputs, source, recipe):
    supplied = {}
    for item in named_inputs:
        name, separator, file = item.partition('=')
        if not separator or not name or not file or name in supplied:
            raise ValueError('provide one distinct calibration input as NAME=FILE per source Tensor parameter')
        supplied[name] = Path(file).resolve(strict=True)
    formals = [source['parameters'][argument] for argument in source['input_order']]
    if set(supplied) != set(formals):
        raise ValueError(f'calibration inputs must name exactly these original Tensor parameters: {formals}')
    for formal, spec in zip(formals, recipe['inputs'], strict=True):
        file = supplied[formal]
        if not file.is_file() or file.stat().st_size != 4 * prod(spec['shape']):
            raise ValueError(f'calibration file does not match the float32 Tensor slot for {formal}')
    return [(formal, str(supplied[formal])) for formal in formals]


def _compile(inputs, *, out, named_inputs, concurrency_contract, concurrency_sha256,
             geometry_contract, geometry_contract_sha256, tensor_budget_bytes,
             graph_warmup_pairs, graph_measured_pairs, entry_function, entry_args, entry_warmup_pairs,
             entry_measured_pairs, allow_ambient_grad, expected_native_calls):
    report = dict(schema='scar.aten-entry-delivery', schema_version=1, discovery=inputs['discovery'],
                  recommendation='KEEP_ORIGINAL', automatic_deployment_authorized=False)
    if inputs['status'] == 'KEEP_ORIGINAL':
        return dict(report, reason='DISCOVERY_KEPT_ORIGINAL', discovery_reason=inputs.get('reason'))
    if concurrency_contract is None:
        return dict(report, reason='CONCURRENCY_CONTRACT_REQUIRED')

    source, target = inputs['source'], inputs['target']
    recipe = read_artifact(inputs['recipe'])
    concurrency = artifact_reference(Path(concurrency_contract).resolve(strict=True), CONTRACT_SCHEMA)
    if concurrency['sha256'] != concurrency_sha256:
        raise ValueError('concurrency declaration differs from its pinned digest')
    target_dir = Path(target['uri']).parent
    report.update(source=source['source'], function=source['function'], code_key=source['code_key'],
                  target=target, concurrency_contract=concurrency, recipe=inputs['recipe'])
    declarations = read_artifact(concurrency)
    allowed = os.sched_getaffinity(0)
    domain = None
    try:
        placement = resolve_aten_cpu_windows(recipe, contract=declarations, target=target_dir, allowed_cpu_ids=allowed)
    except AtenCpuWindowFrontier as error:
        if error.reason != 'TARGET_CPU_DOMAIN_NOT_EXACT_PAIR':
            return dict(report, reason=error.reason)
        try:
            domain = aten_cpu_placement_domain(target_dir, allowed_cpu_ids=allowed)
        except AtenCpuWindowFrontier as domain_error:
            return dict(report, reason=domain_error.reason)
        if not domain['representatives']:
            return dict(report, reason=domain['reason'], cpu_domain=domain)
        placement = propose_cpu_windows(recipe, contract=declarations, target=target_dir,
                                        cpu_ids=domain['representatives'][0]['cpus'])
    if placement['target'] != target:
        raise ValueError('resolved hardware differs from the original F4 target')
    ordered_inputs = _calibration_inputs(named_inputs, source, recipe)
    report['calibration_order'] = [name for name, _ in ordered_inputs]
    sample_paths = [path for _, path in ordered_inputs]

    def compile_graph(destination, schedule):
        result = compile_aten_native_graph(inputs['recipe']['uri'], recipe_sha256=inputs['recipe']['sha256'],
            linkage=inputs['linkage']['uri'], linkage_sha256=inputs['linkage']['sha256'], out=destination,
            concurrency_contract=concurrency['uri'], concurrency_sha256=concurrency['sha256'],
            target=target_dir, cpu_ids=schedule['cpus'])
        if result['cpu_windows'] != schedule:
            raise ValueError('compiled schedule differs from the resolved placement')
        return result

    native = out / ('native' if domain is None else 'cpu-probe')
    compiled = compile_graph(native, placement)
    primitive_observation = None
    if domain is not None:
        probe = artifact_reference(native / 'artifact.json', compiled['schema'])
        choices, chosen = plan_aten_cpu_domain(probe, domain=domain, input_paths=sample_paths,
            warmup_pairs=graph_warmup_pairs, tensor_budget_bytes=tensor_budget_bytes)
        path = out / 'cpu-domain-plan.json'
        path.write_text(json.dumps(choices, sort_keys=True, indent=2, allow_nan=False) + '\n')
        report['cpu_domain_plan'] = artifact_reference(path, choices['schema'])
        if choices['selected_mode'] == 'SERIAL':
            return dict(report, reason='CPU_DOMAIN_MODEL_KEPT_SERIAL')
        if choices['selected_cpu_ids'] != placement['cpus']:
            placement = propose_cpu_windows(recipe, contract=declarations, target=target_dir,
                                            cpu_ids=choices['selected_cpu_ids'])
            native = out / 'native'
            delivered = compile_graph(native, placement)
            if any(delivered['files'][name] != compiled['files'][name]
                   for name in ('graph.hpp', 'operators.hpp', 'pair.cpp', 'aten_native_pair_v1.h')):
                raise ValueError('delivery graph code differs from the measured CPU probe')
            compiled = delivered
        primitive_observation = dict(inputs=choices['inputs'], raw=chosen['primitive_observations'])

    report['native'] = artifact_reference(native / 'artifact.json', compiled['schema'])
    plan = out / 'plan'
    selection = plan_aten_native_graph(report['native']['uri'], bundle_sha256=report['native']['sha256'],
        input_paths=sample_paths, out=plan, warmup_pairs=graph_warmup_pairs,
        measured_pairs=graph_measured_pairs, tensor_budget_bytes=tensor_budget_bytes,
        primitive_observation=primitive_observation)
    report['selection'] = artifact_reference(plan / 'selection.json', selection['schema'])
    report['native_selection'] = selection['selected']
    if selection['selected'] != 'TWO_CPU':
        return dict(report, reason='NATIVE_PLAN_KEPT_SERIAL')

    trial_dir = out / 'trial'
    projection, contract = inputs['projection'], inputs['wrapper_contract']
    trial = build_aten_python_trial(source['source']['path'], source['function'], projection=projection['uri'],
        projection_sha256=projection['sha256'], selection=report['selection']['uri'],
        selection_sha256=report['selection']['sha256'], contract=contract['uri'], contract_sha256=contract['sha256'],
        geometry_contract=geometry_contract, geometry_contract_sha256=geometry_contract_sha256, out=trial_dir)
    if trial['source_proof'] != source:
        raise ValueError('trial source proof differs from the replayed discovery')
    report['trial'] = artifact_reference(trial_dir / 'trial.json', trial['schema'])
    if entry_function is None:
        return dict(report, reason='GUARDED_TRIAL_REQUIRES_ENTRY_VALIDATION')

    verified = out / 'entry-validation'
    validation = validate_aten_python_entry(trial_dir, trial_sha256=report['trial']['sha256'],
        entry_function=entry_function, entry_args=entry_args, out=verified, warmup_pairs=entry_warmup_pairs,
        measured_pairs=entry_measured_pairs, allow_ambient_grad=allow_ambient_grad,
        require_full_coverage=True, expected_native_calls=expected_native_calls)
    return dict(report, recommendation=validation['recommendation'], speedups=validation['speedups'],
                entry_validation_cpu_context=selection['cpu_ids'],
                entry_validation=artifact_reference(verified / 'validation.json', validation['schema']))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-aten-entry', description=__doc__)
    parser.add_argument('--discovery', required=True)
    parser.add_argument('--discovery-sha256', required=True)
    parser.add_argument('--concurrency-contract')
    parser.add_argument('--concurrency-sha256')
    parser.add_argument('--geometry-contract')
    parser.add_argument('--geometry-contract-sha256')
    parser.add_argument('--input', action='append', default=[], help='calibration Tensor as original parameter NAME=FILE')
    parser.add_argument('--tensor-budget-bytes', type=int)
    parser.add_argument('--graph-warmup-pairs', type=int, default=7)
    parser.add_argument('--graph-measured-pairs', type=int, default=31)
    parser.add_argument('--entry-function', help='validate the original callable in paired fresh processes')
    parser.add_argument('--entry-warmup-pairs', type=int, default=1)
    parser.add_argument('--entry-measured-pairs', type=int, default=3)
    parser.add_argument('--allow-ambient-grad', action='store_true')
    parser.add_argument('--expected-native-calls', type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('entry_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.entry_args[:1] == ['--']:
        args.entry_args.pop(0)
    if bool(args.concurrency_contract) != bool(args.concurrency_sha256):
        parser.error('--concurrency-contract and --concurrency-sha256 must be provided together')
    if bool(args.geometry_contract) != bool(args.geometry_contract_sha256):
        parser.error('--geometry-contract and --geometry-contract-sha256 must be provided together')
    if args.entry_function is None and (args.entry_args or args.expected_native_calls is not None or args.allow_ambient_grad):
        parser.error('entry arguments and coverage options require --entry-function')
    out = Path(args.out).resolve()
    inputs = read_aten_region_inputs(args.discovery, discovery_sha256=args.discovery_sha256)
    out.mkdir(parents=True)
    report = _compile(inputs, out=out, named_inputs=args.input, concurrency_contract=args.concurrency_contract,
        concurrency_sha256=args.concurrency_sha256, geometry_contract=args.geometry_contract,
        geometry_contract_sha256=args.geometry_contract_sha256, tensor_budget_bytes=args.tensor_budget_bytes,
        graph_warmup_pairs=args.graph_warmup_pairs, graph_measured_pairs=args.graph_measured_pairs,
        entry_function=args.entry_function, entry_args=args.entry_args, entry_warmup_pairs=args.entry_warmup_pairs,
        entry_measured_pairs=args.entry_measured_pairs, allow_ambient_grad=args.allow_ambient_grad,
        expected_native_calls=args.expected_native_calls)
    path = out / 'delivery.json'
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(path), recommendation=report['recommendation'], reason=report.get('reason'))))
    return 0
