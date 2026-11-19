"""Replay an exact, source-to-stdout dense versus compact Basis comparison."""
from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from statistics import median, stdev

from scar.codegen import native_publication_delivery as delivery
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_compact_basis, native_fixed_entry_effects


SCHEMA = 'scar.native-compact-basis-observed-choice'
OBSERVATION = 'scar.cpu-source-seed-matched-benchmark'
_MODES = {'original', 'dense', 'compact'}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _excluding(row, *keys):
    return {name: value for name, value in row.items() if name not in keys}


def _local(root, reference):
    path = root / reference['uri']
    _require(path.is_relative_to(root) and path.resolve(strict=True) == path,
             'invalid local comparison package member')
    return read_artifact({**reference, 'uri': str(path)})


def _stage_inputs(dense, compact, dense_doc, compact_doc, proof, proof_ref):
    """Check that both private stage plans differ exactly at the certified port."""
    libraries = set()
    for stage, port, representation in (
            ('mesh', 'exact_basis', 'compact_output_header'),
            ('basis', 'admitted_input', 'dense_input_header')):
        before = _local(dense, dense_doc[stage])
        after = _local(compact, compact_doc[stage])
        _require((before['schema_version'], after['schema_version']) == (1, 2)
            and before[port] == after['dense_' + port] == proof['dense_input']
            and after[port] == proof['compact_input']
            and after['compact_proof'] == proof_ref and after['wire'] == proof['wire'],
            'native arm uses a different physical representation witness: ' + stage)
        before_plan, after_plan = before['plan'], after['plan']
        if stage == 'basis':
            _require(before_plan['input_header'] == after_plan[representation],
                'native compact Basis changed the dense reference header')
            after_plan = dict(after_plan, input_header=after_plan[representation])
        _require(before_plan == _excluding(after_plan, representation),
            'numeric operation order changed with the private wire: ' + stage)
        treatment = ('schema_version', 'source', 'executable', 'plan', 'compact_proof', 'proof_member',
                     'wire', port, 'dense_' + port, 'basis_package', 'value_view')
        _require(_excluding(before, *treatment) == _excluding(after, *treatment),
            'non-wire native stage contract changed: ' + stage)
        libraries.update(f'{stage}/{row["uri"]}' for row in before['libraries'])
    topology = _local(dense, dense_doc['topology'])
    libraries.update(f'topology/{row["uri"]}' for row in topology['libraries'])
    return libraries


def _summary(samples):
    return dict(samples_ns=samples, median_ns=median(samples), min_ns=min(samples), max_ns=max(samples),
                sample_stdev_ns=stdev(samples))


def _shared_providers(dense, compact, paths, required):
    _require(type(paths) is list and all(type(name) is str for name in paths)
        and paths == sorted(set(paths)) and required <= set(paths),
        'benchmark does not record the required native provider sharing')
    digests = {}
    for name in paths:
        pair = []
        for root in (dense, compact):
            path = root / name
            _require(path.is_relative_to(root) and path.resolve(strict=True) == path and path.is_file(),
                     'invalid shared executable or library member: ' + name)
            status = path.stat()
            key = status.st_dev, status.st_ino
            if key not in digests:
                with path.open('rb') as stream:
                    digests[key] = file_digest(stream, 'sha256').hexdigest()
            pair.append((status.st_size, digests[key]))
        _require(pair[0] == pair[1], 'archived shared provider bytes differ: ' + name)


def choose(proof_path, effects_path, dense_path, compact_path, benchmark_path, script_path, *,
           threshold_bps=None):
    proof_ref = _reference(proof_path, native_compact_basis.SCHEMA)
    proof = native_compact_basis.check(proof_path)
    effects_ref = _reference(effects_path, native_fixed_entry_effects.SCHEMA)
    effects = native_fixed_entry_effects.check(effects_path)
    dense, compact = (Path(path).resolve(strict=True) for path in (dense_path, compact_path))
    dense_doc, compact_doc = delivery.check(dense), delivery.check(compact)
    _require(dense != compact and dense_doc['schema_version'] == compact_doc['schema_version'] == 6
        and _excluding(dense_doc, 'plan', 'source', 'executable', 'basis', 'mesh')
            == _excluding(compact_doc, 'plan', 'source', 'executable', 'basis', 'mesh'),
        'expected two frozen source-seeded publications with identical non-treatment artifacts')
    plan, other = _local(dense, dense_doc['plan']), _local(compact, compact_doc['plan'])
    _require(_excluding(plan, 'basis', 'mesh') == _excluding(other, 'basis', 'mesh')
        and all(_excluding(plan[stage], 'receipt', 'source') == _excluding(other[stage], 'receipt', 'source')
                for stage in ('basis', 'mesh'))
        and plan['mesh']['source'] == other['mesh']['source']
        and plan['basis']['source'] == proof['dense_input'] and other['basis']['source'] == proof['compact_input']
        and plan['owner_cpu'] == proof['placement']['owner_cpu'],
        'native arms differ beyond the private Basis transport at the same route and CPU placement')
    required_libraries = _stage_inputs(dense, compact, dense_doc, compact_doc, proof, proof_ref)
    # The frozen source derivation and exact original-run operand are different
    # kinds of evidence. Match source/argv and the actual packet independently.
    seed = plan['source_seed']
    seed_origin = _local(dense, seed['provenance'])
    capture_ref = effects['inputs']['capture']
    capture = read_artifact(capture_ref)
    _require(capture_ref == proof['mesh_capture'] and seed_origin['entry'] == capture['entry']
        and seed['argv'] == effects['configuration']['original_argv']
        and bytes.fromhex(seed['wire_hex']) == read_artifact(capture['seed_wire'], json_document=False),
        'native seed bytes or entry configuration differ from exact original source evidence')

    reference = _reference(benchmark_path, OBSERVATION)
    observed = read_artifact(reference)
    environment, argv = observed['environment'], seed['argv']
    outer_cwd = observed['benchmark_cwd']
    _require(observed['schema_version'] == 1
        and observed['scope'] == 'FIXED_ARGV_ORIGINAL_PYTHON_VS_SOURCE_DERIVED_SEED_NATIVE_FRESH_PROCESSES'
        and observed['source_scope'] == 'FIXED_SOURCE_CONFIGURATION_AND_EFFECT_ORACLE'
        and observed['source'] == capture['entry'] and observed['argc'] == len(argv) and observed['argv'] == argv
        and observed['cpu_ids'] == plan['cpu_ids']
        and type(outer_cwd) is str and Path(outer_cwd).is_absolute() and '\0' not in outer_cwd
        and all(environment.get(name) == value for name, value in plan['environment'].items())
        and all(type(name) is str and type(value) is str for name, value in environment.items())
        and observed['stdin'] == 'DEVNULL' and observed['exact_output_all'] is True
        and observed['stdout_sha256'] == effects['inputs']['original_stdout']['sha256']
        and observed['binaries'] == {'dense': dense_doc['executable']['sha256'],
                                    'compact': compact_doc['executable']['sha256']},
        'observation does not match the checked native executables and common launch/output contract')
    _shared_providers(dense, compact, observed['shared_executable_or_library_paths'], required_libraries)
    orders, warmup, measurements = observed['trials'], observed['warmup_ns'], observed['measurements']
    _require(type(orders) is list and len(orders) >= 2
        and all(type(order) is list and len(order) == len(_MODES) and set(order) == _MODES for order in orders)
        and len({tuple(order) for order in orders}) > 1 and warmup.keys() == measurements.keys() == _MODES
        and all(type(duration) is int and duration > 0 for duration in warmup.values()),
        'need complete mixed-order rounds and one recorded warmup per arm')
    output = {}
    for mode, row in measurements.items():
        values = row['samples_ns']
        _require(type(values) is list and len(values) == len(orders)
            and all(type(value) is int and value > 0 for value in values), 'invalid measured samples for ' + mode)
        output[mode] = _summary(values)
    speedup = {mode: output['original']['median_ns'] / output[mode]['median_ns'] for mode in ('dense', 'compact')}
    _require(output == measurements and speedup == observed['speedup'], 'benchmark summary fails cold recalculation')

    design_ref = dense_doc['evidence']['design']
    design = _local(dense, design_ref)
    minimum = design['study']['minimum_reduction_basis_points'] if threshold_bps is None else threshold_bps
    _require(type(minimum) is int and 0 <= minimum < 10000, 'expected a valid observed-gain policy threshold')
    best_dense, worst_compact = output['dense']['min_ns'], output['compact']['max_ns']
    enough = len(orders) >= 3
    profitable = enough and worst_compact < best_dense and worst_compact*10000 <= best_dense*(10000-minimum)
    reason = 'PROFITABLE_WITHIN_OBSERVED_ENVELOPE' if profitable else 'KEEP_DENSE_NO_ROBUST_GAIN'
    if not enough:
        reason = 'KEEP_DENSE_INSUFFICIENT_MATCHED_ROUNDS'
    return dict(schema=SCHEMA, schema_version=1, proof=proof_ref, effects=effects_ref,
        dense=_reference(dense/'artifact.json', delivery.SCHEMA),
        compact=_reference(compact/'artifact.json', delivery.SCHEMA), benchmark=reference,
        benchmark_script=_reference(script_path, 'scar.python-source'),
        launch=dict(source=capture['entry'], argv=argv, cpu_ids=plan['cpu_ids'], owner_cpu=plan['owner_cpu'],
            numa_node=proof['placement']['numa_node'], environment=environment, stdin=observed['stdin'],
            outer_cwd=outer_cwd, native_child_cwd=plan['cwd'], seed_sha256=seed['sha256']),
        policy=dict(threshold_override_basis_points=threshold_bps, minimum_reduction_basis_points=minimum,
            inherited_design={**design_ref, 'uri': str(dense/design_ref['uri'])}),
        timing=dict(arms=output, warmup_ns=warmup, orders=orders, original_to_native_median_speedup=speedup,
            paired_dense_minus_compact_ns=_summary([a-b for a,b in zip(
                output['dense']['samples_ns'], output['compact']['samples_ns'], strict=True)]),
            compact_worst_vs_dense_best_basis_points=((best_dense-worst_compact)*10000)//best_dense,
            scope='RECORDED_MATCHED_FRESH_PROCESS_SAMPLE_ENVELOPES', component_costs_resolved=False,
            exact_stdout_sha256=observed['stdout_sha256'], provider_bytes_replayed=True,
            staged_provider_inode_sharing_recorded=True, timing_rerun_by_replay=False),
        transport=proof['wire'], source_effect_equivalence_proven=False, deployment_authorized=False,
        selected='COMPACT' if profitable else 'DENSE', decision=reason)


def check(path):
    receipt = read_artifact(_reference(path, SCHEMA))
    replay = choose(receipt['proof']['uri'], receipt['effects']['uri'],
        Path(receipt['dense']['uri']).parent, Path(receipt['compact']['uri']).parent,
        receipt['benchmark']['uri'], receipt['benchmark_script']['uri'],
        threshold_bps=receipt['policy']['threshold_override_basis_points'])
    _require(replay == receipt, 'compact Basis choice failed cold replay')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proof', 'effects', 'dense', 'compact', 'benchmark', 'benchmark-script', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--minimum-reduction-bps', type=int)
    args = parser.parse_args(argv)
    inputs = (args.proof, args.effects, args.dense, args.compact, args.benchmark, args.benchmark_script)
    if args.check and not any(inputs) and args.out is None and args.minimum_reduction_bps is None:
        result = check(args.check)
        print(json.dumps(dict(verified=True, decision=result['decision'], selected=result['selected'])))
    elif not args.check and all(inputs) and args.out:
        result = choose(*inputs, threshold_bps=args.minimum_reduction_bps)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(decision=result['decision'], selected=result['selected'])))
    else:
        parser.error('use --check RECEIPT or supply both packages, proof, effects, benchmark, script and output')


if __name__ == '__main__':
    main()
