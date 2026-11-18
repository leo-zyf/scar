"""Differentially run an explicit ATen trial at its live Python function boundary."""

import argparse
import json
import os
from importlib.util import module_from_spec, spec_from_file_location
from math import prod
from pathlib import Path
from statistics import variance
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.codegen.native_provider_bundle import _content
from scar.optimization.paired_entry_selection import _metrics
from scar.runtime.aten_python_trial import _read_manifest, _verify, install_aten_python_trial


def _original_module(proof):
    source = Path(proof['source']['path'])
    source_bytes = _verify(source, proof['source'], retain=True)
    spec = spec_from_file_location('_scar_aten_trial_original', source)
    if spec is None or spec.loader is None:
        raise ValueError('the original Python source has no import loader')
    module = module_from_spec(spec)
    exec(compile(source_bytes, str(source), 'exec', dont_inherit=True), vars(module))
    return module


def _input_receipts(manifest, input_paths):
    if input_paths is None:
        return manifest['inputs']
    if manifest.get('input_guard', {}).get('kind') != 'VALUE_INDEPENDENT_GEOMETRY':
        raise ValueError('alternate inputs require an explicit value-independent geometry trial')
    recipe = manifest['recipe']
    if len(input_paths) != len(recipe['inputs']):
        raise ValueError('provide all alternate raw input files in native recipe order')
    inputs = []
    for tensor, path in zip(recipe['inputs'], input_paths, strict=True):
        path = Path(path).resolve(strict=True)
        receipt = dict(path=str(path), **_content(path))
        if receipt['size_bytes'] != 4 * prod(tensor['shape']):
            raise ValueError(f'alternate input bytes do not match the guarded Tensor geometry: {path}')
        inputs.append(receipt)
    return inputs


def _inputs(manifest, torch, receipts=None):
    recipe = manifest['recipe']
    proof = manifest['source_proof']
    args = [None] * len(proof['parameters'])
    if receipts is None:
        receipts = manifest['inputs']
    for index, tensor, receipt in zip(proof['input_order'], recipe['inputs'], receipts, strict=True):
        shape = tensor['shape']
        _verify(Path(receipt['path']), receipt)
        args[index] = torch.from_file(receipt['path'], False, prod(shape), dtype=torch.float32).reshape(shape)
    if any(value is None for value in args):
        raise ValueError('trial validation requires every original argument to have a retained input')
    return tuple(args)


def _check_outputs(original, candidate, inputs, torch, return_kind):
    if return_kind == 'TENSOR':
        original, candidate = (original,), (candidate,)
    if type(original) is not tuple or type(candidate) is not tuple or len(original) != len(candidate):
        raise ValueError('Python trial changed the original result structure')
    for reference, actual in zip(original, candidate, strict=True):
        if (type(reference) is not torch.Tensor or type(actual) is not torch.Tensor
                or (reference.shape, reference.dtype, reference.device, reference.stride(), reference.requires_grad)
                != (actual.shape, actual.dtype, actual.device, actual.stride(), actual.requires_grad)
                or not torch.equal(reference.detach().reshape(-1).view(torch.uint8),
                                   actual.detach().reshape(-1).view(torch.uint8))
                or any(torch._C._is_alias_of(actual, value) for value in inputs)):
            raise ValueError('Python trial changed output bytes, tensor metadata or input ownership')


def _timed(function, inputs):
    start = perf_counter_ns()
    result = function(*inputs)
    return perf_counter_ns() - start, result


def _measure(trial, digest, manifest, warmup, repetitions, torch, receipts):
    module = _original_module(manifest['source_proof'])
    proof = manifest['source_proof']
    name, kind = proof['function'], proof['return_kind']
    original = getattr(module, name)
    inputs = _inputs(manifest, torch, receipts)
    variable_values = manifest.get('input_guard', {}).get('kind') == 'VALUE_INDEPENDENT_GEOMETRY'
    snapshots = tuple(value.clone() for value in inputs)
    mask = os.sched_getaffinity(0)
    samples = []
    verification = {}
    with torch.no_grad(), install_aten_python_trial(module, trial, expected_sha256=digest,
                                                   exclusive_inputs=True, provider_report=verification) as counters:
        guarded = getattr(module, name)
        for i in range(warmup + repetitions):
            first_original = i % 2 == 0
            modes = (original, guarded) if first_original else (guarded, original)
            first, second = (_timed(mode, inputs) for mode in modes)
            baseline, candidate = (first, second) if first_original else (second, first)
            _check_outputs(baseline[1], candidate[1], inputs, torch, kind)
            if os.sched_getaffinity(0) != mask:
                raise ValueError('native call changed the Python caller affinity')
            if i >= warmup:
                samples.append(dict(pair=i-warmup, order='AB' if first_original else 'BA',
                                    original_ns=baseline[0], guarded_ns=candidate[0]))
        if counters['native_calls'] != warmup + repetitions or counters['fallback_calls']:
            raise ValueError('matched Python trial did not take the optimized path on each trial call')
        if any(not torch.equal(t.reshape(-1).view(torch.uint8), snapshot.reshape(-1).view(torch.uint8))
               for t, snapshot in zip(inputs, snapshots, strict=True)):
            raise ValueError('Python trial changed the caller Tensor inputs')

        keywords = dict(zip(proof['parameters'], inputs, strict=True))
        _check_outputs(original(*inputs), guarded(**keywords), inputs, torch, kind)
        with torch.enable_grad():
            differentiable = (inputs[0].detach().requires_grad_(True), *inputs[1:])
            _check_outputs(original(*differentiable), guarded(*differentiable), differentiable, torch, kind)
        for i, value in enumerate(inputs):
            if value.numel():
                changed = value.clone()
                changed.reshape(-1)[0] = 1.0 if changed.reshape(-1)[0] == 0 else 0.0
                alternatives = (*inputs[:i], changed, *inputs[i+1:])
                _check_outputs(original(*alternatives), guarded(*alternatives), alternatives, torch, kind)
                break
        expected_reasons = {'CALL_ABI': 1, 'RUNTIME_CONTEXT': 1}
        changed_value_tested = any(value.numel() for value in inputs)
        expected_native = warmup + repetitions
        if changed_value_tested and variable_values:
            expected_native += 1
        if changed_value_tested and not variable_values:
            expected_reasons['INPUT_BYTES'] = 1
        if variable_values:
            for i, value in enumerate(inputs):
                if value.ndim >= 2 and min(value.shape[-2:]) > 1:
                    noncontiguous = value.transpose(-2, -1).contiguous().transpose(-2, -1)
                    alternatives = (*inputs[:i], noncontiguous, *inputs[i+1:])
                    _check_outputs(original(*alternatives), guarded(*alternatives), alternatives, torch, kind)
                    expected_reasons['TENSOR_ABI'] = 1
                    break
        if counters['fallback_reasons'] != expected_reasons or counters['native_calls'] != expected_native:
            raise ValueError('live Python fallback did not honor the tested guard boundaries')
    if getattr(module, name) is not original or os.sched_getaffinity(0) != mask:
        raise ValueError('Python trial did not restore the original function and affinity')

    metrics = {}
    for label, field in (('ORIGINAL', 'original_ns'), ('GUARDED_NATIVE', 'guarded_ns')):
        values = [row[field] for row in samples]
        metrics[label] = dict(_metrics(values), variance_ns2=variance(values))
    return dict(samples=samples, metrics=metrics,
                speedup=metrics['ORIGINAL']['median_ns'] / metrics['GUARDED_NATIVE']['median_ns'],
                matched_faster_pairs=sum(row['guarded_ns'] < row['original_ns'] for row in samples),
                original_restored=True, bitwise_original_results=True, inputs_unchanged=True, counters=counters,
                input_guard_kind=verification.get('input_guard_kind', 'EXACT_BYTES'),
                reference_input_bytes_loaded=verification.get('reference_input_bytes_loaded'),
                changed_value_native_verified=variable_values and changed_value_tested)


def validate_aten_python_trial(trial, *, trial_sha256, out, warmup_pairs, measured_pairs, input_paths=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= warmup_pairs <= 100 or not 3 <= measured_pairs <= 501 or measured_pairs % 2 == 0:
        raise ValueError('trial requires 1–100 warmup pairs and an odd 3–501 measured pairs')
    trial = Path(trial).resolve(strict=True)
    manifest = _read_manifest(trial, trial_sha256)
    receipts = _input_receipts(manifest, input_paths)
    import torch

    mask = os.sched_getaffinity(0)
    if not set(manifest['cpus']) <= mask:
        raise ValueError('trial CPUs are outside the current allowed affinity')
    os.sched_setaffinity(0, manifest['cpus'])
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        evidence = _measure(trial, trial_sha256, manifest, warmup_pairs, measured_pairs, torch, receipts)
    finally:
        os.sched_setaffinity(0, mask)
    report = dict(schema='scar.aten-python-trial-validation', schema_version=1,
        scope='WARM_FULL_PYTHON_FUNCTION_WITH_GUARDS', trial=str(trial / 'trial.json'), trial_sha256=trial_sha256,
        warmup_pairs=warmup_pairs, measured_pairs=measured_pairs, cpu_ids=manifest['cpus'],
        immutable_reference_bytes=(0 if evidence['input_guard_kind'] == 'VALUE_INDEPENDENT_GEOMETRY'
                                   else sum(row['size_bytes'] for row in manifest['inputs'])),
        validated_inputs=receipts,
        complete_application_performance_measured=False, source_replacement_authorized=False, **evidence)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-trial-validation-', dir=out.parent) as tmp:
        stage = Path(tmp)
        (stage / 'validation.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stage.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar validate-aten-python-trial', description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--trial-sha256', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--warmup-pairs', type=int, default=7)
    parser.add_argument('--measured-pairs', type=int, default=31)
    parser.add_argument('--input', action='append', dest='input_paths',
                        help='alternate raw float32 inputs in recipe order for a geometry trial')
    args = parser.parse_args(argv)
    report = validate_aten_python_trial(**vars(args))
    print(json.dumps(dict(speedup=report['speedup'], faster_pairs=report['matched_faster_pairs'],
                          report=str(Path(args.out) / 'validation.json'))))
    return 0
