"""Replay a pinned, scoped whole-entry caller-pipeline measurement.

This is an admission adapter for the existing live guarded ABI, not another
optimizer. A region proof or a selection dictionary is never cost evidence.
"""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys

from .compiled_deployment import entry_context, _raw


SCHEMA = 'scar.caller-pipeline-deployment'
DESIGN = 'scar.caller-pipeline-validation-design'
SAMPLE = 'scar.caller-pipeline-validation-sample'
EXECUTION = 'scar.caller-pipeline-delivery-execution'
TRUST = 'EXPLICIT_EXPECTED_DEPLOYMENT_SHA256_FROM_VALIDATOR_NOT_A_SIGNATURE'


def reference(path, schema):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(uri=str(path), sha256=sha256(raw).hexdigest(), size_bytes=len(raw), schema=schema)


def environment_sha256():
    """Freeze inherited environment without retaining secrets or their values."""
    return sha256(json.dumps(sorted(os.environ.items()), separators=(',', ':')).encode()).hexdigest()


def runner_identity():
    root = Path(__file__).resolve().parents[1]
    names = ('runtime/caller_pipeline_delivery.py', 'runtime/caller_pipeline_deployment.py',
        'runtime/caller_pipeline_cli.py', 'runtime/caller_pipeline_guard.py',
        'runtime/compiled_deployment.py', 'runtime/entry_runner.py', 'runtime/native_loop.py',
        'runtime/stateful_guard.py', 'runtime/native_stateful.py', 'runtime/stateful_contract.py',
        'runtime/bundle_metadata.py', 'trace/code_snapshot.py')
    return sha256(json.dumps([(name, sha256((root/name).read_bytes()).hexdigest()) for name in names],
        separators=(',', ':')).encode()).hexdigest()


def provider_identity():
    """Pin NumPy distribution declarations and the actual ndarray/ufunc core.

    Irrelevant tests and unused extension modules are not hashed. Declarations
    are not claimed to prove every imported module's actual bytes; the actual
    loaded math provider has a separate source/binary receipt.
    """
    from importlib.metadata import distribution
    dist = distribution('numpy')
    files = dist.files
    if files is None: raise ValueError('NumPy distribution files are not declared')
    names = [str(name) for name in files if str(name) in ('numpy/__init__.py', 'numpy/_core/__init__.py')
        or str(name).startswith('numpy/_core/_multiarray_umath.') and str(name).endswith('.so')]
    if len(names) != 3: raise ValueError('NumPy declared ndarray/ufunc core receipts required')
    rows = []
    for name in sorted(names):
        path = Path(dist.locate_file(name)).resolve()
        rows.append(dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
    declarations = {}
    for name in ('METADATA', 'RECORD'):
        text = dist.read_text(name)
        if text is None: raise ValueError('NumPy distribution declaration missing')
        declarations[name] = sha256(text.encode()).hexdigest()
    return dict(name='numpy', version=dist.version, distribution_declarations=declarations,
        actual_ndarray_ufunc_core=rows,
        scope='DISTRIBUTION_DECLARATIONS_AND_ACTUAL_NDARRAY_UFUNC_CORE_NOT_FULL_MODULE_OR_SYSTEM_LIBRARY_CLOSURE')


def source_references(entry, contract, source_paths=()):
    paths = {Path(entry).resolve(), Path(contract['source']['path']).resolve()}
    paths.update(Path(row['source']['path']).resolve() for row in contract['callees'])
    paths.update(Path(path).resolve() for path in source_paths)
    return [reference(path, 'scar.caller-pipeline-program-source') for path in sorted(paths)]


def bundle_reference(directory, expected_admission_sha256):
    from .caller_pipeline_guard import inspect_bundle
    metadata = inspect_bundle(directory, expected_admission_sha256=expected_admission_sha256)
    return dict(directory=metadata['directory'], admission_sha256=metadata['admission_sha256'],
        runtime_sha256=metadata['admission']['runtime_sha256'],
        native=metadata['contract']['native']), metadata


def execution_counts(report, *, bundle, context, identity, environment, design=None):
    """Validate observed installed-boundary coverage, never infer global coverage."""
    if (report.get('schema'), report.get('schema_version')) != (EXECUTION, 1):
        return 0, 0, False
    if (report.get('mode') != 'MEASURE_CANDIDATE' or report.get('bundle') != bundle
            or report.get('entry_context') != context or report.get('runner_identity') != identity
            or report.get('environment_sha256') != environment
            or report.get('context_unchanged') is not True
            or design is not None and report.get('validation_design') != design):
        return 0, 0, False
    inner = report.get('execution', {})
    stats = inner.get('statistics', {}); native = stats.get('native_calls'); fallback = stats.get('fallback_calls')
    valid = (type(native) is int and native > 0 and type(fallback) is int and fallback == 0
        and inner.get('schema') == 'scar.entry-runner-execution' and inner.get('schema_version') == 1
        and inner.get('entry_path') == context['entry_path'] and inner.get('argv') == context['argv']
        and inner.get('exit_code') == 0 and inner.get('installed_bindings') == 1
        and not inner.get('frontier') and inner.get('decision') == 'GUARDED_ATTACHMENT'
        and inner.get('observed_boundary_calls') == native)
    return native if type(native) is int else 0, fallback if type(fallback) is int else 0, bool(valid)


def deployment_gate(receipt, *, expected_deployment_sha256, bundle,
        expected_admission_sha256, entry, argv):
    """Replay preliminary candidate evidence, retaining NO-OP for deployment.

    Receipts are producer-pinned local evidence, not signatures. Changed argv,
    frozen sources, environment, interpreter, runtime or ELF bytes retain NO-OP.
    The candidate clock excludes this receipt replay. Final-path cost admission
    is not implemented; even positive candidate gain cannot authorize native.
    """
    def rejected(reason):
        return dict(decision='NO_OP', deployment_authorized=False, reason=reason)
    if receipt is None or expected_deployment_sha256 is None:
        return rejected('complete_entry_performance_receipt_and_expected_hash_required')
    try:
        ref = reference(receipt, SCHEMA)
        if ref['sha256'] != expected_deployment_sha256: raise ValueError('deployment receipt trust pin differs')
        value = json.loads(_raw(ref))
        if ((value.get('schema'), value.get('schema_version')) != (SCHEMA, 1)
                or value.get('trust_policy') != TRUST): raise ValueError('unsupported caller deployment receipt')
        frozen_bundle, _ = bundle_reference(bundle, expected_admission_sha256)
        context = entry_context(entry, argv)
        if value['bundle'] != frozen_bundle or value['entry_context'] != context:
            raise ValueError('entry arguments/context or original bundle differ')
        if value['environment_sha256'] != environment_sha256(): raise ValueError('inherited environment changed')
        if value['runner_identity'] != runner_identity(): raise ValueError('delivery runtime changed')
        for source in value['source_refs']: _raw(source)
        if value['provider_identity'] != provider_identity(): raise ValueError('NumPy source/binary identity changed')
        for evidence in value['evidence']: _raw(evidence)
        design_refs = [row for row in value['evidence'] if row['schema'] == DESIGN]
        if len(design_refs) != 1: raise ValueError('one original frozen validation design required')
        design = json.loads(_raw(design_refs[0]))
        keys = ('bundle', 'entry_context', 'environment_sha256', 'runner_identity', 'source_refs',
            'provider_identity', 'policy', 'validator_implementation')
        if any(design[key] != value[key] for key in keys): raise ValueError('receipt differs from frozen design')
        policy = value['policy']; warmup = policy['warmup']; count = policy['repetitions']; minimum = policy['minimum_gain']
        if (type(warmup) is not int or warmup < 1 or type(count) is not int or count < 3
                or type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum < 1
                or policy.get('frozen_before_measurement') is not True):
            raise ValueError('complete predeclared warmup/measurement/gain policy required')
        expected = {('WARMUP', i) for i in range(warmup)} | {('MEASUREMENT', i) for i in range(count)}
        samples = value['samples']; coordinates = [(row['phase'], row['index']) for row in samples]
        if len(coordinates) != len(expected) or set(coordinates) != expected:
            raise ValueError('complete unique warmup and measurement pairs required')
        measurements = []
        for sample in samples:
            if json.loads(_raw(sample['sample_ref'])) != {k: v for k, v in sample.items() if k != 'sample_ref'}:
                raise ValueError('retained original sample differs')
            baseline, optimized = sample['baseline'], sample['optimized']
            for row in (baseline, optimized):
                if type(row['wall_ns']) is not int or row['wall_ns'] <= 0 or row['returncode'] != 0:
                    raise ValueError('successful positive complete-process samples required')
            if (_raw(baseline['stdout']) != _raw(optimized['stdout'])
                    or _raw(baseline['stderr']) != _raw(optimized['stderr'])):
                raise ValueError('entry stdout/stderr differ')
            report = json.loads(_raw(optimized['execution_report']))
            native, fallback, valid = execution_counts(report, bundle=frozen_bundle, context=context,
                identity=value['runner_identity'], environment=value['environment_sha256'], design=design_refs[0])
            if (not valid or sample['valid'] is not True or sample['native_calls'] != native
                    or sample['fallback_calls'] != fallback):
                raise ValueError('whole-entry candidate or guarded-boundary observations invalid')
            if sample['phase'] == 'MEASUREMENT': measurements.append(sample)
        gain = 1 - statistics.median(row['optimized']['wall_ns'] for row in measurements) / statistics.median(
            row['baseline']['wall_ns'] for row in measurements)
        if gain < minimum: return rejected('not_profitable_at_frozen_whole_entry_gain')
        return dict(**rejected('positive_candidate_gain_only_final_delivery_admission_cost_not_covered'),
            candidate_gain_fraction=gain, evidence_status='PRELIMINARY_CANDIDATE_ONLY',
            final_delivery_cost_admission_supported=False)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        return rejected('entry_performance_or_context_not_admitted:' + str(error))


__all__ = ['deployment_gate', 'entry_context', 'runner_identity', 'environment_sha256',
    'provider_identity', 'source_references', 'reference', 'bundle_reference', 'execution_counts']
