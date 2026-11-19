"""Collect instrumented service samples for one admitted hybrid binary."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.compiler.compile_exported_aten_cuda import ACTIVITY_BINDING
from scar.optimization.native_device_activity_costs import verify_native_activity_sample
from scar.runtime.native_device_probe import (
    _attempt, _check_device_unchanged, _launch_scope, _locked_json, _observe_cuda_preflight, _package,
    _payloads, _receipt_paths, _stat_locks,
)


SCHEMA = 'scar.native-device-activity-observations'


def decode_activity_attempt(attempt, expected, steps):
    if attempt['returncode'] != 0 or attempt['stderr']:
        return None, 'NATIVE_CALIBRATION_FAILED'
    if attempt['outputs'] != expected:
        return None, 'REFERENCE_OUTPUT_MISMATCH'
    try:
        sample = json.loads(attempt['stdout'])
        verify_native_activity_sample(sample, steps)
    except ValueError:
        return None, 'NATIVE_CALIBRATION_WIRE_MISMATCH'
    return sample, None


def observe_device_activities(cpu, cuda, *, witness, expected_identity, binding_ref, inputs, references,
                              warmup_runs, measurement_runs):
    cpu_manifest, cpu_identity = _package(cpu, 'scar.native-exported-aten')
    manifest, cuda_identity = _package(cuda, 'scar.native-exported-aten-cuda')
    identities = dict(CPU=cpu_identity, CUDA=cuda_identity)
    if (identities != expected_identity or manifest.get('activity_calibration_abi') != 1
            or manifest['files'].get(ACTIVITY_BINDING) != {
                key: binding_ref[key] for key in ('sha256', 'size_bytes')}):
        raise ValueError('activity probe requires the admitted instrumented native artifact')
    description = _locked_json(cuda, manifest, 'validation/description.json')
    binding = _locked_json(cuda, manifest, ACTIVITY_BINDING)
    if description['device'] != witness['cuda']['device'] or 'activity_calibration' not in description:
        raise ValueError('activity probe device or capability differs from the admitted artifact')
    if (_receipt_paths(row['path'] for row in inputs) != inputs
            or _receipt_paths(row['path'] for row in references) != references):
        raise ValueError('activity input or reference bytes changed before calibration')
    protected = {row['path'] for row in (*inputs, *references)}
    for root, info in ((cpu, cpu_manifest), (cuda, manifest)):
        protected.update(str(root / name) for name in ('artifact.json', *info['files']))
        protected.update(path for path in info['dependencies'].values() if path is not None)
    previous = _stat_locks(protected)
    cpu_ids = witness['cpu']['cpu_ids']
    scope = _launch_scope(cpu_ids)
    prefix, environment, _ = scope
    context, probe, phase = _observe_cuda_preflight(cuda, manifest, description['device'], cpu_ids, scope)
    runs, failure = [], None
    if phase is None:
        phase = 'COMPLETE'
        expected = _payloads(references)
        arguments = ['--calibrate', *(row['path'] for row in inputs)]
        with TemporaryDirectory(prefix='scar-device-activity-') as scratch:
            for ordinal in range(warmup_runs + measurement_runs):
                attempt = _attempt(cuda, 'CUDA', ordinal, prefix, arguments, len(references), scratch, environment)
                sample, reason = decode_activity_attempt(attempt, expected, binding['activities'])
                if reason is not None:
                    phase = 'CALIBRATION_FAILED'
                    failure = dict(ordinal=ordinal, reason=reason, attempt=attempt)
                    break
                runs.append(dict(ordinal=ordinal, sample=sample, outputs=attempt['outputs']))
        if phase == 'COMPLETE':
            _check_device_unchanged(context, prefix, environment)
    if (_stat_locks(protected) != previous
            or _receipt_paths(row['path'] for row in inputs) != inputs
            or _receipt_paths(row['path'] for row in references) != references):
        raise ValueError('native packages or sample files changed during activity calibration')
    return dict(schema=SCHEMA, schema_version=1, phase=phase, probe=probe, context=context,
                inputs=inputs, references=references, package_identity=identities,
                binding_sha256=binding_ref['sha256'], warmup_runs=warmup_runs,
                measurement_runs=measurement_runs, runs=runs, failure=failure)
