"""Validate an original Python matrix entry against its locked NVIDIA device project."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from scar.analysis.nvidia_stack_target import SCHEMA as TARGET_PROBE, _join, _pci_address, _target as probe_target
from scar.analysis.nvidia_stack_target import probe_nvidia_target
from scar.analysis.nvhpc_source_entry import SCHEMA as SOURCE_CONTRACT
from scar.compiler.compile_exported_aten import _write_record
from scar.compiler.compile_exported_aten_cuda import CUDA_ENVIRONMENT, SCHEMA as PROJECT
from scar.compiler.compile_exported_nvhpc import _contract
from scar.compiler.entry_delivery import NVHPC_DEVICE_ROUTE as ROUTE
from scar.compiler.nvhpc_entry_variant import matrix_entry_inputs
from scar.compiler.nvhpc_source_validation import _collect_pairs, _context, _receipt, _replay_pairs, _timing
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_device_comparison import native_device_probe_reason
from scar.optimization.resource_solver import semantic_sha256
from scar.runtime.native_device_probe import (
    _check_device_unchanged, _cuda_runtime_lock, _launch_scope, _observe_cuda_preflight,
)
from scar.runtime.validate_native_project import verify_native_project


SCHEMA = 'scar.nvhpc-source-device-process-observation'


def qualify_source_device(scope, reference, manifest):
    """Join the replayed device artifact to the original source, target and file ABI."""
    from scar.compiler.entry_delivery import _budget, _entry, _target
    from scar.compiler.plan_aten_boundary import _target as cpu_target

    root, replay = verify_native_project(Path(reference['uri']).parent, reference['sha256'])
    if not replay.get('original_source_entry_replayed') or 'entry_obligations_sha256' not in scope:
        raise ValueError('NVIDIA device delivery requires the original source and F4 obligations')
    stack_target = read_artifact(scope['stack_target'])
    stack = manifest['operator_stack']['reference']
    if stack_target['stack'] != stack or stack_target['target'] != scope['target']:
        raise ValueError('CUDA binary differs from the scoped NVIDIA stack or target graph')
    _target(scope, scope['target'])
    contract_ref = artifact_reference(root/'plan/source-entry.json', SOURCE_CONTRACT)
    contract = read_artifact(contract_ref)
    source = artifact_reference(contract['source']['path'], 'scar.python-source')
    _entry(scope, source['uri'], source['sha256'])
    bindings = json.loads((root/'plan/bindings.json').read_bytes())
    original_cpus = scope['process_affinity_cpu_ids']
    host = bindings.get('host_cpu_ids')
    if host is not None:
        if not set(host) <= set(original_cpus) or bindings.get('host_target', scope['target']) != scope['target']:
            raise ValueError('CUDA hybrid host lane exceeds the original CPU or target scope')
        cpu_target(Path(scope['target']['uri']).parent, tuple(host), allow_cpu_subset=True)
    payload = replay['reserved_payload_bytes']
    kinds = bindings['memory_kinds']
    host_bytes = sum(byte_count for pool, byte_count in payload.items() if kinds[pool] != 'Device')
    device_bytes = sum(byte_count for pool, byte_count in payload.items() if kinds[pool] == 'Device')
    _budget(scope, host_bytes)
    numerics = scope.get('numerical_contract', {'kind': 'EXACT_BYTES'})
    if numerics != _contract(numerics.get('atol'), numerics.get('rtol')):
        raise ValueError('CUDA device validation needs the exact scoped numerical contract')
    return dict(kind='STANDALONE_CPP', source=source, source_contract=contract_ref, target=scope['target'],
        operator_stack=stack, file_inputs=matrix_entry_inputs(contract, scope['argv']),
        output_sizes=[port['nbytes'] for port in contract['outputs']], numerical_contract=numerics,
        launcher=artifact_reference(root/'launch.sh', 'scar.native-source-launcher'), device_ordinal=bindings['device'],
        native_cpu_ids=original_cpus if host is None else host, host_payload_bytes=host_bytes,
        device_payload_bytes=device_bytes, python_runtime_required=False)


def _context_matches(scope, context):
    from scar.compiler.entry_delivery import _environment_digest

    return (context['cpu_ids'] == scope['process_affinity_cpu_ids']
            and context['prefix'] == ['/usr/bin/taskset', '-c', ','.join(map(str, context['cpu_ids']))]
            and context['python']['path'] == scope['python_executable']
            and context['working_directory'] == scope['working_directory']
            and _environment_digest(context['environment']) == scope['environment_sha256'])


def _commands(scope, evidence, context):
    prefix, argv = context['prefix'], scope['argv']
    return dict(ORIGINAL=[*prefix, context['python']['path'], '-B', scope['entry'], *argv],
                NATIVE=[*prefix, evidence['launcher']['uri'], '--trial', *argv])


def _preflight_status(device, probe, pinned, *, evidence, scope):
    env = device['cuda_environment']
    if (device['device_ordinal'] != evidence['device_ordinal'] or device['benchmark_cpus'] != evidence['native_cpu_ids']
            or not set(scope['process_affinity_cpu_ids']) <= set(device['caller_affinity'])
            or not env.keys() <= set(CUDA_ENVIRONMENT) or any(type(value) is not str for value in env.values())):
        raise ValueError('device preflight differs from the admitted launch scope')
    reason = native_device_probe_reason(probe, device=evidence['device_ordinal'])
    identity = device['cuda_identity']
    if reason is not None:
        if identity is not None or pinned is not None:
            raise ValueError('failed device preflight contains a successful identity')
        return 'DEVICE_UNAVAILABLE' if reason == 'CUDA_DEVICE_UNAVAILABLE' else 'PROBE_FAILED'
    if identity is None:
        if pinned is not None or not device['cuda_identity_error']:
            raise ValueError('device identity lookup has no evidence')
        return 'DEVICE_SCOPE_UNVERIFIED'
    root = Path(evidence['artifact']['uri']).parent
    runtime = _cuda_runtime_lock(root, read_artifact(evidence['artifact']))
    if (runtime is None or identity['runtime_library'] != runtime or device['cuda_identity_error'] is not None
            or type(identity['device_count']) is not int or identity['device_count'] <= evidence['device_ordinal']
            or any(type(identity[key]) is not int or identity[key] <= 0 for key in ('runtime_version', 'driver_version'))
            or _pci_address(identity['pci_bus_id']) is None):
        raise ValueError('device preflight did not use the pinned CUDA runtime')
    if (pinned is None or pinned['schema'] != TARGET_PROBE or pinned['schema_version'] != 1
            or pinned['target'] != evidence['target']
            or pinned['scope']['cuda_environment'] != env):
        raise ValueError('device preflight requires the original hardware graph join')
    if pinned['status'] != 'DEVICE_QUERY_SUCCEEDED':
        return 'TARGET_DEVICE_UNBOUND'
    devices = deepcopy(pinned['visible_devices'])
    snapshot, resources = probe_target(evidence['target'])
    _join(devices, resources, snapshot)
    if devices != pinned['visible_devices']:
        raise ValueError('UUID and PCI relation differs from the retained hardware graph')
    row = next((entry for entry in devices if entry['ordinal'] == evidence['device_ordinal']), None)
    if (row is None or row['retained_target']['status'] != 'UUID_AND_PCI_MATCH'
            or row['pci_bus_id'] != _pci_address(identity['pci_bus_id'])):
        return 'TARGET_DEVICE_UNBOUND'
    return 'DEVICE_CAPACITY_INSUFFICIENT' if row['memory_total_bytes'] < evidence['device_payload_bytes'] else None


def _counts(warmup, count):
    if type(warmup) is not int or not 0 <= warmup <= 10 or type(count) is not int or not 3 <= count <= 30:
        raise ValueError('require a bounded warmup and at least three timed process pairs')
    return warmup + count


def validate_source_device(scope, evidence, out, *, warmup_pairs=1, measurement_pairs=3):
    """Check device readiness before executing either arm of the original process trial."""
    from scar.compiler.entry_delivery import _environment_digest

    count = _counts(warmup_pairs, measurement_pairs)
    if _environment_digest(os.environ) != scope['environment_sha256']:
        raise ValueError('caller must use the original frozen environment for this device trial')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = read_artifact(evidence['artifact'])
    root = Path(evidence['artifact']['uri']).parent
    cpus = evidence['native_cpu_ids']
    launch = _launch_scope(cpus)
    device, probe, preliminary = _observe_cuda_preflight(root, manifest, evidence['device_ordinal'], cpus, launch)
    pinned = probe_nvidia_target(target=evidence['target']) if preliminary is None else None
    phase = _preflight_status(device, probe, pinned, evidence=evidence, scope=scope)
    context, commands = None, None
    runs, checks, timing = [], [], None
    if phase is None:
        import torch

        context = _context(scope['process_affinity_cpu_ids'], cuda_environment=device['cuda_environment'])
        if not _context_matches(scope, context):
            raise ValueError('source process setup differs from its frozen scope')
        commands = _commands(scope, evidence, context)
        publications = list(map(Path, scope['argv'][len(evidence['file_inputs']):]))
        runs, checks = _collect_pairs(commands, context, out, publications, evidence['numerical_contract'],
                                     evidence['output_sizes'], count, torch)
        _check_device_unchanged(device, *launch[:2])
        if probe_nvidia_target(target=evidence['target']) != pinned:
            raise ValueError('device or original hardware join changed during the trial')
        if len(checks) == count and checks[-1]['accepted']:
            phase = 'COMPLETE'
            timing = _timing(runs, warmup_pairs, measurement_pairs)
        else:
            phase = 'OUTPUT_CONTRACT_FAILED'
    if dict(artifact=evidence['artifact'], **qualify_source_device(scope, evidence['artifact'], manifest)) != evidence:
        raise ValueError('source or native package changed during device validation')
    report = dict(schema=SCHEMA, schema_version=1, scope_sha256=semantic_sha256(scope), evidence=evidence,
        warmup_pairs=warmup_pairs, measurement_pairs=measurement_pairs, phase=phase,
        device=device, probe=probe, pinned_device=pinned, context=context, commands=commands, runs=runs,
        comparisons=checks, timing=timing, deployment_authorized=False)
    _write_record(out/'result.json', report)
    return report


def replay_source_device(reference, evidence, scope, *, minimum_gain=0.05):
    saved = read_artifact(reference)
    if (reference['schema'] != SCHEMA or saved['schema_version'] != 1 or saved['evidence'] != evidence
            or saved['scope_sha256'] != semantic_sha256(scope) or saved['deployment_authorized'] is not False):
        raise ValueError('device observation differs from its exact artifact and original execution scope')
    phase = _preflight_status(saved['device'], saved['probe'], saved['pinned_device'], evidence=evidence, scope=scope)
    warmup, count = saved['warmup_pairs'], saved['measurement_pairs']
    total = _counts(warmup, count)
    runs, checks, context = saved['runs'], saved['comparisons'], saved['context']
    timing, retained = None, None
    if phase is None:
        import torch

        visible = {name: context['environment'][name] for name in CUDA_ENVIRONMENT if name in context['environment']}
        if (not _context_matches(scope, context) or saved['commands'] != _commands(scope, evidence, context)
                or visible != saved['device']['cuda_environment']):
            raise ValueError('retained device trial does not have its original process commands')
        for provider in (context['python'], *(context['providers'][name] for name in ('torch', 'torch_native', 'numpy'))):
            if _receipt(provider['path']) != provider:
                raise ValueError('retained device trial source provider changed')
        if _replay_pairs(Path(reference['uri']).parent, runs, checks, evidence['numerical_contract'],
                         evidence['output_sizes'], total, torch):
            phase = 'COMPLETE'
            retained = _timing(runs, warmup, count)
            timing = _timing(runs, warmup, count, minimum_gain=minimum_gain)
        else:
            phase = 'OUTPUT_CONTRACT_FAILED'
    elif runs or checks or context is not None or saved['commands'] is not None:
        raise ValueError('device observation includes entry runs after preflight rejection')
    if phase != saved['phase'] or retained != saved['timing']:
        raise ValueError('device observation decision differs from retained runs')
    return timing, dict(phase=phase, device=saved['device'], entry_runs=len(runs),
                        output_contract_verified=phase == 'COMPLETE', deployment_authorized=False)


def main(argv=None):
    from scar.compiler.entry_delivery import qualify_entry_delivery

    parser = argparse.ArgumentParser(prog='scar validate-nvhpc-source-device', description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--domain', required=True, help='original plan-build whole-entry-domain.json')
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--out')
    operation.add_argument('--check')
    parser.add_argument('--warmup-pairs', type=int, default=1)
    parser.add_argument('--measurement-pairs', type=int, default=3)
    args = parser.parse_args(argv)
    domain = read_artifact(artifact_reference(args.domain, 'scar.whole-entry-option-domain'))
    scope = domain['scope']
    ref = artifact_reference(Path(args.project)/'artifact.json', PROJECT)
    if ref['sha256'] != args.manifest_sha256:
        parser.error('supplied project digest differs from the locked artifact')
    admission = qualify_entry_delivery(scope, dict(route=ROUTE, id='source-device-observation', artifact=ref))
    if admission['status'] != 'QUALIFIED':
        parser.error(admission['reason'])
    evidence = admission['evidence']
    if args.check is None:
        saved = validate_source_device(scope, evidence, args.out, warmup_pairs=args.warmup_pairs,
                                       measurement_pairs=args.measurement_pairs)
        phase, runs, timing = saved['phase'], len(saved['runs']), saved['timing']
    else:
        timing, summary = replay_source_device(artifact_reference(args.check, SCHEMA), evidence, scope)
        phase, runs = summary['phase'], summary['entry_runs']
    print(json.dumps(dict(phase=phase, entry_runs=runs, performance_measured=timing is not None,
                          deployment_authorized=False)))
    return 0
