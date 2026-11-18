"""Bind a checked stateful resource plan to one isolated C++ validation run."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import wraps
from hashlib import file_digest
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import sys

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.controlled-stateful-delivery'
VALIDATION = 'scar.controlled-stateful-delivery-validation'


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(), schema)


def _verify_native_resources(reference, contract, stages, protocol):
    """Match the certified fixed order to the actual C++ and ABI resource inputs."""
    from scar.optimization.resource_solver import semantic_sha256

    receipt = read_artifact(reference)
    implementation = contract['implementation']
    identity = contract['identity']
    memory = contract['memory']
    time = stages['time_binding']
    native_lane = time['resources']['native_lane']['id']
    expected = dict(schema='scar.controlled-native-resource-certificate', schema_version=1,
        status='VERIFIED_FIXED_NATIVE_ORDER', native_manifest=implementation['native_manifest'],
        native_stage_plan=identity['stage_plan'], generated_source=implementation['native_members']['source'],
        candidate=identity['candidate'], contract_sha256=semantic_sha256(contract),
        projected_problem_sha256=stages['projected_problem_sha256'],
        task_order=[task['id'] for task in stages['timed_schedule']['tasks']],
        native_cpu_ids=contract['execution']['affinity_stage']['cpu_ids'],
        native_affinity_task_ids=[row['task'] for row in time['task_bindings'] if row['native_lane'] == native_lane],
        private_slots={row['id']:row['capacity_bytes'] for row in memory['private_slots']},
        private_payload_bytes=memory['private_workspace_bytes'],
        borrowed_payload_bytes=memory['borrowed_payload_bytes'],
        scoped_payload_bytes=memory['included_payload_bytes'], budget_bytes=memory['budget_bytes'],
        numeric_durations_proven=False, native_reordering_authorized=False)
    source = protocol['native_members']['source']
    stage_plan = read_artifact(expected['native_stage_plan'])
    native_manifest = read_artifact(expected['native_manifest'])
    if (any(receipt.get(key) != value for key, value in expected.items())
            or any(source[key] != expected['generated_source'][key] for key in ('uri', 'sha256', 'size_bytes'))
            or stage_plan['status'] != 'READY' or stage_plan['budget_satisfied'] is not True
            or stage_plan['resource_binding']['cpu_ids'] != expected['native_cpu_ids']
            or native_manifest['cpu_ids'] != expected['native_cpu_ids']
            or {row['id']:row['capacity_bytes'] for row in stage_plan['workspace_slots']} != expected['private_slots']
            or native_manifest['workspace_payload_bytes'] != expected['private_payload_bytes']
            or native_manifest['external_array_bytes'] != expected['borrowed_payload_bytes']
            or native_manifest['total_region_array_payload_bytes'] != expected['scoped_payload_bytes']
            or native_manifest['memory_budget_bytes'] != expected['budget_bytes']):
        raise ValueError('native resource certificate differs from the fixed C++ entry and its scope')
    return receipt


def _derive(preparation, canonical_model):
    from scar.analysis.stateful_entry_protocol import derive_stateful_entry_protocol
    from scar.compiler.calibrate_stateful_entry import read_canonical_entry_model
    from scar.compiler.plan_controlled_region import check as check_plan

    planned = check_plan(preparation)
    if planned['kind'] != 'STATEFUL_SPECIALIZATION':
        raise ValueError('checked stateful specialization required')
    files = planned['files']
    contract = read_artifact(files['contract'])
    identity = contract['identity']
    native = contract['implementation']['native_manifest']
    model = read_canonical_entry_model(canonical_model)
    protocol = derive_stateful_entry_protocol(native, trial=True, exclusive_inputs=True)
    strict = read_artifact(files['strict_problem'])
    cost = read_artifact(files['entry_cost_use'])
    (operation,) = strict['operations']
    (mode,) = operation['options']
    if (model['specialization'] != planned['admission'] or model['domain'] != identity['domain']
            or model['candidate_id'] != identity['candidate_id']
            or model['profile'] not in planned['entry_profiles']
            or model['entry_protocol'] != protocol
            or mode['parameters']['cost_use']['requested_entry_protocol'] != protocol
            or protocol['native_members']['plan'] != identity['stage_plan']):
        raise ValueError('checked timing, canonical cost and runtime lowering must share one entry protocol')
    specialization = read_artifact(planned['admission'])
    program = read_artifact(specialization['program'])
    entry = _reference(program['entry'], 'scar.python-source')
    stages = read_artifact(files['native_task_projection'])
    if (stages['original_native_order_preserved'] is not True
            or stages['reordering_authorized'] is not False):
        raise ValueError('delivery requires the original executable native stage order')
    resource_reference = files.get('native_resource_certificate')
    if resource_reference is not None:
        _verify_native_resources(resource_reference, contract, stages, protocol)
    schedules = {}
    for name in ('strict_plan', 'controlled_plan', 'native_task_plan'):
        status = read_artifact(files[name])
        schedules[name] = dict(reference=files[name], status=status['status'],
            selected_count=len(status['selected']))
    return dict(schema=SCHEMA, schema_version=1, preparation=preparation,
        canonical_model=canonical_model, entry=entry, specialization=planned['admission'],
        candidate=identity['candidate'], native_artifact=native, native_members=protocol['native_members'],
        native_stage_plan=identity['stage_plan'], native_task_projection=files['native_task_projection'],
        native_resource_certificate=resource_reference,
        entry_loop_context=files.get('entry_loop_context'),
        entry_cost_use=files['entry_cost_use'], entry_cost_status=cost['status'],
        timing_qualified=cost['timing_qualified'], schedules=schedules,
        entry_protocol=protocol, producer=_reference(__file__, 'scar.implementation-source'),
        decision='KEEP', default_backend='NO_OP', explicit_validation_available=True,
        changed_task_order_applied=False, deployment_authorized=False)


def check(admission, *, expected_sha256=None):
    """Independently replay source, resource plan, cost, protocol and member bytes."""
    reference = _reference(admission, SCHEMA)
    if expected_sha256 is not None and reference['sha256'] != expected_sha256:
        raise ValueError('stateful delivery admission checksum differs')
    saved = read_artifact(reference)
    if saved != _derive(saved['preparation'], saved['canonical_model']):
        raise ValueError('stateful delivery admission differs from source, schedule and cost replay')
    return saved


def _episode(module, region, original, selected, entry, argv, token, directory):
    from scar.compiler.stateful_holdout import _owner, _state, _summary
    from scar.runtime.entry_runner import run_original_script

    owner, name = _owner(module, region)
    descriptor = vars(owner)[name]
    signature = inspect.signature(original)
    stdout, stderr = io.StringIO(), io.StringIO()
    trajectory, last = [], None
    output = directory/'program-output'
    directory.mkdir()

    def snapshot(arguments):
        return _summary(_state(region, arguments, contents=True))

    @wraps(original)
    def observe(*args, **kwargs):
        nonlocal last
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        last = bound.arguments
        before = snapshot(last)
        result = selected(*args, **kwargs)
        trajectory.append(dict(before=before, after=snapshot(last)))
        return result

    wrapper = staticmethod(observe) if type(descriptor) is staticmethod else observe
    setattr(owner, name, wrapper)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            run_original_script(entry, [str(output) if token is not None and arg == token else arg
                                        for arg in argv])
        if vars(owner).get(name) is not wrapper:
            raise ValueError('declared entry replaced the observed method descriptor')
        if last is None:
            raise ValueError('declared entry did not call the admitted stateful method')
        final = snapshot(last)
    finally:
        setattr(owner, name, descriptor)
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                files[str(path.relative_to(output))] = dict(
                    size_bytes=os.fstat(stream.fileno()).st_size, sha256=file_digest(stream, 'sha256').hexdigest())
    return dict(trajectory=trajectory, final=final, stdout=stdout.getvalue(),
        stderr=stderr.getvalue(), files=files)


@contextmanager
def _caller_mask(mask):
    original = os.sched_getaffinity(0)
    os.sched_setaffinity(0, mask)
    try:
        if os.sched_getaffinity(0) != mask:
            raise ValueError('caller CPU mask differs from the requested validation mask')
        yield
    finally:
        os.sched_setaffinity(0, original)


def _validate(admission, reference, directory, *, argv, output_token, validate_resources):
    from scar.codegen.stateful import load
    from scar.compiler.stateful_holdout import _owner, _stats
    from scar.compiler.stateful_runtime import _function, install_stateful_regions

    if (type(argv) is not list or any(type(arg) is not str for arg in argv)
            or (output_token is not None and (not output_token or argv.count(output_token) != 1))):
        raise ValueError('explicit string argv and an exact optional output token required')
    protocol = admission['entry_protocol']
    candidate = read_artifact(admission['candidate'])
    entry = admission['entry']
    before_affinity = os.sched_getaffinity(0)
    resource_reference = admission['native_resource_certificate']
    resource = None
    if validate_resources:
        if resource_reference is None:
            raise ValueError('resource validation requires a verified native resource certificate')
        resource = read_artifact(resource_reference)
        native_mask = set(resource['native_cpu_ids'])
        reduced_mask = before_affinity - native_mask
        if not native_mask or not native_mask <= before_affinity or not reduced_mask:
            raise ValueError('resource validation needs both certified CPUs and other available caller CPUs')
    artifact = load(candidate['native_directory'])
    region = artifact.region
    original_source = protocol['original_source']['artifact']
    path = Path(original_source['uri'])
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path.parent))
    try:
        module = importlib.import_module(path.stem)
        if Path(vars(module)['__file__']).resolve() != path:
            raise ValueError('ordinary import resolved a different source module')
        owner, name = _owner(module, region)
        descriptor = vars(owner)[name]
        original = descriptor.__func__ if type(descriptor) is staticmethod else descriptor
        _function(region.source, original)
        baseline = _episode(module, region, original, original, entry['uri'], argv,
            output_token, directory/'original')
        with install_stateful_regions(module, [artifact], **{key:protocol['configuration'][key]
                for key in ('trial', 'exclusive_inputs', 'selection')}) as (guard,):
            from scar.trace.code_snapshot import code_key

            if code_key(guard.__code__) != protocol['shared_call_code_key']:
                raise ValueError('live wrapper differs from the canonical compiler entry')
            actual = _episode(module, region, original, guard, entry['uri'], argv,
                output_token, directory/'candidate')
            stats = _stats(guard)
            resource_validation = None
            if resource is not None:
                if os.sched_getaffinity(0) != before_affinity:
                    raise ValueError('successful native execution changed the caller CPU mask')
                with _caller_mask(reduced_mask):
                    rejected = _episode(module, region, original, guard, entry['uri'], argv,
                        output_token, directory/'resource-fallback')
                    if os.sched_getaffinity(0) != reduced_mask:
                        raise ValueError('resource fallback changed the reduced caller CPU mask')
                after = _stats(guard)
                exact_reasons = dict(stats['reasons'])
                affinity_reason = 'native_precommit_guard:stateful precommit guard: CPU affinity'
                expected_fallbacks = len(rejected['trajectory'])
                exact_reasons[affinity_reason] = exact_reasons.get(affinity_reason, 0) + expected_fallbacks
                checks = dict(
                    native_declined_before_commit=after['native_calls'] == stats['native_calls'],
                    exact_affinity_fallback=after['fallback_calls'] - stats['fallback_calls'] == expected_fallbacks
                        and after['reasons'] == exact_reasons,
                    original_trajectory_and_effects_match=rejected == baseline)
                resource_validation = dict(resource_certificate=resource_reference,
                    certified_native_cpu_ids=resource['native_cpu_ids'],
                    original_caller_cpu_ids=sorted(before_affinity), fallback_caller_cpu_ids=sorted(reduced_mask),
                    native_calls_with_certified_resources=stats['native_calls'],
                    resource_guard_fallback_calls=after['fallback_calls'] - stats['fallback_calls'],
                    guard_statistics_after_fallback=after,
                    restored_caller_cpu_ids=sorted(os.sched_getaffinity(0)),
                    checks=checks, validated=all(checks.values()),
                    assumption='ORIGINAL_CPP_STAGE_GUARDS_AND_SCOPED_ARRAY_PAYLOAD',
                    complete_physical_timing_inferred=False)
        restored = vars(owner)[name] is descriptor
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
    for source in (entry, original_source, *(row['artifact'] for row in protocol['callee_sources'])):
        read_artifact(source, json_document=False)
    checks = {key:baseline[key] == actual[key] for key in baseline}
    checks.update(original_binding_restored=restored, caller_affinity_restored=os.sched_getaffinity(0) == before_affinity,
        native_success=stats['native_calls'] == len(actual['trajectory']) and stats['fallback_calls'] == 0)
    if resource_validation is not None:
        checks['certified_resources_validated'] = resource_validation['validated']
    return _write(directory/'validation.json',
        dict(schema=VALIDATION, schema_version=1, admission=reference,
            entry_protocol=protocol['id'], entry=entry, argv=argv, output_token=output_token,
            comparisons=checks, original=baseline, candidate=actual,
            guard_statistics=stats, validated=all(checks.values()), source_bytes_unchanged=True,
            resource_validation=resource_validation,
            snapshot_scope='DECLARED_ARRAY_BYTES_AND_SCALAR_TYPES_EACH_METHOD_ENTRY_EXIT_AND_POST_DRIVER',
            file_scope='EXPLICIT_OUTPUT_DIRECTORY', performance_measured=False, decision='KEEP',
            deployment_authorized=False))


def prepare(preparation, model, out, *, validation_argv=None, output_token=None, validate_resources=False):
    """Build a delivery receipt and optionally run the two unchanged entry paths."""
    from scar.compiler.calibrate_stateful_entry import SCHEMA as MODEL
    from scar.compiler.plan_controlled_region import SCHEMA as PLAN

    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh stateful delivery directory required')
    if validate_resources and validation_argv is None:
        raise ValueError('resource validation requires an explicit driver trial')
    admission = _derive(_reference(preparation, PLAN), _reference(model, MODEL))
    directory.mkdir(parents=True)
    reference = _write(directory/'admission.json', admission)
    return (reference if validation_argv is None else
        _validate(admission, reference, directory, argv=validation_argv, output_token=output_token,
            validate_resources=validate_resources))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation')
    parser.add_argument('--canonical-model')
    parser.add_argument('--out')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--validate-resources', action='store_true',
        help='also compare precommit fallback with the certified native CPUs unavailable')
    parser.add_argument('--argv-json')
    parser.add_argument('--output-token')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        document = check(args.check)
        result = dict(replayed=True, decision=document['decision'], deployment_authorized=False)
    else:
        if not all((args.preparation, args.canonical_model, args.out)):
            parser.error('--preparation, --canonical-model and --out are required')
        if args.validate != (args.argv_json is not None):
            parser.error('--validate and --argv-json must be supplied together')
        result = prepare(args.preparation, args.canonical_model, args.out,
            validation_argv=None if args.argv_json is None else json.loads(args.argv_json),
            output_token=args.output_token, validate_resources=args.validate_resources)
        if args.validate and not read_artifact(result)['validated']:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
