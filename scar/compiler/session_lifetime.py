"""Publish dynamic session lifetimes from cold-replayed source/resource parents."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-session-lifetime-preparation'


def _reference(path, schema):
    path = Path(path).resolve()
    return artifact_reference(path/'result.json' if path.is_dir() else path, schema)


def _pins():
    root = Path(__file__).resolve().parents[1]
    return {name: artifact_reference(root/name, 'scar.python-source') for name in (
        'compiler/session_lifetime.py', 'optimization/session_lifetime.py',
        'compiler/inspect_workspace_session.py', 'compiler/bind_time_resources.py')}


def _derive(session_reference, resources_reference):
    from scar.compiler.inspect_workspace_session import inspect_workspace_session
    from scar.compiler.bind_time_resources import read_resource_bindings
    from scar.ir.temporal_resources import projection_from_dict
    from scar.ir.versioned_schedule import schedule_from_dict
    from scar.optimization.session_lifetime import derive_session_lifetime

    check, context = inspect_workspace_session(session_reference['uri'], with_context=True)
    if check['compiled'] != session_reference:
        raise ValueError('compiled session publication changed during cold replay')
    receipt = read_resource_bindings(resources_reference['uri'])
    if receipt != read_artifact(resources_reference):
        raise ValueError('resource publication changed during cold replay')
    time = read_artifact(receipt['time_schedule'])
    schedule = schedule_from_dict(read_artifact(time['schedule']))
    resources = projection_from_dict(read_artifact(receipt['projection']))
    inputs = dict(compiled_session=session_reference, resource_preparation=resources_reference,
        session_preparation=check['preparation'], session_plan=check['session_plan'],
        session_native=check['native'], original_native=context['plan']['parent_native_manifest'],
        kernel_time_preparation=receipt['time_schedule'], kernel_schedule=time['schedule'],
        kernel_resources=receipt['projection'])
    source_refs = tuple(dict(role=role, artifact=ref, locator='#') for role, ref in inputs.items())
    projection = derive_session_lifetime(context['plan'], schedule, resources,
        session_plan_ref=inputs['session_plan'], kernel_schedule_ref=inputs['kernel_schedule'],
        kernel_resources_ref=inputs['kernel_resources'], native_reference=inputs['session_native'],
        native_metadata=context['native'], parent_native_metadata=context['original']['native_metadata'],
        source_refs=source_refs)
    for ref in inputs.values():
        read_artifact(ref)
    return projection, inputs


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _report(inputs, projection_reference, projection):
    return dict(schema=SCHEMA, schema_version=1, inputs=inputs,
        projection=projection_reference, implementations=_pins(),
        representation=projection['representation'],
        scope='ONE_SUCCESSFUL_DYNAMIC_SESSION_LEASE_WITH_SHARED_STEP_TEMPLATE',
        summary=dict(owned_array_slots=len(projection['slot_lifetimes']),
            owned_array_payload_bytes=projection['memory']['fixed_owned_array_payload_bytes'],
            shared_body_activity_references=len(projection['shared_step_template']['original_activities']),
            step_count=None, invalidate_count=None, auxiliary_storage_bytes=None,
            complete_session_storage_bytes=None, action_durations_known=False,
            outer_action_cpu_domain_known=False, fixed_parent_count_used_as_session_count=False,
            invalidate_releases_capacity=False),
        source_caller_effect_coverage_proven=False, caller_replacement_authorized=False,
        physical_resource_feasibility_proven=False, target_program_executed=False,
        native_compiled=False, native_libraries_loaded=False, solver_invoked=False,
        performance_measured=False, default_backend='NO_OP', deployment_authorized=False)


def prepare(compiled_session, resources_receipt, out):
    """Cold-admit each parent once and publish a persistent pure projection."""
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh session lifetime output required')
    session = _reference(compiled_session, 'scar.compiled-workspace-session')
    resources = _reference(resources_receipt, 'scar.temporal-resource-preparation')
    projection, inputs = _derive(session, resources)
    output.mkdir(parents=True, exist_ok=False)
    report = _report(inputs, _write(output/'projection.json', projection), projection)
    _write(output/'result.json', report)
    return report


def replay(receipt):
    """Re-admit both parents and fully rederive; return persistent report/data."""
    reference = _reference(receipt, SCHEMA)
    report = read_artifact(reference)
    projection, inputs = _derive(report['inputs']['compiled_session'],
        report['inputs']['resource_preparation'])
    retained = read_artifact(report['projection'])
    expected = _report(inputs, report['projection'], projection)
    if (semantic_sha256(report) != semantic_sha256(expected)
            or semantic_sha256(retained) != semantic_sha256(projection)):
        raise ValueError('session lifetime differs from source/native/resource parent replay')
    read_artifact(reference)
    return report, retained


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--session')
    source.add_argument('--check')
    parser.add_argument('--resources')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.resources or args.out:
            parser.error('check consumes only the saved session lifetime receipt')
        report = replay(args.check)[0]
    else:
        if not args.resources or not args.out:
            parser.error('session lifetime preparation requires --resources and --out')
        report = prepare(args.session, args.resources, args.out)
    print(json.dumps(dict(summary=report['summary'], projection=report['projection'],
        representation=report['representation'], deployment_authorized=False), sort_keys=True))
    return 0


__all__ = ['SCHEMA', 'prepare', 'replay', 'main']

if __name__ == '__main__':
    raise SystemExit(main())
