"""Publish a source-replayed L4 CPU execution certificate."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference
from scar.optimization.pipeline_execution_refinement import (
    refine_pipeline_resource_domain, refine_selected_pipeline_temporal,
)


SCHEMA = 'scar.pipeline-execution-refinement-publication'


def refine_pipeline_temporal(domain, out, *, resource_events=False, resource_domain=None, trial_case=None):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh L4 output directory required')
    if (domain is None) == (resource_domain is None):
        raise ValueError('one original model or resource-event domain required')
    if resource_domain is not None and not resource_events:
        raise ValueError('resource-event domain requires resource-event mode')
    if trial_case is not None and resource_domain is None:
        raise ValueError('an explicit trial case requires its original resource-event domain')
    if resource_domain is not None:
        certificate = refine_pipeline_resource_domain(resource_domain, trial_case=trial_case)
    else:
        certificate = refine_selected_pipeline_temporal(domain, resource_events=resource_events)
    out.mkdir(parents=True)
    path = out/'certificate.json'
    path.write_text(json.dumps(certificate, sort_keys=True, indent=2, allow_nan=False)+'\n')
    report = dict(schema=SCHEMA, schema_version=1,
        layer=('L4_RESOURCE_EVENT_EXECUTION_REFINEMENT' if resource_events
               else 'L4_COMPUTE_ORDER_EXECUTION_REFINEMENT'),
        original_domain=certificate['original_domain'], selected_case_id=certificate['selected_case_id'],
        certificate=artifact_reference(path, certificate['schema']),
        native_compiled=False, runtime_order_executed=False, full_model_timeline_implemented=False,
        target_program_executed=False, performance_measured=False, deployment_authorized=False)
    (out/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--domain')
    source.add_argument('--resource-domain')
    parser.add_argument('--trial-case')
    parser.add_argument('--mode', choices=('compute-order', 'resource-events'), default='compute-order')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = refine_pipeline_temporal(args.domain, args.out,
        resource_events=args.mode == 'resource-events', resource_domain=args.resource_domain,
        trial_case=args.trial_case)
    print(json.dumps(dict(layer=result['layer'], selected_case_id=result['selected_case_id'],
        certificate=result['certificate']['uri'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
