"""L4 compute-order certificate from a completely replayed L3 finite domain."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference
from scar.optimization.pipeline_execution_refinement import refine_selected_pipeline_temporal


SCHEMA = 'scar.pipeline-execution-refinement-publication'


def refine_pipeline_temporal(domain, out):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh L4 output directory required')
    certificate = refine_selected_pipeline_temporal(domain)
    out.mkdir(parents=True)
    path = out/'certificate.json'
    path.write_text(json.dumps(certificate, sort_keys=True, indent=2, allow_nan=False)+'\n')
    report = dict(schema=SCHEMA, schema_version=1, layer='L4_COMPUTE_ORDER_EXECUTION_REFINEMENT',
        original_domain=certificate['original_domain'], selected_case_id=certificate['selected_case_id'],
        certificate=artifact_reference(path, certificate['schema']),
        native_compiled=False, runtime_order_executed=False, full_model_timeline_implemented=False,
        target_program_executed=False, performance_measured=False, deployment_authorized=False)
    (out/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = refine_pipeline_temporal(args.domain, args.out)
    print(json.dumps(dict(layer=result['layer'], selected_case_id=result['selected_case_id'],
        certificate=result['certificate']['uri'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
