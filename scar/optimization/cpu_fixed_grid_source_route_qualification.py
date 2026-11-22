"""Join source-level route evidence without making a physical timing claim."""

import argparse
from pathlib import Path

from scar.analysis import cpu_fixed_grid_startup_graph as startup_graph
from scar.compiler import cpu_fixed_grid_publication_guard_study as guard_study
from scar.compiler import cpu_fixed_grid_source_route_benchmark as source_benchmark
from scar.compiler import observe_cpu_fixed_grid_publication as publication_observation
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import _canonical, read_artifact
from scar.optimization import cpu_fixed_grid_input_route_review as input_route
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN


SCHEMA = 'scar.cpu-fixed-grid-source-route-qualification'


def _require(condition, message):
    if not condition:
        raise ValueError('fixed-grid source route qualification: ' + message)


def derive_cpu_fixed_grid_source_route_qualification(route_ref, *, startup_ref, publication_ref, guard_ref, benchmark_ref):
    route = input_route.verify_cpu_fixed_grid_input_route_review(route_ref)
    startup = startup_graph.verify_cpu_fixed_grid_startup_graph(startup_ref)
    publication = publication_observation.verify_cpu_fixed_grid_publication(publication_ref)
    guard = guard_study.verify_cpu_fixed_grid_publication_guard_study(guard_ref)
    benchmark = source_benchmark.verify_cpu_fixed_grid_source_route_benchmark(benchmark_ref)
    scope = startup['scope']
    package = read_artifact(route['package'])
    shared = dict(source=route['source'], target=route['target'], argv=route['argv'],
                  cpu_ids=scope['cpu_ids'], environment_sha256=scope['environment_sha256'])
    _require(route['native_route'] == 'PROVIDER_MAP' and startup['lifecycle'] == route['lifecycle']
             and all(startup[key] == shared[key] for key in ('source', 'target'))
             and all(scope[key] == package[key] for key in ('argv', 'cpu_ids', 'environment_sha256', 'working_directory'))
             and scope['boundary'] == 'PROCESS_LAUNCH_THROUGH_EXIT'
             and all(all(witness[key] == value for key, value in shared.items())
                     for witness in (publication, guard, benchmark))
             and publication['package'] == guard['route_package'] == benchmark['package'] == route['package']
             and publication['delivery'] == benchmark['original_delivery'] == route['cli_delivery']
             and benchmark['native_project'] == guard['project'] == route['native_project']
             and benchmark['native_executable'] == route['native_executable']
             and guard['working_directory'] == benchmark['working_directory'] == scope['working_directory']
             and benchmark['publication'] == package['publication']
             and benchmark['path_state_at_start'] == 'ABSENT_EXACT_PUBLICATION_PATH',
             'startup, effects and clean timing must identify the same original source scope and native route')

    python = publication['arms']['ORIGINAL_PYTHON']['trace_projection']
    native = publication['arms']['SELECTED_CPP']['trace_projection']
    selection = benchmark['selection']
    _require(selection['policy'].get('strict_minimum_gain') is True
             and selection['policy']['minimum_gain'] == DEFAULT_MINIMUM_GAIN,
             'normalized current strict gain policy required')
    output_match = publication['matching_file_metadata'] and publication['return_content_and_streams_verified']
    guard_effect = (guard['native_existing_file_preserved'] and guard['original_existing_file_overwritten']
                    and not guard['original_user_output_written'])
    observed = (selection['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW' and output_match and guard_effect
                and startup['declared_request_observation_coverage_complete'])
    metrics = selection['metrics']
    original = metrics[selection['incumbent']]['median_ns_twice']
    candidate = metrics[selection['challenger']]['median_ns_twice']
    startup_roles = dict(original=python['external_exec_images'], native=native['external_exec_images'])
    transient = dict(original=python['observed_same_task_create_unlink'], native=native['observed_same_task_create_unlink'])
    return dict(schema=SCHEMA, schema_version=1, input_route_review=route_ref, startup=startup_ref,
        publication=publication_ref, occupied_output_guard=guard_ref, benchmark=benchmark_ref, **shared,
        package=route['package'], cli_delivery=route['cli_delivery'], lifecycle=route['lifecycle'],
        native_project=route['native_project'], native_executable=route['native_executable'],
        scope='EXACT_FRESH_OUTPUT_AND_STABLE_INPUT_FULL_PROCESS_NORMAL_SUCCESS',
        status='SCOPED_NATIVE_SOURCE_ROUTE_REVIEW_ELIGIBLE' if observed else 'SCOPED_SOURCE_ROUTE_REVIEW_OPEN',
        observed_effects=dict(startup_requests=startup['summary'], other_exec_images=startup_roles,
            transient_file_effects=transient,
            output_metadata_matched=output_match, publication_close_before_stdout_observed={
                arm: row['trace_projection']['normal_close_before_stdout_observed']
                for arm, row in publication['arms'].items()},
            occupied_output_observed_difference=guard_effect),
        clean_full_process_timing=dict(basis='DIRECT_EMPIRICAL_PROCESS_ENVELOPES', metrics=metrics,
            policy=selection['policy'], verdict=selection['verdict'], warmup_pairs=selection['warmup_pairs'],
            measurement_pairs=selection['measurement_pairs'], observed_gain_fraction=selection['gain_fraction'],
            speedup=original/candidate, observed_envelopes_separated=selection['observed_envelopes_separated'],
            physical_time_bound=False, used_as_forward_model_bound=False),
        coverage_frontier=[
            dict(role='STARTUP_MODULE_RESOLUTION', status=startup['status'], evidence=startup_ref),
            dict(role='STARTUP_AND_RUNTIME_EFFECTS', status='OPEN',
                 blockers=startup['coverage_frontier'] + publication['unobserved_effect_surfaces']),
            dict(role='FRESH_FILE_PUBLICATION', status='MATCHED_OBSERVED_EFFECTS' if output_match else 'OPEN',
                 blockers=['CONCURRENT_OBSERVERS', 'STORAGE_DURABILITY', 'WRITE_FAILURES']),
            dict(role='OCCUPIED_FILE_PUBLICATION', status='OUTSIDE_SCOPE_OBSERVED_DIVERGENCE' if guard_effect else 'OPEN'),
            dict(role='FORWARD_COST', status='PARTIAL_LIFECYCLE_MODEL_ONLY')],
        whole_entry_effect_equivalence_proven=False, original_entry_action='KEEP_ORIGINAL',
        deployment_authorized=False, target_executed=False)


def verify_cpu_fixed_grid_source_route_qualification(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'versioned qualification required')
    replay = derive_cpu_fixed_grid_source_route_qualification(saved['input_route_review'], startup_ref=saved['startup'],
        publication_ref=saved['publication'], guard_ref=saved['occupied_output_guard'], benchmark_ref=saved['benchmark'])
    _require(_canonical(saved) == _canonical(replay), 'scoped source route qualification differs from cold replay')
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('route', 'startup', 'publication', 'guard', 'benchmark', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = derive_cpu_fixed_grid_source_route_qualification(
        _ref(args.route, input_route.SCHEMA), startup_ref=_ref(args.startup, startup_graph.SCHEMA),
        publication_ref=_ref(args.publication, publication_observation.SCHEMA),
        guard_ref=_ref(args.guard, guard_study.SCHEMA), benchmark_ref=_ref(args.benchmark, source_benchmark.SCHEMA))
    _write(args.out, result)
    print(args.out)
    return 0
