"""Bind fixed-grid input graph, cost observations and the chosen native project."""

from scar.analysis.cpu_fixed_grid_input_route_graph import verify_cpu_fixed_grid_input_route_graph
from scar.codegen.cpu_fixed_grid_native_route_package import verify_cpu_fixed_grid_native_route_package
from scar.compiler.observe_cpu_fixed_grid_input_counters import verify_cpu_fixed_grid_input_counters
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fixed-grid-input-route-review'


def _require(condition, reason):
    if not condition:
        raise ValueError('CPU fixed-grid input route review: ' + reason)


def derive_cpu_fixed_grid_input_route_review(graph_ref, package_ref, *, counters_ref=None):
    graph = verify_cpu_fixed_grid_input_route_graph(graph_ref)
    package = verify_cpu_fixed_grid_native_route_package(package_ref)
    lifecycle = read_artifact(graph['lifecycle'])
    selected = package['selected_runtime']
    _require(package['plan'] == graph['mapped_plan'] and package['owned_copy_delivery'] == lifecycle['delivery']
             and selected == graph['selected_runtime'] and package['hardware'] == graph['target']
             and package['source'] == graph['source'] and package['argv'] == lifecycle['scope']['argv']
             and package['routes']['OWNED_COPY']['project'] == graph['owned_project']
             and package['routes']['PROVIDER_MAP']['project'] == graph['mapped_project']
             and package['original_python_action'] == 'KEEP_ORIGINAL'
             and package['deployment_authorized'] is False,
             'input route graph and C++20 package must name one exact source CLI decision')
    counter_summaries = None
    if counters_ref is not None:
        counters = verify_cpu_fixed_grid_input_counters(counters_ref)
        _require(counters['mapped_plan'] == package['plan'] and counters['observation'] == package['observation']
                 and counters['control_delivery'] == package['owned_copy_delivery']
                 and counters['environment_sha256'] == package['environment_sha256']
                 and all(counters[key] == package[key] for key in ('source', 'argv', 'cpu_ids', 'working_directory'))
                 and counters['target'] == package['hardware'] and counters['clean_benchmark'] is False
                 and all(counters['routes'][name]['original_project'] == route['project']
                         and counters['routes'][name]['original_executable'] == route['executable']
                         for name, route in package['routes'].items()),
                 'phase counters must match both exact C++20 routes')
        counter_summaries = {
            route: {phase: {name: quantity['median'] for name, quantity in values.items()}
                    for phase, values in row['windows'].items()}
            for route, row in counters['routes'].items()}

    return dict(schema=SCHEMA, schema_version=1, graph=graph_ref, package=package_ref, counters=counters_ref,
                mapped_plan=package['plan'], cli_delivery=package['owned_copy_delivery'],
                lifecycle=graph['lifecycle'], target=package['hardware'], source=package['source'],
                argv=package['argv'], native_route=selected, native_project=package['selected_project'],
                native_executable=package['routes'][selected]['executable'],
                phase_counter_medians=counter_summaries, counters_used_for_ranking=False,
                native_scope_status='EXPLICIT_SCOPE_PRELAUNCH_GUARDS_REQUIRED',
                selected_python_action='KEEP_ORIGINAL', original_entry_replacement_authorized=False,
                deployment_authorized=False, target_executed=False)


def verify_cpu_fixed_grid_input_route_review(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and saved['schema_version'] == 1, 'unsupported native input route review')
    replayed = derive_cpu_fixed_grid_input_route_review(saved['graph'], saved['package'], counters_ref=saved['counters'])
    _require(saved == replayed, 'native input route review differs at cold replay')
    return replayed
