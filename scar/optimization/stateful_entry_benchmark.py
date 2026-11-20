"""Replay a clean paired stateful recipe comparison at its measured boundary."""
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.current-stateful-entry-benchmark'
_MODES = ('ORIGINAL', 'GUARDED_NATIVE_TRIAL')


def replay_entry_benchmark(reference, recipe_reference):
    report = read_artifact(reference)
    design = read_artifact(report['design'])
    material = design['material']
    if (reference['schema'] != SCHEMA or report['schema_version'] != 1
            or design['schema'] != SCHEMA + '.design' or design['schema_version'] != 1
            or material['recipe'] != recipe_reference
            or report['sys_profile_and_sys_monitoring_enabled'] is not False
            or report['fresh_process_startup_cost_established'] is not False
            or report['clean_entry_output_files_equal'] is not True):
        raise ValueError('same-recipe clean protected-region and preloaded entry comparisons required')
    for source in material.values():
        read_artifact(source, json_document=False)
    recipe = read_artifact(recipe_reference)
    lowering = read_artifact(material['lowering'])
    if (report['recipe_id'] != recipe['recipe_id'] or lowering['recipe_id'] != recipe['recipe_id']
            or lowering['recipe_sha256'] != recipe_reference['sha256']
            or lowering['native_library_sha256'] != material['native_member:library']['sha256']):
        raise ValueError('comparison lowering differs from the retained recipe')
    replayed = {}
    warmup, measured = design['warmup_pairs'], design['measurement_pairs']
    for scope in ('entry', 'method'):
        raw = report['raw'][scope]
        correctness = raw['correctness']
        rows = raw['rows']
        if len(correctness) != warmup + measured or len(rows) != 2 * len(correctness):
            raise ValueError('comparison does not retain every original paired repetition')
        for index, observation in enumerate(correctness):
            pair = rows[2 * index:2 * index + 2]
            if (observation['pair'] != index or observation['fallback_calls'] != 0
                    or observation['native_calls'] != report['calls_per_episode']
                    or any(row['pair'] != index for row in pair)
                    or pair[0]['stdout_sha256'] != pair[1]['stdout_sha256']
                    or pair[0]['stderr_sha256'] != pair[1]['stderr_sha256']):
                raise ValueError('paired timing context or output differs')
            if scope == 'method' and (observation['final_numeric_state_equal'] is not True
                    or any(sum(observation['method_calls_ns'][row['mode']]) != row['wall_ns'] for row in pair)):
                raise ValueError('region comparison does not reproduce the native call boundaries')
        replayed[scope] = compare_entry_pairs(rows, incumbent=_MODES[0], challenger=_MODES[1],
            warmup_pairs=warmup, measurement_pairs=measured, minimum_gain=design['minimum_gain'])
    if replayed != report['result']:
        raise ValueError('comparison verdict differs from the retained clean repetitions')
    read_artifact(reference)
    return dict(argv=design['argv'], cpu_ids=design['request']['caller_cpu_ids'],
        calls_per_episode=report['calls_per_episode'], warmup_pairs=warmup, measurement_pairs=measured,
        entry_verdict=replayed['entry']['verdict'], method_verdict=replayed['method']['verdict'],
        entry_timer=design['entry_timer'], method_timer=design['method_timer'],
        measured_cost_transferred_to_schedule=False, deployment_authorized=False)
