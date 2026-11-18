"""Replay small paired CSR solver measurements with a frozen source and ABI."""

from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_solve_cost import _duration, _summary

SCHEMA = 'scar.sparse-symbolic-paired-cost'
DESIGN = SCHEMA + '-design'
WORKER = SCHEMA + '-worker'


def project(design_path, worker_paths):
    design_ref = artifact_reference(design_path, DESIGN)
    design = read_artifact(design_ref)
    solve_scope = read_artifact(design['solve_scope'])
    source_scope = read_artifact(solve_scope['source_scope'])
    native = read_artifact(design['native'])
    carrier = solve_scope['source_call']['carrier']['library']
    if (design['schema_version'] != 1 or native['abi_version'] != 1
            or any(native['wheel'][key] != carrier[key] for key in ('uri', 'sha256', 'size_bytes'))
            or design['entry'] != source_scope['entry']['source']
            or design['context']['argv'] != source_scope['entry']['argv']
            or design['context']['cpu_ids'] != source_scope['entry']['affinity']
            or any(design['context']['environment'][key] != value
                   for key, value in source_scope['entry']['thread_environment'].items())):
        raise ValueError('symbolic measurement source, native ABI or launch scope differs')
    for reference in (design['entry'], design['runtime_adapter'], design['fixture_parser'],
                      design['harness'], carrier, native['source'], native['wheel'], native['library'],
                      native['blas']):
        read_artifact(reference, json_document=False)
    binding = {}
    if 'execution_recipe' in design:
        from scar.optimization.sparse_solve_recipe import verify_sparse_solve_recipe

        reference = design['execution_recipe']
        recipe = verify_sparse_solve_recipe(read_artifact(reference))
        if (recipe['artifacts']['source_scope'] != design['solve_scope']
                or recipe['artifacts']['blas'] != native['blas']
                or recipe['resources']['selected_cpu']['cpu_ids'] != design['context']['cpu_ids']
                or recipe['resources']['memory']['explicit_budget_bytes'] != design['budget_bytes']):
            raise ValueError('symbolic measurements do not use the recipe scope, hardware or budget')
        binding['execution_recipe'] = reference
    refs = [artifact_reference(path, WORKER) for path in worker_paths]
    groups = design['groups']
    if (not groups or len(groups) != len(refs) or
            len(set(item['uri'] for item in refs)) != len(refs)):
        raise ValueError('one unique worker per frozen measurement group required')

    diagnostics = []
    timings = {key: [] for key in ('load_once', 'close_once', 'original_first', 'native_first',
                                   'original_warm', 'native_warm', 'symbolic_build', 'first_reference')}
    savings = {key: [] for key in ('first', 'warm', 'observed_scope')}
    retained = []
    all_bitwise = True
    all_unchanged = True
    misses = 0
    qualified_scopes = 0
    for group, reference in zip(groups, refs):
        raw = read_artifact(reference)
        samples = raw['samples']
        if (raw['schema_version'] != 1 or raw['group'] != group or raw['design'] != design_ref
                or raw['context'] != design['context']
                or [sample['iteration'] for sample in samples] != group['iterations']):
            raise ValueError('measured worker identity or iteration order differs')
        entry_cost = _duration(raw['load_clock_ns'])
        close_cost = _duration(raw['close_clock_ns'])
        timings['load_once'].append(entry_cost)
        timings['close_once'].append(close_cost)
        whole_group = -entry_cost - close_cost
        previous_end = raw['load_clock_ns'][1]
        for position, sample in enumerate(samples):
            phase = 'first' if position == 0 else 'warm'
            original_cuts, candidate_cuts = sample['original_clock_ns'], sample['native_clock_ns']
            order = sorted(('original', 'native'), key=lambda mode: sample[mode + '_clock_ns'][0])
            if (order != group['order'] or
                    min(original_cuts[0], candidate_cuts[0]) < previous_end or
                    max(original_cuts[1], candidate_cuts[1]) > raw['close_clock_ns'][0] or
                    max(original_cuts[0], candidate_cuts[0]) < min(original_cuts[1], candidate_cuts[1])):
                raise ValueError('paired solver clock order differs')
            original_ns, native_ns = _duration(original_cuts), _duration(candidate_cuts)
            previous_end = max(original_cuts[1], candidate_cuts[1])
            all_unchanged &= sample['both_inputs_unchanged']
            all_bitwise &= (sample['original_output_sha256'] is not None and
                            sample['original_output_sha256'] == sample['native_output_sha256'])
            if sample['original_info'] != 0 or sample['native_info'] != 0:
                misses += 1
            timings['original_'+phase].append(original_ns)
            timings['native_'+phase].append(native_ns)
            savings[phase].append(original_ns - native_ns)
            whole_group += original_ns - native_ns
        savings['observed_scope'].append(whole_group)
        stats = raw['stats']
        diagnostics.append(stats)
        if (raw['admission_counts'] == dict(builds=1, hits=len(samples)-1) and
                stats['native_calls'] == len(samples) and stats['fallback_calls'] == 0):
            qualified_scopes += 1
        if raw['symbolic_build_ns'] is not None:
            timings['symbolic_build'].append(raw['symbolic_build_ns'])
        if stats['initial_reference_ns'] is not None:
            timings['first_reference'].append(stats['initial_reference_ns'])
        if raw['symbolic_retained_bytes'] is not None:
            retained.append(raw['symbolic_retained_bytes'])
    return dict(schema=SCHEMA, schema_version=1, design=design_ref, workers=refs,
                solve_scope=design['solve_scope'], source_scope=solve_scope['source_scope'],
                native=design['native'], runtime_adapter=design['runtime_adapter'],
                context=design['context'], budget_bytes=design['budget_bytes'],
                timings={key: _summary(values) if values else None for key, values in timings.items()},
                paired_savings={key: _summary(values) for key, values in savings.items()},
                diagnostics=diagnostics,
                resource=dict(symbolic_retained_bytes=max(retained, default=None),
                              max_numeric_workspace_bytes=None),
                qualification=dict(paired_section_only=True, all_bitwise=all_bitwise,
                                   preserved_both_inputs=all_unchanged, nonzero_info_or_rejected=misses,
                                   scopes_with_one_build_and_expected_hits=qualified_scopes,
                                   full_entry_cost_ns=None, automatic_rewrite=False), **binding)


def check(path):
    saved = read_artifact(artifact_reference(Path(path), SCHEMA))
    replay = project(saved['design']['uri'], [reference['uri'] for reference in saved['workers']])
    if saved != replay:
        raise ValueError('paired symbolic cost study no longer replays')
    return saved
