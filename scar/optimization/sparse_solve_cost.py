"""Project causal timing cuts from a measured sparse direct-solver study."""

import json
from pathlib import Path
from statistics import median

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.sparse-direct-solve-cost'
DESIGN = SCHEMA + '-design'
OBSERVATION = SCHEMA + '-observations'

_SECTIONS = ('source_solve', 'direct_csr', 'convert_csc', 'csc_factor', 'factored_solve')


def _duration(cuts):
    start, end = cuts
    if type(start) is not int or type(end) is not int or not 0 < start < end:
        raise ValueError('ordered positive perf_counter_ns cuts required')
    return end - start


def _summary(values):
    return dict(estimate_ns=int(median(values)), lower_ns=min(values), upper_ns=max(values),
                samples=len(values), bound_kind='observed_sample_envelope')


def project(design_path, observations_path):
    """Revalidate the receipts and summarize direct cuts, without extrapolating."""
    design_ref = artifact_reference(design_path, DESIGN)
    observations_ref = artifact_reference(observations_path, OBSERVATION)
    design = read_artifact(design_ref)
    observations = read_artifact(observations_ref)
    if (design['schema_version'] != 1 or observations['schema_version'] != 1
            or observations['design'] != design_ref or observations['context'] != design['context']):
        raise ValueError('solver study version or frozen context differs')
    scope = read_artifact(design['source_scope'])
    entry = scope['entry']
    count = entry['argv_repetitions']
    if (design['entry'] != entry['source'] or design['context']['cpu_ids'] != entry['affinity']
            or design['context']['argv'] != entry['argv']
            or any(design['context']['environment'][key] != value
                   for key, value in entry['thread_environment'].items())
            or scope['observed']['scope_entry_count'] != 1
            or scope['observed']['scope_exit_count'] != 1
            or scope['observed']['region_attempts'] != count
            or scope['observed']['consumer_completions'] != count):
        raise ValueError('solver samples differ from the admitted source scope')
    for reference in [design['entry'], design['harness'], design['fixture_parser'],
                      *design['provider_files'].values()]:
        read_artifact(reference, json_document=False)
    samples = observations['samples']
    indices = [sample['iteration'] for sample in samples]
    if (indices != design['indices'] or len(indices) < 3 or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or observations['data_domains']['iterations'] != count):
        raise ValueError('sparse solver sample coverage differs from the design')
    durations = {name: [] for name in (*_SECTIONS, 'csc_reference_total')}
    previous_end = 0
    for sample in samples:
        cuts = sample['clock_cuts_ns']
        if set(cuts) != set(_SECTIONS):
            raise ValueError('sparse solver cut names differ')
        ordered = sorted(cuts.values())
        actual_order = sorted(('source_solve', 'direct_csr', 'csc_reference'),
                              key=lambda name: cuts['convert_csc' if name == 'csc_reference' else name][0])
        if (cuts['convert_csc'][1] != cuts['csc_factor'][0]
                or cuts['csc_factor'][1] != cuts['factored_solve'][0]
                or sample['call_order'] != actual_order
                or ordered[0][0] < previous_end
                or any(left[1] > right[0] for left, right in zip(ordered, ordered[1:]))):
            raise ValueError('sparse solver cuts differ from the ordered study boundaries')
        previous_end = ordered[-1][1]
        for name, section in cuts.items():
            durations[name].append(_duration(section))
        durations['csc_reference_total'].append(cuts['factored_solve'][1] - cuts['convert_csc'][0])
    return dict(schema=SCHEMA, schema_version=1, design=design_ref, observations=observations_ref,
                source_scope=design['source_scope'], context=design['context'],
                provider=observations['provider'], hardware=observations['hardware'],
                data_domains=observations['data_domains'],
                timings={name: _summary(values) for name, values in durations.items()},
                correctness=dict(direct_csr_all_bitwise=all(s['direct_csr_bitwise'] for s in samples),
                                 csc_reference_all_bitwise=all(s['csc_reference_bitwise']
                                                               for s in samples)),
                attribution=dict(source_solve='Python solver selection and default one-shot solve',
                                 direct_csr='separate public SciPy CSR spsolve call',
                                 csc_reference='separate CSC public conversion/factor/solve path',
                                 gssv_internal_ordering_factor_solve_ns=None,
                                 csc_factor_symbolic_and_numeric_split_ns=None),
                decision=dict(automatic_rewrite=False, full_entry_cost_ns=None,
                              reusable_numeric_factor_proven=False,
                              compatible_symbolic_reuse_abi_proven=False))


def check(path):
    """Cold-check a published projection from pinned, raw clock evidence."""
    path = Path(path)
    result = read_artifact(artifact_reference(path, SCHEMA))
    replayed = project(result['design']['uri'], result['observations']['uri'])
    if result != replayed:
        raise ValueError('sparse solver cost projection does not replay')
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.result), sort_keys=True))
