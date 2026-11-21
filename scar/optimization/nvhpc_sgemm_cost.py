"""Transfer matched NVIDIA SGEMM observations into the scoped CPU point model."""

from dataclasses import asdict
from statistics import median

from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.optimization.cpu_sgemm_probe import SCHEMA


def sgemm_body_costs(projection, references, *, stack, target, cpus):
    observations = {}
    library = stack['libraries']['cpu_blas']
    identity = {key: library[key] for key in ('path', 'sha256', 'size_bytes')}
    read_artifact(asdict(target), json_document=False)
    for reference in references:
        report = read_artifact(reference)
        if (reference['schema'] != SCHEMA or report['schema_version'] != 1 or report['target'] != asdict(target)
                or report['cpu_ids'] != list(cpus) or report['dtype'] != 'FP32'
                or report['layout'] != 'CONTIGUOUS_ROW_MAJOR' or report['operator'] != 'SGEMM'
                or (report['alpha'], report['beta']) != (1, 0)
                or report['sample_scope'] != 'WARM_PREALLOCATED_SINGLE_OPERATOR_CALL'):
            raise ValueError('SGEMM observation differs from the source ABI, CPU cohort or hardware target')
        rows = [(index, row) for index, row in enumerate(report['providers'])
                if row['library'] == identity and row['symbol'] == 'cblas_sgemm' and row['integer_bits'] == 32
                and row['state'] == 'MEASURED' and row['threads'] == len(cpus)]
        dimensions = tuple(report['shape_mnk'])
        if len(rows) != 1 or dimensions in observations:
            raise ValueError('each observed SGEMM shape needs one unique pinned NVIDIA LP64 provider sample')
        index, row = rows[0]
        samples = row['samples_ns']
        if (len(samples) != report['repetitions'] or len(samples) < 3 or report['warmups'] < 1
                or any(type(ns) is not int or ns <= 0 for ns in samples) or int(median(samples)) < 1
                or (row['lower_sample_ns'], row['median_ns'], row['upper_sample_ns']) !=
                   (min(samples), median(samples), max(samples))):
            raise ValueError('invalid SGEMM cost samples')
        observations[dimensions] = CostEstimate(int(median(samples)), None, None, 'EMPIRICAL_MODEL',
            'MATCHED_WARM_CPU_SGEMM_POINT_FORECAST', (dict(artifact=reference, locator=f'#/providers/{index}'),))
    return {row['id']: observations.get((*row['shape'], row['reduction']), CostEstimate(None, None, None,
            'UNKNOWN', 'CPU_SGEMM_COMPONENT_UNPRICED', missing_terms=('MATCHED_WARM_SGEMM_OBSERVATION',)))
            for row in projection['operations']}
