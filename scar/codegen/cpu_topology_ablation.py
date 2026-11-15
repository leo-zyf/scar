"""Frozen independent topology/geometry experiment, never a selectable cost.

This is a small diagnostic ablation of an already rejected additive model.
It compiles three source-independent canonical DAGs using the same native
stage measurement harness. No target region is compiled, executed or fitted.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time

from scar.codegen.cpu_fused_cost import _Layout, _native, fused_features
from scar.ir.execution_recipe import artifact_reference, read_artifact


def _save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _cases():
    specs = [
        ('SERIAL_CHAIN', [('add', 'x0', 'x1'), ('sub', 'v0', 'x2'), ('mul', 'v1', 'a.scale'),
            ('add', 'v2', 'x3'), ('sub', 'v3', 'a.scale'), ('mul', 'v4', 'a.scale'),
            ('add', 'v5', 'a.scale'), ('add', 'v6', 'a.scale')]),
        ('BALANCED_FANIN', [('add', 'x0', 'x1'), ('add', 'x2', 'x3'), ('sub', 'v0', 'a.scale'),
            ('sub', 'v1', 'a.scale'), ('mul', 'v2', 'a.scale'), ('mul', 'v3', 'a.scale'),
            ('add', 'v4', 'v5'), ('add', 'v6', 'a.scale')]),
        ('TWO_INDEPENDENT_CHAINS', [('add', 'x0', 'x1'), ('sub', 'v0', 'a.scale'),
            ('mul', 'v1', 'a.scale'), ('add', 'v2', 'a.scale'), ('sub', 'x2', 'x3'),
            ('mul', 'v4', 'a.scale'), ('add', 'v3', 'v5'), ('add', 'v6', 'a.scale')])]
    result = []
    for index, (family, graph) in enumerate(specs):
        depth = {f'x{i}': 0 for i in range(4)}; depth['a.scale'] = 0
        statements = [f'const double x{i}=a.inputs[{i}][a.offsets[{i}]+row*a.strides[{i}]+col];' for i in range(4)]
        for i, (op, left, right) in enumerate(graph):
            depth['v'+str(i)] = 1 + max(depth[left], depth[right])
            statements.append(f'const double v{i}=({left}{dict(add="+",sub="-",mul="*")[op]}{right});')
        statements.append('a.output[row*a.cols+col]=v7;')
        counts = Counter(op for op, _, _ in graph)
        result.append(dict(id='topology_'+str(index), family=family, array_loads=4,
            arithmetic={op: counts[op] for op in ('add', 'sub', 'mul', 'div', 'neg')},
            dependent_operation_depth=depth['v7'], output_stores=1,
            graph=[dict(id='v'+str(i), op=op, inputs=[left, right]) for i, (op, left, right) in enumerate(graph)],
            body='\n'.join(statements)))
    return result


def _layout(cols, groups, alignment):
    stride = cols + (4 if alignment == 'ALIGNED_EVEN_STRIDE' else 3)
    return [dict(stride=stride, offset=(i//groups)*stride + (2 if alignment == 'ALIGNED_EVEN_STRIDE' else 1)*(i%2),
        storage_group=i % groups) for i in range(4)]


def _compiler_paths(native, cases):
    """Map static packed/scalar paths and GCC line remarks to each DAG."""
    source = Path(native['source']['path']).read_text().splitlines()
    assembly = Path(native['assembly']['path']).read_text().splitlines()
    remarks = Path(native['vectorization_remarks']['path']).read_text().splitlines()
    symbols = []; active = None
    for line in assembly:
        header = re.match(r'^[0-9a-f]+ <(.+)>:$', line)
        if header:
            active = dict(symbol=header[1], lines=[]); symbols.append(active)
        elif active is not None: active['lines'].append(line)
    result = []
    for case in cases:
        start = next(i+1 for i, line in enumerate(source) if 'void '+case['id']+'(' in line)
        end = next((i+1 for i in range(start, len(source)) if
            'void topology_' in source[i] or 'using Kernel=' in source[i]), len(source))
        matching = [entry for entry in symbols if case['id']+'(' in entry['symbol']]
        instructions = [m[1] for entry in matching for line in entry['lines']
            if (m := re.match(r'^\s*[0-9a-f]+:\s+([a-zA-Z][a-zA-Z0-9.]*)', line))]
        notes = [line for line in remarks if (m := re.search(r':(\d+):\d+:', line)) and start <= int(m[1]) < end]
        result.append(dict(primitive=case['id'], family=case['family'], source_lines=[start, end],
            packed_arithmetic={op: instructions.count(op+'pd') for op in ('add', 'sub', 'mul', 'div')},
            scalar_arithmetic={op: instructions.count(op+'sd') for op in ('add', 'sub', 'mul', 'div')},
            static_branch_instructions=sum(op.startswith('j') for op in instructions),
            gcc_remarks=notes, symbols=[entry['symbol'] for entry in matching],
            alias_versioning_reported=any('versioned' in note and 'alias' in note for note in notes),
            runtime_branch_sampled=False, complete_isa_or_register_pressure_proof=False))
    return result


def run_topology_ablation(profile, out):
    """One compile, 24 frozen configs, seven repeats and two warmups each."""
    profile = Path(profile).absolute(); out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh topology diagnostic output required')
    reference = artifact_reference(profile, 'scar.cpu-fused-cost-model'); original = read_artifact(reference)
    if original['schema_version'] != 2: raise ValueError('independent canonical model v2 required')
    context = deepcopy(original['resource_context']); cases = _cases(); capacity = original['cache_geometry']['reported_capacity_bytes']
    before_affinity = sorted(os.sched_getaffinity(0)); thread_env = context['native_thread_environment']
    if before_affinity != context['caller_affinity_cpu_ids'] or any(os.environ.get(key) != value for key, value in thread_env.items()):
        raise ValueError('exact original caller affinity and native thread environment required')
    if len(context['worker_cpu_ids']) != 4: raise ValueError('frozen diagnostic design requires original four-worker context')
    configurations = []
    for primitive, case in enumerate(cases):
        for groups in (1, 2):
            for alignment in ('ALIGNED_EVEN_STRIDE', 'SHIFTED_ODD_STRIDE'):
                for rows, cols in ((256, 384), (512, 640)):
                    parameters = dict(rows=rows, cols=cols, workers=4, arithmetic=case['arithmetic'],
                        array_loads=4, output_stores=1, dependent_operation_depth=case['dependent_operation_depth'],
                        operand_layouts=_layout(cols, groups, alignment), backing_storage_groups=groups)
                    features, geometry = fused_features(parameters, capacity)
                    configurations.append(dict(id='configuration-'+str(len(configurations)),
                        primitive=primitive, family=case['family'], alignment=alignment,
                        parameters=parameters, features=features, geometry_properties=geometry))
    random.Random(7131).shuffle(configurations)
    out.mkdir(parents=True)
    implementation = Path(__file__).resolve(); implementation_raw = implementation.read_bytes()
    started = time.time(); load_before = list(os.getloadavg())
    design = dict(schema='scar.cpu-topology-ablation-design', schema_version=1, created_unix_seconds=started,
        original_model=reference, resource_context=context, cases=cases, configurations=configurations,
        configuration_order_seed=7131, timing_order='FIXED_SHUFFLED_CONFIGURATIONS_SEVEN_CONTIGUOUS_SAMPLES_PER_CONFIG',
        repetitions=7, warmup=2, native_compilations=1, total_stage_executions=216,
        design_scope='DIAGNOSTIC_ABLATION_NOT_NEW_MODEL_ADMISSION_HOLDOUTS',
        implementation=dict(path=str(implementation), sha256=sha256(implementation_raw).hexdigest(), bytes=len(implementation_raw)),
        target_workload_executed=False, coefficients_refitted=False, selectable=False)
    _save(out/'design.json', design)
    design_ref = artifact_reference(out/'design.json', design['schema'])
    # The immutable design is retained before compilation or any measurement.
    handle, function, native = _native(context, out/'native', cases)
    compiled_at = time.time(); observations = []
    for configuration in configurations:
        p = configuration['parameters']; layout = _Layout(); layout.rows=p['rows']; layout.cols=p['cols']; layout.loads=4
        for i, descriptor in enumerate(p['operand_layouts']):
            layout.strides[i]=descriptor['stride']; layout.offsets[i]=descriptor['offset']; layout.groups[i]=descriptor['storage_group']
        samples=(ctypes.c_double*7)(); checksum=ctypes.c_uint64(); cpus=(ctypes.c_int32*2)()
        status=function(configuration['primitive'], ctypes.byref(layout), 7, 2, samples, ctypes.byref(checksum), cpus)
        if status != 0: raise ValueError('independent topology FP/affinity/allocation guard: '+str(status))
        values=list(samples)
        if any(not math.isfinite(value) or value < 0 for value in values): raise ValueError('finite nonnegative native clocks required')
        model_key='backing-groups-'+str(p['backing_storage_groups']); model=original['local_models'][model_key]
        prediction=sum(configuration['features'][key]*coefficient for key, coefficient in model['coefficients'].items())
        median=statistics.median(values)
        observations.append(dict(configuration=configuration['id'], family=configuration['family'], alignment=configuration['alignment'],
            parameters=p, features=configuration['features'], geometry_properties=configuration['geometry_properties'],
            samples_ns=values, median_ns=median, min_ns=min(values), max_ns=max(values), stdev_ns=statistics.stdev(values),
            checksum=checksum.value, caller_cpu_before=cpus[0], caller_cpu_after=cpus[1],
            original_model= dict(artifact=reference, locator='#/local_models/'+model_key), original_model_admitted=model['admitted'],
            original_prediction_ns=prediction, residual_ns=prediction-median, relative_error=abs(prediction-median)/max(median, 1),
            diagnostic_only=True))
        _save(out/'raw-observations.json', dict(schema='scar.cpu-topology-ablation-observations', schema_version=1,
            frozen_design=design_ref, native=native, observations=observations))
    finished = time.time()
    after_affinity=sorted(os.sched_getaffinity(0))
    if (before_affinity != after_affinity or any(os.environ.get(key) != value for key, value in thread_env.items())
            or artifact_reference(profile, reference['schema']) != reference
            or artifact_reference(out/'design.json', design['schema']) != design_ref):
        raise ValueError('retained profile/design or caller environment changed during diagnostic experiment')
    comparisons = []
    for groups in (1, 2):
        for alignment in ('ALIGNED_EVEN_STRIDE', 'SHIFTED_ODD_STRIDE'):
            for shape in ((256, 384), (512, 640)):
                rows = [row for row in observations if row['parameters']['backing_storage_groups']==groups
                    and row['alignment']==alignment and (row['parameters']['rows'], row['parameters']['cols'])==shape]
                lo=min(rows, key=lambda row:row['median_ns']); hi=max(rows, key=lambda row:row['median_ns'])
                comparisons.append(dict(backing_storage_groups=groups, alignment=alignment, shape=shape,
                    family_medians_ns={row['family']: row['median_ns'] for row in rows},
                    family_relative_errors={row['family']: row['relative_error'] for row in rows},
                    largest_median_ratio=hi['median_ns']/max(lo['median_ns'], 1),
                    fastest_slowest_repeat_ranges_disjoint=max(lo['samples_ns']) < min(hi['samples_ns']),
                    causal_attribution_proven=False))
    family_errors = {}
    for case in cases:
        rows = [row for row in observations if row['family']==case['family']]
        family_errors[case['family']] = dict(maximum_relative_error=max(row['relative_error'] for row in rows),
            median_relative_error=statistics.median(row['relative_error'] for row in rows),
            overpredictions=sum(row['residual_ns']>0 for row in rows), observation_count=len(rows))
    report = dict(schema='scar.cpu-topology-ablation', schema_version=1, frozen_design=design_ref,
        raw_observations=artifact_reference(out/'raw-observations.json', 'scar.cpu-topology-ablation-observations'),
        status='OBSERVED_DIAGNOSTIC_ONLY', configuration_count=len(observations),
        compilation_seconds=compiled_at-started, measurement_and_recording_seconds=finished-compiled_at,
        seconds=finished-started, observed_until_unix_seconds=finished,
        comparisons=comparisons, original_model_errors_by_topology=family_errors,
        native=native, compiler_paths=_compiler_paths(native, cases),
        caller_affinity_before=before_affinity, caller_affinity_after=after_affinity,
        caller_affinity_restored=True, host_load=dict(before=load_before, after=list(os.getloadavg()), isolated=False),
        interpretation='Compare dependency structures under the same operation count and geometry. Static compiler '
            'paths do not reveal the branch taken at runtime, and finite-repeat differences are not causal proof.',
        empirical_scatter_is_latency_bound=False, new_model_admitted=False, coefficients_refitted=False,
        target_workload_executed=False, target_variants_ranked=False, selectable=False,
        exportable_as_complete_cost=False, physical_optimality_proven=False)
    _save(out/'result.json', report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True); parser.add_argument('--out', required=True)
    args=parser.parse_args(argv); result=run_topology_ablation(args.profile, args.out)
    print(json.dumps(dict(report=str(Path(args.out).absolute()/'result.json'), seconds=result['seconds'],
        configuration_count=result['configuration_count'], original_model_errors_by_topology=result['original_model_errors_by_topology'],
        selectable=False), sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
