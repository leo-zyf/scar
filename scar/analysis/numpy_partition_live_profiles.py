"""Bind the measured NumPy partition schedule to its exact live-call profile."""

import argparse
import json
from pathlib import Path

from scar.compiler import calibrate_numpy_partition
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.numpy-partition-live-profile'
_ROLES = ('target', 'matrix', 'out')
_GEOMETRY = ('dtype', 'shape', 'strides_bytes')


def _require(condition, reason):
    if not condition:
        raise ValueError('live NumPy partition: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _geometry(sample):
    return {name: {key: sample[name][key] for key in _GEOMETRY} for name in _ROLES}


def _coverage(witness, iteration, geometry):
    equivalent_samples = [row for row in witness['samples'] if _geometry(row) == geometry]
    observed_calls = sum(row['calls'] for row in witness['geometries'] if row['roles'] == geometry)
    total = witness['counts']['calls']
    _require(len(equivalent_samples) <= observed_calls <= total and observed_calls > 0,
             'matched call count is inconsistent with the original capture')
    verified = {row['id'] for row in iteration['samples']}
    coverage = dict(
        original_observed_calls=total,
        identical_geometry_calls=observed_calls,
        other_or_unclassified_geometry_calls=total - observed_calls,
        scope='ORIGINAL_CAPTURE_GEOMETRY_COUNTS_BEFORE_LIVE_GUARDS',
        matched_samples=[{key: row[key] for key in ('id', 'case', 'ordinal', 'caller')}
                         for row in equivalent_samples],
        exact_geometry_independent_holdouts=sum(row['case'] == 'HOLDOUT' and row['id'] in verified
                                               for row in equivalent_samples),
        live_accepted_call_count_proven=False,
    )
    numerically_verified = [_geometry(row) for row in witness['samples'] if row['id'] in verified]
    frontiers = []
    for row in witness['geometries']:
        current = row['roles']
        if current == geometry:
            continue
        if len(current['target']['shape']) < 2:
            reason = 'SCALAR_DISPATCH_NOT_ADMITTED'
        elif current in numerically_verified:
            reason = 'NUMERICAL_SAMPLE_ONLY_NO_MATCHED_SCOPE_TIMING'
        else:
            reason = 'NO_MATCHED_NUMERICAL_AND_TIMING_EVIDENCE'
        frontiers.append(dict(geometry=current, calls=row['calls'], reason=reason))
    return coverage, frontiers


def project(calibration_path):
    """Recheck prepared timing and detached native evidence from retained receipts."""
    calibration_ref = _ref(calibration_path, calibrate_numpy_partition.SCHEMA)
    calibration = calibrate_numpy_partition.check(calibration_ref['uri'])
    _require(calibration == read_artifact(calibration_ref), 'measured receipt changed')
    decision = calibration['selection']
    implementation = decision['chosen_for_measured_scope']
    _require(implementation == 'elided-copy'
             and calibration['timing'] == 'PREPARED_REPEAT_CALLS_EXCLUDING_INGRESS_AND_PUBLICATION'
             and decision['default_entry_backend'] == 'NO_OP',
             'the admitted copy-elided schedule must win this prepared scope')

    package_ref = calibration['package']
    package = read_artifact(package_ref)
    iteration_ref = package['iteration']
    iteration = read_artifact(iteration_ref)
    witness_ref = calibration['source_witness']
    _require(iteration['witness'] == witness_ref, 'native and measured source witnesses differ')
    witness = read_artifact(witness_ref)
    sample_id = calibration['sample']
    observed, = (row for row in witness['samples'] if row['id'] == sample_id)
    projected, = (row for row in iteration['samples'] if row['id'] == sample_id)
    candidate, = (row for row in package['cases'] if row['source_sample'] == sample_id
                  and row['implementation'] == implementation)
    numerical, = (row for row in package['validation'] if row['id'] == candidate['id'])
    provider_ref = package['provider']
    provider = read_artifact(provider_ref)
    _require(calibration['cpu_ids'] == package['cpu_ids'] == witness['cpu_ids']
             and candidate['inputs'] == projected['inputs'] and candidate['expected'] == projected['expected_out']
             and candidate['expected']['sha256'] == numerical['output']['sha256']
             and candidate['expected']['size_bytes'] == numerical['output']['size_bytes']
             and dict(provider['provider'], schema=iteration['provider']['schema']) == iteration['provider']
                 == witness['numpy']['extension']
             and package['linkage']['provider']['exact_machine_closure'] is True
             and observed['numpy_errstate'] == witness['numpy']['errstate']
             and observed['normal_return'] is True and observed['returned_alias'] == 'out'
             and all(step['op'] in ('ADD', 'MULTIPLY') for step in candidate['steps']),
             'provider, processor, arithmetic or witnessed private output differs')
    geometry = _geometry(observed)
    coverage, frontiers = _coverage(witness, iteration, geometry)
    profile = {key: candidate[key] for key in
               ('id', 'source_sample', 'implementation', 'sizes', 'residues', 'steps', 'expected', 'inputs')}
    profile.update(geometry=geometry, selectors=observed['slices'],
                   rounding_mode_raw=observed['rounding_mode_raw'], numpy_errstate=observed['numpy_errstate'],
                   caller=observed['caller'], detached_numerical_validation=numerical['output'])
    guards = dict(
        source='EXACT_CODE_AND_LOADED_BUILTIN_PROVIDER_BINDINGS',
        layout='EXACT_NDARRAY_CLASS_DTYPE_SHAPE_STRIDES_SELECTOR_ORDER_AND_ALIGNED_FULL_SPAN',
        input_placement='MATCHED_INPUT_ADDRESS_RESIDUES_OR_PRIVATE_REPACK_AT_MATCHED_RESIDUES',
        effects='EXCLUSIVE_INPUT_AND_OUTPUT_ACCESS_UNTIL_ATOMIC_WITH_RESPECT_TO_ALLOWED_OBSERVERS',
        float_status='MATCH_STOCK_FENV_AND_ERROR_POLICY_AND_REQUIRE_NO_NEW_EXCEPTIONS_BEFORE_COMMIT',
        publication='COMPUTE_FULLY_IN_PRIVATE_OUTPUT_THEN_COPY_TO_ORIGINAL_OUT_AND_RETURN_IT',
        fallback='NO_OUTPUT_MUTATION_AND_RESTORE_FP_STATE_BEFORE_CALLING_ORIGINAL',
        all_guards_discharged=False,
    )
    measurement = {key: calibration[key] for key in ('timing', 'iterations_per_process', 'warmup_rounds',
                                                       'measured_rounds')}
    measurement.update(selection=decision, live_boundary_profitability='UNMEASURED',
                       whole_application_benchmark=False)
    machine = dict(isa=provider['isa'], floating_mode=provider['floating_mode'],
                   ndarray_c_abi=provider['ndarray_c_abi'], closure=package['linkage']['provider'])
    return dict(schema=SCHEMA, schema_version=1, calibration=calibration_ref, package=package_ref,
                iteration=iteration_ref, witness=witness_ref, source=witness['focus'], entry=witness['entry'],
                source_python=calibration['source_python'], provider=provider_ref, numpy=witness['numpy'],
                cpu_ids=calibration['cpu_ids'], machine=machine, profile=profile, coverage=coverage,
                guards=guards, frontiers=frontiers, measurement=measurement,
                deployment_authorized=False, default_entry_backend='NO_OP')


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    _require(stored == project(stored['calibration']['uri']), 'stored live profile differs from replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--calibration', type=Path)
    sources.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.out is None:
        report = check(args.check)
    elif args.calibration and args.out:
        report = project(args.calibration)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('supply --check RESULT or --calibration CALIBRATION --out RESULT')
    print(json.dumps(dict(profile=report['profile']['id'], coverage=report['coverage']), sort_keys=True))


if __name__ == '__main__':
    main()
