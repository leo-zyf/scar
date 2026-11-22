"""Freeze and validate one final compact caller-delivery implementation.

P is immutable. D has fixed bytes and original sample coordinates, and is
producer-pinned separately. Final evidence does not include D's final digest:
the measured runtime/code/payload therefore has no self-referential hash.
"""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time

from scar.compiler.qualification import _reference, _save, _equal_files, _statistics
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN
from scar.runtime.compiled_deployment import _raw
from scar.runtime.caller_pipeline_deployment import (entry_context, environment_sha256,
    provider_snapshot, bundle_reference, source_references, whole_entry_gain)
from scar.runtime.caller_pipeline_capsule import (PAYLOAD, SLOT, EXECUTION, TRUST, SLOT_BYTES,
    TRIAL, NO_OP, RELEASE, implementation_identity, pack_slot, evaluate_slot, _admit)


DESIGN = 'scar.caller-pipeline-final-validation-design'
SAMPLE = 'scar.caller-pipeline-final-validation-sample'
VALIDATION = 'scar.caller-pipeline-final-validation'


def _bytes(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _publication_contract(directory, exact, seconds, argv):
    if directory is None and not exact and not seconds:
        return None
    if directory is None or not (exact or seconds):
        raise ValueError('publication directory and at least one declared file required')
    root = Path(directory)
    if not root.is_absolute() or str(root.resolve()) != str(root) or argv.count(str(root)) != 1:
        raise ValueError('publication directory must be a canonical absolute path in the exact entry arguments')
    files = {}
    for names, policy in ((exact, 'EXACT_BYTES'), (seconds, 'FINITE_NONNEGATIVE_SECONDS')):
        for name in names:
            path = PurePosixPath(name)
            if (not path.parts or path.is_absolute() or '..' in path.parts or str(path) != name
                    or '\x00' in name or name in files):
                raise ValueError('unique canonical relative publication file names required')
            files[name] = policy
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents):
            raise ValueError('a publication file cannot also be a directory')
    return dict(directory=str(root), files=files)


def _require_fresh_publication(root):
    if os.path.lexists(root):
        raise ValueError('fresh publication directory required: ' + str(root))


def _publication_snapshot(root, contract):
    directories = {PurePosixPath('.'): set()}
    for name in contract['files']:
        child = PurePosixPath(name)
        while child.parts:
            directories.setdefault(child.parent, set()).add(child.name)
            child = child.parent
    for directory in sorted(directories, key=lambda path: len(path.parts)):
        path = root / directory
        if (not stat.S_ISDIR(path.lstat().st_mode)
                or {entry.name for entry in path.iterdir()} != directories[directory]):
            raise ValueError('publication directory entries differ from the declared file tree: ' + str(path))
    files = {}
    for name, policy in contract['files'].items():
        path = root / name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('publication must be a regular file: ' + str(path))
        row = dict(policy=policy)
        if policy == 'FINITE_NONNEGATIVE_SECONDS':
            with path.open('rb') as stream:
                raw = stream.read(257)
            if len(raw) > 256:
                raise ValueError('elapsed publication exceeds the scalar size limit')
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError('elapsed publication requires one finite nonnegative scalar')
            row['seconds'] = value
            digest, size = sha256(raw).hexdigest(), len(raw)
        else:
            h = sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            digest, size = h.hexdigest(), info.st_size
        after = path.lstat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_ctime_ns) != (
                info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns):
            raise ValueError('publication file changed during validation: ' + str(path))
        row['file'] = dict(uri=str(path), sha256=digest, size_bytes=size, schema='scar.validation-raw')
        files[name] = row
    return files


def _publication_match(a, b, contract):
    return all(_equal_files(a[name]['file']['uri'], b[name]['file']['uri'])
               for name, policy in contract['files'].items() if policy == 'EXACT_BYTES')


def _slot(payload, digest, role, samples=(), validation_sha256='0'*64):
    known = {(0 if row['phase'] == 'WARMUP' else 1, row['index']): row for row in samples}
    coordinates = [(0, i) for i in range(payload['policy']['warmup'])] + [(1, i) for i in range(payload['policy']['repetitions'])]
    rows = []
    for phase, index in coordinates:
        sample = known.get((phase, index))
        rows.append(dict(phase=phase, index=index,
            baseline_wall_ns=f"{0 if sample is None else sample['baseline']['wall_ns']:020d}",
            candidate_wall_ns=f"{0 if sample is None else sample['optimized']['wall_ns']:020d}",
            sample_sha256='0'*64 if sample is None else sample['sample_ref']['sha256'],
            valid=0 if sample is None else int(sample['valid'])))
    return dict(schema=SLOT, schema_version=1, payload_sha256=digest, role=role,
        validation_sha256=validation_sha256, producer_sha256=payload['validator_implementation']['sha256'], samples=rows)


def prepare_capsule(bundle, entry, argv, out, *, expected_admission_sha256, source_paths,
        warmup=1, repetitions=3, minimum_gain=DEFAULT_MINIMUM_GAIN, publication_directory=None,
        publication_bytes=(), publication_nonnegative_seconds=()):
    from scar.runtime.entry_runner import _arguments
    entry, argv = _arguments(entry, argv); out = Path(out).resolve()
    if out.exists(): raise ValueError('fresh final capsule directory required')
    if (type(warmup) is not int or warmup < 1 or type(repetitions) is not int or repetitions < 3
            or warmup+repetitions > 12 or type(minimum_gain) not in (int, float)
            or not math.isfinite(minimum_gain) or not 0 <= minimum_gain < 1):
        raise ValueError('bounded warmup>=1, repetitions>=3 and finite minimum gain required')
    publication = _publication_contract(publication_directory, publication_bytes,
        publication_nonnegative_seconds, argv)
    if publication is not None:
        root = Path(publication['directory'])
        _require_fresh_publication(root)
        if root.is_relative_to(out) or out.is_relative_to(root):
            raise ValueError('publication must be outside the capsule directory')
    frozen, metadata = bundle_reference(bundle, expected_admission_sha256)
    if Path(metadata['contract']['source']['path']).resolve() == entry:
        raise ValueError('entry-defined original caller is unsupported')
    # The producer-pinned C2 admission is the prior original F4+ABI authority.
    # Its original proof is byte checked here, not falsely called a new replay.
    original = metadata['contract']['original_proof']; _raw(original)
    provider, provider_witness = provider_snapshot()
    payload = dict(schema=PAYLOAD, schema_version=1, trust_policy=TRUST,
        capsule_directory=str(out), slot_bytes=SLOT_BYTES, bundle=frozen,
        entry_context=entry_context(entry, argv), environment_sha256=environment_sha256(),
        source_refs=source_references(entry, metadata['contract'], source_paths),
        source_scope='EXPLICIT_RETAINED_SOURCE_SET_NOT_TRANSITIVE_IMPORT_PROOF',
        provider_identity=provider, provider_file_witness=provider_witness, runtime_identity=implementation_identity(),
        validator_implementation=_reference(__file__, 'scar.validation-raw'),
        original_regional_proof=original, regional_authority='PRIOR_PRODUCER_PINNED_C2_ORIGINAL_F4_AND_MATCHED_ABI_REPLAY',
        regional_source_derivation_repeated=False,
        policy=dict(warmup=warmup, repetitions=repetitions, minimum_gain=minimum_gain, strict_minimum_gain=True,
            frozen_before_measurement=True),
        reporting=dict(required=True, allowed_root=str(out.parent), new_file_only=True),
        cost_applicability='SAME_IMMUTABLE_PAYLOAD_RUNTIME_FIXED_PAGE_SCHEMA_AND_SHARED_TRIAL_RELEASE_OPERATION_PATH',
        fixed_slot_content_update_is_same_operation_profile=True,
        raw_benchmark_evidence_read_at_runtime=False, physical_clock_identity_guaranteed=False)
    if publication is not None:
        payload['publication'] = publication
    out.mkdir(parents=True); raw = _bytes(payload); (out/'payload.json').write_bytes(raw)
    digest = sha256(raw).hexdigest(); slot = _slot(payload, digest, TRIAL)
    (out/'trial.slot').write_bytes(pack_slot(slot))
    result = dict(payload=_reference(out/'payload.json', PAYLOAD), trial_slot=_reference(out/'trial.slot', SLOT))
    _admit(out, out/'trial.slot', expected_payload_sha256=digest, expected_slot_sha256=result['trial_slot']['sha256'],
        entry=entry, argv=argv, report=out.parent/'prepare-check-never-written.json')
    _save(out/'prepare.json', dict(schema='scar.caller-pipeline-final-capsule-prepare', schema_version=1,
        **result, deployment_authorized=False, target_program_executed=False))
    return result


def _counts(report, payload, payload_ref, trial_ref):
    if (report.get('schema'), report.get('schema_version')) != (EXECUTION, 1): return 0, 0, False
    fixed = (report.get('payload_sha256') == payload_ref['sha256'] and report.get('slot_sha256') == trial_ref['sha256']
        and report.get('authority_role_code') == TRIAL and report.get('explicit_trial') == 1
        and report.get('deployment_authorized') == 0 and report.get('selected') is True
        and report.get('entry_context') == payload['entry_context']
        and report.get('runtime_identity') == payload['runtime_identity']
        and report.get('environment_sha256') == payload['environment_sha256']
        and report.get('context_unchanged') is True and report.get('raw_benchmark_evidence_replayed') is False)
    execution = report.get('execution', {}); stats = execution.get('statistics', {})
    a, b = stats.get('native_calls'), stats.get('fallback_calls')
    valid = (type(a) is int and a > 0 and type(b) is int and b == 0
        and execution.get('entry_path') == payload['entry_context']['entry_path']
        and execution.get('argv') == payload['entry_context']['argv']
        and execution.get('schema') == 'scar.entry-runner-execution' and execution.get('schema_version') == 1
        and execution.get('installed_bindings') == 1 and execution.get('observed_boundary_calls') == a
        and not execution.get('frontier') and execution.get('exit_code') == 0)
    return a if type(a) is int else 0, b if type(b) is int else 0, bool(fixed and valid)


def _prepared(capsule, payload_ref, trial_ref, report):
    payload = json.loads(_raw(payload_ref)); _raw(trial_ref); _raw(payload['validator_implementation'])
    _raw(payload['original_regional_proof'])
    checked, _, _ = _admit(capsule, trial_ref['uri'], expected_payload_sha256=payload_ref['sha256'],
        expected_slot_sha256=trial_ref['sha256'], entry=payload['entry_context']['entry_path'],
        argv=payload['entry_context']['argv'], report=report)
    if checked != payload: raise ValueError('admitted final payload differs from original bytes')
    return payload


def _validation_design(payload, payload_ref, trial_ref):
    design = dict(schema=DESIGN, schema_version=1, payload=payload_ref, trial_slot=trial_ref,
        validator_implementation=payload['validator_implementation'], policy=payload['policy'],
        candidate_path='scar.runtime.caller_pipeline_capsule:run_capsule_entry',
        release_uses_same_admission_execution_report_path=True, decision_slot_bytes=SLOT_BYTES,
        pair_order='ALTERNATE_ORIGINAL_FIRST_AND_CANDIDATE_FIRST',
        process_clock_scope='PARENT_LAUNCH_THROUGH_EXIT_INCLUDING_COMPACT_PAYLOAD_SLOT_ADMISSION_ELF_NATIVE_GUARDS_REPORT',
        runtime_raw_benchmark_replay=False, implementations_compared=1, target_variants_ranked=False,
        arbitrary_external_effect_validation=False)
    if 'publication' in payload:
        design.update(publication=payload['publication'], publication_capture_in_process_clock=False)
    return design


def validate_capsule(capsule, out, *, expected_payload_sha256, expected_trial_slot_sha256):
    capsule = Path(capsule).resolve(); out = Path(out).resolve()
    if out.exists(): raise ValueError('fresh final-path validation output required')
    payload_ref = _reference(capsule/'payload.json', PAYLOAD); trial_ref = _reference(capsule/'trial.slot', SLOT)
    if payload_ref['sha256'] != expected_payload_sha256 or trial_ref['sha256'] != expected_trial_slot_sha256:
        raise ValueError('original prepared capsule pins differ')
    payload = _prepared(capsule, payload_ref, trial_ref, out/'never-written.json')
    context = payload['entry_context']; entry = context['entry_path']; argv = context['argv']; policy = payload['policy']
    publication = payload.get('publication')
    if publication is not None:
        root = Path(publication['directory'])
        _require_fresh_publication(root)
        if root.is_relative_to(out) or out.is_relative_to(root):
            raise ValueError('publication must be outside the validation directory')
    environment = dict(os.environ); out.mkdir(parents=True); (out/'samples').mkdir()
    if publication is not None and root.parent.exists() and root.parent.stat().st_dev != out.stat().st_dev:
        raise ValueError('publication and validation paths must allow an atomic directory move')
    design = _validation_design(payload, payload_ref, trial_ref)
    design_ref = _save(out/'design.json', design); evidence = [design_ref, payload['validator_implementation']]
    samples = []; reasons = []
    def process(command, folder, label):
        stdout = folder/(label+'.stdout'); stderr = folder/(label+'.stderr')
        if publication is not None:
            _require_fresh_publication(root)
        with stdout.open('xb') as a, stderr.open('xb') as b:
            start = time.perf_counter_ns()
            result = subprocess.run(command, stdout=a, stderr=b, env=environment, cwd=context['working_directory'])
            elapsed = time.perf_counter_ns()-start
        value = dict(command=command, wall_ns=elapsed, returncode=result.returncode,
            stdout=_reference(stdout, 'scar.validation-raw'), stderr=_reference(stderr, 'scar.validation-raw'))
        if publication is not None:
            if not root.is_dir() or root.is_symlink():
                raise ValueError('entry did not publish the declared regular directory: ' + str(root))
            snapshot = folder/(label+'.publications')
            root.rename(snapshot)
            value['publications'] = _publication_snapshot(snapshot, publication)
        evidence.extend((value['stdout'], value['stderr'])); return value
    coordinates = [('WARMUP', i) for i in range(policy['warmup'])] + [('MEASUREMENT', i) for i in range(policy['repetitions'])]
    for ordinal, (phase, index) in enumerate(coordinates):
        folder = out/'samples'/(phase.lower()+'-'+str(index)); folder.mkdir(); report_path = folder/'execution.json'
        commands = dict(baseline=[sys.executable, entry, *argv], optimized=[sys.executable, '-m',
            'scar.runtime.caller_pipeline_capsule', '--capsule', str(capsule), '--slot', trial_ref['uri'],
            '--payload-sha256', payload_ref['sha256'], '--slot-sha256', trial_ref['sha256'],
            '--exclusive-inputs', '--authority-role', str(TRIAL), '--report', str(report_path), '--', entry, *argv])
        values = {}
        for label in (('baseline', 'optimized') if ordinal%2 == 0 else ('optimized', 'baseline')):
            values[label] = process(commands[label], folder, label)
        a, b = values['baseline'], values['optimized']; native = fallback = 0; native_only = False
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_bytes()); ref = _reference(report_path, report['schema'])
                b['execution_report'] = ref; evidence.append(ref)
                native, fallback, native_only = _counts(report, payload, payload_ref, trial_ref)
            except (OSError, ValueError, KeyError, TypeError): b['execution_report_invalid'] = True
        stdout_equal = _equal_files(a['stdout']['uri'], b['stdout']['uri'])
        stderr_equal = _equal_files(a['stderr']['uri'], b['stderr']['uri'])
        success = a['returncode'] == b['returncode'] == 0
        try: stable = _prepared(capsule, payload_ref, trial_ref, out/'never-written.json') == payload
        except (OSError, ValueError, KeyError, TypeError, ImportError): stable = False
        sample = dict(schema=SAMPLE, schema_version=1, phase=phase, index=index, baseline=a, optimized=b,
            stdout_match=stdout_equal, stderr_match=stderr_equal, successful_returncodes=success,
            native_calls=native, fallback_calls=fallback, native_only=native_only, frozen_context_unchanged=stable,
            valid=stdout_equal and stderr_equal and success and native_only and stable)
        if publication is not None:
            sample['publications_match'] = _publication_match(a['publications'], b['publications'], publication)
            sample['valid'] = sample['valid'] and sample['publications_match']
        ref = _save(folder/'sample.json', sample); evidence.append(ref); samples.append(dict(**sample, sample_ref=ref))
        if not sample['valid']: reasons.append('ORIGINAL_OUTPUT_EXIT_NATIVE_COVERAGE_OR_CONTEXT_INVALID'); break
    formal = [row for row in samples if row['phase'] == 'MEASUREMENT']
    a = _statistics([row['baseline']['wall_ns'] for row in formal]); b = _statistics([row['optimized']['wall_ns'] for row in formal])
    gain = None if not a['median_ns'] or not b['median_ns'] else 1-b['median_ns']/a['median_ns']
    complete = len(samples) == len(coordinates); valid = complete and all(row['valid'] for row in samples)
    separated = profitable = False
    if valid:
        gain, separated, profitable = whole_entry_gain(a['samples_ns'], b['samples_ns'], policy['minimum_gain'],
                                                       strict_minimum_gain=policy.get('strict_minimum_gain', False))
    if not complete: reasons.append('DECLARED_MEASUREMENTS_INCOMPLETE')
    if valid and not profitable: reasons.append('FINAL_COMPACT_PATH_NOT_PROFITABLE')
    result = dict(schema=VALIDATION, schema_version=1, payload=payload_ref, trial_slot=trial_ref,
        design=design_ref, samples=samples, evidence=evidence, complete=complete, validated=valid,
        decision='EXPORT_PROFITABLE_RELEASE_SLOT' if profitable else 'NO_OP', reasons=reasons,
        performance=dict(baseline=a, optimized=b, gain_fraction=gain,
            sampled_ranges_separated=separated,
            speedup=None if not a['median_ns'] or not b['median_ns'] else a['median_ns']/b['median_ns']),
        runtime_code_and_payload_unchanged=True, decision_slot_identity_separate_from_payload=True,
        final_admission_native_guard_report_path_in_clock=True, implementations_compared=1,
        target_variants_ranked=False, physical_optimality_proven=False, future_clock_guaranteed=False,
        applicability='FIXED_PAGE_CONTENTS_CHANGE_WITH_SAME_VALIDATION_OPERATION_PROFILE_NOT_A_PHYSICAL_CLOCK_IDENTITY')
    if publication is not None:
        result['declared_publications_validated'] = complete and all(row['publications_match'] for row in samples)
    ref = _save(out/'result.json', result)
    return dict(validation=ref, result=result)


def export_cost_slot(capsule, validation, *, expected_payload_sha256, expected_validation_sha256):
    """Replay all raw final-path evidence before exporting one compact decision.

    An arbitrary selection dictionary or the old preliminary receipt is not an
    accepted input. Explicit expected SHA pins retain local producer trust.
    """
    capsule = Path(capsule).resolve(); validation = Path(validation).resolve()
    ref = _reference(validation, VALIDATION)
    if ref['sha256'] != expected_validation_sha256: raise ValueError('final validation pin differs')
    value = json.loads(_raw(ref))
    if (value.get('schema'), value.get('schema_version')) != (VALIDATION, 1): raise ValueError('original final validation required')
    payload_ref, trial_ref = value['payload'], value['trial_slot']
    if payload_ref['sha256'] != expected_payload_sha256: raise ValueError('final payload pin differs')
    payload = _prepared(capsule, payload_ref, trial_ref, capsule.parent/'export-check-never-written.json')
    design = json.loads(_raw(value['design']))
    expected_design = _validation_design(payload, payload_ref, trial_ref)
    if design != expected_design: raise ValueError('frozen final-path design differs')
    for evidence in value['evidence']: _raw(evidence)
    policy = payload['policy']; expected = [('WARMUP', i) for i in range(policy['warmup'])] + [('MEASUREMENT', i) for i in range(policy['repetitions'])]
    if [(row['phase'], row['index']) for row in value['samples']] != expected:
        raise ValueError('complete ordered original sample coordinates required')
    publication = payload.get('publication')
    valid = True
    for row in value['samples']:
        original = json.loads(_raw(row['sample_ref']))
        if original != {k: v for k, v in row.items() if k != 'sample_ref'}: raise ValueError('raw original sample differs')
        a, b = row['baseline'], row['optimized']
        if any(type(r['wall_ns']) is not int or not 0 < r['wall_ns'] < 10**20 for r in (a, b)):
            raise ValueError('positive bounded original process costs required')
        exact = (_raw(a['stdout']) == _raw(b['stdout']) and _raw(a['stderr']) == _raw(b['stderr'])
            and a['returncode'] == b['returncode'] == 0)
        if publication is not None:
            folder = validation.parent/'samples'/(row['phase'].lower()+'-'+str(row['index']))
            for label, sample in (('baseline', a), ('optimized', b)):
                retained = _publication_snapshot(folder/(label+'.publications'), publication)
                if retained != sample['publications']:
                    raise ValueError('retained publication bytes or scalar differ from their original evidence')
            publication_match = _publication_match(a['publications'], b['publications'], publication)
            if row['publications_match'] is not publication_match:
                raise ValueError('publication comparison differs from retained evidence')
            exact = exact and publication_match
        observed = json.loads(_raw(b['execution_report']))
        native, fallback, native_only = _counts(observed, payload, payload_ref, trial_ref)
        if row['native_calls'] != native or row['fallback_calls'] != fallback:
            raise ValueError('retained native observations differ')
        recomputed = exact and native_only and row['frozen_context_unchanged'] is True
        if row['valid'] is not recomputed: raise ValueError('sample validity differs from original evidence')
        valid = valid and recomputed
    provisional = _slot(payload, payload_ref['sha256'], RELEASE, value['samples'], ref['sha256'])
    evaluated = evaluate_slot(provisional, payload, payload_ref['sha256'])
    role = RELEASE if valid and evaluated['approved'] else NO_OP
    slot = _slot(payload, payload_ref['sha256'], role, value['samples'], ref['sha256'])
    path = capsule/'decision.slot'
    if path.exists(): raise ValueError('fresh final decision slot required')
    path.write_bytes(pack_slot(slot)); slot_ref = _reference(path, SLOT)
    report = dict(schema='scar.caller-pipeline-final-slot-export', schema_version=1, payload=payload_ref,
        slot=slot_ref, original_validation=ref, decision='RELEASE' if role == RELEASE else 'NO_OP',
        deployment_authorized=role == RELEASE, raw_original_evidence_replayed=True,
        preliminary_timing_accepted=False, fixed_slot_bytes=SLOT_BYTES,
        measured_runtime_and_payload_unchanged=True, self_referential_cost_identity=False,
        trust_policy=TRUST, target_variant_ranking=False, future_clock_guaranteed=False)
    if publication is not None:
        report['declared_publication_evidence_replayed'] = True
    _save(capsule/'decision-export.json', report)
    return report


def _add_publication_options(parser):
    parser.add_argument('--publication-directory', help='fresh canonical absolute directory passed in the entry argv')
    parser.add_argument('--publication-bytes', default=[], action='append', metavar='RELATIVE_FILE')
    parser.add_argument('--publication-nonnegative-seconds', default=[], action='append', metavar='RELATIVE_FILE')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True); parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--source', action='append', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--warmup', type=int, default=1); parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--minimum-gain', type=float, default=DEFAULT_MINIMUM_GAIN)
    _add_publication_options(parser)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv); command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('unchanged original Python entry required after --')
    out = Path(args.out).resolve(); prepared = prepare_capsule(args.bundle, command[0], command[1:], out/'capsule',
        expected_admission_sha256=args.admission_sha256, source_paths=args.source, warmup=args.warmup,
        repetitions=args.repetitions, minimum_gain=args.minimum_gain, publication_directory=args.publication_directory,
        publication_bytes=args.publication_bytes, publication_nonnegative_seconds=args.publication_nonnegative_seconds)
    measured = validate_capsule(out/'capsule', out/'validation',
        expected_payload_sha256=prepared['payload']['sha256'], expected_trial_slot_sha256=prepared['trial_slot']['sha256'])
    exported = export_cost_slot(out/'capsule', measured['validation']['uri'],
        expected_payload_sha256=prepared['payload']['sha256'], expected_validation_sha256=measured['validation']['sha256'])
    print(json.dumps(dict(prepared=prepared, validation=measured['validation'],
        decision=exported, performance=measured['result']['performance']), sort_keys=True))
    return 0


if __name__ == '__main__': raise SystemExit(main())
