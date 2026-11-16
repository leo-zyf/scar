"""One compact admission and execution path for trial and measured delivery.

The immutable payload/code identity excludes the decision-slot digest. A
fixed-size slot carries either explicitly unmeasured trial coordinates or a
producer-pinned release/rejection derived from complete original samples.
Both use the same parsing, numeric checks, scope checks, native runner and
report writer. No original benchmark evidence is replayed in this process.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics

from .caller_pipeline_deployment import (entry_context, environment_sha256,
    provider_identity, bundle_reference, runner_identity as _base_identity)
from .compiled_deployment import _raw


PAYLOAD = 'scar.caller-pipeline-final-payload'
SLOT = 'scar.caller-pipeline-final-cost-slot'
EXECUTION = 'scar.caller-pipeline-final-entry-execution'
TRUST = 'EXPLICIT_EXPECTED_PAYLOAD_AND_SLOT_SHA256_FROM_BUILD_VALIDATOR_NOT_SIGNATURES'
SLOT_BYTES = 4096
TRIAL, NO_OP, RELEASE = 0, 1, 2
_HEX = re.compile('[0-9a-f]{64}')
_DIGITS = re.compile('[0-9]{20}')


def _need(condition, detail):
    if not condition: raise ValueError(detail)


def _json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            _need(key not in result, 'duplicate capsule JSON key'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite capsule JSON')))


def _receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def implementation_identity():
    # These same files are checked in both the trial and release admission.
    return sha256(json.dumps([_receipt(__file__), _base_identity()],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def pack_slot(value):
    """Fixed bytes and fixed-width timing strings; zeros mean UNMEASURED."""
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    _need(len(raw) < SLOT_BYTES, 'cost slot exceeds the declared fixed page')
    return raw + b' '*(SLOT_BYTES-len(raw)-1) + b'\n'


def evaluate_slot(slot, payload, payload_sha256):
    """Do all structural/numeric work for every role, including the trial.

    Zero trial coordinates are explicitly unknown, never performance evidence.
    No release-only benchmark parsing, hashing, output check or cost arithmetic
    exists in the entry runner. Producer trust is the external slot SHA pin.
    """
    _need(type(slot) is dict and set(slot) == {'schema', 'schema_version', 'payload_sha256',
        'role', 'validation_sha256', 'producer_sha256', 'samples'}, 'exact compact cost-slot fields required')
    _need(slot['schema'] == SLOT and slot['schema_version'] == 1
        and slot['payload_sha256'] == payload_sha256 and type(slot['role']) is int
        and slot['role'] in (TRIAL, NO_OP, RELEASE), 'supported cost slot and payload binding required')
    for name in ('validation_sha256', 'producer_sha256'):
        _need(type(slot[name]) is str and _HEX.fullmatch(slot[name]) is not None, 'cost-slot hash required')
    _need(slot['producer_sha256'] == payload['validator_implementation']['sha256'], 'slot producer differs from frozen validator')
    policy = payload['policy']; warmup = policy['warmup']; count = policy['repetitions']
    _need(type(warmup) is int and warmup >= 1 and type(count) is int and count >= 3
        and warmup+count <= 12 and type(policy['minimum_gain']) in (int, float)
        and math.isfinite(policy['minimum_gain']) and 0 <= policy['minimum_gain'] < 1,
        'bounded complete frozen cost policy required')
    samples = slot['samples']; _need(type(samples) is list and len(samples) == warmup+count, 'all declared compact coordinates required')
    expected = [(0, i) for i in range(warmup)] + [(1, i) for i in range(count)]
    original = []; candidate = []; valid = []; unmeasured = []
    for row, coordinate in zip(samples, expected):
        _need(type(row) is dict and set(row) == {'phase', 'index', 'baseline_wall_ns',
            'candidate_wall_ns', 'sample_sha256', 'valid'}, 'exact cost-coordinate fields required')
        _need(type(row['phase']) is int and type(row['index']) is int
            and (row['phase'], row['index']) == coordinate and type(row['valid']) is int
            and row['valid'] in (0, 1), 'original ordered cost coordinates required')
        for key in ('baseline_wall_ns', 'candidate_wall_ns'):
            _need(type(row[key]) is str and _DIGITS.fullmatch(row[key]) is not None, 'fixed-width cost coordinate required')
        _need(type(row['sample_sha256']) is str and _HEX.fullmatch(row['sample_sha256']) is not None,
            'original sample hash required')
        a, b = int(row['baseline_wall_ns']), int(row['candidate_wall_ns'])
        flags = (row['valid'] == 1, a > 0, b > 0, row['sample_sha256'] != '0'*64)
        valid.append(sum(map(int, flags)) == len(flags))
        unknown_flags = (row['valid'] == 0, a == 0, b == 0, row['sample_sha256'] == '0'*64)
        unmeasured.append(sum(map(int, unknown_flags)) == len(unknown_flags))
        if row['phase'] == 1: original.append(a); candidate.append(b)
    # Every role performs these operations, including zeros in TRIAL slots.
    a = statistics.median(original); b = statistics.median(candidate)
    gain = 1-b/max(a, 1)
    valid_count = sum(map(int, valid)); validation_bound = slot['validation_sha256'] != '0'*64
    unknown_count = sum(map(int, unmeasured)); trial_unknown = unknown_count == len(unmeasured) and not validation_bound
    meets_gain = gain >= policy['minimum_gain']
    _need(slot['role'] != TRIAL or trial_unknown, 'trial coordinates must remain explicitly unmeasured')
    complete = valid_count == len(valid) and validation_bound
    profitable = complete and meets_gain
    approved = slot['role'] == RELEASE and profitable
    return dict(complete=complete, profitable=profitable, approved=approved, role=slot['role'])


def _admit(capsule, slot_path, *, expected_payload_sha256, expected_slot_sha256,
        entry, argv, report):
    _need(type(expected_payload_sha256) is str and _HEX.fullmatch(expected_payload_sha256) is not None
        and type(expected_slot_sha256) is str and _HEX.fullmatch(expected_slot_sha256) is not None,
        'explicit expected payload and slot hashes required')
    capsule = Path(capsule).resolve(); path = capsule/'payload.json'
    _need(path.stat().st_size <= 256*1024, 'bounded immutable capsule payload required')
    raw = path.read_bytes(); _need(sha256(raw).hexdigest() == expected_payload_sha256, 'capsule payload trust pin differs')
    payload = _json(raw)
    _need(payload.get('schema') == PAYLOAD and payload.get('schema_version') == 1
        and payload['trust_policy'] == TRUST and payload['slot_bytes'] == SLOT_BYTES
        and payload['capsule_directory'] == str(capsule)
        and payload['policy']['frozen_before_measurement'] is True,
        'supported original capsule payload required')
    slot_path = Path(slot_path).resolve()
    _need(slot_path.parent == capsule and slot_path.stat().st_size == SLOT_BYTES,
        'one original fixed-page capsule slot required')
    raw = slot_path.read_bytes(); _need(sha256(raw).hexdigest() == expected_slot_sha256, 'cost slot trust pin differs')
    slot = _json(raw); evaluation = evaluate_slot(slot, payload, expected_payload_sha256)
    _need(payload['entry_context'] == entry_context(entry, argv), 'entry argv/context changed')
    _need(payload['environment_sha256'] == environment_sha256(), 'inherited environment changed')
    _need(payload['runtime_identity'] == implementation_identity(), 'final delivery implementation changed')
    report = Path(report).resolve()
    _need(not report.exists() and report.is_relative_to(Path(payload['reporting']['allowed_root']).resolve())
        and payload['reporting']['required'] is True, 'fresh report in frozen storage scope required')
    for source in payload['source_refs']: _raw(source)
    _need(payload['provider_identity'] == provider_identity(), 'NumPy distribution/core identity changed')
    frozen_bundle, _ = bundle_reference(payload['bundle']['directory'], payload['bundle']['admission_sha256'])
    _need(payload['bundle'] == frozen_bundle, 'original guarded ABI bundle changed')
    return payload, slot, evaluation


def run_capsule_entry(entry, argv, *, capsule, slot, expected_payload_sha256,
        expected_slot_sha256, report, exclusive_inputs=False, measure_candidate=False):
    """Run the identical compact path under a trial or approved cost slot.

    The implementation never accepts a selection dictionary. In trial mode,
    only a producer-pinned explicitly UNMEASURED slot is eligible. Normal use
    requires the slot exported after complete whole-entry evidence replay.
    """
    from .entry_runner import _arguments, run_original_script
    entry, argv = _arguments(entry, argv); report = Path(report).resolve()
    _need(not report.exists(), 'fresh execution report required')
    payload = slot_value = evaluation = None; selected = False; reason = 'cost_capsule_not_admitted'
    before = entry_context(entry, argv); environment = environment_sha256(); identity = implementation_identity()
    try:
        payload, slot_value, evaluation = _admit(capsule, slot, expected_payload_sha256=expected_payload_sha256,
            expected_slot_sha256=expected_slot_sha256, entry=entry, argv=argv, report=report)
        trial_allowed = slot_value['role'] == TRIAL and measure_candidate is True
        release_allowed = slot_value['role'] == RELEASE and evaluation['approved'] is True and measure_candidate is False
        selected = exclusive_inputs is True and (trial_allowed or release_allowed)
        reason = 'compact_cost_authority_selected' if selected else 'no_profitable_release_or_explicit_trial_authority'
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        reason = 'capsule_scope_or_cost_not_admitted:' + str(error)
    if selected:
        from .caller_pipeline_cli import run_caller_entry
        result = run_caller_entry(entry, argv, bundle=payload['bundle']['directory'],
            expected_admission_sha256=payload['bundle']['admission_sha256'], exclusive_inputs=True, measure_candidate=True)
        execution = result['execution']; loaded = result['native_libraries_loaded']
    else:
        execution = run_original_script(entry, argv); loaded = False
    stable = (entry_context(entry, argv) == before and environment_sha256() == environment
        and implementation_identity() == identity)
    # Same post-run admission in both trial and release, including source and
    # provider hashes. The report has constant keys and integer role codes.
    try:
        if payload is not None:
            after, after_slot, _ = _admit(capsule, slot, expected_payload_sha256=expected_payload_sha256,
                expected_slot_sha256=expected_slot_sha256, entry=entry, argv=argv, report=report)
            stable = stable and after == payload and after_slot == slot_value
    except (OSError, ValueError, KeyError, TypeError, ImportError): stable = False
    value = dict(schema=EXECUTION, schema_version=1, payload_sha256=expected_payload_sha256,
        slot_sha256=expected_slot_sha256, authority_role_code=-1 if slot_value is None else slot_value['role'],
        context_unchanged=stable, entry_context=before, environment_sha256=environment, runtime_identity=identity,
        selected=selected, deployment_authorized=int(selected and slot_value['role'] == RELEASE),
        native_libraries_loaded=loaded, execution=execution, reason=reason,
        explicit_trial=int(measure_candidate is True), raw_benchmark_evidence_replayed=False,
        full_program_cpp=False, physical_optimality_proven=False, future_clock_guaranteed=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream: json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False); stream.write('\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', required=True); parser.add_argument('--slot', required=True)
    parser.add_argument('--payload-sha256', required=True); parser.add_argument('--slot-sha256', required=True)
    parser.add_argument('--report', required=True); parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--authority-role', required=True, type=int, choices=(TRIAL, RELEASE))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv); command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('unchanged Python entry required after --')
    run_capsule_entry(command[0], command[1:], capsule=args.capsule, slot=args.slot,
        expected_payload_sha256=args.payload_sha256, expected_slot_sha256=args.slot_sha256,
        report=args.report, exclusive_inputs=args.exclusive_inputs, measure_candidate=args.authority_role == TRIAL)
    return 0


if __name__ == '__main__': raise SystemExit(main())
