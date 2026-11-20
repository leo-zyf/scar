"""Plan a bounded native console effect from the exact original call and stdout evidence."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import stat

from scar.compiler.plan_native_controller import SCHEMA as CONTROLLER
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import encode

SCHEMA = 'scar.controller-console-plan'
OBSERVATION = 'scar.controller-console-observation'
_SOURCE = 'scar.source-native-controller'
_OPAQUE = 'scar.source-controller-effects'
_TEXT_CHAIN = ['_io.TextIOWrapper', '_io.BufferedWriter', '_io.FileIO']
_BUFFER_LIMIT = 8192


@dataclass(frozen=True, slots=True)
class ConsoleContract:
    encoding: str
    errors: str
    ascii_only: bool
    append_lf: bool
    destination_fd: int
    destination_kind: str
    publication_horizon: str
    max_buffer_bytes: int
    exclusive_destination_contract_required: bool
    runtime_guards: tuple[str, ...]


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _at(reference, locator='#'):
    return dict(artifact=reference, locator=locator)


def _bytes(text):
    _require(type(text) is str, 'original UTF-8 observation must have canonical byte hex')
    raw = bytes.fromhex(text)
    _require(raw.hex() == text, 'noncanonical byte hex in original stdout observation')
    return raw


def _console_protocol(observation):
    occurrences = observation['occurrences']
    _require(type(occurrences) is list and occurrences, 'at least one exact successful print required')
    _require(observation['exclusive_observed_stdout_write'] is True,
             'unrelated or earlier observed stdout publication prevents one buffered console window')
    stream = occurrences[0]['stdout']
    _require(stream['original'] is True and stream['chain_types'] == _TEXT_CHAIN
             and stream['fd'] == 1 and stream['encoding'].lower() == 'utf-8'
             and type(stream['errors']) is str and bool(stream['errors'])
             and stream['size_bytes'] == 0
             and stream['line_buffering'] is False and stream['write_through'] is False,
             'original buffered UTF-8 sys.stdout chain required')
    mode = stream['mode']
    sink_kind = 'REGULAR_FILE' if stat.S_ISREG(mode) else 'PIPE' if stat.S_ISFIFO(mode) else None
    _require(sink_kind is not None, 'stdout must be its original regular-file or pipe sink')
    scope_end = observation['scope_finish_ns']
    _require(type(scope_end) is int and scope_end > 0, 'original successful scope exit required')
    parts = []
    previous_end = 0
    for row in occurrences:
        start, end = row['before_ns'], row['after_ns']
        _require(row['builtin_print'] is True and row['completion'] == 'C_RETURN'
                 and row['stdout'] == stream
                 and type(start) is int and type(end) is int
                 and previous_end <= start <= end <= scope_end,
                 'exact normal print and stdout identity/order must be retained')
        original = _bytes(row['positional_exact_str_utf8_hex'])
        _require(original.isascii(), 'native console scope requires exact ASCII subset of original UTF-8')
        parts.append(original + b'\n')
        previous_end = end
    merged = b''.join(parts)
    publication = observation['publication']
    observed = _bytes(publication['bytes_hex'])
    _require(publication['kind'] == 'PROCESS_EXIT_STDOUT_WRITE' and publication['fd'] == 1
             and type(publication['timestamp_ns']) is int and publication['timestamp_ns'] > scope_end
             and 0 < len(merged) < _BUFFER_LIMIT and merged == observed
             and read_artifact(publication['stdout'], json_document=False) == observed,
             'one bounded matching stdout write after the original successful scope required')
    read_artifact(publication['syscall_trace'], json_document=False)
    return ConsoleContract(
        'UTF-8', stream['errors'], True, True, 1, sink_kind, 'NORMAL_PROCESS_EXIT', len(merged), True,
        ('MATCH_ORIGINAL_SOURCE_CALL_AND_BUILTIN_PRINT_CONTRACT',
         'EXACT_ONE_ASCII_STRING_AND_DEFAULT_NEWLINE_PER_ORIGINAL_CALL',
         'EXCLUSIVE_FRESH_ORIGINAL_KIND_STDOUT_THROUGH_NORMAL_PROCESS_EXIT',
         'REJECT_BEFORE_PUBLICATION_IF_OWNED_BUFFER_WOULD_EXCEED_OBSERVED_DOMAIN',
         'PUBLISH_ONLY_AFTER_REMAINING_SYNCHRONOUS_EFFECTS_AND_SUCCESS_OUTPUT_COMMITS'),
    )


def _opaque_mapping(reference, source_ref, effect):
    from scar.analysis.source_controller_effects import check

    report = check(reference['uri'])
    _require(report == read_artifact(reference) and report['controller'] == source_ref,
             'read-only opaque-effect projection names a different original controller')
    positions = [ordinal for ordinal, row in enumerate(report['effects'])
                 if (row['id'], row['site']) == (effect['id'], effect['site'])]
    _require(len(positions) == 1, 'opaque-effect projection has no unique original console site')
    return _at(reference, f'#/effects/{positions[0]}')


def _derive(observation_ref, opaque_ref=None):
    from scar.analysis.source_controller_console import read_console_observation

    observation = read_console_observation(observation_ref['uri'])
    _require(observation == read_artifact(observation_ref)
             and (observation['schema'], observation['schema_version']) == (OBSERVATION, 1),
             'original console observation changed during replay')
    plan_ref = observation['controller_plan']
    source_ref = observation['controller_source']
    controller = read_artifact(plan_ref)
    source = read_artifact(source_ref)
    _require(plan_ref['schema'] == CONTROLLER and source_ref['schema'] == _SOURCE
             and controller['source'] == source_ref and source['effect_closure_proven'] is False
             and controller['replay']['runtime_effect_mode'] == 'ORIGINAL_SYNCHRONOUS_EFFECT_PROVIDER_REQUIRED',
             'console needs the exact source controller and original effect authority')
    ordinal = observation['ordinal']
    _require(type(ordinal) is int and 0 <= ordinal < len(source['effects']), 'invalid original effect ordinal')
    effect = source['effects'][ordinal]
    _require(effect['site'] == observation['site'] and effect['source_fact'] == observation['source_fact']
             and effect['positional_count'] == 1 and effect['keyword_names'] == []
             and effect['result_policy'] == 'ORIGINAL_RELEASE_ONLY' and effect['effects_closed'] is False,
             'console requires the exact original one-positional-argument and release-only call site')
    binding = _at(source_ref, f'#/effects/{ordinal}')
    console = _console_protocol(observation)
    effect_site = dict(id=effect['id'], ordinal=ordinal, source_fact=effect['source_fact'], site=effect['site'])
    refs = (binding, _at(observation_ref, '#/occurrences'), _at(observation_ref, '#/publication'))
    cost = CostEstimate(None, None, None, 'UNKNOWN', 'ORIGINAL_CONSOLE_EFFECT_TO_EXIT',
                        refs, ('NO_MATCHED_CLEAN_CONSOLE_EFFECT_SERVICE_COST',))
    opaque = None if opaque_ref is None else _opaque_mapping(opaque_ref, source_ref, effect)
    read_artifact(plan_ref)
    read_artifact(source_ref)
    read_artifact(observation_ref)
    return dict(
        schema=SCHEMA, schema_version=1, observation=observation_ref,
        controller_plan=plan_ref, controller_source=source_ref, opaque_effects=opaque_ref,
        opaque_site_evidence=opaque, effect_site=effect_site, console=encode(console),
        status='CONDITIONAL_EXPLICIT_TRIAL_ONLY', original_effect=refs[0],
        native_console_call_count_observed=len(observation['occurrences']),
        cost=encode(cost), default_backend='NO_OP', whole_entry_authorized=False,
        opaque_callbacks=[dict(id=row['id'], ordinal=index, action='PRESERVE_ORIGINAL_SYNCHRONOUS_PROVIDER')
                          for index, row in enumerate(source['effects']) if index != ordinal],
        frontiers=[
            'STDERR_AND_OTHER_OBSERVER_INTERLEAVINGS_OPEN_OUTSIDE_OBSERVED_SCOPE',
            'OTHER_OR_FUTURE_STDOUT_WRITERS_REQUIRE_EXCLUSIVE_DESTINATION_CONTRACT',
            'ORIGINAL_FLUSH_PARTIAL_WRITE_AND_NON_NORMAL_EXIT_PROTOCOL_OPEN',
            'UNMATCHED_ENCODING_OR_BUFFER_HORIZON_REQUIRES_ORIGINAL_EFFECT',
            'UNQUALIFIED_CALLBACK_EFFECTS_STILL_REQUIRE_ORIGINAL_PROVIDER',
        ],
        native_compiled=False, target_program_executed=False, performance_measured=False,
        deployment_authorized=False,
    )


def prepare_controller_console_plan(observation, out, *, opaque_effects=None):
    reference = _reference(observation, OBSERVATION)
    opaque_ref = _reference(opaque_effects, _OPAQUE) if opaque_effects is not None else None
    report = _derive(reference, opaque_ref)
    with Path(out).resolve().open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def read_controller_console_plan(path):
    reference = _reference(path, SCHEMA)
    report = read_artifact(reference)
    if report != _derive(report['observation'], report['opaque_effects']):
        raise ValueError('controller console plan differs from original stdout and controller replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--observation')
    inputs.add_argument('--check')
    parser.add_argument('--opaque-effects')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.opaque_effects:
            parser.error('--check only consumes the retained plan')
        report = read_controller_console_plan(args.check)
    else:
        if not args.out:
            parser.error('preparation requires --out')
        report = prepare_controller_console_plan(args.observation, args.out, opaque_effects=args.opaque_effects)
    print(json.dumps(dict(status=report['status'], site=report['effect_site']['ordinal'],
        bytes=report['console']['max_buffer_bytes'], destination=report['console']['destination_kind'],
        default_backend=report['default_backend'], deployment_authorized=False), sort_keys=True))


if __name__ == '__main__':
    main()
