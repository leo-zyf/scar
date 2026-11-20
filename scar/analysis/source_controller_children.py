"""Join original controller effect calls to observed synchronous child execution."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.analysis.call_operand_reader import read_call_operands
from scar.analysis.source_controller_effects import SCHEMA as SOURCE
from scar.analysis.source_resource_scopes import _physical_members, derive_resource_scopes
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.providers import ProviderFactReference
from scar.ir.record_codec import encode
from scar.trace.capture_coordinates import bind_capture_coordinates
from scar.trace.files import open_binary
from scar.trace.os_resources import _CALL, _TIME, _arguments, _scan

SCHEMA = 'scar.controller-observed-children'
_SPAWN = frozenset(('clone', 'clone3', 'fork', 'vfork'))
_PROCESS_INTERFACE = _SPAWN | {'wait4', 'execve', 'execveat'}
_WAIT_RESULT = re.compile(r'\[\{WIFEXITED\(s\) && WEXITSTATUS\(s\) == ([0-9]+)\}\]')


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _attempt_key(anchor):
    return anchor.path, anchor.line_number, anchor.line_sha256, anchor.index


def _catalog(source, directory):
    def observe(rows, locator):
        for i, row in enumerate(rows):
            if row['trace_directory'] == directory and row['normal_completion_observed']:
                a = row['attempt']
                yield (a['path'], a['line_number'], a['line_sha256'], a['index']), f'{locator}/{i}'

    result = defaultdict(list)
    for i, effect in enumerate(source['effects']):
        for key, locator in observe(effect['observations'], f'#/effects/{i}/observations'):
            result[key].append(locator)
    for i, body in enumerate(source['bodies']):
        for j, call in enumerate(body['calls']):
            for key, locator in observe(call['observations'], f'#/bodies/{i}/calls/{j}/observations'):
                result[key].append(locator)
    return result


def _exec_argv(reference, call):
    # The shared resource importer omits argv. Read just the selected original
    # receipt and verify its enclosing artifact before retaining that field.
    if call['first_line'] != call['line']:
        raise ValueError('split exec argv rendering unavailable')
    digest, size, selected = sha256(), 0, None
    with open_binary(reference.artifact.uri) as stream:
        for ordinal, raw in enumerate(stream, 1):
            size += len(raw)
            if size > reference.artifact.size_bytes:
                raise ValueError('exec argv trace grew beyond original size')
            digest.update(raw)
            if ordinal == call['line']:
                selected = raw
    if ((size, digest.hexdigest()) != (reference.artifact.size_bytes, reference.artifact.sha256)
            or selected is None):
        raise ValueError('exec argv trace bytes differ')
    line = _TIME.fullmatch(selected.decode('utf-8').rstrip('\r\n'))
    complete = _CALL.fullmatch(line[1]) if line else None
    if complete is None or (complete[1], complete[3]) != ('execve', '0'):
        raise ValueError('original exec has no complete successful rendering')
    executable, argv = (json.loads(value) for value in _arguments(complete[2])[:2])
    if not (type(executable) is str and executable.startswith('/') and type(argv) is list
            and argv and all(type(arg) is str for arg in argv)):
        raise ValueError('exact executable or argument vector unavailable')
    return executable, argv


def project(effects, capture):
    """Rebuild source-attempt, same-scope spawn/wait, and exact exec argv joins."""
    source_ref = _ref(effects, SOURCE)
    source = read_artifact(source_ref)
    capture = Path(capture).resolve(strict=True)
    directory = str(capture / 'trace')
    matches = [row for row in source['trace_inputs'] if row['directory'] == directory]
    if len(matches) != 1:
        raise ValueError('capture not admitted by the controller effect source projection')
    coordinate_path = capture / 'capture-coordinates.json'
    coordinates = bind_capture_coordinates(capture, max_bytes=512_000_000,
                                            max_lines=2_000_000, profile_uri=str(coordinate_path))
    if coordinate_path.read_bytes() != coordinates.payload:
        raise ValueError('capture coordinates differ from retained receipts')
    trace = read_call_operands(directory, matches[0]['project_root'])
    scopes, _, _ = derive_resource_scopes(coordinates, capture, trace)
    catalog = _catalog(source, directory)
    selected = {scope.id: (scope, catalog[_attempt_key(scope.call_attempt)])
                for scope in scopes['scopes'] if _attempt_key(scope.call_attempt) in catalog}
    _, artifacts, _, calls = _scan(capture / 'os', 128_000_000, 2_000_000, interface_work=True)
    members = _physical_members(coordinates, artifacts)
    rows = defaultdict(list)
    exec_by_process = defaultdict(list)
    for call in calls:
        if call['file'] not in members or call['name'] not in _PROCESS_INTERFACE:
            continue
        member = members[call['file']]
        reference = ProviderFactReference(member, f'#/lines/{call["line"]}')
        rows[reference].append(call)
        if call['name'] in ('execve', 'execveat') and call['result'] == 0:
            process = coordinates.process_for(member, call['tid'])
            if process is not None:
                exec_by_process[process.id].append((reference, call))
    work_by_scope = defaultdict(list)
    for row in scopes['work']:
        if row.scope in selected and row.result is not None and row.result > 0:
            work_by_scope[row.scope].append(row)
    births = {process.source: process for process in coordinates.processes}
    children, frontiers = [], []
    for scope_id, (scope, locators) in selected.items():
        work = work_by_scope[scope_id]
        for spawn in (row for row in work if row.syscall in _SPAWN):
            child = births.get(spawn.source)
            matches = [row for row in work if row.syscall == 'wait4' and row.result == spawn.result]
            anchors = dict(source_observations=locators, call_scope=scope, spawn=spawn.source)
            if child is None or child.pid != spawn.result or len(matches) != 1:
                frontiers.append(dict(**anchors, reason='NO_SOURCE_SCOPED_CHILD_INSTANCE_AND_WAIT'))
                continue
            images = exec_by_process[child.id]
            waits = rows[matches[0].source]
            exit_status = (len(waits) == 1 and len(waits[0]['args']) == 4
                           and _WAIT_RESULT.fullmatch(waits[0]['args'][1]))
            if len(images) != 1 or images[0][1]['name'] != 'execve' or not exit_status:
                frontiers.append(dict(**anchors, reason='CHILD_EXEC_OR_NORMAL_COMPLETION_OPEN'))
                continue
            image_ref, image = images[0]
            try:
                executable, argv = _exec_argv(image_ref, image)
            except ValueError:
                frontiers.append(dict(**anchors, reason='EXACT_EXEC_ARGUMENTS_UNAVAILABLE'))
                continue
            children.append(dict(**anchors, child=child, exec=image_ref, executable=executable, argv=argv,
                                 wait=matches[0].source, exit_status=int(exit_status[1]),
                                 same_scope_spawn_and_wait=True, direct_program_origin_proven=False))
    return encode(dict(schema=SCHEMA, schema_version=1, source_effects=source_ref, coordinates=coordinates.artifact,
                       children=children, frontiers=frontiers, environment_and_executable_content_bound=False,
                       child_effects_closed=False, transformation_authorized=False))


def check(path):
    from scar.analysis.source_controller_effects import check as check_source

    reference = _ref(path, SCHEMA)
    saved = read_artifact(reference)
    source_ref = saved['source_effects']
    if check_source(source_ref['uri']) != read_artifact(source_ref):
        raise ValueError('controller source effects differ during child replay')
    derived = project(saved['source_effects']['uri'], Path(saved['coordinates']['uri']).parent)
    if derived != saved:
        raise ValueError('child execution differs from source and OS capture replay')
    return derived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check')
    parser.add_argument('--effects')
    parser.add_argument('--capture')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.effects and args.capture and args.out:
        report = project(args.effects, args.capture)
        with Path(args.out).open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or --effects INPUT --capture DIR --out JSON')
    print(json.dumps(dict(children=len(report['children']), frontiers=len(report['frontiers']))))


if __name__ == '__main__':
    main()
