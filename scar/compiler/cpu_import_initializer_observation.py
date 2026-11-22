"""Observe admitted source initializers in the original and direct spawn processes."""

import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic_ns

from scar.analysis.cpu_entry_import_bindings import _key, _static
from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import cpu_entry_replay, cpu_import_binding_observation as bindings
from scar.compiler.cpu_entry_launch_scope import SCHEMA as ADMISSION, verify_cpu_import_source_admission
from scar.compiler.cpu_original_importtime import _observation, _raw, _ref, _write
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-import-initializer-observation'
_ARCHIVED_MONITOR = '75be92bb1d4883ef1a40e5f37eb129e37929a64c5e63c6b1abc669f9119f46e9'
_SPAWN_CORE = 'spawn-observer.py'
_INITIALIZERS = 'initializers'
_PROCESS = 'scar.cpu-import-initializer-process'
_LIMIT = 128
_LOADER = 'STANDARD_SOURCE_FILE_LOADER_EXEC_MODULE'
_COORDINATES = ('caller', 'position', 'offset', 'code_first_line', 'code_name', 'requested_name', 'level', 'fromlist')
_OUTCOMES = {'PY_RETURN': 'RETURNED', 'PY_UNWIND': 'UNWOUND', 'OPEN': 'OPEN'}


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU import initializer observation: ' + detail)


def _inputs(reference):
    with replay_transaction():
        admitted = verify_cpu_import_source_admission(reference)
        prior = read_artifact(admitted['observation'])
        _need(admitted['schema_version'] == 1 and prior['schema_version'] == 2
              and prior['normalized_output_verified'] is True and admitted['modules'],
              'nonempty admitted sources and a qualified v2 paired observation are required')
        initializer = prior['initializer_graph']
        spec, watch, sites = bindings._admit(prior['paired'], initializer, prior['watch'])
        graph, original, _ = _static(initializer)
        _need(graph['program'] == admitted['program'] and graph['entry_obligations'] == admitted['entry_obligations']
              and prior['entry'] == spec['entry'], 'original F4 request and launch scopes differ')
        sources = {}
        for module in admitted['modules']:
            source = module['source']
            key = module['module_name'], source['uri']
            _need(source['schema'] == bindings._SOURCE and sources.setdefault(key, source) == source,
                  'conflicting source identity for one admitted initializer')
            _raw(source)
    return admitted, prior, spec, watch, sites, original, dict(sorted(sources.items()))


def _invocation(spec, root, sites, sources, *, archived=False):
    config, launch = bindings._invocation(spec, root, sites, follow_spawn=True)
    observer = str(root/bindings._OBSERVER)
    config.update(spawn_observer=str(root/_SPAWN_CORE) if archived else observer,
        initializer_directory=str(root/_INITIALIZERS),
        initializer_sources=[dict(module_name=name, source_path=path, source_bytes=_raw(reference))
                             for (name, path), reference in sources.items()],
        max_initializer_activations=_LIMIT)
    if not archived:
        config['observer_core'] = observer
    return config, launch


def _request(row):
    return {key: row[key] for key in _COORDINATES}


def _valid_request(row):
    return (type(row) is dict and set(row) == set(_COORDINATES)
            and all(type(row[key]) is str and '\0' not in row[key] for key in ('caller', 'code_name', 'requested_name'))
            and type(row['position']) is list and len(row['position']) == 4
            and all(bindings._ordinal(value) for value in row['position'])
            and all(bindings._ordinal(row[key]) for key in ('offset', 'code_first_line', 'level'))
            and type(row['fromlist']) is list and all(type(item) is str for item in row['fromlist']))


def _watched_site(row, sites):
    return row['position'] == sites.get(row['caller'], {}).get(
        (row['code_first_line'], row['code_name'], row['offset'], row['requested_name']))


def _selector(row):
    return _key(row['caller'], row['position'], row['requested_name'], row['level'], row['fromlist'])


def _entries(entries, original):
    sites = defaultdict(list)
    for row in original:
        key = _key(row['file']['uri'], row['span'], row['requested_name'], row['level'], row['fromlist'])
        sites[key].append(f"#/imports/{row['ordinal']}")
    slots = defaultdict(list)
    for index, row in enumerate(entries):
        candidates = sites[_selector(row)]
        row.update(original_import=candidates[0] if len(candidates) == 1 else None,
                   import_start_sequence=None, import_return_sequence=None)
        for module in row['modules']:
            module['module_identity'] = None
        slots[row['process_slot']].append(index)
    return slots


def _sidecar(path, process):
    _need(path.resolve(strict=True) == path and path.is_file() and path.stat().st_size <= bindings._PROCESS_BYTES,
          'initializer sidecar path or size is outside replay limits')
    reference = _ref(path, bindings._RAW)
    raw = ast.literal_eval(_raw(reference).decode('utf-8'))
    _need(type(raw) is dict and set(raw) == {'schema', 'schema_version', 'slot', 'pid', 'ppid', 'cpu_ids',
              'activations', 'activations_omitted', 'rejections', 'import_completions'}
          and raw['schema'] == _PROCESS and type(raw['schema_version']) is int and raw['schema_version'] == 1
          and all(type(raw[key]) is type(process[key]) and raw[key] == process[key] for key in ('slot', 'pid', 'ppid', 'cpu_ids'))
          and type(raw['activations']) is list and len(raw['activations']) <= _LIMIT
          and bindings._ordinal(raw['activations_omitted'])
          and type(raw['import_completions']) is list and len(raw['import_completions']) <= bindings._ENTRY_LIMIT
          and type(raw['rejections']) is dict and all(type(key) is str and bindings._pid(value)
              for key, value in raw['rejections'].items()), 'initializer sidecar identity or bound differs')
    return raw, reference


def _claim_sequences(sequences, *values):
    _need(all(bindings._pid(value) for value in values) and len(values) == len(set(values))
          and not sequences.intersection(values), 'local monitor events do not have distinct positive ordinals')
    sequences.update(values)


def _completions(raw, entries, offsets, lexical, sequences):
    result = {}
    last = 0
    for completion in raw:
        _need(type(completion) is dict and set(completion) ==
              {'entry_index', 'start_sequence', 'return_sequence', 'module_identities'}, 'invalid watched completion')
        index, start, returned = (completion[key] for key in ('entry_index', 'start_sequence', 'return_sequence'))
        _claim_sequences(sequences, start, returned)
        _need(bindings._ordinal(index) and index < len(offsets) and index not in result
              and start < returned and last < returned, 'watched completion is outside the local import sequence')
        last = returned
        entry = entries[offsets[index]]
        _need(type(completion['module_identities']) is list, 'invalid returned module identities')
        by_name = {row['name']: row for row in entry['modules']}
        seen = set()
        for identity in completion['module_identities']:
            _need(type(identity) is dict and set(identity) == {'name', 'identity'}
                  and type(identity['name']) is str and identity['name'] in by_name and identity['name'] not in seen
                  and (identity['identity'] is None or bindings._pid(identity['identity'])
                       and by_name[identity['name']]['activation'] != 'ABSENT'),
                  'module token is outside the completed import cache snapshot')
            seen.add(identity['name'])
            by_name[identity['name']]['module_identity'] = identity['identity']
        entry.update(import_start_sequence=start, import_return_sequence=returned)
        result[index] = _request(lexical[index]), entry, offsets[index]
    return result


def _activation(row, slot, source_map, sites, completions, sequences):
    _need(type(row) is dict and set(row) == {'module_name', 'source_path', 'start_sequence', 'terminal_sequence',
              'terminal', 'code_match', 'loader', 'requester', 'watched_request', 'watched_entry_index', 'module_identity'}
          and type(row['module_name']) is str and type(row['source_path']) is str and row['code_match'] is True
          and row['loader'] == _LOADER and row['terminal'] in _OUTCOMES and bindings._pid(row['module_identity']),
          'unexpected initializer activation shape or loader')
    source = source_map.get((row['module_name'], row['source_path']))
    _need(source is not None, 'initializer source is outside the exact admitted files')
    start, terminal = row['start_sequence'], row['terminal_sequence']
    _claim_sequences(sequences, start)
    if row['terminal'] == 'OPEN':
        _need(terminal is None, 'open initializer has a terminal sequence')
    else:
        _claim_sequences(sequences, terminal)
        _need(start < terminal, 'initializer terminal precedes its entry')
    requester, watched = row['requester'], row['watched_request']
    _need(requester is None or _valid_request(requester), 'invalid lexical requesting frame')
    _need(watched is None or _valid_request(watched) and _watched_site(watched, sites),
          'enclosing request is outside the original watched bytecode')
    import_index = row['watched_entry_index']
    enclosing = enclosing_start = enclosing_return = None
    cache_match = direct = False
    if import_index is not None:
        _need(bindings._ordinal(import_index) and import_index in completions, 'missing exact enclosing completion')
        raw_request, completed, enclosing = completions[import_index]
        enclosing_start, enclosing_return = completed['import_start_sequence'], completed['import_return_sequence']
        _need(watched == raw_request and terminal is not None and enclosing_start < start < terminal < enclosing_return,
              'initializer crosses the paired import request interval')
        cache_match = any(module['name'] == row['module_name'] and module['source'] == source
            and module['module_identity'] == row['module_identity'] for module in completed['modules'])
        direct = (row['terminal'] == 'PY_RETURN' and requester == watched and watched['level'] == 0
                  and watched['requested_name'] == row['module_name'] and completed['original_import'] is not None
                  and cache_match)
    return dict(process_slot=slot, module_name=row['module_name'], source=source,
                outcome=_OUTCOMES[row['terminal']], structural_code_match=True, standard_source_loader=True,
                activation_start_sequence=start, activation_terminal_sequence=terminal,
                enclosing_import=enclosing, enclosing_import_start_sequence=enclosing_start,
                enclosing_import_return_sequence=enclosing_return, requester=requester, watched_request=watched,
                module_identity=row['module_identity'], returned_cache_identity_match=cache_match, direct_import=direct)


def _activations(root, report, original, sites, sources):
    folder = root/_INITIALIZERS
    _need(folder.resolve(strict=True) == folder and folder.is_dir(), 'expected a canonical initializer sidecar directory')
    paths = set(folder.iterdir())
    _need(len(paths) <= bindings._SPAWN_LIMIT + 1 and
          sum(path.stat().st_size for path in paths) <= bindings._TOTAL_PROCESS_BYTES, 'initializer evidence exceeds the byte budget')
    activations = []
    entries = report['entries']
    slots = _entries(entries, original)
    for process in report['processes']:
        slot = process['slot']
        path = folder/f'initializer-process-{slot:04d}.txt'
        process.update(initializer_status='UNOBSERVED_MISSING_SIDECAR', initializer_capture=None,
                       activations_omitted=None, activation_rejections=None)
        if path not in paths:
            continue
        _need(process['status'] == bindings._OBSERVED, 'initializer sidecar has no observed owner process')
        paths.remove(path)
        raw, reference = _sidecar(path, process)
        lexical = ast.literal_eval(_raw(process['capture']).decode('utf-8'))['entries']
        offsets = slots[slot]
        _need(len(lexical) == len(offsets), 'initializer sidecar cannot locate its import entries')
        sequences = set()
        completions = _completions(raw['import_completions'], entries, offsets, lexical, sequences)
        observed = [_activation(row, slot, sources, sites, completions, sequences) for row in raw['activations']]
        ordered = sorted(observed, key=lambda row: row['activation_start_sequence'])
        _need(observed == ordered, 'initializer records lost their process-local entry order')
        activations.extend(ordered)
        process.update(initializer_status='OBSERVED', initializer_capture=reference,
                       activations_omitted=raw['activations_omitted'], activation_rejections=raw['rejections'])
    _need(not paths, 'initializer sidecar has no original or direct spawn slot')
    return activations


def _record(root, admission_ref, inputs):
    admitted, prior, spec, watch, sites, original, sources = inputs
    script = _ref(root/bindings._OBSERVER, bindings._SOURCE)
    stored = _raw(script)
    archived = script['sha256'] == _ARCHIVED_MONITOR
    config, launch = _invocation(spec, root, sites, sources, archived=archived)
    spawn = _ref(root/_SPAWN_CORE, bindings._SOURCE) if archived else script
    request = _ref(root/bindings._CONFIG, bindings._RAW)
    if archived:
        _raw(spawn)
        _need(spawn['sha256'] == bindings._ARCHIVED_SPAWN, 'archived monitor spawn observer changed')
    else:
        _need(stored == bindings._BOOTSTRAP.read_bytes(), 'initializer monitor changed')
    _need(_raw(request) == (repr(config) + '\n').encode(), 'pinned request changed')
    run = _ref(root/'run.json', bindings._STATUS)
    timing = read_artifact(run)
    _need(type(timing['status']) is int and timing['status'] == 0
          and type(timing['started_ns']) is type(timing['reaped_ns']) is int
          and 0 <= timing['started_ns'] < timing['reaped_ns'], 'original diagnostic process failed')
    stdout, stderr = (_ref(root/name, bindings._RAW) for name in ('stdout.bin', 'stderr.bin'))
    _need(not _raw(stderr), 'diagnostic changed original stderr')
    publications = cpu_entry_replay._published(root)
    _observation(spec, stdout, publications)
    report = dict(artifacts=dict(bootstrap=script, spawn_observer=spawn, request=request, run=run,
                                 stdout=stdout, stderr=stderr, publications=publications))
    bindings._add_processes(report, root, spec, sites, timing, observer_core=None if archived else script)
    activations = _activations(root, report, original, sites, sources)
    original_arm = spec['arms']['original']
    report.update(schema=SCHEMA, schema_version=1, admission=admission_ref,
                  observation=admitted['observation'], program=admitted['program'],
                  entry_obligations=admitted['entry_obligations'], paired=prior['paired'], target=admitted['target'],
                  initializer_graph=prior['initializer_graph'], entry=spec['entry'], watch=watch,
                  activations=activations, activation_order='PROCESS_SLOT_THEN_LOCAL_EVENT_ORDER',
                  max_initializer_activations_per_process=_LIMIT, boundary=spec.get('scope', 'ARGV_PROCESS'),
                  original_argv=original_arm['argv'], diagnostic_argv=launch, interpreter=original_arm['executable'],
                  cwd=spec['cwd'], cpu_ids=spec['cpu_ids'], environment_sha256=_environment_digest(original_arm['environment']),
                  process_launches=1, instrumented=True, normalized_output_verified=True, clean_benchmark=False,
                  source_entry_executed=True, importer_effects_closed=False, deployment_authorized=False)
    return report


def measure_cpu_import_initializer_observation(admission_ref, out):
    """Run one original diagnostic with the qualified source and spawn selectors."""
    inputs = _inputs(admission_ref)
    _, _, spec, _, sites, _, sources = inputs
    root = Path(out).absolute()
    _need(root.resolve() == root and not root.exists() and getattr(sys, 'monitoring', None) is not None
          and sites and set(spec['cpu_ids']) <= os.sched_getaffinity(0), 'fresh output and qualified CPython CPU mask required')
    config, launch = _invocation(spec, root, sites, sources)
    (root/'published').mkdir(parents=True, exist_ok=False)
    (root/_INITIALIZERS).mkdir()
    (root/bindings._PROCESSES).mkdir()
    (root/bindings._OBSERVER).write_bytes(bindings._BOOTSTRAP.read_bytes())
    (root/bindings._CONFIG).write_text(repr(config) + '\n', encoding='utf-8')
    transcript = spec['transcript']
    with (root/'stdout.bin').open('xb') as stdout, (root/'stderr.bin').open('xb') as stderr:
        started = monotonic_ns()
        with subprocess.Popen(launch, cwd=spec['cwd'], env=spec['arms']['original']['environment'],
                stdout=subprocess.PIPE if transcript else stdout, stderr=stderr,
                stdin=subprocess.PIPE if transcript else subprocess.DEVNULL) as process:
            launcher = process.pid
            if transcript:
                cpu_entry_replay._pace(process, stdout, transcript)
            status = process.wait()
        reaped = monotonic_ns()
    _write(root/'run.json', dict(schema=bindings._STATUS, started_ns=started, reaped_ns=reaped,
                               status=status, launcher_pid=launcher))
    report = _record(root, admission_ref, inputs)
    _write(root/'result.json', report)
    return report


def verify_cpu_import_initializer_observation(reference):
    """Recompute the exact diagnostic projection from its retained sidecars."""
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and saved['schema_version'] == 1 and path.name == 'result.json'
          and path.resolve(strict=True) == path, 'expected a canonical initializer observation receipt')
    admission = saved['admission']
    expected = _record(path.parent, admission, _inputs(admission))
    _need(saved == expected, 'retained initializer observation differs on cold replay')
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar observe-cpu-import-initializers', description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--admission', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.admission is not None and args.out is not None:
        report = measure_cpu_import_initializer_observation(_ref(args.admission, ADMISSION), args.out)
        reference = _ref(args.out/'result.json', SCHEMA)
    elif args.check is not None and args.out is None:
        reference = _ref(args.check, SCHEMA)
        report = verify_cpu_import_initializer_observation(reference)
    else:
        parser.error('use --admission and --out, or --check')
    print(json.dumps(dict(reference=reference, activations=len(report['activations'])), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
