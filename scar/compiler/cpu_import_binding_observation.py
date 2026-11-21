"""Observe the resolved module cache at selected original CPython import sites."""

import argparse
import ast
import dis
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic_ns
from types import CodeType

from scar.analysis.cpu_entry_import_initializers import SCHEMA as INITIALIZERS
from scar.compiler import cpu_entry_replay
from scar.compiler.cpu_original_importtime import _observation, _pair, _raw, _ref, _write
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-original-import-bindings'
_CONFIG = 'observer-input.txt'
_OBSERVER = 'observer.py'
_CAPTURE = 'import-cache.txt'
_STATUS = 'scar.cpu-original-import-binding-run'
_RAW = 'application/octet-stream'
_SOURCE = 'scar.python-source'
_ORIGIN_KINDS = {'ABSOLUTE_FILE', 'BUILT_IN', 'FROZEN', 'NAMESPACE', 'UNKNOWN'}
_BOOTSTRAP = Path(__file__).resolve().parents[1]/'runtime/original_import_bindings.py'


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU original import bindings: ' + reason)


def _codes(code):
    yield code
    for constant in code.co_consts:
        if type(constant) is CodeType:
            yield from _codes(constant)


def _sites(watch):
    files = {}
    for row in watch:
        reference = row['file']
        saved, spans = files.setdefault(reference['uri'], (reference, set()))
        _need(reference == saved and reference['schema'] == _SOURCE, 'watch source identity differs')
        spans.add(tuple(row['span']))
    result = {}
    for path, (reference, spans) in files.items():
        compiled = compile(_raw(reference), path, 'exec', dont_inherit=True, optimize=0)
        sites = {}
        for code in _codes(compiled):
            for inst in dis.get_instructions(code):
                if inst.opname == 'IMPORT_NAME' and tuple(inst.positions) in spans:
                    key = code.co_firstlineno, code.co_qualname, inst.offset, inst.argval
                    _need(key not in sites, 'watched import bytecode coordinate is ambiguous')
                    sites[key] = list(inst.positions)
        if sites:
            result[path] = sites
    return result


def _admit(paired_ref, initializer_ref, watch):
    from scar.analysis.cpu_entry_import_bindings import derive_cpu_entry_import_watch

    spec = _pair(paired_ref)
    graph = read_artifact(initializer_ref)
    _need(initializer_ref['schema'] == INITIALIZERS and read_artifact(graph['program'])['entry'] == spec['entry']['uri'],
          'initializer graph belongs to a different original script')
    expected = derive_cpu_entry_import_watch(initializer_ref)
    if watch is not None:
        _need(watch == expected, 'watch differs from its original F4 requests')
    original = spec['arms']['original']
    _need(sys.implementation.name == 'cpython' and sys.platform == 'linux'
          and Path(sys.executable).resolve() == Path(original['executable']['uri']).resolve()
          and sys.version_info >= (3, 11) and not original['environment'].get('PYTHONOPTIMIZE'),
          'the original unoptimized CPython bytecode and execution context must match the observer')
    return spec, expected, _sites(expected)


def _invocation(spec, root, sites):
    original = spec['arms']['original']
    argv = list(original['argv'])
    entry = spec['entry']['uri']
    prefix = [original['executable']['uri'], '-B']
    if spec['transcript']:
        prefix.append('-u')
    source_index = len(prefix)
    _need(argv[:source_index+1] == [*prefix, entry] and Path(entry).resolve(strict=True) == Path(entry),
          'only the admitted absolute CPython file invocation and exact flags are supported')
    output = cpu_entry_replay._OUTPUT
    slots = [index for index, argument in enumerate(argv) if output in argument]
    expected = {output + '/' + candidate for candidate in spec['files']}
    _need(len(slots) == len(expected) and all(index > source_index and argv[index] in expected for index in slots)
          and set(argv[index] for index in slots) == expected
          and ('output_slot' not in original or slots == [original['output_slot']]),
          'only verified output slots may be relocated')
    for index in slots:
        argv[index] = argv[index].replace(output, str(root/'published'))
    config = dict(original_argv=argv, source_index=source_index, script_directory=str(Path(entry).parent),
                  import_opcode=dis.opmap['IMPORT_NAME'], sites=sites)
    bootstrap = [*prefix, str(root/_OBSERVER), str(root/_CONFIG), str(root/_CAPTURE)]
    launch = ['/usr/bin/taskset', '-c', ','.join(map(str, spec['cpu_ids'])), '/usr/bin/timeout', '-k', '2',
              str(cpu_entry_replay._TIMEOUT), *bootstrap]
    return config, launch


def _source(origin):
    if origin is None:
        return None
    path = Path(origin)
    _need(path.is_absolute() and '\0' not in origin, 'module origin is not an absolute pathname')
    if path.suffix in ('.py', '.pyw') and path.is_file():
        return _ref(path, _SOURCE)
    return None


def _project(raw, sites):
    observed = ast.literal_eval(raw.decode('utf-8'))
    _need(type(observed) is list, 'expected the raw watched import record list')
    result, sources = [], {}
    for row in observed:
        caller, position, offset, name, level, fromlist = (row[key] for key in
            ('caller', 'position', 'offset', 'requested_name', 'level', 'fromlist'))
        coordinate = row['code_first_line'], row['code_name'], offset, name
        _need(position == sites.get(caller, {}).get(coordinate) and type(position) is list
              and type(offset) is int and offset >= 0 and type(level) is int and level >= 0
              and type(fromlist) is list and all(type(value) is str for value in fromlist),
              'raw import observation does not match the watched source bytecode')
        modules, seen = [], set()
        for value in row['modules']:
            module, origin, kind, before, after, same = (value[key] for key in
                ('name', 'origin', 'origin_kind', 'cached_before', 'present_after', 'same_cached_object'))
            _need(type(module) is str and module and module not in seen and all(type(x) is bool for x in (before, after, same))
                  and kind in _ORIGIN_KINDS and (type(origin) is str if kind == 'ABSOLUTE_FILE' else origin is None)
                  and (after or kind == 'UNKNOWN') and (not same or before and after),
                  'invalid observed module cache snapshot')
            seen.add(module)
            if origin not in sources:
                sources[origin] = _source(origin)
            activation = ('ABSENT' if not after else 'NEW_TO_CACHE' if not before else 'CACHED' if same
                          else 'REPLACED_IN_CACHE')
            modules.append(dict(name=module, origin=origin, origin_kind=kind, activation=activation, source=sources[origin]))
        result.append(dict(caller=caller, position=position, offset=offset, requested_name=name, level=level,
                           fromlist=fromlist, modules=modules))
    return result


def _record(root, paired_ref, initializer_ref, spec, watch, sites):
    config, launch = _invocation(spec, root, sites)
    script = _ref(root/_OBSERVER, _SOURCE)
    request = _ref(root/_CONFIG, _RAW)
    _need(_raw(script) == _BOOTSTRAP.read_bytes() and _raw(request) == (repr(config) + '\n').encode(),
          'observer bootstrap or exact watched request changed')
    run = _ref(root/'run.json', _STATUS)
    status = read_artifact(run)
    _need(type(status['status']) is int and status['status'] == 0
          and type(status['started_ns']) is type(status['reaped_ns']) is int
          and 0 <= status['started_ns'] < status['reaped_ns'], 'observed original process failed')
    capture = _ref(root/_CAPTURE, _RAW)
    stdout, stderr = (_ref(root/name, _RAW) for name in ('stdout.bin', 'stderr.bin'))
    _need(not _raw(stderr), 'diagnostic changed original stderr')
    publications = cpu_entry_replay._published(root)
    _observation(spec, stdout, publications)
    original = spec['arms']['original']
    entries = _project(_raw(capture), sites)
    return dict(schema=SCHEMA, schema_version=1, initializer_graph=initializer_ref, paired=paired_ref,
        entry=spec['entry'], watch=watch, entries=entries, original_argv=original['argv'], diagnostic_argv=launch,
        interpreter=original['executable'], cwd=spec['cwd'], cpu_ids=spec['cpu_ids'],
        environment_sha256=_environment_digest(original['environment']),
        artifacts=dict(bootstrap=script, request=request, capture=capture, run=run,
                       stdout=stdout, stderr=stderr, publications=publications),
        boundary=spec.get('scope', 'ARGV_PROCESS'), observation='WATCHED_CPYTHON_IMPORT_NAME_CACHE_SNAPSHOTS',
        origin_source_bytes='POST_RUN_SOURCE_PATH_READBACK', process_launches=1, instrumented=True,
        source_entry_executed=True, normalized_output_verified=True, clean_benchmark=False,
        module_initializer_execution_proven=False, importer_effects_closed=False, deployment_authorized=False)


def measure(paired_ref, initializer_ref, watch, out):
    """Run one original entry with exact paired input, observing only approved imports."""
    spec, watch, sites = _admit(paired_ref, initializer_ref, watch)
    _need(sites and set(spec['cpu_ids']) <= os.sched_getaffinity(0), 'watched bytecode and bound CPUs must be available')
    root = Path(out).resolve()
    config, launch = _invocation(spec, root, sites)
    (root/'published').mkdir(parents=True, exist_ok=False)
    (root/_OBSERVER).write_bytes(_BOOTSTRAP.read_bytes())
    (root/_CONFIG).write_text(repr(config) + '\n', encoding='utf-8')
    transcript = spec['transcript']
    with (root/'stdout.bin').open('xb') as stdout, (root/'stderr.bin').open('xb') as stderr:
        started = monotonic_ns()
        with subprocess.Popen(launch, cwd=spec['cwd'], env=spec['arms']['original']['environment'],
                stdout=subprocess.PIPE if transcript else stdout, stderr=stderr,
                stdin=subprocess.PIPE if transcript else subprocess.DEVNULL) as process:
            if transcript:
                cpu_entry_replay._pace(process, stdout, transcript)
            status = process.wait()
        reaped = monotonic_ns()
    _write(root/'run.json', dict(schema=_STATUS, started_ns=started, reaped_ns=reaped, status=status))
    report = _record(root, paired_ref, initializer_ref, spec, watch, sites)
    _write(root/'result.json', report)
    return report


def verify_cpu_import_binding_observation(reference):
    """Replay the original launch input, watched bytecode and sealed diagnostic outputs."""
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'exact local import observation receipt required')
    paired, initializers = saved['paired'], saved['initializer_graph']
    spec, watch, sites = _admit(paired, initializers, saved['watch'])
    rebuilt = _record(path.parent, paired, initializers, spec, watch, sites)
    _need(saved == rebuilt, 'retained import cache observation differs on cold replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar observe-cpu-import-bindings', description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--paired', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--initializers', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.paired and args.initializers and args.out:
        report = measure(_ref(args.paired, cpu_entry_replay.SCHEMA), _ref(args.initializers, INITIALIZERS), None, args.out)
        reference = _ref(args.out/'result.json', SCHEMA)
    elif args.check and args.initializers is None and args.out is None:
        reference = _ref(args.check, SCHEMA)
        report = verify_cpu_import_binding_observation(reference)
    else:
        parser.error('use --paired, --initializers and --out, or --check alone')
    print(json.dumps(dict(reference=reference, observed_imports=len(report['entries'])), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
