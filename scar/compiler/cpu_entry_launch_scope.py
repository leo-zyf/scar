"""Project original CPU launch requirements from opt-in retained evidence."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

from scar.analysis.cpu_stream_graph import _OBSERVER, _retained
from scar.analysis.scalar_recursive_driver import CAPTURE
from scar.compiler.cpu_entry_replay import SCHEMA as PAIRED_REPLAY
from scar.compiler.plan_aten_boundary import _target
from scar.ir.execution_recipe import artifact_reference, read_artifact


def _require(condition, message):
    if not condition:
        raise ValueError('retained CPU launch: ' + message)


def _context(interpreter, cpu_ids, cwd, environment):
    path = Path(interpreter)
    directory = Path(cwd)
    _require(path.is_absolute() and directory.is_absolute() and directory.is_dir()
             and directory.resolve(strict=True) == directory,
             'absolute original interpreter and canonical working directory required')
    path = path.resolve(strict=True)
    # These archive schemas do not record Python's version or cache tag. Recover
    # them only from the exact executable already hosting the compiler process.
    _require(path == Path(sys.executable).resolve(strict=True),
             'captured interpreter must match the compiler executable for the Python ABI')
    _require(type(cpu_ids) is list and bool(cpu_ids)
             and all(type(cpu) is int and cpu >= 0 for cpu in cpu_ids)
             and sorted(set(cpu_ids)) == cpu_ids, 'canonical original CPU cohort required')
    _require(type(environment) is dict and all(type(name) is str and name
             and '=' not in name and '\x00' not in name and type(value) is str and '\x00' not in value
             for name, value in environment.items()), 'invalid retained original environment')
    names = {'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'}
    names.update(name for name in environment if name.startswith(('OMP_', 'GOMP_', 'KMP_')))
    environment_sha256 = sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()
    return dict(working_directory=str(directory), python_executable=str(path),
                python_version=sys.version, python_cache_tag=sys.implementation.cache_tag,
                process_affinity_cpu_ids=cpu_ids,
                native_thread_environment={name: environment.get(name) for name in sorted(names)},
                environment_sha256=environment_sha256)


def _paired(reference, entry, argv, stream_capture):
    report = read_artifact(reference)
    _require(reference['schema'] == PAIRED_REPLAY and report['schema_version'] == 1
             and report['capture'] == stream_capture,
             'paired observation and explicit stream capture differ')
    compilation = read_artifact(report['compilation'])
    spec = report['invocation']
    _require(compilation['entry'] == spec['entry'] == entry and compilation['argv'] == argv,
             'paired observation identifies another original entry or argv')
    original = spec['arms']['original']
    read_artifact(original['executable'], json_document=False)
    return _context(original['executable']['uri'], spec['cpu_ids'], spec['cwd'], original['environment'])


def verify_paired_original_scope(scope, reference):
    """Check exact original launch identity for evidence added after planning."""
    report = read_artifact(reference)
    spec = report['invocation']
    original = spec['arms']['original']
    compilation = read_artifact(report['compilation'])
    environment = sha256(json.dumps(sorted(original['environment'].items()), separators=(',', ':')).encode()).hexdigest()
    _require(reference['schema'] == PAIRED_REPLAY
             and compilation['entry'] == spec['entry']
             and (spec['entry']['uri'], spec['entry']['sha256']) == (scope['entry'], scope['entry_sha256'])
             and compilation['argv'] == scope['argv']
             and spec['cpu_ids'] == scope['process_affinity_cpu_ids']
             and spec['cwd'] == scope['working_directory']
             and str(Path(original['executable']['uri']).resolve(strict=True)) == scope['python_executable']
             and environment == scope['environment_sha256'],
             'paired evidence must match the original entry, argv, CPU cohort and environment')
    return report, compilation


def _stream(reference, scope, entry, paired):
    capture = read_artifact(reference)
    _require(reference['schema'] == CAPTURE and capture['source'] == entry,
             'stream capture identifies another original entry')
    launch = json.loads(read_artifact(capture['captured_launch'], json_document=False))
    command = launch['command']
    _require(type(command) is list and all(type(arg) is str and '\x00' not in arg for arg in command)
             and command.count(_OBSERVER) == 1, 'structured original observer launch required')
    observer = command.index(_OBSERVER)
    _require(observer >= 4, 'original observer must name an interpreter')
    if command[:2] == ['/usr/bin/taskset', '-c']:
        value = command[2]
        _require(len(value) <= 10 and value.isascii() and value.isdecimal() and value == str(int(value)),
                 'stream observer must bind one canonical CPU')
        cpus = [int(value)]
    else:
        _require(paired is not None and observer == 4,
                 'stream observer must bind one CPU or have a verified paired observation')
        cpus = paired['process_affinity_cpu_ids']
    delta = _context(command[observer - 4], cpus, launch['cwd'], launch['environment'])
    _require(paired is None or delta == paired, 'paired measurement and stream observer launch differ')
    _retained(reference, dict(scope, **delta, cpu_stream_capture=reference))
    return delta


def project_cpu_entry_launch_scope(scope, *, stream_capture=None, paired_observation=None):
    """Return launch requirements for later mandatory F4 and provider replay."""
    _require(stream_capture is not None or paired_observation is not None,
             'retained original launch authority required')
    entry = artifact_reference(Path(scope['entry']).resolve(strict=True), 'scar.python-source')
    _require(entry['uri'] == scope['entry'] and entry['sha256'] == scope['entry_sha256'],
             'original F4 source identity changed')
    delta = None if paired_observation is None else _paired(paired_observation, entry, scope['argv'], stream_capture)
    if stream_capture is not None:
        delta = _stream(stream_capture, scope, entry, delta)
    cpus = delta['process_affinity_cpu_ids']
    target, _ = _target(Path(scope['target']['uri']).parent, cpus, allow_cpu_subset=True)
    _require(asdict(target) == scope['target'], 'original CPU cohort must belong to the exact F4 target')
    return delta
