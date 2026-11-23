"""Build, link and validate the common C++20 project artifact."""

from decimal import Decimal
from hashlib import sha256
import os
from pathlib import Path
import re
import statistics
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter_ns

from scar.contract import Member, read_artifact, verify_artifact


_TARGET = 'scar_entry'
_SOURCES = {'.cpp', '.cc', '.cxx', '.C', '.c++', '.cp'}


def _cmake_arg(value):
    if any(token in value for token in (';', '$<', '\n', '\r')):
        raise ValueError(f'unsupported CMake input: {value!r}')
    delimiter = '='
    while f']{delimiter}]' in value:
        delimiter += '='
    return f'[{delimiter}[{value}]{delimiter}]'


def _cmake_project(root, config):
    sources = config['sources']
    if not sources or len(sources) != len(set(sources)):
        raise ValueError('C++20 delivery requires distinct source units')
    for name in sources:
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.suffix not in _SOURCES
                or not (root / relative).is_file()):
            raise ValueError(f'missing C++20 translation unit: {name}')
    lines = ['cmake_minimum_required(VERSION 3.20)', 'project(scar_entry LANGUAGES CXX)',
             'add_executable(scar_entry ' + ' '.join(map(_cmake_arg, sources)) + ')',
             'set_target_properties(scar_entry PROPERTIES CXX_STANDARD 20 '
             'CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)']
    includes = ('include/scar_generated', 'src', *config.get('include_dirs', ()))
    for command, arguments in (('target_include_directories', includes),
                               ('target_compile_options', config.get('compile_options', ())),
                               ('target_link_libraries', config.get('libraries', ())),
                               ('target_link_options', config.get('link_options', ()))):
        if arguments:
            lines.append(f'{command}(scar_entry PRIVATE ' + ' '.join(map(_cmake_arg, arguments)) + ')')
    for name, (directories, flags) in config.get('source_settings', {}).items():
        if name not in sources:
            raise ValueError(f'source settings have no C++20 unit: {name}')
        source = _cmake_arg(name)
        if directories:
            lines.append('set(scar_source_includes ' + ' '.join(map(_cmake_arg, directories)) + ')')
            lines.append('list(TRANSFORM scar_source_includes PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")')
            lines.append(f'set_property(SOURCE {source} PROPERTY INCLUDE_DIRECTORIES ${{scar_source_includes}})')
        if flags:
            lines.append(f'set_property(SOURCE {source} PROPERTY COMPILE_OPTIONS '
                         + ' '.join(map(_cmake_arg, flags)) + ')')
    if config.get('runtime_dirs'):
        paths = ' '.join(map(_cmake_arg, config['runtime_dirs']))
        lines.extend((f'set_property(TARGET scar_entry PROPERTY BUILD_RPATH {paths})',
                      f'set_property(TARGET scar_entry PROPERTY INSTALL_RPATH {paths})'))
    lines.append('install(TARGETS scar_entry RUNTIME DESTINATION bin)')
    (root / 'CMakeLists.txt').write_text('\n'.join(lines) + '\n')


def build_project(project, config, compiler, env):
    _cmake_project(project, config)
    logs = []
    with TemporaryDirectory(prefix='.build-', dir=project.parent) as build_dir:
        parallel = min(len(os.sched_getaffinity(0)), 8)
        commands = (['cmake', '-S', str(project), '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release',
                     f'-DCMAKE_CXX_COMPILER={compiler}'],
                    ['cmake', '--build', build_dir, '--parallel', str(parallel)],
                    ['cmake', '--install', build_dir, '--prefix', str(project)])
        for command in commands:
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
            logs.extend((run.stdout, run.stderr))
            if run.returncode:
                raise ValueError(f'C++20 project build failed\n{run.stderr}\n{run.stdout}')
    (project / 'build.log').write_text(''.join(logs))
    executable = project / 'bin' / _TARGET
    listing = subprocess.run(['ldd', str(executable)], env=env, capture_output=True, text=True, timeout=30)
    if listing.returncode:
        raise ValueError(f'could not inspect linked C++20 dependencies\n{listing.stderr}')
    forbidden = ('libpython', 'libtorch_python', 'libgfortran')
    if 'not found' in listing.stdout or any(line.lstrip().startswith(forbidden) for line in listing.stdout.splitlines()):
        raise ValueError('C++20 executable has unresolved or Python/Fortran runtime dependencies')
    (project / 'ldd.txt').write_text(listing.stdout)
    return executable


_TIMING = re.compile(rb'(time:) +[0-9]+\.[0-9]+')
_DECIMAL = re.compile(rb'[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)[DEde][+-]?[0-9]+')


def verify_exports(executable, exports):
    if not exports:
        return
    symbols = subprocess.check_output(['nm', '-P', '--defined-only', str(executable)], text=True, timeout=30)
    missing = set(exports) - {line.split()[0] for line in symbols.splitlines() if line.split()}
    if missing:
        raise ValueError(f'C++20 executable did not preserve link symbols: {sorted(missing)}')


def _scientific_output(data):
    values = [Decimal(value.upper().replace(b'D', b'E').decode()) for value in _DECIMAL.findall(data)]
    return _DECIMAL.sub(b'<decimal>', data).split(), values


def _matches(comparison, original, generated):
    if original.returncode or generated.returncode or original.stderr != generated.stderr:
        return False
    if comparison == 'timed-fields':
        return _TIMING.sub(rb'\1 <timing>', original.stdout) == _TIMING.sub(rb'\1 <timing>', generated.stdout)
    if comparison == 'scientific-text':
        return _scientific_output(original.stdout) == _scientific_output(generated.stdout)
    return original.stdout == generated.stdout


def validate_cases(validation, bundle, executable, out, cpus, seconds, repetitions, warmups):
    cases = validation.get('cases', ())
    if not cases:
        return []
    baseline = validation['baseline']
    original_ref = Member(**baseline['executable']).resolve(bundle)
    original = [original_ref.path, *baseline['argv']]
    verify_artifact(original_ref)
    if any(type(cpu) is not int or cpu not in os.sched_getaffinity(0) for cpu in cpus):
        raise ValueError('selected CPU is unavailable for native validation')
    affinity = ['taskset', '-c', ','.join(map(str, cpus))] if cpus else []
    validated = []
    for case in cases:
        name = case['id']
        comparison = case.get('comparison', 'exact')
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name):
            raise ValueError(f'invalid native case name: {name}')
        if comparison not in ('exact', 'timed-fields', 'scientific-text'):
            raise ValueError(f'unsupported native case comparison: {comparison}')
        folder = out / 'validation' / name
        folder.mkdir(parents=True)
        stdin = read_artifact(Member(**case['input']).resolve(bundle)) if 'input' in case else b''
        arguments = case.get('argv', ())
        arms = {'original': (original, baseline['cwd']), 'generated': ([str(executable), *original[1:]], folder)}
        env = dict(os.environ, LC_ALL='C')
        deadline = monotonic() + seconds
        pairs = []
        for index in range(warmups + repetitions):
            outputs, measurements = {}, {}
            order = tuple(arms) if index % 2 == 0 else tuple(reversed(arms))
            for label in order:
                command, cwd = arms[label]
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'native case exceeded its time budget: {name}')
                start = perf_counter_ns()
                run = subprocess.run([*affinity, *command, *arguments], cwd=cwd, env=env, input=stdin,
                                     capture_output=True, timeout=remaining)
                measurements[label] = dict(ns=perf_counter_ns() - start, exit=run.returncode)
                outputs[label] = run
                (folder / f'{index}-{label}.stdout').write_bytes(run.stdout)
                (folder / f'{index}-{label}.stderr').write_bytes(run.stderr)
            equal = _matches(comparison, outputs['original'], outputs['generated'])
            pairs.append(dict(warmup=index < warmups, order=order, arms=measurements, matches=equal))
            if not equal:
                break
        timing = {}
        if all(pair['matches'] for pair in pairs):
            for label in arms:
                samples = [pair['arms'][label]['ns'] for pair in pairs if not pair['warmup']]
                timing[label] = dict(median_ns=statistics.median(samples), min_ns=min(samples), max_ns=max(samples),
                                     stddev_ns=statistics.pstdev(samples))
        validated.append(dict(case=name, matches=all(pair['matches'] for pair in pairs), comparison=comparison,
                              stdin_sha256=sha256(stdin).hexdigest(), pairs=pairs, timing=timing))
    verify_artifact(original_ref)
    return validated
