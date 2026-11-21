"""Build and check an independent C++20 project from admitted native sources."""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _dependencies


def _cmake_arg(value):
    # CMake expands list separators and generator expressions when applying
    # target properties, even when they were literal tokens in the script.
    if any(token in value for token in (';', '$<', '\n', '\r')):
        raise ValueError('unsupported CMake argument: ' + repr(value))
    delimiter = '='
    while f']{delimiter}]' in value:
        delimiter += '='
    return f'[{delimiter}[{value}]{delimiter}]'


def write_native_project(root, name, *, sources=('src/main.cpp',), include_dirs=(),
                         compile_options=(), link_options=(), libraries=(), runtime_dirs=(),
                         install_dirs=(), install_launcher=False, preserve_install_permissions=False,
                         with_c_provider=False):
    """Write the shared project format with profile-specific source and link inputs."""
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name):
        raise ValueError('expected a CMake target name')
    languages = 'C CXX' if with_c_provider else 'CXX'
    lines = ['cmake_minimum_required(VERSION 3.20)', f'project({name} LANGUAGES {languages})',
             f'add_executable({name} ' + ' '.join(map(_cmake_arg, sources)) + ')',
             f'set_target_properties({name} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)']
    if with_c_provider:
        lines.append(f'set_target_properties({name} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)')
    for command, args in (('target_include_directories', ('include/scar_generated', 'src', *include_dirs)),
                          ('target_compile_options', compile_options), ('target_link_options', link_options),
                          ('target_link_libraries', libraries)):
        if args:
            lines.append(f'{command}({name} PRIVATE ' + ' '.join(map(_cmake_arg, args)) + ')')
    if runtime_dirs:
        paths = ' '.join(map(_cmake_arg, runtime_dirs))
        lines.append(f'set_property(TARGET {name} PROPERTY BUILD_RPATH {paths})')
        lines.append(f'set_property(TARGET {name} PROPERTY INSTALL_RPATH {paths})')
    lines.append(f'install(TARGETS {name} RUNTIME DESTINATION bin)')
    for directory, destination in install_dirs:
        permissions = ' USE_SOURCE_PERMISSIONS' if preserve_install_permissions else ''
        lines.append(f'install(DIRECTORY {_cmake_arg(directory)} DESTINATION {_cmake_arg(destination)}{permissions})')
    if install_launcher:
        lines.append('install(PROGRAMS launch.sh DESTINATION .)')
    (root / 'CMakeLists.txt').write_text('\n'.join(lines) + '\n')


def build_native_project(root, name, compiler, *, expected_libraries=(), require_cpu_only=False):
    """Build into a temporary directory, then check and return the installed ELF."""
    if not compiler:
        raise ValueError('an available C++ compiler is required')
    env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    env.setdefault('PATH', os.defpath)
    cmake = shutil.which('cmake', path=env['PATH'])
    if cmake is None:
        raise ValueError('CMake is required to build the native project')
    cmake = str(Path(cmake).resolve())
    # CXX can name a compiler with options or a compiler launcher.
    env['CXX'] = shlex.join(compiler)
    with TemporaryDirectory(prefix='.scar-cmake-', dir=root.parent) as temporary:
        configure = [cmake, '-S', str(root), '-B', temporary, '-DCMAKE_BUILD_TYPE=Release']
        build = [cmake, '--build', temporary, '--parallel', '2']
        prefix = Path(temporary) / 'install'
        install = [cmake, '--install', temporary, '--prefix', str(prefix)]
        logs = []
        for command in (configure, build, install):
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            logs.append(result.stdout + result.stderr)
            if result.returncode:
                raise ValueError('C++20 project build failed\n' + logs[-1])
        executable = root / 'bin' / name
        executable.parent.mkdir(exist_ok=True)
        (prefix / 'bin' / name).rename(executable)
    listing = subprocess.check_output(['ldd', str(executable)], env=env, text=True)
    dependencies = _dependencies(listing)
    if (any(not dependencies.get(path.name) or Path(dependencies[path.name]).resolve() != path.resolve()
            for path in expected_libraries)
            or any(name.startswith(('libpython', 'libtorch_python')) for name in dependencies)):
        raise ValueError('native project did not link the required non-Python provider libraries')
    if require_cpu_only:
        from scar.runtime.native_cpu_linkage import qualify_cpu_dependencies

        qualify_cpu_dependencies(dependencies)
    validation = root / 'validation'
    validation.mkdir(exist_ok=True)
    (validation / 'build.log').write_text('\n'.join(logs))
    (validation / 'ldd.txt').write_text(listing)
    return executable, dependencies
