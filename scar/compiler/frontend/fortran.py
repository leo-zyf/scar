"""Capture compiler-resolved Fortran units and their source/SSA evidence."""

from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory


def _run(args, cwd, *, accepted=(0,), env=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=180, env=env)
    if result.returncode not in accepted:
        raise ValueError(f'{Path(args[0]).name} failed with status {result.returncode}\n{result.stderr.strip()}')
    return result.stdout


def _tool(config, name):
    path = config.get(name) or shutil.which(name)
    if not path or not Path(path).is_file():
        raise ValueError(f'Fortran entry requires {name}. Supply its path in fortran.{name}.')
    return str(Path(path).resolve())


def _units(manifest):
    root = Path(manifest['source_root']).resolve()
    for row in manifest['units']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Fortran source unit is outside the source root or missing. {path}')
        yield row['path'], path, tuple(row.get('flags', ()))


def capture_fortran(manifest, out):
    """Record compiler-resolved namespaces, kinds, effects, and SSA with source spans."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = manifest.get('fortran', {})
    compiler = _tool(config, 'gfortran')
    units = []
    with TemporaryDirectory(prefix='.fortran-capture-', dir=out) as temporary:
        modules = Path(temporary)
        version = _run([compiler, '--version'], modules).splitlines()[0]
        for index, (name, path, flags) in enumerate(_units(manifest)):
            symbols = out / f'unit-{index}.fortran-original'
            ssa = out / f'unit-{index}.ssa'
            calls = ssa.with_suffix('.cgraph')
            command = [compiler, *flags, '-J', str(modules), '-I', str(modules)]
            ssa_switch = f'-fdump-tree-ssa-raw-blocks-lineno-vops={ssa}'
            text = _run([*command, '-O0', '-c', '-fdump-fortran-original', ssa_switch,
                         f'-fdump-ipa-cgraph={calls}',
                         '-o', str(modules / f'unit-{index}.o'), str(path)], modules)
            symbols.write_text(text, encoding='utf-8')
            units.append({'path': name, 'frontend': 'gfortran', 'format': 'gcc-tree-ssa',
                          'raw': str(ssa), 'call_table': str(calls),
                          'semantics': str(symbols), 'flags': list(flags),
                          'procedures': re.findall(r'^\s*procedure name = (\w+)', text, re.MULTILINE)})
    return {'frontend': 'gfortran', 'version': version, 'units': units}
