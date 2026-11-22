"""Lock each native compilation unit with its own source dialect and flags."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

from scar.artifacts import file_reference
from scar.ir.program.providers import ProviderArtifact


@dataclass(frozen=True, slots=True)
class NativeInputs:
    manifest: dict
    request: ProviderArtifact


def lock_units(path: Path, out: Path) -> NativeInputs:
    manifest = json.loads(path.read_bytes())
    if (manifest.get('schema'), manifest.get('schema_version')) != ('scar.native-entry', 1):
        raise ValueError('expected scar.native-entry schema v1')
    dialects = {'c', 'c++', 'fortran'}
    language = manifest['language']
    if language not in dialects | {'mixed'}:
        raise ValueError(f'unsupported language: {language}')
    root = Path(manifest['source_root']).resolve(strict=True)
    manifest['source_root'] = str(root)
    receipt = manifest['source_lock']['files']
    for name, expected in receipt.items():
        relative = Path(name)
        source = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not source.is_relative_to(root):
            raise ValueError(f'source path outside of root: {name}')
        actual = file_reference(source, 'source')
        if any(actual[key] != expected[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source differs: {name}')
    units = manifest['units']
    names = {unit['path'] for unit in units}
    if not names or len(names) != len(units):
        raise ValueError('target requires uniquely named translation units')
    if names - receipt.keys():
        raise ValueError('translation units missing source lock: ' + ', '.join(sorted(names - receipt.keys())))
    for unit in units:
        if unit.get('language', language) not in dialects:
            raise ValueError(f'translation unit has no language profile: {unit["path"]}')
    if baseline := manifest.get('baseline'):
        cwd = Path(baseline['cwd']).resolve(strict=True)
        argv = baseline['argv']
        if not argv:
            raise ValueError('baseline requires an executable')
        executable = shutil.which(argv[0]) if '/' not in argv[0] else None
        argv[0] = str(Path(executable or cwd / argv[0]).resolve())
        baseline['cwd'] = str(cwd)
    ids = [case['id'] for case in manifest.get('cases', ())]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[\w.-]+', name) or name in {'.', '..'} for name in ids):
        raise ValueError('case IDs must be unique directory names')
    out.mkdir(parents=True, exist_ok=False)
    request = out / 'request.json'
    request.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return NativeInputs(manifest, ProviderArtifact(**file_reference(request, manifest['schema'])))
