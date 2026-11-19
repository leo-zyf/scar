"""Publish the runtime files of an already verified CPU-only native project."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_elf import PACKAGE_RUNPATH, relocate_cpu_executable
from scar.codegen.native_provider_bundle import _content
from scar.runtime.validate_native_cpu_package import verify_native_cpu_package
from scar.runtime.validate_native_project import validate_native_outputs, verify_native_project


def _launch_and_data(project, manifest):
    if manifest['schema'] == 'scar.aten-recurrence-delivery':
        from scar.compiler.compile_aten_recurrence import _render_recurrence_launch

        return _render_recurrence_launch(manifest['cpus']), {}
    from scar.compiler.compile_exported_aten import _render_launch

    profiles = manifest.get('profiles', [None])
    masks, weights = {}, {}
    for label in profiles:
        plan = 'plan' if label is None else f'plan/profiles/{label}'
        execution = json.loads((project / plan / 'execution.json').read_bytes())
        if not masks:
            inputs, outputs = execution['inputs'], execution['output_shapes']
        masks[label] = execution['cpu_ids']
        for row in execution['inputs']:
            if row['kind'] != 'WEIGHT':
                continue
            name = row['path']
            location = Path(name)
            if location.parent != Path('data') or location.as_posix() != name:
                raise ValueError('native weight requires its qualified package-local data path')
            receipt = {key: row[key] for key in ('sha256', 'size_bytes')}
            if receipt != manifest['files'].get(name):
                raise ValueError('native weight differs from its published file ABI')
            weights[name] = receipt
    return _render_launch(inputs, len(outputs), masks), weights


def _copy_member(source, target, expected=None):
    target.parent.mkdir(exist_ok=True)
    found = _content(source, copy_to=target)
    if expected is not None and expected != found:
        raise ValueError(f'packaged file differs from its source receipt: {source.name}')
    return found


def export_native_cpu(project, manifest_sha256, *, out):
    """Keep the verified launch ABI and snapshot the CPU provider shared libraries."""
    project, proof = verify_native_project(project, manifest_sha256)
    if 'cpu_linkage' not in proof:
        raise ValueError('build the source project with --require-cpu-only before packaging')
    manifest_path = project / 'artifact.json'
    manifest = json.loads(manifest_path.read_bytes())
    if _content(manifest_path)['sha256'] != manifest_sha256:
        raise ValueError('native project manifest changed after source verification')
    dependencies = manifest['dependencies']
    provider_root = Path(dependencies['libtorch_cpu.so']).parent
    providers = {name: f'lib/{name}' for name, path in dependencies.items()
                 if path is not None and Path(path).parent == provider_root}
    locks = json.loads((project / 'plan/dependencies.json').read_bytes())
    provider_locks = {row['path']: {key: row[key] for key in ('sha256', 'size_bytes')}
                      for row in (locks['libraries'] if type(locks) is dict else locks)}
    launcher, weights = _launch_and_data(project, manifest)
    if (project / 'launch.sh').read_bytes() != launcher.encode():
        raise ValueError('published launcher differs from its admitted input, output and CPU binding')
    if not {'libtorch_cpu.so', 'libc10.so'} <= providers.keys():
        raise ValueError('the native project needs co-installed CPU provider libraries')
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(project):
        raise ValueError('a fresh output outside the source native project is required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-native-cpu-', dir=out.parent) as temporary:
        stage = Path(temporary)
        files = {}
        for name, relative in providers.items():
            source = dependencies[name]
            if Path(relative).parent != Path('lib'):
                raise ValueError('expected a flat native provider filename')
            files[relative] = _copy_member(Path(source), stage / relative, provider_locks.get(source))
        for relative, receipt in weights.items():
            files[relative] = _copy_member(project / relative, stage / relative, receipt)
        license_paths = sorted(provider_root.parent.parent.glob('torch-*.dist-info/licenses/LICENSE'))
        if len(license_paths) == 1:
            relative = 'licenses/LibTorch.txt'
            files[relative] = _copy_member(license_paths[0], stage / relative)
        executable = manifest['executable']
        source_elf = project / executable
        (stage / 'bin').mkdir()
        relocate_cpu_executable(source_elf, stage / executable, original_runpath=str(provider_root))
        original = manifest['files'][executable]
        if _content(source_elf) != original:
            raise ValueError('native executable changed during relocation')
        files[executable] = _content(stage / executable)
        entry = stage / 'launch.sh'
        entry.write_bytes(launcher.encode())
        entry.chmod(0o755)
        files['launch.sh'] = _content(entry)
        package = dict(schema='scar.native-cpu-package', schema_version=1, executable=executable,
                       source_project=dict(schema=manifest['schema'], manifest_sha256=manifest_sha256,
                                           executable=original),
                       cpu_linkage=proof['cpu_linkage'], providers=providers, files=files,
                       runtime_search_path=PACKAGE_RUNPATH)
        record = stage / 'artifact.json'
        record.write_text(json.dumps(package, sort_keys=True, indent=2, allow_nan=False) + '\n')
        digest = _content(record)['sha256']
        verify_native_cpu_package(stage, digest)
        stage.rename(out)
    return dict(path=str(out / 'artifact.json'), sha256=digest, providers=providers)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar export-native-cpu', description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--project', help='source CPU-only native project')
    mode.add_argument('--check', help='verify a published runtime directory')
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out')
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if args.project:
        if not args.out or arguments or args.reference:
            parser.error('--project requires --out and takes no runtime inputs')
        result = export_native_cpu(args.project, args.manifest_sha256, out=args.out)
    else:
        if args.out or arguments and not args.reference:
            parser.error('--check takes runtime inputs only with --reference')
        root, result = verify_native_cpu_package(args.check, args.manifest_sha256)
        if args.reference:
            result['bitwise_output_receipts'] = validate_native_outputs(root, arguments, args.reference)
    print(json.dumps(result, sort_keys=True))
    return 0
