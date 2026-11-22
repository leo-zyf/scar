"""Render the fixed-count standalone ordered sparse digest provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

from scar.codegen import native_ordered_sparse_export as ordered
from scar.codegen.native_sparse_source import project_stream as project_digest
from scar.codegen.native_sparse_source import project_source_csr_stream
from scar.codegen.sparse_native_export import check_sparse_export
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_ordered_profile import SCHEMA as PROFILE, verify_profile_plan


SCHEMA = 'scar.native-ordered-count-provider'
_MEMBERS = (ordered._ABI_HEADER, ordered._DISPATCH_HEADER, ordered._ORDERED_HEADER)
_PROFILE_FILE = 'profile.json'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _part(root, reference):
    return {**reference, 'uri': str(root/reference['uri'])}


def _arc(profile):
    root = Path(profile['anchor']['package']['uri']).parent
    archive = read_artifact(profile['anchor']['package'])
    detached = read_artifact(profile['anchor']['detached'])
    publication = read_artifact(profile['anchor']['publication'])
    resources = profile['resources']
    source = project_source_csr_stream(resources, detached)
    _require(source['input'] == profile['source_stream_input']
        and source['header'] == profile['source_stream_header'], 'standalone wire differs from the checked scope')
    return root, archive, detached, publication, project_digest(source, publication)


def export_profile_solver(profile_plan_path, out):
    """Bind the specialized schedule to the unchanged numeric and digest providers."""
    profile_path = Path(profile_plan_path).resolve(strict=True)
    profile_ref = artifact_reference(profile_path, PROFILE)
    profile = verify_profile_plan(profile_path)
    anchor, archive, _, publication, protocol = _arc(profile)
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(anchor):
        raise ValueError('a fresh package directory outside the original provider is required')
    out.parent.mkdir(parents=True, exist_ok=True)
    local = ordered._local
    with TemporaryDirectory(prefix='.scar-count-provider-', dir=out.parent) as temporary:
        root = Path(temporary)/'output'
        root.mkdir()
        for directory in ('numeric', 'sha256'):
            copytree(anchor/directory, root/directory)
        for name in _MEMBERS[:2]:
            (root/name).write_bytes((anchor/name).read_bytes())
        blas = archive['blas_member']
        (root/ordered._ORDERED_HEADER).write_bytes(ordered._render(profile['resources'], blas))
        (root/_PROFILE_FILE).write_bytes(read_artifact(profile_ref, json_document=False))
        sha256_ref = archive['publication']['provider']
        crypto = ordered._bundled_crypto(root, sha256_ref, publication)
        stream = ordered._write_stream(root, profile['resources'], protocol, crypto)
        manifest = dict(schema=SCHEMA, schema_version=profile['schema_version'], profile_plan=profile_ref,
            profile_member=local(root, _PROFILE_FILE, PROFILE), recipe_id=profile['resources']['recipe_id'],
            count=profile['resources']['case']['call_count'], width=profile['width'],
            numeric=archive['numeric'], sha256_provider=sha256_ref, blas_member=blas,
            headers=[local(root, name, 'scar.cpp-header') for name in _MEMBERS],
            stream=stream, deployment_authorized=False,
            **({'window': profile['window']} if 'window' in profile else {}))
        _write(root/'artifact.json', manifest)
        _check(root, profile)
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return artifact_reference(out/'artifact.json', SCHEMA)


def _check(root, profile):
    local = ordered._local
    manifest = read_artifact(artifact_reference(root/'artifact.json', SCHEMA))
    anchor, archive, detached, publication, protocol = _arc(profile)
    resources = profile['resources']
    numeric = check_sparse_export(root/'numeric')
    _require(manifest['schema_version'] == profile['schema_version']
        and manifest.get('window') == profile.get('window') and ('window' in manifest) == ('window' in profile)
        and manifest['deployment_authorized'] is False
        and manifest['profile_member'] == local(root, _PROFILE_FILE, PROFILE)
        and manifest['profile_member'] == {**manifest['profile_plan'], 'uri': _PROFILE_FILE}
        and read_artifact(_part(root, manifest['profile_member'])) == profile
        and manifest['count'] == resources['case']['call_count'] and manifest['width'] == profile['width']
        and manifest['recipe_id'] == resources['recipe_id']
        and manifest['numeric'] == archive['numeric'] == local(root, 'numeric/artifact.json', numeric['schema'])
        and manifest['sha256_provider'] == archive['publication']['provider']
        and manifest['blas_member'] == archive['blas_member']
        and manifest['headers'] == [local(root, name, 'scar.cpp-header') for name in _MEMBERS]
        and all((root/name).read_bytes() == (anchor/name).read_bytes() for name in _MEMBERS[:2])
        and (root/ordered._ORDERED_HEADER).read_bytes() == ordered._render(resources, manifest['blas_member']),
        'fixed-count provider differs from its exact plan, ABI or template')
    crypto = ordered._bundled_crypto(root, manifest['sha256_provider'], publication)
    ordered._check_stream(root, resources, manifest['stream'], detached, source_csr=True,
        publication=publication, crypto=crypto)
    _require(read_artifact(_part(root, manifest['stream']['protocol'])) == protocol,
        'fixed-count provider wire or output contract differs')
    return manifest


def check_profile_solver(package):
    root = Path(package).resolve(strict=True)
    manifest = read_artifact(artifact_reference(root/'artifact.json', SCHEMA))
    profile = verify_profile_plan(manifest['profile_plan']['uri'])
    _require(artifact_reference(manifest['profile_plan']['uri'], PROFILE) == manifest['profile_plan'],
        'the native provider was built with a different exact plan')
    return _check(root, profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.profile is None and args.out is None:
        result = check_profile_solver(args.check)
    elif args.check is None and args.profile and args.out:
        result = export_profile_solver(args.profile, args.out)
    else:
        parser.error('use --check PACKAGE or --profile PLAN --out PACKAGE')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
