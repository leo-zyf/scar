"""Lower the ordered sparse recipe to a standalone C++20 dispatch package."""

import argparse
import json
import os
from pathlib import Path
import shlex
from shutil import copytree
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import (
    SCHEMA as PROVIDER_BUNDLE, build_native_provider_bundle, validate_native_provider_bundle,
)
from scar.codegen.native_sparse_source import SCHEMA as PUBLICATION, STREAM as DIGEST_STREAM, SYSTEM_LIBRARIES, project as project_publication, project_stream as project_digest_stream
from scar.codegen.native_sparse_source import SOURCE_STREAM, project_source_csr_stream, render_source_scale
from scar.codegen.sparse_native_export import check_sparse_export, export_sparse_native
from scar.ir.execution_recipe import SCHEMA as RECIPE, artifact_reference, read_artifact
from scar.optimization.native_ordered_resources import SCHEMA as PLAN, project_native_ordered_resources
from scar.optimization.native_ordered_stream import (
    DETACHED_BLAS_SYMBOL, DETACHED_STREAM, SCHEMA as STREAM, project_native_detached_stream,
    project_native_ordered_stream, validate_native_ordered_stream,
)
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.sparse_parallel_recipe import DETACHED as DETACHED_RECIPE, verify_sparse_parallel_recipe


SCHEMA = 'scar.native-ordered-sparse-export'
_ABI_HEADER = 'scar_sparse_provider_v1.hpp'
_ORDERED_HEADER = 'scar_ordered_sparse_v1.hpp'
_DISPATCH_HEADER = 'native_ordered_dispatch_v1.h'
_DISPATCH_SOURCE = Path(__file__).with_name(_DISPATCH_HEADER + '.in')
_ORDERED_SOURCE = Path(__file__).with_name('native_ordered_sparse_v1.h.in')
_WIRE_SOURCE = Path(__file__).with_name('native_ordered_sparse_wire_v1.cpp.in')
_MAIN_SOURCE = Path(__file__).with_name('native_ordered_sparse_main_v1.cpp.in')
_DETACHED_MAIN = Path(__file__).with_name('native_ordered_sparse_detached_v1.cpp.in')
_SOURCE_INGRESS = Path(__file__).with_name('native_ordered_sparse_source_v1.cpp.in')
_PUBLICATION_SOURCE = Path(__file__).with_name('native_ordered_sparse_publication_v1.cpp.in')
_PUBLICATION_FILE = 'publication.json'
_CRYPTO_BUNDLE = 'sha256/bundle.json'
_DETACHED_SOURCE = 'detached-source.json'
_STREAM_PLAN = 'stream.json'
_STREAM_MAIN = 'scar_ordered_sparse.cpp'
_STREAM_EXECUTABLE = 'scar_ordered_sparse'
_RELEASE = 'scar.sparse-parallel-entry-selection'
_RELEASE_FILE = 'release.json'
_CXXFLAGS = ('-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-Wl,-z,defs')


def _render(plan, blas):
    case, placement = plan['case'], plan['placement']
    lanes = [placement['owner'], *placement['workers']]
    identities = [(row['cpu_id'], row['package_id'], -1 if row['die_id'] is None else row['die_id'], row['core_id'])
                  for row in lanes]
    memory = plan['memory']
    bindings = dict(MATRIX_SIZE=case['size'], MATRIX_NNZ=case['nnz'], REQUEST_COUNT=case['call_count'],
        NUMA_NODE=placement['cpu_numa_node'], OWNER_CPU=placement['owner_cpu_id'],
        WORKER_CPUS=', '.join(map(str, placement['worker_cpu_ids'])),
        CPU_PLACEMENT=', '.join('{' + ', '.join(map(str, identity)) + '}' for identity in identities),
        WINDOW=plan['dispatch']['window'], WORKER_LIMIT=memory['worker_address_space_limit_bytes'],
        SYMBOLIC_BUDGET=memory['per_worker_symbolic_budget_bytes'],
        MEMORY_HEADROOM=memory['minimum_launch_headroom_bytes'],
        PAYLOAD_BUDGET=memory['owner_pending_input_bytes'] + memory['owner_pending_result_bound_bytes'],
        BLAS_MEMBER=json.dumps(blas))
    source = _ORDERED_SOURCE.read_text()
    for field, value in bindings.items():
        source = source.replace('@SCAR_' + field + '@', str(value))
    return source.encode()


def _local(root, name, schema):
    path = root / name
    if path.resolve(strict=True) != path:
        raise ValueError('expected a regular package member: ' + name)
    return {**artifact_reference(path, schema), 'uri': name}


def _render_publication(tail, digest):
    before, markers = tail.split('@SCAR_RESULT_SUCCESS_BEGIN@', 1)
    legacy, after = markers.split('@SCAR_RESULT_SUCCESS_END@', 1)
    tail = before + ('            publication.append(row.result);' if digest else legacy) + after
    values = dict(RESULT_SETUP='        DigestPublication publication;\n' if digest else '',
        RESULT_COMPLETE='publication.complete()' if digest else 'publish_record(published, Record::end, 0)',
        RESULT_FAULT='publish_digest_failure' if digest else 'publish_record',
        MAIN_ARGV=' argv' if digest else '',
        MAIN_PARAMETERS=' && (argc != 2 || std::string_view(argv[1]) != "--consume-records")' if digest else '',
        MAIN_USAGE='ordered sparse digest accepts binary stdin and optional --consume-records' if digest
            else 'detached sparse executable accepts binary stdin and no arguments',
        ALTERNATE_MAIN=('        if (argc == 2) {\n'
                        '            return consume_binary_results();\n'
                        '        }\n') if digest else '',
        RESULT_HEADER='' if digest else '        write_exact(output_header);\n')
    for name, value in values.items():
        tail = tail.replace('@SCAR_' + name + '@', value)
    return tail


def _render_main(stream, plan, crypto=None):
    digest = stream['recipe_id'].removeprefix('recipe-v1:')
    wire = _WIRE_SOURCE.read_text().replace('@SCAR_RECIPE_DIGEST@', digest).replace(
        '@SCAR_INPUT_MAGIC@', stream['input']['magic']) + '\n'
    is_digest = stream['schema'] == DIGEST_STREAM
    if is_digest != (crypto is not None):
        raise ValueError('native digest stream and package provider must be selected together')
    if stream['schema'] in (DETACHED_STREAM, SOURCE_STREAM, DIGEST_STREAM):
        includes = ''.join(f'#include <{name}>\n' for name in (
            'algorithm', 'cmath', 'cstring', 'iterator', 'memory', 'utility', 'dlfcn.h'))
        tail = _DETACHED_MAIN.read_text().replace('@SCAR_OWNER_CPU@', str(plan['placement']['owner_cpu_id']))
        if plan['dispatch'].get('credit_policy') == 'INDEPENDENT_ORDERED_WINDOW_V1':
            window = plan['dispatch']['window']
            former = 'constexpr auto window = std::size(sparse::worker_cpus);'
            if (type(window) is not int or not 1 < window < len(plan['placement']['worker_cpu_ids'])
                    or tail.count(former) != 1):
                raise ValueError('independent credits must bind the one detached dispatch window')
            tail = tail.replace(former, f'constexpr std::size_t window = {window};')
        before, ticket = tail.split('@SCAR_TICKET_BEGIN@\n', 1)
        legacy_ticket, after = ticket.split('@SCAR_TICKET_END@', 1)
        source = stream['schema'] in (SOURCE_STREAM, DIGEST_STREAM)
        ingress = ''
        if source:
            expression = render_source_scale(stream['source']['expression'], stream['header']['count'])
            ingress = _SOURCE_INGRESS.read_text().replace('@SCAR_SCALE_EXPRESSION@', expression) + '\n'
        if is_digest:
            includes += '#include <openssl/evp.h>\n'
            wire = wire.replace('\nvoid publish_record(', '\n[[maybe_unused]] void publish_record(')
            ingress += _PUBLICATION_SOURCE.read_text().replace('@SCAR_CRYPTO_PATH@', json.dumps(crypto)) + '\n\n'
        tail = before + ('            fill_source_scale(sequence, request);' if source else legacy_ticket) + after
        tail = (tail.replace('@SCAR_SOURCE_INGRESS@', ingress)
                .replace('@SCAR_SNAPSHOT_READER@', 'read_source_snapshot' if source else 'read_snapshot')
                .replace('@SCAR_LEGACY_UNUSED@', '[[maybe_unused]] ' if source else ''))
        tail = _render_publication(tail, is_digest)
        return (includes + wire + tail.replace('@SCAR_BLAS_SCALE_SYMBOL@', DETACHED_BLAS_SYMBOL)).encode()
    return (wire + _MAIN_SOURCE.read_text()).encode()


def _compile_flags(schema):
    if schema in (SOURCE_STREAM, DIGEST_STREAM):
        return (*_CXXFLAGS[:-1], '-fno-fast-math', '-ffp-contract=off', _CXXFLAGS[-1])
    return _CXXFLAGS


def _write_stream(directory, plan, stream, crypto=None):
    (directory / _STREAM_PLAN).write_text(json.dumps(stream, sort_keys=True, indent=2, allow_nan=False) + '\n')
    (directory / _STREAM_MAIN).write_bytes(_render_main(stream, plan, crypto))
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise ValueError('CXX must select an available C++20 compiler')
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    flags = _compile_flags(stream['schema'])
    install = [('numeric', 'bin')]
    if crypto is not None:
        install.append(('sha256', 'bin'))
    write_native_project(directory, _STREAM_EXECUTABLE, sources=(_STREAM_MAIN,), include_dirs=('.',),
        compile_options=flags[1:-1], link_options=(flags[-1],), libraries=('dl',), install_dirs=install)
    executable, _ = build_native_project(directory, _STREAM_EXECUTABLE, compiler)
    # Preserve the existing colocated stream executable/solver ABI.
    executable.rename(directory / _STREAM_EXECUTABLE)
    return dict(protocol=_local(directory, _STREAM_PLAN, stream['schema']),
        source=_local(directory, _STREAM_MAIN, 'scar.cpp-source'),
        executable=_local(directory, _STREAM_EXECUTABLE, 'scar.native-executable-file'),
        cmake=_local(directory, 'CMakeLists.txt', 'scar.cmake-project'),
        compiler=dict(command=compiler, version=version, options=list(flags)))


def _check_stream(directory, plan, transport, detached=None, *, source_csr=False, publication=None, crypto=None):
    schema = DIGEST_STREAM if publication is not None else SOURCE_STREAM if source_csr else (
        STREAM if detached is None else DETACHED_STREAM)
    if (transport['protocol'] != _local(directory, _STREAM_PLAN, schema)
            or transport['source'] != _local(directory, _STREAM_MAIN, 'scar.cpp-source')
            or transport['executable'] != _local(directory, _STREAM_EXECUTABLE, 'scar.native-executable-file')
            or transport['compiler']['options'] != list(_compile_flags(schema))
            or not os.access(directory / _STREAM_EXECUTABLE, os.X_OK)):
        raise ValueError('native stream executable or protocol differs from its receipt')
    if 'cmake' in transport and transport['cmake'] != _local(directory, 'CMakeLists.txt', 'scar.cmake-project'):
        raise ValueError('native stream project definition differs from its receipt')
    protocol = read_artifact({**transport['protocol'], 'uri': str(directory / _STREAM_PLAN)})
    if source_csr:
        expected = project_source_csr_stream(plan, detached)
        if publication is not None:
            expected = project_digest_stream(expected, publication)
        if semantic_sha256(protocol) != semantic_sha256(expected):
            raise ValueError('native source CSR stream differs from its released integer/float and ingress contracts')
    else:
        validate_native_ordered_stream(plan, protocol, detached=detached)
    if (directory / _STREAM_MAIN).read_bytes() != _render_main(protocol, plan, crypto):
        raise ValueError('native stream entry differs from its recipe and protocol')


def _bundled_crypto(directory, reference, publication):
    if reference != _local(directory, _CRYPTO_BUNDLE, PROVIDER_BUNDLE):
        raise ValueError('ordered digest bundle differs from its local receipt')
    bundle = validate_native_provider_bundle({**reference, 'uri': str(directory / _CRYPTO_BUNDLE)})
    provider, = bundle['members']
    original = publication['provider']['libcrypto']
    if (provider['path'] != bundle['root'] or provider['origin'] != original['uri']
            or any(provider[key] != original[key] for key in ('sha256', 'size_bytes'))
            or publication['provider']['abi'] != 'OPENSSL_EVP_SHA256'
            or publication['provider']['external_config_loaded'] is not False):
        raise ValueError('ordered digest does not ship the admitted OpenSSL implementation')
    return 'sha256/' + bundle['root']


def _check_publication(directory, manifest, selected):
    publication = manifest['publication']
    ref = publication['contract']
    if ref != _local(directory, _PUBLICATION_FILE, PUBLICATION):
        raise ValueError('ordered digest contract differs from its local receipt')
    contract = read_artifact({**ref, 'uri': str(directory / _PUBLICATION_FILE)})
    if ({**contract['release'], 'uri': _RELEASE_FILE} != manifest['qualified_release']
            or contract['execution_recipe'] != selected['execution_recipe']
            or contract['parallel_scope'] != selected['source'] or contract['entry_source'] != selected['entry']):
        raise ValueError('ordered digest publication is not attached to the qualified source release')
    return contract, _bundled_crypto(directory, publication['provider'], contract)


@replay_transaction()
def export_native_ordered(recipe, provider, out, *, detached=False, release=None, source_csr=False, output_contract=None):
    if source_csr and (detached or release is None):
        raise ValueError('native source CSR requires the published release and its own ingress protocol')
    if output_contract is not None and not source_csr:
        raise ValueError('a native output contract requires the canonical source CSR ingress')
    release_ref = None
    if release is not None:
        from scar.compiler.plan_sparse_parallel import qualify

        release_ref = artifact_reference(release, _RELEASE)
        selected = read_artifact(release_ref)
        if recipe is not None or selected.get('decision') != 'PARALLEL' or selected.get('study_parallel') is not False:
            raise ValueError('native delivery requires a published parallel selection')
        with TemporaryDirectory(prefix='scar-ordered-qualification-') as check:
            replayed = qualify(selected['study']['uri'], selected['validation']['uri'], check)
            if read_artifact(replayed) != selected:
                raise ValueError('sparse release differs from its replayed optimization decision')
        recipe = selected['execution_recipe']['uri']
    recipe_ref = artifact_reference(recipe, RECIPE)
    if release_ref is not None and recipe_ref != selected['execution_recipe']:
        raise ValueError('sparse release lost its selected execution recipe')
    source = verify_sparse_parallel_recipe(read_artifact(recipe_ref))
    plan = project_native_ordered_resources(source)
    source_ref = source['artifacts']['detached']
    if source_csr:
        protocol = project_source_csr_stream(plan, read_artifact(source_ref))
    elif detached:
        protocol = project_native_detached_stream(plan, read_artifact(source_ref))
    else:
        protocol = project_native_ordered_stream(plan)
    publication_ref = publication = None
    if output_contract is not None:
        publication_ref = artifact_reference(output_contract, PUBLICATION)
        publication = read_artifact(publication_ref)
        if project_publication(release_ref, publication['entry_dependence'], replay_entry=True) != publication:
            raise ValueError('ordered output contract differs from the original publication routes and provider')
        protocol = project_digest_stream(protocol, publication)
    capability = source['implementation']['backend_capability_record']
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix='.scar-native-ordered-', dir=out.parent) as temporary:
        directory = Path(temporary)
        numeric = directory / 'numeric'
        if provider is None:
            export_sparse_native(capability['serial_recipe']['uri'], capability['serial_lowering']['uri'], numeric)
            provider = numeric
        else:
            provider = Path(provider).resolve(strict=True)
        native = check_sparse_export(provider)
        provenance = native['build_provenance']
        if (provenance['recipe'] != capability['serial_recipe']
                or provenance['lowering'] != capability['serial_lowering']
                or provenance['native'] != capability['native'] or native['abi']['version'] != capability['abi_version']):
            raise ValueError('ordered recipe and standalone numeric provider do not share the same lowered ABI')
        closure_path = provider / native['provider']['uri']
        closure = read_artifact({**native['provider'], 'uri': str(closure_path)})
        matches = [row for row in closure['members'] if row['origin'] == capability['blas']['uri']
                   and row['sha256'] == capability['blas']['sha256']]
        if len(matches) != 1:
            raise ValueError('ordered recipe requires its exact single-thread BLAS member')
        blas = 'provider/' + matches[0]['path']
        if provider != numeric:
            copytree(provider, numeric)
        (directory / _ABI_HEADER).write_bytes((numeric / native['header']['uri']).read_bytes())
        (directory / _DISPATCH_HEADER).write_bytes(_DISPATCH_SOURCE.read_bytes())
        (directory / _ORDERED_HEADER).write_bytes(_render(plan, blas))
        (directory / 'plan.json').write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + '\n')
        publication_members, crypto = None, None
        if publication is not None:
            build_native_provider_bundle(publication['provider']['libcrypto'], directory / 'sha256',
                provider_roots=(), system_sonames=SYSTEM_LIBRARIES)
            (directory / _PUBLICATION_FILE).write_bytes(read_artifact(publication_ref, json_document=False))
            publication_members = dict(contract=_local(directory, _PUBLICATION_FILE, PUBLICATION),
                provider=_local(directory, _CRYPTO_BUNDLE, PROVIDER_BUNDLE))
            crypto = _bundled_crypto(directory, publication_members['provider'], publication)
        stream = _write_stream(directory, plan, protocol, crypto)
        version = 5 if publication is not None else 4 if source_csr else 3 if detached else 2
        boundaries = {2: 'native CSR structure and one binary64 value/RHS frame per ordered request',
                      3: 'prepared CSR template, RHS base and one binary64 scale per ordered request',
                      4: 'canonical source CSR, original RHS and selected row indices',
                      5: 'canonical source CSR, original RHS and selected row indices to explicit native stdout'}
        manifest = dict(schema=SCHEMA, schema_version=version, recipe=recipe_ref, recipe_id=source['recipe_id'],
            numeric=_local(directory, 'numeric/artifact.json', native['schema']), blas_member=blas,
            plan=_local(directory, 'plan.json', PLAN), headers=[_local(directory, name, 'scar.cpp-header')
                for name in (_ABI_HEADER, _DISPATCH_HEADER, _ORDERED_HEADER)],
            caller_boundary=boundaries[version],
            failure='first failed request stops publication; original Python entry is separate',
            stream=stream, cost='unmeasured native stream boundary', deployment_authorized=False)
        if detached or source_csr:
            (directory / _DETACHED_SOURCE).write_bytes(read_artifact(source_ref, json_document=False))
            manifest['detached_source'] = _local(directory, _DETACHED_SOURCE, DETACHED_RECIPE)
        if release_ref is not None:
            (directory / _RELEASE_FILE).write_bytes(read_artifact(release_ref, json_document=False))
            manifest['qualified_release'] = _local(directory, _RELEASE_FILE, _RELEASE)
        if publication_members is not None:
            manifest['publication'] = publication_members
        (directory / 'artifact.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        check_native_ordered_export(directory)
        if out.exists():
            raise FileExistsError(out)
        directory.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check_native_ordered_export(path):
    root = Path(path).resolve(strict=True)
    manifest = read_artifact(artifact_reference(root / 'artifact.json', SCHEMA))
    verification = check_sparse_export(root / 'numeric')
    expected_headers = [_local(root, name, 'scar.cpp-header') for name in (_ABI_HEADER, _DISPATCH_HEADER, _ORDERED_HEADER)]
    version = manifest.get('schema_version')
    if (version not in (1, 2, 3, 4, 5) or ('stream' in manifest) != (version >= 2)
            or ('detached_source' in manifest) != (version >= 3)
            or ('publication' in manifest) != (version == 5)
            or version >= 4 and 'qualified_release' not in manifest or manifest['deployment_authorized'] is not False
            or manifest['headers'] != expected_headers
            or manifest['numeric'] != _local(root, 'numeric/artifact.json', verification['schema'])
            or manifest['plan'] != _local(root, 'plan.json', PLAN)):
        raise ValueError('an ordered package member differs from its receipt')
    plan = read_artifact({**manifest['plan'], 'uri': str(root / 'plan.json')})
    if 'qualified_release' in manifest:
        release = manifest['qualified_release']
        if release != _local(root, _RELEASE_FILE, _RELEASE):
            raise ValueError('ordered package release differs from its local receipt')
        selected = read_artifact({**release, 'uri': str(root / _RELEASE_FILE)})
        dispatch, placement, case, memory = (plan[name] for name in ('dispatch', 'placement', 'case', 'memory'))
        if (selected.get('decision') != 'PARALLEL' or selected.get('study_parallel') is not False
                or selected['execution_recipe'] != manifest['recipe'] or selected['owner_cpu'] != placement['owner_cpu_id']
                or selected['numa_node'] != placement['cpu_numa_node']
                or selected['launch_cpu_ids'] != placement['launch_cpu_ids']
                or selected['workers']['cpu_ids'] != placement['worker_cpu_ids']
                or selected['workers']['window'] != dispatch['window']
                or selected['workers']['address_space_limit_bytes'] != memory['worker_address_space_limit_bytes']
                or selected['symbolic_budget_bytes'] != memory['per_worker_symbolic_budget_bytes']
                or selected['memory_floor_bytes'] != memory['minimum_launch_headroom_bytes']
                or selected['count'] != case['call_count']
                or selected['matrix'] != {name: case[name] for name in ('size', 'nnz')}):
            raise ValueError('ordered C++ schedule differs from the qualified release')
        if version >= 3 and {**manifest['detached_source'], 'uri': selected['recipe']['uri']} != selected['recipe']:
            raise ValueError('ordered C++ source preparation differs from the qualified release')
    closure_path = root / 'numeric' / verification['provider']['uri']
    closure = read_artifact({**verification['provider'], 'uri': str(closure_path)})
    operator, = [row for row in closure['members'] if row['path'] == closure['root']]
    direct = set(operator['elf']['needed']) - set(closure['system_sonames'])
    blas = [row['path'] for row in closure['members'] if row['elf']['soname'] in direct]
    if (len(blas) != 1 or manifest['blas_member'] != 'provider/' + blas[0]
            or manifest['recipe_id'] != plan['recipe_id']
            or (root / _ABI_HEADER).read_bytes() != (root / 'numeric' / verification['header']['uri']).read_bytes()
            or (root / _DISPATCH_HEADER).read_bytes() != _DISPATCH_SOURCE.read_bytes()
            or (root / _ORDERED_HEADER).read_bytes() != _render(plan, manifest['blas_member'])):
        raise ValueError('standalone C++ scheduling does not match its native plan')
    detached = None
    if version >= 3:
        reference = manifest['detached_source']
        if reference != _local(root, _DETACHED_SOURCE, DETACHED_RECIPE):
            raise ValueError('compact export detached source differs from its receipt')
        detached = read_artifact({**reference, 'uri': str(root / _DETACHED_SOURCE)})
    if version >= 2:
        publication, crypto = _check_publication(root, manifest, selected) if version == 5 else (None, None)
        _check_stream(root, plan, manifest['stream'], detached, source_csr=version >= 4,
            publication=publication, crypto=crypto)
    return manifest


def rebuild_native_ordered_project(path, out):
    """Rebuild an independently replayable native export on its existing wire ABI."""
    source = Path(path).resolve(strict=True)
    manifest = check_native_ordered_export(source)
    if 'stream' not in manifest:
        raise ValueError('the native export needs an executable stream recipe')
    plan = read_artifact({**manifest['plan'], 'uri': str(source / 'plan.json')})
    protocol = read_artifact({**manifest['stream']['protocol'], 'uri': str(source / _STREAM_PLAN)})
    crypto = None
    if manifest['schema_version'] == 5:
        release = read_artifact({**manifest['qualified_release'], 'uri': str(source / _RELEASE_FILE)})
        _, crypto = _check_publication(source, manifest, release)
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(source):
        raise ValueError('a fresh output outside the original sparse package is required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-sparse-project-', dir=out.parent) as temporary:
        directory = Path(temporary)
        copytree(source, directory, dirs_exist_ok=True)
        manifest['stream'] = _write_stream(directory, plan, protocol, crypto)
        (directory / 'artifact.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        check_native_ordered_export(directory)
        directory.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--recipe')
    choice.add_argument('--release', help='replay the published parallel decision before exporting its recipe')
    parser.add_argument('--provider', help='reuse a checked numeric bundle instead of exporting the selected provider')
    parser.add_argument('--out')
    parser.add_argument('--check')
    parser.add_argument('--rebuild-project', help='build a checked standalone export using the shared CMake project')
    input_protocol = parser.add_mutually_exclusive_group()
    input_protocol.add_argument('--detached', action='store_true', help='export compact prepared inputs and scalar tickets')
    input_protocol.add_argument('--source-csr', action='store_true', help='export source CSR, RHS, and selected row inputs')
    parser.add_argument('--output-contract', help='lower a qualified independent ordered SHA-256 stdout contract')
    args = parser.parse_args(argv)
    export = (args.recipe or args.release, args.provider, args.out)
    custom_stream = args.detached or args.source_csr or args.output_contract
    if args.rebuild_project and args.out and not any((export[0], args.provider, args.check, custom_stream)):
        print(json.dumps(rebuild_native_ordered_project(args.rebuild_project, args.out)))
    elif args.check and not any(export) and not (custom_stream or args.rebuild_project):
        result = check_native_ordered_export(args.check)
        print(json.dumps(dict(verified=True, recipe_id=result['recipe_id'])))
    elif not args.check and export[0] and args.out and not args.rebuild_project:
        print(json.dumps(export_native_ordered(args.recipe, args.provider, args.out,
            detached=args.detached, release=args.release, source_csr=args.source_csr,
            output_contract=args.output_contract)))
    else:
        parser.error('choose --check PACKAGE, --rebuild-project PACKAGE --out OUT, or supply --release or --recipe with --out')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
