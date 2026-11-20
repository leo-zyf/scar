"""Deliver a runtime count through a checked native sparse application pipeline."""

import argparse
import errno
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import shlex
from shutil import copy2, copytree
from struct import Struct
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_source_triangle_profile as anchor
from scar.codegen import scalar_recursive_entry as host
from scar.codegen.native_project import write_native_project
from scar.codegen.native_publication_delivery import _render
from scar.ir.execution_recipe import read_artifact
from scar.optimization.scalar_score_ir import _require


SCHEMA = 'scar.native-parameterized-sparse-executable'
PLAN = 'scar.native-parameterized-sparse-plan'
_PROGRAM = 'scar_sparse_parameterized'
_INGRESS = 'scar_coo_ingress'
_INGRESS_DIR = 'upstream/coo/bin'
_COUNT = Struct('<I')
_CSR = Struct('<8s32s4I')
_COUNT_OFFSET = Struct('<8s32s2I').size
_HEADER = Path(__file__).with_name('native_sparse_runtime_count.hpp.in')
_FLAGS = ('-O2', '-Wall', '-Wextra', '-Werror', '-pedantic')
_FP_FLAGS = ('-fno-fast-math', '-ffp-contract=off')
_UNCHANGED = ('topology', 'mesh', 'basis')


def _local_ref(root, reference):
    path = (root / reference['uri']).resolve(strict=True)
    _require(path.is_relative_to(root) and str(path.relative_to(root)) == reference['uri'],
             'expected an exact package-relative native input')
    return dict(reference, uri=str(path))


def _member(root, reference):
    return read_artifact(_local_ref(root, reference), json_document=False)


def _replace(text, old, new):
    _require(text.count(old) == 1, 'generated stage does not expose a unique expected source boundary')
    return text.replace(old, new, 1)


def _supervisor(launch, original):
    expected = _render(launch)
    _require(expected == original, 'source supervisor differs from the archived checked stage')
    body = expected.decode('utf-8')
    ingress, = launch['ingress']['command']
    count, = launch['source_seed']['argv']
    original_guard = f'argc != 2 || std::string_view(argv[1]) != {json.dumps(count)}'
    body = _replace(body, 'void run() {', 'void run(const char* count_argument) {')
    argument = f'ingress.launch(root / {json.dumps(ingress)}, nullptr,'
    body = _replace(body, argument, argument.replace('nullptr', 'count_argument'))
    body = _replace(body, original_guard, 'argc != 2')
    original_usage = json.dumps(f'selected source publication requires the fixed argument {count}')
    body = _replace(body, original_usage, '"expected exactly one runtime iteration count"')
    body = _replace(body, '        run();', '        scar_runtime_count(argv[1]);\n        run(argv[1]);')
    return ('#include "scar_sparse_runtime_count.hpp"\n' + body).encode('utf-8')


def _coo(original, old_prefix, output_minimum):
    body = original.decode('utf-8')
    legacy = ', '.join(map(str, old_prefix))
    _require(body.count(legacy) == 1, 'qualified COO stage lacks the original exact wire prefix')
    body = _replace(body, 'void run() {', 'void run(std::uint32_t iterations) {')
    body = _replace(body, 'int main() {', 'int main(int argc, char** argv) {')
    body = _replace(body, '        run();',
                    '        if (argc != 2) throw std::runtime_error("COO requires a checked iteration count");\n'
                    '        run(scar_runtime_count(argv[1]));')
    canonical = ', '.join(map(str, output_minimum))
    before = '    write_exact(header.data() + coarse_header_bytes, header.size() - coarse_header_bytes);'
    publication = (f'    std::array<unsigned char, {len(output_minimum)}> output{{{canonical}}};\n'
                   f'    static_assert(coarse_header_bytes + output.size() == ingress.size());\n'
                   f'    for (std::size_t index = 0; index < {_COUNT.size}; ++index) {{\n'
                   f'        output[{_COUNT_OFFSET} + index] = static_cast<unsigned char>(iterations >> (index * 8));\n'
                   '    }\n'
                   '    write_exact(output.data(), output.size());')
    body = _replace(body, before, publication)
    return ('#include "scar_sparse_runtime_count.hpp"\n' + body).encode('utf-8')


def _bound_solver(solver_path, fixed, plan):
    from scar.codegen import native_parameterized_sparse_solver as solver

    root = Path(solver_path).resolve(strict=True)
    root = root if root.is_dir() else root.parent
    actual = solver.check(root)
    ref = host._ref(root / 'artifact.json', actual['schema'])
    minimum, maximum = (plan['scope'][name] for name in ('minimum_count','maximum_count'))
    legacy = Path(fixed['uri']).parent / 'solver'
    protocol = solver.protocol(legacy, maximum_count=maximum)
    blob = solver.pack_header(protocol, minimum)
    magic, digest, size, nnz, count, flags = _CSR.unpack(blob)
    _require(magic == protocol['input']['magic'].encode('ascii') and digest.hex() == protocol['header']['recipe_digest_hex']
             and size > 0 and nnz > 0 and count == minimum and flags == 0,
             'parameterized solver header differs from the typed sparse count port')
    return root, ref, actual, protocol, blob


def _check_solver_identity(root, actual, protocol, fixed):
    previous = host._ref(fixed / 'solver' / 'artifact.json', 'scar.native-ordered-count-provider')
    observed = read_artifact(_local_ref(root, actual['protocol']))
    _require(dict(actual['anchor'], uri=previous['uri']) == previous and observed['anchor'] == actual['anchor']
             and dict(observed, anchor=previous) == protocol
             and actual['stream']['executable']['uri'] == 'scar_ordered_sparse',
             'runtime-count solver differs from the planned fixed numerical and physical provider')


def _coo_projection(fixed, upstream, old_csr):
    from scar.codegen.native_parameterized_upstream import _metadata_only

    root = fixed / 'upstream' / 'coo'
    manifest = read_artifact(_local_ref(fixed / 'upstream', upstream['packages']['coo']))
    body = _member(root, manifest['members']['main.cpp'])
    profile = read_artifact(_local_ref(fixed / 'upstream', upstream['profile']))
    envelope = bytes(profile['target_header'])[-76:]
    executable = 'bin/' + _INGRESS
    _require(envelope.endswith(old_csr)
             and manifest['profile_binding'] == upstream['profile']
             and manifest['members'][executable]['uri'] == executable,
             'original COO stage does not identify the canonical numerical stream')
    _metadata_only(body, 'coo', envelope)
    retained = {name:_member(root, reference) for name,reference in manifest['members'].items()
                if name.startswith('vendor/')}
    return dict(source=body,prefix=envelope,headers=retained)

def _verify_file(path, reference):
    with path.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    _require(path.stat().st_size == reference['size_bytes'] and digest == reference['sha256'],
             'native runtime member differs from checked immutable bytes')


def _materials(plan_path, solver_path):
    from scar.optimization import native_parameterized_sparse_plan as planner

    plan_ref = host._ref(plan_path, PLAN)
    plan = planner.check(plan_ref['uri'])
    fixed_ref = plan['artifacts']['fixed_package']
    fixed_root = Path(fixed_ref['uri']).parent
    fixed = anchor.check(fixed_root)
    _require(read_artifact(fixed_ref) == fixed and plan['decision']['native_experiment_authorized'] is True
             and plan['decision']['deployment_authorized'] is False
             and plan['scope']['count_port']['encoding'] == 'CANONICAL_ASCII_DECIMAL',
             'expected an admitted canonical decimal count plan')
    solver_root, solver_ref, solver, protocol, output_minimum = _bound_solver(solver_path, fixed_ref, plan)
    # The solver's executable metadata and exact runtime protocol must belong to the same provider.
    _check_solver_identity(solver_root, solver, protocol, fixed_root)

    launch = read_artifact(_local_ref(fixed_root, fixed['plan']))
    upstream = read_artifact(_local_ref(fixed_root, fixed['upstream']))
    csr = _member(fixed_root / 'upstream', upstream['canonical_csr'])
    magic, digest, n, nnz, fixed_count, flags = _CSR.unpack_from(csr)
    source_dims = protocol['header']
    _require(magic == b'SCARPSR1' and digest.hex() == fixed['recipe_id'].removeprefix('recipe-v1:')
             and (n, nnz, flags) == (source_dims['size'],source_dims['nnz'],source_dims['flags'])
             and fixed_count == plan['scope']['fixed_profile_count'],
             'runtime count must preserve all numerical fields of the original upstream')
    legacy_ingress = _coo_projection(fixed_root, upstream, csr[:_CSR.size])
    count_header = _HEADER.read_text()
    count_header = count_header.replace('@SCAR_MIN@', str(plan['scope']['minimum_count']))
    count_header = count_header.replace('@SCAR_MAX@', str(plan['scope']['maximum_count']))
    members = dict(legacy_ingress['headers'])
    members.update({'include/scar_sparse_runtime_count.hpp': count_header.encode(),
                    'src/supervisor.cpp': _supervisor(launch, _member(fixed_root, fixed['source'])),
                    'src/coo.cpp': _coo(legacy_ingress['source'], legacy_ingress['prefix'], output_minimum)})
    reused = {}
    for role in _UNCHANGED:
        root = fixed_root / 'upstream' / role
        manifest = json.loads((root / 'artifact.json').read_bytes())
        refs = (manifest['executable'], *manifest['libraries'])
        for receipt in refs:
            path = _local_ref(root, receipt)
            _verify_file(Path(path['uri']), receipt)
            reused[f'upstream/{role}/{receipt["uri"]}'] = path
    return plan_ref, plan, solver_ref, members, reused, protocol, solver_root


def _cmake(root):
    write_native_project(root, _PROGRAM, sources=('src/supervisor.cpp',), include_dirs=('include',),
                         compile_options=_FLAGS)
    with (root / 'CMakeLists.txt').open('a', encoding='utf8') as stream:
        stream.write(f'''add_executable({_INGRESS} src/coo.cpp)
set_target_properties({_INGRESS} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories({_INGRESS} PRIVATE include vendor)
target_compile_options({_INGRESS} PRIVATE -O2 {' '.join(_FP_FLAGS)})
install(DIRECTORY upstream solver DESTINATION bin USE_SOURCE_PERMISSIONS PATTERN "{_INGRESS}" EXCLUDE)
install(TARGETS {_INGRESS} RUNTIME DESTINATION bin/{_INGRESS_DIR})
''')


def _build(root, compiler):
    env = {key:value for key,value in os.environ.items() if not key.startswith('LD_')}
    env['CXX'] = shlex.join(compiler)
    logs = []
    with TemporaryDirectory(prefix='.scar-count-cmake-', dir=root.parent) as temporary:
        for args in (['cmake','-S',str(root),'-B',temporary,'-DCMAKE_BUILD_TYPE=Release'],
                     ['cmake','--build',temporary,'--parallel','2']):
            child = subprocess.run(args, env=env, capture_output=True, text=True)
            logs.append(child.stdout + child.stderr)
            _require(child.returncode == 0, 'C++20 application build failed\n' + logs[-1])
        ingress = root / _INGRESS_DIR / _INGRESS
        ingress.parent.mkdir(parents=True)
        copy2(Path(temporary) / _PROGRAM, root / _PROGRAM)
        copy2(Path(temporary) / _INGRESS, ingress)
    validation = root / 'validation'
    validation.mkdir()
    (validation / 'build.log').write_text('\n'.join(logs))
    return dict(supervisor=host._elf(root / _PROGRAM), ingress=host._elf(ingress))


def _link(source, destination):
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        copy2(source, destination)
    return destination


def _members(root, names):
    return {name:host._local(root, name, 'application/octet-stream') for name in sorted(names)}


def _validate_csr(root, plan, protocol):
    from scar.codegen.native_parameterized_sparse_solver import pack_header

    fixed = plan['artifacts']['fixed_package']
    original = Path(fixed['uri']).parent
    upstream = original / 'upstream'
    manifest = read_artifact(_local_ref(original, read_artifact(fixed)['upstream']))
    coo = _local_ref(upstream, manifest['wires'][-2])
    csr = _local_ref(upstream, manifest['canonical_csr'])
    numeric = read_artifact(csr,json_document=False)[_CSR.size:]
    count = plan['scope']['maximum_count']
    child = subprocess.run([str(root / _INGRESS_DIR / _INGRESS), str(count)],
                           cwd='/', env={}, input=read_artifact(coo,json_document=False),
                           capture_output=True,timeout=10)
    _require(child.returncode == 0 and not child.stderr and child.stdout == pack_header(protocol,count) + numeric,
             'runtime count ingress changed the checked numerical CSR bytes')
    return dict(checked_count=count,coo=coo,fixed_csr=csr,metadata_bytes=_CSR.size,
                payload_bytes=len(numeric),preserved_payload_sha256=sha256(numeric).hexdigest(),
                emitted_wire_sha256=sha256(child.stdout).hexdigest())


@replay_transaction()
def export(plan_path, solver_path, out):
    plan_ref, plan, solver, members, reused, protocol, solver_root = _materials(plan_path, solver_path)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'a C++20 compiler is required')
    with TemporaryDirectory(prefix='.scar-count-package-', dir=out.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        for name, data in members.items():
            child = root / name
            child.parent.mkdir(parents=True, exist_ok=True)
            child.write_bytes(data)
        for name, reference in reused.items():
            child = root / name
            child.parent.mkdir(parents=True, exist_ok=True)
            _link(reference['uri'], child)
        copytree(solver_root, root / 'solver', copy_function=_link)
        _cmake(root)
        linkage = _build(root, compiler)
        declined = subprocess.run([str(root / _PROGRAM), '0'], cwd='/', env={}, stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=10)
        _require(declined.returncode == 1 and not declined.stdout and declined.stderr,
                 'rejected runtime count created an observable result')
        descriptor = dict(schema=SCHEMA, schema_version=1, parameterized_plan=plan_ref, solver=solver,
                          source=plan['scope']['source'], binary=host._local(root, _PROGRAM, 'scar.elf'),
                          ingress=host._local(root, f'{_INGRESS_DIR}/{_INGRESS}', 'scar.elf'),
                          cxx_standard=20, python_runtime_required=False, deployment_authorized=False,
                          source_count_read_at_runtime=True, protocol=protocol, host_linkage=linkage,
                          csr_boundary_validation=_validate_csr(root,plan,protocol),
                          compiler=dict(command=compiler, version=host._version(compiler)),
                          reused_upstream={name:dict(reference,uri=name) for name, reference in reused.items()},
                          members=_members(root,[*members,'CMakeLists.txt']))
        (root / 'artifact.json').write_text(json.dumps(descriptor,indent=2,sort_keys=True) + '\n')
        root.rename(out)
    return descriptor


@replay_transaction()
def check(path):
    reference = host._ref(path, SCHEMA)
    doc = read_artifact(reference)
    root = Path(reference['uri']).parent
    plan_ref, plan, solver, members, reused, protocol, _ = _materials(doc['parameterized_plan']['uri'],
                                                                    doc['solver']['uri'])
    from scar.codegen import native_parameterized_sparse_solver as provider

    relocated_solver = provider.check(root / 'solver')
    _check_solver_identity(root / 'solver', relocated_solver, protocol, Path(plan['artifacts']['fixed_package']['uri']).parent)
    _verify_file(root/'solver'/'artifact.json', solver)
    _require(doc['schema_version'] == 1 and doc['source'] == plan['scope']['source']
             and doc['cxx_standard'] == 20 and doc['python_runtime_required'] is doc['deployment_authorized'] is False
             and doc['source_count_read_at_runtime'] is True and doc['parameterized_plan'] == plan_ref
             and doc['solver'] == solver and doc['protocol'] == protocol
             and doc['csr_boundary_validation'] == _validate_csr(root,plan,protocol)
             and doc['binary'] == host._local(root,_PROGRAM,'scar.elf')
             and doc['ingress'] == host._local(root,f'{_INGRESS_DIR}/{_INGRESS}','scar.elf')
             and doc['host_linkage'] == dict(supervisor=host._elf(root/_PROGRAM),ingress=host._elf(root/doc['ingress']['uri']))
             and doc['members'] == _members(root,[*members,'CMakeLists.txt'])
             and doc['reused_upstream'] == {name:dict(receipt,uri=name) for name,receipt in reused.items()}
             and all((root / name).read_bytes() == raw for name,raw in members.items()),
             'published runtime-count application differs from the original preparation, protocol or C++20 plan')
    for name, receipt in reused.items():
        _verify_file(root / name, receipt)
    with TemporaryDirectory(prefix='scar-count-cmake-replay-') as temporary:
        _cmake(Path(temporary))
        _require((Path(temporary) / 'CMakeLists.txt').read_bytes() == (root / 'CMakeLists.txt').read_bytes(),
                 'runtime count CMake installation instructions changed')
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('plan','solver','out','check'):
        parser.add_argument('--' + arg,type=Path)
    args = parser.parse_args(argv)
    inputs = (args.plan,args.solver,args.out)
    if args.check and not any(inputs):
        result = check(args.check)
    elif not args.check and all(inputs):
        result = export(*inputs)
    else:
        parser.error('provide --plan, --solver and --out, or --check ARTIFACT')
    print(json.dumps({key:result[key] for key in ('schema','binary','python_runtime_required','host_linkage')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
