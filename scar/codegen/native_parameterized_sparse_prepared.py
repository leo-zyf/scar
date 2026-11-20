"""Deliver an admitted count-only sparse entry with an embedded CSR numerical body."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
from shutil import copytree
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_parameterized_sparse_entry as entry
from scar.codegen import native_parameterized_sparse_solver as solver
from scar.codegen import scalar_recursive_entry as host
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_publication_delivery import FUSED, _render
from scar.ir.execution_recipe import read_artifact
from scar.optimization.scalar_score_ir import _require


SCHEMA = 'scar.native-parameterized-sparse-prepared-executable'
_PROGRAM = 'scar_sparse_prepared'
_SOLVER = 'solver/scar_ordered_sparse'
_FRAGMENT = Path(__file__).with_name('native_sparse_prepared_input.cpp.in')


def _body_include(payload):
    return ''.join('"' + ''.join(f'\\{byte:03o}' for byte in payload[start:start + 32]) + '"\n'
                   for start in range(0, len(payload), 32))


def _render_source(launch):
    fields = ('selected', 'commands', 'environment', 'cwd', 'cpu_ids', 'owner_cpu', 'pipe_capacity_bytes')
    direct = _render({name: launch[name] for name in fields}).decode('utf-8')
    replacements = (
        ('void run() {', '#include "prepared_input.inc"\n\nvoid run(std::uint32_t count) {'),
        ('STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO', 'STDOUT_FILENO, STDERR_FILENO'),
        (f'    const auto digest_program = root / "{_SOLVER}";',
         f'    const auto digest_program = root / "{_SOLVER}";\n'
         '    if (std::filesystem::canonical(digest_program) != digest_program) {\n'
         '        throw std::runtime_error("expected the exact package-relative sparse solver");\n'
         '    }'),
        ('    set_affinity(cpus);\n    if constexpr (split) {',
         '    set_affinity(std::span(&owner, 1));\n'
         '    auto input = stage_prepared(count);\n'
         '    set_affinity(cpus);\n    if constexpr (split) {'),
        ('digest.launch(digest_program, nullptr, STDIN_FILENO, answer[1].get());',
         'digest.launch(digest_program, nullptr, input.get(), answer[1].get());\n'
         '        input.close_now();'),
        ('int main(int argc, char**)', 'int main(int argc, char** argv)'),
        ('argc != 1', 'argc != 2'),
        ('"selected publication accepts canonical CSR on stdin and no arguments"',
         '"expected exactly one runtime iteration count"'),
        ('        run();', '        run(scar_runtime_count(argv[1]));'),
    )
    for old, new in replacements:
        direct = entry._replace(direct, old, new)
    return '#include "scar_sparse_runtime_count.hpp"\n#include <linux/memfd.h>\n' + direct


def _materials(plan_path):
    from scar.optimization import native_parameterized_sparse_prepared as planner

    plan_ref = host._ref(plan_path, planner.SCHEMA)
    plan = planner.check(plan_ref['uri'])
    payload = planner.payload(plan)
    # Planner replay already rechecks the entire ordinary runtime entry and its providers.
    archive = read_artifact(plan['entry'])
    _require(archive['source'] == plan['source'] and archive['protocol'] == plan['protocol'],
             'prepared source, solver protocol or entry lineage differ')
    snapshot = plan['snapshot']
    proof = archive['csr_boundary_validation']
    _require(snapshot['origin'] == proof['fixed_csr'] and snapshot['offset'] == proof['metadata_bytes'] == entry._CSR.size
             and snapshot['size_bytes'] == proof['payload_bytes'] == len(payload)
             and snapshot['sha256'] == proof['preserved_payload_sha256'] == sha256(payload).hexdigest(),
             'prepared numerical body differs from the qualified CSR boundary')
    stages = plan['graph']['stages']
    _require(stages['embedded_payload']['kind'] == 'EMBEDDED_READ_ONLY_EXECUTABLE_BYTES'
             and stages['seal_solver_input']['kind'] == 'ANONYMOUS_SEALED_MEMFD'
             and stages['seal_solver_input']['logical_bytes'] == len(payload) + entry._CSR.size,
             'preparation plan does not describe the generated sealed input')
    runtime = read_artifact(archive['parameterized_plan'])
    fixed_ref = runtime['artifacts']['fixed_package']
    fixed_root = Path(fixed_ref['uri']).parent
    launch = read_artifact(entry._local_ref(fixed_root, read_artifact(fixed_ref)['plan']))
    place = plan['protocol']['placement']
    _require(launch['selected'] == FUSED and launch['commands'] == [[_SOLVER]]
             and launch['cpu_ids'] == place['launch_cpu_ids'] and launch['owner_cpu'] == place['owner_cpu_id']
             and plan['protocol']['width'] == plan['protocol']['window'] == len(place['worker_cpu_ids']),
             'prepared entry must retain the exact ordered native cohort and direct solver mode')
    interval = plan['protocol']['header']['count']
    minimum, maximum = interval['minimum'], interval['maximum']
    _require(interval == {name: runtime['scope'][name + '_count'] for name in ('minimum', 'maximum')},
             'prepared count bounds differ from the original source argument scope')
    header = solver.pack_header(plan['protocol'], minimum)
    fragment = _FRAGMENT.read_text().replace('@SCAR_HEADER@', ', '.join(map(str, header)))
    fragment = fragment.replace('@SCAR_BODY_SIZE@', str(len(payload)))
    fragment = fragment.replace('@SCAR_COUNT_OFFSET@', str(plan['protocol']['header']['count_offset']))
    bound = entry._HEADER.read_text().replace('@SCAR_MIN@', str(minimum)).replace('@SCAR_MAX@', str(maximum))
    members = {'src/main.cpp': _render_source(launch), 'src/prepared_input.inc': fragment,
               'src/prepared_csr.inc': _body_include(payload), 'include/scar_sparse_runtime_count.hpp': bound}
    _require(all('@SCAR_' not in value for value in members.values()), 'incomplete prepared native source input')
    bare = Path(plan['entry']['uri']).parent / 'solver'
    archived_manifest = read_artifact(archive['solver'])
    solver_ref = dict(archive['solver'], uri='solver/artifact.json')
    executable_ref = dict(archived_manifest['stream']['executable'], uri=_SOLVER)
    return plan_ref, plan, members, payload, bare, solver_ref, executable_ref


def _project(root):
    write_native_project(root, _PROGRAM, include_dirs=('include',), compile_options=entry._FLAGS,
                         install_dirs=(('solver', 'bin'),), preserve_install_permissions=True)


def _describe(root, materials):
    plan_ref, plan, sources, payload, _, solver_ref, executable_ref = materials
    local = solver.check(root / 'solver')
    _require(host._local(root, solver_ref['uri'], solver.SCHEMA) == solver_ref
             and host._local(root, _SOLVER, 'scar.native-executable-file') == executable_ref
             and dict(local['stream']['executable'], uri=_SOLVER) == executable_ref,
             'package-local solver differs from the admitted runtime-count entry')
    binary = root / _PROGRAM
    _require(os.access(binary, os.X_OK) and binary.read_bytes().count(payload) == 1,
             'prepared native executable lacks the exact embedded numerical body')
    return dict(schema=SCHEMA, schema_version=1, plan=plan_ref, entry=plan['entry'], source=plan['source'],
                scope=plan['scope'], protocol=plan['protocol'], snapshot=plan['snapshot'],
                solver=solver_ref, solver_executable=executable_ref, binary=host._local(root, _PROGRAM, 'scar.elf'),
                host_linkage=host._elf(binary), members=entry._members(root, [*sources, 'CMakeLists.txt']),
                source_count_read_at_runtime=True, cxx_standard=20, python_runtime_required=False,
                deployment_authorized=False)


@replay_transaction()
def export(plan_path, out):
    materials = _materials(plan_path)
    _, _, sources, _, bare, _, _ = materials
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    with TemporaryDirectory(prefix='.scar-sparse-prepared-', dir=out.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        for name, value in sources.items():
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(value, encoding='utf-8')
        copytree(bare, root / 'solver')
        _project(root)
        binary, _ = build_native_project(root, _PROGRAM, compiler)
        binary = binary.rename(root / _PROGRAM)
        refused = subprocess.run([str(binary), '0'], cwd='/', env={}, stdin=subprocess.DEVNULL,
                                 capture_output=True, timeout=10)
        _require(refused.returncode == 1 and not refused.stdout and refused.stderr,
                 'prepared native argument rejection published a result')
        manifest = _describe(root, materials)
        (root / 'artifact.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return manifest


@replay_transaction()
def check(path):
    reference = host._ref(path, SCHEMA)
    manifest = read_artifact(reference)
    root = Path(reference['uri']).parent
    materials = _materials(manifest['plan']['uri'])
    _require(all((root / name).read_bytes() == source.encode('utf-8') for name, source in materials[2].items()),
             'prepared C++ source differs from the admitted body and supervisor')
    with TemporaryDirectory(prefix='scar-sparse-prepared-replay-') as temporary:
        replay = Path(temporary)
        _project(replay)
        _require((root / 'CMakeLists.txt').read_bytes() == (replay / 'CMakeLists.txt').read_bytes(),
                 'prepared C++20 project differs from the checked delivery')
    _require(manifest == _describe(root, materials), 'prepared package differs from cold replay')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not (args.plan or args.out):
        result = check(args.check)
    elif args.plan and args.out and not args.check:
        result = export(args.plan, args.out)
    else:
        parser.error('provide --plan and --out or --check ARTIFACT')
    print(json.dumps({name: result[name] for name in ('schema', 'binary', 'solver_executable')}, sort_keys=True))


if __name__ == '__main__':
    main()
