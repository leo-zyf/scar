"""Verify a native project and compare its outputs with explicit reference files."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_provider_bundle import _content
from scar.ir.implementation_plan import problem_from_dict
from scar.ir.program import codec


def verify_native_project(directory, manifest_sha256):
    root = Path(directory).resolve(strict=True)
    path = root / 'artifact.json'
    if _content(path)['sha256'] != manifest_sha256:
        raise ValueError('native project manifest differs from its declared digest')
    manifest = json.loads(path.read_bytes())
    files = manifest['files']

    def member(relative):
        path = root / relative
        if relative not in files or path.resolve(strict=True) != path or not path.is_relative_to(root):
            raise ValueError(f'not a local native project member: {relative}')
        return path

    for name, expected in files.items():
        if _content(member(name)) != expected:
            raise ValueError(f'native project member differs: {name}')
    graph = codec.loads(member('plan/program.json').read_text())
    execution = json.loads(member('plan/execution.json').read_text())
    problem = problem_from_dict(json.loads(member('plan/problem.json').read_text()))
    reference = execution['program_graph']
    if reference != dict(path='plan/program.json', **files['plan/program.json']):
        raise ValueError('execution plan does not reference its original program graph')
    known = {operation.id.wire for operation in graph.operations}
    if any(ref.get('operation') not in known for row in problem.operations if row.id != '$return'
           for ref in row.source_refs):
        raise ValueError('physical operation lacks a source program graph node')
    if problem.source_refs != (reference,) or execution['target']['sha256'] != problem.target.sha256:
        raise ValueError('shared physical problem differs from its graph or target binding')
    executable = member(manifest['executable'])
    current = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(executable)], text=True, env={}))
    if current != manifest['dependencies'] or any(name.startswith(('libpython', 'libtorch_python')) for name in current):
        raise ValueError('runtime dependency resolution differs from the checked native project')
    locks = json.loads(member('plan/dependencies.json').read_text())
    for lock in locks['libraries'] if type(locks) is dict else locks:
        library = Path(lock['path'])
        if current.get(library.name) != str(library.resolve()) or _content(library) != {
                key: lock[key] for key in ('sha256', 'size_bytes')}:
            raise ValueError(f'native operator library differs from the build lock: {library.name}')
    return root, dict(graph_regions=len(graph.regions), graph_operations=len(graph.operations),
                      physical_activities=len(problem.operations), native_dependencies_checked=True)


def validate_native_project(directory, manifest_sha256, *, arguments=(), references=()):
    root, receipt = verify_native_project(directory, manifest_sha256)
    report = dict(schema='scar.native-project-validation', schema_version=1,
                  manifest_sha256=manifest_sha256, **receipt)
    if references:
        expected = [_content(Path(path).resolve(strict=True)) for path in references]
        with TemporaryDirectory(prefix='scar-native-validation-') as temporary:
            outputs = [Path(temporary) / str(i) for i in range(len(references))]
            subprocess.run([str(root / 'launch.sh'), *arguments, *map(str, outputs)], env={},
                           check=True, capture_output=True)
            observed = [_content(path) for path in outputs]
        if observed != expected:
            raise ValueError('native program output differs from its reference bytes')
        report['bitwise_output_receipts'] = observed
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar validate-native-project', description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    receipt = validate_native_project(args.project, args.manifest_sha256, arguments=arguments,
                                      references=args.reference)
    print(json.dumps(receipt, sort_keys=True))
    return 0
