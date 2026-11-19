"""Register and resolve an explicitly authorized source triangle provider family."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from scar.analysis import source_triangle_seed
from scar.codegen import native_publication_delivery
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_ordered_profile import ORIGINAL_OBSERVATION
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.triangle-profile-family'
_MODEL = 'scar.program-build'
_ENTRY = 'scar.python-source'
_INTERPRETER = 'scar.profile-interpreter-executable'
_PROBE = """import json, sys
from importlib.util import MAGIC_NUMBER
print(json.dumps(dict(name=sys.implementation.name, executable=sys.executable,
    version=sys.version, optimize=sys.flags.optimize, magic=MAGIC_NUMBER.hex())))
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _local(root, row):
    relative = Path(row['uri'])
    member = root / relative
    _require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == row['uri']
        and member.resolve(strict=True) == member, 'family requires a checked relative provider member')
    return {**row, 'uri': str(member)}


def _model_ref(path):
    path = Path(path).resolve(strict=True)
    return _ref(path / 'manifest.json' if path.is_dir() else path, _MODEL)


def _interpreter_path(path):
    # The executable's lexical environment matters for Python prefix discovery.
    return os.path.abspath(os.fspath(path))


def _verify_interpreter(context, expected_file):
    executable, optimize = context['executable'], context['optimize']
    _require(type(optimize) is int and 0 <= optimize <= 2 and expected_file['uri'] == executable,
        'registered interpreter has an unsupported execution profile')
    read_artifact(expected_file, json_document=False)
    args = [executable, '-I', '-B', '-S']
    if optimize:
        args.append('-' + 'O' * optimize)
    try:
        observed = subprocess.run([*args, '-c', _PROBE], check=True, capture_output=True, timeout=5,
            env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, cwd='/')
        _require(not observed.stderr and json.loads(observed.stdout) == context,
            'registered interpreter execution profile no longer matches')
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('cannot verify the registered interpreter') from error


def _derive(anchor, *, qualify):
    root = Path(anchor).resolve(strict=True)
    anchor_ref = _ref(root / 'artifact.json', native_publication_delivery.SCHEMA)
    indexed = native_publication_delivery.check(root) if qualify else read_artifact(anchor_ref)
    _require(indexed['schema'] == native_publication_delivery.SCHEMA and indexed['schema_version'] == 6
        and indexed['full_python_entry'] is False, 'family requires a source-sealed v6 provider anchor')
    seed_ref = _local(root, indexed['source_seed'])
    seed = source_triangle_seed.check(seed_ref['uri']) if qualify else read_artifact(seed_ref)
    digest_ref = _local(root, indexed['packages'][1])
    digest = read_artifact(digest_ref)
    digest_root = Path(digest_ref['uri']).parent
    release_ref = _local(digest_root, digest['qualified_release'])
    release = read_artifact(release_ref)
    portable_plan = read_artifact(_local(root, indexed['plan']))
    contract = read_artifact(_local(digest_root, digest['publication']['contract']))
    detached = read_artifact(_local(digest_root, digest['detached_source']))
    entry = seed['entry']
    context = release['context']
    _require(seed['schema'] == source_triangle_seed.SCHEMA and seed['schema_version'] == 1
        and digest['schema_version'] == 5 and release['decision'] == 'PARALLEL' and release['study_parallel'] is False
        and entry == contract['entry_source'] == release['entry'] and entry['schema'] == _ENTRY
        and portable_plan['release_sha256'] == release_ref['sha256']
        and (seed['iterations'], seed['argv']) == (release['count'], context['argv'])
        and portable_plan['source_seed']['provenance'] == indexed['source_seed'],
        'source seed, entry, release and executable publication are not the same historic scope')
    read_artifact(entry, json_document=False)
    for source in seed['sources']:
        read_artifact(source, json_document=False)

    calls_ref = seed['entry_calls']
    calls = read_artifact(calls_ref)
    model = calls['program']
    program = read_artifact(model)
    activation_ref = seed['activations']
    activation = read_artifact(activation_ref)
    provenance = seed['provenance']
    design_ref = provenance['frame_design']
    design = read_artifact(design_ref)
    selection_ref = design['source_selection']
    selection = read_artifact(selection_ref)
    _require(model['schema'] == _MODEL and program['entry'] == entry['uri']
        and activation['bounded_call_observations'] == provenance['raw_attempts']
        and activation['frame_observations'] == provenance['frames']
        and design['entry'] == entry and design['argv'] == seed['argv']
        and selection['prior_entry_projection'] == design['source_graph'],
        'selected frontend model, frame plan or constructor selector belongs to another source')
    for ref in (activation['source_catalog'], design['source_graph'], provenance['frames'], provenance['raw_attempts']):
        read_artifact(ref, json_document=False)

    scope = read_artifact(detached['source'])
    original = read_artifact(scope['source_scope'])['observed']
    aggregate_ref = original['exact_count_evidence']
    aggregate = read_artifact(aggregate_ref)
    _require(aggregate_ref['schema'] == ORIGINAL_OBSERVATION and aggregate['schema_version'] == 1
        and aggregate['entry'] == scope['entry']['source'] == entry
        and aggregate['argv'] == seed['argv'] and aggregate['affinity'] == context['cpu_ids']
        and scope['source_body']['code_key'] == detached['body_code_key']
        and detached['body_code_key'] in aggregate['code_keys'].values()
        and original['consumer_completions'] == seed['iterations']
        and aggregate['normal_completion'] is True and aggregate['installed_native_providers'] == 0,
        'historical aggregate is not the matching original execution scope')

    source_root = Path(seed['source_root']).resolve(strict=True)
    _require(source_root.is_dir() and str(source_root) == seed['source_root']
        and _interpreter_path(context['interpreter']['executable']) == context['interpreter']['executable'],
        'family source root or interpreter path is not canonical')
    executable_ref = artifact_reference(context['interpreter']['executable'], _INTERPRETER)
    if qualify:
        _verify_interpreter(context['interpreter'], executable_ref)
    frontier = dict(entry_calls=calls_ref, activation=activation_ref, catalog=activation['source_catalog'],
        frame_design=design_ref, source_selection=selection_ref, selectors=provenance['observed_call_sites'])
    binding = dict(anchor=anchor_ref, source_seed=seed_ref, release=release_ref, entry=entry, model=model,
        source_root=str(source_root), interpreter_executable=executable_ref)
    return dict(schema=SCHEMA, schema_version=1, family_id='triangle-family-v1:' + semantic_sha256(binding),
        **binding, context=context, launch_cpu_ids=release['launch_cpu_ids'], frontend=frontier,
        aggregate=aggregate_ref, scope='EXPLICIT_AUTHORITY_FOR_FRESH_SOURCE_PROFILE_COMPILATION',
        historical_costs_transferred=False, full_original_effects_closed=False)


def register(anchor, out, *, aggregate=None, model=None):
    """Run provider and source qualification once, then publish one explicit authority."""
    result = _derive(anchor, qualify=True)
    _require(model is None or _model_ref(model) == result['model'], 'supplied model is outside the sealed family')
    _require(aggregate is None or _ref(aggregate, ORIGINAL_OBSERVATION) == result['aggregate'],
        'supplied aggregate is outside the sealed historical source scope')
    out = Path(out).absolute()
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return _ref(out, SCHEMA)


def check(path):
    """Replay small authority references without rebuilding the frontend or running the workload."""
    reference = _ref(path, SCHEMA)
    stored = read_artifact(reference)
    _require(stored['schema'] == SCHEMA and stored['schema_version'] == 1,
        'expected a checked triangle profile family')
    anchor = Path(stored['anchor']['uri']).parent
    read_artifact(stored['anchor'], json_document=False)
    current = _derive(anchor, qualify=False)
    _require(stored == current, 'provider family differs from its exact source, model or interpreter authority')
    return stored


def resolve(registry_path, entry, *, interpreter=None, model=None, source_root=None):
    """Resolve one matching explicit authority for the exact source and launch identity."""
    incoming = registry_path if isinstance(registry_path, (list, tuple)) else [registry_path]
    paths = [Path(value).resolve(strict=True) for value in incoming]
    _require(paths and len(paths) == len(set(paths)), 'repeated provider family authority is ambiguous')
    source = _ref(entry, _ENTRY)
    selected_model = _model_ref(model) if model is not None else None
    selected_root = str(Path(source_root).resolve(strict=True)) if source_root is not None else None
    selected_interpreter = _interpreter_path(interpreter) if interpreter is not None else None
    choices = []
    for path in paths:
        candidate = check(path)
        if (candidate['entry'] == source
                and (selected_interpreter is None or candidate['context']['interpreter']['executable'] == selected_interpreter)
                and (selected_model is None or candidate['model'] == selected_model)
                and (selected_root is None or candidate['source_root'] == selected_root)):
            choices.append(candidate)
    _require(len(choices) == 1, 'entry, interpreter and model require exactly one authorized provider family')
    chosen, = choices
    _verify_interpreter(chosen['context']['interpreter'], chosen['interpreter_executable'])
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    create = modes.add_parser('register')
    create.add_argument('--anchor', required=True, type=Path)
    create.add_argument('--out', required=True, type=Path)
    create.add_argument('--aggregate', type=Path)
    create.add_argument('--model', type=Path)
    verify = modes.add_parser('check')
    verify.add_argument('--registry', required=True, type=Path)
    lookup = modes.add_parser('resolve')
    lookup.add_argument('--registry', action='append', required=True, type=Path)
    lookup.add_argument('--entry', required=True, type=Path)
    lookup.add_argument('--interpreter', type=Path)
    lookup.add_argument('--model', type=Path)
    lookup.add_argument('--source-root', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'register':
        result = register(args.anchor, args.out, aggregate=args.aggregate, model=args.model)
    elif args.mode == 'resolve':
        result = resolve(args.registry, args.entry, interpreter=args.interpreter, model=args.model, source_root=args.source_root)
    else:
        result = check(args.registry)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
