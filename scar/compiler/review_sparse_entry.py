"""Review ordered sparse entry studies against the already released implementation."""

import argparse
import json
from pathlib import Path

from scar.codegen.native_ordered_sparse_export import SCHEMA as PACKAGE
from scar.compiler.plan_dead_result_item import SCHEMA as DEAD_ITEM
from scar.compiler.plan_sparse_parallel import EXECUTION, _check_ordered_activity
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs
from scar.runtime.native_detached_entry import _admit_package
from scar.runtime.sparse_entry_review_launch import SCHEMA as DECISION
from scar.runtime.sparse_parallel_entry import SCHEMA as INCUMBENT


DESIGN = 'scar.sparse-entry-review-design'
_EVIDENCE = 'scar.dead-item-entry-validation'
_COMPACT_REPORT = 'scar.native-detached-entry-execution'
_MODES = ('compact', 'dead_item', 'parallel')
_SCOPE = 'wall clock of full-entry fresh process, including launch and receipt'
_RUNTIME = ('runtime/native_detached_entry.py', 'runtime/native_detached_stream.py', 'runtime/dead_result_item.py',
    'runtime/entry_runner.py', 'runtime/csr_region_entry.py', 'runtime/sparse_pipeline_entry.py',
    'codegen/native_ordered_sparse_export.py', 'codegen/sparse_native_export.py', 'codegen/native_provider_bundle.py',
    'compiler/plan_dead_result_item.py')
_TIMING_FIELDS = ('pair', 'mode', 'wall_ns', 'returncode', 'stdout_sha256', 'stderr_sha256')
_PROCESS_FIELDS = ('command', 'wall_ns', 'returncode', 'stdout_sha256', 'stdout_bytes', 'stderr_sha256', 'stderr_bytes')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _write(path, document):
    path = Path(path).resolve()
    with path.open('x') as output:
        json.dump(document, output, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n')
    return artifact_reference(path, document['schema'])


def _released(reference):
    selected = read_artifact(reference)
    _require(reference['schema'] == INCUMBENT and selected['schema_version'] == 1
        and selected['decision'] == 'PARALLEL' and selected['study_parallel'] is False
        and 'validation' in selected and 'execution_recipe' in selected,
        'already released typed incumbent required')
    for dependency in (selected['entry'], *selected['runtime_members']):
        read_artifact(dependency, json_document=False)
    validation = read_artifact(selected['validation'])
    launch_context = {**selected['context'], 'cpu_ids': selected['launch_cpu_ids']}
    _require(validation['clean'] is True and validation['context'] == launch_context
        and validation['selection'] == selected['study'], 'incumbent lacks its original clean release evidence')
    return selected


def prepare(selection, package, dead_item, harness, harness_support, out):
    incumbent = artifact_reference(Path(selection).resolve(), INCUMBENT)
    selected = _released(incumbent)
    package = Path(package).resolve(strict=True)
    native = artifact_reference(package / 'artifact.json', PACKAGE)
    _, manifest, _, _ = _admit_package(package, native['sha256'], selected)
    plan = artifact_reference(Path(dead_item).resolve(), DEAD_ITEM)
    rewrite = read_artifact(plan)
    _require(rewrite['schema_version'] == 2 and rewrite['source_selection'] == incumbent
        and rewrite['auto_apply'] is False and manifest['deployment_authorized'] is False,
        'compact candidates must be the matching explicitly unpromoted studies')
    directory = Path(__file__).resolve().parents[1]
    code = [artifact_reference(directory / name, 'scar.python-source') for name in _RUNTIME]
    return _write(out, dict(schema=DESIGN, schema_version=1, incumbent=incumbent, entry=selected['entry'],
        context=selected['context'], launch_affinity=selected['launch_cpu_ids'], owner_cpu=selected['owner_cpu'],
        execution_recipe=selected['execution_recipe'], native_package=native, dead_item_plan=plan,
        runtime_members=code, harness=artifact_reference(Path(harness).resolve(), 'scar.python-source'),
        harness_support=artifact_reference(Path(harness_support).resolve(), 'scar.python-source'),
        scope=_SCOPE, warmup_pairs=1, measurement_pairs=3, minimum_gain=0.05, variants=list(_MODES)))


def _expected_command(row, frozen):
    selection = frozen['incumbent']
    context = frozen['context']
    mode = row['mode']
    compact = mode != 'parallel'
    module = 'scar.runtime.native_detached_entry' if compact else 'scar.runtime.sparse_parallel_entry'
    command = ['taskset', '-c', ','.join(map(str, frozen['launch_affinity'])),
        context['interpreter']['executable'], '-B', '-m', module, '--exclusive-inputs',
        '--selection', selection['uri'], '--selection-sha256', selection['sha256']]
    if compact:
        native = frozen['native_package']
        command += ['--native-detached', str(Path(native['uri']).parent), '--native-detached-sha256', native['sha256']]
    if mode == 'dead_item':
        plan = frozen['dead_item_plan']
        command += ['--dead-item', plan['uri'], '--dead-item-sha256', plan['sha256']]
    return [*command, '--report', row['receipt']['uri'], '--', frozen['entry']['uri'], *context['argv']]


def _verify_launch(row, design, selected, compact_input_bounds):
    raw = read_artifact(row['receipt'])
    mode = row['mode']
    parallel = mode == 'parallel'
    count = selected['count']
    stats = raw['statistics']
    _require(row['command'] == _expected_command(row, design)
        and raw['schema'] == (EXECUTION if parallel else _COMPACT_REPORT)
        and raw['schema_version'] == 1 and raw['entry_path'] == selected['entry']['uri']
        and raw['argv'] == selected['context']['argv'] and raw['selection_sha256'] == design['incumbent']['sha256']
        and raw['exit_code'] == row['returncode'] == 0 and stats['fallback_calls'] == 0
        and raw['observed_boundary_calls'] == count,
        'process launch, source, guarded result or coverage differed')
    if parallel:
        offers = raw['parallel']
        team = offers['workers']
        _require(raw['applied'] == 'PARALLEL' and stats['native_calls'] == count and offers['matched'] == count
            and offers['serial_calls'] == offers['rejected'] == 0 and team['closed']
            and team['native_calls'] == team['submissions'] == count
            and len(team['publications']) == count and team['pending_snapshot_bytes'] == 0
            and team['retained_after_close_bytes'] == 0, 'incumbent did not complete its released worker lane')
        _check_ordered_activity(team, selected)
        return
    offers = raw['compact']
    stream = offers['transport']
    _require(raw['applied'] == 'CPP_DETACHED' and raw['package_sha256'] == design['native_package']['sha256']
        and stats['native_calls'] == offers['matched'] == count - 1 and stats['reference_calls'] == 1
        and offers['reference_calls'] == 1 and offers['rejected'] == 0
        and compact_input_bounds[0] <= stream['input_bytes'] <= compact_input_bounds[1]
        and stream['accepted'] == count and stream['last_terminal'] == [count, 4, 0]
        and stream['completed'] is True and stream['closed'] is True,
        'compact study did not publish the matching guarded result sequence')
    if mode == 'dead_item':
        elision = raw['dead_item']
        _require(elision['installed_bindings'] == 1 and elision['retained_first'] == 1
            and elision['elided'] == count - 1 and elision['fallback'] == 0,
            'dead result item did not execute throughout its admitted boundary')
    else:
        _require('dead_item' not in raw, 'ordinary compact control contained an extra rewrite')


def qualify(design_path, evidence_path, out):
    design = artifact_reference(Path(design_path).resolve(), DESIGN)
    frozen = read_artifact(design)
    evidence_ref = artifact_reference(Path(evidence_path).resolve(), _EVIDENCE)
    evidence = read_artifact(evidence_ref)
    _require(frozen['schema_version'] == evidence['schema_version'] == 1 and evidence['review_design'] == design
        and evidence['harness_support'] == frozen['harness_support'] and frozen['variants'] == list(_MODES)
        and evidence['selection'] == frozen['incumbent'] and evidence['original'] == frozen['entry']
        and all(evidence[key] == frozen[key] for key in ('context', 'launch_affinity', 'owner_cpu', 'execution_recipe',
            'scope', 'harness', 'native_package', 'dead_item_plan', 'warmup_pairs', 'measurement_pairs')),
        'evidence is outside its frozen clean-process study')
    selected = _released(frozen['incumbent'])
    for reference in (frozen['harness'], frozen['harness_support'], *frozen['runtime_members']):
        read_artifact(reference, json_document=False)
    package = Path(frozen['native_package']['uri']).parent
    _, manifest, _, _ = _admit_package(package, frozen['native_package']['sha256'], selected)
    read_artifact(frozen['dead_item_plan'])
    protocol = manifest['stream']['protocol']
    wire = read_artifact({**protocol, 'uri': str(package / protocol['uri'])})['bytes']
    compact_input_bounds = (wire['input_complete_base'], wire['input_complete_max'])
    orders = [_MODES[pair % len(_MODES):] + _MODES[:pair % len(_MODES)]
        for pair in range(frozen['warmup_pairs'] + frozen['measurement_pairs'])]
    runs = evidence['runs']
    _require([(row['pair'], row['mode']) for row in runs] == [(index, mode) for index, order in enumerate(orders)
            for mode in order] and evidence['orders'] == list(map(list, orders)), 'complete rotated process runs required')
    release_outputs = {(run['stdout_sha256'], run['stderr_sha256'])
        for run in read_artifact(selected['validation'])['runs']}
    _require(len(release_outputs) == 1 and release_outputs == {(row['stdout_sha256'], row['stderr_sha256']) for row in runs},
        'study outputs differ from the incumbent release evidence')
    processes = []
    for row in runs:
        reference = artifact_reference(Path(row['receipt']['uri']).with_suffix('.process.json'), 'scar.process-observation')
        process = json.loads(read_artifact(reference, json_document=False))
        _require(process == {key: row[key] for key in _PROCESS_FIELDS}, 'run differs from its raw process observation')
        _verify_launch(row, frozen, selected, compact_input_bounds)
        processes.append(reference)
    reviews = {candidate: compare_entry_pairs([{key: row[key] for key in _TIMING_FIELDS} for row in runs
        if row['mode'] in ('parallel', candidate)], incumbent='parallel', challenger=candidate,
        warmup_pairs=frozen['warmup_pairs'], measurement_pairs=frozen['measurement_pairs'],
        minimum_gain=frozen['minimum_gain']) for candidate in ('compact', 'dead_item')}
    advance = any(review['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW' for review in reviews.values())
    return _write(out, dict(schema=DECISION, schema_version=1, entry=frozen['entry'], context=frozen['context'],
        incumbent=frozen['incumbent'], design=design, evidence=evidence_ref, processes=processes, comparisons=reviews,
        decision='RELEASE_REVIEW_REQUIRED' if advance else 'KEEP_INCUMBENT', challengers_auto_authorized=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('prepare')
    for field in ('selection', 'package', 'dead-item', 'harness', 'harness-support', 'out'):
        freeze.add_argument('--' + field, required=True)
    review = commands.add_parser('qualify')
    for field in ('design', 'evidence', 'out'):
        review.add_argument('--' + field, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.selection, args.package, args.dead_item, args.harness, args.harness_support, args.out)
    else:
        result = qualify(args.design, args.evidence, args.out)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
