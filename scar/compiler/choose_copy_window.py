"""Replay canonical window costs and publish a finite component-model choice."""
import argparse
import json
from pathlib import Path

from scar.compiler.calibrate_copy_window import read_copy_window_calibration, SCHEMA as CALIBRATION_SCHEMA
from scar.ir.activity_costs import activity_cost_document
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.copy-window-choice-preparation'


def _derive(reference):
    from scar.optimization.copy_window_choice import choose_copy_window
    document, context = read_copy_window_calibration(reference)
    if document != read_artifact(reference):
        raise ValueError('source-derived copy calibration changed during choice preparation')
    original, replacement, choice = choose_copy_window(context['original_schedule'],
        context['original_resources'], context['original_table'], context['rewritten_schedule'],
        context['rewritten_resources'], context['rewritten_table'], context['certificate'], reference, document)
    read_artifact(reference)
    return choice, original, replacement, context


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _report(reference, artifacts, choice, context):
    root = Path(__file__).resolve().parents[1]
    return dict(schema=SCHEMA, schema_version=1, calibration=reference,
        rewrite=context['rewrite_reference'], **artifacts,
        implementations={name:artifact_reference(root/name,'scar.python-source') for name in
            ('compiler/choose_copy_window.py','optimization/copy_window_choice.py')},
        selected_variant=choice['selected_variant'], decision=choice['decision'],
        scope='FINITE_FIXED_CPU_CANONICAL_COPY_WINDOW_COMPONENT_MODEL',
        target_program_executed=False, native_compiled=False, whole_scope_optimality_proven=False,
        deployment_authorized=False, default_runtime_decision='NO_OP')


def prepare_copy_window_choice(calibration, out):
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh copy-choice output required')
    reference = artifact_reference(Path(calibration).resolve(), CALIBRATION_SCHEMA)
    choice, original, replacement, context = _derive(reference)
    directory.mkdir(parents=True)
    documents = dict(choice=choice, original_costs=activity_cost_document(original),
        replacement_costs=activity_cost_document(replacement))
    artifacts = {key:_write(directory/(key+'.json'),value) for key,value in documents.items()}
    report = _report(reference, artifacts, choice, context)
    _write(directory/'result.json', report)
    return report


def replay_copy_window_choice(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'),report.get('schema_version')) != (SCHEMA,1):
        raise ValueError('versioned source-derived copy-window choice required')
    choice, original, replacement, context = _derive(report['calibration'])
    documents = dict(choice=choice, original_costs=activity_cost_document(original),
        replacement_costs=activity_cost_document(replacement))
    expected = _report(report['calibration'], {key:report[key] for key in documents}, choice, context)
    if report != expected or any(semantic_sha256(read_artifact(report[key])) != semantic_sha256(value)
            for key,value in documents.items()):
        raise ValueError('copy-window choice differs from source, raw evidence and complete cost-model replay')
    read_artifact(reference)
    return report, choice, original, replacement, context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--calibration'); source.add_argument('--check')
    parser.add_argument('--out'); args = parser.parse_args(argv)
    if args.check:
        if args.out: parser.error('check only consumes a saved choice receipt')
        report = replay_copy_window_choice(args.check)[0]
    else:
        if not args.out: parser.error('choice preparation requires --out')
        report = prepare_copy_window_choice(args.calibration,args.out)
    print(json.dumps({key:report[key] for key in ('selected_variant','decision','scope',
        'deployment_authorized','default_runtime_decision')},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
