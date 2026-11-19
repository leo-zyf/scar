"""Qualified device alternatives for one immutable FP32 export."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.compiler.aten_export_hybrid_source import _crossings, qualify_hybrid_export
from scar.ir.program import codec
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS
from scar.optimization.native_device_placement_search import search_native_device_placements


_MM = FUNCTIONAL_EXPORT_SCHEMAS['aten.mm.default']


def qualify_device_export(recipe, shapes=None, *, graph, program_ref, input_kinds):
    return qualify_hybrid_export(recipe, shapes, graph=graph,
        program_ref=program_ref, input_kinds=input_kinds, allow_cuda_only=True)


def placement_choices(projection):
    return {row['id']: ('CPU', 'CUDA') if row['schema'] == _MM else ('CPU',)
            for row in projection['operations']}


def placement_proposal(projection, choices, analysis, selected, *, actual_device_timing='UNKNOWN'):
    layout, regions, crossings = None, [], []
    if selected is not None:
        layout = selected['placement']
        placed = [dict(row, placement=layout[row['id']]) for row in projection['operations']]
        regions, crossings = _crossings(projection['inputs'], placed, projection['outputs'])
    return dict(schema='scar.native-export-placement-proposal', schema_version=1,
                domain={name: list(modes) for name, modes in choices.items()}, selected_placement=layout,
                selected_cuda_regions=regions, selected_crossings=crossings,
                analysis=analysis, actual_device_timing=actual_device_timing, default_backend='CPU',
                deployment_authorized=False)


def propose_device_placement(projection, model, *, model_ref):
    from scar.compiler.aten_export_hybrid_plan import plan_placed_hybrid_export

    choices = placement_choices(projection)

    def build(placement):
        return plan_placed_hybrid_export(projection, model, model_ref=model_ref, placement=placement)

    analysis, selected = search_native_device_placements(choices, build)
    return placement_proposal(projection, choices, analysis, selected), selected


def plan_device_export(projection, model, *, model_ref):
    report, selected = propose_device_placement(projection, model, model_ref=model_ref)
    if selected is None or 'CUDA' not in selected['placement'].values():
        raise ValueError('declared placement retains CPU or is unresolved; no CUDA candidate was nominated')
    return selected['problem'], selected['schedule'], selected['bindings'], report


def replay_device_export(projection, model, *, model_ref, problem, schedule, bindings, placement_search):
    expected_problem, expected_schedule, expected_bindings, expected = plan_device_export(
        projection, model, model_ref=model_ref)
    if (expected != placement_search or expected_problem != problem or expected_bindings != bindings
            or {key: value for key, value in expected_schedule.items() if key != 'elapsed_seconds'} !=
            {key: value for key, value in schedule.items() if key != 'elapsed_seconds'}):
        raise ValueError('CUDA placement or schedule differs from its replayed model and source choices')


def _from_cpu(cpu, cpu_sha256, profile, model_bytes, model_ref, *, empirical=None):
    from scar.compiler.native_device_pair import CPU_SCHEMA, _manifest, _member
    from scar.runtime.validate_native_project import verify_native_project

    root, _ = verify_native_project(cpu, cpu_sha256)
    manifest = _manifest(root, cpu_sha256, CPU_SCHEMA)
    family = manifest.get('profiles')
    if (family is None and profile is not None or family is not None and profile not in family):
        raise ValueError('choose exactly one published CPU shape specialization')
    base = 'plan' if profile is None else f'plan/profiles/{profile}'
    execution = json.loads(_member(root, manifest, base + '/execution.json'))
    reference = execution['program_graph']
    graph = codec.loads(_member(root, manifest, reference['path']).decode())
    recipe_file = 'source-recipe.json' if manifest.get('simplified_normal_completion') else 'recipe.json'
    recipe = json.loads(_member(root, manifest, base + '/' + recipe_file))
    projection = qualify_device_export(recipe, graph=graph, program_ref=reference,
        input_kinds={row['id']: row['kind'] for row in execution['inputs']})
    model = json.loads(model_bytes)
    if empirical is None:
        proposal, _ = propose_device_placement(projection, model, model_ref=model_ref)
    else:
        from scar.compiler.native_device_placement_empirical import propose_empirical_device_placement

        proposal = propose_empirical_device_placement(root, cpu_sha256, profile, projection, model, model_ref,
                                                       **empirical)
    return dict(proposal, cpu_manifest_sha256=cpu_sha256, cpu_profile=profile, program=reference,
                declared_model=model_ref)


def _model_reference(raw):
    return dict(path='device-model.json', sha256=sha256(raw).hexdigest(), size_bytes=len(raw))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-native-device-placement', description=__doc__)
    parser.add_argument('--cpu-project', required=True)
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--cpu-profile')
    parser.add_argument('--device-model')
    parser.add_argument('--device-model-sha256')
    parser.add_argument('--empirical', action='store_true', help='use only exact measured treatments')
    parser.add_argument('--input', action='append', default=[], metavar='NAME=FILE')
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('--device-evidence', nargs=3, action='append', default=[],
                        metavar=('CUDA_PROJECT', 'CALIBRATION_JSON', 'SHA256'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out')
    action.add_argument('--check')
    parser.add_argument('--check-sha256')
    args = parser.parse_args(argv)
    if args.check:
        path = Path(args.check).resolve(strict=True)
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != args.check_sha256:
            raise ValueError('device placement proposal differs from its declared digest')
        stored = json.loads(raw)
        model_path = path.with_name('device-model.json')
        if model_path.resolve(strict=True) != model_path:
            raise ValueError('device model must be local to its archived placement proposal')
        model = model_path.read_bytes()
        evidence = stored.get('empirical')
        empirical = None
        if evidence is not None:
            version = stored['analysis']['schema_version']
            if type(version) is not int or version not in (1, 2):
                raise ValueError('unsupported empirical placement analysis version')
            empirical = dict(inputs=evidence['inputs'], references=evidence['references'],
                device_evidence=[(row['project'], row['calibration'], row['sha256']) for row in evidence['device_evidence']],
                service_intervals=version == 2)
        proposal = _from_cpu(args.cpu_project, stored['cpu_manifest_sha256'], stored['cpu_profile'],
                             model, _model_reference(model), empirical=empirical)
        if proposal != stored:
            raise ValueError('device placement proposal differs from its replayed source choices')
    else:
        if not (args.cpu_sha256 and args.device_model and args.device_model_sha256):
            parser.error('--out requires the CPU manifest digest and the declared device model and digest')
        out = Path(args.out).resolve()
        if out.exists():
            raise FileExistsError(out)
        model = Path(args.device_model).read_bytes()
        model_ref = _model_reference(model)
        if model_ref['sha256'] != args.device_model_sha256:
            raise ValueError('device model differs from its declared digest')
        if not args.empirical and (args.input or args.reference or args.device_evidence):
            parser.error('sample inputs, references and device evidence require --empirical')
        empirical = dict(named_inputs=args.input, references=args.reference, device_evidence=args.device_evidence) \
            if args.empirical else None
        proposal = _from_cpu(args.cpu_project, args.cpu_sha256, args.cpu_profile, model, model_ref,
                             empirical=empirical)
        out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.scar-device-placement-', dir=out.parent) as temporary:
            stage = Path(temporary)
            (stage / model_ref['path']).write_bytes(model)
            (stage / 'placement.json').write_text(json.dumps(proposal, sort_keys=True, indent=2) + '\n')
            stage.rename(out)
    print(json.dumps(dict(selected_placement=proposal['selected_placement'],
                          default_backend=proposal['default_backend'], analysis=proposal['analysis']), sort_keys=True))
    return 0
