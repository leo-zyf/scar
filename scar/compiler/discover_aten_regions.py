"""Discover complete source callables from joined F4 and native ATen scopes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

from scar.analysis.aten_python_trial_source import CONTRACT_SCHEMA
from scar.analysis.aten_region_boundary import (
    AtenRegionBoundaryError, AtenRegionBoundaryIndex, qualify_aten_region_recipe,
)
from scar.analysis.build import BuildSession
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.aten_scoped_ports import read_scoped_aten_capture
from scar.trace.capture_coordinates import bind_capture_coordinates


def _region_profiles(scoped):
    activations = defaultdict(list)
    for call in scoped.calls:
        invocation = call['invocation']
        key = invocation['process_id'], invocation['thread_id'], invocation['invocation_id']
        activations[key].append(call)

    profiles = {}
    for calls in activations.values():
        signature = tuple((call['site']['code'], call['site']['offset'],
                           tuple(op['schema'] for op in call['operators'])) for call in calls)
        representative, count = profiles.get(signature, (calls, 0))
        profiles[signature] = representative, count + 1
    return profiles.values()


def discover_aten_regions(build, capture, *, contract, contract_sha256):
    build = Path(build).resolve(strict=True)
    contract_ref = artifact_reference(Path(contract).resolve(strict=True), CONTRACT_SCHEMA)
    if contract_ref['sha256'] != contract_sha256:
        raise ValueError('wrapper contract differs from its declared artifact digest')
    declarations = read_artifact(contract_ref)
    coordinates = bind_capture_coordinates(capture)
    scoped = read_scoped_aten_capture(coordinates)
    coverage = scoped.scopes
    regions = []
    with BuildSession(build) as session:
        if session.report.get('backend') != 'F4':
            raise ValueError('original F4 program build required')
        boundaries = AtenRegionBoundaryIndex(session.view, coordinates, declarations)
        for calls, count in _region_profiles(scoped):
            region = dict(
                sites=[dict(site=call['site'], schemas=[op['schema'] for op in call['operators']])
                       for call in calls],
                representative=dict(invocation=calls[0]['invocation'], attempts=[call['attempt'] for call in calls]),
                observed_activations=count,
            )
            try:
                region['boundary'] = boundaries.qualify(calls)
            except AtenRegionBoundaryError as error:
                region.update(status='KEEP_ORIGINAL', reason=error.reason, detail=str(error))
            else:
                region['status'] = 'CLOSED_SOURCE_CANDIDATE'
            regions.append(region)
        boundaries.check()

    return dict(schema='scar.aten-region-discovery', schema_version=1,
                inputs=dict(program=artifact_reference(build / 'manifest.json', 'scar.program-build'),
                            capture=str(Path(capture).resolve()), wrapper_contract=contract_ref),
                run=coverage['run'], process=coverage['process'],
                coverage={key: coverage[key] for key in ('scope_rows', 'closed_joined_scopes', 'gaps',
                          'collector_availability', 'collector_budget_exhausted', 'collector_errors')},
                regions=regions, source_replacement_authorized=False), declarations


def _stage(command, *args):
    # F4 replay remains in a fresh process, ahead of importing the ATen provider.
    subprocess.run([sys.executable, '-m', 'scar', command, *args], check=True, stdout=subprocess.PIPE)


def _project_single(report, declarations, *, capture_model, capture_model_sha256, out):
    eligible = [region for region in report['regions'] if region['status'] == 'CLOSED_SOURCE_CANDIDATE']
    if len(eligible) != 1:
        return dict(status='KEEP_ORIGINAL', reason='ONE_CLOSED_SOURCE_CANDIDATE_REQUIRED', eligible=len(eligible))
    region, = eligible
    attempts = region['representative']['attempts']
    args = [argument for attempt in attempts
            for argument in ('--call-attempt', str(attempt))]
    linked, projected = out / 'linked', out / 'projected'
    _stage('link-aten-capture', '--capture-model', capture_model, '--capture-model-sha256', capture_model_sha256,
           '--out', str(linked), *args)
    linkage = artifact_reference(linked / 'result.json', 'scar.aten-provider-linkage')
    _stage('project-aten-graph', '--build', str(Path(report['inputs']['program']['uri']).parent),
           '--capture', report['inputs']['capture'], '--linkage', linkage['uri'],
           '--linkage-sha256', linkage['sha256'], '--out', str(projected), *args)
    recipe = artifact_reference(projected / 'recipe.json', 'scar.aten-native-recipe')
    projection = artifact_reference(projected / 'projection.json', 'scar.aten-capture-graph-projection')
    projected_report = read_artifact(projection)
    expected_inputs = dict(program=report['inputs']['program'], capture=report['inputs']['capture'],
                           call_attempts=attempts)
    if (projected_report['inputs'] != expected_inputs or projected_report['recipe_sha256'] != recipe['sha256']
            or any(projected_report[key] != report[key] for key in ('run', 'process'))):
        raise ValueError('projection differs from the discovered F4 and capture identity')
    proof = qualify_aten_region_recipe(region['boundary'], read_artifact(recipe), declarations)
    return dict(status='CLOSED_SOURCE_REPLAY', linkage=linkage, recipe=recipe, projection=projection, source=proof)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar discover-aten-regions', description=__doc__)
    parser.add_argument('--build', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--capture-model', help='link and project the unique closed candidate')
    parser.add_argument('--capture-model-sha256')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if bool(args.capture_model) != bool(args.capture_model_sha256):
        parser.error('--capture-model and --capture-model-sha256 must be provided together')
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    report, declarations = discover_aten_regions(args.build, args.capture, contract=args.contract,
                                                  contract_sha256=args.contract_sha256)
    out.mkdir(parents=True)
    if args.capture_model:
        try:
            report['projection'] = _project_single(report, declarations, capture_model=args.capture_model,
                                                    capture_model_sha256=args.capture_model_sha256, out=out)
        except AtenRegionBoundaryError as error:
            report['projection'] = dict(status='KEEP_ORIGINAL', reason=error.reason, detail=str(error))
        except subprocess.CalledProcessError as error:
            report['projection'] = dict(status='KEEP_ORIGINAL', reason='PROJECTION_STAGE_FAILED',
                                        stage=error.cmd[3], exit_code=error.returncode)
    path = out / 'discovery.json'
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(path), regions=len(report['regions']),
                         candidates=sum(row['status'] == 'CLOSED_SOURCE_CANDIDATE' for row in report['regions']),
                         projection=report.get('projection', {}).get('status'))))
    return 0
