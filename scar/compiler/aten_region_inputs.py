"""Verify the retained discovery inputs before planning an ATen region."""

from pathlib import Path

from scar.analysis.aten_python_trial_source import CONTRACT_SCHEMA
from scar.analysis.aten_region_boundary import qualify_aten_region_recipe
from scar.compiler.aten_native_recipe import SCHEMA as RECIPE_SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact


DISCOVERY_SCHEMA = 'scar.aten-region-discovery'
_PROJECTION_SCHEMA = 'scar.aten-capture-graph-projection'
_LINKAGE_SCHEMA = 'scar.aten-provider-linkage'
_PROGRAM_SCHEMA = 'scar.program-build'
_TARGET_SCHEMA = 'scar.target-hardware-graph'


def _document(reference, schema, versions=(1,)):
    if type(reference) is not dict or reference.get('schema') != schema:
        raise ValueError('expected a retained ' + schema + ' reference')
    value = read_artifact(reference)
    version = value.get('schema_version')
    if type(version) is not int or version not in versions:
        raise ValueError('unsupported ' + schema + ' version')
    return value


def _target_reference(program):
    record = program.get('target')
    if (type(record) is not dict or type(record.get('path')) is not str
            or type(record.get('bytes')) is not int or Path(record['path']).name != 'result.json'):
        raise ValueError('the F4 program must retain its target graph receipt')
    reference = dict(uri=record['path'], sha256=record.get('sha256'), size_bytes=record['bytes'],
                     schema=_TARGET_SCHEMA)
    _document(reference, _TARGET_SCHEMA, versions=(1, 2))
    return reference


def _original_region(report):
    candidates = [region for region in report['regions'] if region['status'] == 'CLOSED_SOURCE_CANDIDATE']
    if len(candidates) != 1:
        raise ValueError('a closed replay must identify one discovered region')
    region, = candidates
    representative, boundary = region['representative'], region['boundary']
    attempts = representative['attempts']
    if (type(attempts) is not list or not attempts
            or any(type(attempt) is not int or attempt < 1 for attempt in attempts)
            or attempts != sorted(set(attempts)) or attempts != boundary['call_attempts']
            or representative['invocation'] != boundary['invocation']
            or boundary['source_replacement_authorized'] is not False):
        raise ValueError('discovered representative differs from its exact original boundary')
    return region


def _verify_projection(report, region, published, projected, recipe, linkage):
    attempts = region['representative']['attempts']
    inputs = report['inputs']
    if (projected['inputs'] != dict(program=inputs['program'], capture=inputs['capture'], call_attempts=attempts)
            or projected['recipe_sha256'] != published['recipe']['sha256']
            or projected['linkage']['linkage_report'] != published['linkage']
            or any(projected[key] != report[key] for key in ('run', 'process'))
            or projected['original_source_decision'] != 'KEEP_ORIGINAL'
            or projected['source_replacement_authorized'] is not False or projected['reordered'] is not False):
        raise ValueError('projection differs from the discovered capture, program or boundary')

    binding = projected['linkage']
    captured = _document(binding['capture_source'], 'scar.aten-capture-source')
    occurrences = binding['occurrences']
    if (any(captured[key] != report[key] for key in ('run', 'process'))
            or captured['occurrences'] != occurrences or captured['linkage_report'] != published['linkage']
            or captured['provider_revision'] != linkage['provider']['revision']
            or captured['replacement_authorized'] is not False):
        raise ValueError('provider linkage differs from retained scoped capture facts')
    if ([event['call_attempt'] for event in occurrences] != attempts
            or [dict(site=event['call_site'], schemas=[event['schema']]) for event in occurrences] != region['sites']
            or [(f"call_{event['call_attempt']}", event['schema']) for event in occurrences]
            != [(operation['id'], operation['schema']) for operation in recipe['operations']]):
        raise ValueError('recipe differs from the observed CALL and provider order')


def read_aten_region_inputs(discovery, *, discovery_sha256):
    """Bind a pinned discovery to its original source, recipe and target bytes."""
    path = Path(discovery).resolve(strict=True)
    discovery_ref = artifact_reference(path, DISCOVERY_SCHEMA)
    if discovery_ref['sha256'] != discovery_sha256:
        raise ValueError('ATen discovery differs from its declared digest')
    report = _document(discovery_ref, DISCOVERY_SCHEMA)
    if report['source_replacement_authorized'] is not False:
        raise ValueError('ATen discovery cannot authorize a source replacement')

    base = dict(discovery=discovery_ref, source_replacement_authorized=False)
    published = report.get('projection')
    if published is None:
        return dict(base, status='KEEP_ORIGINAL', reason='PROJECTION_NOT_REQUESTED')
    if published['status'] == 'KEEP_ORIGINAL':
        return dict(base, status='KEEP_ORIGINAL', reason=published['reason'],
                    **{key: published[key] for key in ('detail', 'eligible', 'stage', 'exit_code') if key in published})
    if published['status'] != 'CLOSED_SOURCE_REPLAY':
        raise ValueError('unknown ATen discovery projection status')

    region = _original_region(report)
    refs = report['inputs']
    program = _document(refs['program'], _PROGRAM_SCHEMA)
    if program['backend'] != 'F4':
        raise ValueError('the discovery must refer to its original F4 program')
    target = _target_reference(program)
    contract = _document(refs['wrapper_contract'], CONTRACT_SCHEMA)
    projected = _document(published['projection'], _PROJECTION_SCHEMA)
    recipe = _document(published['recipe'], RECIPE_SCHEMA)
    linkage = _document(published['linkage'], _LINKAGE_SCHEMA)
    if linkage['status'] != 'EXTERNAL_CPP_CALL_LINKED' or linkage['cxx_standard'] != 20:
        raise ValueError('the discovery must retain external C++20 provider linkage')
    _verify_projection(report, region, published, projected, recipe, linkage)
    proof = qualify_aten_region_recipe(region['boundary'], recipe, contract)
    if proof != published['source']:
        raise ValueError('source proof differs from a replay of the retained Python function')
    return dict(base, status=published['status'], program=refs['program'], target=target,
                wrapper_contract=refs['wrapper_contract'], projection=published['projection'],
                recipe=published['recipe'], linkage=published['linkage'], source=proof,
                representative=region['representative'])
