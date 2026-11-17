"""Bind conditional source summaries to an independently admitted native region.

Source identity and MAY summaries are not purity, effect closure, runtime
state identity, source transformation permission or deployment qualification.
"""
from pathlib import Path
import json

from scar.ir.execution_recipe import _reference, artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.stateful-source-summary-binding'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _ref(value, schema=None):
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        reference = artifact_reference(path, schema or json.loads(path.read_bytes())['schema'])
    else:
        reference = _reference(value)
    _need(schema is None or reference['schema'] == schema, 'source summary binding artifact schema differs')
    return reference


def _entry_source(session, program):
    owner = session._f4
    _need(artifact_reference(owner.path/'manifest.json', 'scar.program-build') == program,
        'entry session differs from its retained ProgramBuild')
    model = owner.catalog.fact_store.model_units[owner.report['f4']['entry_unit']]
    path = Path(owner.report['entry']).absolute()
    reference = artifact_reference(path, 'scar.python-source')
    _need((model['path'], model['raw_sha256']) == (str(path), reference['sha256']),
        'original entry source version differs from its F4 model')
    return reference


def derive_stateful_source_summary_binding(summary, specialization, *, entry_calls):
    """Cold-replay path/ArtifactRef inputs; never import or execute the target.

    Different ProgramBuild receipts may describe the same entry/source version.
    Both are retained and checked; their identity is never inferred from names.
    """
    from scar.analysis.build import BuildSession
    from scar.analysis.bounded_entry_calls import derive_entry_source_witness
    from scar.analysis.interprocedural_summary import verify_interprocedural_summary
    from scar.compiler.native_lowering_view import derive_stateful_lowering_view
    from scar.compiler.stateful_candidate import check_observed_specialization
    from scar.compiler.stateful_model import stateful_from_dict, stateful_digest

    summary_ref = _ref(summary)
    specialization_ref = _ref(specialization, 'scar.observed-stateful-specialization')
    calls_ref = _ref(entry_calls, 'scar.bounded-entry-calls')
    retained_summary = read_artifact(summary_ref)
    calls = read_artifact(calls_ref)
    admission = check_observed_specialization(specialization_ref)
    _need(admission['native_metadata_verified'] is True, 'source-admitted native validation candidate required')
    specialized = read_artifact(specialization_ref)
    candidate = read_artifact(specialized['candidate'])
    region = stateful_from_dict(read_artifact(specialized['source_region']))
    view = derive_stateful_lowering_view(candidate['native_artifact'])
    _need(candidate['region'] == specialized['source_region']
        and candidate['program'] == specialized['program']
        and view['identity']['region_sha256'] == stateful_digest(region),
        'native lowering, candidate and original stateful region differ')
    with BuildSession(Path(calls['program']['uri']).parent) as session:
        checked = verify_interprocedural_summary(session, retained_summary)
        _need(checked == retained_summary, 'source summary verifier did not return the complete retained report')
        _need(checked['entry_projection'] == dict(sha256=semantic_sha256(calls),
            program=calls['program'], producer=calls['producer']),
            'source summary belongs to another original entry projection')
        entry_source = _entry_source(session, calls['program'])
        source = region.source
        witness = derive_entry_source_witness(session, calls,
            path=source.path, sha256=source.sha256, code_key=source.code_key)
        _need(witness['witnessed'] is True and witness['program'] == calls['program'],
            'native source lacks an original conditional entry/source witness')
        _need(witness['requested_source'] == dict(path=str(Path(source.path).absolute()),
            sha256=source.sha256, code_key=source.code_key), 'entry witness changed the native source identity')
    if specialized['program'] == calls['program']:
        native_entry_source = entry_source
    else:
        with BuildSession(Path(specialized['program']['uri']).parent) as native_session:
            native_entry_source = _entry_source(native_session, specialized['program'])
    _need(native_entry_source == entry_source,
        'native admission and summary do not share the same original entry/source version')

    bound = {}
    for match_index, match in enumerate(witness['matches']):
        matches = [(index, row) for index, row in enumerate(checked['functions'])
            if (row['body'], row['point'], row['definition']) == (match['original_body'],
                match['original_body_point'], match['original_function_definition'])]
        _need(len(matches) == 1, 'entry witness must bind exactly one original summary function body/point/definition')
        index, row = matches[0]
        local = row['local']
        _need(type(local) is dict and local['function_id'] == row['function_id']
            and local['body'] == row['body'] and local['body_point'] == row['point']
            and local['source_function'] == row['source_function']
            and local['definition'] == row['definition']
            and local['source'] == dict(path=str(Path(source.path).absolute()), sha256=source.sha256),
            'local summary identity or original source version differs from the native CodeID witness')
        key = row['function_id']
        if key not in bound:
            bound[key] = dict(function_id=key, summary_function=dict(artifact=summary_ref,
                locator='#/functions/'+str(index)), local_summary=dict(artifact=summary_ref,
                locator='#/functions/'+str(index)+'/local'), local_summary_sha256=semantic_sha256(local),
                original_body=row['body'], original_body_point=row['point'],
                original_function_definition=row['definition'], source_function=row['source_function'],
                witness_match_indices=[])
        bound[key]['witness_match_indices'].append(match_index)
    _need(bound, 'no exact original function summary matched the native source witness')
    inputs = {row.id:row for row in region.inputs}
    writes = [dict(action=row.id, target_input=row.target, target_path=list(inputs[row.target].path),
        source_ref=dict(artifact=specialized['source_region'], locator='#/actions/'+str(index)))
        for index, row in enumerate(region.actions) if row.kind == 'write']
    result = dict(schema=SCHEMA, schema_version=1, summary=summary_ref, entry_calls=calls_ref,
        specialization=specialization_ref, native_admission=admission,
        programs=dict(entry_summary=calls['program'], native_admission=specialized['program']),
        entry_source=entry_source, source_identity=witness['requested_source'], entry_witness=witness,
        candidate_id=candidate['candidate_id'], region=specialized['source_region'],
        region_sha256=stateful_digest(region), native_manifest=view['native_manifest'],
        native_identity=view['identity'], bindings=list(bound.values()), native_stateful_write_actions=writes,
        native_external_assumptions=dict(artifact=view['plan'], locator='#/external_assumptions'),
        native_guard_contract=dict(artifact=view['plan'], locator='#/guard_contract'),
        summary_interpretation='CONDITIONAL_MAY_DEPENDENCIES_AND_VISIBLE_EFFECTS_NOT_CLOSED_WORLD_EFFECTS',
        witness_conditions='PRESERVED_UNDISCHARGED_IN_ORIGINAL_MATCHES',
        missing_effects_interpretation='UNKNOWN_NOT_NONE_OR_PURE',
        source_identity_bound=True, purity_proven=False, effects_closed=False,
        source_transform_authorized=False, native_legality_derived_from_summary=False,
        runtime_state_identity_proven=False, future_dispatch_exhaustive=False,
        runtime_guards_discharged=False, whole_entry_legality_proven=False,
        native_loaded=False, compiled=False, target_program_executed=False,
        timing_qualified=False, deployment_authorized=False,
        producer_source=artifact_reference(__file__, 'scar.python-source'))
    for reference in (summary_ref, specialization_ref, calls_ref, candidate['native_artifact'],
            specialized['candidate'], specialized['program'], calls['program'], entry_source,
            specialized['source_region'], witness['source'], result['producer_source']):
        read_artifact(reference, json_document=False)
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def verify_stateful_source_summary_binding(binding, summary, specialization, *, entry_calls):
    """Recompute the complete conditional binding, including both cold parents."""
    retained = (read_artifact(_ref(binding, SCHEMA)) if isinstance(binding, (str, Path))
        or type(binding) is dict and 'uri' in binding else binding)
    expected = derive_stateful_source_summary_binding(summary, specialization, entry_calls=entry_calls)
    _need(expected == retained, 'stateful source summary binding differs from cold replay')
    return expected


__all__ = ['SCHEMA', 'derive_stateful_source_summary_binding', 'verify_stateful_source_summary_binding']
