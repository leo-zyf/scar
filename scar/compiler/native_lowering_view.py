"""Read a retained serial stateful lowering without loading or executing it.

This source map is not F4 admission, binary/source equivalence, a timing
qualification, or permission to install an implementation.
"""
from pathlib import Path

from scar.ir.execution_recipe import _reference, artifact_reference, read_artifact


SCHEMA = 'scar.stateful-native-lowering-view'
_ROOT = Path(__file__).resolve().parents[1]
_FILE_SCHEMAS = dict(region='scar.stateful-array-region', plan='scar.stateful-stage-plan',
    source='scar.cpp-source', library='scar.native-library',
    **{'stateful_abi_v1.h': 'scar.cpp-header', 'scar_abi_v1.h': 'scar.cpp-header'})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(value, schema):
    reference = (artifact_reference(value, schema) if isinstance(value, (str, Path))
        else _reference(value))
    _need(reference['schema'] == schema, 'lowering artifact schema differs: '+schema)
    return reference


def _body_spans(source_lines, body):
    if not body:
        return 'EMPTY_BODY_NO_EMITTED_SPAN', []
    # The existing capture records the renderer's body without this exact
    # outer indentation. Do not strip whitespace or pick an occurrence by order.
    lines = ['            '+line for line in body.split('\n')]
    spans = [dict(start_line=index+1, end_line=index+len(lines))
        for index in range(len(source_lines)-len(lines)+1)
        if source_lines[index:index+len(lines)] == lines]
    _need(spans, 'captured nonempty stage body not found in actual generated C++')
    return ('UNIQUE_BODY_MATCH' if len(spans) == 1 else 'AMBIGUOUS_BODY_MATCHES'), spans


def derive_stateful_lowering_view(native_manifest, *, stage_capture=None):
    """Return a detached JSON source map from an artifact.json path/ArtifactRef.

    Only the existing serial renderer is supported. An optional retained
    capture path/ArtifactRef must equal a fresh complete capture, including its
    bodies and provenance. No compiler, target import or native load occurs.
    """
    import json
    from scar.codegen import stateful
    from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
    from scar.compiler.model import CpuPlacement
    from scar.compiler.stateful_model import stateful_from_dict, stateful_digest

    manifest_ref = _ref(native_manifest, stateful.SCHEMA)
    manifest = read_artifact(manifest_ref)
    _need((manifest.get('schema_version'), manifest.get('stateful_abi_version')) == (1, 1),
        'stateful native artifact and ABI version one required')
    _need(type(manifest.get('files')) is dict and set(manifest['files']) == set(_FILE_SCHEMAS)
        and type(manifest.get('sha256')) is dict and set(manifest['sha256']) == set(_FILE_SCHEMAS),
        'complete original stateful artifact file/hash roles required')
    directory = Path(manifest_ref['uri']).parent
    references, contents = {}, {}
    for role, schema in _FILE_SCHEMAS.items():
        name = manifest['files'][role]
        _need(type(name) is str and Path(name).name == name and name not in ('.', '..'),
            'stateful artifact member must be a local basename')
        reference = artifact_reference(directory/name, schema)
        _need(reference['sha256'] == manifest['sha256'][role], 'stateful artifact checksum differs: '+role)
        references[role] = reference
        contents[role] = read_artifact(reference, json_document=role in ('region', 'plan'))

    region = stateful_from_dict(contents['region'])
    plan = contents['plan']
    _need(plan.get('schema_version') == 1 and plan.get('region_id') == region.id
        and plan.get('region_sha256') == stateful_digest(region), 'actual plan/region identity differs')
    _need(type(plan.get('cpu_threads')) is int and plan['cpu_threads'] == 1
        and plan.get('status') == 'READY', 'lowering view supports ready original serial plans only')
    binding = dict(plan['resource_binding'])
    binding['cpu_ids'] = tuple(binding['cpu_ids'])
    binding['evidence'] = tuple(binding.get('evidence', ()))
    binding = CpuPlacement(**binding)
    _need(len(binding.cpu_ids) == 1 and type(binding.cpu_ids[0]) is int
        and 0 <= binding.cpu_ids[0] < 2**20, 'one bounded actual plan CPU required')
    for key in ('cpu_threads', 'external_array_bytes', 'workspace_payload_bytes',
            'total_region_array_payload_bytes', 'memory_budget_bytes'):
        _need(manifest[key] == plan[key], 'native metadata differs from actual plan: '+key)
    _need(manifest['cpu_ids'] == list(binding.cpu_ids)
        and manifest['external_assumptions'] == plan['external_assumptions'],
        'native metadata CPU/assumptions differ from actual plan')
    core = dict(region_sha256=stateful_digest(region), plan_sha256=stateful._digest(plan),
        provider_runtime=manifest['provider_runtime'])
    _need(all(manifest[key] == value for key, value in core.items())
        and manifest['contract_sha256'] == stateful._digest(core), 'native core identity differs')
    compiler = manifest['compiler']
    _need(type(compiler) is dict and set(compiler) == {'argv', 'identity', 'options'}
        and all(type(compiler[name]) is list and compiler[name]
            and all(type(arg) is str for arg in compiler[name]) for name in ('argv', 'options'))
        and type(compiler['identity']) is str and compiler['identity'], 'recorded compiler context required')
    # The historical producer fingerprint is part of the retained build ID;
    # comparing it with today's full-file fingerprint would revoke old evidence.
    build_id = stateful._digest(dict(source=references['source']['sha256'],
        headers={name:references[name]['sha256'] for name in ('stateful_abi_v1.h', 'scar_abi_v1.h')},
        compiler=compiler['argv'], identity=compiler['identity'], options=compiler['options'],
        implementation_sha256=manifest['implementation_sha256'], contract=core))
    _need(build_id == manifest['build_id'], 'retained stateful build identity differs')

    capture = capture_stateful_stage_sources(region, plan, binding, core)
    _need(capture['full_source_sha256'] == references['source']['sha256'],
        'actual generated C++ differs from original stage renderer replay')
    capture_ref = None
    if stage_capture is not None:
        capture_ref = _ref(stage_capture, 'scar.stateful-stage-source-capture')
        _need(stateful._digest(read_artifact(capture_ref)) == stateful._digest(capture),
            'retained stage capture bodies/provenance differ from fresh renderer capture')
    records = {row['index']:row for row in capture['stages']}
    source_lines = contents['source'].decode('utf-8').splitlines()
    stages = []
    for stage in plan['stages']:
        if stage['kind'] == 'COMMIT':
            continue
        record = records[stage['index']]
        status, spans = _body_spans(source_lines, record['body'])
        stages.append(dict(index=stage['index'], id=stage['id'], kind=stage['kind'],
            source_action_ids=stage['source_action_ids'], expr_ids=stage['expr_ids'],
            body_sha256=record['body_sha256'], mapping_status=status, source_line_spans=spans,
            spans_are_candidates=len(spans) > 1, plan_pointer='#/stages/'+str(stage['index'])))
    wrapper_files = ('compiler/stateful_runtime.py', 'runtime/stateful_guard.py',
        'codegen/stateful.py', 'codegen/cpp.py')
    result = dict(schema=SCHEMA, schema_version=1, native_manifest=manifest_ref,
        source=references['source'], region=references['region'], plan=references['plan'],
        library=references['library'],
        headers={name:references[name] for name in ('stateful_abi_v1.h', 'scar_abi_v1.h')}, compiler=compiler,
        identity={key:manifest[key] for key in ('build_id', 'region_sha256', 'plan_sha256',
            'contract_sha256', 'implementation_sha256')}, region_id=region.id,
        actual_resource_binding=plan['resource_binding'],
        wrapper=dict(family='FULL_ARTIFACT_COMPATIBILITY_ENTRY',
            call_chain=['scar.runtime.stateful_guard.guarded_stateful_function',
                'scar.compiler.stateful_runtime._ArtifactAdapter.run',
                'scar.codegen.stateful.StatefulArtifact.run'],
            source_refs=[artifact_reference(_ROOT/name, 'scar.python-source') for name in wrapper_files],
            entry_selection_proven=False, compact_runtime_equivalence_claimed=False),
        capture=dict(artifact=capture_ref, full_source_sha256=capture['full_source_sha256'],
            provenance=capture['provenance']), stages=stages,
        line_numbering='ONE_BASED_INCLUSIVE_BODY_LINES_IN_ACTUAL_GENERATED_CPP',
        ambiguous_stage_indices=[row['index'] for row in stages if row['spans_are_candidates']],
        omitted_stage_kinds=['COMMIT'], original_external_assumptions=plan['external_assumptions'],
        original_guard_contract=plan['guard_contract'],
        producer_source=artifact_reference(__file__, 'scar.python-source'),
        source_renderer_replayed=True, retained_member_bytes_verified=True,
        source_F4_admitted=False, plan_legality_readmitted=False, binary_source_equivalence_proven=False,
        target_program_executed=False, native_loaded=False, compiled=False,
        performance_measured=False, timing_qualified=False, deployment_authorized=False)
    for reference in (manifest_ref, *references.values(), *result['wrapper']['source_refs'],
            result['producer_source'], *((capture_ref,) if capture_ref is not None else ())):
        read_artifact(reference, json_document=False)
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


__all__ = ['SCHEMA', 'derive_stateful_lowering_view']
