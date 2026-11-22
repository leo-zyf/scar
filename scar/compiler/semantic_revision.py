"""Materialize an existing selected fusion as a consumable semantic revision.

The enclosing producer owns source admission and cost selection. This module
replays the selected OIR, exact fusion body and source-route projection; it
never promotes a conditional structural rewrite to deployment authorization.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.semantic-region-revision'
_PRODUCERS = ('compiler/semantic_revision.py', 'analysis/source_pipeline_graph.py',
    'optimization/semantic_rewrite.py', 'analysis/source_pipeline_graph.py',
    'optimization/pipeline_fusion.py', 'ir/v2/semantic.py', 'ir/v2/codec.py')


def _pins():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in _PRODUCERS}


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _selection(selected_region):
    from scar.analysis.source_pipeline_graph import read_fusion_region, ORIGINAL, FUSION
    from scar.ir.v2.optimization import PlanDisposition

    oir = read_fusion_region(selected_region)
    if len(oir.plans) != 1:
        raise ValueError('one actual OIR selection is required to materialize a revision')
    plan = next(iter(oir.plans.values()))
    alternative = oir.alternatives[plan.selected]
    name = FUSION if plan.disposition is PlanDisposition.REWRITE else ORIGINAL
    if (plan.selected.wire != selected_region['alternative_ids'].get(name)
            or alternative.region != plan.region):
        raise ValueError('selected alternative differs from the source-qualified region')
    return plan, alternative, name == FUSION


def _body(pipeline, selected_region, alternative):
    from scar.compiler.model import ExecutionPlan, document
    from scar.optimization.pipeline_fusion import fuse_pipeline_region, fusion_region_digest

    fused, certificate = fuse_pipeline_region(pipeline)
    expected = selected_region['strategy']['fusion_certificate']['region_sha256']
    if (not certificate['supported'] or fused is None
            or fusion_region_digest(fused) != expected
            or len(alternative.instructions) != 1
            or alternative.instructions[0].replacement['fusion_region_sha256'] != expected):
        raise ValueError('selected OIR replacement differs from the actual fused expression body')
    return json.loads(json.dumps(document(ExecutionPlan(fused, passes=('guarded_full_pipeline_fusion',),
        proof=dict(fusion_certificate=certificate, selected=False,
            runtime_guards_discharged=False, execution_authorized=False))),
        sort_keys=True, allow_nan=False))


def _successor(pipeline, selected_region, source_values, before_ref, body_ref):
    from scar.analysis.source_pipeline_graph import project_source_pipeline_graph
    from scar.ir.v2.ids import OperationDefinitionID
    from scar.ir.v2.semantic import OperationDefinition, OperationKind
    from scar.optimization.semantic_rewrite import rewrite_closed_region, analyze_static_region

    before, projection = project_source_pipeline_graph(pipeline, selected_region, source_values)
    if read_artifact(before_ref) != before.to_dict():
        raise ValueError('before graph differs from admitted original source routes')
    root = OperationDefinitionID(projection['root']['value'])
    selection, alternative, rewrite = _selection(selected_region)
    if rewrite:
        if body_ref is None or read_artifact(body_ref) != _body(pipeline, selected_region, alternative):
            raise ValueError('body expansion is not the exact selected source fusion')
        parent = before.definitions[root]
        replacement = OperationDefinition(
            OperationDefinitionID(alternative.delta.added_operations[0]), OperationKind.REGION,
            'Source-qualified fused region', code_id=parent.code_id,
            source_file=parent.source_file, parent_id=root,
            contract=parent.contract, effect_summary=parent.effect_summary)
        after, mapping = rewrite_closed_region(before, root, replacement, alternative,
            original_graph=before_ref, body_expansion=body_ref)
    else:
        if body_ref is not None:
            raise ValueError('KEEP cannot publish a fused expansion')
        after = before
        mapping = dict(source_graph_unchanged=True, execution_authorized=False)
    old_analysis = analyze_static_region(before, root)
    new_analysis = analyze_static_region(after, root)
    from scar.optimization.macro_pass import FUSION_PASS
    return before, after, dict(schema=SCHEMA, schema_version=1,
        decision='REWRITE' if rewrite else 'KEEP', root=root.as_dict(),
        selection=selection.as_dict(), delta=alternative.delta.as_dict(),
        selected_region_sha256=semantic_sha256(selected_region),
        projection=projection, before_revision=old_analysis['revision'],
        after_revision=new_analysis['revision'], mapping=mapping,
        invalidated_analyses=list(FUSION_PASS.invalidates) if rewrite else [],
        recomputed_analyses=['SOURCE_USE_DEF', 'SOURCE_ORDER', 'STATIC_LAST_USE'],
        outstanding_analyses=(['PHYSICAL_LIFETIME', 'IMPLEMENTATION_DEMAND', 'COST',
            'RESOURCE_SCHEDULE', 'NATIVE_ARTIFACT_BINDING'] if rewrite else []),
        new_full_region_graph_emitted=True, whole_program_graph_rewritten=False,
        original_source_and_execution_evidence_unchanged=True,
        runtime_guards_discharged=False, opaque_effects_preserved=True,
        execution_authorized=False, performance_measured=False, producer_pins=_pins()), old_analysis, new_analysis


def publish_fusion_revision(pipeline, selected_region, source_values, out):
    from scar.analysis.source_pipeline_graph import project_source_pipeline_graph

    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh semantic revision directory required')
    before, _ = project_source_pipeline_graph(pipeline, selected_region, source_values)
    _, alternative, rewrite = _selection(selected_region)
    body = _body(pipeline, selected_region, alternative) if rewrite else None
    output.mkdir(parents=True)
    before_ref = _write(output/'before.json', before.to_dict())
    body_ref = None if body is None else _write(output/'body.json', body)
    _, after, receipt, old_analysis, new_analysis = _successor(
        pipeline, selected_region, source_values, before_ref, body_ref)
    files = dict(before=before_ref, body=body_ref,
        after=_write(output/'after.json', after.to_dict()),
        before_analysis=_write(output/'before-analysis.json', old_analysis),
        after_analysis=_write(output/'after-analysis.json', new_analysis))
    receipt['files'] = files
    return _write(output/'revision.json', receipt)


def check_fusion_revision(reference, pipeline, selected_region, source_values):
    saved = read_artifact(reference)
    if (saved.get('schema'), saved.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('versioned semantic region revision required')
    files = saved['files']
    before, after, expected, old_analysis, new_analysis = _successor(
        pipeline, selected_region, source_values, files['before'], files['body'])
    for name, value in (('before', before.to_dict()), ('after', after.to_dict()),
            ('before_analysis', old_analysis), ('after_analysis', new_analysis)):
        if read_artifact(files[name]) != value:
            raise ValueError('semantic revision graph or analysis differs from source replay: '+name)
    expected['files'] = deepcopy(files)
    if expected != saved:
        raise ValueError('semantic revision selection, mapping or producer differs from replay')
    return saved
