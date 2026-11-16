"""Original-F4 loop guards for the distinct persistent-workspace trial profile.

An explicit call composition is not a proof about a Python caller loop. Native
admission is cold-replayed before sharing the original live guard machinery.
"""
from copy import deepcopy
from pathlib import Path

from scar.compiler.export_stateful import project_stateful_contract
from scar.compiler.stateful_model import stateful_digest
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256
from scar.runtime.stateful_loop_guard import _guarded_stateful_loop_core

SCHEMA = 'scar.stateful-workspace-loop-guard-view'
PROFILE = 'UNMEASURED_TRIAL_SOURCE_LOOP_FIXED_WORKSPACE_REUSE_ABI1'
_IDS = ('build_id', 'region_sha256', 'plan_sha256', 'contract_sha256')


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _source(value):
    return {key: getattr(value, key) for key in ('path', 'sha256', 'code_key', 'qualname')}


def guarded_workspace_loop(original, artifact, *, exclusive_inputs=False, trial=False):
    """Guard a proven original source loop, never an arbitrary composition.

    Default execution calls the original loop once. Explicit trial execution
    retains the shared preflight and committed-prefix/uncommitted-suffix policy.
    """
    from scar.codegen.stateful_workspace import WorkspaceRepeatArtifact, inspect_workspace_repeat
    from scar.compiler.stateful_loop_frontend import StatefulLoopProof
    if type(artifact) is not WorkspaceRepeatArtifact:
        raise TypeError('the distinct admitted workspace-repeat artifact is required')
    # Inspection owns one complete current preparation/source/bytes/identity
    # replay and performs no CDLL/provider construction. Never trust a cached
    # qualified tuple or a source_replayed boolean supplied by the artifact.
    metadata, qualified = inspect_workspace_repeat(artifact.directory)
    _need(type(qualified) is tuple and len(qualified) == 5,
          'complete workspace preparation replay inputs required')
    report, region, step_plan, workspace_plan, context = qualified
    _need(metadata == artifact.metadata
          and (metadata.get('schema'), metadata.get('schema_version'))
              == ('scar.native-stateful-workspace-repeat', 1),
          'workspace artifact object differs from its current admitted metadata')
    proof, loop_plan = context['loop_proof'], context['loop_plan']
    _need(report['source_kind'] == 'ORIGINAL_STATEFUL_LOOP'
          and context['source_kind'] == 'ORIGINAL_STATEFUL_LOOP'
          and type(proof) is StatefulLoopProof and type(loop_plan) is dict,
          'automatic workspace loop replacement requires complete original F4 loop and step proof')
    _need((workspace_plan.get('schema'), workspace_plan.get('schema_version'))
              == ('scar.stateful-workspace-reuse-plan', 1)
          and (loop_plan.get('schema'), loop_plan.get('schema_version')) == ('scar.stateful-loop-plan', 1),
          'distinct current workspace plan and verified original source-loop plan required')
    preparation = metadata['preparation']
    _need((preparation.get('schema'), report.get('schema'), report.get('schema_version'))
              == ('scar.workspace-preparation', 'scar.workspace-preparation', 1)
          and read_artifact(preparation) == report
          and report['source'] == context['source_reference']
          and read_artifact(report['workspace_plan']) == workspace_plan
          and metadata['workspace_plan_sha256'] == semantic_sha256(workspace_plan),
          'workspace guard must retain the exact current preparation and workspace plan')
    base_reference = context['base_native_reference']
    _need(type(base_reference) is dict and metadata['base_native_reference'] == base_reference
          and report['base_native'] == base_reference,
          'workspace loop has no exact original native base reference')
    retained_base = read_artifact(base_reference)
    base = artifact.base; base_report = base.report
    _need((base.directory/'artifact.json').resolve() == Path(base_reference['uri']).resolve()
          and metadata['base_ids'] == {key: retained_base[key] for key in _IDS}
          and {key: base_report[key] for key in _IDS} == metadata['base_ids'],
          'loaded workspace base identities differ from original native source admission')
    _need(stateful_digest(base.region) == stateful_digest(region) == stateful_digest(proof.step)
          and base.plan == step_plan == loop_plan['step_plan'],
          'workspace loop proof, shared step plan and loaded base source differ')
    count = workspace_plan['trip_count']
    _need(type(count) is int and 0 <= count < 2**64 and loop_plan['trip_count'] == count
          and report['trip_count'] == count
          and loop_plan['proof_id'] == proof.id,
          'workspace count or original source-loop proof/plan identity differs')
    step_contract = project_stateful_contract(base, selection=None, require_selection=False)
    _need(step_contract['selection'] is None and step_contract['source'] == _source(proof.step.source),
          'workspace loop must reuse the original unselected step guard contract')
    view = dict(schema=SCHEMA, schema_version=1, profile=PROFILE,
        source=_source(proof.source), step_source=_source(proof.step.source),
        parameters=list(proof.parameters), parameter_map=dict(proof.parameter_map),
        trip_count_parameter=proof.trip_count_parameter, trip_count=count,
        step_global_name=proof.step_global_name,
        call_arguments=dict(positional_parameters=list(proof.call_evidence['positional_parameters']),
            keyword_parameters=[list(pair) for pair in proof.call_evidence['keyword_parameters'].items()]),
        step_contract=step_contract, preparation=preparation, workspace_plan=report['workspace_plan'],
        original_loop_proof_id=proof.id, original_loop_plan_sha256=semantic_sha256(loop_plan),
        source_refs=encode(context['source_refs']), base_native_reference=base_reference,
        source_authority='COMPLETE_ORIGINAL_F4_LOOP_STEP_AND_CURRENT_WORKSPACE_PREPARATION_REPLAY',
        commit_policy='EACH_SUCCESSFUL_STEP_COMMITS_BEFORE_NEXT_STEP',
        failure_policy='RESUME_UNCOMMITTED_ORIGINAL_SUFFIX_WITH_DYNAMIC_CALLEE_LOOKUP',
        cost_status='UNKNOWN', trial_only=True, default_backend='NO_OP', deployment_authorized=False)
    guarded = _guarded_stateful_loop_core(original, artifact, view,
        exclusive_inputs=exclusive_inputs, trial=trial)
    guarded.scar_workspace_guard_view = deepcopy(view)
    return guarded


__all__ = ['SCHEMA', 'PROFILE', 'guarded_workspace_loop']
