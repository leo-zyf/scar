"""Selected stateful lowering."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from scar.ir.execution_recipe import artifact_reference, read_artifact, recipe_id, _reference
from scar.optimization.resource_solver import semantic_sha256


# Replay stateful CPU choices and lower the selected region to C++20.
def lower_selected_stateful(region,plan,out,*,backend_descriptor_sha256,plan_only=False):
    from scar.optimization.stateful_recipe import make_stateful_recipe,verify_stateful_recipe
    if type(plan_only)is not bool:raise TypeError('exact plan-only boolean required')
    out=Path(out).resolve()
    recipe=verify_stateful_recipe(make_stateful_recipe(region,plan,out=out/'recipe'))
    capability=recipe['implementation']['backend_capability_record'];graph=recipe['task_graph']
    if (semantic_sha256(capability)!=backend_descriptor_sha256
            or graph['schema_version']!=2 or graph['executable']is not True):
        raise ValueError('source plan/backend differs from selected executable model')
    result=dict(model_selected_native=True,execution='EXISTING_TRANSACTIONAL_CPU_STATEFUL_RECIPE',
        status='MODEL_SELECTED_PLAN_ONLY',recipe_id=recipe_id(recipe),
        recipe=artifact_reference(out/'recipe/recipe.json',recipe['schema']),
        task_graph_sha256=semantic_sha256(graph),selected_binding=plan['resource_binding'],
        payload={key:plan[key]for key in ('external_array_bytes','workspace_payload_bytes',
            'total_region_array_payload_bytes','memory_budget_bytes','budget_scope')},
        compiled_variant_count=0,native_directory=None,artifact=None,lowering_receipt=None,
        deployment_authorized=False,selected_for_use=False)
    if not plan_only:
        from scar.codegen.recipe import build_recipe
        lowered=build_recipe(recipe,out/'lowered');report=lowered.report
        if (report['recipe_id']!=result['recipe_id'] or report['stateful_abi_version']!=1
                or report['backend_capability_record']!=capability):
            raise ValueError('compiled artifact differs from source recipe/ABI/backend')
        result.update(status='COMPILED_UNMEASURED',compiled_variant_count=1,
            native_directory=str(lowered.native_directory),artifact=report,
            lowering_receipt=artifact_reference(out/'lowered/lowering-receipt.json','scar.execution-recipe-lowering'),
            stateful_abi_version=1,operator_abi_version=1,cxx_standard=20)
    return result



def _require(condition, message):
    if not condition: raise ValueError(message)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def lower_stateful_choices(choice_path, out, *, plan_only=False):
    """Verify the full recorded domain; plan or compile its selected CPU only."""
    from scar.optimization.stateful_cpu_model import verify_stateful_cpu_choices, selected_stateful_plan

    _require(type(plan_only) is bool, 'plan_only must be an exact boolean')
    choice_path = Path(choice_path).absolute()
    if choice_path.is_dir(): choice_path /= 'choice.json'
    directory = Path(out).resolve()
    _require(not directory.exists(), 'fresh stateful choice lowering output required')
    choice_ref = artifact_reference(choice_path, 'scar.stateful-cpu-choice')
    recorded = read_artifact(choice_ref)
    checked = verify_stateful_cpu_choices(choice_path)
    _require(semantic_sha256(recorded) == semantic_sha256(checked),
             'stateful choice differs from complete source/hardware/cost replay')
    domain = read_artifact(checked['domain'])
    selection = checked['selected_id']
    result = dict(schema='scar.lowered-stateful-cpu-choice', schema_version=1,
        upstream_choice=choice_ref, choice_semantic_sha256=semantic_sha256(checked),
        domain=checked['domain'], domain_id=domain['domain_id'],
        selected_id=selection, complete_cost_domain=checked['complete_cost_domain'],
        upstream_decision=checked['decision'], plan_only=plan_only,
        choice_scope=checked['scope'], comparison_model=checked['comparison_model'],
        resource_policy=checked['resource_policy'],
        original_no_op_available=checked['original_no_op_available'],
        model_selected_native=False, compiled_variant_count=0,
        recipe=None, recipe_id=None, native_directory=None, artifact=None,
        lowering_receipt=None, actual_target_duration_seconds=None,
        target_program_executed=False, target_variants_ranked_by_measurement=False,
        selected_for_use=False, deployment_authorized=False,
        physical_optimality_proven=False, whole_program_makespan_optimized=False)
    if selection in (None, 'NO_OP'):
        _require(checked['decision'] == 'NO_OP', 'original execution decision differs')
        result.update(status='ORIGINAL_EXECUTION_RETAINED', execution='UNCHANGED_ORIGINAL_PYTHON',
            reason='MODEL_SELECTED_ORIGINAL' if selection == 'NO_OP' else 'COST_OR_LEGALITY_DOMAIN_UNRESOLVED',
            unresolved=checked['solver'].get('unresolved', []),
            frontier=checked['solver'].get('frontier', []))
        directory.mkdir(parents=True)
        _write(directory / 'lowering.json', result)
        return result

    solver = checked['solver']
    _require(checked['complete_cost_domain'] is True
        and checked['decision'] == 'SELECT_NATIVE_IMPLEMENTATION_FOR_LOWERING'
        and solver.get('status') == 'OPTIMAL_WITHIN_FINITE_MODEL'
        and not solver.get('frontier') and not solver.get('unresolved')
        and solver.get('search_budget_exhausted') is False and solver.get('gap_ticks') == 0
        and checked['physical_optimality_proven'] is False,
        'complete finite-domain model selection required before native lowering')
    variants = checked['variants']
    index = next((i for i, row in enumerate(variants) if row['id'] == selection), None)
    _require(index is not None, 'selected implementation is absent from the retained domain')
    variant = variants[index]
    _require(variant['cost_status'] == 'KNOWN' and not variant['frontier'],
             'selected implementation requires an applicable complete-call cost')
    region, plan = selected_stateful_plan(checked)
    context = variant['context']
    binding = {key: value for key, value in plan['resource_binding'].items() if key != 'evidence'}
    _require(binding == context['cpu_binding']
        and plan['input_shapes'] == context['input_shapes']
        and plan['memory_budget_bytes'] == context['memory_budget_bytes']
        and plan['budget_scope'] == context['budget_scope'],
        'selected source/CPU/shape/payload differs from its modeled call context')

    # The standard producer replays sources, effects, slots, target receipts,
    # provider and compiler context. Nonselected plans are never persisted here.
    result.update(lower_selected_stateful(region,plan,directory,
        backend_descriptor_sha256=context['backend']['descriptor_sha256'],plan_only=plan_only))
    result.update(solver_scope=dict(status=solver['status'], optimality=solver.get('optimality'),
            gap_ticks=solver['gap_ticks'], optimality_scope=domain['enumeration']['optimality_scope']),
        selected_model=dict(duration_ticks=variant['profile']['estimate']['duration_ticks'],
            tick_seconds=variant['profile']['estimate']['tick_seconds'],
            origin=variant['profile']['derivation']['origin'],
            assumptions=variant['profile']['derivation']['assumptions'],
            evidence=dict(artifact=choice_ref, locator=f'#/variants/{index}/profile'),
            uncertainty_scope='RETAINED_MODEL_DERIVATION_NOT_TARGET_MEASUREMENT_OR_PHYSICAL_PROOF'))
    _require(sha256(choice_path.read_bytes()).hexdigest() == choice_ref['sha256'],
             'upstream choice changed during lowering')
    _write(directory / 'lowering.json', result)
    return result


def lower_stateful_bounds(plan_path,out,*,plan_only=False):
    from scar.compiler.plan_stateful_bounds import verify_plan_stateful_bounds
    from scar.optimization.stateful_cpu_model import _domain, _cost, stateful_cpu_cost_context, _family
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region

    _require(type(plan_only)is bool,'exact plan-only boolean required')
    plan_path=Path(plan_path).resolve()
    if plan_path.is_dir():plan_path/= 'result.json'
    reference=artifact_reference(plan_path,'scar.stateful-interval-choice-plan')
    checked=verify_plan_stateful_bounds(plan_path);domain=_domain(checked['original_domain']['uri'])
    out=Path(out).resolve();_require(not out.exists(),'fresh interval lowering output required')
    selected=checked['selected_candidate_id']
    result=dict(schema='scar.lowered-stateful-interval-choice',schema_version=1,
        upstream_plan=reference,domain=checked['original_domain'],selected_id=selected,
        model_selected_native=False,compiled_variant_count=0,native_directory=None,recipe=None,recipe_id=None,
        plan_only=plan_only,selection_status=checked['status'],remaining_frontier=checked['remaining_frontier'],
        selected_for_use=False,deployment_authorized=False,target_program_executed=False,
        target_variants_ranked_by_measurement=False,physical_optimality_proven=False,
        whole_program_makespan_optimized=False,unknown_duration_imputed=False)
    if selected in(None,'NO_OP'):
        _require(checked['selection_decision']=='NO_OP','unselected interval choice cannot enable native code')
        result.update(status='ORIGINAL_EXECUTION_RETAINED',execution='UNCHANGED_ORIGINAL_PYTHON',
            reason='UNRESOLVED_COST_INTERVALS' if selected is None else 'MODEL_SELECTED_ORIGINAL')
    else:
        choice=read_artifact(checked['interval_choice'])
        _require(checked['status']==choice['status']=='FINITE_MODEL_OPTIMAL'
            and not choice['remaining_frontier'] and checked['selected_option_evidence'] is not None
            and checked['selected_cost_profile'] is not None,'finite model selection with a complete selected profile required')
        candidate=next(r for r in domain['candidates'] if r['id']==selected)
        _require(candidate['legal_under_guards'] and not candidate['legality_frontier'],'original selected capability declined')
        context=stateful_cpu_cost_context(domain,selected);profile_ref=checked['selected_cost_profile']
        profile=read_artifact(profile_ref);cost,issues,_=_cost(profile,context)
        option=next(r for r in choice['options'] if r['id']==selected)
        _require(cost is not None and not issues and option['interval']==dict(lower=cost['duration_ticks'],upper=cost['duration_ticks'])
            and cost['tick_seconds']==checked['tick_seconds'],'complete selected cost differs from interval decision')
        previous=read_artifact(checked['original_admission'])
        replayed=[r for r in previous['independently_replayed_new_profiles'] if r['candidate_id']==selected]
        if replayed:
            _require(len(replayed)==1 and replayed[0]['profile']==profile_ref,'selected calibration context differs')
            from scar.optimization.stateful_batch_cost import verify_batch_cpu_cost
            cost_replay=verify_batch_cpu_cost(domain,selected,replayed[0]['worker_result']['uri'])
            _require(cost_replay['profile']==profile_ref,'selected raw source qualification differs')
        else:
            from scar.optimization.stateful_cost_rebase import _model_dependencies
            cost_replay=_model_dependencies(profile,domain,checked['original_domain'],context,selected)
        binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
        region=stateful_from_dict(read_artifact(domain['region']))
        numeric=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],
            cpu_threads=candidate['worker_count'],memory_budget_bytes=domain['request']['memory_budget_bytes'],
            resource_binding=CpuPlacement(**binding))
        _require(semantic_sha256(_family(numeric))==domain['semantic_family_sha256']
            and numeric['input_shapes']==context['input_shapes'] and numeric['guard_contract']==context['guards'],
            'selected numeric graph/shape/effect guards differ from original model')
        result.update(lower_selected_stateful(region,numeric,out,
            backend_descriptor_sha256=context['backend']['descriptor_sha256'],plan_only=plan_only))
        result.update(selected_cost_profile=profile_ref,selected_cost_replay=cost_replay,
            selected_option_evidence=checked['selected_option_evidence'])
    read_artifact(reference);out.mkdir(parents=True,exist_ok=True)
    _write(out/'lowering.json', result)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--choices')
    inputs.add_argument('--plan')
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    if args.choices:
        report = lower_stateful_choices(args.choices, args.out, plan_only=args.plan_only)
    else:
        report = lower_stateful_bounds(args.plan, args.out, plan_only=args.plan_only)
    print(json.dumps({key: report[key] for key in ('status', 'selected_id', 'recipe_id',
        'native_directory', 'compiled_variant_count', 'deployment_authorized')}))
    return 0


# Read a retained serial stateful lowering without loading or executing it.
#
# This source map is not F4 admission, binary/source equivalence, a timing
# qualification, or permission to install an implementation.
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


__all__ = [
    'lower_selected_stateful',
    'lower_stateful_choices',
    'lower_stateful_bounds',
    'SCHEMA',
    'derive_stateful_lowering_view',
    'main',
]


if __name__ == '__main__':
    raise SystemExit(main())
