"""Trial-only control simplification of certified raw-copy windows.

No mathematical body, calling thread, placement, allocation, or boundary is
changed. Loading replays the original preparation and source transformation;
the legacy loader deliberately does not admit this distinct execution plan.
"""
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess

from scar.codegen import cpp, stateful as base_stateful
from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
from scar.compiler.stateful_model import document, stateful_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-stateful-window-order-trial'
CONTROL_SCHEMA = 'scar.stateful-window-order-control'
CERTIFIED_SCHEMA = 'scar.native-certified-window-fusion'
CERTIFIED_CONTROL_SCHEMA = 'scar.certified-window-fusion-control'
_HERE = Path(__file__).resolve().parent
_ORDER = 'PERMUTE_INDEPENDENT_RAW_COPY_TASKS'
_FUSION = 'FUSE_IDENTICAL_INDEPENDENT_RAW_COPY_LOOPS'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(raw):
    return sha256(raw).hexdigest()


def _digest(value):
    return _hash(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _implementation():
    return _digest(dict(base_stateful=base_stateful.implementation_sha256(),
        files={name: _hash((_HERE/name).read_bytes()) for name in
               ('window_order.py', 'stateful_stage_capture.py')}))


def _reference(preparation):
    path = Path(preparation).resolve()
    if path.is_dir():
        path /= 'result.json'
    raw = json.loads(path.read_bytes())
    _need(type(raw) is dict and type(raw.get('schema')) is str,
          'original closed-window preparation receipt required')
    return artifact_reference(path, raw['schema'])


def _derive(preparation, window_id, task_order, *, kind=_ORDER, _qualified=None):
    # Import only after the public preparation is available; no duplicate
    # frontend/cost proof or unverified user selection is used as authority.
    from scar.compiler.plan_closed_windows import replay_closed_window_plans
    reference = _reference(preparation)
    if _qualified is None:
        report, projection, plans, context = replay_closed_window_plans(reference['uri'])
    else:
        _need(type(_qualified) is tuple and len(_qualified) == 4,
              'private qualified parent must be the original complete replay tuple')
        report, projection, plans, context = _qualified
    _need(read_artifact(reference) == report,
          'closed-window preparation changed during complete replay')
    _need(type(projection) is dict and projection.get('schema') == 'scar.closed-window-projection',
          'original source-closed window projection required')
    _need(kind in (_ORDER, _FUSION) and type(window_id) is str,
          'supported explicit raw-copy control and source window required')
    matches = [row for row in projection['windows'] if row['id'] == window_id]
    _need(len(matches) == 1, 'one source-proven original window required')
    window = matches[0]; original_order = window['task_ids']
    if kind == _FUSION:
        planned = [row for row in plans['windows'] if row['window_id'] == window_id]
        _need(len(planned) == 1 and any(candidate['kind'] == _FUSION
              and candidate['task_ids'] == original_order
              for candidate in planned[0]['candidate_transforms']),
              'fusion trial requires the automatically derived graph proposal')
    _need(len(original_order) >= 2, 'at least two original raw-copy tasks required')
    if kind == _FUSION and task_order is None:
        task_order = original_order
    _need(type(task_order) in (list, tuple) and all(type(name) is str for name in task_order),
          'complete original-window task order required')
    task_order = list(task_order)
    _need(len(task_order) == len(original_order) and len(set(task_order)) == len(task_order)
          and set(task_order) == set(original_order)
          and (task_order != original_order if kind == _ORDER else task_order == original_order),
          'order trial needs a nontrivial permutation; fusion preserves original statement order')
    new_index = {name: index for index, name in enumerate(task_order)}
    pairs = {(row['before'], row['after']): row for row in window['pairs']}
    checked_pairs = []
    for i, before in enumerate(original_order):
        for after in original_order[i+1:]:
            if kind == _ORDER and new_index[before] < new_index[after]:
                continue
            pair = pairs.get((before, after))
            _need(pair is not None and pair['status'] == 'PAIRWISE_DATA_INDEPENDENT',
                  'every transformed pair needs original source/data/effect independence')
            proof = pair['proof']
            _need(not proof['storage_region_conflicts'] and not proof['producer_consumer']
                  and proof['world_or_token_conflict'] is False and not proof['non_sequence_dependencies']
                  and not proof['remaining_obligations'] and proof['original_edges_removed'] is False,
                  'transformed pair retains unresolved data/effect conflicts')
            checked_pairs.append(dict(before=before, after=after, proof=deepcopy(proof)))
    _need(checked_pairs, 'independent source/data/effect pair proof required')
    recipe = context['recipe']; schedule = context['schedule']
    region, plan, binding = base_stateful._admit(read_artifact(recipe['artifacts']['region']),
        read_artifact(recipe['artifacts']['stage_plan']))
    _need(plan['cpu_threads'] == 1 and len(binding.cpu_ids) == 1,
          'trial supports only the existing same-caller single-CPU host')
    tasks = {row.id: row for row in schedule.tasks}
    stages = {row['id']: row for row in plan['stages']}
    for name in original_order:
        task = tasks[name]; stage = stages[name]
        _need(task.kind.value == 'SNAPSHOT' and task.capability == 'CPU_HOST_SNAPSHOT'
              and stage['kind'] == 'SNAPSHOT_ROOT' and not task.world_before and not task.world_after,
              'only source-proven private raw-copy tasks can be reordered by this trial')
    indexes = [stages[name]['index'] for name in original_order]
    _need(indexes == list(range(indexes[0], indexes[0]+len(indexes))),
          'original window must contain contiguous complete source-stage bodies')
    runtime, providers, owners, power = base_stateful._providers(region)
    _need(runtime == recipe['implementation']['backend_capability_record']['provider_runtime'],
          'original source recipe and live provider binary differ')
    base_contract = dict(region_sha256=stateful_digest(region), plan_sha256=_digest(plan),
                         provider_runtime=runtime)
    capture = capture_stateful_stage_sources(region, plan, binding, base_contract)
    captured = {row['index']: row for row in capture['stages']}
    bodies = {name: captured[stages[name]['index']]['body'] for name in original_order}
    # Exactly the indentation applied by original stateful._source to every
    # stage_text line. Neither expression generation nor scalar AST is changed.
    def fragment(order):
        return '\n'.join('\n'.join(' '*12+line for line in bodies[name].split('\n')) for name in order)
    original_source = base_stateful._source(region, plan, binding, base_contract)
    _need(_hash(original_source.encode()) == capture['full_source_sha256'],
          'captured source and original host differ')
    before, after = fragment(original_order), fragment(task_order)
    fusion = None
    if kind == _FUSION:
        members = set(original_order)
        materials = {row.id: row for row in schedule.materializations}
        _need(all(materials[name].ready_task not in members
                  for task in original_order for name in tasks[task].reads),
              'fusion cannot delay any consumer inside its original window')
        skeleton = None; inners = []; body_rows = []
        for name in original_order:
            lines = bodies[name].split('\n')
            _need(len(lines) == 5 and lines[-2:] == ['    }', '}'],
                  'raw-copy fusion requires an exact five-line two-loop body')
            row_match = re.fullmatch(r'for \(int64_t row = 0; row < ([0-9]+); \+\+row\) \{', lines[0])
            col_match = re.fullmatch(r'    for \(int64_t col = 0; col < ([0-9]+); \+\+col\) \{', lines[1])
            _need(row_match is not None and col_match is not None
                  and [int(row_match[1]), int(col_match[1])] == list(stages[name]['shape']),
                  'raw-copy fusion needs the exact original concrete iteration domain')
            _need(re.fullmatch(r'        slots\[[0-9]+\]\.get\(\)\[row \* [0-9]+ \+ col\] = '
                  r'static_cast<const double \*>\(arrays\[[0-9]+\]\.data\)\[row \* [0-9]+ \+ col\];', lines[2]) is not None,
                  'raw-copy fusion accepts only the original single load-store assignment')
            candidate = (lines[:2], lines[-2:])
            _need(skeleton is None or candidate == skeleton,
                  'raw-copy loop shapes and control skeletons must match verbatim')
            skeleton = candidate; inners.append(lines[2])
            body_rows.append(dict(task=name, body_sha256=_hash(bodies[name].encode()),
                inner_sha256=_hash(lines[2].encode())))
        fused = '\n'.join([*skeleton[0], *inners, *skeleton[1]])
        after = '\n'.join(' '*12+line for line in fused.split('\n'))
        fusion = dict(domain=list(stages[original_order[0]]['shape']),
            loop_nests_before=len(original_order), loop_nests_after=1,
            original_bodies=body_rows, fused_body_sha256=_hash(fused.encode()),
            every_original_inner_statement_copied_verbatim=True,
            inner_statement_order=list(original_order), window_internal_consumers=False,
            readiness_policy='ALL_WINDOW_OUTPUTS_READY_ONLY_AFTER_COMPLETE_FUSED_LOOP',
            original_component_cost_sum_reused=False, fused_component_cost_status='UNMEASURED')
    _need(original_source.count(before) == 1, 'original continuous raw-copy window source is not unique')
    start = original_source.index(before); end = start+len(before)
    control = dict(schema=CONTROL_SCHEMA, schema_version=1, preparation=reference,
        kind=kind, window_id=window_id, original_task_order=original_order, task_order=task_order,
        inversions=checked_pairs if kind == _ORDER else [],
        all_fusion_pair_proofs=checked_pairs if kind == _FUSION else [],
        fusion=fusion, original_window_sha256=_digest(window),
        policy='SAME_CALLER_SAME_CPU_SERIAL_PRIVATE_RAW_COPY_CONTROL_ONLY',
        source_scope='UNMEASURED_TRIAL_NOT_COMPLETE_BOUNDARY_LOWERING_AUTHORITY',
        no_remaining_data_effect_conflicts=True,
        worker_threads_created=0, cpu_binding_changed=False, boundary_order_changed=False,
        deployment_authorized=False, cost_status='UNMEASURED')
    execution_plan = dict(schema='scar.stateful-window-order-execution-plan', schema_version=1,
        base_plan_sha256=_digest(plan), control_sha256=_digest(control))
    execution_sha = _digest(execution_plan)
    contract = {**base_contract, 'plan_sha256': execution_sha}
    reordered = original_source[:start]+after+original_source[end:]
    table_replacements = []
    for old, new in ((base_contract['plan_sha256'], execution_sha), (_digest(base_contract), _digest(contract))):
        token = '"'+old+'"'
        _need(reordered.count(token) == 1, 'original table identity must occur exactly once')
        reordered = reordered.replace(token, '"'+new+'"')
        table_replacements.append(dict(original=old, replacement=new, occurrences=1))
    mapping = dict(original_full_source_sha256=_hash(original_source.encode()),
        trial_full_source_sha256=_hash(reordered.encode()), capture_full_source_sha256=capture['full_source_sha256'],
        capture_provenance=capture['provenance'], source_char_start=start, source_char_end_exclusive=end,
        original_fragment_sha256=_hash(before.encode()), replacement_fragment_sha256=_hash(after.encode()),
        unchanged_prefix_sha256=_hash(original_source[:start].encode()),
        unchanged_suffix_before_table_identity_sha256=_hash(original_source[end:].encode()),
        table_identity_replacements=table_replacements,
        bodies=[dict(task=name, stage_index=stages[name]['index'], body_sha256=_hash(bodies[name].encode()))
                for name in original_order],
        every_original_body_copied_verbatim=kind == _ORDER,
        every_original_inner_statement_copied_verbatim=True, fusion=fusion,
        mathematical_ast_changed=False,
        outside_window_changed_only_for_explicit_table_identities=True)
    compiler = deepcopy(recipe['implementation']['backend_capability_record']['compiler'])
    _need(compiler['options'] == list(cpp.COMPILER_OPTIONS)
          and ('g++' in compiler['identity'].lower() or 'gcc' in compiler['identity'].lower()),
          'exact original GNU serial C++20 compiler profile required')
    read_artifact(reference)
    return dict(reference=reference, region=region, plan=plan, binding=binding, control=control,
        execution_plan=execution_plan, contract=contract, source=reordered.encode(), source_mapping=mapping,
        compiler=compiler, providers=providers, owners=owners, power=power,
        parent_recipe=recipe, parent_schedule=schedule)


def _identity(derived):
    headers = {name: (_HERE/name).read_bytes() for name in ('stateful_abi_v1.h', 'scar_abi_v1.h')}
    fingerprint = _implementation()
    compiler = derived['compiler']
    payload = dict(source=_hash(derived['source']), headers={name: _hash(raw) for name, raw in headers.items()},
        compiler=compiler, implementation_sha256=fingerprint, contract=derived['contract'],
        execution_plan=derived['execution_plan'], control_sha256=_digest(derived['control']),
        source_mapping=derived['source_mapping'])
    return _digest(payload), fingerprint, headers


def _metadata(derived, directory, build_id, fingerprint, headers):
    plan, binding = derived['plan'], derived['binding']
    files = dict(region='region.json', plan='base-plan.json', control='control.json',
        source='stateful.cpp', library='stateful_window_'+build_id[:20]+'.so',
        **{name: name for name in headers})
    result = dict(schema=derived.get('artifact_schema', SCHEMA), schema_version=1,
        stateful_abi_version=1, build_id=build_id,
        **derived['contract'], contract_sha256=_digest(derived['contract']),
        compiler=deepcopy(derived['compiler']), implementation_sha256=fingerprint,
        base_backend_implementation_sha256=base_stateful.implementation_sha256(),
        files=files, sha256={key: _hash((directory/name).read_bytes()) for key, name in files.items()},
        execution_plan=derived['execution_plan'], execution_plan_sha256=derived['contract']['plan_sha256'],
        base_plan_sha256=_digest(plan), source_mapping=derived['source_mapping'], preparation=derived['reference'],
        cpu_threads=plan['cpu_threads'], cpu_ids=list(binding.cpu_ids),
        memory_budget_bytes=plan['memory_budget_bytes'], workspace_payload_bytes=plan['workspace_payload_bytes'],
        external_array_bytes=plan['external_array_bytes'],
        total_region_array_payload_bytes=plan['total_region_array_payload_bytes'],
        external_assumptions=deepcopy(plan['external_assumptions']),
        execution=('ONE_NATIVE_CALL_SAME_CPU_SERIAL_FUSED_RAW_COPY_LOOPS_THEN_ORIGINAL_STAGES'
            if derived['control']['kind'] == _FUSION else
            'ONE_NATIVE_CALL_SAME_CPU_SERIAL_RAW_COPY_WINDOW_PERMUTATION_THEN_ORIGINAL_STAGES'),
        control_kind=derived['control']['kind'],
        failure_policy='GUARD_FAILURE_ONLY_BEFORE_EXTERNAL_COMMIT_ORIGINAL_ONCE', memory_policy='UNBOUND',
        time_fusion=False, trial_only=True, full_boundary_lowering_authority=False,
        cost_status='UNMEASURED', default_decision='NO_OP', deployment_authorized=False,
        performance_measured=False, target_variants_ranked=False, physical_optimality_proven=False)
    if 'rewrite' in derived:
        result['rewrite'] = deepcopy(derived['rewrite'])
    return result


def _derive_certified(preparation):
    """Reconcile the replacement graph with the existing exact source fusion."""
    from dataclasses import replace
    from scar.compiler.prepare_window_rewrite import replay_window_rewrite
    from scar.ir.activity_costs import activity_cost_document
    from scar.ir.temporal_resources import projection_document
    from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_document
    from scar.optimization.fused_copy_graph import verify_fused_copy_graph
    reference = _reference(preparation)
    report, rewritten, resources, costs, certificate, context = replay_window_rewrite(reference['uri'])
    _need(reference['schema'] == 'scar.window-rewrite-preparation' and read_artifact(reference) == report,
          'original replacement-graph preparation receipt required')
    parent = _derive(context['parent_window_preparation'], context['window_id'], None, kind=_FUSION,
                     _qualified=context.get('qualified_parent'))
    original = context['original_schedule']
    _need(parent['reference'] == report['original_windows']
          and parent['parent_recipe'] == context['recipe'] and parent['parent_schedule'] == original,
          'certified fusion must retain its original window preparation')
    _need(verify_fused_copy_graph(original, rewritten, certificate) == certificate,
          'replacement task graph differs from independent original-graph certificate replay')
    original_ids = parent['control']['original_task_order']; members = set(original_ids)
    _need((certificate.get('schema'), certificate.get('schema_version'))
          == ('scar.fused-copy-graph-certificate', 1)
          and certificate['parent_schedule_sha256'] == _digest(schedule_document(original))
          and certificate['rewritten_schedule_sha256'] == _digest(schedule_document(rewritten))
          and certificate['window_id'] == context['window_id']
          and certificate['original_task_ids'] == original_ids,
          'replacement certificate must bind the exact parent/source window and rewritten graph')
    replacement_id = certificate['replacement_task_id']
    before = {row.id: row for row in original.tasks}; after = {row.id: row for row in rewritten.tasks}
    _need(replacement_id not in before and set(after) == (set(before)-members)|{replacement_id},
          'replacement graph must replace exactly the complete certified window')
    replacement = after[replacement_id]
    expected_pairs = [dict(read=before[name].reads[0], write=before[name].writes[0], source_task=name)
                      for name in original_ids]
    domain = parent['control']['fusion']['domain']
    _need(replacement.kind is K.COPY and replacement.capability == 'CPU_HOST_COPY_REGION'
          and replacement.attrs.get('stage_kind') == 'FUSED_SNAPSHOT_ROOT'
          and replacement.attrs.get('dtype') == 'float64'
          and list(replacement.attrs.get('shape', ())) == domain
          and replacement.attrs.get('child_task_ids') == original_ids
          and replacement.attrs.get('copy_pairs') == expected_pairs
          and list(replacement.reads) == [row['read'] for row in expected_pairs]
          and list(replacement.writes) == [row['write'] for row in expected_pairs]
          and not replacement.world_before and not replacement.world_after
          and not replacement.allocates and not replacement.releases and not replacement.validates,
          'replacement COPY ports, source children or exact raw-copy domain differ')
    task_map = {name: replacement_id if name in members else name for name in before}
    _need(certificate['task_map'] == task_map, 'certificate task map is not complete caller-matched replacement')
    remap = lambda names: tuple(dict.fromkeys(task_map[name] for name in names))
    for name in set(before)-members:
        row = after[name]; old = before[name]
        _need(row.depends_on == remap(old.depends_on) and row.validates == remap(old.validates)
              and replace(row, depends_on=old.depends_on, validates=old.validates) == old,
              'a source anchor or non-window task changed beyond certified port dependency contraction')
    expected_dependencies = tuple(dict.fromkeys(dependency for name in original_ids
        for dependency in before[name].depends_on if dependency not in members))
    _need(replacement.depends_on == expected_dependencies,
          'replacement must retain every original external window prerequisite')
    _need(rewritten.values == original.values and rewritten.storage_slots == original.storage_slots
          and rewritten.region_id == original.region_id and rewritten.call_scope == original.call_scope
          and (rewritten.budget_bytes, rewritten.budget_scope, rewritten.budget_refs)
              == (original.budget_bytes, original.budget_scope, original.budget_refs),
          'replacement changed logical values, physical slots, call scope or original budget')
    materials = {row.id: row for row in original.materializations}
    changed_materials = {row['write'] for row in expected_pairs}
    _need({row.id for row in rewritten.materializations} == set(materials),
          'replacement must retain every original material/version port')
    for row in rewritten.materializations:
        old = materials[row.id]
        _need(row == (replace(old, ready_task=replacement_id) if row.id in changed_materials else old),
              'replacement changed material content identity/version/region beyond readiness contraction')
    _need(resources.target == context['original_resources'].target
          and resources.budget == context['original_resources'].budget,
          'rewritten resources changed the retained target or original memory budget')
    active = [row.id for row in rewritten.temporal.activities if row.task == replacement_id]
    _need(len(active) == 1, 'one replacement COPY lifecycle activity required')
    original_activities = {row.task: row.id for row in original.temporal.activities}
    old_occupancies = {row.activity: row for row in original.temporal.occupancies
                       if row.kind.value == 'EXECUTION_DEMAND'}
    fused_occupancies = [row for row in rewritten.temporal.occupancies
                        if row.kind.value == 'EXECUTION_DEMAND' and row.activity == active[0]]
    _need(len(fused_occupancies) == 1, 'one serial replacement COPY execution demand required')
    fused_occupancy = fused_occupancies[0]
    old_bindings = {row.occupancy: row for row in context['original_resources'].executions}
    new_bindings = {row.occupancy: row for row in resources.executions}
    fused_binding = new_bindings[fused_occupancy.id]
    for name in original_ids:
        old_occupancy = old_occupancies[original_activities[name]]
        old_binding = old_bindings[old_occupancy.id]
        _need((fused_occupancy.resource, fused_occupancy.units)
                  == (old_occupancy.resource, old_occupancy.units)
              and (fused_binding.exact_thread, fused_binding.candidate_threads, fused_binding.units)
                  == (old_binding.exact_thread, old_binding.candidate_threads, old_binding.units),
              'replacement cannot change the original single-caller CPU demand or placement')
    matched_costs = [row for row in costs.costs if active[0] in row.activities]
    _need(len(matched_costs) == 1 and matched_costs[0].activities == tuple(active)
          and matched_costs[0].basis.value == 'UNKNOWN' and matched_costs[0].estimate_ticks is None,
          'certified source fusion currently requires unknown unsummed replacement service cost')
    mapping = []
    for pair, body in zip(expected_pairs, parent['control']['fusion']['original_bodies']):
        _need(pair['source_task'] == body['task'], 'each fused inner must map to its original child task')
        mapping.append(dict(**pair, body_sha256=body['body_sha256'], inner_sha256=body['inner_sha256']))
    documents = dict(schedule=schedule_document(rewritten), resources=projection_document(resources),
        costs=activity_cost_document(costs), certificate=certificate)
    graph_hashes = {name: _digest(value) for name, value in documents.items()}
    for name, value in documents.items():
        _need(_digest(read_artifact(report[name])) == graph_hashes[name],
              'retained rewrite artifact differs from independent typed/source replay: '+name)
    control = dict(schema=CERTIFIED_CONTROL_SCHEMA, schema_version=1, kind=_FUSION,
        preparation=reference, original_windows=parent['reference'], window_id=context['window_id'],
        parent_source_control=parent['control'], replacement_task_id=replacement_id,
        copy_pair_source_mapping=mapping, rewritten_semantic_hashes=graph_hashes,
        rewritten_artifacts={name: deepcopy(report[name]) for name in (*documents, 'lineage')},
        port_and_storage_conservation_checked=True, all_non_window_tasks_preserved=True,
        readiness_policy='ALL_WINDOW_OUTPUTS_READY_ONLY_AFTER_COMPLETE_FUSED_LOOP',
        mathematical_ast_changed=False, worker_threads_created=0, cpu_binding_changed=False,
        cost_status='UNMEASURED', deployment_authorized=False)
    execution_plan = dict(schema='scar.certified-window-fusion-execution-plan', schema_version=1,
        base_plan_sha256=_digest(parent['plan']), control_sha256=_digest(control),
        rewritten_schedule_sha256=graph_hashes['schedule'], certificate_sha256=graph_hashes['certificate'])
    contract = {**parent['contract'], 'plan_sha256': _digest(execution_plan)}
    source = parent['source'].decode()
    for old, new in ((parent['contract']['plan_sha256'], contract['plan_sha256']),
                     (_digest(parent['contract']), _digest(contract))):
        _need(source.count('"'+old+'"') == 1, 'parent fusion table identity must be unique')
        source = source.replace('"'+old+'"', '"'+new+'"')
    source_mapping = deepcopy(parent['source_mapping'])
    source_mapping.update(parent_fusion_source_sha256=_hash(parent['source']),
        trial_full_source_sha256=_hash(source.encode()), replacement_task_id=replacement_id,
        copy_pair_source_mapping=mapping, rewritten_graph_sha256=graph_hashes['schedule'],
        replacement_certificate_sha256=graph_hashes['certificate'])
    source_mapping['table_identity_replacements'][0]['replacement'] = contract['plan_sha256']
    source_mapping['table_identity_replacements'][1]['replacement'] = _digest(contract)
    read_artifact(reference)
    return {**parent, 'artifact_schema': CERTIFIED_SCHEMA, 'reference': reference,
        'control': control, 'execution_plan': execution_plan, 'contract': contract,
        'source': source.encode(), 'source_mapping': source_mapping,
        'rewrite': dict(preparation=reference, replacement_task_id=replacement_id,
            graph_semantic_hashes=graph_hashes, certificate=deepcopy(report['certificate']),
            parent_original_ids=original_ids, cost_status='UNKNOWN', deployment_authorized=False)}


def build_window_order_trial(preparation, window_id, task_order, out):
    """Compile one explicit certified order; no profitability or deployment selection."""
    return _build(_derive(preparation, window_id, task_order), out)


def build_window_fusion_trial(preparation, window_id, out):
    """Fuse exact independent raw-copy loops; never reuse their summed costs."""
    return _build(_derive(preparation, window_id, None, kind=_FUSION), out)


def build_certified_window_fusion(preparation, out):
    """Consume a replayed replacement task graph, retaining unmeasured trial status."""
    return _build(_derive_certified(preparation), out)


def _build(derived, out):
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh window-order trial artifact directory required')
    compiler = derived['compiler']
    _need(compiler['argv'] and shutil.which(compiler['argv'][0]), 'original GNU compiler is unavailable')
    identity = subprocess.run([*compiler['argv'], '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    _need(identity == compiler['identity'], 'original GNU compiler identity changed')
    build_id, fingerprint, headers = _identity(derived)
    directory.mkdir(parents=True)
    (directory/'region.json').write_bytes(_json(document(derived['region'])))
    (directory/'base-plan.json').write_bytes(_json(derived['plan']))
    (directory/'control.json').write_bytes(_json(derived['control']))
    (directory/'stateful.cpp').write_bytes(derived['source'])
    for name, raw in headers.items():
        (directory/name).write_bytes(raw)
    library = directory/('stateful_window_'+build_id[:20]+'.so')
    command = [*compiler['argv'], *compiler['options'], str(directory/'stateful.cpp'), '-o', str(library), '-lm']
    process = subprocess.run(command, capture_output=True, text=True)
    (directory/'compiler-stdout.txt').write_text(process.stdout)
    (directory/'compiler-stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError('window-order trial C++ compilation failed:\n'+process.stderr)
    _need(_implementation() == fingerprint, 'window-order implementation changed during compilation')
    read_artifact(derived['reference'])
    metadata = _metadata(derived, directory, build_id, fingerprint, headers)
    (directory/'artifact.json').write_bytes(_json(metadata))
    return (load_certified_window_fusion(directory) if derived.get('artifact_schema') == CERTIFIED_SCHEMA
            else load_window_order_trial(directory))


def load_window_order_trial(out):
    """Cold replay original scope/pairs/source, then load the original ABI1 host."""
    directory = Path(out).resolve()
    if directory.is_file():
        directory = directory.parent
    metadata = json.loads((directory/'artifact.json').read_bytes())
    _need((metadata.get('schema'), metadata.get('schema_version'), metadata.get('stateful_abi_version'))
          == (SCHEMA, 1, 1), 'source-proven window-order trial artifact required')
    for key, name in metadata['files'].items():
        _need(type(name) is str and Path(name).name == name
              and _hash((directory/name).read_bytes()) == metadata['sha256'][key],
              'window-order artifact checksum differs: '+key)
    control = json.loads((directory/'control.json').read_bytes())
    _need((control.get('schema'), control.get('schema_version')) == (CONTROL_SCHEMA, 1),
          'original explicit window-order control required')
    derived = _derive(control['preparation']['uri'], control['window_id'], control['task_order'],
                      kind=control['kind'])
    _need(derived['reference'] == control['preparation'] and derived['control'] == control,
          'window-order source control differs from preparation replay')
    return _finish_load(directory, metadata, derived)


def load_certified_window_fusion(out):
    """Replay the rewritten graph/certificate and exact source before loading ABI1."""
    directory = Path(out).resolve()
    if directory.is_file():
        directory = directory.parent
    metadata = json.loads((directory/'artifact.json').read_bytes())
    _need((metadata.get('schema'), metadata.get('schema_version'), metadata.get('stateful_abi_version'))
          == (CERTIFIED_SCHEMA, 1, 1), 'certified window-fusion artifact schema required')
    for key, name in metadata['files'].items():
        _need(type(name) is str and Path(name).name == name
              and _hash((directory/name).read_bytes()) == metadata['sha256'][key],
              'certified window-fusion checksum differs: '+key)
    control = json.loads((directory/'control.json').read_bytes())
    _need((control.get('schema'), control.get('schema_version')) == (CERTIFIED_CONTROL_SCHEMA, 1),
          'original certified fusion control required')
    derived = _derive_certified(control['preparation']['uri'])
    _need(derived['reference'] == control['preparation'] and derived['control'] == control,
          'certified fusion control differs from replacement graph replay')
    return _finish_load(directory, metadata, derived)


def _finish_load(directory, metadata, derived):
    build_id, fingerprint, headers = _identity(derived)
    _need(metadata == _metadata(derived, directory, build_id, fingerprint, headers),
          'window-order metadata differs from full source/control replay')
    _need((directory/metadata['files']['source']).read_bytes() == derived['source']
          and (directory/'region.json').read_bytes() == _json(document(derived['region']))
          and (directory/'base-plan.json').read_bytes() == _json(derived['plan'])
          and all((directory/name).read_bytes() == raw for name, raw in headers.items()),
          'trial C++/base region/plan/ABI differs from original guarded replay')
    library = ctypes.CDLL(str(directory/metadata['files']['library']))
    getter = library.scar_get_stateful_table_v1
    getter.argtypes = []; getter.restype = ctypes.POINTER(base_stateful._Table)
    table = getter()
    _need(table and table.contents.abi_version == 1
          and table.contents.struct_size == ctypes.sizeof(base_stateful._Table) and table.contents.run,
          'original stateful table ABI differs')
    region = derived['region']
    _need((table.contents.array_count, table.contents.scalar_count, table.contents.provider_count)
          == (sum(row.kind == 'array' for row in region.inputs),
              sum(row.kind == 'scalar' for row in region.inputs), len(region.providers)),
          'original stateful table input counts differ')
    for key in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        _need(getattr(table.contents, key) == metadata[key].encode(), 'trial embedded identity differs: '+key)
    read_artifact(derived['reference'])
    return base_stateful.StatefulArtifact(region, derived['plan'], directory, metadata, library, table,
        derived['providers'], derived['owners'], derived['power'])


__all__ = ['build_window_order_trial', 'build_window_fusion_trial', 'load_window_order_trial',
           'build_certified_window_fusion', 'load_certified_window_fusion']
