"""Finite-content induction across a closed, transactional workspace loop.

This proves a property under retained source/runtime contracts, not equal
values, cached tensors or a successful COMMIT in isolation. Only the repeated
array finite-input bit scans may be omitted. Source admission stays upstream.
"""
from scar.codegen.cpp import COMPILER_OPTIONS
from scar.compiler.stateful_model import validate_stateful_region, stateful_digest
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_schedule import _metadata, _normalized

SCHEMA = 'scar.workspace-finite-content-property-plan'
_OPS = {'input', 'const', 'slice', 'add', 'sub', 'mul', 'div', 'neg'}
_STAGES = {'SNAPSHOT_ROOT', 'SCALAR', 'COMPUTE', 'WRITE_SHADOW', 'COMMIT'}


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _initialization(workspace):
    """Every private read must be covered by this iteration's actual writes."""
    graph = schedule_from_dict(workspace['shared_step'])
    slots = {s.id: s for s in graph.storage_slots if s.ownership == 'SCHEDULE'}
    materials = {m.id: m for m in graph.materializations}
    definitions = {s: [] for s in slots}; records = []
    positions = {t.id: i for i, t in enumerate(graph.tasks)}
    def covered(slot, start, stop):
        point = start
        for lo, hi in sorted(definitions[slot]):
            if lo <= point:
                point = max(point, hi)
        return point >= stop
    for task in graph.tasks:
        if task.kind in (K.ALLOCATE, K.RELEASE):
            continue
        for name in task.reads:
            material = materials[name]; region = material.region
            if region.slot in slots:
                _need(material.ready_task is not None and positions[material.ready_task] < positions[task.id]
                      and covered(region.slot, region.offset_bytes, region.offset_bytes+region.nbytes),
                      'a private read could observe uninitialized/previous-iteration content')
                records.append((task.id, name, region.slot, region.offset_bytes, region.nbytes, material.ready_task))
        for name in task.writes:
            material = materials[name]; region = material.region
            if region.slot not in slots:
                continue
            if task.kind is K.STATE_WRITE:
                _need(covered(region.slot, region.offset_bytes, region.offset_bytes+region.nbytes),
                      'partial shadow write requires a complete current-iteration base')
            else:
                _need(task.kind in (K.SNAPSHOT, K.COMPUTE), 'unsupported private content definition')
                definitions[region.slot].append((region.offset_bytes, region.offset_bytes+region.nbytes))
    original = workspace['initialization_proof']
    _need(original['rule'] == 'CURRENT_ITERATION_FULL_DEFINITION_BEFORE_EVERY_PRIVATE_READ'
          and records == [(r['task'], r['materialization'], r['slot'], r['extent']['offset_bytes'],
              r['extent']['nbytes'], r['producer']) for r in original['reads']],
          'original initialization proof differs from complete private-read replay')
    return graph, len(records)


def derive_workspace_guard_properties(region, step_plan, workspace_plan, execution_certificate,
        native_metadata, *, source_refs=()):
    """Return PROVEN_CONDITIONAL or concrete profile refusals; perform no I/O."""
    region = validate_stateful_region(region)
    _need(type(source_refs) is tuple and all(type(r) is dict for r in source_refs), 'original reference tuple required')
    region_sha = stateful_digest(region); step_sha = semantic_sha256(step_plan)
    workspace_sha = semantic_sha256(workspace_plan)
    blueprint = execution_certificate['native_blueprint']
    _need((execution_certificate.get('schema'), execution_certificate.get('schema_version'))
              == ('scar.workspace-execution-refinement', 1)
          and execution_certificate['graph_lowering_checked'] is True
          and execution_certificate['execution_scope'] == 'WORKSPACE_LIFETIME_ONLY'
          and execution_certificate['inputs']['step_plan_sha256'] == step_sha
          and execution_certificate['inputs']['workspace_plan_sha256'] == workspace_sha
          and blueprint['workspace_plan_sha256'] == native_metadata['workspace_plan_sha256'] == workspace_sha
          and blueprint['execution_plan_sha256'] == native_metadata['execution_plan_sha256']
          and blueprint['base_ids'] == native_metadata['base_ids']
          and blueprint['base_ids']['region_sha256'] == workspace_plan['step_region_sha256'] == region_sha
          and blueprint['base_ids']['plan_sha256'] == workspace_plan['step_plan_sha256'] == step_sha
          and blueprint['cpp_source']['sha256'] == native_metadata['sha256']['source']
          and blueprint['source_mapping'] == native_metadata['source_mapping']
          and step_plan['region_id'] == region.id and step_plan['region_sha256'] == region_sha,
          'exact original region/step/workspace/native/execution refinement identities required')
    count = workspace_plan['trip_count']
    _need(type(count) is int and 0 <= count < 2**64 and count == execution_certificate['trip_count']
          == native_metadata['trip_count'], 'exact source sequence count differs')
    inputs, nodes, shapes, expr_shapes, kinds, _, _, _, issues = _metadata(region, step_plan['input_shapes'])
    rejected = []
    def refuse(code, field, detail):
        rejected.append(dict(code=code, field=field, detail=detail))
    if count <= 1:
        refuse('NO_REPEATED_INPUT_SCAN', 'trip_count', 'zero/one iteration has no repeated input scan to remove')
    if region.providers or any(n.op not in _OPS for n in region.nodes) or any(s['kind'] not in _STAGES for s in step_plan['stages']):
        refuse('PROVIDER_OR_OPAQUE_OPERATION', 'region/providers', 'finite-output and side-effect closure is not established for providers/opaque work')
    if (step_plan.get('cpu_threads') != 1 or len(step_plan['resource_binding']['cpu_ids']) != 1
            or native_metadata.get('cpu_threads') != 1):
        refuse('NOT_SERIAL_SINGLE_CPU', 'cpu_threads', 'this property profile covers one serial calling thread')
    if issues or step_plan.get('status') != 'READY' or step_plan.get('frontier') or step_plan.get('budget_satisfied') is not True:
        refuse('INCOMPLETE_MEMORY_OR_SHAPE_PLAN', 'step_plan', 'closed bounded shapes/private workspace are required')
    required = {'EXACT_NDARRAY_FLOAT64_CPU_RANK2', 'DISTINCT_ARRAY_INPUTS_HAVE_DISJOINT_STORAGE',
        'NO_FOREIGN_CONCURRENT_INPUT_WRITERS', 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS',
        'ORIGINAL_FLOATING_ENVIRONMENT_AND_FALLBACK_CONTRACT', 'ORDERED_TRANSACTIONAL_SHADOW_WRITES_BEFORE_COMMIT',
        'ALL_SHAPES_ALIAS_BOUNDS_AND_WORKSPACE_CHECKED_BEFORE_EXTERNAL_WRITE',
        'ALL_CPU_AND_FENV_RESTORATION_SUCCEEDS_BEFORE_COMMIT'}
    loop_guards = {'NO_INTERSTEP_PYTHON_CODE_HOOKS_OBSERVERS_OR_CONCURRENT_STATE_ACCESS',
                   'IDENTICAL_ARRAY_IDENTITIES_DESCRIPTORS_SCALARS_AND_PROVIDERS_EACH_STEP'}
    if (not required <= set(step_plan['guard_contract'])
            or not loop_guards <= set(workspace_plan['required_live_guards'])
            or 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS' not in workspace_plan['external_assumptions']
            or step_plan['alias_contract'] != 'PAIRWISE_DISJOINT_ARRAY_INPUT_STORAGE_AND_PRIVATE_WORKSPACE'):
        refuse('MEMORY_OR_ISOLATION_CONTRACT_MISSING', 'guard_contract', 'no hidden/interstep writer, bounded disjoint roots and private shadows are required')
    if (tuple(native_metadata['compiler']['options']) != tuple(COMPILER_OPTIONS)
            or step_plan['numerical_contract']['arithmetic'] != 'ORIGINAL_OPERANDS_AND_DAG_ORDER_NO_REASSOCIATION_OR_CONTRACTION'
            or step_plan['transaction']['external_writes_before_success'] is not False
            or step_plan['transaction']['no_failing_operations_or_GuardFailure_after_first_external_commit'] is not True):
        refuse('STRICT_FP_OR_TRANSACTION_CONTRACT_MISSING', 'numerical_contract', 'original strict FP flags/checks before every commit and original transaction must remain')
    arrays = [r for r in region.inputs if r.kind == 'array']
    scans = [dict(input_id=r.id, array_index=i, shape=shapes.get(r.id),
        element_count=None if r.id not in shapes else shapes[r.id][0]*shapes[r.id][1],
        nbytes=None if r.id not in shapes else 8*shapes[r.id][0]*shapes[r.id][1], transfer_ref=None)
        for i, r in enumerate(arrays)]
    properties, writes, transfers = [], [], []
    init_count = None
    if not rejected:
        graph, init_count = _initialization(workspace_plan)
        epochs = {r.id: 0 for r in arrays}; established = {}
        def prove(value):
            node = nodes[value]
            if value in established:
                return established[value]
            parents = [prove(v) for v in node.inputs]
            if node.op == 'input':
                if node.attrs['kind'] == 'array':
                    _need(node.attrs['epoch'] == epochs[node.attrs['name']], 'input property uses a stale local content epoch')
                    rule = 'FINITE_FULL_ROOT_AT_CURRENT_LOCAL_EPOCH'
                else:
                    rule = 'ORIGINAL_EVERY_ITERATION_FINITE_SCALAR_GUARD'
            elif node.op == 'const':
                rule = 'EXACT_FINITE_SOURCE_CONSTANT'
            elif node.op == 'slice':
                rule = 'BOUNDED_VIEW_SUBSET_OF_FINITE_ROOT'
            else:
                rule = 'FINITE_OPERANDS_CLOSED_IEEE_BINARY64_OP_UNDER_ORIGINAL_FP_SUCCESS'
            record = dict(expression=value, op=node.op, kind=kinds[value], shape=expr_shapes[value],
                operands=list(node.inputs), operand_properties=[p['expression'] for p in parents],
                property='FINITE_CONTENT', rule=rule, source_ref=dict(artifact_role='region',
                    pointer='#/nodes/'+str(next(i for i, n in enumerate(region.nodes) if n.id == value))),
                requires_original_fp_success=node.op in {'add','sub','mul','div','neg'})
            if node.op == 'input':
                record.update(input_id=node.attrs['name'], local_epoch=node.attrs['epoch'])
            established[value] = record; properties.append(record)
            return record
        stage_writes = [s for s in step_plan['stages'] if s['kind'] == 'WRITE_SHADOW']
        stage_commits = [s for s in step_plan['stages'] if s['kind'] == 'COMMIT']
        actions = [a for a in region.actions if a.kind == 'write']
        _need(len(stage_writes) == len(stage_commits) == len(actions), 'complete original ordered write/commit coverage required')
        for action in region.actions:
            proof = prove(action.value)
            if action.kind != 'write':
                continue
            normalized, shape = _normalized(action.bounds, shapes[action.target])
            index = len(writes); write, commit = stage_writes[index], stage_commits[index]
            _need(action.epoch_before == epochs[action.target]
                  and write['target'] == commit['target'] == action.target
                  and write['source_action_ids'] == commit['source_action_ids'] == [action.id]
                  and write['root_expr'] == action.value and write['op'] == action.op
                  and write['normalized_bounds'] == commit['normalized_bounds'] == normalized
                  and write['shape'] == commit['shape'] == shape
                  and commit['input_payloads'] == [step_plan['shadows'][action.target]],
                  'exact source write ROI and final-shadow commit coverage differ')
            writes.append(dict(action=action.id, target=action.target, op=action.op,
                rhs_property=proof['expression'], normalized_bounds=normalized, shape=shape,
                epoch_before=action.epoch_before, epoch_after=action.epoch_after,
                write_stage=write['id'], commit_stage=commit['id'],
                written_roi='FINITE_UNDER_RETAINED_FP_SUCCESS',
                complement='UNCHANGED_FINITE_PRIOR_SHADOW_CONTENT', full_root_after='FINITE_CONTENT',
                extra_obligation='COMPOUND_WRITE_ARITHMETIC_USES_ORIGINAL_FP_FLAGS' if action.op != 'assign' else None))
            epochs[action.target] += 1
        _need(len(established) == len(region.nodes), 'all source expressions must be covered by the property proof')
        values = {v.id: v for v in graph.values}; materials = {m.id: m for m in graph.materializations}
        loop_rows = {r['input_id']: r for r in workspace_plan['loop_model']['inputs']}
        carries = execution_certificate['body']['content_carries']
        for ordinal, row in enumerate(arrays):
            related = [w for w in writes if w['target'] == row.id]
            loop = loop_rows[row.id]; committed = [c['commit_stage'] for c in related]
            _need(loop['kind'] == 'array' and loop['step_path'] == list(row.path)
                  and loop['entry']['local_epoch'] == 0 and loop['exit']['local_epoch'] == epochs[row.id],
                  'source loop local-epoch transfer differs')
            snapshots = [s for s in step_plan['stages'] if s['kind'] == 'SNAPSHOT_ROOT' and s['target'] == row.id]
            if related:
                _need(len(snapshots) == 1 and snapshots[0]['shape'] == shapes[row.id]
                      and loop['role'] == 'ORDERED_ARRAY_CONTENT_CARRY', 'written roots need a complete finite entry snapshot')
                incoming = [m for m in graph.materializations if m.external_ready and values[m.value].attrs.get('input_id') == row.id]
                _need(len(incoming) == 1, 'one original finite-content entry materialization required')
                carry = [c for c in carries if c['after'] == incoming[0].id]
                _need(len(carry) == 1 and carry[0]['iteration_distance'] == 1
                      and materials[carry[0]['before']].ready_task in committed
                      and materials[carry[0]['before']].region == incoming[0].region and carry[0]['source_refs'],
                      'written-root finite property requires the original committed-content carry')
                basis = 'FINITE_FULL_SHADOW_AND_FINITE_COMMITTED_ROIS_PLUS_UNCHANGED_FINITE_EXTERNAL_COMPLEMENT'
                carry_ref = dict(artifact_role='execution_certificate', pointer='#/body/content_carries/'+str(carries.index(carry[0])))
            else:
                _need(not snapshots and loop['role'] == 'GUARDED_LOOP_LIVE_IN_ARRAY'
                      and not loop['ordered_writes'], 'read-only finite transfer needs no source or alias writer')
                basis = 'NO_REGION_WRITES_PLUS_DISJOINT_ROOTS_AND_NO_FOREIGN_OR_INTERSTEP_WRITERS'
                carry_ref = None
            transfers.append(dict(input_id=row.id, shape=shapes[row.id], property='FINITE_CONTENT',
                entry='FIRST_SCAN_IF_i_EQ_0_ELSE_PREVIOUS_SUCCESSFUL_EXIT_PROPERTY',
                local_epoch_exit=epochs[row.id], written=bool(related), source_write_actions=[w['action'] for w in related],
                commit_stages=committed, full_root_exit=basis, content_carry_ref=carry_ref,
                next_entry=dict(iteration='i+1', local_epoch=0, property='FINITE_CONTENT', enabled_when='i+1<N'),
                data_value_equality_claimed=False, runtime_logical_version_inferred=False))
            scans[ordinal]['transfer_ref'] = '#/input_transfers/'+str(ordinal)
    proven = not rejected
    result = dict(schema=SCHEMA, schema_version=1, status='PROVEN_CONDITIONAL' if proven else 'REJECTED',
        native_blueprint=execution_certificate['native_blueprint'], trip_count=count,
        inputs=dict(region_sha256=region_sha, step_plan_sha256=step_sha, workspace_plan_sha256=workspace_sha,
            execution_certificate_sha256=semantic_sha256(execution_certificate),
            native_metadata_sha256=semantic_sha256(native_metadata), source_refs=encode(source_refs)),
        input_scans=scans,
        input_scan_policy=dict(input_table='#/input_scans', first_iteration='ALL_ORIGINAL_INPUT_FINITE_BIT_SCANS',
            later_iterations='SKIP_ONLY_INPUT_FINITE_SCANS_FROM_PROPERTY_INDUCTION' if proven else 'KEEP_ALL_ORIGINAL_SCANS',
            enabled_when='1<=i<N_AND_ALL_PRIOR_NATIVE_STEPS_COMMITTED', zero_trip='NO_FIELDS_OR_SCAN',
            no_interstep_or_cross_invocation_property_cache=True),
        expression_properties=properties, ordered_writes=writes, input_transfers=transfers,
        initialization=dict(private_reads_checked=init_count,
            rule='CURRENT_ITERATION_FULL_DEFINITION_BEFORE_EVERY_PRIVATE_READ',
            source_ref=dict(artifact_role='workspace_plan', pointer='#/initialization_proof'),
            previous_iteration_workspace_content_is_not_a_finite_input=True),
        required_guards=list(workspace_plan['required_live_guards']),
        external_assumptions=list(workspace_plan['external_assumptions']),
        fp_obligations=['FINITE_INITIAL_ARRAY_INPUTS_AND_EVERY_ITERATION_SCALARS',
            'CLOSED_IEEE_BINARY64_ADD_SUB_MUL_DIV_NEG_WITH_FINITE_CONSTANTS',
            'ORIGINAL_FP_INVALID_DIVBYZERO_OVERFLOW_UNDERFLOW_FLAGS_ZERO_BEFORE_COMMIT',
            'ORIGINAL_NUMERIC_GUARDS_AND_AFFINITY_FENV_RESTORATION_SUCCEED',
            'NO_FAST_MATH_REASSOCIATION_OR_CONTRACTION'],
        preserved_checks=['ALL_ABI_SHAPE_ALIAS_MEMORY_PROVIDER_IDENTITY_CHECKS', 'EVERY_ITERATION_SCALAR_FINITE_AND_EXACTNESS',
            'CPU_AFFINITY_AND_CONTEXT_CAPTURE', 'INPUT_SCANS_ON_FIRST_ITERATION',
            'EVERY_ITERATION_FLOATING_ENVIRONMENT_AND_NUMERIC_STATUS', 'EVERY_ITERATION_RESTORE_AND_STATUS_CHECKS',
            'ORIGINAL_DAG_AND_SHADOW_WRITES', 'PER_STEP_TRANSACTIONAL_COMMITS_AND_COMPLETED_CURSOR',
            'ORIGINAL_FAILURE_UNCOMMITTED_SUFFIX'],
        work_counts=dict(basis='SOURCE_LOOP_AND_GUARD_GEOMETRY_NOT_RUNTIME_COUNTERS',
            original_scan_invocations=count*len(scans),
            proposed_scan_invocations=len(scans) if proven else count*len(scans),
            avoided_element_visits=(count-1)*sum(r['element_count'] for r in scans) if proven else 0),
        rejections=rejected, finite_property_is_not_equal_content=True,
        source_admission_performed_by_this_function=False, current_source_admission_required=True,
        trial_only=True, selected_for_use=False, cost_status='UNKNOWN', default_runtime_decision='NO_OP',
        profitability_proven=False, deployment_authorized=False, target_program_executed=False)
    return encode(result)


def verify_workspace_guard_properties(region, step_plan, workspace_plan, execution_certificate,
        native_metadata, property_plan, *, source_refs=()):
    expected = derive_workspace_guard_properties(region, step_plan, workspace_plan,
        execution_certificate, native_metadata, source_refs=source_refs)
    _need(property_plan == expected, 'finite-content property plan differs from complete source-structure induction')
    return expected


__all__ = ['SCHEMA', 'derive_workspace_guard_properties', 'verify_workspace_guard_properties']
