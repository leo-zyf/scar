"""Separate original loop recurrence from detached sparse-solve speculation."""

from collections import defaultdict
import json
from pathlib import Path

from scar.analysis.csr_topology_scope import _Transport, _body, _only, _require
from scar.analysis.program_view import ProgramView
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_declared_catalog import DeclaredSourceCatalog
from scar.analysis.source_control import _enclosed, loop_dependencies
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.sparse_pipeline_scope import SCHEMA as PIPELINE, _interface, check_sparse_pipeline_scope
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.sparse-parallel-scope'


def _entry(port):
    return G.Port(port.template,port.owner,G.PortKind.ENTER)


def _loop_control(view,source,function,flow,scope):
    loop = _only((row for row in loop_dependencies(source.material,function,flow=flow)['loops']
        if row['header'] == scope['header']), 'parallel scope lacks one original shared loop')
    _require(loop['control_backedges'] == scope['control_backedges'] and loop['parent'] is None,
        'shared loop recurrence differs from the parent pipeline')
    header = decode(T.GlobalPort,loop['header'])
    context = BodyInterfaceResolver(view).plan(header.unit,header.port.template)
    step = context.steps[header.port.owner]
    control = step.instruction
    _require(type(control) is P.LoopControl and type(control.condition) is P.IterationCondition,
        'detached execution requires an original iterator loop')
    start = G.Port(context.key,control.body,G.PortKind.ENTER)
    item = G.Port(context.key,control.body,G.PortKind.PARAMETER,len(step.arguments))
    stops = {header.port,*(context.emitter.port(step.id,G.PortKind.EXIT,completion=c) for c in step.completions)}
    adjacency = defaultdict(list)
    for edge in flow.edges:
        if edge.kind is G.EdgeKind.CONTROL:
            adjacency[edge.source].append(edge)
    backedges = [edge for edge in flow.edges if edge.transition is G.Transition.BACKEDGE]
    world = [edge for edge in backedges if edge.kind is G.EdgeKind.STATE]
    explicit_data = [edge for edge in backedges if edge.kind is G.EdgeKind.DATA]
    _require(len(world) == 1 and not explicit_data,'unsupported previous-iteration explicit sparse value carry')
    transport = _Transport(flow)
    iterator = decode(T.GlobalPort,loop['slots'][0]['argument'])
    dependencies = dict(control_backedges=loop['control_backedges'],world_backedges=encode(world),
        explicit_data_backedges=encode(explicit_data), original_loop_item=encode(T.GlobalPort(header.unit,item)),
        iterator=dict(original_reference=encode(iterator), repeated_advance=encode(control.condition,
            record_tags=P.PRODUCTION_RECORD_TAGS),referent_mutable=True),
        original_loop_point=scope['original_operation'])
    return transport, start, item, stops, adjacency, dependencies


def _lexical_recurrences(flow,transport,start,adjacency,stops):
    """Cut each original normal store exit when checking same-iteration reads."""
    inside = _enclosed(start,adjacency,stops)
    stores = tuple(op for op in flow.operations if type(op.instruction) is P.AccessBinding
        and op.instruction.action is P.BindingActionKind.WRITE and op.entry in inside)
    local = []
    lexical_live_ins = []
    for location in flow.locations:
        reads = tuple(read for read in location.reads if _entry(read.value) in inside)
        if not reads:
            continue
        writer = _only(location.writes,'lexical reads have ambiguous reaching definitions')
        candidates = [op for op in stores if transport.path(writer.value,op.inputs[0].port) is not None]
        if not candidates and _entry(writer.value) not in inside:
            lexical_live_ins.append(dict(namespace=encode(location.namespace),definition=encode(writer),
                in_loop_reads=encode(reads),referent_contents_unchanged_proven=False))
            continue
        store = _only(candidates,'lexical writer does not identify one original body STORE')
        normal = G.Port(store.entry.template,store.entry.owner,G.PortKind.EXIT,completion=C.FALLTHROUGH)
        bypass = _enclosed(start,adjacency,{*stops,normal})
        _require(normal in inside and all(_entry(read.value) not in bypass for read in reads),
            'a loop-local read may use a previous iteration definition')
        local.append(dict(namespace=encode(location.namespace),definition=encode(writer),
            original_store=encode(store.entry),normal_exit=encode(normal),
            reads=encode(reads),normal_same_iteration_definition_dominates=True))
    return local, lexical_live_ins, inside


def _call_record(source,flow,observation):
    endpoint = decode(T.GlobalPort,observation['observation']['source_call'])
    return _only((item for item in flow.calls if source_port(source.facts,item.call.result) == endpoint),
        'no original source CALL operands for the selected sparse scope')


def _accumulator(view,source,flow,transport,report,caller_code,inside,solve_result):
    """Follow the modeled fresh hash, ordered updates and final read on one location."""
    native = {}
    for i,row in enumerate(report['observed_targets']):
        record = row['observation']
        if not row['entry_scope_reachable'] or record['caller_site']['code'] != caller_code:
            continue
        carrier = json.loads(record['metadata_json'])['native_boundary']
        qualname = carrier.get('qualname')
        if qualname in ('openssl_sha256','HASH.update','HASH.hexdigest'):
            _require(qualname not in native,'ordered hash ABI has multiple candidate calls')
            native[qualname] = (row,_call_record(source,flow,row),i)
    _require(set(native) == {'openssl_sha256','HASH.update','HASH.hexdigest'},
        'ordered publication requires a retained fresh hash constructor, update and final digest')
    binding = {}
    for key in ('HASH.update','HASH.hexdigest'):
        _,record,_ = native[key]
        callee = source_port(source.facts,record.call.callee).port
        attribute = _only((row for row in source.facts.attributes if row.action is F.AttributeAction.GET
            and transport.path(source_port(source.facts,row.value).port,callee) is not None),
            'ordered publication receiver lacks one original attribute access')
        binding[key] = transport.binding(source_port(source.facts,attribute.receiver).port)
    update_loc,writer,update_binding = binding['HASH.update']
    final_loc,_,final_binding = binding['HASH.hexdigest']
    _require(update_loc == final_loc and len(update_loc.reads) == 2,
        'the ordered accumulator has an additional lexical receiver route')
    update,update_record,update_index = native['HASH.update']
    final,final_record,final_index = native['HASH.hexdigest']
    constructor,constructor_record,constructor_index = native['openssl_sha256']
    _require(_entry(decode(T.GlobalPort,update['observation']['source_call']).port) in inside
        and _entry(decode(T.GlobalPort,final['observation']['source_call']).port) not in inside
        and _entry(decode(T.GlobalPort,constructor['observation']['source_call']).port) not in inside,
        'hash construction, update and publication scopes differ')
    constructor_result = source_port(source.facts,constructor_record.call.result).port
    initial = transport.path(constructor_result,writer.value)
    solution, = update_record.call.positional
    _,solution_writer,solution_binding = transport.binding(source_port(source.facts,solution).port)
    feed = transport.path(solve_result.port,solution_writer.value)
    final_result = source_port(source.facts,final_record.call.result).port
    function = _only((f for f in source.facts.functions if (f.template,f.body) == (flow.body.template,flow.body.region)),
        'publication body differs from its original declaration')
    returned = transport.path(final_result,source_port(source.facts,function.returns).port)
    _require(initial is not None and feed is not None and returned is not None,
        'original accumulation is not ordered from fresh construction through result to return')
    return dict(kind='ORDERED_HASH_ACCUMULATION',receiver=encode(update_loc.namespace),
        original_constructor=f'#/observed_targets/{constructor_index}',constructor_to_binding=encode(initial),
        original_update=f'#/observed_targets/{update_index}',original_final=f'#/observed_targets/{final_index}',
        ordered_world=dict(before=encode(source_port(source.facts,update_record.call.world)),
            after=encode(source_port(source.facts,update_record.call.result_world))),
        update_receiver=update_binding,final_receiver=final_binding,
        solve_result_to_publication=dict(result=encode(solve_result),path=encode(feed),binding=solution_binding),
        original_return_transport=encode(returned),
        requires_original_provider_and_normal_completion=True,
        sole_transitive_loop_effect_proven=False)


class _Expressions:
    """Recover a small original arithmetic tree for a private prediction only."""
    def __init__(self,view,source,flow,transport,formals,item,locals_):
        self.source = source
        self.transport = transport
        self.formals = formals
        self.item = item
        self.local_definitions = {row['definition']['fact']['locator'] for row in locals_}
        self.references = ReferenceSlicer(view.catalog)
        _,self.syntax,_,_,_ = _source_identity(view.catalog,source.unit)
        self.literals = tuple(op for op in flow.operations if type(op.instruction) is P.ConstructValue
            and op.instruction.construction is P.ConstructionKind.LITERAL)

    def span(self,occurrence):
        _,metadata,_ = _metadata(self.syntax,occurrence)
        return [metadata[key] for key in ('line','column','end_line','end_column')]

    def _point(self,index):
        return source_port(self.source.facts,index).port

    def expression(self,point,ancestors=frozenset()):
        _require(point not in ancestors,'cyclic source expression needs its original evaluation')
        nested = ancestors | {point}
        def descend(value):
            return self.expression(value,nested)
        if point in self.formals.values():
            return dict(kind='CALLER_FORMAL',port=encode(T.GlobalPort(self.source.unit,point)))
        if self.transport.path(self.item,point) is not None:
            return dict(kind='CURRENT_ITERATOR_ITEM',port=encode(T.GlobalPort(self.source.unit,self.item)))
        facts = self.source.facts
        constants = [(op,output.port) for op in self.literals for output in op.outputs
            if output.port.ordinal == 0 and output.port.completion is C.FALLTHROUGH
            and self.transport.path(output.port,point) is not None]
        if constants:
            op,port = _only(constants,'prediction operand has more than one original constant source')
            occurrence = port.template.occurrence
            node = self.syntax.nodes[occurrence]
            _require(node.constructor == 'Constant','original constant constructor is not a source scalar')
            schema = next(schema for schema in self.syntax.profile.ast_schema if schema.name == node.constructor)
            value = node.fields[schema.fields.index('value')].scalar.to_python()
            _require(type(value) in (int,float),'speculative arithmetic requires an exact numeric constant')
            return dict(kind='LITERAL',value=value,original=encode(op.entry),span=self.span(occurrence))
        operations = [(i,row) for i,row in enumerate(facts.operations)
            if self.transport.path(self._point(row.result),point) is not None]
        if operations:
            i,row = _only(operations,'prediction operand has multiple original arithmetic producers')
            _require(row.action is F.LanguageOperationKind.BINARY and len(row.operands) == 2
                and row.selector in (F.OperatorSelector.ADD,F.OperatorSelector.DIV,F.OperatorSelector.MULT),
                'prediction requires an unsupported original operation')
            return dict(kind='BINARY',operator=row.selector.value,
                arguments=[descend(self._point(k)) for k in row.operands],
                original=encode(self.references.fact(facts,'operations',i)),
                span=self.span(self._point(row.result).template.occurrence))
        _,write,binding = self.transport.binding(point)
        _require(write.value in self.formals.values() or write.fact.locator in self.local_definitions,
            'prediction reads a binding without a selected same-iteration or entry definition')
        return dict(kind='LEXICAL_VALUE',binding=binding,value=descend(write.value))


def _shadow_inputs(view,source,flow,transport,item,formals,locals_,report,topology,proof):
    _,entry,entry_observation = _interface(report,topology['loop']['ingress'])
    _,edit,edit_observation = _interface(report,topology['copies_and_consumers']['region_interface'])
    _,_,_,edit_formals = _body(view,edit.body)
    diag = edit_formals[proof['region']['inputs']['D']]
    diag_route = _only((r for r in edit.parameters if r.formal.port == diag and r.kind is T.RouteKind.ARGUMENT
        and len(r.actuals) == 1),'no original per-iteration diagonal route')
    source_result = _only((r for r in edit.parameters if r.kind is T.RouteKind.ARGUMENT
        and len(r.actuals) == 1 and r != diag_route
        and r.actuals[0] not in {decode(T.Route,row['region_argument_route']).actuals[0]
            for row in topology['live_ins'].values()}), 'no original per-iteration RHS route')
    expressions = _Expressions(view,source,flow,transport,formals,item,locals_)
    rhs, = source_result.actuals
    diag_value, = diag_route.actuals
    rhs_expr = expressions.expression(rhs.port)
    scale = expressions.expression(diag_value.port)
    def collect(node):
        if node['kind'] == 'CALLER_FORMAL':
            return {decode(T.GlobalPort,node['port'])}
        if node['kind'] == 'LEXICAL_VALUE':
            return collect(node['value'])
        return set().union(*(collect(child) for child in node.get('arguments',())))
    rhs_formals,scalar_formals = collect(rhs_expr),collect(scale)
    formals_used = rhs_formals | scalar_formals
    incoming = [encode(r) for r in entry.parameters if r.formal in formals_used and r.kind is T.RouteKind.ARGUMENT]
    _require(len(incoming) == len(formals_used), 'prediction formals do not match the original caller entry parameters')
    names = {port:name for name,port in formals.items()}
    count = _only(scalar_formals,'expected one original bound parameter for scalar prediction')
    load = _only(rhs_formals-scalar_formals,'expected one original load parameter for RHS prediction')
    code = entry_observation['observation']['target_code']
    bodies = {role:row['caller_binding']['namespace']['name'] for role,row in topology['live_ins'].items()}
    roles = dict(matrix=bodies['matrix'],boundary=bodies['selected_rows'],load=names[load.port],count=names[count.port])
    _require(set(roles.values()) == set(code['parameters']),'original first-entry capture has a different formal ABI')
    iterator_name = _only((row['namespace']['name'] for row in locals_
        if transport.path(item,decode(G.Port,row['definition']['value'])) is not None),
        'iteration item has no original current-iteration lexical receiver')
    _require(scale['kind'] == 'LEXICAL_VALUE' and scale['value']['kind'] == 'BINARY',
        'original diagonal argument is not produced by its own scalar definition')
    bytecode = edit_observation['observation']['caller_site']['position']
    preparation = dict(row_edit_call_span=[bytecode[i] for i in (0,2,1,3)],
        scale_expression_span=scale['value']['span'],iteration_variable=iterator_name,
        scale_binding=scale['binding']['namespace']['name'],original_rhs_route=encode(source_result),
        original_diagonal_route=encode(diag_route),rhs_expression=rhs_expr,diag_expression=scale)
    return dict(source_body=dict(path=code['filename'],code_key=code['key'],parameters=code['parameters'],
        parameter_roles=roles),preparation=preparation,shadow_inputs=dict(caller_code=code,
        inherited_live_ins=topology['live_ins'],rhs_and_scalar_entry_routes=incoming,
        require_owned_current_snapshots=True))


def _project(pipeline):
    topology = read_artifact(pipeline['topology_scope'])
    original = read_artifact(pipeline['source_scope'])
    proof = read_artifact(topology['region_proof'])
    report = read_artifact(topology['source_graph'])
    program = read_artifact(report['program'])
    catalog = program['f4']['catalog']
    reference = artifact_reference(catalog['path'],'scar.source-declared-catalog')
    _require((reference['sha256'],reference['size_bytes']) == (catalog['sha256'],catalog['bytes']),
        'original sparse loop catalog changed')
    view = ProgramView(DeclaredSourceCatalog(Path(catalog['path']).parent))
    body = decode(T.Body,original['f4_loop']['source_body'])
    source,function,flow,formals = _body(view,body)
    transport,start,item,stops,adjacency,dependencies = _loop_control(view,source,function,flow,pipeline['loop'])
    locals_,live_ins,inside = _lexical_recurrences(flow,transport,start,adjacency,stops)
    inputs = _shadow_inputs(view,source,flow,transport,item,formals,locals_,report,topology,proof)
    _, solve, _ = _interface(report,topology['copies_and_consumers']['consumer_interface'])
    publication = _accumulator(view,source,flow,transport,report,inputs['source_body']['code_key'],inside,solve.result)
    return dict(dependences={**dependencies,'current_iteration_lexical_definitions':locals_,
        'loop_live_in_references':live_ins,'ordered_publication':publication},**inputs)


def _contract(solve):
    return dict(mode='DETACHED_NATIVE_RESULT_PREDICTION_WITH_ORDERED_ORIGINAL_CONSUMPTION',
        eligible_worker_payload=['OWNED_CSR_INPUT_SNAPSHOT','OWNED_LOAD_ROWS_AND_EXACT_SCALARS',
            'PRIVATE_ROW_EDIT_AND_NATIVE_SOLVE','PRIVATE_RESULT_AND_STATUS'],
        compute_guard=['ADMITTED_PURE_OR_PROCESS_ISOLATED_SPECULATIVE_PROVIDER',
            'NO_WORKER_PYTHON_CALLBACKS_LOGS_WARNINGS_RNG_OR_EXTERNAL_IO',
            'CURRENT_WORKER_FP_AND_ABI_MATCH_ORIGINAL_RAW_CARRIER'],
        original_raw_dispatch=solve['source_call'],
        acceptance_key=['ENTRY_SCOPE_AND_ORIGINAL_ITERATION','EXACT_CARRIER_AND_PROVIDER_REVISION',
            'DIMENSIONS_NNZ_FLAG_LAYOUT_DTYPE_OPTIONS','CURRENT_MATRIX_DATA_INDICES_INDPTR_RHS_BYTES'],
        comparison='EXACT_BEFORE_ANY_RESULT_PUBLICATION',
        original_effects_preserved=['ORIGINAL_ITERATOR_AND_WORLD_ORDER','ORIGINAL_PYTHON_ROW_EDIT_AND_NORMALIZATION',
            'ORIGINAL_LOGGER_MESSAGES_AND_WARNING_STACKLEVEL','ORIGINAL_STATUS_TO_EXCEPTION_OR_RESULT',
            'ORIGINAL_ACCUMULATOR_UPDATES_AND_RETURN'],
        fallback='ORIGINAL_SERIAL_NATIVE_CALL_FOR_MISSING_INVALID_OR_FAILED_PREDICTION',
        advanced_work_has_no_visible_publication=True, raw_inputs_always_revalidated=True,
        independent_scope_owned_state_required=True, frame_identity_alone_is_not_a_valid_key=True)


def derive_sparse_parallel_scope(pipeline_scope):
    """Replay the original sparse loop and expose conditional private work."""
    _require(pipeline_scope.get('schema') == PIPELINE,'expected an admitted original sparse pipeline scope')
    pipeline = check_sparse_pipeline_scope(pipeline_scope)
    solve = read_artifact(pipeline['solve_scope'])
    projected = _project(pipeline)
    return dict(schema=SCHEMA,schema_version=1,pipeline_scope=pipeline_scope,entry=pipeline['entry'],
        source_scope=pipeline['source_scope'],topology_scope=pipeline['topology_scope'],solve_scope=pipeline['solve_scope'],
        loop=pipeline['loop'],source_graph=pipeline['source_graphs']['row_edit'],**projected,
        speculation_contract=_contract(solve),
        frontiers=['NATIVE_AND_PYTHON_CALLEE_TRANSITIVE_EFFECTS_REMAIN_OPEN',
            'ORIGINAL_LOGGING_AND_WARNINGS_CAN_INVOKE_USER_CALLBACKS',
            'RNG_GLOBAL_STATE_AND_EXTERNAL_IO_ARE_NOT_CLOSED_BY_THE_ABSENCE_OF_F4_DATA_BACKEDGES',
            'LEXICAL_BINDING_DOMINANCE_SAYS_NOTHING_ABOUT_REFERENT_MUTATION',
            'DETACHED_PROVIDER_NUMERIC_AND_STATUS_EQUIVALENCE_REQUIRES_ITS_OWN_CONTRACT'],
        qualification=dict(shared_F4_loop_replayed=True,iterations_unrolled=False,
            sole_ordered_effect_proven=False,serial_fallback_preserved=True,transformation_authorized=False))


def check_sparse_parallel_scope(reference):
    """Cold-rederive the original graph and verify the entire scope receipt."""
    from scar.analysis.replay_transaction import current_transaction

    transaction = current_transaction()
    if transaction is not None and 'uri' in reference:
        key = tuple(reference[name] for name in ('uri', 'sha256', 'size_bytes', 'schema'))
        return transaction.memo('scar.sparse-parallel-scope/v1', key,
            lambda: _check_sparse_parallel_scope(reference), capture_reads=True)
    return _check_sparse_parallel_scope(reference)


def _check_sparse_parallel_scope(reference):
    result = read_artifact(reference) if 'uri' in reference else reference
    _require((result.get('schema'),result.get('schema_version')) == (SCHEMA,1),'expected sparse parallel scope')
    expected = derive_sparse_parallel_scope(result['pipeline_scope'])
    canonical = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    _require(canonical(result) == canonical(expected),'sparse parallel scope differs from original cold replay')
    return expected
