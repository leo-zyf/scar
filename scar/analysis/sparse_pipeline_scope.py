"""Cold-join two admitted sparse regions through their original argument ports."""

import json
from pathlib import Path

from scar.analysis.csr_topology_scope import (
    SCHEMA as TOPOLOGY, _Transport, _body, _only, _require, check_csr_topology_scope,
)
from scar.analysis.program_view import ProgramView
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_declared_catalog import DeclaredSourceCatalog
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import data_reach
from scar.analysis.sparse_solve_scope import SCHEMA as SOLVE, check_sparse_solve_scope
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T, source_productions as P
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.sparse-pipeline-scope'


def _row(report, ref, section):
    prefix = f'#/{section}/'
    _require(ref.startswith(prefix) and ref[len(prefix):].isdecimal(), 'expected an original graph row reference')
    return report[section][int(ref[len(prefix):])]


def _interface(report, reference):
    row = _row(report,reference,'interfaces')
    attempted = _only((observation for observation in report['observed_targets']
        if observation['provider'] == row['provider'] and observation['site_id'] == row['site_id']
        and observation['observation']['evidence'] == row['frame']['selection']),
        'selected sparse pipeline call has no unique original actual attempt')
    _require(row['ordinary_body_entry'] and attempted['entry_scope_reachable'],
        'sparse pipeline call does not carry an ordinary conditional source frame')
    call = decode(T.CallLink,row['interface']['link'])
    return row, call, attempted


def _matching_call(report, old_call, old_attempt):
    identity = old_attempt['operation']
    code = old_attempt['observation']['target_code']
    candidates = []
    for i,row in enumerate(report['interfaces']):
        if row['frame']['operation'] != identity['node'] or row['frame']['caller'] != identity['namespace']:
            continue
        ref = f'#/interfaces/{i}'
        attempt = [item for item in report['observed_targets']
            if item['provider'] == row['provider'] and item['site_id'] == row['site_id']
            and item['observation']['evidence'] == row['frame']['selection']]
        if not attempt or attempt[0]['observation']['target_code'] != code:
            continue
        row,link,observed = _interface(report,ref)
        if observed['operation'] == identity and link == old_call:
            candidates.append(ref)
    return _only(candidates,'the richer native trace changed the older F4 call/CodeID or selected ABI')


def _fixed_pack(view, resolver, link):
    pack = link.argument_packs.positional
    context = resolver.plan(pack.unit,pack.port.template)
    step = context.steps[pack.port.owner]
    instruction = step.instruction
    _require(type(instruction) is P.ApplyPythonProtocol
        and instruction.protocol is P.PythonProtocolKind.BUILD_CALL_POSITIONALS
        and context.result(step) == pack,'sparse pipeline encountered an open positional expansion')
    source = view.unit(pack.unit)
    ordinal,container = _only(((i,row) for i,row in enumerate(source.facts.containers)
        if row.kind is F.ContainerKind.SEQUENCE and source_port(source.facts,row.value) == pack),
        'original fixed positional tuple lacks one F4 content fact')
    items = tuple(source_port(source.facts,i) for i in container.elements)
    _require(items == tuple(context.value(value) for value in step.arguments[:-1]),
        'fixed positional tuple no longer carries the original ordered F4 argument values')
    return items, encode(ReferenceSlicer(view.catalog).fact(source.facts,'containers',ordinal))


def _project_hop(view,resolver,caller_body,link,current):
    _,_,flow,_ = _body(view,caller_body)
    transport = _Transport(flow)
    direct = link.binding is T.ArgumentBinding.DIRECT
    if direct:
        routes = [(route,route.actuals[0]) for route in link.parameters
            if route.kind is T.RouteKind.ARGUMENT and len(route.actuals) == 1]
        pack_fact = None
    else:
        _require(link.binding is T.ArgumentBinding.SYMBOLIC_PACK,
            'only direct and fixed positional tuple routes are admitted')
        items,pack_fact = _fixed_pack(view,resolver,link)
        routes = [(projection,items[projection.index]) for projection in link.projections
            if projection.kind is T.ProjectionKind.POSITIONAL_ITEM
            and projection.pack == link.argument_packs.positional and projection.index < len(items)]
    output = {}
    details = {}
    for kind,formal in current.items():
        matches = []
        for route,actual in routes:
            if actual.unit != caller_body.unit:
                continue
            try:
                _,write,binding = transport.binding(actual.port)
            except ValueError:
                continue
            if write.value == formal.port:
                matches.append((route,binding))
        route,binding = _only(matches,'tracked original value did not reach exactly one next frame formal')
        formal_port = route.formal.port
        _require((route.formal.unit,formal_port.template,formal_port.owner,formal_port.kind) ==
            (link.body.unit,link.body.template,link.body.region,G.PortKind.PARAMETER),
            'sparse argument escaped its next original function ABI')
        output[kind] = route.formal
        details[kind] = dict(original_formal=encode(formal),original_lexical_transport=binding,
            next_original_formal=encode(route.formal),
            selector='DIRECT_ARGUMENT' if direct else 'FIXED_POSITIONAL_ITEM_UNDER_NORMAL_PACK_BINDING',
            original_route=encode(route))
    return output, dict(items=details,fixed_pack_fact=pack_fact,conditions=link.conditions,
        keyword_pack_content_proven=False)


def _native_inputs(view, report, source_call, last_body, current):
    native = _row(report,source_call['original_target_observation'],'observed_targets')
    _require(native['operation'] == source_call['call_point'], 'native target differs from the admitted F4 CALL')
    source,_,flow,_ = _body(view,last_body)
    source_call_port = decode(T.GlobalPort,native['observation']['source_call'])
    call = _only((item for item in flow.calls
        if source_port(source.facts,item.call.result) == source_call_port),
        'native sparse call lacks one original F4 operand list')
    _require(len(call.call.positional) == 7 and call.call.positional_pack is None
        and len(call.call.keywords) == 1 and call.call.keywords[0][0] == 'options',
        'installed sparse carrier argument ABI differs')
    # The seven slots belong to the installed provider signature, not to the
    # Python workload or arbitrary similarly named sparse functions.
    operand_names = ('dimension','nnz','matrix_values','column_indices','row_offsets','rhs','format_flag')
    operands = {role:source_port(source.facts,i) for role,i in zip(operand_names,call.call.positional,strict=True)}
    may_routes = {}
    for role,formal in current.items():
        reached = data_reach(flow,formal,0,0,source.facts)
        originals = [x for x in reached.call_inputs if x.call == call.fact]
        selected = {'matrix': ('matrix_values','column_indices','row_offsets'), 'rhs': ('rhs',)}[role]
        _require({operands[r] for r in selected} <= {x.actual for x in originals},
            'normalized native operands lost their original formal source dependence')
        may_routes[role] = dict(formal=encode(formal),normalized_operands={r:encode(operands[r]) for r in selected},
            original_may_routes=[encode(x) for x in originals if x.actual in {operands[r] for r in selected}])
    return dict(original_call_fact=encode(call.fact),input_ports=encode(operands),
        world_before=encode(source_port(source.facts,call.call.world)),
        world_after=encode(source_port(source.facts,call.call.result_world)),
        may_routes=may_routes,
        original_effect_or_alias_equivalence_proven=False)


def _normalization(report,solve,native):
    source_call = solve['source_call']
    observed = []
    for i,row in enumerate(report['observed_targets']):
        site = row['observation']['caller_site']
        if row['entry_scope_reachable'] and site['code'] == source_call['code_key'] and site['offset'] < source_call['bytecode_offset']:
            observed.append(dict(observation=f'#/observed_targets/{i}',operation=row['operation'],
                original_bytecode_span=site['position']))
    samples = read_artifact(solve['native_observation'])['native_samples']
    return dict(order=['SPARSE_CONTAINER_CONVERSION','FORMAT_AND_RHS_SHAPE_CLASSIFICATION',
        'DUPLICATE_CANONICALIZATION','FLOAT_DTYPE_PROMOTION','SHAPE_AND_SOLVER_BRANCH',
        'INDEX_INTEGER_CAST_COPY_FALSE','NATIVE_CALL'],
        retained_original_call_targets=observed, terminal_call=native,
        sampled_local_index_cast_alias=dict(samples=[row['index'] for row in samples],
            indices_alias_local_matrix=all(row['native_indices_alias_matrix'] for row in samples),
            indptr_alias_local_matrix=all(row['native_indptr_alias_matrix'] for row in samples)),
        actual_cross_region_storage_alias='UNKNOWN', conversion_noop_proven=False,
        index_bytes_before_and_after_normalization_equivalence_proven=False,
        original_normalization_required=True,
        frontiers=['PROVIDER_CALLS_MAY_CHANGE_OR_REPLACE_CSR_STRUCTURE',
            'DUPLICATE_CANONICALIZATION_MAY_WRITE_CONTENT_OR_FLAGS',
            'DTYPE_PROMOTION_AND_ASARRAY_MAY_COPY_OR_REBIND',
            'INDEX_CAST_REQUIRES_CURRENT_DTYPE_AND_LAYOUT',
            'SAMPLED_LOCAL_ALIAS_IS_NOT_CROSS_REGION_ALIAS_PROOF'])


def _compose(topology, solve):
    update_graph = read_artifact(topology['source_graph'])
    solve_graph = read_artifact(solve['source_graph'])
    update_program = read_artifact(update_graph['program'])
    solve_program = read_artifact(solve_graph['program'])
    catalog = solve_program['f4']['catalog']
    metadata = artifact_reference(catalog['path'],'scar.source-declared-catalog')
    _require(update_program['f4']['catalog'] == catalog and (metadata['sha256'],metadata['size_bytes']) ==
        (catalog['sha256'],catalog['bytes']), 'bounded graphs name different retained F4 catalogs')
    view = ProgramView(DeclaredSourceCatalog(Path(catalog['path']).parent))
    copies = topology['copies_and_consumers']
    _,edit,edit_observation = _interface(update_graph,copies['region_interface'])
    _,consumer,consumer_observation = _interface(update_graph,copies['consumer_interface'])
    edit_in_solve_graph = _matching_call(solve_graph,edit,edit_observation)
    consumer_in_solve_graph = _matching_call(solve_graph,consumer,consumer_observation)
    route = solve['source_call']['original_entry_route']
    _require(consumer_in_solve_graph in route, 'edited CSR has no original solve consumer on the native route')
    downstream = route[route.index(consumer_in_solve_graph)+1:]
    current = {}
    item_routes = {}
    for item in copies['original_unpack_and_consumer']['item_routes']:
        role = 'matrix' if item['item'] == copies['original_copy_return']['private_matrix_item'] else 'rhs'
        formal = decode(T.Route,item['consumer_route']).formal
        _require(formal in {r.formal for r in consumer.parameters},'copied output does not enter the original consumer')
        current[role] = formal
        item_routes[role] = dict(return_item=item['item'],original_consumer_formal=encode(formal),
            original_unpack_to_consumer=item)
    _require(set(current) == {'matrix','rhs'}, 'joint region must carry both returned values separately')
    caller = consumer.body
    resolver = BodyInterfaceResolver(view)
    hops = []
    for interface in downstream:
        row,link,observed = _interface(solve_graph,interface)
        prior = _row(solve_graph,consumer_in_solve_graph if not hops else hops[-1]['interface'],'interfaces')
        _require(row['caller_scope'] == prior['callee_scope'],'successive pipeline call scopes are not adjacent')
        current,hop = _project_hop(view,resolver,caller,link,current)
        hops.append(dict(interface=interface,call_point=observed['operation'],
            code=observed['observation']['target_code'],**hop))
        caller = link.body
    native = _native_inputs(view,solve_graph,solve['source_call'],caller,current)
    return dict(regions=dict(row_edit=dict(call_point=edit_observation['operation'],
        interface=edit_in_solve_graph,region_proof=topology['region_proof']),
        sparse_solve=dict(call_point=solve['source_call']['call_point'],source_call=solve['source_call'])),
        value_routes=dict(region_return=encode(edit.returns),original_fixed_unpack=copies['original_unpack_and_consumer']['original_unpack'],
            producer_to_consumer=item_routes,consumer_to_solver=hops,
            solver_formals_to_normalized_native_values=native['may_routes']),
        normalization=_normalization(solve_graph,solve,native))


def derive_sparse_pipeline_scope(topology_scope, solve_scope):
    """Replay both admitted source regions and their matched shared F4 route."""
    _require((topology_scope.get('schema'),solve_scope.get('schema')) == (TOPOLOGY,SOLVE),
        'composition requires independently admitted topology and solve scope artifacts')
    topology = check_csr_topology_scope(topology_scope)
    solve = check_sparse_solve_scope(solve_scope)
    _require(topology['source_scope'] == solve['source_scope'] and topology['entry'] == solve['entry']
        and topology['loop']['original_operation'] == solve['source_provenance']['original_loop'],
        'source loop, original entry bytes, arguments or affinity differ across sparse regions')
    return dict(schema=SCHEMA,schema_version=1,topology_scope=topology_scope,solve_scope=solve_scope,
        source_scope=topology['source_scope'],entry=topology['entry'],loop=topology['loop'],
        source_graphs=dict(row_edit=topology['source_graph'],sparse_solve=solve['source_graph']),
        **_compose(topology,solve),
        composition_contract=dict(shared_normal_iteration_order=['ROW_EDIT','ORIGINAL_SPARSE_NORMALIZATION','SOLVE'],
            retain_original_normal_backedge=True,independent_key_lifetimes=True,
            row_edit_key=topology['guard_contract']['required_hit_key'],
            solver_symbolic_key=solve['guard_contract']['symbolic_key'],
            raw_keys_interchangeable=False,pre_normalization_solver_plan_validated=False,
            keep_independent_fallbacks=True,original_calls_eliminated=False),
        frontiers=['NORMAL_PACK_SIGNATURE_ADMISSION_AND_DYNAMIC_TARGET_VALIDITY_RETAINED',
            'EVERY_ITERATION_CROSS_ARG_CONTENT_VERSION_AND_ALIAS_NOT_CLOSED',
            'NORMALIZED_CSR_KEY_CANNOT_BE_REPLACED_BY_ROW_EDIT_KEY',
            'ORIGINAL_FORMAT_DTYPE_CANONICALIZATION_WARNINGS_AND_EFFECTS_MUST_REMAIN'],
        qualification=dict(original_F4_route_joined=True,unrolled_iterations=False,
            whole_loop_storage_identity_proven=False,transformation_authorized=False))


def check_sparse_pipeline_scope(reference):
    """Recompute the composition from its two original immutable parent receipts."""
    document = read_artifact(reference) if 'uri' in reference else reference
    _require((document.get('schema'),document.get('schema_version')) == (SCHEMA,1),
        'expected a sparse pipeline scope receipt')
    expected = derive_sparse_pipeline_scope(document['topology_scope'],document['solve_scope'])
    canonical = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    _require(canonical(expected) == canonical(document),'sparse pipeline scope differs from cold replay')
    return expected
