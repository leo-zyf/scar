"""Verify source-backed function creation and entry-reference origins.

All endpoints refer to original graph and compiler ports. The source context
can retain compilation products, while every relation is replayed on demand.
"""
from __future__ import annotations

import ast
from dataclasses import replace

from scar.analysis.static.compilation import (
    RetainedSourceCompilation, _reference_span, function_compiler_flags,
    function_compiler_profile, replay_function_compilation, replay_source_compilation,
    retained_compilation,
)
from scar.analysis.static.function_environment import (
    _EnvironmentFacts, _factory, _roles, validate_static_body_namespace,
)
from scar.analysis.static.namespace import normal_reference, protocol_kinds
from scar.ir.program.environments import EnvironmentRole
from scar.ir.program.function_origins import StaticFactoryEntryOrigin
from scar.ir.program.model import ExitKind, OperationResult, ReferenceValue, TerminatorKind
from scar.ir.program.source_language import decode_language_overlay
from scar.ir.program.static_calls import StaticFunctionCreationRoute
from scar.ir.v2.common import SourceReference
from scar.ir.v2.semantic import OperationKind, SemanticRelation
from scar.trace.code_snapshot import code_key


def extract_class_function_creation(body, operation, value, normal_edge):
    """Join a class-suite factory through its original lexical code path.

    Annotation factories without a user DefinitionID retain their compiler
    provenance, but cannot manufacture a SourceDefinitionReference. The
    returned route is only the creation relation, not closure/default state
    or evidence that any decorator returned this function unchanged.
    """
    from scar.ir.program.class_entries import ClassSuiteOverlay, ClassSuiteMethodOverlay
    overlay = body.class_entry
    if type(overlay) not in (ClassSuiteOverlay, ClassSuiteMethodOverlay):
        return None
    factories = [record for record in overlay.factories if record.operation == operation.id]
    if len(factories) != 1:
        raise ValueError('class factory lacks its unique original creation record')
    factory = factories[0]
    if factory.definition is None:
        return None
    context = body.context
    if factory.definition not in context.definitions:
        return None
    links = body.links_by_op.get(operation.id, ())
    qualified, _, _, _ = context.definitions[factory.definition]
    source = factory.source
    normal = normal_reference(body.index, operation.id)
    if (len(links) != 1 or links[0].source != source or qualified.source != source
            or factory.reference != value or normal != (normal_edge, value)):
        raise ValueError('class factory source/normal result differs from its original qualified body')
    declarations = [definition for definition in context.semantic.definitions.values()
        if definition.id == links[0].semantic_operation and definition.metadata.get('phase') == 'definition_time'
        and definition.code_id == source.atom_id.wire and source.atom_id in definition.source_atoms]
    definition = context.semantic.definitions.get(factory.definition)
    if (len(declarations) != 1 or definition is None or definition.metadata.get('phase') != 'call_time'
            or definition.code_id != source.atom_id.wire or source.atom_id not in definition.source_atoms):
        raise ValueError('class factory did not retain its original declaration/call-time definition')
    codes = [row for row in body.uses_by_op.get(operation.id, ()) if row.role == 'function-code']
    if len(codes) != 1:
        raise ValueError('class factory lacks its actual code Use')
    code_use = body.index.use_site(codes[0].use).use
    origin = body.index.values[code_use.value].origin
    records = [record for record in overlay.constants if isinstance(origin, OperationResult)
               and record.operation == origin.operation and record.reference == code_use.value]
    if (code_use.value != factory.code_reference or len(records) != 1
            or records[0].constant_path != factory.constant_path or len(factory.constant_path) != 1):
        raise ValueError('class factory code does not come from its original constant slot')
    compilation = retained_compilation(context, body.metadata.source)
    if compilation.layout_key != overlay.compilation.root_code_layout_key:
        raise ValueError('class factory module layout differs from retained source compilation')
    scopes = compilation.scopes
    parents = [row for row in scopes.get(_reference_span(overlay.source), ())
               if row[0] == overlay.compilation.code_path and isinstance(row[2], ast.ClassDef)]
    if len(parents) != 1 or code_key(parents[0][1]) != overlay.compilation.code_layout_key:
        raise ValueError('class factory parent code is not its original ClassDef compiler scope')
    full_path = overlay.compilation.code_path + factory.constant_path
    children = [row for row in scopes.get(_reference_span(source), ()) if row[0] == full_path
                and isinstance(row[2], (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(children) != 1:
        raise ValueError('class factory child is not a unique original lexical source function')
    _, child, node = children[0]
    parent_code = parents[0][1]
    if (parent_code.co_consts[factory.constant_path[0]] is not child
            or child.co_name != node.name or child.co_qualname != parent_code.co_qualname + '.' + node.name
            or child.co_firstlineno != min([node.lineno] + [d.lineno for d in node.decorator_list])
            or tuple(child.co_freevars) != factory.free_names):
        raise ValueError('class factory child slot/name/qualname/firstline/closure layout differs')
    return StaticFunctionCreationRoute(operation.id, normal_edge, value, code_use.id,
        origin.operation, overlay.compilation.code_layout_key, factory.constant_path, qualified)


def extract_function_creation(body, operation, value, normal_edge):
    """Join an explicit factory's code slot to its original shared body.

    This never treats APPLY_DECORATOR or an arbitrary same-source result as
    the original function. Compilation is of retained text, never exec.
    """
    if body.module is None:
        if body.class_entry is not None:
            return extract_class_function_creation(body, operation, value, normal_edge)
        return verify_nested_function_creation(body, operation, value, normal_edge)
    from scar.analysis.static.lowering.source_mapping_functions import mapping_function_plan
    context = body.context
    links = body.links_by_op.get(operation.id, ())
    if len(links) != 1 or type(links[0].source) is not SourceReference:
        raise ValueError("module factory lacks its unique original declaration source")
    source = links[0].source
    declarations = [d for d in context.semantic.definitions.values()
                    if d.id == links[0].semantic_operation and d.metadata.get("phase") == "definition_time"
                    and d.code_id == source.atom_id.wire and source.atom_id in d.source_atoms]
    definitions = [definition for definition in context.defs_by_atom.get(source.atom_id, ())
                   if context.semantic.definitions[definition].metadata.get("phase") == "call_time"]
    compilation = retained_compilation(context, body.metadata.source)
    nodes = compilation.functions.get(_reference_span(source), ())
    if len(declarations) != 1 or len(definitions) != 1 or len(nodes) != 1:
        raise ValueError("module factory declaration/body source join is not exact")
    codes = [row for row in body.uses_by_op.get(operation.id, ()) if row.role == "function-code"]
    if len(codes) != 1:
        raise ValueError("module factory does not have its original code Use")
    code_use = body.index.use_site(codes[0].use).use
    origin = body.index.values[code_use.value].origin
    if type(origin) is not OperationResult:
        raise ValueError("module factory code is not its original constant operation result")
    records = [row for row in body.module.constants if row.operation == origin.operation]
    if len(records) != 1 or len(records[0].constant_path) != 1 or records[0].source != source:
        raise ValueError("module factory code constant does not retain its exact slot/source")
    parent = compilation.root
    record = records[0]
    if (compilation.layout_key != record.code_key
            or record.code_key != body.module.declaration.compilation.code_layout_key):
        raise ValueError("module factory code-slot layout differs from retained source compilation")
    plan = mapping_function_plan(nodes[0], parent)
    if record.constant_path != (plan.child.arg,) or parent.co_consts[plan.child.arg] is not plan.child.argval:
        raise ValueError("module factory code slot does not select this source function's compiled child")
    definition = definitions[0]
    return StaticFunctionCreationRoute(operation.id, normal_edge, value, code_use.id,
        origin.operation, record.code_key, record.constant_path, context.definitions[definition][0])


def _nested_factory(*, graph, index, document, namespace, source_text, operation, value,
                    normal_edge, compilation=None, compiled_module=None):
    """Shared source/compiler/port proof; qualified source IDs are joined above."""
    if index.graph is not graph:
        raise ValueError('nested factory index belongs to another original graph')
    metadata = validate_static_body_namespace(graph, document, namespace)
    protocols = protocol_kinds(document)
    if protocols.get(operation.id) != 'MAKE_NESTED_FUNCTION':
        return None
    if index.operations.get(operation.id) != operation:
        raise ValueError('nested factory is not its original operation')
    if (operation.terminator is None or operation.terminator.kind is not TerminatorKind.INVOKE
            or tuple(edge.kind for edge in operation.terminator.successors) != (ExitKind.NORMAL, ExitKind.RAISE)
            or normal_reference(index, operation.id) != (normal_edge, value)):
        raise ValueError('nested factory changed its original paired normal/raise reference boundary')
    wire = document.get('static_compilation')
    language_wire = document.get('language')
    if wire is None or language_wire is None:
        return None
    language = decode_language_overlay(language_wire)
    if language.source != metadata.source or language.code_evidence != wire:
        raise ValueError('nested factory compiler/language source provenance differs')
    if compilation is None:
        compilation = RetainedSourceCompilation(metadata.source, source_text, compiled_module=compiled_module)
    else:
        compilation.require(metadata.source, source_text)
    compiler = wire.get('compiler')
    if not function_compiler_profile(compiler):
        return None
    if not function_compiler_flags(compiler):
        raise ValueError('nested factory has an unsupported source compiler profile')
    evidence = replay_function_compilation(compilation, namespace, wire)
    path, parent = evidence.code_path, evidence.code
    parents = [row for row in compilation.scopes.get(_reference_span(namespace.source), ())
               if row[0] == path and row[1] is parent and isinstance(row[2], ast.FunctionDef)]
    if len(parents) != 1:
        raise ValueError('nested factory parent code is not its original lexical source function')
    links = [link for link in metadata.operation_links if link.operation == operation.id]
    if len(links) != 1:
        raise ValueError('nested factory lacks its unique original declaration source')
    source = links[0].source
    if (source.path != metadata.source.path or source.fingerprint != metadata.source.fingerprint
            or links[0].opcode is not None):
        raise ValueError('nested factory declaration source differs from its parent snapshot')
    nodes = [node for node in compilation.functions.get(_reference_span(source), ())
             if isinstance(node, ast.FunctionDef)]
    if len(nodes) != 1:
        raise ValueError('nested factory lacks its exact original FunctionDef AST')
    node = nodes[0]
    from scar.analysis.program_lowering import _Unsupported
    from scar.analysis.static.lowering.source_function_definitions import compiler_plan

    try:
        _, child, _ = compiler_plan(node, parent)
    except _Unsupported:
        return None
    children = [row for row in compilation.scopes.get(_reference_span(source), ())
                if row[0] == path + (child.arg,) and row[1] is child.argval and row[2] is node]
    if len(children) != 1:
        raise ValueError('nested factory child slot does not select its exact original lexical definition')
    sources = [row for row in metadata.use_links if row.operation == operation.id and row.role == 'function-code']
    if len(sources) != 1 or sources[0].index != 0 or sources[0].source != source:
        raise ValueError('nested factory lacks its original function-code Use role/source')
    code_use = index.use_site(sources[0].use)
    if code_use.operation != operation.id or code_use.use.id not in operation.invoke_arguments:
        raise ValueError('nested factory code Use belongs to another operation')
    origin = index.values[code_use.use.value].origin
    if type(origin) is not OperationResult:
        raise ValueError('nested factory code does not come from an original constant operation')
    constant = index.operations.get(origin.operation)
    constants = [row for row in language.constants if row.operation == origin.operation]
    if (constant is None or protocols.get(origin.operation) != 'READ_CODE_CONSTANT'
            or len(constants) != 1 or code_use.use.value not in constant.results
            or constants[0].source != source or constants[0].code_key != evidence.code_layout_key
            or constants[0].constant_path != (child.arg,)
            or parent.co_consts[child.arg] is not child.argval):
        raise ValueError('nested factory code Use changed its original parent constant slot/source')
    return metadata, source, node, links[0], code_use.use.id, constants[0]


def verify_nested_function_creation(body, operation, value, normal_edge):
    """Return a verified qualified creation route, or an unconnected frontier.

    ``body.context`` supplies already verified retained source/semantic records.
    Cache only compilation products in that source context; never import or
    execute its target module. A source name alone never admits a definition.
    """
    if protocol_kinds(body.document).get(operation.id) != 'MAKE_NESTED_FUNCTION':
        return None
    context = body.context
    result = _nested_factory(graph=body.graph, index=body.index, document=body.document,
        namespace=body.namespace, source_text=context.docs['source']['text'], operation=operation,
        value=value, normal_edge=normal_edge,
        compilation=retained_compilation(context, body.metadata.source))
    if result is None:
        return None
    metadata, source, node, link, code_use, constant = result
    declarations = [row for row in context.semantic.definitions.values()
        if row.code_id == source.atom_id.wire and source.atom_id in row.source_atoms
        and row.metadata.get('phase') == 'definition_time']
    definitions = [row for row in context.semantic.definitions.values()
        if row.code_id == source.atom_id.wire and source.atom_id in row.source_atoms
        and row.metadata.get('phase') == 'call_time' and row.kind in (OperationKind.FUNCTION, OperationKind.METHOD)]
    if (len(declarations) != 1 or len(definitions) != 1
            or link.semantic_operation != declarations[0].id or definitions[0].label != node.name):
        raise ValueError('nested factory did not retain its original declaration/body definition IDs')
    slots = [context.semantic.slots[edge.target.id] for edge in context.semantic.edges.values()
             if edge.source.id == declarations[0].id and edge.relation is SemanticRelation.WRITES_SLOT]
    if len(slots) != 1 or slots[0].owner != body.namespace.definition or slots[0].name != node.name:
        raise ValueError('nested factory does not write its original parent lexical name slot')
    qualified = context.definitions.get(definitions[0].id)
    if qualified is None:
        return None
    qualified = qualified[0]
    if qualified.source != source or qualified.snapshot != metadata.source:
        raise ValueError('nested factory qualified body differs from its original source definition')
    return StaticFunctionCreationRoute(operation.id, normal_edge, value, code_use,
        constant.operation, constant.code_key, constant.constant_path, qualified)


def validate_nested_function_creation(creation, *, graph, index, document, namespace,
                                     source_text, compiled_module=None, _compilation=None):
    """Replay the shared compiler/port proof for an already source-checked ID.

    The enclosing query verifies the qualified definition's original lexical
    entry and semantic locator. This reader independently checks its source,
    factory, code Use, parent compiler path and exact relative child slot.
    """
    if type(creation) is not StaticFunctionCreationRoute:
        raise TypeError('nested factory reader requires its typed original creation route')
    creation.__post_init__()
    operation = index.operations.get(creation.factory)
    if operation is None:
        raise ValueError('nested factory reader has no original factory operation')
    result = _nested_factory(graph=graph, index=index, document=document, namespace=namespace,
        source_text=source_text, operation=operation, value=creation.normal_value,
        normal_edge=creation.normal_edge, compiled_module=compiled_module, compilation=_compilation)
    if result is None:
        raise ValueError('nested factory has no connected supported source/compiler proof')
    metadata, source, _, _, code_use, constant = result
    if (creation.definition.source != source or creation.definition.snapshot != metadata.source
            or creation.code_use != code_use or creation.code_constant_operation != constant.operation
            or creation.code_layout_key != constant.code_key or creation.constant_path != constant.constant_path):
        raise ValueError('nested factory relation changed its original qualified compiler/source endpoints')
    return True


def _source_proof(creator, body, creation, source_text, compilation):
    snapshot = creator.metadata.source
    if body.metadata.source != snapshot or creation.definition.snapshot != snapshot:
        raise ValueError("factory origin retained text/source versions differ")
    if compilation is None:
        compilation = RetainedSourceCompilation(snapshot, source_text)
    else:
        compilation.require(snapshot, source_text)
    namespace_root = creation.definition.context.namespace_root
    parent = replay_source_compilation(creator, compilation, namespace_root)
    child = replay_source_compilation(body, compilation, namespace_root)
    if (creation.code_layout_key != parent.code_layout_key
            or child.code_path != parent.code_path + creation.constant_path):
        raise ValueError("factory origin code Use/constant path does not select this body")
    scopes = compilation.scopes
    matches = [row for row in scopes.get(_reference_span(creation.definition.source), ())
        if row[0] == child.code_path and row[1] is child.code
        and isinstance(row[2], (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(matches) != 1:
        raise ValueError("factory origin child is not its exact lexical source/CodeRef occurrence")
    if creator.namespace.source is not None:
        parents = [row for row in scopes.get(_reference_span(creator.namespace.source), ())
            if row[0] == parent.code_path and row[1] is parent.code]
        if len(parents) != 1:
            raise ValueError("factory origin creator path does not match its original lexical source")
    if creator.document.get("class") is not None:
        from scar.ir.program.class_entries import decode_class_entry_overlay
        overlay = decode_class_entry_overlay(creator.document["class"])
        constants = overlay.constants
    else:
        constants = getattr(creator.sources, "constants", None)
    if constants is None:
        constants = decode_language_overlay(creator.document["language"]).constants
    selected = [row for row in constants if row.operation == creation.code_constant_operation]
    if len(selected) != 1 or selected[0].constant_path != creation.constant_path:
        raise ValueError("factory origin constant record changed its original code slot/source")
    if creator.document.get("class") is not None:
        code_value = creator.index.use_site(creation.code_use).use.value
        factories = [row for row in overlay.factories if row.operation == creation.factory]
        if (overlay.compilation.code_layout_key != creation.code_layout_key
                or selected[0].reference != code_value or len(factories) != 1
                or factories[0].definition != body.namespace.definition
                or factories[0].source != body.namespace.source
                or factories[0].code_reference != code_value
                or factories[0].reference != creation.normal_value
                or factories[0].constant_path != creation.constant_path):
            raise ValueError("factory class constant/factory record differs from its exact original compiler source")
    elif (selected[0].code_key != creation.code_layout_key
            or selected[0].source != creation.definition.source):
        raise ValueError("factory origin constant record changed its original code slot/source")


def derive_factory_entry_origins(*, creator_graph, creator_document, creator_namespace,
        body_graph, body_document, body_namespace, creation, source_text,
        creator_index=None, body_index=None, _facts=None, _creator_inputs=None, _body_inputs=None,
        _compilation=None):
    """Derive only object/G reference origins, without selecting a caller.

    The supplied definition/namespace locators must already be verified by the
    enclosing reader.  Both source compilation paths are independently replayed;
    a supplied creation record or a common code name is not accepted as proof.
    """
    if type(creation) is not StaticFunctionCreationRoute:
        raise TypeError("factory origin needs its original typed creation record")
    creation.__post_init__()
    if _facts is None and (_creator_inputs is not None or _body_inputs is not None):
        raise ValueError('factory origin owned inputs require their original private owner')
    facts = _EnvironmentFacts() if _facts is None else _facts
    if type(facts) is not _EnvironmentFacts:
        raise TypeError('factory origin input owner must be _EnvironmentFacts')
    creator = facts.body(creator_graph, creator_document, creator_namespace, creator_index,
                         inputs=_creator_inputs)
    body = facts.body(body_graph, body_document, body_namespace, body_index, inputs=_body_inputs)
    if (body_namespace.definition != creation.definition.definition
            or body_namespace.source != creation.definition.source):
        raise ValueError("factory origin target differs from its original qualified definition")
    factory = _factory(creator.index, creator_document, creator.protocols, creation, inputs=creator)
    _source_proof(creator, body, creation, source_text, _compilation)
    environments = [row for row in body_graph.environments
        if row.region == body_graph.root and row.code_definition == body_namespace.definition]
    if len(environments) != 1:
        raise ValueError("factory origin body lacks its original unique function environment")
    environment = environments[0]
    members = {member.role: member for member in environment.members}
    roles = _roles(creator_document, creator.index, factory, inputs=creator)
    captures = [use for use in factory.uses if roles.get(use.id, (None,))[0] == "function-globals"]
    if len(captures) != 1:
        raise ValueError("factory origin needs its unique original globals capture Use")
    capture = captures[0]
    # Module/nested source roles use a per-role ordinal.  Class-suite source
    # roles use their ordered primitive operand position (code=0, globals=1).
    # They are different existing ABIs, not interchangeable arbitrary indices.
    expected_position = 1 if creator_document.get("class") is not None else 0
    if roles[capture.id] != ("function-globals", expected_position):
        raise ValueError("factory globals role index differs from its original source-lowering ABI")
    if creator_document.get("module") is not None or creator_document.get("class") is not None:
        roots = creator_graph.frame_roots
        if len(roots) != 1:
            raise ValueError("factory origin creator lacks its unique original frame root")
        entry = roots[0].entry
        if creator_document.get("class") is not None:
            from scar.ir.program.source_subjects import ClassFrameABI
            if type(entry) is not ClassFrameABI:
                raise ValueError("factory class origin lacks its original class frame ABI")
            expected_globals = entry.mapping.globals_namespace
        else:
            from scar.ir.program.source_subjects import MappingFramePorts
            if type(entry) is not MappingFramePorts:
                raise ValueError("factory module origin lacks its original mapping frame ABI")
            expected_globals = entry.globals_namespace
    else:
        owners = [environment for environment in creator_graph.environments
            if environment.region == creator_graph.root and environment.code_definition == creator_namespace.definition]
        members_g = [member for environment in owners for member in environment.members
            if member.role is EnvironmentRole.GLOBALS_NAMESPACE]
        if len(owners) != 1 or len(members_g) != 1:
            raise ValueError("nested factory origin lacks its original function globals interface")
        expected_globals = members_g[0].reference
    if capture.value != expected_globals:
        raise ValueError("factory globals capture does not reference the creator authoritative globals port")
    result = []
    for role, use, value in ((EnvironmentRole.FUNCTION_OBJECT, None, creation.normal_value),
            (EnvironmentRole.GLOBALS_NAMESPACE, capture.id, capture.value)):
        member = members.get(role)
        if (member is None or type(creator.index.values[value]) is not ReferenceValue
                or type(body.index.values[member.reference]) is not ReferenceValue):
            raise ValueError("factory origin must connect existing function/reference ports")
        result.append(StaticFactoryEntryOrigin(creator_namespace, creation, body_namespace,
            environment.id, member, use, value))
    return tuple(result)


def validate_factory_entry_origin(origin, **inputs):
    """Freshly replay one derived record; no successful-validation cache."""
    if type(origin) is not StaticFactoryEntryOrigin:
        raise TypeError("expected a typed original factory entry origin")
    origin.__post_init__()
    if origin not in derive_factory_entry_origins(creation=origin.creation, **inputs):
        raise ValueError("factory entry origin changed its original captured/member ports")
    return True


def factory_entry_origin_evidence(origin, *, creator_document, body_document):
    """Original graph/compiler locators, never a fictitious saved origin row."""
    if type(origin) is not StaticFactoryEntryOrigin:
        raise TypeError("expected a typed factory entry origin")
    compiler = ("#/module/declaration/compilation" if creator_document.get("module") is not None else
        "#/class/compilation" if creator_document.get("class") is not None else "#/static_compilation")
    if body_document.get("static_compilation") is None:
        raise ValueError("factory origin evidence lacks original body compilation")
    return (origin.creator.graph, origin.body.graph,
        replace(origin.creator.graph, pointer=compiler), replace(origin.body.graph, pointer="#/static_compilation"))


def factory_entry_origin_ports(*, graph, document, namespace, index=None, _facts=None, _inputs=None):
    """Identify this adapter's original interface scope, not successful origins.

    An empty explicit origin list cannot close a real MAKE normal/capture port.
    Unsupported/unselected factory sources remain interfaces with open origins.
    """
    if _facts is None and _inputs is not None:
        raise ValueError('factory origin ports need the original private input owner')
    owner = _EnvironmentFacts() if _facts is None else _facts
    if type(owner) is not _EnvironmentFacts:
        raise TypeError('factory origin port owner must be _EnvironmentFacts')
    facts = owner.body(graph, document, namespace, index, inputs=_inputs)
    incoming = frozenset(member.reference for environment in graph.environments
        if environment.region == graph.root and environment.code_definition == namespace.definition
        for member in environment.members if member.role in (
            EnvironmentRole.FUNCTION_OBJECT, EnvironmentRole.GLOBALS_NAMESPACE))
    outgoing = set()
    for operation in graph.operations:
        if facts.protocols.get(operation.id) not in ("MAKE_FUNCTION_314", "MAKE_NESTED_FUNCTION"):
            continue
        normal = normal_reference(facts.index, operation.id, blocks=facts.blocks)
        if normal is not None:
            outgoing.add(normal[1])
        roles = _roles(document, facts.index, operation, inputs=facts)
        outgoing.update(use.id for use in operation.uses if roles.get(use.id, (None,))[0] == "function-globals")
    return incoming, frozenset(outgoing)
