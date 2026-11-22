"""Source function try."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import replace
from scar.analysis.environment_lowering import StructuredEnvironmentBuilder
from scar.analysis.program_lowering import _Unsupported, _REF
from scar.analysis.static.lowering.source_module_lowering import PropagationOperands, PropagationModuleBuilder
from scar.ir.program.bindings import BindingState
from scar.ir.program.control import (
    ControlScope,
    ControlScopeKind as S,
    ControlPort,
    LanguageCompletionKind as L,
    ProtocolValueKind as P,
    ProtocolValueType,
    PendingDispatchTerminator,
    PendingSuccessor,
    PendingPayload,
)
from scar.ir.program.cleanup import (
    CleanupInterface,
    CleanupEnter,
    CleanupResume,
    CleanupReplace,
    CleanupRouteTerminator,
    CleanupRouteSuccessor,
    HandledContextDomain,
)
from scar.ir.program.ids import ControlScopeID, ControlPortID, OperationID, ValueID, UseID
from scar.ir.program.model import (
    Behavior,
    Operation,
    OperationResult,
    ProtocolValue,
    ReferenceValue,
    StateVersionValue,
    StateAccess,
    AccessMode,
    Footprint,
    FootprintKind,
    ExitKind,
    Terminator,
    TerminatorKind,
    Use,
    UseRole,
    DataValue,
    LocalBindingValue,
    ValueKind,
    ValueType,
    BlockArgument,
    CompletionResult,
)
from scar.ir.program.source import GapKind, SourceRole, UseSource
from scar.ir.program.structured import PythonProtocolKind as K, ControlEdgeReference
from scar.ir.program.source_language import SourceLanguageCapability as C, FUNCTION_TRY_PROTOCOLS


# Opt-in synchronous function Try over existing C0/C1 control records.
class FunctionTryBuilder(StructuredEnvironmentBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_scope = self.current_owner = None
        self.scope_specs, self.block_owners, self.control_names = {}, {}, {}
        self.incoming_states = {}
        self.control_ports, self.exception_packets, self.pending_completions = [], [], []
        self.cleanup_interfaces, self.cleanup_enters, self.cleanup_resumes, self.cleanup_replacements = [], [], [], []
        self.exception_matches, self.failure_sinks = [], []
        self.try_depth = 0
        self.handler_expression_depth = 0
        self.completion_sinks = []
        self.return_shims = {}
        self.completion_sources = {}
        self.loop_scopes = {}
        node = self.functions[self.entry]
        args = node.args
        self.formal_names = {x.arg for x in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        self.formal_names.update(x.arg for x in (args.vararg, args.kwarg) if x is not None)

    @property
    def frame_region(self):
        return self.function_context

    def new_block(self, *args, **kwargs):
        cursor = super().new_block(*args, **kwargs)
        if self.root_scope is None:
            self.root_scope = self.fresh(ControlScopeID, 'function-control-root')
            self.current_owner = self.root_scope
            self.scope_specs[self.root_scope] = (S.FUNCTION, None, self.source(self.functions[self.entry]))
        self.block_owners[cursor.block] = self.current_owner
        return cursor

    @contextmanager
    def in_owner(self, owner):
        before = self.current_owner
        self.current_owner = owner
        try:
            yield
        finally:
            self.current_owner = before

    new_scope = PropagationModuleBuilder.new_scope
    packet = PropagationModuleBuilder.packet
    pending = PropagationModuleBuilder.pending

    def emit(self, cursor, operation, node, role, old=None):
        super().emit(cursor, operation, node, role, old=old)
        kind = {'pack-propagation-pair': K.FUNCTION_EXCEPTION_PACKET,
                'pack-language-pending': K.FUNCTION_PENDING_COMPLETION,
                'handler-context-enter': K.FUNCTION_HANDLED_CONTEXT_ENTER,
                'cleanup-resume': K.FUNCTION_CLEANUP_RESUME,
                'cleanup-replace': K.FUNCTION_CLEANUP_REPLACE}.get(operation.label)
        if kind is not None:
            self.protocol(operation.id, kind, node)

    def use(self, operation, value, node, role, index, *, semantic_value=None):
        if type(self.values_by_id[value]) is ProtocolValue:
            use = Use(self.fresh(UseID, 'protocol-use'), value, UseRole.PROTOCOL)
            self.use_links.append(UseSource(operation, use.id, self.source(node), role, index, semantic_value))
            return use
        return super().use(operation, value, node, role, index, semantic_value=semantic_value)

    def control_block(self, node, *, extra_types=()):
        names = tuple(sorted(self.locals))
        cursor = self.new_block(node, named=names, extra_types=extra_types)
        self.control_names[cursor.block] = names
        for name in names:
            if name in self.formal_names:
                self.known_binding_states[cursor.env[name]] = BindingState.BOUND
        return cursor

    def carry(self, cursor, target, node):
        values = []
        for index, name in enumerate(self.control_names[target.block]):
            value = cursor.env[name]
            values.append(self.use(self._transfer_id, value, node, 'carried-binding', index))
            self.incoming_binding(target, name, value)
        return values

    def incoming_binding(self, target, name, value):
        states = self.incoming_states.setdefault((target.block, name), [])
        states.append(self.known_binding_states.get(value))
        if states[0] is not None and all(state is states[0] for state in states):
            self.known_binding_states[target.env[name]] = states[0]
        else:
            self.known_binding_states.pop(target.env[name], None)

    def jump(self, cursor, target, names, node):
        for name in names:
            self.incoming_binding(target, name, cursor.env[name])
        return super().jump(cursor, target, names, node)

    def jump_values(self, cursor, target, values, node):
        from scar.ir.program.model import Successor
        identity = self.fresh(OperationID, 'function-control-jump')
        self._transfer_id = identity
        uses = self.carry(cursor, target, node)
        uses += [self.use(identity, value, node, 'forwarded', i) for i, value in enumerate(values)]
        uses.append(self.use(identity, cursor.state, node, 'current-world', 0))
        self.emit(cursor, Operation(identity, 'function-control-jump', (Behavior.CONTROL,), tuple(uses),
            terminator=Terminator(TerminatorKind.JUMP, (Successor(target.block, tuple(x.id for x in uses)),))), node, SourceRole.CONTROL)
        return identity

    def raise_payload_types(self):
        return (_REF, _REF)

    def failure_payload(self, values):
        if len(values) != 2:
            raise ValueError('function Try needs this completion exception/traceback pair')
        return PropagationOperands(*values)

    def exception_reference(self, payload):
        return payload.exception

    def failure_references(self, payload):
        return payload.exception, payload.traceback

    def read_local(self, node, cursor, old):
        token = cursor.env[node.id]
        if self.known_binding_states.get(token) is not BindingState.BOUND:
            # Conditional-expression joins use their own ordinary jump helper;
            # derive their actual binding carries with the same canonical
            # fixed-point implementation rather than guessing BOUND.
            from scar.ir.program.interface_validation import _compute_binding_facts
            from scar.ir.program.model import LocalBindingValue,BlockArgument
            tokens={k:v for k,v in self.values_by_id.items()if type(v)is LocalBindingValue}
            incoming={k:[]for k,v in tokens.items()if type(v.origin)is BlockArgument}
            for operation in self.operations:
                own={u.id:u.value for u in operation.uses}
                for edge in getattr(operation.terminator,'successors',()):
                    for operand,formal in zip(edge.arguments,self.block_arguments[edge.block]):
                        if formal in incoming and operand in own and own[operand]in tokens and tokens[formal].slot==tokens[own[operand]].slot:
                            incoming[formal].append(own[operand])
            facts=_compute_binding_facts(tokens,{r.value:r for r in self.binding_definitions},incoming)
            if facts.get(token)==frozenset((BindingState.BOUND,)):
                self.known_binding_states[token]=BindingState.BOUND
        if self.known_binding_states.get(token) is not BindingState.BOUND:
            # A merge can genuinely contain BOUND and UNBOUND. Keep its real
            # BindingRead NORMAL/RAISE split; do not invent a reaching binding.
            # The canonical binding checker additionally requires the paired
            # failure to reach its own actual ExceptionPacketDefinition.
            from scar.ir.program.model import BindingRead
            normal, value, identity = self.invoke(cursor, node,
                ((token, node, 'local-binding', 0, None),), old=old)
            self.binding_reads.append(BindingRead(identity,
                self.operation_records[identity].uses[0].id, value))
            return normal, value
        return super().read_local(node, cursor, old)

    @contextmanager
    def handler_expression(self):
        self.handler_expression_depth += 1
        try:
            yield
        finally:
            self.handler_expression_depth -= 1

    def old_operation(self, node):
        records = self.old_by_atom.get(self.source(node).atom_id, ())
        if self.handler_expression_depth and not records:
            return None
        return super().old_operation(node)

    def expr(self, node, cursor):
        if self.handler_expression_depth and not self.old_by_atom.get(self.source(node).atom_id, ()):
            from scar.analysis.static.lowering.source_heap_lowering import expression_for_heap_target
            return expression_for_heap_target(self, node, cursor)
        if isinstance(node, ast.GeneratorExp):
            if self.has_language('GENEXPR_CREATION_314_V1'):
                from scar.analysis.static.lowering.source_generator_create import lower_paired_creation
                return lower_paired_creation(self,node,cursor)
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node,
                'FUNCTION_TRY_RESUMABLE_PROPAGATION_ABI_UNIMPLEMENTED')
        if isinstance(node, ast.Call) and (any(isinstance(arg, ast.Starred) for arg in node.args)
                                         or any(kw.arg is None for kw in node.keywords)):
            from scar.analysis.static.lowering.source_call_lowering import lower_kwargs_call
            return lower_kwargs_call(self, node, cursor)
        return super().expr(node, cursor)

    def invoke_protocol(self, cursor, node, values, kind, *, result_type=_REF, failure_handler=None):
        if kind in (K.RERAISE, K.RAISE_EXCEPTION, K.RAISE_WITH_CAUSE, K.FUNCTION_START_PROPAGATION):
            inputs = tuple((value, node, 'protocol-input', i, None) for i, value in enumerate(values))
            normal, payload, operation = self.invoke_results(cursor, node, inputs,
                result_types=(_REF, _REF), failure_handler=failure_handler)
            self.protocol(operation, kind, node)
            return normal, PropagationOperands(*payload), operation
        return super().invoke_protocol(cursor, node, values, kind, result_type=result_type,
                                       failure_handler=failure_handler)

    def raise_landing(self, node):
        with self.in_owner(self.root_scope):
            landing = self.control_block(node, extra_types=(ProtocolValueType(P.EXCEPTION_PACKET),))
        packet = self.block_arguments[landing.block][-2]
        roots = self.frame_exit_references(landing, node, packet)
        bindings = tuple(row for row in roots if type(self.values_by_id[row[0]]) is not ProtocolValue
                         and type(self.values_by_id[row[0]]) is not ReferenceValue)
        if bindings:
            self.frame_transition(landing, node, bindings, K.FUNCTION_FRAME_EXIT_BINDINGS)
        roots = tuple(row for row in roots if row not in bindings)
        self.frame_transition(landing, node, roots,
                              K.FUNCTION_FRAME_EXIT_EXCEPTION_PACKET)
        identity = self.fresh(OperationID, 'function-packet-exit')
        uses = (self.use(identity, packet, node, 'exception-packet', 0),
                self.use(identity, landing.state, node, 'post-world', 0))
        self.emit(landing, Operation(identity, 'function-packet-exit', (Behavior.CONTROL,), uses,
            terminator=Terminator(TerminatorKind.RAISE, exit='raise', values=tuple(x.id for x in uses))), node, SourceRole.EXCEPTION)
        port = ControlPort(self.fresh(ControlPortID, 'function-raise-port'), self.root_scope, L.RAISE, block=landing.block)
        self.control_ports.append(port)
        return port

    def completion_port(self, node, kind, handler=None):
        parent = self.scope_specs[self.current_owner][1] if self.scope_specs[self.current_owner][0] is S.CLEANUP else self.root_scope
        if kind is L.RETURN:
            with self.in_owner(self.root_scope):
                landing = self.control_block(node, extra_types=(_REF,))
            payload = self.block_arguments[landing.block][-2]
            if handler is not None and handler.name:
                self.clear_exception_name(handler, landing)
            super().finish_exit(landing,node,payload,ExitKind.NORMAL)
        else:
            if not self.loop_stack:
                raise _Unsupported(GapKind.CONTROL_FLOW,node,'loop completion has no original active loop')
            loop = self.loop_stack[-1]
            scope,parent = self.loop_scopes[id(loop.node)]
            with self.in_owner(scope):
                landing = self.control_block(node)
            if handler is not None and handler.name:
                self.clear_exception_name(handler,landing)
            if kind is L.BREAK:
                if loop.iterator is not None:
                    self.frame_transition(landing,node,((loop.iterator,node,'broken-iterator'),),K.FRAME_RELEASE_TEMPORARY)
                loop.breaks.append(self.jump(landing,self.loop_after(loop),loop.names,node))
            else:
                edge=self.jump(landing,loop.header,loop.names,node)
                loop.backedges.append(edge);loop.continues.append(edge)
        # This is an internal post-cleanup continuation. The source Return or
        # loop transfer is the actual ordinary terminator at its landing.
        port=ControlPort(self.fresh(ControlPortID,'function-completion-continuation'),
            self.root_scope if kind is L.RETURN else scope,L.FALLTHROUGH,block=landing.block)
        self.control_ports.append(port);self.completion_sources[port.id]=node
        if kind is L.RETURN:self.return_shims[port.id]=landing
        return port

    def dispatch(self, cursor, node, token, ports, *, payload=None):
        identity = self.fresh(OperationID, 'function-pending-dispatch')
        selector = self.use(identity, token, node, 'pending-selector', 0)
        uses, edges = [selector], []
        for index, port in enumerate(ports):
            target = self._cursor_for(port.block)
            self._transfer_id = identity
            carried = self.carry(cursor, target, node)
            state = self.use(identity, cursor.state, node, 'dispatch-world', index)
            uses.extend(carried); uses.append(state)
            arguments = tuple(x.id for x in carried)
            if port.id in self.return_shims:
                need_payload=self.use(identity,payload,node,'retained-return-reference',0)
                uses.append(need_payload);arguments+=(need_payload.id,)
            elif port.kind in (L.RETURN, L.RAISE):
                arguments += (PendingPayload(),)
            edges.append(PendingSuccessor(port.id, target.block, arguments + (state.id,)))
        self.emit(cursor, Operation(identity, 'function-pending-dispatch', (Behavior.CONTROL,), tuple(uses),
            terminator=PendingDispatchTerminator(selector.id, tuple(edges))), node, SourceRole.CONTROL)
        return identity

    def frame_transition(self,cursor,node,references,kind):
        if kind in (K.FRAME_DISCARD,K.FRAME_RELEASE_TEMPORARY) and len(references)>1:
            for reference in references:
                super().frame_transition(cursor,node,(reference,),kind)
            return
        return super().frame_transition(cursor,node,references,kind)

    def for_statement(self,node,cursor):
        parent=self.current_owner;scope=self.new_scope(S.LOOP,node,parent)
        self.loop_scopes[id(node)]=(scope,parent)
        with self.in_owner(scope):
            return super().for_statement(node,cursor)

    def while_statement(self,node,cursor):
        parent=self.current_owner;scope=self.new_scope(S.LOOP,node,parent)
        self.loop_scopes[id(node)]=(scope,parent)
        with self.in_owner(scope):
            return super().while_statement(node,cursor)

    def loop_after(self,loop):
        if loop.after is None:
            scope,parent=self.loop_scopes[id(loop.node)]
            with self.in_owner(parent):
                return super().loop_after(loop)
        return loop.after

    def _cursor_for(self, block):
        from scar.analysis.program_lowering import _Cursor
        env = dict(zip(self.control_names[block], self.block_arguments[block]))
        return _Cursor(block, env, self.block_arguments[block][-1])

    def finish_exit(self, cursor, node, value, kind, *, binding=None):
        if kind is ExitKind.NORMAL:
            if self.completion_sinks:
                return self.completion_sinks[-1][0](cursor,node,L.RETURN,value)
            return super().finish_exit(cursor, node, value, kind, binding=binding)
        if binding is not None:
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, 'FUNCTION_TRY_UNBOUND_LOCAL_EXIT_UNIMPLEMENTED')
        if type(value) is not PropagationOperands:
            cursor, value, _ = self.invoke_protocol(cursor, node, (value,), K.FUNCTION_START_PROPAGATION)
        if self.failure_sinks:
            return self.failure_sinks[-1](cursor, value)
        port = self.raise_landing(node)
        token = self.pending(cursor, node, port, self.packet(cursor, node, value))
        return self.dispatch(cursor, node, token, (port,))

    def route_to_cleanup(self, cursor, node, token, port, target, extra=()):
        identity = self.fresh(OperationID, 'function-cleanup-route')
        selector = self.use(identity, token, node, 'pending-selector', 0)
        self._transfer_id = identity
        carried = self.carry(cursor, target, node)
        forward = self.use(identity, token, node, 'entry-pending', 0)
        extra_uses = tuple(self.use(identity, value, node, 'caught-reference', i) for i, value in enumerate(extra))
        state = self.use(identity, cursor.state, node, 'current-world', 0)
        self.emit(cursor, Operation(identity, 'function-cleanup-route', (Behavior.CONTROL,),
            tuple([selector, *carried, forward, *extra_uses, state]), terminator=CleanupRouteTerminator(selector.id,
                (CleanupRouteSuccessor(port.id, target.block, tuple(x.id for x in carried) + (forward.id,) + tuple(x.id for x in extra_uses) + (state.id,)),))), node, SourceRole.CONTROL)

    def enter_cleanup(self, entry, node, cleanup, *, trailing_references=0):
        token = self.block_arguments[entry.block][-2-trailing_references]
        identity = self.fresh(OperationID, 'function-context-enter')
        use = self.use(identity, token, node, 'entry-pending', 0)
        before = self.use(identity, entry.state, node, 'context-world', 0)
        saved = ProtocolValue(self.fresh(ValueID, 'saved-context'), self.function_context, OperationResult(identity, 0), P.SAVED_HANDLED_CONTEXT)
        after = StateVersionValue(self.fresh(ValueID, 'context-world'), self.function_context, OperationResult(identity, 1), self.world)
        for value in (saved, after):
            self.add_value(value); self.link_value(value, node, SourceRole.EXCEPTION)
        self.emit(entry, Operation(identity, 'handler-context-enter', (Behavior.CONTROL, Behavior.READ, Behavior.WRITE),
            (use, before), (saved.id, after.id),
            (StateAccess(self.world, before.id, AccessMode.READ_WRITE, after.id, Footprint(FootprintKind.WHOLE)),), effects_complete=False), node, SourceRole.CONTROL)
        self.cleanup_enters.append(CleanupEnter(identity, cleanup, use.id, saved.id)); entry.state = after.id
        return token, saved.id

    def cleanup_exit(self, cursor, node, cleanup, original, selected, context, ports, *, resume=False, payload=None):
        parent = self.scope_specs[cleanup][1]
        with self.in_owner(cleanup):
            action = self.control_block(node)
        self.jump_values(cursor, action, (), node)
        with self.in_owner(parent):
            types=(ProtocolValueType(P.PENDING_COMPLETION),)+((_REF,)if payload is not None else())
            corridor = self.control_block(node, extra_types=types)
            finals = {port.id:self.control_block(node,extra_types=types) for port in ports}
        identity = self.fresh(OperationID, 'function-context-restore')
        pending = self.use(identity, original, node, 'entry-pending', 0)
        replacement = None if resume else self.use(identity, selected, node, 'replacement-pending', 0)
        context_use = self.use(identity, context, node, 'saved-context', 0)
        before = self.use(identity, action.state, node, 'current-world', 0)
        after = StateVersionValue(self.fresh(ValueID, 'cleanup-post-world'), self.function_context, OperationResult(identity, 0), self.world)
        self.add_value(after); self.link_value(after, node, SourceRole.STATE)
        uses = (pending, context_use, before) if resume else (pending, replacement, context_use, before)
        self.emit(action, Operation(identity, 'cleanup-resume' if resume else 'cleanup-replace',
            (Behavior.CONTROL, Behavior.READ, Behavior.WRITE), uses, (after.id,),
            (StateAccess(self.world, before.id, AccessMode.READ_WRITE, after.id, Footprint(FootprintKind.WHOLE)),), effects_complete=False), node, SourceRole.CONTROL)
        action.state = after.id; jump = self.jump_values(action, corridor, (selected,)+((payload,)if payload is not None else()), node)
        record = (CleanupResume(identity, cleanup, pending.id, context_use.id, ControlEdgeReference(jump, 0)) if resume
            else CleanupReplace(identity, cleanup, pending.id, replacement.id, context_use.id, ControlEdgeReference(jump, 0)))
        (self.cleanup_resumes if resume else self.cleanup_replacements).append(record)
        token = self.block_arguments[corridor.block][-3 if payload is not None else -2]
        carried_payload=self.block_arguments[corridor.block][-2]if payload is not None else None
        route = self.fresh(OperationID, 'function-post-pop-route')
        selector = self.use(route, token, node, 'pending-selector', 0); self._transfer_id = route
        successors, route_uses = [], [selector]
        for i, port in enumerate(ports):
            final=finals[port.id]
            carried = self.carry(corridor, final, node)
            forward = self.use(route, token, node, 'forwarded-pending', i)
            state = self.use(route, corridor.state, node, 'post-pop-world', i)
            extra=()if carried_payload is None else(self.use(route,carried_payload,node,'retained-completion-reference',i),)
            route_uses.extend((*carried, forward,*extra,state))
            successors.append(CleanupRouteSuccessor(port.id, final.block, tuple(x.id for x in carried) + (forward.id,)+tuple(u.id for u in extra)+(state.id,)))
        self.emit(corridor, Operation(route, 'function-post-pop-route', (Behavior.CONTROL,), tuple(route_uses),
            terminator=CleanupRouteTerminator(selector.id, tuple(successors))), node, SourceRole.CONTROL)
        for port in ports:
            final=finals[port.id]
            token=self.block_arguments[final.block][-3 if payload is not None else -2]
            ref=self.block_arguments[final.block][-2]if payload is not None else None
            self.dispatch(final,node,token,(port,),payload=ref)

    def _try_shape(self, node):
        if self.try_depth or (node.handlers and node.finalbody):
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, 'FUNCTION_TRY_NESTED_OR_COMBINED_UNIMPLEMENTED')
        for child in ast.walk(node):
            if child is not node and isinstance(child, (ast.Try, ast.TryStar, ast.Yield, ast.YieldFrom, ast.Await)):
                raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, child, 'FUNCTION_TRY_NONLOCAL_TRANSFER_OR_NESTED_UNIMPLEMENTED')
        if not node.finalbody and not node.handlers:
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, 'FUNCTION_TRY_WITHOUT_CLEANUP_UNIMPLEMENTED')

    @contextmanager
    def unwind_temporaries(self,cursor,node,base):
        held=tuple(self.active_temporaries[base:]);del self.active_temporaries[base:]
        try:
            for reference in reversed(held):
                self.frame_transition(cursor,node,((reference,node,'function-exception-unwind'),),K.FRAME_RELEASE_TEMPORARY)
            yield
        finally:
            self.active_temporaries.extend(held)

    def bind_exception_name(self, handler, reference, cursor):
        name = handler.name
        if name not in self.locals or name in self.cell_references:
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, handler,
                'FUNCTION_TRY_NONLOCAL_EXCEPTION_ALIAS_UNIMPLEMENTED')
        previous = cursor.env[name]
        self.frame_transition(cursor, handler, ((previous, handler, 'previous-exception-binding'),
            (reference, handler, 'caught-exception-binding')), K.FRAME_BIND)
        self.define_binding(name, cursor, handler, reference)
        return cursor

    def clear_exception_name(self, handler, cursor):
        cursor = self.bind_exception_name(handler, self.literal(handler, None), cursor)
        self.frame_transition(cursor, handler, ((cursor.env[handler.name], handler, 'cleared-exception-binding'),),
                              K.FUNCTION_CLEAR_EXCEPTION_NAME)
        self.define_binding(handler.name, cursor, handler)
        return cursor

    def match_handler(self, handler, current, caught_ref, entry):
        from scar.ir.program.exception_matches import ExceptionMatcher, ExceptionHandlerOperand
        from scar.ir.program.state_addresses import ExistingAccess
        from scar.ir.program.model import BlockArgument
        with self.handler_expression():
            current, handler_ref = self.expr(handler.type, current)
        self.frame_transition(current, handler, ((handler_ref, handler.type, 'handler-type-hold'),), K.FRAME_HOLD_TEMPORARY)
        self.active_temporaries.append(handler_ref)
        def invalid_type(failed, payload):
            held = self.active_temporaries.pop()
            try:
                self.frame_transition(failed, handler, ((held, handler.type, 'handler-type-release'),), K.FRAME_RELEASE_TEMPORARY)
                self.finish_exit(failed, handler, payload, ExitKind.RAISE)
            finally:
                self.active_temporaries.append(held)
        current, validated, operation = self.invoke(current, handler,
            ((handler_ref, handler.type, 'handler-type', 0, None),), failure_handler=invalid_type)
        self.protocol(operation, K.VALIDATE_EXCEPT_TYPE_314, handler)
        identity = self.fresh(OperationID, 'borrowed-function-exception-match')
        exc = self.use(identity, caught_ref, handler, 'caught-exception', 0)
        selected = self.use(identity, validated, handler.type, 'dynamic-handler', 0)
        world = self.use(identity, current.state, handler, 'match-world', 0)
        boolean = DataValue(self.fresh(ValueID, 'exception-match'), self.function_context,
                            OperationResult(identity, 0), 'builtins.bool')
        self.add_value(boolean); self.link_value(boolean, handler, SourceRole.TRUTH)
        self.emit(current, Operation(identity, 'borrowed-function-exception-match',
            (Behavior.COMPUTE, Behavior.READ, Behavior.CONTROL), (exc, selected, world), (boolean.id,),
            (StateAccess(self.world, world.id, AccessMode.READ, footprint=Footprint(FootprintKind.WHOLE)),), effects_complete=True), handler, SourceRole.TRUTH)
        self.protocol(identity, K.MATCH_EXCEPTION_BORROWED_314, handler)
        self.active_temporaries.pop()
        self.frame_transition(current, handler, ((handler_ref, handler.type, 'handler-type-release'),), K.FRAME_RELEASE_TEMPORARY)
        yes, no = self.new_block(handler), self.new_block(handler)
        self.branch(current, boolean.id, yes, no, handler)
        self.exception_matches.append(ExceptionMatcher(identity, exc.id,
            BlockArgument(entry.block, len(self.control_names[entry.block]) + 1),
            ExceptionHandlerOperand(selected.id, self.values_by_id[validated].origin),
            OperationResult(identity, 0), None, self.operations[-1].id, (ExistingAccess(identity, 0),)))
        return yes, no

    def try_except(self, node, cursor):
        self._try_shape(node)
        parent, after = self.current_owner, None
        body_scope = self.new_scope(S.BLOCK, node, parent)
        gate_scope = self.new_scope(S.BLOCK, node, parent)
        cleanup = self.new_scope(S.CLEANUP, node, parent)
        with self.in_owner(body_scope):
            body = self.control_block(node)
        with self.in_owner(gate_scope):
            gate = self.control_block(node, extra_types=(_REF, _REF))
        with self.in_owner(cleanup):
            entry = self.control_block(node, extra_types=(ProtocolValueType(P.PENDING_COMPLETION), _REF))
        self.cleanup_interfaces.append(CleanupInterface(cleanup, gate_scope, entry.block, self.block_arguments[entry.block][-3]))
        self.jump_values(cursor, body, (), node)
        caught_edges = [];protected_holds=len(self.active_temporaries)
        def caught(failed, payload):
            caught_edges.append(failed.block)
            with self.unwind_temporaries(failed,node,protected_holds):
                self.jump_values(failed, gate, (payload.exception, payload.traceback), node)
        self.try_depth += 1
        with self.in_owner(body_scope):
            self.failure_sinks.append(caught)
            try:
                normal = self.statements(node.body, body)
            finally:
                self.failure_sinks.pop()
        if not caught_edges:
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node,
                'FUNCTION_TRY_NO_RAISING_PROTECTED_OPERATION_UNIMPLEMENTED')
        if normal is not None:
            with self.in_owner(parent):
                else_entry = self.control_block(node)
            self.jump_values(normal, else_entry, (), node)
            normal = self.statements(node.orelse, else_entry)
            if normal is not None:
                with self.in_owner(parent):
                    after = self.control_block(node)
                self.jump_values(normal, after, (), node)
        original_port = self.raise_landing(node)
        pair = PropagationOperands(*self.block_arguments[gate.block][-3:-1])
        with self.in_owner(gate_scope):
            token = self.pending(gate, node, original_port, self.packet(gate, node, pair))
            self.route_to_cleanup(gate, node, token, original_port, entry, (pair.exception,))
        with self.in_owner(cleanup):
            original, saved = self.enter_cleanup(entry, node, cleanup, trailing_references=1)
            caught_ref = self.block_arguments[entry.block][-2]
            selection_holds=len(self.active_temporaries)
            def replaced(failed, payload):
                with self.unwind_temporaries(failed,node,selection_holds):
                    port = self.raise_landing(node)
                    fresh = self.pending(failed, node, port, self.packet(failed, node, payload))
                    self.cleanup_exit(failed, node, cleanup, original, fresh, saved, (port,))
            self.failure_sinks.append(replaced)
            current = entry
            try:
                for handler in node.handlers:
                    if handler.type is None:
                        selected, current = current, None
                    else:
                        selected, current = self.match_handler(handler, current, caught_ref, entry)
                    body_owner = self.new_scope(S.BLOCK, handler, cleanup)
                    with self.in_owner(body_owner):
                        handler_body = self.control_block(handler)
                        self.jump_values(selected, handler_body, (), handler)
                        if handler.name:
                            handler_body = self.bind_exception_name(handler, caught_ref, handler_body)
                        hold_base = len(self.active_temporaries)
                        def body_failed(failed, payload):
                            held = tuple(self.active_temporaries[hold_base:])
                            del self.active_temporaries[hold_base:]
                            try:
                                for reference in reversed(held):
                                    self.frame_transition(failed, handler, ((reference, handler, 'function-exception-unwind'),), K.FRAME_RELEASE_TEMPORARY)
                                if handler.name:
                                    self.clear_exception_name(handler, failed)
                                replaced(failed, payload)
                            finally:
                                self.active_temporaries.extend(held)
                        def body_completed(completed,source,kind,value):
                            port=self.completion_port(source,kind,handler)
                            token=self.pending(completed,source,port)
                            self.cleanup_exit(completed,handler,cleanup,original,token,saved,(port,),payload=value if kind is L.RETURN else None)
                        self.failure_sinks.append(body_failed)
                        self.completion_sinks.append((body_completed,len(self.loop_stack)))
                        try:
                            completed = self.statements(handler.body, handler_body)
                        finally:
                            self.completion_sinks.pop();self.failure_sinks.pop()
                        if completed is not None:
                            if after is None:
                                with self.in_owner(parent):
                                    after = self.control_block(node)
                            with self.in_owner(parent):
                                target = self.control_block(handler) if handler.name else after
                            port = ControlPort(self.fresh(ControlPortID, 'handler-fallthrough'), parent, L.FALLTHROUGH, block=target.block)
                            self.control_ports.append(port)
                            selected_pending = self.pending(completed, handler, port)
                            self.cleanup_exit(completed, handler, cleanup, original, selected_pending, saved, (port,))
                            if handler.name:
                                with self.in_owner(parent):
                                    self.clear_exception_name(handler, target)
                                    self.jump_values(target, after, (), handler)
                    if current is None:
                        break
                if current is not None:
                    self.cleanup_exit(current, node, cleanup, original, original, saved, (original_port,), resume=True)
            finally:
                self.failure_sinks.pop(); self.try_depth -= 1
        return after

    def try_finally(self, node, cursor):
        self._try_shape(node)
        parent = self.current_owner
        protected = self.new_scope(S.BLOCK, node, parent)
        cleanup = self.new_scope(S.CLEANUP, node, parent)
        with self.in_owner(protected):
            body = self.control_block(node)
        with self.in_owner(cleanup):
            entry = self.control_block(node, extra_types=(ProtocolValueType(P.PENDING_COMPLETION),_REF))
        self.cleanup_interfaces.append(CleanupInterface(cleanup, protected, entry.block, self.block_arguments[entry.block][-3]))
        self.jump_values(cursor, body, (), node)
        ports = [];protected_holds=len(self.active_temporaries)
        def failed(failure, payload):
            with self.unwind_temporaries(failure,node,protected_holds):
                port = self.raise_landing(node); ports.append(port)
                token = self.pending(failure, node, port, self.packet(failure, node, payload))
                self.route_to_cleanup(failure, node, token, port, entry,(payload.exception,))
        def protected_completed(completed,source,kind,value):
            port=self.completion_port(source,kind);ports.append(port)
            token=self.pending(completed,source,port)
            anchor=value if kind is L.RETURN else next(iter(self.environment_references.values()))
            self.route_to_cleanup(completed,node,token,port,entry,(anchor,))
        self.try_depth += 1
        with self.in_owner(protected):
            self.failure_sinks.append(failed);self.completion_sinks.append((protected_completed,len(self.loop_stack)))
            try:
                normal = self.statements(node.body, body)
            finally:
                self.completion_sinks.pop();self.failure_sinks.pop()
        after = None
        if normal is not None:
            with self.in_owner(parent):
                after = self.control_block(node)
            port = ControlPort(self.fresh(ControlPortID, 'finally-fallthrough'), parent, L.FALLTHROUGH, block=after.block)
            self.control_ports.append(port); ports.append(port)
            with self.in_owner(protected):
                token = self.pending(normal, node, port)
                self.route_to_cleanup(normal, node, token, port, entry,(next(iter(self.environment_references.values())),))
        with self.in_owner(cleanup):
            original, saved = self.enter_cleanup(entry, node, cleanup,trailing_references=1)
            original_carrier=self.block_arguments[entry.block][-2]
            def replaced(failure, payload):
                port = self.raise_landing(node)
                token = self.pending(failure, node, port, self.packet(failure, node, payload))
                self.cleanup_exit(failure, node, cleanup, original, token, saved, (port,))
            def cleanup_completed(completed,source,kind,value):
                port=self.completion_port(source,kind)
                token=self.pending(completed,source,port)
                self.cleanup_exit(completed,node,cleanup,original,token,saved,(port,),payload=value if kind is L.RETURN else None)
            self.failure_sinks.append(replaced);self.completion_sinks.append((cleanup_completed,len(self.loop_stack)))
            try:
                completed = self.statements(node.finalbody, entry)
            finally:
                self.completion_sinks.pop();self.failure_sinks.pop();self.try_depth -= 1
            if completed is not None:
                self.cleanup_exit(completed, node, cleanup, original, original, saved, tuple(ports), resume=True,payload=original_carrier)
        return after

    def statements(self, nodes, cursor):
        for node in nodes:
            if (cursor is not None and isinstance(node,(ast.Break,ast.Continue)) and self.completion_sinks
                    and len(self.loop_stack)<=self.completion_sinks[-1][1]):
                kind=L.BREAK if isinstance(node,ast.Break)else L.CONTINUE
                self.completion_sinks[-1][0](cursor,node,kind,None);cursor=None
            elif cursor is not None and isinstance(node, ast.Try):
                cursor = (self.try_except(node, cursor) if node.handlers else self.try_finally(node, cursor))
            elif cursor is not None and isinstance(node, ast.AugAssign):
                cursor = self.augmented_assignment(node, cursor)
            else:
                cursor = super().statements((node,), cursor)
        return cursor

    def augmented_assignment(self, node, cursor):
        """Read a local or saved heap target, RHS, in-place, then store back.

        The result may alias the old reference or be a new object; neither is
        presumed. __iop__/fallback and mutation stay inside the world INVOKE.
        Heap targets reuse separately saved receiver/index ports, never a
        second evaluation of their original target subtree.
        """
        from scar.analysis.static.lowering.source_heap_lowering import expression_for_heap_target
        from scar.analysis.static.lowering.source_call_lowering import _invoke
        heap = isinstance(node.target, (ast.Attribute, ast.Subscript)) and self.has_language('HEAP_AUGASSIGN_314_V1')
        if not heap and (not isinstance(node.target, ast.Name) or node.target.id not in self.locals
                or node.target.id in self.cell_references):
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node.target,
                'FUNCTION_TRY_AUGMENTED_HEAP_TARGET_UNIMPLEMENTED')
        baseline = len(self.active_temporaries)
        def hold(at, value):
            self.frame_transition(cursor, at, ((value, at, 'augassign-held-reference'),), K.FRAME_HOLD_TEMPORARY)
            self.active_temporaries.append(value)
        try:
            if heap:
                from scar.analysis.static.lowering.source_heap_lowering import augmented_heap_target
                cursor, receiver, key, left = augmented_heap_target(self, node.target, cursor)
            else:
                cursor, left = expression_for_heap_target(self, node.target, cursor)
            if cursor is None:
                return None
            hold(node.target, left)
            cursor, right = self.expr(node.value, cursor)
            if cursor is None:
                return None
            hold(node.value, right)
            cursor, result, _ = _invoke(self, cursor, node,
                ((left, node.target, 'augassign-left', 0, None),
                 (right, node.value, 'augassign-right', 1, None)),
                K.FUNCTION_INPLACE_BINARY,
                ((node.target, left), (node.value, right)), result=True)
            self.gap(GapKind.EFFECT_SUMMARY, node,
                'in-place/fallback dispatch and original assignment are connected; aliasing, mutation and finalizer effects remain unproved')
            if heap:
                from scar.analysis.static.lowering.source_heap_lowering import _hold, _release, store_prepared_heap_target
                _hold(self, cursor, node.target, result)
                cursor = store_prepared_heap_target(self, node.target, result, cursor, receiver, key)
                _release(self, cursor, node.target, result)
                if key is not None:
                    _release(self, cursor, node.target.slice, key)
                _release(self, cursor, node.target.value, receiver)
                self.gap(GapKind.EFFECT_SUMMARY, node,
                    'saved heap target/read/in-place/storeback and normal/raise worlds are connected; physical primitive cleanup and enclosing lifetime remain unproved')
                return cursor
            return self.store(node.target, result, cursor)
        finally:
            del self.active_temporaries[baseline:]

    def build_function(self):
        node = super().build_function()
        packet = ProtocolValueType(P.EXCEPTION_PACKET)
        self.regions = [replace(region, exits=tuple(replace(port, types=(packet, self.state_type))
            if port.kind is ExitKind.RAISE else port for port in region.exits))
            if region.id == self.function_context else region for region in self.regions]
        return node

    def make_graph(self):
        graph = super().make_graph()
        scopes = tuple(ControlScope(identity, self.function_context, parent, kind, source,
            tuple(block for block in self.function_blocks if self.block_owners[block] == identity))
            for identity, (kind, parent, source) in self.scope_specs.items()
            if any(self.block_owners[block] == identity for block in self.function_blocks))
        return replace(graph, control_scopes=scopes, control_ports=tuple(self.control_ports),
            exception_packets=tuple(self.exception_packets), pending_completions=tuple(self.pending_completions),
            handled_context_domains=(HandledContextDomain(self.function_context, self.world),) if self.cleanup_interfaces else (),
            cleanup_interfaces=tuple(self.cleanup_interfaces), cleanup_enters=tuple(self.cleanup_enters),
            cleanup_resumes=tuple(self.cleanup_resumes), cleanup_replacements=tuple(self.cleanup_replacements),
            exception_matches=tuple(self.exception_matches))


# Source checks for the opt-in function propagation/C0/C1 adapter.
def exception_handoff_holds(block_id, operations, protocols, blocks, use_links):
    """Read actual unwind Uses up to this validated exception's cleanup entry.

    This query does not run a handler or certify finalizers/whole-frame
    lifetime. It follows only the single exception corridor already checked
    by the canonical C0/C1 and function-source consumers.
    """
    cursor, seen, released = (block_id, 0), set(), []
    while cursor not in seen:
        seen.add(cursor)
        block = blocks[cursor[0]]
        if cursor[1] >= len(block.operations):
            raise ValueError('function exception handoff falls out of block')
        op = operations[block.operations[cursor[1]]]
        marker = protocols.get(op.id)
        kind = marker.kind if marker is not None else None
        first = use_links.get((op.id, op.uses[0].id)) if op.uses else None
        if kind is K.FRAME_RELEASE_TEMPORARY and first is not None and first.role == 'function-exception-unwind':
            released.append(op.uses[0].value)
        if kind in (K.FRAME_EXIT, K.FUNCTION_FRAME_EXIT_EXCEPTION_PACKET):
            retained = tuple(u.value for u in op.uses
                if (link := use_links.get((op.id, u.id))) is not None and link.role == 'held-temporary')
            return retained + tuple(reversed(released))
        if kind is K.FUNCTION_HANDLED_CONTEXT_ENTER:
            return tuple(reversed(released))
        term = op.terminator
        if term is None:
            cursor = block.id, cursor[1] + 1
        elif term.kind in (TerminatorKind.JUMP, TerminatorKind.PENDING_DISPATCH, TerminatorKind.CLEANUP_ROUTE) and len(term.successors) == 1:
            cursor = term.successors[0].block, 0
        else:
            raise ValueError('function exception handoff lost its single canonical continuation')
    raise ValueError('function exception handoff cycles without a cleanup endpoint')


def validate_function_try_protocols(model, graph, interface, metadata, semantic, language, *, source_tree=None):
    def need(value, detail):
        if not value:
            raise ValueError('function Try: ' + detail)
    actual = {r.operation for r in model.protocols if r.kind.value in FUNCTION_TRY_PROTOCOLS}
    enabled = language is not None and C.FUNCTION_TRY_314_V1 in language.capabilities
    need(language is None or C.HEAP_AUGASSIGN_314_V1 not in language.capabilities or enabled,
         'heap AugAssign requires the existing function propagation capability')
    if not enabled:
        need(not actual, 'new control protocol requires its explicit capability')
        return set()
    need(isinstance(source_tree, ast.Module), 'exact retained source AST required')
    need(not graph.frame_roots and not graph.resumable_creates, 'function capability cannot inherit mapping/resumable ABI')
    values = {r.id: r for r in graph.values}
    operations = {r.id: r for r in graph.operations}
    blocks = {b.id: b for region in graph.regions for b in region.blocks}
    owners = {b: s.id for s in graph.control_scopes for b in s.blocks}
    scopes = {r.id: r for r in graph.control_scopes}
    links = {r.operation: r for r in metadata.operation_links}
    operation_blocks = {identity:b.id for b in blocks.values() for identity in b.operations}
    protocols = {r.operation: r for r in model.protocols}
    uses = {u.id: u for op in graph.operations for u in op.uses}
    roots = [r for r in graph.control_scopes if r.parent is None]
    need(len(roots) == 1 and roots[0].kind is S.FUNCTION and roots[0].source == model.source,
         'exact original function control root required')
    need(all(r.region == graph.root and r.kind in (S.FUNCTION, S.BLOCK, S.CLEANUP, S.LOOP) for r in scopes.values()),
         'only this function local control scopes are implemented')
    state = ValueType(ValueKind.STATE, location=values[interface.world].location)
    ref = ValueType(ValueKind.REFERENCE)
    pairs, incoming = set(), {block: [] for block in blocks}
    normalizers = (K.RERAISE, K.RAISE_EXCEPTION, K.RAISE_WITH_CAUSE, K.FUNCTION_START_PROPAGATION)
    for op in graph.operations:
        if op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE:
            raised = next(s for s in op.completions if s.kind is ExitKind.RAISE)
            need(raised.types == (ref, ref, state), 'invoke lost original exception/traceback/post-world pair')
            for signature, edge in zip(op.completions, op.terminator.successors):
                if signature.kind is ExitKind.RAISE or protocols.get(op.id) is not None and protocols[op.id].kind in normalizers:
                    need(signature.types == (ref, ref, state)
                         and edge.arguments == tuple(CompletionResult(i) for i in range(3)),
                         'propagation must use both results of this exact completion')
                    pairs.add(tuple(blocks[edge.block].arguments[:2]))
        if op.terminator is not None:
            for edge in getattr(op.terminator, 'successors', ()):
                if edge.block in incoming:
                    incoming[edge.block].append((op, edge))
    # Same fixed-point provenance check as the existing module C adapter:
    # forwarding may preserve a pair, never combine two invocation failures.
    changed = True
    while changed:
        changed = False
        for block in blocks.values():
            refs = [v for v in block.arguments if type(values[v]) is ReferenceValue]
            if len(refs) != 2 or tuple(refs) in pairs or not incoming[block.id]:
                continue
            positions = [block.arguments.index(v) for v in refs]
            sources = []
            for op, edge in incoming[block.id]:
                arguments = [edge.arguments[i] for i in positions]
                own = {u.id: u for u in op.uses}
                sources.append(tuple(own[u].value for u in arguments) if all(u in own for u in arguments) else None)
            if all(pair in pairs for pair in sources):
                pairs.add(tuple(refs)); changed = True
    for packet in graph.exception_packets:
        need((uses[packet.exception].value, uses[packet.traceback].value) in pairs,
             'packet must keep one actual completion/forwarded exception-traceback pair')
    for region in graph.regions:
        for port in region.exits:
            if port.kind is ExitKind.RAISE:
                need(port.types == (ProtocolValueType(P.EXCEPTION_PACKET), state), 'function Raise boundary must carry its packet')
    expected = {}
    for record in graph.exception_packets:
        expected[values[record.value].origin.operation] = K.FUNCTION_EXCEPTION_PACKET
    for record in graph.pending_completions:
        expected[values[record.value].origin.operation] = K.FUNCTION_PENDING_COMPLETION
    for rows, kind in ((graph.cleanup_enters, K.FUNCTION_HANDLED_CONTEXT_ENTER),
                       (graph.cleanup_resumes, K.FUNCTION_CLEANUP_RESUME),
                       (graph.cleanup_replacements, K.FUNCTION_CLEANUP_REPLACE)):
        for record in rows:
            expected[record.operation] = kind
    def node_for(reference, cls):
        found = [n for n in ast.walk(source_tree) if isinstance(n, cls) and hasattr(n, 'end_lineno')
                 and (n.lineno, n.end_lineno, n.col_offset, n.end_col_offset) ==
                 (reference.start_line, reference.end_line, reference.start_column, reference.end_column)]
        need(len(found) == 1, 'scope has no unique original AST node')
        atom = semantic.source_atoms.get(reference.atom_id)
        need(atom is not None and atom.reference == reference, 'scope source atom differs')
        return found[0]
    def within(owner, ancestor):
        while owner is not None:
            if owner == ancestor:
                return True
            owner = scopes[owner].parent
        return False

    def check_handler(handler, scope, entry):
        handler_source = next(a.reference for a in semantic.source_atoms.values() if a.kind == 'ExceptHandler' and (a.reference.start_line,a.reference.end_line,a.reference.start_column,a.reference.end_column)==(handler.lineno,handler.end_lineno,handler.col_offset,handler.end_col_offset))
        matches = [r for r in graph.exception_matches if protocols[r.operation].source == handler_source]
        validations = [operations[r.operation] for r in model.protocols
                       if r.kind is K.VALIDATE_EXCEPT_TYPE_314 and r.source == handler_source]
        if handler.type is not None:
            need(len(matches) == len(validations) == 1 and uses[matches[0].input_use].value == entry.arguments[-2],
                 'handler must match its own borrowed original caught reference')
            normal = next(e for e in validations[0].terminator.successors if e.kind is ExitKind.NORMAL)
            need(uses[matches[0].handler.use].value == blocks[normal.block].arguments[0],
                 'matcher did not consume its own handler validation result')
            handler_value = validations[0].uses[0].value
            origin = values[handler_value].origin
            if type(origin) is OperationResult:
                producer = operations[origin.operation]
            else:
                producers = [op for op in graph.operations if op.terminator is not None
                    and op.terminator.kind is TerminatorKind.INVOKE
                    and any(e.kind is ExitKind.NORMAL and e.block == origin.block for e in op.terminator.successors)]
                need(len(producers) == 1, 'handler result has no unique actual evaluation')
                producer = producers[0]
            need(node_for(links[producer.id].source, ast.expr) is handler.type,
                 'handler validation did not evaluate its original AST expression')
            definition = semantic.definitions.get(links[producer.id].semantic_operation)
            if definition is None:
                # The initial explicit capability admitted only this exact
                # ordinary formal-name BindingRead, before the SG bridge.
                formals = {semantic.slots[p.slot].name:p.slot for p in interface.parameters}
                reads = [r for r in graph.binding_reads if r.operation == producer.id and r.normal_result == handler_value]
                need(isinstance(handler.type, ast.Name) and handler.type.id in formals
                     and len(reads) == 1 and values[uses[reads[0].binding].value].slot == formals[handler.type.id],
                     'legacy handler has no exact original formal binding')
            else:
                need(definition.code_id == links[producer.id].source.atom_id.wire
                     and links[producer.id].source.atom_id in definition.source_atoms,
                     'handler evaluation lost exact original SG ownership')
        else:
            need(not matches and not validations, 'bare handler must not fabricate matching/type validation')
        return matches[0] if matches else None

    for record in graph.cleanup_interfaces:
        scope, protected = scopes[record.scope], scopes[record.protected]
        atom = semantic.source_atoms[scope.source.atom_id]
        if atom.kind == 'With':
            need(C.FUNCTION_WITH_314_V1 in language.capabilities, 'With cleanup requires its explicit source capability')
            node = node_for(scope.source, ast.With)
            need(protected.source == scope.source, 'With cleanup/protected source differs')
            suites = ((protected, [*node.body, *(i.optional_vars for i in node.items if i.optional_vars is not None)]),)
            handlers = ()
        elif atom.kind == 'Try':
            node = node_for(scope.source, ast.Try)
            need(protected.source == scope.source, 'cleanup/protected original Try differs')
            if node.finalbody:
                need(not node.handlers, 'combined handler/finally protocol not implemented')
                suites = ((protected, node.body), (scope, node.finalbody))
                handlers = ()
            else:
                need(bool(node.handlers), 'exception cleanup has no original handler clauses')
                suites = ((scope, node.handlers),)
                handlers = node.handlers
        else:
            handler = node_for(scope.source, ast.ExceptHandler)
            node = node_for(protected.source, ast.Try)
            need(len(node.handlers) == 1 and node.handlers[0] is handler and not handler.name and not node.finalbody,
                 'legacy handler cleanup original source differs')
            suites = ((scope, [handler]),)
            handlers = (handler,)
        if handlers:
            gate = blocks[next(iter(protected.blocks))]
            need(len(protected.blocks) == 1, 'handler must keep its single propagation gate')
            route = operations[gate.operations[-1]]
            edges = [e for e in route.terminator.successors if e.block == record.entry]
            need(len(edges) == 1, 'handler has no exact original entry edge')
            entry = blocks[record.entry]
            token = uses[edges[0].arguments[-3]].value
            pending = next((r for r in graph.pending_completions if r.value == token), None)
            packet = next((r for r in graph.exception_packets if pending is not None
                           and r.value == uses[pending.payload].value), None)
            need(packet is not None and uses[edges[0].arguments[-2]].value == uses[packet.exception].value,
                 'handler caught-reference must come from this original pending packet')
            checked = [(handler,check_handler(handler,scope,entry)) for handler in handlers]
            def local_reachable(start):
                seen=set();todo=[start]
                while todo:
                    block=todo.pop()
                    if block in seen or not within(owners[block],scope.id):continue
                    seen.add(block);last=operations[blocks[block].operations[-1]]
                    todo.extend(e.block for e in getattr(last.terminator,'successors',()))
                return seen
            for index,(handler,matcher) in enumerate(checked):
                if matcher is None:continue
                branch=operations[matcher.branch]
                yes,no=branch.terminator.successors
                matched=local_reachable(yes.block);missed=local_reachable(no.block)
                for later,later_match in checked[index+1:]:
                    if later_match is None:continue
                    target=operation_blocks[later_match.operation]
                    need(target in missed and target not in matched,
                         'later handler evaluated on an earlier match or lost clause order')
            for handler,matcher in checked:
                if not handler.name:continue
                source=next(a.reference for a in semantic.source_atoms.values() if a.kind=='ExceptHandler'
                    and (a.reference.start_line,a.reference.end_line,a.reference.start_column,a.reference.end_column)==
                    (handler.lineno,handler.end_lineno,handler.col_offset,handler.end_col_offset))
                bound=[op for op in graph.operations if links[op.id].source==source
                    and protocols.get(op.id) is not None and protocols[op.id].kind is K.FRAME_BIND
                    and len(op.uses)==3 and op.uses[1].value==entry.arguments[-2]]
                need(len(bound)==1,'handler alias must bind this one original caught reference')
                clears={op.id for op in graph.operations if links[op.id].source==source
                    and protocols.get(op.id) is not None and protocols[op.id].kind is K.FUNCTION_CLEAR_EXCEPTION_NAME}
                need(bool(clears),'handler alias omitted all normal/exceptional name cleanup')
                start=operation_blocks[bound[0].id];todo=[(start,blocks[start].operations.index(bound[0].id)+1)];seen=set()
                while todo:
                    block,offset=todo.pop()
                    if(block,offset)in seen:continue
                    seen.add((block,offset));identities=blocks[block].operations[offset:]
                    if any(identity in clears for identity in identities):continue
                    need(block!=entry.id,'handler alias crossed a new caught entry without clearing')
                    last=operations[blocks[block].operations[-1]]
                    need(last.terminator is not None and last.terminator.kind not in (TerminatorKind.RETURN,TerminatorKind.RAISE),
                         'handler completion bypassed as-name clear/delete')
                    for edge in getattr(last.terminator,'successors',()):
                        target=blocks[edge.block]
                        need(within(owners[target.id],scope.id) or links[target.operations[0]].source==source,
                             'handler alias escaped its cleanup corridor before clearing')
                        todo.append((target.id,0))
        for source_scope, suite in suites:
            spans = [(n.lineno, n.end_lineno, n.col_offset, n.end_col_offset) for n in suite]
            for block in blocks.values():
                if not within(owners[block.id], source_scope.id):
                    continue
                for identity in block.operations:
                    source = links[identity].source
                    if source == scope.source or source == protected.source:
                        continue
                    need(source.path == scope.source.path and source.fingerprint == scope.source.fingerprint
                         and any((a, c) <= (source.start_line, source.start_column)
                             and (source.end_line, source.end_column) <= (b, d) for a, b, c, d in spans),
                         'protected/cleanup operation came from another source suite')
    # Source Return may use an internal FALLTHROUGH continuation so canonical
    # C1 route/dispatch-only blocks remain unchanged. Its real reference must
    # reach the frame-closing ordinary RETURN landing through actual SSA Uses.
    incoming = {block:[] for block in blocks}
    for op in graph.operations:
        for edge in getattr(op.terminator,'successors',()):
            incoming[edge.block].append((op,edge))
    value_sources = {}
    for row in metadata.value_links:value_sources.setdefault(row.value,[]).append(row.source)
    for pending in graph.pending_completions:
        constructor=operations[values[pending.value].origin.operation]
        source=links[constructor.id].source
        if semantic.source_atoms[source.atom_id].kind!='Return':continue
        node=node_for(source,ast.Return)
        port=next(p for p in graph.control_ports if p.id==pending.port)
        from scar.ir.program.control import LanguageCompletionKind
        need(port.kind is LanguageCompletionKind.FALLTHROUGH and port.scope==roots[0].id
             and port.block is not None and pending.payload is None,
             'source Return needs its exact internal post-cleanup continuation')
        target=blocks[port.block];last=operations[target.operations[-1]]
        need(last.terminator.kind is TerminatorKind.RETURN and links[last.id].source==source,
             'source Return continuation does not actually close/return')
        returned=uses[last.terminator.values[0]].value
        expected_node=node.value if node.value is not None else node
        expected_span=(expected_node.lineno,expected_node.end_lineno,expected_node.col_offset,expected_node.end_col_offset)
        todo=[returned];seen=set();leaves=0
        while todo:
            value=todo.pop()
            if value in seen:continue
            seen.add(value);record=values[value]
            exact=[r for r in value_sources.get(value,())if (r.path,r.fingerprint)==(source.path,source.fingerprint)
                   and (r.start_line,r.end_line,r.start_column,r.end_column)==expected_span]
            # A cleanup landing parameter can carry the Return's source span
            # without being its literal. Follow its actual incoming Uses until
            # the original None leaf is reached; a same-span label is no proof.
            from scar.ir.program.model import LiteralOrigin
            if exact and (node.value is not None or type(record.origin)is LiteralOrigin):
                if node.value is None:
                    need(type(record.origin)is LiteralOrigin and record.origin.value is None,'empty return changed None')
                leaves+=1;continue
            need(type(record.origin)is BlockArgument,'return carrier did not originate at its exact source expression')
            carried=[(operation,edge)for operation,edge in incoming[record.origin.block]
                     if not hasattr(edge,'port') or edge.port==port.id]
            need(carried,'return carrier has no actual incoming control edge for this completion')
            for operation,edge in carried:
                operand=edge.arguments[record.origin.index]
                own={u.id:u for u in operation.uses}
                need(operand in own,'return carrier cannot be reconstructed from a guessed completion payload')
                todo.append(own[operand].value)
        need(leaves>0,'return carrier has no original expression leaf')
    for block in blocks.values():
        if not block.operations:
            continue
        final = operations[block.operations[-1]]
        if final.terminator is None or final.terminator.kind is not TerminatorKind.RAISE:
            continue
        need(len(block.operations) >= 2, 'packet exit omitted its conditional frame cleanup')
        before = operations[block.operations[-2]]
        need(protocols.get(before.id) is not None and protocols[before.id].kind is K.FUNCTION_FRAME_EXIT_EXCEPTION_PACKET,
             'packet exit has no exact preceding frame protocol')
        expected[before.id] = K.FUNCTION_FRAME_EXIT_EXCEPTION_PACKET
        local_roots = tuple(v for v in block.arguments if type(values[v]) is LocalBindingValue)
        need(not local_roots or len(block.operations) >= 3,
             'exception frame omitted its current local roots')
        if len(block.operations) >= 3:
            binding_exit = operations[block.operations[-3]]
            marker = protocols.get(binding_exit.id)
            if marker is not None and marker.kind is K.FUNCTION_FRAME_EXIT_BINDINGS:
                need(all(type(values[u.value]) is LocalBindingValue for u in binding_exit.uses[:-1])
                     and tuple(u.value for u in binding_exit.uses[:-1]) == local_roots
                     and type(values[binding_exit.uses[-1].value]) is StateVersionValue
                     and not binding_exit.effects_complete
                     and set(binding_exit.behaviors) == {Behavior.READ, Behavior.WRITE, Behavior.MEMORY}
                     and len(binding_exit.accesses) == 1
                     and binding_exit.accesses[0].after == before.uses[-1].value,
                     'frame binding roots must feed the following packet exit current world')
                expected[binding_exit.id] = K.FUNCTION_FRAME_EXIT_BINDINGS
            else:
                need(not local_roots, 'exception frame bypassed exact local-binding cleanup')
        payload = values[before.uses[0].value]
        need(type(payload) is ProtocolValue and payload.protocol is P.EXCEPTION_PACKET,
             'frame exceptional payload is not the selected packet')
        need(not before.effects_complete and set(before.behaviors) == {Behavior.READ, Behavior.WRITE, Behavior.MEMORY}
             and before.terminator is None and len(before.accesses) == 1 and before.accesses[0].mode is AccessMode.READ_WRITE,
             'frame packet cleanup must advance its incomplete coarse world')
        need(all(type(values[u.value]) in (ReferenceValue, LocalBindingValue) for u in before.uses[1:-1]),
             'private pending/context token escaped with frame roots')
        own = {u.id: u for u in final.uses}
        need(tuple(own[u].value for u in final.terminator.values) == (payload.id, before.accesses[0].after),
             'packet/world differs from actual frame post-state')
    for identity in actual:
        kind = protocols[identity].kind
        if kind is K.FUNCTION_INPLACE_BINARY:
            node = node_for(protocols[identity].source, ast.AugAssign)
            operation = operations[identity]
            heap = isinstance(node.target, (ast.Attribute, ast.Subscript))
            need((isinstance(node.target, ast.Name) or heap and C.HEAP_AUGASSIGN_314_V1 in language.capabilities)
                 and isinstance(node.op,
                 (ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.FloorDiv,
                  ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd)),
                 'in-place operation needs an original supported AugAssign and target capability')
            need(operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
                 and len(operation.uses) == 3 and operation.invoke_arguments == tuple(u.id for u in operation.uses)
                 and all(type(values[u.value]) is ReferenceValue for u in operation.uses[:-1])
                 and type(values[operation.uses[-1].value]) is StateVersionValue
                 and not operation.effects_complete and operation.callee is None
                 and tuple(s.types for s in operation.completions) == ((ref, state), (ref, ref, state)),
                 'in-place operation lost its ordered references/world/completions')
            sources = {(r.operation, r.use):r for r in metadata.use_links}
            for use, original, role, index in zip(operation.uses, (node.target, node.value),
                                                ('augassign-left', 'augassign-right'), (0, 1)):
                link = sources[(identity, use.id)]
                need(node_for(link.source, ast.expr) is original and (link.role, link.index) == (role, index),
                     'in-place operands do not use the original target-before-RHS AST')
            normal = blocks[operation.terminator.successors[0].block]
            if not heap:
                assigned = [r for r in graph.binding_definitions if r.payload is not None
                            and uses[r.payload].value == normal.arguments[0]]
                need(len(assigned) == 1 and semantic.slots[values[assigned[0].value].slot].name == node.target.id
                     and semantic.slots[values[assigned[0].value].slot].owner == interface.definition,
                     'in-place result must rebind the original local slot')
            else:
                from scar.ir.semantics_v2 import Opcode
                from scar.ir.program.state_addresses import (AddressCoverage, AddressEffect, Element,
                    InvokeStateOperand, ObjectField, ReferenceOwner)
                target = node.target
                attribute = isinstance(target, ast.Attribute)
                need(not isinstance(target, ast.Subscript) or not any(isinstance(n, ast.Slice) for n in ast.walk(target.slice)),
                     'slice AugAssign has no admitted cleanup contract')
                target_source = sources[(identity, operation.uses[0].id)].source
                opcode = Opcode.ATTRIBUTE if attribute else Opcode.INDEX
                reads = [operations[r.operation] for r in metadata.operation_links
                         if r.source == target_source and r.opcode is opcode]
                stores = [operations[r.operation] for r in model.protocols
                          if r.source == target_source and r.kind is (K.STORE_ATTRIBUTE if attribute else K.STORE_SUBSCRIPT)
                          and operations[r.operation].uses[-2].value == normal.arguments[0]]
                need(len(reads) == len(stores) == 1, 'heap AugAssign needs one original target load and result storeback')
                load, store = reads[0], stores[0]
                saved_count = 1 if attribute else 2
                saved = tuple(u.value for u in load.uses[:-1])
                need(len(saved) == saved_count and load.terminator is not None
                     and load.terminator.kind is TerminatorKind.INVOKE and not load.effects_complete
                     and load.invoke_arguments == tuple(u.id for u in load.uses)
                     and tuple(s.types for s in load.completions) == ((ref,state),(ref,ref,state))
                     and tuple(u.value for u in store.uses[:saved_count]) == saved
                     and len(store.uses) == saved_count + 2,
                     'heap AugAssign re-evaluated or changed its saved receiver/key')
                declaration = semantic.definitions.get(links[load.id].semantic_operation)
                need(declaration is not None and declaration.code_id == target_source.atom_id.wire
                     and target_source.atom_id in declaration.source_atoms
                     and declaration.metadata.get('role') != 'binding' and len(declaration.output_slots) == 1,
                     'heap target load lost its exact original SG evaluation definition')
                load_normal = blocks[load.terminator.successors[0].block]
                need(load_normal.arguments[0] == operation.uses[0].value,
                     'in-place left port is not the actual target load result')
                for use, original, role, index in zip(load.uses, (target.value,target.slice) if not attribute else (target.value,),
                        ('receiver','index'), (0,1)):
                    row = sources[(load.id,use.id)]
                    need(node_for(row.source,ast.expr) is original and (row.role,row.index) == (role,index),
                         'heap load saved ports differ from original receiver/key AST roles')
                # Core validation owns SSA dominance. This profile additionally
                # orders source effects: LHS load normal -> RHS -> in-place ->
                # storeback, including stateful frame-retention operations.
                predecessors = {b:set() for b in blocks}
                for op in operations.values():
                    if op.terminator is not None:
                        for edge in op.terminator.successors:
                            predecessors[edge.block].add(operation_blocks[op.id])
                region = next(r for r in graph.regions if r.id == graph.root)
                dom = {b:({b} if b == region.entry else set(blocks)) for b in blocks}
                changed = True
                while changed:
                    changed = False
                    for block in blocks:
                        if block == region.entry: continue
                        parents = predecessors[block]
                        result = {block} | (set.intersection(*(dom[p] for p in parents)) if parents else set())
                        if result != dom[block]: dom[block]=result;changed=True
                need(load_normal.id in dom[operation_blocks[identity]]
                     and normal.id in dom[operation_blocks[store.id]],
                     'heap target load/in-place/storeback order is not on the normal CFG')
                def world_after(value, anchor, seen=frozenset()):
                    if value == anchor: return True
                    if value in seen or type(values[value]) is not StateVersionValue: return False
                    origin = values[value].origin
                    if type(origin) is OperationResult:
                        before = [u.value for u in operations[origin.operation].uses
                                  if type(values[u.value]) is StateVersionValue]
                    elif type(origin) is BlockArgument:
                        before = []
                        for producer, edge in incoming[origin.block]:
                            argument = edge.arguments[origin.index]
                            if type(argument) is CompletionResult:
                                before.extend(u.value for u in producer.uses if type(values[u.value]) is StateVersionValue)
                            else:
                                before.append(uses[argument].value)
                    else: return False
                    return bool(before) and all(world_after(v,anchor,seen|{value}) for v in before)
                need(world_after(operation.uses[-1].value,load_normal.arguments[-1])
                     and world_after(store.uses[-1].value,normal.arguments[-1]),
                     'heap AugAssign bypassed the actual load/in-place normal post-world')
                need(Behavior.MEMORY in operation.behaviors,
                     'in-place mutation/fallback cannot omit memory effects')
                for block, held in ((load_normal,operation.uses[0].value),(normal,normal.arguments[0])):
                    first = operations[block.operations[0]]
                    need(protocols.get(first.id) is not None and protocols[first.id].kind is K.FRAME_HOLD_TEMPORARY
                         and len(first.uses) == 2 and first.uses[0].value == held,
                         'heap AugAssign must retain each actual load/in-place result before further effects')
                rhs_span = (node.value.lineno,node.value.col_offset),(node.value.end_lineno,node.value.end_col_offset)
                for row in metadata.operation_links:
                    span=(row.source.start_line,row.source.start_column),(row.source.end_line,row.source.end_column)
                    if rhs_span[0] <= span[0] and span[1] <= rhs_span[1]:
                        need(load_normal.id in dom[operation_blocks[row.operation]],
                             'RHS effects precede the saved target load normal completion')
                addresses = {r.site.operation:r for r in graph.state_address_refinements}
                address = addresses.get(load.id)
                selector = ObjectField(target.attr) if attribute else Element(load.uses[1].id)
                need(address is not None and address.coverage is AddressCoverage.PARTIAL
                     and address.site == InvokeStateOperand(load.id,load.uses[-1].id) and address.protocol is None
                     and address.effects == (AddressEffect(ReferenceOwner(load.uses[0].id),(selector,),AccessMode.READ),),
                     'heap target read address lost its exact receiver/key/world Uses')
                # Holds are the existing frame-retention protocol, not a new
                # pending graph or a claim about physical refcounts/finalizers.
                load_pending = exception_handoff_holds(load.terminator.successors[1].block,
                    operations,protocols,blocks,sources)
                need(len(load_pending) >= saved_count and load_pending[-saved_count:] == saved,
                     'heap load failure lost its saved target holds')
                prefix = load_pending[:-saved_count]
                for primitive, suffix in ((operation,saved),(store,saved+(normal.arguments[0],))):
                    pending = exception_handoff_holds(primitive.terminator.successors[1].block,
                        operations,protocols,blocks,sources)
                    need(pending == prefix+suffix, 'heap AugAssign failure changed its ordered retained references')
                release_values = (normal.arguments[0],)+tuple(reversed(saved))
                store_normal = blocks[store.terminator.successors[0].block]
                for key, value in zip(store_normal.operations, release_values):
                    release = operations[key]
                    need(protocols.get(key) is not None and protocols[key].kind is K.FRAME_RELEASE_TEMPORARY
                         and len(release.uses) == 2 and release.uses[0].value == value,
                         'heap AugAssign normal store lost its ordered retained-reference releases')
                need(len(store_normal.operations) >= len(release_values), 'heap store omitted retained-reference releases')
            expected[identity] = kind
        elif kind is K.FUNCTION_START_PROPAGATION:
            node_for(protocols[identity].source, ast.Assert)
            operation = operations[identity]
            constructors = [operations[r.operation] for r in model.protocols
                            if r.kind is K.MAKE_ASSERTION_ERROR and r.source == protocols[identity].source]
            need(len(constructors) == 1, 'assert propagation has no original allocation')
            edge = next(e for e in constructors[0].terminator.successors if e.kind is ExitKind.NORMAL)
            need(operation.uses[0].value == blocks[edge.block].arguments[0],
                 'assert propagation is not its actual newly allocated exception reference')
            expected[identity] = kind
        elif kind is K.FUNCTION_CLEAR_EXCEPTION_NAME:
            from scar.ir.program.bindings import BindingState
            from scar.ir.program.model import LiteralOrigin
            handler = node_for(protocols[identity].source, ast.ExceptHandler)
            op = operations[identity]
            need(handler.name is not None and len(op.uses) == 2
                 and type(values[op.uses[0].value]) is LocalBindingValue
                 and type(values[op.uses[1].value]) is StateVersionValue
                 and not op.effects_complete and op.terminator is None
                 and set(op.behaviors) == {Behavior.READ,Behavior.WRITE,Behavior.MEMORY}
                 and len(op.accesses) == 1 and op.accesses[0].mode is AccessMode.READ_WRITE,
                 'exception alias deletion lost exact binding/current world')
            bound = values[op.uses[0].value]
            slot = semantic.slots[bound.slot]
            need(slot.owner == interface.definition and slot.name == handler.name,
                 'exception alias deletion targets another source local')
            definition = next((r for r in graph.binding_definitions if r.value == bound.id), None)
            need(definition is not None and definition.state is BindingState.BOUND
                 and type(values[uses[definition.payload].value].origin) is LiteralOrigin
                 and values[uses[definition.payload].value].origin.value is None,
                 'exception alias must be overwritten with None before deletion')
            block = next(b for b in blocks.values() if identity in b.operations)
            position = block.operations.index(identity)
            need(position + 1 < len(block.operations), 'exception alias missing unbound definition')
            deleted = operations[block.operations[position+1]]
            records = [r for r in graph.binding_definitions if r.value in deleted.results]
            need(len(records) == 1 and records[0].state is BindingState.UNBOUND
                 and values[records[0].value].slot == bound.slot
                 and links[deleted.id].source == protocols[identity].source,
                 'exception alias deletion must define this same local UNBOUND')
            expected[identity] = kind
        elif kind is K.VALIDATE_EXCEPT_TYPE_314:
            node = node_for(protocols[identity].source, ast.ExceptHandler)
            op = operations[identity]
            need(node.type is not None and op.terminator is not None
                 and op.terminator.kind is TerminatorKind.INVOKE
                 and len(op.uses) == 2 and type(values[op.uses[0].value]) is ReferenceValue
                 and op.invoke_arguments == tuple(u.id for u in op.uses)
                 and not op.effects_complete and op.callee is None
                 and tuple((s.kind, s.types) for s in op.completions) ==
                     ((ExitKind.NORMAL, (ref, state)), (ExitKind.RAISE, (ref, ref, state))),
                 'handler class validation has no actual original expression operand')
            expected[identity] = kind
        elif kind is K.MATCH_EXCEPTION_BORROWED_314:
            node_for(protocols[identity].source, ast.ExceptHandler)
            need(any(row.operation == identity for row in graph.exception_matches),
                 'borrowed match has no canonical matcher authority')
            expected[identity] = kind
        need(expected.get(identity) is kind, 'control marker has no matching canonical constructor/action/source')
    need(set(expected) == actual, 'canonical C0/C1 operations are missing their exact source protocol markers')
    return actual
