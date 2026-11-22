"""Resumption flow."""

from __future__ import annotations

from collections.abc import Mapping
from scar.ir.program.ids import BlockID, OperationID, RegionID
from scar.ir.program.model import (
    Block,
    Operation,
    Region,
    TerminatorKind,
    LiteralOrigin,
    OperationResult,
    ReferenceValue,
)
from scar.ir.literals import LiteralKind, PythonLiteral
from scar.ir.program.literal_sources import ExactLiteralOrigin
from scar.ir.program.protocols import HostTerminal, OwnedUse
from scar.ir.program.resumption import ResumableKind


# Small derived queries over a validated resumable body's ordinary CFG.
_BODY_OUTCOMES = frozenset({
    TerminatorKind.SUSPEND,
    TerminatorKind.RETURN,
    TerminatorKind.RAISE,
})


def first_body_outcome_operations(
    region: Region,
    operations: Mapping[OperationID, Operation],
    start: BlockID,
    *,
    blocks: Mapping[BlockID, Block] | None = None,
    block_regions: Mapping[BlockID, RegionID] | None = None,
) -> tuple[OperationID, ...]:
    """Return terminal body operations reached before the next suspension.

    The caller supplies the already validated graph's operation mapping. Every
    ordinary CFG successor is followed, including INVOKE and typed C0/C1
    routing. A SUSPEND is an outcome for this drive; its temporal successors
    are deliberately not traversed. Operation IDs are sorted only to make the
    result stable; that order is not an execution order.
    """
    if blocks is None:
        blocks = {block.id: block for block in region.blocks}

    def in_region(block_id):
        return block_id in blocks and (block_regions is None or block_regions.get(block_id) == region.id)

    if type(start) is not BlockID:
        raise TypeError("body outcome start must be BlockID")
    if not in_region(start):
        raise ValueError("body outcome start must belong to the region")

    pending = [start]
    seen_blocks: set[BlockID] = set()
    found: set[OperationID] = set()

    while pending:
        block_id = pending.pop()
        if block_id in seen_blocks:
            continue
        seen_blocks.add(block_id)
        block = blocks[block_id]
        if not block.operations:
            raise ValueError("validated body block must have a terminator")
        terminator = operations[block.operations[-1]].terminator
        if terminator is None:
            raise ValueError("validated body block must end in a terminator")
        if terminator.kind in _BODY_OUTCOMES:
            found.add(block.operations[-1])
            continue
        for successor in terminator.successors:
            if not in_region(successor.block):
                raise ValueError("validated body CFG successor leaves its region")
            if successor.block not in seen_blocks:
                pending.append(successor.block)

    return tuple(sorted(found, key=lambda identity: identity.wire))


# Exact created-close site and its original no-body host result routes.
def _none_result(ctx, source, interface):
    use = ctx.owned_use(source.operation, source.use)
    value = ctx.values[use.value]
    if type(value.origin) is LiteralOrigin:
        return value.origin.value is None
    if type(value.origin) is ExactLiteralOrigin:
        return value.origin.literal.kind is LiteralKind.NONE
    if isinstance(value, ReferenceValue) and value.id in ctx.driver_roles(interface.region)[0]:
        return False  # Under this native body profile, the receiver is not None.
    known = []
    for row in ctx.graph.literal_evaluations:
        if not ctx.spend('created_close_literals'):
            return None
        if type(row.literal) is PythonLiteral and ctx.forwards(row.site, source) is True:
            known.append(row.literal.kind is LiteralKind.NONE)
    return known[0] if known and all(item == known[0] for item in known) else None


def created_close_view(ctx, scope):
    """Derived conditional routes; missing contents or native applicability stay gaps."""
    from .protocol_validation import host_terminals
    test = next((row for row in ctx.graph.created_close_tests
                 if row.operation == scope.subject.operation), None)
    if test is None:
        return {'test': None, 'routes': (), 'gaps': ('created_close_site_unbound',)}
    interface = ctx.interfaces[scope.subject.region]
    body = ctx.bodies[interface.body]
    if body.kind is not ResumableKind.GENERATOR:
        return {'test': test, 'routes': (), 'gaps': ('created_close_family_not_implemented',)}
    result = ctx.values[test.result].origin
    routes, gaps = [], []
    for finish in ctx.graph.no_body_finishes:
        if not ctx.spend('created_close_finishes'):
            gaps.append('protocol_budget_exhausted')
            break
        if ctx.region_of[finish.operation] != interface.region:
            continue
        branch = ctx.operations[finish.precondition.operation]
        if ctx.forwards(result, OwnedUse(branch.id, branch.terminator.condition)) is not True:
            continue
        operation = ctx.operations[finish.operation]
        # The native mark's real state result is a route anchor, not a Python
        # result payload or an invented completion.
        terminals = host_terminals(ctx, OperationResult(operation.id, 0))
        if not terminals:
            gaps.append('created_close_host_route_unproved')
        for terminal in terminals:
            host = ctx.operations[terminal.operation]
            uses = ctx.terminal_uses(terminal)
            if host.terminator.kind is not TerminatorKind.RETURN:
                gaps.append('created_close_cleanup_exception_route_unproved')
                verdict = None
            elif len(uses) != 1:
                verdict = False
            else:
                verdict = _none_result(ctx, uses[0], interface)
            if verdict is None:
                gaps.append('created_close_none_result_unproved')
            routes.append({'finish': finish, 'precondition': finish.precondition,
                'terminal': terminal, 'payload_uses': uses, 'declared_none_result': verdict})
    if not routes:
        gaps.append('created_close_finish_or_result_unbound')
    return {'test': test, 'operation': ctx.operations[test.operation], 'routes': tuple(routes),
            'gaps': tuple(dict.fromkeys(gaps)), 'source_runtime_verified': False,
            'transformation_authorized': False}


__all__ = [
    'first_body_outcome_operations',
    'created_close_view',
]
