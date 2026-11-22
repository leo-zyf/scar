"""Stored values."""

from scar.ir.program.model import (
    CompletionResult,
    DataValue,
    InvokeLiteralResult,
    LiteralOrigin,
    OperationResult,
    ReferenceValue,
    StateVersionValue,
    value_type,
    BlockArgument,
    ExitKind,
    TerminatorKind,
)
from scar.ir.program.state_addresses import (
    CellContent,
    DynamicField,
    Element,
    ExistingAccess,
    NamespaceKey,
    ObjectField,
    ReferenceOwner,
)
from scar.ir.execution.common import Unresolved
from scar.ir.execution.coordinates import OccurrencePhase, OccurrencePoint, ValueOccurrence
from scar.ir.execution.common import OperationOccurrenceID, BlockVisitID
from scar.ir.execution.validity_budget import ValidityBudget
from collections import deque


# Demand-bound stored-value receipts over original accesses and state SSA.
#
# Primitive-slot pre/post contracts are supplied model facts. They neither
# interpret arbitrary Python attribute access nor establish stable tensor data.
# No derived copy is installed in the canonical graph or an identity union.
def _flags():
    return dict(source_runtime_verified=False, runtime_identity_verified=False,
                native_identity_verified=False, value_identity_verified=False,
                transformation_authorized=False)


def validate_request(site, occurrence, budget):
    ValidityBudget(budget)
    if type(site) not in (OperationResult, InvokeLiteralResult):
        raise TypeError('stored_value_sources requires an exact original result site')
    site.__post_init__()
    if type(occurrence) is not OperationOccurrenceID:
        raise TypeError('stored_value_sources requires an exact operation occurrence ID')


class _Sources:
    def __init__(self, ctx, shared):
        self.ctx, self.shared = ctx, shared
        self.gaps, self.cursor = [], None
        self.refinements = {row.site: row for row in ctx.program.state_address_refinements}
        self.by_operation = {}
        for row in ctx.model.operations:
            self.by_operation.setdefault(row.operation, []).append(row)

    def gap(self, code):
        if code not in self.gaps:
            self.gaps.append(code)

    def spend(self, kind, *, cost=1, path='stored_value_sources'):
        if self.shared.truncated or not self.shared.spend('stored_value_' + kind, cost, path):
            if self.cursor is None:
                self.cursor = {'phase': kind, 'path': path}
            self.gap('stored_value_budget_exhausted')
            return False
        return True

    def value(self, use):
        return (None if use is None or type(use.value) is Unresolved
                else self.ctx.values.get(use.value))

    def bind(self, row, occurrence, *, write=False):
        ctx = self.ctx
        original = ctx.core_operations[occurrence.operation]
        refinement = self.refinements[row.site]
        effect = refinement.effects[row.effect_index]
        local, inputs = [], {}
        result = {'declaration': row, 'operation': occurrence, 'effect': effect,
                  'inputs': (), 'before': None, 'current': None, 'post': None,
                  'post_point': None, 'result': None, 'source_value': None, 'source_use': None,
                  'selected': None, 'owner': None, 'selectors': (),
                  'completion': None, 'supported': False, 'gaps': ()}

        def gap(code):
            if code not in local:
                local.append(code)

        current = ctx.current(OccurrencePoint(occurrence.id, OccurrencePhase.BEFORE)) or {}
        for use in original.uses:
            if not self.spend('inputs', path=str(occurrence.id)):
                gap('stored_value_budget_exhausted')
                break
            actual = ctx.uses.get((occurrence.id, use.id))
            value = self.value(actual)
            inputs[use.id] = value
            if value is None:
                gap('stored_value_input_unobserved')
            definition = ctx.core_values[use.value]
            if type(definition) is StateVersionValue and (
                    value is None or current.get(definition.location) != value.id):
                gap('stored_value_current_state_unproved')
        result['inputs'] = tuple(inputs.items())
        state_use = (original.accesses[row.site.index].before
                     if type(row.site) is ExistingAccess else row.site.state_use)
        state_definition = ctx.core_values[ctx.core_uses[state_use][1].value]
        location = state_definition.location
        result['location'], result['before'] = location, inputs.get(state_use)
        result['current'] = ctx.values.get(current.get(location))
        if result['before'] is None or result['before'] != result['current']:
            gap('stored_value_current_state_unproved')

        if type(effect.owner) is ReferenceOwner:
            result['owner'] = inputs.get(effect.owner.use)
        else:
            gap('stored_value_single_slot_owner_unproved')
        selectors = []
        # A path through intermediate references may alias a different path.
        # This receipt only interprets one direct cell, never a range/aggregate.
        if len(effect.path) != 1:
            gap('stored_value_single_slot_path_unproved')
        for selector in effect.path:
            if not self.spend('selectors', path=str(occurrence.id)):
                gap('stored_value_budget_exhausted')
                break
            if type(selector) in (ObjectField, CellContent):
                selectors.append((selector, None))
            elif type(selector) in (DynamicField, Element, NamespaceKey):
                use = (selector.name if type(selector) is DynamicField else
                       selector.index if type(selector) is Element else selector.key)
                value = inputs.get(use)
                if value is None or type(ctx.core_values[value.value]) is not DataValue:
                    gap('stored_value_selector_value_unproved')
                selectors.append((selector, value))
            else:
                gap('stored_value_single_slot_path_unproved')
                selectors.append((selector, None))
        result['selectors'] = tuple(selectors)

        selected_edge = (row.completion if write else row.result.edge
                         if type(row.result) is InvokeLiteralResult else None)
        if type(row.site) is ExistingAccess:
            result['selected'] = True
            for identity in original.results:
                if not self.spend('results', path=str(occurrence.id)):
                    gap('stored_value_budget_exhausted')
                    break
                if ctx.results.get((occurrence.id, identity)) is None:
                    gap('stored_value_result_unobserved')
            if write:
                after = original.accesses[row.site.index].after
                result['post'] = ctx.results.get((occurrence.id, after))
                result['post_point'] = OccurrencePoint(occurrence.id, OccurrencePhase.AFTER)
            else:
                result['result'] = ctx.results.get((occurrence.id, original.results[row.result.index]))
        else:
            transfer = ctx.outgoing.get(occurrence.id)
            result['completion'] = transfer
            result['selected'] = (None if transfer is None else
                                  transfer.successor_index == selected_edge.successor_index)
            if result['selected'] is not True:
                gap('stored_value_completion_unobserved' if transfer is None
                    else 'stored_value_completion_not_taken')
            else:
                successor = original.terminator.successors[transfer.successor_index]
                block = ctx.blocks[successor.block]
                signature = next(sig for sig in original.completions if sig.kind is successor.kind)
                for formal_id, argument in zip(block.arguments, successor.arguments):
                    if not self.spend('completion_ports', path=str(occurrence.id)):
                        gap('stored_value_budget_exhausted')
                        break
                    formal = ctx.results.get((transfer.target, formal_id))
                    if formal is None:
                        gap('stored_value_result_unobserved')
                    if type(argument) is CompletionResult:
                        type_ = signature.types[argument.index]
                        if (getattr(type_, 'location', None) == location
                                and type(ctx.core_values[formal_id]) is StateVersionValue):
                            result['post'] = formal
                        if not write and argument.index == row.result.result.index:
                            result['result'] = formal
                    else:
                        actual = ctx.uses.get((occurrence.id, argument))
                        if self.value(actual) is None:
                            gap('stored_value_completion_input_unobserved')
                if write:
                    result['post_point'] = OccurrencePoint(occurrence.id, OccurrencePhase.EDGE,
                                                           transfer.successor_index)
        if write:
            result['source_use'] = ctx.uses.get((occurrence.id, row.payload))
            result['source_value'] = inputs.get(row.payload)
            if result['source_value'] is None:
                gap('stored_value_payload_unobserved')
            post, point = result['post'], result['post_point']
            after = {} if point is None else ctx.current(point) or {}
            if post is None or after.get(location) != post.id:
                gap('stored_value_post_state_unproved')
        elif result['result'] is None:
            gap('stored_value_result_unobserved')
        if result['owner'] is None:
            gap('stored_value_single_slot_owner_unproved')
        result['gaps'] = tuple(local)
        result['supported'] = not local and not self.shared.truncated
        return result

    def copies(self, source, target):
        """Compare proven sources, including owners/keys obtained through loads."""
        return source_pair(self.ctx, source, target, self.shared)

    def address(self, left, right):
        from scar.ir.execution.source_validation import object_identity
        result = {'match': None, 'owner': None, 'selectors': ()}
        a, b = left['owner'], right['owner']
        if a is None or b is None:
            return result
        owner = self.copies(a, b)
        if owner['match'] is not True and not owner['conflict'] and not self.shared.truncated:
            identity = object_identity(self.ctx, a, b, shared=self.shared)
            owner = {**owner, 'identity': identity, 'match': identity['match']}
        result['owner'] = owner
        if owner['match'] is not True:
            result['match'] = owner['match']
            return result
        if len(left['selectors']) != 1 or len(right['selectors']) != 1:
            return result
        (p, x), (q, y) = left['selectors'][0], right['selectors'][0]
        if type(p) is not type(q):
            return result
        if not self.spend('address_comparisons'):
            return result
        if type(p) is ObjectField:
            result['match'] = p.name == q.name
        elif type(p) is CellContent:
            result['match'] = True
        elif type(p) in (DynamicField, Element, NamespaceKey) and x is not None and y is not None:
            a, b = self.ctx.core_values[x.value], self.ctx.core_values[y.value]
            if type(a) is not DataValue or type(b) is not DataValue:
                return result
            comparison = self.copies(x, y)
            result['selectors'] = (comparison,)
            result['match'] = comparison['match']
            if (result['match'] is None and type(a.origin) is LiteralOrigin
                    and type(b.origin) is LiteralOrigin
                    and type(a.origin.value) in (type(None), bool, int, float, str)
                    and type(a.origin.value) is type(b.origin.value)
                    and a.data_type == b.data_type):
                result['match'] = a.origin.value == b.origin.value
        return result


def stored_value_sources(ctx, site, occurrence, *, shared):
    """Conditionally link an original direct-slot load to actual store payloads."""
    if type(shared) is not ValidityBudget:
        raise TypeError('stored-value query requires exact shared ValidityBudget')
    validate_request(site, occurrence, shared.limit)
    original = ctx.occurrences.get(occurrence)
    if original is None:
        raise KeyError(occurrence)
    owner = site.operation if type(site) is OperationResult else site.edge.operation
    if original.operation != owner:
        raise ValueError('stored-value result site belongs to another original operation')
    definition = ctx.core_operations[owner]
    if type(site) is OperationResult:
        if site.index >= len(definition.results):
            raise KeyError(site)
    else:
        if (definition.terminator is None
                or site.edge.successor_index >= len(definition.terminator.successors)):
            raise KeyError(site)
        edge = definition.terminator.successors[site.edge.successor_index]
        signature = next((sig for sig in definition.completions if sig.kind is edge.kind), None)
        if signature is None or site.result.index >= len(signature.types):
            raise KeyError(site)
    query = _Sources(ctx, shared)
    reads = tuple(row for row in ctx.program.stored_value_reads if row.result == site)
    read = None
    if not reads:
        query.gap('stored_value_read_declaration_missing')
    elif query.spend('read'):
        read = query.bind(reads[0], original)
        for gap in read['gaps']:
            query.gap(gap)
    candidates = []
    for row in ctx.program.stored_value_writes:
        for operation in query.by_operation.get(row.site.operation, ()):
            if not query.spend('candidates', path=str(operation.id)):
                break
            binding = query.bind(row, operation, write=True)
            candidate = {'write': row, 'operation': operation, 'binding': binding,
                         'source_value': binding['source_value'], 'source_use': binding['source_use'],
                         'address': None,
                         'interval': None, 'match': None, 'gaps': binding['gaps']}
            candidates.append(candidate)
            if read is None:
                continue
            candidate['address'] = query.address(binding, read)
            if (not binding['supported'] or not read['supported']
                    or candidate['address']['match'] is not True or shared.truncated):
                continue
            if value_type(ctx.core_values[binding['source_value'].value]) != value_type(
                    ctx.core_values[read['result'].value]):
                candidate['gaps'] += ('stored_value_result_kind_mismatch',)
                continue
            from scar.ir.execution.state_preservation import state_preservation
            interval = state_preservation(ctx, binding['post_point'],
                OccurrencePoint(occurrence, OccurrencePhase.BEFORE), (read['location'],), shared=shared)
            candidate['interval'] = interval
            if interval['preserved'] is True and not shared.truncated:
                from .resumable_identity import compatible
                comparison = compatible(ctx, binding['source_value'], read['result'], [],
                                        'stored_value.payload_result')
                candidate['compatibility'] = comparison
                if comparison['status'] != 'CONFLICT':
                    candidate['match'] = True
        if shared.truncated:
            break
    # Separate final-post declarations cannot race to supply incompatible
    # payloads for an uncertain shared cell at that same completion.
    groups = {}
    if query.spend('post_groups', cost=len(candidates)):
        for candidate in candidates:
            key = candidate['operation'].id, candidate['write'].completion
            groups.setdefault(key, []).append(candidate)
    for candidate in candidates:
        if candidate['match'] is not True:
            continue
        for other in groups.get((candidate['operation'].id, candidate['write'].completion), ()):
            if not query.spend('post_consistency'):
                break
            if other is candidate:
                continue
            address = other['address']
            if ((address is None or address['match'] is not False)
                    and other['source_value'] != candidate['source_value']):
                candidate['match'] = None
                candidate['gaps'] += ('stored_value_multiple_post_alias_unproved',)
                query.gap('stored_value_multiple_post_alias_unproved')
                break
    proven = {}
    for candidate in candidates:
        if shared.truncated or ctx.issues:
            candidate['match'] = None
        if candidate['match'] is True:
            value = candidate['source_value']
            proven[value.id] = value
    sources = tuple(proven.values())
    match = True if len(sources) == 1 and not shared.truncated and not ctx.issues else None
    if len(sources) > 1:
        query.gap('stored_value_sources_ambiguous')
    if match is None and not query.gaps:
        query.gap('stored_value_source_unproved')
    if shared.truncated and query.cursor is None:
        query.cursor = {'phase': 'shared_interval', 'path': 'stored_value_sources'}
    source = sources[0] if match is True else None
    recipe = (None if source is None else {'source': source.id, 'target': read['result'].id,
        'rule': 'DECLARED_STORED_VALUE',
        'support': tuple(candidate for candidate in candidates if candidate['match'] is True)})
    return {'site': site, 'operation': original, 'read': read,
        'read_inventory': ctx.program.stored_value_reads,
        'write_inventory': ctx.program.stored_value_writes,
        'operation_inventory': ctx.model.operations,
        'candidates': tuple(candidates), 'sources': sources, 'source_value': source,
        'match': match, 'recipe': recipe, 'gaps': tuple(query.gaps), 'cursor': query.cursor,
        'budget': {'limit': shared.limit, 'used': shared.used,
                   'costs': dict(shared.costs), 'truncated': shared.truncated},
        'preparation': {'operations': len(ctx.model.operations),
                        'refinements': len(query.refinements),
                        'scope': 'original indexing, snapshot replay and detached copying are separate'},
        'conditions': ('direct-slot pre/post semantics are supplied declarations, not arbitrary Python attributes',
                       'object identity and stored-reference source do not establish stable object contents',
                       'selected original ports and interval effects remain independent requirements'),
        **_flags()}


# Demand-bound stored-value edges, separate from original copy ancestry.
#
# Only an exact own READ result can project. Receipts are private to one shared
# query and its fresh context; neither roots nor canonical copy tables change.
_DEPTH_LIMIT = 32




def _receipt(ctx, target, shared, *, declaration=None, operation=None,
             sources_report=None, recipe=None, gaps=(), cursor=None):
    match = True if recipe is not None and not shared.truncated else None
    if shared.truncated:
        recipe = None
        if 'stored_value_projection_budget' not in gaps:
            gaps = (*gaps, 'stored_value_projection_budget')
        if cursor is None:
            cursor = {'phase': 'shared_query', 'target': None if target is None else getattr(target, 'id', None)}
    return {'declaration': declaration,
            'site': None if declaration is None else declaration.result,
            'operation': operation, 'target_value': target,
            'read_inventory': ctx.program.stored_value_reads,
            'write_inventory': ctx.program.stored_value_writes,
            'operation_inventory': ctx.model.operations,
            'sources_report': sources_report, 'recipe': recipe,
            'binding': {'match': match, 'support': () if recipe is None else (recipe,),
                        'compatibility': None, 'gaps': tuple(gaps), **_flags()},
            'status': 'DECLARED' if match is True else 'PARTIAL',
            'gaps': tuple(gaps), 'cursor': cursor,
            'budget': {'limit': shared.limit, 'used': shared.used,
                       'costs': dict(shared.costs), 'truncated': shared.truncated},
            'conditions': ('original direct-slot contracts and interval effects remain conditional',
                           'this receipt adds no original copy edge or global identity union'),
            **_flags()}


def _spend(shared, phase, *, cost=1, target=None):
    path = 'stored_value_projection.' + phase
    if target is not None:
        path += '.' + str(target.id)
    return not shared.truncated and shared.spend('stored_value_projection_' + phase, cost, path)


def _read_index(ctx, shared):
    """Index original result definitions once, charging all inspected ports."""
    indexes = getattr(shared, '_stored_projection_indexes', None)
    if indexes is None:
        indexes = shared._stored_projection_indexes = {}
    if ctx in indexes:
        return indexes[ctx], (), None
    by_value = {}
    for index, declaration in enumerate(ctx.program.stored_value_reads):
        cursor = {'phase': 'read_index', 'read_index': index}
        if not _spend(shared, 'read_index'):
            return None, ('stored_value_projection_budget',), cursor
        site = declaration.result
        owner = site.operation if type(site) is OperationResult else site.edge.operation
        operation = ctx.core_operations.get(owner)
        if operation is None:
            return None, ('stored_value_projection_definition_unresolved',), cursor
        if type(site) is OperationResult:
            if site.index >= len(operation.results):
                return None, ('stored_value_projection_definition_unresolved',), cursor
            identity = operation.results[site.index]
            position = None
        else:
            term = operation.terminator
            if (term is None or term.kind is not TerminatorKind.INVOKE
                    or site.edge.successor_index >= len(term.successors)):
                return None, ('stored_value_projection_definition_unresolved',), cursor
            edge = term.successors[site.edge.successor_index]
            block = ctx.blocks.get(edge.block)
            if (edge.kind is not ExitKind.NORMAL or block is None
                    or len(edge.arguments) != len(block.arguments)):
                return None, ('stored_value_projection_definition_unresolved',), cursor
            produced = []
            for position, (argument, formal) in enumerate(zip(edge.arguments, block.arguments)):
                if not _spend(shared, 'index_landing_ports'):
                    return None, ('stored_value_projection_budget',), {
                        **cursor, 'landing_position': position}
                if type(argument) is CompletionResult and argument.index == site.result.index:
                    produced.append((position, formal))
            if len(produced) != 1:
                return None, ('stored_value_projection_definition_unresolved',), cursor
            position, identity = produced[0]
        by_value.setdefault(identity, []).append((declaration, position))
    completed = {identity: tuple(rows) for identity, rows in by_value.items()}
    indexes[ctx] = completed
    return completed, (), None


def _producer(ctx, target, declaration, position, shared):
    """Resolve this result incarnation, never a sibling static-site visit."""
    if not _spend(shared, 'producer', target=target):
        return None, ('stored_value_projection_budget',), {'phase': 'producer', 'target': target.id}
    definition = ctx.core_values[target.value]
    site = declaration.result
    if type(site) is OperationResult:
        operation = (ctx.occurrences.get(target.origin)
                     if type(target.origin) is OperationOccurrenceID else None)
        original = None if operation is None else ctx.core_operations.get(operation.operation)
        if (type(definition.origin) is not OperationResult or definition.origin != site
                or original is None or original.id != site.operation
                or site.index >= len(original.results) or original.results[site.index] != target.value
                or ctx.results.get((operation.id, target.value)) != target):
            return operation, ('stored_value_projection_producer_unobserved',), None
        return operation, (), None
    visit = ctx.visits.get(target.origin) if type(target.origin) is BlockVisitID else None
    origin = definition.origin
    if (visit is None or type(origin) is not BlockArgument
            or origin.block != visit.block or origin.index != position
            or ctx.results.get((target.origin, target.value)) != target):
        return None, ('stored_value_projection_landing_unobserved',), None
    transfer = ctx.incoming.get(target.origin)
    if transfer is None:
        return None, ('stored_value_projection_transfer_unobserved',), None
    operation = ctx.occurrences.get(transfer.operation)
    original = None if operation is None else ctx.core_operations.get(operation.operation)
    if (operation is None or original is None or original.id != site.edge.operation
            or transfer.target != target.origin or ctx.outgoing.get(operation.id) != transfer
            or transfer.successor_index != site.edge.successor_index):
        return operation, ('stored_value_projection_completion_unresolved',), None
    term = original.terminator
    if term is None or term.kind is not TerminatorKind.INVOKE:
        return operation, ('stored_value_projection_completion_unresolved',), None
    edge = term.successors[transfer.successor_index]
    block = ctx.blocks[visit.block]
    if not _spend(shared, 'selected_completion', target=target):
        return operation, ('stored_value_projection_budget',), {
            'phase': 'selected_completion', 'target': target.id}
    if (edge.kind is not ExitKind.NORMAL or edge.block != visit.block
            or position >= len(edge.arguments) or position >= len(block.arguments)
            or block.arguments[position] != target.value
            or type(edge.arguments[position]) is not CompletionResult
            or edge.arguments[position].index != site.result.index):
        return operation, ('stored_value_projection_own_result_unresolved',), None
    return operation, (), None


def projection(ctx, target, shared):
    """Return a completed conditional stored recipe, or None for a non-READ.

    ACTIVE keys are not partially published cache entries. A cycle increments
    a generation shared with the general source walker, so ancestors cannot
    cache failures dependent on an unfinished parent proof. Depth exhaustion
    truncates the shared analysis rather than granting a successful prefix.
    """
    if type(shared) is not ValidityBudget:
        raise TypeError('stored value projection requires exact shared ValidityBudget')
    if not ctx.program.stored_value_reads:
        return None
    if target is None or type(target) is Unresolved:
        _spend(shared, 'queries')
        return _receipt(ctx, target, shared, gaps=('stored_value_projection_target_unobserved',))
    if type(target) is not ValueOccurrence:
        raise TypeError('stored value projection requires an original ValueOccurrence')
    if ctx.values.get(target.id) != target:
        _spend(shared, 'queries', target=target)
        return _receipt(ctx, target, shared, gaps=('stored_value_projection_target_unobserved',))
    definition = ctx.core_values.get(target.value)
    if type(definition) not in (DataValue, ReferenceValue):
        return None
    if not _spend(shared, 'queries', target=target):
        return _receipt(ctx, target, shared, gaps=('stored_value_projection_budget',),
                        cursor={'phase': 'lookup', 'target': target.id})
    cache = getattr(shared, '_stored_projection_receipts', None)
    if cache is None:
        cache = shared._stored_projection_receipts = {}
    key = ctx, target.id
    if key in cache:
        cached = cache[key]
        if cached is not None and cached['recipe'] is not None and ctx.issues:
            return _receipt(ctx, target, shared, declaration=cached['declaration'],
                            operation=cached['operation'], sources_report=cached['sources_report'],
                            gaps=(*cached['gaps'], 'stored_value_projection_context_conflict'),
                            cursor=cached['cursor'])
        return cached
    index, gaps, cursor = _read_index(ctx, shared)
    if index is None:
        return _receipt(ctx, target, shared, gaps=gaps, cursor=cursor)
    declarations = index.get(target.value, ())
    if not declarations:
        cache[key] = None
        return None
    selected, unresolved = [], []
    for candidate_index, (declaration, position) in enumerate(declarations):
        if not _spend(shared, 'read_candidates', target=target):
            return _receipt(ctx, target, shared, gaps=('stored_value_projection_budget',),
                            cursor={'phase': 'read_candidates', 'target': target.id,
                                    'candidate_index': candidate_index})
        operation, gaps, cursor = _producer(ctx, target, declaration, position, shared)
        if shared.truncated:
            return _receipt(ctx, target, shared, declaration=declaration,
                            operation=operation, gaps=gaps, cursor=cursor)
        if gaps:
            unresolved.append((declaration, operation, gaps, cursor))
        else:
            selected.append((declaration, operation))
    if len(selected) != 1:
        gaps = ['stored_value_projection_read_ambiguous'] if len(declarations) > 1 else []
        for _, _, missing, _ in unresolved:
            gaps.extend(code for code in missing if code not in gaps)
        return _receipt(ctx, target, shared,
                        declaration=declarations[0][0] if len(declarations) == 1 else None,
                        operation=unresolved[0][1] if len(unresolved) == 1 else None,
                        gaps=tuple(gaps),
                        cursor=next((row[3] for row in unresolved if row[3] is not None), None))
    declaration, operation = selected[0]
    active = getattr(shared, '_stored_projection_active', None)
    if active is None:
        active = shared._stored_projection_active = set()
    if key in active:
        shared._stored_projection_cycle_generation = getattr(
            shared, '_stored_projection_cycle_generation', 0) + 1
        return _receipt(ctx, target, shared, declaration=declaration, operation=operation,
                        gaps=('stored_value_projection_cycle',),
                        cursor={'phase': 'active', 'target': target.id})
    if len(active) >= _DEPTH_LIMIT:
        shared.mark_truncated('stored_value_projection.depth',
                              'stored-value source analysis depth limit exceeded')
        return _receipt(ctx, target, shared, declaration=declaration, operation=operation,
                        gaps=('stored_value_projection_depth_limit',),
                        cursor={'phase': 'depth', 'target': target.id, 'limit': _DEPTH_LIMIT})
    generation = getattr(shared, '_stored_projection_cycle_generation', 0)
    active.add(key)
    try:
        from .stored_values import stored_value_sources
        report = stored_value_sources(ctx, declaration.result, operation.id, shared=shared)
    finally:
        active.remove(key)
    gaps = list(report['gaps'])
    recipe = report['recipe']
    source = report['source_value']
    if (report['match'] is not True or shared.truncated or recipe is None or ctx.issues):
        recipe = None
        if ctx.issues:
            gaps.append('stored_value_projection_context_conflict')
    elif (source is None or ctx.values.get(source.id) != source
          or recipe.get('source') != source.id or recipe.get('target') != target.id
          or recipe.get('rule') != 'DECLARED_STORED_VALUE'
          or report['read'] is None or report['read']['result'] != target):
        recipe = None
        gaps.append('stored_value_projection_recipe_unresolved')
    receipt = _receipt(ctx, target, shared, declaration=declaration, operation=operation,
                       sources_report=report, recipe=recipe, gaps=tuple(gaps),
                       cursor=report['cursor'])
    if (not shared.truncated
            and generation == getattr(shared, '_stored_projection_cycle_generation', 0)):
        cache[key] = receipt
    return receipt


# Query-local value-source proofs, separate from unconditional copy ancestry.
#
# A stored edge means the supplied slot contract and the captured state interval
# support that source. It never changes canonical copy recipes or object roots.
# Parallel original-copy and stored-read edges are retained at the same value.
def source_pair(ctx, source, target, shared):
    """Find an actual common source, without equating unrelated candidates.

    Ordinary routes are tried first, so an independent copy proof does not
    require a recursive stored-read proof. If needed, a multiedge search adds
    conditional stored routes at *every* visited value, including values which
    already have an original copy predecessor. Only the chosen proof paths
    undergo endpoint compatibility checks.
    """
    from .pending_payloads import copy_recipe
    from .resumable_identity import compatible

    if type(shared) is not ValidityBudget:
        raise TypeError('value-source query requires exact shared ValidityBudget')
    gaps, search_gaps, projections, support = [], [], [], ()
    comparison, common, conflict = None, None, False
    paths, processed = ({}, {}), (set(), set())
    original, conditional = deque(), deque()

    def spend(kind, cost=1):
        return not shared.truncated and shared.spend('value_source_' + kind, cost,
                                                     path='value.source.' + kind)

    def add(side, identity, child=None, recipe=None):
        nonlocal common
        if identity not in paths[side]:
            paths[side][identity] = None if child is None else (child, recipe)
            original.append((side, identity))
            conditional.append((side, identity))
        if identity in paths[1 - side]:
            common = identity

    def append_original(side, identity):
        nonlocal conflict
        if (side, identity) in processed[0]:
            return
        if not spend('original_nodes'):
            return
        processed[0].add((side, identity))
        recipe, missing, contradicted = copy_recipe(ctx, identity, shared)
        search_gaps.extend(missing)
        conflict |= contradicted
        if recipe is not None:
            follow(side, identity, recipe)

    def follow(side, identity, recipe):
        nonlocal conflict
        if (recipe.get('target') != identity or recipe.get('source') not in ctx.values):
            # A broken supported route is a hard model contradiction, not a
            # reason to choose another candidate or install a guessed source.
            conflict = True
            gaps.append('value_source_recipe_endpoint_conflict')
            return
        add(side, recipe['source'], identity, recipe)

    admitted = spend('pairs')
    present = all(value is not None and type(value) is not Unresolved
                  and ctx.values.get(value.id) == value for value in (source, target))
    if not present:
        gaps.append('value_source_endpoint_unobserved')
    elif admitted:
        add(0, source.id)
        add(1, target.id)
        # First establish any independent original proof. No conditional
        # expansion is necessary for a reflexive or already connected pair.
        while original and common is None and not conflict and not shared.truncated:
            append_original(*original.popleft())
        # Each node can have BOTH predecessors. Newly discovered conditional
        # sources in turn get their original and conditional edges explored.
        while conditional and common is None and not conflict and not shared.truncated:
            while original and common is None and not conflict and not shared.truncated:
                append_original(*original.popleft())
            if common is not None or conflict or shared.truncated:
                break
            side, identity = conditional.popleft()
            if (side, identity) in processed[1]:
                continue
            if not spend('conditional_nodes'):
                break
            processed[1].add((side, identity))
            receipt = projection(ctx, ctx.values[identity], shared)
            if receipt is None:
                continue
            projections.append(receipt)
            search_gaps.extend(receipt['gaps'])
            # ACTIVE-dependent Unknown is only a missing edge. An independent
            # original route may still succeed; a cycle is never its own proof.
            recipe = receipt['recipe']
            if recipe is not None and not shared.truncated:
                follow(side, identity, recipe)

    match = None
    if common is not None and not shared.truncated and not conflict:
        selected = []
        for side in (0, 1):
            cursor = common
            while paths[side][cursor] is not None:
                if not spend('selected_support'):
                    break
                child, recipe = paths[side][cursor]
                selected.append(recipe)
                cursor = child
            if shared.truncated:
                break
        if spend('selected_support'):
            support = tuple(selected)
            comparison = compatible(ctx, source, target, gaps, 'value.source.selected')
            conflict = comparison['status'] == 'CONFLICT'
            for recipe in support:
                if not spend('selected_compatibility'):
                    break
                check = compatible(ctx, ctx.values[recipe['source']], ctx.values[recipe['target']],
                                   gaps, 'value.source.recipe')
                conflict |= check['status'] == 'CONFLICT'
            if not shared.truncated and not conflict and not ctx.issues:
                match = True
    if conflict:
        match = False
    if match is None:
        gaps.extend(search_gaps)
        gaps.append('value_source_unproved')
    if shared.truncated:
        gaps.append('value_source_budget_exhausted')
    stored = any(row['rule'] == 'DECLARED_STORED_VALUE' for row in support)
    return {'source_value': source, 'target_value': target, 'match': match,
        'conflict': conflict, 'common_source': common, 'support': support,
        'compatibility': comparison, 'projections': tuple(projections),
        'binding_kind': ('CONDITIONAL_STORED_VALUE_SOURCE' if stored else 'ORIGINAL_COPY')
                        if match is True else 'UNPROVED_VALUE_SOURCE',
        'conditional_stored_source': stored and match is True,
        'read_inventory': ctx.program.stored_value_reads,
        'write_inventory': ctx.program.stored_value_writes,
        'operation_inventory': ctx.model.operations,
        'gaps': tuple(gaps),
        'cursor': None if not shared.truncated else {
            'phase': 'value_source_search', 'original_pending': len(original),
            'conditional_pending': len(conditional), 'visited': tuple(len(rows) for rows in paths)},
        'budget': {'limit': shared.limit, 'used': shared.used, 'costs': dict(shared.costs),
                   'truncated': shared.truncated},
        'conditions': ('stored source routes depend on supplied direct-slot contracts and state intervals',
                       'source correspondence does not establish stable reference contents or native identity',
                       'original copy recipes and identity roots remain unchanged'),
        'preparation': {'original_operations': len(ctx.model.operations),
                        'original_reads': len(ctx.program.stored_value_reads),
                        'original_writes': len(ctx.program.stored_value_writes),
                        'scope': 'original indexing, fresh-context snapshot replay and detached output copying are separate'},
        **_flags()}


__all__ = [
    'stored_value_sources',
    'validate_request',
    'projection',
    'source_pair',
]
