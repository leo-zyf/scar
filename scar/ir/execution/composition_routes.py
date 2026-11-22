"""Composition routes."""

from __future__ import annotations

from scar.ir.program.control import PendingPayload
from scar.ir.program.drivers import BodyDriveResult
from scar.ir.program.driver_validation import body_outcome_uses
from scar.ir.program.ids import UseID
from scar.ir.program.model import BlockArgument, CompletionResult
from scar.ir.program.resumption import ResumeInput, SavedInput, saved_actual_use
from scar.ir.execution.common import (
    Unresolved,
    EvidencePoint,
    BlockVisitID,
    OperationOccurrenceID,
    RegionActivationID,
    RunID,
    MissingEvidence,
    _identifier,
    _record,
)
from scar.ir.execution.validity_budget import ValidityBudget
from scar.ir.execution.validation import ExecutionIssue
from scar.ir.execution.coordinates import OccurrencePoint, OccurrencePhase
from scar.ir.execution.content import ObservationConsumer
from scar.ir.execution.observation_validation import RequiredEndpoint, RequiredOutput, ContractRef
from copy import deepcopy
from dataclasses import fields, dataclass
from scar.ir.execution.work import WithinOperation
from scar.ir.program.providers import ProviderFactReference


# Query-only scopes for composing explicit contract obligations.
#
# These records are analysis inputs, not execution facts or persistent model
# rows.  A scope selects one exact contract and one run/activation/operation
# owner for a particular query.
@dataclass(frozen=True, slots=True)
class CompositionScope:
    contract: ContractRef
    owner: RunID | RegionActivationID | OperationOccurrenceID
    source: ProviderFactReference | MissingEvidence

    def __post_init__(self) -> None:
        _record(self.contract, ContractRef, "composition contract")
        if type(self.owner) not in (RunID, RegionActivationID, OperationOccurrenceID):
            raise TypeError("composition owner must be an exact RunID, RegionActivationID, or OperationOccurrenceID")
        _identifier(self.owner, type(self.owner), "composition owner")
        if type(self.source) not in (ProviderFactReference, MissingEvidence):
            raise TypeError("composition source must be a ProviderFactReference or MissingEvidence")
        self.source.__post_init__()


# Composition routes.
# Budgeted projections of original execution ports, never another use graph.
#
# This internal query does not detach: the public composition query owns its one
# final copy.  DERIVED means the supplied observed rows were scanned, not that
# capture, effects, identity, or the whole program has been proved complete.
class _Routes:
    def __init__(self, ctx, budget):
        if type(budget) is not ValidityBudget:
            raise TypeError('routes requires an exact shared ValidityBudget')
        self.ctx, self.budget = ctx, budget
        self.operations, self.values, self.transfers, self.calls = [], [], [], []
        self.gaps, self.cursor = [], None
        self.visited = {'operations': 0, 'values': 0, 'transfers': 0, 'calls': 0,
                        'uses': 0, 'results': 0, 'accesses': 0, 'arguments': 0,
                        'call_inputs': 0, 'call_outputs': 0, 'call_scan': 0}

    def spend(self, kind, path):
        if self.cursor is not None:
            return False
        if self.budget.spend('composition_' + kind, path=path):
            return True
        self.cursor = {'stage': kind, 'path': path}
        self.gaps.append(ExecutionIssue('composition_budget_exhausted', path,
                         'shared traversal budget exhausted; unvisited rows remain in the inventory'))
        return False

    def gap(self, code, path, message):
        self.gaps.append(ExecutionIssue(code, str(path), message))

    def value(self, identity):
        return identity if type(identity) is Unresolved else self.ctx.values.get(identity)

    def use(self, occurrence, identity):
        definition = self.ctx.core_uses[identity][1]
        actual = self.ctx.uses.get((occurrence, identity))
        value = None if actual is None else self.value(actual.value)
        if actual is None:
            self.gap('composition_use_unobserved', occurrence,
                     f'original input port {identity} has no captured UseOccurrence')
        elif type(actual.value) is Unresolved:
            self.gap('composition_use_unresolved', occurrence,
                     f'original input port {identity} has an unresolved actual')
        return {'definition': definition, 'actual': actual, 'value': value}

    def argument(self, occurrence, operation, successor, argument, destination, formal):
        """Keep projections with no copy Use as real typed boundary relations."""
        source = saved_actual_use(operation, argument)
        result = {'argument': argument, 'kind': type(argument).__name__,
                  'source': None if source is None else self.use(occurrence.id, source),
                  'projection': None}
        if isinstance(argument, UseID) or type(argument) is SavedInput:
            return result
        if type(argument) is CompletionResult:
            signature = next((row for row in operation.completions
                              if row.kind is successor.kind), None)
            result['projection'] = {'completion': signature, 'index': argument.index,
                                    'exit_kind': successor.kind}
        elif type(argument) is BodyDriveResult:
            original = self.ctx.core_operations[successor.source]
            identity = body_outcome_uses(original)[argument.index]
            result['projection'] = {'operation': original,
                'use': self.ctx.core_uses[identity][1], 'index': argument.index,
                'dynamic_occurrence': None}
        elif type(argument) is ResumeInput:
            result['projection'] = {'slot': successor.inputs[argument.resume_slot],
                'index': argument.resume_slot,
                'ingress': self.ctx.segments[destination.segment].ingress}
        elif type(argument) is PendingPayload:
            from .pending_payloads import projection
            pending = operation.terminator.pending
            receipt = projection(self.ctx, formal, self.budget)
            result['projection'] = {'pending': self.use(occurrence.id, pending),
                                    'port': successor.port,
                                    'payload': receipt}
            for code in () if receipt is None else receipt['gaps']:
                self.gap(code, occurrence.id, 'original pending payload projection remains unresolved')
            if self.budget.truncated and self.cursor is None:
                self.cursor = {'stage': 'pending_payload', 'path': str(occurrence.id)}
        else:
            self.gap('composition_argument_unsupported', occurrence.id,
                     f'original {type(argument).__name__} marker retained without an interpretation')
        return result

    def scan_operations(self):
        ctx = self.ctx
        for number, row in enumerate(ctx.model.operations):
            path = f'operations[{number}]'
            if not self.spend('operations', path):
                return
            self.visited['operations'] += 1
            definition = ctx.core_operations[row.operation]
            visit = ctx.visits[row.visit]
            segment = ctx.segments[visit.segment]
            item = {'occurrence': row, 'definition': definition, 'visit': visit,
                'segment': segment, 'activation': ctx.activations[segment.activation],
                'uses': [], 'results': [], 'accesses': [],
                'before': ctx.before.get(row.id), 'after': ctx.after.get(row.id),
                'effects_complete': definition.effects_complete, 'scan_complete': False}
            self.operations.append(item)
            for ordinal, use in enumerate(definition.uses):
                if not self.spend('uses', f'{path}.uses[{ordinal}]'):
                    return
                self.visited['uses'] += 1
                item['uses'].append({'ordinal': ordinal, **self.use(row.id, use.id)})
            for ordinal, identity in enumerate(definition.results):
                if not self.spend('results', f'{path}.results[{ordinal}]'):
                    return
                self.visited['results'] += 1
                actual = ctx.results.get((row.id, identity))
                item['results'].append({'ordinal': ordinal,
                    'definition': ctx.core_values[identity], 'actual': actual})
                if actual is None:
                    self.gap('composition_result_unobserved', row.id,
                             f'original result port {identity} has no captured value')
            for index, access in enumerate(definition.accesses):
                if not self.spend('accesses', f'{path}.accesses[{index}]'):
                    return
                self.visited['accesses'] += 1
                item['accesses'].append({'index': index, 'access': access,
                    'before': self.use(row.id, access.before),
                    'reference': None if access.reference is None else self.use(row.id, access.reference),
                    'after_definition': None if access.after is None else ctx.core_values[access.after],
                    'after': None if access.after is None else ctx.results.get((row.id, access.after))})
            item['scan_complete'] = True

    def scan_values(self):
        ctx = self.ctx
        for number, row in enumerate(ctx.model.values):
            path = f'values[{number}]'
            if not self.spend('values', path):
                return
            self.visited['values'] += 1
            definition = ctx.core_values[row.value]
            # Exactly the original definition endpoint: never an identity union.
            producer = (ctx.occurrences.get(row.origin) or ctx.visits.get(row.origin)
                        or ctx.activations.get(row.origin))
            route = None
            if isinstance(definition.origin, BlockArgument):
                visit = ctx.visits[row.origin]
                transfer = ctx.incoming.get(row.origin)
                route = {'transfer': transfer, 'argument': None,
                         'ordinal': definition.origin.index,
                         'ingress': ctx.segments[visit.segment].ingress}
                if transfer is not None:
                    occurrence = ctx.occurrences[transfer.operation]
                    operation = ctx.core_operations[occurrence.operation]
                    successor = operation.terminator.successors[transfer.successor_index]
                    route = {'transfer': transfer,
                        'argument': successor.arguments[definition.origin.index],
                        'ordinal': definition.origin.index,
                        'ingress': ctx.segments[visit.segment].ingress}
            self.values.append({'value': row, 'definition': definition,
                                'producer': producer, 'formal_route': route})

    def scan_transfers(self):
        ctx = self.ctx
        for number, transfer in enumerate(ctx.model.transfers):
            path = f'transfers[{number}]'
            if not self.spend('transfers', path):
                return
            self.visited['transfers'] += 1
            occurrence = ctx.occurrences[transfer.operation]
            operation = ctx.core_operations[occurrence.operation]
            successor = operation.terminator.successors[transfer.successor_index]
            destination = ctx.visits[transfer.target]
            block = ctx.blocks[destination.block]
            item = {'transfer': transfer, 'occurrence': occurrence,
                'operation': operation, 'terminator': operation.terminator,
                'successor': successor, 'destination': destination,
                'arguments': [], 'state': ctx.edges.get((occurrence.id, transfer.successor_index)),
                'scan_complete': False}
            self.transfers.append(item)
            for ordinal, argument in enumerate(successor.arguments):
                if not self.spend('arguments', f'{path}.arguments[{ordinal}]'):
                    return
                self.visited['arguments'] += 1
                identity = block.arguments[ordinal]
                item['arguments'].append({'ordinal': ordinal,
                    'formal_definition': ctx.core_values[identity],
                    'formal': ctx.results.get((destination.id, identity)),
                    **self.argument(occurrence, operation, successor, argument, destination,
                                    ctx.results.get((destination.id, identity)))})
            item['scan_complete'] = True

    def scan_calls(self):
        ctx = self.ctx
        # Iterate original observed calls, preserving the empty-link case.
        for number, caller in enumerate(ctx.model.operations):
            if not self.spend('call_scan', f'call_scan[{number}]'):
                return
            self.visited['call_scan'] += 1
            operation = ctx.core_operations[caller.operation]
            links = ctx.call_activations_by_caller.get(caller.id, ())
            if operation.callee is None and not links:
                continue
            if not links:
                if not self.spend('calls', f'calls[{caller.id}]'):
                    return
                self.visited['calls'] += 1
                gap = ExecutionIssue('call_activation_missing', str(caller.id),
                                     'static callee has no declared dynamic activation link')
                self.calls.append({'caller': caller, 'operation': operation,
                    'callee': ctx.regions[operation.callee], 'association': None,
                    'status': 'MISSING', 'inputs': (), 'outputs': (), 'gaps': (gap,),
                    'scan_complete': True})
                continue
            for identity in links:
                path = f'calls[{identity}]'
                if not self.spend('calls', path):
                    return
                self.visited['calls'] += 1
                original = ctx.resolved_call_activations[identity]
                item = {**original, 'operation': operation,
                    'callee': ctx.regions[operation.callee],
                    'inputs': [], 'outputs': [], 'status': 'DECLARED', 'scan_complete': False}
                self.calls.append(item)
                for kind in ('inputs', 'outputs'):
                    for ordinal, route in enumerate(original[kind]):
                        if not self.spend('call_' + kind, f'{path}.{kind}[{ordinal}]'):
                            return
                        self.visited['call_' + kind] += 1
                        item[kind].append(route)
                item['scan_complete'] = True

    def result(self):
        ctx = self.ctx
        for collection in (self.operations, self.transfers, self.calls):
            for row in collection:
                for name in ('uses', 'results', 'accesses', 'arguments', 'inputs', 'outputs'):
                    if name in row and type(row[name]) is list:
                        row[name] = tuple(row[name])
        return {'operations': tuple(self.operations), 'values': tuple(self.values),
            'transfers': tuple(self.transfers), 'calls': tuple(self.calls),
            'gaps': tuple(self.gaps), 'status': 'TRUNCATED' if self.cursor is not None else 'DERIVED',
            'visited': self.visited,
            'pending': {'operations': len(ctx.model.operations) - self.visited['operations'],
                        'values': len(ctx.model.values) - self.visited['values'],
                        'transfers': len(ctx.model.transfers) - self.visited['transfers'],
                        'call_scan': len(ctx.model.operations) - self.visited['call_scan']},
            'cursor': self.cursor, 'budget_used': self.budget.used,
            'source_runtime_verified': False, 'runtime_identity_verified': False,
            'value_identity_verified': False, 'transformation_authorized': False}


def routes(ctx, *, budget: ValidityBudget) -> dict:
    """Project supplied observed ports; caller retains the complete inventory.

    All stages consume the same budget. Pending counts exclude partially read
    rows; ``cursor`` and each row's ``scan_complete`` flag retain that distinction.
    This function neither chooses roots nor classifies unreachable work as dead.
    """
    view = _Routes(ctx, budget)
    for scan in (view.scan_operations, view.scan_values, view.scan_transfers, view.scan_calls):
        if view.cursor is not None:
            break
        scan()
    return view.result()


# Derive a bounded Q ledger for explicit composition query scopes.
#
# The scope is an analysis request, not an execution row. This module keeps the
# original observation requirements, roots, and content reads as source
# inventory; its derived ledger never replaces those canonical rows.
def _contract_key(reference):
    artifact = reference.artifact
    # URI is a locator. Snapshot identity is the exact artifact tuple.
    return artifact.sha256, artifact.schema, artifact.size_bytes, reference.contract


def _scope_key(scope):
    return _contract_key(scope.contract), type(scope.owner), scope.owner


def validate_scopes(scopes):
    """Validate a query-scope tuple without consulting execution or artifacts."""
    if type(scopes) is not tuple:
        raise TypeError("composition scopes must be an exact tuple")
    seen = set()
    for scope in scopes:
        if type(scope) is not CompositionScope:
            raise TypeError("composition scopes must contain exact CompositionScope records")
        scope.__post_init__()
        key = _scope_key(scope)
        if key in seen:
            raise ValueError("composition scope is duplicated")
        seen.add(key)
    return scopes


def prepare_scopes(ctx, scopes):
    """Resolve local owner foreign keys without opening external artifacts.

    A public query can call this before refreshing external snapshots so a bad
    owner fails before any provider artifact is inspected. Contract loading is
    deferred to requirements().
    """
    validate_scopes(scopes)
    prepared = []
    for scope in scopes:
        owner = scope.owner
        run = activation = operation = None
        if type(owner) is RunID:
            if owner not in ctx.runs:
                raise KeyError(f"unknown composition run: {owner.wire}")
            run = owner
        elif type(owner) is RegionActivationID:
            activation_row = ctx.activations.get(owner)
            if activation_row is None:
                raise KeyError(f"unknown composition activation: {owner.wire}")
            activation = owner
            process = ctx.processes.get(activation_row.process)
            if process is None:
                raise KeyError(f"composition activation has unknown process: {activation_row.process.wire}")
            run = process.run
        elif type(owner) is OperationOccurrenceID:
            if owner not in ctx.occurrences:
                raise KeyError(f"unknown composition operation occurrence: {owner.wire}")
            operation = owner
            activation = ctx.activation_of(owner)
            activation_row = ctx.activations.get(activation)
            if activation_row is None:
                raise KeyError(f"composition operation has unknown activation: {activation.wire}")
            process = ctx.processes.get(activation_row.process)
            if process is None:
                raise KeyError(f"composition operation has unknown process: {activation_row.process.wire}")
            run = process.run
        else:
            raise TypeError("invalid composition scope owner")
        prepared.append({
            "scope": scope, "owner": owner, "run": run,
            "activation": activation, "operation": operation,
        })
    return tuple(prepared)


def _point_owner(ctx, point):
    if type(point) is Unresolved:
        point = point.point
    if type(point) is OccurrencePoint:
        if point.operation not in ctx.occurrences:
            return None, None, None
        activation = ctx.activation_of(point.operation)
        activation_row = ctx.activations.get(activation)
        process = None if activation_row is None else ctx.processes.get(activation_row.process)
        return (None if process is None else process.run, activation, point.operation)
    if type(point) is EvidencePoint:
        return point.run, None, None
    return None, None, None


def _actual_owner(ctx, actual):
    from scar.ir.execution.observation_validation import BoundEndpoint, ExternalEndpoint, OutputValue

    if type(actual) is Unresolved:
        return _point_owner(ctx, actual.point)
    if type(actual) is OutputValue:
        value = ctx.values.get(actual.value)
        if value is None:
            return None, None, None
        origin = value.origin
        if type(origin) is OperationOccurrenceID:
            activation = ctx.activation_of(origin) if origin in ctx.occurrences else None
            operation = origin
        elif type(origin) is BlockVisitID:
            visit = ctx.visits.get(origin)
            segment = None if visit is None else ctx.segments.get(visit.segment)
            activation = None if segment is None else segment.activation
            operation = None
        elif type(origin) is RegionActivationID:
            activation, operation = origin, None
        else:
            activation = operation = None
        activation_row = None if activation is None else ctx.activations.get(activation)
        process = None if activation_row is None else ctx.processes.get(activation_row.process)
        return (None if process is None else process.run, activation, operation)
    if type(actual) is BoundEndpoint:
        binding = ctx.requirements.get(actual.binding)
        if binding is None:
            return None, None, None
        owner = binding.owner
        if type(owner) is OperationOccurrenceID and owner in ctx.occurrences:
            activation = ctx.activation_of(owner)
            operation = owner
        elif type(owner) is RegionActivationID:
            activation, operation = owner, None
        else:
            activation = operation = None
        activation_row = None if activation is None else ctx.activations.get(activation)
        process = None if activation_row is None else ctx.processes.get(activation_row.process)
        return (None if process is None else process.run, activation, operation)
    if type(actual) is ExternalEndpoint:
        # The resource target may establish a run/process, but never an
        # activation or operation correspondence.
        from .resources import (
            ExternalRealm, InstanceTarget, ProcessRealm, ProcessTarget,
            RunRealm, ThreadTarget,
        )
        target = actual.target
        process_id = run = None
        if type(target) is InstanceTarget:
            resource = ctx.resources.get(target.instance)
            if resource is not None:
                if type(resource.realm) is RunRealm:
                    run = resource.realm.run
                elif type(resource.realm) is ProcessRealm:
                    process_id = resource.realm.process
                elif type(resource.realm) is ExternalRealm:
                    return None, None, None
        elif type(target) is ProcessTarget:
            process_id = target.process
        elif type(target) is ThreadTarget:
            thread = ctx.threads.get(target.thread)
            process_id = None if thread is None else thread.process
        process = None if process_id is None else ctx.processes.get(process_id)
        if process is not None:
            run = process.run
        return run, None, None
    return None, None, None


def _root_owner_match(ctx, prepared, root):
    """Classify anchor, interval, and actual scope without identity inference."""
    owner = prepared["owner"]
    points = [root.anchor, root.interval.start, root.interval.end]
    point_scopes = [_point_owner(ctx, point) for point in points]
    actual_scope = _actual_owner(ctx, root.actual)
    known_scopes = [row for row in (*point_scopes, actual_scope) if any(part is not None for part in row)]
    if type(owner) is RunID:
        known_runs = [row[0] for row in known_scopes if row[0] is not None]
        if not known_runs:
            return "UNRESOLVED"
        if any(run != owner for run in known_runs):
            return "NO_MATCH"
        return "MATCH"
    if type(owner) is RegionActivationID:
        owner_row = ctx.activations.get(owner)
        owner_process = None if owner_row is None else ctx.processes.get(owner_row.process)
        owner_run = None if owner_process is None else owner_process.run
        known_runs = [row[0] for row in known_scopes if row[0] is not None]
        if owner_run is not None and any(run != owner_run for run in known_runs):
            return "NO_MATCH"
        known_activations = [row[1] for row in known_scopes if row[1] is not None]
        if not known_activations:
            return "UNRESOLVED"
        return ("MATCH" if all(activation == owner for activation in known_activations)
                and point_scopes[0][1] == owner else "UNRESOLVED")
    if type(owner) is OperationOccurrenceID:
        owner_activation = ctx.activation_of(owner) if owner in ctx.occurrences else None
        owner_row = None if owner_activation is None else ctx.activations.get(owner_activation)
        owner_process = None if owner_row is None else ctx.processes.get(owner_row.process)
        owner_run = None if owner_process is None else owner_process.run
        known_runs = [row[0] for row in known_scopes if row[0] is not None]
        if owner_run is not None and any(run != owner_run for run in known_runs):
            return "NO_MATCH"
        known_activations = [row[1] for row in known_scopes if row[1] is not None]
        if owner_activation is not None and any(
                activation != owner_activation for activation in known_activations):
            return "UNRESOLVED"
        anchor_run, anchor_activation, anchor_operation = point_scopes[0]
        if anchor_operation == owner:
            return "MATCH"
        if anchor_run is None and anchor_activation is None and anchor_operation is None:
            return "UNRESOLVED"
        return "UNRESOLVED"
    return "UNRESOLVED"


def _resolve_contracts(ctx, prepared):
    """Load exact requested definitions, retaining missing artifacts as gaps."""
    resolved, cache = [], {}
    for index, item in enumerate(prepared):
        scope = item["scope"]
        key = _contract_key(scope.contract)
        if key not in cache:
            graph = ctx.external_semantic(scope.contract.artifact,
                                          f"composition_scopes[{index}].contract")
            definition = None
            if graph is not None:
                definition = graph.contracts.get(scope.contract.contract)
                if definition is None:
                    raise KeyError(
                        "unknown contract in exact semantic artifact: "
                        f"{scope.contract.contract.wire}")
            cache[key] = graph, definition
        graph, definition = cache[key]
        resolved.append({
            **item,
            "contract": definition,
            "expected_outputs": None if definition is None else definition.required_outputs,
            "required_facets": None if definition is None else tuple(
                facet for facet in definition.facets if facet.required is True),
            "contract_status": "UNRESOLVED" if definition is None else "DECLARED",
        })
    return tuple(resolved)


def _scan_index(rows, budget, cost_kind, key_fn):
    """Build one linear index; return inspected count and an exact cursor."""
    index = {}
    for ordinal, row in enumerate(rows):
        if not budget.spend(cost_kind, path=f"{cost_kind}[{ordinal}]"):
            return index, ordinal, {"stage": cost_kind, "index": ordinal,
                                    "pending": len(rows) - ordinal}
        key = key_fn(row)
        if key is not None:
            index.setdefault(key, []).append(row)
    return index, len(rows), None


def _root_read_links(root, reads_by_root, budget, reads_complete):
    if not reads_complete:
        return {"root": root, "content_reads": (), "status": "READ_SCAN_TRUNCATED",
                "pending_reads": None}
    candidates = reads_by_root.get(root.id, ())
    selected = []
    for index, read in enumerate(candidates):
        if not budget.spend("composition_root_read_joins",
                            path=f"observation_roots[{root.id}].content_reads[{index}]"):
            return {"root": root, "content_reads": tuple(selected),
                    "status": "TRUNCATED", "pending_reads": len(candidates) - index}
        selected.append(read)
    status = ("DECLARED_READS" if selected else
              "NO_CONTENT_READ_DECLARED_PROTOCOL_UNCLASSIFIED")
    return {"root": root, "content_reads": tuple(selected), "status": status,
            "pending_reads": 0}


def requirements(ctx, scopes, *, budget):
    """Return Q obligations, original inventories, and conservative join status.

    The budget charges derived Q scans and joins. The returned top-level
    requirement/root/content-read tuples remain the exact original model tuples,
    including unvisited rows when a derived scan stops. Facets carry no
    endpoint target or cardinality, so they always remain uninterpreted.
    """
    from .resources import _exact
    from .validity_budget import ValidityBudget

    validate_scopes(scopes)
    _exact(budget, ValidityBudget, "composition validity budget")
    prepared = prepare_scopes(ctx, scopes)
    issue_start = len(ctx.issues)
    resolved = _resolve_contracts(ctx, prepared)
    scope_issues = ctx.issues[issue_start:]
    if scope_issues:
        raise ValueError("; ".join(
            f"{item.code} at {item.path}: {item.message}" for item in scope_issues))

    all_requirements = ctx.model.observation_requirements
    all_roots = ctx.model.observation_roots
    all_reads = ctx.model.content_reads

    req_output, req_facet = {}, {}
    req_count = 0 if scopes else len(all_requirements)
    req_cursor = None
    if scopes:
        for ordinal, row in enumerate(all_requirements):
            if not budget.spend("composition_requirement_rows",
                                path=f"observation_requirements[{ordinal}]"):
                req_cursor = {"stage": "requirements", "index": ordinal,
                              "pending": len(all_requirements) - ordinal}
                break
            req_count += 1
            contract_key = _contract_key(row.contract)
            if type(row.target) is RequiredOutput:
                req_output.setdefault((contract_key, row.target.slot), []).append(row)
            elif type(row.target) is RequiredEndpoint:
                req_facet.setdefault((contract_key, row.target.facet), []).append(row)
    requirements_complete = req_count == len(all_requirements)

    roots_by_requirement = {}
    root_count = 0 if scopes else len(all_roots)
    root_cursor = None
    if scopes and not budget.truncated:
        for ordinal, root in enumerate(all_roots):
            if not budget.spend("composition_root_rows", path=f"observation_roots[{ordinal}]"):
                root_cursor = {"stage": "roots", "index": ordinal,
                               "pending": len(all_roots) - ordinal}
                break
            root_count += 1
            roots_by_requirement.setdefault(root.requirement, []).append(root)
    roots_complete = root_count == len(all_roots)

    reads_by_root = {}
    read_count = 0 if scopes else len(all_reads)
    read_cursor = None
    if scopes and not budget.truncated:
        for ordinal, read in enumerate(all_reads):
            if not budget.spend("composition_content_read_rows", path=f"content_reads[{ordinal}]"):
                read_cursor = {"stage": "content_reads", "index": ordinal,
                               "pending": len(all_reads) - ordinal}
                break
            read_count += 1
            if type(read.consumer) is ObservationConsumer:
                reads_by_root.setdefault(read.consumer.root, []).append(read)
    reads_complete = read_count == len(all_reads)

    cursor = req_cursor or root_cursor or read_cursor
    stopped = budget.truncated
    obligations, scope_rows = [], []
    any_incomplete = False
    for scope_index, item in enumerate(resolved):
        contract = item["contract"]
        planned = (() if contract is None else
                   tuple(("OUTPUT", row) for row in item["expected_outputs"]) +
                   tuple(("FACET", row) for row in item["required_facets"]))
        entry = {
            **item, "obligation_start": len(obligations),
            "obligation_count": len(planned), "pending_obligations": (),
            "ledger_status": item["contract_status"],
        }
        if contract is None:
            any_incomplete = True
            scope_rows.append(entry)
            continue
        if stopped:
            entry["ledger_status"] = "TRUNCATED"
            entry["pending_obligations"] = planned
            entry["obligation_cursor"] = 0
            any_incomplete = True
            scope_rows.append(entry)
            continue

        entry_incomplete = False
        for local_index, (kind, target) in enumerate(planned):
            path = f"composition_scopes[{scope_index}].obligations[{local_index}]"
            if not budget.spend("composition_obligations", path=path):
                cursor = {"stage": "obligations", "scope": scope_index,
                          "index": local_index, "pending": len(planned) - local_index}
                entry["ledger_status"] = "TRUNCATED"
                entry["pending_obligations"] = planned[local_index:]
                entry["obligation_cursor"] = local_index
                any_incomplete = stopped = True
                break

            contract_key = _contract_key(item["scope"].contract)
            if kind == "OUTPUT":
                candidate_requirements = tuple(req_output.get((contract_key, target), ()))
            else:
                candidate_requirements = tuple(req_facet.get((contract_key, target.facet), ()))
            matched_roots, unresolved_roots = [], []
            owner_scan_complete = roots_complete and requirements_complete
            root_pending = 0
            pending_requirement_groups = 0
            for requirement_ordinal, requirement in enumerate(candidate_requirements):
                if not roots_complete:
                    continue
                candidates = roots_by_requirement.get(requirement.id, ())
                for root_index, root in enumerate(candidates):
                    if not budget.spend("composition_root_owner_joins",
                                        path=f"{path}.roots[{root_index}]"):
                        pending_requirement_groups = (
                            len(candidate_requirements) - requirement_ordinal - 1)
                        cursor = {"stage": "root_owner_joins", "scope": scope_index,
                                  "obligation": local_index,
                                  "requirement": requirement.id,
                                  "requirement_ordinal": requirement_ordinal,
                                  "index": root_index,
                                  "pending": len(candidates) - root_index,
                                  "pending_root_joins": len(candidates) - root_index,
                                  "pending_requirement_groups": pending_requirement_groups}
                        owner_scan_complete = False
                        root_pending += len(candidates) - root_index
                        stopped = any_incomplete = True
                        break
                    match = _root_owner_match(ctx, item, root)
                    if match == "MATCH":
                        matched_roots.append(root)
                    elif match == "UNRESOLVED":
                        unresolved_roots.append(root)
                if stopped:
                    break

            root_read_rows = []
            for root in (*matched_roots, *unresolved_roots):
                root_read_rows.append(_root_read_links(root, reads_by_root, budget, reads_complete))
                if budget.truncated:
                    cursor = cursor or {"stage": "root_read_joins", "scope": scope_index,
                                        "obligation": local_index}
                    owner_scan_complete = False
                    stopped = any_incomplete = True

            if kind == "OUTPUT":
                if not requirements_complete:
                    status = "REQUIREMENT_SCAN_TRUNCATED"
                elif not candidate_requirements:
                    status = "REQUIREMENT_MISSING"
                elif not owner_scan_complete:
                    status = "ROOT_JOIN_TRUNCATED"
                elif matched_roots:
                    status = "ROOT_PRESENT_OUTPUT_SLOT_UNVERIFIED"
                elif unresolved_roots:
                    status = "ROOT_SCOPE_UNRESOLVED"
                else:
                    status = "ROOT_MISSING"
            else:
                status = "FACET_TARGETS_UNINTERPRETED"
                if not requirements_complete or not owner_scan_complete:
                    status += ":JOIN_TRUNCATED"
                elif candidate_requirements and matched_roots:
                    status += ":DECLARED_ENDPOINT_CANDIDATE"
                elif candidate_requirements and unresolved_roots:
                    status += ":ENDPOINT_SCOPE_UNRESOLVED"
                elif not candidate_requirements:
                    status += ":NO_ENDPOINT_CANDIDATE"

            obligations.append({
                "scope": item["scope"], "kind": kind, "target": target,
                "requirements": candidate_requirements,
                "roots": tuple(matched_roots),
                "unresolved_roots": tuple(unresolved_roots),
                "root_content_reads": tuple(root_read_rows),
                "status": status,
                "requirements_scan_complete": requirements_complete,
                "roots_scan_complete": roots_complete,
                "content_reads_scan_complete": reads_complete,
                "pending_root_joins": root_pending,
                "pending_requirement_groups": pending_requirement_groups,
            })
            if status != "ROOT_PRESENT_OUTPUT_SLOT_UNVERIFIED":
                entry_incomplete = True
            if stopped:
                entry["ledger_status"] = "TRUNCATED"
                entry["pending_obligations"] = planned[local_index + 1:]
                entry["obligation_cursor"] = local_index + 1
                any_incomplete = True
                break

        if entry["ledger_status"] != "TRUNCATED":
            entry["ledger_status"] = "PARTIAL" if entry_incomplete else "DERIVED_FOR_REQUEST"
            any_incomplete = any_incomplete or entry_incomplete
        scope_rows.append(entry)

    if not scopes:
        scope_status = "UNRESOLVED"
    elif budget.truncated:
        scope_status = "TRUNCATED"
    elif any_incomplete:
        scope_status = "PARTIAL"
    else:
        scope_status = "DERIVED_FOR_REQUESTED_SCOPES"

    gaps = list(ctx.gaps)
    for item in budget.gaps:
        issue = ExecutionIssue(item["code"], item["path"], item["message"])
        if issue not in gaps:
            gaps.append(issue)
    if not scopes:
        gaps.append(ExecutionIssue("composition_scope_unresolved", "composition_scopes",
                                   "no exact contract/owner request established the Q denominator"))
    if any(item["contract"] is None for item in resolved):
        gaps.append(ExecutionIssue("composition_contract_unresolved", "composition_scopes",
                                   "an exact semantic contract artifact is unavailable"))
    if any(item["kind"] == "FACET" for item in obligations):
        gaps.append(ExecutionIssue("composition_facet_targets_uninterpreted", "composition_scopes",
                                   "required facets do not declare endpoint targets or cardinality"))
    if any(item["status"] == "ROOT_MISSING" for item in obligations):
        gaps.append(ExecutionIssue("composition_root_missing", "composition_scopes",
                                   "an expected output slot has no scope-matched observation root"))
    if any(read["status"] == "NO_CONTENT_READ_DECLARED_PROTOCOL_UNCLASSIFIED"
           for item in obligations for read in item["root_content_reads"]):
        gaps.append(ExecutionIssue("composition_content_protocol_unclassified", "content_reads",
                                   "a Q root has no linked ContentRead; this does not imply no content dependency"))

    return {
        "scopes": tuple(scope_rows),
        "obligations": tuple(obligations),
        "requirements": all_requirements,
        "roots": all_roots,
        "content_reads": all_reads,
        "gaps": tuple(gaps),
        "scope_status": scope_status,
        "scope_complete_for_request": scope_status == "DERIVED_FOR_REQUESTED_SCOPES",
        "budget": budget,
        "budget_truncated": budget.truncated,
        "cursor": cursor,
        "pending": {
            "requirements": len(all_requirements) - req_count,
            "roots": len(all_roots) - root_count,
            "content_reads": len(all_reads) - read_count,
        },
    }


# Compose original program/execution associations without transformation proof.
#
# The full supplied graphs are retained as the denominator. Derived views share
# one budget and one prepared external snapshot; only the public result detaches.
def _inventory(ctx):
    counts = {field.name: len(getattr(ctx.model, field.name))
              for field in fields(ctx.model) if field.name != 'snapshot'}
    return {'program': ctx.program, 'execution': ctx.model, 'execution_counts': counts,
            'canonical_counts': {'regions': len(ctx.program.regions),
                'operations': len(ctx.program.operations), 'values': len(ctx.program.values),
                'locations': len(ctx.program.locations)},
            'scope': 'all supplied canonical definitions and execution records; not complete source/runtime capture'}


def _reads(ctx, budget):
    from .validity_query import validity_shared
    result = []
    for ordinal, read in enumerate(ctx.model.content_reads):
        if not budget.spend('composition_read', path=f'composition.reads[{ordinal}]'):
            break
        if not budget.remaining:
            budget.mark_truncated(f'composition.reads[{ordinal}]',
                                  'read identity inspected but no budget remains for content reconciliation')
            result.append({'read': read, 'validity': None, 'status': 'TRUNCATED',
                           'query_budget_used': 0})
            break
        start = budget.used
        view = validity_shared(ctx, read.id, budget=budget, include_context_gaps=False)
        result.append({'read': read, 'validity': view,
                       'status': 'TRUNCATED' if view['truncated'] else 'INSPECTED',
                       'query_budget_used': budget.used - start})
        if budget.truncated:
            break
    completed = sum(item['status'] == 'INSPECTED' for item in result)
    cursor = next(({'table': 'content_reads', 'ordinal': ordinal, 'phase': 'validity'}
                   for ordinal, item in enumerate(result) if item['status'] == 'TRUNCATED'), None)
    if cursor is None and len(result) < len(ctx.model.content_reads):
        cursor = {'table': 'content_reads', 'ordinal': len(result), 'phase': 'identity'}
    return {'items': tuple(result), 'visited': len(result), 'completed': completed,
            'total': len(ctx.model.content_reads), 'pending': len(ctx.model.content_reads) - completed,
            'status': 'TRUNCATED' if budget.truncated else 'DERIVED',
            'cursor': cursor}


def _production(ctx, budget):
    registries = (('writes', ctx.resolved_writes), ('outcome_productions', ctx.resolved_produced),
                  ('established_productions', ctx.resolved_establishments))
    result, pending, cursor = {}, {}, None
    for name, registry in registries:
        items = []
        for ordinal, resolved in enumerate(registry.values()):
            if not budget.spend('composition_production', path=f'composition.{name}[{ordinal}]'):
                if cursor is None:
                    cursor = {'table': name, 'ordinal': ordinal}
                break
            items.append(resolved)
        result[name] = tuple(items)
        pending[name] = len(registry) - len(items)
    return {**result, 'pending': pending, 'cursor': cursor,
            'status': 'TRUNCATED' if any(pending.values()) else 'DERIVED',
            'meaning': 'original write/result/actual associations, not a caller-selected complete writer set'}


def _sources(ctx, budget):
    result, cursor = [], None
    for ordinal, binding in enumerate(ctx.model.source_inventories):
        if not budget.spend('composition_inventory', path=f'composition.sources[{ordinal}]'):
            cursor = {'table': 'source_inventories', 'ordinal': ordinal}
            break
        resolution = ctx.source_resolutions.get(binding.id)
        sources = []
        for source_number, source in enumerate(() if resolution is None else resolution.sources):
            if not budget.spend('composition_inventory_source',
                                path=f'composition.sources[{ordinal}].sources[{source_number}]'):
                cursor = {'table': 'source_inventories', 'ordinal': ordinal,
                          'source_ordinal': source_number}
                break
            evidence = None if source.bundle is None else source.bundle.evidence
            edges = []
            for edge_ordinal, edge in enumerate(() if evidence is None else evidence.edges.values()):
                if not budget.spend('composition_source_edge',
                                    path=f'composition.sources[{ordinal}].sources[{source_number}].edges[{edge_ordinal}]'):
                    cursor = {'table': 'source_inventories', 'ordinal': ordinal,
                              'source_ordinal': source_number, 'edge_ordinal': edge_ordinal}
                    break
                edges.append(edge)
            sources.append({'key': source.key, 'raw_artifact': source.raw_artifact,
                'raw_record_count': len(source.raw_records),
                'evidence_artifact': source.evidence_artifact, 'values_artifact': source.values_artifact,
                'semantic_artifact': source.semantic_artifact,
                'projection_replayed': source.projection_replayed,
                'evidence': {'instance_count': None if evidence is None else len(evidence.instances),
                    'edge_count': None if evidence is None else len(evidence.edges),
                    'edges': tuple(edges),
                    'pending_edges': None if evidence is None else len(evidence.edges) - len(edges),
                    'status': 'UNRESOLVED' if evidence is None else 'TRUNCATED' if cursor is not None else 'DERIVED',
                    'meaning': 'original supplied evidence edges, including effects independent of ContentReads; not native complete capture'}})
            if cursor is not None:
                break
        result.append({'binding': binding, 'sources': tuple(sources),
            'total_sources': None if resolution is None else len(resolution.sources),
            'pending_sources': None if resolution is None else len(resolution.sources) - len(sources),
            'format_replayed': None if resolution is None else resolution.format_replayed,
            'gaps': () if resolution is None else resolution.gaps})
        if cursor is not None:
            break
    return {'items': tuple(result), 'total': len(ctx.model.source_inventories),
            'pending': len(ctx.model.source_inventories) - len(result), 'cursor': cursor,
            'status': 'TRUNCATED' if cursor is not None else 'DERIVED',
            'capture_complete': False,
            'meaning': 'supplied manifest/projection counts; not all native writers, aliases or external effects'}


def _lifecycle(ctx, budget):
    work_views, cursor = [], None
    for ordinal, work in enumerate(ctx.model.submissions):
        if not budget.spend('composition_work', path=f'composition.work[{ordinal}]'):
            cursor = {'table': 'submissions', 'ordinal': ordinal}
            break
        operation = (ctx.occurrences[work.origin.operation]
                     if type(work.origin) is WithinOperation else None)
        core = None if operation is None else ctx.core_operations[operation.operation]
        transfer = None if operation is None else ctx.outgoing.get(operation.id)
        successor = (None if transfer is None else core.terminator.successors[transfer.successor_index])
        groups = (('outcomes', ctx.work_outcomes.get(work.id, ())),
                  ('quiescences', ctx.work_quiescences.get(work.id, ())),
                  ('retentions', ctx.work_retentions.get(work.id, ())))
        view = {'submission': work, 'host_occurrence': operation,
                'host_operation': core, 'actual_host_transfer': transfer,
                'actual_host_successor': successor,
                'host_before': None if operation is None else ctx.current(
                    OccurrencePoint(operation.id, OccurrencePhase.BEFORE)), 'counts': {}, 'pending': {}}
        for name, rows in groups:
            selected = []
            for position, row in enumerate(rows):
                if not budget.spend('composition_work_relation',
                                    path=f'composition.work[{ordinal}].{name}[{position}]'):
                    if cursor is None:
                        cursor = {'table': 'submissions', 'ordinal': ordinal,
                                  'relation': name, 'relation_ordinal': position}
                    break
                selected.append(row)
            view[name] = tuple(selected)
            view['counts'][name] = len(rows)
            view['pending'][name] = len(rows) - len(selected)
        work_views.append(view)
        if cursor is not None:
            break
    return {'work': tuple(work_views), 'total_work': len(ctx.model.submissions),
            'pending_work': len(ctx.model.submissions) - len(work_views), 'cursor': cursor,
            'handles': ctx.model.handles, 'target_lifetime_events': ctx.model.target_lifetime_events,
            'status': 'TRUNCATED' if cursor is not None else 'DERIVED',
            'meaning': 'host completion, work terminal, quiescence, borrow release, handle end and retirement remain distinct; no free or persistence proof'}


def resource_composition(ctx, scopes, *, budget):
    """Return one detached projection; the caller already prepared this context."""
    from .composition_requirements import requirements
    from .composition_routes import routes

    observations = requirements(ctx, scopes, budget=budget)
    # Publish the Q-phase receipt, not the mutable traversal controller. Later
    # phases continue charging the same controller without changing this view.
    observations['budget'] = {'limit': budget.limit, 'used': budget.used,
        'remaining': budget.remaining, 'costs': dict(budget.costs),
        'truncated': budget.truncated, 'scope': 'Q scans and joins at the end of the Q phase'}
    reads = _reads(ctx, budget)
    productions = _production(ctx, budget)
    sources = _sources(ctx, budget)
    dependencies = routes(ctx, budget=budget)
    lifecycle = _lifecycle(ctx, budget)
    gaps = list(ctx.gaps)
    for row in (*observations['gaps'], *dependencies['gaps'], *budget.gaps):
        issue = row if type(row) is ExecutionIssue else ExecutionIssue(
            row['code'], str(row['path']), row['message'])
        gaps.append(issue)
    return deepcopy({'requested_scopes': scopes, 'inventory': _inventory(ctx),
        'observations': observations, 'reads': reads, 'production': productions,
        'sources': sources, 'routes': dependencies, 'lifecycle': lifecycle,
        'external_preparation': ctx.external_preparation,
        'gaps': tuple(dict.fromkeys(gaps)),
        'status': 'TRUNCATED' if budget.truncated else 'DERIVED',
        'budget': {'limit': budget.limit, 'used': budget.used, 'remaining': budget.remaining,
                   'costs': dict(budget.costs), 'truncated': budget.truncated,
                   'scope': 'shared derived Q joins, content queries, routes and work/source views; preparation/FK/replay and detached copying separate'},
        'scope_inventory_complete': False, 'source_runtime_verified': False,
        'runtime_visibility_verified': False, 'transformation_authorized': False})


__all__ = [
    'routes',
    'prepare_scopes',
    'requirements',
    'validate_scopes',
    'resource_composition',
    'CompositionScope',
]
