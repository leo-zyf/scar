"""Call demand."""

from __future__ import annotations

from collections import Counter, deque
from copy import copy
import hashlib
import pickle
import heapq
from pathlib import Path
from scar.analysis.program_context_store import PreparedContextStore, ContextStoreLimit
from scar.analysis.program_lowering import prepare_source
from scar.ir.program.function_origins import CallCompilationProfile
from scar.ir.program.functions import BodyLoweringProfile
from dataclasses import dataclass, field
from scar.ir.program.call_composition import (
    BodyExpansionV6,
    DemandOrigin,
    EnteredCallCode,
    EntryRoot,
    LegacyObservation,
    Remainder,
    SourceCallCandidate,
    ExpansionScheduleV6,
)
from scar.ir.program.expansion import ContextStoreBudget, ContextUnavailable
from scar.ir.program.inventory import BodyStatus, DefinitionInventory, FileInventory
from scar.ir.v2.common import SourceReference
from scar.ir.v2.ids import OperationDefinitionID


# Private raw-call/source/inventory composition for inventory v6.
#
# Only this coordinator shares contexts created during the current invocation.
# Public replay starts from raw collector files and current source, never from
# serialized claims about source matches or queue admission.
def _fresh_failure(error):
    """Keep exception fields without retaining or reattaching a heavy frame."""
    result = copy(error)
    result.__traceback__ = result.__context__ = result.__cause__ = None
    return result


class _SourceWorkspace:
    """One preparation per source, backed by the bounded private context store."""

    def __init__(self, project_root, budget, *, entry=None, source_roots=(), inventory_budget=None):
        self.root = Path(project_root).resolve()
        self.store = PreparedContextStore(budget)
        self.preparations = Counter()
        self._errors = {}
        self._source_errors = {}
        self._projection_errors = {}
        self._sealed = set()
        self._projections = {}
        self._observed_paths = set()
        self.entry, self.source_roots, self.inventory_budget = entry, source_roots, inventory_budget
        self.entry_frontier = None
        self.captured_session = None

    def bind_captured_session(self, session):
        from scar.analysis.call_seeds import _CapturedSourceSession
        if type(session) is not _CapturedSourceSession or session.root != self.root:
            raise TypeError('workspace requires the same normal captured source session')
        if self.captured_session is not None and self.captured_session is not session:
            raise ValueError('workspace captured source owner changed')
        self.captured_session = session

    def prime_sources(self, descriptors):
        """Give the entry closure first claim on the private context budget.

        Source-match budget order remains the original code-hash order. Light
        source projections survive a full-context spool failure, so lack of
        scratch space does not turn an otherwise verifiable code into a match.
        """
        self._observed_paths = {code.filename for code in descriptors}
        if self.entry is None:
            return
        from scar.analysis.program_inventory import _ordered_files
        from scar.analysis.source_discovery import _discover_sources_from_call_sources
        self._observed_paths = {code.filename for code in descriptors}
        budget = self.inventory_budget
        self.entry_frontier = _discover_sources_from_call_sources(self.entry,
            project_root=self.root, source_roots=self.source_roots, source_witnesses=(),
            max_files=budget.max_files, max_bytes=budget.max_bytes,
            max_directories=budget.max_directories,_captured_source_session=self.captured_session)
        for file in _ordered_files(self.entry_frontier):
            try:
                context = self.prepare(file.snapshot.path)
                if context.semantics.sources[file.snapshot.path].fingerprint != file.snapshot.fingerprint:
                    raise ValueError("source changed between entry discovery and preparation")
            except (OSError, ValueError, TypeError, KeyError, SyntaxError, RecursionError):
                # The normal inventory path publishes the concrete boundary.
                pass

    def _current_fingerprint(self,path):
        if self.captured_session is not None and self.captured_session.has_path(path):
            return self.captured_session.archived(path).snapshot.fingerprint
        from scar.analysis.source_discovery import _read_source
        try:
            text, _, _ = _read_source(Path(path), 4_000_000)
        except OverflowError as error:
            raise ValueError("source exceeds the existing 4 MB control-model input budget") from error
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest()

    def source_projection(self, path):
        key = str(path)
        self._observed_paths.add(key)
        if key not in self._projections and key not in self._projection_errors:
            context = self.prepare(path)
            if key not in self._projections:
                self._publish_projection(context)
        if key in self._projection_errors:
            raise _fresh_failure(self._projection_errors[key]) from None
        projection = self._projections[key]
        if self._current_fingerprint(path) != projection[0].snapshot.fingerprint:
            raise ValueError("source changed since private source-witness preparation")
        return projection

    def _publish_projection(self, context):
        from scar.analysis.call_edges import _source_witness_projection
        key = str(context.path)
        try:
            self._projections[key] = _source_witness_projection(context)
        except (ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
            self._projection_errors[key] = _fresh_failure(error)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.store.close()

    def prepare(self, path):
        path = Path(path)
        if not path.is_absolute() or path != path.resolve() or not path.is_relative_to(self.root):
            raise ValueError("source workspace path must be canonical and within project root")
        key = str(path)
        if key in self._source_errors:
            raise _fresh_failure(self._source_errors[key]) from None
        if key in self._errors:
            raise ValueError(self._errors[key])
        if key in self._sealed:
            fingerprint = self._current_fingerprint(path)
            context = self.store.get(key)
            if context.semantics.sources[key].fingerprint != fingerprint:
                raise ValueError("source changed since private context preparation")
            return context
        self.preparations[key] += 1
        try:
            context = (self.captured_session.prepare(path) if self.captured_session is not None
                       and self.captured_session.has_path(path) else prepare_source(path,project_root=self.root))
        except (OSError, ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
            self._source_errors[key] = _fresh_failure(error)
            raise
        if key in self._observed_paths:
            self._publish_projection(context)
        try:
            self.store.put(context)
        except (OSError, ValueError, TypeError, KeyError, RecursionError, pickle.PickleError) as error:
            # A scratch budget must not invalidate a code/source match that
            # can be checked using this first context. Later consumers retain
            # an explicit unavailable boundary instead of preparing it again.
            self._errors[key] = (str(error) if isinstance(error, ContextStoreLimit)
                                 else "private context unavailable: " + type(error).__name__)
        else:
            self._sealed.add(key)
        return context

    def measurements(self):
        return {"source_prepare_attempts": sum(self.preparations.values()),
                "distinct_source_prepare_paths": len(self.preparations),
                "source_prepare_max_per_path": max(self.preparations.values(), default=0),
                "source_workspace_failures": len(self._errors) + len(self._source_errors),
                "source_witness_projection_failures": len(self._projection_errors),
                "source_discovery_passes": 2 if self.entry_frontier is not None else 1}


def inventory_from_calls(entry, *, project_root, source_roots, budget, call_trace,
                         call_budget, context_budget, expected_call_sources=None,
                         compilation_profile=CallCompilationProfile.RECORDED_FUTURE_FLAGS,
                         body_profile=BodyLoweringProfile.BOUND_FRAME,_seed_report=None,_workspace=None,
                         coordinates=None):
    from scar.analysis.call_seeds import _read_call_source_seeds
    from scar.analysis.program_regions import expand_inventory
    from scar.analysis.program_inventory import ProgramInventory, _canonical
    from scar.analysis.source_discovery import _discover_sources_from_call_sources
    from scar.ir.program.functions import BodyLoweringProfile

    if type(body_profile) is not BodyLoweringProfile or body_profile not in (
            BodyLoweringProfile.BOUND_FRAME, BodyLoweringProfile.STRUCTURED_BODY, BodyLoweringProfile.STRUCTURED_ENV):
        raise ValueError("call-source inventory requires a frame or structured body profile")

    root = Path(project_root).resolve() if project_root is not None else Path(entry).absolute().parent.resolve()
    from contextlib import nullcontext
    scope=(nullcontext(_workspace) if _workspace is not None else
           _SourceWorkspace(root,context_budget,entry=entry,source_roots=source_roots,inventory_budget=budget))
    with scope as workspace:
        if _seed_report is None:
            report = _read_call_source_seeds(call_trace,root,budget=call_budget,
                _source_workspace=workspace,compilation_profile=compilation_profile,coordinates=coordinates)
        else:
            from scar.analysis.call_seeds import CallSourceSeedReport,_CapturedSourceSession
            if (type(_seed_report)is not CallSourceSeedReport or
                    type(_seed_report._captured_session)is not _CapturedSourceSession or
                    workspace.captured_session is not _seed_report._captured_session):
                raise ValueError('inventory seed reuse requires this live normal workspace/session')
            report=_seed_report
        if expected_call_sources is not None and _canonical(report.to_dict()) != _canonical(expected_call_sources):
            raise ValueError("call seeds differ from current raw trace/source replay")
        discovery = _discover_sources_from_call_sources(entry, project_root=root,
            source_roots=source_roots, source_witnesses=report.eligible_code_links(),
            max_files=budget.max_files, max_bytes=budget.max_bytes,
            max_directories=budget.max_directories,_captured_source_session=workspace.captured_session)
        files, schedule, measurements, interfaces, frames, structured = expand_inventory(
            discovery, None, budget, context_budget, discovery.files, interface_model=True,
            body_profile=body_profile, call_source_report=report,
            source_workspace=workspace)
        scope = _call_source_scope(report, files)
        result = ProgramInventory(discovery, files, budget, expansion=schedule,
            analysis_measurements=measurements, interfaces=interfaces,
            profile=body_profile, frames=frames, structured=structured,
            call_sources=report, call_source_scope=scope,
            source_analysis_measurements=workspace.measurements())
        result._check_profile()
        return result


def derive_call_source_admission(origin, code_source_link, *, source_snapshot,
                                 indexed_definitions, body=None, source_unavailable=False):
    """Derive one static source/body association using the full original join.

    ``body`` is the existing BodyInventory produced/read through the public
    body pipeline, not a caller success flag or standalone region claim.
    This correspondence does not prove effects or any current invocation.
    """
    from scar.ir.program.call_composition import CallSourceAdmission, CallSourceAdmissionStatus as Status, EnteredCallCode
    from scar.ir.program.call_edges import CodeSourceLink, CallLinkStatus
    from scar.ir.program.inventory import BodyInventory, BodyStatus, DefinitionInventory
    from scar.ir.semantics_v2 import SourceSnapshot
    if (type(origin) is not EnteredCallCode or type(code_source_link) is not CodeSourceLink
            or code_source_link.status is not CallLinkStatus.VERIFIED
            or origin.code != code_source_link.code or type(indexed_definitions) is not tuple
            or any(type(item) is not DefinitionInventory for item in indexed_definitions)
            or type(source_unavailable) is not bool):
        raise TypeError("static admission needs exact original origin/link/definition records")
    if source_snapshot is None:
        return CallSourceAdmission(origin, Status.UNAVAILABLE,
            reason="source not admitted by discovery budgets or source boundary")
    if type(source_snapshot) is not SourceSnapshot:
        raise TypeError("static admission source snapshot must be exact")
    link = code_source_link
    if source_snapshot.path != link.snapshot.path or source_snapshot.fingerprint != link.snapshot.fingerprint:
        return CallSourceAdmission(origin, Status.MISMATCH,
            reason="discovered source content differs from the verified code/source snapshot")
    matches = [definition for definition in indexed_definitions
               if definition.id == link.definition and definition.source == link.source]
    if len(matches) != 1:
        status = Status.UNAVAILABLE if not indexed_definitions and source_unavailable else Status.MISMATCH
        return CallSourceAdmission(origin, status,
            reason="verified unique definition was not indexed in the admitted source")
    definition = matches[0]
    if body is not None and (type(body) is not BodyInventory or body.definition != definition.id):
        raise ValueError("static admission body differs from its unique original definition")
    if body is not None and body.graph is not None:
        if (body.status is not BodyStatus.LOWERED_WITH_GAPS or body.metadata is None
                or body.metadata.source != source_snapshot):
            raise ValueError("canonical body status/source contradicts its static admission")
        return CallSourceAdmission(origin, Status.CORE_LINKED,
            source_snapshot, definition.id, definition.source, body.graph.root)
    reason = body.reason if body is not None and body.reason else "indexed source has no supported body interface"
    return CallSourceAdmission(origin, Status.INDEXED_WITHOUT_BODY,
        source_snapshot, definition.id, definition.source, reason=reason)


def derive_entry_source_admission(origin, code_source_link, *, catalog, entry, body):
    """Associate one checked original entry/body, without project inventory slots.

    This is static membership only. Current frame/process membership is checked
    independently by the invocation binder; no callable/environment is inferred.
    """
    from scar.analysis.source_entry_catalog import SourceEntryCatalog, ResolvedSourceEntry
    from scar.ir.program.call_composition import EnteredCallCode, CallSourceAdmission, CallSourceAdmissionStatus
    from scar.ir.program.call_edges import CodeSourceLink, CallLinkStatus
    from scar.ir.program.inventory import BodyInventory, BodyStatus
    if (type(catalog) is not SourceEntryCatalog or type(entry) is not ResolvedSourceEntry
            or type(origin) is not EnteredCallCode or type(code_source_link) is not CodeSourceLink
            or type(body) is not BodyInventory):
        raise TypeError("qualified admission requires exact catalog/entry/origin/link/body")
    owner = catalog.require_entry(entry)
    owner.resolve_definition(entry.reference)
    link, reference = entry.source_link, entry.reference
    if (code_source_link != link or link.status is not CallLinkStatus.VERIFIED
            or origin.code != entry.captured.key or link.code != origin.code
            or link.snapshot != reference.snapshot or link.definition != reference.definition
            or link.source != reference.source or body.definition != reference.definition
            or body.status is not BodyStatus.LOWERED_WITH_GAPS or body.graph is None
            or body.metadata is None or body.metadata.source != reference.snapshot
            or body.construction_operations is not None):
        raise ValueError("qualified body admission contradicts its original code/source/body")
    groups = [row for row in catalog.metadata()["groups"]
              if row["process_id"] == origin.process_id and row["code"] == origin.code
              and row.get("entry_key") == entry.key]
    if len(groups) != 1:
        raise ValueError("qualified admission origin is not one original catalog group")
    return CallSourceAdmission(origin, CallSourceAdmissionStatus.CORE_LINKED,
        reference.snapshot, reference.definition, reference.source, body.graph.root)


def _call_source_scope(report, files):
    """Link only final bodies, after every preparation/interface failure."""
    from scar.ir.program.call_composition import CallSourceScope, EnteredCallCode
    by_path = {file.snapshot.path: file for file in files}
    links = {link.code: link for link in report.eligible_code_links()}
    admissions = []
    for group in report.eligible_groups():
        link = links[group.code]
        file = by_path.get(link.snapshot.path)
        definitions = () if file is None else file.definitions
        body = (next((body for body in file.bodies if body.definition == link.definition), None)
                if file is not None else None)
        admissions.append(derive_call_source_admission(EnteredCallCode(group.process_id, group.code), link,
            source_snapshot=None if file is None else file.snapshot, indexed_definitions=definitions,
            body=body, source_unavailable=file is not None and bool(file.gaps)))
    return CallSourceScope(tuple(admissions))


def replay_inventory_from_calls(document, *, discovery, roots, budget, context_ceiling, call_ceiling):
    from scar.analysis.call_seeds import CallSourceSeedReport
    from scar.analysis.program_inventory import _check_context_budget
    from scar.ir.program.call_composition import CallSourceScope, ExpansionScheduleV6
    from scar.ir.program.frames import FrameModel
    from scar.ir.program.functions import BodyLoweringProfile, FunctionInterfaces
    from scar.ir.record_codec import decode

    if document["runtime_scope"] is not None:
        raise ValueError("inventory v6 cannot silently combine legacy observations and call-source groups")
    schedule = decode(ExpansionScheduleV6, document["expansion"])
    _check_context_budget(schedule.budget, context_ceiling)
    interfaces = decode(FunctionInterfaces, document["interfaces"])
    expected_profile = (BodyLoweringProfile.STRUCTURED_ENV if document["schema_version"] == 8 else
                        BodyLoweringProfile.STRUCTURED_BODY if document["schema_version"] == 7
                        else BodyLoweringProfile.BOUND_FRAME)
    if (decode(BodyLoweringProfile, document["profile"]) is not expected_profile
            or interfaces.profile is not expected_profile):
        raise ValueError("inventory version differs from its call-source body profile")
    if expected_profile is BodyLoweringProfile.BOUND_FRAME:
        decode(tuple[FrameModel, ...], document["frames"])
    else:
        from scar.ir.program.structured import StructuredBodyModel
        decode(tuple[StructuredBodyModel, ...], document["structured"])
    decode(CallSourceScope, document["call_source_scope"])
    model = CallSourceSeedReport._decode_replay_input(document["call_sources"],
        verification_budget=call_ceiling)
    compilation_profile = CallSourceSeedReport._compilation_profile(document["call_sources"])
    if model.project_root != discovery.project_root:
        raise ValueError("call-source project root differs from source discovery")
    return inventory_from_calls(discovery.entry, project_root=discovery.project_root,
        source_roots=roots, budget=budget, call_trace=model.trace_directory,
        call_budget=model.budget, context_budget=schedule.budget,
        expected_call_sources=document["call_sources"], compilation_profile=compilation_profile,
        body_profile=expected_profile)


# Demand queue primitives for inventory v6 call-source expansion.
#
# The queue records reasons for analysis work. Its priority is an analysis
# policy; it does not turn source candidates into runtime call facts.
def _matched_call_sources(report, files: tuple[FileInventory, ...]):
    """Match every structurally eligible D0 group to this indexed source view.

    The join intentionally requires the exact source path and decoded-source
    fingerprint, the exact definition ID, and the exact SourceReference. A
    matching filename or qualname is insufficient. The returned mapping also
    includes eligible non-function definitions so the coordinator can report
    their admission without attempting to lower them as function bodies.
    """
    if report is None:
        return {}
    model = getattr(report, "model", None)
    if model is None or not callable(getattr(report, "eligible_groups", None)):
        raise TypeError("call-source scheduling requires a replayed CallSourceSeedReport")
    links = {link.code: link for link in report.eligible_code_links()}
    by_path = {file.snapshot.path: file for file in files}
    matched = {}
    for group in report.eligible_groups():
        key = group.process_id, group.code
        link = links.get(group.code)
        if link is None or link.snapshot is None or link.definition is None or link.source is None:
            matched[key] = None
            continue
        file = by_path.get(link.snapshot.path)
        if (file is None or file.snapshot.path != link.snapshot.path
                or file.snapshot.fingerprint != link.snapshot.fingerprint):
            matched[key] = None
            continue
        definitions = [definition for definition in file.definitions
                       if definition.id == link.definition and definition.source == link.source]
        matched[key] = (link, definitions[0]) if len(definitions) == 1 else None
    return matched


@dataclass(slots=True)
class _Task:
    definition: OperationDefinitionID
    source: SourceReference
    priority: int
    origins: list[DemandOrigin] = field(default_factory=list)
    _origin_set: set = field(default_factory=set)
    version: int = 0
    started: bool = False
    attempted: bool = False
    construction_operations: int = 0
    status: BodyStatus = BodyStatus.NOT_ATTEMPTED
    operation_budget_charged: int = 0
    construction_cost_known: bool = True

    def add_origin(self, origin):
        if origin not in self._origin_set:
            self._origin_set.add(origin)
            self.origins.append(origin)


class CallDemandQueue:
    """FIFO within three priority frontiers, with one task per definition.

    Priorities are entry-reachable source candidates, orphan observed roots,
    then the static remainder. A task may be promoted before it is processed;
    later origins are retained even after construction has finished.
    """

    ENTRY = 0
    OBSERVED = 1
    REMAINDER = 2

    def __init__(self, functions: dict[OperationDefinitionID, DefinitionInventory]):
        self.functions = functions
        self._tasks = {}
        self._heap = []
        self._serial = 0
        self._processed = []

    def add(self, definition, priority, origin):
        if definition not in self.functions:
            return None
        if type(priority) is not int or priority not in (self.ENTRY, self.OBSERVED, self.REMAINDER):
            raise ValueError("unknown call-demand priority")
        if type(origin) not in (LegacyObservation, EnteredCallCode, SourceCallCandidate, Remainder):
            raise TypeError("call-demand origin must be a typed record")
        task = self._tasks.get(definition)
        if task is None:
            task = _Task(definition, self.functions[definition].source, priority)
            self._tasks[definition] = task
            self._push(task)
        elif type(origin) is not Remainder and not task.started:
            remainder = Remainder()
            if remainder in task._origin_set:
                task._origin_set.remove(remainder)
                task.origins.remove(remainder)
        task.add_origin(origin)
        if not task.started and priority < task.priority:
            task.priority = priority
            task.version += 1
            self._push(task)
        return task

    def add_remainder(self, definition):
        if definition in self._tasks:
            return self._tasks[definition]
        return self.add(definition, self.REMAINDER, Remainder())

    def _push(self, task):
        self._serial += 1
        heapq.heappush(self._heap, (task.priority, self._serial, task.version, task.definition))

    def pop(self):
        while self._heap:
            priority, _, version, definition = heapq.heappop(self._heap)
            task = self._tasks[definition]
            if task.started or version != task.version or priority != task.priority:
                continue
            task.started = True
            self._processed.append(task)
            return task
        return None

    @staticmethod
    def mark_unattempted(task, status):
        if type(status) is not BodyStatus or status is not BodyStatus.NOT_ATTEMPTED:
            raise ValueError("unattempted task must retain NOT_ATTEMPTED status")
        task.status = status

    @staticmethod
    def mark_result(task, *, status, construction_operations, operation_budget_charged,
                    construction_cost_known):
        if type(status) is not BodyStatus or status is BodyStatus.NOT_ATTEMPTED:
            raise ValueError("attempted task needs a terminal body status")
        task.attempted = True
        task.status = status
        task.construction_operations = construction_operations
        task.operation_budget_charged = operation_budget_charged
        task.construction_cost_known = construction_cost_known

    def schedule(self, budget: ContextStoreBudget, entry_roots: tuple[EntryRoot, ...],
                 unavailable_contexts: tuple[ContextUnavailable, ...]) -> ExpansionScheduleV6:
        entries = tuple(BodyExpansionV6(
            task.definition, task.source, tuple(task.origins), task.attempted,
            task.construction_operations, task.status, task.operation_budget_charged,
            task.construction_cost_known,
        ) for task in self._processed)
        return ExpansionScheduleV6(
            budget, entries, tuple(entry_roots), tuple(unavailable_contexts),
        )


def entry_roots_for(files: tuple[FileInventory, ...], module_initializers: dict[str, OperationDefinitionID],
                    entry_path: str) -> tuple[EntryRoot, ...]:
    """Build genuine module-scope roots from indexed source files."""
    roots = []
    for file in files:
        if file.snapshot.path == entry_path and file.snapshot.path in module_initializers:
            roots.append(EntryRoot(file.snapshot, module_initializers[file.snapshot.path]))
    return tuple(roots)


def entry_frontier_initializers(discovery, source_workspace, module_initializers):
    """Return module scopes in the entry's nondeferred local-import closure.

    These scopes contribute only conditional source-call candidates. Their
    presence does not assert that a guarded or dynamically selected import ran.
    """
    frontier = getattr(source_workspace, "entry_frontier", None) or discovery
    files = {file.snapshot.path for file in frontier.files}
    adjacency = {path: [] for path in files}
    for dependency in frontier.imports:
        if dependency.deferred or dependency.owner_qualname != "<module>":
            continue
        targets = adjacency.get(dependency.source.path)
        if targets is None:
            continue
        for candidate in dependency.candidates:
            if candidate.namespace:
                continue
            for path in (*candidate.parent_initializers, candidate.path):
                if path in files and path not in targets:
                    targets.append(path)

    ordered_paths, seen = [], {discovery.entry}
    pending = deque((discovery.entry,))
    while pending:
        path = pending.popleft()
        ordered_paths.append(path)
        for target in adjacency.get(path, ()):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return tuple(module_initializers[path] for path in ordered_paths if path in module_initializers)


__all__ = [
    'CallDemandQueue',
    '_matched_call_sources',
    'entry_frontier_initializers',
    'entry_roots_for',
    'inventory_from_calls',
    'derive_call_source_admission',
    'derive_entry_source_admission',
    'replay_inventory_from_calls',
]
