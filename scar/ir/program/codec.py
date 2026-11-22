"""Strict JSON codec for the immutable program graph records."""
from __future__ import annotations

from dataclasses import fields
import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scar.ir.program.queries import ProgramIndex

from scar.ir.record_codec import decode, encode, loads as _loads

from scar.ir.program.model import LocalBindingValue, Operation, ProgramGraph, ProtocolValue, TerminatorKind
from scar.ir.program.control import (
    ControlScopeKind, PendingDispatchTerminator, ProtocolValueKind, ProtocolValueType,
)
from scar.ir.program.source_subjects import (
    CLASS_FRAME_ABI_RECORD_TAGS, FRAME_ROOT_RECORD_TAGS, STATIC_FRAME_ROOT_RECORD_TAGS,
    ClassFrameABI, FrameKind, ModuleUnitSource, SourceCompilationFrameRoot,
)
from scar.ir.program.cleanup import CleanupRouteTerminator
from scar.ir.program.resumption import (
    ResumeInput, SavedInput, SuspendTerminator,
)
from scar.ir.program.drivers import BodyDriveResult, BodyDriveTerminator
from scar.ir.program.state_addresses import STATE_ADDRESS_RECORD_TAGS
from scar.ir.program.literal_sources import ExactLiteralOrigin, LITERAL_SOURCE_RECORD_TAGS
from scar.ir.program.model import InvokeLiteralResult
from scar.ir.program.protocols import (
    ADMISSION_RECORD_TAGS, AdmissionPredicate, PROTOCOL_RECORD_TAGS,
)
from scar.ir.program.protocol_connections import (
    LOOKUP_FALLBACK_RECORD_TAGS, PROTOCOL_CONNECTION_RECORD_TAGS,
    SessionOccupancyCondition,
    has_async_initial_control_capability, has_delegated_method_lookup_capability,
    has_lookup_fallback_capability,
    has_protocol_connection_capability,
)
from scar.ir.program.exception_matches import EXCEPTION_MATCH_RECORD_TAGS
from scar.ir.program.protocol_await_records import (
    AWAIT_ACQUISITION_RECORD_TAGS, has_await_acquisition_capability,
)
from scar.ir.program.call_effect_expansion import CALL_EFFECT_EXPANSION_RECORD_TAGS
from scar.ir.program.stored_values import STORED_VALUE_RECORD_TAGS, has_stored_value_capability
from scar.ir.program.resources import ResourceRole


SCHEMA = "scar.program-graph"
SCHEMA_VERSION = 32
_GRAPH_FIELDS = frozenset(item.name for item in fields(ProgramGraph))
_BINDING_FIELDS = frozenset({"binding_definitions", "binding_reads"})
_RESOURCE_FIELDS = frozenset({"resources", "resource_requirements", "resource_uses",
                              "resource_locations"})
_ENVIRONMENT_FIELDS = frozenset({"environments", "function_entries"})
_CONTROL_FIELDS = frozenset({"control_scopes", "control_ports", "exception_packets",
                             "pending_completions"})
_PROVIDER_FIELDS = frozenset({"provider_envelopes", "provider_regions", "provider_bindings"})
_NATIVE_FIELDS = frozenset({"native_boundaries", "boundary_relations"})
_CLEANUP_FIELDS = frozenset({"handled_context_domains", "cleanup_interfaces", "cleanup_enters",
                             "cleanup_resumes", "cleanup_replacements", "handled_context_reads"})
_RESUMPTION_FIELDS = frozenset({"resumable_bodies", "frame_context_initials",
                                "frame_context_attaches", "frame_context_detaches"})
_DRIVER_FIELDS = frozenset({"resumable_creates", "raw_driver_requests",
                           "async_request_creates", "request_association_reads",
                           "driver_interfaces", "final_body_gates", "body_drives"})
_PUBLICATION_FIELDS = frozenset({"driver_lifecycles", "body_outcome_commits"})
_NO_BODY_FIELDS = frozenset({"created_close_tests", "no_body_finishes"})
_SESSION_RELEASE_FIELDS = frozenset({"session_updates", "terminal_frame_releases"})
_STATE_ADDRESS_FIELDS = frozenset({"state_address_refinements"})
_LITERAL_FIELDS = frozenset({"literal_evaluations"})
_PROTOCOL_FIELDS = frozenset({"protocol_scopes", "protocol_operations"})
_CONNECTION_FIELDS = frozenset({"protocol_call_inputs", "protocol_result_consumers",
                                "protocol_bindings"})
_EXCEPTION_MATCH_FIELDS = frozenset({"exception_matches"})
_CALL_EFFECT_FIELDS = frozenset({"call_effect_expansions"})
_STORED_VALUE_FIELDS = frozenset({"stored_value_writes", "stored_value_reads"})
_FRAME_ROOT_FIELDS = frozenset({"frame_roots"})
_ATTRIBUTE_LOOKUP_FIELDS = frozenset({"attribute_lookups"})
# Each capability owns the fields it introduced. Intermediate schema versions
# can extend nested tags while keeping the same set of top-level fields.
_ADDED_FIELDS = {
    3: _BINDING_FIELDS,
    4: _RESOURCE_FIELDS,
    5: _ENVIRONMENT_FIELDS,
    6: _CONTROL_FIELDS,
    7: _PROVIDER_FIELDS,
    8: _NATIVE_FIELDS,
    9: _CLEANUP_FIELDS,
    10: _RESUMPTION_FIELDS,
    11: _DRIVER_FIELDS,
    12: _PUBLICATION_FIELDS,
    13: _NO_BODY_FIELDS,
    14: _SESSION_RELEASE_FIELDS,
    15: _STATE_ADDRESS_FIELDS,
    16: _LITERAL_FIELDS,
    17: _PROTOCOL_FIELDS,
    19: _CONNECTION_FIELDS,
    21: _EXCEPTION_MATCH_FIELDS,
    25: _CALL_EFFECT_FIELDS,
    27: _STORED_VALUE_FIELDS,
    29: _FRAME_ROOT_FIELDS,
    32: _ATTRIBUTE_LOOKUP_FIELDS,
}
_FIELDS_BY_VERSION = {
    version: _GRAPH_FIELDS.difference(
        *(group for since, group in _ADDED_FIELDS.items() if since > version))
    for version in range(1, SCHEMA_VERSION + 1)
}
_HEADER_FIELDS = frozenset({"schema", "schema_version"})
_ENVELOPE_FIELDS_BY_VERSION = {
    version: _HEADER_FIELDS | graph_fields for version, graph_fields in _FIELDS_BY_VERSION.items()
}
_V19_CONNECTION_RECORD_TAGS = {record: tag for record, tag in PROTOCOL_CONNECTION_RECORD_TAGS.items()
                              if record is not SessionOccupancyCondition}
_V17_PROTOCOL_RECORD_TAGS = {record: tag for record, tag in PROTOCOL_RECORD_TAGS.items()
                             if record not in ADMISSION_RECORD_TAGS}
_LITERAL_RECORD_TAGS = {**LITERAL_SOURCE_RECORD_TAGS,
                        InvokeLiteralResult: ("kind", "INVOKE_LITERAL_RESULT")}
_CLEANUP_RECORD_TAGS = {CleanupRouteTerminator: ("kind", "CLEANUP_ROUTE")}
_RESUMPTION_RECORD_TAGS = {SuspendTerminator: ("kind", "SUSPEND")}
_V10_RECORD_TAGS = {**_CLEANUP_RECORD_TAGS, **_RESUMPTION_RECORD_TAGS}
_DRIVER_RECORD_TAGS = {
    BodyDriveTerminator: ("kind", "BODY_DRIVE"),
    # CompletionResult has the same dataclass shape. The explicit new tag
    # prevents older wire formats from decoding this projection as that type.
    BodyDriveResult: ("kind", "BODY_DRIVE_RESULT"),
}
_V11_RECORD_TAGS = {**_V10_RECORD_TAGS, **_DRIVER_RECORD_TAGS}


def _has_cleanup_capability(graph: ProgramGraph) -> bool:
    """Detect C1 even when declarations omit the required context domain."""
    return (
        any(getattr(graph, name) for name in _CLEANUP_FIELDS)
        or any(isinstance(value, ProtocolValue)
               and value.protocol is ProtocolValueKind.SAVED_HANDLED_CONTEXT
               for value in graph.values)
        or any(isinstance(operation.terminator, CleanupRouteTerminator)
               for operation in graph.operations)
        or any(isinstance(value_type, ProtocolValueType)
               and value_type.protocol is ProtocolValueKind.SAVED_HANDLED_CONTEXT
               for region in graph.regions for port in region.exits for value_type in port.types)
        or any(isinstance(value_type, ProtocolValueType)
               and value_type.protocol is ProtocolValueKind.SAVED_HANDLED_CONTEXT
               for operation in graph.operations for signature in operation.completions
               for value_type in signature.types)
    )


def _has_resumption_capability(graph: ProgramGraph) -> bool:
    """Detect C2 records, tagged terminators, and protocol values/types."""
    new_protocols = {
        ProtocolValueKind.FRAME_CONTEXT,
        ProtocolValueKind.DRIVE_CALLER_CONTEXT,
    }
    return (
        any(getattr(graph, name) for name in _RESUMPTION_FIELDS)
        or any(type(operation.terminator) is SuspendTerminator
               for operation in graph.operations)
        or any(isinstance(value, ProtocolValue) and value.protocol in new_protocols
               for value in graph.values)
        or any(isinstance(value_type, ProtocolValueType)
               and value_type.protocol in new_protocols
               for region in graph.regions for port in region.exits for value_type in port.types)
        or any(isinstance(value_type, ProtocolValueType)
               and value_type.protocol in new_protocols
               for operation in graph.operations for signature in operation.completions
               for value_type in signature.types)
        or any(isinstance(actual, (SavedInput, ResumeInput))
               for operation in graph.operations
               for edge in getattr(operation.terminator, "successors", ())
               for actual in edge.arguments)
    )


def _has_driver_capability(graph: ProgramGraph) -> bool:
    """Include raw-only boundaries and markers omitted from interface tables."""
    raw = ProtocolValueKind.RAW_DRIVER_REQUEST
    return (
        any(getattr(graph, name) for name in _DRIVER_FIELDS)
        or any(isinstance(value, ProtocolValue) and value.protocol is raw
               for value in graph.values)
        or any(isinstance(value_type, ProtocolValueType) and value_type.protocol is raw
               for region in graph.regions for port in region.exits for value_type in port.types)
        or any(isinstance(value_type, ProtocolValueType) and value_type.protocol is raw
               for operation in graph.operations for signature in operation.completions
               for value_type in signature.types)
        or any(isinstance(operation.terminator, BodyDriveTerminator)
               for operation in graph.operations)
        or any(isinstance(actual, BodyDriveResult)
               for operation in graph.operations
               for edge in getattr(operation.terminator, "successors", ())
               for actual in edge.arguments)
    )


def _has_publication_capability(graph: ProgramGraph) -> bool:
    """Only the two implemented D2.1 tables select publication wire v12."""
    return any(getattr(graph, name) for name in _PUBLICATION_FIELDS)


def _has_no_body_capability(graph: ProgramGraph) -> bool:
    """Include predicate-only declarations without asserting a finish exists."""
    return any(getattr(graph, name) for name in _NO_BODY_FIELDS)


def _has_session_release_capability(graph: ProgramGraph) -> bool:
    """Only implemented session-post/reference-release declarations select v14."""
    return any(getattr(graph, name) for name in _SESSION_RELEASE_FIELDS)


def _has_literal_capability(graph: ProgramGraph) -> bool:
    return bool(graph.literal_evaluations) or any(
        isinstance(value.origin, ExactLiteralOrigin) for value in graph.values)


def _has_protocol_capability(graph: ProgramGraph) -> bool:
    return any(getattr(graph, name) for name in _PROTOCOL_FIELDS)


_has_protocol_connection_capability = has_protocol_connection_capability


def _has_admission_capability(graph: ProgramGraph) -> bool:
    return any(type(row.contract) is AdmissionPredicate for row in graph.protocol_operations)


def _has_session_occupancy_capability(graph: ProgramGraph) -> bool:
    """Only the new read subject selects Program24; old async scopes do not."""
    return any(type(row.contract) is AdmissionPredicate
               and type(row.contract.subject) is SessionOccupancyCondition
               for row in graph.protocol_operations)


def _has_invoke_subset_capability(graph: ProgramGraph) -> bool:
    return any(operation.terminator is not None
               and operation.terminator.kind is TerminatorKind.INVOKE
               and len(operation.terminator.successors) == 1 for operation in graph.operations)


def _has_handle_table_capability(graph: ProgramGraph) -> bool:
    return (any(row.role is ResourceRole.HANDLE_TABLE for row in graph.resources)
            or any(row.role is ResourceRole.HANDLE_TABLE for row in graph.resource_requirements))


def _has_frame_root_capability(graph: ProgramGraph) -> bool:
    return (bool(graph.frame_roots)
            or any(scope.kind is ControlScopeKind.FRAME
                   or type(scope.source) is ModuleUnitSource for scope in graph.control_scopes))


def _has_static_frame_root_capability(graph: ProgramGraph) -> bool:
    return any(type(row) is SourceCompilationFrameRoot for row in graph.frame_roots)


def _has_class_frame_capability(graph: ProgramGraph) -> bool:
    return any(type(row) is SourceCompilationFrameRoot and row.kind is FrameKind.CLASS
               and type(row.entry) is ClassFrameABI for row in graph.frame_roots)


def dumps(graph: ProgramGraph) -> str:
    """Select only present capabilities; preserve historical field/tag/byte profiles."""
    from scar.ir.program.validation import require_valid

    if type(graph) is not ProgramGraph:
        raise TypeError("program graph must be ProgramGraph")
    require_valid(graph)
    if _has_cleanup_capability(graph) and not graph.handled_context_domains:
        raise ValueError("cleanup capability requires explicit handled-context domains")
    has_binding_capability = (
        bool(graph.binding_definitions or graph.binding_reads)
        or any(isinstance(value, LocalBindingValue) for value in graph.values))
    has_resumption = _has_resumption_capability(graph)
    if has_resumption and not graph.resumable_bodies:
        raise ValueError("resumption capability requires an explicit resumable body")
    has_driver = _has_driver_capability(graph)
    has_publication = _has_publication_capability(graph)
    has_no_body = _has_no_body_capability(graph)
    has_session_release = _has_session_release_capability(graph)
    has_addresses = bool(graph.state_address_refinements)
    has_literals = _has_literal_capability(graph)
    has_protocols = _has_protocol_capability(graph)
    has_protocol_connections = _has_protocol_connection_capability(graph)
    has_method_lookup = has_delegated_method_lookup_capability(graph)
    has_exception_matches = bool(graph.exception_matches)
    has_lookup_fallback = has_lookup_fallback_capability(graph)
    has_await_acquisition = has_await_acquisition_capability(graph)
    has_session_occupancy = _has_session_occupancy_capability(graph)
    has_call_effect_expansions = bool(graph.call_effect_expansions)
    has_async_initial_control = has_async_initial_control_capability(graph)
    has_stored_values = has_stored_value_capability(graph)
    has_admission = _has_admission_capability(graph)
    has_invoke_subset = _has_invoke_subset_capability(graph)
    has_frame_roots = _has_frame_root_capability(graph)
    has_static_frame_roots = _has_static_frame_root_capability(graph)
    has_class_frame = _has_class_frame_capability(graph)
    has_attribute_lookups = graph.attribute_lookups is not None
    tags = (_V11_RECORD_TAGS if has_driver else
            _V10_RECORD_TAGS if has_resumption else _CLEANUP_RECORD_TAGS)
    if has_addresses or has_protocols or has_exception_matches:
        tags = {**tags, **STATE_ADDRESS_RECORD_TAGS}
    if has_literals or has_protocols or has_exception_matches or has_stored_values:
        tags = {**tags, **_LITERAL_RECORD_TAGS}
    if has_protocols or has_protocol_connections or has_exception_matches:
        tags = {**tags, **(PROTOCOL_RECORD_TAGS if has_admission else _V17_PROTOCOL_RECORD_TAGS)}
    if has_protocol_connections or has_exception_matches:
        tags = {**tags, **PROTOCOL_CONNECTION_RECORD_TAGS}
    if has_exception_matches or has_lookup_fallback:
        tags = {**tags, **EXCEPTION_MATCH_RECORD_TAGS}
    if has_lookup_fallback:
        tags = {**tags, **LOOKUP_FALLBACK_RECORD_TAGS}
    if has_await_acquisition:
        tags = {**tags, **AWAIT_ACQUISITION_RECORD_TAGS}
    if has_call_effect_expansions:
        tags = {**tags, **CALL_EFFECT_EXPANSION_RECORD_TAGS}
    if has_stored_values:
        tags = {**tags, **STORED_VALUE_RECORD_TAGS}
    if has_frame_roots:
        tags = {**tags, **FRAME_ROOT_RECORD_TAGS}
    if has_static_frame_roots:
        tags = {**tags, **STATIC_FRAME_ROOT_RECORD_TAGS}
    if has_class_frame:
        tags = {**tags, **CLASS_FRAME_ABI_RECORD_TAGS}
    capabilities = (
        (has_attribute_lookups, 32),
        (has_class_frame, 31),
        (has_static_frame_roots, 30),
        (has_frame_roots, 29),
        (_has_handle_table_capability(graph), 28),
        (has_stored_values, 27),
        (has_async_initial_control, 26),
        (has_call_effect_expansions, 25),
        (has_session_occupancy, 24),
        (has_await_acquisition, 23),
        (has_lookup_fallback, 22),
        (has_exception_matches, 21),
        (has_method_lookup, 20),
        (has_protocol_connections, 19),
        (has_admission or has_invoke_subset, 18),
        (has_protocols, 17),
        (has_literals, 16),
        (has_addresses, 15),
        (has_session_release, 14),
        (has_no_body, 13),
        (has_publication, 12),
        (has_driver, 11),
        (has_resumption, 10),
        (graph.handled_context_domains, 9),
        (graph.native_boundaries, 8),
        (graph.provider_envelopes, 7),
        (graph.control_scopes, 6),
        (graph.environments, 5),
        (any(getattr(graph, name) for name in _RESOURCE_FIELDS), 4),
        (has_binding_capability, 3),
    )
    version = next((version for present, version in capabilities if present), 2)
    encoded = encode(graph, record_tags=tags)
    document = {"schema": SCHEMA, "schema_version": version,
                **{name: encoded[name] for name in _FIELDS_BY_VERSION[version]}}
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=(has_literals or has_protocols or has_protocol_connections
                                    or has_exception_matches),
                      allow_nan=False)


def _upgrade_v1_record(record: dict[str, Any]) -> dict[str, Any]:
    """Add only the N2 Operation field omitted by historical schema v1."""
    operations = record.get("operations")
    if type(operations) is not list:
        # Let the strict typed decoder report malformed collection shapes.
        return record
    upgraded_operations = []
    for operation in operations:
        if type(operation) is not dict:
            # Let the strict typed decoder report malformed operation records.
            upgraded_operations.append(operation)
            continue
        if "completions" in operation:
            raise ValueError("schema version 1 cannot contain Operation.completions")
        if "invoke_arguments" in operation:
            raise ValueError("schema version 1 cannot contain Operation.invoke_arguments")
        terminator = operation.get("terminator")
        if type(terminator) is dict and terminator.get("kind") == "INVOKE":
            raise ValueError("schema version 1 cannot contain INVOKE terminators")
        upgraded = dict(operation)
        upgraded["completions"] = []
        upgraded["invoke_arguments"] = []
        upgraded_operations.append(upgraded)

    values = record.get("values")
    if type(values) is list:
        for value in values:
            if (type(value) is dict and "referent_identity" in value
                    and type(value.get("origin")) is dict
                    and set(value["origin"]) == {"value"}):
                raise ValueError("schema version 1 cannot contain literal ReferenceValue origins")

    upgraded_record = dict(record)
    upgraded_record["operations"] = upgraded_operations
    return upgraded_record


def _decode_program_graph(payload: str) -> ProgramGraph:
    """Shared exact v1–v32 wire decoding; callers must validate the graph."""
    if type(payload) is not str:
        raise TypeError("program graph JSON payload must be str")
    document: Any = _loads(payload)
    if type(document) is not dict:
        raise ValueError("program graph document has incorrect top-level fields")
    if not _HEADER_FIELDS <= document.keys():
        raise ValueError("program graph document has incorrect top-level fields")
    if document["schema"] != SCHEMA:
        raise ValueError("unsupported program graph schema")
    version = document["schema_version"]
    if type(version) is not int or version not in _FIELDS_BY_VERSION:
        raise ValueError("unsupported program graph schema version")
    if document.keys() != _ENVELOPE_FIELDS_BY_VERSION[version]:
        raise ValueError("program graph document has incorrect top-level fields")
    record = {name: document[name] for name in _FIELDS_BY_VERSION[version]}
    # Historical schemas never recorded attribute lookups. Preserve that
    # absence separately from the empty typed tables introduced by other versions.
    if version == 32 and record["attribute_lookups"] is None:
        raise ValueError("schema version 32 requires explicitly recorded attribute lookup capability")
    missing = _GRAPH_FIELDS - record.keys()
    record.update({name: None if name == "attribute_lookups" else [] for name in missing})
    if version < 29:
        # The expanded typed source union and enum must not expand old wire.
        scopes = record.get("control_scopes")
        if type(scopes) is list:
            for scope in scopes:
                if type(scope) is not dict:
                    continue
                source = scope.get("source")
                if (scope.get("kind") == "FRAME" or type(source) is dict
                        and (source.get("source_kind") == "MODULE_UNIT" or "snapshot" in source)):
                    raise ValueError("historical schema cannot contain frame-root source capability")
    if version == 1:
        record = _upgrade_v1_record(record)
    if version < 9:
        # C0's dispatch wire is unchanged. C1 routes use an explicit new tag;
        # accepting it under an older schema would silently upgrade capability.
        operations = record.get("operations")
        if type(operations) is list:
            for operation in operations:
                term = operation.get("terminator") if type(operation) is dict else None
                if type(term) is dict and term.get("kind") == "CLEANUP_ROUTE":
                    raise ValueError("historical schema cannot contain cleanup protocol capability")
    if version < 11:
        # Inspect only canonical protocol-bearing fields. Capability rejection
        # must not depend on whether a containing legacy record constructs
        # successfully (for example a malformed completion signature).
        def raw_protocol(item):
            return type(item) is dict and item.get("protocol") == "RAW_DRIVER_REQUEST"

        values = record.get("values")
        if type(values) is list and any(raw_protocol(value) for value in values):
            raise ValueError("historical schema cannot contain driver capability")
        regions = record.get("regions")
        if type(regions) is list:
            for region in regions:
                ports = region.get("exits") if type(region) is dict else None
                if type(ports) is not list:
                    continue
                for port in ports:
                    types = port.get("types") if type(port) is dict else None
                    if type(types) is list and any(raw_protocol(item) for item in types):
                        raise ValueError("historical schema cannot contain driver capability")
        operations = record.get("operations")
        if type(operations) is list:
            for operation in operations:
                signatures = operation.get("completions") if type(operation) is dict else None
                if type(signatures) is list:
                    for signature in signatures:
                        types = signature.get("types") if type(signature) is dict else None
                        if type(types) is list and any(raw_protocol(item) for item in types):
                            raise ValueError("historical schema cannot contain driver capability")
                term = operation.get("terminator") if type(operation) is dict else None
                if type(term) is not dict:
                    continue
                if term.get("kind") == "BODY_DRIVE":
                    raise ValueError("historical schema cannot contain driver capability")
                edges = term.get("successors")
                if type(edges) is not list:
                    continue
                for edge in edges:
                    arguments = edge.get("arguments") if type(edge) is dict else None
                    if type(arguments) is list and any(
                            type(argument) is dict
                            and argument.get("kind") == "BODY_DRIVE_RESULT"
                            for argument in arguments):
                        raise ValueError("historical schema cannot contain driver capability")
    if version < 10:
        operations = record.get("operations")
        if type(operations) is list:
            for operation in operations:
                term = operation.get("terminator") if type(operation) is dict else None
                if type(term) is dict and term.get("kind") == "SUSPEND":
                    raise ValueError("historical schema cannot contain resumable-body capability")
    tags = (_V11_RECORD_TAGS if version >= 11 else
            _V10_RECORD_TAGS if version == 10 else
            _CLEANUP_RECORD_TAGS if version == 9 else None)
    if version >= 15:
        tags = {**tags, **STATE_ADDRESS_RECORD_TAGS}
    if version >= 16:
        tags = {**tags, **_LITERAL_RECORD_TAGS}
    if version >= 17:
        tags = {**tags, **(PROTOCOL_RECORD_TAGS if version >= 18 else _V17_PROTOCOL_RECORD_TAGS)}
    if version >= 19:
        tags = {**tags, **(PROTOCOL_CONNECTION_RECORD_TAGS if version >= 24
                          else _V19_CONNECTION_RECORD_TAGS)}
    if version >= 21:
        tags = {**tags, **EXCEPTION_MATCH_RECORD_TAGS}
    if version >= 22:
        tags = {**tags, **LOOKUP_FALLBACK_RECORD_TAGS}
    if version >= 23:
        tags = {**tags, **AWAIT_ACQUISITION_RECORD_TAGS}
    if version >= 25:
        tags = {**tags, **CALL_EFFECT_EXPANSION_RECORD_TAGS}
    if version >= 27:
        tags = {**tags, **STORED_VALUE_RECORD_TAGS}
    if version >= 29:
        tags = {**tags, **FRAME_ROOT_RECORD_TAGS}
    if version >= 30:
        tags = {**tags, **STATIC_FRAME_ROOT_RECORD_TAGS}
    if version >= 31:
        tags = {**tags, **CLASS_FRAME_ABI_RECORD_TAGS}
    raw_frame_roots = record.get("frame_roots")
    if version < 30 and type(raw_frame_roots) is list and any(
            type(item) is dict and item.get("frame_root_kind") == "SOURCE_COMPILATION"
            for item in raw_frame_roots):
        raise ValueError("historical schema cannot contain source-compilation frame roots")
    if version < 31 and type(raw_frame_roots) is list and any(
            type(item) is dict and type(item.get("entry")) is dict
            and item["entry"].get("entry_kind") == "STATIC_CLASS_FRAME_ABI_314_V1"
            for item in raw_frame_roots):
        raise ValueError("historical schema cannot contain static class-frame ABI capability")
    graph = decode(ProgramGraph, record, record_tags=tags)
    if version < 29 and _has_frame_root_capability(graph):
        raise ValueError("historical schema cannot contain frame-root capability")
    if version == 29 and not graph.frame_roots:
        raise ValueError("schema version 29 requires explicit frame-root capability")
    has_static_frame_roots = _has_static_frame_root_capability(graph)
    if version < 30 and has_static_frame_roots:
        raise ValueError("historical schema cannot contain source-compilation frame roots")
    if version == 30 and not has_static_frame_roots:
        raise ValueError("schema version 30 requires explicit source-compilation frame-root capability")
    has_class_frame = _has_class_frame_capability(graph)
    if version < 31 and has_class_frame:
        raise ValueError("historical schema cannot contain static class-frame ABI capability")
    if version == 31 and not has_class_frame:
        raise ValueError("schema version 31 requires explicit static class-frame ABI capability")
    has_handle_table = _has_handle_table_capability(graph)
    if version < 28 and has_handle_table:
        raise ValueError("historical schema cannot contain handle-table resource role")
    if version == 28 and not has_handle_table:
        raise ValueError("schema version 28 requires explicit handle-table resource capability")
    has_stored_values = has_stored_value_capability(graph)
    if version < 27 and has_stored_values:
        raise ValueError("historical schema cannot contain stored value capability")
    if version == 27 and not has_stored_values:
        raise ValueError("schema version 27 requires explicit stored value capability")
    has_async_initial_control = has_async_initial_control_capability(graph)
    if version < 26 and has_async_initial_control:
        raise ValueError("historical schema cannot contain async initial control capability")
    if version == 26 and not has_async_initial_control:
        raise ValueError("schema version 26 requires explicit async initial control capability")
    if version == 25 and not graph.call_effect_expansions:
        raise ValueError("schema version 25 requires explicit call effect expansion capability")
    has_session_occupancy = _has_session_occupancy_capability(graph)
    if version < 24 and has_session_occupancy:
        raise ValueError("historical schema cannot contain session occupancy capability")
    if version == 24 and not has_session_occupancy:
        raise ValueError("schema version 24 requires explicit session occupancy capability")
    has_await_acquisition = has_await_acquisition_capability(graph)
    if version < 23 and has_await_acquisition:
        raise ValueError("historical schema cannot contain await acquisition capability")
    if version == 23 and not has_await_acquisition:
        raise ValueError("schema version 23 requires explicit await acquisition capability")
    has_lookup_fallback = has_lookup_fallback_capability(graph)
    if version < 22 and has_lookup_fallback:
        raise ValueError("historical schema cannot contain lookup fallback capability")
    if version == 22 and not has_lookup_fallback:
        raise ValueError("schema version 22 requires explicit lookup fallback capability")
    if version < 21 and graph.exception_matches:
        raise ValueError("historical schema cannot contain exception matcher capability")
    if version == 21 and not graph.exception_matches:
        raise ValueError("schema version 21 requires explicit exception matcher capability")
    has_method_lookup = has_delegated_method_lookup_capability(graph)
    if version < 20 and has_method_lookup:
        raise ValueError("historical schema cannot contain delegated method lookup capability")
    if version < 18 and _has_admission_capability(graph):
        raise ValueError("historical schema cannot contain admission predicate capability")
    if version < 18 and _has_invoke_subset_capability(graph):
        raise ValueError("historical schema cannot contain single-completion INVOKE capability")
    if version == 18 and not (_has_admission_capability(graph) or _has_invoke_subset_capability(graph)):
        raise ValueError("schema version 18 requires explicit admission predicate capability or single-completion INVOKE")
    if version < 19 and _has_protocol_connection_capability(graph):
        raise ValueError("historical schema cannot contain protocol connection capability")
    if version == 19 and not _has_protocol_connection_capability(graph):
        raise ValueError("schema version 19 requires explicit protocol connection capability")
    if version == 20 and not has_method_lookup:
        raise ValueError("schema version 20 requires explicit delegated method lookup capability")
    if version < 17 and _has_protocol_capability(graph):
        raise ValueError("historical schema cannot contain protocol capability")
    if version == 17 and not _has_protocol_capability(graph):
        raise ValueError("schema version 17 requires explicit protocol capability")
    if version < 16 and _has_literal_capability(graph):
        raise ValueError("historical schema cannot contain exact literal capability")
    if version == 16 and not _has_literal_capability(graph):
        raise ValueError("schema version 16 requires exact literal capability")
    if version < 11 and _has_driver_capability(graph):
        raise ValueError("historical schema cannot contain driver capability")
    if version < 9 and _has_cleanup_capability(graph):
        raise ValueError("historical schema cannot contain cleanup protocol capability")
    if version < 10 and _has_resumption_capability(graph):
        raise ValueError("historical schema cannot contain resumable-body capability")
    if _has_resumption_capability(graph) and not graph.resumable_bodies:
        raise ValueError("resumption capability requires explicit resumable body")
    if version < 6 and (
            any(isinstance(v, ProtocolValue) for v in graph.values)
            or any(isinstance(o.terminator, PendingDispatchTerminator) for o in graph.operations)
            or any(isinstance(t, ProtocolValueType) for r in graph.regions for p in r.exits for t in p.types)
            or any(isinstance(t, ProtocolValueType) for o in graph.operations for s in o.completions for t in s.types)):
        raise ValueError("historical schema cannot contain control protocol capability")
    if version == 3 and not (
            graph.binding_definitions or graph.binding_reads
            or any(isinstance(value, LocalBindingValue) for value in graph.values)):
        raise ValueError("schema version 3 requires local-binding graph capability")
    if version == 4 and not any(getattr(graph, name) for name in _RESOURCE_FIELDS):
        raise ValueError("schema version 4 requires explicit resource graph capability")
    if version == 5 and not graph.environments:
        raise ValueError("schema version 5 requires explicit environment graph capability")
    if version == 6 and not graph.control_scopes:
        raise ValueError("schema version 6 requires explicit lexical control capability")
    if version == 7 and not graph.provider_envelopes:
        raise ValueError("schema version 7 requires explicit provider provenance capability")
    if version == 8 and not graph.native_boundaries:
        raise ValueError("schema version 8 requires explicit native boundary capability")
    if version == 9 and not graph.handled_context_domains:
        raise ValueError("schema version 9 requires explicit handled-context domain capability")
    if version == 10 and not _has_resumption_capability(graph):
        raise ValueError("schema version 10 requires explicit resumable-body capability")
    if version == 11 and not _has_driver_capability(graph):
        raise ValueError("schema version 11 requires explicit driver capability")
    if version == 12 and not _has_publication_capability(graph):
        raise ValueError("schema version 12 requires explicit body-publication capability")
    if version == 13 and not _has_no_body_capability(graph):
        raise ValueError("schema version 13 requires explicit no-body capability")
    if version == 14 and not _has_session_release_capability(graph):
        raise ValueError("schema version 14 requires explicit session/release capability")
    if version == 15 and not graph.state_address_refinements:
        raise ValueError("schema version 15 requires explicit state-address capability")
    return graph


def loads(payload: str) -> ProgramGraph:
    """Decode exact capability profiles v1–v32 without upgrading old wire."""
    from scar.ir.program.validation import require_valid

    graph = _decode_program_graph(payload)
    # The strict decoder already checks all local record invariants.
    require_valid(graph)
    return graph


def _loads_with_index(payload: str) -> tuple[ProgramGraph, "ProgramIndex"]:
    """Strictly load one graph and its index with one semantic validation.

    Both readers share every original schema, record and capability check.
    ProgramIndex runs the full graph validator and consumes its internally
    prepared index/facts. No caller can supply a validation verdict or reuse
    an index from different input bytes; nothing is cached globally here.
    """
    graph = _decode_program_graph(payload)
    from scar.ir.program.queries import ProgramIndex
    return graph, ProgramIndex(graph)


__all__ = ["SCHEMA", "SCHEMA_VERSION", "dumps", "loads"]
