"""Native memory binding."""

from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode
from scar.ir.execution.common import EvidencePoint
from scar.ir.execution.resources import AllocationTarget
from scar.ir.execution.snapshot import SnapshotError
from scar.ir.execution.validity import TargetLifetimeEvent, LifetimeEventKind
from collections import Counter, defaultdict
from scar.ir.v2.ids import StorageAllocationID
from scar.ir.v2.values import ValueGraph
from scar.ir.execution.endpoint_protocol import same_fact_reference


# Independent interpretation of the pinned Torch allocator receipt contract.
#
# The caller must bind the exact raw artifact bytes and final ValueGraph through
# ExternalSnapshots before calling this module.  No URI is opened here.  A match
# concerns profiler allocator receipts, never physical allocation, CUDA completion,
# or permission to transform.  Source stage/effect validation remains separate.
PROFILE = "TORCH_KINETO_ALLOCATOR_V1"
PROVIDER_REVISION = "cf30153c4c131c8164ee7798e5022d810682e2cb"


def _integer(value, minimum=None):
    return type(value) is int and (minimum is None or value >= minimum)


class NativeMemoryResolver:
    """Replay raw receipt slots once; resolve individual canonical lifetime rows.

    ``birth_verified`` / ``free_verified`` mean exact receipt correspondence,
    with ``free_verified`` additionally requiring an unambiguous slot replay.
    They remain conditional on the explicit provider and ordering contract.
    ``ordering_qualified`` requires a declared pinned revision and an explicitly
    disabled provider timestamp adjustment.  None is never treated as False.
    The caller must separately map EvidencePoint.process to the declared pid.
    """

    def __init__(self, raw_kineto, kineto_artifact, scope, pid, *,
                 provider_revision=None, adjust_timestamps=None,
                 protocol_reference=None):
        if type(raw_kineto) is not dict or type(raw_kineto.get("traceEvents")) is not list:
            raise ValueError("native memory requires an exact traceEvents array")
        if type(kineto_artifact) is not ProviderArtifact:
            raise TypeError("native memory requires a ProviderArtifact")
        kineto_artifact.__post_init__()
        if type(scope) is not str or not scope or not _integer(pid, 1):
            raise ValueError("native memory requires a nonempty scope and positive pid")
        if provider_revision is not None and type(provider_revision) is not str:
            raise TypeError("provider_revision must be a string or None")
        if adjust_timestamps is not None and type(adjust_timestamps) is not bool:
            raise TypeError("adjust_timestamps must be an exact bool or None")
        if protocol_reference is not None:
            if type(protocol_reference) is not ProviderFactReference:
                raise TypeError("protocol_reference must be a ProviderFactReference or None")
            protocol_reference.__post_init__()
        self.artifact = kineto_artifact
        self.scope, self.pid = scope, pid
        self.provider_revision = provider_revision
        self.adjust_timestamps = adjust_timestamps
        self.protocol_reference = protocol_reference
        self.epochs = {}
        self.input_reasons = []
        slots, indices = defaultdict(list), Counter()
        for index, event in enumerate(raw_kineto["traceEvents"]):
            if type(event) is not dict or event.get("name") != "[memory]":
                continue
            args = event.get("args")
            # Ev Idx is a provider-wide key, including diagnostics and malformed
            # receipts.  Such rows cannot make another receipt's key unique.
            if type(args) is dict and _integer(args.get("Ev Idx"), 0):
                indices[args["Ev Idx"]] += 1
            if (event.get("ph") != "i" or event.get("cat") != "cpu_instant_event"
                    or type(args) is not dict):
                if event.get("pid") == pid:
                    self.input_reasons.append("malformed_memory_receipt")
                continue
            if type(args.get("Bytes")) is int and args["Bytes"] == 0:
                continue  # A diagnostic is not an allocator birth or free.
            fields = (event.get("pid"), event.get("tid"), args.get("Device Type"),
                      args.get("Device Id"), args.get("Addr"), args.get("Bytes"))
            process, thread, device_type, device_id, address, size = fields
            if not (_integer(process, 1) and _integer(thread, 1)
                    and _integer(device_type, 0) and _integer(device_id)
                    and _integer(address, 1) and _integer(size)):
                if event.get("pid") == pid:
                    self.input_reasons.append("invalid_memory_receipt_fields")
                continue
            ev_idx = args.get("Ev Idx") if _integer(args.get("Ev Idx"), 0) else None
            item = {"index": index, "ev_idx": ev_idx, "process": process,
                    "thread": thread, "device_type": device_type,
                    "device_id": device_id, "address": address, "size": size}
            slots[(process, device_type, device_id, address)].append(item)
        for entries in slots.values():
            self._slot(entries, indices)

    def _reference(self, index):
        return ProviderFactReference(self.artifact, "#/traceEvents/" + str(index))

    def _reference_text(self, index):
        return "sha256:" + self.artifact.sha256 + "#/traceEvents/" + str(index)

    def _slot(self, entries, indices):
        reasons = []
        if len({item["thread"] for item in entries}) != 1:
            reasons.append("address_slot_has_multiple_host_threads")
        if any(not _integer(item["ev_idx"], 0) or indices[item["ev_idx"]] != 1
               for item in entries):
            reasons.append("address_slot_lacks_unique_provider_event_order")
        ordered = sorted(entries, key=lambda item: item["index"] if reasons else item["ev_idx"])
        births, active, seen_birth = [], None, False
        for item in ordered:
            if item["size"] > 0:
                if active is not None:
                    reasons.append("overlapping_allocator_births")
                epoch = {"birth": item, "free": None, "generation": len(births)}
                births.append(epoch)
                active, seen_birth = epoch, True
            elif active is None:
                # A capture may begin after an allocation was born.  Such an
                # orphan is not evidence against a later independently born epoch.
                if seen_birth:
                    reasons.append("free_without_an_open_observed_epoch")
            elif -item["size"] != active["birth"]["size"]:
                reasons.append("allocator_free_size_mismatch")
                active = None
            else:
                active["free"], active = item, None
        reasons = sorted(set(reasons))
        for epoch in births:
            birth, free = epoch["birth"], epoch["free"]
            if reasons:
                free = None
            metadata = {
                "profile": PROFILE, "birth_index": birth["index"],
                "birth_ev_idx": birth["ev_idx"],
                "free_index": free["index"] if free else None,
                "free_ev_idx": free["ev_idx"] if free else None,
                "address": birth["address"], "process": birth["process"],
                "thread": birth["thread"], "device_type": birth["device_type"],
                "device_id": birth["device_id"],
                "generation": birth["index"] if reasons else epoch["generation"],
                "birth_reference": self._reference_text(birth["index"]),
                "free_reference": self._reference_text(free["index"]) if free else None,
                "ordering_status": "UNRESOLVED" if reasons else "PROVIDER_THREAD_ORDER",
                "lifetime_status": ("UNRESOLVED" if reasons else
                                    "MATCHED_PROFILER_FREE" if free else "OPEN_AT_CAPTURE_END"),
            }
            allocation_id = StorageAllocationID(self.scope + ":allocation:kineto:" + str(birth["index"]))
            self.epochs[allocation_id] = {"birth": birth, "free": free,
                                          "metadata": metadata, "reasons": reasons}

    def resolve(self, row, values):
        """Return a bounded non-authorizing result; never silently accept a gap.

        ``values`` must be the caller's exact catalog-validated final ValueGraph
        for row.target.artifact.  This method does not rehash the entire graph
        per receipt and cannot replace that snapshot check.
        """
        result = {"status": "UNRESOLVED", "matched": False,
                  "profile_supported": self.provider_revision == PROVIDER_REVISION,
                  "birth_verified": False, "free_verified": False,
                  "ordering_qualified": False, "reasons": [], "conditions": [],
                  "physical_allocation_established": False, "resource_ready": False,
                  "transformation_authorized": False}

        def fail(reason, status="MISMATCH"):
            result["status"] = status
            result["reasons"].append(reason)
            return result

        if type(row) is not TargetLifetimeEvent or type(row.target) is not AllocationTarget:
            return fail("native_memory_requires_an_allocation_lifetime_row")
        if type(values) is not ValueGraph or row.target.artifact.schema != ValueGraph.SCHEMA:
            return fail("native_memory_requires_the_bound_final_value_graph")
        if self.protocol_reference is None:
            return fail("validated_profile_reference_not_supplied", "UNRESOLVED")
        if not same_fact_reference(row.protocol, self.protocol_reference):
            return fail("lifetime_protocol_reference_mismatch")
        epoch = self.epochs.get(row.target.allocation)
        allocation = values.allocations.get(row.target.allocation)
        if epoch is None or allocation is None:
            return fail("allocation_has_no_matching_raw_birth")
        birth, free = epoch["birth"], epoch["free"]
        if birth["process"] != self.pid:
            return fail("allocation_process_differs_from_profile_pid")
        metadata = allocation.metadata.get("native_memory") if type(allocation.metadata) is dict else None
        expected = epoch["metadata"]
        if (type(metadata) is not dict or metadata.keys() != expected.keys()
                or any(type(metadata[key]) is not type(value) or metadata[key] != value
                       for key, value in expected.items())):
            return fail("allocation_metadata_disagrees_with_independent_raw_replay")
        if (type(allocation.nbytes) is not int or allocation.nbytes != birth["size"]
                or allocation.lifetime_scope != self.scope):
            return fail("allocation_extent_or_scope_disagrees_with_raw_birth")
        device = ("cpu" if (birth["device_type"], birth["device_id"]) == (0, -1)
                  else "cuda:" + str(birth["device_id"])
                  if birth["device_type"] == 1 and birth["device_id"] >= 0
                  else "provider-device:" + str(birth["device_type"]) + ":" + str(birth["device_id"]))
        if (type(allocation.device) is not str or allocation.device != device
                or allocation.allocator_token != "kineto-addr:" + str(birth["address"])):
            return fail("allocation_device_or_address_token_disagrees_with_raw_birth")
        if type(row.kind) is not LifetimeEventKind:
            return fail("unsupported_lifetime_kind")
        event = birth if row.kind is LifetimeEventKind.LIVE else free
        if event is None:
            return fail("no_unambiguous_observed_free_for_this_epoch", "UNRESOLVED")
        fact = self._reference(event["index"])
        if (type(row.at) is not EvidencePoint or row.at.process is None
                or not same_fact_reference(row.at.source, fact)
                or not same_fact_reference(row.source, fact)):
            return fail("lifetime_point_or_source_is_not_the_exact_raw_receipt")
        result.update(birth_index=birth["index"], free_index=free["index"] if free else None,
                      event_index=event["index"], generation=expected["generation"])
        if not result["profile_supported"]:
            return fail("pinned_provider_revision_not_declared", "UNSUPPORTED")
        result["birth_verified"] = True
        reasons = sorted(set(epoch["reasons"] + self.input_reasons))
        if reasons:
            result["reasons"] = reasons
            return result
        if self.adjust_timestamps is True:
            return fail("provider_timestamp_adjustment_does_not_qualify_receipt_order", "UNRESOLVED")
        result["free_verified"] = row.kind is LifetimeEventKind.RETIRED
        result["ordering_qualified"] = self.adjust_timestamps is False
        if self.adjust_timestamps is None:
            result["conditions"].append("provider_adjust_timestamps_must_be_disabled")
        result["conditions"].extend(("caller_bound_exact_raw_and_final_value_graph",
                                     "caller_bound_evidence_process_to_profile_pid",
                                     "profiler_allocator_receipts_only"))
        result["matched"], result["status"] = True, "CONDITIONAL"
        return result


# Bind native allocator claims to exact caller-supplied profile/raw snapshots.
#
# This adapter opens no URI and creates no program occurrence.  A matched claim
# remains scoped to profiler allocator receipts, not physical memory or readiness.
PROFILE_SCHEMA = "scar.native-memory-profile"
_CONTRACT_FIELDS = {"kineto", "scope", "pid", "provider_revision",
                    "adjust_timestamps", "capture_manifest"}


class _Unavailable(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def _float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    return value


def _document(ctx, artifact):
    payload = ctx.snapshots.payload(artifact)
    if payload is None:
        raise _Unavailable("exact " + artifact.schema + " payload was not supplied")
    document = json.loads(payload, object_pairs_hook=_pairs, parse_constant=_constant,
                          parse_float=_float)
    if type(document) is not dict:
        raise ValueError(artifact.schema + " must be a JSON object")
    return document


def _shape(document, fields, name):
    if type(document) is not dict or set(document) != fields:
        raise ValueError(name + " has missing or unknown fields")


def _capture(ctx, artifact, contract, kineto):
    if artifact.schema != "scar.execution-evidence-collection":
        raise ValueError("capture artifact has unsupported schema")
    document = _document(ctx, artifact)
    if (document.get("schema") != artifact.schema
            or type(document.get("schema_version")) is not int or document["schema_version"] != 1
            or type(document.get("pid")) is not int or document["pid"] != contract["pid"]):
        raise ValueError("capture schema/version/process does not match native contract")
    configuration, provider = document.get("configuration"), document.get("provider")
    if (type(configuration) is not dict or configuration.get("profile_memory") is not True
            or type(provider) is not dict
            or provider.get("torch_git_version") != contract["provider_revision"]):
        raise ValueError("capture memory setting/provider revision does not match native contract")
    experimental = configuration.get("experimental_config")
    observed = experimental.get("adjust_timestamps") if type(experimental) is dict else None
    observed = observed if type(observed) is bool else None
    declared = contract["adjust_timestamps"]
    if type(declared) is not type(observed) or declared != observed:
        raise ValueError("declared timestamp adjustment does not match capture configuration")
    artifacts = document.get("artifacts")
    captured = artifacts.get("kineto") if type(artifacts) is dict else None
    if (type(captured) is not dict or type(captured.get("path")) is not str
            or not captured["path"] or type(captured.get("size_bytes")) is not int
            or captured["size_bytes"] != kineto.size_bytes):
        raise ValueError("capture Kineto artifact has no matching decompressed size/path")
    current = PurePosixPath(kineto.uri).name
    current = current[:-3] if current.endswith(".gz") else current
    if PurePosixPath(captured["path"]).name != current:
        raise ValueError("capture Kineto basename does not match retained raw artifact")


def _binding(ctx, reference):
    document = _document(ctx, reference.artifact)
    _shape(document, {"schema", "schema_version", "contract"}, "native memory profile")
    if (document["schema"] != PROFILE_SCHEMA or type(document["schema_version"]) is not int
            or document["schema_version"] != 1):
        raise ValueError("unsupported native memory profile schema/version")
    contract = document["contract"]
    _shape(contract, _CONTRACT_FIELDS, "native memory contract")
    if (type(contract["scope"]) is not str or not contract["scope"]
            or type(contract["pid"]) is not int or contract["pid"] <= 0
            or type(contract["provider_revision"]) is not str or not contract["provider_revision"]
            or (contract["adjust_timestamps"] is not None
                and type(contract["adjust_timestamps"]) is not bool)):
        raise ValueError("native memory contract has invalid scope/process/provider/ordering fields")
    kineto = decode(ProviderArtifact, contract["kineto"])
    if kineto.schema != "pytorch.kineto.chrome-trace.v1":
        raise ValueError("native memory contract requires the exact Kineto artifact schema")
    capture = contract["capture_manifest"]
    if capture is not None:
        _capture(ctx, decode(ProviderArtifact, capture), contract, kineto)
    elif contract["adjust_timestamps"] is not None:
        raise ValueError("timestamp adjustment cannot be declared without a capture manifest")
    resolver = NativeMemoryResolver(_document(ctx, kineto), kineto, contract["scope"], contract["pid"],
        provider_revision=contract["provider_revision"], adjust_timestamps=contract["adjust_timestamps"],
        protocol_reference=reference)
    return resolver, contract


def _unmatched(reason, status="UNRESOLVED"):
    return {"status": status, "matched": False, "profile_supported": False,
            "birth_verified": False, "free_verified": False, "ordering_qualified": False,
            "reasons": [reason], "conditions": [], "physical_allocation_established": False,
            "resource_ready": False, "transformation_authorized": False}


def resolve_native_memory_claim(ctx, row, path):
    """Resolve only the fixed native profile; generic protocols return None.

    ctx supplies snapshots, external_values(), runs/processes, and issue()/gap().
    Missing exact snapshots remain gaps; malformed declarations are issues.
    Cached bindings live only inside this context and never open artifact URIs.
    """
    reference = getattr(row, "protocol", None)
    if type(reference) is not ProviderFactReference or reference.artifact.schema != PROFILE_SCHEMA:
        return None
    try:
        reference.__post_init__()
        if reference.locator != "#/contract":
            raise ValueError("native memory profile must select #/contract")
        if (type(row) is not TargetLifetimeEvent or type(row.target) is not AllocationTarget
                or type(row.at) is not EvidencePoint or row.at.process is None):
            raise ValueError("native memory claim needs an allocation target and process EvidencePoint")
        key = (reference.artifact.sha256, reference.artifact.schema,
               reference.artifact.size_bytes, reference.locator)
        cache = getattr(ctx, "_native_memory_binding_cache", None)
        if cache is None:
            cache = ctx._native_memory_binding_cache = {}
        if key not in cache:
            try:
                cache[key] = (_binding(ctx, reference), None, None)
            except _Unavailable as error:
                cache[key] = (None, "gap", str(error))
            except (SnapshotError, ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as error:
                cache[key] = (None, "issue", str(error))
        binding, diagnostic, message = cache[key]
        if diagnostic:
            getattr(ctx, diagnostic)("native_memory_binding", path, message)
            return _unmatched(message, "MISMATCH" if diagnostic == "issue" else "UNRESOLVED")
        resolver, contract = binding
        process = ctx.processes.get(row.at.process)
        if (process is None or type(process.pid) is not int or process.pid != contract["pid"]
                or process.run != row.at.run or row.at.run not in ctx.runs):
            raise ValueError("native memory EvidencePoint process/run does not match contract pid")
        values = ctx.external_values(row.target.artifact, path)
        if values is None:
            return _unmatched("exact final allocation ValueGraph was not supplied or is invalid")
        result = resolver.resolve(row, values)
        if result["status"] == "MISMATCH":
            ctx.issue("native_memory_claim", path, "; ".join(result["reasons"]))
        elif not result["matched"]:
            ctx.gap("native_memory_claim", path, "; ".join(result["reasons"]))
        if result["matched"]:
            result["conditions"] = [item for item in result["conditions"]
                if item not in ("caller_bound_exact_raw_and_final_value_graph",
                                "caller_bound_evidence_process_to_profile_pid")]
            if contract["capture_manifest"] is not None:
                result["conditions"].append("capture_association_uses_basename_and_decompressed_size")
        return result
    except (SnapshotError, ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as error:
        ctx.issue("native_memory_binding", path, str(error))
        return _unmatched(str(error), "MISMATCH")


__all__ = [
    'resolve_native_memory_claim',
    'NativeMemoryResolver',
    'PROFILE',
    'PROVIDER_REVISION',
    'PROFILE_SCHEMA',
]
