"""Dispatch evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path
from collections import Counter
from hashlib import sha256
from scar.ir.v2 import EvidenceClaim, EvidenceKind, ObjectID, ObjectInstance
from scar.ir.v2.evidence import (
    EvidenceEdge,
    EvidenceEndpoint,
    EvidenceNodeKind as N,
    EvidenceRelation,
)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode, encode
from scar.trace.reader import open_binary, resolve as resolve_retained_path
from scar.trace.native_import import _canonical_hash, _constant, _float, _pairs


# Bounded production dispatch observations bound to real ETO marker slots.
#
# No Tensor contents, versions, hashes or copies are observed. A callback scope
# does not enclose the original dispatcher's already-started RecordFunction.
PRODUCTION_PROFILE = "scar.dispatch-marker-collector-v1"
PROVIDER_REVISION = "cf30153c4c131c8164ee7798e5022d810682e2cb"


class DispatchEvidence:
    def __init__(self, out, recorder, *, nonce, provider_revision,
                 max_events=400000, max_bytes=256000000, max_slots=1024,
                 max_depth=16, max_items=4096, max_frames=32):
        limits = dict(max_events=max_events, max_bytes=max_bytes, max_slots=max_slots,
                      max_depth=max_depth, max_items=max_items, max_frames=max_frames)
        if any(type(value) is not int or value < 1 for value in limits.values()):
            raise ValueError("dispatch budgets must be positive exact integers")
        if provider_revision != PROVIDER_REVISION or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            raise ValueError("unsupported dispatch provider or nonce")
        self.pid, self.owner_ident = os.getpid(), threading.get_ident()
        self.owner_native_tid = threading.get_native_id()
        if recorder.pid != self.pid or recorder.closed:
            raise ValueError("dispatch recorder must be live in the owner process")
        self.out, self.recorder, self.nonce = Path(out).resolve(), recorder, nonce
        self.provider_revision, self.limits = provider_revision, limits
        self.event_path = self.out / f"dispatch-observations.{self.pid}.{nonce}.jsonl"
        self.meta_path = self.out / f"dispatch-meta.{self.pid}.{nonce}.json"
        if self.meta_path.exists() or self.meta_path.is_symlink():
            raise ValueError("dispatch output must be fresh")
        self.fd = os.open(self.event_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._digest = hashlib.sha256()
        self._buffer = bytearray()
        self._buffer_limit = 65536
        self._lock = threading.RLock()
        self._mode = self._top_mode = self._torch = None
        self._enter = self._exit = None
        self._sequence = self._events = self._bytes = self._dropped = 0
        self._errors, self._gaps = [], []
        self._installed = self._closed = self._stopped = self._io_failed = False
        self._mode_popped = False
        self.recorder.observer_files.add(__file__)

    def _error(self, stage, error):
        if len(self._errors) < 16:
            self._errors.append({"stage": stage, "type": type(error).__name__})

    def _gap(self, code):
        if code not in self._gaps and len(self._gaps) < 32:
            self._gaps.append(code)

    def _emit(self, kind, fields):
        with self._lock:
            if self._closed or self._io_failed or os.getpid() != self.pid:
                return False
            if self._events >= self.limits["max_events"]:
                self._dropped += 1; self._stopped = True
                self._gap("event_budget_exceeded"); return False
            row = {"index": self._events + 1, "kind": kind, "process_id": self.pid, **fields}
            data = (json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
            if len(data) > 2_000_000 or self._bytes + len(data) > self.limits["max_bytes"]:
                self._dropped += 1; self._stopped = True
                self._gap("byte_or_line_budget_exceeded"); return False
            try:
                self._bytes += len(data)
                self._digest.update(data)
                if len(self._buffer) + len(data) > self._buffer_limit:
                    self._flush()
                if len(data) >= self._buffer_limit:
                    self._write_all(data)
                else:
                    self._buffer.extend(data)
            except OSError as error:
                self._io_failed = self._stopped = True
                self._error("write", error); return False
            self._events += 1
            return True

    def _write_all(self, data):
        if os.getpid() != self.pid:
            raise OSError("forked process cannot flush parent dispatch records")
        offset = 0
        while offset < len(data):
            size = os.write(self.fd, data[offset:])
            if size <= 0:
                raise OSError("dispatch sidecar made no write progress")
            offset += size

    def _flush(self):
        if self._buffer:
            self._write_all(self._buffer)
            self._buffer.clear()

    def _leaves(self, value, prefix):
        leaves, gaps, seen_containers = [], [], set()
        items = 0
        def visit(item, path, depth):
            nonlocal items
            items += 1
            if items > self.limits["max_items"] or depth > self.limits["max_depth"]:
                raise ValueError("dispatch container budget exceeded")
            kind = type(item)
            if kind is self._torch.Tensor or kind is self._torch.nn.Parameter:
                if len(leaves) >= self.limits["max_slots"]:
                    raise ValueError("dispatch Tensor slot budget exceeded")
                leaves.append((path, item)); return
            if kind in (tuple, list, dict):
                if id(item) in seen_containers:
                    gaps.append({"path": list(path), "reason": "cyclic_container"}); return
                seen_containers.add(id(item))
                try:
                    if kind is dict:
                        for key, child in item.items():
                            if type(key) is not str:
                                gaps.append({"path": list(path), "reason": "non_string_mapping_key"}); continue
                            visit(child, (*path, key), depth + 1)
                    else:
                        for index, child in enumerate(item):
                            visit(child, (*path, index), depth + 1)
                finally:
                    seen_containers.remove(id(item))
            elif kind not in (type(None), bool, int, float, str, self._torch.dtype,
                               self._torch.device, self._torch.layout, self._torch.memory_format):
                gaps.append({"path": list(path), "reason": "unqualified_leaf_or_tensor_subclass"})
        visit(value, prefix, 0)
        return leaves, gaps

    def start(self, torch):
        if os.getpid() != self.pid or threading.get_ident() != self.owner_ident or self._installed:
            raise ValueError("dispatch mode start requires its installing process/thread")
        if getattr(torch.version, "git_version", None) != self.provider_revision:
            raise ValueError("dispatch provider revision mismatch")
        from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode
        self._torch, self._top_mode = torch, _get_current_dispatch_mode
        self._enter = torch.autograd._record_function_with_args_enter
        self._exit = torch.autograd._record_function_with_args_exit
        if _get_current_dispatch_mode() is not None:
            self._gap("preexisting_foreign_dispatch_mode_not_instrumented")
            return False
        owner = self
        class Observe(TorchDispatchMode):
            supports_higher_order_operators = True
            @classmethod
            def _should_skip_dynamo(cls):
                return False
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return owner._dispatch(func, args, kwargs)
        self._mode = Observe()
        self._forward_adapter_code = Observe.__torch_dispatch__.__code__
        self._mode.__enter__()
        self._installed = True
        witness = getattr(self.recorder, 'collector_witness', None)
        if witness is not None:
            callbacks = ('_dispatch','_leaves','_call_target') if getattr(self.recorder,'process_ports',False) else ('_dispatch','_leaves')
            witness.register('dispatch', self, callbacks, hook=Observe.__torch_dispatch__)
        return True

    def _call_target(self, func, args, kwargs):
        token = None
        if getattr(self.recorder,'process_ports',False):
            try:
                frame = sys._getframe()
                adapter = frame.f_back
                while adapter is not None and adapter.f_code is not self._forward_adapter_code:
                    adapter = adapter.f_back
                if adapter is not None:
                    token = self.recorder._begin_target_forward(frame, 'DISPATCH', adapter_frame=adapter)
            except Exception as error:
                self._error('target_forward_begin', error)
            finally:
                frame = adapter = None  # No frame/Tensor lifetime extension through a frame-local cycle.
        try:
            # Exactly one original call, including on observation failure.
            return func(*args, **({} if kwargs is None else kwargs))
        finally:
            if token is not None:
                try:
                    self.recorder._end_target_forward(token)
                except Exception as error:
                    self._error('target_forward_end', error)

    def _dispatch(self, func, args, kwargs):
        if (self._stopped or self._closed or os.getpid() != self.pid
                or threading.get_ident() != self.owner_ident):
            if getattr(self.recorder,'process_ports',False):
                return self._call_target(func, args, kwargs)
            return func(*args, **({} if kwargs is None else kwargs))
        witness = getattr(self.recorder, 'collector_witness', None)
        if witness is not None and not witness.guard('dispatch', sys._getframe().f_code):
            self._stopped = True
            if getattr(self.recorder,'process_ports',False):
                return self._call_target(func, args, kwargs)
            return func(*args, **({} if kwargs is None else kwargs))
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        span = f"scar-dispatch:{self.nonce}:{self.pid}:{sequence}:span"
        output_name = f"scar-dispatch:{self.nonce}:{self.pid}:{sequence}:output"
        handle = None
        input_leaves, output_leaves = [], []
        captured, input_gaps = None, []
        try:
            input_leaves, input_gaps = self._leaves((args, {} if kwargs is None else kwargs), ())
            captured = self.recorder.observe_dispatch(sys._getframe().f_back, input_leaves,
                max_frames=self.limits["max_frames"], max_slots=self.limits["max_slots"])
            if captured is None:
                self._gap("call_recorder_unavailable")
            else:
                operator = str(func) if type(func) is self._torch._ops.OpOverload else None
                if self._emit("DISPATCH_BEGIN", {"sequence": sequence, "marker": span,
                    "observation": captured, "slot_gaps": input_gaps, "operator": operator}):
                    handle = self._enter(span, *(tensor for _, tensor in input_leaves))
        except Exception as error:
            self._error("input_observation", error)
        try:
            if getattr(self.recorder,'process_ports',False):
                result = self._call_target(func, args, kwargs)
            else:
                result = func(*args, **({} if kwargs is None else kwargs))
        except BaseException:
            self._emit("DISPATCH_END", {"sequence": sequence, "outcome": "RAISED",
                       "span_enter_returned": handle is not None, "output_marker": None})
            raise
        else:
            output_entered = False
            if handle is not None:
                output_handle = None
                try:
                    output_leaves, output_gaps = self._leaves(result, ())
                    observation = self.recorder.observe_dispatch(sys._getframe().f_back, output_leaves,
                        max_frames=self.limits["max_frames"], max_slots=self.limits["max_slots"])
                    if observation is not None and self._emit("DISPATCH_OUTPUT", {
                        "sequence": sequence, "marker": output_name, "observation": observation,
                        "slot_gaps": output_gaps}):
                        output_handle = self._enter(output_name, *(tensor for _, tensor in output_leaves))
                        output_entered = True
                except Exception as error:
                    self._error("output_observation", error)
                finally:
                    if output_handle is not None:
                        try:
                            self._exit(output_handle)
                        except Exception as error:
                            self._error("output_marker_exit", error); self._stopped = True
            self._emit("DISPATCH_END", {"sequence": sequence, "outcome": "RETURNED",
                "span_enter_returned": handle is not None,
                "output_marker": output_name if output_entered else None})
            return result
        finally:
            if handle is not None:
                try:
                    self._exit(handle)
                except Exception as error:
                    self._error("span_exit", error); self._stopped = True
            input_leaves.clear(); output_leaves.clear()

    def stop(self):
        """Stop observations before the owning profiler stops; never pop foreign modes."""
        self._stopped = True
        if not self._installed or os.getpid() != self.pid:
            return
        if threading.get_ident() != self.owner_ident:
            self._gap("close_on_different_thread"); return
        if self._top_mode() is not self._mode:
            self._gap("foreign_top_mode_preserved_owned_mode_inert"); return
        try:
            self._mode.__exit__(None, None, None)
            self._installed = False; self._mode_popped = True
        except Exception as error:
            self._error("mode_exit", error)

    def close(self, *, native_manifest=None):
        if self._closed or os.getpid() != self.pid:
            return
        self.stop()
        with self._lock:
            self._closed = True
            try:
                if not self._io_failed:
                    self._flush()
            except OSError as error:
                self._io_failed = self._stopped = True
                self._error("flush", error)
            finally:
                self._buffer.clear()
            os.close(self.fd)
            native = None
            if native_manifest is not None:
                path = Path(native_manifest)
                try:
                    with path.open("rb") as stream:
                        payload = stream.read(2_000_001)
                    if len(payload) > 2_000_000:
                        raise OSError("native collector manifest exceeds metadata bound")
                    native = {"path": str(path), "raw_sha256": hashlib.sha256(payload).hexdigest(),
                              "raw_bytes": len(payload)}
                except OSError as error:
                    self._error("native_manifest", error)
            metadata = {"schema": "scar.dispatch-observation-trace", "schema_version": 1,
                "profile": PRODUCTION_PROFILE, "provider_revision": self.provider_revision,
                "process_id": self.pid, "owner_python_ident": self.owner_ident,
                "owner_native_tid": self.owner_native_tid, "nonce": self.nonce,
                "limits": self.limits, "events": self._events, "dispatch_attempts": self._sequence,
                "bytes": self._bytes, "dropped_events": self._dropped, "errors": self._errors,
                "gaps": self._gaps, "io_failed": self._io_failed, "mode_popped": self._mode_popped,
                "file": {"path": str(self.event_path), "raw_sha256": self._digest.hexdigest(),
                         "raw_bytes": self._bytes, "lines": self._events},
                "native_manifest": native,
                "scope": "owner process and installing thread dispatch callbacks only",
                "reference_contents_observed": False, "tensor_versions_observed": False,
                "direct_python_operator_correspondence_established": False,
                "logical_versions_established": False, "state_effects_established": False,
                "resource_readiness_established": False, "whole_model_established": False,
                "transformation_authorized": False}
            payload = (json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            fd = os.open(self.meta_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(fd, payload[offset:])
                    if written <= 0:
                        raise OSError("dispatch metadata made no write progress")
                    offset += written
            finally:
                os.close(fd)


# Decode production dispatch slots into the existing native evidence bundle.
#
# Markers identify exact provider observations. Their actual ctrl_deps path is
# retained; a dispatch callback is never promoted to a direct Python/ATen call.
_MARKER = re.compile(r"scar-dispatch:([0-9a-f]{32}):([1-9][0-9]*):([1-9][0-9]*):(span|output)")
_WEAK = re.compile(r"weak:([1-9][0-9]*):([1-9][0-9]*)")


def _retained_artifact(coordinates, actual_path, raw_sha256, raw_bytes, schema):
    """Resolve an explicitly read local file to its original declared artifact.

    A content hash alone cannot select between distinct retained files. New
    coordinate reports publish local_path separately from the logical URI;
    historical reports require an exact URI/path match instead. No URI opens
    a file, no basename/selector inference participates in this lookup.
    """
    if coordinates is None:
        return ProviderArtifact(str(Path(actual_path).absolute()), raw_sha256, raw_bytes, schema)
    local_path = str(Path(actual_path).resolve())
    matches = []
    for member in coordinates.report["membership"]:
        declared = decode(ProviderArtifact, member["artifact"])
        selected_path = member.get("local_path", declared.uri)
        if (selected_path == local_path and declared.sha256 == raw_sha256
                and declared.size_bytes == raw_bytes):
            matches.append(declared)
    if len(matches) != 1:
        raise ValueError("dispatch input has no unique exact retained path/raw-byte membership")
    return matches[0]


def import_dispatch_bridge(result, observations_path, metadata_path, call_events_path,
                           call_codes_path, call_meta_path, *, native_manifest_path,
                           coordinates=None, max_bytes=512000000, max_events=400000):
    """Append exact Object→ProviderReference associations; return result.

    Every opened path is an explicit caller argument. Metadata paths select no
    external files. Missing context/weak identity/slot qualification remains a
    gap; contradictory marker identities or byte receipts are rejected.
    """
    if any(type(value) is not int or value < 1 for value in (max_bytes, max_events)):
        raise ValueError("dispatch import budgets must be positive exact integers")
    if result.report["provider_revision"] != PROVIDER_REVISION:
        raise ValueError("unsupported dispatch provider revision")
    if "dispatch_bridge" in result.report:
        raise ValueError("this native bundle already has a dispatch bridge")
    documents, artifacts, lines, input_paths, remaining = {}, {}, {}, {}, max_bytes
    for key, path, schema in (
        ("observations", observations_path, "scar.dispatch-observation.events.v1"),
        ("metadata", metadata_path, "scar.dispatch-observation-trace"),
        ("calls", call_events_path, "scar.call-edge-events.v1"),
        ("codes", call_codes_path, "scar.call-edge-codes.v1"),
        ("call_meta", call_meta_path, "scar.call-edge-trace"),
        ("native_meta", native_manifest_path, "scar.execution-evidence-collection")):
        actual_path = resolve_retained_path(path).resolve()
        input_paths[key] = str(actual_path)
        with open_binary(actual_path) as stream:
            payload = stream.read(remaining + 1)
        if len(payload) > remaining:
            raise ValueError("dispatch import exceeds combined raw byte budget")
        remaining -= len(payload)
        artifact = _retained_artifact(coordinates, actual_path if coordinates is not None else path,
                                      sha256(payload).hexdigest(), len(payload), schema)
        artifacts[key] = artifact
        def decode(raw):
            return json.loads(raw, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)
        if key in ("observations", "calls", "codes"):
            numbered = [(number, raw) for number, raw in enumerate(payload.splitlines(), 1) if raw.strip()]
            if key == "observations" and len(numbered) > max_events:
                raise ValueError("dispatch observation event budget exceeded")
            documents[key] = [decode(raw) for _, raw in numbered]
            lines[key] = [number for number, _ in numbered]
        else:
            documents[key] = decode(payload)
    meta, native, call_meta = documents["metadata"], documents["native_meta"], documents["call_meta"]
    pid, nonce = meta.get("process_id"), meta.get("nonce")
    if (meta.get("schema") != "scar.dispatch-observation-trace" or meta.get("schema_version") != 1
            or meta.get("profile") != PRODUCTION_PROFILE or meta.get("provider_revision") != PROVIDER_REVISION
            or type(pid) is not int or pid != result.report["pid"]
            or type(nonce) is not str or re.fullmatch(r"[0-9a-f]{32}", nonce) is None):
        raise ValueError("dispatch producer profile/process/nonce mismatch")
    declaration, native_declaration = meta.get("file", {}), meta.get("native_manifest", {})
    for declaration_item, key in ((declaration, "observations"), (native_declaration, "native_meta")):
        if (declaration_item.get("raw_sha256"), declaration_item.get("raw_bytes")) != (
                artifacts[key].sha256, artifacts[key].size_bytes):
            raise ValueError("dispatch producer exact raw receipt mismatch: " + key)
    if (declaration.get("lines") != len(documents["observations"])
            or meta.get("events") != len(documents["observations"])
            or meta.get("bytes") != artifacts["observations"].size_bytes):
        raise ValueError("dispatch producer event/byte count mismatch")
    dispatch_config = native.get("configuration", {}).get("dispatch_observations", {})
    if (native.get("schema") != "scar.execution-evidence-collection" or native.get("schema_version") != 1
            or native.get("pid") != pid or native.get("status") != "CLOSED" or native.get("failures") != []
            or native.get("provider", {}).get("torch_git_version") != PROVIDER_REVISION
            or dispatch_config.get("profile") != PRODUCTION_PROFILE or dispatch_config.get("started") is not True
            or dispatch_config.get("events") != declaration.get("path")
            or Path(dispatch_config.get("metadata", "")).name != Path(metadata_path).name.removesuffix(".gz")):
        raise ValueError("native receipt does not bind this production dispatch collector")
    for family, native_key in (("execution_trace", "eto"), ("kineto", "kineto")):
        item = native.get("artifacts", {}).get(family, {})
        actual = result.report["artifacts"].get(native_key)
        expected_name = ("execution-trace" if family == "execution_trace" else "kineto") + f".{pid}.{nonce}.json"
        if (actual is None or Path(item.get("path", "")).name != expected_name
                or Path(actual["path"]).name.removesuffix(".gz") != expected_name
                or item.get("size_bytes") != actual["size_bytes"]):
            raise ValueError("native artifact differs from the dispatch nonce capture")
    if (call_meta.get("schema") != "scar.call-edge-trace" or call_meta.get("schema_version") != 1
            or call_meta.get("process_id") != pid or call_meta.get("events") != len(documents["calls"])
            or call_meta.get("codes") != len(documents["codes"])):
        raise ValueError("call collector schema/process/count mismatch")
    codes, entries, exits, indices = {}, {}, {}, set()
    for row in documents["codes"]:
        if type(row) is not dict or row.get("key") in codes:
            raise ValueError("invalid/duplicate call code")
        codes[row["key"]] = row
    for number, row in zip(lines["calls"], documents["calls"], strict=True):
        if (type(row) is not dict or row.get("process_id") != pid
                or type(row.get("index")) is not int or row["index"] in indices):
            raise ValueError("invalid/duplicate call event")
        indices.add(row["index"])
        if row.get("kind") == "call_enter":
            if row.get("invocation_id") in entries or row.get("code") not in codes:
                raise ValueError("duplicate/unbound retained entry")
            entries[row["invocation_id"]] = (number, row)
        elif row.get("kind") == "call_exit":
            exits[row["invocation_id"]] = row["index"]
    membership_verified = coordinates is not None
    if coordinates is not None:
        for artifact in artifacts.values():
            if coordinates.process_for(artifact, pid) != coordinates.target_process:
                raise ValueError("dispatch/call artifact lacks verified capture membership")
        native_artifact = result.report["artifacts"]["eto"]
        provider = _retained_artifact(coordinates, resolve_retained_path(native_artifact["path"]).resolve(),
            native_artifact["sha256"], native_artifact["size_bytes"], "pytorch.execution-trace.v1")
        if coordinates.process_for(provider, pid) != coordinates.target_process:
            raise ValueError("native graph belongs to a different capture")
    evidence, values = result.bundle.evidence, result.bundle.values
    native_nodes = {instance.metadata["node_id"]: instance for instance in evidence.instances.values()
                    if instance.metadata.get("provider") == "ETO"}
    marker_nodes = {}
    for instance in native_nodes.values():
        name = result.bundle.semantic.definitions[instance.definition].label
        if name.startswith(f"scar-dispatch:{nonce}:"):
            if name in marker_nodes:
                raise ValueError("duplicate production marker node")
            marker_nodes[name] = instance
    scope = "dispatch-reference:" + _canonical_hash({"native_scope": result.report["scope"],
        "profile": PRODUCTION_PROFILE, "artifacts": {key: artifact.sha256 for key, artifact in artifacts.items()}})
    grouped, gaps, paths, observations, consumed = {}, Counter(), [], [], set()
    pending_objects, pending_edges = {}, []
    def pointer(key, suffix):
        return "sha256:" + artifacts[key].sha256 + suffix
    for number, row in zip(lines["observations"], documents["observations"], strict=True):
        if type(row) is not dict or row.get("process_id") != pid or type(row.get("sequence")) is not int:
            raise ValueError("invalid dispatch event process/sequence")
        sequence, kind = row["sequence"], row.get("kind")
        group = grouped.setdefault(sequence, {})
        if kind not in ("DISPATCH_BEGIN", "DISPATCH_OUTPUT", "DISPATCH_END") or kind in group:
            raise ValueError("invalid/duplicate dispatch event kind")
        group[kind] = row
        if kind == "DISPATCH_END":
            continue
        match = _MARKER.fullmatch(row.get("marker", ""))
        phase = "span" if kind == "DISPATCH_BEGIN" else "output"
        if match is None or match.groups() != (nonce, str(pid), str(sequence), phase):
            raise ValueError("dispatch marker identity mismatch")
        marker = marker_nodes.get(row["marker"])
        if marker is None:
            gaps["marker_enter_not_retained"] += 1; continue
        consumed.add(row["marker"])
        observation = row.get("observation", {})
        if observation.get("process_id") != pid or type(observation.get("native_tid")) is not int:
            raise ValueError("dispatch thread observation missing")
        native_tid_verified = marker.metadata.get("kineto_reference") is not None
        if native_tid_verified:
            if (marker.metadata.get("raw_pid") != pid
                    or marker.metadata.get("raw_tid") != observation["native_tid"]):
                raise ValueError("actual RF-correlated Kineto host thread differs from dispatch observation")
        else:
            gaps["marker_host_native_tid_unverified"] += 1
        context = observation.get("context")
        if context is None:
            gaps["retained_ancestor_missing"] += 1
        else:
            found = entries.get(context.get("invocation"))
            if found is None:
                raise ValueError("dispatch retained ancestor is absent")
            entry_number, entry = found
            if (context.get("kind") != "retained ancestor context, not a proven direct caller"
                    or context.get("code") != entry["code"]
                    or context.get("source_file") != codes[entry["code"]]["filename"]
                    or observation.get("python_ident") != entry["thread_id"]
                    or type(context.get("bytecode_offset")) is not int or context["bytecode_offset"] < 0
                    or type(observation.get("call_event_index")) is not int
                    or not entry["index"] <= observation["call_event_index"] < exits.get(entry["invocation_id"], float("inf"))):
                raise ValueError("dispatch retained ancestor/context is contradictory")
        slots = [slot for slot in marker.metadata.get("tensor_slots", ()) if slot["role"] == "inputs"]
        leaves = observation.get("tensor_references")
        if (type(leaves) is not list or len(leaves) != len(slots)
                or [slot["ordered_path"] for slot in slots] != [[i] for i in range(len(slots))]):
            raise ValueError("dispatch marker does not contain the exact flattened Tensor slots")
        ancestry, current, seen = [], marker, set()
        while current is not None and current.id not in seen:
            seen.add(current.id)
            ancestry.append({"instance": current.id.wire, "node_id": current.metadata["node_id"],
                "rf_id": current.metadata["rf_id"], "ctrl_deps": current.metadata["ctrl_deps"],
                "raw_reference": current.metadata["raw_reference"]})
            parent = current.metadata["ctrl_deps"]
            current = native_nodes.get(parent) if parent != current.metadata["node_id"] else None
        observations.append({"sequence": sequence, "phase": phase, "marker_instance": marker.id.wire,
            "source_line": number,
            "retained_ancestor": context, "python_ident": observation["python_ident"],
            "native_tid": observation["native_tid"], "native_tid_verified": native_tid_verified,
            "actual_provider_ancestry": ancestry})
        seen_paths = set()
        for ordinal, (leaf, slot) in enumerate(zip(leaves, slots, strict=True)):
            weak = _WEAK.fullmatch(leaf.get("ref", ""))
            path = leaf.get("path")
            if type(path) is not list or any(type(part) not in (int, str) for part in path):
                raise ValueError("invalid exact Tensor payload path")
            if tuple(path) in seen_paths:
                raise ValueError("duplicate Tensor payload path")
            seen_paths.add(tuple(path))
            if weak is None or int(weak[1]) != pid or leaf.get("identity_qualified") is not True:
                gaps["weak_reference_unqualified"] += 1; continue
            identifier = ObjectID(scope + ":" + leaf["ref"])
            claim = EvidenceClaim(EvidenceKind.INFERRED, (
                pointer("observations", f"#line/{number}"), pointer("metadata", "#"),
                pointer("native_meta", "#"), slot["raw_reference"]), scope=scope,
                assumptions=("fixed production collector passed the same held Tensor to this exact marker argument",
                    "association is local to this observation; no persistent content/storage identity",
                    "retained ancestor is not a direct caller",
                    "capture membership verified" if membership_verified else "caller supplied these artifacts as one capture"))
            if identifier not in values.objects and identifier not in pending_objects:
                pending_objects[identifier] = ObjectInstance(identifier, scope, (claim,), metadata={
                    "weak_token": leaf["ref"], "process_id": pid, "lifetime": "UNKNOWN",
                    "identity_basis": "existing call recorder weak-reference epoch"})
            edge_id = scope + f":slot:{number}:{ordinal}"
            pending_edges.append(EvidenceEdge(edge_id, EvidenceRelation.OBJECT_REFERENCE_ASSOCIATION,
                EvidenceEndpoint(N.OBJECT, identifier), EvidenceEndpoint(N.PROVIDER_REFERENCE, slot["provider_reference"]), claim))
            paths.append({"edge_id": edge_id, "sequence": sequence, "phase": phase,
                "original_payload_path": path, "provider_argument_ordinal": ordinal,
                "provider_role": "inputs", "marker_instance": marker.id.wire})
        if row.get("slot_gaps"):
            gaps["unqualified_payload_slots"] += len(row["slot_gaps"])
    for sequence, group in grouped.items():
        end = group.get("DISPATCH_END")
        if "DISPATCH_BEGIN" not in group or end is None:
            gaps["incomplete_dispatch_phase_inventory"] += 1; continue
        if end.get("outcome") == "RAISED":
            if "DISPATCH_OUTPUT" in group or end.get("output_marker") is not None:
                raise ValueError("raised dispatch falsely declares an output observation")
        elif end.get("outcome") == "RETURNED":
            if "DISPATCH_OUTPUT" not in group:
                gaps["returned_dispatch_output_unobserved"] += 1
            else:
                span_node = marker_nodes.get(group["DISPATCH_BEGIN"]["marker"])
                output_node = marker_nodes.get(group["DISPATCH_OUTPUT"]["marker"])
                if span_node is not None and output_node is not None and (
                        output_node.metadata["ctrl_deps"] != span_node.metadata["node_id"]
                        or output_node.metadata["raw_tid"] != span_node.metadata["raw_tid"]):
                    raise ValueError("output observation is not a real child of its dispatch span")
        else:
            raise ValueError("unknown dispatch outcome")
    if set(marker_nodes) - consumed:
        gaps["provider_markers_without_retained_sidecar"] += len(set(marker_nodes) - consumed)
    if meta.get("dropped_events") or meta.get("errors") or meta.get("gaps") or meta.get("io_failed"):
        gaps["collector_declared_partial"] += 1
    for item in pending_objects.values():
        values.add_object(item)
        evidence.register_external(EvidenceEndpoint(N.OBJECT, item.id))
    for edge in pending_edges:
        evidence.add_edge(edge)
    result.bundle.assert_valid()
    result.report["dispatch_bridge"] = {"schema": "scar.dispatch-observation-import", "schema_version": 1,
        "profile": PRODUCTION_PROFILE, "scope": scope, "production_collector": True,
        "capture_membership_verified": membership_verified,
        "artifacts": {key: encode(artifact) for key, artifact in artifacts.items()},
        "retained_input_paths": input_paths,
        "dispatches": len(grouped), "marker_nodes": len(marker_nodes), "slot_associations": len(paths),
        "outcomes": dict(Counter(group.get("DISPATCH_END", {}).get("outcome", "OPEN") for group in grouped.values())),
        "observations": observations, "paths": paths, "gaps": dict(gaps),
        "direct_python_operator_correspondence_established": False,
        "logical_versions_established": False, "world_effects_established": False,
        "resource_readiness_established": False, "whole_model_established": False,
        "transformation_authorized": False}
    return result


__all__ = [
    'DispatchEvidence',
    'PRODUCTION_PROFILE',
    'PROVIDER_REVISION',
    'import_dispatch_bridge',
]
