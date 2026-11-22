"""Runtime evidence about allocation, materialization and ordering."""
from __future__ import annotations

from collections import defaultdict

from scar.ir import ExecutionGraph, Opportunity, ProofStatus, proof


def _residency_legality_proofs(support: list[int]):
    return [
        proof(
            "no_logical_invalidation", "legality",
            "the logical version is not invalidated while the resident copy is reused",
            ProofStatus.UNKNOWN, evidence="UNKNOWN",
            reason="the observed transfers do not prove all foreign and alias writes",
            supporting_events=support,
        ),
        proof(
            "residency_lifetime_safe", "legality",
            "destination storage lifetime, stream ordering, consumers and escapes permit retention",
            ProofStatus.UNKNOWN, evidence="UNKNOWN",
            reason="lifetime, ordering and escape evidence is incomplete",
            supporting_events=support,
        ),
    ]


_DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2, "float64": 8,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1,
    "uint8": 1, "bool": 1,
}

# Runtime serializers may report a dtype as ``torch.float32``, ``float32``
# or a NumPy spelling.  These are representations of the same contract, so
# normalise the spelling before estimating residency cost.
_DTYPE_ALIASES = {
    "float": "float32", "float32": "float32", "torch.float": "float32",
    "torch.float32": "float32", "float16": "float16", "torch.float16": "float16",
    "bfloat16": "bfloat16", "torch.bfloat16": "bfloat16",
    "float64": "float64", "torch.float64": "float64",
    "int64": "int64", "torch.int64": "int64", "long": "int64",
    "int32": "int32", "torch.int32": "int32", "int16": "int16",
    "torch.int16": "int16", "int8": "int8", "torch.int8": "int8",
    "uint8": "uint8", "torch.uint8": "uint8", "bool": "bool", "torch.bool": "bool",
}


def _bytes(value: dict) -> int | None:
    try:
        if "shape" not in value:
            return None
        size = 1
        for dimension in value.get("shape", []):
            size *= int(dimension)
        dtype = str(value.get("dtype")).lower()
        dtype = _DTYPE_ALIASES.get(dtype, dtype)
        itemsize = _DTYPE_BYTES.get(dtype)
        return size * itemsize if itemsize is not None else None
    except (TypeError, ValueError):
        return None


def materialization_candidates(graph: ExecutionGraph) -> list[Opportunity]:
    transfers = defaultdict(list)
    physical = defaultdict(list)
    for idx, event in enumerate(graph.events):
        if event.kind == "transfer" and event.inputs and event.outputs:
            logical = event.inputs[0].get("logical_version")
            if logical and event.metadata.get("physical") is True:
                transfers[(logical, event.outputs[0].get("device"))].append((idx, event))
        elif event.kind == "cuda_memcpy":
            # Kineto observes the physical copy. Only a unique logical mapping
            # associates an activity with a residency candidate.
            logical = event.metadata.get("logical_version")
            confidence = event.metadata.get("mapping_confidence")
            if logical and confidence in {"Observed", "Inferred"}:
                destination = event.resource.get("device", "unknown")
                direction = event.resource.get("direction", "unknown")
                physical[(logical, direction, destination)].append((idx, event))
    result = []
    for (version, device), values in transfers.items():
        if len(values) > 1:
            sizes = [_bytes(event.outputs[0]) for _, event in values]
            support = [i for i, _ in values]
            result.append(Opportunity(
                kind="ResidencyCandidate", code_id=None, evidence="Observed",
                applicability="same_logical_version_repeated_transfer",
                guard="no intervening invalidation and destination storage remains live",
                reason=f"logical version {version} materialized {len(values)} times on {device}",
                supporting_events=support, decision="rejected",
                backend="persistent_residency",
                rejection_reason="residency lifetime and invalidation guard is incomplete",
                estimated_memory_bytes=max(sizes) if all(size is not None for size in sizes) else None,
                proof_obligations=[
                    proof(
                        "repeated_physical_materialization", "applicability",
                        "one logical version is physically materialized on the same destination more than once",
                        ProofStatus.PROVEN, evidence="Observed",
                        reason=f"observed {len(values)} physical transfer events",
                        supporting_events=support,
                    ),
                ] + _residency_legality_proofs(support),
            ))

    for (logical, direction, destination), values in physical.items():
        if len(values) < 2:
            continue
        sizes = [event.resource.get("bytes") for _, event in values]
        support = [idx for idx, _ in values]
        mapping_status = (ProofStatus.PROVEN
                          if all(event.metadata.get("mapping_confidence") == "Observed"
                                 for _, event in values) else ProofStatus.UNKNOWN)
        result.append(Opportunity(
            kind="ResidencyCandidate", code_id=None,
            evidence="Observed" if mapping_status == ProofStatus.PROVEN else "Inferred",
            applicability="same_logical_version_repeated_physical_materialization",
            guard=("logical-version invalidation, storage lifetime, stream ordering, "
                   "consumer/escape behavior and residency memory budget must be proven"),
            reason=(f"physical {direction} materialization observed {len(values)} times for "
                    f"logical version {logical} on resource {destination}"),
            supporting_events=support,
            expected_savings_ns=sum(event.duration_ns or 0 for _, event in values[1:]),
            estimated_memory_bytes=max(sizes) if all(isinstance(size, int) for size in sizes) else None,
            decision="rejected", backend="persistent_residency",
            rejection_reason="residency lifetime and invalidation guard is incomplete",
            proof_obligations=[
                proof(
                    "repeated_physical_materialization", "applicability",
                    "the same logical version has repeated physical copy activities",
                    ProofStatus.PROVEN, evidence="Observed",
                    reason=f"Kineto observed {len(values)} {direction} copies",
                    supporting_events=support,
                ),
                proof(
                    "physical_to_logical_mapping", "applicability",
                    "each physical copy is uniquely correlated to the stated logical version",
                    mapping_status,
                    evidence="Observed" if mapping_status == ProofStatus.PROVEN else "Inferred",
                    reason=("runtime correlation is observed" if mapping_status == ProofStatus.PROVEN
                            else "signature join is inferred and needs a runtime correlation ID"),
                    supporting_events=support,
                ),
            ] + _residency_legality_proofs(support),
        ))
    return result


_ALLOC_OPS = {
    "aten::empty", "aten::empty_strided", "aten::empty.memory_format",
    "aten::zeros", "aten::ones", "aten::clone", "aten::new_empty",
}


def allocation_candidates(graph: ExecutionGraph) -> list[Opportunity]:
    out: list[Opportunity] = []
    for idx, event in enumerate(graph.events):
        if event.kind != "torch_op" or not event.code_id:
            continue
        name = str(event.code_id).lower()
        if name not in _ALLOC_OPS:
            continue
        count = int(event.resource.get("count", 0))
        if count < 2:
            continue
        memory_bytes = max(
            abs(int(event.resource.get("self_device_memory_usage", 0))),
            abs(int(event.resource.get("device_memory_usage", 0))),
            abs(int(event.resource.get("self_cpu_memory_usage", 0))),
            abs(int(event.resource.get("cpu_memory_usage", 0))),
        )
        out.append(Opportunity(
            kind="AllocationReuseCandidate", code_id=event.code_id,
            evidence="Observed", applicability="repeated_runtime_allocation",
            guard=("allocation lifetime, alias overlap, writes, escapes, peak-memory "
                   "budget, and allocator ordering must be proven"),
            reason=(f"profiler observed {count} executions of allocation-like operator"
                    + (f"; measured memory delta {memory_bytes} bytes" if memory_bytes else
                       "; memory delta was unavailable")),
            supporting_events=[idx], expected_savings_ns=int(event.duration_ns or 0),
            # Aggregated net allocation deltas are not retained/peak memory.
            # Lifetime analysis is needed before estimating buffer residency.
            estimated_memory_bytes=None,
            decision="rejected", rejection_reason="allocation lifetime and alias guard is incomplete",
            proof_obligations=[
                proof(
                    "repeated_allocation", "applicability",
                    "the allocation action executes repeatedly in the observed run",
                    ProofStatus.PROVEN, evidence="Observed",
                    reason=f"profiler count is {count}", supporting_events=[idx],
                ),
                proof(
                    "buffer_lifetimes_do_not_overlap", "legality",
                    "old and new buffer live ranges permit one storage allocation to be reused",
                    ProofStatus.UNKNOWN, evidence="UNKNOWN",
                    reason="aggregate profiler memory deltas do not prove allocation lifetimes",
                    supporting_events=[idx],
                ),
                proof(
                    "buffer_aliases_and_escapes_safe", "legality",
                    "no live alias or escaped consumer observes destructive buffer reuse",
                    ProofStatus.UNKNOWN, evidence="UNKNOWN",
                    reason="alias and escape sets are incomplete",
                    supporting_events=[idx],
                ),
            ],
        ))
    return out


_SCALAR_NAMES = {"aten::item", "aten::_local_scalar_dense", "item"}
_BARRIER_NAMES = {"cudadevicesynchronize", "aten::synchronize"}


def synchronization_candidates(graph: ExecutionGraph) -> list[Opportunity]:
    result: list[Opportunity] = []
    dynamic = defaultdict(list)
    for idx, event in enumerate(graph.events):
        if event.kind == "cuda_barrier":
            dynamic[str(event.metadata.get("operation", "unknown"))].append((idx, event))
    observed_names = {name.lower() for name in dynamic}
    for name, calls in dynamic.items():
        duration = sum(event.duration_ns or 0 for _, event in calls)
        result.append(Opportunity(
            kind="SynchronizationCandidate", code_id=None, evidence="Observed",
            applicability="measured_runtime_wait",
            guard="stream/event, host consumer, escape and exception dependencies must prove removable waiting",
            reason=(f"Kineto observed {len(calls)} {name} calls with {duration} ns total host API duration; "
                    "this is not an estimate of removable time"),
            supporting_events=[idx for idx, _ in calls], expected_savings_ns=None,
            decision="rejected", rejection_reason="consumer or ordering guard is incomplete",
            proof_obligations=[
                proof(
                    "runtime_wait_observed", "applicability",
                    "a physical host or device wait occurs in the observed execution",
                    ProofStatus.PROVEN, evidence="Observed",
                    reason=f"Kineto observed {len(calls)} wait activities",
                    supporting_events=[idx for idx, _ in calls],
                ),
                proof(
                    "wait_has_no_required_consumer", "legality",
                    "removing or moving the wait preserves stream, host consumer, escape, exception and external ordering",
                    ProofStatus.UNKNOWN, evidence="UNKNOWN",
                    reason="the trace does not contain a complete happens-before and consumer proof",
                    supporting_events=[idx for idx, _ in calls],
                ),
            ],
        ))
    for idx, event in enumerate(graph.events):
        if event.kind != "torch_op" or not event.code_id:
            continue
        name = str(event.code_id)
        lowered = name.lower()
        if lowered in observed_names:
            continue  # A profiler aggregate is not another set of dynamic calls.
        count = int(event.resource.get("count", event.invocation_id or 0))
        if name in _SCALAR_NAMES or lowered in _BARRIER_NAMES:
            if count < 1:
                continue
            scalar = name in _SCALAR_NAMES
            result.append(Opportunity(
                kind="DeferredMaterializationCandidate" if scalar else "SynchronizationCandidate",
                code_id=name, evidence="Observed",
                applicability="repeated_host_scalar_materialization" if scalar else "explicit_device_barrier",
                guard=("consumer, escape, exception, ordering, and contract effects must prove "
                       "that materialization can be delayed" if scalar else
                       "stream/event dependencies and host-visible ordering must prove that the barrier can be removed"),
                reason=(f"profiler observed {count} host scalar materializations" if scalar else
                        f"profiler observed {count} explicit device synchronization calls"),
                supporting_events=[idx], expected_savings_ns=event.duration_ns or 0,
                decision="rejected",
                rejection_reason="consumer or ordering guard is incomplete",
                proof_obligations=[
                    proof(
                        "host_materialization_or_barrier_observed", "applicability",
                        ("a device value is materialized for host scalar use"
                         if scalar else "an explicit device synchronization is invoked"),
                        ProofStatus.PROVEN, evidence="Observed",
                        reason=f"profiler observed {count} calls", supporting_events=[idx],
                    ),
                    proof(
                        "defer_or_remove_preserves_consumers", "legality",
                        "all consumers, branches, exceptions, escapes and ordering remain equivalent",
                        ProofStatus.UNKNOWN, evidence="UNKNOWN",
                        reason="aggregate operator evidence lacks complete consumer and ordering dependencies",
                        supporting_events=[idx],
                    ),
                ],
            ))
    return result
