"""Repeated regions and loop instances from dynamic execution evidence."""
from __future__ import annotations

from collections import defaultdict
from itertools import islice

from scar.ir import ExecutionGraph, Knowledge, Opportunity, ProofStatus, proof


def _effect_status(calls) -> ProofStatus:
    effects = [event.effect for _, event in calls]
    if all(effect.safe_for_exact_reuse for effect in effects):
        return ProofStatus.PROVEN
    for effect in effects:
        if (effect.writes or effect.allocates or effect.frees or effect.aliases or
                effect.escapes or any(value == Knowledge.KNOWN for value in (
                    effect.rng_effect, effect.may_raise, effect.external_effect,
                    effect.ordering_effect))):
            return ProofStatus.DISPROVEN
    return ProofStatus.UNKNOWN


def _reuse_proofs(calls, *, same_inputs: bool, effects: ProofStatus,
                  per_invocation_inputs: bool = True):
    support = [idx for idx, _ in calls]
    input_status = (ProofStatus.PROVEN if same_inputs and per_invocation_inputs else
                    ProofStatus.DISPROVEN if per_invocation_inputs else
                    ProofStatus.UNKNOWN)
    return [
        proof(
            "same_input_versions", "applicability",
            "every repeated invocation reads the same logical versions and regions",
            input_status, evidence="Observed" if per_invocation_inputs else "UNKNOWN",
            reason=("input region signatures are identical" if input_status == ProofStatus.PROVEN
                    else "input region signatures changed" if input_status == ProofStatus.DISPROVEN
                    else "aggregate profiler record has no per-invocation inputs"),
            supporting_events=support,
        ),
        proof(
            "effects_allow_exact_reuse", "legality",
            "all invocations have complete effects with no mutation, allocation, alias, escape, RNG, exception, external or ordering effect",
            effects, evidence="Observed" if effects != ProofStatus.UNKNOWN else "UNKNOWN",
            reason=("complete exact-reuse effect contracts" if effects == ProofStatus.PROVEN
                    else "a visible effect disproves exact reuse" if effects == ProofStatus.DISPROVEN
                    else "one or more effect fields are incomplete"),
            supporting_events=support,
        ),
    ]


def _input_signature(event) -> tuple:
    # Shape alone is not a region identity: slices and transposed/expanded
    # views can share a logical version while selecting different storage.
    # Include offset and strides so a repeated-region candidate cannot join
    # distinct aliases accidentally.
    return tuple((x.get("storage_id"), x.get("logical_version"),
                  x.get("offset", 0), tuple(x.get("shape", [])),
                  tuple(x.get("strides", [])), x.get("dtype"),
                  x.get("device")) for x in event.inputs)


def _operator_candidate(code_id, calls, *, profiler_count: int | None = None) -> Opportunity:
    """Build a conservative operator candidate from dynamic call evidence.

    Dispatch mode supplies per-call tensor regions; the profiler aggregate
    supplies only a count and time.  Both are useful observations, but neither
    is treated as proof of purity.  In particular, an empty input signature
    can mean allocation or RNG rather than a reusable constant.
    """
    dispatch = profiler_count is None
    first = calls[0][1]
    if dispatch:
        signature = _input_signature(first)
        same_inputs = all(_input_signature(event) == signature for _, event in islice(calls, 1, None))
        duration = sum(event.duration_ns or 0 for _, event in islice(calls, 1, None))
        effect_status = _effect_status(calls)
        applicability = "same_input_logical_versions" if same_inputs else "changed_input_logical_versions"
        reason = (f"operator {code_id} executed {len(calls)} times with the same "
                  "observed input region signature"
                  if same_inputs else
                  f"operator {code_id} repeated with changing input region signatures")
        guard = ("storage epochs/logical versions, alias overlap, writes, RNG, "
                 "stream ordering, and contract must be proven before reuse")
        missing = "operator effects or invalidation guard is incomplete"
    else:
        same_inputs = False
        duration = first.duration_ns or 0
        effect_status = ProofStatus.UNKNOWN
        applicability = "dynamic_operator_count_gt_one"
        reason = f"torch operator executed {profiler_count} times in the profiled process"
        guard = "operator inputs, logical versions, effects, and contract must be captured before reuse"
        missing = "profiler aggregate has no per-invocation input/effect guard"
    safe = same_inputs and effect_status == ProofStatus.PROVEN
    return Opportunity(
        kind="RepeatedOperatorCandidate", code_id=code_id, evidence="Observed",
        applicability=applicability, guard=guard, reason=reason,
        supporting_events=[idx for idx, _ in calls], expected_savings_ns=duration,
        decision="proposed" if safe else "rejected",
        backend="exact_reuse" if safe else None,
        rejection_reason=None if safe else missing,
        proof_obligations=_reuse_proofs(
            calls, same_inputs=same_inputs, effects=effect_status,
            per_invocation_inputs=dispatch),
    )


def _python_candidate(code_id, calls) -> Opportunity:
    """Report repeated user-Python work without assuming it is pure.

    Python calls cover parsing, indexing, preprocessing, dispatch and arbitrary
    user code.  Their input records can be compared when the tracer captured
    them, but globals, registries, callbacks and external effects remain
    outside the default contract.  Consequently this detector is useful for
    discovery and accounting while the default decision stays conservative.
    """
    effect_status = _effect_status(calls)
    support = [idx for idx, _ in calls]
    duration = sum(event.duration_ns or 0 for _, event in islice(calls, 1, None))
    safe = effect_status == ProofStatus.PROVEN
    return Opportunity(
        kind="PythonReuseCandidate", code_id=code_id, evidence="Observed",
        applicability="same_input_logical_versions",
        guard=("Python globals/defaults, callbacks, external effects, exceptions, "
               "logical versions, escapes and ordering must be proven before reuse"),
        reason=f"Python region {code_id} executed {len(calls)} times with the same observed input signature",
        supporting_events=support, expected_savings_ns=duration,
        decision="proposed" if safe else "rejected",
        backend="exact_reuse" if safe else None,
        rejection_reason=None if safe else (
            "visible effects make exact reuse illegal" if effect_status == ProofStatus.DISPROVEN
            else "Python effect, callback, or invalidation knowledge is incomplete"),
        proof_obligations=_reuse_proofs(
            calls, same_inputs=True, effects=effect_status),
    )


def repeated_regions(graph: ExecutionGraph) -> list[Opportunity]:
    module_groups = defaultdict(list)
    python_groups = defaultdict(list)
    dispatch_groups = defaultdict(list)
    aggregates = []
    for idx, event in enumerate(graph.events):
        if event.kind == "module_call" and event.code_id:
            module_groups[(event.code_id, _input_signature(event))].append((idx, event))
        elif event.kind == "python_call" and event.code_id:
            python_groups[(event.code_id, _input_signature(event))].append((idx, event))
        elif event.kind == "torch_dispatch" and event.code_id:
            dispatch_groups[event.code_id].append((idx, event))
        elif event.kind == "torch_op":
            count = int(event.resource.get("count", event.invocation_id or 0))
            if count > 1:
                aggregates.append((idx, event, count))

    # Emit the original family order, using the global event ordinals throughout.
    out: list[Opportunity] = []
    for (code_id, _signature), calls in module_groups.items():
        if len(calls) < 2:
            continue
        duration = sum(event.duration_ns or 0 for _, event in islice(calls, 1, None))
        effect_status = _effect_status(calls)
        safe = effect_status == ProofStatus.PROVEN
        out.append(Opportunity(
            kind="ReuseCandidate", code_id=code_id,
            evidence="Observed", applicability="same_input_logical_versions",
            guard="input storage epochs and logical versions unchanged; effects and contract proven exact",
            reason=f"{len(calls)} dynamic executions share the same input region signature",
            supporting_events=[idx for idx, _ in calls], expected_savings_ns=duration,
            decision="proposed" if safe else "rejected",
            backend="exact_reuse" if safe else None,
            rejection_reason=None if safe else
                ("visible effects make exact reuse illegal"
                 if effect_status == ProofStatus.DISPROVEN else
                 "effect knowledge is incomplete or input logical versions changed"),
            proof_obligations=_reuse_proofs(
                calls, same_inputs=True, effects=effect_status),
        ))

    # User Python spans are a first-class avoidable-work source (parse,
    # indexing, preprocessing, dispatch).  Keep them separate from module
    # calls because a Python span has no implicit tensor purity contract.
    for (code_id, _signature), calls in python_groups.items():
        if len(calls) > 1:
            out.append(_python_candidate(code_id, calls))

    # Per-invocation operator evidence comes from TorchDispatchMode. It gives
    # tensor regions and versions, while effects are intentionally UNKNOWN;
    # therefore this path can explain *why* a candidate is rejected without
    # silently turning an observed repeat into an optimization.
    for code_id, calls in dispatch_groups.items():
        if len(calls) > 1:
            out.append(_operator_candidate(code_id, calls))

    # Aggregate evidence comes from torch.profiler. It gives reliable dynamic
    # counts and time, but no complete per-invocation signature.
    for idx, event, count in aggregates:
        out.append(_operator_candidate(event.code_id, [(idx, event)], profiler_count=count))
    return out


def loop_invariant_candidates(graph: ExecutionGraph) -> list[Opportunity]:
    """Suggest repeated module work observed across a bytecode loop back-edge.

    Line tracing only labels actions after a back-edge, so this detector is
    deliberately conservative and may see a suffix of the loop rather than
    every iteration. It emits a candidate only when at least two distinct
    observed iterations contain the same module CodeID. Hoisting is not
    assumed legal: effects, aliases, exceptions, loop control and consumers
    still require a backend-specific proof.
    """
    grouped = defaultdict(list)
    for idx, event in enumerate(graph.events):
        if event.kind != "module_call" or not event.code_id:
            continue
        metadata = event.metadata
        if metadata.get("loop_iteration") is None:
            continue
        key = (metadata.get("loop_parent_invocation_id",
                           metadata.get("parent_invocation_id")),
               metadata.get("loop_target_offset"), event.code_id)
        grouped[key].append((idx, event))

    result: list[Opportunity] = []
    for (parent, offset, code_id), calls in grouped.items():
        iterations = {event.metadata.get("loop_iteration") for _, event in calls}
        if len(iterations) < 2:
            continue
        signatures = {_input_signature(event) for _, event in calls}
        same_inputs = len(signatures) == 1 and () not in signatures
        duration = sum(event.duration_ns or 0 for _, event in islice(calls, 1, None))
        effect_status = _effect_status(calls)
        effects_known = effect_status == ProofStatus.PROVEN
        eligible = same_inputs and effects_known
        if not same_inputs:
            rejection = "input logical versions or region signatures changed across iterations"
        elif not effects_known:
            rejection = "loop effects or invalidation guard is incomplete"
        else:
            rejection = "planner must measure exact-reuse guard and lookup cost"
        result.append(Opportunity(
            kind="LoopInvariantCandidate", code_id=code_id, evidence="Observed",
            applicability=("same_inputs_across_loop_iterations" if same_inputs
                           else "changing_inputs_across_loop_iterations"),
            guard=("loop control, input logical versions, aliases, effects, exceptions, "
                   "consumer/escape ordering and measured hoisting cost must be proven"),
            reason=(f"module executed in {len(iterations)} observed loop iterations"
                    + (" with the same input region signature" if same_inputs else
                       " with changing input region signatures")),
            supporting_events=[idx for idx, _ in calls],
            expected_savings_ns=duration,
            decision="proposed" if eligible else "rejected",
            backend="exact_reuse" if eligible else None,
            rejection_reason=None if eligible else rejection,
            proof_obligations=(
                [proof(
                    "repeated_loop_scope", "applicability",
                    "the same action occurs in at least two observed iterations of one dynamic loop",
                    ProofStatus.PROVEN, evidence="Observed",
                    reason=f"observed in {len(iterations)} loop iterations",
                    supporting_events=[idx for idx, _ in calls],
                    details={"parent_invocation_id": parent,
                             "loop_target_offset": offset},
                )]
                + _reuse_proofs(calls, same_inputs=same_inputs,
                                effects=effect_status)
                + [proof(
                    "hoisting_preserves_loop_control", "legality",
                    "moving the action outside the loop preserves exceptions, branches, ordering and consumers",
                    ProofStatus.UNKNOWN, evidence="UNKNOWN",
                    reason="dynamic repetition does not prove control equivalence",
                    supporting_events=[idx for idx, _ in calls],
                )]
            ),
        ))
    return result
