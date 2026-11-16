"""Adapt one fused C++ operator to the complete original pipeline contract.

The original calls and global callee guards remain authoritative. Fusion is
admitted by source-graph and representation-strategy replay, then executed by
one existing ABI-1 operator boundary. No pipeline workspace is allocated.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

from scar.compiler.model import document as operator_document
from scar.compiler.pipeline_model import (
    PipelinePlan, document, pipeline_digest, pipeline_from_dict, read_pipeline,
)
from scar.optimization.hardware import bind_cpu_placement
from scar.optimization.resource_solver import semantic_sha256
from . import cpp

GuardFailure = cpp.GuardFailure
SCHEMA = "scar.native-fused-pipeline-artifact"
_FILES = {"pipeline": "pipeline.json", "strategy": "strategy.json", "memory_plan": "memory-plan.json"}


def _digest(raw):
    return sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def fused_implementation_sha256():
    """Version this adapter separately from the existing C++ cost context."""
    return semantic_sha256({"fused_pipeline.py": _digest(Path(__file__).read_bytes()),
                            "operator_runtime_implementation_sha256": cpp.runtime_implementation_sha256()})


def _sources(pipeline):
    for source in (pipeline.source, *(call.callee.source for call in pipeline.calls)):
        _require(bool(source.code_key) and Path(source.path).is_absolute(),
                 "original pipeline/callee source versions are required")
        _require(_digest(Path(source.path).read_bytes()) == source.sha256,
                 "original pipeline/callee source snapshot differs")


def _admit(pipeline, strategy):
    from scar.optimization.pipeline_fusion import fuse_pipeline_region, verify_pipeline_fusion
    from scar.optimization.pipeline_strategy import (
        fused_execution_plan, verify_pipeline_strategy,
    )
    strategy = verify_pipeline_strategy(pipeline, strategy)
    _require(strategy["selected"]["kind"] == "FULL_FUSION",
             "fused backend requires the selected FULL_FUSION representation")
    raw, _ = fuse_pipeline_region(pipeline)
    _require(raw is not None, "complete fusion source graph is unsupported")
    verify_pipeline_fusion(pipeline, raw, strategy["fusion_certificate"])
    # The strategy helper independently replays raw -> optimized plan. Its
    # source operation map belongs to that bridge, not to the raw certificate.
    expected = fused_execution_plan(pipeline, strategy)
    cpp._supported(expected)
    _require(expected.region.source == pipeline.source and
             expected.region.parameters == pipeline.parameters and expected.resource_binding is None,
             "fusion must retain the original outer source and parameter contract")
    memory = strategy["selected_memory_plan"]
    shape = memory["output_shape"]
    _require(len(shape) == 2 and all(type(n) is int and 0 <= n <= sys.maxsize // 8 for n in shape),
             "fusion output shape exceeds host bounds")
    _require(type(memory["final_bytes"]) is int and memory["final_bytes"] == shape[0] * shape[1] * 8
             and memory["final_bytes"] <= sys.maxsize and memory["total_workspace_bytes"] == 0
             and memory["peak_allocation_bytes"] == memory["final_bytes"],
             "fusion requires only one fresh final payload and zero workspace")
    budget = memory["memory_budget_bytes"]
    _require(type(budget) is int and memory["final_bytes"] <= budget <= sys.maxsize,
             "fusion output exceeds the declared payload budget")
    _require(set(memory["input_shapes"]) == set(pipeline.parameters),
             "fusion shapes must cover the original outer parameters")
    for shape in memory["input_shapes"].values():
        _require(len(shape) == 2 and all(type(n) is int and 0 <= n <= sys.maxsize // 8 for n in shape)
                 and shape[0] * shape[1] * 8 <= sys.maxsize,
                 "fusion input shape exceeds host bounds")
    return strategy, expected, memory


def _compatible(expected, operator, memory):
    actual = operator.plan
    cpp._supported(actual)
    # Reconstruct the sole allowed binding transformation, including target
    # receipts. Arbitrary proof/cost/loop/DAG edits cannot masquerade as pinning.
    bound = bind_cpu_placement(expected, actual.resource_binding) if actual.resource_binding is not None else expected
    _require(operator_document(actual) == operator_document(bound),
             "fused operator differs from the replayed plan or admitted CPU binding")
    budget = actual.cost.get("memory_budget_bytes")
    _require(budget is not None and memory["final_bytes"] <= budget <= memory["memory_budget_bytes"],
             "fused operator cannot relax the selected final payload budget")
    if operator.timed_admission is not None:
        guard = operator.timed_admission["must_guard"]
        _require(guard["input_shapes"] == memory["input_shapes"] and
                 guard["output_shape"] == memory["output_shape"] and
                 guard["output_nbytes"] == memory["final_bytes"] and
                 guard["memory_budget_bytes"] == budget and guard["input_layout"] == "C_CONTIGUOUS",
                 "fused operator timed obligations differ from the selected representation")


def _operator_receipt(operator):
    metadata = operator.report
    files = {kind: {"path": str(operator.directory / name),
                    "sha256": _digest((operator.directory / name).read_bytes())}
             for kind, name in metadata["files"].items()}
    files["manifest"] = {"path": str(operator.metadata_path),
                         "sha256": _digest(operator.metadata_path.read_bytes())}
    return {"artifact_directory": str(operator.directory), "build_id": metadata["build_id"],
            "plan_sha256": metadata["plan_sha256"], "library_sha256": metadata["sha256"]["library"],
            "cpu_threads": operator.plan.cpu_threads,
            "cpu_ids": list(operator.plan.resource_binding.cpu_ids) if operator.plan.resource_binding else None,
            "source_code_key": operator.plan.region.source.code_key, "files": files}


def _identity(pipeline, strategy, memory, operator_receipt, fingerprint):
    return semantic_sha256({"schema": SCHEMA, "schema_version": 1,
        "pipeline_sha256": pipeline_digest(pipeline), "strategy_sha256": semantic_sha256(strategy),
        "memory_plan_sha256": semantic_sha256(memory), "operator": operator_receipt,
        "fused_implementation_sha256": fingerprint})


def _manifest(pipeline, strategy, memory, operator, binding, fingerprint, directory):
    return {"schema": SCHEMA, "schema_version": 1, "backend": "cpp",
        "build_id": _identity(pipeline, strategy, memory, binding, fingerprint),
        "pipeline_sha256": pipeline_digest(pipeline), "strategy_sha256": semantic_sha256(strategy),
        "memory_plan_sha256": semantic_sha256(memory), "fused_implementation_sha256": fingerprint,
        "operator_runtime_implementation_sha256": operator.report.get("runtime_implementation_sha256"),
        "files": dict(_FILES), "sha256": {key: _digest((directory / name).read_bytes()) for key, name in _FILES.items()},
        "operators": [binding], "kernel_count": 1, "native_ffi_calls_per_run": 1,
        "original_call_count": len(pipeline.calls), "original_callee_guards_retained": True,
        "execution": "ONE_CPP_OPERATOR_RUN_OVER_REPLAYED_COMPLETE_FUSION",
        "workspace_bytes": 0, "final_bytes": memory["final_bytes"],
        "incremental_payload_bytes": memory["peak_allocation_bytes"], "memory_budget_bytes": memory["memory_budget_bytes"],
        "budget_scope": "INCREMENTAL_PAYLOAD_FRESH_FINAL_ONLY",
        "budget_excludes": ["EXTERNAL_INPUTS", "ALLOCATOR_METADATA_AND_ALIGNMENT_OVERHEAD", "THREAD_SCRATCH", "PROCESS_RSS"],
        "final_ownership": "FRESH_NUMPY_ARRAY_CALLER_OWNED", "memory_policy": "UNBOUND",
        "numa_page_binding": False, "asynchronous_transfers": False,
        "source_guard_scope": "SNAPSHOT_BUILD_LOAD_ORIGINAL_PIPELINE_AND_ALL_CALLEES_EXTERNAL_ENTRY_CODE_ID_GUARDS",
        "cost_scope": "COMPLETE_FUSION_REPRESENTATION_PAYLOAD_ONLY_ADAPTER_TIME_UNCALIBRATED",
        "duration_predicted": False, "physical_optimality_proven": False, "deployment_authorized": False}


@dataclass
class FusedPipelineArtifact:
    pipeline: PipelinePlan
    directory: Path
    _strategy: dict
    _memory: dict
    _metadata: dict
    _operator: cpp.CppArtifact

    @property
    def memory_plan(self):
        return json.loads(json.dumps(self._memory))

    @property
    def report(self):
        return {**json.loads(json.dumps(self._metadata)), "artifact_directory": str(self.directory),
                "metadata_path": str(self.directory / "artifact.json"),
                "source_path": str(self._operator.source_path),
                "library_path": str(self._operator.library_path)}

    def run(self, *arrays):
        import numpy as np
        if len(arrays) != len(self.pipeline.parameters):
            raise GuardFailure("fused pipeline input count differs")
        if any('EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR' in call.plan.region.assumptions
               for call in self.pipeline.calls):
            from scar.runtime.array_copy_guard import guard_copy_inputs,CopyGuardFailure
            try:guard_copy_inputs(arrays)
            except CopyGuardFailure as error:raise GuardFailure(str(error))from error
        for parameter, array in zip(self.pipeline.parameters, arrays):
            if (type(array) is not np.ndarray or array.shape != tuple(self._memory["input_shapes"][parameter])
                    or not array.flags.c_contiguous):
                raise GuardFailure("fused pipeline input shape/type/layout differs from selected strategy")
        # The existing boundary verifies ownership, dtype, bounds, alignment,
        # output budget, affinity and floating status, and returns a fresh final.
        return self._operator.run(*arrays)


def build(pipeline, out, *, strategy, operator):
    """Admit a selected complete fusion and retain the original pipeline guards."""
    pipeline = read_pipeline(pipeline) if isinstance(pipeline, (str, Path)) else pipeline_from_dict(document(pipeline))
    strategy, expected, memory = _admit(pipeline, strategy)
    _sources(pipeline)
    operator = cpp.load(operator.directory if isinstance(operator, cpp.CppArtifact) else operator)
    _compatible(expected, operator, memory)
    binding = _operator_receipt(operator)
    fingerprint = fused_implementation_sha256()
    directory = Path(out).resolve()
    _require(not directory.exists(), "fresh fused pipeline output directory required")
    directory.mkdir(parents=True)
    for key, value in (("pipeline", document(pipeline)), ("strategy", strategy), ("memory_plan", memory)):
        (directory / _FILES[key]).write_bytes(_json_bytes(value))
    metadata = _manifest(pipeline, strategy, memory, operator, binding, fingerprint, directory)
    (directory / "artifact.json").write_bytes(_json_bytes(metadata))
    return load(directory)


def load(out):
    path = Path(out).resolve(); directory = path if path.is_dir() else path.parent
    metadata = json.loads((directory / "artifact.json").read_bytes())
    _require((metadata.get("schema"), metadata.get("schema_version")) == (SCHEMA, 1),
             "unsupported fused pipeline artifact schema")
    _require(metadata["files"] == _FILES,
             "unexpected or missing fused pipeline sidecars")
    for key, name in metadata["files"].items():
        _require(Path(name).name == name and _digest((directory / name).read_bytes()) == metadata["sha256"][key],
                 "fused pipeline sidecar receipt differs: " + key)
    pipeline = read_pipeline(directory / metadata["files"]["pipeline"])
    strategy, expected, memory = _admit(pipeline, directory / metadata["files"]["strategy"])
    _sources(pipeline)
    _require(json.loads((directory / metadata["files"]["memory_plan"]).read_bytes()) == memory,
             "fused pipeline memory plan differs from strategy replay")
    _require(len(metadata["operators"]) == 1, "complete fusion requires exactly one native operator")
    binding = metadata["operators"][0]
    for receipt in binding["files"].values():
        _require(_digest(Path(receipt["path"]).read_bytes()) == receipt["sha256"],
                 "fused pipeline operator receipt differs")
    operator = cpp.load(binding["artifact_directory"])
    _compatible(expected, operator, memory)
    _require(_operator_receipt(operator) == binding, "fused pipeline operator binding differs")
    _require(metadata["pipeline_sha256"] == pipeline_digest(pipeline)
             and metadata["strategy_sha256"] == semantic_sha256(strategy)
             and metadata["memory_plan_sha256"] == semantic_sha256(memory),
             "fused pipeline semantic receipt differs")
    _require(metadata["build_id"] == _identity(pipeline, strategy, memory, binding,
             metadata["fused_implementation_sha256"]), "fused pipeline build identity differs")
    _require(metadata == _manifest(pipeline, strategy, memory, operator, binding,
             metadata["fused_implementation_sha256"], directory), "fused pipeline report differs from admitted execution")
    return FusedPipelineArtifact(pipeline, directory, strategy, memory, metadata, operator)


__all__ = ["FusedPipelineArtifact", "GuardFailure", "SCHEMA", "build", "load", "fused_implementation_sha256"]
