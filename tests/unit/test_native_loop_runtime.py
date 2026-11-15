"""The deployment boundary reads a pinned small bundle, without planner imports."""
import builtins
import ctypes
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import struct
import sys

import pytest

from scar.runtime.loop_contract import canonical_sha256, TRUST_POLICY


def test_pinned_light_bundle_abi_guards_and_fresh_array_without_planner(tmp_path, monkeypatch):
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("runtime profile requires Linux x86_64")
    original_import = builtins.__import__
    forbidden = ("numpy", "scar.compiler", "scar.optimization", "scar.analysis", "scar.codegen", "scar.ir")
    def minimal_import(name, *args, **kwargs):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            raise AssertionError("deployment imported forbidden module: " + name)
        return original_import(name, *args, **kwargs)
    module_path = Path(__file__).resolve().parents[2] / "scar/runtime/native_loop.py"
    spec = importlib.util.spec_from_file_location("scar.runtime._native_loop_unit", module_path)
    native = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, native)
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", minimal_import)
        spec.loader.exec_module(native)
    assert ctypes.sizeof(native._Array) == 64 and ctypes.sizeof(native._Table) == 56
    cpu = min(os.sched_getaffinity(0))
    native_contract = dict(schema="scar.native-loop-contract", schema_version=2,
        loop_sha256="1" * 64, strategy_sha256="2" * 64, execution_plan_sha256="3" * 64,
        trip_count=7, state_shape=[2, 3], output_shape=[2, 3], final_bytes=48,
        memory_budget_bytes=48, layout_sha256="4" * 64)
    ids = {key:native_contract[key] for key in ("loop_sha256", "strategy_sha256", "execution_plan_sha256")}
    ids.update(build_id="5" * 64, contract_sha256=canonical_sha256(native_contract))
    def source(name):
        return dict(path=str(tmp_path / (name + ".py")), sha256="6" * 64,
                    code_key="pycode-v2:" + "7" * 64, qualname=name)
    header = bytearray(64); header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", header, 16, 3, 62)
    library_path = tmp_path / "loop.so"; library_path.write_bytes(bytes(header) + b"unit ABI fixture")
    library_hash = sha256(library_path.read_bytes()).hexdigest()
    assumptions = ["SYNCHRONOUS_CPU_EXECUTION", "NO_FOREIGN_CONCURRENT_INPUT_WRITERS",
                   "FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT"]
    contract = dict(schema="scar.loop-runtime-contract", schema_version=1, profile="cpu-f64-rank2-loop-v1",
        native=dict(library=library_path.name, library_sha256=library_hash, loop_abi_version=1,
                    ids=ids, contract=native_contract), source=source("recurrence"), parameters=["steps", "x"],
        state_parameter="x", trip_count_parameter="steps",
        callee_guards=[dict(environment_path=["advance"], source=source("advance")),
                       dict(environment_path=["advance", "leaf"], source=source("leaf"))],
        array=dict(shape=[2, 3], dtype="float64", device="cpu", layout="C", alignment_bytes=8, final_bytes=48),
        trip_count=7, memory_budget_bytes=48,
        cpu=dict(ids=[cpu], threads=1, architecture="x86_64", os="linux", pointer_bits=64,
                 byteorder="little", required_isa="SSE2_X86_64"), external_assumptions=assumptions,
        floating_environment_policy="PRESERVE_CALLER_AND_WORKER_ENVIRONMENT_NOT_NUMPY_STATUS_FLAG_EQUIVALENCE",
        selection=dict(decision="TRANSFORM_UNDER_CONTRACT", selected=True,
            artifact={key:ids[key] for key in ("build_id", "loop_sha256", "strategy_sha256")},
            external_assumptions=assumptions))
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(contract, sort_keys=True))
    admission = dict(schema="scar.loop-runtime-admission", schema_version=1,
        runtime_contract_sha256=sha256(runtime_path.read_bytes()).hexdigest(), library_sha256=library_hash,
        native_ids=ids, producer=dict(name="SCAR", version="unit", exporter_sha256="8" * 64),
        evidence=[dict(role="unshipped-build-proof", path=str(tmp_path / "missing-proof.json"),
                       sha256="9" * 64, bytes=123)], trust_policy=TRUST_POLICY)
    admission_path = tmp_path / "admission.json"; admission_path.write_text(json.dumps(admission, sort_keys=True))
    pin = sha256(admission_path.read_bytes()).hexdigest()
    # An optional build report is deliberately not valid JSON and not trusted.
    (tmp_path / "export-report.json").write_text("not runtime input")
    calls, failure = [], [0]
    @native._Run
    def execute(state, count, output):
        calls.append(count)
        if failure[0]: return failure[0]
        ctypes.memmove(output.contents.data, state.contents.data, 48)
        return 0
    table = native._Table(1, ctypes.sizeof(native._Table), 7, execute,
        ids["loop_sha256"].encode(), ids["strategy_sha256"].encode(),
        ids["execution_plan_sha256"].encode(), ids["contract_sha256"].encode())
    class Getter:
        def __call__(self): return ctypes.pointer(table)
    class Library:
        scar_get_loop_table_v1 = Getter()
        keep_callback_alive = execute
    with monkeypatch.context() as context:
        context.setattr(native.ctypes, "CDLL", lambda path: Library())
        context.setattr(builtins, "__import__", minimal_import)
        artifact = native.load(tmp_path, expected_admission_sha256=pin)
        with pytest.raises(ValueError, match="trust pin"):
            native.load(tmp_path, expected_admission_sha256="0" * 64)
        table.trip_count = 6
        with pytest.raises(ValueError, match="ABI/count"):
            native.load(tmp_path, expected_admission_sha256=pin)
        table.trip_count = 7
        table.contract_sha256 = b"0" * 64
        with pytest.raises(ValueError, match="embedded identity"):
            native.load(tmp_path, expected_admission_sha256=pin)
        table.contract_sha256 = ids["contract_sha256"].encode()
        context.setattr(native.os, "sched_getaffinity", lambda _: set())
        with pytest.raises(native.GuardFailure, match="allocation unavailable"):
            native.load(tmp_path, expected_admission_sha256=pin)
    import numpy as np
    values = np.arange(6, dtype=np.float64).reshape(2, 3); values[0, 0] = -0.0
    output = artifact.run(values, 7)
    assert calls == [7] and output.flags.owndata and not np.shares_memory(output, values)
    np.testing.assert_array_equal(output.view(np.uint64), values.view(np.uint64))
    clone = artifact.contract; clone["trip_count"] = 1
    assert artifact.contract["trip_count"] == 7 and artifact.report["admission_sha256"] == pin
    for steps in (0, -1, 6, True, np.int64(7)):
        with pytest.raises(native.GuardFailure, match="trip count"):
            artifact.run(values, steps)
    for value in (values.astype(np.float32), values.T,
                  np.arange(12.).reshape(4, 3)[::2], np.frombuffer(bytes(48), dtype=np.float64).reshape(2, 3)):
        with pytest.raises(native.GuardFailure): artifact.run(value, 7)
    unaligned = np.ndarray((2, 3), dtype=np.float64, buffer=bytearray(49), offset=1)
    with pytest.raises(native.GuardFailure, match="aligned"):
        artifact.run(unaligned, 7)
    failure[0] = 3
    with pytest.raises(native.GuardFailure, match="floating-point"):
        artifact.run(values, 7)
    raw = runtime_path.read_bytes(); runtime_path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="byte SHA256"):
        native.load(tmp_path, expected_admission_sha256=pin)
    runtime_path.write_bytes(raw)
    raw = library_path.read_bytes(); library_path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="library byte SHA256"):
        native.load(tmp_path, expected_admission_sha256=pin)
