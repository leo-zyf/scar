"""A full original entry code can attach at its observed module STORE edge."""
import dis
from hashlib import sha256
import os
import sys
from types import CodeType, ModuleType, SimpleNamespace

import pytest

from scar.runtime import loop_guard, main_attachment
from scar.runtime.loop_contract import canonical_sha256
from scar.trace.code_snapshot import code_key


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="CPython local monitoring requires 3.12+")
def test_main_full_code_readiness_guards_and_cleanup(tmp_path, monkeypatch, capsys):
    import numpy as np
    entry = tmp_path / "main.py"
    holder_name = "main_attachment_holder_" + tmp_path.name.replace("-", "_")
    holder = ModuleType(holder_name)
    sys.modules[holder_name] = holder
    entry.write_text("from __future__ import annotations\nimport numpy as np\n"
        f"import {holder_name} as holder\n"
        "def body(state: NotEvaluated):\n    return state + 1.0\n"
        "def evolve(state, steps):\n    for _ in range(steps):\n"
        "        state = body(state)\n    return state\n"
        "holder.saved = [evolve]\nholder.namespace = globals()\n"
        "print(evolve(np.zeros((1, 2)), 2).tolist())\n"
        "print(evolve(np.zeros((1, 2)), 3).tolist())\n"
        "def replacement(state, steps):\n    return state - 90.0\n"
        "evolve = replacement\nraise SystemExit(0)\n")
    raw = entry.read_bytes()
    code = compile(raw, str(entry), "exec", flags=0, dont_inherit=True, optimize=sys.flags.optimize)
    codes = {child.co_name: child for child in code.co_consts if type(child) is CodeType}
    digest = sha256(raw).hexdigest()
    def source(name):
        return dict(path=str(entry), sha256=digest, code_key=code_key(codes[name]), qualname=name)
    contract = dict(source=source("evolve"),
        callee_guards=[dict(environment_path=["body"], source=source("body"))],
        array=dict(shape=[1, 2]), trip_count=2, state_parameter="state", trip_count_parameter="steps",
        cpu=dict(ids=sorted(os.sched_getaffinity(0))[:1]))
    instructions = tuple(dis.get_instructions(code, adaptive=False, show_caches=False))
    anchors = [dict(store_offset=before.offset, after_offset=after.offset, store_opname=before.opname,
        binding=before.argval, predecessor_execution_proven=False, guard_readiness_proven=False)
        for before, after in zip(instructions, instructions[1:])
        if before.opname in ("STORE_NAME", "STORE_GLOBAL") and before.argval in {"body", "evolve"}]
    plan = dict(schema="scar.entry-attachment-plan", schema_version=1, entry=str(entry),
        runtime_contract_sha256=canonical_sha256(contract),
        interpreter=dict(implementation=sys.implementation.name, version=list(sys.version_info[:3]),
                         cache_tag=sys.implementation.cache_tag), limits=dict(max_bytes=8 * 1024 * 1024),
        modules=[dict(path=str(entry), mode="ENTRY_MAIN_LOCAL_POST_STORE_CANDIDATES", attachment_target=True,
            frontiers=[], compile_recipe=dict(filename=str(entry), mode="exec", flags=0,
                dont_inherit=True, optimize=sys.flags.optimize), source_sha256=digest,
            module_code_key=code_key(code), required_global_names=["body", "evolve"], anchors=anchors)])
    calls = []
    artifact = SimpleNamespace(contract=contract, run=lambda state, steps: (calls.append(steps) or state + steps))
    monkeypatch.setattr(loop_guard, "validate_contract", lambda value: value)
    old = (sys.modules.get("__main__"), sys.argv, list(sys.argv), sys.path, list(sys.path),
           sys.meta_path, list(sys.meta_path), [sys.monitoring.get_tool(i) for i in range(6)])
    try:
        report = main_attachment.execute_main(entry, ["--unchanged"], artifact=artifact,
            attachment_plan=plan, exclusive_inputs=True)
        assert report["statistics"] == dict(native_calls=1, fallback_calls=1,
            reasons={"trip_count_outside_exact_specialization": 1})
        assert report["installed_bindings"] == 1 and report["observed_boundary_calls"] == 2
        assert report["attachments"] == [dict(source=str(entry), module="__main__", natural_import=False)]
        assert report["boundary_calls_observation_complete"] is False
        assert report["monitoring"]["observed_normal_store_edges"] == 2
        assert report["system_exit_zero"] is True
        assert calls == [2]
        assert capsys.readouterr().out == "[[2.0, 2.0]]\n[[3.0, 3.0]]\n"
        assert holder.namespace["evolve"].__name__ == "replacement"
        assert holder.namespace["body"].__annotations__ == {"state": "NotEvaluated"}
        np.testing.assert_array_equal(holder.saved[0](np.zeros((1, 2)), 2), np.full((1, 2), 2.0))
        assert calls == [2]  # Escaped proxy becomes original Python after cleanup.
        plan["modules"][0]["anchors"][0]["after_offset"] += 2
        rejected = main_attachment.execute_main(entry, [], artifact=artifact,
            attachment_plan=plan, exclusive_inputs=True)
        assert rejected["decision"] == "NO_OP" and calls == [2]
        assert "admission_rejected" in rejected["reason"]
        assert sys.modules.get("__main__") is old[0]
        assert sys.argv is old[1] and sys.argv == old[2]
        assert sys.path is old[3] and sys.path == old[4]
        assert sys.meta_path is old[5] and sys.meta_path == old[6]
        assert [sys.monitoring.get_tool(i) for i in range(6)] == old[7]
    finally:
        sys.modules.pop(holder_name, None)
