"""An ordinary from-import is attached after source execution, then restored."""
from hashlib import sha256
import os
from pathlib import Path
import sys
from types import CodeType, SimpleNamespace

import pytest

from scar.runtime import entry_runner, bundle_metadata, native_loop, loop_guard
from scar.trace.code_snapshot import code_key


def test_import_attachment_restores_state_preserves_rebind_and_noop(tmp_path, monkeypatch, capsys):
    import numpy as np
    module_name = "runner_source_" + tmp_path.name.replace("-", "_")
    helper_name = "runner_helper_" + tmp_path.name.replace("-", "_")
    source_path = tmp_path / (module_name + ".py")
    source_path.write_text("def body(state):\n    return state + 1.0\n\n"
                           "def evolve(state, steps):\n    for _ in range(steps):\n"
                           "        state = body(state)\n    return state\n")
    codes = {value.co_name:value for value in compile(source_path.read_text(), str(source_path), "exec").co_consts
             if isinstance(value, CodeType)}
    def source(name):
        return dict(path=str(source_path), sha256=sha256(source_path.read_bytes()).hexdigest(),
                    code_key=code_key(codes[name]), qualname=name)
    contract = dict(source=source("evolve"), callee_guards=[dict(environment_path=["body"], source=source("body"))],
        array=dict(shape=[1, 2]), trip_count=2, state_parameter="state", trip_count_parameter="steps",
        cpu=dict(ids=sorted(os.sched_getaffinity(0))[:1]))
    calls, loads = [], []
    artifact = SimpleNamespace(contract=contract, run=lambda state, steps: (calls.append(steps) or state + steps))
    # The runner fixture exercises the real live code/source/global guard; the
    # bundle/schema/ELF boundary is independently tested by native_loop tests.
    monkeypatch.setattr(loop_guard, "validate_contract", lambda value:value)
    monkeypatch.setattr(bundle_metadata, "inspect_bundle", lambda *args, **kwargs:dict(contract=contract))
    monkeypatch.setattr(native_loop, "load", lambda *args, **kwargs:(loads.append(1) or artifact))
    helper = tmp_path / (helper_name + ".py")
    helper.write_text(f"from {module_name} import evolve\nsaved = [evolve]\n")
    entry = tmp_path / "main.py"
    entry.write_text(f"import {helper_name} as helper\nfrom {module_name} import evolve\n"
        f"import {module_name} as source\nimport numpy as np\nimport sys\n"
        "state = np.array([[0.0, 1.0]])\n"
        "print(evolve(state, 2).tolist(), sys.argv[1:])\n"
        "print(helper.evolve(state, 3).tolist())\n"
        "def replacement(state, steps):\n    return state - 99.0\n"
        "source.evolve = replacement\n"
        "sys.path.append('entry-added-path')\nsys.meta_path.append(object())\n")
    old = (sys.argv, list(sys.argv), sys.path, list(sys.path), sys.meta_path, list(sys.meta_path))
    try:
        report = entry_runner.run_script(entry, ["--flag"], bundle="unused", expected_admission_sha256="a" * 64,
                                         exclusive_inputs=True)
        assert report["statistics"] == dict(native_calls=1, fallback_calls=1,
            reasons={"trip_count_outside_exact_specialization":1})
        assert report["installed_bindings"] == 1 and report["observed_boundary_calls"] == 2
        assert report["attachments"][0]["natural_import"] is True
        assert report["boundary_calls_observation_complete"] is False
        assert "[[2.0, 3.0]] ['--flag']" in capsys.readouterr().out
        source_module = sys.modules[module_name]; helper_module = sys.modules[helper_name]
        original = helper_module.evolve
        assert source_module.evolve.__name__ == "replacement"  # Do not overwrite an application's rebind.
        assert helper_module.evolve is not source_module.evolve
        before = len(calls)
        np.testing.assert_array_equal(helper_module.saved[0](np.zeros((1, 2)), 2), np.full((1, 2), 2.0))
        assert len(calls) == before  # Escaped proxy is inert after the scope.
        source_module.evolve = original
        assert sys.argv is old[0] and sys.argv == old[1]
        assert sys.path is old[2] and sys.path == old[3]
        assert sys.meta_path is old[4] and sys.meta_path == old[5]
        clean = tmp_path / "clean.py"; clean.write_text("print('original-only')\n")
        count = len(loads)
        disabled = entry_runner.run_script(clean, [], bundle="missing", expected_admission_sha256="bad",
                                           exclusive_inputs=True, native_enabled=False)
        assert disabled["decision"] == "NO_OP" and len(loads) == count
        implicit = entry_runner.run_script(clean, [], bundle="missing", expected_admission_sha256="bad")
        assert implicit["decision"] == "NO_OP" and len(loads) == count
        saved_path = contract["source"]["path"]; contract["source"]["path"] = str(clean)
        same_entry = entry_runner.run_script(clean, [], bundle="unused", expected_admission_sha256="a" * 64,
                                             exclusive_inputs=True)
        assert same_entry["decision"] == "NO_OP" and len(loads) == count
        contract["source"]["path"] = saved_path
        for code in (0, 7):
            exit_entry = tmp_path / "exit.py"; exit_entry.write_text(f"raise SystemExit({code})\n")
            if code == 0:
                assert entry_runner.run_original_script(exit_entry, [])["system_exit_zero"] is True
            else:
                with pytest.raises(SystemExit) as error:
                    entry_runner.run_original_script(exit_entry, [])
                assert error.value.code == 7
        failure = tmp_path / "failure.py"
        failure.write_text(f"from {module_name} import evolve\nraise RuntimeError('original exception')\n")
        with pytest.raises(RuntimeError, match="original exception"):
            entry_runner.run_script(failure, [], bundle="unused", expected_admission_sha256="a" * 64,
                                    exclusive_inputs=True)
        assert source_module.evolve is original
        assert sys.argv is old[0] and sys.path is old[2] and sys.meta_path is old[4]
    finally:
        sys.modules.pop(module_name, None); sys.modules.pop(helper_name, None)
