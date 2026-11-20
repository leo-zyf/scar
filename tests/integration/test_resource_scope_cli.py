"""Owned OS events attach to actual nested Python and native call boundaries."""
import shutil
import sys

import pytest

from scar.analysis.source_resource_scopes import ResourceScopeStore, build_resource_scopes
from scar.ir.program.resource_scopes import ObservedPythonResourceScope, ScopeEventKind
from test_call_operand_cli import cli


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("strace") is None,
                    reason="owned Linux syscall capture requires strace")
def test_nested_python_and_native_resource_scope(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    entry = project / "entry.py"
    entry.write_text(
        "import os, sys\n"
        "def publish(path):\n"
        "    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)\n"
        "    os.write(fd, b'result')\n"
        "    os.close(fd)\n"
        "publish(sys.argv[1])\n",
        encoding="utf-8",
    )
    capture = tmp_path / "capture"
    target = tmp_path / "data"
    run = cli("trace", "--project-root", project, "--call-operands", "--execution-evidence",
              "--resource-scopes", "--os-capture-root", capture, "--out", capture / "trace",
              "--", sys.executable, "-B", entry, target, cwd=project)
    assert run.returncode == 0, run.stdout + run.stderr
    assert target.read_bytes() == b"result"

    projection = tmp_path / "scopes"
    build_resource_scopes(capture, projection, project_root=str(project))
    store = ResourceScopeStore(projection)
    scopes = {scope.id: scope for scope in store.rows["scopes"]}
    python, = (scope for scope in scopes.values() if isinstance(scope, ObservedPythonResourceScope))
    assert python.end.kind is ScopeEventKind.PY_EXIT
    assert python.frame_entry.index < python.frame_exit.index
    assert python.normal_completion_proven is False
    write, = (row for row in store.rows["work"] if row.syscall == "write")
    native = scopes[write.scope]
    assert native.end.kind is ScopeEventKind.C_RETURN
    assert python.begin.sequence < native.begin.sequence < native.end.sequence < python.end.sequence
    assert write.result == 6
    assert write.direct_program_origin_proven is False
    assert all(row.syscall != "pwrite64" for row in store.rows["work"])
