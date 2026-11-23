"""B3 verifies the C++20 source product and builds its declared CMake target."""

import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import digest, document, open_product, write_json


def build(manifest) -> Path:
    path, revision, package = open_product(manifest, "cpp20.json", "scar.cpp20-project")
    if package["project"] != "project":
        raise ValueError("invalid C++20 project location")
    root = (path.parent / package["project"]).resolve(strict=True)
    if root.parent != path.parent.resolve():
        raise ValueError("C++20 project directory escapes its package")
    supplied = set()
    for member in package["members"]:
        relative = Path(member["path"])
        source = (root / relative).resolve(strict=True)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not source.is_relative_to(root)
            or str(relative) in supplied
        ):
            raise ValueError(f"invalid C++20 project member {relative}")
        data = source.read_bytes()
        if len(data) != member["size"] or digest(data) != member["sha256"]:
            raise ValueError(f"C++20 project member changed: {relative}")
        supplied.add(str(relative))
    sources = package["sources"]
    if not sources or not {"CMakeLists.txt", package["header"], *sources} <= supplied:
        raise ValueError("incomplete C++20 project")
    with TemporaryDirectory(prefix="scar-cmake-", dir=root.parent) as scratch:
        folder = Path(scratch)
        compile_command = ["cmake", "--build", str(folder / "build")]
        if not os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"):
            parallel = min(len(sources), len(os.sched_getaffinity(0)), 8)
            compile_command.extend(("--parallel", str(parallel)))
        commands = (
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(folder / "build"),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            compile_command,
            [
                "cmake",
                "--install",
                str(folder / "build"),
                "--prefix",
                str(folder / "install"),
            ],
        )
        log = folder / "build.log"
        try:
            with log.open("w", encoding="utf-8") as out:
                for argv in commands:
                    result = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT)
                    if result.returncode:
                        out.flush()
                        raise ValueError(
                            f"C++20 project build failed\n{log.read_text(errors='replace')[-8000:]}"
                        )
            with log.open(errors="replace") as output:
                for line in output:
                    if re.search(
                        r"\bwarning(?:\s*#\d+)?\s*:|^CMake Warning\b", line, re.I
                    ):
                        raise ValueError(
                            f"C++20 project emitted a build warning: {line.strip()}"
                        )
            executable = folder / "install/bin/scar_entry"
            if not os.access(executable, os.X_OK):
                raise ValueError("CMake did not install the executable")
            target = root / "bin/scar_entry"
            target.parent.mkdir(exist_ok=True)
            os.replace(executable, target)
        finally:
            if log.exists():
                os.replace(log, root / "build.log")
    data = target.read_bytes()
    write_json(
        path.parent / "build.json",
        document(
            "scar.cpp20-executable",
            stage="B3",
            project_sha256=revision,
            executable=dict(
                path=str(target.relative_to(path.parent.resolve())),
                size=len(data),
                sha256=digest(data),
            ),
            warnings=0,
        ),
    )
    return target
