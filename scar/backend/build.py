"""B3 verifies the C++20 source product and builds its declared CMake target."""

import fcntl
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4

from scar.contract import digest, document, open_product, validate_images, write_json
from scar.host import cpu_availability


def _image_outputs(package, supplied):
    images, root = package["images"], package["root_image"]
    validate_images(images, root)
    all_sources = []
    for image in images:
        folder = f"images/{image['id']}"
        header = folder + "/include/scar/program.hpp"
        sources = image.get("sources")
        if (
            image.get("directory") != folder
            or image.get("header") != header
            or not isinstance(sources, list)
            or not sources
            or any(
                not isinstance(s, str)
                or re.fullmatch(
                    re.escape(folder) + r"/src/(?:program|unit_[0-9]+)\.cpp", s
                )
                is None
                for s in sources
            )
            or len(set(sources)) != len(sources)
            or not {
                folder + "/CMakeLists.txt",
                folder + "/include/scar/runtime.hpp",
                header,
                *sources,
            }
            <= supplied
        ):
            raise ValueError(f"incomplete C++20 project for image {image['id']}")
        if image["id"] == root and package["header"] != header:
            raise ValueError("root C++20 image header mismatch")
        all_sources.extend(sources)
    if package["sources"] != all_sources:
        raise ValueError("C++20 image source ownership mismatch")
    return {image["id"]: image["output"] for image in images}


def _publish_images(root, installed, manifest, receipt):
    """Switch all binaries together; a failed rebuild retains the prior complete set."""
    link = root / "bin"
    release = root / (".scar-bin-" + uuid4().hex)
    switch = installed.parent / "publish-link"
    switch.symlink_to(release.name, target_is_directory=True)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        prior = os.readlink(link) if link.is_symlink() else None
        if (
            link.exists()
            and prior is None
            or prior is not None
            and not re.fullmatch(r"\.scar-bin-[0-9a-f]{32}", prior)
        ):
            raise ValueError(
                "existing bin directory is not a published program image set"
            )
        installed.rename(release)
        switched = False
        try:
            os.replace(switch, link)
            switched = True
            write_json(manifest, receipt)
        except BaseException:
            if switched:
                if prior is None:
                    link.unlink()
                else:
                    switch.symlink_to(prior, target_is_directory=True)
                    os.replace(switch, link)
            release.rename(installed)
            raise
        if prior is not None and (root / prior).is_dir():
            (root / prior).rename(installed.parent / "previous-binaries")
    finally:
        os.close(descriptor)


def build(manifest) -> Path:
    path, revision, package = open_product(manifest, "cpp20.json", "scar.cpp20-project")
    if package.get("kind") == "source_frame_library":
        raise ValueError(
            "a source-frame interface library needs a consuming native entry"
        )
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
    bundled = package.get("kind") == "image_bundle"
    names = (
        _image_outputs(package, supplied) if bundled else {"scar_entry": "scar_entry"}
    )
    root_image = package["root_image"] if bundled else "scar_entry"
    availability = cpu_availability()
    requested = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    if requested and (not requested.isdecimal() or int(requested) < 1):
        raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer")
    parallel = min(
        len(sources), availability.worker_budget, int(requested) if requested else 8
    )
    with TemporaryDirectory(prefix="scar-cmake-", dir=root.parent) as scratch:
        folder = Path(scratch)
        compile_command = ["cmake", "--build", str(folder / "build")]
        compile_command.extend(("--parallel", str(parallel)))
        configure_command = [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(folder / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        # Historical sealed projects predate the compiler-worker input. Their
        # CMake files must remain untouched and cannot consume this new setting.
        if "build_options" in package or any(
            "build_options" in image for image in package.get("images", ())
        ):
            configure_command.append(f"-DSCAR_COMPILER_WORKER_LIMIT={parallel}")
        commands = (
            configure_command,
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
            installed = folder / "install/bin"
            if not installed.is_dir() or {f.name for f in installed.iterdir()} != set(
                names.values()
            ):
                raise ValueError("CMake did not install every declared executable")
            executables = {}
            for image_id, output in names.items():
                binary = installed / output
                if (
                    binary.is_symlink()
                    or not binary.is_file()
                    or not os.access(binary, os.X_OK)
                ):
                    raise ValueError(f"CMake did not install executable {output!r}")
                data = binary.read_bytes()
                executables[image_id] = dict(
                    path=f"project/bin/{output}", size=len(data), sha256=digest(data)
                )
            fields = (
                dict(
                    kind="image_bundle", root_image=root_image, executables=executables
                )
                if bundled
                else {}
            )
            manifest = path.parent / "build.json"
            receipt = document(
                "scar.cpp20-executable",
                stage="B3",
                project_sha256=revision,
                executable=executables[root_image],
                warnings=0,
                build_workers=parallel,
                execution_host=availability.as_record(),
                **fields,
            )
            target = root / "bin" / names[root_image]
            if bundled:
                _publish_images(root, installed, manifest, receipt)
            else:
                target.parent.mkdir(exist_ok=True)
                os.replace(installed / names[root_image], target)
                write_json(manifest, receipt)
        finally:
            if log.exists():
                os.replace(log, root / "build.log")
    return target
