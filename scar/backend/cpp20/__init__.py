"""B2 produces a relocatable, formatted C++20 project from a scheduled execution IR."""

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import (
    PHYSICAL,
    digest,
    document,
    file_receipt,
    json_bytes,
    member,
    open_product,
    publish,
    read_blob,
    read_image_product,
    read_member,
    validate_images,
    write_json,
)
from scar.host import cpu_availability


def _format_workers(jobs, file_count):
    return max(1, min(jobs, 4, file_count, cpu_availability().worker_budget))


def _format_sources(formatter, files, jobs):
    files = sorted(files, key=lambda path: (-path.stat().st_size, str(path)))
    workers = _format_workers(jobs, len(files))
    command = [formatter, "--style=file", "-i"]
    if workers == 1:
        subprocess.run([*command, *files], check=True)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [
            pool.submit(subprocess.run, [*command, path], check=True) for path in files
        ]
        _, pending = wait(tasks, return_when=FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
    # All active writers have exited before an error can release the staging directory.
    for task in tasks:
        if not task.cancelled():
            task.result()


_STYLE = "BasedOnStyle: LLVM\nStandard: c++20\n"
_HEADER = "include/scar/program.hpp"
_METADATA = (
    "ir_sha256",
    "openmp",
    "runtime_libraries",
    "coordinate_maps",
    "record_field_offset_checks",
)

# The two pinned packages install headers without their source-archive LICENSE.
# Keep the shared BSD terms and both upstream notices with redistributed headers.
_XTENSOR_NOTICE = b"""BSD 3-Clause License

xtensor
Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht
Copyright (c) 2016, QuantStack

xtl
Copyright (c) 2017, Sylvain Corlay, Johan Mabille and Wolf Vollprecht
Copyright (c) 2017, QuantStack
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def _xtensor_headers(receipt):
    if (
        receipt.get("schema") != "scar.cpu-sdk"
        or receipt.get("schema_version") != 1
        or {k: v["version"] for k, v in receipt.get("packages", {}).items()}
        != {"xtensor": "0.27.1", "xtl": "0.8.0"}
    ):
        raise ValueError("incompatible installed operator SDK receipt")
    headers = {}
    for name, sha in receipt["files"].items():
        if not name.startswith("include/"):
            continue
        if not re.fullmatch(
            r"include/(?:xtensor\.hpp|(?:xtensor|xtl)/(?:[\w-]+/)*[\w-]+\.hpp)", name
        ) or not re.fullmatch("[a-f0-9]{64}", sha):
            raise ValueError(f"invalid xtensor CPU SDK header {name!r}")
        headers[name] = sha
    if (
        not {
            "include/xtensor/containers/xarray.hpp",
            "include/xtensor/containers/xadapt.hpp",
            "include/xtl/xtl_config.hpp",
        }
        <= headers.keys()
    ):
        raise ValueError("incomplete xtensor CPU SDK header closure")
    return headers


def _seal_xtensor(destination, binding):
    # Consume exactly the installation selected by B1. Its location is not
    # needed by B2b or by the delivered project's native build.
    if binding.get("provider") != "xtensor-cpu" or not isinstance(
        binding.get("prefix"), str
    ):
        raise ValueError("missing scheduled native operator SDK identity")
    prefix = Path(binding["prefix"]).resolve(strict=True)
    raw = (prefix / "scar-sdk.json").read_bytes()
    if digest(raw) != binding.get("receipt_sha256"):
        raise ValueError("installed operator SDK differs from its scheduled receipt")
    receipt = json.loads(raw)
    if receipt.get("packages") != binding.get("packages") or len(
        receipt.get("files", {})
    ) != binding.get("validated_files"):
        raise ValueError(
            "scheduled operator SDK package contract differs from its receipt"
        )

    def seal(data):
        sha = digest(data)
        path = f"operator-sdk/{sha}"
        publish(destination / path, data)
        return dict(path=path, sha256=sha, size=len(data))

    headers = {}
    for name, sha in _xtensor_headers(receipt).items():
        path = (prefix / name).resolve(strict=True)
        if not path.is_relative_to(prefix):
            raise ValueError(f"operator SDK header escapes its selected prefix: {name}")
        data = path.read_bytes()
        if digest(data) != sha:
            raise ValueError(f"operator SDK header changed since scheduling: {name}")
        headers[name] = seal(data)
    return dict(
        provider="xtensor-cpu",
        receipt=seal(raw),
        headers=headers,
        notice=seal(_XTENSOR_NOTICE),
    )


def _install_xtensor(folder, directory, binding):
    if not isinstance(binding, dict) or binding.get("provider") != "xtensor-cpu":
        raise ValueError(
            "target has no sealed XTensor SDK; regenerate it from its physical schedule"
        )
    headers = _xtensor_headers(read_member(folder, binding["receipt"]))
    members = binding["headers"]
    if headers.keys() != members.keys():
        raise ValueError(
            "target native SDK header set differs from the selected receipt"
        )
    vendor = directory / "vendor" / "xtensor-cpu"
    for name, sha in headers.items():
        if members[name]["sha256"] != sha:
            raise ValueError(
                f"target native SDK header differs from its selected receipt: {name}"
            )
        output = vendor / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(read_blob(folder, members[name]))
    (vendor / "LICENSE").write_bytes(read_blob(folder, binding["notice"]))


def _validate_mpi_sdk(receipt):
    if (
        not isinstance(receipt, dict)
        or receipt.get("provider") != "mpi-c"
        or receipt.get("api") != "c"
        or receipt.get("semantics_version") != 1
        or not re.fullmatch(r"[0-9]+\.[0-9]+", str(receipt.get("standard_version")))
        or not all(
            isinstance(receipt.get(k), list) and receipt[k]
            for k in ("libraries", "headers")
        )
    ):
        raise ValueError("target requires a qualified MPI C SDK receipt")
    for item in (*receipt["headers"], *receipt["libraries"]):
        if not isinstance(item, dict) or not re.fullmatch(
            r"[a-f0-9]{64}", str(item.get("sha256"))
        ):
            raise ValueError("invalid scheduled MPI SDK fingerprint")
    if not any(h.get("relative") == "mpi.h" for h in receipt["headers"]):
        raise ValueError("MPI SDK receipt omits the public header")
    for header in receipt["headers"]:
        index, relative = header.get("root_index"), header.get("relative")
        if (
            type(index) is not int
            or index < 0
            or not isinstance(relative, str)
            or not re.fullmatch(r"[A-Za-z0-9_+.-]+(?:/[A-Za-z0-9_+.-]+)*", relative)
            or any(part in {".", ".."} for part in relative.split("/"))
        ):
            raise ValueError("invalid scheduled MPI SDK header location")


def _mpi_checks(receipt):
    """Recheck FindMPI's effective include roots and ordered binary link inputs at build time."""
    lines = [
        f'if(NOT MPI_CXX_VERSION STREQUAL "{receipt["standard_version"]}")',
        '    message(FATAL_ERROR "MPI C API version differs from its physical schedule")',
        "endif()",
        "list(LENGTH MPI_CXX_LIBRARIES scar_mpi_libraries)",
        f'if(NOT scar_mpi_libraries EQUAL {len(receipt["libraries"])})',
        '    message(FATAL_ERROR "MPI link inputs differ from their physical schedule")',
        "endif()",
        "function(scar_verify_mpi_file path expected)",
        '    file(SHA256 "${path}" actual)',
        "    if(NOT actual STREQUAL expected)",
        '        message(FATAL_ERROR "MPI SDK fingerprint mismatch: ${path}")',
        "    endif()",
        "endfunction()",
    ]
    for header in receipt["headers"]:
        index, relative = header["root_index"], header["relative"]
        lines.extend(
            (
                f"list(GET MPI_CXX_INCLUDE_DIRS {index} scar_mpi_dir)",
                f'scar_verify_mpi_file("${{scar_mpi_dir}}/{relative}" "{header["sha256"]}")',
            )
        )
    for index, lib in enumerate(receipt["libraries"]):
        lines.extend(
            (
                f"list(GET MPI_CXX_LIBRARIES {index} scar_mpi_library)",
                f'scar_verify_mpi_file("${{scar_mpi_library}}" "{lib["sha256"]}")',
            )
        )
    return lines


def _package(root, sources, openmp, libraries, jobs, image=None, mpi=None):
    unknown = libraries - {
        "MPI",
        "Threads",
        "DynamicLoader",
        "libgfortran.so.5",
        "XTensor",
    }
    if unknown:
        raise ValueError(
            f"unsupported C++ project link requirements: {sorted(unknown)}"
        )
    formatter = shutil.which("clang-format-18") or shutil.which("clang-format")
    if formatter is None:
        raise ValueError("clang-format is required for LLVM-style C++20 delivery")
    if image is None:
        (root / ".clang-format").write_text(_STYLE, encoding="utf-8")
    generated = [root / name for name in sources]
    generated.extend(sorted((root / "include").rglob("*.hpp")))
    _format_sources(formatter, generated, jobs)
    target = f"scar_{image['id']}" if image else "scar_entry"
    lines = (
        []
        if image
        else [
            "cmake_minimum_required(VERSION 3.20)",
            "project(scar_delivery LANGUAGES CXX)",
        ]
    )
    lines += [
        f"add_executable({target}",
        *(f"    {source}" for source in sources),
        ")",
        f"target_include_directories({target} PRIVATE include)",
        f"set_target_properties({target} PROPERTIES CXX_STANDARD 20",
        "    CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)",
        'if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")',
        f"    target_compile_options({target} PRIVATE -Werror -fno-fast-math -frounding-math -ffp-contract=off)",
        'elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")',
        f"    target_compile_options({target} PRIVATE -Werror -Kieee -Mnofma)",
        "endif()",
    ]
    if "MPI" in libraries:
        lines.extend(
            ("set(MPI_ASSUME_NO_BUILTIN_MPI TRUE)", "set(MPI_CXX_SKIP_MPICXX TRUE)")
        )
    for enabled, package in ((openmp, "OpenMP"), ("MPI" in libraries, "MPI")):
        if enabled:
            lines.extend(
                (
                    f"find_package({package} REQUIRED COMPONENTS CXX)",
                    f"target_link_libraries({target} PRIVATE {package}::{package}_CXX)",
                )
            )
    if "MPI" in libraries:
        lines.extend(_mpi_checks(mpi))
        lines.append(
            f"target_compile_definitions({target} PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)"
        )
    if "Threads" in libraries:
        lines.extend(
            (
                "find_package(Threads REQUIRED)",
                f"target_link_libraries({target} PRIVATE Threads::Threads)",
            )
        )
    if "DynamicLoader" in libraries:
        lines.append(f'target_link_libraries({target} PRIVATE "${{CMAKE_DL_LIBS}}")')
    if "libgfortran.so.5" in libraries:
        lines.extend(
            (
                "find_library(SCAR_GNU_IO_LIBRARY NAMES libgfortran.so.5 REQUIRED)",
                f'target_link_libraries({target} PRIVATE "${{SCAR_GNU_IO_LIBRARY}}")',
            )
        )
    if "XTensor" in libraries:
        lines.append(
            f'target_include_directories({target} SYSTEM PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/vendor/xtensor-cpu/include")'
        )
    if image:
        lines.append(
            f'set_target_properties({target} PROPERTIES OUTPUT_NAME "{image["output"]}")'
        )
    lines.append(f"install(TARGETS {target} RUNTIME DESTINATION bin)")
    (root / "CMakeLists.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lower_image(path, revision, scheduled, destination):
    from .lowering import Lowerer
    from .model import encode_unit

    if scheduled.get("status") != "READY" or scheduled.get("kind") == "image_bundle":
        raise ValueError("expected a ready single-image physical schedule")
    physical = read_member(path.parent, scheduled["physical"])
    if (
        physical.get("schema"),
        physical.get("schema_version"),
        physical.get("ir_sha256"),
    ) != (
        "scar.physical-plan",
        scheduled["schema_version"],
        scheduled["ir_sha256"],
    ) or physical.get(
        "status", "READY"
    ) != "READY":
        raise ValueError("physical plan does not belong to its scheduled IR package")
    module = dict(
        scheduled["module"],
        functions=[read_member(path.parent, f) for f in scheduled["functions"]],
    )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lowerer = Lowerer(module, physical)
    units = lowerer.units()
    members = [
        dict(output=name, tree=member(destination, "target-units", encode_unit(unit)))
        for name, unit in units.items()
    ]
    support = Path(__file__).with_name("runtime.hpp").read_bytes()
    runtime = destination / "runtime" / (digest(support) + ".hpp")
    publish(runtime, support)
    headers = []
    lowerer.used_sdk_headers.add("exceptions.hpp")
    if "format.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.add("array.hpp")
    if "array_io.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.update({"array.hpp", "host.hpp", "text.hpp"})
    if "host.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.add("exceptions.hpp")
    for name in sorted(lowerer.used_sdk_headers):
        if name not in {
            "array.hpp",
            "array_io.hpp",
            "host.hpp",
            "exceptions.hpp",
            "format.hpp",
            "text.hpp",
            "sequence.hpp",
        }:
            raise ValueError(f"unregistered target SDK header {name!r}")
        data = Path(__file__).with_name(name).read_bytes()
        sealed = destination / "runtime" / (digest(data) + ".hpp")
        publish(sealed, data)
        headers.append(
            dict(output="include/scar/" + name, blob=file_receipt(destination, sealed))
        )
    operator_sdks = {}
    mpi = physical.get("operator_backends", {}).get("mpi")
    if mpi is not None or "MPI" in lowerer.libraries.links:
        _validate_mpi_sdk(mpi)
        lowerer.libraries.links.add("MPI")
        operator_sdks["MPI"] = member(destination, "operator-sdk", mpi)
    metadata = dict(
        openmp=lowerer.openmp,
        runtime_libraries=sorted(lowerer.libraries.links),
        coordinate_maps=lowerer.coordinate_maps,
        record_field_offset_checks=lowerer.layout_scope,
    )
    if "XTensor" in lowerer.libraries.links:
        operator_sdks["XTensor"] = _seal_xtensor(
            destination, physical["operator_backends"]["ndarray"]
        )
    if operator_sdks:
        metadata["operator_sdks"] = operator_sdks
    product = document(
        "scar.cpp20-target-ir",
        stage="B2a",
        scheduled_ir_sha256=revision,
        units=members,
        ir_sha256=physical.get("ir_sha256"),
        runtime_sha256=digest(support),
        runtime=file_receipt(destination, runtime),
        sdk_headers=headers,
        **metadata,
    )
    path = destination / "target-ir.json"
    write_json(path, product)
    return path


def _check_launch_targets(folder, plans):
    workers, launches = {}, []
    for image_id, plan in plans.items():
        _, _, physical = read_image_product(
            folder, plan, "physical", "physical-plan.json", PHYSICAL
        )
        bindings = physical.get("operator_backends", {})
        sdk = bindings.get("mpi")
        if isinstance(sdk, dict) and physical.get("status") == "READY":
            workers[image_id] = (
                plan["output"],
                physical["ir_sha256"],
                digest(json_bytes(sdk)),
            )
        launches.extend(bindings.get("process", {}).get("launches", {}).items())
    for site, launch in launches:
        expected = workers.get(launch["image_id"])
        actual = tuple(
            launch.get(key)
            for key in (
                "worker_output",
                "worker_ir_sha256",
                "worker_mpi_receipt_sha256",
            )
        )
        if actual != expected:
            raise ValueError(
                f"MPI launch {site!r} differs from the sealed worker physical plan"
            )


def lower_scheduled(source, destination):
    """Lower each link image in its own symbol namespace."""
    path, revision, scheduled = open_product(
        source, "scheduled-ir.json", "scar.scheduled-ir"
    )
    if scheduled.get("kind") != "image_bundle":
        return _lower_image(path, revision, scheduled, destination)
    if scheduled.get("status") != "READY":
        raise ValueError("incomplete physical execution schedule")
    images, root = scheduled["images"], scheduled["root_image"]
    validate_images(images, root)
    physical = read_member(path.parent, scheduled["physical"])
    if (
        physical.get("schema") != PHYSICAL
        or physical.get("schema_version") != scheduled["schema_version"]
        or physical.get("ir_sha256") != scheduled["ir_sha256"]
        or physical.get("kind") != "image_bundle"
        or physical.get("status") != "READY"
        or physical.get("root_image") != root
        or physical.get("planning_scope") != "per_image"
    ):
        raise ValueError("physical plan does not belong to the scheduled image bundle")
    validate_images(physical.get("images"), root)
    plans = {plan["id"]: plan for plan in physical["images"]}
    if set(plans) != {image["id"] for image in images}:
        raise ValueError("physical plan image set differs from the schedule")
    _check_launch_targets(path.parent, plans)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = []
    for image in images:
        child, child_hash, data = read_image_product(
            path.parent, image, "scheduled", "scheduled-ir.json", "scar.scheduled-ir"
        )
        if data.get("module", {}).get("entry") != image["entry"]:
            raise ValueError(f"physical image {image['id']} entry mismatch")
        plan = plans[image["id"]]
        if (plan["entry"], plan["output"]) != (image["entry"], image["output"]):
            raise ValueError(f"physical image {image['id']} identity mismatch")
        declared, selected = plan.get("physical", {}), data.get("physical", {})
        if any(declared.get(key) != selected.get(key) for key in ("sha256", "size")):
            raise ValueError(f"physical image {image['id']} receipt mismatch")
        compiled = _lower_image(
            child, child_hash, data, destination / "images" / image["id"]
        )
        targets.append(
            dict(
                id=image["id"],
                entry=image["entry"],
                output=image["output"],
                target=file_receipt(destination, compiled),
            )
        )
    target = destination / "target-ir.json"
    write_json(
        target,
        document(
            "scar.cpp20-target-ir",
            stage="B2a",
            kind="image_bundle",
            root_image=root,
            images=targets,
            ir_sha256=scheduled["ir_sha256"],
            scheduled_ir_sha256=revision,
        ),
    )
    return target


def _print_image(path, product, directory, runtime, jobs, image=None):
    from .model import from_data
    from .printer import print_unit

    if "runtime" in product:
        runtime = read_blob(path.parent, product["runtime"])
    elif runtime is None:
        runtime = Path(__file__).with_name("runtime.hpp").read_bytes()
    if product.get("kind") == "image_bundle" or digest(runtime) != product.get(
        "runtime_sha256"
    ):
        raise ValueError("target image requires a matching C++20 runtime revision")
    (directory / "include/scar").mkdir(parents=True)
    (directory / "src").mkdir()
    (directory / "include/scar/runtime.hpp").write_bytes(runtime)
    for header in product.get("sdk_headers", ()):
        if header["output"] not in {
            "include/scar/array.hpp",
            "include/scar/array_io.hpp",
            "include/scar/host.hpp",
            "include/scar/exceptions.hpp",
            "include/scar/format.hpp",
            "include/scar/text.hpp",
            "include/scar/sequence.hpp",
        }:
            raise ValueError("invalid target SDK header output")
        target = directory / header["output"]
        if target.exists():
            raise ValueError("duplicate target SDK header")
        target.write_bytes(read_blob(path.parent, header["blob"]))
    sources = []
    local_headers, sdk_headers = set(), set()
    for part in product["units"]:
        output = part["output"]
        if output != _HEADER and not re.fullmatch(
            r"src/(program|unit_[0-9]+)\.cpp", output
        ):
            raise ValueError(f"invalid C++ translation unit path {output!r}")
        file = directory / output
        if file.exists():
            raise ValueError(f"duplicate C++ translation unit {output!r}")
        tree = from_data(read_member(path.parent, part["tree"]))
        local_headers.update(name for kind, name in tree.includes if kind == "local")
        sdk_headers.update(name for kind, name in tree.includes if kind == "system")
        file.write_text(print_unit(tree), encoding="utf-8")
        if output != _HEADER:
            sources.append(output)
    if not sources or not (directory / _HEADER).is_file():
        raise ValueError("incomplete target translation-unit set")
    include_root = (directory / "include").resolve()
    for name in local_headers:
        header = (include_root / name).resolve()
        if not header.is_relative_to(include_root) or not header.is_file():
            raise ValueError(f"target package is missing local header {name!r}")
    libraries = set(product["runtime_libraries"])
    if sdk_headers & {"mpi.h", "mpio.h"} and "MPI" not in libraries:
        raise ValueError("target MPI header requires a qualified MPI SDK")
    bindings = product.get("operator_sdks", {})
    if not isinstance(bindings, dict) or bindings.keys() - (
        libraries & {"XTensor", "MPI"}
    ):
        raise ValueError("target declares an unused or unsupported native operator SDK")
    if (
        local_headers & {"scar/array.hpp", "scar/array_io.hpp", "scar/format.hpp"}
        and "XTensor" not in libraries
    ):
        raise ValueError("target array header requires a sealed XTensor SDK")
    if "XTensor" in libraries:
        _install_xtensor(path.parent, directory, bindings.get("XTensor"))
    mpi = None
    if "MPI" in libraries:
        if "MPI" not in bindings:
            raise ValueError(
                "target lacks a qualified MPI SDK; regenerate it from its execution IR"
            )
        mpi = read_member(path.parent, bindings["MPI"])
        _validate_mpi_sdk(mpi)
        write_json(directory / "mpi-sdk.json", mpi)
    _package(directory, sources, product["openmp"], libraries, jobs, image, mpi)
    return sources


def print_target(target, destination, *, jobs=4):
    """Print and atomically publish one CMake project for all declared images."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError("formatting jobs must be a positive integer")
    path, revision, product = open_product(
        target, "target-ir.json", "scar.cpp20-target-ir"
    )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    project = destination / "project"
    if project.exists():
        raise ValueError(f"output project already exists at {project}")
    runtime = None
    manifest = destination / "cpp20.json"
    with TemporaryDirectory(prefix=".scar-project-", dir=destination) as tmp:
        staging = Path(tmp) / "project"
        if product.get("kind") == "image_bundle":
            root, images = product["root_image"], product["images"]
            validate_images(images, root)
            staging.mkdir()
            (staging / ".clang-format").write_text(_STYLE, encoding="utf-8")
            cmake = [
                "cmake_minimum_required(VERSION 3.20)",
                "project(scar_delivery LANGUAGES CXX)",
            ]
            projects = []
            for image in images:
                child, child_hash, data = read_image_product(
                    path.parent,
                    image,
                    "target",
                    "target-ir.json",
                    "scar.cpp20-target-ir",
                )
                directory = Path("images") / image["id"]
                sources = _print_image(
                    child, data, staging / directory, runtime, jobs, image
                )
                cmake.append(f"add_subdirectory({directory.as_posix()})")
                projects.append(
                    dict(
                        id=image["id"],
                        entry=image["entry"],
                        output=image["output"],
                        directory=directory.as_posix(),
                        header=(directory / _HEADER).as_posix(),
                        sources=[(directory / s).as_posix() for s in sources],
                        target_ir_sha256=child_hash,
                    )
                )
            (staging / "CMakeLists.txt").write_text(
                "\n".join(cmake) + "\n", encoding="utf-8"
            )
            root_project = next(p for p in projects if p["id"] == root)
            header, first_source = root_project["header"], root_project["sources"][0]
            sources = [s for p in projects for s in p["sources"]]
            metadata = dict(
                kind="image_bundle",
                root_image=root,
                images=projects,
                ir_sha256=product["ir_sha256"],
            )
        else:
            sources = _print_image(path, product, staging, runtime, jobs)
            header, first_source = _HEADER, sources[0]
            metadata = {key: product[key] for key in _METADATA}
        members = [
            file_receipt(staging, f) for f in sorted(staging.rglob("*")) if f.is_file()
        ]
        receipt = document(
            "scar.cpp20-project",
            stage="B2b",
            target_ir_sha256=revision,
            project="project",
            sources=sources,
            header=header,
            members=members,
            **metadata,
        )
        staging.rename(project)
        try:
            write_json(manifest, receipt)
        except BaseException:
            project.rename(staging)
            raise
    return dict(
        target_ir=str(path),
        cpp20_manifest=str(manifest),
        project=str(project),
        source=str(project / first_source),
        sources=[str(project / s) for s in sources],
        header=str(project / header),
        **metadata,
    )
