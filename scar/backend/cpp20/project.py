"""B2b: print a typed target package and seal its formatted C++20 project."""

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import (
    digest,
    document,
    file_receipt,
    json_bytes,
    open_product,
    read_blob,
    read_image_product,
    read_member,
    validate_images,
    write_json,
)
from scar.host import cpu_availability
from .receipts import (
    SDK_HEADERS,
    gmp_files,
    is_digest,
    pocketfft_files,
    validate_mpi_sdk,
    xtensor_headers,
)

_STYLE = "BasedOnStyle: LLVM\nStandard: c++20\n"

_HEADER = "include/scar/program.hpp"

_CMAKE_PROJECT = (
    "cmake_minimum_required(VERSION 3.20)",
    "project(scar_delivery LANGUAGES CXX)",
    'if(DEFINED SCAR_COMPILER_WORKER_LIMIT AND NOT "${SCAR_COMPILER_WORKER_LIMIT}" MATCHES "^[1-9][0-9]*$")',
    '    message(FATAL_ERROR "SCAR_COMPILER_WORKER_LIMIT must be a positive integer")',
    "endif()",
)

_METADATA = (
    "ir_sha256",
    "build_options",
    "openmp",
    "runtime_libraries",
    "coordinate_maps",
    "record_field_offset_checks",
)


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


def _install_xtensor(folder, directory, binding):
    if not isinstance(binding, dict) or binding.get("provider") != "xtensor-cpu":
        raise ValueError(
            "target has no sealed XTensor SDK; regenerate it from its physical schedule"
        )
    headers = xtensor_headers(read_member(folder, binding["receipt"]))
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


def _install_sdk_files(folder, vendor, files):
    for name, receipt in files.items():
        output = vendor / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(read_blob(folder, receipt))


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


def _package(
    root,
    sources,
    openmp,
    libraries,
    jobs,
    options,
    image=None,
    mpi=None,
    numeric_compiler=None,
):
    unknown = libraries - {
        "MPI",
        "Threads",
        "DynamicLoader",
        "libgfortran.so.5",
        "XTensor",
        "PocketFFT",
        "GMP",
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
    objects = f"{target}.objects"
    lines = [] if image else list(_CMAKE_PROJECT)
    lines += [
        f"add_library({objects} OBJECT",
        *(f"    {source}" for source in sources),
        ")",
        f"add_executable({target})",
        f"target_link_libraries({target} PRIVATE {objects})",
        f"target_include_directories({objects} PRIVATE include)",
        f"set_target_properties({objects} {target} PROPERTIES CXX_STANDARD 20",
        "    CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)",
        'if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")',
        f"    target_compile_options({objects} PRIVATE -Werror -fno-fast-math -frounding-math -ffp-contract=off)",
        'elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")',
        f"    target_compile_options({objects} PRIVATE -Werror -Kieee -Mnofma)",
        "endif()",
    ]
    if numeric_compiler:
        lines.extend(
            (
                'file(SHA256 "${CMAKE_CXX_COMPILER}" scar_numeric_compiler_sha256)',
                f'if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR NOT scar_numeric_compiler_sha256 STREQUAL "{numeric_compiler}")',
                '    message(FATAL_ERROR "Function numeric options require their qualified GNU C++ compiler")',
                "endif()",
            )
        )
    interprocedural = options["interprocedural"]
    if interprocedural["mode"] == "gnu_lto":
        lines.extend(
            (
                'if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")',
                '    message(FATAL_ERROR "This target requires GNU C++ link-time optimization")',
                "endif()",
                f"set(scar_lto_workers {interprocedural['workers']})",
                "if(DEFINED SCAR_COMPILER_WORKER_LIMIT AND SCAR_COMPILER_WORKER_LIMIT LESS scar_lto_workers)",
                '    set(scar_lto_workers "${SCAR_COMPILER_WORKER_LIMIT}")',
                "endif()",
                f'target_compile_options({objects} PRIVATE "-flto=${{scar_lto_workers}}" -fno-fat-lto-objects)',
                f'target_link_options({target} PRIVATE "-flto=${{scar_lto_workers}}")',
                "if(scar_lto_workers EQUAL 1)",
                f"    target_link_options({target} PRIVATE -flto-partition=one)",
                "endif()",
            )
        )
    if "MPI" in libraries:
        lines.extend(
            ("set(MPI_ASSUME_NO_BUILTIN_MPI TRUE)", "set(MPI_CXX_SKIP_MPICXX TRUE)")
        )
    for enabled, package in ((openmp, "OpenMP"), ("MPI" in libraries, "MPI")):
        if enabled:
            lines.extend(
                (
                    f"find_package({package} REQUIRED COMPONENTS CXX)",
                    f"target_link_libraries({objects} PUBLIC {package}::{package}_CXX)",
                )
            )
    if "MPI" in libraries:
        lines.extend(_mpi_checks(mpi))
        lines.append(
            f"target_compile_definitions({objects} PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)"
        )
    if "Threads" in libraries:
        lines.extend(
            (
                "find_package(Threads REQUIRED)",
                f"target_link_libraries({objects} PUBLIC Threads::Threads)",
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
            f'target_include_directories({objects} SYSTEM PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/vendor/xtensor-cpu/include")'
        )
    if "PocketFFT" in libraries:
        lines.append(
            f'target_include_directories({objects} SYSTEM PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/vendor/pocketfft/include")'
        )
    if "GMP" in libraries:
        lines.extend(
            (
                f'target_include_directories({objects} SYSTEM PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/vendor/gmp/include")',
                f'target_link_libraries({target} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/vendor/gmp/lib/libgmp.a")',
            )
        )
    if image:
        lines.append(
            f'set_target_properties({target} PROPERTIES OUTPUT_NAME "{image["output"]}")'
        )
    lines.append(f"install(TARGETS {target} RUNTIME DESTINATION bin)")
    (root / "CMakeLists.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _numeric_target_compiler(product, actual):
    """Verify the target-only receipt against the definitions being printed."""
    audit = product.get("numeric_permission_consumption")
    if audit is None and not actual:
        return None
    if (
        not isinstance(audit, dict)
        or type(audit.get("semantics_version")) is not int
        or audit["semantics_version"] != 1
        or audit.get("scope") != "function_definitions"
        or not is_digest(audit.get("binding_sha256"))
        or not isinstance(audit.get("functions"), dict)
        or not audit["functions"]
    ):
        raise ValueError("missing or invalid target numeric permission receipt")
    expected = {}
    for row in audit["functions"].values():
        if not isinstance(row, dict) or not is_digest(
            row.get("scheduled_function_sha256")
        ):
            raise ValueError("invalid target numeric function member")
        if row.get("options") == [] and "target_signature_sha256" not in row:
            continue
        sha = row.get("target_signature_sha256")
        if not is_digest(sha) or sha in expected:
            raise ValueError("invalid or duplicate numeric target definition")
        expected[sha] = row.get("options")
    compiler = audit.get("compiler_sha256")
    if expected != actual or (
        not is_digest(compiler) if actual else compiler is not None
    ):
        raise ValueError(
            "numeric permissions do not match the printed target definitions"
        )
    return compiler


def _print_image(path, product, directory, jobs, image=None):
    from .model import (
        build_options,
        from_data,
        numeric_definition,
        validate_build_options,
    )
    from .printer import print_unit

    options = (
        validate_build_options(product["build_options"])
        if "build_options" in product
        else build_options()
    )
    if "runtime" in product:
        runtime = read_blob(path.parent, product["runtime"])
        runtime_sha = product["runtime"]["sha256"]
    else:
        runtime = Path(__file__).with_name("runtime.hpp").read_bytes()
        runtime_sha = digest(runtime)
    if product.get("kind") == "image_bundle" or runtime_sha != product.get(
        "runtime_sha256"
    ):
        raise ValueError("target image requires a matching C++20 runtime revision")
    (directory / "include/scar").mkdir(parents=True)
    (directory / "src").mkdir()
    (directory / "include/scar/runtime.hpp").write_bytes(runtime)
    for header in product.get("sdk_headers", ()):
        if header["output"] not in SDK_HEADERS.values():
            raise ValueError("invalid target SDK header output")
        target = directory / header["output"]
        if target.exists():
            raise ValueError("duplicate target SDK header")
        target.write_bytes(read_blob(path.parent, header["blob"]))
    sources = []
    numeric_definitions = {}
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
        declarations = list(tree.declarations)
        while declarations:
            decl = declarations.pop()
            declarations.extend(decl.members)
            if decl.attributes and (signature := numeric_definition(decl)):
                sha = digest(json_bytes(signature))
                if sha in numeric_definitions:
                    raise ValueError("duplicate attributed numeric definition")
                numeric_definitions[sha] = signature["options"]
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
        libraries & {"XTensor", "MPI", "PocketFFT", "GMP"}
    ):
        raise ValueError("target declares an unused or unsupported native operator SDK")
    if (
        local_headers
        & {"scar/array.hpp", "scar/array_io.hpp", "scar/format.hpp", "scar/fourier.hpp"}
        and "XTensor" not in libraries
    ):
        raise ValueError("target array header requires a sealed XTensor SDK")
    if "XTensor" in libraries:
        _install_xtensor(path.parent, directory, bindings.get("XTensor"))
    if "scar/fourier.hpp" in local_headers and "PocketFFT" not in libraries:
        raise ValueError("target Fourier header requires a sealed PocketFFT SDK")
    if "PocketFFT" in libraries:
        _install_sdk_files(
            path.parent,
            directory / "vendor/pocketfft",
            pocketfft_files(bindings.get("PocketFFT")),
        )
    if "scar/integer.hpp" in local_headers and "GMP" not in libraries:
        raise ValueError("target integer header requires a sealed GMP SDK")
    if "GMP" in libraries:
        _install_sdk_files(
            path.parent, directory / "vendor/gmp", gmp_files(bindings.get("GMP"))
        )
    mpi = None
    if "MPI" in libraries:
        if "MPI" not in bindings:
            raise ValueError(
                "target lacks a qualified MPI SDK; regenerate it from its execution IR"
            )
        mpi = read_member(path.parent, bindings["MPI"])
        validate_mpi_sdk(mpi)
        write_json(directory / "mpi-sdk.json", mpi)
    compiler = _numeric_target_compiler(product, numeric_definitions)
    _package(
        directory,
        sources,
        product["openmp"],
        libraries,
        jobs,
        options,
        image,
        mpi,
        numeric_compiler=compiler,
    )
    return sources, options


def _math_product_metadata(product):
    """Carry the target-local numerical receipts after verifying their C++ product."""
    metadata = (
        {"numeric_permission_consumption": product["numeric_permission_consumption"]}
        if "numeric_permission_consumption" in product
        else {}
    )
    key = "math_interface_consumption"
    if key not in product:
        return metadata
    audit = product[key]
    fields = {
        "semantics_version",
        "scope",
        "binding_sha256",
        "selected",
        "vector_eligibility",
        "functions",
        "sites",
        "interfaces",
    }
    if not isinstance(audit, dict) or set(audit) != fields:
        raise ValueError("invalid target math consumption receipt")
    by_interface = audit.get("interfaces")
    if (
        type(audit.get("semantics_version")) is not int
        or audit["semantics_version"] != 1
        or audit.get("scope") != "scheduled_operations"
        or (audit.get("selected"), audit.get("vector_eligibility"))
        != ("preserve_operation", "not_qualified")
        or not isinstance(audit.get("binding_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", audit["binding_sha256"])
        or type(audit.get("functions")) is not int
        or type(audit.get("sites")) is not int
        or not 1 <= audit["functions"] <= audit["sites"]
        or not isinstance(by_interface, dict)
        or not all(
            isinstance(k, str) and k and type(v) is int and v > 0
            for k, v in by_interface.items()
        )
        or sum(by_interface.values()) != audit["sites"]
    ):
        raise ValueError("invalid or unsupported target math consumption receipt")
    return {**metadata, key: audit}


def _print_frame_library(path, product, staging, jobs):
    from .model import unit_from_data
    from .printer import print_unit

    if len(product["units"]) != 1 or product["units"][0]["output"] != "regions.hpp":
        raise ValueError("invalid source-frame library unit table")
    unit = unit_from_data(read_member(path.parent, product["units"][0]["tree"]))
    if not unit.once or unit.includes != (("system", "array"), ("local", "import.hpp")):
        raise ValueError("invalid source-frame library includes")
    staging.mkdir()
    (staging / ".clang-format").write_text(_STYLE, encoding="utf-8")
    (staging / "regions.hpp").write_text(print_unit(unit), encoding="utf-8")
    (staging / "import.hpp").write_bytes(read_blob(path.parent, product["runtime"]))
    formatter = shutil.which("clang-format-18") or shutil.which("clang-format")
    if not formatter:
        raise ValueError("clang-format is required for LLVM-style C++20 delivery")
    _format_sources(formatter, [staging / "regions.hpp", staging / "import.hpp"], jobs)
    (staging / "CMakeLists.txt").write_text(
        "\n".join(
            (
                *_CMAKE_PROJECT,
                "add_library(scar_source_frames INTERFACE)",
                'target_include_directories(scar_source_frames INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")',
                "target_compile_features(scar_source_frames INTERFACE cxx_std_20)",
                "",
            )
        ),
        encoding="utf-8",
    )


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
    manifest = destination / "cpp20.json"
    with TemporaryDirectory(prefix=".scar-project-", dir=destination) as tmp:
        staging = Path(tmp) / "project"
        if product.get("kind") == "source_frame_library":
            _print_frame_library(path, product, staging, jobs)
            sources, header, first_source = [], "regions.hpp", "regions.hpp"
            metadata = dict(
                kind="source_frame_library",
                sites=product["sites"],
                ir_sha256=product["ir_sha256"],
            )
        elif product.get("kind") == "image_bundle":
            root, images = product["root_image"], product["images"]
            validate_images(images, root)
            staging.mkdir()
            (staging / ".clang-format").write_text(_STYLE, encoding="utf-8")
            cmake = list(_CMAKE_PROJECT)
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
                sources, options = _print_image(
                    child, data, staging / directory, jobs, image
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
                        build_options=options,
                        target_ir_sha256=child_hash,
                        **_math_product_metadata(data),
                    )
                )
            # Link processes may each consume the entire compiler budget for LTO.
            # Keep all object compilations parallel, then order the image links.
            targets = [f"scar_{image['id']}" for image in images]
            cmake.extend(
                [
                    "add_custom_target(scar.objects.ready DEPENDS",
                    *(f"    $<TARGET_OBJECTS:{t}.objects>" for t in targets),
                    ")",
                    "add_dependencies(scar.objects.ready "
                    + " ".join(f"{t}.objects" for t in targets)
                    + ")",
                ]
            )
            previous = "scar.objects.ready"
            for target in targets:
                cmake.append(f"add_dependencies({target} {previous})")
                previous = target
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
            sources, options = _print_image(path, product, staging, jobs)
            header, first_source = _HEADER, sources[0]
            image = dict(product, build_options=options)
            metadata = {key: image[key] for key in _METADATA}
            metadata.update(_math_product_metadata(product))
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
