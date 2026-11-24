"""B1 qualifies the C ABI of the selected native MPI installation."""

from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import digest, walk_operations

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INTERFACE = re.compile(r"P?MPI_[A-Za-z0-9_]+\Z")
_SCALARS = dict(
    void="void",
    bool="bool",
    char="char",
    f32="float",
    f64="double",
    f80="long double",
    **{
        f"{s}{n}": f"std::{c}int{n}_t"
        for s, c in (("i", ""), ("u", "u"))
        for n in (8, 16, 32, 64)
    },
)

_CMAKE = """cmake_minimum_required(VERSION 3.20)
project(scar_mpi_qualification LANGUAGES CXX)
set(MPI_ASSUME_NO_BUILTIN_MPI TRUE)
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(probe probe.cpp)
set_target_properties(probe PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_compile_definitions(probe PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_link_libraries(probe PRIVATE MPI::MPI_CXX ${CMAKE_DL_LIBS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM|NVHPC")
  target_compile_options(probe PRIVATE -Werror)
endif()
file(WRITE "${CMAKE_BINARY_DIR}/selection" "${MPI_CXX_VERSION}\n${CMAKE_CXX_COMPILER}\n${MPI_CXX_COMPILER}\n")
file(APPEND "${CMAKE_BINARY_DIR}/selection" "launcher=${MPIEXEC_EXECUTABLE}\nrank_option=${MPIEXEC_NUMPROC_FLAG}\n")
foreach(dir IN LISTS MPI_CXX_INCLUDE_DIRS)
  file(APPEND "${CMAKE_BINARY_DIR}/selection" "include=${dir}\n")
endforeach()
foreach(lib IN LISTS MPI_CXX_LIBRARIES)
  file(APPEND "${CMAKE_BINARY_DIR}/selection" "library=${lib}\n")
endforeach()
"""


def _identifier(value, pattern=_IDENT):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid MPI C ABI identifier {value!r}")
    return value


def dependency(node):
    """Recognize both tagged operations and externally bound type-only uses."""
    return (
        node.get("provider") == "mpi"
        or "mpi_interface" in node
        or "mpi_interfaces" in node
    )


def _wire_type(spec, records):
    if isinstance(spec, str) and spec in _SCALARS:
        return _SCALARS[spec]
    if isinstance(spec, dict):
        kind = spec.get("kind")
        if kind in {"ptr", "const", "volatile"}:
            wrapper = {"ptr": "Ptr", "const": "Const", "volatile": "Volatile"}[kind]
            return f"{wrapper}<{_wire_type(spec.get('to' if kind == 'ptr' else 'of'), records)}>"
        if kind == "record" and spec.get("name") in records:
            return records[spec["name"]]
        if kind == "function":
            return _prototype(spec, spec["params"], records)
    raise ValueError(f"MPI C ABI type has no closed target binding: {spec!r}")


def _prototype(decl, params, records):
    types = [_wire_type(spec, records) for spec in (decl["result"], *params)]
    constructor = "Variadic" if decl.get("variadic") else "Function"
    return f"{constructor}<{', '.join(types)}>"


def _interfaces(module):
    defined = {f["name"] for f in module.get("functions", ())}
    externs = {
        e["name"]: e
        for e in module.get("externs", ())
        if e.get("provider") == "mpi" and e["name"] not in defined
    }
    records = {
        r["name"]: _identifier(r.get("symbol"))
        for r in module.get("records", ())
        if r.get("provider") == "mpi"
    }
    globals_ = {
        g["name"]: g for g in module.get("globals", ()) if g.get("provider") == "mpi"
    }
    if not (externs or records or globals_):
        raise ValueError(
            "MPI requires a bound external interface, SDK type or SDK global"
        )
    specs, calls, addresses, ops = [], set(), set(), set()
    for symbol in sorted(set(records.values())):
        specs.append(f"SCAR_TYPE({symbol})")
    for name in sorted(globals_):
        specs.append(
            f"SCAR_GLOBAL({_identifier(name)}, {_wire_type(globals_[name]['type'], records)})"
        )
    for name, decl in sorted(externs.items()):
        prototype = _prototype(decl, (p["type"] for p in decl["params"]), records)
        specs.append(f"SCAR_FUNCTION({_identifier(name, _INTERFACE)}, {prototype})")
    for node in walk_operations(
        [module.get(k, []) for k in ("functions", "globals", "records")]
    ):
        op, family = node["op"], node.get("operator_family")
        if op == "function_ref" and node.get("name") in externs:
            addresses.add(node["name"])
        names = (
            [node.get("callee")]
            if op == "call"
            else node.get("targets", ()) if op == "indirect_call" else ()
        )
        bound = externs.keys() & names
        annotated = "mpi_interface" in node or "mpi_interfaces" in node
        if family == "mpi" or annotated:
            if "mpi_interfaces" in node:
                values = node["mpi_interfaces"]
                valid = (
                    "mpi_interface" not in node
                    and isinstance(values, list)
                    and all(isinstance(s, str) for s in values)
                    and len(values) == len(bound) > 1
                    and set(values) == bound
                )
            else:
                valid = len(bound) == 1 and node.get("mpi_interface") in bound
            if not valid:
                raise ValueError(
                    "MPI call annotation disagrees with its bound external targets"
                )
        if bound:
            if family not in {None, "mpi"}:
                raise ValueError("MPI call has a conflicting operator family")
            ops.add(op)
            calls.update(bound)
    return (
        dict(
            api="c",
            semantics_version=1,
            operations=sorted(ops),
            interfaces=sorted(externs),
        ),
        dict(
            called_or_possible=sorted(calls),
            address_taken=sorted(addresses),
            types=sorted(set(records.values())),
            globals=sorted(globals_),
        ),
        "\n".join(specs) + "\n",
    )


def _run(args, directory):
    result = subprocess.run(
        args,
        cwd=directory,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=90,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError("MPI C++20 qualification failed: " + detail[-2400:])
    return result.stdout


def _artifact(path):
    path = path.resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), sha256=digest(data), size=len(data))


def _probe(specs):
    template = Path(__file__).with_name("probe.cpp")
    source = template.read_bytes()
    with TemporaryDirectory(prefix="scar-mpi-sdk-") as temp:
        root = Path(temp)
        (root / template.name).write_bytes(source)
        (root / "bindings.inc").write_text(specs, encoding="ascii")
        (root / "CMakeLists.txt").write_text(_CMAKE, encoding="ascii")
        build = root / "build"
        _run(
            [
                "cmake",
                "-S",
                ".",
                "-B",
                "build",
                "-G",
                "Unix Makefiles",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            root,
        )
        _run(["cmake", "--build", "build", "--target", "probe"], root)
        header_version, compiler, wrapper, *selection = (
            (build / "selection").read_text().splitlines()
        )
        includes = [
            Path(v.removeprefix("include=")).resolve()
            for v in selection
            if v.startswith("include=")
        ]
        libraries = [
            _artifact(Path(v.removeprefix("library=")))
            for v in selection
            if v.startswith("library=")
        ]
        if not libraries:
            raise ValueError("MPI::MPI_CXX supplied no identifiable link library")
        # The make generator's tested TU records the actual header reads. Select
        # only those reached under the MPI target's explicit include roots.
        dep = build / "CMakeFiles/probe.dir/probe.cpp.o.d"
        operands = dep.read_text().replace("\\\n", " ").split(":", 1)[1]
        headers = {}
        for operand in shlex.split(operands):
            path = Path(operand.replace("$$", "$")).resolve()
            for index, directory in enumerate(includes):
                if path.is_relative_to(directory):
                    relative = path.relative_to(directory).as_posix()
                    headers[index, relative] = dict(
                        _artifact(path), root_index=index, relative=relative
                    )
                    break
        if not any(name == "mpi.h" for _, name in headers):
            raise ValueError(
                "MPI header selected by compiler is outside the identified SDK include roots"
            )
        executable = build / "probe"
        output = _run([str(executable)], root).rstrip("\x00\n")
        location, api, implementation = output.split("\n", 2)
        if api != header_version or not implementation:
            raise ValueError("MPI SDK header or library version query is inconsistent")
        version_provider = dict(kind="unverified")
        if location:
            path = Path(location).resolve(strict=True)
            if path == executable and any(
                Path(lib["path"]).suffix == ".a" for lib in libraries
            ):
                version_provider = dict(kind="statically_linked_probe")
            else:
                loaded = _artifact(path)
                if loaded["sha256"] not in {lib["sha256"] for lib in libraries}:
                    raise ValueError(
                        "runtime MPI version provider differs from the selected link inputs"
                    )
                version_provider = dict(kind="loaded_shared_library", **loaded)
        options = dict(
            v.split("=", 1)
            for v in selection
            if v.startswith(("launcher=", "rank_option="))
        )
        return dict(
            provider="mpi-c",
            device="cpu",
            api="c",
            semantics_version=1,
            standard_version=api,
            library_version=implementation.strip(),
            version_provider=version_provider,
            launcher_candidate=dict(
                path=options["launcher"], rank_option=options["rank_option"]
            ),
            headers=[headers[k] for k in sorted(headers)],
            libraries=libraries,
            probe=dict(
                kind="cxx20_compile_link_query",
                sha256=digest(source),
                bindings_sha256=digest(specs.encode("ascii")),
                compiler=compiler,
                wrapper=wrapper,
                target="MPI::MPI_CXX",
                initialized=False,
                header_scope="transitive probe includes under FindMPI include roots",
                library_scope="FindMPI direct link inputs",
            ),
            rank_selection="unmodeled",
            transport_cost="unmodeled",
        )


def qualify(module, requested, *, explicit):
    expected, uses, specs = _interfaces(module)
    if explicit and (
        not isinstance(requested, dict)
        or requested.keys() != expected.keys()
        or requested.get("api") != "c"
        or type(requested.get("semantics_version")) is not int
        or requested["semantics_version"] != 1
        or any(
            not isinstance(requested[k], list)
            or not all(isinstance(v, str) for v in requested[k])
            or sorted(requested[k]) != expected[k]
            for k in ("operations", "interfaces")
        )
    ):
        raise ValueError(
            "MPI requirement differs from its bound C interfaces and operation sites"
        )
    try:
        binding = _probe(specs)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"MPI C++20 qualification unavailable: {exc}") from exc
    # A legacy IR may lack source SDK hashes. Preserve this distinction even
    # when the deployment probe succeeded.
    source = module.get("resource_model", {}).get("software_headers", {})
    public = {"mpi.h", "mpio.h"}
    roots = {Path(path).parent for path in source if Path(path).name == "mpi.h"}
    expected_headers = {}
    for header in binding["headers"]:
        relative = header["relative"]
        paths = {str(root / relative) for root in roots}
        name = Path(relative).name
        if name in public:
            paths.update(path for path in source if Path(path).name == name)
        for path in paths:
            expected_headers.setdefault(path, set()).add(
                (header["sha256"], header["size"])
            )
    checked = []
    for path, stamp in source.items():
        if path not in expected_headers and Path(path).name not in public:
            continue
        if (stamp.get("sha256"), stamp.get("size")) not in expected_headers.get(
            path, set()
        ):
            raise ValueError(
                f"source MPI header differs from the qualified deployment SDK: {path}"
            )
        checked.append(path)
    binding.update(
        uses,
        interfaces=expected["interfaces"],
        operations=expected["operations"],
        source_header_evidence="matched" if checked else "unavailable",
        source_headers_checked=sorted(checked),
        source_header_count=len(checked),
        source_header_scope="recorded public MPI headers and matching probe dependencies under their source SDK roots",
    )
    return expected, binding
