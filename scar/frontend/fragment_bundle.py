"""Seal executable fragments and compiler-proved ABI declarations in the graph bundle."""

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from ..contract import (
    NATIVE_GRAPH_VERSION,
    Artifact,
    artifact,
    json_digest,
    local_member,
    read_artifact,
    write_json,
)

from .c_fragments import adapt_c_source, collect_c_adaptations, read_c_family_types
from .fortran_fragments import lower_fortran_units

_GNU_CFI_HEADER = "ISO_Fortran_binding.h"
_GNU_CFI_ROUTINES = {
    "CFI_establish",
    "CFI_is_contiguous",
    "CFI_section",
    "CFI_setpointer",
}


def _compile_settings(flags, source_root, staged_root):
    include_dirs = []
    # The generated CMake Release profile defines NDEBUG independently of
    # the original source command. Apply the caller's macro flags after it.
    compile_options = ["-UNDEBUG"]
    tokens = iter(flags)
    for flag in tokens:
        if flag.startswith(("-std=", "--std=")):
            continue
        if flag in ("-std", "--std"):
            if next(tokens, None) is None:
                raise ValueError("missing language standard in native compiler flags")
            continue
        if not flag.startswith("-I"):
            compile_options.append(flag)
            continue
        location = next(tokens, None) if flag == "-I" else flag[2:]
        if not location:
            raise ValueError("missing include directory in native compiler flags")
        include_dir = (source_root / location).resolve()
        if not include_dir.is_relative_to(source_root):
            if not include_dir.is_dir():
                raise ValueError(f"target include directory is unavailable: {location}")
            include_dirs.append(str(include_dir))
        else:
            include_dirs.append(
                (staged_root / include_dir.relative_to(source_root)).as_posix()
            )
    return tuple(include_dirs), tuple(compile_options)


def _locked_source(root, name, lock):
    path = (root / name).resolve()
    if (
        Path(name).is_absolute()
        or not path.is_relative_to(root)
        or ".." in Path(name).parts
    ):
        raise ValueError(f"source path outside the locked root: {name}")
    data = path.read_bytes()
    size = lock.get("size_bytes", lock.get("size"))
    if len(data) != size or sha256(data).hexdigest() != lock["sha256"]:
        raise ValueError(f"locked source changed: {name}")
    return data


def _stage(project, name, data):
    target = project / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)


def _stage_gnu_cfi(header, project, headers):
    if header is not None:
        origin, digest = header["path"], header["sha256"]
        data = read_artifact(header)
    elif headers:
        origin, digest = headers[0]["path"], headers[0]["sha256"]
        data = Path(origin).read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError("the GNU CFI binding header changed after type capture")
    else:
        raise ValueError("the Fortran plan must carry its GNU CFI binding header")
    for record in headers:
        if (record["path"], record["sha256"]) != (origin, digest):
            raise ValueError(
                "the native call was typed against a different GNU CFI binding header"
            )
    symbols = {symbol for record in headers for symbol in record["symbols"]}
    unsupported = symbols - _GNU_CFI_ROUTINES
    if unsupported:
        raise ValueError(
            f"unsupported standalone GNU CFI routines: {sorted(unsupported)}"
        )

    # It remains a toolchain header when sealed into the relocatable project.
    _stage(
        project,
        "include/scar_generated/" + _GNU_CFI_HEADER,
        b"#pragma GCC system_header\n" + data,
    )
    return bool(symbols)


def _stage_native_sources(manifest, evidence, project, regions):
    """Describe compatibility inputs from pinned source and compiler evidence."""
    profile = manifest["language"]
    units = {unit["path"]: unit for unit in manifest["units"]}
    languages = {name: unit.get("language", profile) for name, unit in units.items()}
    material = manifest["source_lock"]["files"]
    if (
        not units
        or len(units) != len(manifest["units"])
        or not units.keys() <= material.keys()
    ):
        raise ValueError("every translation unit needs locked source bytes")
    if set(languages.values()) - {"c", "c++", "fortran"}:
        raise ValueError("the C++20 backend requires C, C++ or Fortran source units")
    mixed = len(set(languages.values())) > 1
    root = Path(manifest["source_root"]).resolve()
    fortran = [
        units[name] for name, language in languages.items() if language == "fortran"
    ]
    source_bytes = {
        name: _locked_source(root, name, lock) for name, lock in material.items()
    }
    captures = evidence.get("compiler_types", {})
    project.mkdir()
    bridge = None
    if fortran:
        config = dict(manifest, units=fortran, language="fortran")
        compiler_units = {
            row["unit"]: row
            for row in evidence.get("units", ())
            if row["language"] == "fortran"
        }
        bridge = lower_fortran_units(
            config, project, source_bytes, captures, compiler_units, mixed=mixed
        )
        for name, data in bridge.files.items():
            _stage(project, name, data)
    if not mixed and fortran:
        build = dict(bridge.settings)
        if "f2c-cfi-v1" in build["runtime_support"]:
            _stage_gnu_cfi(evidence.get("gnu_cfi_header"), project, ())
        build["libraries"] = (*manifest.get("libraries", ()), *build["libraries"])
        for key in ("link_options", "runtime_dirs"):
            if manifest.get(key):
                build[key] = manifest[key]
        return build, {}, bridge
    c_units = [units[name] for name, language in languages.items() if language == "c"]
    cxx_units = [
        units[name]
        for name, language in languages.items()
        if language == "c++" and fortran
    ]
    c_root = Path("src/c-abi") if mixed else Path("src")
    cxx_root = Path("src/cxx") if mixed else Path("src")
    decls, casts, headers, cfi_headers = collect_c_adaptations(
        c_units, source_bytes, mixed, captures
    )
    for unit in cxx_units:
        typed = read_c_family_types(unit, captures, source_bytes[unit["path"]])
        cfi_headers.extend(typed.get("gnu_cfi", ()))
    if (decls.keys() | casts.keys() | headers.keys()) - material.keys():
        raise ValueError(
            "C translation referenced project files missing from the source lock"
        )

    settings, anchors = {}, {}
    for name, data in source_bytes.items():
        language = languages.get(name)
        unit_regions = regions.get(name, ())
        if language == "c" or (language is None and mixed and c_units):
            output = c_root / name
            if language == "c":
                output = output.with_suffix(".cpp")
                settings[output.as_posix()] = _compile_settings(
                    units[name]["flags"], root, c_root
                )
            fragment, body_spans = adapt_c_source(
                name, data, decls[name], casts[name], headers[name], unit_regions
            )
            _stage(project, output, fragment)
            anchors.update(
                {region: (output.as_posix(), span) for region, span in body_spans}
            )
        if language in (None, "c++"):
            output = cxx_root / name
            _stage(project, output, data)
            anchors.update(
                {
                    row["region"]: (output.as_posix(), row["source"]["span"])
                    for row in unit_regions
                }
            )
            if language == "c++":
                settings[output.as_posix()] = _compile_settings(
                    units[name]["flags"], root, cxx_root
                )

    fortran_libraries = ()
    runtime_support = ()
    cfi_runtime = False
    if bridge:
        build = bridge.settings
        settings.update(
            {name: ((), build["compile_options"]) for name in build["sources"]}
        )
        fortran_libraries = build["libraries"]
        runtime_support = build["runtime_support"]
        if cfi_headers or "f2c-cfi-v1" in runtime_support:
            cfi_runtime = _stage_gnu_cfi(
                evidence.get("gnu_cfi_header"), project, cfi_headers
            )
        if cfi_runtime:
            runtime_support += ("gnu-cfi-v1",)

    profiles = (manifest,) if profile == "c" else (manifest, manifest.get("cxx", {}))
    linkage = {
        key: tuple(value for profile in profiles for value in profile.get(key, ()))
        for key in ("libraries", "link_options", "runtime_dirs")
    }
    linkage["libraries"] = (*linkage["libraries"], *fortran_libraries)
    common = set(settings.values())
    if cfi_runtime:
        common.add(((), ()))
    include_dirs, compile_options = next(iter(common)) if len(common) == 1 else ((), ())
    configuration = dict(
        sources=tuple(settings),
        include_dirs=include_dirs,
        compile_options=compile_options,
        source_settings=settings if len(common) > 1 else {},
        **linkage,
    )
    if runtime_support:
        configuration["runtime_support"] = runtime_support
    return configuration, anchors, bridge


def _abi_declaration(export):
    arguments = []
    for row in export["arguments"]:
        argument = {
            key: row[key]
            for key in (
                "wrapper_name",
                "c_type",
                "transport_type",
                "by_value",
                "read_only",
            )
        }
        if "character_length" in row:
            argument["character_length"] = row["character_length"]
        if descriptor := row.get("descriptor"):
            argument["descriptor"] = {
                key: descriptor[key]
                for key in ("rank", "cfi_type", "lower_bounds", "contiguous", "intent")
                if key in descriptor
            }
        arguments.append(argument)
    result = export["result"]
    return dict(
        symbol=export["symbol"],
        transport=export["transport"],
        arguments=arguments,
        result=(
            {key: result[key] for key in ("c_type", "transport_type")}
            if result
            else None
        ),
    )


def _fortran_anchor(region, export, capture, path, digest, compiler_types):
    abi, source = region["fortran_abi"], region["source"]
    if (
        abi["profile"] != "BIND_C_RANK1_ASSUMED_SIZE"
        or abi["procedure"] != export["procedure"]
        or abi["symbol"] != region["symbol"]
        or source["sha256"] != export["source_sha256"]
        or region["return_type"] != "void"
        or export["result"] is not None
        or capture is None
        or capture.get("types") != compiler_types
        or compiler_types is None
        or compiler_types["sha256"] != abi["semantic_sha256"]
        or capture.get("ssa", {}).get("sha256") != abi["ssa_sha256"]
    ):
        return None
    parameters, bindings = [], {}
    for arg in export["arguments"]:
        name = arg["name"]
        if (
            arg["by_value"]
            and arg["shape"] is None
            and arg["c_type"] in {"int", "double"}
        ):
            parameters.append(
                dict(
                    name=name,
                    type="i32" if arg["c_type"] == "int" else "f64",
                    passing="VALUE",
                )
            )
        elif not arg["by_value"] and "lower" in arg and arg["c_type"] == "double":
            pointer = "ptr<const f64>" if arg["read_only"] else "ptr<f64>"
            parameters.append(
                dict(name=name, type=pointer, lower=arg["lower"], passing="ADDRESS")
            )
        else:
            return None
        bindings[name] = arg["wrapper_name"]
    if parameters != abi["parameters"] or region["parameters"] != [
        dict(name=row["name"], type=row["type"]) for row in parameters
    ]:
        return None
    domain = dict(
        parameter=abi["bounds"]["parameter"],
        minimum=1,
        maximum=abi["bounds"]["maximum"],
    )
    return dict(
        path=path,
        symbol=export["symbol"],
        abi_sha256=digest,
        bindings=bindings,
        authority="TYPED_ABI_EXPORT",
        count_domain=domain,
    )


def seal_compiled_fragments(
    target, evidence, out, *, regions=()
) -> tuple[Artifact, list[dict]]:
    """Seal source leaves and typed declarations for independently lowering the graph."""
    out = Path(out)
    out.mkdir(parents=True)
    pending = out / "inputs"
    by_unit = defaultdict(list)
    for region in regions:
        by_unit[region["source"]["path"]].append(region)
    configuration, anchors, bridge = _stage_native_sources(
        target, evidence, pending, by_unit
    )
    fragments = {}
    for directory in ("src", "include"):
        for path in sorted((pending / directory).rglob("*")):
            if path.is_file():
                name = path.relative_to(pending).as_posix()
                fragments[name] = asdict(local_member(artifact(path), out))
    exports = bridge.exports if bridge else ()
    wrapper = bridge.wrapper if bridge else None
    declarations = [_abi_declaration(row) for row in exports]
    bridge_exports = {
        (row["unit"], row["symbol"]): (row, json_digest(declaration))
        for row, declaration in zip(exports, declarations, strict=True)
    }
    captures = {
        row["unit"]: row
        for row in evidence.get("units", ())
        if row["language"] == "fortran"
    }
    retained = []
    for region in regions:
        if region["region"] in anchors:
            path, span = anchors[region["region"]]
            fragment = {
                "path": path,
                "span": span,
                "sha256": fragments[path]["sha256"],
                "slice_sha256": region["source"]["slice_sha256"],
            }
            retained.append({**region, "fragment": fragment})
        elif "fortran_abi" in region and (
            binding := bridge_exports.get((region["source"]["path"], region["symbol"]))
        ):
            unit = region["source"]["path"]
            export, digest = binding
            fragment = _fortran_anchor(
                region,
                export,
                captures.get(unit),
                wrapper,
                digest,
                evidence["compiler_types"].get(unit),
            )
            if fragment is not None:
                lowered = dict(region, fragment=fragment)
                del lowered["fortran_abi"]
                retained.append(lowered)
    seals = {row["region"]: json_digest(row) for row in retained}
    abi_units = {wrapper: declarations} if declarations else {}
    return (
        write_json(
            out / "native-units.json",
            dict(
                schema="scar.native-units-ir",
                schema_version=NATIVE_GRAPH_VERSION,
                configuration=configuration,
                fragments=fragments,
                abi_units=abi_units,
                exports=sorted(row["symbol"] for row in exports),
                region_seals=seals,
            ),
        ),
        retained,
    )
