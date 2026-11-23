"""Convert validated Fortran bodies to portable C++ fragments and ABI records."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..contract import read_artifact

_FIXED_SUFFIXES = {".f", ".for", ".ftn"}
_FIXED_FLAGS = {
    "-std=legacy",
    "-std=gnu",
    "-ffixed-form",
    "-ffixed-line-length-72",
    "-fno-fast-math",
    "-ffp-contract=off",
}
_CPP_LINE_MARKER = re.compile(rb'(?m)^#[ \t]+[0-9]+[ \t]+"[^\r\n]*')


@dataclass(frozen=True)
class FortranFragments:
    settings: dict
    files: dict[str, bytes]
    wrapper: str | None
    exports: list[dict]


def _normalize_freeform_sources(
    manifest, units, config, compiler_types, compiler_units, project, mixed
):
    env = os.environ.copy()
    if "fparser_path" in config:
        dependency = Path(config["fparser_path"]).resolve()
        if not (dependency / "fparser").is_dir():
            raise ValueError("fortran.fparser_path must contain the fparser package")
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(dependency), env.get("PYTHONPATH")))
        )
    with TemporaryDirectory(prefix=".fortran-freeform-", dir=project) as temporary:
        work = Path(temporary)
        normalization_units = []
        for index, (name, data) in enumerate(units):
            if name not in compiler_types:
                raise ValueError(f"no sealed Fortran type evidence for {name}")
            snapshot = work / "sources" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data)
            types = work / f"unit-{index}.fortran-original"
            types.write_bytes(read_artifact(compiler_types[name]))
            normalization_units.append(
                dict(
                    unit=name,
                    source=str(snapshot),
                    semantics=str(types),
                    signatures=compiler_units.get(name, {}).get(
                        "fortran_signatures", ()
                    ),
                    passthrough=Path(name).suffix.lower() in _FIXED_SUFFIXES,
                )
            )
        request = work / "normalization-request.json"
        request.write_text(
            json.dumps(
                dict(
                    entry=manifest.get("entry", "main"),
                    mixed=mixed,
                    units=normalization_units,
                )
            ),
            encoding="utf-8",
        )
        normalizer = Path(__file__).with_name("fortran_normalize.py")
        run = subprocess.run(
            [sys.executable, "-B", str(normalizer), str(request)],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
        if run.returncode:
            raise ValueError(
                f"Fortran normalization failed with status {run.returncode}\n{run.stderr.strip()}"
            )
        return json.loads(run.stdout)


def _run(args, cwd):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=180)
    if result.returncode not in (0, 4):
        raise ValueError(
            f"{Path(args[0]).name} failed with status {result.returncode}\n{result.stderr.strip()}"
        )


def _translator(path):
    if path is None:
        path = shutil.which("f2c")
    if (
        type(path) is not str
        or not Path(path).is_absolute()
        or not Path(path).is_file()
    ):
        raise ValueError(
            "Fortran transport requires an absolute f2c path or f2c on PATH"
        )
    return str(Path(path).resolve())


_POINTER_ADJUSTMENTS = "/* Parameter adjustments */"
_FUNCTION_BODY = "/* Function Body */"
_CPP_NONCODE = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\''
)
_POINTER_SHIFT = re.compile(r"^\s*(?:--(\w+)|(\w+)\s*-=\s*(\w+))\s*;", re.MULTILINE)
_INDIRECT_CALL = re.compile(r"\(\s*\*\s*(\w+)\s*\)\s*\(\s*(?!\s*\))")
_DESCRIPTOR_ARRAY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _mask_cpp(source):
    return _CPP_NONCODE.sub(lambda match: re.sub(r"[^\n]", " ", match[0]), source)


def _next_bracket_end(source, opening):
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "[":
            depth += 1
        elif source[index] == "]":
            depth -= 1
            if not depth:
                return index
    raise ValueError("unbalanced subscript in generated f2c source")


def _apply_spans(source, edits):
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _repair_f2c_pointers(source, descriptor_arrays=()):
    """Subtract one-based offsets in subscripts without forming invalid base pointers."""
    changes = []
    for marker in re.finditer(re.escape(_POINTER_ADJUSTMENTS), source):
        body = source.find(_FUNCTION_BODY, marker.end())
        if body < 0:
            raise ValueError("missing function body after f2c parameter adjustments")
        closing = re.search(r"^}", source[body:], re.MULTILINE)
        if not closing:
            raise ValueError("missing function end in generated f2c source")
        stop = body + closing.start()
        base = marker.end()
        prefix = source[base:body]
        mask = _mask_cpp(source[base:stop])
        for shift in _POINTER_SHIFT.finditer(prefix):
            decremented, offset_pointer, offset = shift.groups()
            pointer = decremented or offset_pointer
            offset = "1" if decremented else offset
            changes.append(
                (marker.end() + shift.start(), marker.end() + shift.end(), "")
            )
            references = re.compile(r"\b" + re.escape(pointer) + r"\b\s*")
            for use in references.finditer(mask, shift.end()):
                opening = use.end()
                if opening == len(mask) or mask[opening] != "[":
                    raise ValueError(
                        f"unsupported shifted pointer use in f2c output. {pointer}"
                    )
                closing = _next_bracket_end(mask, opening)
                if pointer in descriptor_arrays:
                    changes.extend(
                        (
                            (
                                base + use.start(),
                                base + opening + 1,
                                f"(*scar_cfi_element({pointer}, (",
                            ),
                            (base + closing, base + closing + 1, f") - ({offset})))"),
                        )
                    )
                else:
                    changes.extend(
                        (
                            (base + opening + 1, base + opening + 1, "("),
                            (base + closing, base + closing, f") - ({offset})"),
                        )
                    )
    repaired = _apply_spans(source, changes)
    if descriptor_arrays:
        mask = _mask_cpp(repaired)
        for pointer in descriptor_arrays:
            if re.search(r"\b" + re.escape(pointer) + r"\s*\[", mask):
                raise ValueError(
                    f"unlowered CFI element access in f2c output. {pointer}"
                )
    return repaired


def _repair_f2c_source(source, descriptor_arrays=()):
    source = _repair_f2c_pointers(source, descriptor_arrays)
    mask = _mask_cpp(source)
    edits = [
        (call.start(), call.end(), f"scar_f2c_call({call[1]}, ")
        for call in _INDIRECT_CALL.finditer(mask)
    ]
    literal = r'("(?:\\.|[^"\\])*")'
    # Literal arguments to libf2c's STOP and output calls are read-only.
    prefix = (
        r"(?:\b(?:s_stop\s*\(\s*|do_fio\s*\(\s*&\w+\s*,\s*"
        r"|do_lio\s*\(\s*&\w+\s*,\s*&\w+\s*,\s*)"
        r"|\bs_copy\s*\(\s*\w+\s*,\s*)"
    )
    for match in re.finditer(prefix + literal + r"\s*,", source):
        edits.append((*match.span(1), f"const_cast<char*>({match[1]})"))
    return '#include "scar_f2c_call.hpp"\n' + _apply_spans(source, edits)


def _translate_sources(sources, translator, directory, project):
    translator = _translator(translator)
    with TemporaryDirectory(prefix=".f2c-", dir=project) as temporary:
        staging = Path(temporary)
        cohorts = {False: [], True: []}
        for index, (source, extended, _) in enumerate(sources):
            filename = f"fortran-{index}.f"
            (staging / filename).write_bytes(source)
            cohorts[extended].append(filename)

        prototypes = {}
        for _ in range(4):
            for extended, members in cohorts.items():
                if members:
                    options = ["-C++", "-Ps", "-g", *(["-f"] if extended else ())]
                    _run([translator, *options, *prototypes, *members], staging)
            inferred = {
                path.name: path.read_bytes() for path in sorted(staging.glob("*.P"))
            }
            if inferred == prototypes and inferred:
                break
            prototypes = inferred
        else:
            raise ValueError("f2c could not stabilize compiler-generated prototypes")

        return {
            f"{directory}/fortran-{index}.cpp": _repair_f2c_source(
                (staging / f"fortran-{index}.c").read_text(encoding="utf-8"), arrays
            ).encode()
            for index, (_, _, arrays) in enumerate(sources)
        }


def lower_fortran_units(
    manifest, project, source_bytes, compiler_types, compiler_units, *, mixed
):
    """Use Fortran toolchain inputs once and return source-language-free fragments."""
    config = manifest.get("fortran", {})
    if not config.get("f2c_header") or not config.get("f2c_archive"):
        raise ValueError(
            "Fortran entries require fortran.f2c_header and fortran.f2c_archive"
        )
    units, fixed = [], []
    for row in manifest["units"]:
        name = row["path"]
        data = source_bytes[name]
        evidence = compiler_units.get(name, {})
        receipt = evidence.get("normalization_source") or evidence.get(
            "preprocessed_source"
        )
        if receipt is not None:
            data = _CPP_LINE_MARKER.sub(b"", read_artifact(receipt))
        units.append((name, data))
        if Path(name).suffix.lower() in _FIXED_SUFFIXES:
            if set(row.get("flags", ())) - _FIXED_FLAGS:
                raise ValueError(f"unsupported fixed-form compiler options in {name}")
            fixed.append((data, False, ()))

    normalized = dict(sources={}, exports=[], descriptor_units={})
    if len(fixed) != len(units):
        normalized = _normalize_freeform_sources(
            manifest, units, config, compiler_types, compiler_units, project, mixed
        )
    sources = [
        *fixed,
        *(
            (source.encode(), True, normalized["descriptor_units"].get(f"{name}.f", ()))
            for name, source in normalized["sources"].items()
        ),
    ]
    directory = "src/fortran-abi" if mixed else "src"
    tool = config.get("f2c")
    translator = str(Path(tool).absolute()) if tool else None
    files = _translate_sources(sources, translator, directory, project)
    destinations = list(files)
    header = Path(config["f2c_header"]).read_text(encoding="utf-8")
    # libf2c only reads the supplied format, including when it is a string literal.
    header = re.sub(r"\bchar\s*\*\s*cifmt\b", "const char *cifmt", header)
    files["include/scar_generated/f2c.h"] = header.encode()

    exports = normalized["exports"]
    digests = {name: sha256(source_bytes[name]).hexdigest() for name, _ in units}
    for export in exports:
        export["source_sha256"] = digests[export["unit"]]
        for index, argument in enumerate(export["arguments"]):
            argument["wrapper_name"] = f"arg{index}"
    wrapper = f"{directory}/scar_c_abi.cpp" if exports else None
    if wrapper:
        destinations.append(wrapper)
    support = "f2c-cfi-v1" if normalized["descriptor_units"] else "f2c-v1"
    settings = dict(
        sources=destinations,
        libraries=(str(Path(config["f2c_archive"]).absolute()), "m"),
        runtime_support=(support,),
        compile_options=(
            "-ffp-contract=off",
            "-fno-fast-math",
            "-Werror=write-strings",
        ),
    )
    return FortranFragments(settings, files, wrapper, exports)
