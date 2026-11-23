"""Import GNU Fortran source, type and calling-convention evidence."""

from dataclasses import asdict
from filecmp import cmp
from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from ..contract import artifact

_FORTRAN_SCALARS = {
    "integer(kind=4)": "i32",
    "integer(kind=8)": "i64",
    "real(kind=4)": "f32",
    "real(kind=8)": "f64",
    "logical(kind=4)": "fortran-logical32",
}
_FORTRAN_DECL = re.compile(
    r"^  (?:static )?((?:integer|real|logical)\(kind=[48]\)) ([A-Za-z_][\w.]*)\s*(?:= .*?)?;$",
    re.M,
)
_FORTRAN_PROCEDURE = re.compile(r"^[ \t]*procedure name = (\w+)[ \t]*$", re.M)
_FORTRAN_SYMBOL = re.compile(
    r"^[ \t]*symtree: '(\w+)'[ \t]*\|\| symbol: '\w+'(?:[ \t]*\|\| binding_label: '([^']+)')?[ \t]*\n",
    re.M,
)
_FORTRAN_ARRAY = re.compile(
    r"^[ \t]*Array spec:\(1 \[0\] AS_ASSUMED_SIZE ([01]) \(\) \)[ \t]*$", re.M
)
_FORTRAN_SIGNATURE = re.compile(r"^(void|\w+\(kind=\d+\)) \w+ \((.*)\)$", re.M)
_FORTRAN_PARAMETER = re.compile(
    r"(\w+\(kind=\d+\))(\[[^\]]+\])? (?:([*&]) )?(?:restrict )?(\w+)"
)
_FREE_FORM_SUFFIXES = {".f90", ".f95", ".f03", ".f08"}
_DOUBLE_PLUS = re.compile(
    rb"""'(?:[^'\r\n]|'')*'|"(?:[^"\r\n]|"")*"|![^\r\n]*|"""
    rb"""\+(?:[ \t]*&[ \t]*(?:![^\r\n]*)?\r?\n[ \t]*&?)?[ \t]*(?P<sign>\+)"""
)


def _procedure_signature(preamble):
    match = _FORTRAN_SIGNATURE.search(preamble)
    if match is None:
        return None
    result, parameters = match.groups()
    arguments = []
    for declaration in parameters.split(", ") if parameters else ():
        parameter = _FORTRAN_PARAMETER.fullmatch(declaration)
        if parameter is None:
            return None
        dtype, array, reference, name = parameter.groups()
        arguments.append(
            dict(name=name, dtype=dtype, by_value=reference is None, array=bool(array))
        )
    return dict(result=result, arguments=arguments)


def _counted_procedure_declarations(dump, semantics, ssa):
    """Read GNU's exact BIND(C) formal declarations without a second compiler pass."""
    result, ambiguous = {}, set()
    scopes = list(_FORTRAN_PROCEDURE.finditer(dump))
    for index, scope in enumerate(scopes):
        stop = scopes[index + 1].start() if index + 1 < len(scopes) else len(dump)
        text = dump[scope.end() : stop].split("\n  code:", 1)[0]
        entries = list(_FORTRAN_SYMBOL.finditer(text))
        fields = {
            row[1]: (
                row[2],
                text[
                    row.end() : (
                        entries[i + 1].start() if i + 1 < len(entries) else len(text)
                    )
                ],
            )
            for i, row in enumerate(entries)
        }
        procedure = scope[1]
        label, owner = fields.get(procedure, (None, ""))
        if (
            not label
            or not re.fullmatch(r"[A-Za-z_]\w*", label)
            or "BIND(C) SUBROUTINE)" not in owner
        ):
            continue
        formals = re.search(
            r"^[ \t]*Formal arglist:[ \t]+([A-Za-z_][\w \t]*)$", owner, re.M
        )
        if not formals:
            continue
        names = formals[1].split()
        parameters = []
        for name in names:
            _, details = fields.get(name, (None, ""))
            attributes = re.search(
                r"^[ \t]*attributes:[ \t]*\(([^\n]*)\)[ \t]*$", details, re.M
            )
            if not attributes:
                break
            roles = set(attributes[1].split())
            if not any(role == "DUMMY" or role.startswith("DUMMY(") for role in roles):
                break
            if roles & {
                "OPTIONAL",
                "VOLATILE",
                "ASYNCHRONOUS",
                "TARGET",
                "POINTER",
                "ALLOCATABLE",
                "COARRAY",
            }:
                break
            lower = _FORTRAN_ARRAY.search(details)
            if (
                "(INTEGER 4 C_INTEROP)" in details
                and "VALUE" in roles
                and "DIMENSION" not in roles
            ):
                parameters.append({"name": name, "type": "i32", "passing": "VALUE"})
            elif (
                "(REAL 8 C_INTEROP)" in details
                and lower
                and "DIMENSION" in roles
                and "VALUE" not in roles
            ):
                type_ = "ptr<const f64>" if "DUMMY(IN)" in roles else "ptr<f64>"
                parameters.append(
                    {
                        "name": name,
                        "type": type_,
                        "passing": "ADDRESS",
                        "lower": int(lower[1]),
                    }
                )
            else:
                break
        counts = [p["name"] for p in parameters if p["passing"] == "VALUE"]
        lowers = {p["lower"] for p in parameters if p["passing"] == "ADDRESS"}
        if (
            len(parameters) != len(names)
            or len(lowers) != 1
            or len(counts) != 1
            or len(names) < 3
        ):
            continue
        lower = next(iter(lowers))
        minimum, maximum = -(2**31) + (lower == 0), 2**31 - 1 - (lower == 1)
        abi = {
            "profile": "BIND_C_RANK1_ASSUMED_SIZE",
            "procedure": procedure,
            "symbol": label,
            "semantic_sha256": semantics,
            "ssa_sha256": ssa,
            "induction_lower": lower,
            "parameters": parameters,
            "min_count": minimum,
            "max_count": maximum,
            "bounds": {"parameter": counts[0], "minimum": minimum, "maximum": maximum},
        }
        if label in result:
            ambiguous.add(label)
        result[label] = {
            "parameters": [{"name": p["name"], "type": p["type"]} for p in parameters],
            "fortran_abi": abi,
        }
    return {
        label: declaration
        for label, declaration in result.items()
        if label not in ambiguous
    }


def capture_flags(scratch):
    return [
        "-J",
        str(scratch),
        "-I",
        str(scratch),
        "-O0",
        "-fdump-fortran-original",
        "-save-temps=obj",
    ]


def _prove_dialect_normalization(source, data, compiler, flags, cwd, object_file):
    # Keep the lexical footprint and the original compiler's executable semantics.
    candidate = _DOUBLE_PLUS.sub(
        lambda token: token[0][:-1] + b" " if token["sign"] else token[0], data
    )
    if candidate == data:
        return None
    with TemporaryDirectory(prefix="dialect-", dir=object_file.parent) as temporary:
        work = Path(temporary)
        replay = work / source.name
        replay.write_bytes(candidate)
        normalized_object = work / object_file.name
        command = [
            compiler,
            *flags,
            "-I",
            str(source.parent),
            "-I",
            str(object_file.parent),
            "-J",
            str(work),
            "-O0",
            "-x",
            "f95",
            "-nocpp",
            "-fpreprocessed",
            "-c",
            str(replay),
            "-o",
            str(normalized_object),
        ]
        rebuilt = subprocess.run(command, cwd=cwd, capture_output=True, timeout=180)
        if rebuilt.returncode or not cmp(object_file, normalized_object, shallow=False):
            return None
        if any(
            not (object_file.parent / module.name).is_file()
            or not cmp(object_file.parent / module.name, module, shallow=False)
            for pattern in ("*.mod", "*.smod")
            for module in work.glob(pattern)
        ):
            return None
    return candidate


def seal_source_evidence(
    evidence, output, object_file, semantic_dump, *, source, compiler, flags, cwd
):
    typed_path = output / "types.fortran-original"
    typed_path.write_bytes(semantic_dump)
    normalization_input = source
    for suffix in (".f", ".f90"):
        preprocessed_path = object_file.with_suffix(suffix)
        if preprocessed_path.is_file():
            retained_source = output / f"preprocessed{suffix}"
            shutil.move(preprocessed_path, retained_source)
            evidence["preprocessed_source"] = asdict(artifact(retained_source))
            normalization_input = retained_source
            break
    if (
        normalization_input.suffix.lower() in _FREE_FORM_SUFFIXES
        and "-ffixed-form" not in flags
    ):
        candidate = _prove_dialect_normalization(
            source, normalization_input.read_bytes(), compiler, flags, cwd, object_file
        )
        if candidate is not None:
            normalized = output / "normalized-freeform.f90"
            normalized.write_bytes(candidate)
            evidence["normalization_source"] = asdict(artifact(normalized))
            evidence["normalization_object_sha256"] = artifact(object_file).sha256
    return artifact(typed_path)


def read_gimple_types(preamble):
    return dict(
        scalar_types={
            name: _FORTRAN_SCALARS[dtype]
            for dtype, name in _FORTRAN_DECL.findall(preamble)
            if dtype in _FORTRAN_SCALARS
        },
        fortran_signature=_procedure_signature(preamble),
    )


def seal_procedure_evidence(
    evidence, functions, symbols, semantic_dump, flags, *, counted
):
    public = {
        row["symbol"] for row in symbols.values() if row["public"] and row["definition"]
    }
    evidence["fortran_signatures"] = [
        dict(name=body["name"], symbol=body["symbol"], **body["fortran_signature"])
        for body in functions
        if body["symbol"] in public and body["fortran_signature"] is not None
    ]
    if (
        counted
        and "ssa" in evidence
        and Path(evidence["unit"]).suffix.lower() in _FREE_FORM_SUFFIXES
        and "-ffixed-form" not in flags
    ):
        return _counted_procedure_declarations(
            semantic_dump.decode(errors="replace"),
            evidence["types"]["sha256"],
            evidence["ssa"]["sha256"],
        )
    return {}


@lru_cache(maxsize=8)
def gnu_cfi_header(compiler):
    try:
        location = subprocess.check_output(
            [compiler, "-print-file-name=include/ISO_Fortran_binding.h"],
            text=True,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    header = Path(location)
    if not header.is_absolute() or not header.is_file():
        return None
    return header.resolve(strict=True)
