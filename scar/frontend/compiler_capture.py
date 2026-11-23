"""Collect typed source, SSA and call-graph evidence for one compilation unit."""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess

from ..contract import artifact, write_json
from . import gnu_fortran
from .clang import capture_unit, counted_function_bodies, typed_call_occurrences
from .gcc_ir import parse_call_graph, parse_functions

_COMPILER = {"c": "gcc", "c++": "g++", "fortran": "gfortran"}
_LOOSE_FLOAT = {
    "-Ofast",
    "-ffast-math",
    "-funsafe-math-optimizations",
    "-fassociative-math",
    "-ffinite-math-only",
    "-freciprocal-math",
    "-fno-signed-zeros",
    "-fno-trapping-math",
}


COMPILED_LANGUAGES = frozenset(_COMPILER)


@dataclass(frozen=True, slots=True)
class CompilationUnitEvidence:
    receipts: dict
    functions: list[dict]
    symbols: dict
    counted_declarations: dict
    call_sites: dict


def select_compiler(target, unit):
    language = unit.get("language", target["language"])
    config = target.get("fortran", {}) if language == "fortran" else {}
    name = unit.get("compiler") or config.get("gfortran") or _COMPILER[language]
    executable = shutil.which(name)
    if not executable:
        raise ValueError(f"compiler not found: {name}")
    # GCC and G++ infer their default language from the command name.
    return str(Path(executable).absolute())


def capture_unit_evidence(
    target, unit, compiler, out, scratch, cfi
) -> CompilationUnitEvidence:
    root = Path(target["source_root"])
    language = unit.get("language", target["language"])
    ssa, ipa = out / "body.ssa", out / "symbols.cgraph"
    object_file = scratch / f"{out.name}.o"
    flags = [*unit["flags"], *unit.get("gcc_flags", ())]
    is_fortran = language == "fortran"
    extra = gnu_fortran.capture_flags(scratch) if is_fortran else ()
    command = [
        compiler,
        *flags,
        *extra,
        f"-fdump-tree-ssa-raw-blocks-lineno-vops={ssa}",
        f"-fdump-ipa-cgraph={ipa}",
        "-c",
        str(root / unit["path"]),
        "-o",
        str(object_file),
    ]
    result = subprocess.run(
        command, cwd=root, capture_output=True, timeout=180, check=False
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"compiler capture failed for {unit['path']}: {message}")
    evidence = {"unit": unit["path"], "language": language, "command": command}
    if result.stderr:
        (out / "diagnostics.txt").write_bytes(result.stderr)
        evidence["diagnostics"] = asdict(artifact(out / "diagnostics.txt"))
    if is_fortran:
        typed = gnu_fortran.seal_source_evidence(
            evidence,
            out,
            object_file,
            result.stdout,
            source=root / unit["path"],
            compiler=compiler,
            flags=flags,
            cwd=root,
        )
        bodies, call_sites = {}, {}
    else:
        document = capture_unit(root, unit, cfi=cfi)
        bodies = (
            {}
            if _LOOSE_FLOAT.intersection(flags)
            else counted_function_bodies(document)
        )
        call_sites = typed_call_occurrences(document)
        typed = write_json(out / "types.json", document)
    evidence["types"] = asdict(typed)
    functions = []
    if ssa.is_file():
        evidence["ssa"] = asdict(artifact(ssa))
        functions = parse_functions(
            ssa.read_text(errors="replace"),
            read_types=gnu_fortran.read_gimple_types if is_fortran else None,
        )
    if ipa.is_file():
        evidence["symbols"] = asdict(artifact(ipa))
        symbols = parse_call_graph(ipa.read_text(errors="replace"))
    else:
        symbols = {}
    if is_fortran:
        bodies = gnu_fortran.seal_procedure_evidence(
            evidence,
            functions,
            symbols,
            result.stdout,
            flags,
            counted=not _LOOSE_FLOAT.intersection(flags),
        )
    return CompilationUnitEvidence(evidence, functions, symbols, bodies, call_sites)
