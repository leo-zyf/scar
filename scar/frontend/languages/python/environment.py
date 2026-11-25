"""CPython launch context and source identity for F1 observations."""

import builtins
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.machinery import SourceFileLoader
import marshal
import os
from pathlib import Path
import sys
from types import BuiltinFunctionType, MemberDescriptorType, ModuleType


@cache
def interpreter_image():
    path = Path(sys.executable).resolve(strict=True)
    checksum, size = sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
            size += len(block)
    return dict(path=str(path), sha256=checksum.hexdigest(), size=size)


def _type_name(value):
    # A finder may be a class or an instance with arbitrary attribute hooks.
    cls = value if issubclass(type(value), type) else type(value)
    module = type.__dict__["__module__"].__get__(cls)
    name = type.__dict__["__qualname__"].__get__(cls)
    return f"{module}.{name}" if type(module) is str else name


def _import_callable():
    namespace = vars(builtins)
    if "__import__" not in namespace:
        return dict(kind="missing", type=None)
    value = namespace["__import__"]
    builtin = (
        type(value) is BuiltinFunctionType
        and value.__self__ is builtins
        and value.__module__ == "builtins"
        and value.__name__ == "__import__"
    )
    return dict(kind="builtin_import" if builtin else "other", type=_type_name(value))


@dataclass(frozen=True)
class ImportEnvironment:
    entries: tuple
    analysis: dict
    cache_states: dict

    def for_providers(self, names):
        """Project the captured state onto the source providers selected by F1."""
        selected = {
            name: self.cache_states.get(name, "absent") for name in sorted(set(names))
        }
        return dict(
            semantics_version=1,
            entries=list(self.entries),
            analysis={**self.analysis, "selected_module_cache": selected},
        )


def capture_import_environment(entries):
    executable = Path(sys.executable)
    for entry in entries:
        declared = entry["declaration"].get("interpreter")
        if declared is not None:
            path = Path(declared)
            # Different virtual environments can symlink to the very same ELF.
            if not (
                path.is_file()
                and path.parent.samefile(executable.parent)
                and path.samefile(executable)
            ):
                raise ValueError(
                    f"declared Python interpreter {declared} differs from the "
                    f"analysis interpreter {executable}; run SCAR with {declared}"
                )
    if type(sys.modules) is not dict:
        raise ValueError("cannot snapshot a custom analysis-process module registry")
    cache_states = {
        key: (
            "present_none"
            if value is None
            else (
                "present_module"
                if issubclass(type(value), ModuleType)
                else "present_other"
            )
        )
        for key, value in sys.modules.copy().items()
        if type(key) is str
    }
    analysis = dict(
        scope="analysis_process_snapshot",
        confidence="observed",
        interpreter=dict(
            executable=str(executable),
            implementation=sys.implementation.name,
            version=sys.version.split()[0],
            prefix=sys.prefix,
            base_prefix=sys.base_prefix,
            image=interpreter_image(),
        ),
        flags={
            name: getattr(sys.flags, name)
            for name, slot in vars(type(sys.flags)).items()
            if isinstance(slot, MemberDescriptorType)
        },
        search_path=[
            str.__str__(path) for path in sys.path if issubclass(type(path), str)
        ],
        meta_path=[_type_name(finder) for finder in sys.meta_path],
        import_callable=_import_callable(),
        audit_hooks="unknown",
        deployment_equivalence="unproven",
        application_initial_cache="unproven",
    )
    return ImportEnvironment(entries, analysis, cache_states)


def code_digest(code):
    return sha256(marshal.dumps(code, 2)).hexdigest()


def run_script(script, arguments, initial_directory, *, data=None):
    """Run a source script with its own main namespace and ordinary path entry."""
    sys.argv = [script, *arguments]
    if not getattr(sys.flags, "safe_path", False) and not sys.flags.isolated:
        sys.path[0] = str(Path(script).resolve().parent)
    filename = os.path.join(initial_directory, script)
    application = ModuleType("__main__")
    application.__dict__.update(
        __file__=filename,
        __package__=None,
        __spec__=None,
        __cached__=None,
        __loader__=SourceFileLoader("__main__", filename),
        __builtins__=builtins,
    )
    sys.modules["__main__"] = application
    code = compile(
        data if data is not None else Path(filename).read_bytes(),
        filename,
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    exec(code, application.__dict__)
