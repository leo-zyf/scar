"""F1: discover local imports and capture syntax, lexical scopes and source bindings."""

import ast
import builtins
from collections import deque
from dataclasses import dataclass, replace
from functools import cache
from hashlib import sha256
from importlib.util import resolve_name
from io import BytesIO, TextIOWrapper
from pathlib import Path
import symtable
import sys
import tokenize


@dataclass(frozen=True)
class LexicalScope:
    name: str
    line: int
    local: frozenset
    declared_global: frozenset
    free: frozenset
    cell: frozenset
    children: dict

    @classmethod
    def capture(cls, table):
        symbols = table.get_symbols()
        children = [cls.capture(c) for c in table.get_children()]
        local = frozenset(s.get_name() for s in symbols if s.is_local())
        return cls(
            table.get_name(),
            table.get_lineno(),
            local,
            frozenset(s.get_name() for s in symbols if s.is_declared_global()),
            frozenset(s.get_name() for s in symbols if s.is_free() or s.is_nonlocal()),
            local & frozenset(n for c in children for n in c.free),
            {(c.name, c.line): c for c in children},
        )


@dataclass(frozen=True)
class ParsedUnit:
    path: Path
    root: Path
    abi: dict
    tree: ast.Module | None
    scopes: LexicalScope | None
    builtins: frozenset
    libraries: frozenset
    exports: dict
    source_hash: str
    diagnostics: tuple = ()

    @property
    def source_digests(self):
        return {self.path.relative_to(self.root).as_posix(): self.source_hash}

    compiler_id = f"{sys.implementation.name} {sys.version.split()[0]}"


def parse_unit(
    path, flags=(), *, root, abi=None, source_bytes=None, source_hashes=None
):
    """F1: read source once and capture both CPython syntax and lexical binding."""
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes() if source_bytes is None else source_bytes
    source_hash = sha256(raw).hexdigest()
    if (
        source_hashes
        and relative in source_hashes
        and source_hashes[relative] != source_hash
    ):
        raise ValueError(f"Python source snapshot differs from input lock: {relative}")
    try:
        buffer = BytesIO(raw)
        encoding, _ = tokenize.detect_encoding(buffer.readline)
        buffer.seek(0)
        with TextIOWrapper(buffer, encoding=encoding) as source:
            text = source.read()
        tree = ast.parse(text, filename=str(path))
        scopes = LexicalScope.capture(symtable.symtable(text, str(path), "exec"))
    except (SyntaxError, UnicodeError) as error:
        location = dict(
            file=path.relative_to(root).as_posix(),
            line=getattr(error, "lineno", None),
            column=getattr(error, "offset", None),
        )
        issue = dict(
            code="python.syntax",
            detail=getattr(error, "msg", str(error)),
            field="syntax",
            **location,
        )
        return ParsedUnit(
            path,
            root,
            abi or {},
            None,
            None,
            frozenset(),
            frozenset(),
            {},
            source_hash,
            (issue,),
        )
    module = (abi or {}).get("module") or ".".join(
        path.relative_to(root).with_suffix("").parts
    )
    return ParsedUnit(
        path,
        root,
        abi or {},
        tree,
        scopes,
        frozenset(vars(builtins)),
        frozenset(sys.stdlib_module_names),
        _catalog(tree, module),
        source_hash,
    )


def _relative(node, module, package):
    if not node.level:
        return node.module or ""
    parent = module if package else module.rpartition(".")[0]
    try:
        return resolve_name("." * node.level + (node.module or ""), parent)
    except (ValueError, ImportError):
        return ""


def provider(name, libraries):
    base = name.split(".", 1)[0]
    return (
        "python_stdlib" if base in libraries or base == "builtins" else "python_package"
    )


def _function(module, qualified):
    return f"py:{module}:{qualified}"


def _literal_fact(value):
    return {"class": type(value).__name__, "values": [value]}


def _catalog(tree, module):
    """Export lexical names and initial values. Mutable globals remain state edges."""
    exports = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports[node.name] = {
                "class": "type" if isinstance(node, ast.ClassDef) else "function",
                "symbol": _function(module, node.name),
            }
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name):
                    fact = {}
                    if isinstance(value, ast.Constant) and type(value.value) in (
                        bool,
                        str,
                        int,
                        float,
                        type(None),
                    ):
                        fact = _literal_fact(value.value)
                    exports[target.id] = {**fact, "state": _function(module, target.id)}
    return exports


@dataclass(frozen=True)
class ParsedProject:
    profiles: tuple
    units: tuple[ParsedUnit, ...]
    catalog: dict
    source_digests: dict


def capture_project(
    entry: Path, root: Path, *, buffers=None, source_hashes=None
) -> ParsedProject:
    """Discover local imports on the same syntax trees that F2 will normalize."""
    root = Path(root).resolve()
    entry = Path(entry)
    entry = (root / entry).resolve() if not entry.is_absolute() else entry.resolve()

    def module_path(path):
        relative = path.relative_to(root).with_suffix("")
        pieces = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        return ".".join(pieces) or "__main__"

    @cache
    def resolve(module):
        if not module or not all(part.isidentifier() for part in module.split(".")):
            return None
        stem = root.joinpath(*module.split("."))
        # CPython's FileFinder tries the package before its sibling module.
        for path in (stem / "__init__.py", stem.with_suffix(".py")):
            if path.is_file() and path.resolve().is_relative_to(root):
                return path.resolve()
        return None

    entry_module = module_path(entry)
    catalog = {}
    by_path = {}
    modules = {}
    pending = deque([(entry_module, entry)])
    scheduled = {entry_module}

    def enqueue(module):
        parts = module.split(".")
        for i in range(1, len(parts) + 1):
            candidate = ".".join(parts[:i])
            if candidate in scheduled:
                continue
            path = resolve(candidate)
            if path:
                scheduled.add(candidate)
                pending.append((candidate, path))

    while pending:
        module, path = pending.popleft()
        package = path.name == "__init__.py"
        relative = path.relative_to(root).as_posix()
        facts = by_path.get(path)
        if facts is None:
            facts = parse_unit(
                path,
                root=root,
                abi={"module": module},
                source_bytes=(buffers or {}).get(relative),
                source_hashes=source_hashes,
            )
            by_path[path] = facts
        elif facts.tree is not None and module != facts.abi["module"]:
            # Different import names can refer to one physical source via symlinks.
            facts = replace(facts, exports=_catalog(facts.tree, module))
        modules[module] = facts
        catalog[module] = {
            "path": relative,
            "package": package,
            "exports": facts.exports,
        }
        if facts.tree is None:
            continue
        for node in ast.walk(facts.tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    enqueue(name.name)
            elif isinstance(node, ast.ImportFrom):
                source = _relative(node, module, package)
                if source:
                    enqueue(source)
                    for name in node.names:
                        enqueue(source + "." + name.name)
    profiles = tuple(
        dict(
            path=spec["path"],
            language="python",
            abi=dict(
                kind="python_module",
                module=name,
                main=name == entry_module,
                project=catalog,
            ),
        )
        for name, spec in catalog.items()
    )
    return ParsedProject(
        profiles,
        tuple(replace(modules[p["abi"]["module"]], abi=p["abi"]) for p in profiles),
        catalog,
        {
            unit.path.relative_to(root).as_posix(): unit.source_hash
            for unit in by_path.values()
        },
    )


def project_units(entry: Path, root: Path) -> list[dict]:
    return list(capture_project(entry, root).profiles)
