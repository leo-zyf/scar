"""F1: discover local imports and capture syntax, lexical scopes and source bindings."""

import ast
import builtins
from collections import deque
from dataclasses import dataclass, field, replace
from functools import cache
from hashlib import sha256
from io import BytesIO, TextIOWrapper
from math import isfinite
from pathlib import Path
import symtable
import sys
import tokenize
import unicodedata

from scar.contract import digest, json_bytes
from .imports import Imports, relative_module as _relative


@dataclass(frozen=True)
class LexicalScope:
    name: str
    line: int
    local: frozenset
    declared_global: frozenset
    free: frozenset
    cell: frozenset
    forwarded: frozenset
    children: dict

    @classmethod
    def capture(cls, table):
        symbols = table.get_symbols()
        children = [cls.capture(c) for c in table.get_children()]
        local = frozenset(s.get_name() for s in symbols if s.is_local())
        nested_free = frozenset(
            n for child in children for n in child.free | child.forwarded
        )
        is_class = table.get_type() == "class"
        return cls(
            table.get_name(),
            table.get_lineno(),
            local,
            frozenset(s.get_name() for s in symbols if s.is_declared_global()),
            frozenset(s.get_name() for s in symbols if s.is_free() or s.is_nonlocal()),
            nested_free & ({"__class__"} if is_class else local),
            nested_free - {"__class__"} if is_class else frozenset(),
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
    providers: dict = field(default_factory=dict)
    runtime_facts: dict = field(default_factory=dict)
    source_text: str = ""
    text_profiles: dict = field(default_factory=dict)

    @property
    def source_digests(self):
        return {self.path.relative_to(self.root).as_posix(): self.source_hash}

    compiler_id = f"{sys.implementation.name} {sys.version.split()[0]}"


@cache
def _text_profiles():
    """Capture single-code-point printability for the selected interpreter."""
    intervals = []
    start = None
    for codepoint in range(0x110000):
        if chr(codepoint).isprintable():
            if start is not None:
                intervals.append([start, codepoint - 1])
                start = None
        elif start is None:
            start = codepoint
    if start is not None:
        intervals.append([start, 0x10FFFF])
    profile = dict(
        unicode_version=unicodedata.unidata_version,
        interpreter=ParsedUnit.compiler_id,
        nonprintable=intervals,
    )
    return {digest(json_bytes(profile)): profile}


def parse_unit(
    path,
    flags=(),
    *,
    root,
    abi=None,
    source_bytes=None,
    source_hashes=None,
    _imports=None,
):
    """F1: read source once and capture both CPython syntax and lexical binding."""
    path, root = Path(path).resolve(), Path(root).resolve()
    imports = _imports or Imports(root, ParsedUnit.compiler_id)
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
            runtime_facts=imports.runtime,
            text_profiles=_text_profiles(),
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
        providers=imports.providers_for(tree),
        runtime_facts=imports.runtime,
        source_text=text,
        text_profiles=_text_profiles(),
    )


def provider(name, libraries):
    base = name.split(".", 1)[0]
    return (
        "python_stdlib" if base in libraries or base == "builtins" else "python_package"
    )


def _function(module, qualified):
    return f"py:{module}:{qualified}"


def _literal_fact(value):
    return {"class": type(value).__name__, "values": [value]}


def scope_nodes(statements):
    """Visit syntax evaluated in this lexical frame without entering child bodies."""
    pending = list(reversed(statements))
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            values = [*getattr(node, "decorator_list", ()), node.args]
        elif isinstance(node, ast.ClassDef):
            values = [*node.decorator_list, *node.bases, *node.keywords]
        elif isinstance(node, ast.comprehension):
            values = [
                node.iter,
                *node.ifs,
            ]  # Its iteration variable belongs to the comprehension.
        else:
            values = list(ast.iter_child_nodes(node))
        pending.extend(reversed(values))


def assigned_names(node):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        yield node.id
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            yield alias.asname or alias.name.split(".")[0]
    elif (
        isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name
    ):
        yield node.name


def _catalog(tree, module):
    """Index declarations and state slots; runtime bindings follow execution order."""
    exports = {}
    for node in scope_nodes(tree.body):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.Constant
        ):
            v = node.value.value
            if (
                type(v) in {str, int, bool, type(None)}
                or type(v) is float
                and isfinite(v)
            ) and not (isinstance(v, str) and len(v) > 4096):
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                ):
                    if isinstance(target, ast.Name):
                        slot = exports.setdefault(
                            target.id, {"state": _function(module, target.id)}
                        )
                        sites = slot.setdefault("literal_sites", [])
                        if len(sites) < 16:
                            sites.append({"line": node.lineno, "value": v})
        for name in assigned_names(node):
            state = exports.setdefault(name, {"state": _function(module, name)})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                state.setdefault("definitions", []).append(
                    {
                        "line": node.lineno,
                        "class": (
                            "type" if isinstance(node, ast.ClassDef) else "function"
                        ),
                        "decorated": bool(node.decorator_list),
                    }
                )
    for item in exports.values():
        definitions = item.get("definitions", ())
        for d in definitions:
            d["symbol"] = item["state"] + (
                f"@{d['line']}" if len(definitions) > 1 else ""
            )
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
    imports = Imports(root, ParsedUnit.compiler_id)

    def module_path(path):
        relative = path.relative_to(root).with_suffix("")
        pieces = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        return ".".join(pieces) or "__main__"

    resolve = cache(imports.project_source)

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
                _imports=imports,
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
