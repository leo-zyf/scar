"""F1: discover local imports and capture syntax, lexical scopes and source bindings."""

import ast
import builtins
from collections import deque
from dataclasses import dataclass, field, replace
import dis
from functools import cache
from hashlib import sha256
from io import BytesIO, TextIOWrapper
from math import isfinite
from pathlib import Path
import symtable
import sys
import tokenize
from types import CodeType
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
    tuple_sites: dict = field(default_factory=dict)

    @property
    def source_digests(self):
        return {self.path.relative_to(self.root).as_posix(): self.source_hash}

    compiler_id = f"{sys.implementation.name} {sys.version.split()[0]}"


def tuple_position(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _tuple_sites(tree, text, path, relative, source_hash):
    """Match actual tuple constructors to exact UTF-8 source spans."""
    if sys.implementation.name != "cpython" or not hasattr(CodeType, "co_positions"):
        return {}
    nodes, syntax = {}, deque([tree])
    while syntax:
        node = syntax.popleft()
        syntax.extend(ast.iter_child_nodes(node))
        if (
            isinstance(node, ast.Tuple)
            and isinstance(node.ctx, ast.Load)
            and not any(isinstance(e, ast.Starred) for e in node.elts)
        ):
            nodes[tuple_position(node)] = node
    if not nodes:
        return {}
    optimize = sys.flags.optimize
    program = compile(text, str(path), "exec", dont_inherit=True, optimize=optimize)
    unit = digest(json_bytes([relative, source_hash, ParsedUnit.compiler_id, optimize]))
    fresh, conflict = object(), object()
    origins = {}

    def observe(node, origin):
        if origins.setdefault(node, origin) is not origin:
            origins[node] = conflict

    pending = [program]
    while pending:
        code = pending.pop()
        pending.extend(c for c in reversed(code.co_consts) if isinstance(c, CodeType))
        for instruction in dis.get_instructions(code):
            node = nodes.get(instruction.positions)
            if node is None:
                continue
            size = len(node.elts)
            opcode = getattr(instruction, "baseopname", instruction.opname)
            if opcode == "BUILD_TUPLE" and instruction.arg == size:
                observe(node, fresh if size else ())
            elif opcode in {"LOAD_CONST", "RETURN_CONST"}:
                value = code.co_consts[instruction.arg]
                if type(value) is tuple and len(value) == size:
                    observe(node, value)

    sites, slots = {}, {}
    # Parents precede children. A folded constant supplies its direct tuple
    # members, never intermediate operands of an expression folded into one member.
    for span, node in nodes.items():
        value = origins.get(node)
        if value is fresh:
            sites[span] = {"kind": "fresh"}
        elif type(value) is tuple:
            if not value:
                sites[span] = {
                    "kind": "constant_pool",
                    "pool": "cpython:empty_tuple",
                    "slot": 0,
                }
                continue
            slot = slots.setdefault(id(value), len(slots))
            sites[span] = {"kind": "constant_pool", "unit": unit, "slot": slot}
            for child, member in zip(node.elts, value):
                if (
                    isinstance(child, ast.Tuple)
                    and nodes.get(tuple_position(child)) is child
                    and type(member) is tuple
                    and len(member) == len(child.elts)
                ):
                    observe(child, member)
    return sites


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
        tuple_sites = _tuple_sites(tree, text, path, relative, source_hash)
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
        tuple_sites=tuple_sites,
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


class ProjectCapture:
    """Share file syntax within one build, keeping each script's import environment."""

    def __init__(self, root, *, buffers=None, source_hashes=None):
        self.root = Path(root).resolve()
        self.buffers = buffers or {}
        self.source_hashes = source_hashes
        self.parsed = {}
        self.resolved = {}
        self.finders = {}
        self.projects = {}

    def _read(self, path, module, imports):
        key = (path, imports)
        if key not in self.resolved:
            facts = self.parsed.get(path)
            if facts is None:
                relative = path.relative_to(self.root).as_posix()
                facts = parse_unit(
                    path,
                    root=self.root,
                    abi={"module": module},
                    source_bytes=self.buffers.get(relative),
                    source_hashes=self.source_hashes,
                    _imports=imports,
                )
                self.parsed[path] = facts
            elif facts.tree is not None:
                # The same bytes can import a different helper in a different image.
                facts = replace(facts, providers=imports.providers_for(facts.tree))
            self.resolved[key] = facts
        facts = self.resolved[key]
        if facts.tree is not None and module != facts.abi["module"]:
            return replace(facts, exports=_catalog(facts.tree, module))
        return facts

    def capture(self, entry, *, search_root=None) -> ParsedProject:
        entry = (self.root / entry).resolve()
        directory = Path(search_root).resolve() if search_root else entry.parent
        key = (directory, entry)
        if key not in self.projects:
            if directory not in self.finders:
                self.finders[directory] = Imports(
                    self.root, ParsedUnit.compiler_id, search_root=directory
                )
            self.projects[key] = _capture_project(
                entry, self.root, directory, self.finders[directory], self._read
            )
        return self.projects[key]


def _capture_project(entry, root, directory, imports, read) -> ParsedProject:
    """Discover local imports on the same syntax trees that F2 will normalize."""

    def module_path(path):
        relative = path.relative_to(directory).with_suffix("")
        pieces = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        return ".".join(pieces) or "__main__"

    resolve = cache(imports.project_source)

    entry_module = module_path(entry)
    catalog = {}
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
        facts = read(path, module, imports)
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
            for unit in modules.values()
        },
    )


def capture_project(
    entry: Path, root: Path, *, buffers=None, source_hashes=None
) -> ParsedProject:
    capture = ProjectCapture(root, buffers=buffers, source_hashes=source_hashes)
    return capture.capture(entry, search_root=root)


def project_units(entry: Path, root: Path) -> list[dict]:
    return list(capture_project(entry, root).profiles)
