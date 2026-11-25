"""F1: discover local imports and capture syntax, lexical scopes and source bindings."""

import ast
import builtins
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import dis
from functools import cache
from pathlib import Path
import symtable
import sys
from types import CodeType
import unicodedata

from scar.contract import digest, json_bytes
from .imports import (
    Imports,
    numeric_wire_value,
    relative_module as _relative,
    source_member,
)


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
    source_key: str = ""
    source_origins: dict = field(default_factory=dict)
    import_value_uses: frozenset = frozenset()
    mapping_builders: dict = field(default_factory=dict)

    @property
    def source_digests(self):
        return {self.source_key: self.source_hash}

    compiler_id = f"{sys.implementation.name} {sys.version.split()[0]}"


def tuple_position(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _container_sites(tree, path, relative, source_hash):
    """Capture container construction at exact source spans in one compiler pass."""
    if sys.implementation.name != "cpython" or not hasattr(CodeType, "co_positions"):
        return {}, {}
    nodes, mappings, syntax = {}, {}, deque([tree])
    while syntax:
        node = syntax.popleft()
        syntax.extend(ast.iter_child_nodes(node))
        if isinstance(node, ast.Dict):
            mappings[tuple_position(node)] = []
        elif (
            isinstance(node, ast.Tuple)
            and isinstance(node.ctx, ast.Load)
            and not any(isinstance(e, ast.Starred) for e in node.elts)
        ):
            nodes[tuple_position(node)] = node
    if not (nodes or mappings):
        return {}, {}
    optimize = sys.flags.optimize
    program = compile(tree, str(path), "exec", dont_inherit=True, optimize=optimize)
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
            opcode = getattr(instruction, "baseopname", instruction.opname)
            if instruction.positions in mappings and opcode in {
                "BUILD_MAP",
                "BUILD_CONST_KEY_MAP",
                "MAP_ADD",
                "DICT_UPDATE",
            }:
                mappings[instruction.positions].append((opcode, instruction.arg))
            node = nodes.get(instruction.positions)
            if node is None:
                continue
            size = len(node.elts)
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
    return sites, mappings


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
    _source_key=None,
    _source_origin=None,
):
    """F1: read source once and capture both CPython syntax and lexical binding."""
    path, root = Path(path).resolve(), Path(root).resolve()
    imports = _imports or Imports(root, ParsedUnit.compiler_id)
    relative = _source_key or path.relative_to(root).as_posix()
    snapshot = imports.snapshot(path, source_bytes)
    source_hash = snapshot.sha256
    if _source_origin and _source_origin["sha256"] != source_hash:
        raise ValueError(f"Python provider and captured source disagree: {relative}")
    origins = (
        {relative: {**_source_origin, "size": len(snapshot.raw)}}
        if _source_origin
        else {}
    )
    if (
        source_hashes
        and relative in source_hashes
        and source_hashes[relative] != source_hash
    ):
        raise ValueError(f"Python source snapshot differs from input lock: {relative}")
    try:
        text, tree = snapshot.text, snapshot.tree
        scopes = LexicalScope.capture(symtable.symtable(text, str(path), "exec"))
        tuple_sites, mapping_builders = _container_sites(
            tree, path, relative, source_hash
        )
    except (SyntaxError, UnicodeError, LookupError) as error:
        location = dict(
            file=relative,
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
            source_key=relative,
            source_origins=origins,
        )
    module = (abi or {}).get("module") or ".".join(
        path.relative_to(root).with_suffix("").parts
    )
    value_uses = set()
    providers = (
        imports.providers_for(tree, module, snapshot.package, value_uses=value_uses)
        if not origins
        else {}
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
        providers=providers,
        runtime_facts=imports.runtime,
        source_text=text,
        text_profiles=_text_profiles(),
        tuple_sites=tuple_sites,
        source_key=relative,
        source_origins=origins,
        import_value_uses=frozenset(value_uses),
        mapping_builders=mapping_builders,
    )


def provider(name, libraries):
    base = name.split(".", 1)[0]
    return (
        "python_stdlib" if base in libraries or base == "builtins" else "python_package"
    )


def _function(module, qualified):
    return f"py:{module}:{qualified}"


def _literal_fact(value):
    if isinstance(numeric_wire_value(value), dict):
        return {"class": type(value).__name__}
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
    elif isinstance(node, ast.MatchMapping) and node.rest:
        yield node.rest


def _catalog(tree, module):
    """Index declarations and state slots; runtime bindings follow execution order."""
    exports = {}
    for node in scope_nodes(tree.body):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.Constant
        ):
            v = node.value.value
            if (
                type(v) in {str, int, float, bool, type(None)}
                and not isinstance(numeric_wire_value(v), dict)
                and not (isinstance(v, str) and len(v) > 4096)
            ):
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
        self.snapshots = {}

    def _read(self, path, module, imports, provider):
        external = provider["status"] != "project"
        member = source_member(provider) if external else None
        if external and not member:
            raise ValueError(
                f"selected provider has no verified Python source: {module}"
            )
        relative = (
            f"@python/{provider['provider_id']}/{member}"
            if external
            else path.relative_to(self.root).as_posix()
        )
        key = (relative, module, imports)
        if key not in self.resolved:
            facts = self.parsed.get(relative)
            if facts is None:
                facts = parse_unit(
                    path,
                    root=self.root,
                    abi={"module": module},
                    source_bytes=self.buffers.get(relative),
                    source_hashes=self.source_hashes,
                    _imports=imports,
                    _source_key=relative,
                    _source_origin=(
                        dict(
                            module=module,
                            provider_id=provider["provider_id"],
                            origin=provider["origin"],
                            sha256=provider["sha256"],
                            member=member,
                            status=provider["status"],
                            loader_kind=provider.get("loader_kind", provider["loader"]),
                        )
                        if external
                        else None
                    ),
                )
                self.parsed[relative] = facts
            elif facts.path != path:
                raise ValueError(
                    f"two physical sources resolve to one Python unit: {relative}"
                )
            elif facts.tree is not None:
                # The same bytes can import a different helper in a different image.
                value_uses = set()
                facts = replace(
                    facts,
                    exports=(
                        facts.exports
                        if module == facts.abi["module"]
                        else _catalog(facts.tree, module)
                    ),
                    providers=(
                        {}
                        if external
                        else imports.providers_for(
                            facts.tree,
                            module,
                            path.name == "__init__.py",
                            value_uses=value_uses,
                        )
                    ),
                    import_value_uses=frozenset(value_uses),
                )
            self.resolved[key] = facts
        return self.resolved[key]

    def capture(self, entry, *, search_root=None) -> ParsedProject:
        entry = (self.root / entry).resolve()
        directory = Path(search_root).resolve() if search_root else entry.parent
        key = (directory, entry)
        if key not in self.projects:
            if directory not in self.finders:
                self.finders[directory] = Imports(
                    self.root,
                    ParsedUnit.compiler_id,
                    search_root=directory,
                    snapshots=self.snapshots,
                )
            self.projects[key] = _capture_project(
                entry, self.root, directory, self.finders[directory], self._read
            )
        return self.projects[key]


def _import_nodes(tree, requested):
    """Visit the initialization suite and code that a requested operation may invoke."""
    definitions = defaultdict(list)
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in scope_nodes(tree.body):
        if isinstance(node, (*functions, ast.ClassDef)):
            definitions[node.name].append(node)

    pending = [tree]
    visited = set()

    def callable_body(node):
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, functions) and member.name in {
                    "__init__",
                    "__new__",
                }:
                    pending.extend(member.body)
        else:
            pending.extend(node.body)

    for name, line in requested:
        for node in definitions.get(name, ()):
            if line is None or line == node.lineno:
                callable_body(node)
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        yield node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for definition in definitions.get(node.func.id, ()):
                callable_body(definition)
        if isinstance(node, (*functions, ast.Lambda)):
            # Definitions evaluate their headers. Calling the resulting function
            # is a separate event; a merely present nested import opens no file.
            pending.append(node.args)
            pending.extend(getattr(node, "decorator_list", ()))
            pending.extend(getattr(node, "type_params", ()))
            if annotation := getattr(node, "returns", None):
                pending.append(annotation)
        else:
            pending.extend(ast.iter_child_nodes(node))


def _capture_project(entry, root, directory, imports, read) -> ParsedProject:
    """Use one ordinary unit per local or requested verified Python source module."""

    def module_path(path):
        relative = path.relative_to(directory).with_suffix("")
        pieces = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        return ".".join(pieces) or "__main__"

    entry_module = module_path(entry)
    catalog = {}
    modules = {}
    origins = {
        entry_module: dict(
            status="project",
            module=entry_module,
            origin=str(entry),
            file=entry.relative_to(root).as_posix(),
        )
    }
    requests = defaultdict(set)
    pending = deque([entry_module])
    scheduled = {entry_module}

    def enqueue(name, *, requester, directly_requested, body=None):
        record = imports.record(name)
        if record["status"] != "project" and (
            record.get("source_interface") or not source_member(record)
        ):
            return
        # A verified standard-library source can contribute its complete F2
        # unit on direct demand. Its own dependencies remain typed frontier
        # records unless independently requested by project/distribution code.
        if requester["status"] == "verified_stdlib" or (
            record["status"] == "verified_stdlib" and not directly_requested
        ):
            return
        changed = name not in origins
        origins.setdefault(name, record)
        if body is not None and body not in requests[name]:
            requests[name].add(body)
            changed = True
        if changed and name not in scheduled:
            pending.append(name)
            scheduled.add(name)

    while pending:
        module = pending.popleft()
        scheduled.remove(module)
        origin = origins[module]
        path = Path(origin["origin"])
        package = path.name == "__init__.py"
        facts = read(path, module, imports, origin)
        if facts.tree is not None and origin["status"] != "project":
            value_uses = set()
            facts = replace(
                facts,
                providers=imports.providers_for(
                    facts.tree,
                    module,
                    package,
                    nodes=_import_nodes(facts.tree, requests[module]),
                    value_uses=value_uses,
                ),
                import_value_uses=frozenset(value_uses),
            )
        modules[module] = facts
        catalog[module] = {
            "path": facts.source_key,
            "package": package,
            "exports": facts.exports,
            "source_provider": origin,
            "graph_backed": facts.tree is not None,
        }
        for name, record in facts.providers.items():
            enqueue(name, requester=origin, directly_requested=True)
            if record.get("source_interface") or not record.get("exports"):
                continue
            enqueue(
                name,
                requester=origin,
                directly_requested=True,
                body=("__getattr__", None),
            )
            for member, export in record["exports"].items():
                # Executing `from pkg import f` inspects the binding. The
                # implementation of f is demanded only when its value is used.
                if (name, member) not in facts.import_value_uses:
                    continue
                for candidate in export.get("candidates", ()):
                    if candidate.get("kind", "definition") == "definition":
                        enqueue(
                            candidate["module"],
                            requester=origin,
                            directly_requested=candidate["module"] in facts.providers,
                            body=(candidate["name"], candidate["line"]),
                        )

    for name, facts in modules.items():
        if origins[name]["status"] != "verified_stdlib":
            continue
        providers = dict(facts.providers)
        for target, record in facts.providers.items():
            if (
                target not in modules
                and not record.get("source_interface")
                and (source_member(record) or record["status"] == "project")
            ):
                providers[target] = dict(
                    record,
                    source_expansion=dict(
                        status="frontier", reason="stdlib_dependency", requested_by=name
                    ),
                )
        modules[name] = replace(facts, providers=providers)

    def visible_modules(name):
        # Catalog membership is image-wide. A translation profile needs only
        # names this source can resolve; repeating every export of every unit
        # in each profile makes serialization quadratic in the package graph.
        visible = {name, *modules[name].providers}
        for provider in modules[name].providers.values():
            for export in provider.get("exports", {}).values():
                visible.update(c["module"] for c in export.get("candidates", ()))
        return {
            module: catalog[module] for module in sorted(visible) if module in catalog
        }

    profiles = tuple(
        dict(
            path=spec["path"],
            language="python",
            abi=dict(
                kind="python_module",
                module=name,
                main=name == entry_module,
                project=visible_modules(name),
            ),
        )
        for name, spec in catalog.items()
    )
    return ParsedProject(
        profiles,
        tuple(replace(modules[p["abi"]["module"]], abi=p["abi"]) for p in profiles),
        catalog,
        {unit.source_key: unit.source_hash for unit in modules.values()},
    )


def capture_project(
    entry: Path, root: Path, *, buffers=None, source_hashes=None
) -> ParsedProject:
    capture = ProjectCapture(root, buffers=buffers, source_hashes=source_hashes)
    return capture.capture(entry, search_root=root)


def project_units(entry: Path, root: Path) -> list[dict]:
    return list(capture_project(entry, root).profiles)
