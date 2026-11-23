"""F1: discover local imports and capture syntax, lexical scopes and source bindings."""

import ast
import builtins
from collections import deque
from dataclasses import dataclass, field, replace
from functools import cache
from hashlib import sha256
from importlib.machinery import BuiltinImporter, ExtensionFileLoader, FrozenImporter
from importlib.machinery import PathFinder, SourceFileLoader, SourcelessFileLoader
from importlib.util import resolve_name
from importlib.metadata import PackageNotFoundError, distribution
from io import BytesIO, TextIOWrapper
import locale
from math import isfinite
import os
from pathlib import Path
import symtable
import sys
import sysconfig
import tokenize


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

    @property
    def source_digests(self):
        return {self.path.relative_to(self.root).as_posix(): self.source_hash}

    compiler_id = f"{sys.implementation.name} {sys.version.split()[0]}"


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
    imports = _imports or _Imports(root)
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
    )


def _digest_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@cache
def _interpreter_digest():
    """Fingerprint the interpreter image supplying built-in and frozen code."""
    images = [Path(sys.executable).resolve(strict=True)]
    if sysconfig.get_config_var("Py_ENABLE_SHARED"):
        directory = sysconfig.get_config_var("LIBDIR")
        library = sysconfig.get_config_var("INSTSONAME") or sysconfig.get_config_var(
            "LDLIBRARY"
        )
        if directory and library and (image := Path(directory, library)).is_file():
            images.append(image)
    parts = [
        ParsedUnit.compiler_id,
        sys.version,
        sys.platform,
        *map(_digest_file, images),
    ]
    return sha256("\0".join(parts).encode()).hexdigest()


class _Imports:
    """One capture's standard-finder results. No loader executes a target module."""

    supported = frozenset(
        {"os", "shutil", "sys", "time", "math", "subprocess", "numpy"}
    )

    def __init__(self, root):
        self.root = root
        self.search_path = [str(root), *sys.path]
        self.specs = {}
        self.providers = {}
        locale_encoding = (
            locale.getencoding()
            if hasattr(locale, "getencoding")
            else locale.getpreferredencoding(False)
        )
        self.runtime = dict(
            interpreter=ParsedUnit.compiler_id,
            platform=sys.platform,
            os_name=os.name,
            default_text_encoding="utf-8" if sys.flags.utf8_mode else locale_encoding,
            filesystem_encoding=sys.getfilesystemencoding(),
            stdin_encoding=getattr(sys.stdin, "encoding", None),
            stdin_errors=getattr(sys.stdin, "errors", None),
            stdout_encoding=getattr(sys.stdout, "encoding", None),
            stdout_errors=getattr(sys.stdout, "errors", None),
            utf8_mode=bool(sys.flags.utf8_mode),
        )

    def top(self, name):
        if name not in self.specs:
            self.specs[name] = (
                BuiltinImporter.find_spec(name)
                or FrozenImporter.find_spec(name)
                or PathFinder.find_spec(name, self.search_path)
            )
        return self.specs[name]

    def project_source(self, name):
        parts = name.split(".")
        if not all(part.isidentifier() for part in parts):
            return None
        spec = self.top(parts[0])
        for part in parts[1:]:
            locations = list(spec.submodule_search_locations or ()) if spec else []
            if not any(Path(p).resolve().is_relative_to(self.root) for p in locations):
                return None
            # A child is looked up on its parent's search path. Passing its leaf
            # to FileFinder avoids consulting sys.modules for an unexecuted parent.
            spec = PathFinder.find_spec(part, locations)
        if spec and isinstance(spec.loader, SourceFileLoader):
            source = Path(spec.origin).resolve()
            if source.is_relative_to(self.root):
                return source
        return None

    def stdlib(self, name):
        spec = self.top(name)
        if not spec or not spec.origin:
            raise ValueError("no interpreter module resolved")
        if spec.loader in (BuiltinImporter, FrozenImporter):
            loader = "builtin" if spec.loader is BuiltinImporter else "frozen"
            origin, digest, basis = (
                spec.origin,
                _interpreter_digest(),
                "interpreter_image",
            )
        else:
            roots = [
                Path(sysconfig.get_path(k)).resolve() for k in ("stdlib", "platstdlib")
            ]
            roots.extend(r / "lib-dynload" for r in tuple(roots))
            if shared := sysconfig.get_config_var("DESTSHARED"):
                roots.append(Path(shared).resolve())
            trusted = PathFinder.find_spec(name, [str(r) for r in dict.fromkeys(roots)])
            source = Path(spec.origin).resolve(strict=True)
            if (
                not isinstance(
                    spec.loader,
                    (SourceFileLoader, ExtensionFileLoader, SourcelessFileLoader),
                )
                or not trusted
                or not trusted.origin
                or source != Path(trusted.origin).resolve(strict=True)
                or not any(source.is_relative_to(r) for r in roots)
            ):
                raise ValueError(
                    f"module resolves outside interpreter stdlib: {source}"
                )
            loader, origin, digest, basis = (
                "path",
                str(source),
                _digest_file(source),
                "provider_file",
            )
        return dict(
            status="verified_stdlib",
            module=name,
            loader=loader,
            origin=origin,
            sha256=digest,
            hash_basis=basis,
            **self.runtime,
        )

    def identify(self, name):
        try:
            if name != "numpy":
                result = self.stdlib(name)
                if name == "os":
                    # os.path is an alias registered by os, even though os has no __path__.
                    path_module = {"posix": "posixpath", "nt": "ntpath"}.get(os.name)
                    if path_module:
                        try:
                            path_source = self.stdlib(path_module)
                        except (OSError, ValueError) as error:
                            path_source = dict(
                                status="unresolved",
                                module=path_module,
                                reason=str(error),
                            )
                        result["aliases"] = {"os.path": path_source}
                return result
            spec = self.top(name)
            if not spec or not spec.origin:
                raise ValueError("module has no resolved filesystem origin")
            package = distribution(name)
            origin = Path(spec.origin).resolve(strict=True)
            expected = Path(package.locate_file(name + "/__init__.py")).resolve(
                strict=True
            )
            if origin != expected:
                raise ValueError("import resolves outside the installed distribution")
            return dict(
                status="verified_distribution",
                distribution=name,
                version=package.version,
                origin=str(origin),
                sha256=_digest_file(origin),
                resolution="standard path finder; provider code not executed",
            )
        except (OSError, ValueError, PackageNotFoundError) as error:
            if name != "numpy" and (source := self.project_source(name)):
                return dict(
                    status="project",
                    module=name,
                    origin=str(source),
                    file=source.relative_to(self.root).as_posix(),
                )
            return dict(status="unresolved", reason=str(error))

    def providers_for(self, tree):
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".", 1)[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names.add(node.module.split(".", 1)[0])
        supported = names & self.supported
        for name in sorted(supported - self.providers.keys()):
            self.providers[name] = self.identify(name)
        return {n: self.providers[n] for n in sorted(supported)}


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
    imports = _Imports(root)

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
