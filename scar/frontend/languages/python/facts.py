"""F1: discover local imports and capture syntax, lexical scopes and source bindings."""

import ast
from base64 import urlsafe_b64encode
import builtins
from collections import deque
from dataclasses import dataclass, field, replace
from functools import cache, cached_property
from hashlib import new as new_hash, sha256
from importlib.machinery import BuiltinImporter, ExtensionFileLoader, FrozenImporter
from importlib.machinery import PathFinder, SourceFileLoader, SourcelessFileLoader
from importlib.util import resolve_name
from importlib.metadata import distributions, packages_distributions
from io import BytesIO, TextIOWrapper
import locale
from math import isfinite
import os
from pathlib import Path
import symtable
import sys
import sysconfig
import tokenize
import unicodedata

from scar.contract import digest, json_bytes


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


def _digest_file(path, algorithm="sha256"):
    digest = new_hash(algorithm)
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

    def __init__(self, root):
        self.root = root
        self.search_path = list(
            dict.fromkeys(str(Path(p or ".").resolve()) for p in (root, *sys.path))
        )
        roots = [
            Path(sysconfig.get_path(k)).resolve() for k in ("stdlib", "platstdlib")
        ]
        roots.extend(r / "lib-dynload" for r in tuple(roots))
        if shared := sysconfig.get_config_var("DESTSHARED"):
            roots.append(Path(shared).resolve())
        self.stdlib_path = list(dict.fromkeys(map(str, roots)))
        self.find = cache(self._find)
        self.file_hash = cache(_digest_file)
        self.named_distributions = cache(
            lambda name: tuple(distributions(name=name, path=self.search_path))
        )
        self.members = cache(self._members)
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

    def _find(self, name, standard=False):
        parent, _, leaf = name.rpartition(".")
        if not leaf.isidentifier():
            return None
        if parent:
            owner = self.find(parent, standard)
            locations = list(owner.submodule_search_locations or ()) if owner else []
            if not locations:
                return None
        else:
            locations = self.stdlib_path if standard else self.search_path
        # Use the leaf on the parent's search path. This also works for a
        # namespace package whose parent has never been placed in sys.modules.
        return (
            BuiltinImporter.find_spec(name, locations if parent else None)
            or FrozenImporter.find_spec(name)
            or PathFinder.find_spec(leaf, locations)
        )

    def project_source(self, name):
        spec = self.find(name)
        if spec and isinstance(spec.loader, SourceFileLoader):
            source = Path(spec.origin).resolve()
            if source.is_relative_to(self.root):
                return source
        return None

    def stdlib(self, name, spec):
        if spec.loader in (BuiltinImporter, FrozenImporter):
            loader = "builtin" if spec.loader is BuiltinImporter else "frozen"
            origin, digest, basis = (
                spec.origin,
                _interpreter_digest(),
                "interpreter_image",
            )
        else:
            trusted = self.find(name, True)
            source = Path(spec.origin).resolve(strict=True)
            if (
                not isinstance(
                    spec.loader,
                    (SourceFileLoader, ExtensionFileLoader, SourcelessFileLoader),
                )
                or not trusted
                or not trusted.origin
                or source != Path(trusted.origin).resolve(strict=True)
                or not any(source.is_relative_to(r) for r in self.stdlib_path)
            ):
                return None
            loader, origin, digest, basis = (
                "path",
                str(source),
                self.file_hash(source),
                "provider_file",
            )
        return dict(
            status="verified_stdlib",
            module=name,
            loader=loader,
            origin=origin,
            sha256=digest,
            hash_basis=basis,
            proof_scope="resolved_module",
            **self.runtime,
        )

    @cached_property
    def import_distributions(self):
        # The standard index also covers distribution names that bear no
        # resemblance to their import names and contributed namespaces.
        return packages_distributions()

    @staticmethod
    def _members(dist):
        """Index this candidate's file claims. Only selected files are hashed."""
        members = {}
        directory = dist.locate_file("")
        if not isinstance(directory, (str, os.PathLike)):
            return members
        directory = str(Path(directory).resolve())
        for member in dist.files or ():
            relative = str(member)
            if relative.endswith((".py", ".pyc", ".so", ".pyd")):
                absolute = os.path.normcase(
                    os.path.normpath(os.path.join(directory, relative))
                )
                members.setdefault(absolute, []).append(member)
        return members

    def distribution_source(self, name, origin):
        key = os.path.normcase(os.path.abspath(origin))
        root_name = name.split(".", 1)[0]

        def claimed(packages):
            return [
                (dist, member)
                for package in sorted(set(packages))
                for dist in self.named_distributions(package)
                for member in self.members(dist).get(key, ())
            ]

        claims = claimed([root_name])
        if not claims or self.find(root_name).origin is None:
            claims = claimed(self.import_distributions.get(root_name, ())) or claims
        unresolved = dict(status="unresolved", module=name, origin=str(origin))
        if not claims:
            return dict(
                unresolved,
                resolution="unowned",
                reason="selected file has no installed distribution member",
            )
        if len(claims) != 1:
            return dict(
                unresolved,
                resolution="ambiguous",
                reason="multiple installed distributions list this file",
                distributions=sorted(
                    dist.metadata.get("Name", "") for dist, _ in claims
                ),
            )
        dist, member = claims[0]
        path = Path(origin).resolve(strict=True)
        digest = self.file_hash(path)
        if member.hash:
            expected = member.hash
            try:
                actual = (
                    digest
                    if expected.mode == "sha256"
                    else self.file_hash(path, expected.mode)
                )
            except (TypeError, ValueError):
                return dict(
                    unresolved,
                    resolution="record_algorithm",
                    reason=f"unsupported file-record digest: {expected.mode}",
                )
            if urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip(
                "="
            ) != expected.value.rstrip("="):
                return dict(
                    unresolved,
                    resolution="record_mismatch",
                    reason="selected bytes differ from the installed file record",
                )
        metadata = dist.metadata
        if not metadata.get("Name") or not metadata.get("Version"):
            return dict(
                unresolved,
                resolution="incomplete_metadata",
                reason="file owner has no distribution name or version",
            )
        return dict(
            status="verified_distribution",
            module=name,
            distribution=metadata["Name"],
            version=metadata["Version"],
            member=str(member),
            origin=str(path),
            sha256=digest,
            hash_basis="provider_file",
            proof_scope=(
                "installed_file_with_digest"
                if member.hash
                else "installed_file_membership"
            ),
            resolution="standard path finder; provider code not executed",
        )

    def identify(self, name):
        if name == "os.path":
            # os installs this checked alias without exposing a package search path.
            alias = self.record("os").get("aliases", {}).get(name)
            if alias:
                return dict(alias, module=name, canonical_module=alias["module"])
        spec = self.find(name)
        if spec is None:
            return dict(
                status="unresolved",
                module=name,
                resolution="missing",
                reason="standard finders found no module",
            )
        if not spec.origin and spec.submodule_search_locations is not None:
            return dict(
                status="namespace",
                module=name,
                loader="namespace",
                locations=list(spec.submodule_search_locations),
                proof_scope="ordered_search_locations",
            )
        if name.split(".", 1)[0] in sys.stdlib_module_names:
            result = self.stdlib(name, spec)
            if result:
                if name == "os" and (
                    path_name := {"posix": "posixpath", "nt": "ntpath"}.get(os.name)
                ):
                    result["aliases"] = {"os.path": self.record(path_name)}
                return result
        if source := self.project_source(name):
            return dict(
                status="project",
                module=name,
                origin=str(source),
                file=source.relative_to(self.root).as_posix(),
            )
        if isinstance(
            spec.loader, (SourceFileLoader, SourcelessFileLoader, ExtensionFileLoader)
        ):
            return self.distribution_source(name, spec.origin)
        return dict(
            status="unresolved",
            module=name,
            origin=spec.origin,
            resolution="loader",
            reason="selected loader has no supported distribution file identity",
        )

    def record(self, name):
        if name not in self.providers:
            try:
                self.providers[name] = self.identify(name)
            except (OSError, ValueError, TypeError) as error:
                self.providers[name] = dict(
                    status="unresolved",
                    module=name,
                    resolution="metadata",
                    reason=str(error),
                )
        return self.providers[name]

    def providers_for(self, tree):
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names.add(node.module)
                for alias in node.names:
                    candidate = node.module + "." + alias.name
                    if alias.name != "*" and (
                        self.find(candidate)
                        or candidate == "os.path"
                        and self.record("os").get("aliases")
                    ):
                        names.add(candidate)
        # A qualified import first loads every parent. From-list submodules
        # remain alternatives to attributes of the already initialized module.
        prefixes = {
            ".".join(parts[:i])
            for name in names
            for parts in [name.split(".")]
            for i in range(1, len(parts) + 1)
        }
        return {name: self.record(name) for name in sorted(prefixes)}


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
