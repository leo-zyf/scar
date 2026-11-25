"""F1 static import resolution, installed file identity and export provenance."""

import ast
from base64 import urlsafe_b64encode
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cache, cached_property
from hashlib import new as new_hash, sha256
from importlib.machinery import BuiltinImporter, ExtensionFileLoader, FrozenImporter
from importlib.machinery import PathFinder, SourceFileLoader, SourcelessFileLoader
from importlib.metadata import distributions, packages_distributions
from importlib.util import resolve_name
from io import BytesIO
import locale
from math import isfinite
import os
from pathlib import Path
import sys
import sysconfig
import tokenize

from scar.contract import python_provider_id
from scar.source_contracts import python_source_boundary
from .environment import interpreter_image


def _digest_file(path, algorithm="sha256"):
    digest = new_hash(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@cache
def _interpreter_digest(compiler_id):
    """Fingerprint the interpreter image supplying built-in and frozen code."""
    image_hashes = [interpreter_image()["sha256"]]
    if sysconfig.get_config_var("Py_ENABLE_SHARED"):
        directory = sysconfig.get_config_var("LIBDIR")
        library = sysconfig.get_config_var("INSTSONAME") or sysconfig.get_config_var(
            "LDLIBRARY"
        )
        if directory and library and (image := Path(directory, library)).is_file():
            image_hashes.append(_digest_file(image))
    parts = [
        compiler_id,
        sys.version,
        sys.platform,
        *image_hashes,
    ]
    return sha256("\0".join(parts).encode()).hexdigest()


class Imports:
    """One capture's standard-finder results. No loader executes a target module."""

    def __init__(self, root, compiler_id, *, search_root=None, snapshots=None):
        self.root = root
        self.compiler_id = compiler_id
        self.snapshots = {} if snapshots is None else snapshots
        self.search_path = list(
            dict.fromkeys(
                str(Path(p or ".").resolve()) for p in (search_root or root, *sys.path)
            )
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
        self.installed_directory = cache(
            lambda path: next(iter(distributions(path=[str(path)])), None) is not None
        )
        self.members = cache(self._members)
        self.providers = {}
        self.exports = ExportSources(self)
        self.module_imports = cache(self._module_imports)
        locale_encoding = (
            locale.getencoding()
            if hasattr(locale, "getencoding")
            else locale.getpreferredencoding(False)
        )
        self.runtime = dict(
            interpreter=compiler_id,
            platform=sys.platform,
            os_name=os.name,
            default_text_encoding="utf-8" if sys.flags.utf8_mode else locale_encoding,
            filesystem_encoding=sys.getfilesystemencoding(),
            filesystem_errors=sys.getfilesystemencodeerrors(),
            stdin_encoding=getattr(sys.stdin, "encoding", None),
            stdin_errors=getattr(sys.stdin, "errors", None),
            stdout_encoding=getattr(sys.stdout, "encoding", None),
            stdout_errors=getattr(sys.stdout, "errors", None),
            utf8_mode=bool(sys.flags.utf8_mode),
        )

    def snapshot(self, path, raw=None):
        """Share exactly the bytes inspected for ownership, exports and grammar."""
        if path not in self.snapshots:
            self.snapshots[path] = _Source(
                path, path.read_bytes() if raw is None else raw
            )
        return self.snapshots[path]

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
                _interpreter_digest(self.compiler_id),
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
                (
                    self.snapshot(source).sha256
                    if isinstance(spec.loader, SourceFileLoader)
                    else self.file_hash(source)
                ),
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

    def distribution_source(self, name, origin, loader):
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
        source = self.snapshot(path) if loader == "source" else None
        digest = source.sha256 if source is not None else self.file_hash(path)
        if member.hash:
            expected = member.hash
            try:
                actual = (
                    digest
                    if expected.mode == "sha256"
                    else (
                        new_hash(expected.mode, source.raw).hexdigest()
                        if source is not None
                        else self.file_hash(path, expected.mode)
                    )
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
            loader=loader,
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
            # A script's environment may live inside its source tree. Establish
            # ownership from the selected import directory and actual RECORD.
            depth = name.count(".") + (spec.submodule_search_locations is not None)
            directory = Path(spec.origin).parents[depth]
            if self.installed_directory(directory):
                claimed = self.distribution_source(name, spec.origin, "source")
                if claimed.get("resolution") != "unowned":
                    return claimed
            return dict(
                status="project",
                module=name,
                origin=str(source),
                file=source.relative_to(self.root).as_posix(),
            )
        for loader, kind in (
            (SourceFileLoader, "source"),
            (SourcelessFileLoader, "bytecode"),
            (ExtensionFileLoader, "extension"),
        ):
            if isinstance(spec.loader, loader):
                return self.distribution_source(name, spec.origin, kind)
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
                record = self.identify(name)
                if record.get("status") == "verified_distribution":
                    record["provider_id"] = python_provider_id(record)
                    if interface := python_source_boundary(record):
                        record["source_interface"] = interface
                self.providers[name] = record
            except (OSError, ValueError, TypeError) as error:
                self.providers[name] = dict(
                    status="unresolved",
                    module=name,
                    resolution="metadata",
                    reason=str(error),
                )
        return self.providers[name]

    def _module_imports(self, name):
        """Static finder evidence for direct, absolute imports in this module suite."""
        record = self.record(name)
        if record.get("status") != "verified_stdlib" or record.get("loader") != "path":
            return None
        path = Path(record["origin"])
        if path.suffix != ".py":
            return None
        source, error = self.exports.read_file(path, record["sha256"])
        if source is None:
            return dict(scope="absolute_module_suite", error=error)
        order = []
        for statement in source.tree.body:
            if isinstance(statement, ast.Import):
                order.extend(alias.name for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom) and not statement.level:
                if statement.module is not None:
                    order.append(statement.module)
        keys = (
            "module",
            "status",
            "loader",
            "origin",
            "sha256",
            "hash_basis",
            "proof_scope",
            "resolution",
            "reason",
            "file",
            "interpreter",
            "platform",
        )
        modules = {
            dependency: {
                key: value
                for key, value in self.record(dependency).items()
                if key in keys
            }
            for dependency in dict.fromkeys(order)
        }
        return dict(scope="absolute_module_suite", order=order, modules=modules)

    def providers_for(
        self, tree, module="", package=False, *, nodes=None, value_uses=None
    ):
        names = set()
        aliases = defaultdict(set)
        requests = defaultdict(set)
        accesses = set()
        for node in ast.walk(tree) if nodes is None else nodes:
            if isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name)
                    local = a.asname or a.name.split(".")[0]
                    aliases[local].add((a.name if a.asname else local, ()))
            elif isinstance(node, ast.ImportFrom):
                base = relative_module(node, module, package)
                if not base:
                    continue
                names.add(base)
                for alias in node.names:
                    candidate = base + "." + alias.name
                    if alias.name == "*":
                        continue
                    requests[base].add(alias.name)
                    aliases[alias.asname or alias.name].add((base, (alias.name,)))
                    if (
                        self.find(candidate)
                        or candidate == "os.path"
                        and self.record("os").get("aliases")
                    ):
                        names.add(candidate)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                path = _dotted(node)
                if path:
                    accesses.add(path)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                accesses.add((node.id,))
        for root, *parts in sorted(accesses):
            for target, prefix in aliases.get(root, ()):
                namespace = target
                qualified = (*prefix, *parts)
                for i, part in enumerate(qualified):
                    member = ".".join(qualified[i:])
                    requests[namespace].add(member)
                    if value_uses is not None:
                        value_uses.add((namespace, member))
                    child = namespace + "." + part
                    if not self.find(child):
                        break
                    names.add(child)
                    namespace = child
        # A qualified import first loads every parent. From-list submodules
        # remain alternatives to attributes of the already initialized module.
        prefixes = {
            ".".join(parts[:i])
            for name in names
            for parts in [name.split(".")]
            for i in range(1, len(parts) + 1)
        }
        records = {name: self.record(name) for name in sorted(prefixes)}
        for name, record in records.items():
            if (direct := self.module_imports(name)) is not None:
                records[name] = dict(record, direct_imports=direct)
        additions = defaultdict(dict)
        for module, members in requests.items():
            if records.get(module, {}).get("status") != "verified_distribution":
                continue
            for member in sorted(members):
                evidence = self.exports.get(module, member)
                if (
                    "." not in member
                    and member != "__getattr__"
                    and any(c["kind"] == "literal" for c in evidence["candidates"])
                ):
                    hook = self.exports.get(module, "__getattr__")
                    if hook["candidates"] or any(
                        u["reason"] not in {"unbound", "deleted"}
                        for u in hook["unresolved"]
                    ):
                        additions[module]["__getattr__"] = hook
                        evidence = dict(
                            evidence,
                            on_missing=dict(
                                kind="package_attribute_missing",
                                module=module,
                                name=member,
                                hook="__getattr__",
                            ),
                        )
                additions[module][member] = evidence
                root, _, tail = module.partition(".")
                if tail and records[root].get("status") == "verified_distribution":
                    # Retain the qualified spelling at the original import's root.
                    additions[root].setdefault(tail + "." + member, evidence)
        for name, exports in additions.items():
            records[name] = dict(records[name], exports=exports)
        return records


def relative_module(node, module, package):
    if not node.level:
        return node.module or ""
    parent = module if package else module.rpartition(".")[0]
    try:
        return resolve_name("." * node.level + (node.module or ""), parent)
    except (ValueError, ImportError):
        return ""


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return (node.id, *reversed(parts)) if isinstance(node, ast.Name) else ()


def _target_names(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Starred):
        yield from _target_names(node.value)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for item in node.elts:
            yield from _target_names(item)


def _namespace(target):
    return (
        isinstance(target, ast.Call)
        and _dotted(target.func) in {("globals",), ("locals",), ("vars",)}
        and not target.args
        and not target.keywords
    )


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_IMPLICIT = object()
_DYNAMIC = object()


def _literal_value(node):
    """A literal token or a single numeric sign, without evaluating user code."""
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    return _DYNAMIC


def numeric_wire_value(value):
    """Keep source numeric tokens lossless when JSON cannot encode the number."""
    if type(value) is float and not isfinite(value):
        return {"float_hex": value.hex()}
    if type(value) is int and value.bit_length() > 2048:
        limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
        if limit and value.bit_length() > 3 * limit:
            try:
                str(value)
            except ValueError:
                return {"integer_hex": hex(value)}
    return value


def _literal_source(value):
    source_type = "builtins." + type(value).__name__
    kind = {
        str: "text",
        bool: "bool",
        type(None): "unit",
        int: "i64",
        float: "f64",
    }.get(type(value))
    if kind is None:
        return {"source_type": source_type}
    fields = {"source_type": source_type}
    if kind == "i64" and not -(1 << 63) <= value < 1 << 63:
        kind = "python_int"
        fields["representation"] = "requires_integer_range_contract"
        encoded = numeric_wire_value(value)
        if isinstance(encoded, dict):
            return {
                **fields,
                "literal": {"type": kind, "value_hex": encoded["integer_hex"]},
            }
    elif kind == "f64" and not isfinite(value):
        return {
            **fields,
            "literal": {"type": "python_float", "value_hex": value.hex()},
            "representation": "requires_nonfinite_float_contract",
        }
    return {**fields, "literal": {"type": kind, "value": value}}


def _strings(value):
    if isinstance(value, (ast.List, ast.Tuple)) and all(
        isinstance(item, ast.Constant) and type(item.value) is str
        for item in value.elts
    ):
        return {item.value for item in value.elts}
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        first, last = _strings(value.left), _strings(value.right)
        if isinstance(first, set) and isinstance(last, set):
            return first | last
    return _DYNAMIC


def _star_names(tree):
    """Explicit, locally calculable __all__, implicit public bindings, or unknown."""
    names = _IMPLICIT
    mutators = set()
    for stmt in tree.body:
        if isinstance(stmt, _DEFINITIONS):
            if any(
                isinstance(n, ast.Name) and n.id == "__all__" for n in ast.walk(stmt)
            ):
                mutators.add(stmt.name)
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                incoming = _strings(stmt.value)
                if isinstance(stmt, ast.AugAssign):
                    names = (
                        names | incoming
                        if isinstance(stmt.op, ast.Add)
                        and isinstance(names, set)
                        and isinstance(incoming, set)
                        else _DYNAMIC
                    )
                else:
                    names = incoming
                continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            path = _dotted(call.func)
            if len(call.args) == 1 and not call.keywords and isinstance(names, set):
                arg = call.args[0]
                if (
                    path == ("__all__", "append")
                    and isinstance(arg, ast.Constant)
                    and type(arg.value) is str
                ):
                    names = names | {arg.value}
                    continue
                if path == ("__all__", "extend") and isinstance(
                    items := _strings(arg), set
                ):
                    names = names | items
                    continue
        if any(
            isinstance(n, ast.Name)
            and n.id == "__all__"
            or isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in mutators
            for n in ast.walk(stmt)
        ):
            names = _DYNAMIC
    return names


@dataclass(frozen=True)
class _Binding:
    kind: str
    node: ast.stmt | None = None
    module: str = ""
    path: tuple = ()
    guards: tuple = ()


@dataclass(frozen=True)
class _Source:
    path: Path
    raw: bytes

    @cached_property
    def sha256(self):
        return sha256(self.raw).hexdigest()

    @cached_property
    def text(self):
        encoding, _ = tokenize.detect_encoding(BytesIO(self.raw).readline)
        return self.raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")

    @cached_property
    def tree(self):
        return ast.parse(self.text, filename=str(self.path))

    @cached_property
    def lines(self):
        return self.text.split("\n")

    @cached_property
    def star_names(self):
        return _star_names(self.tree)

    @property
    def package(self):
        return self.path.name == "__init__.py"

    @cached_property
    def string_keys(self):
        return _loop_keys(self.tree)


def _loop_keys(tree):
    """Known namespace subscript keys in a loop over fixed, unrebound strings."""
    keys = {}
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
            continue
        domain = _strings(loop.iter)
        if not isinstance(domain, set):
            continue
        name = loop.target.id
        body = [n for stmt in loop.body for n in ast.walk(stmt)]
        if any(
            isinstance(n, ast.Name)
            and n.id == name
            and not isinstance(n.ctx, ast.Load)
            or isinstance(n, _DEFINITIONS)
            and n.name == name
            or isinstance(n, ast.alias)
            and (n.asname or n.name.partition(".")[0]) == name
            for n in body
        ):
            continue
        for n in body:
            if (
                isinstance(n, ast.Subscript)
                and _namespace(n.value)
                and isinstance(n.slice, ast.Name)
                and n.slice.id == name
            ):
                keys[n] = domain
    return keys


def _guarded(bindings, condition):
    return [replace(b, guards=(condition, *b.guards)) for b in bindings]


def _statement_lines(definition, lines):
    locations = {}

    def visit(node):
        if isinstance(node, ast.stmt):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return
            locations[str(node.lineno)] = lines[node.lineno - 1].strip()
            if isinstance(node, _DEFINITIONS):
                return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in definition.body:
        visit(stmt)
    return locations


class ExportSources:
    """Trace requested namespace bindings to possible sources in verified bytes."""

    def __init__(self, imports):
        self.imports = imports
        self.source = cache(self._source)
        self.read_file = cache(self._read_file)
        self.bindings = cache(self._bindings)
        self.get = cache(self._get)

    def _source(self, module):
        record = self.imports.record(module)
        if record.get("status") != "verified_distribution":
            return None, record.get("resolution", record["status"])
        if record.get("proof_scope") != "installed_file_with_digest":
            return None, "missing_distribution_file_digest"
        path = Path(record["origin"])
        if path.suffix != ".py":
            return None, "no_python_source"
        return self.read_file(path, record["sha256"])

    def _read_file(self, path, expected):
        try:
            source = self.imports.snapshot(path)
            if source.sha256 != expected:
                return None, "source_snapshot_changed"
            _ = source.tree
        except (OSError, SyntaxError, UnicodeError, LookupError) as error:
            return None, "source_read: " + str(error)
        return source, None

    def _bindings(self, module, name, before):
        source, _ = self.source(module)
        missing = _Binding("unbound")

        def scan(statements, state):
            for node in statements:
                if before and (node.lineno, node.col_offset) >= before:
                    continue

                def guard(branch, test=""):
                    return (module, node.lineno, branch, test)

                if isinstance(node, _DEFINITIONS):
                    if node.name == name:
                        state = [_Binding("definition", node)]
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname or alias.name.partition(".")[0]
                        if local == name:
                            state = [
                                _Binding(
                                    "module",
                                    node,
                                    alias.name if alias.asname else local,
                                )
                            ]
                elif isinstance(node, ast.ImportFrom):
                    owner = relative_module(node, module, source.package)
                    for alias in node.names:
                        if alias.name == "*":
                            other, _ = self.source(owner)
                            public = other.star_names if other else _DYNAMIC
                            if (
                                isinstance(public, set)
                                and name not in public
                                or public is _IMPLICIT
                                and name.startswith("_")
                            ):
                                continue
                            binding = _Binding("from", node, owner, (name,))
                            if isinstance(public, set):
                                state = [binding]
                            else:
                                state = [
                                    *_guarded(
                                        state, guard("star_member_absent", owner)
                                    ),
                                    *_guarded(
                                        [binding], guard("star_member_present", owner)
                                    ),
                                ]
                        elif (alias.asname or alias.name) == name:
                            state = [_Binding("from", node, owner, (alias.name,))]
                elif isinstance(
                    node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)
                ):
                    targets = (
                        node.targets
                        if isinstance(node, (ast.Assign, ast.Delete))
                        else [node.target]
                    )
                    direct = any(
                        isinstance(t, ast.Name) and t.id == name for t in targets
                    )
                    if direct or any(name in _target_names(t) for t in targets):
                        if isinstance(node, ast.Delete):
                            state = [_Binding("deleted", node)]
                        elif isinstance(node, ast.AnnAssign) and node.value is None:
                            continue
                        else:
                            rhs = (
                                node.value
                                if direct and not isinstance(node, ast.AugAssign)
                                else None
                            )
                            path = _dotted(rhs)
                            kind = "alias" if path else "computed_binding"
                            if _literal_value(rhs) is not _DYNAMIC:
                                kind = "literal"
                            state = [_Binding(kind, node, module, path)]
                    for t in targets:
                        if isinstance(t, ast.Subscript) and _namespace(t.value):
                            if isinstance(t.slice, ast.Constant):
                                if t.slice.value == name:
                                    state = [_Binding("namespace_write", node)]
                            elif (
                                t in source.string_keys
                                and name not in source.string_keys[t]
                            ):
                                continue
                            else:
                                state = [
                                    *_guarded(state, guard("namespace_write_misses")),
                                    _Binding("namespace_write", node),
                                ]
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    f = node.value.func
                    if _dotted(f) in {("exec",), ("eval",)} or (
                        isinstance(f, ast.Attribute)
                        and _namespace(f.value)
                        and f.attr
                        in {
                            "update",
                            "clear",
                            "pop",
                            "popitem",
                            "setdefault",
                            "__setitem__",
                            "__delitem__",
                        }
                    ):
                        state = [
                            *_guarded(state, guard("namespace_write_misses")),
                            _Binding("dynamic_namespace", node),
                        ]
                else:
                    branches = []
                    if isinstance(node, ast.If):
                        cond = ast.unparse(node.test)
                        arms = [
                            (guard("if", cond), node.body),
                            (guard("else", cond), node.orelse),
                        ]
                    elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                        # A handler can observe a prefix of the try body.
                        normal = scan(node.body, state)
                        alternatives = [
                            *state,
                            *normal,
                            _Binding("exceptional_prefix", node),
                        ]
                        arms = [(guard("try_completed"), [*node.body, *node.orelse])]
                        for h in node.handlers:
                            handled = scan(h.body, alternatives)
                            if normal != state or handled != alternatives:
                                branches += _guarded(
                                    handled,
                                    guard(
                                        "except", ast.unparse(h.type) if h.type else ""
                                    ),
                                )
                        if normal == state and not branches and not node.orelse:
                            state = scan(node.finalbody, state)
                            continue
                    elif isinstance(
                        node,
                        (
                            ast.For,
                            ast.AsyncFor,
                            ast.While,
                            ast.With,
                            ast.AsyncWith,
                            ast.Match,
                        ),
                    ):
                        bodies = (
                            [c.body for c in node.cases]
                            if isinstance(node, ast.Match)
                            else [node.body, getattr(node, "orelse", [])]
                        )
                        arms = [
                            (guard(type(node).__name__.lower(), str(i)), body)
                            for i, body in enumerate(bodies)
                        ]
                        arms.append((guard("fallthrough"), []))
                    else:
                        continue
                    results = [(test, scan(body, state)) for test, body in arms]
                    if branches or any(new != state for _, new in results):
                        for test, new in results:
                            branches += _guarded(new, test)
                        state = list(dict.fromkeys(branches))
                    if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                        state = scan(node.finalbody, state)
            return state

        return scan(source.tree.body, [missing])

    def _get(self, module, member):
        candidates, unresolved = {}, []

        def walk(owner, path, before=None, active=frozenset(), guards=(), via=()):
            key = (owner, path, before)

            def issue(reason, conditions=guards, line=None, **fields):
                item = dict(
                    module=owner,
                    name=".".join(path),
                    reason=reason,
                    conditions=list(conditions),
                    **fields,
                )
                if line:
                    item["line"] = line
                if item not in unresolved:
                    unresolved.append(item)

            if key in active:
                issue("import_or_alias_cycle")
                return
            source, error = self.source(owner)
            if error:
                issue(error)
                return
            record = self.imports.record(owner)
            first, *rest = path
            for binding in self.bindings(owner, first, before):
                node = binding.node
                conditions = (
                    *guards,
                    *(
                        dict(module=m, line=l, branch=b, test=t)
                        for m, l, b, t in binding.guards
                    ),
                )
                step = dict(
                    module=owner,
                    sha256=record["sha256"],
                    member=record["member"],
                    provider_id=record["provider_id"],
                    name=first,
                    line=node.lineno if node else None,
                    kind=binding.kind,
                )
                chain = (*via, step)

                def follow(module, member, limit=None, extra=()):
                    walk(
                        module,
                        tuple(member),
                        limit,
                        active | {key},
                        (*conditions, *extra),
                        chain,
                    )

                if binding.kind in {"definition", "literal"} and not rest:
                    if binding.kind == "literal":
                        fields = _literal_source(_literal_value(node.value))
                        if "literal" not in fields:
                            issue(
                                "unsupported_literal_type",
                                conditions,
                                node.lineno,
                                **fields,
                                provider_id=step["provider_id"],
                                via=list(chain),
                            )
                            continue
                    else:
                        fields = dict(
                            locations=_statement_lines(node, source.lines),
                            decorators=[
                                dict(line=d.lineno, expression=ast.unparse(d))
                                for d in node.decorator_list
                            ],
                        )
                    identity = (
                        owner,
                        node.lineno,
                        node.col_offset,
                        first,
                        binding.kind,
                    )
                    route = dict(conditions=list(conditions), via=list(chain))
                    if identity in candidates:
                        previous = candidates[identity]
                        if any(previous[k] != v for k, v in route.items()):
                            alternatives = previous.setdefault("alternate_paths", [])
                            if route not in alternatives:
                                alternatives.append(route)
                        continue
                    item = {
                        k: record[k]
                        for k in (
                            "module",
                            "origin",
                            "member",
                            "distribution",
                            "version",
                            "sha256",
                            "provider_id",
                        )
                    }
                    item.update(
                        kind=binding.kind,
                        name=first,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        # One-based UTF-8 byte columns; end is exclusive.
                        column=node.col_offset + 1,
                        end_column=node.end_col_offset + 1,
                        **fields,
                        **route,
                    )
                    candidates[identity] = item
                elif binding.kind == "alias":
                    follow(
                        owner, (*binding.path, *rest), (node.lineno, node.col_offset)
                    )
                elif binding.kind == "from":
                    target = binding.module
                    # A from-list inspecting its own package sees only its executed prefix.
                    limit = (node.lineno, node.col_offset) if target == owner else None
                    follow(target, (*binding.path, *rest), limit)
                    submodule = ".".join((target, *binding.path))
                    if rest and self.imports.find(submodule):
                        guard = dict(
                            module=target,
                            branch="package_attribute_missing",
                            name=binding.path[0],
                        )
                        follow(submodule, rest, extra=(guard,))
                elif binding.kind == "module" and rest:
                    follow(binding.module, rest)
                else:
                    issue(binding.kind, conditions, step["line"])

        walk(module, tuple(member.split(".")))
        return dict(
            status="source_candidates",
            candidates=list(candidates.values()),
            unresolved=unresolved,
        )
