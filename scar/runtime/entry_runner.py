"""Scoped automatic attachment at ordinary imported Python source boundaries.

The numerical kernel remains admitted separately. This runner neither plans
nor profiles calls, and makes no complete alias/call-coverage claim. Entry-file
definitions, custom source loaders and partial modules remain original Python.
"""
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import importlib.abc
from importlib.machinery import PathFinder, SourceFileLoader
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

_STATEFUL = 'scar.stateful-runtime-contract'


def runner_implementation_sha256():
    package = Path(__file__).resolve().parents[1]
    names = ("runtime/entry_runner.py", "runtime/native_loop.py", "runtime/loop_guard.py",
             "runtime/loop_contract.py", "runtime/bundle_metadata.py", "runtime/entry_cli.py",
             "runtime/__init__.py", "entrypoint.py", "__main__.py", "trace/code_snapshot.py", "trace/__init__.py",
             "optimization/deployment_gate.py", "optimization/deployment_cost.py", "optimization/__init__.py")
    names += ('runtime/native_stateful.py', 'runtime/stateful_guard.py', 'runtime/stateful_contract.py')
    files = [{"path": name, "sha256": sha256((package / name).read_bytes()).hexdigest()} for name in names]
    implementation = dict(name=sys.implementation.name, cache_tag=sys.implementation.cache_tag,
                          version=list(sys.implementation.version), sys_version=sys.version)
    return sha256(json.dumps(dict(files=files, python=implementation), sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def _file(module):
    value = vars(module).get("__file__") if type(module) is ModuleType else None
    return Path(value).resolve() if type(value) is str else None


class _Loader(importlib.abc.Loader):
    def __init__(self, original, scope, name):
        self.original, self.scope, self.name = original, scope, name

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        # Preserve the ordinary source loader seen by the application's code.
        module.__loader__ = self.original
        module.__spec__.loader = self.original
        self.original.exec_module(module)
        if sys.modules.get(self.name) is module:
            self.scope.attach(module, self.name, natural=True)
        else:
            self.scope.frontier.append("source_module_replaced_itself")

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, scope): self.scope = scope

    def find_spec(self, fullname, path=None, target=None):
        # Replace the existing PathFinder position, preserving custom finders'
        # precedence and performing each ordinary path search only once.
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and type(spec.origin) is str and Path(spec.origin).resolve() == self.scope.source:
            if type(spec.loader) is SourceFileLoader:
                spec.loader = _Loader(spec.loader, self.scope, fullname)
            else:
                self.scope.frontier.append("source_uses_nonstandard_loader")
        return spec


class _Scope:
    def __init__(self, artifact):
        self.stateful = artifact.contract.get('schema') == _STATEFUL
        if self.stateful:
            from .stateful_guard import guarded_stateful_function, SourceMismatch
            make_guard = guarded_stateful_function
        else:
            from .loop_guard import guarded_loop_function, SourceMismatch
            make_guard = guarded_loop_function
        self.artifact, self.make_guard, self.mismatch = artifact, make_guard, SourceMismatch
        self.source = Path(artifact.contract["source"]["path"]).resolve()
        self.name = artifact.contract["source"]["qualname"]
        self.active, self.bindings, self.seen, self.frontier = True, [], set(), []

    def attach(self, module, name, *, natural):
        if not self.active or id(module) in self.seen: return
        self.seen.add(id(module))
        parts = self.name.split('.')
        if (type(module) is not ModuleType or _file(module) != self.source
                or any(not part.isidentifier() for part in parts)
                or not self.stateful and len(parts) != 1):
            self.frontier.append("source_has_no_supported_module_or_class_boundary"); return
        spec = vars(module).get("__spec__")
        if not natural and (spec is None or getattr(spec, "_initializing", False)
                            or type(getattr(spec, "loader", None)) is not SourceFileLoader):
            self.frontier.append("preloaded_source_not_a_completed_standard_module"); return
        owner, attribute = module, parts[-1]
        try:
            for part in parts[:-1]:
                if type(owner) not in (ModuleType, type):
                    raise ValueError('custom boundary owner')
                owner = vars(owner)[part]
            if type(owner) not in (ModuleType, type):
                raise ValueError('custom boundary owner')
            descriptor = vars(owner).get(attribute)
            if type(descriptor) is staticmethod:
                original, wrap, receiver_owner = descriptor.__func__, staticmethod, None
            else:
                original, wrap = descriptor, lambda value: value
                receiver_owner = owner if type(owner) is type else None
            if self.stateful:
                guarded = self.make_guard(original, self.artifact,
                    owner=receiver_owner, exclusive_inputs=True)
            else:
                guarded = self.make_guard(original, self.artifact, exclusive_inputs=True)
        except (self.mismatch, ValueError, KeyError) as error:
            self.frontier.append("source_guard_rejected:" + str(error)); return
        scope = self
        @wraps(original)
        def binding(*args, **kwargs):
            return guarded(*args, **kwargs) if scope.active else original(*args, **kwargs)
        binding.scar_stats = guarded.scar_stats
        binding.scar_artifact = self.artifact
        installed = wrap(binding)
        setattr(owner, attribute, installed)
        self.bindings.append(dict(module=module, module_name=name, original=original,
            owner=owner, attribute=attribute, descriptor=descriptor, installed=installed,
            binding=binding, stats=guarded.scar_stats, natural_import=natural))

    def finish(self):
        self.active = False
        replacements = {id(row["binding"]): (row["binding"], row["original"]) for row in self.bindings}
        # Undo direct module-global from-import aliases. Escaped references in
        # containers/closures become inert original-call-throughs after scope.
        for module in tuple(sys.modules.values()):
            if type(module) is not ModuleType: continue
            namespace = vars(module)
            for name, value in tuple(namespace.items()):
                pair = replacements.get(id(value))
                if pair is not None and value is pair[0]: namespace[name] = pair[1]
        for row in self.bindings:
            if vars(row['owner']).get(row['attribute']) is row['installed']:
                setattr(row['owner'], row['attribute'], row['descriptor'])


@contextmanager
def _script_state(entry, argv, scope):
    old_argv, old_path, old_meta = sys.argv, sys.path, sys.meta_path
    argv_values, path_values, meta_values = list(old_argv), list(old_path), list(old_meta)
    sys.argv = [str(entry), *argv]
    sys.path = [str(entry.parent), *path_values[1:]]
    try:
        if scope is not None:
            if sum(finder is PathFinder for finder in meta_values) == 1:
                proxy = _Finder(scope)
                sys.meta_path = [proxy if finder is PathFinder else finder for finder in meta_values]
                for name, module in tuple(sys.modules.items()):
                    if type(module) is ModuleType and _file(module) == scope.source:
                        scope.attach(module, name, natural=False)
            else:
                scope.frontier.append("standard_pathfinder_position_unavailable")
        yield
    finally:
        try:
            if scope is not None: scope.finish()
        finally:
            old_argv[:] = argv_values; old_path[:] = path_values; old_meta[:] = meta_values
            sys.argv, sys.path, sys.meta_path = old_argv, old_path, old_meta


def _execute(entry, argv, scope, reason):
    exited = False
    with _script_state(entry, argv, scope):
        try:
            runpy.run_path(str(entry), run_name="__main__")
        except SystemExit as error:
            code = error.code
            if code is not None and not (isinstance(code, int) and int.__int__(code) == 0): raise
            exited = True
    stats = dict(native_calls=0, fallback_calls=0, reasons={})
    attachments = []
    if scope is not None:
        for row in scope.bindings:
            stats["native_calls"] += row["stats"]["native_calls"]
            stats["fallback_calls"] += row["stats"]["fallback_calls"]
            for key, count in row["stats"]["reasons"].items():
                stats["reasons"][key] = stats["reasons"].get(key, 0) + count
            attachments.append(dict(source=str(scope.source), module=row["module_name"],
                                    natural_import=row["natural_import"]))
    return dict(schema="scar.entry-runner-execution", schema_version=1, entry_path=str(entry), argv=list(argv),
        decision="GUARDED_ATTACHMENT" if attachments else "NO_OP", reason=reason,
        statistics=stats, installed_bindings=len(attachments),
        observed_boundary_calls=stats["native_calls"] + stats["fallback_calls"],
        boundary_calls_observation_complete=False,
        boundary_observation_scope="INSTALLED_BINDINGS_ONLY_NO_COMPLETE_TOP_LEVEL_OR_ESCAPED_ALIAS_CALL_TRACE",
        attachments=attachments, frontier=list(scope.frontier) if scope is not None else [reason],
        system_exit_zero=exited, exit_code=0, full_program_cpp=False)


def _arguments(entry, argv):
    entry = Path(entry).resolve()
    if not entry.is_file() or entry.suffix != ".py": raise ValueError("a regular Python .py entry is required")
    argv = list(argv)
    if any(type(value) is not str for value in argv): raise ValueError("entry argv must contain exact strings")
    return entry, argv


def run_original_script(entry, argv):
    entry, argv = _arguments(entry, argv)
    return _execute(entry, argv, None, "original_entry_requested")


def run_script(entry, argv, *, bundle, expected_admission_sha256, exclusive_inputs=False, native_enabled=True):
    entry, argv = _arguments(entry, argv)
    if native_enabled is not True:
        return _execute(entry, argv, None, "native_disabled_by_policy")
    if exclusive_inputs is not True:
        return _execute(entry, argv, None, "exclusive_inputs_not_explicitly_true")
    from .bundle_metadata import inspect_bundle
    metadata = inspect_bundle(bundle, expected_admission_sha256=expected_admission_sha256)
    if Path(metadata["contract"]["source"]["path"]).resolve() == entry:
        return _execute(entry, argv, None, "entry_defined_loop_main_installation_not_supported")
    if metadata['contract'].get('schema') == _STATEFUL:
        from .native_stateful import load
    else:
        from .native_loop import load
    artifact = load(bundle, expected_admission_sha256=expected_admission_sha256)
    return _execute(entry, argv, _Scope(artifact), "ordinary_import_source_boundary")


__all__ = ["run_script", "run_original_script", "runner_implementation_sha256"]
