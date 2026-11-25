"""M2: bind verified source host APIs to language-independent effect contracts."""

from contextlib import contextmanager
import math

from scar.contract import (
    EXECUTION_ANNOTATIONS,
    digest,
    json_bytes,
    python_provider_id,
    validate_source_literal,
    validate_python_entry_environment,
    walk_operations,
)
from . import enumerations, namespace_operations, scalars, sequences, text
from .values import Value, argument_evidence, literal, sequence_kind, source_site, wire

_STREAM = "handle:text_stream"
_REQUIRED = object()
# Names and defaults belong to source binding. The emitted interfaces do not.
# Each item is (source parameter, value type, default); None is a real default.
_SIGNATURES = {
    "os.path.exists": ("fs.exists", "bool", [("path", "text", _REQUIRED)]),
    "os.path.isfile": ("fs.is_file", "bool", [("path", "text", _REQUIRED)]),
    "os.path.isdir": ("fs.is_dir", "bool", [("path", "text", _REQUIRED)]),
    "os.mkdir": (
        "fs.mkdir",
        "unit",
        [("path", "text", _REQUIRED), ("mode", "i64", 0o777), ("dir_fd", "unit", None)],
    ),
    "os.makedirs": (
        "fs.makedirs",
        "unit",
        [
            ("name", "text", _REQUIRED),
            ("mode", "i64", 0o777),
            ("exist_ok", "bool", False),
        ],
    ),
    "os.chdir": ("process.chdir", "unit", [("path", "text", _REQUIRED)]),
    "os.getcwd": ("process.getcwd", "text", []),
    "shutil.rmtree": (
        "fs.remove_tree",
        "unit",
        [("path", "text", _REQUIRED), ("ignore_errors", "bool", False)],
    ),
    "shutil.copyfile": (
        "fs.copy_file",
        "text",
        [
            ("src", "text", _REQUIRED),
            ("dst", "text", _REQUIRED),
            ("follow_symlinks", "bool", True),
        ],
    ),
    "shutil.copy": (
        "fs.copy",
        "text",
        [
            ("src", "text", _REQUIRED),
            ("dst", "text", _REQUIRED),
            ("follow_symlinks", "bool", True),
        ],
    ),
    "shutil.copy2": (
        "fs.copy2",
        "text",
        [
            ("src", "text", _REQUIRED),
            ("dst", "text", _REQUIRED),
            ("follow_symlinks", "bool", True),
        ],
    ),
    "shutil.copytree": (
        "fs.copy_tree",
        "text",
        [("src", "text", _REQUIRED), ("dst", "text", _REQUIRED)],
    ),
    "time.time": ("time.wall", "f64", []),
    "time.sleep": ("time.sleep", "unit", [("secs", "f64", _REQUIRED)]),
}
_METHODS = {
    "write": ("stream.write", "i64", [("s", "text", _REQUIRED)]),
    "flush": ("stream.flush", "unit", []),
    "close": ("stream.close", "unit", []),
    "read": ("stream.read", "text", [("size", "i64", -1)]),
    "readline": ("stream.readline", "text", [("size", "i64", -1)]),
}
_MEMBERS = {
    "enum": {"Enum"},
    "glob": {"glob"},
    "importlib": {"import_module"},
    "os": {"path", "mkdir", "makedirs", "chdir", "getcwd"},
    "os.path": {"join", "exists", "isfile", "isdir"},
    "shutil": {"rmtree", "copyfile", "copy", "copy2", "copytree"},
    "time": {"time", "sleep"},
    "sys": {"exit"},
    "math": {"pi", "e", "tau"}
    | scalars.MATH_FUNCTIONS.keys()
    | scalars.MATH_CLASSIFICATIONS,
    "subprocess": {"Popen", "PIPE", "STDOUT", "CalledProcessError"},
}

_GLOB_PROFILE = "cpython-3.12.3-posix-glob-v1"
_GLOB_SHA = "c9e5f9ae0752660ede63328a456f58f87c29500b31f58c1b813458b00fceb6d5"
# Source files defining the qualified matcher and module initialization. Builtins
# and frozen modules must belong to one captured image of the named interpreter.
_GLOB_IMPORTS = {
    "contextlib": (
        "path",
        "8b7a477f978a8532852fd81e241c78182516bc4975d672d580a5848a76e11eb6",
    ),
    "os": ("frozen", None),
    "re": ("path", "8ff3c37c63b917fcf8dc8d50993a502292a3dc159e41de4f4018c72a53d1c07b"),
    "fnmatch": (
        "path",
        "6683da36e47af523f3f41e18ad244d837783e19e98911cc0b7415dea81494ebc",
    ),
    "itertools": ("builtin", None),
    "stat": ("frozen", None),
    "sys": ("builtin", None),
}


def _glob_profile(frame, node, source):
    if not (
        source.get("interpreter") == "cpython 3.12.3"
        and source.get("platform") == "linux"
        and source.get("os_name") == "posix"
        and source.get("filesystem_encoding") == "utf-8"
        and source.get("filesystem_errors") == "surrogateescape"
        and source.get("loader") == "path"
        and source.get("hash_basis") == "provider_file"
        and source.get("sha256") == _GLOB_SHA
    ):
        frame.fail(
            node,
            "glob_source_profile",
            f"glob requires the audited {_GLOB_PROFILE} source and filesystem encoding",
        )
    direct = source.get("direct_imports", {})
    if (
        direct.get("scope") != "absolute_module_suite"
        or direct.get("order") != list(_GLOB_IMPORTS)
        or set(direct.get("modules", {})) != _GLOB_IMPORTS.keys()
    ):
        frame.fail(
            node,
            "glob_import_sources",
            "glob requires ordered source evidence for its direct module imports",
        )
    image_hashes = set()
    for name, (loader, digest) in _GLOB_IMPORTS.items():
        dep = direct["modules"][name]
        sha = dep.get("sha256", "")
        same_source = (
            dep.get("module") == name
            and dep.get("status") == "verified_stdlib"
            and dep.get("interpreter") == source["interpreter"]
            and dep.get("platform") == source["platform"]
            and dep.get("loader") == loader
            and dep.get("proof_scope") == "resolved_module"
        )
        if loader == "path":
            correct = (
                bool(dep.get("origin"))
                and dep.get("hash_basis") == "provider_file"
                and sha == digest
            )
        else:
            origin = "built-in" if loader == "builtin" else "frozen"
            correct = (
                dep.get("origin") == origin
                and dep.get("hash_basis") == "interpreter_image"
                and isinstance(sha, str)
                and len(sha) == 64
                and all(c in "0123456789abcdef" for c in sha)
            )
            image_hashes.add(sha)
        if not (same_source and correct):
            frame.fail(
                node,
                "glob_import_sources",
                f"glob dependency {name} lacks its audited standard-library identity",
            )
    if len(image_hashes) != 1:
        frame.fail(
            node,
            "glob_import_sources",
            "glob dependencies refer to conflicting interpreter images",
        )


def _glob_operator(frame, node, op, kind, reads, writes, **fields):
    frame.project.require_operator("filesystem_glob", op, profile=_GLOB_PROFILE)
    return dict(
        op=op,
        operator_family="filesystem_glob",
        profile=_GLOB_PROFILE,
        type=wire(kind),
        effects=["IO", "STATE", "MEM"],
        may_raise=True,
        reads=reads,
        writes=writes,
        **source_site(node),
        **fields,
    )


def _glob_import(frame, node, source, steps):
    _glob_profile(frame, node, source)
    operation = _glob_operator(
        frame,
        node,
        "fs_glob_import",
        "unit",
        [
            "module.registry",
            "process.import_hooks",
            "process.audit_hooks",
            "filesystem",
        ],
        [
            "module.registry",
            "process.audit_events",
            "import.external_effects",
            "re.cache",
        ],
        scope="process_module_registry",
        module="glob",
        source_protocol="python.import",
        source_provider="stdlib:glob",
        source_sha256=source["sha256"],
        source_effects=node.get("effects", ["UNKNOWN"]),
        binding={
            "request": node["request"]["form"],
            "member": node.get("member"),
            "lookup": "source_import_protocol",
            "cached": "reuse_verified_module",
            "unmodified_exports": ["glob.glob"],
        },
        initialization={
            "policy": "initialize_once_on_success",
            "cold_imports": list(_GLOB_IMPORTS),
            "module_suite": [
                "bind_exports",
                "compile_magic_regex_text_and_bytes",
                "read_directory_open_flags",
            ],
            "cold_import_effects": "preserve_dependency_initializers_hooks_and_failures",
            "identity_guard": "pristine_registry_or_matching_verified_bindings",
            "transitive_imports": "preserve_provider_semantics",
        },
    )
    operation["effects"] = list(
        dict.fromkeys(operation["source_effects"] + operation["effects"])
    )
    frame.effect(operation, "unit", steps, True)


def _glob(frame, node, args, keywords, steps, discard):
    if len(args) > 1:
        frame.fail(node, "glob_arguments", "only pathname is positional in glob.glob")
    pattern, _, _, recursive, hidden = _bind(
        frame,
        node,
        args,
        keywords,
        [
            ("pathname", "text", _REQUIRED),
            ("root_dir", "unit", None),
            ("dir_fd", "unit", None),
            ("recursive", "bool", False),
            ("include_hidden", "bool", False),
        ],
    )
    if recursive.exact is not False or hidden.exact is not False:
        frame.fail(
            node,
            "glob_options",
            "this glob profile requires recursive=False and include_hidden=False",
        )
    operation = _glob_operator(
        frame,
        node,
        "fs_glob",
        sequence_kind("text"),
        [
            "process.cwd",
            "filesystem",
            "process.audit_hooks",
            "module.registry",
            "fnmatch.cache",
        ],
        [
            "new.sequence",
            "process.audit_events",
            "audit_hook.external_effects",
            "filesystem.access_times",
            "fnmatch.cache",
            "re.cache",
        ],
        interface="fs.glob",
        source_provider="stdlib:glob",
        pattern=frame.scalar(pattern, node),
        options=dict(root_dir=None, dir_fd=None, recursive=False, include_hidden=False),
        result=dict(
            cardinality="runtime",
            allocation="fresh_list",
            order="source_scandir_traversal",
            duplicates="preserve",
        ),
        semantics=dict(
            path="posix_split_join_preserve_input_spelling",
            pattern="fnmatch_unicode_codepoints_per_component_case_sensitive",
            expansion="ordered_directory_components_without_recursive_double_star",
            hidden="wildcard_component_requires_literal_leading_dot_for_hidden_name",
            literal="lexists_including_dangling_symlink",
            trailing_separator="directories_only",
            directory_symlinks="follow_for_directory_descent",
            spelling="no_shell_tilde_or_environment_expansion",
            directory_snapshot="materialize_each_scandir_in_source_order_before_filter",
            errors="skip_entry_or_directory_oserror_preserve_partial_listing",
            literal_errors="posixpath_lexists_isdir_return_false_on_oserror_or_valueerror",
            other_errors="preserve_source_exception_at_originating_operation",
            audit=["glob.glob", "glob.glob/2", "os.scandir_per_attempt"],
            audit_failures="preserve_at_the_source_exception_boundary",
            filesystem_encoding="utf-8",
            filesystem_errors="surrogateescape",
        ),
        **argument_evidence([pattern]),
    )
    return sequences.produced(
        frame,
        node,
        "text",
        operation,
        steps,
        text_order=sequences.text_order_atom(pattern),
        discard=discard,
    )


def supports_import(module):
    """Whether this layer has a namespace and initialization contract for a module."""
    return module in _MEMBERS


def array_effect_free(target):
    """These closed interfaces may affect IO/state, but never write array storage."""
    if target.kind == "host_method":
        return target.value[1] in _METHODS
    return target.kind == "host_callable" and target.value in (
        _SIGNATURES.keys()
        | {
            "builtins.open",
            "builtins.input",
            "builtins.str",
            "builtins.repr",
            "builtins.ascii",
            "builtins.format",
            "sys.exit",
            "os.path.join",
            "os.makedirs",
            "os.chdir",
            "os.getcwd",
        }
        | {
            "math." + name
            for name in scalars.MATH_FUNCTIONS.keys() | scalars.MATH_CLASSIFICATIONS
        }
    )


def readonly_callable(base, name):
    """Resolve an array-readonly namespace member without executing an operation."""
    if name not in _MEMBERS.get(base.value, ()):
        return None
    candidate = Value("host_callable", base.value + "." + name, base.facts)
    return candidate if array_effect_free(candidate) else None


def builtin(frame, node, name):
    if name in {"open", "str", "repr", "ascii", "format", "input"}:
        return Value(
            "host_callable", "builtins." + name, node.get("source_runtime", {})
        )
    return None


def import_effects(occurrence, *, origin=None):
    """Describe an import's named ports separately from unconstrained callbacks.

    Source identity qualifies which protocol is requested, never the application's
    cache state or the absence of initializer, audit or attribute-hook effects.
    """
    module = occurrence["resolved_module"]
    parts = module.split(".")
    direct = [".".join(parts[:index]) for index in range(1, len(parts) + 1)]
    conditional = [
        module + "." + name
        for name in occurrence["request"].get("fromlist", ())
        if isinstance(name, str) and name.isidentifier()
    ]
    identity = dict(occurrence.get("import_effects", {}).get("source_identity", {}))
    if origin is not None:
        identity = {
            "provider_id": python_provider_id(origin),
            **{
                key: origin[key]
                for key in ("module", "status", "sha256", "proof_scope")
                if key in origin
            },
        }
    return dict(
        semantics_version=1,
        scope="source_import_occurrence",
        occurrence=occurrence["id"],
        module=module,
        direct_registry_keys=direct,
        conditional_registry_keys=list(dict.fromkeys(conditional)),
        namespace_writes=dict(
            direct=direct,
            conditional=list(dict.fromkeys(conditional)),
            other="UNKNOWN",
        ),
        other_registry_writes="UNKNOWN",
        external_effect="UNKNOWN",
        reads=[
            "module.registry",
            "module.namespace",
            "process.import_hooks",
            "process.audit_hooks",
            "filesystem",
        ],
        writes=[
            "requested_registry_entries",
            "module.namespace",
            "import.external_effects",
        ],
        source_effects=occurrence["source_effects"],
        producer_sites=list(occurrence["producer_sites"]),
        effect_coverage=occurrence["effect_coverage"],
        source_identity=identity,
        completion="normal_return_of_original_import_protocol",
        obligations=list(occurrence["obligations"]),
    )


def _refresh_import_effects(occurrence, *, origin=None):
    result = import_effects(occurrence, origin=origin)
    shared = occurrence.setdefault("import_effects", {})
    shared.clear()
    shared.update(result)
    return shared


class ImportOccurrences:
    """One source IMPORT_NAME, its individual bindings and its effect producers."""

    def __init__(self, functions, *, entry_environment=None):
        if entry_environment is not None:
            validate_python_entry_environment(entry_environment)
        self.entry_environment = entry_environment
        self.observed_bindings = {}
        self.observed_requests = {}
        sample = (entry_environment or {}).get("import_binding_sample", {})
        for event in sample.get("events", ()):
            if "site" not in event:
                continue
            if event.get("phase") == "before_IMPORT_NAME":
                self.observed_requests.setdefault(event["site"], []).append(
                    {
                        key: event[key]
                        for key in (
                            "code_sha256",
                            "instruction",
                            "import_callable",
                            "registry_state",
                        )
                    }
                )
            else:
                self.observed_bindings.setdefault(event["site"], []).append(
                    event["result"]
                )
        self.rows = {}
        self.nodes = {}
        self.last = {}
        self.initialization_points = {}
        self.provider_region = None
        self.comparisons = {}
        for owner, fn in functions.items():
            work = [fn.get("body", [])]
            while work:
                part = work.pop()
                if isinstance(part, list):
                    work.extend(reversed(part))
                elif isinstance(part, dict):
                    if part.get("op") == "python_import":
                        self.get(owner, part)
                    else:
                        work.extend(reversed(tuple(part.values())))

    def get(self, owner, node):
        key = (owner, node.get("site") or id(node))
        if key in self.nodes:
            return self.nodes[key][1]
        req = node.get("request", {})
        request = {
            k: req.get(k, default)
            for k, default in (
                ("form", None),
                ("module", node.get("module")),
                ("level", 0),
                ("fromlist", []),
            )
        }
        site = node.get("site") or f"{owner}/import/{len(self.nodes)}"
        previous = self.last.get(owner, {})
        index = req.get("binding_index", 0)
        if (
            request["form"] == "from"
            and index
            and previous.get("request") == request
            and previous.get("location") == node.get("location")
            and len(previous["bindings"]) == index
        ):
            row = previous
        else:
            identity = digest(json_bytes([owner, site, request]))
            source = node.get("import_resolution") or node.get("source_provider") or {}
            row = self.rows.setdefault(
                identity,
                dict(
                    id=identity,
                    owner=owner,
                    site=site,
                    **({"location": node["location"]} if node.get("location") else {}),
                    request=request,
                    resolved_module=node.get("module"),
                    source_provider_id=python_provider_id(source),
                    source_effects=node.get("effects", ["UNKNOWN"]),
                    producer_sites=[],
                    bindings=[],
                    effect_coverage="UNPROVEN",
                    activation="unvisited",
                    obligations=[
                        "cache_or_cold_initialization",
                        "import_hooks_audit_and_failures",
                    ],
                ),
            )
        binding = dict(
            name=node.get("binding", {}).get("name"),
            member=node.get("member"),
            site=site,
        )
        if site in self.observed_requests:
            row["observed_before_import"] = self.observed_requests[site]
        if site in self.observed_bindings:
            binding["observed_after_store"] = self.observed_bindings[site]
        if binding not in row["bindings"]:
            row["bindings"].append(binding)
        self.last[owner] = row
        self.nodes[key] = (node, row)
        _refresh_import_effects(
            row,
            origin=node.get("import_resolution") or node.get("source_provider") or {},
        )
        return row

    def begin(self, row, node, steps):
        if row["id"] not in self.initialization_points:
            marker = dict(
                op="block", body=[], pending_import_effect=True, **source_site(node)
            )
            self.initialization_points[row["id"]] = (marker, node)
            steps.append(marker)

    def _record_provider_operations(self, row):
        if self.provider_region is None or self.provider_region[0] is not row:
            return
        _, steps, start = self.provider_region
        for operation in walk_operations(
            steps[start:], exclude_fields=EXECUTION_ANNOTATIONS
        ):
            if (
                operation.get("source_protocol") == "python.import"
                and "source_import_occurrence" not in operation
            ):
                self.producer(row, operation, "source_import")

    @contextmanager
    def external_request(self, frame, row, node, origin, steps, *, primary):
        """Resolve provider bindings while retaining the one source loader, even on a partial prefix."""
        if primary:
            self.begin(row, node, steps)
        previous = self.provider_region
        self.provider_region = (row, steps, len(steps))
        try:
            yield
        finally:
            try:
                self._record_provider_operations(row)
                source = (
                    origin
                    if origin.get("module") == node["module"]
                    else origin.get("aliases", {}).get(node["module"], {})
                )
                if primary and not row["producer_sites"] and source.get("sha256"):
                    retain_import_result(
                        frame.project,
                        row,
                        self.initialization_points[row["id"]],
                        dict(
                            module=node["module"],
                            source_provider_id=python_provider_id(source),
                            source_sha256=source["sha256"],
                        ),
                    )
            finally:
                self.provider_region = previous

    def producer(self, row, operation, role):
        operation.setdefault("site", row["site"])
        operation["source_import_occurrence"] = row["id"]
        item = dict(
            owner=row["owner"],
            site=operation.get("site", row["site"]),
            opcode=operation["op"],
            role=role,
            **{
                k: operation[k] for k in ("operator_family", "callee") if k in operation
            },
        )
        if operation.get("namespace_binding", {}).get(
            "retains"
        ) == "actual_IMPORT_NAME_result" and operation.get("state_capture"):
            item["captures_import_result_and_state"] = True
        if item not in row["producer_sites"]:
            row["producer_sites"].append(item)
        row["effect_coverage"] = "MODELED" if role == "source_import" else "PARTIAL"
        if role == "source_import":
            row["obligations"] = ["qualify_equivalent_source_import_on_target"]
            if item.get("captures_import_result_and_state"):
                row["obligations"].append("qualify_ordered_source_bindings_and_release")
        operation["import_effects"] = _refresh_import_effects(row)
        for comparison in self.comparisons.get(row["id"], ()):
            comparison["occurrence_effect_coverage"] = row["effect_coverage"]

    def comparison(self, row, operation):
        self.comparisons.setdefault(row["id"], []).append(operation)

    def empty_markers(self):
        return {
            id(marker)
            for marker, _ in self.initialization_points.values()
            if not marker["body"]
        }

    def product(self, skipped, omitted):
        for row in self.rows.values():
            if row["activation"] == "unvisited" and (
                row["owner"] in omitted
                or all((row["owner"], b["site"]) in skipped for b in row["bindings"])
            ):
                row["activation"] = "proven_inactive"
        return dict(
            semantics_version=1,
            scope="source_import_occurrences",
            occurrences=list(self.rows.values()),
            **(
                {"entry_environment": self.entry_environment}
                if self.entry_environment is not None
                else {}
            ),
        )


def registry_import(frame, node, origin, occurrence):
    """Separate the loader's named cache ports from unconstrained callbacks."""
    name = node["module"]
    stored = frame.provider_sources.get(name, {})
    if not (
        origin.get("status") in {"verified_stdlib", "verified_distribution"}
        and origin.get("proof_scope")
        in {"resolved_module", "installed_file_with_digest"}
        and stored.get("provider_id") == python_provider_id(origin)
        and stored.get("sha256") == origin.get("sha256")
        and stored.get("sha256")
    ):
        return None
    effects = _refresh_import_effects(occurrence, origin=origin)
    operation = dict(
        op="module_import_preserves",
        type="bool",
        operator_family="module_registry",
        source_protocol="module_state_comparison",
        occurrence=occurrence["id"],
        producer_sites=occurrence["producer_sites"],
        occurrence_effect_coverage=occurrence["effect_coverage"],
        module=name,
        request=node["request"],
        source_provider_id=stored["provider_id"],
        source_sha256=stored["sha256"],
        direct_registry_keys=effects["direct_registry_keys"],
        conditional_registry_keys=effects["conditional_registry_keys"],
        import_effects=effects,
        effects=["CTRL", "STATE"],
        may_raise=False,
        reads=["occurrence.before", "occurrence.after"],
        writes=[],
        ordering_effect="read_completed_occurrence_before_normal_continuation",
        checked_interval="state_immediately_before_and_after_this_import",
        checks={},
        **source_site(node),
    )
    frame.project.source_imports.comparison(occurrence, operation)
    return operation


def source_namespace(frame, node, module):
    occurrence = frame.project.source_imports.get(frame.name, node)
    source = frame.provider_sources.get(module, {})
    if (
        source.get("status") not in {"verified_stdlib", "verified_distribution"}
        or not source.get("sha256")
        or source.get("provider_id") != python_provider_id(source)
    ):
        frame.fail(
            node,
            "source_namespace",
            f"{module} has no sealed source namespace identity",
        )
    return dict(
        module=module,
        occurrence=occurrence["id"],
        source_provider_id=source["provider_id"],
        source_sha256=source["sha256"],
        source=source,
    )


def literal_exports(frame, module):
    source = frame.provider_sources.get(module, {})
    return any(
        candidate.get("kind") == "literal"
        for export in source.get("exports", {}).values()
        for candidate in export.get("candidates", ())
    )


def source_provider(frame, node, origin, steps):
    namespace = Value(
        "source_namespace",
        node["module"],
        {"source_namespace": source_namespace(frame, node, node["module"])},
    )
    if node["request"]["form"] == "from":
        return source_export(
            frame, node, namespace.facts["source_namespace"], node["member"], steps
        )
    return namespace


def _initialize_source_namespace(frame, node, namespace):
    ledger = frame.project.source_imports
    row = ledger.rows[namespace["occurrence"]]
    ledger._record_provider_operations(row)
    if any(producer["role"] == "source_import" for producer in row["producer_sites"]):
        return row
    point = ledger.initialization_points.get(row["id"])
    if point is None:
        frame.fail(
            node,
            "source_import_position",
            "the namespace lookup has no preceding source IMPORT_NAME position",
        )
    return retain_import_result(frame.project, row, point, namespace)


def retain_import_result(project, row, point, namespace):
    """Insert the actual loader call at the source occurrence, shared by its users."""
    marker, original = point
    operation = dict(
        op="module_namespace_initialize",
        type="unit",
        operator_family="module_registry",
        semantics_version=1,
        source_protocol="python.import",
        module=namespace["module"],
        source_provider_id=namespace["source_provider_id"],
        source_sha256=namespace["source_sha256"],
        request=row["request"],
        effects=["CTRL", "STATE", "IO", "MEM", "OPAQUE"],
        source_effects=row["source_effects"],
        may_raise=True,
        reads=[
            "module.registry",
            "process.import_hooks",
            "process.audit_hooks",
            "filesystem",
        ],
        writes=[
            "requested_registry_entries",
            "module.namespace",
            "import.external_effects",
        ],
        allocates="UNKNOWN",
        frees="UNKNOWN",
        aliases="UNKNOWN",
        escapes="UNKNOWN",
        rng_effect="UNKNOWN",
        external_effect="UNKNOWN",
        ordering_effect="source_import_order",
        namespace_binding=dict(
            occurrence=row["id"],
            instance_scope="per_successful_dynamic_import",
            retains="actual_IMPORT_NAME_result",
            consumers="following_IMPORT_FROM_lookups_or_plain_store_consume_the_returned_object_in_source_order",
            qualification="require_actual_source_lookup_store_release_and_error_paths_for_each_binding",
            bindings=row["bindings"],
        ),
        state_capture=dict(
            occurrence=row["id"],
            before="immediately_before_original_IMPORT_NAME",
            after="on_normal_return_with_original_result_before_following_binding",
            exceptional="propagate_without_check_or_retry",
        ),
        initialization=dict(
            policy="one_original_IMPORT_NAME_at_its_source_position",
            fromlist="perform_complete_source_fromlist_precheck_before_any_IMPORT_FROM",
            hooks="preserve_import_callbacks_audit_and_failures",
            unknown_namespace="do_not_assume_candidate_is_current_binding",
        ),
        **source_site(original),
    )
    namespace_operations.require_registry_operator(project, operation["op"])
    project.source_imports.producer(row, operation, "source_import")
    marker["body"].append(dict(op="expr", value=operation))
    return row


def source_export(frame, node, namespace, name, steps):
    if not namespace:
        frame.fail(
            node,
            "source_namespace",
            "the source namespace binding was not preserved across this activation",
        )
    source = namespace["source"]
    export = source.get("exports", {}).get(name, {})
    candidates = [
        candidate
        for candidate in export.get("candidates", ())
        if candidate.get("kind") == "literal"
    ]
    if export.get("status") != "source_candidates" or not candidates:
        frame.fail(
            node,
            "source_export",
            f"no literal source candidate for {namespace['module']}.{name}",
        )
    try:
        expected = validate_source_literal(candidates[0])
        kind = expected["type"]
        if kind not in {"unit", "bool", "i64", "f64", "text"}:
            raise ValueError(
                "source literal needs another native representation contract"
            )
        for candidate in candidates:
            if (
                json_bytes(validate_source_literal(candidate)) != json_bytes(expected)
                or candidate["via"][0].get("provider_id")
                != namespace["source_provider_id"]
                or candidate["via"][0].get("sha256") != namespace["source_sha256"]
                or any(
                    not step.get("provider_id") or not step.get("sha256")
                    for step in candidate["via"]
                )
            ):
                raise ValueError(
                    "inconsistent source literal alternatives or export path"
                )
    except ValueError as error:
        frame.fail(
            node,
            "source_export_representation",
            str(error),
        )
    value = expected["value"]
    row = _initialize_source_namespace(frame, node, namespace)
    capture = digest(
        json_bytes(
            [frame.name, node.get("site"), source_site(node), frame.temporary(), name]
        )
    )
    common = dict(
        operator_family="module_registry",
        semantics_version=1,
        capture=capture,
        instance_scope="per_dynamic_lookup_in_current_activation",
        namespace_binding=dict(
            occurrence=row["id"],
            source_provider_id=namespace["source_provider_id"],
            source_sha256=namespace["source_sha256"],
            module=namespace["module"],
        ),
        member=name,
        lookup=("IMPORT_FROM" if node.get("op") == "python_import" else "attribute"),
        **source_site(node),
    )
    operation = dict(
        op="module_export_matches",
        type="bool",
        **common,
        candidate=dict(
            source_type=candidates[0]["source_type"],
            literal=expected,
            provenance=candidates,
            unresolved=export.get("unresolved", []),
            **({"on_missing": export["on_missing"]} if "on_missing" in export else {}),
        ),
        checks=[
            "capture_actual_source_lookup_once",
            "exact_builtin_source_type",
            "binary64_bits" if kind == "f64" else "exact_primitive_value",
        ],
        lookup_effects="preserve___getattr___exceptions_and_IMPORT_FROM_fallback_without_repeating_lookup",
        effects=["CTRL", "STATE", "IO", "MEM", "OPAQUE"],
        may_raise=True,
        reads=["source.import_binding", "module.namespace"],
        writes=["lookup.capture", "import.external_effects"],
        allocates="UNKNOWN",
        frees="UNKNOWN",
        aliases="UNKNOWN",
        escapes="UNKNOWN",
        rng_effect="UNKNOWN",
        external_effect="UNKNOWN",
    )
    for op in (
        "module_export_matches",
        "module_export_value",
        "module_export_continuation_required",
    ):
        namespace_operations.require_registry_operator(frame.project, op)
    matched = frame.effect(operation, "bool", steps)
    steps.append(
        dict(
            op="if",
            condition=matched.value,
            then=[],
            **{
                "else": [
                    dict(
                        op="expr",
                        value=dict(
                            op="module_export_continuation_required",
                            type="void",
                            **common,
                            noreturn=True,
                            effects=["CTRL", "OPAQUE"],
                            reads=["lookup.capture", "source.continuation"],
                            writes=["source.continuation"],
                            external_effect="UNKNOWN",
                            semantics="transfer_to_source_equivalent_continuation_with_the_captured_actual_object_not_raise_or_repeat_lookup",
                        ),
                    )
                ]
            },
        )
    )
    facts = {
        **literal(value).facts,
        "source_export_binding": dict(
            normal_edge="exact_source_type_and_value_checked",
            identity="same_captured_object_requires_identity_protocol_if_observed",
        ),
    }
    return frame.effect(
        dict(
            op="module_export_value",
            type=wire(kind),
            **common,
            effects=["STATE"],
            reads=["lookup.capture"],
            writes=[],
            projection="actual_captured_object_after_successful_match_not_constant_rematerialization",
        ),
        kind,
        steps,
        facts=facts,
    )


def _importlib_namespace(frame, node, source, steps):
    stored = frame.provider_sources.get("importlib", {})
    if not (
        source.get("interpreter", "").startswith("cpython 3.")
        and source.get("proof_scope") == "resolved_module"
        and source.get("loader") == "path"
        and source.get("hash_basis") == "provider_file"
        and source.get("origin")
        and stored.get("sha256") == source["sha256"]
        and stored.get("origin") == source["origin"]
        and stored.get("provider_id") == python_provider_id(source)
    ):
        frame.fail(
            node,
            "importlib_source",
            "importlib requires a sealed CPython standard-library source and matching unit provenance",
        )
    family, operation = "module_registry", "module_namespace_initialize"
    previous = frame.project.imports.registry_entry("importlib")
    known = (
        previous.get("state") == "ready"
        and previous.get("source_provider_id") == stored["provider_id"]
    )
    namespace_operations.require_registry_operator(frame.project, operation)
    frame.effect(
        dict(
            op=operation,
            operator_family=family,
            type="unit",
            semantics_version=1,
            source_protocol="python.import",
            scope="process_module_registry",
            module="importlib",
            source_provider_id=stored["provider_id"],
            source_sha256=source["sha256"],
            interpreter=source["interpreter"],
            request=node["request"],
            effects=list(
                dict.fromkeys(
                    [*node.get("effects", ["UNKNOWN"]), "CTRL", "STATE", "IO", "MEM"]
                )
            ),
            reads=[
                "module.registry",
                "process.import_hooks",
                "process.audit_hooks",
                "filesystem",
            ],
            writes=[
                "module.registry",
                "process.audit_events",
                "import.external_effects",
            ],
            may_raise=True,
            allocates="UNKNOWN",
            frees="UNKNOWN",
            aliases="UNKNOWN",
            escapes="UNKNOWN",
            rng_effect="UNKNOWN",
            external_effect="UNKNOWN",
            ordering_effect="UNKNOWN",
            initialization=dict(
                policy="reuse_or_initialize_once_on_success",
                entry_state="inspect_runtime_cache",
                cold="preserve_source_initializer_dependencies_hooks_and_failures",
                cached="preserve_import_request_and_current_module_identity",
                qualified_export="unmodified_cpython_import_module_and_bootstrap",
                unproven_binding="requires_equivalent_import_protocol",
                **({"expected_instance": previous} if known else {}),
            ),
            **source_site(node),
        ),
        "unit",
        steps,
        True,
    )
    proof = (
        previous
        if known
        else dict(
            state="ready",
            source_provider_id=stored["provider_id"],
            owner=frame.name,
            **source_site(node),
        )
    )
    frame.project.imports.bind_registry("importlib", proof)
    return proof


def _import_module(frame, node, target, args, keywords):
    name, package = _bind(
        frame,
        node,
        args,
        keywords,
        [("name", "text", _REQUIRED), ("package", ("unit", "text"), None)],
    )
    if type(name.exact) is not str:
        frame.fail(
            node,
            "import_module_name",
            "a dynamic module name requires runtime import dispatch and registry lookup",
        )
    module = name.exact
    if not module:
        frame.fail(
            node,
            "import_module_name",
            "an empty module name requires the source ValueError contract",
        )
    if module.startswith("."):
        anchor = package.exact
        suffix = module.lstrip(".")
        level = len(module) - len(suffix)
        if type(anchor) is not str or not anchor or level > len(anchor.split(".")):
            frame.fail(
                node,
                "import_module_package",
                "relative import needs a known package and a valid parent, or its source exception contract",
            )
        parent = anchor.rsplit(".", level - 1)[0] if level > 1 else anchor
        module = ".".join((parent, suffix)) if suffix else parent
    proof = frame.project.imports.registry_entry("importlib")
    if proof.get("state") != "ready" or proof != target.facts.get("namespace_instance"):
        frame.fail(
            node,
            "import_module_binding",
            "importlib's qualified namespace and bootstrap availability are no longer proven",
        )
    frame.project.read_registry("importlib", proof, node)
    return frame.cached_module(node, module)


def import_provider(frame, node, steps):
    source = node.get("source_provider", {})
    module = node["module"]
    if (
        source.get("status") != "verified_stdlib"
        or source.get("module") != module.split(".")[0]
        or source.get("platform") not in {"linux", "linux2"}
        or not source.get("sha256")
    ):
        frame.fail(
            node,
            "host_provider",
            "host APIs require a pinned POSIX standard-library identity",
        )
    identity = source["module"]
    key = "stdlib:" + identity
    prior = frame.project.providers.get(key)
    if prior and prior["sha256"] != source["sha256"]:
        frame.fail(
            node,
            "host_provider",
            "one program resolves conflicting host provider identities",
        )
    if module not in _MEMBERS:
        frame.fail(node, "host_provider", f"no host namespace contract for {module}")
    primary = not node["request"].get("binding_index", 0)
    if module == "glob":
        if prior and prior.get("direct_imports") != source.get("direct_imports"):
            frame.fail(
                node,
                "glob_import_sources",
                "one program resolves conflicting glob dependencies",
            )
        if primary:
            _glob_import(frame, node, source, steps)
    namespace_facts = source
    if module == "enum":
        enumerations.qualify_source(frame, node, source)
        instance = source_namespace(frame, node, module)
        _initialize_source_namespace(frame, node, instance)
        namespace_facts = {**source, "source_namespace": instance}
    elif literal_exports(frame, module):
        namespace_facts = {
            **source,
            "source_namespace": source_namespace(frame, node, module),
        }
    if module == "importlib":
        namespace_facts = {
            **source,
            "namespace_instance": (
                _importlib_namespace(frame, node, source, steps)
                if primary
                else frame.project.imports.registry_entry("importlib")
            ),
        }
    if module == "os.path":
        _path_identity(frame, node, source)
    frame.project.providers[key] = dict(
        source,
        contract={
            "enum": enumerations.PROFILE,
            "glob": _GLOB_PROFILE,
            "importlib": namespace_operations.REGISTRY_PROFILE,
        }.get(module, "closed host APIs with explicit resource effects"),
    )
    namespace = Value("host_namespace", module, namespace_facts)
    if node["request"]["form"] == "from":
        return member(frame, node, namespace, node["member"], steps)
    if module == "os.path" and not node.get("alias"):
        return Value("host_namespace", "os", source)
    return namespace


def member(frame, node, base, name, steps):
    if base.kind in {"enum_class", "enum_member"}:
        return enumerations.member(frame, node, base, name, steps)
    if base.kind == _STREAM:
        if name not in _METHODS:
            frame.fail(node, "stream_member", f"no closed stream method for {name}")
        return Value("host_method", (base, name))
    if base.kind == "source_namespace":
        return source_export(
            frame, node, base.facts.get("source_namespace"), name, steps
        )
    if base.kind != "host_namespace":
        return None
    if name not in _MEMBERS.get(base.value, ()):
        return source_export(
            frame, node, base.facts.get("source_namespace"), name, steps
        )
    qualified = base.value + "." + name
    if qualified == "enum.Enum":
        for opcode in (
            "module_attribute_resolve",
            "module_export_continuation_required",
        ):
            namespace_operations.require_registry_operator(frame.project, opcode)
        return enumerations.base(frame, node, base.facts["source_namespace"], steps)
    if qualified in {"subprocess.PIPE", "subprocess.STDOUT"}:
        return literal(-1 if name == "PIPE" else -2)
    if qualified == "os.path":
        _path_identity(frame, node, base.facts)
        return Value("host_namespace", qualified, base.facts)
    if base.value == "math" and name in {"pi", "e", "tau"}:
        return literal(getattr(math, name))
    return Value("host_callable", qualified, base.facts)


def _path_identity(frame, node, source):
    alias = source.get("aliases", {}).get("os.path", {})
    if alias.get("status") != "verified_stdlib" or not alias.get("sha256"):
        frame.fail(
            node,
            "host_provider",
            "the imported path implementation requires its own verified identity",
        )


def _bind(frame, node, args, keywords, parameters):
    if len(args) > len(parameters):
        frame.fail(node, "host_arguments", "too many positional arguments")
    unknown = set(keywords) - {p[0] for p in parameters}
    if unknown:
        frame.fail(
            node,
            "host_arguments",
            f"unclosed host keyword arguments: {sorted(unknown)}",
        )
    bound = []
    for i, (name, kind, default) in enumerate(parameters):
        if i < len(args):
            if name in keywords:
                frame.fail(node, "host_arguments", f"multiple values for {name}")
            value = args[i]
        elif name in keywords:
            value = keywords[name]
        elif default is not _REQUIRED:
            value = literal(default)
        else:
            frame.fail(node, "host_arguments", f"missing argument {name}")
        if (
            kind == "f64"
            and value.kind == "i64"
            and max(map(abs, value.bounds)) <= 1 << 53
        ):
            value = Value(
                "f64", dict(op="cast", type="f64", value=value.value), value.facts
            )
        allowed = kind if isinstance(kind, tuple) else (kind,)
        if value.kind not in allowed:
            frame.fail(
                node, "host_arguments", f"{name} requires {kind}, got {value.kind}"
            )
        bound.append(value)
    return bound


def _emit(frame, node, interface, result, args, steps, discard, *, result_facts=None):
    filesystem = interface.startswith("fs.")
    reads = ["process.cwd"] if filesystem else []
    writes = []
    if filesystem:
        reads.append("filesystem")
        if interface not in {"fs.exists", "fs.is_file", "fs.is_dir"}:
            writes.append("filesystem")
    if interface == "process.chdir":
        writes.append("process.cwd")
    if interface == "process.getcwd":
        reads.append("process.cwd")
    if interface.startswith("stream."):
        reads.append("stream.state")
        writes.append("stream.state")
        if interface == "stream.open":
            reads += ["process.cwd", "filesystem"]
            if args[1].exact[:1] != "r":
                writes.append("filesystem")
        elif interface in {"stream.write", "stream.flush", "stream.close"}:
            writes.append("filesystem")
        else:
            reads.append("filesystem")
    if interface.startswith("time."):
        reads.append("host.clock")
    if interface == "console.input":
        reads.append("stdin.state")
        writes += ["stdin.state", "stdout.state", "stderr.state"]
    if interface == "process.exit":
        writes.append("process.completion")
    effects = ["VAL"] if interface == "path.join" else ["IO", "STATE"]
    return frame.effect(
        dict(
            op="host_call",
            interface=interface,
            semantics_version=1,
            args=[frame.scalar(value, node) for value in args],
            type=wire(result),
            effects=effects,
            reads=reads,
            writes=writes,
            may_raise=True,
            **source_site(node),
            **argument_evidence(args),
            **({"result_facts": result_facts} if result_facts else {}),
            **({"noreturn": True} if interface == "process.exit" else {}),
        ),
        result,
        steps,
        discard,
        facts=result_facts,
    )


def _open(frame, node, target, args, keywords, steps, discard):
    parameters = [
        ("file", "text", _REQUIRED),
        ("mode", "text", "r"),
        ("buffering", "i64", -1),
        ("encoding", ("unit", "text"), None),
        ("errors", ("unit", "text"), None),
        ("newline", ("unit", "text"), None),
        ("closefd", "bool", True),
        ("opener", "unit", None),
    ]
    bound = _bind(frame, node, args, keywords, parameters)
    path, mode, buffering, encoding, errors, newline, closefd, opener = bound
    if mode.exact not in {"r", "rt", "w", "wt", "a", "at", "x", "xt"}:
        frame.fail(
            node, "stream_mode", "text stream mode must be a closed r/w/a/x mode"
        )
    default_encoding = target.facts.get("default_text_encoding", "")
    encoding_name = default_encoding if encoding.kind == "unit" else encoding.exact
    if not isinstance(encoding_name, str) or encoding_name.lower().replace(
        "-", ""
    ) not in {"utf8", "utf_8"}:
        frame.fail(
            node, "stream_encoding", "text streams require a proven UTF-8 encoding"
        )
    if (
        buffering.exact != -1
        or errors.kind != "unit"
        and errors.exact != "strict"
        or newline.kind != "unit"
        or closefd.exact is not True
        or opener.kind != "unit"
    ):
        frame.fail(
            node,
            "stream_options",
            "nondefault buffering, errors, newline or opener requires another stream contract",
        )
    return _emit(
        frame,
        node,
        "stream.open",
        _STREAM,
        [path, mode, literal(encoding_name), literal("universal")],
        steps,
        discard,
        result_facts={"stream_origin": {"kind": "named_text_file"}},
    )


def invoke(frame, node, target, args, keywords, steps, discard):
    if target.kind == "host_method":
        receiver, method = target.value
        interface, result, parameters = _METHODS[method]
        values = _bind(frame, node, args, keywords, parameters)
        return _emit(
            frame, node, interface, result, [receiver, *values], steps, discard
        )
    if target.kind != "host_callable":
        return None
    name = target.value
    if name == "importlib.import_module":
        return _import_module(frame, node, target, args, keywords)
    if name == "glob.glob":
        return _glob(frame, node, args, keywords, steps, discard)
    if name.startswith("math."):
        return scalars.math_call(
            frame, node, name.removeprefix("math."), args, keywords, steps, discard
        )
    if name == "builtins.open":
        return _open(frame, node, target, args, keywords, steps, discard)
    if name == "builtins.input":
        return _input(frame, node, target, args, keywords, steps, discard)
    if name == "sys.exit":
        return _exit(frame, node, args, keywords, steps, discard)
    if name in {"builtins.str", "builtins.repr", "builtins.ascii", "builtins.format"}:
        builtin_name = name.removeprefix("builtins.")
        if builtin_name == "str":
            if not args and set(keywords) == {"object"}:
                args, keywords = [keywords["object"]], {}
            if not args and not keywords:
                return literal("")
        arities = {1, 2} if builtin_name == "format" else {1}
        if keywords or len(args) not in arities:
            frame.fail(
                node,
                "text_arguments",
                f"{builtin_name} needs its scalar positional signature",
            )
        conversion = "none" if builtin_name == "format" else builtin_name
        spec = args[1] if len(args) == 2 else literal("")
        return text.render(frame, node, args[0], conversion, spec, steps, discard)
    if name == "os.path.join":
        if not args or keywords or any(v.kind != "text" for v in args):
            frame.fail(
                node,
                "path_arguments",
                "path.join requires positional text path components",
            )
        facts = {}
        origin = target.facts.get("aliases", {}).get("os.path", {})
        if (
            origin.get("module") == "posixpath"
            and origin.get("status") == "verified_stdlib"
            and origin.get("sha256")
            and all(type(v.exact) is str for v in args)
        ):
            # POSIX join resets at each absolute component and preserves a final
            # empty component. It neither normalizes dots nor consults the filesystem.
            joined = args[0].exact
            for value in args[1:]:
                part = value.exact
                if part.startswith("/"):
                    joined = part
                else:
                    joined += (
                        "/" if joined and not joined.endswith("/") else ""
                    ) + part
            facts["exact"] = joined
        return _emit(
            frame, node, "path.join", "text", args, steps, discard, result_facts=facts
        )
    interface, result, parameters = _SIGNATURES[name]
    values = _bind(frame, node, args, keywords, parameters)
    if name in {"os.mkdir", "os.makedirs"}:
        if values[1].exact != 0o777:
            frame.fail(
                node,
                "filesystem_mode",
                "nondefault directory permissions need an explicit mode contract",
            )
        if name == "os.mkdir" and values[2].kind != "unit":
            frame.fail(
                node,
                "filesystem_directory",
                "relative directory descriptors need a separate filesystem contract",
            )
        values = values[:1] if name == "os.mkdir" else [values[0], values[2]]
    if name in {"shutil.copyfile", "shutil.copy", "shutil.copy2"}:
        if values[-1].exact is not True:
            frame.fail(
                node,
                "filesystem_links",
                "non-following symbolic links need a separate copy contract",
            )
        values = values[:2]
    return _emit(frame, node, interface, result, values, steps, discard)


def _format(frame, node, value, steps, discard):
    return frame.effect(
        dict(
            op="text_format",
            value=frame.scalar(value, node),
            type="text",
            scalar_format=text.SCALAR_FORMAT,
        ),
        "text",
        steps,
        discard,
    )


def _input(frame, node, target, args, keywords, steps, discard):
    if keywords or len(args) > 1 or args and args[0].kind != "text":
        frame.fail(
            node,
            "console_arguments",
            "input requires at most one positional text prompt",
        )
    facts = target.facts
    encodings = [facts.get(key, "") for key in ("stdin_encoding", "stdout_encoding")]
    if any(
        not isinstance(value, str)
        or value.lower().replace("-", "") not in {"utf8", "utf_8"}
        for value in encodings
    ):
        frame.fail(
            node,
            "console_encoding",
            "console input requires proven UTF-8 input and output encodings",
        )
    errors = facts.get("stdin_errors")
    if (
        errors not in {"strict", "surrogateescape"}
        or errors == "surrogateescape"
        and facts.get("stdout_errors") != errors
    ):
        frame.fail(
            node,
            "console_encoding",
            "the input error policy requires a compatible native text-output contract",
        )
    return _emit(
        frame,
        node,
        "console.input",
        "text",
        [args[0] if args else literal(""), literal(errors)],
        steps,
        discard,
    )


def _exit(frame, node, args, keywords, steps, discard):
    if keywords or len(args) > 1:
        frame.fail(
            node, "exit_arguments", "exit requires at most one positional status"
        )
    value = args[0] if args else literal(None)
    if value.kind == "unit":
        status, message, output = literal(0), literal(""), literal(None)
    elif value.kind in {"i64", "bool"}:
        status = (
            value
            if value.kind == "i64"
            else Value(
                "i64", dict(op="cast", type="i64", value=value.value), value.facts
            )
        )
        message = _format(frame, node, value, steps, False)
        output = literal(None)
    elif value.kind == "text":
        status, message, output = literal(1), value, value
    else:
        frame.fail(
            node,
            "exit_arguments",
            "exit status requires a closed scalar display and integer-code contract",
        )
    error = frame.effect(
        dict(
            op="error_make",
            category="system_exit",
            message=message.value,
            exit_code=status.value,
            exit_message=output.value,
            type=wire("handle:error"),
        ),
        "handle:error",
        steps,
    )
    return _emit(frame, node, "process.exit", "unit", [error], steps, discard)
