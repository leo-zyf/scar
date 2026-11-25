"""M2 state and data contracts for verified plotting operations."""

from pathlib import PurePosixPath

from scar.contract import python_source_boundary

from .values import Value, rank, wire

_PROFILE = "pyplot-3.11.2"
_ROOT = "matplotlib"
_PYPLOT = "matplotlib.pyplot"
_CONTEXT = "handle:graphics_context"
_LINES = "handle:graphics_lines"
_SLOT = "@graphics/current_context"
_API = {"plot", "grid", "savefig"}
_MPLOT3D = "mpl_toolkits.mplot3d"
_AXES3D = _MPLOT3D + ".axes3d"
_PROJECTION_SOURCES = {
    _MPLOT3D: (
        "mpl_toolkits/mplot3d/__init__.py",
        "7c7f4774c7c532f8db216ab2da08d09e6da6dda7b50af2e2b87e8bc0a1e8d242",
    ),
    _AXES3D: (
        "mpl_toolkits/mplot3d/axes3d.py",
        "6823b0e69fe54255ee82de392aeefac5e2813f23204eec8b1ad540a0357aee35",
    ),
}


def handles_import(node):
    if node["module"] == _MPLOT3D:
        return node["request"]["form"] == "from" and node.get("member") == "Axes3D"
    return node["module"] == _PYPLOT or (
        node["module"] == _ROOT
        and node["request"]["form"] == "from"
        and node["member"] == "pyplot"
    )


def _verified(frame, node, source, module):
    member = {
        _ROOT: "matplotlib/__init__.py",
        _PYPLOT: "matplotlib/pyplot.py",
        _MPLOT3D: _PROJECTION_SOURCES[_MPLOT3D][0],
    }[module]
    digest = source.get("sha256")
    if not (
        python_source_boundary(source) == _PROFILE
        and source.get("module") == module
        and source.get("distribution", "").lower() == _ROOT
        and source.get("version") == "3.11.2"
        and source.get("member") == member
        and source.get("origin")
        and source.get("proof_scope") == "installed_file_with_digest"
        and isinstance(digest, str)
        and len(digest) == 64
        and all(ch in "0123456789abcdef" for ch in digest)
        and (
            module not in _PROJECTION_SOURCES
            or digest == _PROJECTION_SOURCES[module][1]
        )
    ):
        frame.fail(
            node,
            "graphics_provider",
            f"{module} requires a verified Matplotlib 3.11.2 distribution member and content digest",
        )
    return {
        key: source[key] for key in ("module", "member", "origin", "sha256", "version")
    }


class Graphics:
    def __init__(self):
        self.sources = None
        self.operations = set()
        self.extensions = {}

    def _operation(self, node, op, kind, reads, writes, **arguments):
        self.operations.add(op)
        return dict(
            op=op,
            operator_family="graphics",
            type=wire(kind),
            effects=["STATE", "IO", "MEM"],
            may_raise=True,
            reads=reads,
            writes=writes,
            location=node.get("location", {}),
            **({"site": node["site"]} if node.get("site") is not None else {}),
            **arguments,
        )

    def import_provider(self, frame, node, steps):
        if node["module"] == _MPLOT3D:
            return self._projection_import(frame, node, steps)
        request = node["request"]["form"]
        root = _verified(frame, node, node.get("source_provider", {}), _ROOT)
        if node["module"] == _PYPLOT:
            child = node.get("import_resolution", {})
            selection = "qualified_module"
        else:
            alternate = node.get("member_resolution", {})
            if alternate.get("module") != _PYPLOT or alternate.get("when") != {
                "kind": "package_attribute_missing",
                "module": _ROOT,
                "name": "pyplot",
            }:
                frame.fail(
                    node,
                    "graphics_import_binding",
                    "the from-list needs separate source evidence for its conditional plotting submodule",
                )
            child = alternate.get("source", {})
            selection = "package_attribute_then_submodule"
        pyplot = _verified(frame, node, child, _PYPLOT)
        if root["origin"].rpartition("/")[0] != pyplot["origin"].rpartition("/")[0]:
            frame.fail(
                node,
                "graphics_provider",
                "the package and selected plotting module originate in different installations",
            )
        sources = [root, pyplot]
        if self.sources is not None and self.sources != sources:
            frame.fail(
                node,
                "graphics_provider",
                "the program resolves different plotting provider identities",
            )
        if self.sources is None:
            self.sources = sources
            frame.project.providers[_PYPLOT] = dict(
                distribution="matplotlib",
                version="3.11.2",
                modules=sources,
                import_effects=node["effects"],
                contract=_PROFILE,
            )
            context = self._operation(
                node,
                "graphics_context",
                _CONTEXT,
                ["process.cwd", "process.environment", "filesystem", "module.registry"],
                [
                    "module.registry",
                    "graphics.context",
                    "graphics.configuration",
                    "filesystem",
                ],
                profile=_PROFILE,
                scope="process",
                source_protocol="python.import",
                initialization="import_once",
                configuration="source_runtime",
                current_figure="preserve_existing",
                current_axes="preserve_existing",
                import_binding=dict(
                    selection=selection,
                    existing_module="match_verified_identity",
                    package_attribute="absent_or_identical_to_verified_submodule",
                    initial_registry="pristine_or_verified",
                ),
            )
            frame.project.globals[_SLOT] = _CONTEXT
            steps.append(
                dict(
                    op="assign",
                    target=Value(_CONTEXT, None).at(_SLOT, scope="global").value,
                    value=context,
                )
            )
        if node["module"] == _PYPLOT and request == "from":
            return self.member(
                frame, node, Value("graphics_namespace", _PYPLOT), node["member"]
            )
        namespace = _ROOT if request == "import" and not node.get("alias") else _PYPLOT
        return Value("graphics_namespace", namespace)

    def _projection_import(self, frame, node, steps):
        if self.sources is None:
            frame.fail(
                node,
                "graphics_context",
                "a projection import requires the previously verified shared Matplotlib context and package identity",
            )
        namespace = node.get("source_provider", {})
        locations = namespace.get("locations")
        package_source = node.get("import_resolution", {})
        package = _verified(frame, node, package_source, _MPLOT3D)
        package_path = PurePosixPath(package["origin"])
        if not (
            namespace.get("module") == "mpl_toolkits"
            and namespace.get("status") == "namespace"
            and namespace.get("loader") == "namespace"
            and namespace.get("proof_scope") == "ordered_search_locations"
            and isinstance(locations, list)
            and locations
            and all(
                isinstance(path, str)
                and PurePosixPath(path).is_absolute()
                and ".." not in PurePosixPath(path).parts
                for path in locations
            )
            and package_path.is_absolute()
            and ".." not in package_path.parts
            and len(package_path.parents) > 2
            and str(package_path.parents[1]) in locations
            and package_path.parents[2]
            == PurePosixPath(self.sources[0]["origin"]).parent.parent
        ):
            frame.fail(
                node,
                "graphics_provider",
                "the namespace locations and selected projection package do not match the shared Matplotlib installation",
            )
        export = package_source.get("exports", {}).get("Axes3D", {})
        candidates = export.get("candidates", ())
        if (
            export.get("status") != "source_candidates"
            or export.get("unresolved") != []
            or len(candidates) != 1
            or candidates[0].get("kind", "definition") != "definition"
        ):
            frame.fail(
                node,
                "graphics_import_binding",
                "Axes3D requires one unconditional sealed export path",
            )
        selected = candidates[0]
        class_member, class_digest = _PROJECTION_SOURCES[_AXES3D]
        via = selected.get("via", ())
        expected = [
            dict(
                kind="from",
                module=_MPLOT3D,
                member=package["member"],
                name="Axes3D",
                sha256=package["sha256"],
            ),
            dict(
                kind="definition",
                module=_AXES3D,
                member=class_member,
                name="Axes3D",
                sha256=class_digest,
            ),
        ]
        route = [{key: step.get(key) for key in expected[0]} for step in via]
        if not (
            selected.get("distribution") == _ROOT
            and selected.get("version") == "3.11.2"
            and selected.get("module") == _AXES3D
            and selected.get("member") == class_member
            and selected.get("name") == "Axes3D"
            and selected.get("sha256") == class_digest
            and selected.get("conditions") == []
            and route == expected
            and selected.get("origin") == str(package_path.parent / "axes3d.py")
        ):
            frame.fail(
                node,
                "graphics_import_binding",
                "Axes3D export identity does not match the qualified Matplotlib projection profile",
            )
        sources = [
            package,
            {
                key: selected[key]
                for key in ("module", "member", "origin", "sha256", "version")
            },
        ]
        receipt = dict(
            distribution=_ROOT,
            version="3.11.2",
            modules=sources,
            namespace={
                key: namespace[key]
                for key in ("module", "status", "loader", "proof_scope", "locations")
            },
            export=dict(
                name="Axes3D", qualification="sealed_source_candidate", path=route
            ),
            import_effects=node.get("effects", ["UNKNOWN"]),
            contract=_PROFILE,
        )
        previous = self.extensions.get(_MPLOT3D)
        if previous is not None and previous != receipt:
            frame.fail(
                node,
                "graphics_provider",
                "the program resolves conflicting projection import identities",
            )
        if previous is None:
            self.extensions[_MPLOT3D] = receipt
            frame.project.providers[_MPLOT3D] = receipt
        operation = self._operation(
            node,
            "graphics_projection_import",
            "unit",
            [
                "module.registry",
                "module.namespace",
                "graphics.context",
                "graphics.configuration",
                "graphics.projection_registry",
                "graphics.class.shared_state",
                "process.environment",
                "process.cwd",
                "filesystem",
            ],
            [
                "module.registry",
                "module.namespace",
                "graphics.projection_registry",
                "graphics.class.shared_state",
                "graphics.configuration",
                "filesystem",
            ],
            context=Value(_CONTEXT, None).at(_SLOT, scope="global").value,
            profile=_PROFILE,
            scope="process",
            source_protocol="python.import",
            initialization="import_once",
            cache="sys.modules",
            exported_binding=dict(
                module=_MPLOT3D, name="Axes3D", source_module=_AXES3D
            ),
            projection_registration="preserve_source_import_closure",
            class_initialization="preserve_decorators_and_shared_axis_state",
            sources=sources,
            import_binding=dict(
                selection="namespace_package_export",
                namespace_locations=locations,
                existing_module="match_verified_identity",
                package_attribute="match_qualified_source_export",
                initial_registry="pristine_or_verified",
            ),
        )
        if not node["request"].get("binding_index", 0):
            steps.append(dict(op="expr", value=operation))
        return Value(
            "graphics_class",
            _AXES3D + ".Axes3D",
            dict(profile=_PROFILE, provider=_MPLOT3D),
        )

    def member(self, frame, node, base, name):
        if base.value == _ROOT and name == "pyplot":
            return Value("graphics_namespace", _PYPLOT)
        if base.value != _PYPLOT or name not in _API:
            frame.fail(
                node,
                "graphics_member",
                f"no graphics state contract for {base.value}.{name}",
            )
        return Value("graphics_callable", name)

    def invoke(self, frame, node, name, args, kwargs, steps, discard):
        context = Value(_CONTEXT, None).at(_SLOT, scope="global").value
        if name == "plot":
            if len(args) != 2 or any(rank(v.kind) not in {1, 2} for v in args):
                frame.fail(
                    node,
                    "graphics_plot_data",
                    "plot requires two positional rank-one or rank-two binary64 arrays",
                )
            if kwargs.keys() - {"color", "marker", "linestyle"} or any(
                v.kind != "text" for v in kwargs.values()
            ):
                frame.fail(
                    node,
                    "graphics_plot_style",
                    "plot color, marker and linestyle require text values",
                )
            kind = _LINES
            operation = self._operation(
                node,
                "graphics_plot",
                kind,
                [
                    "graphics.context",
                    "graphics.configuration",
                    "array.storage",
                    "filesystem",
                    "process.environment",
                ],
                [
                    "graphics.context",
                    "graphics.current_figure",
                    "graphics.current_axes",
                    "graphics.artists",
                    "graphics.canvas",
                    "filesystem",
                ],
                context=context,
                x=args[0].value,
                y=args[1].value,
                style=[
                    dict(name=key, value=value.value) for key, value in kwargs.items()
                ],
                ensure="current_figure_and_axes",
                series="columns",
                data_capture="snapshot_at_call",
            )
        elif name == "grid":
            if not args and set(kwargs) == {"visible"}:
                args, kwargs = [kwargs["visible"]], {}
            if len(args) != 1 or args[0].kind != "bool" or kwargs:
                frame.fail(
                    node,
                    "graphics_grid",
                    "grid requires one Boolean visibility with default axis and tick selection",
                )
            kind = "unit"
            operation = self._operation(
                node,
                "graphics_grid",
                kind,
                [
                    "graphics.context",
                    "graphics.configuration",
                    "filesystem",
                    "process.environment",
                ],
                [
                    "graphics.context",
                    "graphics.current_figure",
                    "graphics.current_axes",
                    "graphics.canvas",
                    "filesystem",
                ],
                context=context,
                visible=args[0].value,
                which="major",
                axis="both",
                ensure="current_figure_and_axes",
            )
        elif name == "savefig":
            if not args and set(kwargs) == {"fname"}:
                args, kwargs = [kwargs["fname"]], {}
            if len(args) != 1 or args[0].kind != "text" or kwargs:
                frame.fail(
                    node,
                    "graphics_save",
                    "savefig requires one text file path and source-runtime defaults",
                )
            kind = "unit"
            operation = self._operation(
                node,
                "graphics_save",
                kind,
                [
                    "graphics.context",
                    "graphics.configuration",
                    "graphics.current_figure",
                    "graphics.current_axes",
                    "graphics.artists",
                    "graphics.canvas",
                    "filesystem",
                    "process.cwd",
                    "process.environment",
                    "host.clock",
                ],
                [
                    "graphics.context",
                    "graphics.current_figure",
                    "graphics.canvas",
                    "filesystem",
                ],
                context=context,
                path=args[0].value,
                format="suffix_or_configured_default",
                ensure="current_figure",
                redraw="idle_after_success",
            )
        else:
            frame.fail(node, "graphics_member", f"no graphics operation for {name}")
        return frame.effect(operation, kind, steps, discard)

    def requirement(self):
        return dict(
            profile=_PROFILE,
            operations=sorted(self.operations),
            context="shared_process",
            output="source_renderer_file",
        )
