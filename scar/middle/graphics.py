"""M2 state and data contracts for verified plotting operations."""

from .values import Value, rank, wire

_PROFILE = "pyplot-3.11.2"
_ROOT = "matplotlib"
_PYPLOT = "matplotlib.pyplot"
_CONTEXT = "handle:graphics_context"
_LINES = "handle:graphics_lines"
_SLOT = "@graphics/current_context"
_API = {"plot", "grid", "savefig"}


def handles_import(node):
    return node["module"] == _PYPLOT or (
        node["module"] == _ROOT
        and node["request"]["form"] == "from"
        and node["member"] == "pyplot"
    )


def _verified(frame, node, source, module):
    member = {_ROOT: "matplotlib/__init__.py", _PYPLOT: "matplotlib/pyplot.py"}[module]
    digest = source.get("sha256")
    if not (
        source.get("status") == "verified_distribution"
        and source.get("module") == module
        and source.get("distribution", "").lower() == _ROOT
        and source.get("version") == "3.11.2"
        and source.get("member") == member
        and source.get("origin")
        and source.get("proof_scope") == "installed_file_with_digest"
        and isinstance(digest, str)
        and len(digest) == 64
        and all(ch in "0123456789abcdef" for ch in digest)
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
            **arguments,
        )

    def import_provider(self, frame, node, steps):
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
