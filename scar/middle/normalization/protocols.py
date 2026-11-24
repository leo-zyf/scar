"""Close lexical Python protocols from the bound project operation graph."""

from scar.contract import walk_operations
from . import (
    arrays,
    expressions,
    exceptions,
    graphics,
    host_protocols,
    objects,
    processes,
    products,
    scalars,
    sequences,
    text,
    values,
)
from .values import Value as _Value, literal as _literal


def _slot(module, name):
    return f"py:{module}:{name}"


def _reachable(roots, edges):
    successors = {}
    for edge in edges:
        successors.setdefault(edge["owner"], set()).add(edge["target"])
    result, pending = set(), list(roots)
    while pending:
        current = pending.pop()
        if current not in result:
            result.add(current)
            pending.extend(successors.get(current, ()))
    return result


def _join(paths, sequence_domain=None):
    """Only bindings present on every control predecessor are definitely bound."""
    result = {}
    for name, first in paths[0].items():
        incoming = [path.get(name) for path in paths[1:]]
        if all(values.identical(v, first) for v in incoming):
            result[name] = first
        elif values.is_sequence(first.kind) and all(
            v is not None
            and values.is_sequence(v.kind)
            and isinstance(v.value, dict)
            and v.value.get("op") == first.value.get("op") == "var"
            and (v.value.get("name"), v.value.get("scope"))
            == (first.value.get("name"), first.value.get("scope"))
            for v in incoming
        ):
            if sequence_domain is not None and all(
                sequence_domain.compatible(first.kind, v.kind) for v in incoming
            ):
                kind = sequence_domain.resolve_kind(first.kind)
                facts = (
                    first.facts
                    if all(values.identical(v.facts, first.facts) for v in incoming)
                    else {}
                )
                result[name] = _Value(
                    kind, {**first.value, "type": values.wire(kind)}, facts
                )
        elif values.is_array(first.kind) and all(
            v is not None
            and values.is_array(v.kind)
            and isinstance(v.value, dict)
            and isinstance(first.value, dict)
            and v.value.get("op") == first.value.get("op") == "var"
            and (v.value.get("name"), v.value.get("scope"))
            == (first.value.get("name"), first.value.get("scope"))
            for v in incoming
        ):
            kind = values.join_arrays([first.kind, *(v.kind for v in incoming)])
            facts = {}
            if "shape" in first.facts and all(
                values.identical(v.facts.get("shape"), first.facts["shape"])
                for v in incoming
            ):
                facts["shape"] = first.facts["shape"]
            result[name] = _Value(
                kind, {**first.value, "type": values.wire(kind)}, facts
            )
        elif all(
            v is not None
            and v.kind == first.kind
            and values.identical(v.value, first.value)
            for v in incoming
        ):
            facts = {}
            if first.kind == "i64":
                bounds = [v.bounds for v in [first, *incoming]]
                facts["range"] = (
                    min(lo for lo, _ in bounds),
                    max(hi for _, hi in bounds),
                )
            if "stream_origin" in first.facts and all(
                values.identical(
                    v.facts.get("stream_origin"), first.facts["stream_origin"]
                )
                for v in incoming
            ):
                facts["stream_origin"] = first.facts["stream_origin"]
            result[name] = _Value(first.kind, first.value, facts)
    return result


def _global_bindings(functions):
    """Prove single module bindings without loop/reentrant or alias writes."""
    writes, repeated = {}, set()
    for function in functions.values():
        pending = [(function["body"], function["python_scope"]["kind"] != "module")]
        while pending:
            node, multiple = pending.pop()
            if isinstance(node, list):
                pending.extend((item, multiple) for item in node)
                continue
            if not isinstance(node, dict):
                continue
            op = node.get("op")
            multiple = multiple or op in {"for", "while", "exception_handler"}
            for target in values.written_targets(node):
                if target.get("op") == "var" and target.get("scope") == "global":
                    name = target["name"]
                    writes[name] = writes.get(name, 0) + 1
                    if multiple:
                        repeated.add(name)
            pending.extend((child, multiple) for child in node.values())
    return set(writes), {
        name for name, count in writes.items() if count == 1 and name not in repeated
    }


class _Unclosed(Exception):
    def __init__(self, name, node, field, detail):
        self.issue = dict(
            code="python." + field,
            symbol=name,
            location=node.get("location"),
            **({"site": node["site"]} if node.get("site") is not None else {}),
            detail=detail,
        )
        super().__init__(detail)


class _Region:
    """A statement boundary and its in-progress child regions in the target buffer."""

    def __init__(self, steps, parent, source):
        self.steps = steps
        self.start = len(steps) if parent is None or steps is parent.steps else 0
        self.end = self.start
        self.source = source
        self.current = None
        self.children = []

    def boundary(self, node):
        self.current, self.end = node, len(self.steps)
        self.children.clear()

    def fragment(self):
        record = dict(
            completion="complete" if self.current is None else "prefix",
            body=self.steps[self.start : self.end],
        )
        if self.current is None:
            return record
        node = self.current
        frontier = dict(operator=node["op"], **values.source_site(node), regions=[])
        # Constant-folded branches may share the caller's destination. Their
        # operations remain inside the corresponding source branch in this record.
        before_child = min(
            (child.start for child in self.children if child.steps is self.steps),
            default=len(self.steps),
        )
        setup = []
        for statement in self.steps[self.end : before_child]:
            if statement["op"] not in {"decl", "assign", "expr"}:
                break
            setup.append(statement)
        if setup:
            frontier["evaluation"] = setup
        for child in self.children:
            path = next(
                (
                    [key]
                    for key in ("body", "then", "else", "finally")
                    if node.get(key) is child.source
                ),
                None,
            )
            if path is None:
                path = next(
                    (
                        ["handlers", i, "body"]
                        for i, handler in enumerate(node.get("handlers", ()))
                        if handler.get("body") is child.source
                    ),
                    None,
                )
            if path is not None:
                frontier["regions"].append(dict(path=path, **child.fragment()))
        record["frontier"] = frontier
        return record


class _Project:
    def __init__(self, functions):
        self.functions = functions
        self.initializers = {
            scope["module"]: name
            for name, f in functions.items()
            if (scope := f.get("python_scope", {})).get("kind") == "module"
        }
        self.names = {}
        self.globals = {}
        self.types = {}
        self.closed = {}
        self.activations = {}
        self.prefixes = []
        self.result_kinds = {}
        self.result_facts = {}
        self.providers = {}
        self.array_ops = set()
        self.process_ops = set()
        self.process_requests = {}
        self.required_operators = {}
        self.execution_images = []
        self.array_state = arrays.LiteralState()
        self.external_array_writes = {}
        self.objects = objects.ObjectModel()
        self.graphics = graphics.Graphics()
        self.decisions = []
        self.skipped_sites = {}
        self.live_sites = set()
        self.actual_calls = set()
        self.active = []
        self.global_reads = {}
        self.shadow, self.single_bind_globals = _global_bindings(functions)
        self.sequences = sequences.Domains(functions, self.single_bind_globals)

    def require_operator(self, family, operation, **contract):
        requirement = self.required_operators.setdefault(
            family, dict(operations=set(), **contract)
        )
        if any(requirement.get(key) != value for key, value in contract.items()):
            raise ValueError("conflicting logical operator requirements")
        requirement["operations"].add(operation)

    def operator_requirements(self):
        return {
            family: {**requirement, "operations": sorted(requirement["operations"])}
            for family, requirement in self.required_operators.items()
        }

    def global_read(self, name, value):
        for active in self.active:
            if self.functions[active]["python_scope"]["kind"] == "function":
                self.global_reads[active][name] = (value.kind, value.value)

    def inactive_bodies(self, activation):
        """Only proven edge removal can turn a required body into an inactive body."""
        if not activation:
            return set()
        roots, edges = activation["roots"], activation["edges"]
        removed, kept = [], []
        for edge in edges:
            key = edge["owner"], edge.get("site")
            if (
                edge.get("site") is not None
                and key in self.skipped_sites
                and key not in self.live_sites
            ):
                removed.append(dict(edge, proof=self.skipped_sites[key]))
            else:
                kept.append(edge)
        kept.extend(
            dict(owner=owner, target=target) for owner, target in self.actual_calls
        )
        inactive = (
            (_reachable(roots, edges) - _reachable(roots, kept))
            .intersection(self.functions)
            .difference(self.closed)
        )
        if removed:
            self.decisions.append(
                dict(
                    code="inactive_activation_edges",
                    roots=roots,
                    removed_edges=removed,
                    inactive_bodies=sorted(inactive),
                    proof="root reachability changes only after excluding proven untaken source sites; hard edges and actual calls remain",
                )
            )
        return inactive

    def compile(self, name, types, *, source=None):
        fn = self.functions[name]
        if name in self.active:
            raise _Unclosed(
                name,
                fn,
                "cycle",
                "recursive calls or partially initialized modules need a recursive contract",
            )
        if name in self.closed:
            if len(self.types[name]) != len(types) or not all(
                self.sequences.compatible(left, right)
                for left, right in zip(self.types[name], types)
            ):
                raise _Unclosed(
                    name,
                    fn,
                    "argument_types",
                    f"call types {types} conflict with earlier {self.types[name]}",
                )
            for slot, required in self.global_reads.get(name, {}).items():
                current = self.names.get(slot)
                if current is None or (current.kind, current.value) != required:
                    raise _Unclosed(
                        name,
                        fn,
                        "name_binding",
                        f"cached function requires initialized binding {slot} at this call",
                    )
                self.global_read(slot, current)
            return self.closed[name]
        scope = fn.get("python_scope", {})
        kind = scope.get("kind")
        if (
            kind not in {"module", "function"}
            or kind == "function"
            and fn.get("call_protocol", {}).get("kind") != "regular"
        ):
            raise _Unclosed(
                name,
                fn,
                "activation",
                "a regular function or module initialization contract is required",
            )
        if len(types) != len(fn["params"]):
            raise _Unclosed(
                name,
                fn,
                "arity",
                "the invocation must bind every formal parameter once",
            )
        if source is not None:
            self.activations[name] = dict(
                owner=self.active[-1], **values.source_site(source)
            )
        self.active.append(name)
        self.global_reads[name] = {}
        saved_contents = self.array_state.contents if kind == "function" else None
        if saved_contents is not None:
            self.array_state.contents = {}
        try:
            frame = _Frame(self, fn, types)
            try:
                result = frame.lower()
            except _Unclosed:
                self.prefixes.append(frame.prefix_fragment())
                raise
            self.types[name] = types
            self.closed[name] = result
            return result
        finally:
            if saved_contents is not None:
                self.array_state.contents = saved_contents
            self.active.pop()


class _Frame:
    def __init__(self, project, fn, types):
        self.project, self.fn = project, fn
        self.name, self.scope = fn["name"], fn["python_scope"]
        self.module = self.scope["module"]
        self.initializing = self.scope["kind"] == "module"
        self.env = {}
        self.locals = {}
        self.returns = set()
        self.return_facts = None
        self.next_temp = 0
        self.loops = []
        self.handler_depth = 0
        self.region = None
        self.root_region = None
        self.source_expression = None
        self.dynamic = 0
        self.params = []
        self.project.sequences.bind_frame(fn, types)
        for formal, kind in zip(fn["params"], types):
            if formal.get("cell") or formal.get("binding") not in {
                "positional",
                "posonly",
                "keywordonly",
            }:
                self.fail(
                    formal,
                    "formal",
                    "captured or variadic parameter needs a binding contract",
                )
            name = formal["name"]
            self.params.append(dict(name=name, type=values.wire(kind)))
            self.env[name] = _Value(kind, None).at(name)
            self.locals[name] = kind

    def fail(self, node, field, message, *, source_evidence=None):
        error = _Unclosed(self.name, node, field, message)
        if source_evidence:
            error.issue["source_evidence"] = source_evidence
        raise error

    def scalar(self, value, node):
        if not values.runtime(value.kind):
            self.fail(
                node, "value_contract", "symbolic binding escapes its closed namespace"
            )
        return value.value

    def skip_expression(self, node, reason):
        for part in walk_operations(node):
            if part.get("site") is not None:
                self.project.skipped_sites[self.name, part["site"]] = reason

    def temporary(self):
        self.next_temp += 1
        return f"@evaluation/{self.next_temp}"

    def effect(self, node, kind, steps, discard=False, *, facts=None):
        source = self.source_expression or (
            self.region.current if self.region is not None else None
        )
        node = {**values.source_site(source or {}), **node}
        if discard:
            steps.append(dict(op="expr", value=node))
            return _literal(None)
        name = self.temporary()
        steps.append(dict(op="decl", name=name, type=values.wire(kind), init=node))
        return _Value(kind, None, facts or {}).at(name)

    def global_value(self, name, value):
        self.project.global_read(name, value)
        if (
            self.initializing
            or not values.runtime(value.kind)
            or value.value["op"] != "var"
        ):
            return value
        facts = {}
        if name in self.project.single_bind_globals:
            if value.kind in values.SCALARS:
                facts = value.facts
            elif values.is_array(value.kind) and "shape" in value.facts:
                # A fixed handle and immutable region layout do not prove content.
                facts = {"shape": value.facts["shape"]}
            elif values.is_product(value.kind):
                facts = products.returned_facts(value)
            elif "stream_origin" in value.facts:
                facts = {"stream_origin": value.facts["stream_origin"]}
        return _Value(value.kind, value.value, facts)

    def ref(self, node):
        name = node["name"]
        if node.get("cell"):
            self.fail(node, "cell", "a lexical cell needs a closed activation lifetime")
        if node.get("scope") == "global":
            result = self.project.names.get(name)
            if result is not None:
                result = self.global_value(name, result)
        elif node["op"] == "python_builtin":
            slot = _slot(self.module, name)
            result = self.env.get(name, self.project.names.get(slot))
            if result is None and slot not in self.project.shadow:
                if name in {
                    "print",
                    "range",
                    "int",
                    "float",
                    "abs",
                    "min",
                    "max",
                    "len",
                    "tuple",
                    "sorted",
                }:
                    result = _Value("builtin", name)
                else:
                    result = exceptions.builtin(name) or host_protocols.builtin(
                        self, node, name
                    )
                    if result is None:
                        self.fail(
                            node,
                            "provider_contract",
                            f"builtins.{name} needs a typed provider contract",
                        )
        else:
            result = self.env.get(name)
        if result is None:
            self.fail(
                node,
                "name_binding",
                f"{name} has no definitely initialized binding at this point",
            )
        return result

    def array_readonly_call(self, node, rebound):
        """A nonactivating lookup for loop effects; unresolved dispatch stays open."""

        for part in walk_operations(node.get("target", {})):
            if part["op"] in {"var", "python_builtin"} and (
                part["name"] in rebound or _slot(self.module, part["name"]) in rebound
            ):
                return False

        def lookup(expr):
            op = expr.get("op")
            if op in {"var", "python_builtin"}:
                name = expr["name"]
                if op == "python_builtin":
                    slot = _slot(self.module, name)
                    found = self.env.get(name, self.project.names.get(slot))
                    if found is None and slot not in self.project.shadow:
                        if name in {
                            "print",
                            "range",
                            "int",
                            "float",
                            "abs",
                            "min",
                            "max",
                            "len",
                            "tuple",
                            "sorted",
                        }:
                            return _Value("builtin", name)
                        return exceptions.builtin(name) or host_protocols.builtin(
                            self, expr, name
                        )
                    return found
                scope = (
                    self.project.names if expr.get("scope") == "global" else self.env
                )
                return scope.get(name)
            if op == "member":
                base = lookup(expr["base"])
                if base and base.kind == "module":
                    return self.project.names.get(_slot(base.value, expr["field"]))
                if base and base.kind == "host_namespace":
                    return host_protocols.readonly_callable(base, expr["field"])
            return None

        target = lookup(node.get("target", {}))
        return target is not None and (
            target.kind in {"builtin", "exception_class"}
            or host_protocols.array_effect_free(target)
        )

    def store(self, target, value, steps):
        if target.get("op") == "unpack":
            items = target["items"]
            for item in items:
                if item.get("op") == "starred" or item.get("construct") == "Starred":
                    self.fail(
                        item,
                        "unpack_starred",
                        "a starred target requires a list of the remaining values",
                    )
            if values.is_product(value.kind):
                value = self.effect(value.value, value.kind, steps, facts=value.facts)
            elements = products.elements(self, target, value, steps)
            if elements is None:
                self.fail(
                    target,
                    "unpack_iterable",
                    f"unpacking {value.kind} requires a closed iterator contract",
                )
            if len(items) != len(elements):
                self.fail(
                    target,
                    "unpack_arity",
                    f"assignment expects {len(items)} items, received {len(elements)}",
                )
            # Check each nested sequence upon reaching its target. Earlier targets
            # can rebind names used to evaluate a later receiver or index.
            for item, element in zip(items, elements):
                self.store(item, element, steps)
            return
        if target.get("op") == "index":
            arrays.index(self, target, steps, rhs=value)
            return
        if target.get("op") == "member":
            if values.runtime(value.kind) and value.value["op"] != "literal":
                value = self.effect(value.value, value.kind, steps)
            base = self.evaluate(target["base"], steps)
            self.project.objects.write(
                self, target, base, target["field"], value, steps
            )
            return
        if target.get("op") != "var" or target.get("cell"):
            self.fail(
                target,
                "store",
                "attribute, subscript or closure assignment needs a state contract",
            )
        name = target["name"]
        global_ = target.get("scope") == "global"
        if global_ and not self.initializing:
            self.fail(target, "global_write", "a function writes a module namespace")
        if global_ and name in {
            _slot(self.module, field)
            for field in (
                "__builtins__",
                "__name__",
                "__package__",
                "__path__",
                "__spec__",
                "__loader__",
            )
        }:
            self.fail(
                target, "rebind", "assignment changes the module or import environment"
            )
        scope = self.project.names if global_ else self.env
        previous = scope.get(name)
        if value.kind == "sequence_method":
            value = sequences.persist_method(self, target, value, steps)
        if global_ and value.kind == "project_method" and not self.dynamic:
            value = objects.persist_method(self, target, value, steps)
        if (
            global_
            and value.kind in {"array_copy_method", "array_reshape_method"}
            and not self.dynamic
        ):
            value = arrays.persist_method(self, target, value, steps)
        if global_ and value.kind == "process_method" and not self.dynamic:
            value = processes.persist_method(self, target, value, steps)
        if not values.runtime(value.kind):
            if self.dynamic or value.kind == "range":
                self.fail(
                    target,
                    "conditional_binding",
                    f"{value.kind} needs a persistent runtime binding here",
                )
            if (
                previous is not None
                and previous != value
                and not (
                    self.initializing
                    and global_
                    and previous.kind == value.kind == "project_class"
                )
            ):
                self.fail(target, "rebind", f"symbolic identity of {name} changes")
            scope[name] = value
            return
        if previous is not None and not self.project.sequences.compatible(
            previous.kind, value.kind
        ):
            self.fail(
                target,
                "rebind",
                f"{name} changes type from {previous.kind} to {value.kind}",
            )
        types = self.project.globals if global_ else self.locals
        if name in types and not self.project.sequences.compatible(
            types[name], value.kind
        ):
            self.fail(
                target, "local_type", f"{name} has incompatible reaching value types"
            )
        types[name] = (
            values.join_arrays([types[name], value.kind])
            if name in types and values.is_array(value.kind)
            else value.kind
        )
        variable = value.at(name, **({"scope": "global"} if global_ else {}))
        scope[name] = variable
        steps.append(dict(op="assign", target=variable.value, value=value.value))

    @property
    def provider_sources(self):
        initializer = self.project.functions.get(self.scope.get("initializer"), {})
        scope = initializer.get("python_scope", {})
        if scope.get("kind") == "module" and scope.get("module") == self.module:
            return scope.get("provider_sources", {})
        return {}

    def _import_evidence(self, module):
        """Summarize where a name could come from when initialization is unclosed."""
        sources = {}
        for name, record in self.provider_sources.items():
            if name != module and not name.startswith(module + "."):
                continue
            row = {
                k: record[k]
                for k in (
                    "status",
                    "provider_id",
                    "member",
                    "sha256",
                    "resolution",
                    "reason",
                )
                if k in record
            }
            exports = {}
            for member, evidence in record.get("exports", {}).items():
                exports[member] = dict(
                    candidates=[
                        {
                            **{
                                k: candidate[k]
                                for k in ("module", "member", "name", "line", "sha256")
                                if k in candidate
                            },
                            **(
                                {"conditional_paths": True}
                                if candidate.get("conditions")
                                or any(
                                    route.get("conditions")
                                    for route in candidate.get("alternate_paths", ())
                                )
                                else {}
                            ),
                        }
                        for candidate in evidence.get("candidates", ())
                    ],
                    unresolved=sorted(
                        {u["reason"] for u in evidence.get("unresolved", ())}
                    ),
                )
            if exports:
                row["exports"] = exports
            sources[name] = row
        return (
            dict(scope="source_candidates_only", modules=sources) if sources else None
        )

    def import_origin(self, node):
        """Resolve the requested module independently of its execution contract."""
        module = node["module"]
        root = module.partition(".")[0]
        local = node.get("provider") == "project"
        source = node.get("source_provider", {})
        exact = "import_resolution" in node
        if not exact and local:
            return None
        resolved = node.get("import_resolution") if exact else source
        if not isinstance(resolved, dict) or not isinstance(source, dict):
            self.fail(
                node, "import_identity", f"invalid resolution record for {module}"
            )
        # A per-site identity and its sealed unit must agree independently of
        # which subset of exports either record was asked to describe.
        identities = [(root, source)]
        if exact:
            identities.append((module, resolved))
        for requested, evidence in identities:
            stored = self.provider_sources.get(requested)
            if (
                stored
                and evidence
                and any(
                    stored.get(key) != evidence.get(key)
                    for key in (
                        "module",
                        "status",
                        "origin",
                        "sha256",
                        "distribution",
                        "version",
                        "member",
                    )
                )
            ):
                self.fail(
                    node,
                    "import_identity",
                    f"site and unit disagree about the source of {requested}",
                )
        if exact and resolved.get("module") != module:
            self.fail(
                node,
                "import_identity",
                f"source evidence for {resolved.get('module')!r} cannot identify {module}",
            )
        status = resolved.get("status")
        if status == "project":
            if not local or module not in self.project.initializers:
                self.fail(
                    node,
                    "import_initializer",
                    f"the resolved project module {module} has no matching graph initializer",
                )
            return None
        if status == "namespace":
            self.fail(
                node,
                "import_namespace",
                f"{module} is a namespace package whose search locations and bindings need an initialization contract",
            )
        if status not in {"verified_stdlib", "verified_distribution"}:
            kind = resolved.get("resolution")
            detail = (
                resolved.get("reason") or "no verified origin for this import request"
            )
            code = "import_missing" if kind == "missing" else "import_resolution"
            origin = f" at {resolved['origin']}" if resolved.get("origin") else ""
            self.fail(
                node, code, f"resolution of {module}{origin} is incomplete ({detail})"
            )
        if local:
            self.fail(
                node,
                "import_identity",
                f"project binding for {module} conflicts with the resolved external source",
            )
        if exact:
            digest = resolved.get("sha256")
            if (
                not resolved.get("origin")
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                self.fail(
                    node,
                    "import_identity",
                    f"{module} lacks an origin and matching file or interpreter digest",
                )
            if status == "verified_distribution" and not (
                resolved.get("distribution") and resolved.get("version")
            ):
                self.fail(
                    node,
                    "import_identity",
                    f"{module} lacks the identity and version of its owning distribution",
                )
        if exact and module == root:
            keys = ["module", "status", "origin", "sha256"]
            if status == "verified_distribution":
                keys += ["distribution", "version"]
            if any(source.get(key) != resolved.get(key) for key in keys):
                self.fail(
                    node,
                    "import_identity",
                    f"root provider and exact resolution disagree for {module}",
                )
        if module != root:
            alias = source.get("aliases", {}).get(module)
            if exact and alias is not None:
                if resolved.get("canonical_module") != alias.get("module") or any(
                    resolved.get(k) != alias.get(k)
                    for k in ("status", "origin", "sha256")
                ):
                    self.fail(
                        node,
                        "import_identity",
                        f"the selected source and provider alias disagree for {module}",
                    )
            elif not exact:
                if (
                    not alias
                    or alias.get("status") != status
                    or not alias.get("sha256")
                ):
                    self.fail(
                        node,
                        "import_resolution",
                        f"the graph identifies {root} without a source identity for {module}",
                    )
        return resolved

    def imported(self, node, steps):
        req = node.get("request", {})
        mod = node["module"]
        if self.dynamic:
            self.fail(
                node,
                "conditional_import",
                f"runtime-dependent initialization of {mod} requires an import-state contract",
            )
        if not self.initializing:
            self.fail(
                node,
                "import_scope",
                f"function-local import of {mod} needs an activation-time initialization and binding contract",
            )
        if req.get("form") not in {"from", "import"} or node.get("member") == "*":
            self.fail(
                node,
                "import_form",
                f"import of {mod} needs an explicit request form and exported binding",
            )
        if not req.get("level") and "module" in req and req["module"] != mod:
            self.fail(
                node,
                "import_identity",
                f"absolute request for {req['module']} points to a different module {mod}",
            )
        origin = self.import_origin(node)
        if origin is not None:
            status = origin["status"]
            if req.get("level"):
                self.fail(
                    node,
                    "import_parent",
                    f"relative external import of {mod} needs a closed parent package",
                )
            if status == "verified_stdlib" and host_protocols.supports_import(mod):
                value = host_protocols.import_provider(self, node, steps)
                receipt = self.project.providers["stdlib:" + mod.partition(".")[0]]
                receipt["import_effects"] = node.get("effects", ["UNKNOWN"])
            elif status == "verified_distribution" and mod == "numpy":
                value = arrays.import_provider(self, node)
            elif status == "verified_distribution" and graphics.handles_import(node):
                value = self.project.graphics.import_provider(self, node, steps)
            else:
                identity = (
                    f"{origin['distribution']} {origin['version']}"
                    if origin.get("distribution") and origin.get("version")
                    else "the standard library"
                )
                self.fail(
                    node,
                    "import_semantics",
                    f"{mod} resolves to {identity} at {origin.get('origin')}. Its initialization, effects and exported bindings have no execution contract.",
                    source_evidence=self._import_evidence(mod),
                )
            self.store(node["binding"], value, steps)
            return

        def initialize(symbol):
            self.project.actual_calls.add((self.name, symbol))
            target = self.project.functions.get(symbol, {})
            scope = target.get("python_scope", {})
            if scope.get("kind") != "module":
                self.fail(
                    node, "import_initializer", f"missing module initializer {symbol}"
                )
            if symbol not in self.project.closed:
                self.project.compile(symbol, (), source=node)
                steps.append(
                    dict(
                        op="expr",
                        value=dict(
                            op="call",
                            callee=symbol,
                            args=[],
                            **values.source_site(node),
                        ),
                    )
                )
            parent, _, leaf = scope["module"].rpartition(".")
            if parent:
                slot = _slot(parent, leaf)
                value = _Value("module", scope["module"])
                if slot in self.project.names and self.project.names[slot] != value:
                    self.fail(
                        node,
                        "rebind",
                        f"import replaces existing package member {slot}",
                    )
                self.project.names[slot] = value

        for symbol in node["initializers"]:
            initialize(symbol)
        if self.project.initializers.get(mod) not in self.project.closed:
            self.fail(
                node, "import_initializer", f"incomplete or circular import of {mod}"
            )
        for optional in node.get("conditional_initializers", ()):
            when = optional["when"]
            if when.get("kind") != "package_attribute_missing":
                self.fail(
                    node, "import_guard", "unsupported conditional submodule binding"
                )
            if _slot(when["module"], when["name"]) not in self.project.names:
                if _slot(when["module"], "__getattr__") in self.project.names:
                    self.fail(
                        node,
                        "import_guard",
                        "package implements dynamic attribute lookup",
                    )
                initialize(optional["initializer"])
        if req["form"] == "import":
            value = _Value("module", mod if node.get("alias") else mod.split(".")[0])
        else:
            value = self.project.names.get(_slot(mod, node["member"]))
            if value is None:
                self.fail(
                    node,
                    "import_binding",
                    f"{mod} has no bound member {node['member']}",
                )
        self.store(node["binding"], value, steps)

    def evaluate(self, node, steps, discard=False, *, truth_context=False):
        previous = self.source_expression
        self.source_expression = node
        try:
            return self._evaluate(node, steps, discard, truth_context=truth_context)
        finally:
            self.source_expression = previous

    def _evaluate(self, node, steps, discard=False, *, truth_context=False):
        op = node["op"]
        if (
            op == "compare_chain"
            and len(node.get("operators", ())) == 1
            and node["operators"][0] in {"is", "isnot"}
        ):
            return self._evaluate(
                {
                    **node,
                    "op": "binary",
                    "operator": node["operators"][0],
                    "right": node["operands"][0],
                },
                steps,
                discard,
                truth_context=truth_context,
            )
        if node.get("site") is not None:
            self.project.live_sites.add((self.name, node["site"]))
        if truth_context:
            if op in {"binary", "compare_chain", "short_circuit"} and not (
                op == "binary" and node.get("operator") in {"is", "isnot"}
            ):
                return expressions.evaluate(self, node, steps, truth_context=True)
            return self.truth(self.evaluate(node, steps), node)
        if op == "literal":
            value = node["value"]
            if type(value) not in (bool, int, float, str, type(None)) or (
                type(value) is int and not -(1 << 63) <= value < 1 << 63
            ):
                self.fail(
                    node,
                    "literal_contract",
                    "literal requires another numeric or object representation",
                )
            if type(value) is str:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    self.fail(node, "encoding", "text contains an unpaired surrogate")
            return _literal(value)
        if op in {"var", "python_builtin"}:
            return self.ref(node)
        if op == "binary" and node.get("operator") in {"is", "isnot"}:
            left = self.evaluate(node["left"], steps)
            if values.runtime(left.kind) and left.value.get("op") != "literal":
                left = self.effect(left.value, left.kind, steps, facts=left.facts)
            right = self.evaluate(node["right"], steps)
            result = sequences.identity(node, left, right)
            return (
                result
                if result is not None
                else expressions.apply_binary(
                    self, node, node["operator"], left, right, steps, discard
                )
            )
        if op in {"format_string", "format_field"}:
            return text.evaluate(self, node, steps, discard)
        if op == "aggregate":
            return arrays.capture_sequence(self, node, steps)
        if op == "index":
            return arrays.index(self, node, steps, discard=discard)
        if op == "process_stream":
            base = self.evaluate(node["base"], steps)
            result = processes.member(self, node, base, node["field"], steps)
            if result is None:
                self.fail(
                    node,
                    "process_receiver",
                    "the process channel has no proven receiver",
                )
            return result
        if op == "member":
            base = self.evaluate(node["base"], steps)
            if values.object_name(base.kind) is not None:
                return self.project.objects.member(
                    self, node, base, node["field"], steps
                )
            if base.kind == "project_class":
                method = self.project.objects.layouts[base.value].methods.get(
                    node["field"]
                )
                if method:
                    return _Value("function", method)
                self.fail(
                    node,
                    "class_attribute",
                    f"no ordinary method binding for {base.value}.{node['field']}",
                )
            if base.kind == "handle:text_stream" or processes.is_process(base.kind):
                base = self.effect(base.value, base.kind, steps, facts=base.facts)
            process_member = processes.member(self, node, base, node["field"], steps)
            if process_member is not None:
                return process_member
            host_member = host_protocols.member(self, node, base, node["field"])
            if host_member is not None:
                return host_member
            if base.kind == "provider_namespace":
                return arrays.provider_member(self, node, node["field"])
            if base.kind == "graphics_namespace":
                return self.project.graphics.member(self, node, base, node["field"])
            if values.is_array(base.kind):
                return arrays.member(self, node, base, node["field"], steps)
            sequence_member = sequences.member(self, node, base, node["field"])
            if sequence_member is not None:
                return sequence_member
            if base.kind == "module":
                slot = _slot(base.value, node["field"])
                value = self.project.names.get(slot)
                if value is not None:
                    return self.global_value(slot, value)
            self.fail(
                node,
                "attribute",
                f"unbound module or dynamic attribute {node['field']}",
            )
        if op in {
            "call",
            "library_call",
            "python_invoke",
            "process_launch",
            "process_wait",
            "process_poll",
            "process_communicate",
        }:
            return self.call(node, steps, discard)
        if op == "unary" and node["operator"] != "not":
            return arrays.scalar_unary(
                self, node, self.evaluate(node["value"], steps), steps, discard
            )
        if op == "unary" and node["operator"] == "not":
            value = self.evaluate(node["value"], steps, truth_context=True)
            if value.value["op"] == "literal":
                return _literal(not value.value["value"])
            return _Value(
                "bool", dict(op="unary", operator="not", type="bool", value=value.value)
            )
        if op in {"binary", "compare_chain", "short_circuit"}:
            return expressions.evaluate(self, node, steps, discard)

        self.fail(node, "operation_contract", f"unclosed expression {op}")

    def call(self, node, steps, discard):
        target = self.evaluate(node["target"], steps)
        if target.kind == "sequence_method":
            receiver, method = target.value
            target = _Value(
                target.kind,
                (
                    self.effect(
                        receiver.value, receiver.kind, steps, facts=receiver.facts
                    ),
                    method,
                ),
            )
        if target.kind == "project_method":
            method, receiver = target.value
            if receiver.value.get("scope") == "global":
                receiver = self.effect(receiver.value, receiver.kind, steps)
            target = _Value("project_method", (method, receiver))

        def argument(expr):
            value = self.evaluate(expr, steps)
            if values.runtime(value.kind) and value.value["op"] != "literal":
                return self.effect(value.value, value.kind, steps, facts=value.facts)
            return value

        args = [argument(v) for v in node["args"]]
        kwargs = {}
        for item in node.get("keywords", ()):
            key = item["name"]
            if key is None or key in kwargs:
                self.fail(node, "keyword_binding", "dynamic or repeated keyword")
            kwargs[key] = argument(item["value"])
        sequence_call = sequences.invoke(
            self, node, target, args, kwargs, steps, discard
        )
        if sequence_call is not None:
            return sequence_call
        if target.kind == "project_class":
            return self.project.objects.construct(
                self, node, target, args, kwargs, steps
            )
        if target.kind == "project_method":
            method, receiver = target.value
            return self.invoke_project(
                node, method, [receiver, *args], kwargs, steps, discard
            )
        if target.kind in {
            "array_constructor",
            "array_copy_method",
            "array_reshape_method",
        }:
            return arrays.construct(self, node, target, args, kwargs, steps, discard)
        process_call = processes.invoke(
            self, node, target, args, kwargs, steps, discard
        )
        if process_call is not None:
            return process_call
        if target.kind == "builtin" and target.value in {"min", "max"}:
            return scalars.extreme(self, node, target.value, args, kwargs, steps)
        if target.kind == "builtin" and target.value == "len":
            if not kwargs and len(args) == 1 and values.is_sequence(args[0].kind):
                return sequences.size(args[0])
            if kwargs or len(args) != 1 or not products.is_sequence(args[0]):
                self.fail(node, "length_contract", "len requires one closed sequence")
            return _literal(
                len(values.product_members(args[0].kind))
                if values.is_product(args[0].kind)
                else len(args[0].value)
            )
        if target.kind == "builtin" and target.value == "tuple":
            if (
                kwargs
                or len(args) != 1
                or args[0].kind != "tuple"
                and not values.is_product(args[0].kind)
            ):
                self.fail(
                    node,
                    "product_constructor",
                    "tuple conversion requires an existing immutable tuple",
                )
            return args[0]
        if target.kind == "builtin" and target.value in {"int", "float", "abs"}:
            result = scalars.construct(self, node, target.value, args, kwargs)
            self.project.decisions.append(
                dict(
                    code=(
                        "numeric_intrinsic"
                        if target.value == "abs"
                        else "numeric_conversion"
                    ),
                    symbol=self.name,
                    location=node.get("location"),
                    constructor=target.value,
                    source_types=[v.kind for v in args],
                    source_facts=[
                        {k: v for k, v in item.facts.items() if k in {"exact", "range"}}
                        for item in args
                    ],
                    result=result.kind,
                    result_facts=result.facts,
                    proof="source numeric protocol and proven representation domain; arguments retain evaluation order",
                )
            )
            return result
        if target.kind == "graphics_callable":
            return self.project.graphics.invoke(
                self, node, target.value, args, kwargs, steps, discard
            )
        if target.kind == "exception_class":
            return exceptions.construct(
                self, node, target, args, kwargs, steps, discard
            )
        if target.kind == "builtin" and target.value == "range":
            if kwargs or not 1 <= len(args) <= 3 or any(v.kind != "i64" for v in args):
                self.fail(
                    node,
                    "range_contract",
                    "range requires one to three index-width integers",
                )
            if len(args) == 3 and (
                type(args[2].exact) is not int or args[2].exact == 0
            ):
                self.fail(node, "range_step", "a nonzero constant step is required")
            return _Value("range", tuple(args))
        if target.kind == "builtin" and target.value == "print":
            args = [text.display(self, node, value, steps) for value in args]
            if any(
                a.kind not in {"bool", "i64", "f64", "numpy_f64", "text", "unit"}
                for a in args
            ):
                self.fail(
                    node,
                    "print_contract",
                    "no text formatting contract for this scalar or aggregate",
                )
            if set(kwargs) - {"sep", "end", "flush", "file"}:
                self.fail(node, "print_contract", "unrecognized print keyword")
            sep, end = (kwargs.get(k, _literal(None)) for k in ("sep", "end"))
            sep = _literal(" ") if sep.kind == "unit" else sep
            end = _literal("\n") if end.kind == "unit" else end
            flush = kwargs.get("flush", _literal(False))
            stream = kwargs.get("file", _literal(None))
            if (
                sep.kind != "text"
                or end.kind != "text"
                or flush.kind != "bool"
                or stream.kind not in {"unit", "handle:text_stream"}
            ):
                self.fail(
                    node,
                    "print_contract",
                    "print requires a text stream, text separators and a Boolean flush flag",
                )
            call = dict(
                op="text_print",
                args=[self.scalar(a, node) for a in args],
                sep=sep.value,
                end=end.value,
                flush=flush.value,
                type="unit",
                effects=["IO", "STATE"],
                encoding="utf-8",
                **values.source_site(node),
                **values.argument_evidence(args, sep=sep, end=end, flush=flush),
                scalar_format=dict(
                    boolean_true="True",
                    boolean_false="False",
                    unit="None",
                    integer_base=10,
                ),
                keyword_order=[
                    "stream" if k == "file" else k
                    for k in kwargs
                    if k != "file" or stream.kind != "unit"
                ],
            )
            if stream.kind != "unit":
                call["stream"] = stream.value
            if any(a.kind in {"f64", "numpy_f64"} for a in args):
                call["scalar_format"]["float"] = "shortest_binary64"
            return self.effect(call, "unit", steps, discard)
        host_call = host_protocols.invoke(
            self, node, target, args, kwargs, steps, discard
        )
        if host_call is not None:
            return host_call
        if target.kind != "function" or target.value not in self.project.functions:
            self.fail(
                node,
                "dispatch_contract",
                f"a callable has no reachable project body ({target.kind}: {target.value})",
            )
        return self.invoke_project(node, target.value, args, kwargs, steps, discard)

    def invoke_project(self, node, name, args, kwargs, steps, discard):
        formals = self.project.functions[name]["params"]
        if len(args) > len(formals):
            self.fail(
                node, "argument_binding", f"too many positional arguments for {name}"
            )
        bound = []
        for i, formal in enumerate(formals):
            param, mode = formal["name"], formal.get("binding")
            if (
                i < len(args)
                and mode in {"posonly", "positional"}
                and param not in kwargs
            ):
                bound.append(args[i])
            elif (
                i >= len(args)
                and mode in {"positional", "keywordonly"}
                and param in kwargs
            ):
                bound.append(kwargs.pop(param))
            else:
                self.fail(
                    node,
                    "argument_binding",
                    f"missing, repeated or unsupported binding of {name}.{param}",
                )
        if kwargs:
            self.fail(node, "argument_binding", f"unconsumed keywords for {name}")
        operands = [self.scalar(v, node) for v in bound]
        function = self.project.compile(name, tuple(v.kind for v in bound), source=node)
        parameter_types = [param["type"] for param in function["params"]]
        if self.project.sequences.resolve(
            parameter_types, strict=False
        ) != self.project.sequences.resolve(
            [values.wire(v.kind) for v in bound], strict=False
        ):
            self.fail(
                node,
                "argument_types",
                "bound arguments disagree with the closed parameter types",
            )
        result = _Value(
            self.project.result_kinds[name], None, self.project.result_facts[name]
        )
        result_fact = (
            None if function.get("noreturn") else values.scalar_evidence(result)
        )
        self.project.actual_calls.add((self.name, name))
        if name not in self.project.external_array_writes:
            self.project.external_array_writes[name] = arrays.may_write_external(
                function, self.project.closed
            )
        if self.project.external_array_writes[name]:
            self.project.array_state.invalidate()
        self.project.sequences.invalidate()
        call = dict(
            op="call",
            callee=name,
            args=operands,
            **values.source_site(node),
            **values.argument_evidence(bound),
            **({"result_facts": result_fact} if result_fact is not None else {}),
            type=function["result"],
            parameter_types=parameter_types,
            **({"noreturn": True} if function.get("noreturn") else {}),
        )
        return self.effect(
            call,
            result.kind,
            steps,
            discard,
            facts=result.facts,
        )

    def record_return(self, value):
        self.returns.add(value.kind)
        facts = products.returned_facts(value)
        if self.return_facts is None:
            self.return_facts = facts
        elif not values.identical(self.return_facts, facts):
            self.return_facts = {}

    def truth(self, value, node):
        if values.is_sequence(value.kind):
            return _Value(
                "bool",
                dict(
                    op="binary",
                    operator="ne",
                    type="bool",
                    left=sequences.size(value).value,
                    right=_literal(0).value,
                ),
            )
        if products.is_sequence(value):
            return _literal(
                bool(values.product_members(value.kind))
                if values.is_product(value.kind)
                else bool(value.value)
            )
        if values.object_name(value.kind) is not None:
            return _literal(True)
        if values.is_array(value.kind):
            self.fail(
                node,
                "array_truth",
                "an array requires an explicit scalar truth or reduction contract",
            )
        expr = self.scalar(value, node)
        if "exact" in value.facts:
            return _literal(bool(value.exact))
        if value.kind in {"bool", "numpy_bool", "boolean_choice"}:
            return _Value("bool", value.value, value.facts)
        if value.kind == "unit":
            return _literal(False)
        if value.kind not in {"i64", "f64", "numpy_f64", "binary64_choice", "text"}:
            self.fail(
                node, "truth_contract", f"{value.kind} has no closed truth conversion"
            )
        return _Value(
            "bool",
            dict(
                op="binary",
                operator="ne",
                left=expr,
                right=_literal(
                    {
                        "i64": 0,
                        "f64": 0.0,
                        "numpy_f64": 0.0,
                        "binary64_choice": 0.0,
                        "text": "",
                    }[value.kind]
                ).value,
                type="bool",
            ),
        )

    def snapshot(self):
        return self.env.copy(), self.project.names.copy()

    def restore(self, state):
        self.env, self.project.names = (scope.copy() for scope in state)

    def merge(self, states):
        self.env, self.project.names = (
            _join(group, self.project.sequences) for group in zip(*states)
        )

    def conditional(self, node, steps):
        check = self.evaluate(node["condition"], steps, truth_context=True)
        if check.value["op"] == "literal":
            self.skip_expression(
                node["else"] if check.value["value"] else node["then"],
                dict(
                    rule="constant_condition",
                    guard_site=node.get("site"),
                    truth=check.value["value"],
                ),
            )
            return self.sequence(
                node["then"] if check.value["value"] else node["else"], steps
            )
        start = self.snapshot()
        arms, exits = [], []
        self.dynamic += 1
        for side in ("then", "else"):
            self.restore(start)
            out = []
            if self.sequence(node[side], out):
                exits.append(self.snapshot())
            arms.append(out)
        self.dynamic -= 1
        steps.append(
            dict(op="if", condition=check.value, then=arms[0], **{"else": arms[1]})
        )
        if exits:
            self.merge(exits)
        return bool(exits)

    def range_loop(self, node, steps):
        domain = self.evaluate(node["range"], steps)
        if (
            values.is_sequence(domain.kind)
            and node.get("protocol") == "iterator"
            and node["target"].get("op") == "var"
        ):
            return sequences.iteration(self, node, domain, steps)
        if (
            domain.kind == "handle:text_stream"
            and node.get("protocol") == "iterator"
            and node["target"].get("op") == "var"
        ):
            stream = self.effect(domain.value, domain.kind, steps, facts=domain.facts)
            setup = []
            line = processes.next_line(self, stream, setup)
            condition = dict(
                op="binary",
                operator="ne",
                type="bool",
                left=line.value,
                right=_literal("").value,
            )
            return (
                dict(op="while", body=[], condition=condition),
                line,
                None,
                False,
                False,
                setup,
            )
        if (
            node.get("protocol") != "iterator"
            or domain.kind != "range"
            or node["target"].get("op") != "var"
        ):
            self.fail(
                node,
                "iterator_contract",
                "counted iteration requires a closed range and one lexical target",
            )
        args = domain.value
        start, stop = (_literal(0), args[0]) if len(args) == 1 else args[:2]
        step = args[2] if len(args) == 3 else _literal(1)
        stride = step.exact
        if type(stride) is not int or not stride:
            self.fail(node, "range_step", "a nonzero constant step is required")
        # Range operands are already captured in source order by call().
        positive = stride > 0
        al, ah = start.bounds
        bl, bh = stop.bounds
        empty = al >= bh if positive else ah <= bl
        nonempty = ah < bl if positive else al > bh
        distance = bh - al if positive else ah - bl
        count = max(0, (distance + abs(stride) - 1) // abs(stride))
        low, high = (al, bh - 1) if positive else (bl + 1, ah)
        cursor = self.temporary()
        iv = _Value("i64", None, {"range": (low, high)}).at(cursor)
        condition = dict(
            op="binary",
            operator="lt" if positive else "gt",
            type="bool",
            left=iv.value,
            right=stop.value,
        )
        advance = dict(
            op="binary", operator="add", type="i64", left=iv.value, right=step.value
        )
        if not empty and (high + stride > (1 << 63) - 1 or low + stride < -(1 << 63)):
            # The terminal mathematical index may exceed int64. Substituting
            # the exclusive end when that happens leaves precisely the same visits.
            limit = (1 << 63) - 1 - stride if positive else -(1 << 63) - stride
            guard = dict(
                op="binary",
                operator="gt" if positive else "lt",
                left=iv.value,
                right=_literal(limit).value,
                type="bool",
            )
            advance = dict(
                op="select",
                condition=guard,
                then=stop.value,
                **{"else": advance},
                type="i64",
            )
        loop = dict(
            op="for",
            init=[dict(op="decl", name=cursor, type="i64", init=start.value)],
            condition=condition,
            step=dict(
                op="update", operator="set", target=iv.value, value=advance, type="i64"
            ),
            body=[],
        )
        return loop, iv, count, empty, nonempty, []

    def widen_loop(self, node, iv, count):
        body = node["body"]
        writes = {}
        nested = False
        for op in walk_operations(body):
            if op["op"] == "exception_handler" and (count is None or count > 1):
                alias = op.get("binding")
                if alias and alias.get("op") == "var":
                    env = (
                        self.project.names
                        if alias.get("scope") == "global"
                        else self.env
                    )
                    env.pop(alias["name"], None)
            if op["op"] in {"for", "while"}:
                nested = True
            for dest in values.written_targets(op):
                if dest.get("op") == "var":
                    writes.setdefault(
                        (dest.get("scope") == "global", dest["name"]), []
                    ).append(op)
        if iv is not None:
            target = node["target"]
            index = (target.get("scope") == "global", target["name"])
            writes.setdefault(index, [])
        else:
            index = None

        def delta(port):
            if port.get("op") == "literal" and type(port.get("value")) is int:
                return (port["value"],) * 2
            if port.get("op") == "var":
                key = port.get("scope") == "global", port["name"]
                if key == index and not writes[key]:
                    return iv.bounds
                binding = (self.project.names if key[0] else self.env).get(key[1])
                if key not in writes and binding and binding.kind == "i64":
                    return binding.bounds
            return None

        for (global_, name), sites in writes.items():
            env = self.project.names if global_ else self.env
            value = env.get(name)
            if value is None:
                continue
            if not values.runtime(value.kind):
                self.fail(
                    node,
                    "loop_binding",
                    f"loop-carried symbolic binding {name} needs a runtime representation",
                )
            facts = {}
            if value.kind == "i64" and count and not nested and sites:
                down = up = 0
                for site in sites:
                    rhs = site.get("value", {})
                    op = site.get("operator")
                    if (
                        site["op"] == "assign"
                        and site["target"].get("op") == "var"
                        and rhs.get("op") == "binary"
                        and rhs["left"].get("op") == "var"
                        and rhs["left"]["name"] == name
                        and (rhs["left"].get("scope") == "global") == global_
                    ):
                        op, rhs = rhs["operator"], rhs["right"]
                    elif site["op"] != "update":
                        break
                    interval = delta(rhs)
                    if op not in {"add", "sub"} or interval is None:
                        break
                    a, b = interval
                    if op == "sub":
                        a, b = -b, -a
                    down += min(a, 0)
                    up += max(b, 0)
                else:
                    lower = value.bounds[0] + (count - 1) * down
                    upper = value.bounds[1] + (count - 1) * up
                    if -(1 << 63) <= lower <= upper < 1 << 63:
                        facts["range"] = (lower, upper)
            env[name] = _Value(value.kind, value.value, facts)

    def refine_loop_test(self, test):
        if test.get("op") != "binary":
            return
        op = test["operator"]
        left, right = test["left"], test["right"]
        if right.get("op") == "var" and left.get("op") == "literal":
            left, right = right, left
            op = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge", "eq": "eq"}.get(op)
        if (
            left.get("op") != "var"
            or right.get("op") != "literal"
            or type(right["value"]) is not int
        ):
            return
        env = self.project.names if left.get("scope") == "global" else self.env
        v = env.get(left["name"])
        if v is None or v.kind != "i64":
            return
        low, high = v.bounds
        b = right["value"]
        if op in {"le", "lt", "eq"}:
            high = min(high, b - (op == "lt"))
        if op in {"ge", "gt", "eq"}:
            low = max(low, b + (op == "gt"))
        if low <= high:
            env[left["name"]] = _Value("i64", v.value, {"range": (low, high)})

    def loop(self, node, steps):
        counted = node["op"] == "for"
        allocation_mark = self.project.array_state.enter_loop(self, node)
        start = self.snapshot()
        if counted:
            loop, iv, count, empty, nonempty, test_setup = self.range_loop(node, steps)
        else:
            iv, count = None, None
            loop = dict(op="while", body=[])
            test_setup = []
        self.widen_loop(node, iv, count)
        header = self.snapshot()
        flag = self.temporary() if node.get("else") else None
        if counted:
            condition = loop["condition"]
        else:
            test = self.evaluate(node["condition"], test_setup, truth_context=True)
            empty, nonempty = test.exact is False, test.exact is True
            condition = test.value
            self.refine_loop_test(condition)
        loop["condition"] = condition
        if test_setup or flag:
            # Mark exhaustion at the condition edge. A pending break can be
            # replaced by a continue or a return inside source-level finally.
            finish = []
            if flag:
                steps.append(
                    dict(op="decl", name=flag, type="bool", init=_literal(False).value)
                )
                finish.append(
                    dict(
                        op="assign",
                        target=_Value("bool", None).at(flag).value,
                        value=_literal(True).value,
                    )
                )
            loop["condition"] = _literal(True).value
            loop["body"].extend(test_setup)
            loop["body"].append(
                dict(
                    op="if",
                    condition=dict(
                        op="unary", operator="not", value=condition, type="bool"
                    ),
                    then=[*finish, dict(op="break")],
                    **{"else": []},
                )
            )
        flow = {"break": [], "continue": []}
        self.loops.append(flow)
        self.dynamic += 1
        if not empty:
            if counted:
                self.store(node["target"], iv, loop["body"])
            if self.sequence(node["body"], loop["body"]):
                flow["continue"].append(self.snapshot())
        else:
            self.skip_expression(
                node["body"], dict(rule="empty_loop", guard_site=node.get("site"))
            )
        self.dynamic -= 1
        self.loops.pop()
        broke = flow["break"]
        steps.append(loop)
        if empty:
            normal = [start]
        elif counted:
            normal = ([start] if not nonempty else []) + flow["continue"]
        else:
            normal = [] if nonempty else [header, *flow["continue"]]
        exits = list(broke)
        if normal:
            self.merge(normal)
            else_body = []
            self.dynamic += bool(broke)
            if self.sequence(node.get("else", []), else_body):
                exits.append(self.snapshot())
            self.dynamic -= bool(broke)
            if else_body:
                if broke:
                    steps.append(
                        dict(
                            op="if",
                            condition=dict(op="var", type="bool", name=flag),
                            then=else_body,
                            **{"else": []},
                        )
                    )
                else:
                    steps.extend(else_body)
        if exits:
            self.merge(exits)
        self.project.array_state.leave_loop(allocation_mark)
        return bool(exits)

    def sequence(self, body, steps):
        parent = self.region
        self.region = region = _Region(steps, parent, body)
        if parent is None:
            self.root_region = region
        else:
            parent.children.append(region)
        try:
            falls = self._sequence(body, steps, region)
            region.boundary(None)
            return falls
        finally:
            self.region = parent

    def _sequence(self, body, steps, region):
        def stop():
            self.skip_expression(
                body[index + 1 :],
                dict(
                    rule="no_normal_successor",
                    guard_site=node.get("site"),
                    operator=node["op"],
                ),
            )
            return False

        for index, node in enumerate(body):
            region.boundary(node)
            if node.get("site") is not None:
                self.project.live_sites.add((self.name, node["site"]))
            op = node["op"]
            first = len(steps)
            if (
                op == "decl"
                and node.get("initial_binding") == "unbound"
                or op == "pass"
            ):
                continue
            if op == "expr":
                self.evaluate(node["value"], steps, discard=True)
            elif op == "assign":
                self.store(node["target"], self.evaluate(node["value"], steps), steps)
            elif op == "python_import":
                self.imported(node, steps)
            elif op == "python_define_class":
                self.store(
                    node["target"], self.project.objects.define(self, node), steps
                )
            elif op == "python_define":
                if not objects.plain_definition(node):
                    self.fail(
                        node,
                        "function_creation",
                        "definition requires default, annotation, type parameter or decorator evaluation",
                    )
                self.store(node["target"], _Value("function", node["symbol"]), steps)
            elif op == "raise":
                exceptions.throw(self, node, steps)
                return stop()
            elif op == "try":
                if not exceptions.protected(self, node, steps):
                    return stop()
            elif op == "with_context":
                if not exceptions.resources(self, node, steps):
                    return stop()
            elif op == "return":
                value = (
                    self.evaluate(node["value"], steps)
                    if "value" in node
                    else _literal(None)
                )
                steps.append(dict(op="return", value=self.scalar(value, node)))
                self.record_return(value)
                return stop()
            elif op == "update" and node["target"].get("op") in {"var", "member"}:
                target = node["target"]
                receiver = None
                if target["op"] == "member":
                    receiver = self.evaluate(target["base"], steps)
                    receiver = self.effect(
                        self.scalar(receiver, target), receiver.kind, steps
                    )
                    left = self.project.objects.member(
                        self, target, receiver, target["field"], steps
                    )
                else:
                    left = self.ref(target)
                left = self.effect(left.value, left.kind, steps, facts=left.facts)
                right = self.evaluate(node["value"], steps)
                if arrays.update(self, node, left, right, steps):
                    result = left
                else:
                    result = expressions.apply_binary(
                        self, node, node["operator"], left, right, steps
                    )
                if receiver is not None:
                    self.project.objects.write(
                        self, target, receiver, target["field"], result, steps
                    )
                else:
                    self.store(target, result, steps)
            elif op in {"break", "continue"} and self.loops:
                flow = self.loops[-1]
                flow[op].append(self.snapshot())
                steps.append(dict(op=op))
                return stop()
            elif op == "if":
                if not self.conditional(node, steps):
                    return stop()
            elif op in {"for", "while"}:
                if not self.loop(node, steps):
                    return stop()
            else:
                self.fail(node, "statement_contract", f"unclosed statement {op}")
            for statement in steps[first:]:
                if statement["op"] in {"decl", "expr", "assign"}:
                    value = (
                        statement.get("init")
                        if statement["op"] == "decl"
                        else statement.get("value")
                    )
                    if value and value.get("noreturn"):
                        return stop()
        return True

    def prefix_fragment(self):
        region = self.root_region
        record = region.fragment() if region else dict(body=[])
        record.update(
            completion="prefix",
            symbol=self.name,
            params=self.params,
            **values.source_site((region.current if region else None) or self.fn),
        )
        if self.name in self.project.activations:
            record["activation"] = self.project.activations[self.name]
        return record

    def lower(self):
        if self.initializing:
            self.project.names[_slot(self.module, "__name__")] = _literal(
                "__main__" if self.scope["main"] else self.module
            )
        body = []
        falls = self.sequence(self.fn["body"], body)
        if falls and not self.initializing:
            body.append(dict(op="return", value=_literal(None).value))
            self.record_return(_literal(None))
        if self.returns and all(values.is_sequence(kind) for kind in self.returns):
            first = next(iter(self.returns))
            for kind in self.returns:
                if not self.project.sequences.compatible(first, kind):
                    self.fail(
                        self.fn,
                        "result_contract",
                        "return paths have heterogeneous sequence element protocols",
                    )
            self.returns = {self.project.sequences.resolve_kind(first)}
        if len(self.returns) > 1 or self.initializing and self.returns:
            self.fail(
                self.fn,
                "result_contract",
                f"incompatible return paths {sorted(self.returns)}",
            )
        result = "void" if self.initializing else next(iter(self.returns), "unit")
        self.project.result_kinds[self.name] = result
        self.project.result_facts[self.name] = self.return_facts or {}
        parameters = {p["name"] for p in self.params}
        declarations = [
            dict(op="decl", name=n, type=values.wire(t))
            for n, t in self.locals.items()
            if n not in parameters
        ]
        return dict(
            name=self.name,
            params=self.params,
            result=values.wire(result),
            location=self.fn.get("location"),
            body=[*declarations, *body],
            **({"noreturn": True} if not falls and not self.returns else {}),
        )


def normalize_protocols(module):
    """Certify each reachable protocol, retaining closed signatures on a partial result."""
    functions = {
        f["name"]: f
        for f in module["functions"]
        if f.get("execution_kind") == "python_protocol"
    }
    if not functions:
        return module, [], []
    project = _Project(functions)
    project.execution_images = module.get("execution_images", [])
    project.text_profiles = module.get("text_profiles", {})
    issue = None
    try:
        entry = module["entry"]
        if (
            entry not in functions
            or functions[entry].get("python_scope", {}).get("kind") != "module"
        ):
            raise _Unclosed(
                entry,
                functions.get(entry, {}),
                "project_entry",
                "module entry required for closed Python name binding",
            )
        project.compile(entry, ())
        omitted = set(module.get("python_candidate_only", ())).difference(
            project.closed
        )
        omitted.update(project.objects.class_bodies)
        omitted.update(project.inactive_bodies(module.get("activation_graph")))
        for name, fn in functions.items():
            if name not in project.closed and name not in omitted:
                raise _Unclosed(
                    name,
                    fn,
                    "reachable_protocol",
                    "no proven activation or call signature for this reachable body",
                )
        for obligation in module.get("protocol_obligations", ()):
            if obligation.get("symbol") in omitted or project.objects.discharged(
                obligation
            ):
                continue
            if obligation.get("field") not in {
                "dispatch",
                "native_binding",
                "provider_binding",
            }:
                raise _Unclosed(
                    obligation["symbol"],
                    {
                        "location": {
                            k: obligation[k]
                            for k in ("file", "line", "column")
                            if k in obligation
                        }
                    },
                    "frontend_obligation",
                    obligation["detail"],
                )
        try:
            project.sequences.freeze(project)
        except ValueError as error:
            raise _Unclosed(
                entry, functions[entry], "sequence_element_type", str(error)
            ) from error
    except _Unclosed as error:
        issue = error.issue
        project.sequences.freeze(project, partial=True)
    contracts = [
        dict(
            symbol=name,
            params=[p["type"] for p in fn["params"]],
            result=fn["result"],
        )
        for name, fn in project.closed.items()
    ]
    module["protocol_analysis"] = dict(
        entry=module["entry"],
        closed=contracts,
        single_bind_globals=sorted(project.single_bind_globals),
        status="INCOMPLETE" if issue else "READY",
        **(
            {"operator_requirements": project.operator_requirements()}
            if project.required_operators
            else {}
        ),
        **({"first_gap": issue} if issue else {}),
        **(
            {"process_requirements": list(project.process_requests.values())}
            if project.process_requests
            else {}
        ),
        **(
            {"graphics_operations": sorted(project.graphics.operations)}
            if project.graphics.operations
            else {}
        ),
        **(
            {
                "providers": list(project.providers.values()),
                "array_operations": sorted(project.array_ops),
            }
            if project.providers
            else {}
        ),
    )
    if issue:
        module["protocol_analysis"].update(
            closed_fragments=[
                dict(
                    symbol=name,
                    completion="complete",
                    function=fn,
                    **values.source_site(functions[name]),
                    **(
                        {"activation": project.activations[name]}
                        if name in project.activations
                        else {}
                    ),
                )
                for name, fn in project.closed.items()
            ],
            prefix_fragments=list(reversed(project.prefixes)),
        )
        return module, project.decisions, [issue]
    changes = project.decisions + [
        dict(
            **contract,
            code="closed_python_protocol",
            proof="ordered module initialization, reaching lexical bindings and consistent call types",
        )
        for contract in contracts
    ]
    for layout in project.objects.layouts.values():
        methods = set(layout.methods.values())
        changes.append(
            dict(
                code="closed_instance_layout",
                record=layout.name,
                namespace=layout.body_symbol,
                fields=sorted(layout.fields),
                activated_methods=sorted(methods & project.closed.keys()),
                inactive_methods=sorted(methods - project.closed.keys()),
                proof="ordinary class and method bindings close throughout the entry call graph with shared identity, checked field presence and observable writes",
            )
        )
    module["functions"] = [
        project.closed.get(fn["name"], fn)
        for fn in module["functions"]
        if fn["name"] not in omitted
    ]
    module["records"].extend(
        layout.record() for layout in project.objects.layouts.values()
    )
    module["globals"].extend(
        dict(op="decl", name=n, type=values.wire(t)) for n, t in project.globals.items()
    )
    if project.array_ops:
        module.setdefault("operator_requirements", {})["ndarray"] = dict(
            device="cpu", element="f64", operations=sorted(project.array_ops)
        )
    if project.process_ops:
        module.setdefault("operator_requirements", {})["process"] = dict(
            operations=sorted(project.process_ops),
            binding=(
                "explicit_targets" if project.process_requests else "declared_images"
            ),
        )
    if project.required_operators:
        module.setdefault("operator_requirements", {}).update(
            project.operator_requirements()
        )
    if project.graphics.operations:
        module.setdefault("operator_requirements", {})[
            "graphics"
        ] = project.graphics.requirement()
    module["externs"] = [
        e for e in module["externs"] if e.get("calling_convention") != "python"
    ]
    declarations = module["declarations"]
    for symbol in omitted:
        declarations.pop(symbol, None)
    module.pop("python_class_candidates", None)
    module.pop("python_candidate_only", None)
    for name, fn in project.closed.items():
        # Keep independent native declarations for the downstream ABI comparison.
        # Only graph-level Python object placeholders acquire the inferred contract.
        variants = [
            d
            for d in declarations.get(name, ())
            if all(
                not (isinstance(t, dict) and t.get("kind") == "python_object")
                for t in (d["result"], *(p["type"] for p in d["params"]))
            )
        ]
        declarations[name] = [
            *variants,
            {k: fn[k] for k in ("name", "params", "result")},
        ]
    module.pop("protocol_obligations", None)
    return project.sequences.resolve(module), changes, []
