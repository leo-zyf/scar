"""Resolve program entries and reconstruct executable operations from graph ports."""

from collections import defaultdict
from .contracts import publish_program
from scar.contract import (
    digest,
    json_bytes,
    python_provider_id,
    read_member,
    semantic_form as semantics,
    validate_images,
    walk_operations as _operations,
    validate_text_profiles,
    validate_numeric_contract,
    validate_source_literal,
)


def _provider_sources(records):
    """Keep the source identity independent of this unit's export queries."""
    sources = {}
    for module, record in records.items():
        if record.get("module") != module:
            raise ValueError(
                f"provider record does not identify requested module {module}"
            )
        for export in record.get("exports", {}).values():
            for candidate in export.get("candidates", ()):
                if candidate.get("kind") == "literal":
                    validate_source_literal(candidate)
        if record.get("status") not in {"verified_distribution", "verified_stdlib"}:
            sources[module] = record
            continue
        sources[module] = dict(record, provider_id=python_provider_id(record))
    return sources


_SITE_OPERATIONS = {
    "map",
    "for",
    "assign",
    "binary",
    "if",
    "while",
    "try",
    "with_context",
    "expr",
    "return",
    "raise",
    "break",
    "continue",
    "call",
    "library_call",
    "intrinsic",
    "member",
    "process_launch",
    "process_stream",
    "process_wait",
    "process_poll",
    "process_communicate",
    "python_invoke",
    "python_import",
    "python_define_class",
    "compare_chain",
    "short_circuit",
}


def _definition(graph):
    nodes = graph["nodes"]
    active = set()

    def read(value):
        if isinstance(value, list):
            return [read(v) for v in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"port"}:
            port = value["port"]
            if type(port) is not int or not 0 <= port < len(nodes) or port in active:
                raise ValueError(f"invalid operand port {port} in {graph['id']}")
            active.add(port)
            result = read(nodes[port])
            if result["op"] in _SITE_OPERATIONS:
                result["site"] = f"{graph['id']}/{port}"
            active.remove(port)
            return result
        return {key: read(field) for key, field in value.items()}

    return {**read(graph["signature"]), "body": read(graph["entry"])}


def _validate_numeric_evidence(unit, payload):
    """Bind numerical source evidence without reopening compiler inputs."""
    facts = unit.get("source_facts", {})
    contract = facts.get("source_numeric_contract")
    if contract is None:
        return
    if (
        contract.get("scope") != "compilation_unit"
        or not contract.get("source", {}).get("sha256")
        or contract.get("source", {}).get("sha256")
        != facts.get("files", {}).get(unit.get("path"))
        or contract.get("source_flags") != unit.get("flags")
    ):
        raise ValueError("numerical source contract disagrees with captured unit")
    headers = facts.get("external_headers", {})
    for interface in contract.get("source_math_interfaces", ()):
        evidence = interface.get("evidence", {})
        header = headers.get(evidence.get("file"))
        if (
            not header
            or evidence.get("sha256") != header.get("sha256")
            or type(evidence.get("line")) is not int
            or evidence["line"] < 1
        ):
            raise ValueError("mathematical interface lacks its captured header")
    for function in payload["functions"]:
        signature = function["signature"]
        validate_numeric_contract(signature)
        captured = signature.get("source_numeric_contract")
        if captured is not None and captured != contract:
            raise ValueError("function numerical evidence disagrees with its unit")
        if function.get("source_defined") and captured is None:
            raise ValueError("source function lost its captured numerical contract")


def _activation_edges(graph):
    """Preserve call-edge provenance independently of source name resolution."""
    owner = graph["signature"]["name"]
    if graph["signature"].get("calling_convention") != "python":
        return [
            dict(owner=owner, target=target, site=None, role="summary")
            for target in graph["calls"]
        ]
    edges, attributed = [], set()
    for port, node in enumerate(graph["nodes"]):
        targets = []
        hooks = list(node.get("missing_attribute_hooks", ()))
        if hook := node.get("missing_attribute_hook"):
            hooks.append(hook)
        for hook in hooks:
            when = hook.get("when", {})
            if (
                not isinstance(hook.get("symbol"), str)
                or when.get("kind") != "package_attribute_missing"
                or not isinstance(when.get("module"), str)
                or not isinstance(when.get("name"), str)
            ):
                raise ValueError("invalid module attribute activation candidate")
            attributed.add(hook["symbol"])
            edges.append(
                dict(
                    owner=owner,
                    target=hook["symbol"],
                    site=f"{graph['id']}/{port}",
                    role="conditional_attribute",
                    condition=when,
                )
            )
        if node["op"] == "call" and isinstance(node.get("callee"), str):
            targets.append((node["callee"], "call"))
        if node["op"] == "python_invoke":
            targets.extend(
                (v["symbol"], "candidate") for v in node.get("candidate_callees", ())
            )
        if node["op"] == "python_import":
            targets.extend((v, "import") for v in node.get("initializers", ()))
            for conditional in node.get("conditional_initializers", ()):
                attributed.add(conditional["initializer"])
                edges.append(
                    dict(
                        owner=owner,
                        target=conditional["initializer"],
                        site=f"{graph['id']}/{port}",
                        role="conditional_import",
                        condition=conditional["when"],
                    )
                )
        if node["op"] == "python_define_class":
            targets.append((node["body_symbol"], "class_creation"))
        for target, role in targets:
            attributed.add(target)
            edges.append(
                dict(
                    owner=owner, target=target, site=f"{graph['id']}/{port}", role=role
                )
            )
        if node["op"] == "function_ref":
            edges.append(
                dict(
                    owner=owner, target=node["name"], site=None, role="escaped_callable"
                )
            )
    edges.extend(
        dict(owner=owner, target=target, site=None, role="summary")
        for target in graph["calls"]
        if target not in attributed
    )
    return edges


def _global_definition(previous, current):
    """Resolve declarations and common storage before checking global ODR."""
    if previous.get("external") or current.get("external"):
        if semantics(previous["type"]) != semantics(current["type"]):
            raise ValueError("global declaration and definition have different types")
        return current if previous.get("external") else previous
    if previous.get("linkage") == current.get("linkage") == "common":
        a, b = previous["type"], current["type"]
        if (
            not isinstance(a, dict)
            or not isinstance(b, dict)
            or a.get("kind") != b.get("kind")
            or a.get("kind") != "array"
            or a.get("of") != b.get("of")
            or a.get("of") != "u8"
            or len(a.get("shape", ())) != 1
            or len(b.get("shape", ())) != 1
        ):
            raise ValueError(
                "common storage requires a declared one-dimensional byte extent"
            )
        x, y = a["shape"][0], b["shape"][0]
        if type(x) is not int or type(y) is not int or min(x, y) < 0:
            raise ValueError("common storage extent must be a nonnegative byte count")
        exact_a = previous.get("common_extent") == "exact"
        exact_b = current.get("common_extent") == "exact"
        if exact_a and x < y or exact_b and y < x:
            raise ValueError("common storage declarations have incompatible extents")
        if (
            "init" in previous
            and "init" in current
            and semantics(previous["init"]) != semantics(current["init"])
        ):
            raise ValueError("common storage has conflicting initializers")
        result = {
            **previous,
            "type": {**a, "shape": [max(x, y)]},
            "align_bytes": max(
                previous.get("align_bytes", 1), current.get("align_bytes", 1)
            ),
            "common_extent": "exact" if exact_a or exact_b else "minimum",
        }
        if "init" in current:
            result["init"] = current["init"]
        return result
    if semantics(previous) != semantics(current):
        raise ValueError("global definitions disagree")
    return previous


def _parameter_identity(spec):
    while isinstance(spec, dict) and spec.get("kind") in {"const", "volatile"}:
        spec = spec["of"]
    return spec


def _record_definition(previous, current):
    if previous.get("opaque"):
        return current
    if current.get("opaque"):
        return previous
    a = {k: v for k, v in previous.items() if k != "methods"}
    b = {k: v for k, v in current.items() if k != "methods"}
    if semantics(a) != semantics(b):
        raise ValueError("record layouts disagree across compilation units")
    methods = {m["symbol"]: m for m in previous.get("methods", ())}
    for method in current.get("methods", ()):
        name = method["symbol"]
        if name in methods:

            def signature(m):
                return {
                    **{k: v for k, v in m.items() if k not in {"params", "order"}},
                    "params": [_parameter_identity(p["type"]) for p in m["params"]],
                }

            if signature(methods[name]) != signature(method):
                raise ValueError(f"method contracts disagree for {name}")
        else:
            methods[name] = method
    return {
        **previous,
        "methods": sorted(
            methods.values(), key=lambda m: (m.get("order", 0), m["symbol"])
        ),
    }


def _function_pointer(spec):
    while isinstance(spec, dict) and spec["kind"] == "const":
        spec = spec["of"]
    if isinstance(spec, dict) and spec["kind"] == "ptr":
        pointee = spec["to"]
        if isinstance(pointee, dict) and pointee["kind"] == "function":
            return pointee
    return None


def _close_callbacks(functions, externs, globals_, entry):
    """Prove closed, ABI-compatible target sets for calls through incoming values."""
    nodes = {name: list(_operations(fn["body"])) for name, fn in functions.items()}
    nodes[None] = list(_operations(list(globals_.values())))
    sites = [
        (caller, n)
        for caller, rows in nodes.items()
        for n in rows
        if n["op"] == "indirect_call"
    ]
    if not sites:
        return []
    incoming = defaultdict(list)
    addressed = set()
    for caller, rows in nodes.items():
        for node in rows:
            if node["op"] == "call":
                incoming[node["callee"]].append((caller, node["args"]))
            elif node["op"] == "function_ref":
                addressed.add(node["name"])
    declarations = {**externs, **functions}

    def compatible(actual, expected):
        # Top-level parameter qualifiers do not affect the passed value;
        # qualifiers inside pointers and the distinction between ref/ptr do.
        def by_value(t):
            while isinstance(t, dict) and t["kind"] in {"const", "volatile"}:
                t = t["of"]
            return t

        return (
            actual["result"] == expected["result"]
            and list(map(by_value, actual["params"]))
            == list(map(by_value, expected["params"]))
            and bool(actual.get("variadic")) == bool(expected.get("variadic"))
        )

    stable = {}

    def stable_parameter(owner, index):
        key = (owner, index)
        if key not in stable:
            param = functions[owner]["params"][index]["name"]
            var = {"op": "var", "name": param}
            count, permitted = 0, 0
            for n in nodes[owner]:
                op = n["op"]
                count += op == "var" and n["name"] == param
                if (op == "decl" or op == "each") and n["name"] == param:
                    stable[key] = False
                    return False
                if op == "indirect_call":
                    permitted += n["target"] == var
                elif op == "call" and n["callee"] in functions:
                    for arg, formal in zip(n["args"], functions[n["callee"]]["params"]):
                        permitted += (
                            arg == var and _function_pointer(formal["type"]) is not None
                        )
                elif op == "unary" and n["operator"] == "not":
                    permitted += n["value"] == var
                elif op == "binary" and n["operator"] in {"ne", "eq"}:
                    permitted += (n["left"] == var) + (n["right"] == var)
                elif op in {"if", "while", "for", "select"}:
                    permitted += n["condition"] == var
                elif op == "cast" and n["type"] == "bool":
                    permitted += n["value"] == var
            stable[key] = count == permitted
        return stable[key]

    issues = []
    for caller, site in sites:
        signature = site.get("signature")
        pending = [(caller, site["target"])]
        targets, visited = set(), set()
        try:
            if not isinstance(signature, dict) or signature.get("kind") != "function":
                raise ValueError("missing callback signature")
            while pending:
                owner, value = pending.pop()
                if (
                    value["op"] == "unary"
                    and value["operator"] == "addr"
                    and value["value"]["op"] == "function_ref"
                ):
                    value = value["value"]
                if (
                    value["op"] == "cast"
                    and value.get("conversion") == "function_decay"
                ):
                    value = value["value"]
                if value["op"] == "select":
                    pending.extend((owner, value[k]) for k in ("then", "else"))
                elif value["op"] == "function_ref":
                    name = value["name"]
                    fn = declarations.get(name)
                    if fn is None or not compatible(
                        {**fn, "params": [p["type"] for p in fn["params"]]}, signature
                    ):
                        raise ValueError(
                            f"incompatible or unresolved callback target {name}"
                        )
                    targets.add(name)
                elif value["op"] == "var" and owner in functions:
                    if owner == entry or owner in addressed:
                        raise ValueError(
                            f"{owner} can receive callbacks from an open caller"
                        )
                    params = functions[owner]["params"]
                    index = next(
                        (i for i, p in enumerate(params) if p["name"] == value["name"]),
                        None,
                    )
                    if index is None or not stable_parameter(owner, index):
                        raise ValueError(
                            f"{owner}.{value['name']} has no immutable incoming binding"
                        )
                    pointee = _function_pointer(params[index]["type"])
                    if pointee is None or not compatible(pointee, signature):
                        raise ValueError("forwarded callback signature mismatch")
                    if (owner, index) in visited:
                        continue
                    visited.add((owner, index))
                    if not incoming[owner]:
                        raise ValueError(f"{owner} has no closed incoming calls")
                    for origin, args in incoming[owner]:
                        if index >= len(args):
                            raise ValueError(f"missing callback argument for {owner}")
                        pending.append((origin, args[index]))
                else:
                    raise ValueError(f"callback target depends on {value['op']}")
            if not targets:
                raise ValueError("no concrete callback targets were found")
            site["targets"] = sorted(targets)
        except ValueError as error:
            issues.append(
                dict(
                    code="open_indirect_call",
                    symbol=caller,
                    detail=str(error),
                    **site.get("location", {}),
                )
            )
    return issues


class _PythonClasses:
    """Link class namespace definitions without inventing a method call."""

    def __init__(self):
        self.functions = set()
        self.creation = defaultdict(dict)
        self.declarations = defaultdict(dict)
        self.methods = set()

    def collect(self, graph):
        signature = graph["signature"]
        scope = signature.get("python_scope")
        if not scope or not graph.get("source_defined"):
            return
        name, owner = signature["name"], signature.get("owner_class")
        self.functions.add(name)
        if scope["kind"] == "function" and owner:
            self.methods.add((owner, name))
        for node in graph["nodes"]:
            if node["op"] == "python_define_class":
                self.creation[name][(node["symbol"], node["body_symbol"])] = None
            elif scope["kind"] == "class" and owner and node["op"] == "python_define":
                self.declarations[owner, name][node["symbol"]] = None

    def candidates(self, reachable, definitions):
        classes, optional = {}, set()
        pending = sorted(reachable.intersection(self.creation))
        while pending:
            name = pending.pop()
            targets = set()
            for owner, body in self.creation.get(name, ()):
                methods = [
                    m
                    for m in self.declarations.get((owner, body), ())
                    if (owner, m) in self.methods
                ]
                classes[owner] = dict(body_symbol=body, methods=methods)
                targets.update((body, *methods))
            if name in optional:
                for _, function in definitions[name]:
                    targets.update(function["calls"])
            for target in sorted((targets & self.functions) - reachable - optional):
                optional.add(target)
                pending.append(target)
        return classes, optional


def _build_bindings(graph, build, captured):
    """Check source-build provenance without authorizing removal of the recipe."""
    images = {image["id"]: image for image in graph.get("images", [])}
    artifacts = {
        node["id"]: node
        for node in captured["nodes"]
        if node["kind"] == "BUILD_ARTIFACT"
    }
    bindings, seen = build.get("bindings", []), set()
    if not isinstance(bindings, list):
        raise ValueError("build image bindings must be a list")
    for binding in bindings:
        image = images.get(binding["image_id"])
        artifact = artifacts.get(binding["target_artifact"])
        if (
            image is None
            or artifact is None
            or image["id"] in seen
            or binding["target_artifact"] != image.get("build_artifact")
            or binding["source_path"] != artifact["path"]
            or binding["unit_indices"] != image["units"]
            or binding["trace_sha256"] != captured["trace_sha256"]
            or binding["trace_sha256"] != build["trace_sha256"]
            or binding["evidence"] != "captured_compile_and_link"
            or binding["obligations"]
            != dict(
                historical_input_versions="unproven",
                recipe_effects="unproven",
                diagnostic_stream="unobserved",
            )
        ):
            raise ValueError("build image binding disagrees with the sealed capture")
        seen.add(image["id"])
    return bindings


def bind(graph, folder):
    from scar.contract import validate_artifact_flow

    image_context = {}
    if "images" in graph:
        # Unit indices belong to the full graph, already checked before projection.
        images = graph["images"]
        validate_images(images, graph["root_image"])
        current = graph.get("current_image")
        if not any(i["id"] == current and i["entry"] == graph["entry"] for i in images):
            raise ValueError("image binding requires the selected image and its entry")
        image_context = dict(
            current_image=current,
            execution_images=[
                {key: image[key] for key in ("id", "entry", "output")}
                for image in images
            ],
        )
    definitions = defaultdict(list)
    roots = {graph["entry"]}
    all_issues, records = [], {}
    build_bindings = []
    artifact_flow = None
    if "artifact_flow" in graph:
        artifact_flow = validate_artifact_flow(
            read_member(folder, graph["artifact_flow"])
        )
    if build := graph.get("build"):
        build_graph = read_member(folder, build["graph"])
        if build_graph["diagnostics"] != build["diagnostics"]:
            raise ValueError("build diagnostics disagree with the sealed build graph")
        build_bindings = _build_bindings(graph, build, build_graph)
        all_issues.extend(build_graph["diagnostics"])
    layout_issues = []
    active_objects = set()
    field_calls = defaultdict(set)
    constructor_initializers = {}
    text_profiles = {}
    software_headers = {}
    python_classes = _PythonClasses()
    activation_edges = []
    for index, unit in enumerate(graph["units"]):
        for source, receipt in (
            unit.get("source_facts", {}).get("external_headers", {}).items()
        ):
            previous = software_headers.setdefault(source, dict(receipt))
            if previous.get("sha256") != receipt.get("sha256") or (
                previous.get("size") is not None
                and receipt.get("size") is not None
                and previous["size"] != receipt["size"]
            ):
                all_issues.append(
                    dict(
                        code="source_header_revision",
                        detail="translation units used different external header contents",
                        file=source,
                    )
                )
            else:
                if previous.get("size") is None and receipt.get("size") is not None:
                    previous["size"] = receipt["size"]
                modes = set(
                    previous.get(
                        "evidence_sources",
                        [previous.get("evidence", "parser_owned_buffer")],
                    )
                )
                modes.add(receipt.get("evidence", "parser_owned_buffer"))
                if len(modes) > 1:
                    previous["evidence_sources"] = sorted(modes)
        for identity, profile in validate_text_profiles(
            unit.get("source_facts", {}).get("text_profiles", {})
        ).items():
            if identity in text_profiles and text_profiles[identity] != profile:
                raise ValueError("conflicting source Unicode profiles")
            text_profiles[identity] = profile
        roots.update(unit["initializer_calls"])
        active_objects.update(unit.get("initializer_objects", ()))
        for owner, calls in unit.get("record_initializers", {}).items():
            field_calls[owner].update(calls)
        for fn in unit["functions"]:
            definitions[fn["name"]].append((index, fn))
        all_issues += unit["diagnostics"]
        payload = read_member(folder, unit["graph"])
        _validate_numeric_evidence(unit, payload)
        roots.update(
            n["name"]
            for n in payload["global_graph"]["nodes"]
            if n["op"] == "function_ref"
        )
        for fn in payload["functions"]:
            python_classes.collect(fn)
            activation_edges.extend(_activation_edges(fn))
            signature = fn["signature"]
            if signature.get("method", {}).get("kind") == "constructor":
                constructor_initializers[signature["name"]] = signature.get(
                    "initializers", []
                )
        for record in payload["records"]:
            name = record["name"]
            if "initializer_graph" in record:
                initializers = {
                    n["name"]: n["init"]
                    for n in _definition(record.pop("initializer_graph"))["body"]
                }
                for field in record.get("fields", ()):
                    if field["name"] in initializers:
                        field["init"] = initializers[field["name"]]
            previous = records.get(name)
            if previous is None:
                records[name] = record
            else:
                try:
                    records[name] = _record_definition(previous, record)
                except ValueError as error:
                    layout_issues.append(
                        dict(
                            code="conflicting_record_layout",
                            symbol=name,
                            detail=str(error),
                        )
                    )
    reachable = set()
    pending = list(roots)
    constructor_owners = {
        m["symbol"]: name
        for name, record in records.items()
        for m in record.get("methods", ())
        if m["kind"] == "constructor"
    }
    default_objects = set()
    while True:
        previous = len(reachable), len(active_objects), len(default_objects)
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            reachable.add(name)
            if name in constructor_owners:
                active_objects.add(constructor_owners[name])
            for _, fn in definitions.get(name, ()):
                pending.extend(fn["calls"])
                active_objects.update(fn.get("objects", ()))
        for owner in sorted(active_objects):
            record = records.get(owner, {})
            active_objects.update(b["name"] for b in record.get("bases", ()))
            # Constructing a polymorphic object instantiates its concrete
            # vtable and cleanup protocol, including inherited implementations.
            for method in record.get("methods", ()):
                if (method.get("virtual") or method["kind"] == "destructor") and not (
                    method.get("pure")
                    or method.get("deleted")
                    or method.get("defaulted")
                ):
                    if method["symbol"] not in reachable:
                        pending.append(method["symbol"])
            constructors = [
                m for m in record.get("methods", ()) if m["kind"] == "constructor"
            ]
            if owner in default_objects:
                for method in constructors:
                    if (
                        not method["params"]
                        and not method.get("deleted")
                        and method["symbol"] not in reachable
                    ):
                        pending.append(method["symbol"])
            contexts = [
                constructor_initializers.get(m["symbol"], [])
                for m in constructors
                if m["symbol"] in reachable
                or owner in default_objects
                and not m["params"]
            ]
            if owner in default_objects and not constructors:
                contexts.append([])
            contexts = [
                ctx for ctx in contexts if not any(i["kind"] == "delegate" for i in ctx)
            ]
            for base in record.get("bases", ()):
                if any(
                    not any(
                        i["kind"] == "base" and i["name"] == base["name"] for i in ctx
                    )
                    for ctx in contexts
                ):
                    default_objects.add(base["name"])
            for field in record.get("fields", ()):
                implicit = any(
                    not any(
                        i["kind"] == "member" and i["name"] == field["name"]
                        for i in ctx
                    )
                    for ctx in contexts
                )
                if implicit and "init" in field:
                    pending.extend(
                        c
                        for c in field.get("calls", field_calls[owner])
                        if c not in reachable
                    )
                spec = field["type"]
                while isinstance(spec, dict) and spec.get("kind") in {
                    "array",
                    "const",
                    "volatile",
                }:
                    spec = spec["of"]
                if not isinstance(spec, dict) or spec.get("kind") != "record":
                    continue
                active_objects.add(spec["name"])
                if implicit and "init" not in field:
                    default_objects.add(spec["name"])
        if not pending and previous == (
            len(reachable),
            len(active_objects),
            len(default_objects),
        ):
            break
    class_candidates, candidate_only = python_classes.candidates(reachable, definitions)
    covered = reachable | candidate_only
    issues = [d for d in all_issues if not d.get("symbol") or d["symbol"] in covered]

    def issue(code, symbol, detail):
        issues.append(dict(code=code, symbol=symbol, detail=detail))

    signatures = defaultdict(list)
    functions, globals_, externs, state_evidence = {}, {}, {}, {}
    for unit in graph["units"]:
        payload = read_member(folder, unit["graph"])
        for decl in payload["externs"]:
            name = decl["name"]
            if name in reachable:
                signatures[name].append(decl)
                if decl.get("provider"):
                    externs[name] = decl
        for decl in _definition(payload["global_graph"])["body"]:
            name = decl["name"]
            try:
                globals_[name] = (
                    _global_definition(globals_[name], decl)
                    if name in globals_
                    else decl
                )
            except ValueError as error:
                issue("conflicting_global", name, str(error))
        for fn in payload["functions"]:
            name = fn["signature"]["name"]
            if name not in covered:
                continue
            if "result" in fn["signature"] and "params" in fn["signature"]:
                signatures[name].append(fn["signature"])
            protocol = (
                not fn["defined"]
                and fn.get("source_defined")
                and fn["signature"].get("calling_convention") == "python"
            )
            if fn["defined"] or protocol:
                body = _definition(fn)
                if protocol:
                    body["execution_kind"] = "python_protocol"
                    scope = body.get("python_scope", {})
                    if scope.get("kind") == "module" and (
                        providers := unit.get("source_facts", {}).get("providers")
                    ):
                        scope["provider_sources"] = _provider_sources(providers)
                if name in functions and semantics(functions[name]) != semantics(body):
                    issue(
                        "conflicting_definition",
                        name,
                        "multiple non-equivalent definitions",
                    )
                else:
                    functions[name] = body
                    state_evidence[name] = dict(
                        graph_id=fn["id"],
                        states=fn.get("states", {}),
                        accesses=fn.get("accesses", []),
                        state_flow=fn.get("state_flow"),
                    )
    referenced_globals = {
        n["name"]
        for n in _operations(
            [
                list(functions.values()),
                [g for g in globals_.values() if not g.get("external")],
                [records[name] for name in active_objects if name in records],
            ]
        )
        if n["op"] == "var"
    }
    globals_ = {
        name: g
        for name, g in globals_.items()
        if not g.get("external") or name in referenced_globals
    }
    provided_methods = {
        m["symbol"]
        for r in records.values()
        for m in r.get("methods", ())
        if m.get("defaulted") or m.get("implicit")
    }
    for name in sorted(reachable):
        if name in functions:
            externs.pop(name, None)
        elif (
            name not in externs
            and name not in provided_methods
            and not any(d.get("symbol") == name for d in issues)
        ):
            issue(
                "unresolved_call",
                name,
                "reachable symbol requires a definition or external ABI",
            )
    if graph["entry"] not in functions:
        issue(
            "missing_entry_body", graph["entry"], "no executable definition for entry"
        )
    used_records = set(active_objects)

    def record_references(value):
        if isinstance(value, list):
            for item in value:
                record_references(item)
        elif isinstance(value, dict):
            if value.get("kind") == "record":
                used_records.add(value["name"])
            for item in value.values():
                record_references(item)

    record_references(
        [list(functions.values()), list(globals_.values()), list(externs.values())]
    )
    expanded = set()
    while used_records - expanded:
        for name in sorted(used_records - expanded):
            expanded.add(name)
            record = records.get(name)
            if record is None:
                issue(
                    "missing_record", name, "referenced record needs a type declaration"
                )
                continue
            record_references(record)
            used_records.update(b["name"] for b in record.get("bases", ()))
    issues.extend(d for d in layout_issues if d["symbol"] in used_records)
    issues.extend(_close_callbacks(functions, externs, globals_, graph["entry"]))
    issues = list({json_bytes(d): d for d in issues}.values())
    protocol_names = {
        name
        for name, function in functions.items()
        if function.get("execution_kind") == "python_protocol"
    }
    obligations, binding_issues = [], []
    for problem in issues:
        deferred = problem.get("symbol") in protocol_names and problem[
            "code"
        ].startswith("python.")
        (obligations if deferred else binding_issues).append(problem)
    issues = binding_issues
    return (
        publish_program(
            dict(
                entry=graph["entry"],
                **({"entry_abi": graph["entry_abi"]} if graph.get("entry_abi") else {}),
                **image_context,
                activation_graph=dict(
                    roots=sorted(roots),
                    edges=[e for e in activation_edges if e["owner"] in covered],
                ),
                functions=list(functions.values()),
                globals=list(globals_.values()),
                externs=list(externs.values()),
                records=[r for name, r in records.items() if name in used_records],
                state_evidence=state_evidence,
                text_profiles=text_profiles,
                **(
                    dict(
                        python_class_candidates=class_candidates,
                        python_candidate_only=sorted(candidate_only),
                    )
                    if class_candidates
                    else {}
                ),
                **({"protocol_obligations": obligations} if protocol_names else {}),
                declarations={
                    name: [
                        dict(
                            name=name,
                            result=s["result"],
                            params=s["params"],
                            variadic=bool(s.get("variadic")),
                        )
                        for s in rows
                    ]
                    for name, rows in signatures.items()
                },
                abi_environment=dict(
                    architecture=graph.get("platform", {}).get("architecture")
                ),
                resource_model=dict(
                    resources=graph.get("resources", []),
                    topology=graph.get("topology", []),
                    platform=graph.get("platform", {}),
                    software_headers=software_headers,
                    **(
                        {"artifact_flow": artifact_flow}
                        if artifact_flow is not None
                        else {}
                    ),
                    **({"build_bindings": build_bindings} if build_bindings else {}),
                ),
            ),
            "M1",
            complete=not issues,
        ),
        issues,
        len(reachable),
    )
