"""Encode one definition as a hierarchical port graph, including mutable effects."""

from scar.contract import digest, json_bytes
from .state_flow import state_flow

_REGIONS = {"body", "then", "else", "init"}
_CONTROL = {
    "for",
    "each",
    "map",
    "if",
    "while",
    "do_while",
    "switch",
    "block",
    "return",
    "break",
    "continue",
}


def function_graph(function, revision):
    name = function["name"]
    identity = digest(json_bytes([revision, name, function]))[:32]
    nodes, accesses, calls, domains = [], [], set(function.get("calls", [])), {}
    objects, dispatch = set(), []

    def cell_domain(info):
        cell = info.get("cell")
        return f"activation/{cell['owner']}/{cell['name']}" if cell else None

    def declare(scope, variable, slot, info):
        domain = cell_domain(info) or f"{slot}/{variable}"
        scope[variable] = domain
        domains[domain] = dict(
            name=variable, type=info.get("type"), kind=slot.split("/")[0]
        )
        if info.get("cell"):
            domains[domain].update(kind="cell", activation=info["cell"])

    def encode(value, scope, mode="read"):
        if isinstance(value, list):
            return [encode(item, scope, mode) for item in value]
        if not isinstance(value, dict):
            return value
        if "op" not in value:
            return {k: encode(v, scope) for k, v in value.items()}
        number = len(nodes)
        op = value["op"]
        row = {"op": op}
        nodes.append(row)
        if op in _CONTROL:
            scope = scope.copy()
        if op in {"decl", "buffer", "runtime_array"}:
            declare(scope, value["name"], f"local/{number}", value)
            if "init" in value or op == "buffer":
                accesses.append(
                    dict(node=number, domain=scope[value["name"]], mode="write")
                )
        elif op in {"map", "each"}:
            variable = value["index"] if op == "map" else value["name"]
            declare(scope, variable, f"iteration/{number}", value)
        if op == "call":
            calls.add(value["callee"])
            accesses.append(dict(node=number, domain="memory", mode="unknown"))
        elif op == "function_ref":
            calls.add(value["name"])
        elif op == "indirect_call":
            calls.update(value.get("targets", ()))
            accesses.append(dict(node=number, domain="memory", mode="unknown"))
        elif op in {"object_construct", "object_new"}:
            objects.add(value["type"]["name"])
            if value.get("symbol"):
                calls.add(value["symbol"])
            accesses.append(dict(node=number, domain="memory", mode="unknown"))
        elif op == "object_dispatch":
            if value.get("qualified"):
                calls.add(value["symbol"])
            else:
                calls.discard(value["symbol"])
                dispatch.append(dict(owner=value["owner"], slot=value["slot"]))
            accesses.append(dict(node=number, domain="memory", mode="unknown"))
        elif op == "object_destroy":
            if value.get("destructor"):
                calls.add(value["destructor"])
            accesses.append(dict(node=number, domain="memory", mode="unknown"))
        elif op in {"load_bytes", "store_bytes"}:
            accesses.append(
                dict(
                    node=number,
                    domain="memory",
                    mode="read" if op == "load_bytes" else "write",
                    view=number,
                )
            )
        elif op == "var":
            variable = value["name"]
            domain = (
                cell_domain(value)
                or (scope.get(variable) if value.get("scope") != "global" else None)
                or f"global/{variable}"
            )
            domains.setdefault(domain, dict(name=variable, kind="global"))
            if value.get("cell"):
                domains[domain].update(kind="cell", activation=value["cell"])
            accesses.append(dict(node=number, domain=domain, mode=mode))
        elif op in {"index", "member"} or (
            op == "unary" and value["operator"] == "deref"
        ):
            accesses.append(dict(node=number, domain="memory", mode=mode, view=number))
        elif op in {
            "allocate",
            "release",
            "unsupported",
            "library_construct",
            "library_call",
            "variadic",
            "runtime_array",
        }:
            accesses.append(dict(node=number, domain="memory", mode="unknown"))
        if isinstance(value.get("object_lifetime"), dict):
            lifetime = value["object_lifetime"]
            objects.add(lifetime["type"])
            calls.update(
                lifetime[protocol]
                for protocol in ("constructor", "destructor")
                if lifetime.get(protocol)
            )
        if op == "python_import":
            calls.update(value.get("initializers", ()))
        if isinstance(value.get("effects"), list) and set(value["effects"]) & {
            "STATE",
            "IO",
            "OPAQUE",
            "MEM",
            "XFER",
            "ORDER",
        }:
            accesses.append(dict(node=number, domain="memory", mode="unknown"))
        for state in value.get("state_reads", ()):
            accesses.append(dict(node=number, domain=f"external/{state}", mode="read"))
        for state in value.get("state_writes", ()):
            accesses.append(dict(node=number, domain=f"external/{state}", mode="write"))
        # Declaration in a for initializer binds the condition, update and body.
        keys = (["init"] if "init" in value else []) + [
            k for k in value if k not in {"op", "init"}
        ]
        for key in keys:
            field = value[key]
            if key in _REGIONS and isinstance(field, list):
                inner = scope if key == "init" else scope.copy()
                row[key] = [encode(child, inner) for child in field]
            else:
                access = (
                    "write"
                    if op in {"assign", "for", "python_import"}
                    and key in {"target", "binding"}
                    else "read"
                )
                if op == "update" and key == "target":
                    access = "write" if value["operator"] == "set" else "read_write"
                if (
                    op == "unary"
                    and key == "value"
                    and value["operator"] in {"preinc", "postinc", "predec", "postdec"}
                ):
                    access = "read_write"
                if op == "unary" and key == "value" and value["operator"] == "addr":
                    access = "address"
                row[key] = encode(field, scope, access)
        return {"port": number}

    scope = {}
    for param in function.get("params", ()):
        declare(scope, param["name"], "formal", param)
    signature = {
        k: v
        for k, v in function.items()
        if k not in {"body", "partial_body", "calls", "initializers"}
    }
    if "initializers" in function:
        signature["initializers"] = encode(function["initializers"], scope)
    body = encode(function.get("body") or function.get("partial_body", []), scope)
    return dict(
        id=identity,
        signature=signature,
        defined=isinstance(function.get("body"), list),
        entry=body,
        nodes=nodes,
        states=domains,
        accesses=accesses,
        calls=sorted(calls),
        objects=sorted(objects),
        dispatch=dispatch,
        state_flow=state_flow(nodes, body, accesses),
    )
