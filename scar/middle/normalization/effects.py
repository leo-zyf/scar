"""M2 caller-visible storage writes derived from a closed canonical call graph."""

from scar.contract import walk_operations
from scar.middle.contracts import has_open_effect

FRESH_ARRAYS = frozenset(
    {
        "array_create",
        "array_iota",
        "array_literal",
        "array_copy",
        "array_broadcast_copy",
        "array_unary",
        "array_binary",
        "array_from_sequence",
    }
)
_MUTATIONS = {
    "array_update": ("arrays", "left"),
    "array_store": ("arrays", "base"),
    "sequence_append": ("sequences", "base"),
}


def _write_origins(expression, parameters, assignments):
    """Formal inputs that may own this storage, or None for an open origin."""
    pending = [(expression, None)]
    active, resolved, ports = set(), set(), set()
    while pending:
        value, exiting = pending.pop()
        if exiting is not None:
            active.remove(exiting)
            resolved.add(exiting)
            continue
        op = value["op"]
        if op in FRESH_ARRAYS or op in {"sequence_make", "sequence_sorted_copy"}:
            continue
        if op in {"array_view", "array_reshape"}:
            pending.append(
                (value["base"] if op == "array_view" else value["value"], None)
            )
            continue
        if op != "var" or value.get("scope") == "global":
            return None
        name = value["name"]
        if name in resolved:
            continue
        if name in active:
            return None
        if name in parameters:
            ports.add(parameters[name])
        elif name not in assignments:
            return None
        active.add(name)
        pending.append((None, name))
        pending.extend((source, None) for source in assignments.get(name, ()))
    return ports


def call_writes(function, functions):
    """For each mutable domain return written argument positions, or None if open.

    Trace views and assignments inside each function, then substitute actual
    argument origins on each call edge. Memoization ends at the current call:
    deferred source effects may populate a callee body before the next call.
    """
    cache = {}
    unknown = {"arrays": None, "sequences": None}

    def summarize(body):
        name = body["name"]
        if name in cache:
            return cache[name]
        cache[name] = unknown  # Conservative until the full body has been visited.
        if body.get("recursive_prototype") or "body" not in body:
            return unknown
        parameters = {p["name"]: i for i, p in enumerate(body["params"])}
        assignments = {}
        targets = {"arrays": [], "sequences": []}
        open_domains = set()
        for node in walk_operations(body["body"]):
            op = node["op"]
            if has_open_effect(node):
                return unknown
            if op == "decl" and "init" in node:
                assignments.setdefault(node["name"], []).append(node["init"])
            elif op == "assign":
                target = node["target"]
                if target["op"] == "var" and target.get("scope") != "global":
                    assignments.setdefault(target["name"], []).append(node["value"])
            elif op in _MUTATIONS:
                domain, key = _MUTATIONS[op]
                targets[domain].append(node[key])
            elif op == "call":
                callee = functions.get(node.get("callee"))
                if callee is None:
                    return unknown
                for domain, ports in summarize(callee).items():
                    if ports is None:
                        open_domains.add(domain)
                    else:
                        targets[domain].extend(node["args"][i] for i in ports)
            else:
                writes = node.get("writes", ())
                if "array.storage" in writes:
                    open_domains.add("arrays")
                if "sequence.state" in writes:
                    open_domains.add("sequences")
        result = {}
        for domain, expressions in targets.items():
            ports = set()
            result[domain] = None
            if domain not in open_domains:
                for expression in expressions:
                    owners = _write_origins(expression, parameters, assignments)
                    if owners is None:
                        break
                    ports.update(owners)
                else:
                    result[domain] = frozenset(ports)
        cache[name] = result
        return result

    return summarize(function)
