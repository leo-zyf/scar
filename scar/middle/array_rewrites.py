"""Exact local rewrites and array lifetime proof for the closed elementwise IR."""

from collections import Counter
import math

_BINARY = {"add", "sub", "mul", "div"}


def canonical_value_aliases(graph):
    representatives, rewrites = {}, []
    for operation in graph.operations:
        if operation.kind != "CONSTANT" or operation.effects != ("PURE",):
            continue
        dtype = operation.attributes.get("dtype")
        if type(dtype) is not str or "value" not in operation.attributes:
            continue
        value = operation.attributes["value"]
        if type(value) is float and not math.isfinite(value):
            continue
        if type(value) not in (bool, int, float, str, type(None)):
            continue
        # Exact payload equality leaves object identity and operation execution intact.
        payload = value.hex() if type(value) is float else value
        key = operation.region, dtype, type(value), payload
        canonical = representatives.setdefault(key, operation.id)
        if canonical != operation.id:
            rewrites.append(
                dict(
                    rule="CANONICAL_LITERAL_VALUE",
                    operation=operation.id,
                    canonical=canonical,
                    region=operation.region,
                    dtype=dtype,
                    scope="TYPED_PAYLOAD",
                    source_edit=False,
                )
            )
    return tuple(rewrites)


def _reads(expression):
    kind, *arguments = expression
    if kind == "read":
        return {arguments[0]}
    children = arguments[1:] if kind == "call" else arguments
    if kind in _BINARY or kind in ("neg", "call"):
        return set().union(*map(_reads, children))
    return set()


def _key(expression, versions):
    kind, *arguments = expression
    if kind == "read":
        return kind, arguments[0], versions[arguments[0]]
    if kind == "constant":
        (value,) = arguments
        return kind, type(value), value.hex() if type(value) is float else value
    if kind in _BINARY or kind == "neg":
        keys = tuple(_key(argument, versions) for argument in arguments)
        if all(key is not None for key in keys):
            return kind, keys
    return None


def _writes(rows):
    names = Counter()
    for row in rows:
        if row[0] == "store":
            names[row[1]] += 1
        elif row[0] == "repeat":
            names.update(_writes(row[1]))
    return names


def _prefix_invariants(rows, available):
    writes = _writes(rows)
    stable = available - writes.keys()
    prefix = []
    for row in rows:
        if row[0] != "store":
            break
        _, name, value = row
        if (
            name in available
            or writes[name] != 1
            or not _reads(value) <= stable
            or _key(value, dict.fromkeys(stable, 0)) is None
        ):
            break
        prefix.append(row)
        stable.add(name)
    return prefix


def _forward(rows, available, ledger, sites):
    versions = dict.fromkeys(available, 0)
    by_value, result = {}, []

    def store(row):
        _, name, value = row
        site = sites[id(row)]
        key = _key(value, versions)
        canonical = by_value.get(key)
        if canonical is not None and versions[canonical[0]] == canonical[1]:
            ledger.append(("EXACT_EXPRESSION_REUSE", name, site, canonical[2]))
            value = ["read", canonical[0]]
        versions[name] = versions.get(name, -1) + 1
        if key is not None and value[0] in (_BINARY | {"neg"}):
            by_value[key] = (name, versions[name], site)
        updated = ["store", name, value]
        sites[id(updated)] = site
        result.append(updated)

    for row in rows:
        if row[0] == "store":
            store(row)
        elif row[0] == "repeat":
            body = row[1]
            prefix = _prefix_invariants(body, set(versions))
            for invariant in prefix:
                ledger.append(
                    (
                        "HOIST_STABLE_LOOP_PREFIX",
                        invariant[1],
                        sites[id(invariant)],
                        None,
                    )
                )
                store(invariant)
            remaining = body[len(prefix) :]
            result.append(["repeat", _forward(remaining, set(versions), ledger, sites)])
            # Loop writes have an unknown number of intervening value versions.
            for name in _writes(remaining).keys() & versions.keys():
                versions[name] += 1
            by_value.clear()
        else:
            result.append(row)
    return result


def _body_reads(rows):
    reads = set()
    for row in rows:
        if row[0] == "repeat":
            reads.update(_body_reads(row[1]))
        else:
            reads.update(_reads(row[-1]))
    return reads


def _dead_aliases(rows, live, ledger, sites):
    result = []
    for row in reversed(rows):
        kind = row[0]
        if kind == "repeat":
            body = row[1]
            keep = live | _body_reads(body)
            result.append(["repeat", _dead_aliases(body, keep, ledger, sites)])
            live |= keep
            continue
        value = row[-1]
        if kind == "store":
            name = row[1]
            if name not in live and value[0] in ("read", "constant"):
                ledger.append(("REMOVE_DEAD_LOCAL_BINDING", name, sites[id(row)], None))
                continue
            live.discard(name)
        live.update(_reads(value))
        result.append(row)
    return result[::-1]


def _subtree(region, regions):
    pending, found = [region], []
    while pending:
        item = pending.pop()
        found.append(item)
        pending.extend(regions[child] for child in reversed(item.children))
    return found


def _store_rows(rows):
    for row in rows:
        if row[0] == "store":
            yield row
        elif row[0] == "repeat":
            yield from _store_rows(row[1])


def _bind_stores(rows, regions, operations):
    body = [operations[key] for region in regions for key in region.operations]
    induction = {
        value
        for operation in body
        if operation.kind == "ITERATION"
        for value in operation.outputs
    }
    stores = [
        operation
        for operation in body
        if operation.kind == "ASSIGN" and not induction.intersection(operation.inputs)
    ]
    if any(
        operation.effects != ("STATE_WRITE",)
        or operation.attributes.get("source_span") is None
        for operation in stores
    ):
        return None
    stores.sort(key=lambda operation: operation.attributes["source_span"])
    statements = list(_store_rows(rows))
    if [operation.name for operation in stores] != [
        statement[1] for statement in statements
    ]:
        return None
    return {
        id(statement): operation.id
        for statement, operation in zip(statements, stores, strict=True)
    }


def _work(rows, count, callees, calls):
    def expression(row):
        kind, *args = row
        if kind == "call":
            name, *parameters = args
            calls[name] += 1
            return callees[name] + sum(expression(item)[0] for item in parameters), True
        if kind in _BINARY:
            return 1 + sum(expression(item)[0] for item in args), True
        if kind == "neg":
            inner, array = expression(args[0])
            return inner + int(array), array
        return 0, kind == "read"

    work = 0
    for row in rows:
        if row[0] == "repeat":
            work += count * _work(row[1], count, callees, calls)
        else:
            work += expression(row[-1])[0]
    return work


def _inline_single_use(functions, entry, counts):
    definitions = {row["name"]: row for row in functions}
    inline = {name for name, uses in counts.items() if uses == 1 and name != entry}
    if not inline:
        return functions
    occupied = {
        name
        for row in functions
        for name in (*row["parameters"], *_writes(row["body"]))
    }
    ordinal = 0

    def fresh():
        nonlocal ordinal
        while True:
            name = f"scar_inline_value_{ordinal}"
            ordinal += 1
            if name not in occupied:
                return name

    def store(value, output):
        name = fresh()
        output.append(["store", name, value])
        return ["read", name]

    def expression(row, frame, output):
        kind, *args = row
        if kind == "read":
            return ["read", frame.get(args[0], args[0])]
        if kind == "constant":
            return row
        if kind == "call":
            name, *arguments = args
            if name not in inline:
                return [
                    kind,
                    name,
                    *(expression(arg, frame, output) for arg in arguments),
                ]
            callee = definitions[name]
            local = {}
            for parameter, arg in zip(callee["parameters"], arguments, strict=True):
                local[parameter] = store(expression(arg, frame, output), output)[1]
            for statement in callee["body"]:
                value = expression(statement[-1], local, output)
                if statement[0] == "return":
                    return store(value, output)
                local[statement[1]] = store(value, output)[1]
            raise ValueError("inline array function has no result")
        return [kind, *(expression(arg, frame, output) for arg in args)]

    def statements(rows):
        output = []
        for row in rows:
            if row[0] == "repeat":
                output.append(["repeat", statements(row[1])])
            else:
                value = expression(row[-1], {}, output)
                output.append([*row[:-1], value])
        return output

    return [
        dict(row, body=statements(row["body"]))
        for row in functions
        if row["name"] not in inline
    ]


def rewrite_elementwise_ir(graph, program, count):
    """Bind the array program to the source graph before lowering its local values."""
    regions = {region.id: region for region in graph.regions}
    operations = {operation.id: operation for operation in graph.operations}
    functions = {}
    for region in graph.regions:
        if region.kind == "FUNCTION":
            functions.setdefault(region.name, []).append(region)
    names = tuple(row["name"] for row in program["functions"])
    if (
        graph.target.get("entry") != program["entry"]
        or not names
        or len(set(names)) != len(names)
        or any(len(functions.get(name, ())) != 1 for name in names)
    ):
        return None, ()
    bindings = {functions[name][0].id: name for name in names}
    owners, contents = {}, {}
    for region, name in bindings.items():
        contents[name] = _subtree(regions[region], regions)
        for part in contents[name]:
            owners.update(dict.fromkeys(part.operations, name))
    calls = {name: Counter() for name in names}
    call_sites = []
    for edge in graph.edges:
        if (
            edge.kind == "CALL_CANDIDATE"
            and edge.source in owners
            and edge.target in bindings
        ):
            calls[owners[edge.source]][bindings[edge.target]] += 1
            call_sites.append(edge)

    rewritten, proofs, work, transformed_work = [], [], {}, {}
    for function in program["functions"]:
        name = function["name"]
        observed = Counter()
        work[name] = _work(function["body"], count, work, observed)
        if observed != calls[name]:
            return None, ()
        sites = _bind_stores(function["body"], contents[name], operations)
        if sites is None:
            return None, ()
        local = []
        body = _forward(function["body"], set(function["parameters"]), local, sites)
        body = _dead_aliases(body, set(), local, sites)
        transformed_work[name] = _work(body, count, transformed_work, Counter())
        rewritten.append(dict(function, body=body))
        for rule, target, original, producer in local:
            identities = [original] if producer is None else [original, producer]
            proofs.append(
                dict(
                    rule=rule,
                    region=operations[original].region,
                    operations=identities,
                    binding=target,
                    scope="CLOSED_FP64_ELEMENTWISE_IR",
                )
            )

    entry = program["entry"]
    counts = Counter()
    for targets in calls.values():
        counts.update(targets)
    rewritten = _inline_single_use(rewritten, entry, counts)
    inlined = [name for name in names if name != entry and counts[name] == 1]
    if inlined:
        proofs.append(
            dict(
                rule="INLINE_SINGLE_CONSUMER_ARRAY_REGIONS",
                region=functions[entry][0].id,
                regions=[functions[name][0].id for name in inlined],
                operations=[
                    edge.source
                    for edge in call_sites
                    if bindings[edge.target] in inlined
                ],
                original_function_count=len(names),
                transformed_function_count=len(rewritten),
                scope="CLOSED_FP64_ELEMENTWISE_IR",
            )
        )
    proofs.append(
        dict(
            rule="FUSE_ELEMENTWISE_ARRAY_LIFETIMES",
            region=functions[entry][0].id,
            regions=[part.id for rows in contents.values() for part in rows],
            operations=[
                key for key in owners if operations[key].kind in ("ARITHMETIC", "CALL")
            ],
            original_array_expression_results=work[entry],
            transformed_lane_arithmetic=transformed_work[entry],
            intermediate_array_materializations_in_transformed_body=0,
            reference_scope="ORIGINAL_LOGICAL_COMPUTE_IR",
            scope="SPECIALIZED_NORMAL_COMPLETION",
            conditions=[
                "EXACT_FP64_ARRAY_VALUES",
                "POSITIVE_EXACT_TRIP_COUNT",
                "CLOSED_CALLS",
                "NO_OBSERVABLE_INTERMEDIATE_IDENTITY",
                "PRESERVED_OPERAND_ORDER",
            ],
        )
    )
    return dict(program, functions=rewritten), tuple(proofs)
