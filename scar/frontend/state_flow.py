"""Compute conservative reaching state versions for structured graph regions."""

from collections import defaultdict


def state_flow(nodes, entry, accesses):
    boundaries = []
    bound_loops = {
        edge["node"]
        for edge in accesses
        if edge.get("phase") == "iteration" and edge["mode"] in {"write", "bind"}
    }
    for port, row in enumerate(nodes):
        op = row["op"]
        if (
            op == "with_context"
            or op == "for"
            and (
                "iterator" in row
                or row.get("protocol") in {"iterator", "async_iterator"}
            )
        ):
            boundaries.append((port, "conservative_iterator_protocol"))
        elif op in {"map", "each"} and port not in bound_loops:
            # An older F3 shard can bind a same-named extent to the inner
            # variable. That lexical distinction is absent from these facts.
            boundaries.append((port, "conservative_iteration_binding"))
        elif (
            op == "decl"
            and row.get("storage") in {"static", "thread_local"}
            and "init" in row
        ):
            boundaries.append((port, "conservative_static_initialization"))
        elif op in {
            "goto",
            "switch",
            "case",
            "default",
            "unsupported",
            "try",
            "exception_handler",
            "raise",
            "throw",
            "yield",
            "await",
        }:
            boundaries.append((port, "conservative_cfg"))
    if boundaries:
        return {
            "status": boundaries[0][1],
            "boundaries": [p for p, _ in boundaries],
            "versions": [],
        }

    by_node = defaultdict(list)
    escaped = {edge["domain"] for edge in accesses if edge["mode"] == "address"}
    for effect in accesses:
        by_node[effect["node"]].append(effect)
    records, cache = [], {}
    regions = {"body", "then", "else", "init"}

    def ports(v):
        if isinstance(v, list):
            for x in v:
                yield from ports(x)
        elif isinstance(v, dict):
            if set(v) == {"port"}:
                yield v["port"]
            else:
                for x in v.values():
                    yield from ports(x)

    def direct_effects(port):
        reads, writes = set(), set()
        for effect in by_node.get(port, ()):
            domain = effect["domain"]
            if "binding" in effect:
                reads.add(effect["binding"])
            # An escaped local or global is also covered by the shared memory
            # token, so an opaque call invalidates reaching reads through it.
            domains = {domain}
            if domain in escaped or domain.startswith("global/"):
                domains.add("memory")
            mode = effect["mode"]
            if mode in {"read", "read_write", "unknown"}:
                reads.update(domains)
            if mode in {"write", "bind", "read_write", "unknown"}:
                writes.update(domains)
        return reads, writes

    def footprint(port, nested):
        key = port, nested
        if key in cache:
            return cache[key]
        reads, writes = direct_effects(port)
        for field, value in nodes[port].items():
            if field in regions and isinstance(value, list) and not nested:
                continue
            for child in ports(value):
                used, changed = footprint(child, True)
                reads.update(used)
                writes.update(changed)
        cache[key] = reads, writes
        return reads, writes

    def version(env, domain):
        return env.get(domain, "entry")

    def apply(env, port, phase, use, change):
        if not use and not change:
            return
        before = {d: version(env, d) for d in sorted(use | change)}
        token = f"{port}/{phase}"
        after = {d: token for d in sorted(change)}
        records.append(
            dict(
                kind="ACCESS",
                node=port,
                phase=phase,
                read=sorted(use),
                incoming=before,
                outgoing=after,
            )
        )
        env.update(after)

    # A compound statement summary cannot turn a conditional expression write
    # into a definite clobber. The F3 ports remain available for a finer pass.
    for port, row in enumerate(nodes):
        op = row["op"]
        if op == "binary" and row.get("operator") in {"land", "lor"}:
            conditional = [row["right"]]
        elif op == "select":
            conditional = [row["then"], row["else"]]
        elif op in {"short_circuit", "compare_chain"}:
            conditional = row["operands"][1:]
        else:
            continue
        if any(footprint(child, True)[1] for child in ports(conditional)):
            return dict(
                status="conservative_conditional_effects",
                boundaries=[port],
                versions=[],
            )

    transfers = []

    def join(port, phase, paths):
        paths = [env for env in paths if env is not None]
        if len(paths) < 2:
            return paths[0] if paths else None
        merged = {}
        for domain in sorted(set().union(*(e.keys() for e in paths))):
            inputs = list(dict.fromkeys(version(e, domain) for e in paths))
            if len(inputs) == 1:
                merged[domain] = inputs[0]
            else:
                merged[domain] = f"{port}/{phase}"
                records.append(
                    dict(
                        kind="MERGE",
                        node=port,
                        phase=phase,
                        domain=domain,
                        inputs=inputs,
                        output=merged[domain],
                    )
                )
        return merged

    def evaluate(row, fields, env):
        for field in fields:
            for child in ports(row.get(field)):
                apply(env, child, field, *footprint(child, True))

    def lexical_slots(handles):
        slots = set()
        pending = [handle["port"] for handle in handles]
        while pending:
            node = pending.pop()
            for edge in by_node.get(node, ()):
                if edge.get("lexical") or (
                    edge.get("phase") == "iteration"
                    and edge["mode"] in {"bind", "write"}
                ):
                    slots.add(edge["domain"])
            for key in regions:
                region = nodes[node].get(key)
                if isinstance(region, list):
                    pending.extend(ports(region))
        return slots

    def constant(handle):
        if isinstance(handle, dict) and "port" in handle:
            expr = nodes[handle["port"]]
            if expr["op"] == "literal" and type(expr.get("value")) in {
                int,
                float,
                bool,
            }:
                return expr["value"]
        return None

    def loop_region(port, row, env, outer):
        op = row["op"]
        initializers = row.get("init", [])
        initializer_slots = lexical_slots(initializers)
        env = sequence(initializers, env, outer)
        if env is None:
            return None
        evaluate(row, ("extent", "range"), env)
        test = constant(row.get("condition"))
        counted = op in {"map", "each"}
        extent = constant(row.get("extent")) if counted else None
        if op == "for" and row.get("condition") is None:
            test = True
        if (counted and extent is not None and extent <= 0) or (
            not counted and op != "do_while" and test is not None and not test
        ):
            evaluate(row, ("condition",), env)
            result = sequence(row.get("else", []), env, outer)
            if result is not None:
                for domain in initializer_slots:
                    result.pop(domain, None)
            return result

        # A symbolic header represents every iteration. Only reachable paths
        # feed the latch. A break goes directly to the loop's outer successor.
        changed = set()
        for field in ("condition", "body", "step"):
            for child in ports(row.get(field)):
                changed.update(footprint(child, True)[1])
        iteration_bindings = {
            access["domain"]
            for access in by_node[port]
            if access.get("phase") == "iteration"
            and access["mode"] in {"bind", "write"}
        }
        # Automatic variables declared in the body start fresh each iteration.
        # A for-initializer variable instead spans the header, body and step.
        temporary = lexical_slots(row["body"]) | iteration_bindings
        changed.difference_update(temporary)
        initial = {domain: version(env, domain) for domain in changed}
        header = env.copy()
        for domain in changed:
            header[domain] = f"{port}/header"
        if not counted and op != "do_while":
            evaluate(row, ("condition",), header)
        frame = {
            "port": port,
            "break": [],
            "continue": [],
            "latch": (
                "step"
                if op == "for"
                else "condition" if op in {"while", "do_while"} else "advance"
            ),
        }
        iteration = header.copy()
        if counted:
            apply(iteration, port, "iteration", *direct_effects(port))
        fallthrough = sequence(row["body"], iteration, frame)
        for path in [fallthrough, *frame["continue"], *frame["break"]]:
            if path is not None:
                for domain in temporary:
                    path.pop(domain, None)
        latch = join(port, "latch", [fallthrough, *frame["continue"]])
        back = latch
        normal = None
        if counted:
            # An unknown cardinality admits the empty domain; a known positive
            # one must execute the body before the range is exhausted.
            normal = join(port, "exhausted", [env if extent is None else None, latch])
        elif op == "do_while":
            if latch is not None:
                evaluate(row, ("condition",), latch)
                normal = latch if test is None or not test else None
                back = latch if test is None or test else None
        else:
            if latch is not None:
                evaluate(row, ("step",), latch)
            normal = header if test is None or not test else None

        # A loop else suite runs after condition failure/exhaustion; break bypasses it.
        if normal is not None and row.get("else"):
            normal = sequence(row["else"], normal.copy(), outer)
        finished = join(port, "exit", [normal, *frame["break"]])
        for domain in sorted(changed):
            carried = version(back, domain) if back is not None else None
            token = f"{port}/header"
            carries = carried is not None and carried != token
            records.append(
                dict(
                    kind="LOOP_CARRY" if carries else "LOOP_ENTRY",
                    node=port,
                    domain=domain,
                    initial=initial[domain],
                    next=carried if carries else None,
                    header=token,
                    exit=(
                        version(finished, domain)
                        if finished is not None and domain not in initializer_slots
                        else None
                    ),
                )
            )
        if finished is not None:
            for domain in initializer_slots:
                finished.pop(domain, None)
        return finished

    def sequence(seq, env, loop):
        for handle in seq:
            if env is None:
                break
            port = handle["port"]
            row = nodes[port]
            op = row["op"]
            if op == "block":
                env = sequence(row["body"], env, loop)
            elif op == "if":
                apply(env, port, "condition", *footprint(port, False))
                truth = constant(row.get("condition"))
                left = (
                    sequence(row["then"], env.copy(), loop)
                    if truth is None or truth
                    else None
                )
                right = (
                    sequence(row.get("else", []), env.copy(), loop)
                    if truth is None or not truth
                    else None
                )
                env = join(port, "join", [left, right])
            elif op in {"map", "each", "for", "while", "do_while"}:
                env = loop_region(port, row, env, loop)
            elif op in {"break", "continue", "return"}:
                apply(env, port, "effect", *footprint(port, True))
                if op == "return":
                    transfers.append(
                        dict(node=port, target=None, phase="function_exit")
                    )
                elif loop is not None:
                    loop[op].append(env)
                    transfers.append(
                        dict(
                            node=port,
                            target=loop["port"],
                            phase="exit" if op == "break" else loop["latch"],
                        )
                    )
                else:
                    boundaries.append((port, "conservative_cfg"))
                env = None
            else:
                apply(env, port, "effect", *footprint(port, True))
        return env

    sequence(entry, {}, None)
    if boundaries:
        return dict(
            status=boundaries[0][1], boundaries=[p for p, _ in boundaries], versions=[]
        )
    return dict(
        status="structured",
        control_scope="normal_completion",
        access_granularity="statement_may_access",
        versions=records,
        transfers=transfers,
    )
