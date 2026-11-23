"""Compute conservative reaching state versions for structured graph regions."""

from collections import defaultdict


def state_flow(nodes, entry, accesses):
    if any(
        n["op"] == "for" and "iterator" in n or n["op"] == "with_context" for n in nodes
    ):
        return {"status": "conservative_iterator_protocol", "versions": []}
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

    def footprint(port, nested):
        key = port, nested
        if key in cache:
            return cache[key]
        reads, writes = set(), set()
        for effect in by_node[port]:
            domain = effect["domain"]
            # An escaped local or global is also covered by the shared memory
            # token, so an opaque call invalidates reaching reads through it.
            domains = {domain}
            if domain in escaped or domain.startswith("global/"):
                domains.add("memory")
            mode = effect["mode"]
            if mode in {"read", "read_write", "unknown"}:
                reads.update(domains)
            if mode in {"write", "read_write", "unknown"}:
                writes.update(domains)
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

    def sequence(seq, env):
        for handle in seq:
            port = handle["port"]
            row = nodes[port]
            op = row["op"]
            if op == "block":
                sequence(row["body"], env)
            elif op == "if":
                apply(env, port, "condition", *footprint(port, False))
                left, right = env.copy(), env.copy()
                sequence(row["then"], left)
                sequence(row.get("else", []), right)
                for d in sorted(left.keys() | right.keys()):
                    a, b = version(left, d), version(right, d)
                    if a == b:
                        env[d] = a
                    else:
                        env[d] = f"{port}/join"
                        records.append(
                            dict(
                                kind="MERGE",
                                node=port,
                                domain=d,
                                inputs=[a, b],
                                output=env[d],
                            )
                        )
            elif op in {"map", "for", "while", "each", "do_while"}:
                if isinstance(row.get("init"), list):
                    sequence(row["init"], env)
                changed = set()
                for f in ("body", "condition", "step"):
                    for child in ports(row.get(f)):
                        changed.update(footprint(child, True)[1])
                initial = {d: version(env, d) for d in changed}
                env.update({d: f"{port}/header" for d in changed})
                # Calls and scalar updates in the condition/update can mutate
                # state too. A loop exit conservatively includes such writes.
                for f in ("extent", "range", "condition"):
                    for child in ports(row.get(f)):
                        apply(env, child, f, *footprint(child, True))
                sequence(row["body"], env)
                for child in ports(row.get("step")):
                    apply(env, child, "step", *footprint(child, True))
                for d in sorted(changed):
                    back = version(env, d)
                    records.append(
                        dict(
                            kind="LOOP_CARRY",
                            node=port,
                            domain=d,
                            initial=initial[d],
                            next=back,
                            header=f"{port}/header",
                            exit=f"{port}/exit",
                        )
                    )
                    env[d] = f"{port}/exit"
            else:
                apply(env, port, "effect", *footprint(port, True))

    exits = {h["port"] for h in entry[-1:]}
    if any(
        n["op"] in {"goto", "switch", "break", "continue", "unsupported", "do_while"}
        or n["op"] == "return"
        and i not in exits
        for i, n in enumerate(nodes)
    ):
        # The original CFG is still in the operation table. Expose the precise
        # analysis boundary rather than threading state through a lexical jump.
        return {"status": "conservative_cfg", "versions": []}
    sequence(entry, {})
    return {"status": "structured", "versions": records}
