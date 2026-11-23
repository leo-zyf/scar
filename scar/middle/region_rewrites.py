"""Prove graph rewrites from typed, executable region contracts."""

from collections import Counter
import math

from scar.contract import json_digest


def _reads(expression, index, parameters):
    kind, dtype, *operands = expression
    if kind == "load" and dtype == "f64" and len(operands) == 2:
        pointer, subscript = operands
        if (
            parameters.get(pointer) in ("ptr<f64>", "ptr<const f64>")
            and subscript == index
        ):
            return (pointer,)
    elif kind == "index" and expression == index:
        return ()
    elif expression == ["add", "i32", index, ["constant", "i32", 1]]:
        # i < n and n is a positive signed int32 count.
        return ()
    elif kind == "parameter" and len(operands) == 1 and dtype in ("i32", "f64"):
        if parameters.get(operands[0]) == dtype:
            return ()
    elif kind == "constant" and len(operands) == 1:
        value = operands[0]
        if dtype == "i32" and type(value) is int and -(2**31) <= value < 2**31:
            return ()
        if dtype == "f64" and (
            (type(value) is float and math.isfinite(value))
            or (type(value) is int and abs(value) <= 2**53)
        ):
            return ()
    elif (
        kind == "cast"
        and dtype == "f64"
        and len(operands) == 1
        and operands[0][1] == "i32"
    ):
        return _reads(operands[0], index, parameters)
    elif kind in ("add", "sub", "mul", "div") and dtype == "f64" and len(operands) == 2:
        if all(operand[1] == "f64" for operand in operands):
            left, right = (_reads(operand, index, parameters) for operand in operands)
            if left is not None and right is not None:
                return (*left, *right)
    return None


def _counted_store(loop, parameters):
    if (
        loop.get("kind") != "for"
        or loop.get("index_type") != "i32"
        or type(loop.get("index")) is not str
        or loop.get("start") != ["constant", "i32", 0]
        or loop.get("comparison") != "lt"
        or type(loop.get("step")) is not int
        or loop["step"] != 1
        or len(loop.get("body", ())) != 1
    ):
        return None
    stop = loop.get("stop", ())
    if (
        len(stop) != 3
        or stop[:2] != ["parameter", "i32"]
        or parameters.get(stop[2]) != "i32"
    ):
        return None
    index = ["index", "i32", loop["index"]]
    store = loop["body"][0]
    if (
        store.get("kind") != "store"
        or store.get("type") != "f64"
        or store.get("index") != index
        or parameters.get(store.get("pointer")) != "ptr<f64>"
        or store["value"][1] != "f64"
    ):
        return None
    reads = _reads(store["value"], index, parameters)
    if reads is None:
        return None
    return store, reads, index


def _accesses(row):
    return Counter(
        (
            access["kind"],
            access["pointer"],
            access["loop"],
            tuple(access["index"]),
            access["type"],
        )
        for access in row["effects"]["accesses"]
    )


def _fuse(row):
    if (
        (row["schema"], row["schema_version"], row["return_type"])
        != ("scar.native-region", 1, "void")
        or row["effects"]["external"] != "NONE"
        or row["effects"]["floating_exceptions"] != "PRESERVE_PROFILE"
        or len(row["body"]) != 2
    ):
        return None
    parameters = {item["name"]: item["type"] for item in row["parameters"]}
    if len(parameters) != len(row["parameters"]):
        return None
    first, second = row["body"]
    producer, consumer = (_counted_store(loop, parameters) for loop in row["body"])
    if producer is None or consumer is None or first["stop"] != second["stop"]:
        return None
    produced, reads, index = producer
    copied, copy_reads, consumer_index = consumer
    source, target = produced["pointer"], copied["pointer"]
    if (
        source == target
        or source in reads
        or target in reads
        or copy_reads != (source,)
        or copied["value"] != ["load", "f64", source, consumer_index]
    ):
        return None
    expected = Counter(
        [
            ("WRITE", source, 0, tuple(index), "f64"),
            ("READ", source, 1, tuple(consumer_index), "f64"),
            ("WRITE", target, 1, tuple(consumer_index), "f64"),
        ]
    )
    expected.update(("READ", name, 0, tuple(index), "f64") for name in reads)
    if _accesses(row) != expected:
        return None
    # Share the same binary64 value across the two observable stores.
    value = ["local", "f64", "produced"]
    definition = dict(
        kind="bind", binding="produced", type="f64", value=produced["value"]
    )
    body = [
        dict(
            first,
            body=[
                definition,
                dict(produced, value=value),
                dict(copied, index=index, value=value),
            ],
        )
    ]
    pointers = [name for name in parameters if name in {source, target, *reads}]
    guard = dict(
        kind="DISJOINT_FP64_RANGES", pointers=pointers, length=first["stop"][2]
    )
    return dict(row, optimized_body=body, guard=guard)


def _subtree_operations(region_id, regions):
    pending, operations = [region_id], set()
    while pending:
        region = regions[pending.pop()]
        operations.update(region.operations)
        pending.extend(region.children)
    return operations


def _within_span(witness, size):
    start, stop = witness["span"]
    return type(start) is int and type(stop) is int and 0 <= start < stop <= size


def _count_limit(row, source, operations, fragments, abi_digests):
    """Match a source anchor or an exported signature to its sealed input."""
    witness = row["fragment"]
    if not _within_span(row["source"], source["size_bytes"]) or any(
        operations[key].attributes.get("symbol") != row["symbol"]
        or operations[key].attributes.get("unit") != row["source"]["path"]
        for key in row["operations"]
    ):
        return None

    authority = witness.get("authority", "DIRECT_SOURCE")
    if authority == "TYPED_ABI_EXPORT":
        digest = abi_digests.get((witness["path"], row["symbol"]))
        if (
            not digest
            or digest != witness["abi_sha256"]
            or witness["symbol"] != row["symbol"]
        ):
            return None
    else:
        fragment = fragments.get(witness["path"])
        if (
            fragment is None
            or fragment["sha256"] != witness["sha256"]
            or not _within_span(witness, fragment["size"])
        ):
            return None
    if authority == "DIRECT_SOURCE":
        if (
            witness["slice_sha256"] == row["source"]["slice_sha256"]
            and "bindings" not in witness
        ):
            return 2**31 - 1
        return None
    if authority not in ("TYPED_ABI_EXPORT", "VERIFIED_ABI_WRAPPER"):
        return None
    domain = witness.get("count_domain", {})
    bindings = witness.get("bindings", {})
    maximum = domain.get("maximum")
    if (
        domain.get("parameter") != row["guard"]["length"]
        or domain.get("minimum") != 1
        or type(maximum) is not int
        or not 0 < maximum < 2**31
        or set(bindings) != {parameter["name"] for parameter in row["parameters"]}
        or len(set(bindings.values())) != len(bindings)
    ):
        return None
    return maximum


def rewrite_counted_regions(graph, rows, fragments, abi_digests, seals=None):
    """Return guarded typed bodies and their source-aligned proof ledger."""
    regions = {region.id: region for region in graph.regions}
    sources = graph.target["source_lock"]["files"]
    operations = {operation.id: operation for operation in graph.operations}
    optimized, proofs, seen = [], [], set()
    for row in rows:
        identity = row["region"]
        if seals is not None and seals.get(identity) != json_digest(row):
            continue
        if (
            seals is None
            and row["fragment"].get("authority", "DIRECT_SOURCE") != "DIRECT_SOURCE"
        ):
            continue
        region = regions.get(identity)
        source = sources.get(row["source"]["path"])
        if (
            region is None
            or region.kind != "FUNCTION"
            or source is None
            or source["sha256"] != row["source"]["sha256"]
            or identity in seen
            or not row["operations"]
            or len(set(row["operations"])) != len(row["operations"])
            or set(row["operations"]) != _subtree_operations(identity, regions)
        ):
            continue
        candidate = _fuse(row)
        if candidate is None:
            continue
        maximum = _count_limit(candidate, source, operations, fragments, abi_digests)
        if maximum is None:
            continue
        if "count_domain" in candidate["fragment"]:
            candidate["guard"]["max_count"] = maximum
        optimized.append(candidate)
        seen.add(identity)
        proofs.append(
            dict(
                rule="FUSE_ADJACENT_COUNTED_PRODUCER_COPY",
                region=identity,
                operations=row["operations"],
                guard=candidate["guard"],
                kept_effects=[
                    "BOTH_OUTPUT_STORES",
                    "FP_ARITHMETIC_ORDER",
                    "ORIGINAL_FALLBACK",
                ],
                conditions=["MASKED_FP_TRAPS"],
                original_loops=2,
                transformed_loops=1,
                eliminated_copy_loads_proven=True,
            )
        )
    return tuple(optimized), tuple(proofs)
