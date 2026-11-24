"""M4 conditional copy/materialization facts over canonical operations only.

Payload bytes describe the logical destination, not measured memory traffic.
Resource domains are inventory candidates; this pass neither chooses placement
nor asserts that a conditional operation actually executed.
"""

from copy import deepcopy
from math import prod

from scar.contract import digest, json_bytes, semantic_form, walk_operations

_LOOPS = {"for", "while", "do_while", "map", "each"}
_BRANCHES = {"if", "select"}
_CALLS = {"call", "indirect_call", "library_call", "process_spawn", "process_request"}
_FILES = {"fs.copy_file", "fs.copy", "fs.copy2", "fs.copy_tree"}
_WIDTHS = {
    "bool": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}
_DOMAINS = {
    "host_memory": {"DRAM", "NUMA_MEMORY"},
    "device_memory": {"DEVICE_MEMORY", "HBM"},
    "filesystem": {"STORAGE_MOUNT"},
}


def _integer(value):
    return (
        value.get("value")
        if isinstance(value, dict)
        and value.get("op") == "literal"
        and type(value.get("value")) is int
        and value["value"] >= 0
        else None
    )


def _slot(value):
    return (
        (value.get("scope"), value["name"])
        if isinstance(value, dict) and value.get("op") in {"var", "decl"}
        else None
    )


def _region(node, shape=None, *, contiguous=False):
    spec = node.get("type", {})
    dtype = (
        spec.get("element")
        if isinstance(spec, dict) and spec.get("kind") == "ndarray"
        else None
    )
    if not isinstance(dtype, str):
        dtype = None
    rank = spec.get("rank") if isinstance(spec, dict) else None
    if shape is None and type(rank) is int:
        shape = [None] * rank
    size = _WIDTHS.get(dtype)
    strides = None
    if contiguous and shape is not None:
        strides = []
        for dimension in reversed(shape):
            strides.insert(0, size)
            size = (
                size * dimension if size is not None and dimension is not None else None
            )
    return dict(
        dtype=dtype,
        shape=shape,
        offset_bytes=0 if contiguous else None,
        strides_bytes=strides,
    )


def _common(left, right):
    """Join fields independently; different origins do not erase common shapes."""
    result = {
        key: value if right.get(key) == value else None for key, value in left.items()
    }
    a, b = left.get("_members"), right.get("_members")
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        result["_members"] = [
            _common(x, y) if isinstance(x, dict) and isinstance(y, dict) else None
            for x, y in zip(a, b)
        ]
    return result


def analyze(module, *, partial=False):
    """Project static conditional copies without reading sources or data values."""
    inventory = module.get("resource_model", {}).get("resources", ())
    resources = sorted({row["id"] for row in inventory})
    candidates = {
        domain: sorted({row["id"] for row in inventory if row.get("kind") in kinds})
        for domain, kinds in _DOMAINS.items()
    }
    dispatchers = sorted(
        row["id"] for row in inventory if row.get("kind") == "CPU_POOL"
    )
    functions = {
        fn["name"]: fn
        for fn in module.get("functions", ())
        if fn.get("execution_kind") != "python_protocol"
    }
    definitions = {fn["name"] for fn in module.get("functions", ())}
    if partial:
        protocol = module.get("protocol_analysis", {})
        for fragment in protocol.get("closed_fragments", ()):
            if fragment.get("completion") == "complete":
                functions.setdefault(fragment["symbol"], fragment["function"])
        for fragment in protocol.get("prefix_fragments", ()):
            functions.setdefault(
                fragment["symbol"], dict(name=fragment["symbol"], body=fragment["body"])
            )
    externs = {
        row["name"]
        for row in module.get("externs", ())
        if row.get("provider") == "libc"
        and row.get("name") in {"memcpy", "memmove"}
        and row["name"] not in definitions
    }
    nodes, edges, copies = [], [], 0

    def identity(owner, path, meaning):
        return (
            "resource:" + digest(json_bytes([owner, path, semantic_form(meaning)]))[:32]
        )

    def projection(
        owner,
        node,
        path,
        scope,
        source_path,
        target_path,
        source_region,
        target_region,
        *,
        domain,
        relation="equal_bytes",
        storage="fresh",
        byte_count=None,
        conditions=(),
    ):
        nonlocal copies
        copy_id = identity(owner, path, node)
        mats = []
        for role, operand_path, region in (
            ("source", source_path, source_region),
            ("target", target_path, target_region),
        ):
            mat_id = identity(owner, path, [copy_id, role])
            reference = dict(owner=owner, path=operand_path)
            materialization = dict(
                id=mat_id,
                kind="MATERIALIZATION",
                owner=owner,
                operand=(
                    None
                    if role == "target"
                    and domain == "filesystem"
                    and storage == "may_create"
                    and node.get("interface") != "fs.copy_file"
                    else reference
                ),
                domain=domain,
                region={
                    **{
                        key: value
                        for key, value in region.items()
                        if not key.startswith("_")
                    },
                    **(
                        {"requested_destination": reference}
                        if role == "target" and domain == "filesystem"
                        else {}
                    ),
                    "storage": dict(
                        kind=(
                            "fresh_allocation"
                            if role == "target" and storage in {"fresh", "fresh_tree"}
                            else "operand_storage"
                        ),
                        operation=copy_id,
                        origin=region.get("_origin"),
                    ),
                },
                logical_value=(
                    dict(kind="operand_version", **reference)
                    if role == "source"
                    else dict(
                        kind={
                            "equal_bytes": "same_as",
                            "broadcast": "broadcast_of",
                            "tree_copy": "tree_copy_of",
                        }[relation],
                        materialization=mats[0],
                    )
                ),
                location=dict(
                    kind=(
                        "filesystem_path"
                        if domain == "filesystem"
                        else "resource_domain"
                    ),
                    namespace="source_inventory",
                    resource_candidates=candidates[domain],
                    binding=(
                        "execution_namespace"
                        if domain == "filesystem"
                        else "runtime_allocator"
                    ),
                ),
                ready=(
                    dict(kind="input_ready", **reference)
                    if role == "source"
                    else dict(kind="operation_complete", operation=copy_id)
                ),
            )
            nodes.append(materialization)
            mats.append(mat_id)
            edges.extend(
                dict(kind="MAY_RESIDE_ON", source=mat_id, target=resource)
                for resource in candidates[domain]
            )
        if byte_count is None:
            shape, width = target_region["shape"], _WIDTHS.get(target_region["dtype"])
            byte_count = (
                dict(kind="constant", value=prod(shape) * width)
                if shape is not None
                and width is not None
                and all(type(n) is int for n in shape)
                else dict(
                    kind=(
                        "result_nbytes" if relation == "broadcast" else "operand_nbytes"
                    ),
                    owner=owner,
                    path=path if relation == "broadcast" else source_path,
                )
            )
        nodes.append(
            dict(
                id=copy_id,
                kind="COPY",
                owner=owner,
                path=path,
                site=node.get("site"),
                scope=deepcopy(scope),
                opcode=(
                    node.get("interface")
                    if node["op"] == "host_call"
                    else node.get("callee") if node["op"] == "call" else node["op"]
                ),
                source=mats[0],
                target=mats[1],
                byte_count=byte_count,
                blocking=True,
                value_relation=relation,
                destination_storage=storage,
                effects=sorted(
                    set(node.get("effects", ()))
                    | {"STATE", "XFER"}
                    | ({"IO"} if domain == "filesystem" else set())
                    | ({"MEM"} if storage in {"fresh", "fresh_tree"} else set())
                ),
                dispatch=dict(
                    kind="host_thread",
                    namespace="source_inventory",
                    resource_candidates=dispatchers,
                    binding="caller_thread",
                ),
                conditions=["operation_reached_and_normally_completed", *conditions],
            )
        )
        edges.extend(
            [
                dict(kind="READS", source=mats[0], target=copy_id),
                dict(kind="PRODUCES", source=copy_id, target=mats[1]),
            ]
        )
        edges.extend(
            dict(kind="MAY_DISPATCH_ON", source=copy_id, target=resource)
            for resource in dispatchers
        )
        copies += 1

    for owner, function in functions.items():
        environment = {}
        for index, parameter in enumerate(function.get("params", ())):
            region = _region(parameter)
            if region["dtype"] is not None:
                region["_origin"] = dict(owner=owner, path=["params", index])
                environment[None, parameter["name"]] = region

        def visit(value, path, scope):
            if isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, [*path, index], scope)
                    if isinstance(child, dict) and child.get("op") in {
                        "return",
                        "break",
                        "continue",
                        "goto",
                        "throw",
                        "raise",
                    }:
                        environment.clear()
                        break
                return None
            if not isinstance(value, dict):
                return None
            op = value.get("op")
            if op == "var":
                return deepcopy(environment.get(_slot(value), _region(value)))
            if op == "product_make":
                members = [
                    visit(item, [*path, "items", index], scope) or _region(item)
                    for index, item in enumerate(value["items"])
                ]
                if any(
                    child["op"] in _CALLS
                    or child["op"] == "store_bytes"
                    or child["op"].startswith("object_")
                    for child in walk_operations(value["items"])
                ):
                    members = [_region(item) for item in value["items"]]
                return {**_region(value), "_members": members}
            if op == "product_get":
                base = visit(value["base"], [*path, "base"], scope)
                members = base.get("_members") if isinstance(base, dict) else None
                index = value.get("index")
                if (
                    isinstance(members, list)
                    and type(index) is int
                    and 0 <= index < len(members)
                    and isinstance(members[index], dict)
                ):
                    return deepcopy(members[index])
                return _region(value)
            if op in _BRANCHES:
                visit(value["condition"], [*path, "condition"], scope)
                before, states, regions = dict(environment), [], []
                for arm in ("then", "else"):
                    environment.clear()
                    environment.update(before)
                    regions.append(
                        visit(
                            value.get(arm, []),
                            [*path, arm],
                            [*scope, dict(kind=op, path=path, arm=arm)],
                        )
                    )
                    states.append(dict(environment))
                environment.clear()
                environment.update(
                    {
                        key: _common(region, states[1][key])
                        for key, region in states[0].items()
                        if key in states[1]
                    }
                )
                return regions[0] if regions[0] == regions[1] else _region(value)
            if op in _LOOPS:
                nested = [*scope, dict(kind=op, path=path)]
                if op == "for":
                    visit(value.get("init", []), [*path, "init"], nested)
                written, opaque = set(), False
                for child in walk_operations(
                    {key: val for key, val in value.items() if key != "init"}
                ):
                    if child["op"] in {"decl", "assign", "update"}:
                        target = child if child["op"] == "decl" else child.get("target")
                        slot = _slot(target)
                        if slot is not None:
                            written.add(slot)
                        else:
                            opaque = True
                    opaque |= (
                        child["op"] in _CALLS
                        or child["op"] == "store_bytes"
                        or child["op"].startswith("object_")
                    )

                def havoc():
                    if opaque:
                        environment.clear()
                    else:
                        for slot in written:
                            environment.pop(slot, None)

                havoc()
                invariant = dict(environment)
                fields = (
                    ("body", "condition")
                    if op == "do_while"
                    else ("condition", "extent", "iterable", "body", "step")
                )
                for field in fields:
                    if field in value:
                        # A phase may also be entered through continue/break
                        # alternatives. Transfer only proven loop invariants.
                        environment.clear()
                        environment.update(invariant)
                        visit(value[field], [*path, field], nested)
                environment.clear()
                environment.update(invariant)
                for declaration in value.get("init", ()):
                    if declaration.get("op") == "decl":
                        environment.pop(_slot(declaration), None)
                return None
            if op == "block":
                visit(value.get("body", []), [*path, "body"], scope)
                for item in value.get("body", ()):
                    if item.get("op") == "decl":
                        environment.pop(_slot(item), None)
                return None
            if op in {
                "try",
                "switch",
                "case",
                "default",
                "exception_handler",
                "resource_scope",
                "schedule_choice",
            }:
                environment.clear()
                for field, child in value.items():
                    visit(
                        child,
                        [*path, field],
                        [*scope, dict(kind=op, path=path, arm=field)],
                    )
                    environment.clear()
                return None
            if op == "decl":
                region = visit(value.get("init"), [*path, "init"], scope)
                if value.get("storage") in {"static", "thread_local", "extern"}:
                    region = None
                region = region or _region(value)
                if region["dtype"] is not None or isinstance(
                    region.get("_members"), list
                ):
                    environment[_slot(value)] = region
                else:
                    environment.pop(_slot(value), None)
                return None
            if op == "assign":
                visit(value.get("target"), [*path, "target"], scope)
                region = visit(value.get("value"), [*path, "value"], scope)
                slot = _slot(value.get("target"))
                if slot is None:
                    environment.clear()
                else:
                    region = region or _region(value["target"])
                    if region["dtype"] is not None or isinstance(
                        region.get("_members"), list
                    ):
                        environment[slot] = region
                    else:
                        environment.pop(slot, None)
                return None
            if op == "binary" and value.get("operator") in {"and", "or", "land", "lor"}:
                visit(value.get("left"), [*path, "left"], scope)
                before = dict(environment)
                visit(
                    value.get("right"),
                    [*path, "right"],
                    [*scope, dict(kind=op, path=path, arm="right")],
                )
                common = {
                    key: _common(region, before[key])
                    for key, region in environment.items()
                    if key in before
                }
                environment.clear()
                environment.update(common)
                return None
            children = {
                field: visit(child, [*path, field], scope)
                for field, child in value.items()
                if isinstance(child, (dict, list))
                and field
                not in {
                    "type",
                    "argument_facts",
                    "state_contract",
                    "state_evidence",
                    "source_evidence",
                    "location",
                    "origin_region",
                }
            }
            shape = (
                [_integer(dim) for dim in value["shape"]]
                if isinstance(value.get("shape"), list)
                else None
            )
            if op in {
                "array_create",
                "array_literal",
                "array_reshape",
                "array_broadcast_copy",
            }:
                region = _region(value, shape, contiguous=True)
                if op == "array_reshape":
                    source = children.get("value") or _region(value.get("value", {}))
                    if (
                        value.get("inference") == "numpy_single_negative_dimension"
                        and shape is not None
                    ):
                        negative = [
                            i
                            for i, dim in enumerate(value["shape"])
                            if dim.get("op") == "literal"
                            and type(dim.get("value")) is int
                            and dim["value"] < 0
                        ]
                        known = [n for i, n in enumerate(shape) if i not in negative]
                        src_shape = source["shape"]
                        if (
                            len(negative) == 1
                            and None not in known
                            and src_shape is not None
                            and None not in src_shape
                        ):
                            divisor, count = prod(known), prod(src_shape)
                            if divisor and count % divisor == 0:
                                shape[negative[0]] = count // divisor
                    region = _region(
                        value, shape, contiguous=value.get("layout") == "C"
                    )
                    region["_origin"] = source.get("_origin")
                    region["offset_bytes"] = source["offset_bytes"]
            elif op == "array_view":
                base = children.get("base") or _region(value.get("base", {}))
                indices = value.get("indices", ())
                region = dict(
                    base,
                    shape=(
                        base["shape"][len(indices) :]
                        if base["shape"] is not None
                        else None
                    ),
                    strides_bytes=(
                        base["strides_bytes"][len(indices) :]
                        if base["strides_bytes"] is not None
                        else None
                    ),
                    offset_bytes=None,
                )
            elif op == "array_copy":
                source = children.get("value") or _region(value.get("value", {}))
                region = _region(value, source["shape"], contiguous=True)
            else:
                region = _region(value)
            if region["dtype"] is not None and op in {
                "array_create",
                "array_literal",
                "array_copy",
                "array_broadcast_copy",
                "array_unary",
                "array_binary",
                "array_iota",
            }:
                region["_origin"] = dict(owner=owner, path=path)
            if op in {"array_copy", "array_broadcast_copy"}:
                projection(
                    owner,
                    value,
                    path,
                    scope,
                    [*path, "value"],
                    path,
                    children.get("value") or _region(value.get("value", {})),
                    region,
                    domain="host_memory",
                    relation=(
                        "broadcast" if op == "array_broadcast_copy" else "equal_bytes"
                    ),
                )
            elif (
                op == "host_call"
                and value.get("interface") in _FILES
                and len(value.get("args", ())) >= 2
            ):
                tree = value["interface"] == "fs.copy_tree"
                file_region = dict(
                    dtype=None,
                    shape=None,
                    offset_bytes=None,
                    strides_bytes=None,
                    path_resolution="execution_namespace_and_destination_kind_unresolved",
                )
                projection(
                    owner,
                    value,
                    path,
                    scope,
                    [*path, "args", 0],
                    [*path, "args", 1],
                    file_region,
                    file_region,
                    domain="filesystem",
                    relation="tree_copy" if tree else "equal_bytes",
                    storage="fresh_tree" if tree else "may_create",
                    byte_count=dict(
                        kind="tree_bytes" if tree else "operand_nbytes",
                        owner=owner,
                        path=[*path, "args", 0],
                    ),
                    conditions=[
                        "source_destination_alias_and_access_contract",
                        "destination_file_or_directory_resolution",
                        "regular_source_and_follow_symlinks_content_contract",
                        "filesystem_and_concurrent_writer_contracts",
                    ],
                )
            elif (
                op == "call"
                and value.get("callee") in externs
                and len(value.get("args", ())) == 3
            ):
                count = _integer(value["args"][2])
                byte_region = dict(
                    dtype="u8", shape=[count], offset_bytes=None, strides_bytes=[1]
                )
                projection(
                    owner,
                    value,
                    path,
                    scope,
                    [*path, "args", 1],
                    [*path, "args", 0],
                    byte_region,
                    byte_region,
                    domain="host_memory",
                    storage="existing",
                    byte_count=(
                        dict(kind="constant", value=count)
                        if count is not None
                        else dict(
                            kind="operand_count", owner=owner, path=[*path, "args", 2]
                        )
                    ),
                    conditions=[
                        "valid_pointer_regions_and_extent",
                        (
                            "nonoverlapping_regions"
                            if value["callee"] == "memcpy"
                            else "overlap_permitted"
                        ),
                    ],
                )
            if (
                op in _CALLS
                or op in {"update", "store_bytes"}
                or op
                and op.startswith("object_")
            ):
                environment.clear()
            return region

        visit(function.get("initializers", []), ["initializers"], [])
        visit(function.get("body", []), ["body"], [])
    return dict(
        schema="scar.resource-flow",
        schema_version=1,
        resources=resources,
        nodes=nodes,
        edges=edges,
        coverage=dict(
            kind="partial" if partial else "canonical",
            functions=len(functions),
            copies=copies,
        ),
    )
