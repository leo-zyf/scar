"""Lower a scheduled import-frame component to typed C++20 constant tables."""

from pathlib import Path

from scar.contract import digest, document, member, read_member, write_json
from .model import (
    Decl,
    Expr,
    Type,
    Unit,
    construct,
    encode_unit,
    integer,
    name,
    named,
    variable,
)
from .receipts import is_digest

_OPERATIONS = dict(
    LOAD_CONST="constant",
    LOAD_SMALL_INT="small_int",
    IMPORT_NAME="import_name",
    STORE_NAME="store_name",
    STORE_GLOBAL="store_global",
    PUSH_NULL="push_null",
    LOAD_NAME="load_name",
    LOAD_ATTR="attribute",
    CALL="call",
    POP_TOP="discard",
)


def _array(symbol, element, items):
    ty = Type("template", name="std::array", args=(element, integer(len(items))))
    return variable(
        symbol, ty, construct(ty, *items, form="brace"), storage=("inline", "constexpr")
    )


def lower_library(path, revision, scheduled, destination):
    from .target import _publish_blob

    if scheduled["status"] != "READY" or scheduled["entry_abi"] != "live_native_frame":
        raise ValueError("source-frame library lacks a qualified component schedule")
    rows = read_member(path.parent, scheduled["regions"])
    if not rows:
        raise ValueError("source-frame library has no scheduled regions")
    destination = Path(destination).resolve()
    definitions, entries = [], []
    instruction, region = named("instruction"), named("region")
    for i, row in enumerate(rows):
        code, minor = row["code_sha256"], row["minor"]
        if not is_digest(code) or minor not in (12, 14) or not row["instructions"]:
            raise ValueError("invalid source-frame code identity")
        steps = []
        previous = -1
        for opcode, offset, operand, spelling in row["instructions"]:
            if (
                opcode not in _OPERATIONS
                or type(offset) is not int
                or not previous < offset < 2**32
                or type(operand) is not int
                or not 0 <= operand < 2**32
                or not isinstance(spelling, str)
            ):
                raise ValueError("invalid source-frame instruction")
            steps.append(
                construct(
                    instruction,
                    name("opcode::" + _OPERATIONS[opcode]),
                    integer(offset),
                    integer(operand),
                    Expr("string", value=spelling),
                    form="brace",
                )
            )
            previous = offset
        symbol = f"steps_{i}"
        definitions.append(_array(symbol, instruction, steps))
        entries.append(
            construct(
                region,
                Expr("string", value=code),
                integer(minor),
                name(symbol),
                form="brace",
            )
        )
    definitions.append(_array("regions", region, entries))
    declarations = tuple(definitions)
    for ns in ("generated", "imports", "scar"):
        declarations = (Decl("namespace", name=ns, members=declarations),)
    tree = Unit(
        once=True,
        includes=(("system", "array"), ("local", "import.hpp")),
        declarations=declarations,
    )
    support = Path(__file__).with_name("import.hpp").read_bytes()
    product = document(
        "scar.cpp20-target-ir",
        kind="source_frame_library",
        scheduled_ir_sha256=revision,
        ir_sha256=scheduled["ir_sha256"],
        units=[
            dict(
                output="regions.hpp",
                tree=member(destination, "target-units", encode_unit(tree)),
            )
        ],
        runtime=_publish_blob(destination, "runtime", support, digest(support), ".hpp"),
        sites=[{k: r[k] for k in ("owner", "site", "occurrence")} for r in rows],
    )
    target = destination / "target-ir.json"
    write_json(target, product)
    return target
