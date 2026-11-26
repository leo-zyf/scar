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
    utf8_literal,
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


_BODY_OPERATIONS = {
    "LOAD_CHILD_CODE": "child_code",
    "MAKE_FUNCTION": "make_function",
    "STORE_NAME": "store_name",
    "STORE_GLOBAL": "store_global",
    "LOAD_CONST": "constant",
    "LOAD_SMALL_INT": "small_int",
    "RETURN_CONST": "return_const",
    "RETURN_VALUE": "return_value",
}


def _body_tables(rows, definitions):
    instruction, slot, body_type = (
        named(n) for n in ("body_instruction", "constant_slot", "body_program")
    )
    entries = []
    for i, row in enumerate(rows):
        if not is_digest(row["code_sha256"]) or row["kind"] not in {
            "module",
            "function",
        }:
            raise ValueError("invalid activation body code identity")
        steps_name, slots_name = f"body_steps_{i}", f"constants_{i}"
        program = row["instructions"]
        if (
            not program
            or program[0][1] != 2
            or any(next_[1] != prev[1] + 2 for prev, next_ in zip(program, program[1:]))
        ):
            raise ValueError("invalid activation body instruction interval")
        # Each declaration consumes one code object and publishes the function
        # before the next declaration. A prefix can stop only after its store.
        pos = 0
        if row["kind"] == "module":
            while pos < len(program) and program[pos][0] == "LOAD_CHILD_CODE":
                group = program[pos : pos + 3]
                if (
                    len(group) != 3
                    or group[1][0] != "MAKE_FUNCTION"
                    or group[1][2] != 0
                    or group[0][3] != group[1][3]
                    or group[2][0] not in {"STORE_NAME", "STORE_GLOBAL"}
                    or not group[2][3]
                ):
                    raise ValueError("invalid function publication in activation body")
                pos += 3
        tail = [ins[0] for ins in program[pos:]]
        if tail not in (["RETURN_CONST"], ["LOAD_CONST", "RETURN_VALUE"]) and not (
            row["kind"] == "function"
            and tail == ["LOAD_SMALL_INT", "RETURN_VALUE"]
            or row["kind"] == "module"
            and not tail
        ):
            raise ValueError("invalid source activation return sequence")
        steps = []
        for opcode, offset, operand, spelling in program:
            if (
                opcode not in _BODY_OPERATIONS
                or any(
                    type(n) is not int or not 0 <= n < 2**32 for n in (offset, operand)
                )
                or not isinstance(spelling, str)
                or opcode in {"LOAD_CHILD_CODE", "MAKE_FUNCTION"}
                and not is_digest(spelling)
            ):
                raise ValueError("invalid activation body instruction")
            steps.append(
                construct(
                    instruction,
                    name("body_opcode::" + _BODY_OPERATIONS[opcode]),
                    integer(offset),
                    integer(operand),
                    Expr("string", value=spelling),
                    form="brace",
                )
            )
        used = {
            str(index)
            for opcode, _, index, _ in program
            if opcode in {"LOAD_CONST", "RETURN_CONST"}
        }
        if used != row["constants"].keys():
            raise ValueError("activation constants do not match their source slots")
        slots = []
        for index, literal in row["constants"].items():
            value, kind = literal["value"], literal["kind"]
            if type(value).__name__ != kind or kind not in {
                "NoneType",
                "str",
                "float",
                "bool",
                "int",
            }:
                raise ValueError("invalid activation constant")
            if row["kind"] == "module" and value is not None:
                raise ValueError("invalid module return constant")
            spelling = (
                value.hex() if kind == "float" else "" if value is None else str(value)
            )
            slots.append(
                construct(
                    slot,
                    integer(int(index)),
                    Expr("string", value=kind),
                    utf8_literal(spelling, "std::string_view"),
                    form="brace",
                )
            )
        returns = program[-1][0] in {"RETURN_CONST", "RETURN_VALUE"}
        if row["resume_offset"] != (None if returns else program[-1][1] + 2):
            raise ValueError("invalid activation continuation")
        definitions.extend(
            (_array(steps_name, instruction, steps), _array(slots_name, slot, slots))
        )
        entries.append(
            construct(
                body_type,
                Expr("string", value=row["symbol"]),
                Expr("string", value=row["code_sha256"]),
                name(steps_name),
                name(slots_name),
                Expr("literal", value=returns),
                integer(row["resume_offset"] or 0),
                form="brace",
            )
        )
    definitions.append(_array("bodies", body_type, entries))


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
                construct(named("body_provider"), name("bodies"), form="brace"),
                form="brace",
            )
        )
    _body_tables(
        read_member(path.parent, scheduled["bodies"]) if "bodies" in scheduled else [],
        definitions,
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
