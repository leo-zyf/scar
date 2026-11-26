"""Lower a scheduled import-frame component to typed C++20 constant tables."""

from pathlib import Path

from scar.contract import (
    digest,
    document,
    json_bytes,
    member,
    read_member,
    valid_python_source_successor,
    write_json,
)
from .model import (
    Decl,
    Expr,
    Stmt,
    call,
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
    "LOAD_DEFAULT_TUPLE": "default_tuple",
    "LOAD_FRAME_CONST": "constant",
    "LOAD_NAME": "load_name",
    "BUILD_LIST": "build_list",
    "BUILD_TUPLE": "build_defaults_tuple",
    "LIST_EXTEND": "list_extend",
    "STORE_SUBSCR": "store_subscript",
    "LOAD_CHILD_CODE": "child_code",
    "SET_FUNCTION_ATTRIBUTE": "set_function_defaults",
    "MAKE_FUNCTION": "make_function",
    "STORE_NAME": "store_name",
    "STORE_GLOBAL": "store_global",
    "LOAD_CONST": "constant",
    "LOAD_SMALL_INT": "small_int",
    "RETURN_CONST": "return_const",
    "RETURN_VALUE": "return_value",
}


def _frame_load(op, n, text):
    return (
        op == "LOAD_NAME"
        and isinstance(text, str)
        and text.isidentifier()
        or op == "LOAD_FRAME_CONST"
        and is_digest(text)
        or op == "LOAD_SMALL_INT"
        and type(n) is int
        and 0 <= n < 256
        and text == ""
    )


def _publication_end(row, pos):
    program = row["instructions"]
    defaults = program[pos][0] == "LOAD_DEFAULT_TUPLE"
    minor = row.get("minor")
    start = pos + int(defaults)
    if _frame_load(program[pos][0], *program[pos][2:]):
        while start < len(program):
            op, _, n, text = program[start]
            if not _frame_load(op, n, text):
                break
            if op == "LOAD_SMALL_INT" and minor != 14:
                raise ValueError("invalid immediate in function defaults")
            start += 1
        if (
            start >= len(program)
            or program[start][0] != "BUILD_TUPLE"
            or program[start][2:] != [start - pos, ""]
        ):
            raise ValueError("invalid stack for function defaults tuple")
        defaults = True
        start += 1
    if defaults and minor not in (12, 13, 14):
        raise ValueError("invalid source version for function defaults")
    split = defaults and minor >= 13
    stop = start + 3 + int(split)
    group = program[start:stop]
    if (
        len(group) != stop - start
        or group[0][0] != "LOAD_CHILD_CODE"
        or group[1][0] != "MAKE_FUNCTION"
        or group[1][2] != int(defaults and not split)
        or group[0][3] != group[1][3]
        or split
        and (group[2][0], group[2][2], group[2][3])
        != ("SET_FUNCTION_ATTRIBUTE", 1, "defaults")
        or group[-1][0] not in {"STORE_NAME", "STORE_GLOBAL"}
        or not group[-1][3]
    ):
        raise ValueError("invalid function publication in activation body")
    return stop


def _validate_subscript(row):
    program = row["instructions"]
    if (
        row["kind"] != "module"
        or program[-1][0] != "STORE_SUBSCR"
        or not valid_python_source_successor(program[-1][1], row["normal_successor"])
        or row["resume_offset"] != row["normal_successor"]["offset"]
    ):
        raise ValueError("invalid module subscription continuation")
    stack = []
    for i, (op, _, n, text) in enumerate(program):
        if type(n) is not int or not 0 <= n < 2**32:
            raise ValueError("invalid subscription instruction argument")
        if _frame_load(op, n, text):
            stack.append("object")
        elif op == "BUILD_LIST" and not text and n <= len(stack):
            if n:
                del stack[-n:]
            stack.append("list")
        elif (
            op == "LIST_EXTEND"
            and not text
            and n == 1
            and len(stack) >= 2
            and stack[-2] == "list"
        ):
            stack.pop()
        elif (
            op == "STORE_SUBSCR"
            and not text
            and n == 0
            and len(stack) == 3
            and i == len(program) - 1
        ):
            stack.clear()
        else:
            raise ValueError("invalid operand stack in module subscription")


def _body_tables(rows, definitions):
    instruction, slot, body_type = (
        named(n) for n in ("body_instruction", "constant_slot", "body_program")
    )
    entries = []
    intervals = {}
    for i, row in enumerate(rows):
        if not is_digest(row["code_sha256"]) or row["kind"] not in {
            "module",
            "function",
        }:
            raise ValueError("invalid activation body code identity")
        steps_name, slots_name = f"body_steps_{i}", f"constants_{i}"
        program = row["instructions"]
        start = row.get("entry_offset", 2)
        if (
            not program
            or type(start) is not int
            or start < 2
            or start % 2
            or (row["kind"] == "function" and start != 2)
            or program[0][1] != start
            or any(next_[1] != prev[1] + 2 for prev, next_ in zip(program, program[1:]))
        ):
            raise ValueError("invalid activation body instruction interval")
        # Each declaration consumes one code object and publishes the function
        # before the next declaration. A prefix can stop only after its store.
        pos = 0
        if "normal_successor" in row:
            _validate_subscript(row)
            pos = len(program)
        elif row["kind"] == "module":
            while pos < len(program):
                if (
                    program[pos][0] == "LOAD_CONST"
                    and pos + 1 < len(program)
                    and program[pos + 1][0] in {"STORE_NAME", "STORE_GLOBAL"}
                ):
                    if not program[pos + 1][3]:
                        raise ValueError("module constant publication lacks a name")
                    pos += 2
                    continue
                op, _, n, text = program[pos]
                if op not in {
                    "LOAD_CHILD_CODE",
                    "LOAD_DEFAULT_TUPLE",
                } and not _frame_load(op, n, text):
                    break
                pos = _publication_end(row, pos)
        tail = [ins[0] for ins in program[pos:]]
        if tail not in (["RETURN_CONST"], ["LOAD_CONST", "RETURN_VALUE"]) and not (
            row["kind"] == "function"
            and tail == ["LOAD_SMALL_INT", "RETURN_VALUE"]
            or row["kind"] == "module"
            and not tail
        ):
            raise ValueError("invalid source activation return sequence")
        steps = []
        slot_types = {
            int(index): ("LOAD_FRAME_CONST", digest(json_bytes(literal["value"])))
            for index, literal in row["constants"].items()
        }
        for opcode, offset, operand, spelling in program:
            tuple_size = 0
            if opcode == "LOAD_DEFAULT_TUPLE":
                if type(spelling) is not int or not 0 < spelling < 2**32:
                    raise ValueError("invalid exact tuple arity in activation body")
                tuple_size, spelling = spelling, ""
            if opcode in {"LOAD_DEFAULT_TUPLE", "LOAD_CHILD_CODE", "LOAD_FRAME_CONST"}:
                identity = (opcode, tuple_size or spelling)
                if slot_types.setdefault(operand, identity) != identity:
                    raise ValueError(
                        "inconsistent identity for an activation constant slot"
                    )
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
                    *((integer(tuple_size),) if tuple_size else ()),
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
            if (
                row["kind"] == "module"
                and tail
                and str(program[pos][2]) == index
                and value is not None
            ):
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
        if "normal_successor" not in row and row["resume_offset"] != (
            None if returns else program[-1][1] + 2
        ):
            raise ValueError("invalid activation continuation")
        intervals.setdefault((row["symbol"], row["code_sha256"]), []).append(
            (
                start,
                (
                    row["resume_offset"] - 2
                    if "normal_successor" in row
                    else program[-1][1]
                ),
            )
        )
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
    for spans in intervals.values():
        spans.sort()
        if any(start <= end for (_, end), (start, _) in zip(spans, spans[1:])):
            raise ValueError("overlapping activation body instruction intervals")
    definitions.append(_array("bodies", body_type, entries))


def lower_library(path, revision, scheduled, destination):
    from .target import _publish_blob

    if scheduled["status"] != "READY" or scheduled["entry_abi"] != "live_native_frame":
        raise ValueError("source-frame library lacks a qualified component schedule")
    rows = read_member(path.parent, scheduled["regions"])
    destination = Path(destination).resolve()
    bodies = (
        read_member(path.parent, scheduled["bodies"]) if "bodies" in scheduled else []
    )
    tree = _frame_tree(rows, bodies)
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


def _frame_tree(rows, bodies, modules=None):
    if not rows:
        raise ValueError("source-frame library has no scheduled regions")
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
    _body_tables(bodies, definitions)
    definitions.append(_array("regions", region, entries))
    if modules is not None:
        module_type = named("project_module")
        definitions.append(
            _array(
                "modules",
                module_type,
                [
                    construct(
                        module_type,
                        *(
                            Expr("string", value=row[k])
                            for k in ("name", "symbol", "code_sha256")
                        ),
                        form="brace",
                    )
                    for row in modules
                ],
            )
        )
    declarations = tuple(definitions)
    for ns in ("generated", "imports", "scar"):
        declarations = (Decl("namespace", name=ns, members=declarations),)
    return Unit(
        once=True,
        includes=(
            ("system", "array"),
            ("local", "import.hpp" if modules is None else "scar/project_import.hpp"),
        ),
        declarations=declarations,
    )


def isolated_units(folder, physical, module):
    from scar.contract import digest, json_bytes

    effects = physical.get("source_import_effects", {})
    certificate = effects.get("isolated_entry")
    if (
        not isinstance(certificate, dict)
        or module.get("isolated_entry_binding_sha256")
        != digest(json_bytes(certificate))
        or certificate.get("entry_environment_sha256")
        != digest(json_bytes(module.get("python_entry_environment")))
        or effects.get("ledger") != module.get("source_imports")
        or certificate.get("ledger_sha256") != effects.get("ledger_sha256")
        or certificate.get("ledger_sha256")
        != digest(json_bytes(read_member(folder, module["source_imports"])))
    ):
        raise ValueError(
            "local module entry does not match its sealed execution domain"
        )
    schedule = read_member(folder, physical["source_frame_schedule"])
    if (
        schedule["kind"] != "source_frame_library"
        or schedule["ir_sha256"] != physical["ir_sha256"]
    ):
        raise ValueError("local module provider belongs to another schedule")
    rows = read_member(folder, schedule["regions"])
    bodies = read_member(folder, schedule["bodies"])
    if certificate["deployment"] != "isolated_project_modules_v1" or certificate[
        "frame_tables_sha256"
    ] != digest(json_bytes([rows, bodies])):
        raise ValueError("local module entry and frame tables disagree")
    rows = sorted(rows, key=lambda r: r["instructions"][0][1])
    if rows[-1]["resume_offset"] != certificate["exit_offset"] or any(
        r["code_sha256"] != certificate["code_sha256"] for r in rows
    ):
        raise ValueError("local module entry does not reach its sealed exit")
    tree = _frame_tree(rows, bodies, certificate["modules"])
    main = Decl(
        "function",
        name="main",
        type=named("int"),
        body=(
            Stmt(
                "return",
                value=call(
                    "scar::imports::run_project",
                    name("scar::imports::generated::regions"),
                    name("scar::imports::generated::modules"),
                ),
            ),
        ),
    )
    return {
        "include/scar/program.hpp": tree,
        "src/program.cpp": Unit(
            includes=(("local", "scar/program.hpp"),), declarations=(main,)
        ),
    }
