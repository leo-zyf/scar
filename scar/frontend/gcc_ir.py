"""Decode GCC SSA blocks and interprocedural symbol records."""

import re

_FUNCTION = re.compile(r"^;; Function (.*?) \(([^,]*), funcdef_no=.*$", re.M)
_BLOCK = re.compile(r"^;;   basic block (\d+), loop depth (\d+).*$", re.M)
_LOCATION = re.compile(
    r"\[(?:(?P<file>[^\[\]\n]+):)?(?P<line>\d+):(?P<column>\d+)(?: discrim \d+)?\]"
)
_QUOTED = re.compile(r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'""")
_GIMPLE_TEXT = re.compile(f"(?:{_QUOTED.pattern})|(?:{_LOCATION.pattern})")
_SSA = re.compile(r"(?<![\w.])(?:[\w.$]*_\d+|\.MEM_\d+)(?:\(D\))?(?![\w.])")
_GIMPLE = re.compile(r"^(gimple_\w+)\s+<(.*)>$")
_PHI = re.compile(r"# gimple_phi <([^,]+), (.*)>")
_PHI_INPUT = re.compile(r"([^, ]+)\((\d+)\)")
_VUSE = re.compile(r"# VUSE <(\.MEM_\d+(?:\(D\))?)>")
_VDEF = re.compile(r"# (\.MEM_\d+) = VDEF <(\.MEM_\d+(?:\(D\))?)>")
_IPA_SYMBOL = re.compile(r"^(.+)/(\d+) \((.*)\)(?: @.*)?$")
_IPA_ID = re.compile(r"/(\d+)(?=\s|$)")
_JUMP = re.compile(r"\bgoto <bb (\d+)>")


def parse_call_graph(raw):
    table = raw.rpartition("Initial Symbol table:")[2].strip().partition("\n\n")[0]
    symbols, current = {}, None
    for line in table.splitlines():
        if match := _IPA_SYMBOL.fullmatch(line):
            symbol, ordinal, display = match.groups()
            current = symbols[ordinal] = {
                "symbol": symbol,
                "display": display,
                "public": False,
                "definition": False,
                "calls": (),
            }
        elif current is not None:
            clause, _, value = line.strip().partition(":")
            if clause == "Calls":
                current["calls"] = tuple(_IPA_ID.findall(value))
            if clause == "Visibility":
                current["public"] = "public" in value.split()
            if clause == "Type":
                current["definition"] = {"function", "definition"} <= set(value.split())
    return symbols


def ssa_names(text):
    return tuple(dict.fromkeys(_SSA.findall(_QUOTED.sub("", text))))


def _parse_block(number, depth, raw):
    operations, successors, branches = [], [], []
    memory, memory_site, in_successor_line = (), None, False
    for line in raw.splitlines():
        if line.startswith(";;    succ:"):
            in_successor_line = True
            part = line.partition(":")[2]
        elif in_successor_line and line.startswith(";;                "):
            part = line[2:]
        else:
            in_successor_line = False
        if in_successor_line:
            successors.extend(
                re.findall(
                    r"(?<!\w)(\d+|EXIT)(?:\s|$)", re.sub(r"\[[^\]]*\]", "", part)
                )
            )
            continue
        site = None

        def capture_location(match):
            nonlocal site
            if match["line"] is None:
                return match[0]
            if site is None:
                site = match.groupdict()
            return ""

        text = _GIMPLE_TEXT.sub(capture_location, line).strip()
        if match := _VUSE.fullmatch(text):
            memory, memory_site = (match[1], None), site
        elif match := _VDEF.fullmatch(text):
            memory, memory_site = (match[2], match[1]), site
        elif match := _PHI.fullmatch(text):
            operations.append(
                {
                    "opcode": "gimple_phi",
                    "text": text,
                    "site": site,
                    "result": match[1],
                    "operands": "",
                    "phi": _PHI_INPUT.findall(match[2]),
                    "memory": (),
                }
            )
        elif match := _GIMPLE.fullmatch(text):
            opcode, operands = match.groups()
            call, result = None, None
            if opcode in {"gimple_assign", "gimple_call"}:
                fields = operands.split(", ", 2)
                if len(fields) > 1 and _SSA.fullmatch(fields[1]):
                    result = fields[1]
                if opcode == "gimple_call":
                    call = fields[0]
                    attached = fields[2] if len(fields) > 2 else ""
                    operands = (
                        ", ".join((call, attached))
                        if _SSA.fullmatch(call)
                        else attached
                    )
                elif result is not None:
                    operands = fields[2] if len(fields) > 2 else ""
            operations.append(
                {
                    "opcode": opcode,
                    "text": text,
                    "site": site or memory_site,
                    "result": result,
                    "operands": operands,
                    "call": call,
                    "memory": memory,
                }
            )
            memory, memory_site = (), None
        elif re.match(r"gimple_", text):
            operations.append(
                {
                    "opcode": "gimple_unparsed",
                    "text": text,
                    "site": site or memory_site,
                    "result": None,
                    "operands": text,
                    "memory": memory,
                }
            )
            memory, memory_site = (), None
        elif match := _JUMP.match(text):
            branches.append(match[1])
    return {
        "number": number,
        "compiler_loop_depth": depth,
        "operations": operations,
        "successors": tuple(dict.fromkeys(successors)),
        "branches": tuple(branches),
    }


def parse_functions(raw, *, read_types=None):
    functions = []
    matches = list(_FUNCTION.finditer(raw))
    for i, function in enumerate(matches):
        finish = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[function.end() : finish]
        blocks = list(_BLOCK.finditer(body))
        if not blocks:
            continue
        preamble = body[: blocks[0].start()]
        locals_ = re.findall(r"^  (int|double) ([A-Za-z_]\w*);$", preamble, re.M)
        prototype = re.search(r"^void \w+ \((.*)\)$", preamble, re.M)
        functions.append(
            {
                "name": function[1],
                "symbol": function[2],
                "locals": {name: type_ for type_, name in locals_},
                "prototype": prototype[1] if prototype else None,
                "scalar_types": {},
                **(read_types(preamble) if read_types else {}),
                "blocks": [
                    _parse_block(
                        block[1],
                        int(block[2]),
                        body[
                            block.end() : (
                                blocks[j + 1].start()
                                if j + 1 < len(blocks)
                                else len(body)
                            )
                        ],
                    )
                    for j, block in enumerate(blocks)
                ],
            }
        )
    return functions


def is_ssa_name(name):
    return _SSA.fullmatch(name) is not None


def instruction_fields(instruction):
    match = _GIMPLE.fullmatch(instruction["text"])
    return tuple(field.strip() for field in match[2].split(",")) if match else None
