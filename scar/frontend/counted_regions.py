"""Project compiler-verified counted loops into the canonical executable region IR."""

from hashlib import sha256
import math
import re

from .gcc_ir import instruction_fields, is_ssa_name

_VERSIONED_NAME = re.compile(r"([A-Za-z_]\w*)_\d+(\(D\))?")


class _OutsideCountedProfile(Exception):
    pass


def _require(condition):
    if not condition:
        raise _OutsideCountedProfile()


def _fields(instruction):
    fields = instruction_fields(instruction)
    _require(fields is not None)
    return fields


def _active_instructions(block):
    return [
        row
        for row in block["operations"]
        if row["opcode"] not in {"gimple_label", "gimple_nop"}
    ]


_ARITHMETIC = {
    "plus_expr": "add",
    "minus_expr": "sub",
    "mult_expr": "mul",
    "rdiv_expr": "div",
}
_FLOAT_LITERAL = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?")
_FORTRAN_REF = re.compile(r"\(\*([^()]+\(D\))\)\[([^][]+)\]")


class _CountedIteration:
    """Decode one acyclic SSA loop body with an exact scalar memory interface."""

    def __init__(self, rows, induction, index, parameters, *, fortran_lower=None):
        self.induction, self.index, self.parameters = induction, index, parameters
        self.fortran_lower = fortran_lower
        self.index_value = ["index", "i32", index]
        self.definitions = {}
        self.used = set()
        self.accesses = []
        self.store = None
        for row in rows:
            _require(row["opcode"] == "gimple_assign")
            args = _fields(row)
            _require(len(args) == 5 and args[-1] == "NULL")
            destination = args[1]
            if destination.startswith("*") or (
                fortran_lower is not None and _FORTRAN_REF.fullmatch(destination)
            ):
                _require(self.store is None and row["memory"] and row["memory"][1])
                self.store = args
            else:
                _require(
                    is_ssa_name(destination) and destination not in self.definitions
                )
                self.definitions[destination] = (args, row["memory"])
        _require(self.store is not None)

    def definition(self, name):
        _require(name in self.definitions and name not in self.used)
        self.used.add(name)
        return self.definitions[name]

    def parameter(self, name, types):
        match = _VERSIONED_NAME.fullmatch(name)
        _require(
            match is not None
            and match[2] == "(D)"
            and self.parameters.get(match[1]) in types
        )
        return match[1]

    def address(self, operand, *, write=False):
        allowed = {"ptr<f64>"} if write else {"ptr<f64>", "ptr<const f64>"}
        if self.fortran_lower is not None:
            access = _FORTRAN_REF.fullmatch(operand)
            _require(access is not None)
            name = self.parameter(access[1], allowed)
            index = access[2]
            if self.fortran_lower == 1:
                offset, memory = self.definition(index)
                _require(
                    not memory
                    and offset[0] == "plus_expr"
                    and offset[3:] == ("-1", "NULL")
                )
                index = offset[2]
            cast, memory = self.definition(index)
            _require(
                not memory
                and cast[0] == "nop_expr"
                and cast[2:] == (self.induction, "NULL", "NULL")
            )
            return name
        _require(operand.startswith("*"))
        fields, memory = self.definition(operand[1:])
        _require(
            not memory and fields[0] == "pointer_plus_expr" and fields[4] == "NULL"
        )
        name = self.parameter(fields[2], allowed)
        scale, memory = self.definition(fields[3])
        _require(not memory and scale[0] == "mult_expr" and scale[3:] == ("8", "NULL"))
        index, memory = self.definition(scale[2])
        _require(
            not memory
            and index[0] == "nop_expr"
            and index[2:] == (self.induction, "NULL", "NULL")
        )
        return name

    def expression(self, value, type_):
        if type_ == "i32" and value == self.induction:
            if self.fortran_lower == 1:
                return ["add", "i32", self.index_value, ["constant", "i32", 1]]
            return self.index_value
        if type_ == "f64" and _FLOAT_LITERAL.fullmatch(value):
            number = float(value)
            _require(math.isfinite(number))
            return ["constant", type_, number]
        if type_ == "i32" and re.fullmatch(r"[+-]?\d+", value):
            number = int(value)
            _require(-(2**31) <= number < 2**31)
            return ["constant", type_, number]
        if value.endswith("(D)"):
            return ["parameter", type_, self.parameter(value, {type_})]
        fields, memory = self.definition(value)
        opcode, _, left, right, _ = fields
        load_opcode = "array_ref" if self.fortran_lower is not None else "mem_ref"
        if opcode == load_opcode and type_ == "f64":
            _require(memory and memory[1] is None and right == "NULL")
            name = self.address(left)
            self.accesses.append(
                {
                    "kind": "READ",
                    "pointer": name,
                    "index": self.index_value,
                    "type": type_,
                }
            )
            return ["load", type_, name, self.index_value]
        _require(not memory)
        if opcode == "float_expr" and type_ == "f64" and right == "NULL":
            return ["cast", type_, self.expression(left, "i32")]
        if opcode in {"ssa_name", "integer_cst", "real_cst"} and right == "NULL":
            return self.expression(left, type_)
        if (
            self.fortran_lower == 1
            and type_ == "i32"
            and (opcode, left, right) == ("plus_expr", self.induction, "-1")
        ):
            return self.index_value
        _require(
            opcode in _ARITHMETIC
            and (type_ == "f64" or opcode != "rdiv_expr")
            and right != "NULL"
        )
        return [
            _ARITHMETIC[opcode],
            type_,
            self.expression(left, type_),
            self.expression(right, type_),
        ]

    def finish(self, increment):
        fields, memory = self.definition(increment)
        _require(
            not memory
            and fields[0] == "plus_expr"
            and fields[2:] == (self.induction, "1", "NULL")
        )
        opcode, destination, source, right, _ = self.store
        _require(opcode in {"ssa_name", "real_cst"} and right == "NULL")
        name = self.address(destination, write=True)
        result = self.expression(source, "f64")
        _require(self.used == self.definitions.keys())
        self.accesses.append(
            {"kind": "WRITE", "pointer": name, "index": self.index_value, "type": "f64"}
        )
        return {
            "kind": "store",
            "pointer": name,
            "index": self.index_value,
            "value": result,
            "type": "f64",
        }


def _counted_bounds(entry, test, fortran):
    _require(
        all(row["opcode"] == "gimple_assign" and not row["memory"] for row in entry)
    )
    _require(test and test[-1]["opcode"] == "gimple_cond" and not test[-1]["memory"])
    condition = _fields(test[-1])
    _require(len(condition) == 5 and condition[3:] == ("NULL", "NULL"))
    if fortran is None:
        _require(len(entry) == 1 and condition[0] == "lt_expr")
        initial = _fields(entry[0])
        _require(
            len(initial) == 5
            and initial[0] == "integer_cst"
            and initial[2:] == ("0", "NULL", "NULL")
        )
        return initial[1], condition[1], condition[2], test[:-1]
    _require(
        len(entry) == 2
        and len(test) >= 2
        and condition[0] == "ne_expr"
        and condition[2] == "0"
        and test[-2]["opcode"] == "gimple_assign"
        and not test[-2]["memory"]
    )
    bound, initial, comparison = _fields(entry[0]), _fields(entry[1]), _fields(test[-2])
    _require(len(bound) == len(initial) == len(comparison) == 5)
    lower = fortran["induction_lower"]
    code, increment = ("ssa_name", "NULL") if lower == 1 else ("plus_expr", "-1")
    _require(
        bound[0] == code
        and bound[3:] == (increment, "NULL")
        and initial[0] == "integer_cst"
        and initial[2:] == (str(lower), "NULL", "NULL")
        and comparison[0] == "gt_expr"
        and comparison[1] == condition[1]
        and comparison[3:] == (bound[1], "NULL")
    )
    return initial[1], comparison[2], bound[2], test[:-2]


def _fortran_prototype(function, abi):
    expected = []
    for parameter in abi["parameters"]:
        base = (
            r"integer\(kind=4\)"
            if parameter["passing"] == "VALUE"
            else r"real\(kind=8\)\[0:\] \* restrict"
        )
        expected.append(f"{base} {re.escape(parameter['name'])}")
    _require(
        function["prototype"] is not None
        and re.fullmatch(", ".join(expected), function["prototype"])
    )


def _counted_body(function, loops, declaration):
    fortran = declaration.get("fortran_abi")
    if fortran is not None:
        _fortran_prototype(function, fortran)
    parameters = {row["name"]: row["type"] for row in declaration["parameters"]}
    blocks = {block["number"]: block for block in function["blocks"]}
    position = function["blocks"][0]["number"]
    body, accesses, visited = [], [], set()
    while position not in visited:
        visited.add(position)
        entry = blocks[position]
        rows = _active_instructions(entry)
        if entry["successors"] == ("EXIT",):
            _require(len(rows) == 1 and rows[0]["text"] == "gimple_return <NULL>")
            break
        _require(len(entry["successors"]) == 1 and entry["successors"][0] in loops)
        header = entry["successors"][0]
        check = blocks[header]
        _require(len(check["branches"]) == 2 and len(check["successors"]) == 2)
        repeat, exit_ = (
            check["branches"] if fortran is None else reversed(check["branches"])
        )
        _require(
            {repeat, exit_} == set(check["successors"])
            and loops[header] == {header, repeat}
            and repeat != header
            and exit_ in blocks
            and blocks[repeat]["successors"] == (header,)
            and not {header, repeat} & visited
        )
        visited.update((header, repeat))
        initial, induction, upper, phis = _counted_bounds(
            rows, _active_instructions(check), fortran
        )
        name = _VERSIONED_NAME.fullmatch(induction)
        _require(name is not None and not name[2])
        if fortran is None:
            _require(function["locals"].get(name[1]) == "int")
        else:
            _require(function["scalar_types"].get(name[1]) == "i32")
        _require(all(row["opcode"] == "gimple_phi" for row in phis))
        index_phis = [row for row in phis if row["result"] == induction]
        _require(
            len(index_phis) == 1
            and all(
                row["result"].startswith(".MEM_") or row is index_phis[0]
                for row in phis
            )
        )
        incoming = {block: value for value, block in index_phis[0]["phi"]}
        _require(
            incoming.keys() == {position, repeat} and incoming[position] == initial
        )
        lower = None if fortran is None else fortran["induction_lower"]
        lane = _CountedIteration(
            _active_instructions(blocks[repeat]),
            induction,
            name[1],
            parameters,
            fortran_lower=lower,
        )
        bound = lane.parameter(upper, {"i32"})
        if fortran is not None:
            _require(bound == fortran["bounds"]["parameter"])
        statement = lane.finish(incoming[repeat])
        accesses.extend({**access, "loop": len(body)} for access in lane.accesses)
        body.append(
            {
                "kind": "for",
                "index": name[1],
                "index_type": "i32",
                "start": ["constant", "i32", 0],
                "stop": ["parameter", "i32", bound],
                "step": 1,
                "comparison": "lt",
                "body": [statement],
            }
        )
        position = exit_
    _require(visited == blocks.keys() and len(body) == len(loops) and bool(body))
    return body, accesses


def _fortran_span(source, abi, operations, locked):
    name = re.escape(abi["procedure"].encode("ascii"))
    beginnings = list(
        re.finditer(
            rb"^[ \t]*subroutine[ \t]+" + name + rb"[ \t]*\(", source, re.M | re.I
        )
    )
    if len(beginnings) != 1:
        return None
    start = beginnings[0].start()
    tail = source[beginnings[0].end() :]
    end = re.search(
        rb"^[ \t]*end[ \t]+subroutine(?:[ \t]+(" + name + rb"))?[ \t]*(?:![^\n]*)?$",
        tail,
        re.M | re.I,
    )
    if not end or re.search(
        rb"^[ \t]*(?:contains\b|subroutine\b|end[ \t]+subroutine\b)",
        tail[: end.start()],
        re.M | re.I,
    ):
        return None
    stop = beginnings[0].end() + end.end()
    if b"#" in source[start:stop]:
        return None
    first, last = source.count(b"\n", 0, start) + 1, source.count(b"\n", 0, stop) + 1
    for operation in operations:
        site = operation.attributes["source"]
        if site is not None:
            if (
                not first <= int(site["line"]) <= last
                or site["file"]
                and site["file"] != locked.path
            ):
                return None
    return start, stop


def extract_counted_region(
    function, loops, declaration, region, operations, unit, locked, source
):
    if not declaration or not loops:
        return None
    fortran = declaration.get("fortran_abi")
    span = (
        declaration["span"]
        if fortran is None
        else _fortran_span(source, fortran, operations, locked)
    )
    if span is None:
        return None
    start, stop = span
    original = source[start:stop]
    if fortran is None and not (
        0 <= start < stop <= len(source)
        and original.startswith(b"{")
        and original.endswith(b"}")
        and not any(
            token in original
            for token in (b"#", b"%:", b"??=", b"\\\n", b"_Pragma", b"__pragma")
        )
    ):
        return None
    try:
        body, accesses = _counted_body(function, loops, declaration)
    except _OutsideCountedProfile:
        return None
    return {
        "schema": "scar.native-region",
        "schema_version": 1,
        "region": region,
        "operations": [row.id for row in operations],
        "symbol": function["symbol"],
        "parameters": declaration["parameters"],
        "return_type": "void",
        "body": body,
        "source": {
            "path": unit,
            "sha256": locked.sha256,
            "span": [start, stop],
            "slice_sha256": sha256(original).hexdigest(),
        },
        "effects": {
            "external": "NONE",
            "floating_exceptions": "PRESERVE_PROFILE",
            "accesses": accesses,
        },
        **({"fortran_abi": fortran} if fortran is not None else {}),
    }
