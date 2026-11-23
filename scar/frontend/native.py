"""Native front end: checked inputs, compiler evidence, and SSA graph import."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import Edge, Operation, Region, artifact, copy_artifact, read_artifact, verify_artifact, write_json

from .clang import capture_unit, counted_function_bodies, gnu_cfi_header, gnu_cfi_overlay, typed_call_occurrences


_COMPILER = {"c": "gcc", "c++": "g++", "fortran": "gfortran"}
_FUNCTION = re.compile(r"^;; Function (.*?) \(([^,]*), funcdef_no=.*$", re.M)
_BLOCK = re.compile(r"^;;   basic block (\d+), loop depth (\d+).*$", re.M)
_LOCATION = re.compile(r"\[(?:(?P<file>[^\[\]\n]+):)?(?P<line>\d+):(?P<column>\d+)(?: discrim \d+)?\]")
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
_VERSIONED_NAME = re.compile(r"([A-Za-z_]\w*)_\d+(\(D\))?")
_LOOSE_FLOAT = {"-Ofast", "-ffast-math", "-funsafe-math-optimizations", "-fassociative-math",
                "-ffinite-math-only", "-freciprocal-math", "-fno-signed-zeros", "-fno-trapping-math"}
_FORTRAN_SCALARS = {
    "integer(kind=4)": "i32", "integer(kind=8)": "i64",
    "real(kind=4)": "f32", "real(kind=8)": "f64",
    "logical(kind=4)": "fortran-logical32",
}
_FORTRAN_DECL = re.compile(
    r"^  (?:static )?((?:integer|real|logical)\(kind=[48]\)) ([A-Za-z_][\w.]*)\s*(?:= .*?)?;$", re.M)


def _lock_sources(target):
    if (target.get("schema"), target.get("schema_version")) != ("scar.native-entry", 1):
        raise ValueError("native input requires scar.native-entry v1")
    if target["language"] not in {*_COMPILER, "mixed"}:
        raise ValueError(f"unsupported native language {target['language']}")
    root = Path(target["source_root"]).resolve(strict=True)
    target["source_root"] = str(root)
    locked = {}
    for name, expected in target["source_lock"]["files"].items():
        relative = Path(name)
        file = (root / relative).resolve(strict=True)
        if relative.is_absolute() or relative.as_posix() != name or not file.is_relative_to(root):
            raise ValueError(f"invalid locked path: {name}")
        found = artifact(file)
        if (found.sha256, found.size) != (expected["sha256"], expected["size_bytes"]):
            raise ValueError(f"locked source changed: {name}")
        locked[name] = found
    units = target["units"]
    names = [unit["path"] for unit in units]
    if not names or len(names) != len(set(names)) or any(name not in locked for name in names):
        raise ValueError("native manifest requires unique locked translation units")
    for unit in units:
        language = unit.get("language", target["language"])
        if language not in _COMPILER:
            raise ValueError(f"no language profile for {unit['path']}")
        if not isinstance(unit["flags"], list) or any(not isinstance(flag, str) for flag in unit["flags"]):
            raise ValueError(f"invalid compiler flags for {unit['path']}")
    return locked


def _lock_cases(target, out):
    baseline = target.get("baseline")
    if not baseline:
        return ()
    cwd = Path(baseline["cwd"]).resolve(strict=True)
    baseline["cwd"] = str(cwd)
    argv = baseline["argv"]
    command = Path(argv[0])
    if not command.is_absolute():
        command = (cwd / command) if "/" in argv[0] else shutil.which(argv[0])
    receipts = [_record_input(baseline, "receipt", command, out / "original", executable=True)]
    for index, case in enumerate(target.get("cases", ())):
        if case.get("stdin") is not None:
            receipts.append(_record_input(case, "input_receipt", cwd / case["stdin"], out / f"case-{index}.stdin"))
    return receipts


def _record_input(row, field, path, destination, *, executable=False):
    if path is None:
        raise ValueError(f"could not resolve native case {field}")
    reference = artifact(path)
    if field in row and row[field] != asdict(reference):
        raise ValueError(f"native case {field} changed: {reference.path}")
    retained = copy_artifact(reference, destination, executable=executable)
    row[field] = asdict(retained)
    return retained


def _compiler(target, unit):
    language = unit.get("language", target["language"])
    config = target.get("fortran", {}) if language == "fortran" else {}
    name = unit.get("compiler") or config.get("gfortran") or _COMPILER[language]
    executable = shutil.which(name)
    if not executable:
        raise ValueError(f"compiler not found: {name}")
    # GCC and G++ infer their default language from the command name.
    return str(Path(executable).absolute())


def _capture(target, unit, compiler, out, scratch, cfi):
    root = Path(target["source_root"])
    language = unit.get("language", target["language"])
    ssa, ipa = out / "body.ssa", out / "symbols.cgraph"
    flags = [*unit["flags"], *unit.get("gcc_flags", ())]
    extra = (["-J", str(scratch), "-I", str(scratch), "-O0", "-fdump-fortran-original"]
             if language == "fortran" else [])
    command = [compiler, *flags, *extra,
               f"-fdump-tree-ssa-raw-blocks-lineno-vops={ssa}", f"-fdump-ipa-cgraph={ipa}",
               "-c", str(root / unit["path"]), "-o", str(scratch / f"{out.name}.o")]
    result = subprocess.run(command, cwd=root, capture_output=True, timeout=180, check=False)
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"compiler capture failed for {unit['path']}: {message}")
    evidence = {"unit": unit["path"], "language": language, "command": command}
    if result.stderr:
        (out / "diagnostics.txt").write_bytes(result.stderr)
        evidence["diagnostics"] = asdict(artifact(out / "diagnostics.txt"))
    if language == "fortran":
        typed_path = out / "types.fortran-original"
        typed_path.write_bytes(result.stdout)
        typed = artifact(typed_path)
        bodies, call_sites = {}, {}
    else:
        document = capture_unit(root, unit, cfi=cfi)
        bodies = {} if _LOOSE_FLOAT.intersection(flags) else counted_function_bodies(document)
        call_sites = typed_call_occurrences(document)
        typed = write_json(out / "types.json", document)
    evidence["types"] = asdict(typed)
    functions = []
    if ssa.is_file():
        evidence["ssa"] = asdict(artifact(ssa))
        functions = _functions(ssa.read_text(errors="replace"))
    if ipa.is_file():
        evidence["symbols"] = asdict(artifact(ipa))
        symbols = _symbols(ipa.read_text(errors="replace"))
    else:
        symbols = {}
    return evidence, functions, symbols, bodies, call_sites


def _symbols(raw):
    table = raw.rpartition("Initial Symbol table:")[2].strip().partition("\n\n")[0]
    symbols, current = {}, None
    for line in table.splitlines():
        if match := _IPA_SYMBOL.fullmatch(line):
            symbol, ordinal, display = match.groups()
            current = symbols[ordinal] = {"symbol": symbol, "display": display,
                                          "public": False, "definition": False, "calls": ()}
        elif current is not None:
            clause, _, value = line.strip().partition(":")
            if clause == "Calls":
                current["calls"] = tuple(_IPA_ID.findall(value))
            if clause == "Visibility":
                current["public"] = "public" in value.split()
            if clause == "Type":
                current["definition"] = {"function", "definition"} <= set(value.split())
    return symbols


def _names(text):
    return tuple(dict.fromkeys(_SSA.findall(_QUOTED.sub("", text))))


def _block(number, depth, raw):
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
            successors.extend(re.findall(r"(?<!\w)(\d+|EXIT)(?:\s|$)", re.sub(r"\[[^\]]*\]", "", part)))
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
            operations.append({"opcode": "gimple_phi", "text": text, "site": site, "result": match[1],
                               "operands": "", "phi": _PHI_INPUT.findall(match[2]), "memory": ()})
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
                    operands = ", ".join((call, attached)) if _SSA.fullmatch(call) else attached
                elif result is not None:
                    operands = fields[2] if len(fields) > 2 else ""
            operations.append({"opcode": opcode, "text": text, "site": site or memory_site, "result": result,
                               "operands": operands, "call": call, "memory": memory})
            memory, memory_site = (), None
        elif re.match(r"gimple_", text):
            operations.append({"opcode": "gimple_unparsed", "text": text, "site": site or memory_site,
                               "result": None, "operands": text, "memory": memory})
            memory, memory_site = (), None
        elif match := _JUMP.match(text):
            branches.append(match[1])
    return {"number": number, "compiler_loop_depth": depth, "operations": operations,
            "successors": tuple(dict.fromkeys(successors)), "branches": tuple(branches)}


def _functions(raw):
    functions = []
    matches = list(_FUNCTION.finditer(raw))
    for i, function in enumerate(matches):
        finish = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[function.end():finish]
        blocks = list(_BLOCK.finditer(body))
        if not blocks:
            continue
        preamble = body[:blocks[0].start()]
        locals_ = re.findall(r"^  (int|double) ([A-Za-z_]\w*);$", preamble, re.M)
        functions.append({"name": function[1], "symbol": function[2], "locals": {name: type_ for type_, name in locals_},
                          "scalar_types": {name: _FORTRAN_SCALARS[type_] for type_, name in _FORTRAN_DECL.findall(preamble)
                                           if type_ in _FORTRAN_SCALARS},
                          "blocks": [
            _block(block[1], int(block[2]), body[block.end():blocks[j + 1].start() if j + 1 < len(blocks) else len(body)])
            for j, block in enumerate(blocks)
        ]})
    return functions


def _natural_loops(blocks):
    numbers = tuple(block["number"] for block in blocks)
    known = set(numbers)
    successors = {block["number"]: tuple(target for target in block["successors"] if target in known)
                  for block in blocks}
    predecessors = defaultdict(set)
    for source, targets in successors.items():
        for target in targets:
            predecessors[target].add(source)
    entry = numbers[0]
    reached, stack = {entry}, [entry]
    while stack:
        for target in successors[stack.pop()]:
            if target not in reached:
                reached.add(target)
                stack.append(target)
    dominators = {key: set(reached) for key in reached}
    dominators[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for key in numbers:
            if key == entry or key not in reached:
                continue
            incoming = [dominators[source] for source in predecessors[key] if source in reached]
            if incoming:
                current = {key} | set.intersection(*incoming)
                if current != dominators[key]:
                    dominators[key] = current
                    changed = True
    loops, backedges = {}, set()
    for source in numbers:
        for target in successors[source]:
            if target not in dominators.get(source, ()):
                continue
            backedges.add((source, target))
            members = loops.setdefault(target, {target})
            pending = [source]
            while pending:
                key = pending.pop()
                if key in members or key not in reached:
                    continue
                members.add(key)
                pending.extend(predecessors[key] - members)
    return loops, backedges


class _OutsideCountedProfile(Exception):
    pass


def _require(condition):
    if not condition:
        raise _OutsideCountedProfile()


def _fields(instruction):
    match = _GIMPLE.fullmatch(instruction["text"])
    _require(match is not None)
    return tuple(field.strip() for field in match[2].split(","))


def _active_instructions(block):
    return [row for row in block["operations"] if row["opcode"] not in {"gimple_label", "gimple_nop"}]


_ARITHMETIC = {"plus_expr": "add", "minus_expr": "sub", "mult_expr": "mul", "rdiv_expr": "div"}
_FLOAT_LITERAL = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?")


class _CountedLane:
    """Decode one acyclic SSA loop body with an exact scalar memory interface."""

    def __init__(self, rows, induction, index, parameters):
        self.induction, self.index, self.parameters = induction, index, parameters
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
            if destination.startswith("*"):
                _require(self.store is None and row["memory"] and row["memory"][1])
                self.store = args
            else:
                _require(_SSA.fullmatch(destination) is not None and destination not in self.definitions)
                self.definitions[destination] = (args, row["memory"])
        _require(self.store is not None)

    def definition(self, name):
        _require(name in self.definitions and name not in self.used)
        self.used.add(name)
        return self.definitions[name]

    def parameter(self, name, types):
        match = _VERSIONED_NAME.fullmatch(name)
        _require(match is not None and match[2] == "(D)" and self.parameters.get(match[1]) in types)
        return match[1]

    def address(self, operand, *, write=False):
        _require(operand.startswith("*"))
        fields, memory = self.definition(operand[1:])
        _require(not memory and fields[0] == "pointer_plus_expr" and fields[4] == "NULL")
        allowed = {"ptr<f64>"} if write else {"ptr<f64>", "ptr<const f64>"}
        name = self.parameter(fields[2], allowed)
        scale, memory = self.definition(fields[3])
        _require(not memory and scale[0] == "mult_expr" and scale[3:] == ("8", "NULL"))
        index, memory = self.definition(scale[2])
        _require(not memory and index[0] == "nop_expr" and index[2:] == (self.induction, "NULL", "NULL"))
        return name

    def expression(self, value, type_):
        if type_ == "i32" and value == self.induction:
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
        if opcode == "mem_ref" and type_ == "f64":
            _require(memory and memory[1] is None and right == "NULL")
            name = self.address(left)
            self.accesses.append({"kind": "READ", "pointer": name, "index": self.index_value, "type": type_})
            return ["load", type_, name, self.index_value]
        _require(not memory)
        if opcode == "float_expr" and type_ == "f64" and right == "NULL":
            return ["cast", type_, self.expression(left, "i32")]
        if opcode in {"ssa_name", "integer_cst", "real_cst"} and right == "NULL":
            return self.expression(left, type_)
        _require(opcode in _ARITHMETIC and (type_ == "f64" or opcode != "rdiv_expr") and right != "NULL")
        return [_ARITHMETIC[opcode], type_, self.expression(left, type_), self.expression(right, type_)]

    def finish(self, increment):
        fields, memory = self.definition(increment)
        _require(not memory and fields[0] == "plus_expr" and fields[2:] == (self.induction, "1", "NULL"))
        opcode, destination, source, right, _ = self.store
        _require(opcode in {"ssa_name", "real_cst"} and right == "NULL")
        name = self.address(destination, write=True)
        result = self.expression(source, "f64")
        _require(self.used == self.definitions.keys())
        self.accesses.append({"kind": "WRITE", "pointer": name, "index": self.index_value, "type": "f64"})
        return {"kind": "store", "pointer": name, "index": self.index_value, "value": result, "type": "f64"}


def _counted_body(function, loops, declaration):
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
        _require(len(rows) == 1 and rows[0]["opcode"] == "gimple_assign" and not rows[0]["memory"])
        initial = _fields(rows[0])
        _require(len(initial) == 5 and initial[0] == "integer_cst" and initial[2:] == ("0", "NULL", "NULL"))
        _require(len(entry["successors"]) == 1 and entry["successors"][0] in loops)
        header = entry["successors"][0]
        check = blocks[header]
        _require(len(check["branches"]) == 2 and len(check["successors"]) == 2)
        repeat, exit_ = check["branches"]
        _require({repeat, exit_} == set(check["successors"]) and loops[header] == {header, repeat}
                 and repeat != header and exit_ in blocks and blocks[repeat]["successors"] == (header,)
                 and not {header, repeat} & visited)
        visited.update((header, repeat))
        test = _active_instructions(check)
        _require(test and test[-1]["opcode"] == "gimple_cond" and not test[-1]["memory"])
        condition = _fields(test[-1])
        _require(len(condition) == 5 and condition[0] == "lt_expr" and condition[3:] == ("NULL", "NULL"))
        _, induction, upper, _, _ = condition
        name = _VERSIONED_NAME.fullmatch(induction)
        _require(name is not None and not name[2] and function["locals"].get(name[1]) == "int")
        phis = [row for row in test[:-1] if row["opcode"] == "gimple_phi"]
        _require(len(phis) == len(test) - 1)
        index_phis = [row for row in phis if row["result"] == induction]
        _require(len(index_phis) == 1 and all(row["result"].startswith(".MEM_") or row is index_phis[0]
                                              for row in phis))
        incoming = {block: value for value, block in index_phis[0]["phi"]}
        _require(incoming.keys() == {position, repeat} and incoming[position] == initial[1])
        lane = _CountedLane(_active_instructions(blocks[repeat]), induction, name[1], parameters)
        bound = lane.parameter(upper, {"i32"})
        statement = lane.finish(incoming[repeat])
        accesses.extend({**access, "loop": len(body)} for access in lane.accesses)
        body.append({"kind": "for", "index": name[1], "index_type": "i32", "start": ["constant", "i32", 0],
                     "stop": ["parameter", "i32", bound], "step": 1, "comparison": "lt", "body": [statement]})
        position = exit_
    _require(visited == blocks.keys() and len(body) == len(loops) and bool(body))
    return body, accesses


def _counted_region(function, loops, declaration, region, operations, unit, locked, source):
    if not declaration or not loops:
        return None
    start, stop = declaration["span"]
    original = source[start:stop]
    if not (0 <= start < stop <= len(source) and original.startswith(b"{") and original.endswith(b"}")
            and not any(token in original for token in (b"#", b"%:", b"??=", b"\\\n", b"_Pragma", b"__pragma"))):
        return None
    try:
        body, accesses = _counted_body(function, loops, declaration)
    except _OutsideCountedProfile:
        return None
    return {"schema": "scar.native-region", "schema_version": 1, "region": region,
            "operations": [row.id for row in operations], "symbol": function["symbol"],
            "parameters": declaration["parameters"], "return_type": "void", "body": body,
            "source": {"path": unit, "sha256": locked.sha256, "span": [start, stop],
                       "slice_sha256": sha256(original).hexdigest()},
            "effects": {"external": "NONE", "floating_exceptions": "PRESERVE_PROFILE", "accesses": accesses}}


_KINDS = {"gimple_assign": "ASSIGN", "gimple_call": "CALL", "gimple_return": "RETURN",
          "gimple_phi": "PHI", "gimple_cond": "BRANCH", "gimple_switch": "BRANCH",
          "gimple_goto": "BRANCH", "gimple_label": "LABEL", "gimple_nop": "NOP"}


def _function_graph(function, unit, source_lock, loops, backedges):
    key = json.dumps([unit, function["symbol"], source_lock.sha256], separators=(",", ":"))
    region = f"fn:{sha256(key.encode()).hexdigest()[:24]}"
    blocks = function["blocks"]
    children = defaultdict(list)
    loop_ids = {header: f"{region}:loop:{header}" for header in loops}
    for header, members in loops.items():
        parents = [(len(other), int(candidate), candidate) for candidate, other in loops.items() if members < other]
        children[loop_ids[min(parents)[2]] if parents else region].append(loop_ids[header])
    regions, operations, edges, calls, uses = [], [], [], [], []
    definitions, boundaries = {}, {}
    scalar_types = function["scalar_types"]
    for block in blocks:
        number = block["number"]
        identity = f"{region}:bb:{number}"
        owners = [(len(members), int(header), header) for header, members in loops.items() if number in members]
        children[loop_ids[min(owners)[2]] if owners else region].append(identity)
        block_ops = []
        for index, instruction in enumerate(block["operations"]):
            op = f"{identity}:op:{index}"
            opcode = instruction["opcode"]
            memory = instruction["memory"]
            phi = instruction.get("phi", [])
            names = [*(_names(instruction["operands"])), *(name for name, _ in phi)]
            if memory:
                names.append(memory[0])
            names = tuple(dict.fromkeys(names))
            output = ([instruction["result"]] if instruction["result"] else [])
            if memory and memory[1]:
                output.append(memory[1])
            for name in output:
                definitions[name] = op
            for name in names:
                uses.append((op, number, name, {pred for value, pred in phi if value == name}))
            effects = ["UNKNOWN"] if opcode not in {"gimple_phi", "gimple_label", "gimple_nop"} else []
            if memory:
                effects.append("STATE_READ")
            if memory and memory[1]:
                effects.append("STATE_WRITE")
            kind = _KINDS.get(opcode, "UNKNOWN")
            attributes = {"unit": unit, "symbol": function["symbol"], "opcode": opcode,
                          "gimple": instruction["text"], "source": instruction["site"]}
            if scalar_types:
                typed = {}
                for name in (*names, *output):
                    value = name.removesuffix("(D)")
                    origin = value if value in scalar_types else value.rpartition("_")[0]
                    if origin in scalar_types:
                        typed[f"{region}:{name}"] = scalar_types[origin]
                if typed:
                    attributes["compiler_scalar_types"] = typed
            if phi:
                attributes["incoming"] = [{"value": f"{region}:{name}", "block": f"{region}:bb:{pred}"}
                                          for name, pred in phi]
            if instruction.get("call") is not None:
                attributes["callee"] = instruction["call"]
                calls.append((op, instruction["call"], instruction["site"], memory))
            operations.append(Operation(op, identity, kind, opcode.removeprefix("gimple_"),
                                        tuple(f"{region}:{name}" for name in names),
                                        tuple(f"{region}:{name}" for name in output),
                                        tuple(effects), attributes))
            if block_ops:
                edges.append(Edge(block_ops[-1], op, "CONTROL"))
            block_ops.append(op)
        regions.append(Region(identity, "BLOCK", f"basic block {number}", tuple(block_ops)))
        boundaries[number] = (block_ops[0] if block_ops else identity, block_ops[-1] if block_ops else identity)
    for block in blocks:
        number = block["number"]
        for target in block["successors"]:
            if target in boundaries:
                kind = "CONTROL_BACKEDGE" if (number, target) in backedges else "CONTROL"
                edges.append(Edge(boundaries[number][1], boundaries[target][0], kind))
    for op, block, name, predecessors in uses:
        producer = definitions.get(name)
        if producer is not None:
            kind = "DATA_BACKEDGE" if any((source, block) in backedges for source in predecessors) else "DATA"
            edges.append(Edge(producer, op, kind, f"{region}:{name}"))
    for header, identity in loop_ids.items():
        regions.append(Region(identity, "LOOP", f"loop headed by basic block {header}",
                              children=tuple(children[identity])))
    regions.append(Region(region, "FUNCTION", function["name"], children=tuple(children[region])))
    return region, regions, operations, edges, calls


def _graph(target, locked, captured):
    regions, operations, edges, records, pending, counted = [], [], [], [], [], []
    definitions, local = defaultdict(list), {}
    root = Path(target["source_root"])
    for (_, functions, symbols, typed_bodies, call_sites), unit in zip(captured, target["units"]):
        name = unit["path"]
        declarations = {row["symbol"]: row for row in symbols.values() if row["definition"]}
        module_children = []
        source = None
        for body in functions:
            loops, backedges = _natural_loops(body["blocks"])
            identity, rs, ops, es, calls = _function_graph(body, name, locked[name], loops, backedges)
            regions.extend(rs)
            operations.extend(ops)
            edges.extend(es)
            module_children.append(identity)
            symbol = body["symbol"]
            if loops and symbol in typed_bodies:
                if source is None:
                    source = read_artifact(locked[name])
                row = _counted_region(body, loops, typed_bodies[symbol], identity, ops, name, locked[name], source)
                if row is not None:
                    counted.append(row)
            local[name, symbol] = identity
            owner = declarations.get(symbol)
            if owner is not None and owner["public"]:
                definitions[symbol].append(identity)
            rows = tuple(symbols[ordinal] for ordinal in owner["calls"] if ordinal in symbols) if owner else ()
            pending.extend((operation, name, symbol, callee, site, memory, rows, call_sites)
                           for operation, callee, site, memory in calls)
        regions.append(Region(f"tu:{name}", "MODULE", name, children=tuple(module_children)))
    for operation, unit, caller, callee, site, memory, rows, call_sites in pending:
        candidates = {row["symbol"] for row in rows if callee in (row["symbol"], row["display"])}
        typed = defaultdict(set)
        if site is not None:
            file = root / (site["file"] or unit)
            if file.is_relative_to(root):
                call_site = file.relative_to(root).as_posix(), int(site["line"]), callee
                column = int(site["column"])
                for symbol, usr, type_, begin, end in call_sites.get(call_site, ()):
                    if begin <= column and (end is None or column < end):
                        typed[symbol].add((usr, type_))
        authorities = ["GCC_IPA"]
        if len(candidates) != 1 and typed:
            observed = {row["symbol"] for row in rows}
            bounded = set(typed).intersection(observed, candidates or observed)
            if len(bounded) == 1:
                candidates = bounded
                authorities.append("CLANG_TYPED_OCCURRENCE")
        symbol = next(iter(candidates)) if len(candidates) == 1 else None
        target_id = None
        if symbol is not None:
            owners = definitions.get(symbol, ())
            target_id = local.get((unit, symbol)) or (owners[0] if len(owners) == 1 else None)
            status = "SOURCE_DEFINITION" if target_id else "AMBIGUOUS_DEFINITION" if owners else "EXTERNAL"
        elif len(candidates) > 1:
            status = "AMBIGUOUS_SYMBOL"
        elif _SSA.fullmatch(callee) or callee.startswith(("OBJ_TYPE_REF", "*")):
            status = "INDIRECT"
        else:
            status = "UNBOUND_SYMBOL"
        if target_id:
            edges.append(Edge(operation, target_id, "CALL", symbol))
        record = {"operation": operation, "unit": unit, "caller": caller, "gimple_target": callee,
                  "symbol": symbol, "target": target_id, "resolution": status, "source": site,
                  "authority": authorities if symbol else [],
                  "effects": {"external": "UNKNOWN",
                              "memory_ssa": "DEFINITION" if memory and memory[1] else "USE" if memory else "UNOBSERVED"}}
        if len(typed.get(symbol, ())) == 1:
            usr, type_ = next(iter(typed[symbol]))
            record["declaration"] = {"usr": usr, "type": type_}
        records.append(record)
    return tuple(regions), tuple(operations), tuple(dict.fromkeys(edges)), records, counted


def build_native(target, reference, out, *, jobs):
    locked = _lock_sources(target)
    case_receipts = _lock_cases(target, out / "cases")
    units = target["units"]
    compilers = [_compiler(target, unit) for unit in units]
    fortran = [index for index, unit in enumerate(units)
               if unit.get("language", target["language"]) == "fortran"]
    header = gnu_cfi_header(compilers[fortran[0]]) if fortran else None
    header_ref = artifact(header) if header else None
    captured = [None] * len(units)
    toolchains = {}
    for compiler in dict.fromkeys(compilers):
        version = subprocess.check_output([compiler, "--version"], text=True, timeout=20).splitlines()[0]
        toolchains[compiler] = version
    with TemporaryDirectory(prefix="scar-native-front-") as temporary, gnu_cfi_overlay(header) as cfi:
        scratch = Path(temporary)

        def run(index):
            folder = out / f"unit-{index:04d}"
            folder.mkdir()
            return _capture(target, units[index], compilers[index], folder, scratch, cfi)

        parallel = [index for index in range(len(units)) if index not in fortran]
        if parallel:
            with ThreadPoolExecutor(max_workers=min(jobs, len(parallel))) as pool:
                for index, result in zip(parallel, pool.map(run, parallel)):
                    captured[index] = result
        for index in fortran:
            captured[index] = run(index)
    regions, operations, edges, calls, counted = _graph(target, locked, captured)
    units_evidence = [item[0] for item in captured]
    evidence = {
        "input": asdict(reference),
        "locked_sources": {name: asdict(receipt) for name, receipt in locked.items()},
        "units": units_evidence, "compiler_types": {unit["unit"]: unit["types"] for unit in units_evidence},
        "toolchains": toolchains, "native_calls": calls,
        "coverage": {"source_resolution": "COMPILER_TYPED_SOURCE_AND_SSA", "runtime": "NOT_CAPTURED"},
    }
    if header_ref:
        evidence["gnu_cfi_header"] = asdict(header_ref)
    from .native_units import prepare_native_units

    try:
        material, typed_regions = prepare_native_units(target, evidence, out / "native-units", regions=counted)
        evidence["native_units"] = asdict(material)
        evidence["native_regions"] = typed_regions
    except (OSError, ValueError) as error:
        evidence["native_units_frontier"] = str(error)
    for receipt in (*locked.values(), *case_receipts, reference):
        verify_artifact(receipt)
    if header_ref:
        read_artifact(header_ref)
    return target, regions, operations, edges, evidence
