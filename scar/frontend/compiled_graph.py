"""Build a linked graph for native translation units from compiler evidence."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import (
    Edge,
    Graph,
    GraphProgram,
    NativeProgram,
    Operation,
    Region,
    artifact,
    copy_artifact,
    local_member,
    read_artifact,
    verify_artifact,
)

from .clang import gnu_cfi_overlay
from .gnu_fortran import gnu_cfi_header
from .compiler_capture import COMPILED_LANGUAGES, capture_unit_evidence, select_compiler
from .counted_regions import extract_counted_region
from .gcc_ir import is_ssa_name, ssa_names


def _lock_sources(target):
    if (target.get("schema"), target.get("schema_version")) != ("scar.native-entry", 1):
        raise ValueError("native input requires scar.native-entry v1")
    if target["language"] not in {*COMPILED_LANGUAGES, "mixed"}:
        raise ValueError(f"unsupported native language {target['language']}")
    root = Path(target["source_root"]).resolve(strict=True)
    target["source_root"] = str(root)
    locked = {}
    for name, expected in target["source_lock"]["files"].items():
        relative = Path(name)
        file = (root / relative).resolve(strict=True)
        if (
            relative.is_absolute()
            or relative.as_posix() != name
            or not file.is_relative_to(root)
        ):
            raise ValueError(f"invalid locked path: {name}")
        found = artifact(file)
        if (found.sha256, found.size) != (expected["sha256"], expected["size_bytes"]):
            raise ValueError(f"locked source changed: {name}")
        locked[name] = found
    units = target["units"]
    names = [unit["path"] for unit in units]
    if (
        not names
        or len(names) != len(set(names))
        or any(name not in locked for name in names)
    ):
        raise ValueError("native manifest requires unique locked translation units")
    for unit in units:
        language = unit.get("language", target["language"])
        if language not in COMPILED_LANGUAGES:
            raise ValueError(f"no language profile for {unit['path']}")
        if not isinstance(unit["flags"], list) or any(
            not isinstance(flag, str) for flag in unit["flags"]
        ):
            raise ValueError(f"invalid compiler flags for {unit['path']}")
    return locked


def _lock_cases(target, out, bundle):
    baseline = target.get("baseline")
    if not baseline:
        return ()
    cwd = Path(baseline["cwd"]).resolve(strict=True)
    baseline["cwd"] = str(cwd)
    argv = baseline["argv"]
    command = Path(argv[0])
    if not command.is_absolute():
        command = (cwd / command) if "/" in argv[0] else shutil.which(argv[0])
    receipts = [
        _record_input(
            baseline, "receipt", command, out / "original", bundle, executable=True
        )
    ]
    for index, case in enumerate(target.get("cases", ())):
        if case.get("stdin") is not None:
            receipts.append(
                _record_input(
                    case,
                    "input_receipt",
                    cwd / case["stdin"],
                    out / f"case-{index}.stdin",
                    bundle,
                )
            )
    return receipts


def _record_input(row, field, path, destination, bundle, *, executable=False):
    if path is None:
        raise ValueError(f"could not resolve native case {field}")
    reference = artifact(path)
    if field in row and row[field] != asdict(reference):
        raise ValueError(f"native case {field} changed: {reference.path}")
    retained = copy_artifact(reference, destination, executable=executable)
    row[field] = asdict(local_member(retained, bundle))
    return retained


def _natural_loops(blocks):
    numbers = tuple(block["number"] for block in blocks)
    known = set(numbers)
    successors = {
        block["number"]: tuple(
            target for target in block["successors"] if target in known
        )
        for block in blocks
    }
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
            incoming = [
                dominators[source] for source in predecessors[key] if source in reached
            ]
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


_KINDS = {
    "gimple_assign": "ASSIGN",
    "gimple_call": "CALL",
    "gimple_return": "RETURN",
    "gimple_phi": "PHI",
    "gimple_cond": "BRANCH",
    "gimple_switch": "BRANCH",
    "gimple_goto": "BRANCH",
    "gimple_label": "LABEL",
    "gimple_nop": "NOP",
}


def _build_function_graph(function, unit, source_lock, loops, backedges):
    key = json.dumps(
        [unit, function["symbol"], source_lock.sha256], separators=(",", ":")
    )
    region = f"fn:{sha256(key.encode()).hexdigest()[:24]}"
    blocks = function["blocks"]
    children = defaultdict(list)
    loop_ids = {header: f"{region}:loop:{header}" for header in loops}
    for header, members in loops.items():
        parents = [
            (len(other), int(candidate), candidate)
            for candidate, other in loops.items()
            if members < other
        ]
        children[loop_ids[min(parents)[2]] if parents else region].append(
            loop_ids[header]
        )
    regions, operations, edges, calls, uses = [], [], [], [], []
    definitions, boundaries = {}, {}
    scalar_types = function["scalar_types"]
    for block in blocks:
        number = block["number"]
        identity = f"{region}:bb:{number}"
        owners = [
            (len(members), int(header), header)
            for header, members in loops.items()
            if number in members
        ]
        children[loop_ids[min(owners)[2]] if owners else region].append(identity)
        block_ops = []
        for index, instruction in enumerate(block["operations"]):
            op = f"{identity}:op:{index}"
            opcode = instruction["opcode"]
            memory = instruction["memory"]
            phi = instruction.get("phi", [])
            names = [*(ssa_names(instruction["operands"])), *(name for name, _ in phi)]
            if memory:
                names.append(memory[0])
            names = tuple(dict.fromkeys(names))
            output = [instruction["result"]] if instruction["result"] else []
            if memory and memory[1]:
                output.append(memory[1])
            for name in output:
                definitions[name] = op
            for name in names:
                uses.append(
                    (op, number, name, {pred for value, pred in phi if value == name})
                )
            effects = (
                ["UNKNOWN"]
                if opcode not in {"gimple_phi", "gimple_label", "gimple_nop"}
                else []
            )
            if memory:
                effects.append("STATE_READ")
            if memory and memory[1]:
                effects.append("STATE_WRITE")
            kind = _KINDS.get(opcode, "UNKNOWN")
            attributes = {
                "unit": unit,
                "symbol": function["symbol"],
                "opcode": opcode,
                "gimple": instruction["text"],
                "source": instruction["site"],
            }
            if scalar_types:
                typed = {}
                for name in (*names, *output):
                    value = name.removesuffix("(D)")
                    origin = (
                        value if value in scalar_types else value.rpartition("_")[0]
                    )
                    if origin in scalar_types:
                        typed[f"{region}:{name}"] = scalar_types[origin]
                if typed:
                    attributes["compiler_scalar_types"] = typed
            if phi:
                attributes["incoming"] = [
                    {"value": f"{region}:{name}", "block": f"{region}:bb:{pred}"}
                    for name, pred in phi
                ]
            if instruction.get("call") is not None:
                attributes["callee"] = instruction["call"]
                calls.append((op, instruction["call"], instruction["site"], memory))
            operations.append(
                Operation(
                    op,
                    identity,
                    kind,
                    opcode.removeprefix("gimple_"),
                    tuple(f"{region}:{name}" for name in names),
                    tuple(f"{region}:{name}" for name in output),
                    tuple(effects),
                    attributes,
                )
            )
            if block_ops:
                edges.append(Edge(block_ops[-1], op, "CONTROL"))
            block_ops.append(op)
        regions.append(
            Region(identity, "BLOCK", f"basic block {number}", tuple(block_ops))
        )
        boundaries[number] = (
            block_ops[0] if block_ops else identity,
            block_ops[-1] if block_ops else identity,
        )
    for block in blocks:
        number = block["number"]
        for target in block["successors"]:
            if target in boundaries:
                kind = (
                    "CONTROL_BACKEDGE" if (number, target) in backedges else "CONTROL"
                )
                edges.append(Edge(boundaries[number][1], boundaries[target][0], kind))
    for op, block, name, predecessors in uses:
        producer = definitions.get(name)
        if producer is not None:
            kind = (
                "DATA_BACKEDGE"
                if any((source, block) in backedges for source in predecessors)
                else "DATA"
            )
            edges.append(Edge(producer, op, kind, f"{region}:{name}"))
    for header, identity in loop_ids.items():
        regions.append(
            Region(
                identity,
                "LOOP",
                f"loop headed by basic block {header}",
                children=tuple(children[identity]),
            )
        )
    regions.append(
        Region(region, "FUNCTION", function["name"], children=tuple(children[region]))
    )
    return region, regions, operations, edges, calls


def _link_translation_units(target, locked, captured):
    regions, operations, edges, records, pending, counted = [], [], [], [], [], []
    definitions, local = defaultdict(list), {}
    root = Path(target["source_root"])
    for unit_evidence, unit in zip(captured, target["units"], strict=True):
        name = unit["path"]
        symbols = unit_evidence.symbols
        typed_bodies = unit_evidence.counted_declarations
        declarations = {
            row["symbol"]: row for row in symbols.values() if row["definition"]
        }
        module_children = []
        source = None
        for body in unit_evidence.functions:
            loops, backedges = _natural_loops(body["blocks"])
            identity, rs, ops, es, calls = _build_function_graph(
                body, name, locked[name], loops, backedges
            )
            regions.extend(rs)
            operations.extend(ops)
            edges.extend(es)
            module_children.append(identity)
            symbol = body["symbol"]
            if loops and symbol in typed_bodies:
                if source is None:
                    source = read_artifact(locked[name])
                row = extract_counted_region(
                    body,
                    loops,
                    typed_bodies[symbol],
                    identity,
                    ops,
                    name,
                    locked[name],
                    source,
                )
                if row is not None:
                    counted.append(row)
            local[name, symbol] = identity
            owner = declarations.get(symbol)
            if owner is not None and owner["public"]:
                definitions[symbol].append(identity)
            rows = (
                tuple(
                    symbols[ordinal] for ordinal in owner["calls"] if ordinal in symbols
                )
                if owner
                else ()
            )
            pending.extend(
                (
                    operation,
                    name,
                    symbol,
                    callee,
                    site,
                    memory,
                    rows,
                    unit_evidence.call_sites,
                )
                for operation, callee, site, memory in calls
            )
        regions.append(
            Region(f"tu:{name}", "MODULE", name, children=tuple(module_children))
        )
    for operation, unit, caller, callee, site, memory, rows, call_sites in pending:
        candidates = {
            row["symbol"] for row in rows if callee in (row["symbol"], row["display"])
        }
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
            target_id = local.get((unit, symbol)) or (
                owners[0] if len(owners) == 1 else None
            )
            status = (
                "SOURCE_DEFINITION"
                if target_id
                else "AMBIGUOUS_DEFINITION" if owners else "EXTERNAL"
            )
        elif len(candidates) > 1:
            status = "AMBIGUOUS_SYMBOL"
        elif is_ssa_name(callee) or callee.startswith(("OBJ_TYPE_REF", "*")):
            status = "INDIRECT"
        else:
            status = "UNBOUND_SYMBOL"
        if target_id:
            edges.append(Edge(operation, target_id, "CALL", symbol))
        record = {
            "operation": operation,
            "unit": unit,
            "caller": caller,
            "gimple_target": callee,
            "symbol": symbol,
            "target": target_id,
            "resolution": status,
            "source": site,
            "authority": authorities if symbol else [],
            "effects": {
                "external": "UNKNOWN",
                "memory_ssa": (
                    "DEFINITION"
                    if memory and memory[1]
                    else "USE" if memory else "UNOBSERVED"
                ),
            },
        }
        if len(typed.get(symbol, ())) == 1:
            usr, type_ = next(iter(typed[symbol]))
            record["declaration"] = {"usr": usr, "type": type_}
        records.append(record)
    return (
        tuple(regions),
        tuple(operations),
        tuple(dict.fromkeys(edges)),
        records,
        counted,
    )


def build_compiled_graph(target, reference, out, *, jobs, bundle):
    locked = _lock_sources(target)
    case_receipts = _lock_cases(target, out / "cases", bundle)
    units = target["units"]
    compilers = [select_compiler(target, unit) for unit in units]
    fortran = [
        index
        for index, unit in enumerate(units)
        if unit.get("language", target["language"]) == "fortran"
    ]
    header = gnu_cfi_header(compilers[fortran[0]]) if fortran else None
    header_ref = artifact(header) if header else None
    captured = [None] * len(units)
    toolchains = {}
    for compiler in dict.fromkeys(compilers):
        version = subprocess.check_output(
            [compiler, "--version"], text=True, timeout=20
        ).splitlines()[0]
        toolchains[compiler] = version
    with (
        TemporaryDirectory(prefix="scar-native-front-") as temporary,
        gnu_cfi_overlay(header) as cfi,
    ):
        scratch = Path(temporary)

        def run(index):
            folder = out / f"unit-{index:04d}"
            folder.mkdir()
            return capture_unit_evidence(
                target, units[index], compilers[index], folder, scratch, cfi
            )

        parallel = [index for index in range(len(units)) if index not in fortran]
        if parallel:
            with ThreadPoolExecutor(max_workers=min(jobs, len(parallel))) as pool:
                for index, result in zip(parallel, pool.map(run, parallel)):
                    captured[index] = result
        for index in fortran:
            captured[index] = run(index)
    regions, operations, edges, calls, counted = _link_translation_units(
        target, locked, captured
    )
    units_evidence = [item.receipts for item in captured]
    evidence = {
        "input": asdict(reference),
        "locked_sources": {name: asdict(receipt) for name, receipt in locked.items()},
        "units": units_evidence,
        "compiler_types": {unit["unit"]: unit["types"] for unit in units_evidence},
        "toolchains": toolchains,
        "native_calls": calls,
        "coverage": {
            "source_resolution": "COMPILER_TYPED_SOURCE_AND_SSA",
            "runtime": "NOT_CAPTURED",
        },
    }
    if header_ref:
        evidence["gnu_cfi_header"] = asdict(header_ref)
    from .fragment_bundle import seal_compiled_fragments

    try:
        material, typed_regions = seal_compiled_fragments(
            target, evidence, out / "native-units", regions=counted
        )
        program = GraphProgram(
            native=NativeProgram(local_member(material, bundle), tuple(typed_regions))
        )
    except (OSError, ValueError) as error:
        program = GraphProgram(unavailable=str(error))
    for receipt in (*locked.values(), *case_receipts, reference):
        verify_artifact(receipt)
    if header_ref:
        read_artifact(header_ref)
    return Graph(target, regions, operations, edges, (), evidence, program)
