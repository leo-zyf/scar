"""Estimate costs for the CPU traversals authorized by the execution IR."""

from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import (
    PHYSICAL,
    digest,
    document,
    control_regions as _control_regions,
    validate_resource_flow,
)
from scar.host import cpu_availability
from .movement import COPY_METRIC, SAMPLE_BYTES, Movement

_PROBE = Path(__file__).with_name("calibration.cpp")


def _worker_caps(host_budget, requested):
    """Apply OpenMP and caller limits to the deployment's conservative budget."""
    caps = {"deployment_worker_budget": host_budget}
    for name in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS"):
        raw = os.environ.get(name, "")
        value = (raw.partition(",")[0] if name == "OMP_NUM_THREADS" else raw).strip()
        if value.isascii() and value.isdecimal() and int(value) > 0:
            caps[name] = int(value)
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError("worker_limit must be a positive integer")
        caps["requested_worker_limit"] = requested
    return caps


def _widths(limit):
    return sorted({1, limit, *(1 << p for p in range(limit.bit_length()))})


def _probe(limit, metrics):
    widths = _widths(limit)
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise ValueError("C++ compiler unavailable for CPU calibration")
    samples = {p: [{}, {}, {}] for p in widths}
    with TemporaryDirectory(prefix="scar-cpu-profile-") as temp:
        directory = Path(temp)
        source, binary = _PROBE, directory / "measure"
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O3",
                "-ffp-contract=off",
                "-fopenmp",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        try:
            result = subprocess.run(
                [
                    str(binary),
                    str(SAMPLE_BYTES),
                    *map(str, widths),
                    "--",
                    *sorted(metrics),
                ],
                capture_output=True,
                timeout=45,
            )
            output = result.stdout
            interrupted = result.returncode != 0
        except subprocess.TimeoutExpired as timeout:
            output = timeout.stdout or b""
            interrupted = True
        # A busy wider team can exhaust the global probe budget. Earlier fully
        # measured widths are still evidence for their own physical alternatives.
        for line in output.rpartition(b"\n")[0].decode().splitlines():
            worker, rep, metric, ns = line.split()
            width, repetition, value = int(worker), int(rep), float(ns)
            if (
                width not in samples
                or not 0 <= repetition < 3
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("invalid target calibration")
            row = samples[width][repetition]
            if metric in row:
                raise ValueError("duplicate target calibration")
            row[metric] = value
    for width, rows in list(samples.items()):
        required = metrics
        if width > 1:
            required = required - {"carry_i64", "guard_i64", "fp_guard", COPY_METRIC}
        if any(row.keys() != required for row in rows):
            if width == 1 or not interrupted:
                raise ValueError("incomplete target calibration")
            del samples[width]
    return samples, compiler


_BYTES = {
    "bool": 1,
    "char": 1,
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


def _scratch_geometry():
    """Deployment cache line and page size, in bytes, or no placement offer."""
    try:
        line = int(
            subprocess.check_output(
                ["getconf", "LEVEL1_DCACHE_LINESIZE"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        )
        page = os.sysconf("SC_PAGESIZE")
        if 0 < line <= page and page % line == 0:
            return line, page
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _unqualified(dtype):
    while isinstance(dtype, dict) and dtype.get("kind") in {"const", "ref"}:
        dtype = dtype.get("of", dtype.get("to"))
    return dtype


def _element(dtype):
    dtype = _unqualified(dtype)
    if (
        isinstance(dtype, dict)
        and dtype.get("kind") == "array"
        and len(dtype.get("shape", ())) > 1
    ):
        return {**dtype, "shape": dtype["shape"][1:]}
    if isinstance(dtype, dict) and dtype.get("kind") in {"ptr", "array"}:
        return _unqualified(dtype.get("to", dtype.get("of")))
    return None


def _arithmetic_type(left, right):
    """Usual arithmetic conversions on the fixed-width CPU C++20 target."""
    left, right = _unqualified(left), _unqualified(right)
    if not all(isinstance(t, str) and t in _BYTES for t in (left, right)):
        return None
    if "f64" in (left, right):
        return "f64"
    if "f32" in (left, right):
        return "f32"
    left, right = ("i32" if _BYTES[t] < 4 else t for t in (left, right))
    width = max(_BYTES[left], _BYTES[right])
    unsigned = any(t.startswith("u") and _BYTES[t] == width for t in (left, right))
    return ("u" if unsigned else "i") + str(width * 8)


def _structural_key(expr):
    """Only direct value expressions can share a quotient and remainder estimate."""
    op = expr["op"]
    if op == "var" and expr.get("scope") != "global":
        return ("var", expr["name"])
    if op == "literal":
        return ("literal", expr.get("type"), expr.get("value"))
    if op == "binary":
        left, right = _structural_key(expr["left"]), _structural_key(expr["right"])
        if left is not None and right is not None:
            return (expr["operator"], left, right)
    return None


def _access_key(value):
    if isinstance(value, list):
        return tuple(map(_access_key, value))
    if isinstance(value, dict):
        return tuple(
            (key, _access_key(child))
            for key, child in sorted(value.items())
            if key not in {"site", "location", "id"}
        )
    return value


def _work(stmt, types, coordinates=()):
    """Count typed operations and logical streams in one B2 traversal."""
    types = {**types, stmt["index"]: "i64"}
    counts = Counter()
    streams = {}
    missing = set()
    locals_ = set()
    replaced = {tuple(c["path"]) for c in coordinates}

    def arithmetic(dtype):
        if dtype:
            counts["arith_" + dtype.replace("u", "i")] += 1

    def visit(x, path, divides, write=False):
        if path in replaced:
            return "i64"
        op = x["op"]
        if op == "literal":
            return x["type"]
        if op == "var":
            dtype = types.get(
                x["name"] if x.get("scope") != "global" else ("global", x["name"])
            )
            if dtype is None:
                missing.add("unresolved variable " + x["name"])
            return _unqualified(dtype)
        if op == "index":
            dtype = visit(x["base"], (*path, "base"), divides)
            for i, idx in enumerate(x["indices"]):
                visit(idx, (*path, "indices", i), divides)
                dtype = _element(dtype)
            if (
                x["base"].get("op") != "var"
                or not isinstance(dtype, str)
                or dtype not in _BYTES
            ):
                missing.add("non-scalar or unresolved indexed memory stream")
            else:
                # Distinct subscripts on one allocation may fetch different elements.
                key = (
                    write,
                    x["base"]["name"],
                    _access_key(x["indices"]),
                )
                streams[key] = _BYTES[dtype]
            return dtype
        if op == "decl":
            dtype = _unqualified(x["type"])
            if (
                not isinstance(dtype, str)
                or dtype not in _BYTES
                or x.get("storage") not in (None, "auto")
            ):
                missing.add("unpriced local storage")
            if x.get("init") is not None:
                actual = visit(x["init"], (*path, "init"), divides)
                if _unqualified(actual) != dtype:
                    missing.add("uncalibrated initializer conversion")
            types[x["name"]] = dtype
            locals_.add(x["name"])
            return dtype
        if op == "assign":
            target = x["target"]
            local = (
                target["op"] == "var"
                and target["name"] in locals_
                and target.get("scope") != "global"
            )
            if target["op"] != "index" and not local:
                missing.add("unpriced scalar or indirect store")
            target_type = visit(target, (*path, "target"), divides, True)
            value_type = visit(x["value"], (*path, "value"), divides)
            if _unqualified(target_type) != _unqualified(value_type):
                missing.add("uncalibrated assignment conversion")
            return target_type
        if op in {"cast", "unary"}:
            source = visit(x["value"], (*path, "value"), divides)
            if op == "cast":
                dtype = _unqualified(x["type"])
                if source != dtype:
                    if (
                        isinstance(source, str)
                        and isinstance(dtype, str)
                        and source[:1] in {"i", "u"}
                        and dtype[:1] in {"i", "u"}
                        and _arithmetic_type(source, dtype)
                    ):
                        arithmetic(_arithmetic_type(source, dtype))
                    else:
                        missing.add("uncalibrated scalar or address conversion")
                return dtype
            operator = x["operator"]
            if operator == "deref":
                # Its cost depends on pointed-to storage, cache behavior and aliasing.
                missing.add("unpriced pointer dereference")
                return _element(source)
            if operator in {"pos", "neg", "bnot", "not"}:
                dtype = _arithmetic_type(source, source)
                if not dtype:
                    missing.add("unresolved unary arithmetic type")
                if operator != "pos":
                    arithmetic(dtype)
                return "bool" if operator == "not" else dtype
            missing.add("unsupported unary " + operator)
            return None
        if op != "binary":
            missing.add("unpriced operation " + op)
            return None
        lhs, rhs = x["left"], x["right"]
        left = visit(lhs, (*path, "left"), divides)
        right = visit(rhs, (*path, "right"), divides)
        dtype = _arithmetic_type(left, right)
        operator = x["operator"]
        if not dtype:
            missing.add("unresolved scalar type for " + operator)
            return None
        if operator in {"div", "mod"}:
            if dtype[0] == "f":
                if operator != "div":
                    missing.add("floating remainder")
                elif rhs["op"] == "literal" and type(rhs.get("value")) in {int, float}:
                    divisor = rhs["value"]
                    if not math.isfinite(divisor) or divisor == 0:
                        missing.add("nonfinite or zero floating divisor")
                    elif abs(divisor) == 1:
                        if divisor < 0:
                            arithmetic(dtype)
                    else:
                        prefix = (
                            "shift" if math.frexp(abs(divisor))[0] == 0.5 else "const"
                        )
                        counts[f"{prefix}_div_{dtype}"] += 1
                else:
                    counts["div_" + dtype] += 1
            elif rhs["op"] == "literal" and type(rhs.get("value")) is int:
                divisor = rhs["value"]
                if divisor == 0:
                    missing.add("zero integer divisor")
                elif abs(divisor) == 1:
                    if divisor < 0 and operator == "div":
                        arithmetic(dtype)
                else:
                    prefix = (
                        "shift" if abs(divisor) & (abs(divisor) - 1) == 0 else "const"
                    )
                    counts[f"{prefix}_{operator}_{dtype}"] += 1
            else:
                numerator, denominator = _structural_key(lhs), _structural_key(rhs)
                if numerator is None or denominator is None:
                    counts[f"{operator}_{dtype}"] += 1
                else:
                    divides.setdefault((dtype, numerator, denominator), set()).add(
                        operator
                    )
        elif operator in {
            "add",
            "sub",
            "mul",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "band",
            "bor",
            "bxor",
            "shl",
            "shr",
            "land",
            "lor",
        }:
            arithmetic(dtype)
        else:
            missing.add("unpriced binary " + operator)
        return (
            "bool"
            if operator in {"eq", "ne", "lt", "le", "gt", "ge", "land", "lor"}
            else dtype
        )

    for i, step in enumerate(stmt["body"]):
        divides = {}
        visit(step, (i,), divides)
        for (dtype, _, _), ops in divides.items():
            group = "divmod" if len(ops) == 2 else next(iter(ops))
            counts[f"{group}_{dtype}"] += 1
    return dict(
        streams=len(streams),
        bytes_per_element=sum(streams.values()),
        operations=dict(sorted(counts.items())),
        priced=not missing,
        unpriced=sorted(missing),
    )


def _candidates(stmt, types):
    linear = _work(stmt, types)
    result = dict(linear=linear, parallel_allowed=stmt.get("independent") is True)
    layout = stmt.get("iteration_layout")
    if not layout or layout["kind"] != "mixed_radix":
        return result
    result["coordinate"] = _work(stmt, types, layout["coordinates"])
    # When an inner positive extent is symbolic, use its lower bound of one.
    # The upper bound accounts for one additional loop increment/test per outer visit.
    axes = layout["axes"]
    outer_rate = 0.0
    inner_elements = 1
    for axis in axes[:-1]:
        if axis.get("op") == "literal":
            inner_elements *= max(1, axis["value"])
        outer_rate += 1 / inner_elements
    result["guard_checks"] = 1 + len(axes) + len(layout["equalities"])
    result["carry_steps_per_element_bound"] = outer_rate
    return result


def _slope(work, row, carry=0.0):
    arithmetic = sum(n * row[key] for key, n in work["operations"].items())
    if carry:
        arithmetic += carry * row["carry_i64"]
    return max(work["bytes_per_element"] * row["byte"], arithmetic)


def _costs(bodies, samples):
    linear = bodies["linear"]
    lines = {
        (w, "linear"): (
            0 if w == 1 else max(row["team"] for row in rows),
            (statistics.median if w == 1 else max)(_slope(linear, row) for row in rows),
        )
        for w, rows in samples.items()
        if w == 1 or bodies["parallel_allowed"]
    }
    coord = bodies.get("coordinate")
    threshold = None
    if coord and coord["priced"]:
        rows = samples[1]
        guard = statistics.median(
            row["guard_i64"] * bodies["guard_checks"] for row in rows
        )
        elapsed = statistics.median(
            _slope(coord, row, bodies["carry_steps_per_element_bound"]) for row in rows
        )
        saving = lines[1, "linear"][1] - elapsed
        lines[1, "mixed_radix"] = (guard, elapsed)
        if saving > 0 and guard / saving < 2**62:
            threshold = max(1, math.floor(guard / saving) + 1)
    return lines, threshold


def _bands(lines):
    # The active B2 traversals are affine between adjacent intersections.
    # Workers greater than the cardinality also fall back in the runtime.
    starts = {0, *(w for w, _ in lines)}
    for a, b in lines.values():
        for c, d in lines.values():
            if b != d:
                n = (c - a) / (b - d)
                if math.isfinite(n) and 0 <= n < 2**62:
                    starts.update((math.floor(n), math.floor(n) + 1))
    bands = []
    for n in sorted(starts):
        legal = (k for k in lines if k[0] == 1 or k[0] <= n)
        width, _ = min(legal, key=lambda k: (lines[k][0] + n * lines[k][1], k))
        if not bands or bands[-1]["workers"] != width:
            bands.append(dict(minimum_elements=n, workers=width))
    return bands


def _fusion_profitable(contenders, guard_ns, extent):
    """Compare actual worker/layout envelopes at every feasible cardinality."""
    if guard_ns is None or not all(contenders):
        return False

    def price(options, n):
        return min(
            a + n * b
            for (width, _), (a, b) in options.items()
            if width == 1 or width <= n
        )

    def gain(n):
        return sum(price(x, n) for x in contenders[1:]) - (
            guard_ns + price(contenders[0], n)
        )

    if extent.get("op") == "literal" and type(extent.get("value")) is int:
        return gain(max(0, extent["value"])) > 0
    slopes = [min(b for _, b in options.values()) for options in contenders]
    if slopes[0] >= sum(slopes[1:]):
        return False
    # The reference is also allowed to pick the best width for each original map.
    points = {1}
    for options in contenders:
        for width, _ in options:
            points.update((max(1, width - 1), width))
        for a, b in options.values():
            for c, d in options.values():
                if b != d:
                    cross = (c - a) / (b - d)
                    if math.isfinite(cross) and 1 <= cross < 2**62:
                        points.update((math.floor(cross), math.ceil(cross)))
    return all(gain(n) >= 0 for n in points)


def _collect(module):
    maps = {}
    has_fusion = False
    globals_ = {("global", g["name"]): g["type"] for g in module.get("globals", [])}

    def scope(items, types):
        nonlocal has_fusion
        for stmt in items:
            op = stmt["op"]
            if op == "schedule_choice":
                # Price each alternative in the same incoming scope. Only choose()
                # can replace this node with the selected branch and any guard.
                has_fusion = True
                scope(stmt["reference"], types.copy())
                scope((stmt["fused"],), types.copy())
                continue
            if op == "decl":
                types[stmt["name"]] = stmt["type"]
            elif op == "buffer":
                types[stmt["name"]] = {"kind": "ptr", "to": stmt["type"]}
            elif op == "runtime_array":
                types[stmt["name"]] = {"kind": "ptr", "to": stmt["element"]}
            if op == "map" and (
                stmt.get("independent") is True or stmt.get("iteration_layout")
            ):
                maps[id(stmt)] = _candidates(stmt, types)
            # for.init declarations are visible to the loop and end with it.
            enclosing = types.copy() if op == "for" else types
            for owner, field, binding in _control_regions(stmt):
                nested = enclosing if field == "init" else enclosing.copy()
                if op == "map":
                    nested[stmt["index"]] = "i64"
                elif op == "each":
                    nested[stmt["name"]] = stmt["type"]
                if isinstance(binding, str):
                    nested[binding] = {"kind": "handle", "resource": "error"}
                scope(owner[field], nested)

    for fn in module["functions"]:
        scope(fn["body"], {**globals_, **{p["name"]: p["type"] for p in fn["params"]}})
    return maps, has_fusion


def _structural_layout(bodies):
    """Use operation counts when a calibrated cost comparison is unavailable."""
    coord = bodies.get("coordinate")
    if not coord:
        return {}
    raw = bodies["linear"]["operations"]
    remaining = coord["operations"]
    divisions = sum(
        raw.get(k, 0) - remaining.get(k, 0)
        for k in raw.keys() | remaining.keys()
        if k.startswith(("div_i", "mod_i", "divmod_i", "div_u", "mod_u", "divmod_u"))
    )
    return dict(
        serial_layout="mixed_radix" if divisions > 0 else "linear",
        minimum_coordinate_elements=(
            max(1, math.ceil(bodies["guard_checks"] / divisions))
            if divisions > 0
            else None
        ),
        coordinate_cost=dict(
            basis="structural_counts",
            guard_checks=bodies["guard_checks"],
            removed_variable_integer_divisions=divisions,
        ),
    )


def schedule(module, mode, worker_limit=None, *, consume_functions=False):
    """Schedule a module, preserving caller data unless B1 owns its decoded bodies.

    With ownership, function bodies can be updated in place. The output gets its
    own module-level bindings so sealing a member cannot overwrite the original
    resource inventory. Other nested metadata is shared read-only in this mode.
    """
    inventory = module.get("resource_model", {})
    movement = None
    if "resource_flow" in inventory:
        flow = inventory["resource_flow"]
        validate_resource_flow(flow, inventory.get("resources", ()))
        movement = Movement(flow, _BYTES)
    deployment_cpu = cpu_availability()
    caps = _worker_caps(deployment_cpu.worker_budget, worker_limit)
    if mode not in {"auto", "serial"}:
        raise ValueError("unknown CPU resource constraint")
    if mode == "serial":
        caps["serial_worker_limit"] = 1
    limit = min(caps.values())
    selected = dict(module) if consume_functions else deepcopy(module)
    maps, has_fusion = _collect(selected)
    report = document(
        PHYSICAL,
        target="cpu-cxx20",
        mode=mode,
        affinity=list(deployment_cpu.cpu_ids),
        deployment_cpu=deployment_cpu.as_record(),
        worker_policy=dict(method="minimum", caps=caps),
        worker_limit=limit,
        workers=1,
        minimum_parallel_elements=None,
        regions={},
        choices=[],
        deployment="unverified",
    )
    samples = {}
    map_probe = bool(maps)
    metrics = movement.required_metrics() if movement else set()
    if map_probe:
        metrics.update(("byte", "team"))
        if has_fusion:
            metrics.add("fp_guard")
        for bodies in maps.values():
            for name in ("linear", "coordinate"):
                work = bodies.get(name)
                if work and work["priced"]:
                    metrics.update(work["operations"])
            if "coordinate" in bodies:
                metrics.update(("guard_i64", "carry_i64"))
    if metrics:
        try:
            probe_limit = (
                limit
                if map_probe and any(b["parallel_allowed"] for b in maps.values())
                else 1
            )
            samples, compiler = _probe(probe_limit, metrics)
            units = {}
            if map_probe:
                units.update(
                    team="ns per team",
                    byte="ns per logical stream byte",
                    operations="ns per operation or named probe group",
                )
            if COPY_METRIC in metrics:
                units[COPY_METRIC] = "ns per payload byte"
            report["calibration"] = dict(
                evidence="observed",
                compiler=compiler,
                source_sha256=digest(_PROBE.read_bytes()),
                metrics=sorted(metrics),
                metric_widths={
                    metric: [
                        width for width, rows in samples.items() if metric in rows[0]
                    ]
                    for metric in sorted(metrics)
                },
                samples=samples,
                repetitions=3,
                units=units,
                coverage="powers of two and the effective probe budget",
                sampled_widths=sorted(samples),
                unmeasured_widths=sorted(set(_widths(probe_limit)) - samples.keys()),
            )
            if map_probe:
                report["calibration"].update(
                    elements_per_sample=1 << 18,
                    constant_divisors=dict(shift=8, other=7),
                )
                report["cost_model"] = dict(
                    evidence="inferred",
                    duration="setup_ns + elements * max(logical_bytes * ns_per_byte, typed_operation_ns)",
                    envelope="serial median and parallel maximum across three observed repetitions",
                    ordinary_arithmetic_proxy="four same-type multiply/add operations per result",
                    coordinate_assumption="axis guards pass; the emitted linear loop remains the guard fallback",
                    coordinate_controls="guard comparison and loop-carry observations are proxies for outer-loop control",
                    constant_division="representative compile-time divisors 8 and 7 for power-of-two and other constants",
                )
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            report["calibration_issue"] = str(error)
    if movement is not None:
        report["movement_plan"] = movement.plan(samples)

    choices = {}
    for site, bodies in maps.items():
        work = bodies["linear"]
        lines, threshold = ({}, None)
        region = dict(
            work=work,
            widths=[dict(minimum_elements=0, workers=1)],
            serial_layout="linear",
        )
        if "coordinate" in bodies:
            region.update(
                coordinate_work=bodies["coordinate"],
                guard_checks=bodies["guard_checks"],
                carry_steps_per_element_bound=bodies["carry_steps_per_element_bound"],
            )
        if samples and work["priced"]:
            lines, threshold = _costs(bodies, samples)
            region.update(
                cost_evidence="inferred",
                candidate_costs=[
                    dict(
                        workers=w,
                        layout=layout,
                        setup_ns=intercept,
                        ns_per_element=slope,
                    )
                    for (w, layout), (intercept, slope) in lines.items()
                ],
            )
            if threshold:
                region.update(
                    serial_layout="mixed_radix", minimum_coordinate_elements=threshold
                )
        else:
            region.update(_structural_layout(bodies))
            region["fallback"] = work["unpriced"] or ["target calibration unavailable"]
        choices[site] = (region, lines)

    geometry = None
    measured_geometry = False

    def place_private_buffer(stmt):
        nonlocal geometry, measured_geometry, used
        if "inline_capacity" in stmt:
            raise ValueError("logical buffer contains a target placement")
        if stmt.get("private_scratch") != "indexed_scalar_access_only":
            return
        dtype = stmt["type"]
        if not isinstance(dtype, str) or dtype not in _BYTES:
            return
        if not measured_geometry:
            geometry = _scratch_geometry()
            measured_geometry = True
        if geometry is None:
            return
        line, page = geometry
        capacity = line // _BYTES[dtype]
        if not capacity or used + line > page:
            return
        used += line
        stmt["inline_capacity"] = capacity
        report.setdefault("allocations", []).append(
            dict(
                function=fn["name"],
                name=stmt["name"],
                site=stmt.get("site"),
                storage="inline_or_heap",
                type=dtype,
                inline_capacity=capacity,
                cache_line_bytes=line,
                function_payload_budget_bytes=page,
                evidence="deployment getconf L1 data cache line and sysconf page size",
            )
        )

    def choose(items):
        output = []
        for stmt in items:
            op = stmt["op"]
            if op == "schedule_choice":
                reference = stmt["reference"]
                fused = stmt["fused"]
                group = [fused, *reference]
                contenders = [choices.get(id(s), ({}, {}))[1] for s in group]
                guard = (
                    statistics.median(row["fp_guard"] for row in samples[1])
                    if samples and "fp_guard" in samples[1][0]
                    else None
                )
                selected_fusion = _fusion_profitable(contenders, guard, fused["extent"])
                report["choices"].append(
                    dict(
                        site=stmt["site"],
                        chosen="fused" if selected_fusion else "reference",
                        guard_ns=guard,
                        cost_evidence=(
                            "inferred"
                            if guard is not None and all(contenders)
                            else "unpriced"
                        ),
                        reference_work=[maps[id(s)]["linear"] for s in reference],
                        fused_work=maps[id(fused)]["linear"],
                        guard=stmt["guard"],
                    )
                )
                if selected_fusion:
                    output.append(
                        dict(
                            op="if",
                            condition={"op": "fp_traps_masked"},
                            then=choose([fused]),
                            **{"else": choose(reference)},
                        )
                    )
                else:
                    output.extend(choose(reference))
                continue
            if op == "buffer":
                place_private_buffer(stmt)
            if op == "runtime_array":
                if stmt.get("scope_control") != "lexical_continuation":
                    raise ValueError(
                        "stack allocation requires the M4 lexical lifetime contract"
                    )
                stmt["allocation"] = "scoped_stack"
                report.setdefault("allocations", []).append(
                    dict(
                        name=stmt["name"],
                        site=stmt.get("site"),
                        storage="scoped_stack",
                        duration="lexical_continuation",
                    )
                )
            for owner, field, _ in _control_regions(stmt):
                owner[field] = choose(owner[field])
            if op == "map":
                region, lines = choices.get(id(stmt), ({}, {}))
                if lines and stmt.get("independent") is True:
                    region["widths"] = _bands(lines)
                    report["workers"] = max(
                        report["workers"], *(r["workers"] for r in region["widths"])
                    )
                report["regions"][stmt["site"]] = region or {"serial_layout": "linear"}
            output.append(stmt)
        return output

    for fn in selected["functions"]:
        used = 0
        fn["body"] = choose(fn["body"])
    return selected, report
