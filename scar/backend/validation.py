"""Verify compiled exports and compare sealed reference and generated runs."""

from decimal import Decimal
from hashlib import sha256
import os
import re
import statistics
import subprocess
from time import monotonic, perf_counter_ns
from scar.contract import Member, read_artifact, verify_artifact

_TIMING = re.compile(rb"(time:) +[0-9]+\.[0-9]+")
_DECIMAL = re.compile(rb"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)[DEde][+-]?[0-9]+")


def verify_exports(executable, exports):
    if not exports:
        return
    symbols = subprocess.check_output(
        ["nm", "-P", "--defined-only", str(executable)], text=True, timeout=30
    )
    missing = set(exports) - {
        line.split()[0] for line in symbols.splitlines() if line.split()
    }
    if missing:
        raise ValueError(
            f"C++20 executable did not preserve link symbols: {sorted(missing)}"
        )


def _scientific_output(data):
    values = [
        Decimal(value.upper().replace(b"D", b"E").decode())
        for value in _DECIMAL.findall(data)
    ]
    return _DECIMAL.sub(b"<decimal>", data).split(), values


def _matches(comparison, original, generated):
    if (
        original.returncode
        or generated.returncode
        or original.stderr != generated.stderr
    ):
        return False
    if comparison == "timed-fields":
        return _TIMING.sub(rb"\1 <timing>", original.stdout) == _TIMING.sub(
            rb"\1 <timing>", generated.stdout
        )
    if comparison == "scientific-text":
        return _scientific_output(original.stdout) == _scientific_output(
            generated.stdout
        )
    return original.stdout == generated.stdout


def validate_cases(
    validation, bundle, executable, out, cpus, seconds, repetitions, warmups
):
    cases = validation.get("cases", ())
    if not cases:
        return []
    baseline = validation["baseline"]
    original_ref = Member(**baseline["executable"]).resolve(bundle)
    original = [original_ref.path, *baseline["argv"]]
    verify_artifact(original_ref)
    if any(type(cpu) is not int or cpu not in os.sched_getaffinity(0) for cpu in cpus):
        raise ValueError("selected CPU is unavailable for native validation")
    affinity = ["taskset", "-c", ",".join(map(str, cpus))] if cpus else []
    validated = []
    for case in cases:
        name = case["id"]
        comparison = case.get("comparison", "exact")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid native case name: {name}")
        if comparison not in ("exact", "timed-fields", "scientific-text"):
            raise ValueError(f"unsupported native case comparison: {comparison}")
        folder = out / "validation" / name
        folder.mkdir(parents=True)
        stdin = (
            read_artifact(Member(**case["input"]).resolve(bundle))
            if "input" in case
            else b""
        )
        arguments = case.get("argv", ())
        arms = {
            "original": (original, baseline["cwd"]),
            "generated": ([str(executable), *original[1:]], folder),
        }
        env = dict(os.environ, LC_ALL="C")
        deadline = monotonic() + seconds
        pairs = []
        for index in range(warmups + repetitions):
            outputs, measurements = {}, {}
            order = tuple(arms) if index % 2 == 0 else tuple(reversed(arms))
            for label in order:
                command, cwd = arms[label]
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"native case exceeded its time budget: {name}")
                start = perf_counter_ns()
                run = subprocess.run(
                    [*affinity, *command, *arguments],
                    cwd=cwd,
                    env=env,
                    input=stdin,
                    capture_output=True,
                    timeout=remaining,
                )
                measurements[label] = dict(
                    ns=perf_counter_ns() - start, exit=run.returncode
                )
                outputs[label] = run
                (folder / f"{index}-{label}.stdout").write_bytes(run.stdout)
                (folder / f"{index}-{label}.stderr").write_bytes(run.stderr)
            equal = _matches(comparison, outputs["original"], outputs["generated"])
            pairs.append(
                dict(
                    warmup=index < warmups,
                    order=order,
                    arms=measurements,
                    matches=equal,
                )
            )
            if not equal:
                break
        timing = {}
        if all(pair["matches"] for pair in pairs):
            for label in arms:
                samples = [
                    pair["arms"][label]["ns"] for pair in pairs if not pair["warmup"]
                ]
                timing[label] = dict(
                    median_ns=statistics.median(samples),
                    min_ns=min(samples),
                    max_ns=max(samples),
                    stddev_ns=statistics.pstdev(samples),
                )
        validated.append(
            dict(
                case=name,
                matches=all(pair["matches"] for pair in pairs),
                comparison=comparison,
                stdin_sha256=sha256(stdin).hexdigest(),
                pairs=pairs,
                timing=timing,
            )
        )
    verify_artifact(original_ref)
    return validated
