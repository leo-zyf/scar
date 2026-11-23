"""Compare complete executables in isolated directories, including process startup."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
from tempfile import TemporaryDirectory
import time


def file_output(path, spec):
    if spec.get("format", "bytes") == "bytes":
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if spec["format"] != "numeric_text":
        raise ValueError("expected bytes or numeric_text")
    digest = hashlib.sha256()
    rows = 0
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or any(
            text.startswith(x) for x in spec.get("comment_prefixes", [])
        ):
            continue
        values = [float(cell.replace("D", "e")) for cell in text.split()]
        digest.update((" ".join(x.hex() for x in values) + "\n").encode())
        rows += 1
    if not rows:
        raise ValueError(f"no numeric data in {path.name}")
    return {"rows": rows, "sha256": digest.hexdigest()}


def run_case(spec, base, output):
    warmups, repeats = spec.get("warmups", 2), spec.get("repetitions", 5)
    arms = spec["arms"]
    if (
        repeats < 1
        or warmups < 0
        or len(arms) < 2
        or len({a["name"] for a in arms}) != len(arms)
    ):
        raise ValueError("need distinct labeled arms and positive repetitions")
    times = {arm["name"]: [] for arm in arms}
    rounds, expected = [], None
    stdin = (base / spec["stdin"]).read_bytes() if spec.get("stdin") else b""
    for i in range(warmups + repeats):
        order = arms[i % len(arms) :] + arms[: i % len(arms)]
        round_ = []
        for arm in order:
            with TemporaryDirectory(prefix="scar-compare-") as temp:
                cwd = Path(temp)
                for relative, source in spec.get("inputs", {}).items():
                    if Path(relative).is_absolute() or ".." in Path(relative).parts:
                        raise ValueError(
                            "input destination must be inside the run directory"
                        )
                    dest = cwd / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(base / source, dest)
                for relative in spec.get("directories", []):
                    if Path(relative).is_absolute() or ".." in Path(relative).parts:
                        raise ValueError("run directory must be relative")
                    (cwd / relative).mkdir(parents=True, exist_ok=True)
                with (
                    (cwd / "stdout.log").open("wb") as stdout,
                    (cwd / "stderr.log").open("wb") as stderr,
                ):
                    start = time.perf_counter_ns()
                    result = subprocess.run(
                        arm["argv"],
                        input=stdin,
                        cwd=cwd,
                        stdout=stdout,
                        stderr=stderr,
                        env={
                            **os.environ,
                            **spec.get("environment", {}),
                            **arm.get("environment", {}),
                        },
                        timeout=spec.get("timeout_seconds", 60),
                    )
                    elapsed = time.perf_counter_ns() - start
                if result.returncode:
                    raise ValueError(
                        f"{arm['name']} exited with {result.returncode}: {(cwd/'stderr.log').read_bytes()[-1024:]!r}"
                    )
                files = spec.get("outputs", [{"path": "stdout.log"}])
                observed = {f["path"]: file_output(cwd / f["path"], f) for f in files}
                if expected is None:
                    expected = observed
                if observed != expected:
                    raise ValueError(
                        f"output mismatch in {arm['name']}: {observed} expected {expected}"
                    )
                if i >= warmups:
                    times[arm["name"]].append(elapsed)
                round_.append(dict(arm=arm["name"], wall_ns=elapsed))
        rounds.append(dict(warmup=i < warmups, runs=round_))
    summary = {
        name: dict(
            median_ns=statistics.median(values),
            min_ns=min(values),
            max_ns=max(values),
            stdev_ns=statistics.pstdev(values),
        )
        for name, values in times.items()
    }
    reference = summary[arms[0]["name"]]["median_ns"]
    for row in summary.values():
        row["speedup_vs_original"] = reference / row["median_ns"]
    report = dict(
        status="MATCHED",
        timing_scope="whole process including startup and I/O; setup excluded",
        platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)),
        specification=spec,
        warmups=warmups,
        repetitions=repeats,
        outputs=expected,
        summary=summary,
        rounds=rounds,
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_case(
                json.loads(args.case.read_bytes()), args.case.resolve().parent, args.out
            ),
            indent=2,
        )
    )
