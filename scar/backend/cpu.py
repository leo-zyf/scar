"""Choose Linux CPU schedules using measured team, arithmetic and memory costs."""

from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import PHYSICAL, digest, document

_PROBE = r"""
#include <omp.h>
#include <cfenv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
using clock_type = std::chrono::steady_clock;
double ns(clock_type::time_point t) {
    return std::chrono::duration<double, std::nano>(clock_type::now()-t).count();
}
int main(int argc, char **argv) {
    constexpr int n = 1<<20;
    std::vector<double> a(n, 1.001), b(n, 1.002), out(n);
    volatile double sink = 0;
    omp_set_dynamic(0);
    for(int j=1; j<argc; ++j) {
        int p=std::atoi(argv[j]);
        std::vector<int> marks(p);
        for(int rep=-1; rep<3; ++rep) {
            auto t=clock_type::now();
            for(int k=0; k<8; ++k) {
                if(p==1) marks[0]=k;
                else {
                    std::fenv_t initial;
                    std::fegetenv(&initial);
                    int raised=0;
                    #pragma omp parallel num_threads(p) reduction(|:raised)
                    {
                        std::fenv_t previous;
                        std::fegetenv(&previous);
                        std::fesetenv(&initial);
                        marks[omp_get_thread_num()]=k;
                        raised |= std::fetestexcept(FE_ALL_EXCEPT);
                        std::fesetenv(&previous);
                    }
                    if(raised) std::feraiseexcept(raised);
                }
                sink=marks.back();
            }
            double setup=ns(t)/8;
            t=clock_type::now();
            if(p==1) for(int i=0;i<n;++i) out[i]=a[i];
            else {
                #pragma omp parallel for num_threads(p) schedule(static)
                for(int i=0;i<n;++i) out[i]=a[i];
            }
            double copy=ns(t); sink=out[rep+1];
            t=clock_type::now();
            if(p==1) for(int i=0;i<n;++i) out[i]=(a[i]*b[i]+a[i])*b[i]+b[i];
            else {
                #pragma omp parallel for num_threads(p) schedule(static)
                for(int i=0;i<n;++i) out[i]=(a[i]*b[i]+a[i])*b[i]+b[i];
            }
            double math=ns(t); sink=out[rep+1];
            t=clock_type::now();
            if(p==1) for(int i=0;i<n;++i) out[i]=a[i]/b[i];
            else {
                #pragma omp parallel for num_threads(p) schedule(static)
                for(int i=0;i<n;++i) out[i]=a[i]/b[i];
            }
            double div=ns(t); sink=out[rep+1];
            if(rep>=0) std::printf("%d %.9g %.9g %.9g %.9g\n",p,setup,copy/(16*n),math/(4*n),div/n);
        }
    }
    return sink==0;
}
"""


def _limits():
    available = sorted(os.sched_getaffinity(0))
    limit = len(available)
    try:
        group = next(
            s[3:]
            for s in Path("/proc/self/cgroup").read_text().splitlines()
            if s.startswith("0::")
        )
        root = Path("/sys/fs/cgroup")
        path = root / group.lstrip("/")
        for folder in (path, *path.parents):
            if not folder.is_relative_to(root):
                break
            quota = folder / "cpu.max"
            if quota.is_file():
                amount, period = quota.read_text().split()
                if amount != "max":
                    limit = min(limit, max(1, int(amount) // int(period)))
    except (OSError, StopIteration, ValueError):
        pass
    for var in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS"):
        token = os.environ.get(var, "").split(",")[0]
        if token.isdigit() and int(token):
            limit = min(limit, int(token))
    return available, limit


def _probe(limit):
    widths = sorted(
        {1, limit, *(2**p for p in range(limit.bit_length()) if 2**p <= limit)}
    )
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise ValueError("C++ compiler unavailable for CPU calibration")
    samples = {p: [] for p in widths}
    with TemporaryDirectory(prefix="scar-cpu-profile-") as temp:
        directory = Path(temp)
        source, binary = directory / "measure.cpp", directory / "measure"
        source.write_text(_PROBE)
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
        result = subprocess.run(
            [str(binary), *map(str, widths)],
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
        for line in result.stdout.splitlines():
            width, *times = line.split()
            samples[int(width)].append(list(map(float, times)))
    if any(
        len(rows) != 3
        or not all(math.isfinite(v) and v > 0 for row in rows for v in row)
        for rows in samples.values()
    ):
        raise ValueError("incomplete target calibration")
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


def _element_bytes(t):
    while isinstance(t, dict):
        t = t.get("to", t.get("of"))
    return _BYTES.get(t)


def _work(stmt, element_sizes):
    reads, writes = set(), set()
    ops = divisions = 0
    pure = True

    def visit(x, write=False):
        nonlocal ops, divisions, pure
        if isinstance(x, list):
            for item in x:
                visit(item)
        if not isinstance(x, dict):
            return
        if "op" in x and x["op"] not in {
            "index",
            "assign",
            "binary",
            "unary",
            "cast",
            "var",
            "literal",
        }:
            pure = False
        if x.get("op") == "unary" and x["operator"] not in {
            "pos",
            "neg",
            "not",
            "bnot",
        }:
            pure = False
        if x.get("op") == "index":
            base = x["base"]
            if base.get("op") != "var":
                pure = False
            (writes if write else reads).add(base.get("name"))
            visit(x["indices"])
            return
        if x.get("op") == "assign":
            visit(x["target"], True)
            visit(x["value"])
            return
        if x.get("op") == "binary":
            if x["operator"] == "div":
                divisions += 1
            else:
                ops += 1
        for field in x.values():
            visit(field)

    visit(stmt["body"])
    known = pure and all(element_sizes.get(base) for base in reads | writes)
    return dict(
        streams=len(reads) + len(writes),
        bytes_per_element=sum(element_sizes.get(x) or 0 for x in [*reads, *writes]),
        scalar_ops=ops,
        divisions=divisions,
        priced=known,
    )


def _bands(work, samples):
    lines = {}
    for width, rows in samples.items():
        costs = [
            max(
                work["bytes_per_element"] * byte,
                work["scalar_ops"] * arith + work["divisions"] * div,
            )
            for _, byte, arith, div in rows
        ]
        # Parallel estimates include the slowest calibration sample. A tied
        # prediction favors fewer workers.
        lines[width] = (
            0 if width == 1 else max(row[0] for row in rows),
            statistics.median(costs) if width == 1 else max(costs),
        )
    starts = {0}
    for a, b in lines.values():
        for c, d in lines.values():
            if b != d:
                n = (c - a) / (b - d)
                if math.isfinite(n) and 0 <= n < 2**62:
                    starts.add(math.floor(n) + 1)
    bands = []
    for n in sorted(starts):
        width = min(lines, key=lambda p: (lines[p][0] + n * lines[p][1], p))
        if not bands or bands[-1]["workers"] != width:
            bands.append(dict(minimum_elements=n, workers=width))
    return bands, lines


def schedule(module, mode, worker_limit=None):
    available, limit = _limits()
    if worker_limit is not None:
        limit = min(limit, worker_limit)
    report = document(
        PHYSICAL,
        target="cpu-cxx20",
        mode=mode,
        affinity=available,
        worker_limit=limit,
        workers=1,
        minimum_parallel_elements=None,
        regions={},
        choices=[],
        deployment="unverified",
    )
    samples = {}

    def maps(v):
        if isinstance(v, dict):
            if v.get("op") == "map" and v.get("independent"):
                yield v
            for child in v.values():
                yield from maps(child)
        elif isinstance(v, list):
            for child in v:
                yield from maps(child)

    if mode == "auto" and next(maps(module["functions"]), None) is not None:
        try:
            samples, compiler = _probe(limit)
            report["calibration"] = dict(
                compiler=compiler,
                source_sha256=digest(_PROBE.encode()),
                metrics=[
                    "team_ns",
                    "ns_per_byte",
                    "ns_per_arithmetic",
                    "ns_per_divide",
                ],
                samples=samples,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["calibration_issue"] = str(error)

    def choose(items, element_sizes):
        output = []
        for stmt in items:
            if stmt["op"] == "schedule_choice":
                baseline = [_work(s, element_sizes) for s in stmt["reference"]]
                merged = _work(stmt["fused"], element_sizes)
                selected = False
                if samples and merged["priced"] and all(w["priced"] for w in baseline):
                    _, flat = _bands(merged, {1: samples[1]})
                    reference = sum(
                        _bands(work, {1: samples[1]})[1][1][1] for work in baseline
                    )
                    selected = flat[1][1] < reference
                report["choices"].append(
                    dict(
                        site=stmt["site"],
                        chosen="fused" if selected else "reference",
                        reference_work=baseline,
                        fused_work=merged,
                        guard=stmt["guard"],
                    )
                )
                if selected:
                    output.append(
                        dict(
                            op="if",
                            condition={"op": "fp_traps_masked"},
                            then=choose([stmt["fused"]], element_sizes),
                            **{"else": choose(stmt["reference"], element_sizes)},
                        )
                    )
                else:
                    output.extend(choose(stmt["reference"], element_sizes))
                continue
            if stmt["op"] in {"decl", "buffer"}:
                element_sizes[stmt["name"]] = _element_bytes(stmt["type"])
            for field in ("body", "then", "else"):
                if isinstance(stmt.get(field), list):
                    stmt[field] = choose(stmt[field], element_sizes.copy())
            if stmt["op"] == "map" and stmt.get("independent") and samples:
                work = _work(stmt, element_sizes)
                bands, lines = (
                    _bands(work, samples)
                    if work["priced"]
                    else ([dict(minimum_elements=0, workers=1)], {})
                )
                report["regions"][stmt["site"]] = dict(
                    work=work, widths=bands, estimated_setup_and_ns_per_element=lines
                )
                report["workers"] = max(
                    report["workers"], *(row["workers"] for row in bands)
                )
            output.append(stmt)
        return output

    selected = deepcopy(module)
    for fn in selected["functions"]:
        fn["body"] = choose(
            fn["body"], {p["name"]: _element_bytes(p["type"]) for p in fn["params"]}
        )
    return selected, report
