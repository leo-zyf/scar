"""Execute gate criteria from a manifest and write evidence for each test case.

Usage: python scripts/run_gate.py gates/cpu.json --out /tmp/scar-gates.json
Reports bind both the Git revision and the tested source hashes. A missing or
skipped test fails its criterion; one successful suite cannot fill every row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "-c", "-o", "--exclude-standard", "--",
         "scar", "tests", "scripts", "gates", ".github/workflows", "pyproject.toml", "sitecustomize.py"],
        cwd=ROOT,
    ).decode(errors="surrogateescape")
    digests = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in sorted(set(files.split("\0")) - {""})
        if (path := ROOT / name).is_file()
    }
    combined = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return combined, digests


def _run_tests(nodes, evidence_dir, junit):
    command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q",
               f"--basetemp={evidence_dir}", *nodes, f"--junitxml={junit}"]
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    cases = []
    if junit.is_file():
        for case in ET.parse(junit).iter("testcase"):
            failed = any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
            cases.append({"class": case.get("classname", ""), "name": case.get("name", ""),
                          "status": "FAIL" if failed else "PASS"})
    return result, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--evidence-dir", type=Path, help="new directory for retained test artifacts")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    criteria = manifest.get("criteria")
    if (not isinstance(criteria, list) or not criteria
            or any(not isinstance(row, dict) or not row.get("id")
                   or type(row.get("isolated", False)) is not bool
                   or not isinstance(row.get("tests"), list) or not row["tests"]
                   or any(not isinstance(node, str) or not node for node in row["tests"])
                   for row in criteria)
            or len({row["id"] for row in criteria}) != len(criteria)):
        code_hash, hashes = fingerprint()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report = {"schema": "scar.gate-report", "schema_version": 2,
                  "gate": manifest.get("gate"), "status": "FAIL",
                  "revision": revision, "source_sha256": code_hash, "tested_files": hashes,
                  "errors": ["manifest requires nonempty unique criteria and test selectors"]}
        out = args.out or ROOT / "artifacts/reports/gates/invalid-manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        out.with_suffix(".md").write_text(
            f"# Gate manifest failure\n\nStatus: **FAIL**\n\nRevision: `{revision}`\n\n"
            + "\n".join(report["errors"]) + "\n")
        print(json.dumps(report))
        return 1
    nodes = {node for row in criteria for node in row["tests"]}
    isolated = {node for row in criteria if row.get("isolated") for node in row["tests"]}
    batches = (nodes - isolated, isolated)
    code_hash, hashes = fingerprint()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="scar-gate-") as temp:
        evidence_dir = Path(temp) / "pytest"
        if args.evidence_dir is not None:
            evidence_dir = args.evidence_dir.absolute()
            evidence_dir.mkdir(parents=True, exist_ok=False)
        runs = []
        cases = []
        for index, batch in enumerate(batches):
            if batch:
                result, collected = _run_tests(sorted(batch), evidence_dir / str(index),
                                               Path(temp) / f"results-{index}.xml")
                runs.append(result)
                cases.extend(collected)
    rows = []
    for row in criteria:
        evidence = []
        for node in row["tests"]:
            parts = node.split("::")
            file = parts[0]
            test = parts[-1] if len(parts) > 1 else ""
            classname = file.removesuffix(".py").replace("/", ".")
            if len(parts) > 2:
                classname += "." + ".".join(parts[1:-1])
            matched = [case for case in cases if (
                (case["class"] == classname and
                 (not test or case["name"] == test or case["name"].startswith(test + "[")))
                or (not test and case["class"].startswith(classname + "."))
                or (test and case["class"] == classname + "." + test))]
            evidence.append({"node": node, "cases": matched,
                             "status": "PASS" if matched and all(case["status"] == "PASS"
                                                                  for case in matched) else "FAIL"})
        rows.append({**row, "evidence": evidence,
                     "status": "PASS" if evidence and all(item["status"] == "PASS"
                                                           for item in evidence) else "FAIL"})
    unchanged = fingerprint()[0] == code_hash
    returncode = next((run.returncode for run in runs if run.returncode != 0), 0)
    passed = returncode == 0 and unchanged and all(row["status"] == "PASS" for row in rows)
    stdout = "\n".join(run.stdout for run in runs)
    stderr = "\n".join(run.stderr for run in runs)
    report = {"schema": "scar.gate-report", "schema_version": 2,
              "gate": manifest["gate"], "status": "PASS" if passed else "FAIL",
              "revision": revision, "worktree_status": dirty, "source_sha256": code_hash,
              "tested_files": hashes, "source_unchanged_during_test": unchanged,
              "scope": manifest["scope"], "limitations": manifest.get("limitations", []),
              "criteria": rows, "command": runs[0].args[:-1],
              "commands": [run.args[:-1] for run in runs], "returncode": returncode,
              "duration_seconds": round(time.monotonic() - started, 3),
              "stdout": stdout, "stderr": stderr, "python": sys.version}
    out = args.out or ROOT / "artifacts/reports/gates" / f"{manifest['gate'].lower()}-verified.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    out.with_suffix(".md").write_text(
        f"# {manifest['gate']} independent criteria report\n\n"
        f"Status: **{report['status']}**\n\nRevision: `{revision}`\n\n"
        f"Source digest: `{code_hash}`\n\n"
        "| Criterion | Status |\n| --- | --- |\n" +
        "".join(f"| {row['id']} | {row['status']} |\n" for row in rows) +
        "\nScope: " + manifest["scope"] + "\n\n" +
        "\n".join("- " + text for text in manifest.get("limitations", [])) + "\n")
    print(json.dumps({"gate": manifest["gate"], "status": report["status"],
                      "report": str(out), "tests": len(cases)}))
    if not passed:
        print(stdout)
        print(stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
