"""F1 records build-process observations and takes a current snapshot of named files."""

import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import digest

from .process_tree import decode_process_tree, wrapper_mappings

_ENVIRONMENT = {
    "PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "LD_LIBRARY_PATH",
    "LANG",
}
_PREFIXES = ("LC_", "OMP_", "GOMP_", "KMP_", "MKL_", "OPENBLAS_", "BLIS_")
_DRIVER = re.compile(
    r"(?:[\w.+]+-)*(?:gcc|g\+\+|gfortran|cc|c\+\+|clang(?:\+\+)?)(?:-[0-9.]+)?"
)
_BUILD_TOOL = re.compile(
    r"(?:[\w.+]+-)*(?:as|ar|ranlib|ld(?:\.bfd|\.gold|\.lld)?|collect2|cc1(?:plus)?|f951)(?:-[0-9.]+)?"
)
_SOURCE = {
    ".c",
    ".cc",
    ".cp",
    ".cpp",
    ".cxx",
    ".c++",
    ".h",
    ".hpp",
    ".f",
    ".for",
    ".ftn",
    ".f77",
    ".f90",
    ".f95",
    ".f03",
    ".f08",
    ".f18",
    ".s",
}
_INSPECT_ENV = {"LC_ALL": "C", "PATH": os.defpath}


def _issue(facts, code, detail, **context):
    facts["issues"].append(dict(code=code, detail=detail, **context))


def _integer(value):
    if type(value) is int or isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    raise ValueError("expected an integer")


def _started(row):
    execution = row["execution"]
    argv = execution["arguments"]
    directory = execution["working_dir"]
    executable = execution["executable"]
    environment = execution.get("environment", {})
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(a, str) and "\0" not in a for a in argv)
        or not isinstance(executable, str)
        or not executable
        or "\0" in executable
        or not isinstance(directory, str)
        or "\0" in directory
        or not Path(directory).is_absolute()
        or not isinstance(environment, dict)
        or not all(isinstance(v, str) for v in environment.values())
    ):
        raise ValueError("invalid Bear execution record")
    pid, ppid = _integer(row["pid"]), _integer(row.get("ppid", 0))
    if pid <= 0 or ppid < 0:
        raise ValueError("invalid process identity")
    record = dict(
        executable=executable,
        arguments=argv,
        directory=directory,
        pid=pid,
        ppid=ppid,
        environment={
            k: v
            for k, v in environment.items()
            if k in _ENVIRONMENT or k.startswith(_PREFIXES)
        },
    )
    reporter = environment.get("INTERCEPT_REPORT_COMMAND")
    if isinstance(reporter, str) and "\0" not in reporter and os.path.isabs(reporter):
        record["instrumentation_reporter"] = reporter
    return record


def _invocations(raw, facts):
    starts, ends, invalid = {}, {}, set()
    for line, payload in enumerate(io.BytesIO(raw), 1):
        if not payload.strip():
            continue
        try:
            event = json.loads(payload)
            identifier = _integer(event["rid"])
            if not 0 <= identifier < 1 << 64:
                raise ValueError("invalid Bear process receipt identity")
            rid = str(identifier)
            timestamp = event["timestamp"]
            kinds = event.keys() & {"started", "terminated", "signalled"}
            if not isinstance(timestamp, str) or not timestamp or len(kinds) != 1:
                raise ValueError("expected one timestamped Bear 3 event")
            kind = kinds.pop()
            if kind == "started":
                value = dict(id=rid, started=timestamp, **_started(event[kind]))
                table = starts
            else:
                status = number = None
                if kind == "signalled":
                    number = _integer(event[kind]["number"])
                    # Bear's supervisor emits notifications for stop and continue
                    # independently of the process's eventual terminal status.
                    if number in (signal.SIGSTOP, signal.SIGCONT):
                        continue
                elif "status" in event[kind]:
                    status = _integer(event[kind]["status"])
                table, value = ends, (timestamp, status, number)
            if rid in table:
                invalid.add(rid)
                _issue(
                    facts,
                    "duplicate_build_event",
                    "ambiguous repeated process event",
                    id=rid,
                    line=line,
                )
            else:
                table[rid] = value
        except (KeyError, TypeError, ValueError, AttributeError):
            _issue(
                facts,
                "invalid_build_event",
                "expected a Bear 3 interception event",
                line=line,
            )
    for rid in sorted(ends.keys() - starts.keys()):
        _issue(
            facts,
            "missing_build_start",
            "process outcome has no corresponding start",
            id=rid,
        )
    for rid, row in starts.items():
        if rid in invalid:
            continue
        finished, status, number = ends.get(rid, (None, None, None))
        row.update(finished=finished, status=status, signal=number)
        if finished is None or status is None and number is None:
            _issue(
                facts,
                "missing_build_exit",
                "no exit status or terminating signal recorded",
                id=rid,
            )
        elif number is not None or status != 0:
            _issue(
                facts,
                "unsuccessful_build_command",
                "recorded command did not exit successfully",
                id=rid,
                status=status,
                signal=number,
            )
        facts["invocations"].append(row)
    if not starts:
        _issue(facts, "empty_build_capture", "no process start was observed")


def _timestamp_ns(value):
    """Decode Bear's timezone-qualified ISO clock without a floating point roundtrip."""
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?" r"(Z|[+-]\d{2}:\d{2})",
        value,
    )
    if match is None:
        raise ValueError("expected a timezone-qualified ISO process timestamp")
    seconds, fraction, zone = match.groups()
    stamp = datetime.fromisoformat(seconds + zone.replace("Z", "+00:00"))
    elapsed = stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (elapsed.days * 86400 + elapsed.seconds) * 10**9 + int(
        (fraction or "").ljust(9, "0")
    )


def _process_parents(facts):
    """Only a unique, closed observed lifetime can identify a process parent."""
    by_pid, intervals, invalid_pids = defaultdict(list), {}, set()
    for row in facts["invocations"]:
        try:
            start = _timestamp_ns(row["started"])
            finish = (
                _timestamp_ns(row["finished"]) if row["finished"] is not None else None
            )
            if finish is not None and finish < start:
                raise ValueError("process exit precedes its start")
            intervals[row["id"]] = (start, finish)
            by_pid[row["pid"]].append(row)
        except (TypeError, ValueError):
            invalid_pids.add(row["pid"])
            facts.setdefault("process_diagnostics", []).append(
                dict(code="invalid_process_timestamp", invocation_id=row["id"])
            )

    def parents(pid, timestamp, child=None):
        if pid in invalid_pids:
            return []
        return [
            row
            for row in by_pid[pid]
            if row["id"] != child
            and intervals[row["id"]][0] <= timestamp
            and (
                intervals[row["id"]][1] is None or timestamp <= intervals[row["id"]][1]
            )
        ]

    for row in facts["invocations"]:
        if row["id"] not in intervals:
            continue
        choices = parents(row["ppid"], intervals[row["id"]][0], row["id"])
        if len(choices) == 1 and intervals[choices[0]["id"]][1] is not None:
            row["parent_invocation_id"] = choices[0]["id"]
    return intervals, parents


def _process_event(event, limit):
    """Validate one audit intent; retaining a location does not prove its binding."""

    def integer(value, minimum=0):
        if type(value) is not int or not minimum <= value < 1 << 64:
            raise ValueError("invalid process observation integer")
        return value

    def text(value, *, nullable=False, absolute=False):
        if nullable and value is None:
            return value
        if (
            not isinstance(value, str)
            or "\0" in value
            or absolute
            and not os.path.isabs(value)
        ):
            raise ValueError("invalid process observation text or path")
        return value

    def unresolved(value):
        if not isinstance(value, list):
            raise ValueError("invalid unresolved process fields")
        return [text(item) for item in value]

    sequence = integer(event["sequence"])
    if event["kind"] != "PROCESS_LAUNCH_INTENT" or sequence >= limit:
        raise ValueError("invalid process observation kind or sequence")
    result = dict(
        kind=event["kind"],
        sequence=sequence,
        parent_pid=integer(event["parent_pid"], 1),
        thread_id=integer(event["thread_id"]),
        wall_time_ns=integer(event["wall_time_ns"]),
        monotonic_ns=integer(event["monotonic_ns"]),
        executable=text(event["executable"], nullable=True),
        cwd=text(event["cwd"], nullable=True, absolute=True),
        caller=None,
        unresolved=unresolved(event["unresolved"]),
    )
    child = event.get("child_pid")
    outcome = event.get("launch_outcome", "unobserved")
    if outcome not in {"unobserved", "started"}:
        raise ValueError("invalid observed launch outcome")
    if child is not None:
        child = integer(child, 1)
    if (outcome == "started") != (child is not None):
        raise ValueError("observed launch outcome disagrees with its child PID")
    result.update(child_pid=child, launch_outcome=outcome)
    arguments = event["arguments"]
    if arguments is not None and (not isinstance(arguments, list) or not arguments):
        raise ValueError("invalid process observation argument vector")
    result["arguments"] = (
        [text(arg) for arg in arguments] if arguments is not None else None
    )
    caller = event["caller"]
    if caller is not None:
        path = text(caller["path"])
        if (
            not path
            or os.path.isabs(path)
            or any(p in {"", ".", ".."} for p in path.split("/"))
        ):
            raise ValueError("caller path must be source-root-relative")
        location = caller["location"]
        if location["file"] != path:
            raise ValueError("caller location disagrees with its source path")
        coordinates = {"file": path, "line": integer(location["line"], 1)}
        for field in ("end_line", "column", "end_column"):
            value = location[field]
            coordinates[field] = None if value is None else integer(value, 1)
        if (
            coordinates["end_line"] is not None
            and coordinates["end_line"] < coordinates["line"]
        ):
            raise ValueError("caller location has a reversed line range")
        if type(caller["source_code_matches_snapshot"]) is not bool:
            raise ValueError("invalid caller code snapshot evidence")
        verified = caller["source_code_matches_snapshot"]
        for key in ("source_sha256", "code_sha256"):
            if caller[key] is None and not verified:
                continue
            if not isinstance(caller[key], str) or not re.fullmatch(
                "[0-9a-f]{64}", caller[key]
            ):
                raise ValueError("invalid caller content digest")
        result["caller"] = dict(
            path=path,
            file=text(caller["file"], absolute=True),
            location=coordinates,
            function=text(caller["function"]),
            qualname=text(caller["qualname"], nullable=not verified),
            first_line=integer(caller["first_line"], 1),
            bytecode_offset=integer(caller["bytecode_offset"]),
            source_sha256=caller["source_sha256"],
            code_sha256=caller["code_sha256"],
            source_code_matches_snapshot=caller["source_code_matches_snapshot"],
            unresolved=unresolved(caller["unresolved"]),
        )
    return result


def _process_observation(raw, facts, intervals, parents, tree=None):
    """Join intent and execution only when the same capture proves a unique match."""
    result = dict(sha256=digest(raw), bindings=[], unresolved=[], diagnostics=[])
    if tree is not None:
        result["process_tree"] = {
            key: tree[key] for key in ("provider", "sha256", "diagnostics", "coverage")
        }
    facts["process_observation"] = result
    try:
        envelope = json.loads(raw)
        if (
            envelope["schema"] != "scar.process-observation"
            or type(envelope["schema_version"]) is not int
            or envelope["schema_version"] != 1
            or envelope["provider"] != "python-audit"
            or not isinstance(envelope["events"], list)
            or not isinstance(envelope["source_root"], str)
            or "\0" in envelope["source_root"]
            or not os.path.isabs(envelope["source_root"])
            or os.path.normpath(envelope["source_root"]) != facts["root"]
        ):
            raise ValueError("unsupported process observation envelope or source root")
        for name in ("collector_pid", "event_limit", "dropped_events"):
            value = envelope[name]
            if (
                type(value) is not int
                or not (1 if name == "collector_pid" else 0) <= value < 1 << 64
            ):
                raise ValueError("invalid process observation envelope count")
        interpreter = envelope["interpreter"]
        for key in ("implementation", "version", "executable"):
            if (
                not isinstance(interpreter[key], str)
                or not interpreter[key]
                or "\0" in interpreter[key]
            ):
                raise ValueError("invalid process observation interpreter")
        if not os.path.isabs(interpreter["executable"]):
            raise ValueError("interpreter executable must be absolute")
        result.update(
            {
                k: envelope[k]
                for k in (
                    "schema",
                    "schema_version",
                    "provider",
                    "collector_pid",
                    "source_root",
                    "event_limit",
                    "dropped_events",
                )
            }
        )
        result["interpreter"] = {
            k: interpreter[k] for k in ("implementation", "version", "executable")
        }
        result["events"] = []
        for event in envelope["events"]:
            try:
                result["events"].append(_process_event(event, envelope["event_limit"]))
            except (KeyError, TypeError, ValueError, AttributeError):
                result["diagnostics"].append(dict(code="invalid_process_intent"))
        if len(envelope["events"]) > envelope["event_limit"]:
            raise ValueError("process observation exceeds its declared event limit")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        result["events"] = []
        result["diagnostics"].append(
            dict(code="invalid_process_observation", detail=str(error))
        )
        return

    def key(row, process=False):
        if (
            row["executable"] is None
            or row["arguments"] is None
            or (row["cwd"] if not process else row["directory"]) is None
            or not process
            and row["child_pid"] is None
        ):
            return None
        return (
            row["ppid"] if process else row["parent_pid"],
            row["pid"] if process else row["child_pid"],
            row["executable"],
            tuple(row["arguments"]),
            row["directory"] if process else row["cwd"],
        )

    invocations, proposals = defaultdict(list), []
    sequences = Counter(event["sequence"] for event in result["events"])
    for row in facts["invocations"]:
        # Re-keying uses directory only; do not equate argv[0] with executable.
        invocations[key(row, True)].append(row)
    for event in result["events"]:
        sequence, match = event["sequence"], key(event)
        choices = invocations.get(match, []) if match is not None else []
        reason = "no_matching_execution"
        parent = parents(event["parent_pid"], event["wall_time_ns"])
        if result["diagnostics"]:
            reason = "invalid_observation_events"
        elif result["dropped_events"]:
            reason = "incomplete_intent_observation"
        elif sequences[sequence] != 1:
            reason = "duplicate_intent_sequence"
        elif event["launch_outcome"] != "started" or event["child_pid"] is None:
            reason = "unobserved_child_pid"
        elif match is None:
            reason = "unresolved_launch_key"
        elif len(parent) != 1 or intervals[parent[0]["id"]][1] is None:
            reason = "unproven_parent_lifetime"
        else:
            eligible = [
                (child, None)
                for child in choices
                if child["id"] in intervals
                and event["wall_time_ns"]
                <= intervals[child["id"]][0]
                <= intervals[parent[0]["id"]][1]
                and child.get("parent_invocation_id") == parent[0]["id"]
            ]
            if tree is not None:
                eligible.extend(
                    wrapper_mappings(
                        tree, event, parent[0], facts["invocations"], intervals
                    )
                )
            if len(eligible) == 1:
                child, mapping = eligible[0]
                proposals.append(
                    dict(
                        sequence=sequence,
                        invocation_id=child["id"],
                        parent_invocation_id=parent[0]["id"],
                        evidence=(
                            "observed_child_pid_and_kernel_wrapper_mapping"
                            if mapping is not None
                            else "observed_child_pid_and_process_identity"
                        ),
                        **(
                            {"instrumentation_mapping": mapping}
                            if mapping is not None
                            else {}
                        ),
                    )
                )
                continue
            reason = (
                "ambiguous_execution_lifetime"
                if eligible
                else "inconsistent_process_time" if choices else "no_matching_execution"
            )
        result["unresolved"].append(
            dict(
                sequence=sequence,
                reason=reason,
                candidates=[row["id"] for row in choices],
            )
        )
    observations = Counter(row["invocation_id"] for row in proposals)
    for binding in proposals:
        if observations[binding["invocation_id"]] == 1:
            result["bindings"].append(binding)
        else:
            result["unresolved"].append(
                dict(
                    sequence=binding["sequence"],
                    reason="duplicate_child_process_observation",
                    candidates=[binding["invocation_id"]],
                )
            )


class _Files:
    def __init__(self, facts):
        self.facts = facts
        self.missing = set()

    def capture(self, path, kind, *, required=True):
        lexical = os.path.abspath(path)
        records = self.facts["files"]
        aliases = self.facts["path_aliases"]
        try:
            path = Path(path).resolve()
        except (OSError, RuntimeError, ValueError) as error:
            if required:
                _issue(
                    self.facts,
                    "unavailable_build_file",
                    str(error),
                    path=lexical,
                    kind=kind,
                )
            return lexical
        key = str(path)
        if aliases.get(lexical, key) != key or key != lexical and lexical in records:
            _issue(
                self.facts,
                "ambiguous_build_path",
                "path spelling identifies different files",
                path=lexical,
            )
        elif key != lexical:
            aliases[lexical] = key
        if key in records:
            if kind == "executable":
                records[key]["kind"] = kind
            return key
        if key in self.missing:
            return key
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("not a regular file")
            h, size = hashlib.sha256(), 0
            text = bytearray() if kind in ("dependency", "response") else None
            with path.open("rb") as source:
                before = os.fstat(source.fileno())
                for block in iter(lambda: source.read(1 << 20), b""):
                    h.update(block)
                    size += len(block)
                    if text is not None:
                        text.extend(block)
                after = os.fstat(source.fileno())
            if before.st_mtime_ns != after.st_mtime_ns or size != after.st_size:
                raise ValueError("file changed during capture")
            records[key] = dict(sha256=h.hexdigest(), size=size, kind=kind)
            if text is not None:
                records[key]["text"] = text.decode("utf-8")
        except (OSError, ValueError) as error:
            if required:
                self.missing.add(key)
                _issue(
                    self.facts,
                    "unavailable_build_file",
                    str(error),
                    path=key,
                    kind=kind,
                )
        return key

    def utility(self, name):
        path = shutil.which(name)
        if path:
            key = self.capture(path, "tool")
            if key in self.facts["files"]:
                try:
                    result = subprocess.run(
                        [key, "--version"],
                        capture_output=True,
                        env=_INSPECT_ENV,
                        check=True,
                    )
                    line = result.stdout.decode().splitlines()[0]
                    version = line.removeprefix("bear ") if name == "bear" else line
                    self.facts["tools"][name] = dict(
                        path=key,
                        version=version,
                        sha256=self.facts["files"][key]["sha256"],
                    )
                    return key, version
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    pass
        _issue(
            self.facts, "unavailable_build_inspector", f"cannot identify local {name}"
        )
        return None, None


def _compilations(raw, bear, files):
    facts = files.facts
    try:
        with TemporaryDirectory(prefix="scar-bear-") as folder:
            events, database = (
                Path(folder) / "events.json",
                Path(folder) / "commands.json",
            )
            events.write_bytes(raw)
            subprocess.run(
                [bear, "citnames", "--input", str(events), "--output", str(database)],
                cwd=folder,
                env=_INSPECT_ENV,
                check=True,
                capture_output=True,
            )
            rows = json.loads(database.read_bytes())
        if not isinstance(rows, list):
            raise ValueError("expected a compilation database")
    except (OSError, ValueError, subprocess.SubprocessError):
        _issue(
            facts,
            "build_database_decode",
            "Bear citnames could not decode the supplied interception",
        )
        return
    for index, row in enumerate(rows):
        if not (
            isinstance(row, dict)
            and all(
                isinstance(row.get(k), str) and row[k]
                for k in ("directory", "file", "output")
            )
            and Path(row["directory"]).is_absolute()
            and isinstance(row.get("arguments"), list)
            and row["arguments"]
            and all(isinstance(v, str) for v in row["arguments"])
        ):
            _issue(
                facts,
                "invalid_build_compilation",
                "expected directory, file, output and arguments",
                row=index,
            )
            continue
        cwd = Path(row["directory"])
        record = {k: row[k] for k in ("arguments", "directory", "file", "output")}
        record["canonical_file"] = files.capture(cwd / row["file"], "source")
        files.capture(cwd / row["output"], "object")
        facts["compilations"].append(record)


def _named_artifact(token):
    path = Path(token)
    suffix = path.suffix.lower()
    if suffix in _SOURCE:
        return "source"
    if suffix in (".o", ".obj", ".lo"):
        return "object"
    if suffix == ".a" or re.search(r"\.so(?:\.\d+)*$", path.name):
        return "library"
    if suffix in (".d", ".rsp"):
        return "dependency" if suffix == ".d" else "response"
    return None


def _command_files(row, files):
    cwd, argv = Path(row["directory"]), row["arguments"]
    exe = row["executable"]
    if "/" in exe:
        row["canonical_executable"] = files.capture(cwd / exe, "tool")
    else:
        search = os.pathsep.join(
            str(cwd / entry)
            for entry in row["environment"].get("PATH", "").split(os.pathsep)
        )
        resolved = shutil.which(exe, path=search)
        if resolved:
            row["canonical_executable"] = files.capture(resolved, "tool")
        else:
            row["canonical_executable"] = None
            _issue(
                files.facts,
                "unresolved_build_tool",
                "executable unavailable on recorded PATH",
                id=row["id"],
                executable=exe,
            )
    driver = _DRIVER.fullmatch(Path(exe).name)
    if not driver and not _BUILD_TOOL.fullmatch(Path(exe).name):
        return None
    output = None
    args = iter(argv[1:])
    for arg in args:
        if arg in ("-MT", "-MQ", "-D", "-U", "-x", "-dumpbase", "-dumpbase-ext"):
            next(args, None)
        elif arg in ("-o", "-MF"):
            value = next(args, None)
            if not value:
                _issue(
                    files.facts,
                    "missing_build_operand",
                    f"missing operand for {arg}",
                    id=row["id"],
                )
            elif arg == "-o":
                output = value
            elif value != "-":
                files.capture(cwd / value, "dependency")
        elif arg.startswith("-MF") and len(arg) > 3 and arg[3:] != "-":
            files.capture(cwd / arg[3:], "dependency")
        elif driver and arg.startswith("-o") and len(arg) > 2:
            output = arg[2:]
        elif arg.startswith("@") or arg.startswith("-Wl,@"):
            files.capture(cwd / arg.partition("@")[2], "response")
        elif not arg.startswith("-") and (kind := _named_artifact(arg)):
            # Lower-level tools also name temporary compiler files which may have
            # legitimately disappeared. Their complete argv is already retained.
            files.capture(cwd / arg, kind, required=bool(driver))
    if not driver or not output or output == "-" or row["status"] != 0:
        return None
    if "-fsyntax-only" in argv:
        return None
    if "-M" in argv or "-MM" in argv:
        kind = "dependency"
    elif "-S" in argv or "-E" in argv:
        kind = "source"
    elif "-c" in argv or "-r" in argv:
        kind = "object"
    else:
        kind = "library" if "-shared" in argv else "executable"
    path = files.capture(cwd / output, kind)
    return (
        path
        if kind in ("library", "executable") and path in files.facts["files"]
        else None
    )


def _libraries(outputs, files):
    tool, _ = files.utility("readelf")
    if not tool:
        return
    for output in sorted(outputs):
        try:
            result = subprocess.run(
                [tool, "-W", "-d", "--", output],
                env=_INSPECT_ENV,
                check=True,
                capture_output=True,
            )
            info = dict(needed=[], rpath=[], runpath=[], observed_at="capture")
            for tag, text in re.findall(
                r"\((NEEDED|RPATH|RUNPATH)\)[^\n]*?\[([^\n]*)\]", result.stdout.decode()
            ):
                info[tag.lower()].extend([text] if tag == "NEEDED" else text.split(":"))
            files.facts["libraries"][output] = info
        except (OSError, ValueError, subprocess.SubprocessError):
            _issue(
                files.facts,
                "unreadable_elf_dependencies",
                "readelf could not inspect declared ELF dependencies",
                path=output,
            )


def capture_build(
    raw: bytes,
    *,
    root: Path,
    provider: str,
    process_observation: bytes | None = None,
    process_tree: bytes | None = None,
) -> dict:
    """Decode an already locked capture without replaying any observed command."""
    facts = dict(
        provider=provider,
        trace_sha256=digest(raw),
        root=str(root.resolve(strict=True)),
        invocations=[],
        compilations=[],
        files={},
        path_aliases={},
        libraries={},
        tools={},
        issues=[],
    )
    if provider != "bear-3-intercept":
        _issue(
            facts, "unsupported_build_provider", "expected Bear 3 interception JSONL"
        )
        return facts
    _invocations(raw, facts)
    intervals, parents = _process_parents(facts)
    if process_observation is not None:
        tree = decode_process_tree(process_tree) if process_tree is not None else None
        _process_observation(process_observation, facts, intervals, parents, tree)
    files = _Files(facts)
    if process_tree is not None:
        files.utility("strace")
    for row in facts["invocations"]:
        if reporter := row.get("instrumentation_reporter"):
            files.capture(reporter, "tool")
    bear, version = files.utility("bear")
    if bear and re.fullmatch(r"3\.\d+\.\d+(?:[-+][\w.-]+)?", version):
        _compilations(raw, bear, files)
        if not facts["compilations"]:
            _issue(
                facts,
                "no_observed_compilation",
                "no compilation profile was recovered from this build",
            )
    elif bear:
        _issue(
            facts,
            "unsupported_build_decoder",
            "Bear 3 citnames is required",
            version=version,
        )
    outputs = {
        path for row in facts["invocations"] if (path := _command_files(row, files))
    }
    if outputs:
        _libraries(outputs, files)
    return facts
