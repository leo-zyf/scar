"""Decode locked process-only strace receipts without running or inspecting a target."""

from collections import defaultdict
import os
import re

from scar.contract import digest

_CALLS = {"execve", "execveat", "clone", "clone3", "fork", "vfork"}
_ESCAPES = dict(zip('abfnrtv\\"', b'\a\b\f\n\r\t\v\\"'))
_CLONE_FLAGS = {
    "CLONE_VM",
    "CLONE_FS",
    "CLONE_FILES",
    "CLONE_SIGHAND",
    "CLONE_PIDFD",
    "CLONE_PTRACE",
    "CLONE_VFORK",
    "CLONE_PARENT",
    "CLONE_THREAD",
    "CLONE_NEWNS",
    "CLONE_SYSVSEM",
    "CLONE_SETTLS",
    "CLONE_PARENT_SETTID",
    "CLONE_CHILD_CLEARTID",
    "CLONE_DETACHED",
    "CLONE_UNTRACED",
    "CLONE_CHILD_SETTID",
    "CLONE_NEWCGROUP",
    "CLONE_NEWUTS",
    "CLONE_NEWIPC",
    "CLONE_NEWUSER",
    "CLONE_NEWPID",
    "CLONE_NEWNET",
    "CLONE_IO",
    "CLONE_CLEAR_SIGHAND",
    "CLONE_INTO_CGROUP",
    "CLONE_NEWTIME",
    "SIGCHLD",
    "0",
}


def _nanoseconds(stamp):
    seconds, _, fraction = stamp.partition(".")
    if not seconds.isdigit() or not fraction.isdigit() or len(fraction) > 9:
        raise ValueError("invalid decimal process timestamp")
    return int(seconds) * 10**9 + int(fraction.ljust(9, "0"))


def _quoted(text, start):
    """Decode strace C byte strings, including octal filesystem byte escapes."""
    if text[start : start + 1] != '"':
        raise ValueError("expected a complete quoted byte string")
    result, position = bytearray(), start + 1
    while position < len(text):
        value = text[position]
        position += 1
        if value == '"':
            if 0 in result:
                raise ValueError("process string contains a NUL byte")
            return os.fsdecode(bytes(result)), position
        if value != "\\":
            result.extend(os.fsencode(value))
            continue
        if position == len(text):
            break
        escape = text[position]
        position += 1
        if escape in _ESCAPES:
            result.append(_ESCAPES[escape])
        elif escape in "01234567":
            match = re.match(r"[0-7]{0,2}", text[position:])
            digits = escape + match[0]
            result.append(int(digits, 8))
            position += len(match[0])
        elif escape == "x" and re.fullmatch(
            r"[0-9a-fA-F]{2}", text[position : position + 2]
        ):
            result.append(int(text[position : position + 2], 16))
            position += 2
        else:
            raise ValueError("unsupported process string escape")
    raise ValueError("truncated process string")


def _execution(arguments, kind):
    position = 0
    if kind == "execveat":
        prefix = re.match(r"(?:AT_FDCWD|-?\d+),\s*", arguments)
        if prefix is None or not re.search(r",\s*0$", arguments):
            raise ValueError("execveat descriptor or flags are unresolved")
        position = prefix.end()
    executable, position = _quoted(arguments, position)
    match = re.match(r",\s*\[", arguments[position:])
    if match is None:
        raise ValueError("process argument vector is not fully captured")
    position += match.end()
    argv = []
    while True:
        position += len(arguments[position:]) - len(arguments[position:].lstrip())
        if arguments[position : position + 1] == "]":
            position += 1
            break
        value, position = _quoted(arguments, position)
        argv.append(value)
        separator = re.match(r"\s*(,|\])\s*", arguments[position:])
        if separator is None:
            raise ValueError("truncated process argument vector")
        position += separator.end()
        if separator[1] == "]":
            break
    if not executable or not argv or not arguments[position:].lstrip().startswith(","):
        raise ValueError("invalid process execution arguments")
    if kind == "execveat" and not os.path.isabs(executable):
        raise ValueError("relative execveat identity needs descriptor state")
    return executable, argv


def decode_process_tree(raw):
    """Read the fixed strace-process-v1 format; undecoded calls remain boundaries."""
    tree = dict(
        provider="strace-process-v1",
        sha256=digest(raw),
        executions=[],
        forks=[],
        diagnostics=[],
        uncertain_pids=set(),
    )
    pending, threads, failed = {}, 0, 0

    def issue(code, line, pid=None, *, uncertain=True):
        tree["diagnostics"].append(dict(code=code, line=line, pid=pid))
        if uncertain:
            tree["uncertain_pids"].add(pid)

    for number, line in enumerate(
        raw.decode("utf-8", "surrogateescape").splitlines(), 1
    ):
        header = re.fullmatch(r"(\d+)\s+(\d+\.\d+)\s+(.+)", line)
        if header is None:
            if line.strip():
                issue("unrecognized_process_record", number)
            continue
        pid, stamp, body = header.groups()
        pid = int(pid)
        try:
            clock = _nanoseconds(stamp)
            if pid <= 0:
                raise ValueError("invalid task PID")
        except ValueError:
            issue("invalid_process_record_clock", number, pid)
            continue
        if body.startswith(("--- ", "+++ ")):
            continue
        resumed = re.match(r"<\.\.\. (\w+) resumed>(.*)", body)
        start = clock
        if resumed is not None:
            previous = pending.pop(pid, None)
            if previous is None or previous[0] != resumed[1] or previous[1] > clock:
                issue("unmatched_resumed_process_call", number, pid)
                continue
            _, start, prefix = previous
            body = prefix + resumed[2]
        elif body.endswith("<unfinished ...>"):
            call = re.match(r"(\w+)\(", body)
            if call is None or call[1] not in _CALLS or pid in pending:
                issue("ambiguous_unfinished_process_call", number, pid)
                pending.pop(pid, None)
            else:
                pending[pid] = (
                    call[1],
                    clock,
                    body[: -len("<unfinished ...>")].rstrip(),
                )
            continue
        call = re.fullmatch(r"(\w+)\((.*)\)\s+=\s+(-?\d+)(?:\s+.*)?", body)
        if call is None or call[1] not in _CALLS:
            issue("unsupported_process_call", number, pid)
            continue
        kind, arguments, returned = call.groups()
        returned = int(returned)
        if returned < 0:
            failed += 1
            issue("failed_process_call", number, pid, uncertain=False)
            continue
        try:
            if kind.startswith("exec"):
                if returned != 0:
                    raise ValueError("unexpected execution return value")
                executable, argv = _execution(arguments, kind)
                tree["executions"].append(
                    dict(
                        pid=pid,
                        executable=executable,
                        arguments=argv,
                        time_ns=start,
                        completed_time_ns=clock,
                    )
                )
            else:
                if returned <= 0:
                    raise ValueError("process creation did not return a child PID")
                if "..." in arguments:
                    raise ValueError("truncated process creation arguments")
                if kind.startswith("clone"):
                    flags = re.search(r"\bflags=([^,}]+)", arguments)
                    if flags is None or not re.fullmatch(
                        r"(?:[A-Z][A-Z0-9_]*|0)(?:\|[A-Z][A-Z0-9_]*)*", flags[1]
                    ):
                        raise ValueError("unresolved clone flags")
                    modes = set(flags[1].split("|"))
                    if modes - _CLONE_FLAGS:
                        raise ValueError("unsupported clone flags")
                    if "CLONE_THREAD" in modes:
                        threads += 1
                        continue
                    if modes & {"CLONE_PARENT", "CLONE_NEWPID"}:
                        raise ValueError(
                            "clone does not establish a direct PID-namespace parent"
                        )
                elif arguments.strip():
                    raise ValueError("unexpected fork arguments")
                tree["forks"].append(
                    dict(
                        parent_pid=pid,
                        child_pid=returned,
                        syscall=kind,
                        time_ns=start,
                        completed_time_ns=clock,
                    )
                )
        except ValueError as error:
            issue(str(error), number, pid)
    for pid in pending:
        issue("unfinished_process_call", len(raw.splitlines()), pid)
    tree["coverage"] = dict(
        scope="process_syscalls_only",
        successful_execs=len(tree["executions"]),
        process_forks=len(tree["forks"]),
        thread_clones_excluded=threads,
        failed_calls=failed,
        undecoded_calls=sum(
            row["code"] != "failed_process_call" for row in tree["diagnostics"]
        ),
        thread_group_resolution="not_collected",
    )
    return tree


def wrapper_mappings(tree, intent, parent, invocations, intervals):
    """Prove the one direct Bear supervisor hop using explicit kernel identities."""
    reporter = parent.get("instrumentation_reporter")
    published = intent["child_pid"]
    if not reporter or published is None or None in tree["uncertain_pids"]:
        return []
    if published in tree["uncertain_pids"]:
        return []
    executions = defaultdict(list)
    for row in tree["executions"]:
        executions[row["pid"]].append(row)
    matches = []
    for child in invocations:
        if (
            child.get("instrumentation_reporter") != reporter
            or child["ppid"] != intent["parent_pid"]
            or child["executable"] != intent["executable"]
            or child["arguments"] != intent["arguments"]
            or child["directory"] != intent["cwd"]
            or child.get("parent_invocation_id") != parent["id"]
            or child["id"] not in intervals
            or child["pid"] in tree["uncertain_pids"]
        ):
            continue
        child_start = intervals[child["id"]][0]
        finish = intervals[parent["id"]][1]
        if finish is None or not intent["wall_time_ns"] <= child_start <= finish:
            continue
        for fork in tree["forks"]:
            if fork["parent_pid"] != published or fork["child_pid"] != child["pid"]:
                continue
            prior = [
                e for e in executions[published] if e["time_ns"] <= fork["time_ns"]
            ]
            if not prior:
                continue
            wrapper = max(prior, key=lambda e: e["time_ns"])
            argv = wrapper["arguments"]
            if (
                wrapper["executable"] != reporter
                or argv[0] != reporter
                or len(argv) < 7
                or argv[1] != "--destination"
            ):
                continue
            options = argv[3:]
            if options[:1] == ["--verbose"]:
                options = options[1:]
            if options != [
                "--execute",
                intent["executable"],
                "--",
                *intent["arguments"],
            ]:
                continue
            for target in executions[child["pid"]]:
                if (
                    target["executable"] != child["executable"]
                    or target["arguments"] != child["arguments"]
                    or not intent["wall_time_ns"]
                    <= wrapper["time_ns"]
                    <= wrapper["completed_time_ns"]
                    <= fork["time_ns"]
                    <= target["time_ns"]
                    <= child_start
                ):
                    continue
                matches.append(
                    (
                        child,
                        dict(
                            provider=tree["provider"],
                            sha256=tree["sha256"],
                            published_pid=published,
                            reported_pid=child["pid"],
                            wrapper_executable=reporter,
                            wrapper_exec_time_ns=wrapper["time_ns"],
                            fork_time_ns=fork["time_ns"],
                            target_exec_time_ns=target["time_ns"],
                        ),
                    )
                )
    return matches
