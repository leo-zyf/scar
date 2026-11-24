"""Observe CPython subprocess launch intents and bind callers to source snapshots."""

import argparse
import atexit
import builtins
from hashlib import sha256
from importlib.machinery import SourceFileLoader
from itertools import islice
import json
import marshal
import os
from pathlib import Path, PurePath
import platform
import sys
from tempfile import NamedTemporaryFile
import threading
import time
from types import CodeType, ModuleType
import warnings

_PATH_TYPES = (str, bytes, type(Path()), type(PurePath()))


def _path_text(value):
    # Never invoke arbitrary application-defined __str__ or __fspath__ methods.
    if type(value) not in _PATH_TYPES:
        return None
    value = os.fsdecode(value)
    return None if "\0" in value else value


def _position(frame, relative):
    line, end_line, column, end_column = frame.f_lineno, None, None, None
    offset = frame.f_lasti
    positions = getattr(frame.f_code, "co_positions", None)
    if positions is not None and sys.implementation.name == "cpython" and offset >= 0:
        span = next(islice(positions(), offset // 2, None), None)
        if span is not None:
            start, end_line, left, right = span
            line = start if start is not None else line
            column = None if left is None else left + 1
            end_column = None if right is None else right + 1
    return dict(
        file=relative,
        line=line,
        end_line=end_line,
        column=column,
        end_column=end_column,
    )


def _code_hash(code):
    return sha256(marshal.dumps(code, 2)).hexdigest()


def _compiled_codes(data, filename):
    # This second compilation is evidence collection. Application warnings were
    # already handled when its actual code was compiled.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pending = [
            compile(
                data, filename, "exec", dont_inherit=True, optimize=sys.flags.optimize
            )
        ]
    index = {}
    while pending:
        code = pending.pop()
        key = (getattr(code, "co_qualname", code.co_name), code.co_firstlineno)
        index.setdefault(key, []).append(code)
        pending.extend(item for item in code.co_consts if isinstance(item, CodeType))
    return index


class _Capture:
    def __init__(self, root, limit):
        self.root = root
        self.initial_directory = os.getcwd()
        self.collector_pid = os.getpid()
        self.limit = limit
        self.dropped = 0
        self.events = []
        self.sources = {}
        self.callers = []
        self.pending = {}
        self.lock = threading.RLock()
        self.finished = False
        self.tool = None
        self.interpreter = dict(
            implementation=sys.implementation.name,
            version=platform.python_version(),
            executable=sys.executable,
        )

    def monitor_returns(self):
        monitor = getattr(sys, "monitoring", None)
        if monitor is None or sys.implementation.name != "cpython":
            return
        import subprocess

        self.init_code = subprocess.Popen.__init__.__code__
        self.execute_code = subprocess.Popen._execute_child.__code__
        # Prefer the two slots without a conventional debugger/profiler owner.
        for tool in (3, 4, 2, 1, 0, 5):
            if monitor.get_tool(tool) is not None:
                continue
            try:
                monitor.use_tool_id(tool, "scar.process-observation")
            except ValueError:
                continue
            self.tool = tool
            try:
                monitor.register_callback(tool, monitor.events.PY_RETURN, self.returned)
                monitor.set_local_events(tool, self.init_code, monitor.events.PY_RETURN)
            except Exception:
                self.stop_monitor()
            break

    def stop_monitor(self):
        tool, self.tool = self.tool, None
        if tool is None:
            return
        monitor = sys.monitoring
        if monitor.get_tool(tool) == "scar.process-observation":
            try:
                monitor.set_local_events(tool, self.init_code, 0)
            finally:
                try:
                    monitor.register_callback(tool, monitor.events.PY_RETURN, None)
                finally:
                    monitor.free_tool_id(tool)

    def returned(self, code, offset, result):
        try:
            if self.finished or os.getpid() != self.collector_pid:
                return
            frame = sys._getframe(1)
            if frame.f_code is not self.init_code or code is not self.init_code:
                return
            owner = frame.f_locals.get("self")
            with self.lock:
                record = self.pending.pop(id(owner), None)
                if record is not None:
                    pid = object.__getattribute__(owner, "__dict__").get("pid")
                    if type(pid) is int and pid > 0:
                        record.update(child_pid=pid, launch_outcome="started")
        except Exception:
            # A missing return witness leaves the original intent unobserved.
            pass

    def caller(self):
        frame = sys._getframe(1)
        try:
            while frame is not None:
                code = frame.f_code
                filename = code.co_filename
                if (
                    frame.f_globals is not globals()
                    and filename
                    and not filename.startswith("<")
                ):
                    absolute = os.path.abspath(
                        os.path.join(self.initial_directory, filename)
                    )
                    path = Path(absolute).resolve()
                    if path.is_relative_to(self.root):
                        relative = path.relative_to(self.root).as_posix()
                        location = _position(frame, relative)
                        unresolved = []
                        if location["column"] is None or location["end_column"] is None:
                            unresolved.append("instruction_columns_unavailable")
                        if not hasattr(code, "co_qualname"):
                            unresolved.append("runtime_qualname_unavailable")
                        caller = dict(
                            path=relative,
                            file=absolute,
                            location=location,
                            function=code.co_name,
                            qualname=getattr(code, "co_qualname", None),
                            first_line=code.co_firstlineno,
                            bytecode_offset=frame.f_lasti,
                            source_sha256=None,
                            code_sha256=None,
                            source_code_matches_snapshot=False,
                            unresolved=unresolved,
                        )
                        with self.lock:
                            first = path not in self.sources
                            if first:
                                self.sources[path] = None
                        if first:
                            try:
                                data = path.read_bytes() if path.is_file() else None
                            except OSError:
                                data = None
                            with self.lock:
                                self.sources[path] = data
                        with self.lock:
                            self.callers.append((caller, code, path))
                        return caller
                frame = frame.f_back
            return None
        finally:
            del frame

    def __call__(self, event, args):
        if event != "subprocess.Popen":
            return
        record = None
        try:
            if os.getpid() != self.collector_pid or self.finished:
                return
            wall, monotonic = time.time_ns(), time.monotonic_ns()
            identity = None
            if self.tool is not None:
                frame = sys._getframe(1)
                if frame.f_code is self.execute_code or frame.f_code is self.init_code:
                    owner = frame.f_locals.get("self")
                    if owner is not None:
                        identity = id(owner)
                    del owner
                del frame
            with self.lock:
                if self.finished:
                    return
                if identity is not None:
                    self.pending.pop(identity, None)
                if len(self.events) == self.limit:
                    self.dropped += 1
                    return
                unresolved = []
                record = dict(
                    kind="PROCESS_LAUNCH_INTENT",
                    sequence=len(self.events),
                    parent_pid=self.collector_pid,
                    child_pid=None,
                    launch_outcome="unobserved",
                    thread_id=threading.get_native_id(),
                    wall_time_ns=wall,
                    monotonic_ns=monotonic,
                    executable=None,
                    arguments=None,
                    cwd=None,
                    caller=None,
                    unresolved=unresolved,
                )
                self.events.append(record)
                if identity is not None:
                    self.pending[identity] = record
            executable, arguments, directory = args[:3]
            record["executable"] = _path_text(executable)
            if record["executable"] is None:
                unresolved.append("executable_not_text")
            if type(arguments) in (list, tuple):
                words = [_path_text(arg) for arg in arguments]
                if all(word is not None for word in words):
                    record["arguments"] = words
            if record["arguments"] is None:
                unresolved.append("argument_vector_not_text")
            try:
                requested = (
                    _path_text(directory) if directory is not None else os.curdir
                )
                if requested:
                    record["cwd"] = os.path.abspath(requested)
            except OSError:
                pass
            if record["cwd"] is None:
                unresolved.append("cwd_unavailable")
            record["caller"] = self.caller()
            if record["caller"] is None:
                unresolved.append("no_caller_within_source_root")
        except Exception as error:
            if record is not None:
                record["unresolved"].append("capture_failed:" + type(error).__name__)

    def verify(self):
        hashes, indexes, code_hashes = {}, {}, {}

        def fingerprint(code):
            key = id(code)
            if key not in code_hashes:
                code_hashes[key] = _code_hash(code)
            return code_hashes[key]

        for caller, code, path in self.callers:
            try:
                caller["code_sha256"] = fingerprint(code)
                data = self.sources[path]
                if data is None:
                    caller["unresolved"].append("caller_source_unreadable")
                    continue
                if path not in hashes:
                    hashes[path] = sha256(data).hexdigest()
                caller["source_sha256"] = hashes[path]
                key = (path, code.co_filename)
                if key not in indexes:
                    indexes[key] = _compiled_codes(data, code.co_filename)
                functions = indexes[key].get(
                    (getattr(code, "co_qualname", code.co_name), code.co_firstlineno),
                    (),
                )
                matches = sum(
                    fingerprint(candidate) == caller["code_sha256"]
                    for candidate in functions
                )
                caller["source_code_matches_snapshot"] = matches == 1
                if matches != 1:
                    caller["unresolved"].append(
                        "ambiguous_source_code"
                        if matches > 1
                        else "runtime_code_differs_from_source_snapshot"
                    )
            except Exception as error:
                caller["unresolved"].append(
                    "code_verification_failed:" + type(error).__name__
                )
        self.callers.clear()
        self.sources.clear()

    def finish(self, output):
        # A forked process inherits the audit hook and atexit registry. Only the
        # collecting interpreter owns this report and its lock.
        if os.getpid() != self.collector_pid:
            return
        with self.lock:
            self.finished = True
            self.pending.clear()
        self.stop_monitor()
        self.verify()
        report = dict(
            schema="scar.process-observation",
            schema_version=1,
            provider="python-audit",
            interpreter=self.interpreter,
            collector_pid=self.collector_pid,
            source_root=str(self.root),
            event_limit=self.limit,
            dropped_events=self.dropped,
            events=self.events,
        )
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as temp:
            name = Path(temp.name)
            try:
                json.dump(
                    report,
                    temp,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                temp.write("\n")
                temp.close()
                os.replace(name, output)
            finally:
                name.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if not 0 <= options.limit < 1 << 64:
        parser.error("--limit requires an unsigned 64-bit count")
    root = options.source_root.resolve(strict=True)
    if not root.is_dir():
        parser.error("--source-root must be a directory")
    output = options.out.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = _Capture(root, options.limit)
    atexit.register(capture.finish, output)
    capture.monitor_returns()
    sys.addaudithook(capture)
    sys.argv = [options.script, *options.arguments]
    if not getattr(sys.flags, "safe_path", False) and not sys.flags.isolated:
        sys.path[0] = str(Path(options.script).resolve().parent)
    filename = os.path.join(capture.initial_directory, options.script)
    application = ModuleType("__main__")
    application.__dict__.update(
        __file__=filename,
        __package__=None,
        __spec__=None,
        __cached__=None,
        __loader__=SourceFileLoader("__main__", filename),
        __builtins__=builtins,
    )
    sys.modules["__main__"] = application
    code = compile(
        Path(filename).read_bytes(),
        filename,
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    exec(code, application.__dict__)


if __name__ == "__main__":
    main()
