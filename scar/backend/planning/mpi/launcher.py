"""Qualify fixed-rank, single-host launches against a sealed worker's MPI SDK."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

from scar.contract import digest, json_bytes
from ..runtime import QualificationFailure, qualification_check

_SHELL_LINE = re.compile(
    r"[ \t]*[A-Za-z0-9_./,:@%+=-]+(?:[ \t]+[A-Za-z0-9_./,:@%+=-]+)*[ \t]*\Z"
)


def _command(node, output):
    launcher, source = node.get("launcher"), node.get("source_command")
    if (
        not isinstance(launcher, dict)
        or set(launcher)
        != {"kind", "ranks", "semantics_version", "source_reference", "rank_option"}
        or launcher["kind"] != "mpi"
        or type(launcher["semantics_version"]) is not int
        or launcher["semantics_version"] != 1
    ):
        raise ValueError("requires the fixed-rank MPI launcher v1 contract")
    count = launcher["ranks"]
    if not isinstance(count, dict) or (count.get("op"), count.get("type")) != (
        "literal",
        "i64",
    ):
        raise ValueError("MPI launch count must be a literal i64")
    ranks = count.get("value")
    if type(ranks) is not int or not 0 < ranks < (1 << 63):
        raise ValueError("MPI launch count must be a positive signed 64-bit integer")
    if not isinstance(source, dict) or set(source) != {
        "form",
        "argv",
        "text",
        "shell_path",
        "runtime",
    }:
        raise ValueError("MPI launch lacks its original command contract")
    argv, shell = source["argv"], node.get("shell")
    if (
        not isinstance(argv, list)
        or len(argv) < 4
        or any(not isinstance(a, str) or "\0" in a for a in argv)
    ):
        raise ValueError("MPI launch requires a closed argument vector")
    tool, flag, raw, worker, *arguments = argv
    if (
        not tool
        or tool.startswith("-")
        or "=" in tool
        or ":" in argv
        or tool != launcher["source_reference"]
        or flag != launcher["rank_option"]
        or flag not in {"-n", "-np", "--np"}
        or not raw.isascii()
        or not raw.isdecimal()
        or raw.lstrip("0") != str(ranks)
        or worker != "./" + output
        or any(arg.startswith("-") for arg in arguments)
    ):
        raise ValueError(
            "requires one rank flag, one declared worker, and unambiguous positional worker arguments"
        )
    operands = node.get("argv")
    if (
        not isinstance(operands, list)
        or len(operands) != len(argv) - 3
        or any(
            not isinstance(n, dict)
            or n.get("type") != "text"
            or (n.get("op") == "literal" and n.get("value") != word)
            for n, word in zip(operands, argv[3:])
        )
    ):
        raise ValueError("worker operands differ from the closed MPI command")
    if shell is True:
        text, runtime = source["text"], source["runtime"]
        if (
            source["form"] != "posix_shell"
            or source["shell_path"] != "/bin/sh"
            or not isinstance(runtime, dict)
            or runtime.get("os_name") != "posix"
            or runtime.get("platform") not in {"linux", "linux2"}
            or not isinstance(text, str)
            or not _SHELL_LINE.fullmatch(text)
            or text.split() != argv
        ):
            raise ValueError(
                "shell launch requires exact simple POSIX words and the default Linux shell"
            )
    elif (
        shell is not False
        or source["form"] != "argv"
        or source["shell_path"] is not None
        or source["text"] is not None
    ):
        raise ValueError("direct MPI launch requires the original argv convention")
    return tool, flag, ranks, shell


def _query(argv):
    result = subprocess.run(
        argv, capture_output=True, text=True, errors="replace", timeout=15
    )
    if result.returncode:
        raise ValueError(
            "launcher identity query failed: " + result.stderr.strip()[-500:]
        )
    return result.stdout.strip()


def _installed(lookup, path):
    """Identify a Debian file by canonical owner, source build and verified installed bytes."""
    actual = path.resolve(strict=True)
    query = "/usr/bin/dpkg-query"
    owners = [
        line.removesuffix(": " + str(actual))
        for line in _query([query, "-S", str(actual)]).splitlines()
        if line.endswith(": " + str(actual))
    ]
    if len(owners) != 1 or not re.fullmatch(r"[a-z0-9][a-z0-9.+:-]*", owners[0]):
        raise ValueError(f"no unique installed package owns {actual}")
    package = owners[0]
    fields = (
        "${db:Status-Abbrev}\t${source:Package}\t${source:Version}\t${Architecture}"
    )
    status, source, version, architecture = _query(
        [query, "-W", "-f=" + fields, "--", package]
    ).split("\t")
    if status.strip() != "ii":
        raise ValueError(f"incomplete package installation for {actual}")
    manifest = _query([query, "--control-show", package, "md5sums"])
    rel = actual.relative_to("/").as_posix()
    recorded = [
        line.split(None, 1)[0]
        for line in manifest.splitlines()
        if line.endswith("  " + rel)
    ]
    sha, checksum, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with actual.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 18), b""):
            sha.update(chunk)
            checksum.update(chunk)
            size += len(chunk)
    if recorded != [checksum.hexdigest()]:
        raise ValueError(
            f"installed bytes differ from the package manifest for {actual}"
        )
    return dict(
        lookup=lookup,
        path=str(path),
        realpath=str(actual),
        sha256=sha.hexdigest(),
        size=size,
        package=package,
        manifest_md5=recorded[0],
        provider=dict(
            method="debian_source_package_and_installed_file",
            source_package=source,
            source_version=version,
            architecture=architecture,
        ),
    )


def _tools(reference, shell, sdk):
    candidate = sdk.get("launcher_candidate", {}).get("path")
    reporting = sdk.get("version_provider", {})
    if (
        not candidate
        or not Path(candidate).is_absolute()
        or reporting.get("kind") != "loaded_shared_library"
    ):
        raise ValueError(
            "the worker SDK does not identify its native launcher and MPI shared library"
        )
    if "/" in reference and not Path(reference).is_absolute():
        raise ValueError(
            "a relative launcher path needs a deployment working-directory binding"
        )
    location = shutil.which(reference)
    if not location or not Path(location).samefile(candidate):
        raise ValueError(
            "source launcher resolution differs from the worker SDK's FindMPI executable"
        )
    lib = _installed("MPI_Get_library_version", Path(reporting["path"]))
    tool = _installed(reference, Path(location).absolute())
    if (
        tool["provider"] != lib["provider"]
        or lib["sha256"] != reporting["sha256"]
        or tool["provider"]["source_package"] != "openmpi"
    ):
        raise ValueError(
            "launcher and worker MPI library lack the same supported installed provider build"
        )
    version = _query([tool["path"], "--version"]).splitlines()
    if not version:
        raise ValueError("qualified MPI launcher returned no version information")
    tool["version"] = version[0]
    tool["findmpi_executable"] = candidate
    tool["matched_mpi_library"] = lib
    interpreter = None
    if shell:
        interpreter = _installed("/bin/sh", Path("/bin/sh"))
        if interpreter["provider"]["source_package"] != "dash" or "%" in os.environ.get(
            "PATH", ""
        ):
            raise ValueError(
                "requires the qualified noninteractive dash command lookup"
            )
        resolved = _query(
            ["/bin/sh", "-c", 'command -v "$1"', "scar-lookup", reference]
        )
        if (
            not resolved
            or not Path(resolved).is_absolute()
            or not Path(resolved).samefile(candidate)
        ):
            raise ValueError(
                "the default shell would not execute the qualified launcher"
            )
    # Generated guards use the Linux kernel's SHA-256 implementation.
    with socket.socket(socket.AF_ALG, socket.SOCK_SEQPACKET) as family:
        family.bind(("hash", "sha256"))
        with family.accept()[0] as operation:
            if operation.recv(32).hex() != hashlib.sha256(b"").hexdigest():
                raise ValueError("kernel SHA-256 guard unavailable")
    return tool, interpreter


def qualify_launcher(node, target, worker_mpi, cache):
    if not isinstance(worker_mpi, dict) or worker_mpi.get("provider") != "mpi-c":
        raise QualificationFailure(
            "target_platform",
            "mpi.launcher_worker_sdk",
            "MPI launcher requires the sealed worker image's independently qualified C SDK",
            image_id=target["image_id"],
        )
    with qualification_check(
        "semantic_contract",
        "mpi.launcher_command",
        image_id=target["image_id"],
        site=node.get("site"),
        fields=["launcher", "source_command", "argv"],
    ):
        reference, flag, ranks, shell = _command(node, target["output"])
    revision = digest(json_bytes(worker_mpi))
    key = (
        worker_mpi.get("launcher_candidate", {}).get("path"),
        worker_mpi.get("version_provider", {}).get("sha256"),
        reference,
        shell,
    )
    if key not in cache:
        with qualification_check(
            "target_platform",
            "mpi.launcher_identity",
            source_reference=reference,
            worker_mpi_sha256=revision,
        ):
            try:
                cache[key] = _tools(reference, shell, worker_mpi)
            except AttributeError as exc:
                raise ValueError(
                    f"native MPI launcher qualification unavailable: {exc}"
                ) from exc
    tool, interpreter = cache[key]
    stamp = digest(json_bytes(node["source_command"]))
    site = node.get("site") or "command/" + digest(
        json_bytes([node["image_id"], stamp])
    )
    if not isinstance(site, str):
        raise ValueError("MPI launch binding needs a stable site identity")
    return site, dict(
        kind="mpi",
        semantics_version=1,
        image_id=target["image_id"],
        worker_ir_sha256=target["ir_sha256"],
        worker_mpi_receipt_sha256=revision,
        worker_output=target["output"],
        tool=tool,
        shell=interpreter,
        ranks=ranks,
        rank_option=flag,
        rank_selection="source_fixed",
        placement="launcher_default",
        capacity="unmodeled",
        deployment_scope="single_host_immutable_generation",
        tool_resolution="planning_lookup_with_runtime_path_and_content_verification",
        worker_guard="same_file_or_equal_bytes",
        source_command_sha256=stamp,
    )
