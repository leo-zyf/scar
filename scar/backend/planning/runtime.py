"""B1 qualification of source-declared external execution interfaces."""

import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import digest, json_bytes, walk_operations

_EXECUTION = set(
    "system popen fork vfork execl execle execlp execv execve execvp execvpe "
    "fexecve posix_spawn posix_spawnp dlopen dlmopen".split()
)
_PROFILE = "posix_libc_system_v1"
_PARAMS = [{"kind": "ptr", "to": {"kind": "const", "of": "char"}}]
_ABI = dict(
    symbol="system",
    provider="libc",
    header="stdlib.h",
    parameters=_PARAMS,
    result="i32",
    variadic=False,
)
_CONTRACT = dict(
    semantics_version=1,
    source_interface="system",
    count_unit="bytes",
    command_read="through_first_nul_when_nonnull",
    null_command="query_command_processor_availability",
    process_effect="external_command_processor_when_nonnull",
    return_value="implementation_status_not_direct_exit_code",
)


def import_effects(module, bindings):
    """Keep source occurrence coverage separate from native interface supply."""
    ledger = module.get("source_imports")
    if ledger is None:
        return None
    if (
        ledger.get("semantics_version") != 1
        or ledger.get("scope") != "source_import_occurrences"
        or not isinstance(ledger.get("occurrences"), list)
    ):
        raise ValueError("invalid source import occurrence product")
    operations = {
        (
            fn["name"],
            node.get("site"),
            node["op"],
            node.get("source_import_occurrence"),
        ): node
        for fn in module["functions"]
        for node in walk_operations(fn["body"])
    }
    rows, identities = [], set()
    counts = dict(visited=0, unvisited=0, proven_inactive=0)
    for occurrence in ledger["occurrences"]:
        identity = occurrence["id"]
        activation = occurrence["activation"]
        if identity in identities or activation not in counts:
            raise ValueError("ambiguous source import occurrence or activation")
        identities.add(identity)
        counts[activation] += 1
        producers = []
        for ref in occurrence["producer_sites"]:
            key = (ref["owner"], ref.get("site"), ref["opcode"], identity)
            operation = operations.get(key)
            family = ref.get("operator_family")
            supplied = operation is not None and family in bindings
            producers.append(
                dict(
                    **ref,
                    operation_present=operation is not None,
                    native_interface_supplied=supplied,
                    evidence_scope="registered target interface only; not a proof of the complete import effects",
                )
            )
        rows.append(
            dict(
                id=identity,
                owner=occurrence["owner"],
                site=occurrence.get("site"),
                activation=activation,
                source_coverage=occurrence["effect_coverage"],
                producers=producers,
                obligations=occurrence["obligations"],
                complete_effect_equivalence=(
                    "unproven"
                    if activation == "visited"
                    else "not_an_execution_observation"
                ),
            )
        )
    return dict(
        semantics_version=1,
        scope="source_import_occurrences",
        ledger_sha256=digest(json_bytes(ledger)),
        coverage="INCOMPLETE" if counts["visited"] else "NO_VISITED_IMPORTS",
        activation_counts=counts,
        occurrences=rows,
        qualification="Existing native API qualification does not discharge omitted initialization, callbacks, audit or failure effects",
    )


def _file(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), size=len(data), sha256=digest(data))


def _shell_observation():
    path = "/bin/sh"
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            return dict(path=path, observation="present", file_kind="non_regular")
        return dict(
            path=path,
            observation="present",
            executable=os.access(path, os.X_OK),
            target=_file(path),
        )
    except FileNotFoundError:
        return dict(path=path, observation="absent")
    except OSError as exc:
        return dict(path=path, observation="unobserved", errno=exc.errno)


def _run(argv):
    result = subprocess.run(
        argv, capture_output=True, text=True, check=False, timeout=30
    )
    if result.returncode:
        raise ValueError(
            f"runtime ABI qualification failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _qualify_glibc(module, declaration):
    resources = module.get("resource_model", {})
    captured = resources.get("platform", {})
    if (
        platform.system() != "Linux"
        or captured.get("provider") != "linux-proc-sysfs"
        or captured.get("architecture") != platform.machine()
    ):
        raise ValueError("requires matching captured and deployment Linux CPU ABIs")
    source_header = declaration.get("location", {}).get("file")
    expected = resources.get("software_headers", {}).get(source_header)
    if not expected:
        raise ValueError("source system declaration has no captured SDK header receipt")
    compiler = shlex.split(os.environ.get("CXX") or "c++")
    cc = shutil.which(compiler[0]) if compiler else None
    if not cc:
        raise ValueError("target C++ compiler is unavailable")
    source = Path(__file__).with_name("system_probe.cpp")
    with TemporaryDirectory(prefix="scar-system-abi-") as build:
        binary, dependencies = (Path(build) / name for name in ("probe", "probe.d"))
        _run(
            [
                cc,
                *compiler[1:],
                "-std=c++20",
                "-fPIC",
                "-Werror",
                "-MD",
                "-MF",
                dependencies,
                source,
                "-ldl",
                "-o",
                binary,
            ]
        )
        lines = _run([binary]).splitlines()
        if len(lines) != 3 or not lines[2].isdigit():
            raise ValueError("incomplete target libc address/ABI evidence")
        provider, version, pointer_bits = lines
        # The depfile lists the headers actually used by this compilation.
        paths = shlex.split(
            dependencies.read_text().replace("\\\n", " ").partition(":")[2]
        )
        headers = [
            _file(p)
            for p in sorted(set(paths))
            if Path(p).name in {"stdlib.h", "libc-version.h"}
        ]
        if not any(
            Path(h["path"]).name == "stdlib.h"
            and (h["sha256"], h["size"])
            == (expected.get("sha256"), expected.get("size"))
            for h in headers
        ):
            raise ValueError(
                "target system declaration header differs from captured stdlib.h"
            )
    return dict(
        evidence="linked_cxx20_prototype_and_loaded_symbol_address",
        probe=dict(sha256=digest(source.read_bytes()), calls_system=False),
        compiler=_file(cc),
        compiler_arguments=compiler[1:],
        source_header=dict(path=source_header, **expected),
        headers=headers,
        header_scope="declaration_and_glibc_identity_headers_used_by_probe",
        architecture=platform.machine(),
        c_int_bits=32,
        pointer_bits=int(pointer_bits),
        libc=_file(provider),
        libc_version=version,
        shell=_shell_observation(),
        scope="qualification_host; deployment requires compatible libc and POSIX environment",
        descendant_tool_identities="unknown",
    )


def _system_site(module, owner, node, declaration):
    boundary = node.get("runtime_boundary") or {}
    effect = node.get("artifact_effect") or {}
    site = node.get("site")
    if not isinstance(site, str):
        raise ValueError("missing source execution site")
    graph_id, _, port = site.rpartition("/")
    if not graph_id or not port.isascii() or not port.isdecimal():
        raise ValueError("missing source graph port for system")
    flow = module.get("resource_model", {}).get("artifact_flow") or {}
    if not isinstance(flow, dict) or not flow.get("sha256"):
        raise ValueError("missing sealed graph member for the source command request")
    operation = "source_operation:" + digest(json_bytes([graph_id, int(port)]))
    request = "artifact:command_request:" + digest(
        json_bytes([module.get("current_image"), [owner, operation]])
    )
    args = node.get("args")
    if (
        not isinstance(effect, dict)
        or not isinstance(args, list)
        or len(args) != 1
        or not isinstance(args[0], dict)
        or "op" not in args[0]
        or args[0].get("type", _PARAMS[0])
        not in (_PARAMS[0], {"kind": "ptr", "to": "char"})
    ):
        raise ValueError("system requires one C string pointer operand")
    required = dict(
        semantics_version=1,
        profile=_PROFILE,
        site=site,
        source_abi=_ABI,
        source_operation=operation,
        command_request=request,
        command_operand=["args", 0],
        execution="synchronous",
        result="c_system_status",
        descendant_closure="open",
    )
    if (
        node.get("op") != "call"
        or node.get("callee") != "system"
        or node.get("parameter_types") != _PARAMS
        or node.get("type", "i32") != "i32"
        or declaration.get("provider") != "libc"
        or declaration.get("header") != "stdlib.h"
        or declaration.get("result") != "i32"
        or declaration.get("variadic")
        or [p.get("type") for p in declaration.get("params", ())] != _PARAMS
        or effect.get("interface") != "process.system"
        or effect.get("semantics_version") != 1
        or effect.get("operands") != {"command": ["args", 0]}
        or effect.get("defaults") != {}
        or effect.get("contract") != _CONTRACT
        or not {
            "linked_external_binding",
            "call_argument_binding",
            "defined_execution",
            "command_nullness_and_terminated_bytes",
            "command_processor_identity",
            "source_environment_and_cwd",
            "external_process_effects_unmodeled",
        }.issubset(effect.get("obligations", ()))
        or boundary != required
    ):
        raise ValueError(
            "source libc declaration, call, or command request does not match its sealed runtime boundary"
        )
    if (
        node.get("external_effect") != "UNKNOWN"
        or not {"CTRL", "IO", "ORDER", "OPAQUE", "STATE"}.issubset(
            node.get("effects", ())
        )
        or not {
            "command.c_string",
            "process.cwd",
            "process.environment",
            "process.signals",
            "external.state",
        }.issubset(node.get("reads", ()))
        or not {
            "process.children",
            "process.signals",
            "thread.errno",
            "external.state",
        }.issubset(node.get("writes", ()))
    ):
        raise ValueError("missing external execution effect barrier")
    return dict(
        interface="system",
        qualification="api_bound",
        descendant_closure="open",
        call_sha256=digest(json_bytes(node)),
        command_sha256=digest(json_bytes(args[0])),
        command_request=request,
        command_operand=["args", 0],
        result="c_system_status",
        execution="synchronous",
        command="runtime_c_string_or_null",
        expanded_argv="unknown",
        tools="unknown",
        cost="unknown",
        external_effect="UNKNOWN",
    )


def qualify(module, *, probe=True):
    """Return qualified runtime sites and unresolved process/loading boundaries."""
    owners = [(f["name"], f) for f in module.get("functions", ())]
    definitions = {name for name, _ in owners}
    external = {
        d["name"]: d
        for d in module.get("externs", ())
        if d["name"] not in definitions
        and (
            d["name"] == "system"
            or d.get("provider") in {"libc", "posix"}
            and d["name"].rsplit("::", 1)[-1] in _EXECUTION
        )
    }
    if not external:
        return None, []
    sites, issues = {}, []
    initializers = (
        (None, row)
        for collection in ("globals", "records")
        for row in module.get(collection, ())
    )
    for owner, row in [*owners, *initializers]:
        for node in walk_operations(row):
            targets = (
                [node.get("callee")]
                if node["op"] == "call"
                else node.get("targets", ()) if node["op"] == "indirect_call" else ()
            )
            for name in targets:
                if name not in external:
                    continue
                try:
                    if name != "system" or node["op"] != "call":
                        raise ValueError(
                            "external execution or dynamic loading requires a bound implementation for this call"
                        )
                    binding = _system_site(module, owner, node, external[name])
                    if node["site"] in sites:
                        raise ValueError("ambiguous runtime call site")
                    sites[node["site"]] = binding
                except ValueError as exc:
                    issues.append(
                        dict(
                            stage="B1",
                            code="missing_execution_boundary_binding",
                            symbol=name,
                            provider=external[name].get("provider"),
                            site=node.get("site"),
                            location=node.get("location"),
                            detail=str(exc),
                        )
                    )
    if not sites or issues or not probe:
        return None, issues
    try:
        deployment = _qualify_glibc(module, external["system"])
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        return None, [
            dict(
                stage="B1",
                code="missing_execution_boundary_binding",
                symbol="system",
                sites=sorted(sites),
                detail=str(exc),
            )
        ]
    return (
        dict(
            provider="glibc",
            profile=_PROFILE,
            qualification="api_bound",
            descendant_closure="open",
            declaration_sha256=digest(json_bytes(external["system"])),
            sites=sites,
            calling_contract=dict(
                null_command="libc_processor_availability",
                nonnull_command="unchanged_null_terminated_bytes",
                result="unchanged_libc_int_including_failure_or_wait_status",
                wait_signals_errno="handled_by_the_original_c_runtime_interface",
                ordering="opaque_external_effect_barrier",
                baseline_domain="source_pointer_and_producer_preconditions",
            ),
            deployment=deployment,
        ),
        [],
    )
