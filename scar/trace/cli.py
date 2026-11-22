"""Command line adapters for traced execution and capture import."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def _trace(args) -> int:
    out = Path(args.out).resolve()
    capture_plan_path = getattr(args, 'capture_plan', None)
    capture_plan_sha256 = getattr(args, 'capture_plan_sha256', None)
    port_profile = getattr(args, 'capture_profile', None)
    program_ports = port_profile in ('program-ports','process-ports')
    if program_ports and capture_plan_path is not None:
        raise SystemExit('--capture-profile and --capture-plan are independent, exclusive scopes')
    if (capture_plan_path is None) != (capture_plan_sha256 is None):
        raise SystemExit('--capture-plan and --capture-plan-sha256 are required together')
    lean_plan = None
    if capture_plan_path is not None:
        from scar.trace.lean_capture import read_plan
        capture_plan_path = Path(capture_plan_path).resolve()
        lean_plan = read_plan(capture_plan_path, expected_sha256=capture_plan_sha256,
            max_codes=args.max_call_codes)
        if args.project_root is not None and str(Path(args.project_root).resolve()) != lean_plan.namespace_root:
            raise SystemExit('--project-root differs from the capture plan namespace')
    stream_interfaces = getattr(args, 'stream_interfaces', False) or program_ports
    resource_scopes=getattr(args,'resource_scopes',False)
    torch_call_scopes = getattr(args, 'torch_call_scopes', False)
    resumptions=getattr(args,'resumptions',False)
    if getattr(args,'stream_backing',False) and not stream_interfaces:
        raise SystemExit('--stream-backing requires --stream-interfaces')
    native_callables = getattr(args, 'native_callables', False)
    compile_events = getattr(args, 'compile_events', False)
    python_targets = getattr(args, 'python_call_targets', False) or lean_plan is not None or stream_interfaces or program_ports or native_callables or compile_events or resource_scopes
    operands = getattr(args, 'call_operands', False) or python_targets or torch_call_scopes
    control_codes = tuple(dict.fromkeys(getattr(args, 'control_code', ()) or ()))
    if control_codes:
        import re
        if not operands or len(control_codes) > 32 or any(re.fullmatch(r'pycode-v2:[0-9a-f]{64}', key) is None
                for key in control_codes):
            raise SystemExit('--control-code requires --call-operands and up to 32 exact CodeIDs')
    epoch_region = getattr(args, 'stateful_allocation_epochs', None)
    if epoch_region is not None and not operands:
        raise SystemExit('--stateful-allocation-epochs requires --call-operands')
    os_capture_root = getattr(args, 'os_capture_root', None)
    if resource_scopes and os_capture_root is None:
        raise SystemExit('--resource-scopes requires --os-capture-root')
    if os_capture_root is not None:
        if not 1 <= args.archive_workers <= 32:
            raise SystemExit('--archive-workers must be between 1 and 32')
        os_capture_root = Path(os_capture_root).absolute()
        if os_capture_root.exists() or out != os_capture_root / 'trace':
            raise SystemExit('--os-capture-root requires a fresh root and explicit --out ROOT/trace')
        if not (operands and getattr(args, 'execution_evidence', False)):
            raise SystemExit('--os-capture-root requires --call-operands --execution-evidence')
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[2])
    env["SCAR_TRACE_DIR"] = str(out)
    env["SCAR_TRACE_DEBUG"] = "1"
    env.pop('SCAR_TRACE_CALL_EDGES',None)
    env.pop('SCAR_CALL_TRACE_OWNER_PID',None)
    env.pop('SCAR_CALL_OPERANDS',None)
    env.pop('SCAR_PYTHON_CALL_TARGETS',None)
    env.pop('SCAR_NATIVE_CALLABLE_PROBE',None)
    for key in tuple(env):
        if key.startswith('SCAR_COMPILE_'):env.pop(key,None)
    env.pop('SCAR_STREAM_INTERFACES',None)
    env.pop('SCAR_RESOURCE_SCOPES',None)
    env.pop('SCAR_TORCH_CALL_SCOPES', None)
    env.pop('SCAR_RESUMPTIONS',None)
    env.pop('SCAR_STREAM_BACKING_OBSERVATIONS',None)
    env.pop('SCAR_OPERAND_CAPTURE',None)
    env.pop('SCAR_CAPTURE_PLAN',None)
    env.pop('SCAR_CAPTURE_PLAN_SHA256',None)
    for key in ('SCAR_PYTHON_CAPTURE_PROFILE','SCAR_SOURCE_MAX_FILES','SCAR_SOURCE_MAX_FILE_BYTES','SCAR_SOURCE_MAX_BYTES','SCAR_CALL_MAX_CODES'):
        env.pop(key,None)
    env.pop('SCAR_EXECUTION_EVIDENCE',None)
    env.pop('SCAR_TENSOR_PHYSICAL',None)
    env.pop('SCAR_RESOURCE_SAMPLER',None)
    env.pop('SCAR_RESOURCE_INTERVAL',None)
    env.pop('SCAR_TRACE_LINES',None)
    env.pop('SCAR_CAPTURE_TRANSPORT',None)
    env.pop('SCAR_SNAPSHOT_BUFFER_BYTES',None)
    env.pop('SCAR_CAPTURE_PHASES',None)
    env.pop('SCAR_REGION_METADATA_EPOCH_TRACKER',None)
    env.pop('SCAR_CONTROL_CODES',None)
    for key in ('SCAR_OS_CAPTURE_NONCE','SCAR_OS_CAPTURE_INTENT','SCAR_OS_CAPTURE_INTENT_SHA256'):
        env.pop(key,None)
    execution_evidence = getattr(args, "execution_evidence", False)
    native_inputs = getattr(args, 'native_tensor_inputs', False)
    if native_inputs and not execution_evidence:
        raise SystemExit('--native-tensor-inputs requires --execution-evidence')
    physical = getattr(args, "tensor_physical", False) or native_inputs
    if physical and not execution_evidence:
        raise SystemExit("--tensor-physical requires --execution-evidence")
    if torch_call_scopes and not execution_evidence:
        raise SystemExit('--torch-call-scopes requires --execution-evidence')
    if execution_evidence and not (getattr(args, "calls_only", False) or operands or resumptions):
        raise SystemExit("--execution-evidence requires --calls-only or --call-operands")
    if getattr(args,"calls_only",False) or operands or resumptions:
        if args.snapshot_buffer_bytes < 1:
            raise SystemExit('--snapshot-buffer-bytes must be positive')
        env['SCAR_CAPTURE_TRANSPORT'] = args.capture_transport
        env['SCAR_SNAPSHOT_BUFFER_BYTES'] = str(args.snapshot_buffer_bytes)
        if args.lines:
            raise SystemExit("calls-only does not collect line tracing; use a separate trace")
        if (any(out.glob('call-events.*.jsonl*')) or any(out.glob('call-codes.*.jsonl*'))
                or any(out.glob('call-operands.*.jsonl*')) or any(out.glob('call-operand-meta.*.json*'))
                or any(out.glob('function-environments.*.jsonl*'))
                or any(out.glob('function-environment-meta.*.json*'))
                or any(out.glob('python-call-targets.*.jsonl*'))
                or any(out.glob('python-call-target-meta.*.json*'))
                or any(out.glob('native-callables.*.jsonl*'))
                or any(out.glob('native-callable-meta.*.json*'))
                or any(out.glob('compilation-meta.*.json'))
                or any(out.glob('compilation-disabled.*.json'))
                or any(out.glob('compilation-events.*.jsonl*'))
                or any(out.glob('compilation-sources.*.jsonl*'))
                or any(out.glob('lean-plan.*.json'))
                or any(out.glob('stream-interfaces.*.jsonl*'))
                or any(out.glob('stream-interface-meta.*.json*'))
                or (execution_evidence and any(out.glob('execution-evidence.*.json')))):
            raise SystemExit("calls-only requires a fresh output directory")
        env['SCAR_TRACE_CALL_EDGES'] = '1'
        if control_codes:env['SCAR_CONTROL_CODES'] = json.dumps(control_codes)
        if epoch_region is not None:
            from scar.compiler.input_metadata import read_input_boundary
            from scar.trace.numpy_allocation_epochs import build_tracker
            region = Path(epoch_region).resolve()
            if not read_input_boundary(region).stateful_inputs:
                raise SystemExit('--stateful-allocation-epochs requires a stateful array region')
            env['SCAR_REGION_METADATA_PLAN'] = str(region)
            env['SCAR_REGION_METADATA_OUT'] = str(out)
            env['SCAR_REGION_METADATA_EPOCH_TRACKER'] = str(build_tracker(out / 'numpy-epochs'))
        if resumptions:env['SCAR_RESUMPTIONS']='1'
        env['SCAR_CALL_PROJECT_ROOT'] = str(Path(args.project_root or (
            lean_plan.namespace_root if lean_plan is not None else Path.cwd())).resolve())
        if compile_events:
            env['SCAR_COMPILE_EVENTS']='1'
            for name,suffix in (('max_compile_events','MAX_EVENTS'),('max_compile_sources','MAX_SOURCES'),
                    ('max_compile_source_bytes','MAX_SOURCE_BYTES'),('max_compile_total_source_bytes','MAX_TOTAL_SOURCE_BYTES'),
                    ('max_compile_stream_bytes','MAX_STREAM_BYTES')):
                value=getattr(args,name)
                if type(value) is not int or value<1:raise SystemExit('compile audit limits must be positive')
                env['SCAR_COMPILE_'+suffix]=str(value)
            if args.max_compile_source_bytes>1_000_000:
                raise SystemExit('--max-compile-source-bytes cannot exceed 1000000')
        env['SCAR_CALL_MAX_EVENTS'] = str(args.max_call_events)
        env['SCAR_CALL_MAX_BYTES'] = str(args.max_call_bytes)
        if args.max_call_references < 1:
            raise SystemExit('call reference limit must be positive')
        env['SCAR_CALL_MAX_REFERENCES'] = str(args.max_call_references)
        if args.max_call_codes < 1:
            raise SystemExit('call code limit must be positive')
        env['SCAR_CALL_MAX_CODES'] = str(args.max_call_codes)
        if args.max_call_events < 1 or args.max_call_bytes < 1:
            raise SystemExit("call budgets must be positive")
        if operands:
            if args.max_operand_events < 1 or args.max_operand_bytes < 1:
                raise SystemExit("operand budgets must be positive")
            capture = (port_profile if program_ports else 'lean-ports' if lean_plan is not None
                       else getattr(args, 'operand_capture', 'windows'))
            if capture not in ('windows', 'instructions', 'lean-ports', 'program-ports', 'process-ports'):
                raise SystemExit("operand capture must be windows or instructions")
            env['SCAR_CALL_OPERANDS'] = '1'
            if python_targets:
                env['SCAR_PYTHON_CALL_TARGETS'] = 'PYTHON_CALL_TARGETS_V2'
            if native_callables:
                from scar.trace.native_callable_probe import build_probe
                env['SCAR_NATIVE_CALLABLE_PROBE'] = str(build_probe(out / 'native-probe'))
            if stream_interfaces:
                env['SCAR_STREAM_INTERFACES'] = '1'
                if getattr(args,'stream_backing',False):
                    env['SCAR_STREAM_BACKING_OBSERVATIONS'] = '1'
            if resource_scopes:env['SCAR_RESOURCE_SCOPES']='1'
            if torch_call_scopes:
                env['SCAR_TORCH_CALL_SCOPES'] = '1'
            env['SCAR_OPERAND_CAPTURE'] = capture
            env['SCAR_OPERAND_MAX_EVENTS'] = str(args.max_operand_events)
            env['SCAR_OPERAND_MAX_BYTES'] = str(args.max_operand_bytes)
            if program_ports:
                from scar.trace.captured_sources import PROGRAM_PORTS_PROFILE, PROCESS_PORTS_PROFILE
                env['SCAR_PYTHON_CAPTURE_PROFILE'] = (PROCESS_PORTS_PROFILE if port_profile == 'process-ports'
                                                      else PROGRAM_PORTS_PROFILE)
                env['SCAR_STREAM_BACKING_OBSERVATIONS'] = '1'
                for name, value in (('FILES', args.max_source_files), ('FILE_BYTES', args.max_source_file_bytes),
                                    ('BYTES', args.max_source_bytes)):
                    if type(value) is not int or value < 1:
                        raise SystemExit('source archive limits must be positive integers')
                    env['SCAR_SOURCE_MAX_' + name] = str(value)
            if lean_plan is not None:
                env['SCAR_CAPTURE_PLAN'] = str(capture_plan_path)
                env['SCAR_CAPTURE_PLAN_SHA256'] = capture_plan_sha256
        if execution_evidence:
            env['SCAR_EXECUTION_EVIDENCE'] = '1'
            if physical:
                env['SCAR_TENSOR_PHYSICAL'] = '1'
            if native_inputs:
                env['SCAR_NATIVE_TENSOR_INPUTS'] = '1'
    if args.lines:
        env["SCAR_TRACE_LINES"] = "1"
    if args.resources:
        env["SCAR_RESOURCE_SAMPLER"] = "1"
        env["SCAR_RESOURCE_INTERVAL"] = str(args.resource_interval)
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("trace requires '-- command'")
    start = time.perf_counter()
    postprocess_timeout = getattr(args, 'os_postprocess_timeout', None)
    if postprocess_timeout is not None and (type(postprocess_timeout) is not int or postprocess_timeout <= 0):
        raise SystemExit('--os-postprocess-timeout must be positive')
    if postprocess_timeout is not None and os_capture_root is None:
        raise SystemExit('--os-postprocess-timeout requires --os-capture-root')
    if os_capture_root is None:
        result = subprocess.run(cmd, env=env)
    else:
        from scar.trace.os_capture_launch import OSCaptureBudgets, launch_os_capture
        os_budgets = OSCaptureBudgets(max_os_bytes=args.max_os_bytes,
            max_raw_bytes=args.max_capture_bytes, max_stored_bytes=args.max_archive_bytes,
            max_os_lines=args.max_os_lines, max_seconds=args.os_timeout,
            max_files=getattr(args,'max_archive_files',4096))
        result, _ = launch_os_capture(cmd, env=env, capture_root=os_capture_root,
            strace_executable=args.strace_executable, budgets=os_budgets,
            selection_profile=args.os_profile,postprocess_seconds=postprocess_timeout)
    (out / "run.json").write_text(json.dumps({"command": cmd, "returncode": result.returncode,
                                                "wall_s": time.perf_counter() - start}, indent=2))
    if os_capture_root is not None and result.returncode == 0:
        from scar.trace.os_capture_launch import archive_owned_capture, publish_bytes
        from scar.trace.control_transfers import bind_capture_coordinates
        archive_owned_capture(os_capture_root, budgets=os_budgets,workers=args.archive_workers)
        coordinate_path = os_capture_root / 'capture-coordinates.json'
        coordinates = bind_capture_coordinates(os_capture_root, max_bytes=os_budgets.max_raw_bytes,
            max_lines=os_budgets.max_os_lines, profile_uri=str(coordinate_path), publish_os_profile=True)
        publish_bytes(coordinate_path, coordinates.payload)
        print(json.dumps({'os_capture': str(os_capture_root), 'coordinates': str(coordinate_path),
            'os_capture_provider': coordinates.report.get('os_capture_provider'),
            'whole_model_established': False}), flush=True)
    return result.returncode


def _import_capture(args) -> int:
    from scar.trace.capture_import import import_capture
    report = import_capture(args.directory, args.out, max_bytes=args.max_bytes, max_lines=args.max_lines,
        schema_contracts=args.schema_contracts, bundle_partitions=args.bundle_partitions,
        os_interface_work=args.os_interface_work,
        progress=lambda record: print(json.dumps(record), flush=True))
    print(json.dumps({"out": str(Path(args.out).resolve()), "coordinate_identity_consistent": report["coordinate_identity_consistent"],
                      "whole_model_established": report["whole_model_established"]}))
    return 0


def trace_main(argv=None):
    parser = argparse.ArgumentParser(prog="scar trace", description="Trace a child command.")
    parser.add_argument("--out", default="artifacts/traces/run")
    parser.add_argument("--lines", action="store_true",
                       help="record user python_line and bytecode loop back-edge events")
    parser.add_argument("--resources", action="store_true",
                       help="sample traced-process CPU and nvidia-smi GPU resources")
    parser.add_argument("--resource-interval", type=float, default=0.1,
                       help="resource sampling interval in seconds (with --resources)")
    parser.add_argument('--calls-only',action='store_true',help='record bounded Python call sites and reference identities without Torch hooks or tensor values')
    parser.add_argument('--call-operands',action='store_true',help='record bounded instruction/load/call evidence; implies calls-only (CPython 3.14)')
    parser.add_argument('--python-call-targets', action='store_true',
                       help='record explicit v2 Python callee and receiver roles; implies call-operands (CPython 3.14.7)')
    parser.add_argument('--native-callables', action='store_true',
                       help='capture actual native callable type/owner/ABI with an isolated optional C probe; implies Python call targets (CPython 3.14.7)')
    parser.add_argument('--compile-events',action='store_true',
                       help='passively retain bounded compilation text and executed CodeID observations; implies Python call targets')
    parser.add_argument('--max-compile-events',type=int,default=4096)
    parser.add_argument('--max-compile-sources',type=int,default=1024)
    parser.add_argument('--max-compile-source-bytes',type=int,default=512000)
    parser.add_argument('--max-compile-total-source-bytes',type=int,default=8000000)
    parser.add_argument('--max-compile-stream-bytes',type=int,default=16000000)
    parser.add_argument('--capture-plan', help='use a source-derived lean interface capture plan; implies Python call targets')
    parser.add_argument('--capture-plan-sha256', help='expected capture plan SHA256')
    parser.add_argument('--capture-profile', choices=('program-ports','process-ports'),
                       help='CALL-only monitoring and source bytes: project descendants (V1), or installed target-process Python threads (V2)')
    parser.add_argument('--max-source-files',type=int,default=1024)
    parser.add_argument('--max-source-file-bytes',type=int,default=2_000_000)
    parser.add_argument('--max-source-bytes',type=int,default=64_000_000,
                       help='raw source bytes read once per observed original file (ports profiles only)')
    parser.add_argument('--execution-evidence',action='store_true',
                       help='collect native ETO/Kineto evidence alongside the call recorder when the target imports Torch; requires calls-only or call-operands')
    parser.add_argument('--torch-call-scopes', action='store_true',
                       help='record bounded CPython native CALL scopes in the owned Torch trace; requires execution-evidence')
    parser.add_argument('--tensor-physical',action='store_true',
                       help='also export owned profiler marker Tensor metadata and allocation epochs after close; requires execution-evidence')
    parser.add_argument('--native-tensor-inputs', action='store_true',
                       help='retain boxed Tensor input geometry, dtype and layout for native graph projection; implies tensor-physical')
    parser.add_argument('--operand-capture',choices=('windows','instructions'),default='windows',
                       help='capture bounded call windows (default) or every project instruction')
    parser.add_argument('--project-root',help='source boundary for lightweight call-site recording; defaults to target cwd')
    parser.add_argument('--max-call-events',type=int,default=100000)
    parser.add_argument('--max-call-bytes',type=int,default=64000000)
    parser.add_argument('--max-call-codes',type=int,default=4096)
    parser.add_argument('--max-call-references',type=int,default=8192,
        help='bounded active weak identities shared by frame/operand observers; no application contents')
    parser.add_argument('--capture-transport',choices=('jsonl','snapshot'),default='jsonl',
        help='JSONL in callbacks, or bounded raw snapshots with JSONL conversion after collection stops')
    parser.add_argument('--snapshot-buffer-bytes',type=int,default=8 * 1024 * 1024,
        help='DRAM byte ceiling per snapshot stream; writer threads perform IO only')
    parser.add_argument('--max-operand-events',type=int,default=100000)
    parser.add_argument('--max-operand-bytes',type=int,default=32000000)
    parser.add_argument('--os-capture-root',help='opt in to owned syscall observation, lossless retention and final coordinates; requires fresh ROOT and --out ROOT/trace')
    parser.add_argument('--os-profile',choices=('program-interfaces','all'),default='program-interfaces',
        help='explicit OS scope; program-interfaces requests seccomp filtering and excludes unlisted internals')
    parser.add_argument('--archive-workers',type=int,default=4,help='1..32 bounded compression/IO workers after capture closes')
    parser.add_argument('--strace-executable',default='strace',help='strace executable for the owned opt-in OS capture')
    parser.add_argument('--max-os-bytes',type=int,default=128000000)
    parser.add_argument('--stream-interfaces', action='store_true', help='observe builtin stream calls and C completion without reading contents or wrapping objects')
    parser.add_argument('--resumptions',action='store_true',help='observe actual resumable activation boundaries without retaining frames or yielded values')
    parser.add_argument('--stream-backing',action='store_true',
        help='also observe exact builtin stream backing handles; requires --stream-interfaces')
    parser.add_argument('--resource-scopes',action='store_true',
        help='opt in to exact Python and native call scope markers for OS/source joins; requires --os-capture-root')
    parser.add_argument('--stateful-allocation-epochs', metavar='REGION',
        help='observe bounded NumPy allocation epochs at accepted original stateful frame entries')
    parser.add_argument('--control-code', action='append', metavar='CODE_ID',
        help='capture local branches and jumps in this original code, repeatable')
    parser.add_argument('--max-os-lines',type=int,default=2000000)
    parser.add_argument('--max-capture-bytes',type=int,default=256000000)
    parser.add_argument('--max-archive-bytes',type=int,default=128000000)
    parser.add_argument('--max-archive-files',type=int,default=4096,
                       help='lossless archive transport member count, including per-TID OS files, native chunks and collector members; not event coverage')
    parser.add_argument('--os-timeout',type=int,default=300,
        help='recording seconds with --os-postprocess-timeout; otherwise original total provider wall limit')
    parser.add_argument('--os-postprocess-timeout',type=int,
        help='explicit finite shutdown/conversion seconds after qualified main atexit; total <= os-timeout + this, plus 5s termination grace; absent keeps old total budget')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return _trace(parser.parse_args(argv))


def import_main(argv=None):
    parser = argparse.ArgumentParser(prog="scar import-capture")
    parser.add_argument("directory", help="successful SCAR collector capture directory containing trace/ and os/")
    parser.add_argument("--max-bytes", type=int, default=1000000000)
    parser.add_argument("--max-lines", type=int, default=2000000)
    parser.add_argument("--schema-contracts", action="store_true", help="recover ordered provider ports using the exact matching installed Torch schema parser")
    parser.add_argument("--bundle-partitions", action="store_true", help="store native graph records as bounded chunks instead of one nested bundle JSON")
    parser.add_argument("--os-interface-work", action="store_true", help="retain kernel interfaces and typed work in bounded partitions; effects and resource closure remain explicit")
    parser.add_argument("--out", required=True, help="fresh assembled capture directory with per-stage progress")
    return _import_capture(parser.parse_args(argv))
