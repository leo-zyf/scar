"""Pinned compact scoped installation; no compiler or graph imports."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy

from .native_loop import _small_raw, _json, GuardFailure
from .native_workspace_session import inspect_bundle as inspect_session, load as load_session
from .session_scope import SessionScope

ADMISSION = 'scar.workspace-session-scope-admission'


def implementation_sha256():
    from .native_workspace_session import implementation_sha256 as parent
    root = Path(__file__).resolve().parents[1]
    names = ('runtime/session_scope_bundle.py','runtime/session_scope.py',
             'runtime/entry_runner.py','runtime/array_copy_guard.py','entrypoint.py')
    return sha256(json.dumps(dict(parent=parent(), files=[(n,sha256((root/n).read_bytes()).hexdigest())
        for n in names]),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inspect_scope_bundle(bundle, *, expected_admission_sha256):
    directory = Path(bundle).resolve(); raw = _small_raw(directory/'scope-admission.json')
    if sha256(raw).hexdigest() != expected_admission_sha256:
        raise GuardFailure('scope admission pin differs')
    admission = _json(raw); raw = _small_raw(directory/'scope.json'); spec = _json(raw)
    if (admission.get('schema') != ADMISSION or admission.get('schema_version') != 1
            or admission['runtime_implementation_sha256'] != implementation_sha256()
            or admission['scope_sha256'] != sha256(raw).hexdigest()
            or spec.get('schema') != 'scar.workspace-session-scope' or spec.get('schema_version') != 1
            or spec['cross_activation_residency_authorized'] is not False
            or spec['default_backend'] != 'NO_OP' or spec['trial_only'] is not True):
        raise GuardFailure('source scope/runtime policy differs from producer admission')
    parent = inspect_session(directory,expected_admission_sha256=admission['session_admission_sha256'])
    return dict(directory=directory,admission=admission,spec=spec,parent=parent,library_loaded=False)


def load_scope(bundle, *, expected_admission_sha256, trial=False, exclusive_inputs=False):
    admitted = inspect_scope_bundle(bundle,expected_admission_sha256=expected_admission_sha256)
    session = load_session(admitted['directory'],
        expected_admission_sha256=admitted['admission']['session_admission_sha256'],
        trial=trial is True and exclusive_inputs is True)
    return SessionScope(session,admitted['spec'],trial=trial,exclusive_inputs=exclusive_inputs)


def run_script(entry, argv, *, bundle, expected_admission_sha256, trial=False, exclusive_inputs=False):
    from .entry_runner import _arguments, _script_state
    entry, argv = _arguments(entry,argv)
    scope = load_scope(bundle,expected_admission_sha256=expected_admission_sha256,
        trial=trial,exclusive_inputs=exclusive_inputs)
    exited = False
    with scope.install():
        with _script_state(entry,argv,None):
            try: runpy.run_path(str(entry),run_name='__main__')
            except SystemExit as error:
                if error.code is not None and not (type(error.code) is int and error.code == 0): raise
                exited = True
    return dict(schema='scar.workspace-session-entry-execution',schema_version=1,
        entry=str(entry),entry_sha256=sha256(entry.read_bytes()).hexdigest(),argv=argv,
        scope_admission_sha256=expected_admission_sha256,runtime_implementation_sha256=implementation_sha256(),
        scope=scope.report,session_statistics=dict(scope.session.stats),
        explicit_trial=trial is True,system_exit_zero=exited,exit_code=0,
        source_caller_body_rewritten=False,outer_caller_control_replaced=False,
        cross_activation_residency_authorized=False,performance_measured=False,whole_program_cpp=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--validate-only',action='store_true');parser.add_argument('--exclusive-inputs',action='store_true')
    parser.add_argument('--report');parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('a Python script entry is required after --')
    if args.report and Path(args.report).exists():parser.error('report requires a fresh path')
    report=run_script(command[0],command[1:],bundle=args.bundle,
        expected_admission_sha256=args.admission_sha256,trial=args.validate_only,exclusive_inputs=args.exclusive_inputs)
    if args.report:
        with Path(args.report).open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
