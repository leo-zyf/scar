"""Attach one source-proven stateful loop at an ordinary import boundary.

Default execution remains Python. Explicit validation executes the unchanged
transactional step ABI through its admitted C++ control adapter, without any
compiler, planner or graph import at runtime.
"""
import argparse
import json
from pathlib import Path


def run_stateful_loop_entry(entry,argv,*,bundle,expected_admission_sha256,
        exclusive_inputs=False,trial=False):
    from .native_stateful_repeat import inspect_bundle
    from .entry_runner import _Scope,_execute,_arguments,run_original_script
    entry,argv=_arguments(entry,argv)
    contract=inspect_bundle(bundle,expected_admission_sha256=expected_admission_sha256)['contract']
    gate=dict(decision='NO_OP',deployment_authorized=False,reason='unmeasured_loop_requires_explicit_validation')
    scope=None
    if trial is True and exclusive_inputs is True and Path(contract['source']['path']).resolve()!=entry:
        from .native_stateful_repeat import load
        from .stateful_loop_guard import guarded_stateful_loop
        artifact=load(bundle,expected_admission_sha256=expected_admission_sha256)
        class LoopScope(_Scope):
            def __init__(self):
                self.stateful=False;self.artifact=artifact;self.mismatch=ValueError
                self.source=Path(contract['source']['path']).resolve();self.name=contract['source']['qualname']
                self.active=True;self.bindings=[];self.seen=set();self.frontier=[]
                self.make_guard=lambda original,artifact,exclusive_inputs:guarded_stateful_loop(
                    original,artifact,exclusive_inputs=exclusive_inputs,trial=True)
        scope=LoopScope();gate.update(decision='EXPLICIT_UNMEASURED_VALIDATION',reason='source_proven_loop_trial')
    execution=(_execute(entry,argv,scope,'ordinary_import_stateful_loop_boundary')if scope is not None
        else run_original_script(entry,argv))
    return dict(schema='scar.stateful-loop-entry-execution',schema_version=1,execution=execution,
        loop_statistics=[]if scope is None else[dict(row['stats'])for row in scope.bindings],
        deployment_gate=gate,native_libraries_loaded=scope is not None,
        performance_measured=False,release_authority=False,whole_program_cpp=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--exclusive-inputs',action='store_true');parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--report',required=True);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command[1:]if args.command[:1]==['--']else args.command
    if not command:parser.error('unchanged Python entry required after --')
    report=Path(args.report).resolve()
    if report.exists():parser.error('fresh report path required')
    result=run_stateful_loop_entry(command[0],command[1:],bundle=args.bundle,
        expected_admission_sha256=args.admission_sha256,exclusive_inputs=args.exclusive_inputs,trial=args.validate_only)
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
