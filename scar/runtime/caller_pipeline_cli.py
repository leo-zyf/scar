"""Run an unchanged entry with a compact caller-pipeline validation bundle.

Cost is unresolved for this host profile. The ordinary execution is the
default; an explicit candidate measurement can attach the admitted boundary.
"""
import argparse
import json
from pathlib import Path


def run_caller_entry(entry, argv, *, bundle, expected_admission_sha256,
        exclusive_inputs=False, measure_candidate=False):
    from .caller_pipeline_guard import _admit
    from .entry_runner import _arguments
    entry,argv=_arguments(entry,argv)
    admission=_admit(bundle,expected_admission_sha256)
    return _run_with_admission(entry,argv,admission,
        exclusive_inputs=exclusive_inputs,measure_candidate=measure_candidate)


def _run_with_admission(entry,argv,admission,*,exclusive_inputs,measure_candidate):
    from .caller_pipeline_guard import _load_admitted,guarded_caller_pipeline
    from .entry_runner import _Scope,_execute,run_original_script
    contract=admission.contract
    gate=dict(decision='NO_OP',deployment_authorized=False,
        reasons=['caller_pipeline_cost_and_deployment_plan_not_admitted'])
    scope=None
    if measure_candidate is True:
        if exclusive_inputs is not True:
            gate['reasons']=['exclusive_inputs_not_explicitly_true']
        elif Path(contract['source']['path']).resolve()==entry:
            gate['reasons']=['entry_defined_caller_attachment_not_supported']
        else:
            artifact=_load_admitted(admission)

            class CallerScope(_Scope):
                def __init__(self):
                    self.stateful=False;self.artifact=artifact;self.mismatch=ValueError
                    self.source=Path(contract['source']['path']).resolve()
                    self.name=contract['source']['qualname']
                    self.active=True;self.bindings=[];self.seen=set();self.frontier=[]
                    self.make_guard=lambda original,artifact,exclusive_inputs:guarded_caller_pipeline(
                        original,artifact,exclusive_inputs=exclusive_inputs,measure_candidate=True)

            scope=CallerScope()
            gate=dict(decision='MEASURE_ADMITTED_CANDIDATE',deployment_authorized=False,
                reasons=['explicit_candidate_measurement_is_not_cost_approval'])
    execution=(_execute(entry,argv,scope,'ordinary_import_caller_boundary')if scope is not None
        else run_original_script(entry,argv))
    return dict(schema='scar.caller-pipeline-entry-execution',schema_version=1,
        deployment_gate=gate,execution=execution,
        default_no_op=measure_candidate is not True,
        native_libraries_loaded=scope is not None,import_attachment_enabled=scope is not None,
        mode='MEASUREMENT_ONLY'if measure_candidate else 'ORIGINAL_NO_OP',
        performance_measured=False,full_program_cpp=False,whole_program_performance_proven=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--exclusive-inputs',action='store_true')
    parser.add_argument('--measure-candidate','--validate-only',action='store_true',
        help='explicit guarded execution for validation; no timing or deployment approval')
    parser.add_argument('--report')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command[1:]if args.command[:1]==['--']else args.command
    if not command:parser.error('an unchanged Python entry is required after --')
    report_path=Path(args.report).resolve()if args.report else None
    if report_path is not None and report_path.exists():parser.error('fresh --report path required')
    result=run_caller_entry(command[0],command[1:],bundle=args.bundle,
        expected_admission_sha256=args.admission_sha256,exclusive_inputs=args.exclusive_inputs,
        measure_candidate=args.measure_candidate)
    if report_path is not None:
        report_path.parent.mkdir(parents=True,exist_ok=True)
        with report_path.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
