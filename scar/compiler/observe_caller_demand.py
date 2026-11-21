"""Count starts at one original F4 caller boundary without measuring time."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import FunctionType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.entry_runner import _Scope, _arguments, _execute, _file
from scar.trace.code_snapshot import code_key


def observe_caller_demand(inputs, entry, argv, out, *, max_calls=4096):
    from scar.compiler.project_pipeline_temporal import INPUTS, open_temporal_inputs
    from scar.runtime.caller_pipeline_deployment import entry_context

    if type(max_calls) is not int or max_calls < 1:
        raise ValueError('positive caller observation budget required')
    monitor = getattr(sys, 'monitoring', None)
    if monitor is None or sys.gettrace() is not None or sys.getprofile() is not None:
        raise ValueError('untraced CPython with local monitoring required')
    entry, argv = _arguments(entry, argv)
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh caller observation path required')
    reference = artifact_reference(Path(inputs), INPUTS)
    with open_temporal_inputs(reference) as (pipeline, *_):
        source = pipeline.source
    pinned_source = Path(source.path).resolve()
    if pinned_source == entry or not source.qualname.isidentifier():
        raise ValueError('ordinary imported source caller required')
    context = entry_context(entry, argv)
    tool = next((index for index in range(6) if monitor.get_tool(index) is None), None)
    if tool is None:
        raise ValueError('no CPython monitoring tool ID available')

    class DemandScope(_Scope):
        def __init__(self):
            self.source = pinned_source
            self.codes = set()
            self.frontier = []
            self.bindings = []
            self.count = 0
            self.truncated = False

        def attach(self, module, name, *, natural):
            function = vars(module).get(source.qualname)
            if (_file(module) != self.source or type(function) is not FunctionType
                    or code_key(function.__code__) != source.code_key):
                self.frontier.append('original_caller_code_not_matched')
                return
            code = function.__code__
            self.codes.add(code)
            monitor.set_local_events(tool, code, monitor.events.PY_START)

        def record(self, code, offset):
            if self.count >= max_calls:
                self.truncated = True
                return monitor.DISABLE
            self.count += 1

        def finish(self):
            for code in self.codes:
                monitor.set_local_events(tool, code, 0)

    scope = DemandScope()
    monitor.use_tool_id(tool, 'scar_caller_demand')
    try:
        monitor.register_callback(tool, monitor.events.PY_START, scope.record)
        # _execute needs an empty binding list for reporting. We install code
        # events only, leaving the original functions and their aliases intact.
        execution = _execute(entry, argv, scope, 'original_caller_demand_observation')
    finally:
        scope.finish()
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.free_tool_id(tool)
    read_artifact(reference)
    if entry_context(entry, argv) != context:
        raise ValueError('original entry context changed during observation')
    if sha256(pinned_source.read_bytes()).hexdigest() != source.sha256:
        raise ValueError('original caller source changed during observation')
    report = dict(schema='scar.caller-entry-demand-observation', schema_version=1,
        original_inputs=reference, entry_context=context, caller_path=str(pinned_source),
        caller_code_key=source.code_key, caller_source_sha256=source.sha256,
        recorded_starts=scope.count, count_budget=max_calls, count_truncated=scope.truncated,
        attached_code_objects=len(scope.codes), frontier=scope.frontier,
        scope='ORIGINAL_MATCHED_CODE_STARTS_AFTER_STANDARD_IMPORT_FOR_EXACT_ARGV',
        whole_entry_coverage_proven=False, guaranteed_future_count=False,
        target_values_retained=False, performance_measured=False, native_executed=False,
        exit_code=execution['exit_code'])
    with output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--max-calls', type=int, default=4096)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python entry required after --')
    observe_caller_demand(args.inputs, command[0], command[1:], args.report, max_calls=args.max_calls)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
