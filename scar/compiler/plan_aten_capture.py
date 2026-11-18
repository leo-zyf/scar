"""Replay original F4 and ATen evidence into the common planning problem."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.analysis.aten_source_calls import corroborate_aten_routes, project_aten_source_calls
from scar.analysis.build import BuildSession
from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import problem_document
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode
from scar.optimization.aten_observed_plan import build_observed_aten_problem
from scar.optimization.constructive_planner import construct_implementations
from scar.trace.aten_scoped_ports import decode_scoped_aten_ports, read_scoped_aten_capture
from scar.trace.capture_coordinates import bind_capture_coordinates


def derive_aten_capture_plan(build, capture, attempts):
    """Keep the source-only checks ahead of the installed provider parser."""
    build = Path(build).resolve()
    program = artifact_reference(build / 'manifest.json', 'scar.program-build')
    coordinates = bind_capture_coordinates(capture)
    scoped = read_scoped_aten_capture(coordinates, call_attempts=attempts)
    with BuildSession(build) as session:
        if session.report.get('backend') != 'F4':
            raise ValueError('original F4 program build required')
        source = project_aten_source_calls(session.view, scoped)
        receipt = session.report['target']
        target = decode(ProviderArtifact, artifact_reference(receipt['path'], 'scar.target-hardware-graph'))
        if (target.size_bytes, target.sha256) != (receipt['bytes'], receipt['sha256']):
            raise ValueError('original planning target changed during source replay')
    native = decode_scoped_aten_ports(scoped)
    corroborate_aten_routes(source, native)
    problem, projection = build_observed_aten_problem((program,), target, source, native)
    plan = construct_implementations(problem, baseline_only=True)
    return dict(inputs=dict(program=program, capture=str(Path(capture).resolve()),
                            call_attempts=[row['attempt'] for row in source['calls']]),
                problem=problem_document(problem), source=source, native=native,
                projection=projection, execution_plan=plan)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-aten-capture', description=__doc__)
    parser.add_argument('--build', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--call-attempt', action='append', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    records = derive_aten_capture_plan(args.build, args.capture, args.call_attempt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-plan-aten-', dir=out.parent) as temp:
        root = Path(temp)
        for name, value in records.items():
            (root / (name.replace('_', '-') + '.json')).write_text(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        root.rename(out)
    print(json.dumps(dict(status=records['execution_plan']['status'], decision=records['projection']['decision'],
        calls=len(records['source']['calls']), reference_routes=len(records['source']['data_routes']), output=str(out))))
    return 0
