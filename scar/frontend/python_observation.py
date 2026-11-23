"""Bounded source-local Python call observation for a declared input case."""

from dataclasses import asdict
import json
import subprocess
import sys

from scar.contract import Edge, artifact, verify_artifact, write_json

_OBSERVE = """
import json
from pathlib import Path
import runpy
import sys

source, entry, request, output = sys.argv[1:]
spec = json.loads(Path(request).read_bytes())
arguments = []
for row in spec['arguments']:
    if 'npy' in row:
        import numpy
        arguments.append(numpy.load(row['npy'], allow_pickle=False))
    else:
        arguments.append(row['scalar'])

events = []
def qualified(code):
    return getattr(code, 'co_qualname', code.co_name)

def profile(frame, event, unused):
    if event != 'call' or frame.f_code.co_filename != source:
        return
    caller = frame.f_back
    arrays = {
        name: {'shape': list(value.shape), 'strides': list(value.strides), 'dtype': str(value.dtype)}
        for name, value in frame.f_locals.items()
        if type(value).__module__ == 'numpy' and type(value).__name__ == 'ndarray'
    }
    events.append({'callee': qualified(frame.f_code), 'arrays': arrays,
                   'caller': qualified(caller.f_code) if caller.f_code.co_filename == source else None,
                   'line': caller.f_lineno})
    if len(events) >= spec['event_limit']:
        sys.setprofile(None)

function = runpy.run_path(source)[entry]
sys.setprofile(profile)
try:
    function(*arguments)
finally:
    sys.setprofile(None)
payload = {'events': events, 'truncated': len(events) >= spec['event_limit']}
Path(output).write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
"""


def observe_python(path, entry, spec, out, regions, operations):
    """Publish exact observed calls and add only uniquely anchored graph edges."""
    if not isinstance(entry, str) or not entry:
        raise ValueError("observation requires a named Python entry")
    limit = spec.get("event_limit", 256)
    timeout = spec.get("timeout_seconds", 10)
    if (
        type(limit) is not int
        or not 1 <= limit <= 4096
        or type(timeout) is not int
        or not 1 <= timeout <= 60
    ):
        raise ValueError("invalid bounded observation limits")
    arguments, receipts = [], []
    for arg in spec["arguments"]:
        if set(arg) == {"npy"}:
            reference = artifact(path.parent / arg["npy"])
            arguments.append(dict(npy=reference.path))
            receipts.append(reference)
        elif set(arg) == {"scalar"} and type(arg["scalar"]) in {
            int,
            float,
            bool,
            str,
            type(None),
        }:
            arguments.append(arg)
        else:
            raise ValueError("observation arguments require scalar or npy payloads")
    request = out / "observation-input.json"
    write_json(
        request,
        dict(
            arguments=arguments,
            event_limit=limit,
            inputs=[asdict(ref) for ref in receipts],
        ),
    )
    report = out / "observation.json"
    try:
        run = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _OBSERVE,
                str(path),
                entry,
                str(request),
                str(report),
            ],
            cwd=path.parent,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            "bounded Python observation exceeded its time limit"
        ) from error
    if run.returncode:
        raise ValueError(f"bounded observation failed: {run.stderr.strip()}")
    for reference in receipts:
        verify_artifact(reference)

    functions = {}
    for region in regions:
        if region.kind == "FUNCTION":
            functions.setdefault(region.name, []).append(region.id)
    by_id = {region.id: region for region in regions}
    parent = {child: region.id for region in regions for child in region.children}
    sites = {}
    for op in operations:
        if op.kind != "CALL" or not op.attributes.get("source_span"):
            continue
        owner = by_id[op.region]
        while owner.kind != "FUNCTION" and owner.id in parent:
            owner = by_id[parent[owner.id]]
        if owner.kind == "FUNCTION":
            sites.setdefault((owner.name, op.attributes["source_span"][0]), []).append(
                op.id
            )
    reference = artifact(report)
    events = json.loads(report.read_bytes())["events"]
    edges = set()
    for event in events:
        caller = sites.get((event["caller"], event["line"]), ())
        callee = functions.get(event["callee"], ())
        if (
            len(caller) == len(callee) == 1
            and len(functions.get(event["caller"], ())) == 1
        ):
            edges.add(Edge(caller[0], callee[0], "OBSERVED_CALL", event["callee"]))
    return reference, tuple(sorted(edges, key=lambda edge: (edge.source, edge.target)))
