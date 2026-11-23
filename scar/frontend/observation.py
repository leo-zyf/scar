"""Observe a declared numeric function invocation, retaining only input metadata and calls."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

_CAPTURE = """
import json, runpy, sys
from pathlib import Path
import numpy as np
source, entry, output = sys.argv[1:]
spec = json.load(sys.stdin)
args = [np.load(a['npy'], allow_pickle=False) if 'npy' in a else a['scalar'] for a in spec['arguments']]
meta = [dict(dtype=str(a.dtype),shape=list(a.shape),strides=list(a.strides),c_order=bool(a.flags.c_contiguous))
        if type(a) is np.ndarray else dict(scalar=a) for a in args]
events = []
def call(frame, event, unused):
    if event == 'call' and frame.f_code.co_filename == source:
        parent=frame.f_back
        events.append(dict(kind='OBSERVED_CALL',callee=frame.f_code.co_name,
                           caller=parent.f_code.co_name if parent.f_code.co_filename==source else None,
                           line=parent.f_lineno))
        if len(events) >= spec['event_limit']:
            sys.setprofile(None)
func = runpy.run_path(source)[entry]
sys.setprofile(call)
try:
    func(*args)
finally:
    sys.setprofile(None)
Path(output).write_text(json.dumps(dict(arguments=meta,events=events,limit_reached=len(events)>=spec['event_limit'])))
"""


def observe(manifest, root, spec):
    request = spec["observe"]
    args = []
    for a in request["arguments"]:
        if set(a) == {"npy"}:
            args.append({"npy": str((manifest.parent / a["npy"]).resolve(strict=True))})
        elif set(a) == {"scalar"} and type(a["scalar"]) in (int, float, bool):
            args.append(a)
        else:
            raise ValueError("observation expects numeric scalars or .npy arrays")
    limit = request.get("event_limit", 256)
    timeout = request.get("timeout_seconds", 10)
    if (
        type(limit) is not int
        or not 1 <= limit <= 4096
        or type(timeout) is not int
        or not 1 <= timeout <= 60
    ):
        raise ValueError("invalid bounded Python observation")
    source = (root / spec["source"]).resolve(strict=True)
    with TemporaryDirectory(prefix="scar-observe-") as temp:
        capture = Path(temp) / "events.json"
        run = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _CAPTURE,
                str(source),
                spec["entry"],
                str(capture),
            ],
            cwd=source.parent,
            input=json.dumps(dict(arguments=args, event_limit=limit)),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if run.returncode:
            raise ValueError(f"observation failed: {run.stderr[-1500:]}")
        return json.loads(capture.read_bytes())


def array_abi(entry, evidence):
    args = evidence["arguments"]
    if len(args) < 2 or type(args[-1].get("scalar")) is not int:
        raise ValueError(
            "binary array entry requires arrays and one integer trip count"
        )
    arrays = args[:-1]
    shape = arrays[0].get("shape")
    if not shape or not all(
        a.get("dtype") == "float64" and a.get("shape") == shape and a.get("c_order")
        for a in arrays
    ):
        raise ValueError(
            "binary array entry requires equal-shaped contiguous float64 arrays"
        )
    return dict(
        kind="binary_f64_arrays",
        shape=shape,
        count=args[-1]["scalar"],
        inputs=len(arrays),
        entry=entry,
    )
