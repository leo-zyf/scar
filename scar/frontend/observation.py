"""Observe a declared numeric function invocation, retaining only input metadata and calls."""

import json
from pathlib import Path
import re
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


_SPAN = ("line", "end_line", "column", "end_column")


def _binding_sample(sample, line):
    """Validate the collector payload before the graph supplies the port identity."""
    if not isinstance(sample, dict):
        raise ValueError("invalid import binding sample fields")
    loc = sample.get("location")
    if (
        not isinstance(loc, dict)
        or set(loc) != set(_SPAN)
        or any(type(loc[k]) is not int or loc[k] < 1 for k in _SPAN)
        or loc["line"] != line
        or (loc["end_line"], loc["end_column"]) <= (loc["line"], loc["column"])
        or type(sample.get("owner")) is not str
        or not sample["owner"]
    ):
        raise ValueError("invalid import binding sample position")
    if sample.get("phase") == "before_IMPORT_NAME":
        i = sample.get("instruction")
        binding = sample.get("import_callable")
        if not (
            set(sample)
            == {
                "owner",
                "location",
                "phase",
                "code_sha256",
                "instruction",
                "import_callable",
                "registry_state",
            }
            and isinstance(sample["code_sha256"], str)
            and re.fullmatch("[a-f0-9]{64}", sample["code_sha256"])
            and isinstance(i, dict)
            and set(i) == {"opcode", "offset", "index", "name"}
            and i["opcode"] == "IMPORT_NAME"
            and all(type(i[k]) is int and i[k] >= 0 for k in ("offset", "index"))
            and type(i["name"]) is str
            and all(part.isidentifier() for part in i["name"].split("."))
            and sample["registry_state"]
            in ("absent", "present_none", "present_module", "present_other", "unknown")
            and isinstance(binding, dict)
            and (
                binding.get("kind") in ("absent", "other", "unknown")
                and set(binding) == {"kind"}
                or binding.get("kind") in ("builtin_function", "python_function")
                and not set(binding) - {"kind", "module", "qualname"}
                and all(type(v) is str and 0 < len(v) <= 4096 for v in binding.values())
            )
        ):
            raise ValueError("invalid pre-import instruction sample")
        return sample
    if (
        set(sample)
        != {"owner", "location", "binding_index", "member", "binding", "result"}
        or any(
            type(sample[k]) is not str or not sample[k] for k in ("member", "binding")
        )
        or type(sample["binding_index"]) is not int
        or sample["binding_index"] < 0
    ):
        raise ValueError("invalid import binding sample fields")
    result = sample["result"]
    if isinstance(result, dict):
        kind = result.get("kind")
        if kind in ("function", "type") and set(result) == {
            "kind",
            "module",
            "qualname",
        }:
            if all(
                type(result[k]) is str and 0 < len(result[k]) <= 4096
                for k in ("module", "qualname")
            ):
                return sample
        if kind in ("absent", "other", "none") and set(result) == {"kind"}:
            return sample
        if kind in ("int", "bool") and set(result) == {"kind", "value"}:
            value = result["value"]
            if (
                type(value) is (int if kind == "int" else bool)
                and value.bit_length() <= 64
            ):
                return sample
        if kind == "module" and set(result) == {
            "kind",
            "name",
            "same_registry_instance",
        }:
            if (
                type(result["name"]) is str
                and result["name"]
                and type(result["same_registry_instance"]) in (bool, type(None))
            ):
                return sample
    raise ValueError("invalid import binding sample value")


def _plain_instruction_key(owner, loc, code_sha, instruction):
    return (
        owner,
        *(loc[k] for k in _SPAN),
        code_sha,
        *(instruction[k] for k in ("opcode", "offset", "index", "name")),
    )


def link_import_observation(observed, units, images, folder):
    """Join observations of imports and completed bindings to source ports in one image."""
    from scar.contract import digest, json_loads, read_member

    image, entry_sha, raw = observed
    record = json_loads(raw)
    if (
        not isinstance(record, dict)
        or record.get("schema") != "scar.python-import-observation"
        or record.get("schema_version") != 1
        or record.get("scope")
        not in (
            "instrumented_script_successful_module_bindings",
            "instrumented_script_import_sites",
        )
        or entry_sha != record.get("entry_sha256")
        or not isinstance(record.get("source_sha256"), str)
        or not re.fullmatch("[a-f0-9]{64}", record["source_sha256"])
        or type(record.get("line")) is not int
        or record["line"] < 1
        or type(record.get("interpreter")) is not str
        or not record["interpreter"]
        or not isinstance(record.get("events"), list)
        or len(record["events"]) > 4096
        or type(record.get("dropped")) is not int
        or record["dropped"] < 0
    ):
        raise ValueError("invalid import observation or mismatched script bytes")
    if images is not None:
        selected = next(v for v in images if v["id"] == image)
        units = [units[i] for i in selected["units"]]
    ports = {}
    for unit in units:
        facts = unit["source_facts"]
        if (
            unit["language"] != "python"
            or facts.get("compiler") != record.get("interpreter")
            or facts["files"].get(unit["path"]) != record["source_sha256"]
        ):
            continue
        for fn in read_member(folder, unit["graph"])["functions"]:
            scope = fn["signature"].get("python_scope", {})
            if scope.get("kind") != "module":
                continue
            owner = "__main__" if scope.get("main") else scope["module"]
            for port, node in enumerate(fn["nodes"]):
                if node["op"] != "python_import":
                    continue
                loc, req = node["location"], node["request"]
                if req["form"] == "from":
                    key = (
                        owner,
                        *(loc[k] for k in _SPAN),
                        req["binding_index"],
                        node["member"],
                        node.get("alias") or node["member"],
                    )
                elif req["form"] == "import" and (
                    proof := node.get("source_import_store")
                ):
                    code = scope.get("source_code", {})
                    ins = proof["import_name"]
                    if not (
                        code.get("source_key") == unit["path"] == loc["file"]
                        and code.get("source_sha256") == record["source_sha256"]
                        and proof.get("code_sha256") == code.get("code_sha256")
                        and proof["position"]
                        == [
                            loc["line"],
                            loc["end_line"],
                            loc["column"] - 1,
                            loc["end_column"] - 1,
                        ]
                        and req["level"] == 0
                        and req["fromlist"] == []
                        and ins["name"] == req["module"]
                    ):
                        continue
                    key = _plain_instruction_key(owner, loc, proof["code_sha256"], ins)
                else:
                    continue
                ports.setdefault(key, []).append(f"{fn['id']}/{port}")
    events = []
    for raw_sample in record["events"]:
        sample = _binding_sample(raw_sample, record["line"])
        if "phase" in sample:
            if record["scope"] != "instrumented_script_import_sites":
                raise ValueError("pre-import sample requires an instruction capture")
            key = _plain_instruction_key(
                sample["owner"],
                sample["location"],
                sample["code_sha256"],
                sample["instruction"],
            )
        else:
            key = (
                sample["owner"],
                *(sample["location"][k] for k in _SPAN),
                sample["binding_index"],
                sample["member"],
                sample["binding"],
            )
        sites = ports.get(key, [])
        joined = (
            {"site": sites[0]}
            if len(sites) == 1
            else {
                "unresolved": (
                    "no_matching_port" if not sites else "ambiguous_source_port"
                )
            }
        )
        events.append({**sample, **joined})
    return dict(
        image=image,
        capture_sha256=digest(raw),
        entry_sha256=entry_sha,
        source_sha256=record["source_sha256"],
        interpreter=record["interpreter"],
        match_basis="entry_and_observed_module_source_revisions_and_port",
        scope=record["scope"],
        dropped=record["dropped"],
        events=events,
    )


def attach_import_observation(index, raw_path, *, image=None):
    """Attach one import sample using the sealed script image, without its sources."""
    from scar.contract import (
        GRAPH,
        digest,
        json_bytes,
        member,
        open_product,
        publish,
        read_blob,
        read_member,
        validate_images,
    )

    index, _, graph = open_product(Path(index).resolve(), "graph.json", GRAPH)
    folder, units = index.parent, graph["units"]
    images, entry = graph.get("images"), graph["entry"]
    selected = units
    if images is not None:
        validate_images(images, graph["root_image"], unit_count=len(units))
        if image is None and len(images) == 1:
            image = images[0]["id"]
        target = next((row for row in images if row["id"] == image), None)
        if target is None:
            raise ValueError("select a declared script image with --image")
        entry, selected = target["entry"], [units[i] for i in target["units"]]
    elif image is not None:
        raise ValueError("single-program graph has no named image")

    # The compact index identifies candidate units. Bind the entry to the
    # actual module initializer in its checksummed graph member.
    revisions = []
    for unit in selected:
        if unit["language"] != "python" or not any(
            fn["name"] == entry for fn in unit["functions"]
        ):
            continue
        for fn in read_member(folder, unit["graph"])["functions"]:
            signature = fn["signature"]
            scope = signature.get("python_scope", {})
            if (
                signature["name"] == entry == scope.get("initializer")
                and scope.get("kind") == "module"
                and scope.get("main") is True
                and scope.get("source_key") == unit["path"]
            ):
                revisions.append(unit["source_facts"]["files"].get(unit["path"]))
    if (
        len(revisions) != 1
        or not isinstance(revisions[0], str)
        or not re.fullmatch("[a-f0-9]{64}", revisions[0])
    ):
        raise ValueError("selected image requires one sealed Python script entry")

    sample = link_import_observation(
        (image, revisions[0], Path(raw_path).read_bytes()), units, images, folder
    )
    facts = graph.setdefault("source_facts", {})
    if "import_binding_sample" in facts:
        previous = facts["import_binding_sample"]
        if (
            not isinstance(previous, dict)
            or previous.get("image") != image
            or read_blob(folder, previous["evidence"]) != json_bytes(sample)
        ):
            raise ValueError("graph already contains a different import observation")
    else:
        receipt = member(folder, "observations", sample)
        read_blob(folder, receipt)
        facts["import_binding_sample"] = dict(image=image, evidence=receipt)

    data = json_bytes(graph)
    derived = folder / f"graph-{digest(data)}.json"
    if derived.exists():
        if derived.read_bytes() != data:
            raise ValueError("existing content-addressed graph has different bytes")
    else:
        publish(derived, data)
    return derived


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Attach an import observation to a sealed graph."
    )
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--image")
    parser.add_argument("--import-observation", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        index = attach_import_observation(
            args.graph, args.import_observation, image=args.image
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    print(index)


if __name__ == "__main__":
    main()
