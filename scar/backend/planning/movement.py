"""Price eligible payload copies and preserve their dynamic ordering constraints."""

from statistics import median

COPY_METRIC = "memcpy_payload_byte"
SAMPLE_BYTES = 1 << 23


def _contiguous(region, scalar_bytes):
    """Prove row-major byte order, including unit axes and symbolic outer extent."""
    shape, strides = region.get("shape"), region.get("strides_bytes")
    if shape is None or strides is None or len(shape) != len(strides):
        return False
    stride = scalar_bytes.get(region.get("dtype"))
    if stride is None:
        return False
    if 0 in shape:
        return True
    for extent, actual in zip(reversed(shape), reversed(strides)):
        if extent == 1:
            continue
        if stride is None or actual != stride:
            return False
        stride = None if extent is None else stride * extent
    return True


def _proxy_gap(operation, source, target, scalar_bytes):
    domains = {source["domain"], target["domain"]}
    if "filesystem" in domains:
        return "execution_mount_and_physical_io_unmeasured"
    if domains != {"host_memory"}:
        return "execution_device_and_transfer_route_unmeasured"
    if operation["value_relation"] != "equal_bytes":
        return "data_traversal_differs_from_contiguous_copy"
    opcode = operation["opcode"]
    if opcode == "memcpy":
        # A successful memcpy requires valid disjoint byte ranges.
        return None
    if opcode == "memmove":
        return "source_destination_overlap_unresolved"
    if opcode != "array_copy" or operation["destination_storage"] != "fresh":
        return "no_matching_copy_probe"
    regions = [m["region"] for m in (source, target)]
    if regions[0].get("dtype") != regions[1].get("dtype") or not all(
        _contiguous(r, scalar_bytes) for r in regions
    ):
        return "contiguous_equal_representation_unproven"
    return None


class Movement:
    """Index the validated M4 flow once before collecting B1 calibration."""

    def __init__(self, flow, scalar_bytes):
        self.coverage = flow["coverage"]
        self.materializations = {
            n["id"]: n for n in flow["nodes"] if n["kind"] == "MATERIALIZATION"
        }
        self.copies = []
        for node in flow["nodes"]:
            if node["kind"] == "COPY":
                source = self.materializations[node["source"]]
                target = self.materializations[node["target"]]
                self.copies.append(
                    (node, _proxy_gap(node, source, target, scalar_bytes))
                )

    def required_metrics(self):
        return {COPY_METRIC} if any(gap is None for _, gap in self.copies) else set()

    def plan(self, samples):
        values = [row[COPY_METRIC] for row in samples.get(1, ()) if COPY_METRIC in row]
        coefficient = median(values) if len(values) == 3 else None
        coefficients = {}
        if coefficient is not None:
            coefficients[COPY_METRIC] = dict(
                evidence="observed",
                operation="std::memcpy",
                unit="ns per payload byte",
                workers=1,
                sample_payload_bytes=SAMPLE_BYTES,
                buffers="disjoint_contiguous_pretouched",
                warmup=1,
                samples=values,
                minimum=min(values),
                median=coefficient,
                maximum=max(values),
            )

        copies = []
        for operation, gap in self.copies:
            source = self.materializations[operation["source"]]
            target = self.materializations[operation["target"]]
            count = operation["byte_count"]
            component = None
            if gap is None and coefficient is None:
                gap = "target_copy_coefficient_unavailable"
            if gap is None:
                # A symbolic count refers to this invocation's evaluated IR operand.
                component = dict(
                    evidence="inferred",
                    expression=dict(
                        op="multiply",
                        left=dict(
                            kind="logical_payload_bytes", operation=operation["id"]
                        ),
                        right=dict(
                            kind="coefficient", metric=COPY_METRIC, statistic="median"
                        ),
                    ),
                    estimated_ns=(
                        count["value"] * coefficient
                        if count["kind"] == "constant"
                        else None
                    ),
                )
            pending = (
                [gap]
                if gap
                else ["size_scaling_cache_and_runtime_placement_unmeasured"]
            )
            if operation["destination_storage"] == "fresh":
                pending.append("allocation_and_page_fault_cost_unmeasured")
            copies.append(
                dict(
                    id=operation["id"],
                    opcode=operation["opcode"],
                    owner=operation["owner"],
                    path=operation["path"],
                    site=operation.get("site"),
                    scope=operation["scope"],
                    conditions=operation.get("conditions", []),
                    dispatch=operation["dispatch"],
                    source=source["id"],
                    target=target["id"],
                    destination_storage=operation["destination_storage"],
                    value_relation=operation["value_relation"],
                    effects=operation["effects"],
                    byte_count=count,
                    ordering=dict(
                        blocking=operation["blocking"],
                        start_after=[source["ready"], {"kind": "operands_evaluated"}],
                        ready_on_success=target["ready"],
                        continuation="after_blocking_operation_returns",
                    ),
                    cost=dict(
                        copy_component=component,
                        total_duration_ns=None,
                        unresolved=pending,
                    ),
                )
            )
        return dict(
            coverage=self.coverage,
            policy="preserve_execution_ir_order",
            instance="per_dynamic_copy",
            byte_count_basis="logical_destination_payload",
            coefficients=coefficients,
            materializations={
                ident: {
                    key: m.get(key)
                    for key in (
                        "domain",
                        "operand",
                        "region",
                        "location",
                        "logical_value",
                        "ready",
                    )
                }
                for ident, m in self.materializations.items()
            },
            copies=copies,
        )
