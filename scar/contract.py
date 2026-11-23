"""Shared wire types for graph, schedule, source and resource handoffs."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


GRAPH_SCHEMA = "scar.graph-bundle"
PLAN_SCHEMA = "scar.compiler-plan"
WIRE_VERSION = 3


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    size: int

    def __post_init__(self):
        if (not Path(self.path).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
                or type(self.size) is not int or self.size < 0):
            raise ValueError("invalid artifact identity")


@dataclass(frozen=True)
class Member:
    """A file relative to the product manifest that owns it."""

    path: str
    sha256: str
    size: int

    def resolve(self, root):
        name = Path(self.path)
        root = Path(root).resolve(strict=True)
        if (name.is_absolute() or name.as_posix() != self.path or self.path == "."
                or ".." in name.parts or (root / name).resolve(strict=True) != root / name):
            raise ValueError(f"invalid local product member {self.path}")
        return Artifact(str(root / name), self.sha256, self.size)


@dataclass(frozen=True)
class Operation:
    id: str
    region: str
    kind: str
    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    id: str
    kind: str
    name: str
    operations: tuple[str, ...] = ()
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    value: str | None = None


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    capacity: int | None = None
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Graph:
    target: dict
    regions: tuple[Region, ...]
    operations: tuple[Operation, ...]
    edges: tuple[Edge, ...]
    resources: tuple[Resource, ...]
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    operation: str
    resource: str
    start_ns: int | None = None
    duration_ns: int | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class FragmentModule:
    """Opaque C++20-compatible leaves retained by the source adapters."""

    artifact: Member
    kind: str = field(default="FRAGMENT_MODULE", init=False)


@dataclass(frozen=True)
class ElementwiseEntry:
    """A typed entry contract for float64 operations over independent elements."""

    program: dict
    shape: tuple[int, int]
    count: int
    kind: str = field(default="ELEMENTWISE_ENTRY", init=False)


@dataclass(frozen=True)
class ExecutableIR:
    """The sole code generation input. Source language is absent from this layer."""

    bodies: tuple[FragmentModule | ElementwiseEntry, ...] = ()
    cpu_ids: tuple[int, ...] = ()
    cxx_standard: int = 20


@dataclass(frozen=True)
class Plan:
    graph: Artifact
    validation: dict
    rewrites: tuple[dict, ...]
    schedule: tuple[Step, ...]
    lowering: ExecutableIR
    decision: dict


def _identity(path, data):
    return Artifact(str(path), sha256(data).hexdigest(), len(data))


def artifact(path):
    path = Path(path).resolve(strict=True)
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
            size += len(chunk)
    return Artifact(str(path), digest.hexdigest(), size)


def read_artifact(reference):
    if isinstance(reference, dict):
        reference = Artifact(**reference)
    path = Path(reference.path)
    data = path.read_bytes()
    if _identity(path, data) != reference:
        raise ValueError(f"artifact changed: {path}")
    return data


def verify_artifact(reference):
    if isinstance(reference, dict):
        reference = Artifact(**reference)
    if artifact(reference.path) != reference:
        raise ValueError(f"artifact changed: {reference.path}")


def copy_artifact(reference, destination, *, executable=False):
    if isinstance(reference, dict):
        reference = Artifact(**reference)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    size = 0
    with Path(reference.path).open("rb") as original, destination.open("xb") as output:
        for chunk in iter(lambda: original.read(2**20), b""):
            digest.update(chunk)
            output.write(chunk)
            size += len(chunk)
    if (size, digest.hexdigest()) != (reference.size, reference.sha256):
        destination.unlink()
        raise ValueError(f"artifact changed: {reference.path}")
    if executable:
        destination.chmod(0o755)
    return Artifact(str(destination), reference.sha256, reference.size)


def local_member(reference, root):
    return Member(Path(reference.path).relative_to(Path(root).resolve()).as_posix(), reference.sha256, reference.size)


def write_json(path, document):
    path = Path(path).resolve()
    encoded = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode("utf-8")
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(encoded)
            stream.flush()
            os.link(temp, path)
        finally:
            temp.unlink()
    return _identity(path, encoded)


def _open(path, filename, schema):
    path = Path(path)
    if path.is_dir():
        path = path / filename
    path = path.resolve(strict=True)
    data = path.read_bytes()
    document = json.loads(data)
    if type(document) is not dict or document.pop("schema", None) != schema:
        raise ValueError(f"invalid {schema} document")
    if document.pop("schema_version", None) != WIRE_VERSION:
        raise ValueError(f"unsupported {schema} version")
    return _identity(path, data), document


def _rows(rows, model, tuple_fields=()):
    return tuple(model(**{**row, **{key: tuple(row[key]) for key in tuple_fields}}) for row in rows)


def validate_graph(graph):
    regions = {row.id: row for row in graph.regions}
    operations = {row.id: row for row in graph.operations}
    identities = (*regions, *operations, *(row.id for row in graph.resources))
    count = len(graph.regions) + len(graph.operations) + len(graph.resources)
    if len(set(identities)) != count or any(not isinstance(key, str) or not key for key in identities):
        raise ValueError("graph IDs must be nonempty and globally unique")
    assigned, parent = set(), {}
    for region in graph.regions:
        for key in region.operations:
            if key in assigned or key not in operations or operations[key].region != region.id:
                raise ValueError(f"operation ownership differs in region {region.id}")
            assigned.add(key)
        for child in region.children:
            if child not in regions or child in parent:
                raise ValueError(f"invalid nested region {child}")
            parent[child] = region.id
    if assigned != operations.keys():
        raise ValueError("every operation must belong to one region")
    visited = set()
    for start in parent:
        node, active = start, set()
        while node in parent and node not in visited:
            if node in active:
                raise ValueError("region containment cycle")
            active.add(node)
            node = parent[node]
        visited.update(active)
    known = set(identities)
    for edge in graph.edges:
        if edge.source not in known or edge.target not in known or not edge.kind:
            raise ValueError("graph edge has unresolved endpoint or type")
    for resource in graph.resources:
        if resource.capacity is not None and (type(resource.capacity) is not int or resource.capacity < 0):
            raise ValueError(f"invalid resource capacity: {resource.id}")
    return graph


def write_graph(path, graph):
    return write_json(path, dict(schema=GRAPH_SCHEMA, schema_version=WIRE_VERSION, **asdict(validate_graph(graph))))


def open_graph(path):
    reference, value = _open(path, "graph.json", GRAPH_SCHEMA)
    graph = Graph(
        value["target"],
        _rows(value["regions"], Region, ("operations", "children")),
        _rows(value["operations"], Operation, ("inputs", "outputs", "effects")),
        _rows(value["edges"], Edge),
        _rows(value["resources"], Resource),
        value["evidence"],
    )
    return reference, validate_graph(graph)


def read_graph(path):
    return open_graph(path)[1]


def validate_plan(plan):
    steps = {step.operation: step for step in plan.schedule}
    if len(steps) != len(plan.schedule):
        raise ValueError("a plan operation can be scheduled only once")
    for step in plan.schedule:
        for value in (step.start_ns, step.duration_ns):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"invalid schedule time for {step.operation}")
        for key in step.depends_on:
            if key == step.operation or key not in steps:
                raise ValueError(f"unbound schedule dependency for {step.operation}")
    lowering = plan.lowering
    if (lowering.cxx_standard != 20 or len(lowering.bodies) > 1
            or len(set(lowering.cpu_ids)) != len(lowering.cpu_ids)
            or any(type(cpu) is not int or cpu < 0 for cpu in lowering.cpu_ids)):
        raise ValueError("invalid executable IR target or CPU allocation")
    for body in lowering.bodies:
        if isinstance(body, ElementwiseEntry):
            if (not lowering.cpu_ids or len(body.shape) != 2
                    or any(type(size) is not int or size <= 0 for size in body.shape)
                    or type(body.count) is not int or not 0 < body.count < 2**63):
                raise ValueError("invalid elementwise entry shape or count")
        elif not isinstance(body, FragmentModule):
            raise ValueError("unsupported executable IR body")
    return plan


def write_plan(path, plan):
    return write_json(path, dict(schema=PLAN_SCHEMA, schema_version=WIRE_VERSION, **asdict(validate_plan(plan))))


def _lowering_from_dict(document):
    if set(document) != {"bodies", "cpu_ids", "cxx_standard"}:
        raise ValueError("invalid executable IR fields")
    bodies = []
    for original in document["bodies"]:
        row = dict(original)
        kind = row.pop("kind")
        if kind == "ELEMENTWISE_ENTRY":
            row["shape"] = tuple(row["shape"])
            bodies.append(ElementwiseEntry(**row))
        elif kind == "FRAGMENT_MODULE":
            bodies.append(FragmentModule(Member(**row["artifact"])))
        else:
            raise ValueError(f"unsupported executable IR body {kind}")
    return ExecutableIR(tuple(bodies), tuple(document["cpu_ids"]), document["cxx_standard"])


def open_plan(path):
    reference, value = _open(path, "plan.json", PLAN_SCHEMA)
    value["graph"] = Artifact(**value["graph"])
    value["rewrites"] = tuple(value["rewrites"])
    value["schedule"] = _rows(value["schedule"], Step, ("depends_on",))
    value["lowering"] = _lowering_from_dict(value["lowering"])
    plan = validate_plan(Plan(**value))
    return reference, plan


def read_plan(path):
    return open_plan(path)[1]
