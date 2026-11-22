"""Products exchanged between analysis, planning and C++20 delivery."""

from dataclasses import asdict, dataclass
from pathlib import Path

from scar.artifacts import file_reference
from scar.ir.execution_recipe import read_artifact
from scar.ir.program.providers import ProviderArtifact


@dataclass(frozen=True, slots=True)
class GraphBundle:
    """A program snapshot, its resource topology and the original evidence.

    A declaration graph can be retained before any executable region has
    been admitted. A declaration is evidence, not a substitute for program.
    """

    request: ProviderArtifact
    program: ProviderArtifact | None
    hardware: ProviderArtifact
    evidence: ProviderArtifact
    calls: ProviderArtifact | None
    report: dict
    declarations: ProviderArtifact | None = None

    @classmethod
    def read_native(cls, directory: Path) -> 'GraphBundle':
        """Open the graph-only publication and verify its role bindings."""
        directory = directory.resolve(strict=True)
        result = read_artifact(file_reference(directory / 'result.json', 'scar.native-entry-result'))
        if result.get('schema_version') != 1 or result.get('artifacts') is None:
            raise ValueError('expected a published native graph result')
        references = result['artifacts']
        schemas = {
            'request': 'scar.native-entry', 'program': 'scar.program-graph',
            'hardware': 'scar.target-hardware-graph', 'calls': 'scar.native-call-links',
            'graph_evidence': 'scar.native-graph-evidence',
        }
        if not isinstance(references, dict) or set(references) != schemas.keys():
            raise ValueError('native graph handoff requires request, program, hardware, calls and evidence')
        if any(not isinstance(references[name], dict) or references[name].get('schema') != schema
               for name, schema in schemas.items()):
            raise ValueError('native graph artifact schema differs from its role')
        parts = {name: ProviderArtifact(**reference) for name, reference in references.items()}
        evidence = parts['graph_evidence']
        if Path(evidence.uri) != directory / 'graph-report.json':
            raise ValueError('native graph report is outside its publication')
        report = read_artifact(evidence)
        if report.get('schema_version') != 1 or not report.get('entry'):
            raise ValueError('native graph report lacks a validated entry')
        for name in ('request', 'program', 'hardware', 'calls'):
            if report.get(name) != references[name]:
                raise ValueError(f'native graph report and result disagree on {name}')
        read_artifact(parts['calls'])
        return cls(parts['request'], parts['program'], parts['hardware'], evidence, parts['calls'], report)

    @classmethod
    def from_python_session(cls, session) -> 'GraphBundle':
        """Use the admitted catalog and exact manifest from this graph session."""
        reference = ProviderArtifact(**session.publication)
        report = session.report
        if (reference.schema != 'scar.program-build' or report.get('schema_version') != 1
                or session._f4 is None):
            raise ValueError('original F4 program build required')
        if not report.get('workflow_complete'):
            raise ValueError('F4 build did not finish graph admission')
        if report.get('target') is None:
            raise ValueError('program build has no target hardware snapshot')
        hardware = _legacy_reference(report['target'], 'scar.target-hardware-graph')
        declarations = _legacy_reference(report['f4']['catalog'], session.view.catalog.report['schema'])
        return cls(reference, None, hardware, reference, None, report, declarations)

    def references(self) -> dict:
        return {role: asdict(reference) for role, reference in (
            ('request', self.request), ('program', self.program), ('hardware', self.hardware),
            ('graph_evidence', self.evidence), ('calls', self.calls), ('declarations', self.declarations),
        ) if reference is not None}


def _legacy_reference(receipt, schema):
    return ProviderArtifact(uri=receipt['path'], size_bytes=receipt['bytes'],
                            sha256=receipt['sha256'], schema=schema)


def verify_source_lock(request):
    root = Path(request['source_root'])
    for name, expected in request['source_lock']['files'].items():
        relative = Path(name)
        source = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not source.is_relative_to(root):
            raise ValueError(f'source path outside of root: {name}')
        actual = file_reference(source, 'scar.native-source')
        if any(actual[key] != expected[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source differs: {name}')


@dataclass(frozen=True, slots=True)
class ExecutionPlanRef:
    """An entry decision sealed to its original problem and selected plan."""

    entry: str
    artifacts: dict
    decision: dict
    scope: str

    def __post_init__(self):
        if not self.entry or not self.scope:
            raise ValueError('execution plan requires an entry and a qualified scope')
        self.problem
        self.schedule

    def _reference(self, role: str, schema: str) -> ProviderArtifact:
        reference = self.artifacts.get(role)
        if not isinstance(reference, dict) or reference.get('schema') != schema:
            raise ValueError(f'execution plan has no {role} reference for {schema}')
        return ProviderArtifact(**reference)

    @property
    def problem(self) -> ProviderArtifact:
        from scar.ir.implementation_plan import SCHEMA

        return self._reference('planning_problem', SCHEMA)

    @property
    def schedule(self) -> ProviderArtifact:
        from scar.ir.implementation_plan import PLAN_SCHEMA

        return self._reference('execution_plan', PLAN_SCHEMA)

    def read_pair(self) -> tuple[dict, dict]:
        """Read both inputs against their original schema and byte identity."""
        return read_artifact(self.problem), read_artifact(self.schedule)

    def document(self) -> dict:
        return dict(schema='scar.compiler-plan', schema_version=1, **asdict(self))

    @classmethod
    def read(cls, path: Path) -> 'ExecutionPlanRef':
        value = read_artifact(file_reference(path, 'scar.compiler-plan'))
        version = value.pop('schema_version', None)
        if (type(version) is not int or version != 1
                or set(value) != {'schema', 'entry', 'artifacts', 'decision', 'scope'}):
            raise ValueError('unsupported compiler plan envelope')
        del value['schema']
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ScheduledLoop:
    """A region-specific lowering with a retained source proof and schedule."""

    loop_id: str
    folder: Path
    proof: dict
    source_motion: dict
    schedule: dict
    composite: bool
    parallel: bool
    lowering: dict
