"""Products exchanged between analysis, planning and C++20 delivery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING

from scar.artifacts import file_reference
from scar.ir.execution_recipe import open_artifact, read_artifact
from scar.ir.program.providers import ProviderArtifact

if TYPE_CHECKING:
    from scar.ir.program.model import ProgramGraph


@dataclass(frozen=True, slots=True)
class NativeInputs:
    manifest: dict
    request: ProviderArtifact
    standards: dict[str, str | None]


def source_standard(flags) -> str | None:
    """Locate the selected dialect without interpreting a compiler's names."""
    arguments = iter(flags)
    standard = None
    for flag in arguments:
        if flag.startswith(('-std=', '--std=')):
            standard = flag.partition('=')[2]
        elif flag in ('-std', '--std'):
            standard = next(arguments, None)
            if not standard:
                raise ValueError('missing language standard in native compiler flags')
    return standard


def lock_native_inputs(path: Path, out: Path, *, languages=('c', 'c++', 'fortran')) -> NativeInputs:
    manifest = json.loads(path.read_bytes())
    if (manifest.get('schema'), manifest.get('schema_version')) != ('scar.native-entry', 1):
        raise ValueError('expected scar.native-entry schema v1')
    dialects = set(languages)
    language = manifest['language']
    if language not in dialects | {'mixed'}:
        raise ValueError(f'unsupported language: {language}')
    root = Path(manifest['source_root']).resolve(strict=True)
    manifest['source_root'] = str(root)
    verify_source_lock(manifest)
    receipt = manifest['source_lock']['files']
    units = manifest['units']
    names = {unit['path'] for unit in units}
    if not names or len(names) != len(units):
        raise ValueError('target requires uniquely named translation units')
    if names - receipt.keys():
        raise ValueError('translation units missing source lock: ' + ', '.join(sorted(names - receipt.keys())))
    standards = {}
    for unit in units:
        if unit.get('language', language) not in dialects:
            raise ValueError(f'translation unit has no language profile: {unit["path"]}')
        standards[unit['path']] = source_standard(unit['flags'])
    if baseline := manifest.get('baseline'):
        cwd = Path(baseline['cwd']).resolve(strict=True)
        argv = baseline['argv']
        if not argv:
            raise ValueError('baseline requires an executable')
        executable = shutil.which(argv[0]) if '/' not in argv[0] else None
        argv[0] = str(Path(executable or cwd / argv[0]).resolve())
        baseline['cwd'] = str(cwd)
    ids = [case['id'] for case in manifest.get('cases', ())]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[\w.-]+', name) or name in {'.', '..'} for name in ids):
        raise ValueError('case IDs must be unique directory names')
    out.mkdir(parents=True, exist_ok=False)
    request = out / 'request.json'
    request.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return NativeInputs(manifest, ProviderArtifact(**file_reference(request, manifest['schema'])), standards)


@dataclass(frozen=True, slots=True)
class EntryEvidence:
    """Optional inputs whose proof checks belong to whole-entry replay."""

    costs: str | Path | None = None
    forward_cpu: bool = False
    launch_probes: tuple[str | Path, ...] = ()
    team_probe: str | Path | None = None
    native_phase_cost: str | Path | None = None
    python_import_cost: str | Path | None = None
    paired_observation: str | Path | None = None
    import_bindings: str | Path | None = None
    import_sources: str | Path | None = None
    initializer_observation: str | Path | None = None
    array_route: str | Path | None = None
    array_route_cost: str | Path | None = None
    fixed_grid_cli: str | Path | None = None
    fixed_grid_mapped_input: str | Path | None = None
    fixed_grid_lifecycle_cost: str | Path | None = None
    fixed_grid_input_review: str | Path | None = None
    fixed_grid_qualification: str | Path | None = None
    fixed_grid_forward: str | Path | None = None

    @property
    def needs_paired_replay(self):
        return any(item is not None for item in (
            self.paired_observation, self.import_bindings, self.import_sources,
            self.initializer_observation, self.array_route, self.fixed_grid_cli,
        ))


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
    native_model: ProgramGraph | None = None

    @classmethod
    def read_native(cls, directory: Path) -> 'GraphBundle':
        """Open the graph-only publication and verify its role bindings."""
        directory = directory.resolve(strict=True)
        _, result = open_artifact(directory / 'result.json', 'scar.native-entry-result')
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
    from scar.analysis.replay_transaction import current_transaction

    root = Path(request['source_root'])
    transaction = current_transaction()
    for name, expected in request['source_lock']['files'].items():
        relative = Path(name)
        source = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not source.is_relative_to(root):
            raise ValueError(f'source path outside of root: {name}')
        if transaction is not None:
            try:
                transaction.watch(source, expected['sha256'], expected['size_bytes'])
            except ValueError as error:
                raise ValueError(f'locked source differs: {name}') from error
            continue
        actual = file_reference(source, 'scar.native-source')
        if any(actual[key] != expected[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source differs: {name}')


def native_bound_targets(call):
    """Return the in-graph bindings of a direct or finite indirect call."""
    if call['status'] == 'BOUND':
        return (call,)
    if call['status'] == 'FINITE_TARGETS':
        return tuple(target for target in call['targets'] if target['status'] == 'BOUND')
    return ()


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
        return cls.open(path)[1]

    @classmethod
    def open(cls, path: Path) -> tuple[dict, 'ExecutionPlanRef']:
        reference, document = open_artifact(path, 'scar.compiler-plan')
        return reference, cls._from_document(document)

    @classmethod
    def read_reference(cls, reference: dict) -> 'ExecutionPlanRef':
        return cls._from_document(read_artifact(reference))

    @classmethod
    def _from_document(cls, value: dict) -> 'ExecutionPlanRef':
        version = value.get('schema_version')
        if (value.get('schema') != 'scar.compiler-plan' or type(version) is not int or version != 1
                or set(value) != {'schema', 'schema_version', 'entry', 'artifacts', 'decision', 'scope'}):
            raise ValueError('unsupported compiler plan envelope')
        return cls(value['entry'], value['artifacts'], value['decision'], value['scope'])


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


@dataclass(frozen=True, slots=True)
class ScheduledArrayLoop:
    """A fixed-count array recurrence with its original proof and bound body plan."""

    loop_id: str
    folder: Path
    proof: dict
    strategy: dict
    schedule: dict
    hardware: dict
