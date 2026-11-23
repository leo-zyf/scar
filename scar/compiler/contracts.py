"""Products exchanged between analysis, planning and C++20 delivery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING

from scar.artifacts import _durable_bytes, file_reference
from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact
from scar.ir.program.providers import ProviderArtifact

if TYPE_CHECKING:
    from scar.ir.program.model import ProgramGraph


@dataclass(frozen=True, slots=True)
class NativeInputs:
    manifest: dict
    request: ProviderArtifact
    standards: dict[str, str | None]


def read_native_metadata(reference):
    """Share decoded native metadata for one content-guarded handoff."""
    from scar.analysis.replay_transaction import detached_replay

    receipt = asdict(reference) if isinstance(reference, ProviderArtifact) else reference
    identity = tuple(sorted(receipt.items()))
    return detached_replay('scar.native-input-metadata.v1', identity, lambda: read_artifact(reference))


def publish_plan(path, document):
    """Persist the versioned plan used by downstream replay."""
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _durable_bytes(path, encoded, replace=False)
    return artifact_reference(path, document['schema'])


def cpp20_capability(profile: str, *, operator_abi_version: int = 1) -> dict:
    """Describe the target ABI used by an admitted delivery profile."""
    profiles = ('CPP20_NATIVE_ENTRY', 'CPP_SERIAL_TEMPORAL', 'CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL')
    if profile not in profiles or type(operator_abi_version) is not int or operator_abi_version != 1:
        raise ValueError('unqualified C++20 operator ABI or delivery profile')
    report = dict(schema='scar.cpp20-backend-capability', schema_version=1, profile=profile,
                  cxx_standard=20, operator_abi_version=operator_abi_version)
    if profile == 'CPP20_NATIVE_ENTRY':
        report['target_kind'] = 'EXECUTABLE'
    else:
        report.update(target_kind='SHARED_OPERATOR', temporal_control_abi_version=1)
    return report


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
    model: ProgramGraph | None = None
    call_document: dict | None = None
    publication: ProviderArtifact | None = None
    entry_regions: tuple[str, ...] = ()

    @classmethod
    def publish(cls, directory, *, program, hardware, request, evidence, calls, entry_regions,
                model=None, report=None, call_document=None):
        """Publish the shared frontend handoff after typed graph construction."""
        roles = dict(program=program, hardware=hardware, request=request, graph_evidence=evidence, calls=calls)
        document = dict(schema='scar.graph-bundle', schema_version=2,
                        artifacts={k: asdict(v) if isinstance(v, ProviderArtifact) else v for k, v in roles.items()},
                        entry_regions=list(entry_regions))
        graph = cls._admit(document, model=model, report=report, call_document=call_document)
        path = Path(directory).resolve(strict=True) / 'graph-report.json'
        raw = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        _durable_bytes(path, raw, replace=False)
        reference = ProviderArtifact(str(path), sha256(raw).hexdigest(), len(raw), 'scar.graph-bundle')
        return replace(graph, publication=reference)

    @classmethod
    def open(cls, path):
        path = Path(path).resolve(strict=True)
        if path.is_dir():
            path /= 'graph-report.json'
        reference, document = open_artifact(path, 'scar.graph-bundle')
        return cls._admit(document, publication=ProviderArtifact(**reference))

    @staticmethod
    def _roles(document):
        if (document.get('schema'), document.get('schema_version')) != ('scar.graph-bundle', 2):
            raise ValueError('expected version 2 compiler graph')
        roles = document.get('artifacts')
        if (set(document) != {'schema', 'schema_version', 'artifacts', 'entry_regions'}
                or type(roles) is not dict or set(roles) != {'request', 'program', 'hardware', 'graph_evidence', 'calls'}
                or type(document['entry_regions']) is not list
                or not all(type(entry) is str and entry for entry in document['entry_regions'])):
            raise ValueError('compiler graph requires typed artifacts and entry regions')
        parts = {name: ProviderArtifact(**reference) for name, reference in roles.items()}
        if (parts['program'].schema, parts['hardware'].schema) != (
                'scar.program-graph', 'scar.target-hardware-graph'):
            raise ValueError('typed program and target topology required')
        if parts['request'].schema == 'scar.native-entry' and (
                parts['graph_evidence'].schema, parts['calls'].schema) != (
                'scar.native-graph-evidence', 'scar.native-call-links'):
            raise ValueError('native graph requires typed call and frontend evidence')
        return parts

    @classmethod
    def _admit(cls, document, *, model=None, report=None, call_document=None, publication=None):
        from scar.ir.program.codec import loads

        parts = cls._roles(document)
        if model is None:
            model = loads(read_artifact(parts['program'], json_document=False).decode())
        entries = tuple(document['entry_regions'])
        if not entries or len(set(entries)) != len(entries) or not set(entries) <= {r.id.wire for r in model.regions}:
            raise ValueError('graph entries must name distinct retained regions')
        if report is None:
            report = read_native_metadata(parts['graph_evidence'])
        if call_document is None:
            call_document = read_native_metadata(parts['calls'])
        if parts['request'].schema == 'scar.native-entry':
            for role in ('request', 'program', 'hardware', 'calls'):
                if report.get(role) != document['artifacts'][role]:
                    raise ValueError(f'native graph evidence differs on {role}')
            if entries != (report.get('entry'),):
                raise ValueError('native graph evidence and entry binding disagree')
        return cls(parts['request'], parts['program'], parts['hardware'], parts['graph_evidence'], parts['calls'],
                   report, model=model, call_document=call_document, publication=publication,
                   entry_regions=entries)

    def references(self) -> dict:
        return {role: asdict(reference) for role, reference in (
            ('request', self.request), ('program', self.program), ('hardware', self.hardware),
            ('graph_evidence', self.evidence), ('calls', self.calls),
        ) if reference is not None}


def python_build_receipts(session):
    """Read the two parent receipts of the admitted Python source session."""
    reference = ProviderArtifact(**session.publication)
    report = session.report
    if (reference.schema != 'scar.program-build' or report.get('schema_version') != 1
            or session._f4 is None):
        raise ValueError('original F4 program build required')
    if not report.get('workflow_complete'):
        raise ValueError('F4 build did not finish graph admission')
    hardware = report.get('target')
    if hardware is None:
        raise ValueError('program build has no target hardware snapshot')
    return reference, ProviderArtifact(uri=hardware['path'], size_bytes=hardware['bytes'],
                                        sha256=hardware['sha256'], schema='scar.target-hardware-graph')


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


def entry_sources(artifacts):
    roles = ('request', 'program', 'graph_evidence', 'calls', 'simplified', 'rewrites', 'cost')
    return tuple(dict(role=role, **artifacts[role]) for role in roles)


def entry_cpus(hardware):
    from scar.analysis.replay_transaction import current_transaction
    from scar.analysis.target_graph import TargetHardwareGraph
    from scar.ir.program.target_hardware import CpuExecutor

    target = ProviderArtifact(**hardware)

    def admitted_cpus():
        graph = TargetHardwareGraph(Path(target.uri).parent, expected=target)
        cpu_ids = {cpu for executor in graph.resources() if isinstance(executor, CpuExecutor)
                   for cpu in executor.cpu_ids}
        cpu_ids.intersection_update(graph.inventory['cpu']['online_cpu_ids'] or ())
        if graph.report['schema_version'] == 1:
            allowed = graph.inventory.get('execution_constraints', {}).get('allowed_cpu_ids')
            if allowed is not None:
                cpu_ids.intersection_update(allowed)
        for constraints in graph.constraints():
            if constraints.allowed_cpu_ids is not None:
                cpu_ids.intersection_update(constraints.allowed_cpu_ids)
        return frozenset(cpu_ids)

    transaction = current_transaction()
    retained = (admitted_cpus() if transaction is None
                else transaction.memo('scar.native-entry.cpu-admission/v1', target, admitted_cpus))
    return os.sched_getaffinity(0).intersection(retained)


def entry_baseline_only(problem):
    return 'PHYSICAL_COST_BASELINE_ONLY' in problem.assumptions


def native_delivery(cpu, variant):
    binding = dict(profile='CPP20_NATIVE_ENTRY', version=1, device='CPU', cpu_ids=[cpu])
    if variant is not None:
        binding['trial_variant'] = variant
    return binding


def verify_native_binding(reference, problem, plan, verification):
    """Validate the selected entry against its original hardware and cost evidence."""
    artifacts = reference.artifacts
    if artifacts['cost']['schema'] != 'scar.native-entry-cost':
        raise ValueError('native delivery requires a native cost artifact')
    observation = read_artifact(artifacts['cost'])
    if (problem.target != ProviderArtifact(**artifacts['hardware'])
            or problem.source_refs != entry_sources(artifacts)
            or observation['request'] != artifacts['request'] or observation['exit'] != 0):
        raise ValueError('native planning problem does not bind the admitted inputs')
    if (len(problem.operations) != 1 or problem.operations[0].id != 'entry'
            or problem.data_edges or problem.activities or problem.timing_constraints
            or problem.reserved_memory):
        raise ValueError('native C++20 entry lowering requires one admitted execution region')

    selected = plan['selected'][0]
    mode = next(option for option in problem.operations[0].options if option.id == selected['option'])
    cpu = observation['cpu']
    if (type(cpu) is not int or cpu not in entry_cpus(artifacts['hardware'])
            or mode.parameters.get('region') != reference.entry
            or mode.demands != {f'cpu:{cpu}': 1}
            or mode.workspace_bytes or selected['start_ns'] != 0
            or mode.proof_refs != problem.source_refs):
        raise ValueError('selected native implementation has no admitted CPU binding')

    variant = observation.get('variants', {}).get(mode.id)
    if mode.parameters.get('delivery') != native_delivery(cpu, variant):
        raise ValueError('selected native implementation has no admitted C++20 delivery binding')
    version = observation['schema_version']
    rewrites = mode.parameters.get('rewrites', [])
    if version == 1 and (variant is not None or not plan['baseline_only'] or mode.id != 'original'):
        raise ValueError('a baseline observation can only deliver the original implementation')
    if version not in (1, 2) or (version == 2 and variant is None):
        raise ValueError('selected native implementation has no calibrated project')
    if mode.id == 'canonical':
        if mode.lowering != 'canonicalize_scalar_copies' or not rewrites or observation.get('calibration_error'):
            raise ValueError('canonical delivery requires a qualified source revision')
    elif mode.id != 'original' or mode.lowering != 'preserve_entry' or rewrites:
        raise ValueError('unsupported native delivery implementation')

    binding = dict(cpus=[cpu], lowering=mode.lowering, selected=plan['selected'],
                   rewrite_operations=rewrites, verification=verification)
    if 'entry_effect_scope' in mode.parameters:
        scope = mode.parameters['entry_effect_scope']
        if scope not in ('CLOSED_BODY', 'OPEN_BODY'):
            raise ValueError('unsupported native entry effect scope')
        binding['entry_effect_scope'] = scope
    if variant is not None:
        binding['trial_variant'] = variant
    return observation, binding


def native_compiler_types(manifest, evidence):
    """Retain the exact adapter type receipts required by C++20 translation."""
    typed_units = {unit['path'] for unit in manifest['units']
                   if unit.get('language', manifest['language']) in ('c', 'fortran')}
    needs_fortran_abi = any(unit.get('language', manifest['language']) == 'fortran'
                            for unit in manifest['units'])
    retained_units = typed_units | {unit['path'] for unit in manifest['units']
                                   if needs_fortran_abi and unit.get('language', manifest['language']) == 'c++'}
    compiler_types = {row['unit']: row['artifacts']['types'] for row in evidence.get('units', ())
                      if row['unit'] in retained_units and 'types' in row['artifacts']}
    if not typed_units <= compiler_types.keys():
        raise ValueError('C and Fortran translation require original type evidence for every unit')
    return compiler_types


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
    graph: dict | None = None
    lowering: dict | None = None

    def __post_init__(self):
        if not self.entry or not self.scope:
            raise ValueError('execution plan requires an entry and a qualified scope')
        self.problem
        self.schedule
        if (self.graph is None) != (self.lowering is None):
            raise ValueError('versioned delivery requires both the graph and lowering receipts')
        if self.graph is not None:
            if ProviderArtifact(**self.graph).schema != 'scar.graph-bundle':
                raise ValueError('compiler plan requires the canonical graph')
            ProviderArtifact(**self.lowering)

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
        fields = asdict(self)
        if self.graph is None:
            del fields['graph'], fields['lowering']
        return dict(schema='scar.compiler-plan', schema_version=1 if self.graph is None else 2, **fields)

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
        fields = {'entry', 'artifacts', 'decision', 'scope'}
        if version == 2:
            fields.update(('graph', 'lowering'))
        if (value.get('schema') != 'scar.compiler-plan' or type(version) is not int or version not in (1, 2)
                or set(value) != {'schema', 'schema_version', *fields}
                or (version == 2 and (value['graph'] is None or value['lowering'] is None))):
            raise ValueError('unsupported compiler plan envelope')
        return cls(**{name: value[name] for name in fields})

    def read_graph_binding(self) -> dict:
        """Verify the plan's graph identity without reopening source analysis."""
        from scar.analysis.replay_transaction import pinned_file_reference

        if self.graph is None:
            raise ValueError('canonical graph binding requires a version 2 plan')
        document = read_native_metadata(self.graph)
        GraphBundle._roles(document)
        if self.entry not in document['entry_regions']:
            raise ValueError('selected plan entry is absent from the original program graph')
        for role, reference in document['artifacts'].items():
            if self.artifacts.get(role) != reference:
                raise ValueError(f'compiler plan differs from graph artifact {role}')
            if pinned_file_reference(reference['uri'], reference['schema']) != reference:
                raise ValueError(f'graph artifact changed before delivery: {role}')
        return document


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
