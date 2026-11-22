"""Replay inherited compiler contexts through capture and inventory."""
from collections import defaultdict
from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest

from scar.analysis.call_demand import inventory_from_calls
from scar.analysis.call_seeds import CallSourceSeedReport, read_call_source_seeds
from scar.analysis.program_inventory import ProgramInventory
from scar.ir.program.call_edges import CallEdgeBudget
from scar.ir.program.function_origins import CallCompilationProfile
from scar.ir.program.expansion import ContextStoreBudget
from scar.ir.program.inventory import InventoryBudget
from tests.fixtures.capture_cli import run_trace


LEGACY = CallCompilationProfile.LEGACY_ZERO_FLAGS
RECORDED = CallCompilationProfile.RECORDED_FUTURE_FLAGS


@dataclass(frozen=True)
class Capture:
    project: Path
    entry: Path
    trace: Path
    marker: Path
    sources: dict


def _capture(directory, *, mixed=False, nested=False):
    project = directory / 'project'
    project.mkdir(parents=True)
    helper = project / 'helper.py'
    helper.write_text('def identity(value):\n    return value\n'
                      'def relay(value):\n    return identity(value)\n' +
                      ('def make():\n    def nested(value):\n        return value\n    return nested\n' if nested else ''))
    marker = project / 'executed'
    entry = project / 'entry.py'
    entry.write_text('from helper import relay' + (', make' if nested else '') + '\n'
                     'def run(value):\n    return relay(value)\n'
                     'value = object()\nassert run(value) is value\n' +
                     ('assert make()(value) is value\n' if nested else '') +
                     f'open({str(marker)!r}, "w").write("ran")\n')
    driver = directory / 'driver.py'
    # No flags are patched into captured descriptors. compile() is actually
    # called inside an executing parent carrying future annotations, then the
    # resulting module/function code is automatically recorded by public CLI.
    driver.write_text('from __future__ import annotations\n'
                      'import sys\nfrom pathlib import Path\nfrom types import ModuleType\n'
                      f'project = Path({str(project)!r})\n'
                      f'for inherited in {([False, True] if mixed else [True])!r}:\n'
                      '    module = ModuleType("helper")\n'
                      '    source = project / "helper.py"\n'
                      '    code = (compile(source.read_text(), str(source), "exec") if inherited else\n'
                      '            compile(source.read_text(), str(source), "exec", flags=0, dont_inherit=True))\n'
                      '    exec(code, module.__dict__)\n'
                      '    sys.modules["helper"] = module\n'
                      '    source = project / "entry.py"\n'
                      '    code = (compile(source.read_text(), str(source), "exec") if inherited else\n'
                      '            compile(source.read_text(), str(source), "exec", flags=0, dont_inherit=True))\n'
                      '    exec(code, {})\n')
    trace = directory / 'trace'
    process = run_trace(driver, trace, '--calls-only', project_root=project)
    assert process.returncode == 0, process.stdout + process.stderr
    assert marker.read_text() == 'ran'
    marker.unlink()
    sources = {path: path.read_bytes() for path in project.iterdir() if path.is_file()}
    return Capture(project, entry, trace, marker, sources)


@pytest.fixture(scope='module')
def inherited(tmp_path_factory):
    return _capture(tmp_path_factory.mktemp('inherited-compilation'))


@pytest.fixture(scope='module')
def mixed(tmp_path_factory):
    return _capture(tmp_path_factory.mktemp('mixed-compilation'), mixed=True, nested=True)


def _assert_not_executed(capture):
    assert not capture.marker.exists()
    assert all(path.read_bytes() == content for path, content in capture.sources.items())


def _entered(capture, document):
    groups = {group['code'] for group in document['model']['groups']}
    return {row['code']['key']: row['code'] for row in document['model']['codes']
            if row['code']['key'] in groups
            and Path(row['code']['filename']).is_relative_to(capture.project)}


def _assert_source_links(document, codes):
    links = {link['code']: link for link in document['model']['code_links']}
    assert codes and all(links[key]['status'] == 'VERIFIED' for key in codes)
    for key, code in codes.items():
        assert links[key]['snapshot']['path'] == code['filename']
        assert links[key]['optimize_levels']


def _assert_core_admissions(document, expected_names):
    names = {definition['id']['wire']: definition['name']
             for file in document['files'] for definition in file['definitions']}
    roots = {body['definition']['wire']: body['graph']['root']
             for file in document['files'] for body in file['bodies'] if body['graph'] is not None}
    admissions = [admission for admission in document['call_source_scope']['admissions']
                  if admission['definition'] is not None
                  and names[admission['definition']['wire']] in expected_names]
    assert {names[item['definition']['wire']] for item in admissions} == set(expected_names)
    assert all(item['status'] == 'CORE_LINKED'
               and item['core_region'] == roots[item['definition']['wire']] for item in admissions)
    assert document['transformation_authorized'] is False
    assert document['summary']['whole_program_graph_established'] is False



def _seed(capture, profile=RECORDED):
    return read_call_source_seeds(capture.trace, capture.project, compilation_profile=profile).to_dict()


def _inventory(capture, profile=RECORDED):
    return inventory_from_calls(
        capture.entry, project_root=capture.project, source_roots=(), budget=InventoryBudget(),
        call_trace=capture.trace, call_budget=CallEdgeBudget(), context_budget=ContextStoreBudget(),
        compilation_profile=profile).to_dict()


@pytest.mark.parametrize('profile', [LEGACY, RECORDED])
def test_inherited_capture_and_nested_inventory_replay(inherited, profile):
    seed = _seed(inherited, profile)
    codes = _entered(inherited, seed)
    assert {'<module>', 'run', 'relay', 'identity'} <= {code['qualname'] for code in codes.values()}
    assert all(code['flags'] & 0x1000000 for code in codes.values())
    document = _inventory(inherited, profile)
    assert document['schema_version'] == 6 and document['call_sources'] == seed
    result = ProgramInventory.from_dict(document)
    if profile is RECORDED:
        assert seed['compilation_profile'] == RECORDED.value
        _assert_source_links(seed, codes)
        _assert_core_admissions(document, {'run', 'relay', 'identity'})
    else:
        assert seed['schema_version'] == 1 and 'compilation_profile' not in seed
        assert result.call_sources.eligible_groups() == () and result.call_source_scope.admissions == ()
        links = {link.code: link for link in result.call_sources.model.code_links}
        assert all(links[key].status.value == 'MISMATCH' for key in codes)
    assert CallSourceSeedReport.from_dict(seed).to_dict() == seed
    assert result.to_dict() == document
    _assert_not_executed(inherited)


def test_same_source_compilation_contexts_keep_separate_admission(mixed):
    document = _seed(mixed)
    codes = _entered(mixed, document)
    by_source = defaultdict(set)
    for code in codes.values():
        by_source[code['filename'], code['qualname']].add(bool(code['flags'] & 0x1000000))
    assert all(contexts == {False, True} for contexts in by_source.values())
    assert any(name == 'make.<locals>.nested' for _, name in by_source)
    _assert_source_links(document, codes)
    inventory = _inventory(mixed)
    assert inventory['call_sources'] == document
    _assert_core_admissions(inventory, {'run', 'relay', 'identity'})
    definitions = [task['definition']['wire'] for task in inventory['expansion']['entries']]
    assert len(definitions) == len(set(definitions))
    assert ProgramInventory.from_dict(inventory).to_dict() == inventory
    _assert_not_executed(mixed)


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('profile', [LEGACY, RECORDED])
def test_profile_and_budget_precede_raw_replay(inherited, nested, profile):
    document = _inventory(inherited, profile) if nested else _seed(inherited, profile)
    read = ProgramInventory.from_dict if nested else CallSourceSeedReport.from_dict
    budget = replace(CallEdgeBudget(), max_events=1)
    budgets = {'call_verification_budget' if nested else 'verification_budget': budget}
    moved = inherited.trace.with_name('trace-temporarily-unavailable')
    inherited.trace.rename(moved)
    try:
        with pytest.raises(ValueError, match='independent verifier budget'):
            read(document, **budgets)
        seed = document['call_sources'] if nested else document
        seed['compilation_profile'] = 'INVENTED_COMPILER_CONTEXT'
        with pytest.raises(ValueError, match='profile|fields|schema|version'):
            read(document)
    finally:
        moved.rename(inherited.trace)


def test_flags_tampering_cannot_match_by_source_name_or_line(inherited, tmp_path):
    trace = tmp_path / 'changed-calls'
    trace.mkdir()
    for path in inherited.trace.iterdir():
        (trace / path.name).write_bytes(path.read_bytes())
    codes_path, = trace.glob('call-codes.*.jsonl')
    rows = [json.loads(line) for line in codes_path.read_text().splitlines()]
    changed = next(row for row in rows if row['qualname'] == 'identity'
                   and row['filename'] == str(inherited.project / 'helper.py'))
    original_key = changed['key']
    changed['flags'] &= ~0x1000000
    codes_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    metadata_path, = trace.glob('call-meta.*.json')
    metadata = json.loads(metadata_path.read_text())
    metadata['bytes'] = sum(path.stat().st_size for family in ('codes', 'events')
                            for path in trace.glob(f'call-{family}.*.jsonl'))
    metadata_path.write_text(json.dumps(metadata))
    document = read_call_source_seeds(trace, inherited.project).to_dict()
    link, = (link for link in document['model']['code_links'] if link['code'] == original_key)
    assert link['status'] == 'MISMATCH' and link['definition'] is None and link['source'] is None
    assert any(issue['kind'] == 'CODE_MISMATCH' for issue in link['issues'])
    assert CallSourceSeedReport.from_dict(document).to_dict() == document
    _assert_not_executed(inherited)
