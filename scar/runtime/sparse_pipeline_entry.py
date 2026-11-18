"""Execute compatible sparse stages in one otherwise original Python entry."""

from contextlib import ExitStack, contextmanager
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path
import sys

from scar.runtime.entry_runner import _Loader, _arguments, _execute, _file, _script_state


class _SourceScopes:
    def __init__(self):
        self.scopes = []
        self.frontier = []

    def attach(self, module, name, *, natural):
        for scope in self.scopes:
            scope.attach(module, name, natural=natural)


class _Finder(MetaPathFinder):
    def __init__(self, sources):
        self.sources = sources

    def find_spec(self, fullname, path=None, target=None):
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and type(spec.origin) is str:
            selected = self.sources.get(Path(spec.origin).resolve())
            if selected is not None:
                if type(spec.loader) is SourceFileLoader:
                    spec.loader = _Loader(spec.loader, selected, fullname)
                else:
                    selected.frontier.append('source_uses_nonstandard_loader')
        return spec


class _Pipeline:
    def __init__(self, row, solve):
        self.admissions = {'row': row, 'solve': solve}
        self.sources = {}
        for scope, _ in self.admissions.values():
            if scope is not None:
                selected = self.sources.get(scope.source)
                if selected is None:
                    selected = self.sources[scope.source] = _SourceScopes()
                selected.scopes.append(scope)

    def stages(self):
        result = {}
        for name, (scope, reason) in self.admissions.items():
            attachments = []
            statistics = dict(native_calls=0, fallback_calls=0, reasons={})
            stage = dict(reason=reason, attachments=attachments, statistics=statistics,
                activation=None, release=None, frontier=[reason])
            stage['topology' if name == 'row' else 'symbolic'] = None
            if scope is not None:
                stage['frontier'] = [*scope.frontier, *self.sources[scope.source].frontier]
                stage['activation'] = dict(scope.activation)
                for binding in scope.bindings:
                    _add_stats(statistics, binding['stats'])
                    attachments.append(dict(source=str(scope.source), module=binding['module_name'],
                        natural_import=binding['natural_import']))
                if name == 'row':
                    retained = scope.topology
                    live = getattr(scope.try_enforce, 'topology_stats', None)
                    stage['topology'] = retained
                else:
                    retained = scope.kernel
                    live = None if scope.candidate is None else scope.candidate.stats
                    stage['symbolic'] = retained
                    stage['activation']['available'] = scope.candidate is not None
                if retained is not None and live is not None:
                    stage['release'] = dict(retained_before_close_bytes=retained['retained_bytes'],
                        retained_after_close_bytes=live['retained_bytes'], closed=live['closed'])
            result[name] = stage
        return result


def _add_stats(total, item):
    total['native_calls'] += item['native_calls']
    total['fallback_calls'] += item['fallback_calls']
    for key, count in item['reasons'].items():
        total['reasons'][key] = total['reasons'].get(key, 0) + count


def _coordinates(reference, schema):
    if (type(reference) is not dict or set(reference) != {'uri', 'sha256', 'size_bytes', 'schema'}
            or reference['schema'] != schema or type(reference['uri']) is not str
            or type(reference['sha256']) is not str or type(reference['size_bytes']) is not int):
        raise ValueError('complete sparse stage artifact reference required')
    path = Path(reference['uri'])
    if not path.is_absolute() or path.stat().st_size != reference['size_bytes']:
        raise ValueError('sparse stage reference size differs')
    return path, reference['sha256']


def admit(entry, argv, *, row_selection=None, solve_selection=None, exclusive=False, study=False):
    """Prepare independent scopes from pinned selections without running the entry."""
    entry, argv = _arguments(entry, argv)
    row = (None, 'ROW_STAGE_NOT_REQUESTED')
    solve = (None, 'SOLVE_STAGE_NOT_REQUESTED')
    if row_selection is not None:
        from scar.runtime import csr_region_entry

        try:
            path, key = _coordinates(row_selection, csr_region_entry.SELECTION)
        except (ValueError, OSError):
            row = (None, 'CSR_SELECTION_OR_SOURCE_CHANGED')
        else:
            row = csr_region_entry.admit(entry, path, key, study=study, exclusive=exclusive, argv=argv)
    if solve_selection is not None:
        from scar.runtime import sparse_solve_entry

        try:
            path, key = _coordinates(solve_selection, sparse_solve_entry.SELECTION)
        except (ValueError, OSError):
            solve = (None, 'SPARSE_SOLVE_SELECTION_OR_SOURCE_CHANGED')
        else:
            # The launcher authorizes this receipt before passing it here.
            solve = sparse_solve_entry.admit(entry, argv, path, key, study=True, exclusive=exclusive)
    return _Pipeline(row, solve)


@contextmanager
def _install(group):
    if not group.sources:
        yield
        return
    original = sys.meta_path
    entries = list(original)
    try:
        with ExitStack() as cleanup:
            for selected in group.sources.values():
                for scope in selected.scopes:
                    cleanup.callback(scope.finish)
            if sum(finder is PathFinder for finder in entries) == 1:
                proxy = _Finder(group.sources)
                sys.meta_path = [proxy if finder is PathFinder else finder for finder in entries]
                for name, module in tuple(sys.modules.items()):
                    selected = group.sources.get(_file(module))
                    if selected is not None:
                        selected.attach(module, name, natural=False)
            else:
                for selected in group.sources.values():
                    selected.frontier.append('standard_pathfinder_position_unavailable')
            yield
    finally:
        original[:] = entries
        sys.meta_path = original


@contextmanager
def script_state(entry, argv, group):
    """Install the same import scopes for a scoped diagnostic driver."""
    entry, argv = _arguments(entry, argv)
    with _script_state(entry, argv, None), _install(group):
        yield


def run(entry, argv, *, row_selection=None, solve_selection=None, exclusive=False, study=False):
    entry, argv = _arguments(entry, argv)
    group = admit(entry, argv, row_selection=row_selection, solve_selection=solve_selection,
        exclusive=exclusive, study=study)
    with _install(group):
        reason = 'SCOPED_SPARSE_PIPELINE' if group.sources else 'ORIGINAL_EXECUTION_CONDITIONS'
        report = _execute(entry, argv, None, reason)
    stages = group.stages()
    row = stages['row']['statistics']['native_calls'] > 0
    solve = stages['solve']['statistics']['native_calls'] > 0
    decisions = {(False, False): 'KEEP', (True, False): 'ROW_ONLY',
        (False, True): 'SOLVE_ONLY', (True, True): 'ROW_AND_SOLVE'}
    for stage in stages.values():
        _add_stats(report['statistics'], stage['statistics'])
        report['attachments'].extend(stage['attachments'])
    report.update(schema='scar.sparse-pipeline-execution', decision=decisions[row, solve], stages=stages,
        installed_bindings=len(report['attachments']),
        observed_boundary_calls=report['statistics']['native_calls'] + report['statistics']['fallback_calls'],
        frontier=[item for stage in stages.values() for item in stage['frontier']])
    return report
