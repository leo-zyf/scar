"""Owner-local cold admission materials; never a serialized proof token.

Each independent controller replay opens a new context. Only this context's
factories populate its registries; consumers cannot assert prior validation.
"""
from copy import deepcopy
from collections.abc import Mapping
import os
from pathlib import Path
import shlex
import shutil
import sys
from threading import get_ident
import weakref

from scar.analysis.replay_transaction import _file_version
from scar.ir.execution_recipe import _reference, artifact_reference, read_artifact


_ROOT = Path(__file__).resolve().parents[1]


def _environment():
    import numpy as np
    from numpy._core import _multiarray_umath
    try:
        command=tuple(shlex.split(os.environ.get('CXX','g++')))
    except ValueError:
        command=()
    executable=shutil.which(command[0]) if command else None
    resolved=str(Path(executable).resolve()) if executable is not None else None
    return (dict(os.environ), str(Path.cwd().resolve()), tuple(sorted(os.sched_getaffinity(0))),
        sys.version, np.__version__, id(np.power), id(np.ufunc), id(_multiarray_umath.power),
        int(_multiarray_umath._get_ndarray_c_version()), (command,executable,resolved))


def _reference_key(reference):
    return tuple(reference[name] for name in ('uri', 'sha256', 'size_bytes', 'schema'))


def require_admission_session(session, program):
    """Accept only the exact live session owned by its original context."""
    from scar.analysis.build import BuildSession
    owner = getattr(session, '_scar_admission_context_owner', None)
    context = owner() if type(owner) is weakref.ReferenceType else None
    if type(session) is not BuildSession or type(context) is not AdmissionContext:
        raise ValueError('borrowed session needs its live admission context owner')
    context._live()
    reference = _reference(program)
    if context._sessions.get(_reference_key(reference), (None,))[0] is not session:
        raise ValueError('borrowed session belongs to another complete ProgramBuild reference')
    context._guard_session(session, reference)
    return session


class AdmissionContext:
    def __init__(self):
        self.owner = (os.getpid(), get_ident())
        self.closed = False
        self._entered = False
        self._sessions = {}
        self._stateful = {}
        self._watched = {}
        self._environment = None
        self._transaction = None
        self._fact_resolvers = {}
        self._fact_routes = {}

    def __reduce__(self):
        raise TypeError('admission contexts cannot be serialized')

    def __deepcopy__(self, memo):
        raise TypeError('admission contexts cannot be copied')

    def _live(self):
        from scar.analysis.replay_transaction import current_transaction
        if self.owner != (os.getpid(), get_ident()) or self.closed or not self._entered:
            raise ValueError('live owner-thread admission context required')
        if current_transaction() is not self._transaction:
            raise ValueError('admission context belongs to another metadata replay scope')
        if _environment() != self._environment:
            raise ValueError('admission environment, caller affinity or NumPy identity changed')

    def watch(self, reference):
        self._live()
        if _reference(reference)['schema']=='scar.source-facts-unit':
            raise ValueError('logical source fact artifacts require their original program/catalog route')
        self._watch_current(reference)

    def _watch_current(self, reference):
        reference = _reference(reference)
        version = _file_version(reference['uri'])
        if version[1:3] != (reference['sha256'], reference['size_bytes']):
            raise ValueError('admission dependency byte identity differs: '+reference['uri'])
        previous = self._watched.get(version[0])
        if previous is not None and previous != version:
            raise ValueError('admission dependency version changed: '+version[0])
        self._watched[version[0]] = version

    def _watch_tree(self, value, *, _program=None):
        self._live()
        pending, references, facts = [value], {}, {}
        while pending:
            row = pending.pop()
            if isinstance(row,Mapping):
                if (set(row)=={'artifact','locator'} and isinstance(row['artifact'],Mapping)
                        and row['artifact'].get('schema')=='scar.source-facts-unit'):
                    ref=_reference(dict(row['artifact']));key=_reference_key(ref)
                    facts.setdefault(key,(ref,set()))[1].add(row['locator'])
                    continue
                if {'uri', 'sha256', 'size_bytes', 'schema'} <= row.keys():
                    ref = {name:row[name] for name in ('uri', 'sha256', 'size_bytes', 'schema')}
                    if ref['schema']=='scar.source-facts-unit':
                        ref=_reference(ref);facts.setdefault(_reference_key(ref),(ref,set()))
                    else:references[_reference_key(ref)] = ref
                elif {'path', 'sha256'} <= row.keys() and type(row['path']) is str:
                    path = Path(row['path'])
                    if path.is_absolute():
                        size = row.get('bytes', row.get('size_bytes'))
                        if type(size) is not int:
                            size = path.stat().st_size
                        ref = dict(uri=str(path), sha256=row['sha256'], size_bytes=size, schema='scar.admission-input')
                        references[_reference_key(ref)] = ref
                pending.extend(row.values())
            elif type(row) in (tuple, list):
                pending.extend(row)
        for reference in references.values():
            self._watch_current(reference)
        if facts:self._watch_fact_materials(facts,_program)

    def _watch_fact_materials(self, facts, program):
        from scar.analysis.bounded_entry_calls import _resolve_fact_artifact
        from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
        if program is None:
            raise ValueError('source fact material needs an explicit original ProgramBuild owner')
        program=_reference(program);owner_key=_reference_key(program)
        if owner_key not in self._sessions:
            raise ValueError('source fact material program is not owned by this admission context')
        session=self._sessions[owner_key][0];catalog=session._f4.catalog
        previous=self._fact_resolvers.get(owner_key)
        if previous is not None and previous.catalog is not catalog:
            raise ValueError('source fact material catalog owner changed')
        resolver=previous
        for key,(reference,locators) in facts.items():
            artifact=ProviderArtifact(**reference);cache_key=(owner_key,key)
            saved=self._fact_routes.get(cache_key)
            if saved is None or not set(locators)<=saved[1]:
                wanted=set(locators)|(set(saved[1]) if saved is not None else set())
                resolver,route=_resolve_fact_artifact(catalog,artifact,resolver=resolver,locators=sorted(wanted))
                if saved is not None and saved[0]!=route:
                    raise ValueError('source fact material route changed')
                self._fact_routes[cache_key]=(route,frozenset(wanted))
                self._fact_resolvers[owner_key]=resolver
            else:
                route=resolver.route(ProviderFactReference(artifact,'#/unit'))
                if route!=saved[0]:raise ValueError('source fact material route changed')
            resolver.validate_catalog()
            for physical in (route.physical,*route.catalogs):
                self._watch_current(dict(uri=physical.uri,sha256=physical.sha256,
                    size_bytes=physical.size_bytes,schema=physical.schema))

    def _producers(self):
        self._live()
        paths = {_ROOT/'compiler'/name for name in ('admission_context.py', 'controlled_resources.py',
            'source_summary_binding.py', 'stateful_candidate.py', 'selected_stateful_lowering.py')}
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, '__file__', None) if name.startswith('scar.') else None
            if filename:
                path = Path(filename).resolve()
                if path.suffix == '.py' and path.is_relative_to(_ROOT):
                    paths.add(path)
        for path in paths:
            previous=self._watched.get(str(path))
            reference=(artifact_reference(path,'scar.python-producer')if previous is None else
                dict(uri=previous[0],sha256=previous[1],size_bytes=previous[2],schema='scar.python-producer'))
            self._watch_current(reference)

    def _guard_session(self, session, reference):
        from scar.analysis.build_f4 import _validate_catalog_parents
        from scar.analysis.build import _rules
        owner = session._f4
        if owner is None or owner.closed or artifact_reference(owner.path/'manifest.json', reference['schema']) != reference:
            raise ValueError('admission ProgramBuild session is closed or changed')
        if owner.report != read_artifact(reference) or owner.reader_rules != _rules():
            raise ValueError('admission ProgramBuild report or reader rules changed')
        self.watch(reference)
        _validate_catalog_parents(owner.catalog)
        self._watch_tree(owner._admitted,_program=reference)
        for source in tuple(owner.view.units.values()):
            source.check()
        for name, provider in tuple(owner.providers.items()):
            if name.startswith('bounded_call_bindings:'):
                provider.check()

    def program(self, reference):
        from scar.analysis.build import BuildSession
        self.validate()
        reference = _reference(reference)
        if reference['schema'] != 'scar.program-build':
            raise ValueError('original ProgramBuild reference required')
        self.watch(reference)
        key = _reference_key(reference)
        if key not in self._sessions:
            session = BuildSession(Path(reference['uri']).parent)
            self._sessions[key] = (session, deepcopy(reference))
            session._scar_admission_context_owner = weakref.ref(self)
        session = self._sessions[key][0]
        self._guard_session(session, reference)
        self._producers()
        return session

    def stateful(self, reference):
        self.validate()
        reference = _reference(reference)
        if reference['schema'] != 'scar.observed-stateful-specialization':
            raise ValueError('original stateful specialization reference required')
        self.watch(reference)
        key = _reference_key(reference)
        if key not in self._stateful:
            value = self._derive_stateful(reference)
            self._watch_tree(value,_program=value['specialization']['program'])
            self._producers()
            self.validate()
            self._stateful[key] = deepcopy(value)
        return deepcopy(self._stateful[key])

    def _derive_stateful(self, reference):
        from scar.codegen import stateful, cpp
        from scar.compiler.selected_stateful_lowering import derive_stateful_lowering_view
        from scar.compiler.stateful_candidate import check_observed_specialization
        from scar.compiler.stateful_model import document
        specialized = read_artifact(reference)
        session = self.program(specialized['program'])
        admission = check_observed_specialization(reference, _session=session)
        if admission['native_metadata_verified'] is not True:
            raise ValueError('compiled source-admitted stateful candidate required')
        candidate = read_artifact(specialized['candidate'])
        domain = read_artifact(specialized['domain'])
        native = read_artifact(candidate['native_artifact'])
        directory = Path(candidate['native_artifact']['uri']).parent
        region, plan, placement = stateful._admit(directory/native['files']['region'], directory/native['files']['plan'])
        view = derive_stateful_lowering_view(candidate['native_artifact'])
        if (plan['cpu_threads'] != 1 or native['cpu_threads'] != 1
                or native['implementation_sha256'] != stateful.implementation_sha256()
                or native['compiler']['options'] != list(cpp.COMPILER_OPTIONS)):
            raise ValueError('stateful implementation or original compiler profile differs')
        provider = native['provider_runtime']['loaded_binary']
        provider_ref = artifact_reference(provider['path'], 'scar.numpy-native-provider-binary')
        if (provider_ref['sha256'], provider_ref['size_bytes']) != (provider['sha256'], provider['bytes']):
            raise ValueError('original provider binary stamp differs')
        members = {name:dict(ref, schema='scar.native-member') for name, ref in dict(
            region=view['region'], plan=view['plan'], source=view['source'], library=view['library'], **view['headers']).items()}
        return dict(admission=admission, specialization=specialized, candidate=candidate, domain=domain,
            region=document(region), plan=plan, placement=plan['resource_binding'], native=native,
            view=view, members=members, provider_binary=provider_ref)

    def validate(self):
        self._live()
        for version in tuple(self._watched.values()):
            self._watch_current(dict(uri=version[0], sha256=version[1], size_bytes=version[2], schema='scar.admission-input'))
        for session, reference in tuple(self._sessions.values()):
            self._guard_session(session, reference)
        for key,resolver in self._fact_resolvers.items():
            if self._sessions[key][0]._f4.catalog is not resolver.catalog:
                raise ValueError('source fact material catalog owner changed')
            resolver.validate_catalog()

    def close(self):
        if self.owner != (os.getpid(), get_ident()):
            raise ValueError('only the admission owner can close its context')
        if self.closed:
            return
        self.closed = True
        try:
            for session, _ in self._sessions.values():
                session.close()
        finally:
            self._sessions.clear(); self._stateful.clear(); self._watched.clear()
            self._fact_resolvers.clear();self._fact_routes.clear()

    def __enter__(self):
        from scar.analysis.replay_transaction import current_transaction
        if self._entered or self.closed or self.owner != (os.getpid(), get_ident()):
            raise ValueError('admission context cannot be reopened or transferred')
        self._environment = _environment(); self._transaction = current_transaction(); self._entered = True
        executable=self._environment[-1][-1]
        if executable is not None:
            version=_file_version(executable)
            self._watch_current(dict(uri=version[0],sha256=version[1],size_bytes=version[2],schema='scar.compiler-executable'))
        self._producers()
        return self

    def __exit__(self, *exception):
        try:
            self.validate()
        finally:
            self.close()


__all__ = ['AdmissionContext']
