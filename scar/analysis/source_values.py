"""Source values."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from scar.analysis.constants_v2 import (
    ConstantStatus,
    BuiltinRuntime,
    ConstantRuntimeRequirement,
    EvaluationBudget,
    EvaluationUsage,
    RULE_VERSION,
    _BINARY,
    _BudgetExceeded,
    _Unsupported,
    _builtin,
    _literal,
    _literal_size,
    _python_value,
    _reserve,
)
from scar.ir.literals import (
    PythonLiteral,
    _preflight_literal,
    literal_builtin_type,
    LiteralKind,
    require_legacy_literal,
)
from scar.ir.program import normal_graph as F, normal_graph as T, literal_validation as V, source_bytes as X, normal_graph as G, source_productions as P
import argparse
import json
import pathlib
import posixpath
import sys
from hashlib import sha256
from io import BytesIO, TextIOWrapper
from pathlib import Path, PurePosixPath
from scar.analysis.normal_facts import SourceFactStore
from scar.analysis.source_bytes import read_source_bytes
from scar.analysis.source_launch import read_script_entry
from scar.analysis.source_syntax import _digest
from scar.analysis.static.query import _Fragments
from scar.artifacts import write_table
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.artifacts import _json
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.python_bindings import SourceCallSite
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.record_codec import decode, encode


# Project standard pathlib source-text candidates without executing a target.
#
# Path terms retain original source sites. Lexical path alternatives are guarded
# by declared cwd, platform and filesystem-version conditions; resolve performs
# no target filesystem lookup. Text-open decoding uses captured raw bytes and
# has its own encoding/newline interpretation, independent of F0 PEP263 decoding.
_BASE = ('DECLARED_SCRIPT_ENTRY_CONTEXT_SELECTED', 'SOURCE_SCOPE_AND_NORMAL_COMPLETION_SELECTED',
         'STANDARD_LIBRARY_PATHLIB_REFERENCE_SELECTED', 'CURRENT_NAMESPACE_REFERENCE_ALTERNATIVE_SELECTED')
_EXPORTS = {'Path': ('POSIX', True), 'PurePath': ('POSIX', False),
            'PosixPath': ('POSIX', True), 'PurePosixPath': ('POSIX', False),
            'WindowsPath': ('WINDOWS', True), 'PureWindowsPath': ('WINDOWS', False)}


@dataclass(frozen=True)
class _String:
    value: str
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PathValue:
    expression: str
    lexical: str | None
    export: str
    trace: frozenset[int]
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class _Method:
    path: _PathValue
    receiver: int
    name: str


def _conditions(*groups):
    return tuple(dict.fromkeys(item for group in groups for item in group))


class _Projection:
    def __init__(self, store, unit, entry, limit):
        if type(store) is not SourceFactStore or type(limit) is not int or limit < 1:
            raise ValueError('an original SourceFactStore and positive path limit are required')
        self.store, self.unit, self.entry, self.limit = store, unit, entry, limit
        self.ports, self.f0s, self.manifests = {}, {}, {}
        self.paths, self.path_points, self.reads, self.candidates, self.obligations = {}, {}, {}, {}, set()
        self.values, self.adj, self.watch = defaultdict(set), defaultdict(set), defaultdict(set)
        self.queue = deque()
        self.owners = {(span.table.value, i): group.owner for group in unit.scope_groups
            for span in group.tables for i in range(span.start, span.stop)}
        model = Path(store.model_for(unit.unit))
        self.syntax = _Fragments(model).read(store.model_units[unit.unit]['artifacts']['source_syntax'])
        f0, _, _ = self.f0(unit.unit)
        if (entry.root != unit.root_scope or entry.source.unit != unit.unit
                or entry.source.f0_document_sha256 != _digest(f0)
                or entry.source.f2_document_sha256 != _digest(self.syntax)
                or self.syntax['input']['f0_document_sha256'] != _digest(f0)):
            raise ValueError('script entry and original F0/F2/root interfaces differ')
        self.fields = {row['name']:{name:i for i,name in enumerate(row['fields'])}
                       for row in self.syntax['profile']['ast_schema']}
        self.provider = None
        stdlib = str(Path(pathlib.__file__).absolute())
        for identity, row in store.model_units.items():
            if row['path'] == stdlib:
                _, raw, locator = self.f0(identity)
                if raw == Path(stdlib).read_bytes():
                    self.provider = locator
                    break
        for source, target in unit.aliases:
            self.adj[source].add(target)
        for access in unit.names:
            ns = access.namespace
            channel = ('namespace', ns.kind, ns.owner, ns.name)
            self.adj[access.value if access.write else channel].add(channel if access.write else access.value)
        self.attributes = defaultdict(list)
        for i, access in enumerate(unit.attributes):
            if access.action in (F.AttributeAction.GET, F.AttributeAction.IMPORT_MEMBER):
                self.attributes[access.receiver].append((i, access))
        for i, call in enumerate(unit.calls):
            for point in (call.callee, *call.positional, *(p for _,p in call.keywords)):
                self.watch[point].add(('call', i))
        for i, operation in enumerate(unit.operations):
            if operation.action is F.LanguageOperationKind.BINARY and operation.selector is F.OperatorSelector.DIV:
                for point in operation.operands:
                    self.watch[point].add(('join', i))
        for item in unit.constants:
            value = item.literal.to_python()
            if type(value) is str:
                self.offer(item.value, _String(value))
        for binding in entry.globals:
            value = binding.literal.to_python()
            if type(value) is str:
                reads = set(binding.reads)
                for access in unit.names:
                    if not access.write and access.namespace == binding.namespace and self.gp(access.value) in reads:
                        self.offer(access.value, _String(value, _conditions(entry.conditions,
                            ('DECLARED_INITIAL_NAMESPACE_VALUE_PRESERVED_AT_READ',))))
        if self.provider is not None:
            for imported in unit.imports:
                if imported.module == 'pathlib':
                    self.offer(imported.value, ('module', 'pathlib'))
        else:
            for imported in unit.imports:
                if imported.module == 'pathlib':
                    self.oblige(self.gp(imported.value), 'RETAINED_STANDARD_LIBRARY_PATHLIB_PROVIDER_NOT_AVAILABLE')

    def gp(self, index):
        if index not in self.ports:
            port = self.unit.ports[index]
            if type(port) is dict:
                port = decode(G.Port, port, record_tags=P.PRODUCTION_RECORD_TAGS)
            self.ports[index] = T.GlobalPort(self.unit.unit, port)
        return self.ports[index]

    def f0(self, identity):
        if identity not in self.f0s:
            model = Path(self.store.model_for(identity))
            manifest = (model/'manifest.json').read_bytes()
            if model in self.manifests and self.manifests[model] != manifest:
                raise ValueError('source model changed during projection')
            self.manifests[model] = manifest
            rows = json.loads(manifest)['units']
            ordinal, = [i for i,row in enumerate(rows) if row['key'] == identity]
            descriptor = self.store.model_units[identity]['artifacts']['source_bytes']
            document = _Fragments(model).read(descriptor)
            raw = read_source_bytes(document)
            declared = ProviderFactReference(ProviderArtifact(str(model/'manifest.json'),
                sha256(manifest).hexdigest(), len(manifest), 'scar.static-program'),
                '#/units/'+str(ordinal)+'/artifacts/source_bytes')
            locator = SourceArtifactLocator(ProviderArtifact(str(model/descriptor['path']),
                descriptor['sha256'], descriptor['bytes'], 'scar.source-bytes+gzip-json'),
                descriptor['decoded_sha256'], descriptor['decoded_bytes'], 'gzip-canonical-json', declaration=declared)
            self.f0s[identity] = document, raw, locator
        return self.f0s[identity]

    def oblige(self, port, reason):
        self.obligations.add(X.SourceTextObligation(port, reason))

    def offer(self, point, value):
        if value in self.values[point]:
            return
        if type(value) is _PathValue and sum(type(v) is _PathValue for v in self.values[point]) >= self.limit:
            if type(point) is int:
                self.oblige(self.gp(point), 'PATH_ALTERNATIVE_LIMIT_REACHED_NO_EXHAUSTIVE_VALUE_CLAIM')
            return
        self.values[point].add(value)
        self.queue.append((point, value))

    def site(self, call):
        return SourceCallSite(self.gp(call.callee), tuple(map(self.gp, call.positional)),
            tuple((n,self.gp(p)) for n,p in call.keywords), self.gp(call.result),
            self.gp(call.world), self.gp(call.result_world),
            None if call.positional_pack is None else self.gp(call.positional_pack),
            None if call.keyword_pack is None else self.gp(call.keyword_pack))

    def expression(self, kind, table, ordinal, result, operands, export, **sites):
        payload = dict(operation=kind, result=self.gp(result), operands=tuple(map(self.gp, operands)),
                       export=export, source_table=table, source_ordinal=ordinal)
        identity = sha256(_json(encode(payload, record_tags=P.PRODUCTION_RECORD_TAGS))).hexdigest()
        self.paths.setdefault(identity, X.SourcePathExpression(identity, self.gp(result),
            self.owners[table,ordinal], kind, tuple(map(self.gp, operands)), self.provider, export,
            table, ordinal, conditions=_BASE, **sites))
        self.path_points[identity] = result
        return identity

    def path_value(self, expression, result, lexical, export, previous=(), conditions=()):
        trace = frozenset().union(*(p.trace for p in previous))
        if result in trace:
            self.oblige(self.gp(result), 'CYCLIC_PATH_EXPRESSION_RETAINED_WITHOUT_UNROLLING')
            return
        self.offer(result, _PathValue(expression, lexical, export, trace | {result},
            _conditions(*(p.conditions for p in previous), conditions)))

    def attribute(self, ordinal, access, value):
        if value == ('module', 'pathlib') and access.name in _EXPORTS:
            self.offer(access.value, ('constructor', access.name))
        if type(value) is not _PathValue:
            return
        if access.name == 'parent':
            site = T.AttributeSite(T.AttributeOrigin.ATTRIBUTE, access.action, self.gp(access.receiver),
                access.name, self.gp(access.value), None, self.gp(access.world), self.gp(access.result_world))
            expression = self.expression(X.SourcePathOperation.PARENT, 'attributes', ordinal,
                access.value, (access.receiver,), value.export, attribute=site)
            lexical = None if value.lexical is None else str(PurePosixPath(value.lexical).parent)
            self.path_value(expression, access.value, lexical, value.export, (value,))
        elif access.name in ('resolve', 'read_text'):
            if _EXPORTS[value.export][1]:
                self.offer(access.value, _Method(value, access.receiver, access.name))
            else:
                self.oblige(self.gp(access.value), 'PURE_PATH_HAS_NO_FILESYSTEM_METHOD:'+access.name)

    def parameters(self, call):
        names = ('encoding', 'errors', 'newline')
        supplied = dict(call.keywords)
        note = []
        packed = call.positional_pack is not None or call.keyword_pack is not None
        if packed:
            note.append('READ_TEXT_ARGUMENT_PACK_BINDING_NOT_IMPLEMENTED')
        if len(call.positional)>3 or any(name not in names for name in supplied):
            note.append('READ_TEXT_SIGNATURE_ACCEPTANCE_NOT_ESTABLISHED')
        for name, point in zip(names, call.positional):
            if name in supplied:
                note.append('READ_TEXT_DUPLICATE_PARAMETER:'+name)
            supplied[name] = point
        node = self.syntax['nodes'][self.gp(call.result).port.template.occurrence]
        literal_by_point = {}
        if node['constructor'] == 'Call':
            fields = self.fields['Call']
            for point, child in zip(call.positional, node['fields'][fields['args']]['items']):
                expr = self.syntax['nodes'][child['reference']]
                if expr['constructor'] == 'Constant':
                    literal_by_point[point] = (decode(PythonLiteral,expr['fields'][self.fields['Constant']['value']]['scalar']),
                                               P.ProductionOccurrence(child['reference']))
            for child in node['fields'][fields['keywords']]['items']:
                keyword = self.syntax['nodes'][child['reference']]
                name = decode(PythonLiteral,keyword['fields'][self.fields['keyword']['arg']]['scalar']).to_python()
                ref = keyword['fields'][self.fields['keyword']['value']]['reference']
                expr = self.syntax['nodes'][ref]
                if name in supplied and expr['constructor'] == 'Constant':
                    literal_by_point[supplied[name]] = (decode(PythonLiteral,expr['fields'][self.fields['Constant']['value']]['scalar']),
                                                       P.ProductionOccurrence(ref))
        parameters = []
        for name in names:
            if name in supplied:
                literal, occurrence = literal_by_point.get(supplied[name], (None, None))
                parameters.append(X.TextReadParameter(name,self.gp(supplied[name]),literal,False,occurrence))
                if literal is None:
                    note.append('READ_TEXT_PARAMETER_VALUE_NOT_RESOLVED:'+name)
            elif packed:
                parameters.append(X.TextReadParameter(name,None,None,False))
            else:
                parameters.append(X.TextReadParameter(name,None,PythonLiteral.from_python(None),True))
        return tuple(parameters), tuple(note)

    def call(self, ordinal, call, callee):
        if type(callee) is tuple and callee[:1] == ('constructor',):
            export = callee[1]
            expression = self.expression(X.SourcePathOperation.CONSTRUCT, 'calls', ordinal,
                call.result, call.positional, export, call=self.site(call))
            if call.keywords or call.positional_pack is not None or call.keyword_pack is not None or len(call.positional)>1:
                self.oblige(self.gp(call.result), 'PATH_CONSTRUCTOR_SIGNATURE_NOT_IMPLEMENTED')
                self.path_value(expression, call.result, None, export)
                return
            values = self.values[call.positional[0]] if call.positional else (_String('.'),)
            found = False
            for value in tuple(values):
                if type(value) not in (_String, _PathValue):
                    continue
                found = True
                lexical = value.value if type(value) is _String else value.lexical
                conditions = list(value.conditions)
                if export in ('Path', 'PurePath'):
                    conditions.append('TARGET_PATHLIB_FLAVOUR_IS_POSIX')
                if _EXPORTS[export][0] != 'POSIX':
                    lexical = None
                    self.oblige(self.gp(call.result), 'WINDOWS_PATH_VALUE_PROJECTION_NOT_IMPLEMENTED')
                elif lexical is not None:
                    lexical = str(PurePosixPath(lexical))
                self.path_value(expression,call.result,lexical,export,
                    (value,) if type(value) is _PathValue else (),tuple(conditions))
            if not found:
                self.path_value(expression,call.result,None,export)
        elif type(callee) is _Method:
            path = callee.path
            if callee.name == 'resolve':
                expression = self.expression(X.SourcePathOperation.RESOLVE, 'calls', ordinal,
                    call.result, (callee.receiver,*call.positional), path.export, call=self.site(call))
                lexical = path.lexical
                conditions = ('DECLARED_CWD_SELECTED_AND_PRESERVED',
                    'PATH_RESOLUTION_HAS_NO_SYMLINK_REDIRECTION', 'PATH_RESOLVE_COMPLETES_NORMALLY')
                if lexical is not None:
                    lexical = posixpath.normpath(posixpath.join(self.entry.inputs.cwd,lexical))
                if call.positional_pack is not None or call.keyword_pack is not None or call.positional or call.keywords:
                    lexical = None
                    self.oblige(self.gp(call.result), 'RESOLVE_ARGUMENT_BINDING_NOT_IMPLEMENTED')
                self.path_value(expression,call.result,lexical,path.export,(path,),conditions)
            else:
                parameters, obligations = self.parameters(call)
                read = X.SourceTextRead(self.site(call),self.owners['calls',ordinal],self.gp(callee.receiver),
                    path.expression,parameters,_BASE,obligations)
                self.reads[ordinal,path.expression] = read
                self.text_candidates(read,path)

    def join(self, ordinal, operation):
        if len(operation.operands) != 2:
            return
        left, right = operation.operands
        site = T.LanguageOperationSite(operation.action,operation.selector,tuple(map(self.gp,operation.operands)),
            self.gp(operation.result),self.gp(operation.world),self.gp(operation.result_world))
        for path in tuple(self.values[left]):
            if type(path) is not _PathValue:
                continue
            expression = self.expression(X.SourcePathOperation.JOIN,'operations',ordinal,
                operation.result,operation.operands,path.export,language=site)
            strings = [value for value in self.values[right] if type(value) is _String]
            if not strings:
                self.path_value(expression,operation.result,None,path.export,(path,))
            for value in strings:
                lexical = None if path.lexical is None else str(PurePosixPath(path.lexical)/value.value)
                self.path_value(expression,operation.result,lexical,path.export,(path,),value.conditions)

    def text_candidates(self, read, path):
        if path.lexical is None:
            return
        lexical = path.lexical
        path_conditions = list(path.conditions)
        if not posixpath.isabs(lexical):
            lexical = posixpath.normpath(posixpath.join(self.entry.inputs.cwd,lexical))
            path_conditions.extend(('DECLARED_CWD_SELECTED_AND_PRESERVED','RELATIVE_PATH_HAS_NO_SYMLINK_REDIRECTION'))
        matches = [identity for identity,row in self.store.model_units.items() if row['path'] == lexical]
        if not matches:
            self.oblige(read.site.result,'NO_RETAINED_F0_ARTIFACT_AT_LEXICAL_PATH_CANDIDATE')
        for identity in matches:
            f0, raw, locator = self.f0(identity)
            values = {p.name:p.literal.to_python() for p in read.parameters if p.literal is not None}
            encoding = values.get('encoding')
            errors = values.get('errors')
            newline = values.get('newline')
            default_encoding = 'encoding' in values and encoding is None
            conditions, obligations = [], list(read.obligations)
            if default_encoding:
                encoding = 'utf-8'
                conditions.append('TEXT_OPEN_DEFAULT_ENCODING_SELECTS_UTF8')
                obligations.append('DEFAULT_ENCODING_CONTEXT_NOT_DECLARED_CONDITIONAL_UTF8_INTERPRETATION_ONLY')
            codecs = {'utf-8':'utf-8','utf8':'utf-8','utf-8-sig':'utf-8-sig',
                      'ascii':'ascii','latin-1':'latin-1','latin1':'latin-1','iso-8859-1':'latin-1'}
            encoding = codecs.get(encoding.lower()) if type(encoding) is str else None
            errors = 'strict' if 'errors' in values and errors is None else errors
            newline_ready = 'newline' in values and (newline is None or newline in ('','\n','\r','\r\n'))
            if encoding is None:
                obligations.append('TEXT_OPEN_ENCODING_NOT_SUPPORTED_OR_NOT_RESOLVED')
            if errors not in ('strict','ignore','replace'):
                obligations.append('TEXT_OPEN_ERRORS_NOT_SUPPORTED_OR_NOT_RESOLVED')
            if not newline_ready:
                obligations.append('TEXT_OPEN_NEWLINE_NOT_SUPPORTED_OR_NOT_RESOLVED')
            text_sha, characters = None, None
            if not read.obligations and encoding is not None and errors in ('strict','ignore','replace') and newline_ready:
                try:
                    with TextIOWrapper(BytesIO(raw),encoding=encoding,errors=errors,newline=newline) as stream:
                        text = stream.read()
                    text_sha, characters = sha256(text.encode('utf-8')).hexdigest(), len(text)
                except UnicodeError:
                    obligations.append('CAPTURED_BYTES_DO_NOT_HAVE_A_SUCCESSFUL_CANDIDATE_TEXT_INTERPRETATION')
            interpretation = X.TextOpenInterpretation(encoding,errors,newline,newline_ready,default_encoding,
                conditions=tuple(conditions),obligations=tuple(obligations))
            candidate = X.SourceTextCandidate(read.site.result,path.expression,identity,f0['path'],locator,
                _digest(f0),sha256(raw).hexdigest(),len(raw),interpretation,text_sha,characters,
                _conditions(_BASE,path_conditions,conditions,('PATH_CANDIDATE_SELECTS_CAPTURED_ARTIFACT',
                    'CAPTURED_FILE_VERSION_PRESERVED_UNTIL_READ','TEXT_READ_COMPLETES_NORMALLY')),
                tuple(obligations))
            self.candidates[_json(encode(candidate,record_tags=P.PRODUCTION_RECORD_TAGS))] = candidate

    def run(self):
        while self.queue:
            point, value = self.queue.popleft()
            for target in tuple(self.adj.get(point,())):
                self.offer(target,value)
            for ordinal, access in self.attributes.get(point,()):
                self.attribute(ordinal,access,value)
            for kind, ordinal in tuple(self.watch.get(point,())):
                if kind == 'call':
                    call = self.unit.calls[ordinal]
                    for callee in tuple(self.values[call.callee]):
                        self.call(ordinal,call,callee)
                else:
                    self.join(ordinal,self.unit.operations[ordinal])
        concrete = {v.expression for values in self.values.values() for v in values
                    if type(v) is _PathValue and v.lexical is not None}
        for expression in self.paths.values():
            if expression.id not in concrete:
                self.oblige(expression.result,'PATH_INPUT_OR_PLATFORM_VALUE_NOT_RESOLVED_SYMBOLIC_EXPRESSION_RETAINED')
        for read in self.reads.values():
            if not any(candidate.read_result==read.site.result for candidate in self.candidates.values()):
                self.oblige(read.site.result,'READ_TEXT_PATH_HAS_NO_CAPTURED_SOURCE_CANDIDATE')
        if (self.store.path/'result.json').read_bytes() != self.store.raw:
            raise ValueError('source fact catalog changed during projection')
        for model, manifest in self.manifests.items():
            if (model/'manifest.json').read_bytes() != manifest:
                raise ValueError('source model changed during projection')
        path_values = {}
        for expression in self.paths.values():
            for value in self.values[self.path_points[expression.id]]:
                if type(value) is _PathValue and value.expression == expression.id and value.lexical is not None:
                    candidate = X.SourcePathCandidate(expression.id,expression.result,value.lexical,
                        _conditions(_BASE,value.conditions))
                    path_values[_json(encode(candidate,record_tags=P.PRODUCTION_RECORD_TAGS))] = candidate
        return X.SourceTextOverlay(self.unit.unit,tuple(self.paths.values()),
            tuple(path_values[key] for key in sorted(path_values)),tuple(self.reads.values()),
            tuple(self.candidates.values()),tuple(sorted(self.obligations,key=lambda item:(str(item.port),item.reason))))


def source_text_candidates(store, unit, entry, *, max_path_values=8):
    """Use actual parent-store receipts; result remains conditional source data."""
    return _Projection(store,unit,entry,max_path_values).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--facts',required=True)
    parser.add_argument('--entry-contract',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    store = SourceFactStore(args.facts)
    entry = read_script_entry(args.entry_contract,store)
    overlay = source_text_candidates(store,store.load(entry.source.unit),entry)
    out = Path(args.out).absolute()
    if out.exists():
        raise ValueError('fresh source-text output required')
    _durable_mkdir(out)
    tables = {name:write_table(out/(name+'.jsonl.gz'),getattr(overlay,name))
              for name in ('paths','path_values','reads','candidates','obligations')}
    report = dict(schema='scar.source-text-candidates',schema_version=1,
        status='CONDITIONAL_CAPTURED_SOURCE_TEXT_CANDIDATES',source_facts=str(store.path),
        source_facts_sha256=sha256(store.raw).hexdigest(),entry_contract=str(Path(args.entry_contract).absolute()),
        entry_result_sha256=sha256((Path(args.entry_contract)/'result.json').read_bytes()).hexdigest(),
        entry_sha256=sha256((Path(args.entry_contract)/'entry.json').read_bytes()).hexdigest(),
        unit=overlay.unit,tables=tables,target_executed=False,source_body_activated=False,
        exact_runtime_text_proven=False,first_stage_complete=False,
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        types_sha256=sha256(Path(X.__file__).read_bytes()).hexdigest(),
        interpretation='TEXT_OPEN_RAW_BYTES_SEPARATE_FROM_F0_PEP263;DEFAULT_UTF8_IS_CONDITIONAL')
    _durable_bytes(out/'result.json',_json(report),replace=False)
    print(json.dumps(report),flush=True)


# Bounded builtin values for original F5 language-operation interfaces.
#
# This adapter invokes only the existing exact-builtin checker.  Results concern
# conditional builtin content, not object identity, runtime dispatch, effects,
# source authenticity, or execution of a target program.
_UNARY_SELECTORS = {
    F.OperatorSelector.U_ADD: 'POS', F.OperatorSelector.U_SUB: 'NEG',
    F.OperatorSelector.INVERT: 'INVERT', F.OperatorSelector.NOT: 'NOT',
}
_BINARY_SELECTORS = {
    F.OperatorSelector.ADD: 'ADD', F.OperatorSelector.SUB: 'SUB',
    F.OperatorSelector.MULT: 'MUL', F.OperatorSelector.DIV: 'TRUE_DIV',
    F.OperatorSelector.FLOOR_DIV: 'FLOOR_DIV', F.OperatorSelector.MOD: 'MOD',
    F.OperatorSelector.POW: 'POW', F.OperatorSelector.L_SHIFT: 'LSHIFT',
    F.OperatorSelector.R_SHIFT: 'RSHIFT', F.OperatorSelector.BIT_AND: 'BIT_AND',
    F.OperatorSelector.BIT_OR: 'BIT_OR', F.OperatorSelector.BIT_XOR: 'BIT_XOR',
}


@dataclass(frozen=True, slots=True)
class BuiltinValueResult:
    status: ConstantStatus
    literal: PythonLiteral | None
    reason: str = ''
    exception: str | None = None
    requirements: tuple[ConstantRuntimeRequirement, ...] = ()

    def __post_init__(self):
        if type(self.status) is not ConstantStatus:
            raise TypeError('builtin result requires ConstantStatus')
        if self.status is ConstantStatus.CONSTANT:
            if type(self.literal) is not PythonLiteral:
                raise ValueError('a constant result requires a typed literal')
        elif self.literal is not None:
            raise ValueError('a failed evaluation cannot supply a constant literal')


class BuiltinValueEvaluator:
    """Memoized primitive evaluations under one cumulative budget.

    A cache miss charges one step and one logical value node for each operand
    root plus the result.  Literal trees also obey the existing size/depth/item
    checks.  Cached outcomes do not repeat evaluation or consume budget again;
    terminal budget refusals are not inserted as unbounded new cache entries.
    These resource limits do not replace the caller's finite-height value
    domain or establish that any original operation executes.
    """

    def __init__(self, budget=EvaluationBudget()):
        if type(budget) is not EvaluationBudget:
            raise TypeError('budget must be EvaluationBudget')
        budget.__post_init__()
        self.budget = budget
        self.runtime = BuiltinRuntime.current()
        self.rule_version = RULE_VERSION
        self._state = dict(nodes=0, steps=0, work=0, output_bytes=0)
        self._cache = {}
        self._calls = self._hits = self._misses = 0
        self._statuses = {}

    @property
    def usage(self):
        return EvaluationUsage(**self._state)

    @property
    def stats(self):
        return dict(calls=self._calls, cache_hits=self._hits,
            cache_misses=self._misses, cache_entries=len(self._cache),
            usage=dict(self._state), statuses=dict(self._statuses))

    @staticmethod
    def opcode_for(action, selector):
        if type(action) is not F.LanguageOperationKind or type(selector) is not F.OperatorSelector:
            raise TypeError('original language-operation enums required')
        if action is F.LanguageOperationKind.UNARY:
            return _UNARY_SELECTORS.get(selector)
        if action in (F.LanguageOperationKind.BINARY, F.LanguageOperationKind.INPLACE_BINARY):
            return _BINARY_SELECTORS.get(selector)
        return None

    def _return(self, result):
        key = result.status.value
        self._statuses[key] = self._statuses.get(key, 0) + 1
        return result

    def evaluate(self, action, selector, ordered_literals):
        """Return a typed outcome without executing arbitrary Python values."""
        opcode = self.opcode_for(action, selector)
        if (type(ordered_literals) is not tuple
                or any(type(item) is not PythonLiteral for item in ordered_literals)):
            raise TypeError('ordered_literals must be a tuple of exact PythonLiteral records')
        self._calls += 1
        if opcode is None:
            return self._return(BuiltinValueResult(ConstantStatus.NOT_CONSTANT, None,
                'no fixed builtin checker for ' + action.value + '/' + selector.value))
        arity = 1 if action is F.LanguageOperationKind.UNARY else 2
        if len(ordered_literals) != arity:
            return self._return(BuiltinValueResult(ConstantStatus.NOT_CONSTANT, None,
                'builtin opcode requires ' + str(arity) + ' ordered operands'))
        # Shared literal grammar validates before recursive dataclass hashing;
        # exact-only kinds are rejected, never silently coerced to legacy data.
        try:
            for item in ordered_literals:
                require_legacy_literal(item)
        except (ValueError, TypeError) as error:
            return self._return(BuiltinValueResult(ConstantStatus.NOT_CONSTANT, None, str(error)))
        key = opcode, ordered_literals
        if key in self._cache:
            self._hits += 1
            return self._return(self._cache[key])
        self._misses += 1
        if self._state['steps'] >= self.budget.max_steps:
            return self._return(BuiltinValueResult(ConstantStatus.BUDGET_EXCEEDED, None,
                'constant step budget exceeded'))
        nodes = len(ordered_literals) + 1
        if nodes > self.budget.max_nodes - self._state['nodes']:
            return self._return(BuiltinValueResult(ConstantStatus.BUDGET_EXCEEDED, None,
                'constant value-node budget exceeded'))
        self._state['nodes'] += nodes
        self._state['steps'] += 1
        requirements = [ConstantRuntimeRequirement.BUILTIN_RUNTIME_MATCH]
        if ((opcode in _BINARY or opcode in ('POS', 'NEG'))
                and (opcode == 'TRUE_DIV' or any(item.kind is LiteralKind.FLOAT64 for item in ordered_literals))):
            requirements.append(ConstantRuntimeRequirement.FLOATING_ENVIRONMENT_MATCH)
        requirements = tuple(requirements)
        try:
            sizes = tuple(_literal_size(item, self.budget) for item in ordered_literals)
            values = tuple(_python_value(item) for item in ordered_literals)
            value = _builtin(opcode, values, sizes, self.budget, self._state, ordered_literals)
            literal = _literal(value)
            size = _literal_size(literal, self.budget)
            _reserve(size, self._state, self.budget)
            self._state['output_bytes'] += size.bytes
            result = BuiltinValueResult(ConstantStatus.CONSTANT, literal,
                requirements=requirements)
        except _BudgetExceeded as error:
            result = BuiltinValueResult(ConstantStatus.BUDGET_EXCEEDED, None, str(error))
        except _Unsupported as error:
            result = BuiltinValueResult(ConstantStatus.NOT_CONSTANT, None, str(error))
        except (ArithmeticError, IndexError, ValueError) as error:
            result = BuiltinValueResult(ConstantStatus.KNOWN_EXCEPTION, None,
                'fixed builtin operation raises ' + type(error).__name__,
                exception=type(error).__name__, requirements=requirements)
        self._cache[key] = result
        return self._return(result)


# Independent finite-height scalar fields, without source-solver admission.
#
# Only original GlobalPorts and LanguageOperationSites are referenced.  A small
# retained literal set supplies conditional may witnesses; open content is a
# field property, never an invented unknown object or source operation.  Worlds
# remain on the original site and do not become scalar-reference aliases.
@dataclass(slots=True)
class _Field:
    literals: dict = field(default_factory=dict)
    builtin_types: set = field(default_factory=set)
    content_open: bool = False
    widened: bool = False


@dataclass(frozen=True, slots=True)
class _Junction:
    """Private transfer index; never an identity in the program graph."""
    key: object


def _requirements(values):
    # The reused kernel owns legacy report enums. Convert them at this
    # analysis boundary; the canonical IR never imports an analysis module.
    return tuple(sorted({V.ScalarRuntimeRequirement(item.value) for item in values},
                        key=lambda item: item.value))


class ScalarValueFlow:
    """Monotone bounded fields; draining proves only this agenda is empty.

    Per port, at most K literals are admitted, each with a finite set of
    runtime requirements.  Open/widened flags only grow.  Operator premises
    range over those bounded sets, so feedback such as x = x + 1 terminates.
    Source and derivation witnesses establish neither must values nor identity.
    """

    def __init__(self, *, limit=4, evaluator=None):
        if type(limit) is not int or limit < 1:
            raise ValueError('literal limit must be a positive exact integer')
        if evaluator is not None and type(evaluator) is not BuiltinValueEvaluator:
            raise TypeError('the fixed BuiltinValueEvaluator is required')
        self.limit = limit
        self.evaluator = BuiltinValueEvaluator() if evaluator is None else evaluator
        self._fields = {}
        self._junction_fields = {}
        self._groups = {}
        self._operations = {}
        self._supported = {}
        self._watch = defaultdict(set)
        self._adj = defaultdict(set)
        self._agenda = deque()
        self._queued = set()
        self._processed = defaultdict(dict)
        self._sources = {}
        self._derivations = {}
        self._obligations = {}
        self._events = 0

    def _field(self, port):
        if type(port) is not T.GlobalPort:
            raise TypeError('an original GlobalPort is required')
        return self._state(port)

    def _state(self, port):
        if type(port) is _Junction:
            return self._junction_fields[port]
        if type(port) is not T.GlobalPort:
            raise TypeError('an original GlobalPort is required')
        if port not in self._fields:
            self._fields[port] = _Field()
        return self._fields[port]

    def _enqueue(self, kind, key):
        event = kind, key
        if event not in self._queued:
            self._queued.add(event)
            self._agenda.append(event)

    def _mark_open(self, port):
        state = self._state(port)
        if state.content_open:
            return False
        state.content_open = True
        self._enqueue('port', port)
        return True

    def _accept(self, port, literal, requirements=()):
        state = self._state(port)
        builtin = literal_builtin_type(literal)
        changed = self._accept_types(port, (builtin.__module__ + '.' + builtin.__qualname__,))
        if literal in state.literals:
            previous = state.literals[literal]
            merged = frozenset((*previous, *requirements))
            if merged != previous:
                state.literals[literal] = merged
                self._enqueue('port', port)
                changed = True
            return changed
        if state.widened:
            return changed
        if len(state.literals) >= self.limit:
            state.widened = True
            self._mark_open(port)
            self._enqueue('port', port)
            if type(port) is T.GlobalPort:
                self._obligations['limit', port] = V.ScalarObligation(port,
                    V.ScalarObligationKind.LITERAL_LIMIT,
                    'per-port literal limit exceeded: ' + str(self.limit))
            return True
        state.literals[literal] = frozenset(requirements)
        self._enqueue('port', port)
        return True

    def _accept_types(self, port, types):
        state = self._state(port)
        new = set(types) - state.builtin_types
        if not new:
            return False
        state.builtin_types.update(new)
        self._enqueue('port', port)
        return True

    def offer(self, port, literal):
        """Admit a direct literal observation, not an object-identity fact."""
        if type(literal) is not PythonLiteral:
            raise TypeError('a typed PythonLiteral is required')
        _preflight_literal(literal)
        changed = self._accept(port, literal)
        if literal in self._field(port).literals:
            self._sources[port, literal] = V.ScalarLiteralSource(port, literal)
        return changed

    def open(self, port, reason):
        if type(reason) is not str or not reason:
            raise ValueError('an open scalar field needs a nonempty reason')
        self._field(port)
        self._obligations['source_open', port, reason] = V.ScalarObligation(port,
            V.ScalarObligationKind.SOURCE_OPEN, reason)
        return self._mark_open(port)

    def connect(self, source, target):
        """Forward scalar fields without asserting reference equality."""
        self._field(source)
        self._field(target)
        if source == target or target in self._adj[source]:
            return False
        self._adj[source].add(target)
        self._enqueue('port', source)
        return True

    def connect_group(self, key, group):
        """Store O(inputs + outputs), without expanding their product.

        A bounded private join shares the union of scalar alternatives. Its
        widening is propagated as open content, never as a complete value.
        """
        if type(group) is not V.ScalarTransferGroup:
            raise TypeError('a typed scalar transfer group is required')
        if not group.sources or not group.targets:
            raise ValueError('a scalar transfer group needs both endpoints')
        if key in self._groups:
            if self._groups[key] != group:
                raise ValueError('scalar transfer key selects another relation')
            return False
        for port in (*group.sources, *group.targets):
            self._field(port)
        junction = _Junction(key)
        self._junction_fields[junction] = _Field()
        self._groups[key] = group
        self._adj[junction].update(group.targets)
        for source in group.sources:
            self._adj[source].add(junction)
            self._enqueue('port', source)
        return True

    def register(self, key, site):
        if type(site) is not T.LanguageOperationSite:
            raise TypeError('an original LanguageOperationSite is required')
        if (type(site.operands) is not tuple or any(type(p) is not T.GlobalPort
                for p in (*site.operands, site.result, site.world, site.result_world))):
            raise TypeError('language operation interfaces require original GlobalPorts')
        opcode = self.evaluator.opcode_for(site.action, site.selector)
        if key in self._operations:
            if self._operations[key] != site:
                raise ValueError('scalar operation key selects another source interface')
            return False
        self._operations[key] = site
        self._field(site.result)
        for operand in site.operands:
            self._field(operand)
            self._watch[operand].add(key)
        arity = 1 if site.action is F.LanguageOperationKind.UNARY else 2
        self._supported[key] = opcode is not None and len(site.operands) == arity
        if not self._supported[key]:
            self._obligations['unsupported', key] = V.ScalarObligation(site.result,
                V.ScalarObligationKind.UNSUPPORTED_OPERATOR,
                'no fixed builtin checker for the original operator/signature',
                site=site, inputs=site.operands, status=V.ScalarEvaluationStatus.NOT_CONSTANT)
            self._mark_open(site.result)
        self._enqueue('operation', key)
        return True

    def _forward(self, source):
        state = self._state(source)
        for target in tuple(self._adj.get(source, ())):
            for literal, requirements in tuple(state.literals.items()):
                self._accept(target, literal, requirements)
            self._accept_types(target, state.builtin_types)
            if state.content_open:
                if type(target) is T.GlobalPort:
                    self._obligations['upstream', source, target] = V.ScalarObligation(target,
                        V.ScalarObligationKind.UPSTREAM_OPEN, 'upstream scalar content is open',
                        inputs=(source,) if type(source) is T.GlobalPort else ())
                self._mark_open(target)
        for key in tuple(self._watch.get(source, ())):
            self._enqueue('operation', key)

    def _evaluate(self, key):
        site = self._operations[key]
        states = tuple(self._field(p) for p in site.operands)
        if any(state.content_open for state in states):
            self._obligations['operand_open', key] = V.ScalarObligation(site.result,
                V.ScalarObligationKind.OPERAND_OPEN, 'operand scalar content is open',
                site=site, inputs=site.operands)
            self._mark_open(site.result)
        if not self._supported[key]:
            return
        if any(not state.literals for state in states):
            self._obligations['input_absent', key] = V.ScalarObligation(site.result,
                V.ScalarObligationKind.INPUT_LITERAL_ABSENT,
                'an original operand has no retained literal candidate',
                site=site, inputs=site.operands)
            return
        self._obligations.pop(('input_absent', key), None)
        for literals in product(*(tuple(state.literals) for state in states)):
            if self._field(site.result).widened and (key, literals) not in self._derivations:
                continue
            inherited = frozenset(r for state, literal in zip(states, literals)
                                 for r in state.literals[literal])
            if self._processed[key].get(literals) == inherited:
                continue
            self._processed[key][literals] = inherited
            premises = tuple(V.ScalarPremise(port, literal,
                _requirements(state.literals[literal]))
                for port, state, literal in zip(site.operands, states, literals))
            outcome = self.evaluator.evaluate(site.action, site.selector, literals)
            requirements = _requirements((*inherited, *outcome.requirements))
            if outcome.status is ConstantStatus.CONSTANT:
                self._accept(site.result, outcome.literal, requirements)
                if outcome.literal in self._field(site.result).literals:
                    self._derivations[key, literals] = V.ScalarDerivation(site,
                        premises, V.ScalarPremise(site.result, outcome.literal, requirements),
                        requirements)
            else:
                self._obligations['evaluation', key, literals] = V.ScalarObligation(site.result,
                    V.ScalarObligationKind.EVALUATION_FAILURE, outcome.reason,
                    site=site, inputs=site.operands, premises=premises,
                    status=V.ScalarEvaluationStatus(outcome.status.value), exception=outcome.exception,
                    requirements=requirements)
                if outcome.status is not ConstantStatus.KNOWN_EXCEPTION:
                    self._mark_open(site.result)

    def drain(self):
        """Reach the bounded field fixed point; obligations can remain open."""
        while self._agenda:
            event = self._agenda.popleft()
            self._queued.remove(event)
            self._events += 1
            kind, key = event
            if kind == 'port':
                self._forward(key)
            else:
                self._evaluate(key)
        return True

    def snapshot(self, port):
        state = self._field(port)
        return V.ScalarSnapshot(port, tuple(state.literals),
            tuple(sorted(state.builtin_types)), state.content_open, state.widened)

    def snapshots(self):
        """Publish original-port fields; private transfer joins stay private."""
        return (self.snapshot(port) for port in self._fields)

    def derivations(self):
        return tuple(self._derivations.values())

    def obligations(self):
        return tuple(self._obligations.values())

    def operations(self):
        return tuple(self._operations.values())

    def sources(self):
        return tuple(self._sources.values())

    def connections(self):
        return tuple(V.ScalarReferenceFlow(source, target)
            for source, targets in self._adj.items() for target in targets
            if type(source) is T.GlobalPort and type(target) is T.GlobalPort)

    def groups(self):
        return tuple(self._groups.values())

    @property
    def stats(self):
        return dict(ports=len(self._fields), operations=len(self._operations),
            connections=sum(len(v) for v in self._adj.values()),
            transfer_groups=len(self._groups), private_junctions=len(self._junction_fields),
            literals=sum(len(s.literals) for s in self._fields.values()),
            open_ports=sum(s.content_open for s in self._fields.values()),
            widened_ports=sum(s.widened for s in self._fields.values()),
            literal_limit=self.limit, events=self._events, pending=len(self._agenda),
            sources=len(self._sources), derivations=len(self._derivations),
            obligations=len(self._obligations), evaluator=self.evaluator.stats)


def collect_unit(unit: F.UnitFacts, flow=None, *, scopes=None):
    """Collect one retained unit's conditional constraints, without draining.

    Without scopes, collection does not admit or execute a body. With scopes,
    only original admitted table spans are collected, retaining row ordinals.
    Namespace joins are qualified by this unit; no cross-call inference occurs.
    Only referenced ports are decoded, with the original strict record codec.
    """
    from scar.ir.program import normal_graph as G, source_productions as P
    from scar.ir.record_codec import _decode, _record_tags

    flow = ScalarValueFlow() if flow is None else flow
    if type(flow) is not ScalarValueFlow:
        raise TypeError('a ScalarValueFlow is required')
    ports = {}
    tags = _record_tags(P.PRODUCTION_RECORD_TAGS)

    def records(table):
        rows = getattr(unit, table.value)
        if scopes is None:
            yield from enumerate(rows)
        else:
            for group in unit.scope_groups:
                if group.owner in scopes:
                    for span in group.tables:
                        if span.table is table:
                            yield from ((i, rows[i]) for i in range(span.start, span.stop))

    def gp(index):
        if type(index) is not int or not 0 <= index < len(unit.ports):
            raise ValueError('scalar fact references a port outside its original unit')
        if index not in ports:
            port = unit.ports[index]
            if type(port) is dict:
                port = _decode(G.Port, port, tags)
            elif type(port) is not G.Port:
                raise TypeError('original source port record required')
            ports[index] = T.GlobalPort(unit.unit, port)
        return ports[index]

    for _, constant in records(F.FactTable.CONSTANTS):
        flow.offer(gp(constant.value), constant.literal)
    for _, (source, target) in records(F.FactTable.ALIASES):
        flow.connect(gp(source), gp(target))
    namespaces = defaultdict(lambda: (set(), set()))
    namespace_rows = {}
    for index, access in records(F.FactTable.NAMES):
        namespace = access.namespace
        key = unit.unit, namespace.kind, namespace.owner, namespace.name
        namespaces[key][int(not access.write)].add(access.value)
        namespace_rows.setdefault(key, index)
    for key, (writes, reads) in namespaces.items():
        if writes and reads:
            kind, owner, name = key[1:]
            # Endpoint sets retain the exact relation; this private factor
            # avoids the quadratic writes x reads adjacency expansion.
            flow.connect_group(('unit.names', key), V.ScalarTransferGroup(
                'unit.names', namespace_rows[key], 'namespace/'+kind.value+'/'+str(owner)+'/'+name,
                tuple(gp(p) for p in sorted(writes)), tuple(gp(p) for p in sorted(reads)),
                ('FLOW_INSENSITIVE_SOURCE_NAMESPACE_ALTERNATIVES',
                 'SOURCE_SCOPE_AND_NAMESPACE_VERSION_SELECTION_REQUIRED')))
    for index, operation in records(F.FactTable.OPERATIONS):
        flow.register((unit.unit, index), T.LanguageOperationSite(
            operation.action, operation.selector,
            tuple(gp(p) for p in operation.operands), gp(operation.result),
            gp(operation.world), gp(operation.result_world)))
    return flow


__all__ = [
    'ScalarValueFlow',
    'collect_unit',
    'BuiltinValueEvaluator',
    'BuiltinValueResult',
    'source_text_candidates',
    'main',
]


if __name__ == '__main__':
    main()
