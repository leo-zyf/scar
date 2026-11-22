"""Namespace values."""

from scar.ir.program import source_productions as P, name_flow as N, normal_graph as F
from scar.ir.program.structured import PythonProtocolKind as K
from collections import defaultdict
from dataclasses import dataclass
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.source_exec_roots import ExecEndpoint
from scar.ir.program.name_flow import NamespaceInterfaceRef
from scar.ir.program.source_paths import GuardedBuiltinCandidate, BuiltinNamespaceSelection


# Extract actual globals operands from original function-creation steps.
#
# These are source relations, not bindings of a scope to an exec dictionary.
# Decorators and later callable/environment selection retain their own edges.
_FACTORIES = frozenset((K.MAKE_LAMBDA_FUNCTION,K.MAKE_GENEXPR_FUNCTION,K.MAKE_NESTED_FUNCTION))


def extract_captures(view) -> tuple[tuple[N.GlobalsEnvironmentReference, ...],
        tuple[N.FunctionGlobalsCapture, ...], tuple[N.NamespaceGraphObligation, ...]]:
    """Return globals references, function captures, and unresolved factories.

    ``view`` resolves only original plan ports and compilation-scope witnesses.
    A factory's code operand must name its local DeferRegion result. Otherwise
    retain an obligation instead of guessing from a scope or source spelling.
    """
    references,captures,obligations = [],[],[]
    for step in view.plan.steps:
        instruction = step.instruction
        if type(instruction) is P.ReferenceEnvironment:
            if type(instruction.source) is P.GlobalsEnvironment:
                world_in,world_out = view.worlds(step)
                references.append(N.GlobalsEnvironmentReference(view.site(step),view.owner(step.region),
                    instruction.source.scope,view.result(step),world_in,world_out))
            continue
        factory = (type(instruction) is P.BuildFunction or
            type(instruction) is P.ApplyPythonProtocol and instruction.protocol in _FACTORIES)
        if not factory:
            continue

        def unresolved(reason):
            obligations.append(N.NamespaceGraphObligation(view.site(step),reason))

        # The existing factory ABI is code, globals, remaining references,
        # world. Neither globals nor the code is the callable result itself.
        if len(step.arguments) < 3:
            unresolved('FUNCTION_GLOBALS_FACTORY_ARGUMENTS_UNRESOLVED')
            continue
        code = step.arguments[0]
        if type(code) is not P.ProductionResult or code.index != 0:
            unresolved('FUNCTION_GLOBALS_CODE_DECLARATION_UNRESOLVED')
            continue
        declaration_step = view.steps.get(code.step)
        if declaration_step is None or type(declaration_step.instruction) is not P.DeferRegion:
            unresolved('FUNCTION_GLOBALS_CODE_DECLARATION_UNRESOLVED')
            continue
        declaration = declaration_step.instruction
        if type(instruction) is P.BuildFunction and instruction.environment != declaration.environment:
            unresolved('FUNCTION_GLOBALS_DECLARATION_ENVIRONMENT_MISMATCH')
            continue
        scope = declaration.environment.globals_scope
        if type(scope) is not int or scope < 0:
            unresolved('FUNCTION_GLOBALS_DECLARATION_SCOPE_UNRESOLVED')
            continue
        try:
            body = view.owner(declaration.body)
        except KeyError:
            unresolved('FUNCTION_GLOBALS_BODY_OWNER_UNRESOLVED')
            continue
        world_in,world_out = view.worlds(step)
        captures.append(N.FunctionGlobalsCapture(view.site(step),view.owner(step.region),
            view.result(step),view.value(code),view.value(step.arguments[1]),body,scope,world_in,world_out))
    return tuple(references),tuple(captures),tuple(obligations)


# Subscribe to actual builtin-map operands without allocating graph ports.
#
# Globals entry selection occurs at root entry or function creation. Contents
# of the selected mapping remain ordinary may reads at the original use world.
# Standard builtin alternatives carry separately published selection guards.
@dataclass(frozen=True, slots=True)
class _Read:
    connection: str
    access: GlobalPort
    interface: NamespaceInterfaceRef
    key: str
    target: int
    selected_at: ExecEndpoint
    capturing_function: GlobalPort | None
    globals_reference: GlobalPort | None
    read_world: GlobalPort
    conditions: tuple[str, ...]


class BuiltinNamespaceFlow:
    def __init__(self, flow):
        self.flow,self.s = flow,flow.s
        self.watchers = defaultdict(set)
        self.globals_readers = defaultdict(set)
        self.fallbacks,self.selections = [],[]
        self._fallbacks,self._selections,self._seen = set(),set(),set()
        self._admitted = set()

    def _gp(self, point):
        value = self.s.gp(point)
        if type(value) is not GlobalPort:
            raise ValueError('builtin namespace operands require original source ports')
        return value

    def _candidate(self, request, conditions):
        row = GuardedBuiltinCandidate(request.connection,request.access,request.interface,
            request.selected_at,request.capturing_function,
            tuple(dict.fromkeys((*request.conditions,*conditions,
                'STANDARD_BUILTIN_MAP_AND_BINDING_SELECTED',
                'ALL_EARLIER_C2_LOOKUP_STEPS_MISS_AT_READ_WORLD'))))
        if row not in self._fallbacks:
            self._fallbacks.add(row);self.fallbacks.append(row)
        if request.key in self.s.builtin_names:
            self.s.add(request.target,('external','builtins.'+request.key))

    def _watch(self, point, mode, request, container=None, unknown_key=False):
        spec = mode,request,container,unknown_key
        if spec in self.watchers[point]:
            return
        self.watchers[point].add(spec)
        for token in tuple(self.s.values.get(point,())):
            self.consume(point,token)

    def admit(self, connection, access, binding):
        if (binding.descriptor.kind != 'BUILTINS' or access.action is not P.BindingActionKind.READ
                or access.value is None):
            return
        identity = connection.id,access.site,binding.interface
        if identity in self._admitted:
            return
        self._admitted.add(identity)
        if access.owner.owner == connection.root.owner:
            choices = ((self.flow.formal(connection,0),self._gp(self.flow.formal(connection,2)),None,
                'ROOT_FRAME_BUILTINS_SELECTED_AT_ENTRY'),)
        else:
            choices = tuple((self.flow.point(c.globals),c.world_in,c.function,
                'CURRENT_CALLABLE_IS_SELECTED_FUNCTION_CREATION') for c in self.flow.captures
                if c.function.unit == connection.root.owner.unit and c.body.owner == access.owner.owner)
        if not choices:
            # A missing frame/capture is open; a module name is not its globals.
            request = _Read(connection.id,access.site,binding.interface,access.key,
                self.flow.point(access.value),access.world_in,None,None,access.world_in,binding.conditions)
            self._candidate(request,('CURRENT_FRAME_BUILTINS_SELECTION_IS_STANDARD',
                'FRAME_OR_FUNCTION_CREATION_BUILTINS_SELECTION_UNRESOLVED'))
            self.s.add(request.target,('external_value',None))
            return
        for globals_point,selected_at,function,condition in choices:
            request = _Read(connection.id,access.site,binding.interface,access.key,
                self.flow.point(access.value),selected_at,function,self._gp(globals_point),access.world_in,
                (*binding.conditions,condition))
            # May contents cannot establish key absence. This alternative is
            # always guarded, even when the current entries set is empty.
            self._candidate(request,('GLOBALS_BUILTINS_KEY_MISSING_AT_SELECTION',
                'CURRENT_FRAME_FALLBACK_BUILTINS_IS_STANDARD_MAP'))
            # The current frame can carry a nonstandard fallback map. Its
            # operand is not supplied by this interface, so keep that branch
            # open instead of treating the guarded standard candidate as full.
            self.s.add(request.target,('external_value',None))
            self._watch(globals_point,'GLOBALS_MAPPING',request)

    def _source(self, container, source, request, unknown_key):
        conditions = (*request.conditions,'GLOBALS_BUILTINS_ENTRY_SELECTED_AT_RECORDED_WORLD',
            'ENTRY_DEFINITION_REACHES_SELECTION_WORLD_WITHOUT_REPLACEMENT',
            'ALL_EARLIER_C2_LOOKUP_STEPS_MISS_AT_READ_WORLD',
            *(('UNKNOWN_SOURCE_KEY_SELECTS_BUILTINS_STRING',) if unknown_key else ()))
        row = BuiltinNamespaceSelection(request.connection,request.access,request.interface,
            request.selected_at,request.capturing_function,request.globals_reference,
            self._gp(container[2] if container[0]=='source_root_namespace' else container[1]),
            self.s.gp(source),request.read_world,tuple(dict.fromkeys(conditions)))
        if row not in self._selections:
            self._selections.add(row);self.selections.append(row)
        self._watch(source,'BUILTINS_VALUE',request,container,unknown_key)

    def content_write(self, container, key, source):
        if key is not None and key != ('str','__builtins__'):
            return
        for request in tuple(self.globals_readers.get(container,())):
            self._source(container,source,request,key is None)

    def _module(self, request, token):
        name = token[1]
        units = self.s.module_units.get(name,())
        for unit in units:
            key = unit,F.NamespaceKind.MODULE.value,0,request.key
            self.s.reads[key].add(request.target)
            for value in tuple(self.s.ns_values.get(key,())):
                self.s.add(request.target,value)
        if name == 'builtins':
            self._candidate(request,('SELECTED_BUILTINS_VALUE_IS_STANDARD_BUILTINS_MODULE',))
        elif not units:
            self.s.add(request.target,('external_value',None))

    def consume(self, point, token):
        for spec in tuple(self.watchers.get(point,())):
            identity = point,token,spec
            if identity in self._seen:
                continue
            self._seen.add(identity)
            mode,request,container,unknown_key = spec
            if mode == 'GLOBALS_MAPPING':
                if token[0] == 'container' and token[2] == 'MAPPING':
                    self.globals_readers[token].add(request)
                    for key in (('str','__builtins__'),None):
                        for source in tuple(self.s.containers.entries[token].get(key,())):
                            self._source(token,source,request,key is None)
                elif token[0]=='source_root_namespace':
                    self.globals_readers[token].add(request)
                    self.s.frames.watch_root_content(token,'__builtins__',self.content_write)
                else:
                    self.s.add(request.target,('external_value',None))
            elif token[0] == 'container':
                self.s.containers.access(token,('str',request.key),request.target,False)
            elif token[0] == 'module':
                # Read the module dictionary channel, never module __getattr__.
                self._module(request,token)
            else:
                self.s.add(request.target,('external_value',None))
