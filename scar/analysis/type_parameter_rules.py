"""Type parameter rules."""

from collections import defaultdict
from scar.ir.program import source_productions as P
from scar.analysis.source_core_interfaces import REF as _REF
from scar.ir.program.signature_patterns import CountBinding
from scar.ir.program.structured import PythonProtocolKind as K
from hashlib import sha256
import json
from pathlib import Path
from scar.analysis.source_role_materials import source_role_catalog
from scar.analysis.source_syntax import _schema_and_digests
from scar.ir.record_codec import encode


# Type-parameter scopes and lazy requests over original F2/F3 references.
def _require(condition,path,detail):
    if not condition:
        from scar.analysis.source_productions import ProductionError
        raise ProductionError('TYPE_PARAMETER_INTERFACE',str(path),detail)


class TypeParameterInterfaceIndex:
    def __init__(self,source):
        self.source=source
        self.scopes={scope.owner.occurrence:i for i,scope in enumerate(source.context.material.scopes) if scope.kind=='TYPE_PARAMETERS'}
        self.ports=defaultdict(list)
        for i,port in enumerate(source.frontend.roles.ports):self.ports[port.site].append(i)
        self._specs={}

    def lazy(self,site,*,allow_starred=False):
        _,value=self.source.roles.locate(site)
        if value.kind!='NODE':return None
        context=self.source.context.context_at_slot(site).state
        scope=self.source.context.material.scopes[context.environment]
        _require(scope.kind in ('TYPE_PARAMETER_EVALUATOR','TYPE_ALIAS') and scope.anchor==site,site,
            'lazy expression requires its original type-parameter/alias evaluator scope')
        starred=self.source.context.syntax.nodes[value.reference].constructor=='Starred'
        _require(not starred or allow_starred,site,'starred evaluator is permitted only for TypeVarTuple defaults')
        return P.LazyExpressionInterface(site,context.environment,
            tuple(self.source.deferred.captures[context.environment]),starred)

    def spec(self,occurrence):
        if occurrence in self._specs:return self._specs[occurrence]
        roles=self.source.roles;nodes=self.source.context.syntax.nodes
        kind=nodes[occurrence].constructor
        kinds={'TypeVar':P.TypeParameterKind.TYPE_VAR,'TypeVarTuple':P.TypeParameterKind.TYPE_VAR_TUPLE,'ParamSpec':P.TypeParameterKind.PARAM_SPEC}
        _require(kind in kinds,occurrence,'expected an original type-parameter constructor')
        name=P.SlotRef(occurrence,'FIELD',roles.fields[kind]['name'])
        scope=self.source.context.entry_context(occurrence).environment
        data=self.source.frontend.bindings;decl=self.source.frontend.declarations
        ports=[]
        for p in self.ports[name]:
            port=data.ports[p]
            if port.declref is None or port.resolution is None:continue
            binding=data.resolutions[port.resolution].binding
            if 'TYPE_PARAMETER_INTERFACE' in decl.contributions[port.declref].operations and binding is not None and data.bindings[binding].owner==scope:
                ports.append(p)
        _require(len(ports)==1,name,'type parameter needs one original declaration/resolution interface')
        bound=self.lazy(P.SlotRef(occurrence,'FIELD',roles.fields[kind]['bound'])) if kind=='TypeVar' else None
        default=self.lazy(P.SlotRef(occurrence,'FIELD',roles.fields[kind]['default_value']),allow_starred=kind=='TypeVarTuple')
        constraints=bound is not None and nodes[roles.locate(bound.source)[1].reference].constructor=='Tuple'
        result=P.TypeParameterSpec(name,kinds[kind],ports[0],bound,default,constraints)
        self._specs[occurrence]=result
        return result

    def environment(self,owner,world,exception_abi):
        _require(owner in self.scopes,owner,'definition has no original type-parameter scope')
        roles=self.source.roles;nodes=self.source.context.syntax.nodes
        parameters=tuple(self.spec(item.reference) for item in roles.field(owner,'type_params').items)
        _require(bool(parameters),owner,'type-parameter factory requires nonempty original parameters')
        seen=False
        for p in parameters:
            _require(not seen or p.default is not None,p.name,'non-default type parameter follows a default')
            seen|=p.default is not None
        inputs=(tuple(p.default_source for p in self.source.deferred.parameters(owner) if p.default_source is not None)
            if nodes[owner].constructor in ('FunctionDef','AsyncFunctionDef') else ())
        scope=self.scopes[owner]
        return P.TypeParameterEnvironment(scope,tuple(self.source.deferred.captures[scope]),parameters,inputs,world,exception_abi,
            self.source.capture_paths.owned_cells(scope))


# Generic factory activation and sequential type-parameter creation.
#
# Lazy expressions are captured as original evaluator interfaces; their bodies
# are not executed while creating TypeVar/TypeVarTuple/ParamSpec objects.
def lazy_factory(b,region,expression,world):
    owner=P.ProductionOccurrence(expression.source.occurrence)
    globals_ref,world=b.environment(region,P.GlobalsEnvironment(b.rules.source.context.material.root_context.environment),world,source=owner)
    captures,world=b.closure_cells(region,expression.captures,world,source=owner)
    b.slots[expression.source]=P.ProductionDispositionKind.DEFERRED_CHILD
    from scar.analysis.source_callable_rules import build_request_body
    request=build_request_body(b,region,expression)
    return b.emit(region,P.BuildLazyExpression(expression,b.rules.exception_abi,request.id),
        (globals_ref,*captures,world),(_REF,b.world),source=expression.source)


def emit_type_parameters(b,region,world,environment):
    values=[]
    for parameter in environment.parameters:
        owner=P.ProductionOccurrence(parameter.name.occurrence)
        for name in b.role.fields[b.rules.source.context.syntax.nodes[parameter.name.occurrence].constructor]:
            b.auxiliary(parameter.name.occurrence,name,P.ProductionDispositionKind.METADATA)
        b.slots[parameter.name]=P.ProductionDispositionKind.ROLE_PORT
        bound=()
        if parameter.bound is not None:
            evaluator,world=lazy_factory(b,region,parameter.bound,world)
            bound=(evaluator,)
        value,world=b.emit(region,P.CreateTypeParameter(parameter,b.rules.exception_abi),(*bound,world),(_REF,b.world),source=owner)
        if parameter.default is not None:
            evaluator,world=lazy_factory(b,region,parameter.default,world)
            world,=b.emit(region,P.SetTypeParameterDefault(parameter,b.rules.exception_abi),
                (value,evaluator,world),(b.world,),source=owner)
        world,=b.emit(region,P.BindTypeParameter(parameter,b.rules.exception_abi),(value,world),(b.world,),source=owner)
        values.append(value)
    result,world=b.protocol(region,K.BUILD_CALL_POSITIONALS,tuple(values),world,
        (CountBinding('item_count',len(values)),),selectors=(b.site('type_params'),))
    return result,world


def enter_factory(b,root,world,inputs):
    environment=b.rules.source.type_parameters.environment(b.occurrence,b.world,b.rules.exception_abi)
    captures,world=b.closure_cells(root,environment.captures,world)
    arguments=(*captures,*inputs,world)
    region=b.region(b.site('type_params'),(*(_REF for _ in range(len(arguments)-1)),b.world),root,P.ProductionRegionKind.TYPE_PARAMETERS)
    result=b.emit(root,P.InvokeTypeParameterFactory(P.ProductionRegionCall(region.id,arguments),environment),(),(_REF,b.world))
    active,=b.emit(region,P.EnterTypeParameters(environment),tuple(b.parameter(region,i) for i in range(len(region.parameters))),
        (b.world,),may_raise=False)
    parameters,active=emit_type_parameters(b,region,active,environment)
    passed=tuple(b.parameter(region,len(captures)+i) for i in range(len(inputs)))
    b.mark('type_params',P.ProductionDispositionKind.DEFERRED_CHILD)
    return result,region,parameters,passed,active


# F4-A input-shape signatures over the original official AST and B rules.
#
# This catalog partitions syntax forms. It does not infer context, phases,
# execution order, operations or effects, and never parses or executes a target.
_VERSION = 'F4A_SOURCE_INPUT_SHAPE_V1'
_SCHEMA = '8f5cd1f9759dfeb2b1ccaf1a10c4483bad994b674241557e1cdd464be1c4831a'
# Original official ASDL bases; AST contains the seven auxiliary constructors.
_SORTS = (('mod', 4), ('stmt', 28), ('expr', 29), ('expr_context', 3),
    ('boolop', 2), ('operator', 13), ('unaryop', 4), ('cmpop', 10),
    ('excepthandler', 1), ('pattern', 8), ('type_param', 3), ('type_ignore', 1), ('AST', 7))
_FLAGS = dict(authority='SOURCE_INPUT_SHAPE_CLOSURE', context_inference='NOT_COMPUTED',
    phase_inference='NOT_COMPUTED', execution_semantics='NOT_CLAIMED', graph_realization='NOT_PERFORMED')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode('ascii')


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def source_type_catalog() -> dict:
    """Return a fresh catalog; type payloads are unchanged F2 AstTypeSchemas.

    FIELD rows retain their original type. ATTRIBUTE rows have type=None:
    the captured official schema declares their names, not _field_types.
    role_rule references the original B rule, including its selector and role.
    """
    schema = _schema_and_digests()[0]
    schema_sha = _digest(encode(schema))
    roles = source_role_catalog()
    if (schema_sha != _SCHEMA or roles['f2_schema_sha256'] != schema_sha
            or _digest({key: value for key, value in roles.items() if key != 'content_sha256'}) != roles['content_sha256']):
        raise ValueError('input type catalog requires its reviewed original F2/B schema')
    constructors, slots = [], []
    sorts = [dict(name=name, constructors=[]) for name, _ in _SORTS]
    sort_by_name = {name: index for index, (name, _) in enumerate(_SORTS)}
    role_constructors = {row['name']: row for row in roles['constructors']}
    if len(role_constructors) != len(roles['constructors']):
        raise ValueError('B constructor references are not unique')
    consumed_rules = set()
    for schema_index, cls in enumerate(schema):
        if cls.abstract:
            continue
        if len(cls.bases) != 1 or cls.bases[0] not in sort_by_name or cls.name not in role_constructors:
            raise ValueError('official constructor has no closed input-sort partition')
        if tuple(field.name for field in cls.field_types) != cls.fields:
            raise ValueError('F2 ordered field types differ from original field declarations')
        index, sort = len(constructors), sort_by_name[cls.bases[0]]
        original, groups = role_constructors[cls.name], []
        for group, names, role_group in (('FIELD', cls.fields, 'fields'), ('ATTRIBUTE', cls.attributes, 'attributes')):
            references = original[role_group]
            if len(references) != len(names):
                raise ValueError('F2/B original field denominator differs')
            ids = []
            for ordinal, name in enumerate(names):
                rule_id = references[ordinal]
                if type(rule_id) is not int or not 0 <= rule_id < len(roles['rules']) or rule_id in consumed_rules:
                    raise ValueError('B original rule reference is invalid or duplicated')
                rule = roles['rules'][rule_id]
                if (rule['rule'], rule['constructor'], rule['group'], rule['ordinal'], rule['name']) != (rule_id, cls.name, group, ordinal, name):
                    raise ValueError('B rule does not identify the original F2 field')
                consumed_rules.add(rule_id);ids.append(len(slots))
                slots.append(dict(constructor=index, group=group, ordinal=ordinal, role_rule=rule_id,
                    type=encode(cls.field_types[ordinal].type) if group == 'FIELD' else None))
            groups.append(ids)
        constructors.append(dict(name=cls.name, source_schema_index=schema_index, sort=sort,
            fields=groups[0], attributes=groups[1]))
        sorts[sort]['constructors'].append(index)
    if (len(constructors) != 113 or len(slots) != 484
            or len(role_constructors) != 113 or consumed_rules != set(range(len(roles['rules'])))
            or len(roles['rules']) != 484
            or any(len(row['constructors']) != size for row, (_, size) in zip(sorts, _SORTS))
            or sum(row['group'] == 'FIELD' for row in slots) != 196):
        raise ValueError('input-sort or original-field partition is not closed')
    payload = dict(schema='scar.source-type-catalog', schema_version=1, rule_version=_VERSION,
        f2_schema_sha256=schema_sha, role_catalog_sha256=roles['content_sha256'],
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        sorts=sorts, constructors=constructors, slots=slots, **_FLAGS)
    return {**payload, 'content_sha256': _digest(payload)}


def read_source_type_catalog(document) -> dict:
    """Fresh exact wire comparison, without granting execution semantics."""
    expected = source_type_catalog()
    if type(document) is not dict or _canonical(document) != _canonical(expected):
        raise ValueError('input type catalog differs from its reviewed original schema/rules')
    return expected


__all__ = [
    'source_type_catalog',
    'read_source_type_catalog',
    'TypeParameterInterfaceIndex',
    'lazy_factory',
    'emit_type_parameters',
    'enter_factory',
]
