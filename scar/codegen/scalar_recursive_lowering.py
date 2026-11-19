"""Lower one typed scalar recursion region from original structured source."""

import ast
from dataclasses import dataclass
import operator

from scar.analysis import scalar_score_reuse_evidence as source
from scar.codegen.scalar_score_standalone import _literal
from scar.codegen.scalar_table_standalone import _integer
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.optimization.scalar_score_ir import _int64, _require
from scar.trace.code_snapshot import describe_code


_ARITHMETIC = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'multiply', ast.FloorDiv:'floor_div', ast.Mod:'remainder'}
_FOLD = {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.LShift:operator.lshift,ast.Pow:operator.pow}
_COMPARE = {ast.Eq:'==',ast.NotEq:'!=',ast.Lt:'<',ast.Gt:'>',ast.LtE:'<=',ast.GtE:'>='}
_TYPES = {
    'int':'Integer','bool':'bool','str':'std::string','float':'double','state':'State','move':'Move',
    'entry':'Pair','key':'Key','maybe_move':'std::optional<Move>','maybe_int':'std::optional<Integer>',
}


def _name(node):
    return node.id if type(node) is ast.Name else None


def _member(node, owner):
    return node.attr if type(node) is ast.Attribute and _name(node.value)==owner else None


def _literal_value(node, constants):
    if type(node) is ast.Name:
        return constants[node.id]
    if type(node) is ast.Tuple:
        return tuple(_literal_value(arg,constants) for arg in node.elts)
    if type(node) is ast.Dict:
        return {_literal_value(key,constants):_literal_value(value,constants)
                for key,value in zip(node.keys,node.values,strict=True)}
    if type(node) is ast.BinOp and type(node.op) in _FOLD:
        left,right = (_literal_value(arg,constants) for arg in (node.left,node.right))
        _require(_int64(left) and _int64(right) and (type(node.op) not in (ast.LShift,ast.Pow) or 0<=right<=63),
                 'constant arithmetic needs bounded signed integer operands')
        result = _FOLD[type(node.op)](left,right)
        _require(_int64(result), 'constant fold exceeds signed-64 scope')
        return result
    if type(node) is ast.Subscript:
        return _literal_value(node.value,constants)[_literal_value(node.slice,constants)]
    return ast.literal_eval(node)


def _defaults(module):
    constants = {}
    for stmt in module.body:
        if type(stmt) is not ast.Assign or len(stmt.targets)!=1:
            continue
        target, = stmt.targets
        binders = target.elts if type(target) is ast.Tuple else [target]
        names = [_name(node) for node in binders]
        if None in names:
            continue
        try:
            value = _literal_value(stmt.value,constants)
            values = value if type(target) is ast.Tuple else (value,)
            _require(type(values) is tuple and len(values)==len(names),'invalid static tuple binding')
        except (ValueError, TypeError, KeyError, IndexError):
            for name in names:
                constants.pop(name,None)
            continue
        constants.update(zip(names,values,strict=True))
    return constants


def _declared_fields(module, symbol):
    imports = {part.asname or part.name for stmt in module.body if type(stmt) is ast.ImportFrom
               and stmt.module=='collections' and not stmt.level for part in stmt.names if part.name=='namedtuple'}
    matches = [row.value for row in module.body if type(row) is ast.Assign and len(row.targets)==1
               and _name(row.targets[0])==symbol and type(row.value) is ast.Call
               and _name(row.value.func) in imports and not row.value.keywords and len(row.value.args)==2]
    match = source._one(matches,'state map needs one declared immutable pair constructor')
    text = ast.literal_eval(match.args[1])
    _require(type(text) is str,'expected fixed namedtuple fields')
    names = text.replace(',',' ').split()
    _require(len(names)==2 and len(set(names))==2,'state map default needs two exact field ports')
    return names


def _state_shape(module, function, owner, initial_pos):
    members, entry_classes = {}, {}
    clock_aliases = {part.asname or part.name for row in module.body if type(row) is ast.Import
                     for part in row.names if part.name=='time'}

    def declare(name, kind):
        _require(name is not None and name not in members or members.get(name)==kind,'inconsistent receiver state role')
        members[name]=kind

    for node in ast.walk(function):
        if type(node) is ast.AugAssign and (name:=_member(node.target,owner)) is not None:
            declare(name,'int')
        if (type(node) is ast.Call and type(node.func) is ast.Attribute and node.func.attr=='get'
                and (name:=_member(node.func.value,owner)) is not None):
            _require(not node.keywords and len(node.args) in (1,2),'unsupported state map lookup')
            if len(node.args)==1:
                declare(name,'move_map')
            else:
                default=node.args[1]
                _require(type(default) is ast.Call and _name(default.func) is not None and len(default.args)==2
                         and not default.keywords,'ordered numeric map needs an explicit pair default')
                declare(name,'entry_map')
                entry_classes[_name(default.func)]=_declared_fields(module,_name(default.func))
        if type(node) is ast.Compare and len(node.ops)==1:
            left,right=node.left,node.comparators[0]
            field=_member(right,owner)
            if type(node.ops[0]) is ast.In and _name(left)==initial_pos and field is not None:
                declare(field,'set')
            if (type(left) is ast.Call and type(left.func) is ast.Attribute and not left.args and not left.keywords
                    and _name(left.func.value) in clock_aliases and left.func.attr=='time' and field):
                declare(field,'float')
        if type(node) is ast.Delete:
            for target in node.targets:
                if type(target) is ast.Subscript and _member(target.value,owner) is not None and type(target.slice) is ast.Call:
                    for operand in ast.walk(target.slice):
                        if type(operand) is ast.Compare and len(operand.ops)==1 and type(operand.ops[0]) is ast.NotEq:
                            other=_member(operand.comparators[0],owner)
                            if other is not None:
                                declare(other,'state')
    _require(entry_classes and members,'recursive region lacks typed shared state')
    _require(len({tuple(names) for names in entry_classes.values()})==1,'ordered numeric map pair layouts differ')
    return members, entry_classes, clock_aliases


@dataclass(frozen=True)
class _Value:
    kind: str
    text: str


def _block(lines):
    return ['    '+line for line in lines]


def _sequenced(values, result):
    bindings=''.join(f'const auto& a{i} = {text}; ' for i,text in enumerate(values))
    return f'([&] {{ {bindings}return {result}; }}())'


class _Lower:
    def __init__(self,module,code,fn,plan,transition,leaves,constant_overrides):
        args=fn.args.posonlyargs+fn.args.args
        _require(len(args)==5 and not (fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.decorator_list)
                 and len(fn.args.defaults)==1 and type(fn.args.defaults[0]) is ast.Constant
                 and fn.args.defaults[0].value is False,'scoped recursion needs shared owner and four typed parameters')
        self.receiver=args[0].arg
        self.pos=args[1].arg
        self.depth=args[3].arg
        self.formals={arg.arg:_Value(kind,f'formal_{i}')
                      for i,(arg,kind) in enumerate(zip(args[1:],('state','int','int','bool'),strict=True))}
        self.method=fn.name
        members,self.entry_classes,self.clock_aliases=_state_shape(module,fn,self.receiver,self.pos)
        self.members={name:_Value(kind,f'world.field_{i}') for i,(name,kind) in enumerate(sorted(members.items()))}
        self.entry_fields=next(iter(self.entry_classes.values()))
        self.constants={**_defaults(module),**constant_overrides}
        self.used_constants={}
        self.constant_ports={}
        self.fields=plan['state_fields']
        self.outputs=[port['name'] for port in transition['outputs']]
        self.score_plan=transition['score_plan']
        self.sorted_plan=plan
        self.leaves=leaves
        self.locals={}
        self.calls={}
        self.serial=0
        self.nested=None
        self.exceptions=set()
        evidence=read_artifact(transition['evidence'])
        _require(leaves['witness'] is not None,'original recursive state has no admitted witness function')
        name=lambda code:code['qualname'].rsplit('.',1)[-1]
        self.move_roles=dict(factory=name(plan['factory']), transition=name(evidence['callee']['code']),
                             score=name(evidence['score']['code']), rotate=name(leaves['rotation']['code']),
                             rotation_option=leaves['rotation']['mode'], witness=name(leaves['witness']['code']))
        self.codes=code

    def unique(self,tag):
        self.serial+=1
        return f'{tag}_{self.serial}'

    def var(self,key,kind):
        if key in self.formals:
            target=self.formals[key]
            _require(target.kind==kind and target.kind!='state','original formal assignment changed parameter type')
            return target
        if key not in self.locals:
            self.locals[key]=_Value(kind,self.unique('local'))
        target=self.locals[key]
        _require(target.kind==kind,'local type changed across source statements: '+key)
        return target

    def _state_arg(self,pos,names):
        return [f'std::get<{self.outputs.index(name)}>({pos})' for name in names]

    def _truth(self,value):
        if value.kind=='bool':
            return value.text
        if value.kind=='int':
            return f'({value.text} != 0)'
        if value.kind in ('maybe_move','maybe_int'):
            return f'({value.text}.has_value())'
        raise ValueError('unbound truth protocol: '+value.kind)

    def test(self,node,scope):
        if type(node) is ast.BoolOp:
            join=' && ' if type(node.op) is ast.And else ' || '
            return '('+join.join(self.test(arg,scope) for arg in node.values)+')'
        if type(node) is ast.UnaryOp and type(node.op) is ast.Not:
            return f'!({self.test(node.operand,scope)})'
        return self._truth(self.expr(node,scope))

    def _coerce(self,value,kind):
        if value.kind==kind:
            return value.text
        if kind=='int' and value.kind=='bool':
            return f'Integer{{{value.text}}}'
        if value.kind=='none' and kind in ('maybe_move','maybe_int'):
            return f'{_TYPES[kind]}{{}}'
        if (value.kind,kind) in (('int','maybe_int'),('move','maybe_move')):
            return f'{_TYPES[kind]}{{{value.text}}}'
        if (value.kind,kind) in (('maybe_int','int'),('maybe_move','move')):
            return f'read({value.text})'
        raise ValueError(f'unsupported scalar conversion {value.kind} to {kind}')

    def expr(self,node,scope):
        if type(node) is ast.Constant:
            value=node.value
            if value is None:
                return _Value('none','std::nullopt')
            if type(value) is bool:
                return _Value('bool','true' if value else 'false')
            if _int64(value):
                return _Value('int',f'Integer{{{_integer(value)}}}')
            if type(value) is str:
                return _Value('str',f'std::string({_literal(value)})')
        if type(node) is ast.Name:
            name=node.id
            if name in scope:
                return scope[name]
            if name in self.formals:
                return self.formals[name]
            if name in self.locals:
                port=self.locals[name]
                return _Value(port.kind,f'read({port.text})')
            _require(_int64(self.constants.get(name)), 'unbound scoped constant '+name)
            self.used_constants[name]=self.constants[name]
            if name not in self.constant_ports:
                self.constant_ports[name]=f'global_{len(self.constant_ports)}'
            return _Value('int',f'world.{self.constant_ports[name]}')
        if type(node) is ast.Attribute:
            name=_member(node,self.receiver)
            if name in self.members:
                return self.members[name]
            base=self.expr(node.value,scope)
            if base.kind=='state' and node.attr in self.fields:
                kind=self.fields[node.attr]
                _require(kind in _TYPES,'unsupported aggregate field')
                return _Value(kind,self._state_arg(base.text,['field_'+node.attr])[0])
            if base.kind=='entry' and node.attr in self.entry_fields:
                return _Value('int',f'std::get<{self.entry_fields.index(node.attr)}>({base.text})')
        if type(node) is ast.Tuple:
            a,b=(self.expr(arg,scope) for arg in node.elts)
            _require((a.kind,b.kind)==('state','int'),'unsupported tuple expression')
            return _Value('key',f'Key{{{a.text}, {b.text}}}')
        if type(node) is ast.BinOp and type(node.op) in _ARITHMETIC:
            args=[self._coerce(self.expr(arg,scope),'int') for arg in (node.left,node.right)]
            op=_ARITHMETIC[type(node.op)]
            return _Value('int',_sequenced(args,f'{op}(a0, a1)'))
        if type(node) is ast.UnaryOp:
            if type(node.op) is ast.Not:
                return _Value('bool',self.test(node,scope))
            if type(node.op) is ast.USub:
                return _Value('int',f'sub(0, {self._coerce(self.expr(node.operand,scope),"int")})')
        if type(node) is ast.BoolOp:
            # This profile consumes its boolean-valued short-circuit result.
            _require(all(self.expr(arg,scope).kind=='bool' for arg in node.values),
                     'not all short-circuit operands produce an exact boolean')
            return _Value('bool',self.test(node,scope))
        if type(node) is ast.Compare:
            return _Value('bool',self.compare(node,scope))
        if type(node) is ast.IfExp:
            yes,no=[self.expr(child,scope) for child in (node.body,node.orelse)]
            _require(yes.kind==no.kind,'conditional returns incompatible ports')
            return _Value(yes.kind,f'({self.test(node.test,scope)} ? {yes.text} : {no.text})')
        if type(node) is ast.NamedExpr and _name(node.target):
            value=self.expr(node.value,scope)
            target=self.var(node.target.id,value.kind)
            _require(node.target.id in self.locals,'assignment expression must target a scoped local')
            return _Value(value.kind,f'({target.text} = {value.text}, read({target.text}))')
        if type(node) is ast.Call:
            return self.call(node,scope)
        raise ValueError('unsupported original recursive expression: '+ast.dump(node,include_attributes=False))

    def compare(self,node,scope):
        if len(node.ops)>1:
            _require(all(type(x) in (ast.Name,ast.Constant) for x in node.comparators[:-1]),
                     'comparison chain requires stable shared scalar operands')
            return '('+' && '.join(self.compare(ast.Compare(left=a,ops=[op],comparators=[b]),scope)
                for a,op,b in zip((node.left,*node.comparators[:-1]),node.ops,node.comparators,strict=True))+')'
        op=type(node.ops[0])
        left,right=[self.expr(arg,scope) for arg in (node.left,node.comparators[0])]
        if op in (ast.Is,ast.IsNot) and left.kind.startswith('maybe_') and right.kind=='none':
            return ('!' if op is ast.Is else '')+f'({left.text}.has_value())'
        if op in (ast.In,ast.NotIn):
            if (left.kind,right.kind)==('str','str'):
                check='contains(a0, a1)'
            elif (left.kind,right.kind)==('state','set'):
                check='a1.contains(a0)'
            else:
                raise ValueError('unsupported membership domain')
            return ('!' if op is ast.NotIn else '')+_sequenced((left.text,right.text),check)
        if left.kind=='maybe_int' and right.kind=='int':
            left=_Value('int',self._coerce(left,'int'))
        if right.kind=='maybe_int' and left.kind=='int':
            right=_Value('int',self._coerce(right,'int'))
        _require(op in _COMPARE and (left.kind==right.kind or {left.kind,right.kind}=={'int','float'}),
                 'unsupported typed comparison')
        return _sequenced((left.text,right.text),f'a0 {_COMPARE[op]} a1')

    def _fold_iter(self,node,name,scope):
        _require(type(node) is ast.GeneratorExp and len(node.generators)==1,'collection fold requires one generator')
        part=node.generators[0]
        key=_name(part.target)
        _require(key is not None and not(part.is_async or part.ifs),'unsupported short-circuit generator protocol')
        item=self.unique('item')
        when='predicate' if name=='all' else '!predicate'
        visitor=self._move_source(part.iter,scope)
        if visitor is not None:
            test=self.test(node.elt,{**scope,key:_Value('move',item)})
            expression=f'visit_records({visitor.text}, [&](const Move& {item}) {{ const bool predicate = {test}; return {when}; }})'
            return _Value('bool',expression if name=='all' else '!'+expression)
        text=self.expr(part.iter,scope)
        _require(text.kind=='str','short-circuit fold needs source record stream or exact text')
        test=self.test(node.elt,{**scope,key:_Value('str',f'std::string(1, {item})')})
        algorithm='all_of' if name=='all' else 'any_of'
        return _Value('bool',f'([&] {{ const auto& sequence = {text.text}; return std::{algorithm}(sequence.begin(), sequence.end(), [&](char {item}) {{ return {test}; }}); }}())')

    def _move_source(self,call,scope):
        if (type(call) is ast.Call and type(call.func) is ast.Attribute and not(call.args or call.keywords)
                and call.func.attr==self.move_roles['factory']):
            value=self.expr(call.func.value,scope)
            _require(value.kind=='state','move stream expects its original state')
            return value
        return None

    def call(self,node,scope):
        target=node.func
        alias=_name(target)
        if alias in ('any','all') and len(node.args)==1 and not node.keywords:
            return self._fold_iter(node.args[0],alias,scope)
        if alias in self.calls:
            expected,kind,fun=self.calls[alias]
            values=[self.expr(arg,scope) for arg in node.args]
            _require(not node.keywords and len(values)==len(expected),'source callable actual shape changed')
            actuals=[self._coerce(value,port) for value,port in zip(values,expected,strict=True)]
            return _Value(kind,_sequenced(actuals,f'{fun}('+', '.join(f'a{i}' for i in range(len(actuals)))+')'))
        if alias is not None:
            args=[self.expr(arg,scope) for arg in node.args]
            _require(not node.keywords,'unsupported builtin argument binding')
            if alias in ('min','max') and len(args)==2:
                return _Value('int',_sequenced([self._coerce(arg,'int') for arg in args],f'std::{alias}(a0, a1)'))
            if alias in ('abs','int') and len(args)==1:
                value=self._coerce(args[0],'int')
                return _Value('int',f'absolute({value})' if alias=='abs' else value)
            if alias=='len' and len(args)==1 and args[0].kind.endswith('_map'):
                return _Value('int',f'checked_size({args[0].text}.size())')
            if alias in self.entry_classes and len(args)==2:
                return _Value('entry','Pair{'+', '.join(self._coerce(arg,'int') for arg in args)+'}')
            raise ValueError('unknown scoped callable '+str(alias))
        _require(type(target) is ast.Attribute,'unknown recursive callable protocol')
        if _name(target.value) in self.clock_aliases and target.attr=='time' and not(node.args or node.keywords):
            return _Value('float','world.clock()')
        if _member(target,self.receiver)==self.method:
            args=[self.expr(arg,scope) for arg in node.args]
            _require(len(args)==3 and not node.keywords,'recursive port order changed')
            actual=[self._coerce(value,kind) for value,kind in zip(args,('state','int','int'),strict=True)]
            return _Value('int',_sequenced(actual,'recurse(world, table, a0, a1, a2, false)'))
        base=self.expr(target.value,scope)
        if target.attr=='get' and base.kind in ('move_map','entry_map'):
            args=[self.expr(arg,scope) for arg in node.args]
            _require(not node.keywords,'dictionary get must have its exact original positional ports')
            if base.kind=='move_map' and len(args)==1:
                key=self._coerce(args[0],'state')
                return _Value('maybe_move',f'find_move({base.text}, {key})')
            if base.kind=='entry_map' and len(args)==2:
                key,pair=(self._coerce(arg,kind) for arg,kind in zip(args,('key','entry'),strict=True))
                return _Value('entry',_sequenced((key,pair),f'{base.text}.get_or(a0, a1)'))
        _require(base.kind=='state','unsupported original receiver invocation')
        if target.attr==self.move_roles['rotate']:
            _require(not node.args and len(node.keywords)==1 and node.keywords[0].arg==self.move_roles['rotation_option'],
                     'rotation requires its exact original option binding')
            option=self._coerce(self.expr(node.keywords[0].value,scope),'bool')
            return _Value('state',_sequenced((base.text,option),'evaluate_rotation(a0, a1)'))
        if target.attr in (self.move_roles['transition'],self.move_roles['score']):
            _require(len(node.args)==1 and not node.keywords,'scalar state call needs one exact record')
            move=self._coerce(self.expr(node.args[0],scope),'move')
            operation=target.attr==self.move_roles['transition']
            name='state_move' if operation else 'state_score'
            return _Value('state' if operation else 'int',_sequenced((base.text,move),f'{name}(table, a0, a1)'))
        if target.attr==self.move_roles['witness']:
            _require(not(node.args or node.keywords),'witness receiver call changed')
            return _Value('maybe_move',f'{self.leaves["functions"]["witness"]}({base.text})')
        raise ValueError('unbound original state method '+target.attr)

    def _set(self,target,value):
        if _name(target):
            kind=self.locals[target.id].kind if target.id in self.locals else value.kind
            out=self.var(target.id,kind)
            return f'{out.text} = {self._coerce(value,kind)};'
        field=_member(target,self.receiver)
        if field in self.members:
            out=self.members[field]
            return f'{out.text} = {self._coerce(value,out.kind)};'
        if type(target) is ast.Subscript and (_member(target.value,self.receiver) in self.members):
            out=self.members[_member(target.value,self.receiver)]
            key_kind,value_kind={'move_map':('state','move'),'entry_map':('key','entry')}[out.kind]
            key=self._coerce(self.expr(target.slice,{}),key_kind)
            exact=self._coerce(value,value_kind)
            # Python evaluates RHS before the item receiver and key.
            return f'{{ const auto mapped_value = {exact}; const auto key = {key}; {out.text}.insert_or_assign(key, mapped_value); }}'
        raise ValueError('unsupported source assignment target '+ast.dump(target))

    def _delete(self,node):
        _require(type(node) is ast.Subscript and _member(node.value,self.receiver) in self.members,
                 'delete must preserve a known ordered map')
        out=self.members[_member(node.value,self.receiver)]
        key=node.slice
        _require(type(key) is ast.Call and _name(key.func)=='next' and len(key.args)==1 and not key.keywords,
                 'delete needs the original first-key iterator')
        arg=key.args[0]
        if type(arg) is ast.Call and _name(arg.func)=='iter' and len(arg.args)==1 and not arg.keywords:
            _require(_member(arg.args[0],self.receiver)==_member(node.value,self.receiver),
                     'eviction cannot borrow another map order')
            return f'if (!{out.text}.erase_first()) throw Decline{{}};'
        _require(type(arg) is ast.GeneratorExp and len(arg.generators)==1,'unsupported first-key filter')
        part=arg.generators[0]
        _require(not part.is_async and _member(part.iter,self.receiver)==_member(node.value,self.receiver)
                 and _name(arg.elt)==_name(part.target) and len(part.ifs)==1,'eviction iterator or current key changed')
        test=part.ifs[0]
        _require(type(test) is ast.Compare and len(test.ops)==1 and type(test.ops[0]) is ast.NotEq
                 and _name(test.left)==_name(part.target),'eviction must preserve the original excluded key')
        kept=self._coerce(self.expr(test.comparators[0],{}),'state')
        return f'if (!{out.text}.erase_first_except({kept})) throw Decline{{}};'

    def block(self,statements,scope=None,*,in_generator=False):
        scope={} if scope is None else scope
        lines=[]
        for statement in statements:
            node=statement
            if type(node) is ast.Expr and type(node.value) is ast.Constant and type(node.value.value) is str:
                continue
            if type(node) is ast.Assign and len(node.targets)==1:
                target=node.targets[0]
                if type(node.value) is ast.Lambda:
                    args=node.value.args
                    _require(_name(target) and len(args.args)==1 and not(args.posonlyargs or args.kwonlyargs or args.defaults or args.vararg or args.kwarg),
                             'only a scoped one-scalar lambda has a region lowering')
                    formal=self.unique('lambda_arg')
                    value=self.expr(node.value.body,{**scope,args.args[0].arg:_Value('int',formal)})
                    symbol=self.unique('lambda')
                    self.calls[target.id]=('int',),value.kind,symbol
                    lines.append(f'const auto {symbol} = [&](Integer {formal}) {{ return {value.text}; }};')
                elif type(target) is ast.Tuple and type(node.value) is ast.Tuple:
                    staged=[]
                    for dest,expr in zip(target.elts,node.value.elts,strict=True):
                        value=self.expr(expr,scope)
                        name=self.unique('rhs')
                        lines.append(f'const auto {name} = {value.text};')
                        staged.append(self._set(dest,_Value(value.kind,name)))
                    lines.extend(staged)
                else:
                    lines.append(self._set(target,self.expr(node.value,scope)))
            elif type(node) is ast.AugAssign:
                left,right=(self.expr(part,scope) for part in (node.target,node.value))
                if (left.kind,right.kind)==('bool','bool') and type(node.op) is ast.BitOr:
                    value=_Value('bool',_sequenced((left.text,right.text),'bool(a0 | a1)'))
                else:
                    op=_ARITHMETIC[type(node.op)]
                    value=_Value('int',_sequenced((self._coerce(left,'int'),self._coerce(right,'int')),f'{op}(a0, a1)'))
                lines.append(self._set(node.target,value))
            elif type(node) is ast.If:
                lines.append(f'if ({self.test(node.test,scope)}) {{')
                lines.extend(_block(self.block(node.body,scope,in_generator=in_generator)))
                if node.orelse:
                    lines.append('} else {')
                    lines.extend(_block(self.block(node.orelse,scope,in_generator=in_generator)))
                lines.append('}')
            elif type(node) is ast.FunctionDef:
                _require(self.nested is None and not(node.decorator_list or node.args.args or node.args.posonlyargs
                         or node.args.kwonlyargs or node.args.vararg or node.args.kwarg),'generator must be a private lexical declaration')
                symbol=self.unique('stream')
                self.nested=node,symbol
                self.calls[node.name]=(),'stream',symbol
                lines.append('@SCAR_NESTED_GENERATOR@')
            elif type(node) is ast.For:
                names=source._tuple_names(node.target)
                obj=self.expr(node.iter,scope)
                _require(names and obj.kind=='stream' and not node.orelse and not in_generator,
                         'recursive loop must consume its original private ordered stream')
                stream=self.unique('stream_instance')
                lines.append(f'for (auto {stream} = {obj.text}; {stream}.advance();) {{')
                for i,kind in enumerate(('maybe_int','maybe_move')):
                    lines.append('    '+self._set(ast.Name(id=names[i]),_Value(kind,f'std::get<{i}>({stream}.current())')))
                lines.extend(_block(self.block(node.body,scope)))
                lines.append('}')
            elif type(node) is ast.Expr and in_generator:
                emit=node.value
                if type(emit) is ast.Yield and type(emit.value) is ast.Tuple:
                    sent=[self._coerce(self.expr(expr,scope),kind)
                          for expr,kind in zip(emit.value.elts,('maybe_int','maybe_move'),strict=True)]
                    lines.append('co_yield Ranked{'+', '.join(sent)+'};')
                elif type(emit) is ast.YieldFrom and source._span(emit.value)==self.sorted_plan['sorted_site']['position']:
                    position=self.formals[self.pos].text
                    names=[port['name'] for port in self.sorted_plan['inputs']]
                    depth=self._coerce(self.expr(ast.Name(id=self.depth),scope),'int')
                    symbol=self.sorted_plan['predicate']['threshold']
                    threshold=self._coerce(self.expr(ast.Name(id=symbol),scope),'int')
                    args=', '.join(('table',threshold,depth,*self._state_arg(position,names)))
                    lines.append(f'for (const auto& row : ordered({args})) {{')
                    lines.append('    co_yield Ranked{std::get<0>(row), std::get<1>(row)};')
                    lines.append('}')
                else:
                    raise ValueError('unsupported original generator atom')
            elif type(node) is ast.Return:
                _require(not in_generator,'generator returning an external value is outside this scope')
                lines.append(f'return {self._coerce(self.expr(node.value,scope),"int")};')
            elif type(node) is ast.Raise:
                name=_name(node.exc)
                _require(not in_generator and name and not node.cause,'unsupported source exception protocol')
                self.exceptions.add(name)
                lines.append('throw SourceStop{};')
            elif type(node) is ast.Break:
                lines.append('break;')
            elif type(node) is ast.Delete:
                lines.extend(self._delete(target) for target in node.targets)
            else:
                raise ValueError('unsupported original recursive statement '+type(node).__name__)
        return lines

    @staticmethod
    def declarations(slots):
        return ['std::optional<'+_TYPES[value.kind]+'> '+value.text+';' for value in slots.values()]

    def generator(self):
        node,symbol=self.nested
        original=source._one((child for child in self.codes.co_consts if hasattr(child,'co_freevars')
                              and child.co_name==node.name and child.co_firstlineno==node.lineno),
                             'expected one code-bound original generator')
        outer=self.locals
        scope={name:_Value(value.kind,f'read({value.text})') for name,value in outer.items()
               if name in original.co_freevars}
        self.locals={}
        try:
            body=self.block(node.body,scope,in_generator=True)
            storage=self.declarations(self.locals)
        finally:
            self.locals=outer
        return [f'const auto {symbol} = [&]() -> Stream<Ranked> {{',*_block(storage),*_block(body),
                '    co_return;','};']


def lower(region, sorted_plan, transition, scorer, leaves, *, constants=None):
    module,codes=source._compile(region['source'])
    owner=region['region']['owner']
    code=codes[owner['qualname']]
    _require(describe_code(code)==owner and region['contract']['whole_recursive_effects_closed'] is False,
             'lowering requires exact original recursive scope and explicit effects boundary')
    function=_definition(module,code.co_qualname)
    lowerer=_Lower(module,code,function,sorted_plan,transition,leaves,constants or {})
    body=lowerer.block(function.body)
    generator=lowerer.generator()
    body=[item for row in body for item in (generator if row=='@SCAR_NESTED_GENERATOR@' else (row,))]
    _require(lowerer.exceptions,'source exception boundary is missing')
    for name in lowerer.exceptions:
        definition=source._one((row for row in module.body if type(row) is ast.ClassDef and row.name==name),
                               'explicit exception is not its original declaration')
        _require(len(definition.bases)==1 and _name(definition.bases[0])=='Exception'
                 and not(definition.keywords or definition.decorator_list)
                 and all(type(row) is ast.Pass for row in definition.body),'exception construction has open effects')
    fields={name:{'type':value.kind,'field':value.text.split('.')[1]} for name,value in lowerer.members.items()}
    scorer_names=[port['name'] for port in scorer['inputs'] if port['name'].startswith('field_')]
    state_input=[port['name'] for port in transition['inputs'] if port['name'].startswith('field_')]
    score_args=', '.join(('table',*lowerer._state_arg('position',scorer_names),
                         *(f'std::get<{i}>(record)' for i,_ in enumerate(sorted_plan['record_types']))))
    transition_args=', '.join(('table',*lowerer._state_arg('position',state_input),
                         *(f'std::get<{i}>(record)' for i,_ in enumerate(sorted_plan['record_types']))))
    declaration=lowerer.declarations(lowerer.locals)
    world_fields={'entry_map':'EntryMap','move_map':'MoveMap','set':'History','int':'Integer','float':'double','state':'State'}
    storage=[f'{world_fields[port["type"]]} {port["field"]}{{}};' for port in fields.values()]
    storage.extend(f'Integer {slot}{{{_integer(lowerer.used_constants[name])}}};'
                   for name,slot in lowerer.constant_ports.items())
    formal=', '.join((f'const State& {lowerer.formals[lowerer.pos].text}',*(
                _TYPES[value.kind]+' '+value.text for name,value in lowerer.formals.items() if name!=lowerer.pos)))
    cpp='\n'.join(['struct World {',*_block(storage),'    double clock() const { return exact_clock(); }','};',
        f'Integer state_score(const Table& table, const State& position, const Move& record) {{ return evaluate_score({score_args}); }}',
        f'State state_move(const Table& table, const State& position, const Move& record) {{ return evaluate_transition({transition_args}); }}',
        f'Integer recurse(World& world, const Table& table, {formal}) {{',
        '    RecursionLimit frame;',*_block(declaration),*_block(body),'}'])
    return dict(body=cpp,world=fields,constants=lowerer.used_constants,constant_ports=lowerer.constant_ports,
                exceptions=sorted(lowerer.exceptions),
                formals=region['region']['formals'],source=region['source'])
