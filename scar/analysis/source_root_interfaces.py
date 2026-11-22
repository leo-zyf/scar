"""Root and class namespace interfaces over admitted source facts."""
from collections import defaultdict
from scar.ir.program import source_productions as P
from scar.ir.program.protocol_signatures import ExceptionABI


FIELDS={'Module':('body','type_ignores'),'Interactive':('body',),'Expression':('body',)}


_CLASS_NAMES=frozenset(('__name__','__module__','__qualname__','__firstlineno__','__type_params__',
    '__doc__','__static_attributes__','__classdictcell__','__classcell__','__annotate_func__'))


def _require(condition,path,detail,*,code='ROOT_INTERFACE'):
    if not condition:
        from scar.analysis.source_productions import ProductionError
        raise ProductionError(code,str(path),detail)


def clean_docstring(value):
    """Pinned _PyCompile_CleanDoc, retaining blank lines and ASCII-space rules."""
    value=value.expandtabs()
    value.encode('utf-8')  # the compiler requires a valid UTF-8 representation
    lines=value.split('\n')
    nonblank=[len(line)-len(line.lstrip(' ')) for line in lines[1:] if line.lstrip(' ')]
    margin=min(nonblank,default=0)
    return '\n'.join((lines[0].lstrip(' '),*(line[min(margin,len(line)-len(line.lstrip(' '))):] for line in lines[1:])))


class RootInterfaceIndex:
    def __init__(self,source):
        self.source=source;self.syntax=source.context.syntax;self.root=self.syntax.root
        self.scope=source.context.material.root_context.environment
        self.kind=self.syntax.nodes[self.root].constructor
        annotations = source.scope_annotations
        self.annotations = annotations.by_scope[self.scope]
        self.publications = tuple(publication for occurrence in self.annotations
            if (publication := annotations.publication(occurrence, ExceptionABI.EXCEPTION_WORLD)) is not None)
        self.deferred=any(p.publication is P.AnnotationPublicationKind.DEFERRED for p in self.publications)
        self.future=False
        if self.kind in ('Module','Interactive'):
            for ordinal,value in enumerate(source.roles.field(self.root,'body').items):
                owner=value.reference
                if ordinal==0 and self.syntax.nodes[owner].constructor=='Expr':
                    child=source.roles.field(owner,'value').reference
                    if self.syntax.nodes[child].constructor=='Constant' and type(source.roles.field(child,'value').scalar.to_python()) is str:continue
                if self.syntax.nodes[owner].constructor!='ImportFrom':break
                module=source.roles.field(owner,'module').scalar.to_python()
                if module!='__future__' or source.roles.field(owner,'level').scalar.to_python()!=0:break
                self.future |= any(source.roles.field(v.reference,'name').scalar.to_python()=='annotations'
                    for v in source.roles.field(owner,'names').items)
        for kind,fields in FIELDS.items():
            _require(tuple(source.roles.fields[kind])==fields,kind,'original root field grammar differs')

    def interface(self,world,abi):
        _require(self.kind in FIELDS and self.syntax.mode=={'Module':'exec','Interactive':'single','Expression':'eval'}[self.kind],
            self.root,'root constructor and original parser mode differ')
        _require(self.source.context.material.scopes[self.scope].kind=={
            'Module':'ROOT_EXEC','Interactive':'ROOT_INTERACTIVE','Expression':'ROOT_EVAL'}[self.kind],
            self.root,'root requires its original external namespace scope')
        return P.RootBodyInterface(P.ProductionOccurrence(self.root),self.scope,world,abi)

    def store_kind(self,name):
        """Implicit codegen_nameop honors an explicit root global directive."""
        declarations=self.source.frontend.declarations;bindings=self.source.frontend.bindings
        explicit=any(row.scope==self.scope and row.classification=='GLOBAL_EXPLICIT' and row.name.kind=='SOURCE'
            and declarations.names[row.name.index].value==name for row in bindings.resolutions)
        return P.NamespaceActionKind.STORE_GLOBAL if explicit else P.NamespaceActionKind.STORE_NAME

    def interactive_expression(self,occurrence):
        context=self.source.context.entry_context(occurrence)
        return self.kind=='Interactive' and context.owner.kind=='ROOT' and context.environment==self.scope


def _require_class(condition,path,detail):
    _require(condition,path,detail,code='CLASS_NAMESPACE_INTERFACE')


class ClassNamespaceIndex:
    def __init__(self,source):
        self.source=source;self.scopes=source.context.material.scopes;self.nodes=source.context.syntax.nodes
        self.resolutions={(r.scope,source.frontend.namespace.name(r.name)):i for i,r in enumerate(source.frontend.bindings.resolutions)}
        self._qualnames={};self._attributes=None
        self.annotations = source.scope_annotations.by_scope

    def access(self,interface,name,action):
        _require_class(name in _CLASS_NAMES and action in (P.BindingActionKind.READ,P.BindingActionKind.WRITE)
            and (action is P.BindingActionKind.WRITE or name=='__name__'),name,'unsupported implicit class name/access')
        return P.AccessClassName(interface,name,action,self.resolutions.get((interface.scope,name)))

    def protocol(self,interface,name):
        bindings=self.source.frontend.bindings.bindings
        found=[i for i in interface.protocol_bindings if self.source.frontend.namespace.name(bindings[i].name)==name]
        _require_class(len(found)<=1,name,'duplicate original class protocol binding')
        if not found:return None
        index=found[0]
        if name=='__conditional_annotations__':
            active=any(d.scope==interface.scope and d.rule=='CONDITIONAL_ANNOTATION_REQUIREMENT' for d in self.source.frontend.bindings.implicit)
        else:active=bindings[index].cell_required
        return index if active else None

    def _name_global(self,owner):
        kind=self.nodes[owner].constructor
        if kind not in ('ClassDef','FunctionDef','AsyncFunctionDef'):return False
        site=P.SlotRef(owner,'FIELD',self.source.roles.fields[kind]['name'])
        bindings = self.source.frontend.bindings
        for ordinal in self.source.role_ports(site):
            resolution = bindings.ports[ordinal].resolution
            if resolution is not None:
                return bindings.resolutions[resolution].classification == 'GLOBAL_EXPLICIT'
        return False

    def qualname(self,scope):
        if scope in self._qualnames:return self._qualnames[scope]
        current=self.scopes[scope]
        if current.parent is None:return ''
        if current.kind=='TYPE_PARAMETERS':return self.qualname(current.parent)
        owner=current.owner.occurrence;kind=self.nodes[owner].constructor
        _require_class(kind in ('FunctionDef','AsyncFunctionDef','ClassDef','Lambda'),scope,
            'class qualname ancestry must consist of original declaration scopes')
        name='<lambda>' if kind=='Lambda' else self.source.roles.field(owner,'name').scalar.to_python()
        parent=current.parent
        if self.scopes[parent].kind=='TYPE_PARAMETERS':parent=self.scopes[parent].parent
        prefix='' if self._name_global(owner) else self.qualname(parent)
        if prefix and self.scopes[parent].kind in ('FUNCTION','LAMBDA'):prefix+='.<locals>'
        result=prefix+'.'+name if prefix else name
        self._qualnames[scope]=result
        return result

    def first_line(self,scope):
        owner=self.scopes[scope].owner.occurrence
        decorators=self.source.roles.field(owner,'decorator_list').items
        node=self.nodes[decorators[0].reference if decorators else owner]
        schema=next(row for row in self.source.context.syntax.profile.ast_schema if row.name==node.constructor)
        return node.attributes[schema.attributes.index('lineno')].scalar.to_python()

    def attributes(self,scope):
        if self._attributes is None:
            always=defaultdict(set);asserted=defaultdict(set);roles=self.source.roles
            for owner,node in enumerate(self.nodes):
                if node.constructor!='Attribute' or roles.context(owner)!='Store':continue
                base=roles.field(owner,'value').reference
                if self.nodes[base].constructor!='Name' or roles.field(base,'id').scalar.to_python()!='self':continue
                context=self.source.context.entry_context(owner)
                if self.source.context.material.regions[context.region].eligibility!='ELIGIBLE':continue
                parent=roles.parents.get(owner)
                if parent is not None and (self.nodes[parent.occurrence].constructor=='AugAssign' or (
                    self.nodes[parent.occurrence].constructor=='AnnAssign' and roles.field(parent.occurrence,'value').kind!='NODE')):continue
                current=context.environment
                while self.source.frontend.topology.scopes[current].disposition=='INLINED':current=self.scopes[current].parent
                current=self.scopes[current].parent  # compiler stack excludes the current code unit
                while current is not None and self.scopes[current].kind!='CLASS':current=self.scopes[current].parent
                if current is None:continue
                conditional=False;original=owner
                while original in roles.parents:
                    original=roles.parents[original].occurrence
                    conditional |= self.nodes[original].constructor=='Assert'
                name=roles.field(owner,'attr').scalar.to_python()
                (asserted if conditional else always)[current].add(name)
            self._attributes={i:(tuple(sorted(always[i])),tuple(sorted(always[i]|asserted[i])))
                for i,s in enumerate(self.scopes) if s.kind=='CLASS'}
        return self._attributes[scope]

    def deferred_annotations(self,scope):
        if self.source.roots.future:return False
        return any(self.source.scope_annotations.publication(i, ExceptionABI.EXCEPTION_WORLD) is not None
            for i in self.annotations[scope])
