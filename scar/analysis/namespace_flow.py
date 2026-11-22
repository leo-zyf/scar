"""Namespace flow."""

from collections import defaultdict
from scar.analysis.namespace_values import BuiltinNamespaceFlow
from scar.ir.program import source_productions as P, normal_graph as F, normal_graph as G, source_frame_bindings as B
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.source_exec_roots import ExecNamespaceKind
from hashlib import sha256
from scar.analysis.source_templates import _json
from scar.ir.record_codec import encode


# Admit namespace content constraints without turning exec dicts into modules.
#
# The published overlay retains lookup order and worlds. Transfer-domain facts
# remain conditional may alternatives; no singleton is a constant proof here.
class NamespaceFlow:
    def __init__(self, solver, graph):
        self.s,self.graph = solver,graph
        self.accesses = defaultdict(list)
        self.bindings = defaultdict(list)
        self.references = defaultdict(list)
        self.watchers = defaultdict(set)
        self.admitted = set(); self.connections = {}; self.mapping_requests = set()
        self.globals_by_owner = defaultdict(set)
        for access in graph.records('accesses'):
            self.accesses[access.owner.owner].append(access)
        for binding in graph.records('bindings'):
            self.bindings[binding.connection,binding.interface].append(binding)
        for reference in graph.records('globals_bindings'):
            self.references[reference.connection].append(reference)
        self.captures = tuple(graph.records('captures'))
        self.builtins = BuiltinNamespaceFlow(self)

    def point(self, endpoint):
        return self.s.original_point(endpoint)

    def formal(self, connection, index):
        root = connection.root
        return self.point(GlobalPort(root.owner.unit,G.Port(root.template,root.region,G.PortKind.PARAMETER,index)))

    def admit_connection(self, connection):
        if connection.id in self.connections:
            return
        self.connections[connection.id]=connection
        globals_point,locals_point = self.formal(connection,0),self.formal(connection,1)
        if type(connection.globals.value) is GlobalPort:
            self.s.edge(self.point(connection.globals.value),globals_point)
        else:
            if not self.s.frames.connect_exec(connection,globals_point):
                self.s.add(globals_point,('external_value',None))
        if type(connection.locals.value) is GlobalPort:
            self.s.edge(self.point(connection.locals.value),locals_point)
        else:
            # Snapshot content is not fabricated from the caller's fast locals.
            self.s.add(locals_point,('external_value',None))
        self.globals_by_owner[connection.root.owner].add(globals_point)
        for reference in self.references[connection.id]:
            self.s.edge(globals_point,self.point(reference.reference))
        for capture in self.captures:
            if capture.function.unit==connection.root.owner.unit:
                self.globals_by_owner[capture.body.owner].add(self.point(capture.globals))
        for owner in tuple(self.s.active_scopes):
            if owner.unit==connection.root.owner.unit:
                self.admit_scope(owner)

    def _lexical(self, descriptor, key):
        if descriptor.kind in ('LEXICAL_BINDING','CLASS_PROTOCOL_CELL'):
            return F.Namespace(F.NamespaceKind.LEXICAL,descriptor.binding,key)
        if descriptor.kind in ('CLASS_PREPARED_MAPPING','CLASS_ANNOTATION_NAMESPACE'):
            return F.Namespace(F.NamespaceKind.CLASS,descriptor.owner,key)

    def _register(self, receiver, key, value, write):
        request = key,value,write
        if request in self.watchers[receiver]:
            return
        self.watchers[receiver].add(request)
        for token in tuple(self.s.values.get(receiver,())):
            self._request(token,key,value,write)

    def _access(self, connection, access, interface):
        value = None if access.value is None else self.point(access.value)
        write = access.action is P.BindingActionKind.WRITE
        for binding in self.bindings[connection.id,interface]:
            descriptor = binding.descriptor
            lexical = self._lexical(descriptor,access.key)
            if lexical is not None:
                # Existing primitive facts already admit these same channels.
                continue
            if access.action is P.BindingActionKind.DELETE:
                # Deletion stays a state operation in the source graph. This
                # monotone may domain never treats it as a strong kill.
                continue
            if descriptor.kind=='BUILTINS':
                if not write:
                    self.builtins.admit(connection,access,binding)
                continue
            if descriptor.kind=='ROOT_GLOBALS':
                receiver = self.formal(connection,0)
            elif descriptor.kind=='ROOT_LOCALS':
                receiver = self.formal(connection,1)
            elif binding.actual is not None:
                receiver = self.point(binding.actual)
            else:
                if not write:
                    self.s.add(value,('external_value',None))
                continue
            self._register(receiver,access.key,value,write)

    def admit_scope(self, owner):
        for connection in tuple(self.connections.values()):
            if connection.root.owner.unit!=owner.unit or (connection.id,owner) in self.admitted:
                continue
            self.admitted.add((connection.id,owner))
            for access in self.accesses.get(owner,()):
                if access.lookup:
                    for lookup in access.lookup:
                        self._access(connection,access,lookup.interface)
                elif access.target is not None:
                    self._access(connection,access,access.target)

    def consume(self, point, token):
        self.builtins.consume(point,token)
        for key,target,write in tuple(self.watchers.get(point,())):
            self._request(token,key,target,write)

    def _request(self, token, key, target, write):
        request = token,key,target,write
        if request in self.mapping_requests:
            return
        self.mapping_requests.add(request)
        if token[0]=='source_root_namespace':
            self.s.frames.access_root(token,('str',key),target,write)
        elif token[0] in ('container','namespace_mapping'):
            self.s.containers.access(token,('str',key),target,write)
        elif not write:
            self.s.add(target,('external_value',None))


# Mutable dictionary views of original root namespace input references.
#
# The proxy reuses the globals formal point. It is neither a fresh allocation
# nor a module object. Content remains open and flow-insensitive; root bindings
# and declared external input fields retain their separate provenance.
_NAMESPACE_TAGS = frozenset(('source_root_namespace',))
_METHODS = frozenset(('get','update','keys','values','items'))


class RootNamespaceViews:
    def __init__(self, solver, frames):
        self.s,self.frames = solver,frames
        self.roots,self.proxies,self.proxy_units = {},{},{}
        self.input_bindings = {}
        self._mirrored,self._forwarded = set(),set()

    @staticmethod
    def _channel(unit, key):
        return unit,F.NamespaceKind.MODULE.value,0,key

    def admit(self, root_binding, contract=None):
        unit = root_binding.root.owner.unit
        previous = self.roots.get(unit)
        if previous is not None and (previous.root != root_binding.root
                or previous.globals != root_binding.globals or previous.locals != root_binding.locals):
            raise ValueError('root namespace view changed its original input interface')
        self.roots[unit] = root_binding
        proxy = ('container',self.frames.point(root_binding.globals),'MAPPING')
        self.proxies[unit] = proxy;self.proxy_units[proxy] = unit
        # A root input is not a closed source display, including an empty view.
        self.s.containers.open_containers.add(proxy)
        sources = tuple(self.s.containers.namespace_sources.items())
        for channel,points in sources:
            if channel[:3] == (unit,F.NamespaceKind.MODULE.value,0) and type(channel[3]) is str:
                for source in tuple(points):
                    self.name_write(unit,channel[3],source)
        for channel,targets in tuple(self.s.reads.items()):
            if channel[:3] == (unit,F.NamespaceKind.MODULE.value,0) and type(channel[3]) is str:
                for target in tuple(targets):
                    self.name_read(unit,channel[3],target)
        if contract is not None:
            digest = sha256(_json(encode(contract,record_tags=P.PRODUCTION_RECORD_TAGS))).hexdigest()
            if (contract.root != root_binding.root or contract.source.unit != unit
                    or root_binding.entry_contract_sha256 != digest):
                raise ValueError('root namespace external inputs differ from the declared entry contract')
            for index,binding in enumerate(contract.globals):
                if binding.namespace.kind is not F.NamespaceKind.MODULE or binding.namespace.owner != 0:
                    raise ValueError('entry global field has no root namespace key')
                port = B.StaticInputPort(root_binding.root,digest,('globals',index,'literal'))
                record = B.StaticInputBinding(port,binding.literal)
                if port in self.input_bindings:
                    if self.input_bindings[port] != record:
                        raise ValueError('declared entry field changed its literal')
                    continue
                self.input_bindings[port] = record
                point = self.s.interface_point(('static-entry-field',port),port)
                value = binding.literal.to_python()
                self.s.add(point,('constant',type(value).__name__,value))
                self.name_write(unit,binding.namespace.name,point)
        return proxy

    def proxy(self, token):
        if token[0] not in _NAMESPACE_TAGS:
            raise ValueError('root namespace reference token required')
        unit = token[1]
        if unit not in self.proxies:
            binding = self.frames.roots.get(unit)
            if binding is None:
                raise ValueError('root namespace input binding must be admitted before its view')
            self.admit(binding,self.frames.entry_contracts.get(unit))
        return self.proxies[unit]

    def name_write(self, unit, key, source):
        proxy = self.proxies.get(unit)
        if proxy is None or type(key) is not str:
            return
        item = proxy,('str',key),source
        if item in self._mirrored:
            return
        self._mirrored.add(item)
        self.s.containers.write(*item)

    def name_read(self, unit, key, target):
        proxy = self.proxies.get(unit)
        if proxy is not None and type(key) is str:
            self.s.containers.read(proxy,('str',key),target)

    def content_write(self, container, key, source):
        unit = self.proxy_units.get(container)
        if unit is None or key is None or key[0] != 'str':
            # Unknown writes already reach all registered proxy readers, and
            # future name_read subscriptions replay their wildcard contents.
            return
        item = container,key,source
        if item in self._forwarded:
            return
        self._forwarded.add(item)
        channel = self._channel(unit,key[1])
        self.s.containers.namespace_sources[channel].add(source)
        self.s.writes[source].add(channel)
        for value in tuple(self.s.values.get(source,())):
            self.s.namespace_value(channel,value)
        # The ordinary namespace subscriptions are retained too. Register all
        # existing readers so later wildcard writes reach them through the view.
        for target in tuple(self.s.reads.get(channel,())):
            self.name_read(unit,key[1],target)
        self.frames.name_write(unit,key[1],source)

    def access(self, token, key, target, write):
        self.s.containers.access(self.proxy(token),key,target,write)

    def attribute(self, token, name, target):
        proxy = self.proxy(token)
        if name == 'copy':
            self.s.add(target,('container_method',proxy,'copy'))
            return True
        if name in _METHODS:
            self.s.containers.attribute(proxy,name,target)
            return True
        return False
