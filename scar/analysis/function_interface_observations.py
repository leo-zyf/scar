"""Function interface observations."""

from collections import Counter, defaultdict
from pathlib import Path
from scar.analysis.source_declared_calls import _owners
from scar.analysis.source_field_interfaces import _InterfaceOnly, _ancestors
from scar.analysis.source_function_connections import Invocation, connect
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.ir.program import normal_graph as T, normal_graph as F, function_sites as I
from scar.ir.program.ids import RegionID
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.source_references import SourceArtifactLocator
from hashlib import sha256
from scar.analysis.compiler_contexts import CompilerCodeIndex
from scar.analysis.source_consumer_parents import _stamp
from scar.artifacts import _json
from scar.analysis.static.query import _Fragments
from scar.compiler.graph_frontend import _source_identity
from scar.ir.program.call_edges import CapturedCode
from scar.ir.program.compiler_symbols import CompilerCodeRole as Role, CompilerTopologyMaterial
from scar.ir.record_codec import decode, encode
from scar.trace.code_snapshot import describe_code


# Conditional source ABIs selected by already validated bounded CALL targets.
#
# This is an ephemeral reader for ``SourceFunctionCallProvider``.  It reuses the
# ordinary source signature binder and original F4 facts; it does not turn a CALL
# attempt into an invocation, body entry, successful return, or effect proof.
# Repeated attempts select one shared conditional interface per call/code target.
# The raw provider retains every attempt independently.
SCHEMA = 'scar.source-bounded-function-interfaces'
CONDITIONS = (
    'BOUNDED_CALL_ATTEMPT_SELECTS_EXACT_RETAINED_CODE_BODY_ONLY',
    'CURRENT_CALLABLE_IS_RECORDED_SOURCE_FUNCTION_WITH_COMPATIBLE_CALL_PROTOCOL',
    'FUNCTION_CODE_CAPTURE_DEFAULT_AND_NAMESPACE_VERSIONS_MATCH_DECLARATION',
    'SOURCE_ARGUMENT_ABI_HAS_NO_ADDITIONAL_UNMODELED_RECEIVER',
    'SOURCE_ARGUMENT_EVALUATION_AND_SIGNATURE_ACCEPTANCE_REQUIRED',
    'CALL_ATTEMPT_DOES_NOT_PROVE_CALLER_INVOCATION_OR_CALLEE_ENTRY',
    'RETURN_ROUTES_REQUIRE_NORMAL_COMPLETION_NOT_OBSERVED_BY_CALL_CAPTURE',
    'UNOBSERVED_PATHS_FUTURE_DISPATCH_AND_EFFECT_PURITY_REMAIN_OPEN',
)
INSTANCE_DESCRIPTOR_CONDITIONS = (
    *(condition for condition in CONDITIONS
        if condition != 'SOURCE_ARGUMENT_ABI_HAS_NO_ADDITIONAL_UNMODELED_RECEIVER'),
    'ORIGINAL_ATTRIBUTE_GET_AND_ALIAS_PATH_SELECT_RECORDED_CALLABLE_CODE',
    'STANDARD_PYTHON_FUNCTION_INSTANCE_DESCRIPTOR_SELECTED',
    'ATTRIBUTE_GET_RECEIVER_IS_SELECTED_INSTANCE_NOT_A_CLASS_RECEIVER',
    'SELECTED_FUNCTION_IS_CLASS_MEMBER_NOT_INSTANCE_ATTRIBUTE_SHADOW',
    'STANDARD_ATTRIBUTE_LOOKUP_HAS_NO_CUSTOM_GETATTRIBUTE_OR_GETATTR_REDIRECTION',
    'STATICMETHOD_CLASSMETHOD_AND_CUSTOM_DESCRIPTOR_PROTOCOLS_NOT_SELECTED',
    'SOURCE_ARGUMENT_ABI_IS_EXPLICIT_ARGUMENTS_PLUS_SELECTED_DESCRIPTOR_RECEIVER',
    'DESCRIPTOR_AND_RECEIVER_VERSIONS_MATCH_SELECTED_ATTRIBUTE_ACCESS',
    'OBSERVED_CALL_ATTEMPT_DOES_NOT_PROVE_RECEIVER_IDENTITY_OR_DESCRIPTOR_BINDING',
)


class BoundedFunctionInterfaceStore:
    """Adapt a ``SourceCallBindingProvider`` to the existing interface reader.

    ``report.shards`` is an in-memory index, not a declaration of stored shard
    files.  All evidence locators address original plan, observation or F4
    records.  ``interface_fact`` is the provenance hook used by the generic
    frame provider; no synthetic invocation identity or facts file is made.
    """

    def __init__(self, owner):
        self.owner = owner
        owner.check()
        self.catalog = owner.view.catalog
        self.path = Path(owner.plan_path).parent
        self.slicer = ReferenceSlicer(self.catalog)
        self.graphs = {}
        self._functions = {}
        self._interface_facts = {}
        self._interface_attempts = {}
        self._interface_evidence = {}
        self._interface_kinds = {}
        self._attribute_indexes = {}
        self._receiver_bindings = {}
        self._sites = defaultdict(dict)
        raw = owner.materials[owner.plan_path]
        from hashlib import sha256
        artifact = ProviderArtifact(str(owner.plan_path), sha256(raw).hexdigest(),
            len(raw), owner.plan['schema'])
        self.manifest_locator = SourceArtifactLocator(artifact, artifact.sha256,
            artifact.size_bytes, 'canonical-json')
        for index, site in enumerate(owner.plan['sites']):
            unit, query = site['unit'], site['query']
            if query in self._sites[unit]:
                raise ValueError('bounded interface plan repeats an original source CALL')
            self._sites[unit][query] = (site, ProviderFactReference(artifact, '#/sites/'+str(index)))
        call_counts = {unit: len(self.catalog.fact_store.load(unit).calls) for unit in self._sites}
        self.report = dict(schema=SCHEMA, schema_version=1,
            storage='EPHEMERAL_ORIGINAL_FACT_PROJECTION',
            shards={unit:dict(calls=call_counts[unit], selected_calls=len(sites))
                for unit, sites in self._sites.items()},
            counts=dict(units=len(self._sites), calls=sum(call_counts.values()),
                selected_calls=sum(len(sites) for sites in self._sites.values()),
                unselected_calls=sum(call_counts.values())-sum(len(sites) for sites in self._sites.values())),
            semantics='CONDITIONAL_SOURCE_ABI_NOT_OBSERVED_ACTIVATION_OR_RETURN',
            target_executed=False, loops_unrolled=False, first_stage_complete=False)

    def validate(self):
        self.owner.check()

    def _target(self, body):
        """Recover the original function behind the provider's validated body."""
        if body in self._functions:
            return self._functions[body]
        source = self.owner.view.namespaces.get(body.namespace)
        if source is None or body.context is not None:
            raise ValueError('bounded target body lacks its original source namespace')
        matches = [(index, function) for index, function in enumerate(source.facts.functions)
            if source._identity(RegionID, function.template, function.body) == body.node]
        if len(matches) != 1:
            raise ValueError('bounded target body is not one original source function')
        index, function = matches[0]
        result = (source.facts, function,
            self.slicer.fact(source.facts, 'functions', index),
            _InterfaceOnly(self.slicer, source.facts, function))
        self._functions[body] = result
        return result

    def _attempt_groups(self, site):
        operation = self.owner.operations[site['id']]
        groups = {}
        for point in self.owner.by_call[operation.namespace, operation.node]:
            record = self.owner.original(point)
            key = point.namespace, point.node
            body = self.owner.body_targets.get(key)
            # CodeID alone does not identify a callable instance. Grouping only
            # shares a conditional static ABI; defaults/captures stay stateful.
            target = (record.target_kind,
                None if record.target_code is None else record.target_code.key,
                body, record.metadata_json if record.target_code is None else None)
            groups.setdefault(target, []).append((point, record))
        return tuple(groups.items())

    def _attribute_receivers(self, unit, callee):
        """Original GET bases whose result reaches this CALL via alias wires.

        This is a structural query, not a lookup of a runtime attribute or a
        receiver type. Several source GETs can remain conditional alternatives.
        """
        key = unit.unit, callee
        if key in self._receiver_bindings:
            return self._receiver_bindings[key]
        if unit.unit not in self._attribute_indexes:
            incoming = defaultdict(list)
            for source, target in unit.aliases:
                incoming[target].append(source)
            reads = defaultdict(list)
            for ordinal, access in enumerate(unit.attributes):
                if access.action is F.AttributeAction.GET:
                    reads[access.value].append((ordinal, access))
            self._attribute_indexes[unit.unit] = incoming, reads
        incoming, reads = self._attribute_indexes[unit.unit]
        ancestors = _ancestors(incoming, callee)
        aliases = tuple(self.slicer.fact(unit, 'aliases', index)
            for index, (source, target) in enumerate(unit.aliases)
            if source in ancestors and target in ancestors)
        result = tuple((self.slicer.fact(unit, 'attributes', ordinal), access, aliases)
            for ordinal, access in sorted(
                (row for value in ancestors for row in reads.get(value, ())), key=lambda row: row[0]))
        self._receiver_bindings[key] = result
        return result

    def read_unit(self, unit):
        self.validate()
        if unit not in self._sites:
            raise ValueError('source unit has no selected bounded CALL sites')
        if unit in self.graphs:
            return self.graphs[unit]
        caller = self.catalog.fact_store.load(unit)
        owners = _owners(caller)
        offset = self.slicer.store.offsets[unit]
        interfaces, frontiers, queries = [], [], []
        origins = 0
        for qi, call in enumerate(caller.calls):
            call_fact = self.slicer.fact(caller, 'calls', qi)
            selected = self._sites[unit].get(qi)
            selected_fact = call_fact if selected is None else selected[1]
            origin_indices, interface_indices = [], []
            if selected is not None:
                site, _ = selected
                groups = self._attempt_groups(site)
                if not groups:
                    frontiers.append(I.SourceFunctionInterfaceFrontier(qi, None,
                        selected_fact, 'SELECTED_CALL_TARGET_NOT_OBSERVED'))
                invocation = Invocation(offset+call.callee,
                    tuple(offset+p for p in call.positional),
                    tuple((name, offset+p) for name, p in call.keywords),
                    offset+call.result, offset+call.world, offset+call.result_world,
                    None if call.positional_pack is None else offset+call.positional_pack,
                    None if call.keyword_pack is None else offset+call.keyword_pack)
                for oi, ((kind, _, body, _), attempts) in enumerate(groups):
                    point, record = attempts[0]
                    evidence = record.evidence
                    if record.source_call != source_port(caller, call.result):
                        raise ValueError('bounded attempt differs from its original source CALL')
                    if kind not in ('PYTHON_FUNCTION', 'BOUND_PYTHON_METHOD'):
                        frontiers.append(I.SourceFunctionInterfaceFrontier(qi, None, evidence,
                            'OBSERVED_CALLABLE_REQUIRES_OTHER_CALL_PROTOCOL'))
                        continue
                    if body is None:
                        reason = self.owner.body_frontiers.get((point.namespace, point.node),
                            'OBSERVED_PYTHON_BODY_HAS_NO_RETAINED_SOURCE_CORRESPONDENCE')
                        frontiers.append(I.SourceFunctionInterfaceFrontier(qi, None, evidence, reason))
                        continue
                    target, function, definition, mini = self._target(body)
                    origins += 1
                    origin_indices.append(oi)
                    alternatives = []
                    if kind == 'PYTHON_FUNCTION':
                        direct = connect(mini, mini.offsets[target.unit]+function.value,
                            invocation, conditions=CONDITIONS)
                        if direct is not None:
                            alternatives.append((direct, evidence, (evidence,), 'DIRECT_FUNCTION'))
                    receivers = self._attribute_receivers(caller, call.callee)
                    for attribute, access, aliases in receivers:
                        descriptor = connect(mini, mini.offsets[target.unit]+function.value,
                            invocation, receiver=offset+access.receiver,
                            conditions=INSTANCE_DESCRIPTOR_CONDITIONS)
                        if descriptor is not None:
                            alternatives.append((descriptor, attribute,
                                tuple(dict.fromkeys((evidence, attribute, *aliases))),
                                'STANDARD_INSTANCE_FUNCTION_DESCRIPTOR'))
                    if not alternatives:
                        reason = ('BOUND_METHOD_RECEIVER_AND_DESCRIPTOR_ABI_NOT_CAPTURED'
                            if kind == 'BOUND_PYTHON_METHOD' and not receivers else
                            'SOURCE_SIGNATURE_REQUIRES_DIFFERENT_ARGUMENT_OR_DESCRIPTOR_BINDING')
                        frontiers.append(I.SourceFunctionInterfaceFrontier(qi, oi, evidence,
                            reason))
                        continue
                    for connection, selection, binding_evidence, binding_kind in alternatives:
                        mode = ('SOURCE_FUNCTION' if connection.activation is T.FunctionActivation.NORMAL
                            else 'CREATE_DEFERRED_BODY')
                        link = T.CallLink(source_port(caller, call.result), source_port(caller, call.callee),
                            connection.source_callable, connection.body, mode, None,
                            connection.parameters, connection.returns, connection.conditions,
                            argument_packs=connection.argument_packs, projections=connection.projections,
                            binding=connection.binding, input_envelope=connection.input_envelope)
                        ordinal = len(interfaces)
                        interface_indices.append(ordinal)
                        interfaces.append(I.SourceFunctionInterface(qi, oi, call_fact, definition,
                            evidence, connection.activation, link))
                        # A receiver branch has a distinct, original selection
                        # fact. This keeps its frame separate from a direct ABI
                        # alternative without inventing an invocation record.
                        self._interface_facts[unit, ordinal] = selection
                        self._interface_attempts[unit, ordinal] = tuple(r.evidence for _, r in attempts)
                        self._interface_evidence[unit, ordinal] = binding_evidence
                        self._interface_kinds[unit, ordinal] = binding_kind
                        if connection.activation is not T.FunctionActivation.NORMAL:
                            frontiers.append(I.SourceFunctionInterfaceFrontier(qi, oi, evidence,
                                'DEFERRED_CREATION_HAS_NO_ORDINARY_BODY_DATA_OR_WORLD_RETURN'))
            # Query indexes remain original call ordinals: frame caller-owner
            # matching must not index a compact list of selected sites.
            queries.append(I.SourceFunctionInterfaceQuery(qi, call_fact, selected_fact,
                selected_fact, owners[qi], tuple(origin_indices), tuple(interface_indices), (), ()))
        graph = I.SourceFunctionInterfaceUnit(unit, len(caller.calls), origins, (),
            tuple(interfaces), tuple(frontiers), tuple(queries),
            semantics='BOUNDED_TARGET_CONDITIONAL_SOURCE_ABI_NOT_INVOCATION_OR_COMPLETION')
        self.validate()
        self.graphs[unit] = graph
        self.report['shards'][unit]['counts'] = dict(calls=graph.calls,
            interfaces=len(interfaces), connected_calls=len({i.query for i in interfaces}),
            targets=origins, frontiers=len(frontiers),
            frontier_reasons=dict(Counter(f.reason for f in frontiers)))
        return graph

    def interface_fact(self, unit, ordinal):
        self.read_unit(unit)
        return self._interface_facts[unit, ordinal]

    def interface_attempts(self, unit, ordinal):
        """All raw attempts selecting this shared conditional body/ABI."""
        self.read_unit(unit)
        return self._interface_attempts[unit, ordinal]

    def interface_locator(self, unit, ordinal):
        """The original plain-JSON attempt, not a synthetic shard pointer."""
        # A receiver branch's frame selection is an original F4 attribute
        # fact, possibly stored compressed. Never relabel its physical bytes
        # as plain JSON: the raw attempt is an independently real locator.
        fact = self.interface_attempts(unit, ordinal)[0]
        return SourceArtifactLocator(fact.artifact, fact.artifact.sha256,
            fact.artifact.size_bytes, 'canonical-json', fact.locator)

    def interface_evidence(self, unit, ordinal):
        """Raw target selection plus original GET/alias facts for this ABI."""
        self.read_unit(unit)
        return self._interface_evidence[unit, ordinal]

    def coverage_summary(self):
        """Project every selected unit without expanding its source body DAGs."""
        graphs = tuple(self.read_unit(unit) for unit in self._sites)
        interfaces = tuple(row for graph in graphs for row in graph.interfaces)
        selected = tuple((unit, qi) for unit, sites in self._sites.items() for qi in sites)
        connected = {(graph.unit, row.query) for graph in graphs for row in graph.interfaces}
        attempts = sum(len(self.owner.by_call[operation.namespace, operation.node])
            for site in self.owner.plan['sites']
            for operation in (self.owner.operations[site['id']],))
        observed = sum(bool(self.owner.by_call[operation.namespace, operation.node])
            for site in self.owner.plan['sites']
            for operation in (self.owner.operations[site['id']],))
        return dict(schema=SCHEMA, projection_scope='ALL_SELECTED_CALLS_IN_SELECTED_ORIGINAL_SOURCE_UNITS',
            **self.report['counts'], observed_calls=observed, recorded_attempts=attempts,
            interfaces=len(interfaces), connected_selected_calls=len(connected),
            selected_calls_without_interface=len(selected)-len(connected),
            shared_source_bodies=len({row.link.body for row in interfaces}),
            activation_kinds=dict(Counter(row.activation.value for row in interfaces)),
            binding_kinds=dict(Counter(row.link.binding.value for row in interfaces)),
            callable_binding_alternatives=dict(Counter(self._interface_kinds.values())),
            parameter_routes=dict(Counter(route.kind.value for row in interfaces for route in row.link.parameters)),
            argument_projections=sum(len(row.link.projections) for row in interfaces),
            data_return_routes=sum(target==row.link.result for row in interfaces for _, target in row.link.returns),
            world_return_routes=sum(target!=row.link.result for row in interfaces for _, target in row.link.returns),
            frontier_reasons=dict(Counter(f.reason for graph in graphs for f in graph.frontiers)),
            selected_call_invocations_proven=False, callee_entry_or_completion_proven=False,
            argument_receiver_or_content_versions_observed=False,
            source_bodies_copied=False, loops_unrolled=False, first_stage_complete=False)

    def summary(self):
        return self.coverage_summary()


# Join exact Python code descriptors to retained original function bodies.
#
# Filename selects candidate catalog units only. The actual join follows an
# original F4 body scope through F2/C3 compiler construction to a code constant;
# the complete captured descriptor must then match. No target import, execution,
# function-name search, invocation identity or callable environment is inferred.
_FUNCTION_ROLES = {'FunctionDef': Role.FUNCTION_BODY,
    'AsyncFunctionDef': Role.FUNCTION_BODY, 'Lambda': Role.LAMBDA_BODY}


class RetainedCodeBodyIndex:
    """Demand-index the admitted filename candidates, sharing original bodies."""

    def __init__(self, view):
        self.view = view
        self.catalog = view.catalog
        self.sources = {}
        self._units = frozenset(self.catalog.fact_store.rows)
        self._profiles = dict(getattr(self.catalog, 'replay_profiles', {}))
        self._profile_bytes = _json(self._profiles)
        self._observed = frozenset(getattr(self.catalog.fact_store, 'observed_compiled_units', ()))
        if not self._profiles.keys() <= self._observed:
            raise ValueError('compatible replay profiles must belong to admitted compiled units')
        self._filenames = defaultdict(set)
        for unit in self._units:
            model = self.catalog.fact_store.model_units[unit]
            # A retained replay's compiler filename can differ from the F0
            # snapshot's physical path. Both are candidate indexes, not proof.
            self._filenames[model['path']].add(unit)
            profile = self._profiles.get(unit)
            if profile is not None:
                self._filenames[profile['recipe']['filename']].add(unit)
        self._loaded = {}
        self._results = {}
        self._explanations = {}
        self._guard = {}
        self._model_bytes = {}
        self.slicer = ReferenceSlicer(self.catalog)
        path = self.catalog.path/'result.json'
        raw = path.read_bytes()
        self._catalog_artifact = ProviderArtifact(str(path), sha256(raw).hexdigest(),
            len(raw), self.catalog.report['schema'])
        self._guard[path] = _stamp(path)

    def check(self):
        if _json(getattr(self.catalog, 'replay_profiles', {})) != self._profile_bytes:
            raise ValueError('retained code replay profiles changed during this query session')
        if any(_json(self.catalog.fact_store.model_units[unit]) != raw
                for unit, raw in self._model_bytes.items()):
            raise ValueError('retained code source metadata changed during this query session')
        if any(_stamp(path) != stamp for path, stamp in self._guard.items()):
            raise ValueError('retained compiler identity material changed during this query session')
        for source in self.sources.values():
            source.check()

    @staticmethod
    def _descriptor_key(code):
        if type(code) is not CapturedCode:
            raise TypeError('retained body matching needs an exact CapturedCode descriptor')
        code.__post_init__()
        # The fingerprint alone must not let a conflicting filename, flags,
        # qualname or formal layout borrow an earlier descriptor's result.
        return _json(encode(code))

    def _profile_evidence(self, unit, profile):
        if profile is None:
            return None
        matches = [(ordinal, entry) for ordinal, entry in enumerate(self.catalog.report.get('compiled', ()))
            if entry['unit'] == unit and entry['recipe'] == profile['recipe']
            and entry['root_code'] == profile['root_code']]
        if len(matches) != 1:
            raise ValueError('retained replay profile lacks its unique original catalog declaration')
        ordinal, entry = matches[0]
        return dict(catalog=encode(ProviderFactReference(self._catalog_artifact,
            '#/compiled/'+str(ordinal))), material=entry['material'],
            compatible_recipes=entry['compatible_recipes'])

    def _load(self, unit):
        if unit in self._loaded:
            return self._loaded[unit]
        profile = self._profiles.get(unit)
        if unit in self._observed and profile is None:
            result = dict(frontier='OBSERVED_COMPILED_SOURCE_REQUIRES_COMPATIBLE_REPLAY')
            self._loaded[unit] = result
            return result
        if unit in self.catalog.fact_store.original_source_units and profile is None:
            result = dict(frontier='ORIGINAL_COMPILED_CONTEXT_REQUIRES_AUTHENTICATED_REPLAY_PROFILE')
            self._loaded[unit] = result
            return result
        source = self.view.unit(unit)
        source.check()
        self.sources[unit] = source
        model = self.catalog.fact_store.model_units[unit]
        root_path = Path(self.catalog.fact_store.model_for(unit))
        fragment_paths = tuple(root_path/model['artifacts'][kind]['path']
            for kind in ('source_bytes', 'source_syntax', 'compiler_topology'))
        stamps = {path: _stamp(path) for path in fragment_paths}
        model, syntax, codes, recipe, source_sha256 = _source_identity(self.catalog, unit)
        root = codes[0]
        root_descriptor = describe_code(root)
        if profile is not None and root_descriptor != profile['root_code']:
            raise ValueError('retained compile-only root differs from its complete replay descriptor')
        topology_descriptor = model['artifacts']['compiler_topology']
        topology = decode(CompilerTopologyMaterial,
            _Fragments(root_path).read(topology_descriptor))
        paths = CompilerCodeIndex(syntax, topology, root)
        if any(_stamp(path) != stamp for path, stamp in stamps.items()):
            raise ValueError('retained compiler fragments changed during body correspondence')
        self._guard.update(stamps)
        self._model_bytes[unit] = _json(model)
        # Several paths can carry structurally identical code. Keep every
        # original construction path rather than selecting by dictionary order.
        code_paths = defaultdict(list)
        for path, actual in paths.codes.items():
            code_paths[_json(describe_code(actual))].append(path)
        functions = defaultdict(list)
        syntax_descriptor = model['artifacts']['source_syntax']
        syntax_artifact = ProviderArtifact(str(root_path/syntax_descriptor['path']),
            syntax_descriptor['sha256'], syntax_descriptor['bytes'], 'scar.source-syntax')
        topology_artifact = ProviderArtifact(str(root_path/topology_descriptor['path']),
            topology_descriptor['sha256'], topology_descriptor['bytes'], 'scar.compiler-topology')
        profile_evidence = self._profile_evidence(unit, profile)
        for ordinal, function in enumerate(source.facts.functions):
            node = syntax.nodes[function.template.occurrence]
            role = _FUNCTION_ROLES.get(node.constructor)
            scope = function.body_scope
            if role is None or scope is None:
                continue
            if (scope.owner.unit != unit or scope.template != function.template
                    or scope.region != function.body):
                raise ValueError('original source function body scope differs from its F4 declaration')
            for origin in paths.by_scope.get(scope.source_scope, ()):
                if origin.role is not role:
                    continue
                actual = paths.codes[origin.constant_path]
                descriptor = describe_code(actual)
                body = self.view.source_region_ref(unit, function.template, function.body)
                proof = dict(unit=unit, body=encode(body),
                    function_fact=encode(self.slicer.fact(source.facts, 'functions', ordinal)),
                    body_scope=encode(scope), compiler_path=encode(origin),
                    constant_path=list(origin.constant_path), compiled_root=root_descriptor,
                    recipe=dict(recipe), recipe_evidence=profile_evidence,
                    source_sha256=source_sha256,
                    syntax=encode(ProviderFactReference(syntax_artifact,
                        '#/nodes/'+str(function.template.occurrence))),
                    topology=encode(ProviderFactReference(topology_artifact,
                        '#/blocks/'+str(origin.block))))
                functions[_json(descriptor)].append((body, proof))
        result = dict(functions=dict(functions), code_paths=dict(code_paths),
            compiler_frontiers=paths.frontiers, frontier=None)
        self._loaded[unit] = result
        self.check()
        return result

    def match(self, code):
        key = self._descriptor_key(code)
        self.check()
        if key in self._results:
            return self._results[key]
        units = tuple(sorted(self._filenames.get(code.filename, ())))
        matches, frontiers, compiler_matches = [], [], []
        for unit in units:
            material = self._load(unit)
            if material['frontier'] is not None:
                frontiers.append(dict(unit=unit, reason=material['frontier']))
                continue
            matches.extend(material['functions'].get(key, ()))
            if key in material['code_paths']:
                compiler_matches.append(dict(unit=unit,
                    constant_paths=[list(path) for path in material['code_paths'][key]]))
        bodies = tuple(dict.fromkeys(body for body, _ in matches))
        if len(bodies) == 1:
            result = bodies[0], None
        elif len(bodies) > 1:
            result = None, 'AMBIGUOUS_ORIGINAL_CODE_BODY'
        elif not units:
            result = None, 'OBSERVED_PYTHON_BODY_OUTSIDE_ADMITTED_CATALOG'
        elif compiler_matches:
            result = None, 'RETAINED_COMPILER_CODE_HAS_NO_ORDINARY_FUNCTION_BODY_PROTOCOL'
        elif frontiers:
            result = None, 'RETAINED_CODE_CANDIDATE_REQUIRES_EXPLICIT_COMPILER_PROFILE'
        else:
            result = None, 'NO_EXACT_RETAINED_COMPILER_CODE_DESCRIPTOR'
        self._results[key] = result
        self._explanations[key] = dict(schema='scar.retained-code-body-correspondence',
            schema_version=1, code=encode(code), candidate_units=list(units),
            matches=[proof for _, proof in matches], compiler_matches=compiler_matches,
            candidate_frontiers=frontiers, frontier_reason=result[1],
            body=encode(result[0]), target_executed=False,
            callable_environment_or_invocation_proven=False)
        self.check()
        return result

    def explain(self, code):
        key = self._descriptor_key(code)
        self.match(code)
        # Do not expose mutable internal proof records to callers.
        import json
        return json.loads(_json(self._explanations[key]))


__all__ = [
    'BoundedFunctionInterfaceStore',
    'RetainedCodeBodyIndex',
    'SCHEMA',
    'CONDITIONS',
    'INSTANCE_DESCRIPTOR_CONDITIONS',
]
