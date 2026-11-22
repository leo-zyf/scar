"""Join exact Python code descriptors to retained original function bodies.

Filename selects candidate catalog units only. The actual join follows an
original F4 body scope through F2/C3 compiler construction to a code constant;
the complete captured descriptor must then match. No target import, execution,
function-name search, invocation identity or callable environment is inferred.
"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from scar.analysis.compiler_contexts import CompilerCodeIndex
from scar.analysis.source_consumer_parents import _stamp
from scar.analysis.source_references import ReferenceSlicer
from scar.analysis.source_templates import _json
from scar.analysis.static.query import _Fragments
from scar.compiler.graph_frontend import _source_identity
from scar.ir.program.call_edges import CapturedCode
from scar.ir.program.code_identity import CompilerCodeRole as Role
from scar.ir.program.compiler_symbols import CompilerTopologyMaterial
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode
from scar.trace.code_snapshot import describe_code


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


__all__ = ['RetainedCodeBodyIndex']
