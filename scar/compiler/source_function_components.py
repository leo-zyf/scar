"""Conditional function targets through admitted F4 globals/import exports.

Module candidates come only from retained catalog mappings. Namespace queries
keep original import/export/function facts; no filesystem search, importlib or
function-name matching establishes a callable target.
"""
from collections import defaultdict,deque
from dataclasses import dataclass

from scar.analysis.source_namespace_interfaces import NamespaceReferenceResolver
from scar.analysis.source_reference_slice import source_port
from scar.compiler.pipeline_frontend import _Callee, _global_functions
from scar.ir.program import normal_graph as F, namespace_routes as N
from scar.ir.record_codec import encode


@dataclass(frozen=True)
class DeclaredFunctionTarget:
    unit: str
    function: object
    evidence: dict

    @property
    def key(self):return self.unit,self.function.template


class SourceFunctionComponents:
    def __init__(self,view):
        self.view=view;self.resolver=NamespaceReferenceResolver(view.catalog)
        self.bindings={};self.frontiers={};self._queries={}

    def dependency_units(self,roots):
        """Follow admitted import facts even when output is source-filtered."""
        seen=set();pending=deque(roots)
        while pending:
            unit=pending.popleft()
            if unit in seen:continue
            seen.add(unit);facts=self.view.unit(unit).facts
            for row in facts.imports:
                for module in (row.module,row.requested_module):
                    if module:pending.extend(self.resolver.modules.get(module,()))
        return tuple(sorted(seen))

    def targets(self,unit):
        if unit in self.bindings:return self.bindings[unit]
        facts=self.view.unit(unit).facts;writes=defaultdict(list)
        for i,row in enumerate(facts.names):
            if row.write and row.namespace.kind is F.NamespaceKind.MODULE:writes[row.namespace.name].append((i,row))
        result={};frontiers={}
        for name,rows in writes.items():
            candidates={};evidence=[];issues=[]
            for ordinal,row in rows:
                point=source_port(facts,row.value)
                if point not in self._queries:self._queries[point]=self.resolver.resolve(point)
                query=self._queries[point];used=set()
                for edge in query.edges:used.update(edge.evidence)
                for origin in query.origins:used.update(origin.evidence)
                for frontier in query.frontiers:used.update(frontier.evidence)
                # Distinct retained module providers are genuine alternatives,
                # never ordered by file path or silently selected by name.
                for state in query.states:
                    provider,local=self.resolver.slicer.local(state.port)
                    for import_i,imported in enumerate(provider.imports):
                        ref=self.resolver.slicer.fact(provider,'imports',import_i)
                        if ref not in used:continue
                        for module in (imported.module,imported.requested_module):
                            options=self.resolver.modules.get(module,()) if module else ()
                            if len(options)>1:issues.append(dict(code='AMBIGUOUS_ADMITTED_MODULE_PROVIDER',module=module,units=list(options)))
                for origin in query.origins:
                    if origin.kind is not N.NamespaceOriginKind.FUNCTION_DEFINITION:continue
                    terminal=query.states[origin.state]
                    if terminal.selectors:continue
                    provider,local=self.resolver.slicer.local(terminal.port)
                    functions=[fn for fn in provider.functions if fn.value==local]
                    if len(functions)!=1:continue
                    function=functions[0];candidates[provider.unit,function.template]=function
                if any(frontier.reason in ('NAMESPACE_SELECTOR_DEPTH_FRONTIER','NAMESPACE_REFERENCE_STATE_BUDGET_FRONTIER')
                        for frontier in query.frontiers):issues.append(dict(code='DECLARED_FUNCTION_QUERY_LIMIT'))
                evidence.append(dict(writer=encode(self.resolver.slicer.fact(facts,'names',ordinal)),
                    reference=encode(query),selection='CONDITIONAL_ON_ACTUAL_FUNCTION_GLOBAL_CODE_ID_GUARDS'))
            if issues or len(candidates)!=1:
                frontiers[name]=dict(code='AMBIGUOUS_DECLARED_FUNCTION_TARGET' if len(candidates)>1 else
                    'DECLARED_FUNCTION_TARGET_OPEN',function_candidates=[dict(unit=key[0],template=encode(key[1]))
                        for key in candidates],issues=issues)
                continue
            (target_unit,template),function=next(iter(candidates.items()))
            result[name]=DeclaredFunctionTarget(target_unit,function,dict(kind='original_f4_namespace_function_route',
                owner_unit=unit,global_name=name,target_unit=target_unit,target_template=encode(template),writers=evidence))
        self.bindings[unit]=result;self.frontiers[unit]=frontiers
        return result

    def globals(self,unit,components,*,fused=False):
        """Preserve old direct-definition paths; add proven import candidates."""
        facts=self.view.unit(unit).facts
        plans={key[1]:(value.fused_body if fused else value.execution) for key,value in components.items()
            if key[0]==unit and (not fused or value.compositions)}
        result=_global_functions(facts,plans)
        for name,target in self.targets(unit).items():
            component=components.get(target.key)
            if component is None or fused and not component.compositions:continue
            if name in result:continue
            plan=component.fused_body if fused else component.execution
            result[name]=[_Callee(name,target.function,plan)]
        # Existing local declaration ambiguity must not be hidden by a unique
        # component subset: every selected global still needs one source fn.
        for name in tuple(result):
            if name not in self.targets(unit):result.pop(name)
        return result

    def imported_route(self,unit,name):
        target=self.targets(unit).get(name)
        return target.evidence if target is not None and target.unit!=unit else None

    def unresolved_dependencies(self,unit,function,components,failures,trail=()):
        """Explain missing components through original owner-scoped name reads.

        A cycle here is a source component dependency, not an observed dynamic
        invocation. No recursive component is substituted by a guessed leaf.
        """
        key=unit,function.template
        if key in trail:return [dict(code='CYCLIC_COMPONENT_DEPENDENCY',unit=unit,template=encode(function.template))]
        if len(trail)>=64:return [dict(code='COMPONENT_DEPENDENCY_DEPTH_FRONTIER',unit=unit)]
        if function.body_scope is None:return [dict(code='FUNCTION_BODY_SCOPE_NOT_RECORDED',unit=unit)]
        facts=self.view.unit(unit).facts;names=set()
        for group in facts.scope_groups:
            if group.owner!=function.body_scope.owner:continue
            for span in group.tables:
                if span.table is not F.FactTable.NAMES:continue
                names.update(row.namespace.name for row in facts.names[span.start:span.stop]
                    if not row.write and row.namespace.kind is F.NamespaceKind.MODULE)
        result=[];targets=self.targets(unit)
        for name in sorted(names):
            if name in self.frontiers[unit]:
                result.append(dict(global_name=name,**self.frontiers[unit][name]));continue
            target=targets.get(name)
            if target is None or target.key in components:continue
            result.append(dict(code='UNRESOLVED_FUNCTION_COMPONENT',global_name=name,target_unit=target.unit,
                target_template=encode(target.function.template),failure=failures.get(target.key),
                dependencies=self.unresolved_dependencies(target.unit,target.function,components,failures,(*trail,key))))
        return result


__all__=['DeclaredFunctionTarget','SourceFunctionComponents']
