"""Sourced interval comparison of complete alternatives for one fixed task.

This is not a resource scheduler or a physical performance oracle. All options
have the same task contract, context scope and tick. Their complete placements
are conditionally feasible according to the supplied original legality facts.
An unbounded cost remains unbounded; it is never submitted as duration zero.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass,is_dataclass
import math
import re
from typing import Any

from scar.ir.execution_recipe import read_artifact
from scar.ir.program.providers import ProviderFactReference
from scar.optimization.scope_accounting import _pointer
from .resource_solver import semantic_sha256


TIE_POLICY='MINIMUM_FEASIBLE_UPPER_THEN_LEXICOGRAPHIC_ID'


@dataclass(frozen=True,slots=True)
class CostInterval:
    lower: int
    upper: int | None


@dataclass(frozen=True,slots=True)
class IntervalOption:
    id: str
    contract_sha256: str
    scope: dict[str,Any]
    interval: CostInterval
    resource_demands: dict[str,int]
    legality_conditions: tuple[str,...]
    interval_evidence: tuple[ProviderFactReference,...]
    scope_evidence: tuple[ProviderFactReference,...]
    legality_evidence: tuple[ProviderFactReference,...]


def _need(value,message):
    if not value:raise ValueError(message)


def _wire(value):return asdict(value) if is_dataclass(value) else deepcopy(value)


def solve_interval_choices(options,*,contract_sha256,scope,tick_seconds,tie_policy=TIE_POLICY):
    """Return a replayable interval choice while retaining every original option.

    A finite upper endpoint is a feasible incumbent within the supplied model.
    Other options are excluded only when their lower endpoint is strictly larger.
    Selection is possible when that incumbent is provably no worse than every
    remaining alternative. Possible equal costs are retained under an explicit
    minimum-upper/ID policy; uniqueness and physical optimality are not claimed.
    """
    _need(isinstance(contract_sha256,str) and re.fullmatch('[0-9a-f]{64}',contract_sha256),
        'explicit task contract SHA required')
    _need(isinstance(scope,dict) and bool(scope),'explicit nonempty comparison scope required')
    _need(type(tick_seconds)in(int,float) and math.isfinite(tick_seconds) and tick_seconds>0,
        'positive finite common tick required')
    _need(tie_policy==TIE_POLICY,'unsupported or implicit tie policy')
    _need(isinstance(options,(list,tuple)) and options,'nonempty original finite option domain required')
    scope=deepcopy(scope);scope_sha=semantic_sha256(scope);cache={};references={}

    def evidence(records,matcher,label):
        _need(isinstance(records,(list,tuple)) and records,'retained '+label+' FactRefs required')
        matched=False
        for supplied in records:
            ref=_wire(supplied)
            _need(set(ref)=={'artifact','locator'},'original typed FactRef fields required')
            artifact=ref['artifact'];key=semantic_sha256(artifact)
            if key not in cache:
                cache[key]=read_artifact(artifact);references[key]=artifact
            matched=matcher(_pointer(cache[key],ref['locator'])) or matched
        _need(matched,'matching retained '+label+' evidence required')

    normalized=[];seen=set()
    for supplied in options:
        row=_wire(supplied)
        _need(set(row)==set(IntervalOption.__dataclass_fields__),'exact interval-option fields required')
        identifier=row['id'];_need(isinstance(identifier,str) and identifier and identifier not in seen,
            'nonempty unique option id required');seen.add(identifier)
        _need(row['contract_sha256']==contract_sha256 and semantic_sha256(row['scope'])==scope_sha,
            'option task contract or comparison scope differs')
        interval=_wire(row['interval']);_need(set(interval)=={'lower','upper'},'lower/upper interval required')
        low,high=interval['lower'],interval['upper']
        _need(type(low)is int and low>=0 and (high is None or type(high)is int and high>=low),
            'nonnegative integer lower and compatible finite/infinite upper required')
        demands=row['resource_demands'];conditions=list(row['legality_conditions'])
        _need(isinstance(demands,dict) and all(isinstance(k,str) and k and type(v)is int and v>0
            for k,v in demands.items()),'explicit complete-option resource demand required')
        _need(conditions and all(isinstance(c,str) and c for c in conditions),
            'nonempty original legality conditions required')
        expected_legality=dict(contract_sha256=contract_sha256,resource_demands=demands,conditions=conditions)
        evidence(row['legality_evidence'],lambda value:isinstance(value,dict) and all(
            value.get(k)==v for k,v in expected_legality.items()),'contract/resource/legality')
        evidence(row['scope_evidence'],lambda value:isinstance(value,dict) and
            semantic_sha256(value)==scope_sha,'comparison scope')
        def cost_matches(value):
            if not isinstance(value,dict) or value.get('tick_seconds')!=tick_seconds:return False
            declared=value.get('interval_ticks',value.get('interval'))
            if isinstance(declared,dict):
                return type(declared.get('lower'))is int and declared.get('lower')==low and declared.get('upper')==high
            return (high is not None and low==high and type(value.get('duration_ticks'))is int
                and value['duration_ticks']==low)
        evidence(row['interval_evidence'],cost_matches,'cost interval and tick')
        normalized.append({**row,'interval':interval,'legality_conditions':conditions})
    finite=[r for r in normalized if r['interval']['upper']is not None]
    incumbent=min(finite,key=lambda r:(r['interval']['upper'],r['id'])) if finite else None
    upper=None if incumbent is None else incumbent['interval']['upper']
    compared=[];frontier=[];excluded=[];possible_ties=[]
    for row in normalized:
        low=row['interval']['lower'];dominated=upper is not None and low>upper
        proof=None
        if dominated:
            excluded.append(row['id'])
            proof=dict(rule='STRICT_LOWER_EXCEEDS_FEASIBLE_UPPER',candidate_lower_ticks=low,
                incumbent_upper_ticks=upper,incumbent_id=incumbent['id'],
                candidate_evidence=row['interval_evidence'],incumbent_evidence=incumbent['interval_evidence'])
        elif incumbent is None or row['id']!=incumbent['id'] and low<upper:
            frontier.append(dict(option_id=row['id'],interval=row['interval'],evidence=row['interval_evidence'],
                reason='NO_FEASIBLE_FINITE_UPPER_BOUND' if incumbent is None
                    else 'COULD_BE_STRICTLY_BETTER_THAN_INCUMBENT'))
        elif row['id']!=incumbent['id'] and low==upper:
            possible_ties.append(row['id'])
        compared.append(dict(**row,strictly_dominated=dominated,exclusion_certificate=proof,
            retained_in_original_domain=True))
    optimal=incumbent is not None and not frontier
    for artifact in references.values():read_artifact(artifact,json_document=False)
    return dict(schema='scar.finite-interval-choice',schema_version=1,options=compared,
        option_domain_sha256=semantic_sha256(normalized),contract_sha256=contract_sha256,
        scope=scope,scope_sha256=scope_sha,tick_seconds=tick_seconds,tie_policy=tie_policy,
        status='FINITE_MODEL_OPTIMAL' if optimal else 'UNRESOLVED',
        selected_id=incumbent['id'] if optimal else None,
        known_upper_incumbent=None if incumbent is None else dict(id=incumbent['id'],
            interval=incumbent['interval'],source=incumbent['interval_evidence']),
        strictly_excluded_option_ids=excluded,remaining_frontier=frontier,
        possible_equal_cost_option_ids=possible_ties,
        original_option_count=len(normalized),retained_option_count=len(compared),
        optimum_duration_interval_ticks=incumbent['interval'] if optimal else None,
        unique_optimum_proven=False,physical_optimality_proven=False,deployment_authorized=False,
        lower_bound_is_physical_runtime_bound=False,
        authority='FINITE_SOURCED_OPTION_INTERVAL_MODEL_WITH_ORIGINAL_LEGALITY_CONDITIONS',
        resource_scheduling_performed=False,complete_call_alternatives_only=True,
        original_option_removed=False,unknown_duration_imputed=False,
        source_artifacts=list(references.values()))
