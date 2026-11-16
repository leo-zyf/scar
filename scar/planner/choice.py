"""Pure median policy for one explicitly listed finite component-model domain.

The caller admits the source facts and defines the compared scope/context.
Empirical ranges, residuals, feasibility and runtime authorization are outside
this policy; no missing median is treated as a free implementation.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math

from scar.ir.program.providers import ProviderFactReference


@dataclass(frozen=True, slots=True)
class FiniteMedianOption:
    id: str
    scope: str
    context: dict
    median_ticks: int | float | None
    source_refs: tuple[ProviderFactReference, ...]


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _context_key(context):
    _need(type(context) is dict and context, 'explicit nonempty comparison context required')
    return json.dumps(context, sort_keys=True, separators=(',', ':'), allow_nan=False)


def choose_finite_medians(options, *, original_id, scope, context,
                          tick_numerator, tick_denominator):
    """Choose the lowest fully known median, preferring original on a tie.

    Completeness concerns only the supplied finite list, never the physical
    implementation space or its unmodeled residuals. Source references are
    retained and structurally checked, not read or qualified by this function.
    """
    _need(type(original_id) is str and original_id, 'explicit original option ID required')
    _need(type(scope) is str and scope, 'explicit nonempty comparison scope required')
    context_key = _context_key(context)
    _need(all(type(v) is int and v > 0 for v in (tick_numerator, tick_denominator)),
          'positive rational common tick required')
    _need(type(options) in (tuple, list) and options, 'nonempty explicitly listed option domain required')
    rows = []; seen = set(); unknown = []
    for option in options:
        _need(type(option) is FiniteMedianOption, 'typed finite median option required')
        _need(type(option.id) is str and option.id and option.id not in seen,
              'unique nonempty option ID required')
        seen.add(option.id)
        _need(option.scope == scope and _context_key(option.context) == context_key,
              'option comparison scope or context differs')
        _need(type(option.source_refs) is tuple and option.source_refs
              and all(type(ref) is ProviderFactReference for ref in option.source_refs),
              'nonempty original source FactRefs required for every option')
        for ref in option.source_refs:
            ref.__post_init__()
        value = option.median_ticks
        _need(value is None or type(value) in (int, float) and value >= 0
              and (type(value) is int or math.isfinite(value)),
              'median must be a nonnegative finite number or explicit UNKNOWN None')
        if value is None:
            unknown.append(option.id)
        rows.append(asdict(option))
    _need(original_id in seen, 'original option must remain in the finite domain')
    minimum = None if unknown else min(row['median_ticks'] for row in rows)
    ties = [] if unknown else sorted(row['id'] for row in rows if row['median_ticks'] == minimum)
    selected = None if unknown else original_id if original_id in ties else ties[0]
    return dict(schema='scar.finite-median-choice', schema_version=1,
        original_id=original_id, options=rows, scope=scope, context=deepcopy(context),
        tick_numerator=tick_numerator, tick_denominator=tick_denominator,
        status='UNRESOLVED' if unknown else 'FINITE_MODEL_MINIMUM',
        selected_id=selected, minimum_median_ticks=minimum,
        tied_minimum_ids=ties, unknown_option_ids=unknown,
        tie_policy='ORIGINAL_THEN_LEXICOGRAPHIC_ID',
        explicit_option_count=len(rows), explicit_domain_medians_complete=not unknown,
        authority='SUPPLIED_FINITE_COMPONENT_MODEL_MEDIANS_ONLY',
        source_evidence_replayed=False, caller_source_admission_required=True,
        unknown_duration_imputed=False, residuals_cancelled=False,
        empirical_ranges_used_for_selection=False, physical_optimality_proven=False,
        execution_authorized=False, deployment_authorized=False)


__all__ = ['FiniteMedianOption', 'choose_finite_medians']
