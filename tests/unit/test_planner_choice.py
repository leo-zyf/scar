from dataclasses import replace

import pytest

from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.planner.choice import FiniteMedianOption, choose_finite_medians


def test_finite_median_policy_retains_original_unknown_and_context():
    scope = 'ONE_CANONICAL_COMPONENT_WINDOW'
    context = {'cpu_ids': [2], 'shape': [8, 9], 'contract': 'source-qualified-test-model'}
    artifact = ProviderArtifact('test:canonical-facts', 'a' * 64, 1, 'test.model')
    def option(identifier, median):
        return FiniteMedianOption(identifier, scope, context, median,
                                  (ProviderFactReference(artifact, '#/' + identifier),))
    original = option('original', 8)
    candidate = option('candidate', 5)
    def choose(rows):
        return choose_finite_medians(rows, original_id='original', scope=scope,
            context=context, tick_numerator=1, tick_denominator=1_000_000_000)

    result = choose([original, candidate])
    assert result['selected_id'] == 'candidate' and result['minimum_median_ticks'] == 5
    assert result['status'] == 'FINITE_MODEL_MINIMUM'
    assert result['execution_authorized'] is result['deployment_authorized'] is False
    assert result['physical_optimality_proven'] is result['residuals_cancelled'] is False
    assert [row['id'] for row in result['options']] == ['original', 'candidate']
    assert result['options'][1]['source_refs'][0]['locator'] == '#/candidate'

    tie = choose([original, replace(candidate, median_ticks=8)])
    assert tie['selected_id'] == 'original'
    assert tie['tied_minimum_ids'] == ['candidate', 'original']
    assert choose([original, replace(candidate, median_ticks=9)])['selected_id'] == 'original'
    unknown = choose([original, replace(candidate, median_ticks=None)])
    assert unknown['status'] == 'UNRESOLVED' and unknown['selected_id'] is None
    assert unknown['minimum_median_ticks'] is None and unknown['unknown_option_ids'] == ['candidate']
    assert unknown['options'][1]['median_ticks'] is None and unknown['unknown_duration_imputed'] is False
    assert choose([replace(original, median_ticks=None), candidate])['selected_id'] is None
    for bad in (replace(candidate, context={'cpu_ids': [3]}),
                replace(candidate, scope='OTHER_SCOPE'), replace(candidate, source_refs=()),
                replace(candidate, median_ticks=float('nan')),
                replace(candidate, median_ticks=float('inf')), replace(candidate, median_ticks=-1)):
        with pytest.raises(ValueError):
            choose([original, bad])
    with pytest.raises(ValueError, match='original option'):
        choose([candidate])
