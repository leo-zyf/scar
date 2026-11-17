"""One metadata-only regression for default/alias/kwargs guard witnesses."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scar.ir.execution_recipe import artifact_reference
from scar.optimization import cost_context_compatibility as compatibility


class CostContextCompatibilityTest(unittest.TestCase):
    def test_pinned_alias_kwargs_default_and_unknown_runtime_metadata(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def source(name, text):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                return artifact_reference(path, 'scar.python-source')

            def document(name, schema, **values):
                path = root / name
                path.write_text(json.dumps(dict(schema=schema, schema_version=1, **values)))
                return artifact_reference(path, schema)

            source('runtime/stateful_guard.py', '''
def guarded_stateful_function(original, artifact, *, trial=False):
    def call():
        if not trial:
            return 'nontrial'
        return 'trial'
    return call
def install_stateful_regions(module, artifacts, *, trial=False):
    return guarded_stateful_function(module, artifacts, trial=trial)
''')
            source('compiler/stateful_runtime.py', '''
from scar.runtime import stateful_guard as lower
def install_stateful_regions(module, artifacts, *, trial=False):
    return lower.install_stateful_regions(module, artifacts, trial=trial)
''')
            source('compiler/unrelated_name.py', '''
SCHEMA = 'scar.stateful-scope-boundary-calibration'
def collect():
    from scar.runtime.stateful_guard import guarded_stateful_function as wrap
    options = {}
    return wrap(None, None, **options)
''')
            harness = source('harness.py', '''
import scar.compiler.stateful_runtime as wrappers
def execute():
    options = dict(trial=True)
    return wrappers.install_stateful_regions(None, [], **options)
''')
            pins = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.glob('*/*.py'))}
            region = document('region.json', 'scar.stateful-array-region')
            plan = document('plan.json', 'scar.stateful-stage-plan')
            domain = document('domain.json', 'scar.stateful-cpu-domain', domain_id='domain:one', base_stage_plan=plan)
            special = document('special.json', 'scar.observed-stateful-specialization',
                domain=domain, source_region=region, input_shapes={'input:0': [3, 3]})
            compiler = dict(identity='test compiler', options=['-O3'])
            native = document('native.json', 'scar.native-stateful-artifact',
                compiler=compiler, cpu_ids=[0], provider_runtime={'id': 'provider'})
            observations = document('scope.json', 'scar.stateful-scope-boundary-calibration',
                domain_id='domain:one', candidate_id='candidate:one',
                domain_source=dict(region=region, stage_plan=plan),
                probe=dict(implementation_files=pins, compiler=compiler, source_fragments=dict(cpus=[0])),
                canonical_structure=dict(inputs=[dict(id='input:0', kind='array', shape=[3, 3])]),
                observed_caller_affinity_before=[0], runtime_path=['recorded canonical chain'],
                resource_context=dict(backend=dict(provider_runtime={'id': 'provider'})))

            def holdout(harness):
                return document('holdout.json', 'scar.real-caller-cost-holdout',
                    harness=harness, specialization=special, native_artifact=native,
                    explicit_trial=True, protocol=dict(caller_cpu_ids=[0]),
                    frozen_model_binding=dict(candidate_id='candidate:one'))

            with patch.object(compatibility, '_ROOT', root):
                actual = holdout(harness)
                report = compatibility.check_stateful_cost_compatibility(observations, actual)
                self.assertEqual(report['result']['mismatched_fields'], ['source_configuration.trial'])
                self.assertFalse(report['source_witnesses']['canonical']['trial'])
                self.assertTrue(report['source_witnesses']['actual']['trial'])
                self.assertEqual(len(report['source_witnesses']['actual']['chain']), 3)
                recorded = next(row for row in report['fields'] if row['name'] == 'recorded_guard_regime.trial')
                self.assertEqual(recorded['relation'], 'UNKNOWN')
                self.assertFalse(report['result']['automatic_cost_transfer_qualified'])
                compatibility.verify_cost_compatibility(report, observations, actual)
                changed = deepcopy(report)
                changed['result']['qualified_profile'] = 'forged'
                with self.assertRaises(ValueError):
                    compatibility.verify_cost_compatibility(changed, observations, actual)

                # A local alias of kwargs passed to an unknown function may be
                # mutated.  It cannot become a proved trial=True witness.
                harness = source('harness.py', '''
from scar.compiler import stateful_runtime as wrappers
def execute():
    options = {'trial': True}
    alias = options
    unknown_mutation(options)
    return wrappers.install_stateful_regions(None, [], **alias)
''')
                unresolved = compatibility.check_stateful_cost_compatibility(observations, holdout(harness))
                self.assertEqual(unresolved['source_witnesses']['actual']['status'], 'UNKNOWN')
                self.assertIn('source_configuration.trial', unresolved['result']['unknown_fields'])
                self.assertIsNone(unresolved['result']['qualified_profile'])


if __name__ == '__main__':
    unittest.main()
