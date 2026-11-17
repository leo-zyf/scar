"""One metadata-only regression for entry-cost result reuse and stale evidence."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import stateful_entry_cost as entry


class StatefulEntryReplayTest(unittest.TestCase):
    def test_exact_report_detachment_input_miss_cold_check_and_transitive_change(self):
        with TemporaryDirectory() as directory:
            leaf = Path(directory)/'source.json'
            leaf.write_text(json.dumps(dict(schema='test.entry-evidence', value=7)))
            reference = artifact_reference(leaf, 'test.entry-evidence')
            profiles = [dict(artifact=reference, locator='#', request_version=1)]
            domain = dict(schema='test.inline-domain', nested=dict(shape=[3, 3]))
            calls = []

            def derive(domain, candidate_id, profiles):
                calls.append(candidate_id)
                # The complete evidence is reached only during the original
                # producer, so this exercises automatic transitive read capture.
                source = read_artifact(profiles[0]['artifact'])
                return dict(schema=entry.SCHEMA, candidate_id=candidate_id,
                    domain=deepcopy(domain), profiles=deepcopy(profiles),
                    cost=dict(basis='UNKNOWN', evidence=source))

            live = dict(test_live_context=1, compiler=dict(command=['metadata-fixture-compiler']))
            with patch.object(entry, '_entry_live_key', return_value=live) as live_key, \
                    patch.object(entry, '_derive_stateful_entry_cost_uncached', side_effect=derive):
                with replay_transaction() as first:
                    report = entry.derive_stateful_entry_cost(domain, 'candidate:one', profiles)
                    saved = deepcopy(report)
                    report['cost']['evidence']['value'] = -1
                    self.assertEqual(entry.verify_stateful_entry_cost(saved, domain, 'candidate:one', profiles), saved)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(first.statistics['hits'], 1)
                    with self.assertRaises(ValueError):
                        entry.verify_stateful_entry_cost(report, domain, 'candidate:one', profiles)
                    self.assertEqual(len(calls), 1)
                    profiles[0]['request_version'] = 2
                    changed = entry.derive_stateful_entry_cost(domain, 'candidate:one', profiles)
                    self.assertEqual(changed['profiles'][0]['request_version'], 2)
                    self.assertEqual(len(calls), 2)

                # An independent check owns a new cold transaction, rather
                # than trusting the successful previous transaction's report.
                with replay_transaction() as second:
                    entry.verify_stateful_entry_cost(changed, domain, 'candidate:one', profiles)
                    self.assertEqual(len(calls), 3)
                    self.assertEqual(second.statistics['misses'], 1)

                # A nested source mutation cannot be hidden behind the same
                # top-level inline domain, profile reference or query key.
                with self.assertRaises(ValueError):
                    with replay_transaction():
                        entry.derive_stateful_entry_cost(domain, 'candidate:one', profiles)
                        self.assertEqual(len(calls), 4)
                        leaf.write_text(json.dumps(dict(schema='test.entry-evidence', value=8)))
                        entry.derive_stateful_entry_cost(domain, 'candidate:one', profiles)
                self.assertEqual(len(calls), 4)

                class ForeignDict(dict):
                    def items(self):
                        raise AssertionError('cache key must not inspect dict subclasses')

                @dataclass
                class ForeignRecord:
                    value: int

                    def __getattribute__(self, name):
                        if name == 'value':
                            raise AssertionError('cache key must not materialize arbitrary dataclasses')
                        return object.__getattribute__(self, name)

                live_calls = live_key.call_count
                with patch.object(entry, '_derive_stateful_entry_cost_uncached', return_value={'uncached': True}) as uncached:
                    with replay_transaction() as unsupported:
                        self.assertEqual(entry.derive_stateful_entry_cost(ForeignDict(), 'candidate:one'), {'uncached': True})
                        self.assertEqual(entry.derive_stateful_entry_cost(domain, 'candidate:one', [ForeignRecord(1)]), {'uncached': True})
                        self.assertEqual(uncached.call_count, 2)
                        self.assertEqual(unsupported.statistics['misses'], 0)
                self.assertEqual(live_key.call_count, live_calls)


if __name__ == '__main__':
    unittest.main()
