"""One feature-projection regression: retain versions and missing quantities."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scar.analysis import gcc_cost_features as features
from scar.analysis.gcc_optimization import parse_gcc_optimization_record
from scar.compiler.collect_native_optimization import _write_feedback
from scar.ir.execution_recipe import artifact_reference


class GCCVectorCostFeaturesTest(unittest.TestCase):
    def test_multiple_models_and_missing_field_are_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.cpp'; source.write_text('int x;\n'*12)
            source_ref = artifact_reference(source, 'scar.cpp-source')
            lines = ['Cost model analysis: \n', '  Vector inside of loop cost: 11\n',
                '  Vector prologue cost: 2\n', '  Vector epilogue cost: 3\n',
                '  Scalar iteration cost: 4\n', '  Scalar outside cost: 0\n',
                '  Vector outside cost: 5\n', '  prologue iterations: 0\n',
                '  epilogue iterations: 1\n', '  Calculated minimum iters for profitability: 2\n']
            record = [dict(format='1', generator=dict(name='GNU C++20', version='13.3.0')),
                [dict(id='0x1', name='vect', type='gimple', num=1, optgroups=['loop'])],
                [dict(kind='note', **{'pass':'0x1'}, function='run',
                    location=dict(file=str(source), line=10),
                    message=lines + [line for line in lines if 'Scalar outside cost:' not in line])]]
            raw_path = root/'source.cpp.opt-record.json.gz'
            raw_path.write_bytes(gzip.compress(json.dumps(record).encode(), mtime=0))
            record_ref = artifact_reference(raw_path, 'gcc.optimization-record-json-gzip')
            projection = parse_gcc_optimization_record(record_ref, source_ref=source_ref,
                stage_spans=[dict(id='stage:4', start_line=10, end_line=10)])
            feedback_ref = _write_feedback(root/'feedback.json.gz', projection)
            collection = dict(schema='scar.gcc-native-optimization-collection', schema_version=1,
                source=source_ref, files=dict(record=record_ref, feedback=feedback_ref), subject=None)
            collection_path = root/'result.json'; collection_path.write_text(json.dumps(collection))
            collection_ref = artifact_reference(collection_path, collection['schema'])
            # Unit fixture isolates text projection; actual retained collections
            # are separately exercised through the unchanged cold checker.
            with patch.object(features, 'check_gcc_optimization', return_value=collection) as checked:
                report = features.project_gcc_vector_costs(collection_ref)
                checked.assert_called_once_with(collection_ref)
            self.assertEqual(len(report['blocks']), 2)
            first, second = report['blocks']
            self.assertNotEqual(first['id'], second['id'])
            self.assertEqual(first['features']['scalar_outside_cost']['value'], 0)
            self.assertEqual(first['status'], 'ALL_REQUESTED_FIELDS_REPORTED')
            self.assertIsNone(second['features']['scalar_outside_cost']['value'])
            self.assertEqual(second['features']['scalar_outside_cost']['status'], 'UNKNOWN')
            self.assertEqual(len(report['stage_blocks']['stage:4']), 2)
            self.assertEqual(first['features']['vector_inside_cost']['unit'], 'GCC_TARGET_RELATIVE_COST')
            self.assertEqual(first['features']['prologue_iterations']['unit'], 'GCC_MODEL_ITERATIONS')
            self.assertFalse(report['qualifications']['best_version_selected'])
            self.assertFalse(report['qualifications']['stage_costs_summed'])
            self.assertIsNone(report['qualified_profile'])


if __name__ == '__main__':
    unittest.main()
