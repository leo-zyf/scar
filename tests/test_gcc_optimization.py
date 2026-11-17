"""One bounded metadata regression for nested GCC record projections."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scar.analysis.gcc_optimization import parse_gcc_optimization_record
from scar.ir.execution_recipe import artifact_reference


class GCCOptimizationTest(unittest.TestCase):
    def test_nested_inline_many_to_many_identity_opaque_and_budgets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'main.c'; source.write_text('int x;\n'*30)
            other = root/'other'/'main.c'; other.parent.mkdir(); other.write_bytes(source.read_bytes())
            source_ref = artifact_reference(source, 'scar.c-source')
            record = [dict(format='1', generator=dict(name='GNU C17', version='13.3.0', target='test-target')),
                [dict(id='0x001', type='gimple', name='loop', num=7, optgroups=['loop'], children=[
                    dict(id='0x002', type='gimple', name='inline', num=8, optgroups=[])])],
                [dict(kind='scope', function='outer', **{'pass': '0x001'},
                    location=dict(file=str(source), line=10, column=2), message=['__FULL_SCOPE_BODY__'],
                    count=dict(value=100, quality='guessed_local'), children=[
                        dict(kind='success', function='callee', **{'pass': '0x002'},
                            location=dict(file=str(other), line=20, column=1),
                            inlining_chain=[dict(fndecl='callee'), dict(fndecl='outer', site=dict(file='main.c', line=20, column=3))],
                            count=dict(value=50, quality='guessed_local'),
                            message=['__FULL_SUCCESS_BODY__', dict(location=dict(file=str(source), line=20, column=9))]),
                        dict(kind='failure', **{'pass': '0x002'}, message=['__FULL_MISSED_BODY__'],
                            count=dict(value=0, quality='uninitialized'),
                            impl_location=dict(file=str(source), line=20, function='compiler_internal')),
                        dict(kind='vendor-diagnostic', **{'future/kind': dict(site=dict(file='main.c', line=21)),
                            'message': ['__FULL_OPAQUE_BODY__']}),
                        17])]]
            encoded = json.dumps(record).encode()
            path = root/'main.c.opt-record.json.gz'; path.write_bytes(gzip.compress(encoded, mtime=0))
            reference = artifact_reference(path, 'scar.gcc-optimization-record-gzip')
            spans = [dict(id='stage:4', start_line=10, end_line=20),
                dict(id='stage:7', start_line=20, end_line=21, mapping_status='CALLER_SUPPLIED')]
            report = parse_gcc_optimization_record(reference, source_ref=source_ref, stage_spans=spans,
                compilation_directory=root)
            self.assertEqual(report['observed_wire_header']['format'], '1')
            self.assertEqual(report['summary']['pass_nodes'], 2)
            self.assertEqual(report['summary']['diagnostic_nodes'], 5)
            self.assertEqual(report['passes'][0]['children'], ['#/1/0/children/0'])
            success, failure, opaque, literal = report['diagnostics'][1:]
            self.assertEqual(success['parent'], '#/2/0')
            self.assertEqual(success['pass_ids'], ['#/1/0/children/0'])
            self.assertEqual(success['stage_ids'], ['stage:4', 'stage:7'])
            self.assertEqual(success['raw_message_pointer'], '#/2/0/children/0/message')
            self.assertEqual(success['count']['status'], 'ESTIMATED')
            self.assertEqual(failure['count']['status'], 'UNINITIALIZED')
            self.assertEqual(failure['stage_ids'], [])
            self.assertEqual(failure['location_ids'], [])  # no inherited or compiler-implementation source location
            self.assertEqual(opaque['opaque_fields']['future/kind'], '#/2/0/children/2/future~1kind')
            self.assertEqual(opaque['stage_ids'], ['stage:7'])
            self.assertEqual(literal['record_type'], 'OPAQUE_DIAGNOSTIC')
            locations = {row['id']: row for row in report['locations']}
            primary = locations[success['primary_location_ids'][0]]
            self.assertEqual(primary['stage_span_indices'], [])
            self.assertEqual(primary['mapping_status'], 'DIFFERENT_SOURCE_PATH')
            inline = locations[success['inlining_chain'][1]['location_ids'][0]]
            self.assertEqual(inline['stage_span_indices'], [0, 1])
            projected = json.dumps(report)
            self.assertNotIn('__FULL_', projected)
            self.assertFalse(report['qualifications']['compiler_counts_are_runtime_observations'])
            self.assertFalse(report['qualifications']['all_compiler_transformations_closed'])

            unresolved = parse_gcc_optimization_record(reference, source_ref=source_ref, stage_spans=spans)
            self.assertEqual(unresolved['diagnostics'][3]['stage_ids'], [])
            self.assertTrue(any(row['mapping_status'] == 'RELATIVE_PATH_WITHOUT_COMPILATION_DIRECTORY'
                for row in unresolved['locations']))
            for budget in (dict(max_compressed_bytes=reference['size_bytes']-1),
                    dict(max_decoded_bytes=len(encoded)-1), dict(max_nodes=5), dict(max_depth=2)):
                with self.assertRaises(ValueError):
                    parse_gcc_optimization_record(reference, source_ref=source_ref, **budget)
            bad = deepcopy(reference); bad['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                parse_gcc_optimization_record(bad, source_ref=source_ref)
            bad_source = deepcopy(source_ref); bad_source['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                parse_gcc_optimization_record(reference, source_ref=bad_source)
            with self.assertRaises(ValueError):
                parse_gcc_optimization_record(reference, source_ref=source_ref,
                    stage_spans=[dict(id='outside', start_line=31, end_line=31)])


if __name__ == '__main__':
    unittest.main()
