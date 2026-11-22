"""Independently replay a retained ordered activation publication.

Stored tables are compared with original source-derived interfaces in a fresh
BuildSession. Byte receipts alone never authorize source connections, runtime
invocations, content equality or transformations.
"""
import argparse
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import time

from scar.analysis.build import BuildSession
from scar.analysis.inspect_ordered_activations import _pins,_rows,validate_public_interfaces
from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
from scar.analysis.source_call_view import SourceFunctionCallProvider
from scar.artifacts import read_record_table as _table
from scar.analysis.stateful_dependency_view import StatefulDependencyProvider,read_stateful_graph
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.record_codec import decode


SCHEMA='scar.ordered-activation-independent-replay'
_PUBLICATION_SCHEMA='scar.ordered-activation-interfaces'
_FALSE_FLAGS=('independent_cold_semantic_rederivation','iterations_unrolled',
    'global_logical_versions_proven','transformation_authorized','target_executed')
_ZERO_FIELDS=('source_bodies_copied','runtime_invocations_created')


def _header(report):
    if (report.get('schema'),report.get('schema_version'))!=(_PUBLICATION_SCHEMA,1):
        raise ValueError('unsupported ordered activation publication')
    if report.get('pins')!=_pins():raise ValueError('ordered activation producer changed')
    if report.get('typed_table_readback')is not True:
        raise ValueError('ordered activation typed publication readback is absent')
    if any(report.get(name)is not False for name in _FALSE_FLAGS):
        raise ValueError('ordered activation publication authority flags differ')
    if any(type(report.get(name))is not int or report[name]!=0 for name in _ZERO_FIELDS):
        raise ValueError('ordered activation publication copied bodies or runtime invocations')
    for name,schema in (('program','scar.program-build'),('original_region','scar.stateful-array-region'),
            ('shared_body','scar.stateful-public-graph')):
        if type(report.get(name))is not dict or report[name].get('schema')!=schema:
            raise ValueError('ordered activation lacks original '+name+' reference')
        read_artifact(report[name])


def _read_tables(path,report,provider):
    rows=_rows(provider)
    if set(report.get('tables',()))!=set(rows):
        raise ValueError('ordered activation table domain differs from its original interfaces')
    absent=object();counts={}
    for name,(cls,expected)in rows.items():
        receipt=report['tables'][name]
        if receipt.get('path')!=name+'.jsonl.gz':
            raise ValueError('ordered activation table is outside its declared local table: '+name)
        count=0
        for wire,original in zip_longest(_table(path,receipt),expected,fillvalue=absent):
            if wire is absent or original is absent or decode(cls,wire)!=original:
                raise ValueError('ordered activation '+name+' differs from its original source derivation')
            count+=1
        counts[name]=count
    return counts


def read_ordered_activations(path):
    """Cold-read a publication directory or result.json; return a new receipt.

    This function owns and closes its fresh source session. The publication is
    never edited. Its source-replayable profile is checked, without importing
    the target program or asserting whole-program modeling completeness.
    """
    started=time.perf_counter();path=Path(path).resolve()
    report_path=path if path.is_file()else path/'result.json'
    path=report_path.parent
    publication=artifact_reference(report_path,_PUBLICATION_SCHEMA)
    report=read_artifact(publication);_header(report);pins=dict(report['pins'])
    replayer_sha256=sha256(Path(__file__).read_bytes()).hexdigest()
    program_path=Path(report['program']['uri'])
    if program_path.name!='manifest.json':raise ValueError('original build reference is not its manifest')
    with BuildSession(program_path.parent)as session:
        if artifact_reference(session.path/'manifest.json','scar.program-build')!=report['program']:
            raise ValueError('independent session opened a different original program build')
        ordered=StatefulDependencyProvider(session.view,report['original_region']['uri'])
        if ordered.reference!=report['original_region']:
            raise ValueError('independent ordered source replay uses another original region')
        body_report,_=read_stateful_graph(Path(report['shared_body']['uri']).parent,session.view,_provider=ordered)
        if body_report['original_region']!=report['original_region']:
            raise ValueError('shared ordered body cites another original region')
        providers={name:p for name,p in session.session.providers.items()if type(p)is SourceFunctionCallProvider}
        provider=OrderedRegionActivationProvider(ordered,providers)
        if provider.summary()!=report['summary']:
            raise ValueError('ordered activation summary differs from independently replayed original interfaces')
        counts=_read_tables(path,report,provider)
        validation=validate_public_interfaces(provider)
        if validation!=report['validation']:
            raise ValueError('ordered activation endpoint validation differs from independent public queries')
        provider.check()
        # All references and producing rules are checked again after semantic
        # replay; a changed parent cannot survive by keeping a stale report.
        if pins!=_pins()or sha256(Path(__file__).read_bytes()).hexdigest()!=replayer_sha256:
            raise ValueError('ordered activation/replay rules changed during independent read')
        for reference in (publication,report['program'],report['original_region'],report['shared_body']):
            read_artifact(reference,json_document=False)
        summary=provider.summary()
    return dict(schema=SCHEMA,schema_version=1,publication=publication,
        program=report['program'],original_region=report['original_region'],shared_body=report['shared_body'],
        producer_pins=pins,replayer_sha256=replayer_sha256,table_counts=counts,summary=summary,
        validation=validation,independent_cold_semantic_rederivation=True,
        publication_declared_independent_cold_semantic_rederivation=False,
        original_body_bytes_copied=0,iterations_unrolled=False,runtime_invocations_created=0,
        global_logical_versions_proven=False,transformation_authorized=False,target_executed=False,
        seconds=time.perf_counter()-started)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication',help='ordered interfaces directory or result.json')
    parser.add_argument('--out',help='fresh independent replay receipt JSON path')
    args=parser.parse_args(argv);result=read_ordered_activations(args.publication)
    if args.out:
        output=Path(args.out).absolute();output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x')as stream:
            stream.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(schema=result['schema'],tables=result['table_counts'],validation=result['validation'],
        seconds=result['seconds'],target_executed=False),sort_keys=True))
    return 0


__all__=['read_ordered_activations']


if __name__=='__main__':raise SystemExit(main())
