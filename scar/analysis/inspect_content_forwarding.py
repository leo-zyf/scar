"""Run the builtin-slot theorem pass on the original shared program graph."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import time

from scar.analysis.source_content_forwarding import SourceContentForwardingProvider
from scar.analysis.normal_output import write_table
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program.dependencies import DependencyKind as K,DependencyMeaning as Meaning,DependencyStrength as S


def inspect_content_forwarding(view,out,*,program=None):
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh source content proof report required')
    out.mkdir(parents=True);started=time.perf_counter();units={}
    for unit in sorted(view.catalog.fact_store.rows):
        state=view.state(unit);state.content()
        provider=SourceContentForwardingProvider(state.content_lineage).discover()
        folder=out/unit;folder.mkdir()
        # The parent access table is stored once: proof ordinals cite this
        # session's actual demand order rather than a guessed canonical order.
        tables=dict(content_accesses=write_table(folder/'content_accesses.jsonl.gz',provider.content.accesses),
            proofs=write_table(folder/'proofs.jsonl.gz',provider.proofs.values()),
            refusals=write_table(folder/'refusals.jsonl.gz',provider.refusals.values()),
            boundaries=write_table(folder/'boundaries.jsonl.gz',provider.discovery_boundaries.values()))
        queries=[]
        for proof in provider.proofs.values():
            source=view.source_port(proof.payload);target=view.source_port(proof.result)
            forward=view.dependencies.outgoing(source,kinds=(K.PROJECTION,)).edges
            matches=[e for e in forward if e.target==target and e.meaning is Meaning.REFERENCE
                and e.strength is S.MAY and e.conditions]
            if not matches or any(e not in view.dependencies.incoming(target,kinds=(K.PROJECTION,)).edges for e in matches):
                raise ValueError('original forwarding proof is not a bidirectional public graph edge')
            queries.append(dict(rule=proof.rule,kind=proof.producer.kind.value,
                static_facts=list(proof.static_facts),runtime_checks=list(proof.runtime_checks),
                external_assumptions=list(proof.external_assumptions),public_edges=len(matches)))
        provider.check()
        units[unit]=dict(summary=provider.summary(),tables=tables,public_queries=queries,
            content_access_refs_resolve_to_saved_parent_table=True)
    from scar.analysis import source_content_forwarding,source_content_lineage
    from scar.ir.program import source_content_forwarding as types
    pins={name:sha256(Path(module.__file__).read_bytes()).hexdigest()for name,module in (
        ('reporter',__import__(__name__,fromlist=['__file__'])),('theorem',source_content_forwarding),
        ('types',types),('lineage',source_content_lineage))}
    result=dict(schema='scar.source-content-forwarding-report',schema_version=1,
        catalog=artifact_reference(view.catalog.path/'result.json',view.catalog.report['schema']),
        program=None if program is None else artifact_reference(Path(program)/'manifest.json','scar.program-build'),
        pins=pins,units=units,seconds=time.perf_counter()-started,
        automatic_discovery=True,target_executed=False,unconditional_transformations=0,
        logical_version_equality_proven=False,first_stage_complete=False)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    from scar.analysis.build import BuildSession
    from scar.analysis.program_view import ProgramView
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--build');source.add_argument('--catalog')
    parser.add_argument('--out',required=True);args=parser.parse_args(argv)
    with BuildSession(args.build)if args.build else nullcontext(None)as session:
        view=session.view if session is not None else ProgramView(args.catalog)
        result=inspect_content_forwarding(view,args.out,program=args.build)
    print(json.dumps(dict(units=len(result['units']),seconds=result['seconds'],
        conditional_proofs=sum(r['summary']['conditional_reference_proofs']for r in result['units'].values()),
        refused_pairs=sum(r['summary']['refusals']for r in result['units'].values()),
        target_executed=False,unconditional_transformations=0)))
    return 0


if __name__=='__main__':raise SystemExit(main())
