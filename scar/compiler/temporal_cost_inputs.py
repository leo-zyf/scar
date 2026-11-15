"""Project retained CPU observations into independent loop-cost fact shards.

The caller owns original source/proof admission. This module routes versioned
calibration formats; it neither executes a target nor invents missing costs.
"""
import json
from pathlib import Path


def project_loop_calibrations(domain,calibrations,out):
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.compiler.calibrate_temporal_cpu import project_temporal_resource_fact_batch
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh loop cost input publication required')
    out.mkdir(parents=True);facts=[];sources=[];batches=[];skipped=[];materials=[]
    composite=domain['schema']=='scar.composite-temporal-cpu-domain'
    for value in calibrations:
        path=Path(value).resolve()
        if path.is_dir():path=path/'result.json'
        document=json.loads(path.read_bytes());schema=document.get('schema')
        reference=artifact_reference(path,schema);read_artifact(reference)
        sources.append(reference)
        materials.append((path,document,reference))
    base_calibrations=[path for path,document,_ in materials
        if document.get('schema')=='scar.temporal-cpu-resource-calibration']
    for index,(path,document,reference) in enumerate(materials):
        schema=document['schema']
        if schema=='scar.temporal-resource-calibration':
            facts.append(path);continue
        if schema=='scar.temporal-cpu-resource-calibration':
            batch=project_temporal_resource_fact_batch(domain,path,out/('calibration-'+str(index)))
            facts.extend(Path(row['observations']['uri']) for row in batch['candidates'])
            batches.append(artifact_reference(out/('calibration-'+str(index))/'result.json',batch['schema']))
            continue
        # Additional typed calibration protocols are handled below, never
        # classified by a program, source filename or mathematical variable.
        if schema=='scar.composite-temporal-boundary-calibration':
            if not composite:
                skipped.append(dict(calibration=reference,reason='COMPOSITE_BOUNDARY_INAPPLICABLE_TO_DIRECT_LOOP'))
                continue
            from scar.compiler.calibrate_temporal_cpu import project_composite_boundary_facts
            projected=project_composite_boundary_facts(domain,path,out/('calibration-'+str(index)),
                base_calibrations=base_calibrations)
            for row in projected['candidates']:facts.append(Path(row['observations']['uri']))
            batches.append(artifact_reference(out/('calibration-'+str(index))/'result.json',projected['schema']))
            continue
        raise ValueError('independent supported loop resource calibration required: '+str(schema))
    for reference in sources:read_artifact(reference,json_document=False)
    report=dict(schema='scar.loop-cost-inputs',schema_version=1,domain_id=domain['id'],
        calibrations=sources,batches=batches,skipped=skipped,
        facts=[artifact_reference(path,'scar.temporal-resource-calibration') for path in facts],
        source_independent_projections=True,target_program_executed=False)
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return facts,artifact_reference(out/'result.json',report['schema'])
