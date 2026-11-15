"""Inspect cold compiled-leaf admission and one original on-demand read.

The catalog owns every directory, unit, offset and immutable byte receipt.
This query copies no body or tensor values and executes no target program.
It keeps original reader validation authoritative when a leaf is requested.
"""
import argparse
import json
from pathlib import Path
import resource
import time


def _leaf_state(catalog):
    stores=catalog.interface_store.compiled
    return {unit:dict(primitive_fact_units=sorted(material.fact_store.loaded),
        composite_fact_cached=unit in catalog.fact_store.loaded,
        **stores[unit].admission_summary())
        for unit,material in sorted(catalog.materials.items())}


def inspect_catalog_admission(catalog,out,*,probe_unit=None):
    """Inspect an original observed catalog once, with one bounded leaf query.

    An already admitted, still cold catalog avoids repeating graph admission.
    A path calls read_source_catalog exactly once. Peak RSS is the process
    high-water mark (Linux KiB), not an isolated allocation measurement.
    """
    from scar.analysis.source_declared_catalog import read_source_catalog
    from scar.analysis.compiled_local_interfaces import _counts
    from scar.ir.execution_recipe import artifact_reference
    from scar.ir.program.normal_links import UnitFacts
    from scar.ir.program.source_interfaces import UnitSourceInterfaces
    from scar.ir.program.suspension_interfaces import WitnessSource

    started=time.perf_counter();rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh catalog admission inspection output required')
    path_input=isinstance(catalog,(str,Path));admitted=time.perf_counter()
    catalog=read_source_catalog(catalog)
    admission_seconds=time.perf_counter()-admitted
    if catalog.report.get('schema')!='scar.observed-source-catalog':
        raise ValueError('original observed source catalog with compiled-leaf directory required')
    if not catalog.materials:raise ValueError('compiled-leaf catalog is empty')
    wrappers=catalog.interface_store.compiled;entries={row['unit']:row for row in catalog.report['compiled']}
    if len(entries)!=len(catalog.report['compiled']) or entries.keys()!=catalog.materials.keys() or entries.keys()!=wrappers.keys():
        raise ValueError('original material/interface directory coverage differs')
    declared=catalog.report['source_units']
    if (catalog.fact_store.report['units']!=dict(catalog.fact_store.rows)
            or len(declared)!=len(catalog.fact_store.rows)
            or set(declared)!=set(catalog.fact_store.rows)):
        raise ValueError('original fact directory differs from retained catalog')
    if any(catalog.fact_store.offsets[u]!=offset for u,offset in catalog.base.fact_store.offsets.items()):
        raise ValueError('base source fact offsets changed')

    leaves=[];expected_offset=catalog.base.fact_store.size
    for unit,material in sorted(catalog.materials.items()):
        entry=entries[unit];wrapper=wrappers[unit];facts=material.fact_store
        descriptor=material.report['tables']['facts'];ports=material.report['counts']['ports']
        if (material.unit!=unit or wrapper.material is not material or facts.unit!=unit
                or facts.size!=ports or catalog.fact_store.offsets[unit]!=expected_offset
                or catalog.fact_store.rows[unit]['facts']!=str(material.path/facts.row['facts'])
                or entry['material']['path']!=str(material.path/'result.json')
                or entry['interfaces']['path']!=str(wrapper.path/'result.json')
                or wrapper.report['material']!=entry['material']
                or wrapper.report['counts']!=dict(wrapper.declared_counts)):
            raise ValueError('original leaf identity/directory/offset/count receipt differs')
        if ports and (catalog.fact_store.locate(expected_offset)!=(unit,0)
                or catalog.fact_store.locate(expected_offset+ports-1)!=(unit,ports-1)):
            raise ValueError('original composed fact point routing differs')
        leaves.append(dict(unit=unit,material=entry['material'],interfaces=entry['interfaces'],
            facts=dict(path=str(material.path/descriptor['path']),sha256=descriptor['sha256'],
                bytes=descriptor['bytes'],decoded_bytes=descriptor['decoded_bytes'],
                decoded_sha256=descriptor['decoded_sha256']),offset=expected_offset,ports=ports,
            declared_interface_counts=dict(wrapper.declared_counts)))
        expected_offset+=ports
    if catalog.fact_store.size!=expected_offset:
        raise ValueError('composed source fact denominator changed')
    before=_leaf_state(catalog)
    if any(state['primitive_fact_units'] or state['composite_fact_cached'] or state['typed_interfaces_loaded']
            or any(state['read_counts'].values()) for state in before.values()):
        raise ValueError('freshly admitted catalog with cold compiled facts/interfaces required')
    eligible=[row for row in leaves if any(row['declared_interface_counts'][kind] for kind in ('iterations','bodies','mappings'))]
    automatic_probe=probe_unit is None
    if automatic_probe:
        if not eligible:raise ValueError('one declared nonempty interface family required for bounded probe')
        probe_unit=min(eligible,key=lambda row:(row['facts']['decoded_bytes'],row['unit']))['unit']
    if probe_unit not in entries:raise ValueError('probe unit is absent from original compiled directory')
    if not any(wrappers[probe_unit].declared_counts[k] for k in ('iterations','bodies','mappings')):
        raise ValueError('probe needs an original nonempty interface family')

    probe_started=time.perf_counter();wrapper=wrappers[probe_unit]
    facts=catalog.fact_store.load(probe_unit);typed=catalog.interface_store.load(facts)
    if (type(facts)is not UnitFacts or facts.unit!=probe_unit
            or facts is not catalog.materials[probe_unit].fact_store.loaded[probe_unit]
            or len(facts.ports)!=catalog.materials[probe_unit].fact_store.size
            or type(typed)is not UnitSourceInterfaces or typed.unit!=probe_unit
            or typed is not wrapper.loaded[probe_unit] or _counts(typed)!=dict(wrapper.declared_counts)):
        raise ValueError('on-demand original facts/interface type or coverage differs')
    witnesses=[entry.site.witness for entry in typed.iterations]
    witnesses.extend(site.witness for entry in typed.bodies for site in entry.site.suspensions)
    witnesses.extend(merge.witness for entry in typed.mappings for merge in entry.merges)
    if any(witness.unit!=probe_unit or witness.source is not WitnessSource.SEALED_COMPILED_LOCAL_MATERIAL
            or witness.template_result_sha256!=wrapper.report['material']['sha256']
            or witness.model_manifest_sha256 is not None for witness in witnesses):
        raise ValueError('on-demand original interface witness selects another material')
    probe_seconds=time.perf_counter()-probe_started
    validate_started=time.perf_counter();catalog.validate_parents();validation_seconds=time.perf_counter()-validate_started
    after=_leaf_state(catalog)
    if not after[probe_unit]['typed_interfaces_loaded'] or after[probe_unit]['read_counts']['unit_reads']!=1:
        raise ValueError('first query did not admit exactly one original typed interface unit')
    if any(after[u]!=before[u] for u in entries if u!=probe_unit):
        raise ValueError('bounded probe loaded or changed an unqueried leaf')
    report=dict(schema='scar.catalog-admission-inspection',schema_version=1,
        catalog=artifact_reference(catalog.path/'result.json','scar.observed-source-catalog'),
        original_leaf_directory=leaves,before_probe=before,after_probe=after,
        summary=dict(source_units=len(catalog.fact_store.rows),compiled_leaves=len(leaves),
            base_source_units=len(catalog.base.fact_store.rows),fact_points=catalog.fact_store.size,
            directory_units_offsets_counts_preserved=True,all_compiled_leafs_cold_before_probe=True,
            original_reader_types_and_witnesses_validated=True,queried_compiled_leafs=1,
            unqueried_compiled_leafs_still_cold=len(leaves)-1),
        probe=dict(unit=probe_unit,selection='MIN_DECLARED_FACT_DECODED_BYTES_WITH_NONEMPTY_INTERFACE_FAMILY'
            if automatic_probe else 'EXPLICIT_ORIGINAL_UNIT',
            interface_counts=_counts(typed),witnesses_checked=len(witnesses),
            witness_source=WitnessSource.SEALED_COMPILED_LOCAL_MATERIAL.value,
            material=entries[probe_unit]['material']),
        admission=dict(input='PATH'if path_input else 'ALREADY_ADMITTED_COLD_OBJECT',
            read_source_catalog_path_calls=int(path_input),graph_admission='EXISTING_EAGER_COMPILED_MATERIAL_GRAPH_UNCHANGED',
            unqueried_typed_coverage_claimed=False),
        seconds=dict(catalog_admission=admission_seconds if path_input else None,
            admitted_object_selection=admission_seconds if not path_input else None,
            probe=probe_seconds,final_parent_validation=validation_seconds,total=time.perf_counter()-started),
        process_peak_rss_kib=dict(before=rss_before,after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope='PROCESS_HIGH_WATER_MARK_NOT_EXCLUSIVE_QUERY_ALLOCATION'),
        target_executed=False,tensor_contents_recorded=False,source_bodies_copied=False,
        hardware_reprobed=False,transformation_authorized=False,first_stage_complete=False)
    out.mkdir(parents=True);(out/'result.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--probe-unit')
    args=parser.parse_args(argv)
    report=inspect_catalog_admission(args.catalog,args.out,probe_unit=args.probe_unit)
    print(json.dumps(dict(result=str(Path(args.out)/'result.json'),summary=report['summary'],
        seconds=report['seconds'],process_peak_rss_kib=report['process_peak_rss_kib']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
