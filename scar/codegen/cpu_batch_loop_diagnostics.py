"""Static, nonauthorizing instruction/row-path audit of a retained batch case.

The finite-row path audit is restricted to an even-width vectorized loop whose
alias checks fall through under the canonical separate live allocations. It
uses the retained compiler assembly and exact LLVM-MCA tool. It never executes
the target or the canonical native library and never emits selectable costs.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from scar.ir.execution_recipe import artifact_reference,read_artifact
from .cpu_fused_cost import _receipt,_require,_save
from . import cpu_instruction_cost as S
from . import cpu_parallel_instruction_cost as P


def _instructions(lines):
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith(('#','.'))]


def _mca(tool,source,out,iterations):
    _require(_receipt(Path(tool['executable']['path']))==tool['executable'],'MCA tool bytes differ')
    command=[tool['executable']['path'],'-mcpu='+tool['cpu_model'],'-iterations='+str(iterations),
        '--noalias=true','--json',str(source)]
    process=subprocess.run(command,capture_output=True,text=True)
    out.write_text(process.stdout);out.with_suffix('.stderr').write_text(process.stderr)
    _require(process.returncode==0,'static finite-path MCA failed: '+process.stderr[-600:])
    result=json.loads(process.stdout)
    return dict(command=command,input=_receipt(source),output=_receipt(out),
        summary=result['CodeRegions'][0]['SummaryView'],static_only=True,
        noalias_model_not_observed_cache_or_address_behavior=True)


def _fp_dependencies(instructions,info):
    ready={};written=set();liveins=set();records=[];store_ready=[]
    for index,(text,metadata) in enumerate(zip(instructions,info)):
        mnemonic=text.split()[0];regs=re.findall(r'%xmm\d+',text)
        destination=re.search(r',\s*(%xmm\d+)\s*$',text)
        dest=destination[1] if destination else None
        sources=regs[:-1] if mnemonic.startswith('mov') and dest else regs
        for reg in sources:
            if reg not in written:liveins.add(reg)
        finish=max((ready.get(reg,0) for reg in sources),default=0)+metadata['Latency']
        if dest:ready[dest]=finish;written.add(dest)
        if metadata['mayStore']:store_ready.append(finish)
        records.append(dict(instruction=index,text=text,latency=metadata['Latency'],
            read_fp_registers=sources,write_fp_register=dest,within_block_ready_cycle=finish))
    return dict(records=records,fp_read_before_write=sorted(liveins),fp_written=sorted(written),
        possible_fp_loop_carried_registers=sorted(liveins&written),
        within_iteration_fp_register_dependency_to_store=max(store_ready,default=0),
        model_scope='SINGLE_ITERATION_REGISTER_DATAFLOW_LATENCIES_NOT_PER_ELEMENT_SERIAL_TIME',
        memory_alias_dependencies_or_cache_latency_proven=False)


def diagnose_batch_loop(sample_result,configuration,out):
    sample_ref=artifact_reference(sample_result,'scar.cpu-batch-context-result');sample=read_artifact(sample_ref)
    model=read_artifact(sample['fused_model']);raw=read_artifact(model['raw_observations'])
    pack=read_artifact(model['pack']);analysis_all=read_artifact(pack['instruction_analysis'])['analyses']
    row=next(r for r in raw['observations'] if r['id']==configuration)
    case=pack['registry'][row['case_index']];analysis=analysis_all[str(row['kernel_index'])]
    selected=analysis['vector'];_require(selected is not None and row['cols']%2==0,'even vector path audit required')
    assembly=Path(pack['assembly']['path']);_require(_receipt(assembly)==pack['assembly'],'assembly bytes differ')
    lines=P._lex_assembly(assembly.read_text()).splitlines();name=case['id']+'_g'+str(row['groups'])
    start=next(i for i,l in enumerate(lines) if l.startswith('_ZN') and name in l and l.endswith(':'))
    end=next(i for i in range(start+1,len(lines)) if lines[i].startswith('\t.size'))
    labels={line.strip()[:-1]:i for i,line in enumerate(lines[start:end],start)
        if re.match(r'^\.L\d+:$',line.strip())}
    backward=[]
    for i,line in enumerate(lines[start:end],start):
        branch=re.match(r'\s*j\w+\s+(\.L\d+)',line)
        if branch and branch[1] in labels and labels[branch[1]]<i:
            first=labels[branch[1]]
            backward.append(dict(label=branch[1],start=first+1,end=i+1,
                instruction_count=len(_instructions(lines[first:i+1])),
                contains_selected_vector=first+1<selected['start'] and i+1>selected['end']))
    outers=[r for r in backward if r['contains_selected_vector']]
    outer=min(outers,key=lambda r:r['end']-r['start'])
    before=lines[outer['start']-1:selected['start']-1]
    inner=lines[selected['start']-1:selected['end']]
    # The retained GCC route after the vector inner loop checks cols == cols&~1.
    # For the declared even shape its je skips the single scalar-tail block.
    bridge=[];jump_target=None
    for line in lines[selected['end']:outer['end']]:
        bridge.append(line)
        jump=re.match(r'\s*je\s+(\.L\d+)',line)
        if jump:
            jump_target=jump[1];break
    _require(jump_target in labels and labels[jump_target]>selected['end'],
        'unsupported even-row continuation CFG')
    after=lines[labels[jump_target]:outer['end']]
    _require(not any(re.search(r'\bcall\w*\b',line) for line in before+inner+bridge+after),
        'opaque call on audited row path')
    block=read_artifact(selected['block']);mca=json.loads(Path(block['result']['path']).read_text())['CodeRegions'][0]
    _require(_receipt(Path(block['result']['path']))==block['result'],'original MCA bytes differ')
    inner_instructions=_instructions(inner);iterations=row['cols']//2
    path=[*_instructions(before),*(instruction for _ in range(iterations) for instruction in inner_instructions),
        *_instructions(bridge),*_instructions(after)]
    out=Path(out).resolve();_require(not out.exists(),'fresh loop diagnostic path required');out.mkdir(parents=True)
    path_source=out/'one-even-row.s';path_source.write_text('\n'.join(path)+'\n')
    finite_inner=_mca(model['tool'],Path(block['input']['path']),out/'finite-inner.mca.json',iterations)
    finite_row=_mca(model['tool'],path_source,out/'finite-row.mca.json',1)
    row_count=(row['rows']+row['workers']-1)//row['workers']
    coefficient=model['coefficients']['effective_ns_per_mca_cycle']
    value=dict(schema='scar.cpu-batch-static-row-path-diagnostics',schema_version=1,
        implementation=_receipt(Path(__file__)),sample=sample_ref,model=sample['fused_model'],pack=model['pack'],
        configuration=row,case=case,source=pack['source'],assembly=pack['assembly'],analysis=analysis,
        cfg=dict(function=name,function_start_line=start+1,function_end_line=end+1,backedges=backward,
            selected_outer_row_backedge=outer,even_tail_skip_target=jump_target,
            prefix_source_range=[outer['start'],selected['start']-1],
            suffix_source_range=[labels[jump_target]+1,outer['end']],
            row_prefix_instruction_count=len(_instructions(before)),
            inner_iteration_instruction_count=len(inner_instructions),
            row_bridge_instruction_count=len(_instructions(bridge)),
            row_suffix_instruction_count=len(_instructions(after)),
            route_conditions=['EVEN_COLUMN_COUNT','NONALIAS_LIVE_INPUT_OUTPUT_AND_ARGS_ALLOCATIONS',
                'NONFINAL_ROW_CONTINUATION_TO_LARGEST_OUTER_BACKEDGE'],
            route_observed=False,full_dynamic_cfg_proven=False),
        original_inner_mca_summary=mca['SummaryView'],
        inner_memory_instruction_counts=dict(loads=sum(r['mayLoad'] for r in mca['InstructionInfoView']['InstructionList']),
            stores=sum(r['mayStore'] for r in mca['InstructionInfoView']['InstructionList']),
            stack_references=selected['stack_memory_references']),
        dependencies=_fp_dependencies(inner_instructions,mca['InstructionInfoView']['InstructionList']),
        finite_inner=finite_inner,finite_row=finite_row,
        diagnostic_arithmetic=dict(largest_logical_rows=row_count,vector_iterations_per_row=iterations,
            original_inner_only_cycle_demand=S._cycle_demand(analysis,row['rows'],row['cols'],row['workers']),
            finite_inner_cycles_all_rows=finite_inner['summary']['TotalCycles']*row_count,
            finite_nonfinal_row_cycles_all_rows=finite_row['summary']['TotalCycles']*row_count,
            outer_path_and_finite_pipeline_delta_cycles=(finite_row['summary']['TotalCycles']-
                iterations*selected['block_throughput_cycles'])*row_count,
            original_effective_ns_per_mca_cycle=coefficient,
            outer_path_and_finite_pipeline_delta_ns_under_original_mapping=(finite_row['summary']['TotalCycles']-
                iterations*selected['block_throughput_cycles'])*row_count*coefficient,
            new_cost_or_holdout_admission=False),
        omitted_measurements=['PER_WORKER_TIMES_AND_BARRIER_ARRIVALS','ACTUAL_BASE_ADDRESSES_AND_PAGE_OFFSETS',
            'CACHE_MISSES_AND_COHERENCE','RUNTIME_TAKEN_VECTOR_OR_ALIAS_FALLBACK_PATH'],
        limitations=['STATIC_MCA_USES_ITS_OWN_CACHE_AND_NOALIAS_ASSUMPTIONS',
            'ROW_ROUTE_AND_FINAL_ROW_CONTROL_ARE_EXPLICIT_ABSTRACT_PATH_CONDITIONS',
            'ALREADY_REVEALED_HOLDOUT_IS_DIAGNOSTIC_ONLY_NOT_NEW_BLIND_VALIDATION'],
        target_workload_executed=False,canonical_probe_executed=False,target_variants_ranked=False,
        numerical_model_refitted=False,cost_selection_authorized=False)
    _save(out/'result.json',value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',required=True);parser.add_argument('--configuration',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args(argv)
    result=diagnose_batch_loop(args.sample,args.configuration,args.out)
    print(json.dumps(result['diagnostic_arithmetic'],sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
