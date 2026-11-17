"""Callee sharing preserves CALL multiplicity and intermediate-value reads."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from scar.analysis.array_call_effects import derive_array_effects, derive_composite_call_effects, verify_composite_call_effects
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.pipeline_loop_frontend import (_Component, _component, CompositeInvariantLoopProof,
    FunctionGlobalGuard)
from scar.compiler.pipeline_model import RegionCall, CalleeGuard, compose_pipeline, prepare_pipeline_leaf
from scar.optimization.optimizer import optimize_region


def _composite_case(tmp_path):
    """Synthetic metadata, not an original-F4-admitted source proof."""
    path = tmp_path/'original.py'
    path.write_text('# inert source bytes for metadata receipts; never executed\n')
    digest = sha256(path.read_bytes()).hexdigest()

    def source(name, letter):
        return SourceRef(str(path),digest,name,1,0,'pycode-v2:'+letter*64)

    prepare_source, combine_source = source('prepare','a'), source('combine','b')
    prepare = ArrayRegion('prepare',prepare_source,('fixed',),(
        Expr('f','input',attrs={'name':'fixed'}),Expr('prepared','mul',('f','f'))),'prepared')
    combine = ArrayRegion('combine',combine_source,('payload','prepared'),(
        Expr('p','input',attrs={'name':'payload'}),Expr('f','input',attrs={'name':'prepared'}),
        Expr('out','add',('p','f'))),'out')
    first, second = (prepare_pipeline_leaf(optimize_region(region)) for region in (prepare,combine))
    known = {(prepare_source.path,prepare_source.code_key):_Component(first),
        (combine_source.path,combine_source.code_key):_Component(second)}
    wrapper_source = source('wrapper','c')
    wrapper_pipeline = compose_pipeline(wrapper_source,('payload','fixed'),(
        RegionCall('prepare',first,('arg:1',),'prepared',CalleeGuard(prepare_source,('prepare',))),
        RegionCall('combine',second,('arg:0','prepared'),'out',CalleeGuard(combine_source,('combine',)))),
        'out',id='pipeline:wrapper')
    wrapper = _component(wrapper_pipeline,known)
    known[wrapper_source.path,wrapper_source.code_key] = wrapper
    relay_source = source('relay','d')
    relay_pipeline = compose_pipeline(relay_source,('payload','fixed'),(
        RegionCall('first-wrapper',wrapper.execution,('arg:0','arg:1'),'middle',CalleeGuard(wrapper_source,('wrapper',))),
        RegionCall('second-wrapper',wrapper.execution,('middle','arg:1'),'out',CalleeGuard(wrapper_source,('wrapper',)))),
        'out',id='pipeline:relay')
    relay = _component(relay_pipeline,known)
    loop_source = source('loop','e')
    guards = (FunctionGlobalGuard(loop_source,'relay',relay_source,('relay',)),
        *(replace(guard,global_path=('relay',*guard.global_path)) for guard in relay.guards))
    return CompositeInvariantLoopProof('composite-invariant-loop:metadata',loop_source,
        ('initial','fixed','count'),relay.fused_body,'payload','count',
        {'payload':'initial','fixed':'fixed'},relay_source,'relay',guards,relay.compositions,())


def test_shared_callee_occurrences_preserve_direct_reads_provenance_and_unknown_scopes(tmp_path, monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('metadata query attempted compilation or native loading')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    monkeypatch.setattr('ctypes.CDLL',forbidden)
    proof = _composite_case(tmp_path)
    report = derive_composite_call_effects(proof)
    assert verify_composite_call_effects(report,proof)==report
    assert report['status']=='CONDITIONAL'
    assert report['input_writes']['status']=='CONDITIONAL_NONE'
    assert report['fresh_result']['status']=='CONDITIONAL_FRESH'
    assert len(report['callees'])==4 and len(report['calls'])==4
    assert report['summary']['source_call_templates_per_root_call']==6
    assert {row['multiplicity_per_root_call'] for row in report['calls'] if row['pipeline_id']=='pipeline:wrapper'}=={2}
    assert {row['multiplicity_per_root_call'] for row in report['calls'] if row['pipeline_id']=='pipeline:relay'}=={1}
    combined = next(row for row in report['calls'] if row['pipeline_id']=='pipeline:wrapper' and row['call_id']=='combine')
    held = next(row for row in combined['direct_reads'] if row['formal']=='prepared')
    assert held['kind']=='IMMUTABLE_CALL_RESULT_READ' and held['actual_value']=='prepared'
    assert held['input_provenance']==['fixed'] and not held['provenance_is_additional_input_read']
    assert not any(row['kind']=='CALLER_INPUT_REFERENT_READ' and row['input_provenance']==['fixed']
        for row in combined['direct_reads'])
    assert report['caller_exposed_effects']['rng_effect']['status']=='CONDITIONAL_NONE'
    assert report['caller_exposed_effects']['external_effect']['status']=='CONDITIONAL_NONE'
    assert report['caller_exposed_effects']['external_effect']['allocator_and_release_callback_effects']=='UNKNOWN'
    assert report['caller_exposed_effects']['frees']['status']=='UNKNOWN'
    assert report['caller_exposed_effects']['may_raise']['status']=='UNKNOWN'
    assert report['caller_exposed_effects']['aliases']['read_read_aliasing_allowed']
    assert {'NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS','NO_INTERSTEP_CALLBACKS_OR_HOOKS',
        'NO_FOREIGN_CONCURRENT_INPUT_WRITERS'}<=set(report['input_writes']['obligations'])
    assert not report['source_f4_replayed'] and not report['runtime_guards_discharged']
    assert not report['source_replacement_authorized'] and not report['whole_python_purity_proven']
    forged = deepcopy(report)
    next(row for row in forged['calls'] if row['pipeline_id']=='pipeline:wrapper')['multiplicity_per_root_call']=1
    with pytest.raises(ValueError,match='composition/ABI replay'):
        verify_composite_call_effects(forged,proof)
    original = proof.compositions[0].pipeline.calls[0].plan
    broken = replace(original,region=replace(original.region,nodes=(*original.region.nodes[:-1],
        replace(original.region.nodes[-1],op='opaque_protocol'))))
    unknown = derive_array_effects(broken)
    assert unknown['status']=='UNRESOLVED' and unknown['input_writes']['status']=='UNKNOWN'
    assert unknown['effects']['rng_effect']['status']=='UNKNOWN'
    assert unknown['blockers'][0]['operations'][0]['op']=='opaque_protocol'
    descriptor = replace(original.region,nodes=(original.region.nodes[0],),output=original.region.nodes[0].id)
    alias = derive_array_effects(descriptor)
    assert alias['fresh_result']['status']=='UNKNOWN' and alias['input_writes']['status']=='CONDITIONAL_NONE'
    assert alias['read_formals']==[] and alias['effects']['aliases']['result']=='ALIASES_FORMAL_INPUT_DESCRIPTOR_NOT_FRESH'
