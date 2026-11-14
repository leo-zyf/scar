"""Sourced SSE2 lane/chain layout for one exact elementwise recurrence.

Register counts are an abstract SSA storage model, not compiler spill proofs.
Canonical dependent/independent timings are effective proxies including their
loop control; they are not pure instruction latencies or whole-loop forecasts.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import math
from pathlib import Path

from scar.compiler.model import document
from scar.compiler.loop_model import loop_digest
from scar.ir.program.providers import ProviderFactReference
from scar.ir.record_codec import decode, loads
from .cost_model import _pointer
from .hardware import _target_materials
from .resource_solver import semantic_sha256


_FINITE_MASK_IMPLEMENTATION = 'SSE2_INTEGER_EXPONENT_AND_CMPEQ32_MOVEMASK_PD'


def _ratio(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def _registers(plan):
    nodes = plan.region.nodes
    constants = [node.id for node in nodes if node.op == 'const']
    arithmetic = [node for node in nodes if node.op not in ('input', 'const')]
    last = {}
    for index, node in enumerate(arithmetic):
        for operand in node.inputs:
            last[operand] = index
    last[plan.region.output] = len(arithmetic)
    live = {node.id for node in nodes if node.op == 'input'}
    peak, rows = len(live), []
    for index, node in enumerate(arithmetic):
        first = node.inputs[0]
        reuse = first if first in live and last.get(first) == index else None
        demand = len(live) + int(reuse is None)
        peak = max(peak, demand)
        rows.append(dict(expression=node.id, live_before=sorted(live),
                         reuse_first_operand=reuse, vector_registers_during_definition=demand))
        if reuse is not None:
            live.discard(reuse)
        live.add(node.id)
        for operand in set(node.inputs):
            if last.get(operand) == index:
                live.discard(operand)
    return constants, arithmetic, peak, rows


def _profile(value):
    if isinstance(value, (str, Path)):
        path = Path(value).absolute(); raw = path.read_bytes(); data = loads(raw)
        return data, dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw),
                          semantic_sha256=semantic_sha256(data))
    return deepcopy(value), None


def plan_loop_layout(loop, strategy, execution_plan, *, hardware_profile):
    """Choose SSE2 chains from declared ISA/register constraints and sourced proxies."""
    from scar.codegen.loop import _admit, layout_compiler_options

    strategy, plan = _admit(loop, strategy, execution_plan)
    profile, receipt = _profile(hardware_profile)
    if not isinstance(profile, dict):
        profile = {}
    frontier, costs, cache, contexts = [], [], {}, set()

    def issue(field, reason, cost=False):
        (costs if cost else frontier).append(dict(field=field, reason=reason))

    def evidence(records, field, expected, cost=False, finite_mask=False):
        if not isinstance(records, list) or not records:
            issue(field, 'matching retained FactReference required', cost); return False
        matches = 0
        for index, record in enumerate(records):
            try:
                reference = decode(ProviderFactReference, record); artifact = reference.artifact
                key = (artifact.uri, artifact.sha256, artifact.size_bytes, artifact.schema)
                if key not in cache:
                    if not Path(artifact.uri).is_absolute(): raise ValueError('absolute local source required')
                    raw = Path(artifact.uri).read_bytes()
                    if len(raw) != artifact.size_bytes or sha256(raw).hexdigest() != artifact.sha256:
                        raise ValueError('source SHA/size differs')
                    data = loads(raw)
                    if data.get('schema') != artifact.schema: raise ValueError('source schema differs')
                    cache[key] = data
                actual = _pointer(cache[key], reference.locator)
                if semantic_sha256(actual) != semantic_sha256(expected):
                    raise ValueError('source value/contract differs')
                if cost and contexts and key not in contexts:
                    raise ValueError('measurement source is not bound to the original calibration context')
                if finite_mask and cache[key].get('finite_mask_implementation') != _FINITE_MASK_IMPLEMENTATION:
                    raise ValueError('integer SSE2 finite-mask implementation evidence required')
                if field == 'context.evidence': contexts.add(key)
                matches += 1
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                issue(f'{field}.{index}', str(error), cost)
        return matches == len(records)

    def metric(record, field, finite_mask=False):
        if not isinstance(record, dict):
            issue(field, 'effective canonical measurement unavailable; unknown is not zero', True); return None
        value = record.get('value')
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or record.get('unit') != 'ns/vector':
            issue(field, 'finite positive ns/vector value required', True); return None
        if not evidence(record.get('evidence'), field + '.evidence', value, True, finite_mask): return None
        return Fraction(str(value))

    if (profile.get('schema'), profile.get('schema_version')) != ('scar.cpu-simd-primitives', 1):
        issue('hardware_profile.schema', 'scar.cpu-simd-primitives v1 required')
    context = profile.get('context', {}); isa = profile.get('isa', {}); registers = profile.get('register_model', {})
    binding = plan.resource_binding
    for key, expected in (('target_path', binding.target_path), ('target_sha256', binding.target_sha256),
                          ('executor_id', binding.executor_id)):
        if context.get(key) != expected: issue('context.' + key, 'must match the bound original hardware snapshot')
    measured = context.get('measured_cpu_id')
    if type(measured) is not int or measured not in binding.cpu_ids:
        issue('context.measured_cpu_id', 'measured CPU must belong to the requested allocation')
    if context.get('cpu_ids') not in (list(binding.cpu_ids), [measured]):
        issue('context.cpu_ids', 'calibration CPU context must match the allocation or measured single CPU')
    compiler = context.get('compiler', {})
    if (not isinstance(compiler.get('argv'), list) or not compiler['argv']
            or any(not isinstance(v, str) or not v for v in compiler['argv'])
            or not isinstance(compiler.get('identity'), str) or not compiler['identity']):
        issue('context.compiler', 'full sourced compiler prefix/identity required')
    if compiler.get('options') != list(layout_compiler_options(plan)):
        issue('context.compiler.options', 'must match the SSE2 backend option authority')
    evidence(context.get('evidence'), 'context.evidence', {k: v for k, v in context.items() if k != 'evidence'})
    _, _, resources, _ = _target_materials(binding.target_path)
    executor = next(resource for resource in resources if str(resource.id) == binding.executor_id)
    if context.get('cpu_model') not in executor.model_names:
        issue('context.cpu_model', 'model must match the original typed CPU executor')
    required_isa = dict(architecture='x86_64', name='SSE2_X86_64', lanes=2, vector_registers=16, supported=True)
    if any(semantic_sha256(isa.get(k)) != semantic_sha256(v) for k, v in required_isa.items()):
        issue('isa', 'only sourced, supported x86_64 SSE2 two-lane layouts are implemented')
    evidence(isa.get('evidence'), 'isa.evidence', {k: v for k, v in isa.items() if k != 'evidence'})
    evidence(registers.get('evidence'), 'register_model.evidence', {k: v for k, v in registers.items() if k != 'evidence'})
    reserve, cap = registers.get('reserved_vector_registers'), registers.get('max_vector_chains')
    if type(reserve) is not int or not 0 <= reserve < 16 or type(cap) is not int or not 1 <= cap <= 16:
        issue('register_model', 'explicit bounded reserve and chain cap required')
    constants, arithmetic, peak, live_rows = _registers(plan)
    counts = Counter(node.op for node in arithmetic)
    rates, measurements = {}, profile.get('measurements', {})
    for op in sorted(counts):
        row = measurements.get('ops', {}).get(op, {})
        dependent = metric(row.get('dependent_ns_per_vector'), 'measurements.ops.' + op + '.dependent_ns_per_vector')
        independent = metric(row.get('independent_ns_per_vector'), 'measurements.ops.' + op + '.independent_ns_per_vector')
        rates[op] = (dependent, independent)
    guard = metric(measurements.get('finite_mask', {}).get('ns_per_vector'),
                   'measurements.finite_mask.ns_per_vector', True)
    register_fit = max(0, min(cap, (16 - reserve - len(constants)) // peak)) if not frontier and peak else 0
    columns = strategy['state_shape'][1]
    geometry_fit = columns // 2
    max_fit = min(register_fit, geometry_fit)
    if not register_fit: issue('register_model.feasibility', 'no independent chain fits the declared vector-register budget')
    if not geometry_fit: issue('state_shape.columns', 'row geometry has no complete two-lane SIMD vector')
    critical, issue_work, chosen, candidates = None, None, None, []
    if max_fit and not costs:
        finish = {}
        for node in plan.region.nodes:
            finish[node.id] = (Fraction(0) if node.op in ('input', 'const') else
                               max(finish[key] for key in node.inputs) + rates[node.op][0])
        critical = finish[plan.region.output] + guard
        issue_work = sum((counts[op] * rates[op][1] for op in counts), Fraction(0)) + guard
        candidates = [(max(critical / chains, issue_work), chains) for chains in range(1, max_fit + 1)]
        chosen = min(candidates)[1]
    elif max_fit:
        chosen = max_fit
    selected = None if frontier or chosen is None else dict(
        kind='SSE2_F64_TEMPORAL_BLOCK_V1', isa='SSE2_X86_64', simd_lanes=2, vector_chains=chosen,
        tile_elements=2 * chosen, lane_mapping='CONTIGUOUS_ROW_CHAINS', scalar_tail=True)
    predicted = None if chosen is None else reserve + len(constants) + chosen * peak
    return dict(schema='scar.loop-layout', schema_version=1,
        status='PLANNED_UNDER_MODEL' if selected else 'UNRESOLVED', selected=selected, frontier=frontier, cost_frontier=costs,
        loop_sha256=loop_digest(loop), strategy_sha256=semantic_sha256(strategy),
        execution_plan_sha256=semantic_sha256(document(plan)), profile_sha256=semantic_sha256(profile),
        hardware_profile=profile if receipt is None else None, hardware_profile_receipt=receipt,
        state_shape=strategy['state_shape'], input_shapes=strategy['input_shapes'], trip_count=strategy['trip_count'],
        payload_plan_sha256=semantic_sha256(strategy['selected_memory_plan']),
        register_model=dict(vector_registers=16, reserved_vector_registers=reserve, shared_constant_vectors=len(constants),
                            per_chain_live_peak=peak, max_register_feasible_vector_chains=register_fit,
                            max_feasible_vector_chains=max_fit,
                            predicted_vector_registers=predicted, live_ranges=live_rows,
                            policy='SSE2_TWO_ADDRESS_FIRST_OPERAND_LAST_USE_REUSE_NO_COMMUTATION',
                            actual_compiler_register_allocation_verified=False, compiler_spills_verified=False),
        model=dict(kind='EFFECTIVE_CANONICAL_DEPENDENCY_THROUGHPUT_PROXY',
                   selection_policy='MIN_PROXY_THEN_FEWEST_CHAINS' if not costs else 'MAX_REGISTER_AND_GEOMETRY_FEASIBLE_CHAINS_NO_TIME_MODEL',
                   effective_dependency_proxy_ns=None if critical is None else _ratio(critical),
                   effective_issue_proxy_ns=None if issue_work is None else _ratio(issue_work),
                   candidates=[dict(vector_chains=n, proxy_ns_per_vector_iteration=_ratio(score)) for score, n in candidates],
                   includes_canonical_loop_control=True, pure_instruction_latency_proven=False,
                   finite_mask_implementation=_FINITE_MASK_IMPLEMENTATION,
                   parallel_scaling='UNKNOWN_NOT_USED', whole_loop_duration_predicted=False),
        geometry=dict(rows=strategy['state_shape'][0], columns=columns,
                      max_vector_chains_per_row=geometry_fit,
                      full_tiles_per_row=None if chosen is None else columns // (2 * chosen),
                      scalar_tail_elements_per_row=None if chosen is None else columns % (2 * chosen)),
        abstract_vector_storage_bytes_per_worker=None if predicted is None else 16 * predicted,
        scratch_budget_bytes_per_worker=256,
        scratch_budget_scope='ABSTRACT_SIMD_REGISTER_STORAGE_ONLY_NOT_NATIVE_STACK_CAP',
        actual_stack_bytes_per_worker=None, target_variants_executed=False, physical_optimality_proven=False,
        time_optimality_proven=False, deployment_authorized=False,
        runtime_guards=['X86_64_PROCESS_AND_SSE2_CPUID_ON_BOUND_WORKERS', 'NO_HIGHER_ISA_COMPILER_TARGET',
                        'ORIGINAL_LOOP_BODY_CALLEES_RANGE_SHAPE_AND_EXACT_COUNT',
                        'PACKED_INITIAL_AND_EVERY_ITERATION_FINITE_WITH_STICKY_FENV',
                        'ORIGINAL_SCALAR_TAIL_ORDER_AND_NO_FMA_OR_REASSOCIATION'],
        external_assumptions=deepcopy(strategy['external_assumptions']))


def verify_loop_layout(loop, strategy, execution_plan, layout):
    """Pure replay of profile receipts, exact body/binding and layout constraints."""
    value = loads(Path(layout).read_bytes()) if isinstance(layout, (str, Path)) else deepcopy(layout)
    if (value.get('schema'), value.get('schema_version')) != ('scar.loop-layout', 1):
        raise ValueError('versioned loop layout required')
    receipt = value['hardware_profile_receipt']
    profile = value['hardware_profile']
    if receipt is not None:
        raw = Path(receipt['path']).read_bytes()
        if len(raw) != receipt['bytes'] or sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('SIMD hardware profile receipt differs')
        profile = receipt['path']
    expected = plan_loop_layout(loop, strategy, execution_plan, hardware_profile=profile)
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('loop layout differs from sourced ISA/register/cost replay')
    return value


__all__ = ['plan_loop_layout', 'verify_loop_layout']
