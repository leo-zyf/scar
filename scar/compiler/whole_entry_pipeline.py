"""Replay an explicit entry optimization boundary on the build main chain."""
from hashlib import sha256
import json
from pathlib import Path
from subprocess import CalledProcessError

from scar.compiler.operator_stack import verify_stack_components_once
from scar.ir.execution_recipe import artifact_reference, read_artifact


REQUEST = 'scar.whole-entry-request'
DOMAIN = 'scar.whole-entry-option-domain'
SELECTION = 'scar.whole-entry-main-chain'
LEGALITIES = 'scar.whole-entry-source-legalities'
COST_INPUTS = 'scar.whole-entry-cost-inputs'
STACK_TARGET = 'scar.whole-entry-stack-target'
OBSERVED_COSTS = 'scar.whole-entry-observed-costs'
OPTION_SPACES = ('DISCOVER_ALL', 'EXACT_REGISTERED_SET', 'EXACT_GENERATED_FAMILY')
FIXED_GRID_CPU = 'ATEN_FIXED_GRID_CPU_FILE_ENTRY'
NVIDIA_HPC = 'NVIDIA_HPC'
LIBTORCH = 'LIBTORCH_ATEN'
_ROUTE_STACKS = {'ATEN_SOURCE_ENTRY': LIBTORCH, 'PARAMETERIZED_SPARSE': 'SPARSE_DIRECT',
                 'NVHPC_SOURCE_ENTRY': NVIDIA_HPC, 'NVHPC_SOURCE_DEVICE_ENTRY': NVIDIA_HPC}


def _write(path, value):
    with path.open('x') as output:
        json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n')
    return artifact_reference(path, value['schema'])


def read_request(value):
    ref = value if isinstance(value, dict) else artifact_reference(Path(value).resolve(), REQUEST)
    document = read_artifact(ref)
    required = {'schema', 'schema_version', 'argv', 'boundary', 'registered_implementations', 'original_cost'}
    optional = {'option_space', 'audit_retained_validation', 'variant_source', 'operator_stack', 'operator_stack_lock',
                'numerical_contract', 'sgemm_observations', 'search_cpu_cohorts', 'nvhpc_device_profile', 'tensor_flow',
                'cpu_provider_catalog', 'cpu_paired_observation', 'cpu_stream_capture', 'use_captured_launch'}
    if (set(document) - optional != required or document['schema'] != REQUEST
            or type(document['schema_version']) is not int or document['schema_version'] not in (1, 2)
            or document['boundary'] != 'PROCESS_LAUNCH_THROUGH_EXIT' or type(document['argv']) is not list
            or any(type(arg) is not str or '\x00' in arg for arg in document['argv'])
            or type(document['registered_implementations']) is not list):
        raise ValueError('exact original entry argv and whole-process evidence request required')
    if (document.get('option_space', 'DISCOVER_ALL') not in OPTION_SPACES
            or type(document.get('audit_retained_validation', False)) is not bool
            or 'operator_stack' in document and document['operator_stack'] not in (NVIDIA_HPC, LIBTORCH)):
        raise ValueError('unrecognized whole-entry option space, audit mode or operator stack')
    if 'operator_stack_lock' in document:
        stack = document['operator_stack_lock']
        if (type(stack) is not dict or stack.get('schema') != 'scar.operator-stack'
                or document.get('operator_stack', NVIDIA_HPC) != NVIDIA_HPC):
            raise ValueError('the NVIDIA stack policy requires an exact SDK lock reference')
    if 'numerical_contract' in document:
        from scar.compiler.compile_exported_nvhpc import _contract

        contract = document['numerical_contract']
        if type(contract) is not dict or contract != _contract(contract.get('atol'), contract.get('rtol')):
            raise ValueError('a declared entry numerical contract is required')
    variant_source = document.get('variant_source')
    if variant_source is not None:
        if (type(variant_source) is not dict or set(variant_source) != {'family', 'proof'}
                or variant_source['family'] != FIXED_GRID_CPU):
            raise ValueError('known whole-entry variant family and its pinned source proof required')
        read_artifact(variant_source['proof'])
    space = document.get('option_space', 'DISCOVER_ALL')
    infer_nvhpc = _infer_nvhpc(document)
    infer_catalog = _infer_cpu_catalog(document)
    if 'cpu_provider_catalog' in document:
        from scar.compiler.compile_cpu_entry import CATALOG_SCHEMA

        catalog = document['cpu_provider_catalog']
        if not infer_catalog or type(catalog) is not dict or catalog.get('schema') != CATALOG_SCHEMA:
            raise ValueError('a CPU provider catalog requires an unregistered CPU-only entry domain')
        if 'numerical_contract' in document:
            raise ValueError('CPU catalog entries use the provider’s declared numerical contract')
        read_artifact(catalog)
    if 'cpu_paired_observation' in document:
        from scar.compiler.cpu_entry_replay import SCHEMA as CPU_REPLAY

        observation = document['cpu_paired_observation']
        if not infer_catalog or type(observation) is not dict or observation.get('schema') != CPU_REPLAY:
            raise ValueError('a retained CPU observation requires its original provider catalog')
        paired = read_artifact(observation)
        if paired['capture'] != document.get('cpu_stream_capture'):
            raise ValueError('a retained CPU observation must use the declared stream capture')
    if 'cpu_stream_capture' in document:
        from scar.analysis.scalar_recursive_driver import CAPTURE

        capture = document['cpu_stream_capture']
        if not infer_catalog or type(capture) is not dict or capture.get('schema') != CAPTURE:
            raise ValueError('a captured CPU stream requires its original provider catalog')
        read_artifact(capture)
    if 'use_captured_launch' in document:
        value = document['use_captured_launch']
        if (type(value) is not bool or not infer_catalog
                or not ({'cpu_stream_capture', 'cpu_paired_observation'} & document.keys())):
            raise ValueError('captured launch projection requires an original CPU stream or paired observation')
    if 'search_cpu_cohorts' in document and (not infer_nvhpc or type(document['search_cpu_cohorts']) is not bool):
        raise ValueError('CPU cohort search requires a source-derived NVIDIA domain')
    if 'sgemm_observations' in document:
        from scar.optimization.cpu_sgemm_probe import SCHEMA as SGEMM_OBSERVATION

        observations = document['sgemm_observations']
        if (not infer_nvhpc or type(observations) is not list
                or any(type(row) is not dict or row.get('schema') != SGEMM_OBSERVATION for row in observations)):
            raise ValueError('CPU SGEMM observations require a source-derived NVIDIA domain')
        for observation in observations:
            read_artifact(observation)
    if 'nvhpc_device_profile' in document:
        from scar.compiler.nvhpc_device_entry_variant import PROFILE_SCHEMAS

        profile = document['nvhpc_device_profile']
        if not infer_nvhpc or type(profile) is not dict or profile.get('schema') not in PROFILE_SCHEMAS:
            raise ValueError('a device profile requires the source-derived NVIDIA entry domain')
        read_artifact(profile)
    if 'tensor_flow' in document:
        from scar.analysis.nvhpc_source_effects import TENSOR_FLOW

        flow = document['tensor_flow']
        if not infer_nvhpc or type(flow) is not dict or flow.get('schema') != TENSOR_FLOW:
            raise ValueError('Tensor-flow evidence requires an observed source-derived NVIDIA entry')
        read_artifact(flow)
    if ((space == 'EXACT_GENERATED_FAMILY' and
            ((variant_source is None and not (infer_nvhpc or infer_catalog)) or document['registered_implementations'])) or
            (space == 'EXACT_REGISTERED_SET' and variant_source is not None)):
        raise ValueError('finite option policy and variant provenance must agree')
    candidates = {}
    for item in document['registered_implementations']:
        if type(item) is not dict or set(item) != {'route', 'artifact', 'validation', 'cost'}:
            raise ValueError('each registered implementation must cite its route, artifact and optional evidence')
        if type(item['route']) is not str or not item['route']:
            raise ValueError('registered delivery route must be named')
        artifact = item['artifact']
        read_artifact(artifact)
        key = json.dumps((item['route'], artifact['sha256']), separators=(',', ':'), allow_nan=False)
        candidate_id = 'native:'+sha256(key.encode()).hexdigest()
        if candidate_id in candidates:
            raise ValueError('duplicate registered native implementation')
        candidates[candidate_id] = item
    return ref, document, candidates


def _infer_nvhpc(request):
    return ('operator_stack_lock' in request and request.get('variant_source') is None
            and not request['registered_implementations']
            and request.get('option_space') != 'EXACT_REGISTERED_SET')


def _infer_cpu_catalog(request):
    return ('cpu_provider_catalog' in request and request.get('variant_source') is None
            and 'operator_stack_lock' not in request and 'operator_stack' not in request
            and request['schema_version'] == 1 and not request['registered_implementations']
            and request.get('option_space') != 'EXACT_REGISTERED_SET')


def _scope(plan, request):
    from scar.runtime.caller_pipeline_deployment import entry_context, environment_sha256

    manifest = read_artifact(plan['program'])
    context = entry_context(manifest['entry'], request['argv'])
    target = plan['hardware']
    scope = dict(context, boundary=request['boundary'], entry=context['entry_path'],
        target_sha256=target['sha256'],
        target=dict(uri=target['path'], sha256=target['sha256'], size_bytes=target['bytes'],
                    schema='scar.target-hardware-graph'),
        option_space=request.get('option_space', 'DISCOVER_ALL'),
        native_output_contract='STANDALONE_CPP',
        program_sha256=plan['program']['sha256'], source_admission_sha256=plan['source_admission']['sha256'],
        memory_budget_bytes=plan['memory_budget_bytes'], input_shapes=plan['input_shape_constraints'],
        environment_sha256=environment_sha256())
    if plan.get('entry_obligations') is not None:
        scope['entry_obligations_sha256'] = plan['entry_obligations']['sha256']
    if request['schema_version'] == 2 or 'operator_stack' in request or 'operator_stack_lock' in request:
        scope['operator_stack'] = request.get('operator_stack', NVIDIA_HPC)
    if 'numerical_contract' in request:
        scope['numerical_contract'] = request['numerical_contract']
    if request.get('search_cpu_cohorts', False):
        scope['search_cpu_cohorts'] = True
    if 'nvhpc_device_profile' in request:
        scope['nvhpc_device_profile'] = request['nvhpc_device_profile']
    if 'tensor_flow' in request:
        scope['tensor_flow'] = request['tensor_flow']
    for key in ('cpu_provider_catalog', 'cpu_paired_observation', 'cpu_stream_capture'):
        if key in request:
            scope[key] = request[key]
    if request.get('use_captured_launch', False):
        from scar.compiler.cpu_entry_launch_scope import project_cpu_entry_launch_scope

        scope.update(project_cpu_entry_launch_scope(scope, stream_capture=request.get('cpu_stream_capture'),
                                                    paired_observation=request.get('cpu_paired_observation')))
    return scope


def _candidate(id_, row):
    return dict(id=id_, route=row['route'], artifact=row['artifact'], validation=row['validation'])


def _stack_target(scope, reference, *, device=None):
    from scar.analysis.nvidia_stack_target import probe_nvidia_target, verify_nvidia_target
    from scar.compiler.operator_stack import load_operator_stack

    stack, document = load_operator_stack(reference)
    if device is None:
        device = probe_nvidia_target(target=scope['target'])
    else:
        verify_nvidia_target(device, target=scope['target'])
    libraries = document['libraries']
    visible = device['status'] == 'DEVICE_QUERY_SUCCEEDED'
    bound = [row['ordinal'] for row in device['visible_devices']
             if row['retained_target']['status'] == 'UUID_AND_PCI_MATCH']
    return dict(schema=STACK_TARGET, schema_version=1, stack=stack, target=scope['target'],
        device=device, cpu_blas_installed='cpu_blas' in libraries,
        cuda_provider_installed={'cudart', 'cublas', 'cublaslt'} <= libraries.keys(),
        cuda_device_evidence='TARGET_DEVICE_BOUND' if bound else 'VISIBLE_DEVICE_UNBOUND' if visible else device['status'],
        bound_cuda_ordinals=bound, allocation_or_performance_qualified=False)


def _discover(scope, request):
    if _infer_nvhpc(request):
        from scar.compiler.nvhpc_entry_variant import derive_nvhpc_entries

        cpu = derive_nvhpc_entries(scope, request.get('sgemm_observations', []))
        if 'nvhpc_device_profile' not in request:
            return cpu
        from scar.compiler.nvhpc_device_entry_variant import derive_nvhpc_device_entry

        device = derive_nvhpc_device_entry(scope, request['nvhpc_device_profile'])
        return dict(cpu, options=[*cpu['options'], *device['options']], device_discovery=device,
                    coverage_frontier=[*cpu['coverage_frontier'], *device['coverage_frontier']])
    if _infer_cpu_catalog(request):
        from scar.compiler.cpu_catalog_entry_variant import derive_cpu_catalog_entry

        return derive_cpu_catalog_entry(scope, request['cpu_provider_catalog'])
    source = request.get('variant_source')
    if source is None:
        return None
    from scar.compiler.aten_entry_variants import derive_aten_entry_variants

    return derive_aten_entry_variants(scope, source['proof'])


def _check_operator_stack(scope, stack, status, reason, *, reference=None):
    if 'operator_stack' not in scope or status != 'LEGAL':
        return status, reason
    if stack != scope['operator_stack']:
        return 'INELIGIBLE', 'OPERATOR_STACK_DIFFERS_FROM_WHOLE_ENTRY_POLICY'
    if stack == NVIDIA_HPC:
        target = scope.get('stack_target')
        if target is None or reference != read_artifact(target)['stack']:
            return 'INELIGIBLE', 'OPERATOR_STACK_LOCK_DIFFERS_FROM_ENTRY_TARGET'
    return status, reason


def _legalities(candidates, scope, discovery):
    from scar.optimization.resource_solver import semantic_sha256

    scope_sha = semantic_sha256(scope)
    facts = []
    if candidates:
        from scar.compiler.entry_delivery import qualify_entry_delivery

        for id_, item in sorted(candidates.items()):
            qualification = qualify_entry_delivery(scope, _candidate(id_, item))
            status, reason = 'UNRESOLVED', qualification['reason']
            if qualification['status'] == 'QUALIFIED':
                status, reason = 'LEGAL', 'STANDALONE_CPP_ARTIFACT_REPLAYED'
                if qualification['evidence']['python_runtime_required']:
                    status, reason = 'INELIGIBLE', 'NATIVE_ROUTE_REQUIRES_PYTHON_AT_RUNTIME'
            evidence = qualification.get('evidence') or {}
            status, reason = _check_operator_stack(scope, _ROUTE_STACKS.get(item['route']), status, reason,
                                                   reference=evidence.get('operator_stack'))
            if item['route'] in ('NVHPC_SOURCE_ENTRY', 'NVHPC_SOURCE_DEVICE_ENTRY') and status == 'LEGAL':
                status, reason = 'UNRESOLVED', 'ORIGINAL_IMPORT_INITIALIZATION_EFFECTS_OPEN'
            facts.append(dict(option_id=id_, scope_sha256=scope_sha,
                legal_status=status, reason=reason, qualification=qualification))
    if discovery is not None:
        known = set(candidates) | {'original'}
        for option in discovery['options']:
            id_ = option['id']
            if id_ in known:
                raise ValueError('duplicate discovered entry option identity')
            known.add(id_)
            nvhpc = _nvhpc_option(option) or _nvhpc_device_option(option)
            stack_reference = option['proof']['emission_plan']['operator_stack'] if nvhpc else None
            status, reason = _check_operator_stack(scope, NVIDIA_HPC if nvhpc else LIBTORCH,
                option['legal_status'], option['reason'], reference=stack_reference)
            facts.append(dict(option_id=id_, scope_sha256=scope_sha,
                legal_status=status, reason=reason,
                generated_option=option))
    return dict(schema=LEGALITIES, schema_version=1, scope_sha256=scope_sha, options=facts)


def _derive(plan, request, candidates, scope, legalities, legality_reference, discovery):
    options = [dict(id='original', legal_status='LEGAL', legality_evidence=[], cost=request['original_cost'])]
    for index, fact in enumerate(legalities['options']):
        id_, legal = fact['option_id'], fact['legal_status']
        item = candidates.get(id_)
        options.append(dict(id=id_, legal_status=legal, cost=item['cost'] if item and legal == 'LEGAL' else None,
            legality_evidence=[dict(artifact=legality_reference, locator=f'#/options/{index}')]))
    frontier = [] if discovery is None else list(discovery['coverage_frontier'])
    if scope['option_space'] == 'DISCOVER_ALL':
        reason = ('NO_COMPLETE_WHOLE_ENTRY_OPTION_GENERATOR' if discovery is None
                  else 'UNENUMERATED_ENTRY_IMPLEMENTATION_FAMILIES')
        frontier.append(dict(layer='ENTRY_IMPLEMENTATION_DOMAIN', reason=reason,
            regional_domains=len(plan['regions']), invariant_loops=len(plan['invariant_loops']),
            composite_invariant_loops=len(plan['composite_invariant_loops'])))
    return dict(schema=DOMAIN, schema_version=1, scope=scope, baseline_id='original', options=options,
        coverage_frontier=frontier)


def _bind_tensor_flow(plan, scope):
    if 'tensor_flow' not in scope:
        return None
    from scar.analysis.entry_tensor_flow import bind_source_tensor_flow

    if plan.get('entry_obligations') is None:
        raise ValueError('Tensor-flow binding requires original F4 entry obligations')
    return bind_source_tensor_flow(plan['program'], scope, scope['tensor_flow'],
                                   entry_obligations=plan['entry_obligations'])


def _bind_cpu_observation(plan, scope, discovery, *, paired_evidence=None, checked_pair=None):
    paired = paired_evidence or scope.get('cpu_paired_observation')
    capture = scope.get('cpu_stream_capture')
    if paired is None and capture is None:
        return None
    from scar.analysis.cpu_provider_graph import bind_cpu_provider_graph, bind_cpu_stream_capture_graph

    program = read_artifact(plan['program'])
    admission = read_artifact(plan['source_admission'])
    if (program['target'] != plan['hardware']
            or admission['source']['filter_path'] != plan.get('source_filter')):
        raise ValueError('CPU observation admission differs from the original hardware or source filter')
    if paired is None:
        return bind_cpu_stream_capture_graph(plan['program'], scope, capture,
            source_admission=plan['source_admission'], entry_obligations=plan.get('entry_obligations'),
            provider=(discovery or {}).get('matched_authority'))
    flow = bind_cpu_provider_graph(plan['program'], scope, paired,
        source_admission=plan['source_admission'], entry_obligations=plan.get('entry_obligations'),
        checked_pair=checked_pair)
    if flow['provider'] != (discovery or {}).get('matched_authority'):
        raise ValueError('retained CPU observation belongs to a different catalog authority')
    return flow


def _attach_cpu_pair(plan, scope, discovery, path, current, out, checked_pair):
    from scar.analysis.replay_transaction import current_transaction
    from scar.compiler.cpu_catalog_entry_trial import _compilation, _observation_summary
    from scar.compiler.cpu_entry_launch_scope import verify_paired_original_scope
    from scar.compiler.cpu_entry_replay import CheckedCpuPair, SCHEMA

    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    paired, compilation = verify_paired_original_scope(scope, reference)
    pinned = scope.get('cpu_paired_observation')
    matches = [row for row in discovery['options'] if _cpu_catalog_option(row)
               and row['proof']['emission_plan']['authority'] == compilation['provider']]
    if (pinned not in (None, reference) or paired['capture'] != scope.get('cpu_stream_capture')
            or compilation['catalog'] != scope['cpu_provider_catalog']
            or len(matches) != 1):
        raise ValueError('paired evidence differs from the retained CPU provider, capture or catalog')
    option, = matches
    emission = option['proof']['emission_plan']
    compiled = _compilation(paired['compilation'], emission)
    review = _observation_summary(reference, paired, compiled, emission, scope)
    if pinned == reference:
        graph_ref = current
    else:
        checked_pair = CheckedCpuPair(reference) if current_transaction() is not None else None
        graph = _bind_cpu_observation(plan, scope, discovery, paired_evidence=reference, checked_pair=checked_pair)
        graph_ref = _write(out/'whole-entry-cpu-provider-graph.json', graph)
    audit = dict(option_id=option['id'], legal_status=option['legal_status'], reason='RETAINED_SCOPED_CPU_PAIR',
        provider_replay=review, validation=dict(reference=reference, timing=None,
            exact_declared_effects_replayed=False, provider_boundary_replayed=True))
    return graph_ref, audit, checked_pair


def _retain_projection(out, filename, report, retained):
    if retained is None:
        return _write(out/filename, report)
    if read_artifact(retained) != report:
        raise ValueError('retained planning projection differs from original evidence: ' + filename)
    return retained


def _cpu_frontiers(plan, scope, entry, out, *, priced_domain=None, import_frontier=None, import_source_effects=None,
                   import_activations=None):
    if 'cpu_provider_catalog' not in scope or plan.get('entry_obligations') is None:
        return {}
    from scar.analysis.entry_effect_frontier import derive_entry_effect_frontier
    from scar.optimization.cpu_entry_decision_ledger import derive_cpu_entry_decision_ledger

    graph = entry.get('cpu_provider_graph')
    effect = _retain_projection(out, 'whole-entry-effect-frontier.json',
        derive_entry_effect_frontier(plan['entry_obligations'], graph), entry.get('entry_effect_frontier'))
    domain = priced_domain or entry['domain']
    decision = _retain_projection(out, 'whole-entry-decision-ledger.json',
        derive_cpu_entry_decision_ledger(domain, entry['legalities'], provider_graph=graph,
                                         effect_frontier=effect, import_frontier=import_frontier,
                                         import_source_effects=import_source_effects, import_activations=import_activations),
        entry.get('cpu_decision_ledger') if domain == entry['domain'] and import_frontier is None else None)
    return dict(entry_effect_frontier=effect, cpu_decision_ledger=decision)


def _attach_native_phase_cost(frontiers, nomination, path):
    from scar.optimization.cpu_native_lifecycle_cost import SCHEMA, verify_cpu_native_lifecycle_cost

    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    diagnostic = verify_cpu_native_lifecycle_cost(reference)
    measured = read_artifact(diagnostic['lifecycle_graph'])
    planned = read_artifact(frontiers['cpu_entry_lifecycle'])
    if measured['capacity_contract'] is None or planned['capacity_contract'] is None:
        raise ValueError('native phase observations require a bound CPU candidate launch contract')
    measured_capacity = read_artifact(measured['capacity_contract'])
    planned_capacity = read_artifact(planned['capacity_contract'])
    if (diagnostic['option_id'] != nomination or diagnostic['scope_sha256'] != planned['scope_sha256']
            or any(measured[key] != planned[key] for key in ('program', 'target', 'scope_sha256'))
            or any(measured_capacity[key] != planned_capacity[key] for key in ('native_launch', 'evidence', 'guard'))):
        raise ValueError('native phase observations differ from the current CPU entry, provider or launch scope')
    frontiers['cpu_native_phase_cost'] = reference


def _attach_python_import_cost(plan, scope, frontiers, path):
    from scar.compiler.cpu_entry_launch_scope import verify_paired_original_scope
    from scar.optimization.cpu_python_import_cost import SCHEMA, verify_cpu_python_import_cost
    from scar.optimization.resource_solver import semantic_sha256

    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    cost = verify_cpu_python_import_cost(reference)
    graph = read_artifact(cost['import_graph'])
    observed = read_artifact(cost['observation'])
    if (cost['scope_sha256'] != semantic_sha256(scope) or graph['program'] != plan['program']
            or graph['entry_obligations']['sha256'] != plan['entry_obligations']['sha256']):
        raise ValueError('import diagnostic belongs to a different F4 program or entry proof')
    verify_paired_original_scope(scope, observed['paired'])
    frontiers['cpu_python_import_cost'] = reference


def _attach_cpu_native_array_route(scope, generated, frontiers, path, cost_path=None):
    from scar.codegen.cpu_array_native_route_package import SCHEMA, verify_cpu_array_native_route_package

    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    package = verify_cpu_array_native_route_package(reference)
    route_plan = read_artifact(package['plan'])
    original = package['routes'][package['fallback_route']]
    matching = []
    for option in generated.values():
        if not _cpu_catalog_option(option):
            continue
        emission = option['proof']['emission_plan']
        contract, context = emission['contract'], emission['context']
        profile = contract['profile']
        if (emission['entry'] == package['source'] and emission['argv'] == package['argv']
                and emission['catalog'] == scope['cpu_provider_catalog'] and emission['target'] == scope['target']
                and profile.get('retention') == route_plan['retention']
                and profile['runtime'] == original['package'] and profile['holdout'] == package['holdout']
                and profile['recipe'] == package['recipe'] and contract['legacy_hardware'] == package['hardware']
                and context['cpu_ids'] == package['cpu_ids']
                and context['cwd'] == route_plan['workload']['cwd']
                and context['original_environment'] == package['source_environment']
                and all(context['scope_matches'].values())):
            matching.append(option['id'])
    if len(matching) != 1:
        raise ValueError('native array route must match one exact F4 source, provider profile and hardware scope')
    frontiers['cpu_native_array_route'] = reference
    frontiers['cpu_native_array_route_option_id'] = matching[0]
    if cost_path is not None:
        from scar.optimization.cpu_array_native_route_forward import SCHEMA, _verify

        cost_ref = artifact_reference(Path(cost_path).resolve(strict=True), SCHEMA)
        cost = _verify(cost_ref, (reference, package))
        if (cost['package'] != reference or cost['proof'] != route_plan['proof']
                or cost['scope'] != package['scope'] or cost['hardware'] != package['hardware']):
            raise ValueError('native array route timeline differs from the exact F4 option and launch package')
        frontiers['cpu_native_array_route_cost'] = cost_ref


def _attach_cpu_fixed_grid_cli(plan, scope, generated, frontiers, path, out, mapped_path=None):
    from scar.analysis.cpu_fixed_grid_cli_effect_bridge import derive_cpu_fixed_grid_cli_effect_bridge
    from scar.analysis.cpu_fixed_grid_cli_resources import derive_cpu_fixed_grid_cli_resources
    from scar.compiler.cpu_fixed_grid_cli_delivery import SCHEMA
    from scar.optimization.cpu_fixed_grid_cli_review import derive_cpu_fixed_grid_cli_review

    delivery_ref = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    delivery = read_artifact(delivery_ref)
    provider = delivery['provider']
    matching = [row for row in generated.values() if _cpu_catalog_option(row)
                and row['proof']['emission_plan']['authority'] == provider]
    if provider['schema'] != 'scar.cpu-fixed-grid-source-provider' or len(matching) != 1:
        raise ValueError('fixed-grid source CLI must name one original CPU catalog option')
    option, = matching
    bridge = derive_cpu_fixed_grid_cli_effect_bridge(
        plan['program'], plan['entry_obligations'], provider, scope['argv'])
    bridge_ref = _write(out/'whole-entry-fixed-grid-source-effects.json', bridge)
    review = derive_cpu_fixed_grid_cli_review(option, scope, bridge_ref, delivery_ref)
    inputs = option['proof']['emission_plan']['contract']['inputs']
    resources = derive_cpu_fixed_grid_cli_resources(delivery['project'], scope['target'], scope['argv'], inputs)
    frontiers.update(
        cpu_fixed_grid_cli_effects=bridge_ref,
        cpu_fixed_grid_cli_resources=_write(out/'whole-entry-fixed-grid-resources.json', resources),
        cpu_fixed_grid_cli_review=_write(out/'whole-entry-fixed-grid-cli-review.json', review),
    )
    if mapped_path is not None:
        from scar.optimization.cpu_fixed_grid_mapped_input_plan import (
            SCHEMA as MAPPED_PLAN, verify_cpu_fixed_grid_mapped_input_plan,
        )

        mapped_ref = artifact_reference(Path(mapped_path).resolve(strict=True), MAPPED_PLAN)
        mapped = verify_cpu_fixed_grid_mapped_input_plan(mapped_ref)
        if (read_artifact(mapped['proof'])['project'] != delivery['project']
                or mapped['observation'] is None
                or read_artifact(mapped['observation'])['control_delivery'] != delivery_ref):
            raise ValueError('mapped inputs do not belong to the exact scoped fixed-grid native incumbent')
        frontiers['cpu_fixed_grid_mapped_input_plan'] = mapped_ref


def _cpu_import_frontiers(plan, scope, out, binding_path, source_path=None, activation_path=None):
    from scar.analysis.cpu_entry_import_effects import derive_cpu_entry_import_effects
    from scar.analysis.cpu_entry_import_initializers import derive_cpu_entry_import_initializers

    admission_ref = None
    activation_ref = None
    if activation_path is not None:
        from scar.compiler.cpu_import_initializer_observation import SCHEMA as INITIALIZER_OBSERVATION

        activation_ref = artifact_reference(Path(activation_path).resolve(strict=True), INITIALIZER_OBSERVATION)
        activation = read_artifact(activation_ref)
        if source_path is None:
            source_path = activation['admission']['uri']
    if source_path is not None:
        from scar.compiler.cpu_import_source_admission import SCHEMA as ADMISSION

        admission_ref = artifact_reference(Path(source_path).resolve(strict=True), ADMISSION)
        admission = read_artifact(admission_ref)
        if activation_ref is not None and activation['admission'] != admission_ref:
            raise ValueError('initializer observation requires its original source admission')
        if admission['program'] != plan['program'] or admission['entry_obligations'] != plan['entry_obligations']:
            raise ValueError('import source admission differs from the original F4 program')

    graph = derive_cpu_entry_import_initializers(plan['entry_obligations'])
    if binding_path is None and admission_ref is None:
        initializers = _write(out/'whole-entry-import-initializers.json', graph)
    else:
        from scar.compiler.cpu_entry_launch_scope import verify_paired_original_scope
        from scar.compiler.cpu_import_binding_observation import SCHEMA as OBSERVATION

        observation_ref = (admission['observation'] if binding_path is None else
                           artifact_reference(Path(binding_path).resolve(strict=True), OBSERVATION))
        if admission_ref is not None and observation_ref != admission['observation']:
            raise ValueError('import source admission requires its original binding observation')
        observation = read_artifact(observation_ref)
        initializers = observation['initializer_graph']
        paired, _ = verify_paired_original_scope(scope, observation['paired'])
        if (read_artifact(initializers) != graph
                or scope.get('cpu_paired_observation') not in (None, observation['paired'])
                or paired['capture'] != scope.get('cpu_stream_capture')):
            raise ValueError('import bindings differ from the original F4 initializer graph or captured launch')
    effects = _write(out/'whole-entry-import-effects.json', derive_cpu_entry_import_effects(initializers))
    result = dict(cpu_entry_import_initializers=initializers, cpu_entry_import_effects=effects)
    if binding_path is not None or admission_ref is not None:
        from scar.analysis.cpu_entry_import_bindings import derive_cpu_entry_import_bindings
        from scar.optimization.cpu_entry_import_frontier import derive_cpu_entry_import_frontier

        bindings = _write(out/'whole-entry-import-bindings.json',
                          derive_cpu_entry_import_bindings(initializers, observation_ref))
        result['cpu_entry_import_bindings'] = bindings
        result['cpu_entry_import_frontier'] = _write(out/'whole-entry-import-frontier.json',
                                                      derive_cpu_entry_import_frontier(effects, bindings))
    if admission_ref is not None:
        from scar.optimization.cpu_import_source_effect_plan import derive_cpu_import_source_effect_plan

        result['cpu_entry_import_source_effects'] = _write(out/'whole-entry-import-source-effects.json',
                                                           derive_cpu_import_source_effect_plan(admission_ref))
    if activation_ref is not None:
        from scar.optimization.cpu_import_initializer_activation_plan import derive_cpu_import_activation_plan

        result['cpu_entry_import_activations'] = _write(out/'whole-entry-import-activations.json',
            derive_cpu_import_activation_plan(result['cpu_entry_import_source_effects'], activation_ref))
    return result


def _forward_cpu_entry(out, frontiers, nomination, probes, team_probe, phase_cost):
    from scar.analysis.cpu_entry_lifecycle_graph import derive_cpu_entry_lifecycle_graph
    from scar.optimization.cpu_entry_forward_cost import LAUNCH_PROXY, TEAM_PROBE, derive_cpu_entry_forward_cost
    from scar.optimization.cpu_entry_memory_lifetimes import derive_cpu_entry_memory_lifetimes
    from scar.optimization.cpu_entry_schedule_constraints import derive_cpu_entry_schedule_constraints

    ledger = frontiers['cpu_decision_ledger']
    if phase_cost is not None:
        from scar.analysis.cpu_native_publication_order import derive_cpu_native_publication_order
        from scar.optimization.cpu_native_lifecycle_cost import SCHEMA

        diagnostic = read_artifact(artifact_reference(Path(phase_cost).resolve(strict=True), SCHEMA))
        frontiers['cpu_native_publication_order'] = _write(out/'whole-entry-native-publication-order.json',
            derive_cpu_native_publication_order(diagnostic['phase_sites']))
    lifecycle = _write(out/'whole-entry-cpu-lifecycle.json',
        derive_cpu_entry_lifecycle_graph(frontiers['cpu_launch_resources'], decision_ledger=ledger,
            capacity_contract=frontiers.get('cpu_capacity_contract'),
            original_publication_order=frontiers.get('cpu_original_publication_order'),
            native_publication_order=frontiers.get('cpu_native_publication_order')))
    frontiers['cpu_entry_lifecycle'] = lifecycle
    if nomination is None:
        if probes or team_probe is not None or phase_cost is not None:
            raise ValueError('CPU component probes require one independently nominated native entry')
        return
    refs = [artifact_reference(Path(path).resolve(strict=True), LAUNCH_PROXY) for path in probes]
    if team_probe is not None:
        refs.append(artifact_reference(Path(team_probe).resolve(strict=True), TEAM_PROBE))
    constraints = _write(out/'whole-entry-cpu-schedule-constraints.json',
                         derive_cpu_entry_schedule_constraints(lifecycle))
    frontiers['cpu_entry_schedule_constraints'] = constraints
    memory = _write(out/'whole-entry-cpu-memory-lifetimes.json', derive_cpu_entry_memory_lifetimes(constraints))
    frontiers['cpu_entry_memory_lifetimes'] = memory
    frontiers['cpu_entry_forward_cost'] = _write(out/'whole-entry-cpu-forward-cost.json',
        derive_cpu_entry_forward_cost(ledger, nomination, refs, lifecycle_graph=lifecycle,
                                     schedule_constraints=constraints, memory_lifetimes=memory))
    if phase_cost is not None:
        _attach_native_phase_cost(frontiers, nomination, phase_cost)


@verify_stack_components_once()
def prepare_whole_entry(plan, request, out):
    reference, document, candidates = read_request(request)
    scope = _scope(plan, document)
    stack_target = None
    if 'operator_stack_lock' in document:
        stack_target = _write(out/'whole-entry-stack-target.json', _stack_target(scope, document['operator_stack_lock']))
        scope['stack_target'] = stack_target
    discovery = _discover(scope, document)
    legalities = _legalities(candidates, scope, discovery)
    legal_ref = _write(out/'whole-entry-legalities.json', legalities)
    domain = _derive(plan, document, candidates, scope, legalities, legal_ref, discovery)
    entry = dict(request=reference, domain=_write(out/'whole-entry-domain.json', domain),
        legalities=legal_ref, scope=scope, complete_declared_domain=not domain['coverage_frontier'])
    if discovery is not None:
        entry['variant_discovery'] = _write(out/'whole-entry-variants.json', discovery)
    flow = _bind_tensor_flow(plan, scope)
    if flow is not None:
        entry['tensor_flow_binding'] = _write(out/'whole-entry-tensor-flow.json', flow)
    cpu_observation = _bind_cpu_observation(plan, scope, discovery)
    if cpu_observation is not None:
        entry['cpu_provider_graph'] = _write(out/'whole-entry-cpu-provider-graph.json', cpu_observation)
    entry.update(_cpu_frontiers(plan, scope, entry, out))
    return entry


def _attach_costs(domain, original_ref, source, out):
    if source is None:
        return domain, original_ref, None
    reference = artifact_reference(Path(source).resolve(), COST_INPUTS)
    document = read_artifact(reference)
    if (set(document) != {'schema', 'schema_version', 'domain', 'costs'} or document['schema_version'] != 1
            or document['domain'] != original_ref or type(document['costs']) is not dict):
        raise ValueError('cost inputs must cite the exact original entry option domain')
    pending = dict(document['costs'])
    options = []
    for row in domain['options']:
        id_ = row['id']
        if id_ in pending:
            cost = pending.pop(id_)
            if row['legal_status'] == 'INELIGIBLE' or row['cost'] not in (None, cost) or type(cost) is not dict:
                raise ValueError('a cost input cannot replace evidence or price an ineligible option: ' + id_)
            row = dict(row, cost=cost)
        options.append(row)
    if pending:
        raise ValueError('cost inputs name options outside the independently replayed domain')
    priced = dict(domain, options=options)
    return priced, _write(out/'whole-entry-priced-domain.json', priced), reference


def _nvhpc_option(option):
    return option.get('implementation_family') == 'NVIDIA_CPU_MATRIX_FILE_ENTRY'


def _nvhpc_device_option(option):
    return option.get('implementation_family') == 'NVIDIA_DEVICE_MATRIX_FILE_ENTRY'


def _cpu_catalog_option(option):
    return option.get('implementation_family') == 'CATALOG_CPU_ENTRY'


def _emit_generated(option, scope, out, *, plan_only, verify_entry, capacity_contract=None, retained_pair_ref=None,
                    checked_pair=None):
    if _nvhpc_option(option):
        from scar.compiler.nvhpc_entry_trial import emit_nvhpc_entry

        return emit_nvhpc_entry(option, scope, out, plan_only=plan_only, verify_entry=verify_entry)
    if _nvhpc_device_option(option):
        from scar.compiler.nvhpc_device_entry_trial import emit_nvhpc_device_entry

        return emit_nvhpc_device_entry(option, scope, out, plan_only=plan_only, verify_entry=verify_entry)
    if _cpu_catalog_option(option):
        from scar.compiler.cpu_catalog_entry_trial import emit_cpu_catalog_entry

        return emit_cpu_catalog_entry(option, scope, out, plan_only=plan_only, verify_entry=verify_entry,
            capacity_contract=capacity_contract, retained_pair_ref=retained_pair_ref, checked_pair=checked_pair)

    from scar.compiler.entry_delivery import HANDOFF, replay_entry_delivery

    plan = option['proof']['emission_plan']
    result = dict(schema=HANDOFF, schema_version=1, selected_id=option['id'], scope=scope,
        final_action='NATIVE_COMPILATION_DEFERRED', reason='plan-only preserved the selected C++ emission plan',
        selected_emission=plan, native_compiled=False, target_executed=False,
        operator_stack=dict(family=LIBTORCH, retained_body=plan['body']),
        qualification=None, validation=None, deployment_authorized=False)
    if plan_only:
        return result

    from scar.compiler.aten_source_entry_plan import emit_planned_source_entry

    try:
        native = emit_planned_source_entry(plan, out/'native-entry')
    except (OSError, ValueError, CalledProcessError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason='native emission failed: ' + str(error))
    if read_artifact(native['artifact'])['source_entry_plan'] != plan:
        raise ValueError('generated C++ project differs from the selected entry plan')
    candidate = dict(id=option['id'], route='ATEN_SOURCE_ENTRY', artifact=native['artifact'], validation=None)
    admission = replay_entry_delivery(scope, selected_id=option['id'], candidate=candidate)
    result.update(admission, native=native, native_compiled=True)
    if not verify_entry or admission['final_action'] != 'CANDIDATE_VALIDATION_REQUIRED':
        return result

    from scar.compiler.aten_source_entry import validate_source_entry

    inputs = scope['argv'][plan['abi']['argv']['input_root'] - 1]
    steps = option['proof']['steps']
    validation_dir = out/'native-validation'
    try:
        validation = validate_source_entry(native['project'], native['manifest_sha256'], inputs, validation_dir,
            steps=steps, argv=scope['argv'], original_cpu_ids=scope['process_affinity_cpu_ids'],
            expected_environment_sha256=scope['environment_sha256'])
    except (OSError, ValueError, CalledProcessError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason='native validation failed: ' + str(error),
                    validation_attempted=True, target_executed=None)
    receipt = artifact_reference(validation_dir/'result.json', validation['schema'])
    candidate['validation'] = receipt
    inspected = replay_entry_delivery(scope, selected_id=option['id'], candidate=candidate)
    return dict(result, **inspected, target_executed=True)


def _observed_costs(audit, delivery, scope_sha, out):
    from scar.optimization.whole_entry_choice import LIFECYCLE_PHASES

    results = [*audit, dict(option_id=delivery.get('selected_id'), validation=delivery.get('validation'))]
    comparisons = []
    for row in results:
        validation = row.get('validation') or {}
        timing = validation.get('timing')
        if timing is None:
            continue
        facts = []
        for option, mode in (('original', 'ORIGINAL'), (row['option_id'], 'NATIVE')):
            observed = timing['metrics'][mode]
            facts.append(dict(option_id=option, scope_sha256=scope_sha, kind='COMPLETE_INTERVAL',
                basis='EMPIRICAL_ENVELOPE', interval_ns=dict(lower=observed['min_ns'], upper=observed['max_ns']),
                lifecycle_coverage=dict.fromkeys(LIFECYCLE_PHASES, 'ACCOUNTED'),
                coverage_basis='WHOLE_PROCESS_ELAPSED_INCLUDING_ALL_ACTUAL_PHASES',
                process_validation=validation['reference'], median_ns=observed['median_ns'],
                physical_duration_bounds=False))
        comparisons.append(dict(challenger=row['option_id'], facts=facts))
    if not comparisons:
        return None
    return _write(out/'whole-entry-observed-costs.json', dict(schema=OBSERVED_COSTS, schema_version=1,
        comparisons=comparisons, separate_component_times_available=False))


def _review_source_trials(discovery, generated, domain, scope, out, *, capacity_contract=None, paired_audit=None,
                          checked_pair=None):
    """Review the nominated device and permit a CPU trial if no entry ran."""
    eligible = [generated[row['id']] for row in domain['options']
                if row['id'] in generated and row['legal_status'] != 'INELIGIBLE']
    cpu = [option for option in eligible if _nvhpc_option(option) or _cpu_catalog_option(option)]
    device = [option for option in eligible if _nvhpc_device_option(option)]
    search = (discovery or {}).get('cohort_search')
    if search is not None:
        cpu = [option for option in cpu if option['id'] == search['trial_option_id']]
    audit = []

    def emit(option, folder):
        folder.mkdir(parents=True, exist_ok=True)
        evidence = {} if capacity_contract is None else dict(capacity_contract=capacity_contract)
        if paired_audit is not None and paired_audit['option_id'] == option['id']:
            evidence['retained_pair_ref'] = paired_audit['validation']['reference']
        if checked_pair is not None and _cpu_catalog_option(option):
            evidence['checked_pair'] = checked_pair
        trial = _emit_generated(option, scope, folder, plan_only=False, verify_entry=True, **evidence)
        audit.append(dict(option_id=trial['selected_id'], legal_status=option['legal_status'],
                          validation=trial['validation'], reason=trial['reason']))
        return trial

    device_trial = emit(device[0], out/'device-trial') if len(device) == 1 else None
    can_try_cpu = device_trial is None or device_trial['target_executed'] is False
    cpu_trial = emit(cpu[0], out) if can_try_cpu and len(cpu) == 1 else None
    return cpu_trial, device_trial, audit


@verify_stack_components_once()
def replay_whole_entry(plan, out, *, plan_only, entry_costs=None, verify_entry=False, enforce_cpu_capacity=False,
                       cpu_entry_forward=False, cpu_launch_probes=(), cpu_team_probe=None, cpu_native_phase_cost=None,
                       cpu_python_import_cost=None, cpu_paired_evidence=None, cpu_import_bindings=None,
                       cpu_import_sources=None, cpu_initializer_observation=None, cpu_native_array_route=None,
                       cpu_native_array_route_cost=None, cpu_fixed_grid_cli_evidence=None,
                       cpu_fixed_grid_mapped_input_plan=None):
    from scar.analysis.build_admission import verify_build_admission
    from scar.analysis.replay_transaction import current_transaction
    from scar.compiler.cpu_entry_replay import CheckedCpuPair
    from scar.compiler.entry_delivery import replay_entry_delivery
    from scar.optimization.whole_entry_choice import solve_whole_entry_choice

    entry = plan['whole_entry']
    _, request, candidates = read_request(entry['request'])
    if enforce_cpu_capacity and not (verify_entry and not plan_only and _infer_cpu_catalog(request)):
        raise ValueError('CPU capacity enforcement requires an explicit CPU catalog trial')
    forward = cpu_entry_forward or bool(cpu_launch_probes) or cpu_team_probe is not None or cpu_native_phase_cost is not None
    if forward and not _infer_cpu_catalog(request):
        raise ValueError('CPU forward cost analysis requires a CPU catalog domain')
    if cpu_python_import_cost is not None and not _infer_cpu_catalog(request):
        raise ValueError('CPU Python import observations require a CPU catalog domain')
    if cpu_paired_evidence is not None and not _infer_cpu_catalog(request):
        raise ValueError('CPU paired evidence requires a CPU catalog domain')
    if cpu_native_array_route is not None and (not _infer_cpu_catalog(request) or plan.get('entry_obligations') is None):
        raise ValueError('CPU native array routes require a catalog domain with original F4 obligations')
    if cpu_fixed_grid_cli_evidence is not None and (not _infer_cpu_catalog(request) or plan.get('entry_obligations') is None):
        raise ValueError('fixed-grid source CLI evidence requires a catalog domain with original F4 obligations')
    if cpu_fixed_grid_cli_evidence is not None and 'cpu_paired_observation' not in request:
        raise ValueError('fixed-grid source CLI evidence requires the frozen original paired launch')
    if cpu_fixed_grid_mapped_input_plan is not None and cpu_fixed_grid_cli_evidence is None:
        raise ValueError('mapped input decisions require the scoped fixed-grid source CLI evidence')
    if cpu_native_array_route_cost is not None and cpu_native_array_route is None:
        raise ValueError('CPU native array route costs require the exact C++ route package')
    if ((cpu_import_bindings is not None or cpu_import_sources is not None or cpu_initializer_observation is not None)
            and (not _infer_cpu_catalog(request) or plan.get('entry_obligations') is None)):
        raise ValueError('CPU import bindings require a CPU catalog domain with original F4 obligations')
    if cpu_paired_evidence is None and 'cpu_paired_observation' not in request and 'cpu_stream_capture' not in request:
        verify_build_admission(plan['source_admission'], program=plan['program'], hardware=plan['hardware'],
                               source_filter=plan.get('source_filter'))
    scope_request = dict(request, use_captured_launch=True) if cpu_fixed_grid_cli_evidence is not None else request
    live_scope = _scope(plan, scope_request)
    if 'operator_stack_lock' in request:
        stack_target = entry['scope']['stack_target']
        retained = read_artifact(stack_target)
        if retained != _stack_target(live_scope, request['operator_stack_lock'], device=retained['device']):
            raise ValueError('NVIDIA operator stack and original device scope differ from retained evidence')
        live_scope['stack_target'] = stack_target
    if live_scope != entry['scope']:
        retained = entry['scope']
        changed = sorted(key for key in live_scope.keys() | retained.keys() if live_scope.get(key) != retained.get(key))
        raise ValueError('whole-entry scope changed in ' + ', '.join(changed))
    discovery = _discover(live_scope, request)
    if (('variant_discovery' in entry) != (discovery is not None) or discovery is not None
            and read_artifact(entry['variant_discovery']) != discovery):
        raise ValueError('entry variant set differs from mechanical proof replay')
    flow = _bind_tensor_flow(plan, live_scope)
    flow_ref = entry.get('tensor_flow_binding')
    if (flow_ref is None) != (flow is None) or flow_ref is not None and read_artifact(flow_ref) != flow:
        raise ValueError('Tensor-flow binding differs from original graph replay')
    pinned = live_scope.get('cpu_paired_observation')
    checked_pair = CheckedCpuPair(pinned) if pinned is not None and current_transaction() is not None else None
    cpu_observation = _bind_cpu_observation(plan, live_scope, discovery, checked_pair=checked_pair)
    cpu_observation_ref = entry.get('cpu_provider_graph')
    if ((cpu_observation_ref is None) != (cpu_observation is None) or cpu_observation_ref is not None
            and read_artifact(cpu_observation_ref) != cpu_observation):
        raise ValueError('CPU provider observation differs from original graph replay')
    frontier_entry, paired_audit = entry, None
    if cpu_paired_evidence is not None:
        attached, paired_audit, checked_pair = _attach_cpu_pair(plan, live_scope, discovery, cpu_paired_evidence,
                                                              cpu_observation_ref, out, checked_pair)
        if attached != cpu_observation_ref:
            cpu_observation_ref = attached
            frontier_entry = dict(entry, cpu_provider_graph=attached, entry_effect_frontier=None,
                                  cpu_decision_ledger=None)
    if flow is None and cpu_observation_ref is None and plan.get('entry_obligations') is not None:
        from scar.analysis.entry_obligations import verify_entry_obligations

        verify_entry_obligations(plan['entry_obligations'], program=plan['program'])
    legalities = _legalities(candidates, live_scope, discovery)
    domain = _derive(plan, request, candidates, live_scope, legalities, entry['legalities'], discovery)
    if read_artifact(entry['domain']) != domain or read_artifact(entry['legalities']) != legalities:
        raise ValueError('retained whole-entry domain differs from original program and evidence replay')
    domain, priced_domain, cost_inputs = _attach_costs(domain, entry['domain'], entry_costs, out)
    import_frontiers = (_cpu_import_frontiers(plan, live_scope, out, cpu_import_bindings, cpu_import_sources,
                                               cpu_initializer_observation)
                        if _infer_cpu_catalog(request) and plan.get('entry_obligations') is not None else {})
    cpu_frontiers = _cpu_frontiers(plan, live_scope, frontier_entry, out, priced_domain=priced_domain,
                                  import_frontier=import_frontiers.get('cpu_entry_import_frontier'),
                                  import_source_effects=import_frontiers.get('cpu_entry_import_source_effects'),
                                  import_activations=import_frontiers.get('cpu_entry_import_activations'))
    cpu_frontiers.update(import_frontiers)
    if paired_audit is not None:
        from scar.analysis.cpu_entry_publication_order import derive_cpu_entry_publication_order

        cpu_frontiers['cpu_original_publication_order'] = _write(out/'whole-entry-original-publication-order.json',
            derive_cpu_entry_publication_order(cpu_observation_ref))
    if cpu_python_import_cost is not None:
        _attach_python_import_cost(plan, live_scope, cpu_frontiers, cpu_python_import_cost)
    generated = {} if discovery is None else {option['id']: option for option in discovery['options']}
    if cpu_native_array_route is not None:
        _attach_cpu_native_array_route(live_scope, generated, cpu_frontiers, cpu_native_array_route,
                                       cpu_native_array_route_cost)
    if cpu_fixed_grid_cli_evidence is not None:
        _attach_cpu_fixed_grid_cli(plan, live_scope, generated, cpu_frontiers, cpu_fixed_grid_cli_evidence, out,
                                   cpu_fixed_grid_mapped_input_plan)
    choice = solve_whole_entry_choice(domain, expected_option_ids=['original', *sorted(candidates), *sorted(generated)])
    selected = choice['selected_option_id']
    capacity_contract = None
    if forward or enforce_cpu_capacity and selected != domain['baseline_id']:
        from scar.analysis.cpu_launch_resources import derive_cpu_launch_resources
        from scar.optimization.cpu_entry_capacity_contract import derive_cpu_entry_capacity_contract

        eligible = [row['id'] for row in domain['options'] if row['legal_status'] != 'INELIGIBLE'
                    and row['id'] in generated and _cpu_catalog_option(generated[row['id']])]
        fallback = eligible[0] if len(eligible) == 1 and (selected is None or forward) else None
        nomination = selected if selected in eligible else fallback
        if forward or nomination is not None:
            cpu_frontiers['cpu_launch_resources'] = _write(out/'whole-entry-cpu-launch-resources.json',
                derive_cpu_launch_resources(cpu_frontiers['entry_effect_frontier']))
        if nomination is not None:
            capacity = _write(out/'whole-entry-cpu-capacity-contract.json',
                derive_cpu_entry_capacity_contract(cpu_frontiers['cpu_decision_ledger'], nomination))
            cpu_frontiers['cpu_capacity_contract'] = capacity
            if enforce_cpu_capacity:
                capacity_contract = capacity
        if forward:
            _forward_cpu_entry(out, cpu_frontiers, nomination, cpu_launch_probes, cpu_team_probe, cpu_native_phase_cost)
    if selected in generated:
        retained_pair_ref = paired_audit['validation']['reference'] if (
            paired_audit is not None and paired_audit['option_id'] == selected) else None
        delivery = _emit_generated(generated[selected], live_scope, out, plan_only=plan_only, verify_entry=verify_entry,
            capacity_contract=capacity_contract, retained_pair_ref=retained_pair_ref, checked_pair=checked_pair)
    else:
        candidate = _candidate(selected, candidates[selected]) if selected in candidates else None
        delivery = replay_entry_delivery(live_scope, selected_id=selected, candidate=candidate)
    audit = [] if paired_audit is None else [paired_audit]
    trial, device_trial = None, None
    if selected is None and verify_entry and not plan_only:
        trial, device_trial, new_audit = _review_source_trials(discovery, generated, domain, live_scope, out,
            capacity_contract=capacity_contract, paired_audit=paired_audit, checked_pair=checked_pair)
        audit.extend(new_audit)
    if request.get('audit_retained_validation', False):
        for fact in legalities['options']:
            id_ = fact['option_id']
            if id_ != selected and fact.get('qualification', {}).get('status') == 'QUALIFIED':
                observation = replay_entry_delivery(live_scope, selected_id=id_,
                    candidate=_candidate(id_, candidates[id_]))
                audit.append(dict(option_id=id_, legal_status=fact['legal_status'],
                    validation=observation['validation'], reason=observation['reason']))
    report = dict(schema=SELECTION, schema_version=1, domain=priced_domain, original_domain=entry['domain'],
        cost_inputs=cost_inputs, variant_discovery=entry.get('variant_discovery'),
        source_admission=plan['source_admission'], entry_obligations=plan.get('entry_obligations'),
        selection=choice, delivery=delivery,
        retained_validation_audit=audit,
        trial=trial, plan_only=plan_only,
        native_compiled=any(row.get('native_compiled', False) for row in (delivery, trial, device_trial) if row is not None),
        target_executed=(trial or device_trial or delivery).get('target_executed', False), **cpu_frontiers)
    if 'nvhpc_device_profile' in request:
        report['device_trial'] = device_trial
    if flow_ref is not None:
        report['tensor_flow_binding'] = flow_ref
    if cpu_observation_ref is not None:
        report['cpu_provider_graph'] = cpu_observation_ref
    report['observed_costs'] = _observed_costs(audit, delivery, choice['scope_sha256'], out)
    return dict(reference=_write(out/'whole-entry-selection.json', report), **report)
