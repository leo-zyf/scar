"""Bind a replayed source file entry to an admitted standalone ATen schedule."""

from hashlib import sha256
from math import prod
from pathlib import PurePosixPath

from scar.ir.v2.schemas import canonical_json
from scar.runtime.native_cpu_linkage import qualify_cpu_dependencies


SCHEMA = 'scar.aten-source-entry-binding'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(document):
    return sha256(canonical_json(document).encode()).hexdigest()


def _bind_files(abi, recipe):
    argv = abi['argv']
    roles = ('input_root', 'steps', 'output')
    indexes = [argv[role] for role in roles]
    _require(type(argv['count']) is int and argv['count'] == len(roles)
             and all(type(index) is int for index in indexes)
             and sorted(indexes) == list(range(1, len(roles) + 1)),
             'source file entry requires one input root, step count and output argument')
    inputs = abi['inputs']
    _require(inputs and len(inputs) == len(recipe['inputs'])
             and all(type(row['load_order']) is int for row in inputs)
             and sorted(row['load_order'] for row in inputs) == list(range(len(inputs))),
             'source file order does not cover every native RHS input')
    bindings = []
    formals = set()
    for port, (source, rhs) in enumerate(zip(inputs, recipe['inputs'], strict=True)):
        name = source['path']
        shape = source['shape']
        _require(type(name) is str and bool(name) and name not in ('.', '..')
                 and '\x00' not in name and PurePosixPath(name).name == name,
                 'source input must be a fixed file name relative to its argument root')
        _require(type(source['port']) is int and source['port'] == port
                 and type(source['formal_index']) is int and source['formal_index'] >= 0
                 and source['formal_index'] not in formals and source['dtype'] == 'FLOAT32'
                 and shape == rhs['shape'] and bool(shape)
                 and all(type(dim) is int and dim > 0 for dim in shape),
                 'source input binding or Tensor geometry differs from its native RHS port')
        formals.add(source['formal_index'])
        bindings.append(dict(source, value=rhs['id'], payload_bytes=4 * prod(shape)))
    return bindings


def _bind_recurrence(abi, execution, recipe, bindings, manifest):
    solver = abi['solver']
    provider = manifest['method_provider']
    state_port = solver['state_input_index']
    _require(type(state_port) is int and 0 <= state_port < len(bindings)
             and state_port == execution['state_input_index']
             and solver['step_size'] == execution['step_size']
             and provider['api'] == 'torchdiffeq.odeint' and provider['callbacks'] == 'NONE'
             and solver['method'] == provider['method'] and len(recipe['outputs']) == 1,
             'source solver does not match the verified native recurrence')
    state = execution['inputs'][0]
    operations = execution['operations']
    finals = [row for row in operations if row['id'] == execution['output']]
    _require(len(finals) == 1 and state['dtype'] == finals[0]['dtype'] == 'FLOAT32'
             and state['shape'] == finals[0]['shape'] == bindings[state_port]['shape'],
             'source carried state and final publication differ from native Tensor geometry')
    control = execution['control']
    carry = dict(distance=1, body_input=state['id'], body_output=execution['output'],
                 initial_entry_input=state_port)
    fixed_ports = [port for port in range(len(bindings)) if port != state_port]
    _require(control['kind'] == 'FIXED_COUNT_LOOP' and control['body_count'] == 1
             and control['carried_state'] == carry and control['invariant_entry_inputs'] == fixed_ports
             and control['invariant_step_size_input'] == execution['inputs'][1]['id'],
             'source step count may drive only one shared body with its admitted state carry')
    rhs_bindings = [dict(operation=row['id'], arguments=[row['arguments'][0] if port == state_port
                    else {'entry_input': port} for port in range(len(bindings))])
                    for row in operations if row['op'] == 'RHS']
    _require(bool(rhs_bindings) and execution['rhs_bindings'] == rhs_bindings,
             'native RHS does not retain its state and fixed file port bindings')
    return state_port, rhs_bindings


def verify_source_entry_binding(abi, execution, rhs_recipe, *, manifest):
    """Replay the source-to-physical boundary after native project verification.

    The original verifier is the authority for activity times, execution order,
    lifetimes, and actual file and dependency identity. This bridge treats the
    observed step body's estimate separately from the still-unmeasured entry.
    """
    _require(abi['schema'] == 'scar.fixed-grid-file-entry' and abi['schema_version'] == 1
             and execution['schema'] == 'scar.aten-recurrence-plan' and execution['schema_version'] == 1
             and rhs_recipe['schema'] == 'scar.aten-native-recipe' and rhs_recipe['schema_version'] == 1
             and manifest['schema'] == 'scar.aten-recurrence-delivery' and manifest['abi_version'] == 1
             and manifest['cxx_standard'] == 20 and manifest.get('source_entry_replayable') is True,
             'source entry requires a replayable source and an admitted C++20 ATen recurrence')
    bindings = _bind_files(abi, rhs_recipe)
    state_port, rhs_bindings = _bind_recurrence(abi, execution, rhs_recipe, bindings, manifest)
    publication = abi['publication']
    _require(publication['file'] == 'FINAL_TENSOR_RAW_BYTES' and publication['order'] == ['FILE', 'STDOUT'],
             'only the original single final Tensor file and subsequent stdout are admitted')
    domain = abi['domain']
    _require(domain['normal_completion_only'] is True
             and domain['inputs'] == 'STABLE_READ_ONLY_EXACT_SIZE_REGULAR_CPU_FLOAT32_NATIVE_ENDIAN_FILES'
             and domain['output'] == 'FRESH_FILE_WITH_EXISTING_PARENT'
             and domain['step_count_syntax'] == 'ASCII_DECIMAL_DIGITS_WITHOUT_SIGN_OR_WHITESPACE'
             and type(domain['step_count_min']) is int and domain['step_count_min'] == 1
             and type(domain['step_count_max']) is int and domain['step_count_max'] == (1 << 31) - 1,
             'source recurrence requires stable files, fresh output and a positive decimal step count')

    placement = execution['placement']
    mode, cpus = placement['mode'], placement['cpus']
    _require(mode in ('SERIAL', 'READY_DAG') and placement == dict(mode=manifest['rhs_mode'],
             cpus=manifest['cpus'], lanes=manifest['lanes'], operation_order=manifest['operation_order'])
             and len(cpus) == (1 if mode == 'SERIAL' else 3)
             and all(type(cpu) is int and 0 <= cpu < 1024 for cpu in cpus)
             and len(cpus) == len(set(cpus)), 'source entry cannot change the admitted CPU placement')
    linkage = qualify_cpu_dependencies(manifest['dependencies'])
    provider = manifest['provider']
    _require(manifest.get('cpu_linkage') == linkage and provider['name'] == 'LibTorch'
             and {PurePosixPath(row['path']).name for row in provider['libraries']} == {'libtorch_cpu.so', 'libc10.so'}
             and all(manifest['dependencies'].get(PurePosixPath(row['path']).name) == row['path']
                     for row in provider['libraries']),
             'source entry requires the verified CPU-only LibTorch dependency closure')

    file_bytes = sum(row['payload_bytes'] for row in bindings)
    output_bytes = bindings[state_port]['payload_bytes']
    storage = execution['storage']
    choice = manifest['alternatives'][mode]
    peak, budget = execution['peak_tensor_storage_bytes'], manifest['tensor_budget_bytes']
    _require(storage['scope'] == 'GRAPH_HELD_TENSOR_BACKING_STORAGE'
             and storage['raw_file_backing_bytes'] == file_bytes
             and storage['previous_step_state_bytes'] == output_bytes and storage['fixed_step_size_bytes'] == 8
             and type(budget) is int and type(peak) is int and file_bytes + output_bytes + 8 <= peak <= budget
             and storage['budget_bytes'] == budget and storage['peak_tensor_storage_bytes'] == peak
             and choice['within_budget'] is True and choice['peak_tensor_storage_bytes'] == peak,
             'source file and carried Tensor sizes differ from the verified storage budget')
    estimate = choice.get('body_estimate_ns')
    _require(execution['timing_basis'] == 'MEASURED_CPU_BODY_PER_STEP' and type(estimate) is int and estimate > 0,
             'source-entry comparison requires the admitted measured-step resource plan')

    return dict(schema=SCHEMA, schema_version=1,
        source_contract_sha256=_digest(abi), execution_plan_sha256=_digest(execution),
        rhs_recipe_sha256=_digest(rhs_recipe), program_graph=execution['program_graph'],
        target=execution['target'], placement=placement, provider=provider, cpu_linkage=linkage,
        source_argv=abi['argv'], files=bindings, read_order=[row['port'] for row in sorted(bindings,
             key=lambda row: row['load_order'])],
        recurrence=dict(control=execution['control'], rhs_bindings=rhs_bindings, step_size=abi['solver']['step_size'],
                        step_count=dict(minimum=domain['step_count_min'], maximum=domain['step_count_max'])),
        publication=publication, source_domain=domain,
        storage=dict(scope=storage['scope'], peak_bytes=peak, budget_bytes=budget,
                     excluded=storage['excluded_bytes']),
        costs=dict(process_startup_ns=None, input_load_ns=None, file_publication_ns=None,
                   complete_process_ns=None, native_body_per_step_estimate_ns=estimate,
                   body_estimate_scope='REPLAYED_NATIVE_STEP_SCHEDULE_ONLY',
                   logical_input_payload_bytes=file_bytes, logical_final_payload_bytes=output_bytes,
                   physical_io_bytes_proven=False),
        decision='CANDIDATE_FOR_EXACT_SOURCE_ENTRY_COMPARISON',
        original_execution_available=True, deployment_authorized=False)
