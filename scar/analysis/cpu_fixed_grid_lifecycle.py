"""Project the observed fixed-grid CLI domain into two source-backed lifecycles."""

import ast
from hashlib import sha256
from pathlib import Path
import shlex

from scar.analysis.cpu_fixed_grid_cli_effect_bridge import verify_cpu_fixed_grid_cli_effect_bridge
from scar.analysis.cpu_fixed_grid_cli_resources import verify_cpu_fixed_grid_cli_resources
from scar.analysis.fixed_grid_entry_abi import _span
from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.aten_recurrence_source import FRESH_SOURCE_FILE
from scar.compiler.cpu_fixed_grid_cli_delivery import verify_cpu_fixed_grid_cli_delivery
from scar.ir.execution_recipe import _canonical, read_artifact


SCHEMA = 'scar.cpu-fixed-grid-lifecycle'
_MAIN = 'src/main.cpp'
_IO = 'src/runtime/file_io.hpp'
_STEP = 'src/regions/step.hpp'
_LOOP = 'plan/execution.json'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid lifecycle: ' + detail)


def _one(rows, detail):
    rows = tuple(rows)
    _need(len(rows) == 1, detail)
    return rows[0]


def _anchor(reference, raw, start, end):
    _need(0 <= start < end <= len(raw), 'invalid source range')
    return dict(source=reference, byte_start=start, byte_end=end, selection_sha256=sha256(raw[start:end]).hexdigest())


def _site(reference, raw, start, end, *, after=False, operation, indent=None):
    span = _anchor(reference, raw, start, end)
    first = raw.rfind(b'\n', 0, start) + 1
    last = raw.find(b'\n', end)
    _need(not after or last >= 0, 'no line boundary after source marker')
    offset = last + 1 if after else first
    if indent is None:
        line = raw[first:raw.index(b'\n', first)]
        indent = line[:len(line)-len(line.lstrip(b' '))]
        _need(start == first + len(indent), 'expected source anchor at a statement start')
    _need(all(value == 32 for value in indent), 'insertion boundary is not space-indented')
    return dict(**span, operation=operation, byte_offset=offset, indentation=indent.decode('ascii'),
                position='AFTER' if after else 'BEFORE')


def _operation(kind, sources=(), *, resources=(), **metadata):
    return dict(kind=kind, source=list(sources), resources=list(resources), **metadata)


def _python_sites(source, raw, abi, entry):
    module = ast.parse(raw, filename=source['uri'])
    fn = _one((row for row in module.body if type(row) is ast.FunctionDef and row.name == entry),
              'original source CLI function is missing or ambiguous')
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1]+len(line))
    nodes = {row['id']: (index, row) for index, row in enumerate(abi['graph']['nodes'])}

    def bounds(span):
        line, left, end_line, right = span
        return offsets[line-1]+left, offsets[end_line-1]+right

    def mark(span, operation, *, after=False):
        return _site(source, raw, *bounds(span), operation=operation, after=after)

    def span(role):
        return nodes[role][1]['source_span']

    def statement(inside):
        begin, end = bounds(inside)
        return _one((row for row in fn.body
                     if (limit := bounds(_span(row)))[0] <= begin and limit[1] >= end),
                    'ABI occurrence does not belong to one original entry statement')

    imports = [row for row in module.body if type(row) in (ast.Import, ast.ImportFrom)]
    loads = sorted({statement(n['source_span']) for _, n in nodes.values() if n['kind'] == 'CPU_TENSOR_FILE_READ'},
                   key=lambda row: bounds(_span(row)))
    _need(bool(imports) and bool(loads), 'source must expose the admitted startup and file materialization')
    repeat = _span(statement(span('recurrence')))
    setup = nodes['thread_setup'][1]['source_spans']
    sites = dict(
        IMPORTS_BEGIN=mark(_span(imports[0]), 'initialization'),
        IMPORTS_RETURN=mark(_span(imports[-1]), 'initialization', after=True),
        MAIN_ENTER=mark(_span(fn.body[0]), 'provider_setup'),
        PROVIDER_SETUP_RETURN=mark(setup[-1], 'provider_setup', after=True),
        INPUTS_BEGIN=mark(_span(loads[0]), 'ingress'),
        INPUTS_READY=mark(_span(loads[-1]), 'recurrence', after=True),
        RECURRENCE_BEGIN=mark(repeat, 'recurrence'),
        RECURRENCE_RETURN=mark(repeat, 'recurrence', after=True),
        FILE_PUBLICATION_BEGIN=mark(span('file'), 'file_publication'),
        FILE_PUBLICATION_RETURN=mark(span('file'), 'file_publication', after=True),
        STDOUT_BEGIN=mark(span('stdout'), 'stdout_publication'),
        STDOUT_SUBMIT_END=mark(span('stdout'), 'stdout_publication', after=True),
    )
    anchors = {role: _anchor(source, raw, *bounds(row['source_span']))
               for role, (_, row) in nodes.items() if 'source_span' in row}
    anchors['thread_setup'] = [_anchor(source, raw, *bounds(value)) for value in setup]
    return sites, anchors, nodes


def _cpp_sites(project, manifest, placement):
    root = Path(project['uri']).parent
    refs = {member: dict(uri=str(root/member), schema='scar.native-cpp-source', **manifest['files'][member])
            for member in (_MAIN, _IO, _STEP)}
    raw = {name: read_artifact(ref, json_document=False) for name, ref in refs.items()}

    def match(member, text):
        data = raw[member]
        start = data.find(text)
        _need(start >= 0 and data.find(text, start+len(text)) == -1,
              'expected one replayed C++ statement or function anchor: ' + text.decode('ascii'))
        return start, start+len(text)

    def mark(member, text, operation, *, after=False, indent=None):
        return _site(refs[member], raw[member], *match(member, text), operation=operation, after=after, indent=indent)

    def anchor(member, text):
        return _anchor(refs[member], raw[member], *match(member, text))

    team = b'graph::CpuTeam team(std::array<int, 3>{' if placement['mode'] == 'READY_DAG' else b'at::NoGradGuard no_grad;'
    call = b'auto result = repeat(inputs, count'
    emit = b'write_fresh_source_file(destinations[0], outputs[0].const_data_ptr<float>(), sizes[0]);'
    submit = b'stdout_record(outputs, sizes);'
    sites = dict(
        MAIN_ENTER=mark(_MAIN, b'const bool probe = argc > 1', 'ingress'),
        RUNTIME_BEGIN=mark(_MAIN, b'at::set_num_threads(1);', 'provider_setup'),
        TEAM_CONSTRUCTOR_RETURN=mark(_MAIN, team, 'provider_setup', after=True),
        INPUTS_BEGIN=mark(_IO, b'std::array<std::vector<float>, graph::input_count> backing;', 'ingress'),
        INPUTS_READY=mark(_IO, b'const std::array<at::Tensor, graph::output_count> outputs = runner(inputs);', 'recurrence'),
        RECURRENCE_BEGIN=mark(_MAIN, call, 'recurrence'),
        RECURRENCE_RETURN=mark(_MAIN, call, 'recurrence', after=True),
        FILE_PUBLICATION_BEGIN=mark(_IO, emit, 'file_publication'),
        FILE_PUBLICATION_RETURN=mark(_IO, emit, 'file_publication', after=True),
        STDOUT_BEGIN=mark(_IO, submit, 'stdout_publication'),
        STDOUT_SUBMIT_END=mark(_IO, submit, 'stdout_publication', after=True),
    )
    sites['INVOKE_RETURN'] = dict(sites['STDOUT_SUBMIT_END'], operation='teardown')
    anchors = dict(
        ingress=[anchor(_MAIN, b'const auto count = parse_step_count('),
                 anchor(_MAIN, b'if (std::fegetround() != FE_TONEAREST)')],
        preflight=anchor(_IO, b'auto absolute = fs::canonical(parent) / path.filename();'),
        provider_setup=[anchor(_MAIN, b'at::set_num_interop_threads(1);'), anchor(_MAIN, team)],
        materialization=[anchor(_IO, b'backing[i] = read_tensor(paths[i], checked_bytes(shape));'),
                         anchor(_IO, b'inputs[i] = at::from_blob(backing[i].data()')],
        recurrence=[anchor(_MAIN, call), anchor(_STEP, b'for (std::int32_t iteration = 0; iteration < count; ++iteration)')],
        terminal=[anchor(_STEP, b'return {std::move(state)};'), anchor(_IO, b'!output.is_contiguous()')],
        file=[anchor(_IO, emit), anchor(_IO, b'if (close(fd) == -1) throw std::system_error')],
        stdout=[anchor(_IO, submit), anchor(_MAIN, b'}, load_order, [&](const auto&, const auto&) { std::cout')],
        invoke=anchor(_MAIN, b'return invoke(arguments.data(), [&](const auto& inputs) {'),
    )
    _need(sites['RECURRENCE_BEGIN']['byte_offset'] < sites['RECURRENCE_RETURN']['byte_offset']
          and sites['INPUTS_BEGIN']['byte_offset'] < sites['INPUTS_READY']['byte_offset']
          < sites['FILE_PUBLICATION_BEGIN']['byte_offset'] < sites['FILE_PUBLICATION_RETURN']['byte_offset']
          <= sites['STDOUT_BEGIN']['byte_offset'] < sites['STDOUT_SUBMIT_END']['byte_offset'],
          'generated site order differs from the admitted source CLI')
    return sites, anchors


def _launch_environment(delivery):
    launch = delivery['launch']
    launcher = read_artifact(delivery['native_launcher'], json_document=False)
    line = _one((row for row in launcher.decode('ascii').splitlines() if row.startswith('exec env -i ')),
               'reviewed native launcher lacks the qualified clear-environment exec')
    tokens = shlex.split(line)
    _need(tokens[:3] == ['exec', 'env', '-i'] and tokens.count('taskset') == 1,
          'native launcher environment scope differs')
    environment = dict(token.split('=', 1) for token in tokens[3:tokens.index('taskset')])
    _need(tokens[tokens.index('taskset')+1:tokens.index('taskset')+3]
          == ['-c', ','.join(map(str, delivery['cpu_ids']))], 'native launcher changed the reviewed CPU scope')
    return launch['provider_environment'], environment


def _node_routes(delivery, obligations, effects, resources, abi, projection):
    source, project = delivery['source'], delivery['project']
    manifest = read_artifact(project)
    _need(manifest['source_entry_cli'].get('source_file_publication') == FRESH_SOURCE_FILE
          and 'mapped_input_proof' not in manifest['source_entry_cli'],
          'expected the verified owned-copy and exclusive-output C++ route')
    physical = read_artifact(resources['binding'])
    raw = read_artifact(source, json_document=False)
    py_sites, py, nodes = _python_sites(source, raw, abi, effects['entry'])
    cpp_sites, cpp = _cpp_sites(project, manifest, delivery['native_placement'])
    py_env, cpp_env = _launch_environment(delivery)
    observed = delivery['launch']['commands']
    files = {row['port']: (index, row) for index, row in enumerate(resources['file_reads'])}
    read_order = physical['read_order']
    _need(set(read_order) == files.keys() and len(set(read_order)) == len(read_order),
          'file materialization order differs from the original physical binding')
    lane_rows = resources['cpu_lanes']
    cpu_scope = sorted({r['thread'] for r in lane_rows})
    memory = sorted({d for r in lane_rows for d in r['memory_candidates']})
    _need(delivery['cpu_ids'] == [r['cpu_id'] for r in lane_rows], 'physical lane inventory differs from the verified cohort')
    owner = [lane_rows[0]['thread']]
    file_role, stdout_role = effects['effects']['file'], effects['effects']['stdout']
    output = resources['file_publication']

    def storage_for(role):
        storage = role['storage']
        return [value for value in [storage['endpoint'], *storage['backing_devices']] if value is not None]

    def edges(order):
        return [dict(source=first, target=second, kind='NORMAL_ORDER', scope='SAME_ENTRY')
                for first, second in zip(order, order[1:])]

    def route(*, python):
        cpu, file_source, stdout_source = (cpu_scope, [py['file']], [py['stdout']]) if python else (owner, cpp['file'], cpp['stdout'])
        path = 'ORIGINAL_PYTHON' if python else 'NATIVE_CPP'
        import_declarations = [f'#/imports/{i}' for i in range(len(obligations['imports']))] if python else []
        ops = {
            'launch': _operation('EXECUTE_REVIEWED_COMMAND', resources=cpu_scope),
            'initialization': _operation('PYTHON_IMPORTS_AND_RUNTIME' if python else 'ELF_AND_LIBTORCH_INITIALIZATION',
                [py_sites['IMPORTS_BEGIN'], py_sites['IMPORTS_RETURN']] if python else (),
                resources=cpu_scope, original_import_requests=import_declarations, complete_effects_proven=False),
            'ingress': _operation('PARSE_SOURCE_ARGV' if python else 'PARSE_VALIDATE_SOURCE_ARGV',
                [py[key] for key in ('root', 'steps', 'output')] if python else cpp['ingress'], resources=cpu_scope),
            'provider_setup': _operation('SET_PROVIDER_THREAD_LIMITS' if python else 'SET_PROVIDER_THREAD_LIMITS_AND_PLACEMENT',
                py['thread_setup'] if python else cpp['provider_setup'], resources=cpu_scope),
            'recurrence': _operation('SOLVE_FIXED_GRID' if python else 'RUN_COUNTED_NATIVE_RECURRENCE',
                [py['recurrence']] if python else cpp['recurrence'], resources=[*cpu_scope, *memory],
                count=delivery['steps'], loop=dict(control=resources['loop']['control'], loops_unrolled=False,
                    shared_region_reused=True, runtime_internal_effects_proven=False),
                original_operation=effects['value_bindings']['recurrence']['operation_pointer'] if python else None),
            'terminal': _operation('EXTRACT_FINAL_CONTIGUOUS_VALUE' if python else 'VALIDATE_FINAL_CPU_FLOAT32_VALUE',
                [py['terminal']] if python else cpp['terminal'], resources=[*cpu, *memory]),
            'file_publication': _operation('WRITE_AND_CLOSE_FINAL_TENSOR', file_source,
                resources=[*cpu, *storage_for(output)], logical_bytes=output['bytes'],
                original_operation=file_role['entry_operation'] if python else None, physical_durability_proven=False),
            'stdout_publication': _operation('SUBMIT_SOURCE_JSON_LINE', stdout_source,
                resources=cpu, original_operation=stdout_role['entry_operation'] if python else None,
                stream_drain_proven=False),
            'teardown': _operation('LANGUAGE_AND_PROCESS_EXIT', () if python else (cpp['invoke'],),
                resources=cpu_scope, effects_proven=False),
        }
        if not python:
            ops['io_preflight'] = _operation('RESOLVE_AND_VALIDATE_FRESH_OUTPUT', (cpp['preflight'],),
                                             resources=[*cpu, *storage_for(output)])
        load_ids = []
        for port in read_order:
            index, row = files[port]
            role = f'load:{port}'
            load_ids.append(role)
            ordinal = abi['inputs'][port]['load_order']
            _need(nodes[f'tensor:{ordinal}'][1]['shape'] == row['shape'],
                  'file source occurrence changed its admitted tensor geometry')
            ops[role] = _operation('PRIVATE_FILE_BACKED_TENSOR_VIEW' if python else 'EAGER_FILE_READ_AND_BORROWED_TENSOR',
                [py[f'tensor:{ordinal}']] if python else cpp['materialization'],
                resources=[*cpu, *memory, *storage_for(row)],
                port=port, ordinal=ordinal, physical_port=f'#/file_reads/{index}', logical_bytes=row['bytes'],
                physical_read_or_first_touch_proven=False)
        head = ['launch', 'initialization', 'provider_setup', 'ingress'] if python else [
            'launch', 'initialization', 'ingress', 'provider_setup', 'io_preflight']
        ordered = [*head, *load_ids, 'recurrence', 'terminal', 'file_publication', 'stdout_publication', 'teardown']
        links = edges(ordered)
        for identity in load_ids:
            links.append(dict(source=identity, target='recurrence', kind='TENSOR_INPUT', scope='SAME_ENTRY',
                              port=ops[identity]['port']))
        links.extend((dict(source='ingress', target='recurrence', kind='ARGV_STEP_COUNT', scope='SAME_ENTRY'),
                      dict(source='ingress', target='file_publication', kind='ARGV_OUTPUT_PATH', scope='SAME_ENTRY'),
                      dict(source='terminal', target='file_publication', kind='FINAL_TENSOR_BYTES', scope='SAME_ENTRY')))
        fields = {row['kind'] for row in abi['publication']['stdout']}
        if 'FINAL_SHAPE' in fields:
            links.append(dict(source='terminal', target='stdout_publication', kind='FINAL_TENSOR_METADATA',
                              scope='SAME_ENTRY'))
        if 'STEP_COUNT' in fields:
            links.append(dict(source='ingress', target='stdout_publication', kind='ARGV_STEP_COUNT', scope='SAME_ENTRY'))
        return dict(command=observed[path], working_directory=delivery['working_directory'],
            observed_launch_environment=py_env, source_effective_environment=py_env if python else cpp_env,
            executable=effects['context']['interpreter'] if python else delivery['native_executable'],
            formal_cpu_cohort=delivery['cpu_ids'], lane_placement=delivery['native_placement'] if not python else None,
            operations=ops, edges=links, sites=py_sites if python else cpp_sites,
            recurrence_binding=projection if not python else None)

    return dict(original_python=route(python=True), native_cpp=route(python=False))


def derive_cpu_fixed_grid_lifecycle(delivery_ref, effects_ref, resources_ref):
    """Verify exact normal-success receipts and project both entries without workload execution."""
    with replay_transaction():
        delivered = verify_cpu_fixed_grid_cli_delivery(delivery_ref)
        effects = verify_cpu_fixed_grid_cli_effect_bridge(effects_ref)
        resources = verify_cpu_fixed_grid_cli_resources(resources_ref)
        _need(delivered['status'] == 'SCOPED_SOURCE_CLI_VALIDATED'
              and all(delivered['validation'][field] for field in ('exact_stdout', 'exact_stderr', 'exact_exit_status',
                                                                  'bitwise_final_file', 'exact_publication_mode'))
              and effects['status'] == 'PROVEN_SCOPED_NORMAL_SUCCESS_SOURCE_CLI_EFFECT_BRIDGE'
              and effects['normal_completion_order']['status'] == 'PROVEN_SCOPED_NORMAL_SUCCESS'
              and delivered['target'] == effects['target'] == resources['target']
              and delivered['argv'] == effects['argv'] == resources['argv']
              and delivered['source'] == effects['source'] and delivered['provider'] == effects['provider']
              and delivered['parent_project'] == effects['source_project'] and delivered['project'] == resources['project']
              and delivered['working_directory'] == effects['context']['cwd']
              and delivered['steps'] == effects['fixed_grid']['steps'] == resources['loop']['steps']
              and delivered['validated_final_tensor']['size_bytes'] == resources['file_publication']['bytes']
              and effects['fixed_grid']['inputs'] == resources['inputs']
              and all((before['path'], before['sha256'], before['size_bytes'])
                    == (after['uri'], after['sha256'], after['size_bytes'])
                    for before, after in zip(delivered['file_inputs'], resources['inputs'], strict=True)),
              'delivery, original effects and resources must pin one exact source and invocation')
        binding = read_artifact(resources['binding'])
        abi = read_artifact(resources['abi'])
        _need(binding['source_contract_sha256'] == effects['source_contract_sha256']
              and binding['placement'] == delivered['native_placement'],
              'source ABI or native placement differs from the verified lifecycle')
        project = delivered['project']
        manifest = read_artifact(project)
        plan = dict(uri=str(Path(project['uri']).parent/_LOOP), schema='scar.aten-recurrence-plan', **manifest['files'][_LOOP])
        routes = _node_routes(delivered, read_artifact(effects['entry_obligations']), effects, resources, abi, plan)
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_SCOPED_NORMAL_SUCCESS_FIXED_GRID_LIFECYCLE',
        delivery=delivery_ref, effects=effects_ref, resources=resources_ref, target=delivered['target'],
        source=delivered['source'], project=project,
        scope=dict(argv=delivered['argv'], cpu_ids=delivered['cpu_ids'], steps=delivered['steps'],
                   working_directory=delivered['working_directory'], environment_sha256=delivered['environment_sha256'],
                   boundary='PROCESS_LAUNCH_THROUGH_EXIT', source_contract_sha256=effects['source_contract_sha256']),
        validated_final_tensor=delivered['validated_final_tensor'], validated_source_stdout=delivered['validated_source_stdout'],
        routes=routes, shared_stage_costs_assumed=False, physical_storage_io_bytes_proven=False,
        performance_bounds_proven=False, whole_entry_effect_equivalence_proven=False,
        coverage_frontier=['LOADER_IMPORT_AND_PROVIDER_INTERNAL_EFFECTS', 'FILE_FIRST_TOUCH_PAGE_FAULTS_AND_CACHE_STATE',
            'PHYSICAL_NUMA_RESIDENCY', 'STORAGE_DURABILITY_AND_STDOUT_DRAIN', 'TEARDOWN_AND_FAILURE_PATHS'],
        target_executed=False, transformation_authorized=False)


def verify_cpu_fixed_grid_lifecycle(reference):
    _need(reference['schema'] == SCHEMA, 'unexpected fixed-grid lifecycle schema')
    saved = read_artifact(reference)
    expected = derive_cpu_fixed_grid_lifecycle(saved['delivery'], saved['effects'], saved['resources'])
    _need(_canonical(saved) == _canonical(expected), 'saved fixed-grid lifecycle differs on cold replay')
    return saved
