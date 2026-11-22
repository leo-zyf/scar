"""Qualify private file-backed Tensor input for a checked fixed-grid CPU CLI."""

import ast
import json
from math import prod
from pathlib import Path
import re
import subprocess

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler.aten_entry_delivery import ABI_FILE, BINDING_FILE
from scar.ir.execution_recipe import _canonical, artifact_reference, read_artifact
from scar.runtime.validate_native_project import verify_native_project


SCHEMA = 'scar.cpu-fixed-grid-mapped-input-proof'
_PROJECT = 'scar.aten-recurrence-delivery'
_FILE_IO = 'src/runtime/file_io.hpp'
_STEP = 'src/regions/step.hpp'
_OPERATORS = 'src/providers/operators.hpp'
_TEAM = 'src/runtime/aten_native_team_v1.h'
_READY = 'src/runtime/aten_ready_dag_v1.h'
_DECLARED_SCHEMA = re.compile(rb'^  static constexpr const char\* schema_str = "([^"\n]+)";$', re.M)


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid mapped inputs: ' + detail)


def _member(root, manifest, name, schema='application/octet-stream'):
    path = root / name
    _need(path == path.resolve(strict=True) and path.is_relative_to(root),
          'checked source project member has an indirect path')
    reference = dict(uri=str(path), schema=schema, **manifest['files'][name])
    raw = read_artifact(reference, json_document=False)
    return reference, raw


def _document(root, manifest, name, schema):
    reference, raw = _member(root, manifest, name, schema)
    value = json.loads(raw)
    _need(value['schema'] == schema, 'typed project member differs from its declared schema')
    return reference, value


def _file_symbols(library):
    """Check exported ATen mapping entry points without loading the operator library."""
    wanted = ('at::_ops::from_file::call(', 'at::cpu::from_file(')
    symbols = {stem: [] for stem in wanted}
    with subprocess.Popen(['/usr/bin/readelf', '-W', '-C', '--dyn-syms', library], env={'LC_ALL': 'C'},
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) as process:
        for line in process.stdout:
            for stem in wanted:
                if stem in line:
                    cols = line.split(None, 7)
                    if len(cols) == 8 and cols[3:6] == ['FUNC', 'GLOBAL', 'DEFAULT'] and cols[6].isdigit():
                        symbols[stem].append(cols[7].rstrip())
        _need(process.wait() == 0 and all(symbols.values()), 'pinned LibTorch CPU mapping exports are unavailable')
    return sorted(symbol for rows in symbols.values() for symbol in rows)


def _provider(manifest, schemas):
    # This parser only reads function schema declarations. Its code is matched
    # byte-for-byte with the tooling shipped next to the linked CPU provider.
    from torchgen import model

    installed = manifest['provider']
    cpu = Path(manifest['dependencies']['libtorch_cpu.so'])
    package = cpu.parent.parent
    version_file = package / 'version.py'
    version_ref = artifact_reference(version_file, 'scar.aten-provider-version-source')
    version = {}
    for node in ast.parse(read_artifact(version_ref, json_document=False)).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ('__version__', 'git_version', 'cuda'):
                _need(target.id not in version and isinstance(node.value, ast.Constant),
                      'provider version is not a unique literal')
                version[target.id] = node.value.value
    _need(version == dict(__version__=installed['version'], git_version=installed['revision'], cuda=None),
          'native project is not linked to the exact CPU declaration package')
    parsers = []
    for file in ('__init__.py', 'model.py', 'utils.py', 'code_template.py'):
        linked = artifact_reference(package.parent / 'torchgen' / file, 'scar.aten-schema-parser-source')
        host = artifact_reference(Path(model.__file__).parent / file, 'scar.aten-schema-parser-source')
        _need(linked['sha256'] == host['sha256'] and linked['size_bytes'] == host['size_bytes'],
              'schema parser differs from the linked ATen provider')
        parsers.append(dict(provider=linked, parser=host))

    def declare(selector, exact=None):
        op, _, _ = selector.partition('.')
        _need(bool(re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*', op)), 'invalid ATen schema selector')
        header = artifact_reference(package/'include'/'ATen'/'ops'/(op + '_ops.h'), 'scar.aten-generated-ops-header')
        candidates = [text.decode('ascii') for text in _DECLARED_SCHEMA.findall(read_artifact(header, json_document=False))
                      if text.split(b'(', 1)[0].decode('ascii') == selector]
        _need(len(candidates) == 1 and (exact is None or exact == 'aten::'+candidates[0]),
              'retained operator differs from the generated ATen declaration: ' + selector)
        signature = model.FunctionSchema.parse(candidates[0])
        _need(len(signature.returns) == 1 and str(signature.returns[0].type) == 'Tensor',
              'mapped input proof requires a single Tensor result declaration')
        return signature, dict(schema_text='aten::' + candidates[0], generated_header=header,
            input_aliases=[None if p.annotation is None else str(p.annotation) for p in signature.arguments.flat_all],
            output_alias=None if signature.returns[0].annotation is None else str(signature.returns[0].annotation),
            declared_mutable=signature.is_mutable)

    declarations = {}
    for schema in sorted(set(schemas)):
        selector = schema.removeprefix('aten::').partition('(')[0]
        signature, row = declare(selector, exact=schema)
        _need(not signature.is_mutable and row['output_alias'] is None and not any(row['input_aliases']),
              'input may be written or returned by a recurrence/RHS operator')
        declarations[schema] = row
    schema, mapping = declare('from_file')
    _need([p.name for p in schema.arguments.flat_all] == ['filename', 'shared', 'size', 'dtype', 'layout', 'device', 'pin_memory']
          and not schema.is_mutable and mapping['output_alias'] is None and not any(mapping['input_aliases']),
          'ATen file mapping declaration is outside the supported private CPU route')
    schema, reshape = declare('view')
    _need([str(p.type) for p in schema.arguments.flat_all] == ['Tensor', 'SymInt[]']
          and not schema.is_mutable and reshape['input_aliases'] == ['a', None] and reshape['output_alias'] == 'a',
          'ATen view must preserve exactly one mapping alias without a write')
    dispatch = artifact_reference(package/'include'/'ATen'/'ops'/'from_file_cpu_dispatch.h',
                                  'scar.aten-generated-cpu-dispatch-header')
    _need(b'namespace cpu {' in read_artifact(dispatch, json_document=False)
          and b'TORCH_API at::Tensor from_file(' in read_artifact(dispatch, json_document=False),
          'the exact provider has no declared CPU file mapping route')
    proof = dict(name=installed['name'], version=installed['version'], revision=installed['revision'],
                 library_locks=installed['libraries'], version_source=version_ref,
                 schema_parsers=parsers, from_file=mapping, view=reshape, from_file_cpu_header=dispatch,
                 defined_mapping_exports=_file_symbols(str(cpu)))
    return proof, declarations


def _uses(operations, value, schema_field):
    return [dict(operation=row['id'], operation_index=index, argument=argument, schema=row[schema_field])
            for index, row in enumerate(operations)
            for argument, operand in enumerate(row['arguments']) if operand.get('value') == value]


def _inputs(abi, binding, execution, rhs):
    loop = execution['control']
    state = execution['state_input_index']
    carry = loop['carried_state']
    numerical = execution['operations']
    rhs_outputs = rhs['outputs']
    _need(loop['kind'] == 'FIXED_COUNT_LOOP' and loop['body_count'] == 1
          and carry['initial_entry_input'] == state and carry['distance'] == 1
          and carry['body_output'] == execution['output'] != carry['body_input']
          and carry['body_input'] == execution['inputs'][0]['id']
          and carry['body_output'] in {row['id'] for row in numerical}
          and len(rhs_outputs) == 1 and rhs_outputs[0] in {row['id'] for row in rhs['operations']}
          and len(rhs['inputs']) == len(abi['inputs']) == len(binding['files'])
          and loop['invariant_entry_inputs'] == [port for port in range(len(rhs['inputs'])) if port != state],
          'the recurrence cannot keep or publish a borrowed mapping as a result')

    rhs_sites = {item['operation']: item['arguments'] for item in execution['rhs_bindings']}
    _need(set(rhs_sites) == {row['id'] for row in numerical if row['op'] == 'RHS'},
          'recurrence schedules an unqualified RHS expansion')
    for row in numerical:
        if row['op'] == 'RHS':
            routes = [row['arguments'][0] if port == state else {'entry_input': port} for port in range(len(rhs['inputs']))]
            _need(rhs_sites[row['id']] == routes, 'RHS receives a different alias or argument route')
    result = []
    for index, (bound, declared, rhs_input) in enumerate(zip(binding['files'], abi['inputs'], rhs['inputs'], strict=True)):
        shape = rhs_input['shape']
        elements = prod(shape)
        _need(bound == dict(declared, value=rhs_input['id'], payload_bytes=4*elements)
              and declared['port'] == index and declared['shape'] == shape and declared['dtype'] == 'FLOAT32'
              and len(shape) == 2 and all(type(extent) is int and extent > 0 for extent in shape)
              and elements < 1 << 61, 'source file port or size differs from the native RHS')
        reads = _uses(rhs['operations'], rhs_input['id'], 'schema')
        _need(bool(reads), 'input has no qualified numerical consumer')
        direct = _uses(numerical, carry['body_input'], 'native_schema') if index == state else []
        result.append(dict(bound, element_count=elements, role='INITIAL_CARRIED_STATE' if index == state else 'INVARIANT_RHS_INPUT',
            rhs_reads=reads, recurrence_reads=[row for row in direct if row['schema'] is not None],
            rhs_entry_sites=[name for name, operands in rhs_sites.items()
                             if index != state or operands[index] == {'value': carry['body_input']}],
            scoped_declared_input_write=False, scoped_graph_result_alias=False, scoped_graph_escape=False))
    return result


def _source_material(root, manifest, placement):
    raw = {}
    members = {}
    for name in (_FILE_IO, _STEP, _OPERATORS, 'src/main.cpp', *(() if placement['mode'] == 'SERIAL' else (_TEAM, _READY))):
        members[name], raw[name] = _member(root, manifest, name, 'scar.native-cpp-source')
    _need(all(raw[_FILE_IO].count(text) == 1 for text in
        (b'std::array<std::vector<float>, graph::input_count> backing;',
         b'backing[i] = read_tensor(paths[i], checked_bytes(shape));',
         b'inputs[i] = at::from_blob(backing[i].data(), at::IntArrayRef(shape.data(), shape.size()),'))
        and b'at::from_file(' not in raw[_FILE_IO]
        and raw[_STEP].count(b'auto state = inputs[state_index];') == 1
        and raw[_STEP].count(b'rhs_inputs[state_index].reset();') == 1
        and raw[_STEP].count(b'return {std::move(state)};') == 1,
          'generated source does not expose the expected borrowed-input lifecycle')
    if placement['mode'] == 'READY_DAG':
        _need(raw[_READY].count(b'team.run_lanes_with_abort(used, run_lane, abort);') == 1
              and raw[_TEAM].count(b'complete_.wait(lock, [this] { return !workers_[0].job && !workers_[1].job; });') == 1
              and b'worker.job = {};' in raw[_TEAM],
              'parallel RHS may retain a borrowed task across return')
    return members


def derive_cpu_fixed_grid_mapped_inputs(project_ref):
    """Check normal-success file-to-Tensor legality without loading workload inputs."""
    _need(project_ref['schema'] == _PROJECT and Path(project_ref['uri']).name == 'artifact.json',
          'expected a sealed native fixed-grid source CLI project')
    with replay_transaction():
        manifest = read_artifact(project_ref)
        root, checked = verify_native_project(Path(project_ref['uri']).parent, project_ref['sha256'])
        descriptor = manifest.get('source_entry_cli', {})
        _need(checked.get('source_entry_cli_replayed') is True and checked['native_dependencies_checked'] is True
              and manifest['cxx_standard'] == 20 and descriptor.get('abi') == ABI_FILE
              and descriptor.get('physical_binding') == BINDING_FILE and 'mapped_input_proof' not in descriptor,
              'baseline must be the checked native source CLI with an eager reader')
        abi_ref, abi = _document(root, manifest, ABI_FILE, 'scar.fixed-grid-file-entry')
        binding_ref, binding = _document(root, manifest, BINDING_FILE, 'scar.aten-source-entry-binding')
        execution_ref, execution = _document(root, manifest, 'plan/execution.json', 'scar.aten-recurrence-plan')
        rhs_ref, rhs = _document(root, manifest, 'plan/rhs-recipe.json', 'scar.aten-native-recipe')
        schedule, _ = _document(root, manifest, 'plan/schedule.json', 'scar.implementation-execution-plan')
        target = binding['target']
        read_artifact(target)
        source_row = abi['solver']['source']
        source = dict(uri=source_row['path'], schema='scar.python-source',
                      sha256=source_row['sha256'], size_bytes=source_row['size_bytes'])
        original_source, expected = _member(root, manifest, 'plan/source-entry.py', 'scar.python-source')
        _need(expected == read_artifact(source, json_document=False)
              and abi['domain'] == binding['source_domain'] and abi['domain']['normal_completion_only'] is True
              and abi['domain']['inputs'] == 'STABLE_READ_ONLY_EXACT_SIZE_REGULAR_CPU_FLOAT32_NATIVE_ENDIAN_FILES'
              and abi['domain']['output'] == 'FRESH_FILE_WITH_EXISTING_PARENT'
              and abi['argv'] == binding['source_argv'] and abi['publication'] == binding['publication']
              and execution['placement'] == binding['placement'] and execution['target'] == target,
              'source, CLI ports or fixed-grid schedule differ from the admitted mapping domain')
        placement = binding['placement']
        native = _source_material(root, manifest, placement)
        inputs = _inputs(abi, binding, execution, rhs)
        provider, declarations = _provider(manifest, [row['schema'] for row in rhs['operations']]
            + [row['native_schema'] for row in execution['operations'] if row['op'] != 'RHS'])
    payload = sum(row['payload_bytes'] for row in inputs)
    _need(payload == binding['costs']['logical_input_payload_bytes'] == execution['storage']['raw_file_backing_bytes'],
          'explicit eager-copy payload differs from original storage accounting')
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_SCOPED_MAPPED_INPUT_MATERIALIZATION',
        project=project_ref, parent_manifest_sha256=descriptor['parent_manifest_sha256'], source=source,
        archived_source=original_source, source_contract_sha256=binding['source_contract_sha256'],
        abi=abi_ref, physical_binding=binding_ref, execution=execution_ref, rhs_recipe=rhs_ref,
        schedule=schedule, rhs_schedule=execution['rhs_schedule'], target=target, placement=placement,
        scope=dict(source_argv=abi['argv'], source_domain=abi['domain'], publication=abi['publication'],
                   from_file_shared=False, normal_success_only=True, stable_inputs_required_for_full_lifetime=True,
                   distinct_fresh_output_required=True, provider_schema_conformance_required=True),
        inputs=inputs, input_read_order=binding['read_order'], provider=provider, operator_declarations=declarations,
        native_members=native, logical_copy_payload_bytes=payload,
        materialization=dict(baseline='EAGER_IFSTREAM_READ_VECTOR_THEN_BORROWED_FROM_BLOB',
            candidate='ATEN_FROM_FILE_PRIVATE_CPU_FLOAT32_THEN_SHAPE_VIEW',
            replaces_source_cli_only=True, source_read_order_preserved=True,
            original_python_uses_same_file_operator=True, file_path_and_shape_unchanged=True),
        effects=dict(graph_input_writes_absent=True, graph_result_alias_absent=True, graph_input_escape_absent=True,
            synchronous_rhs_completion=True, general_native_internal_effects_closed=False,
            file_stability_observed=False, failure_equivalence_proven=False, whole_entry_effect_equivalence_proven=False),
        page_residency_proven=False, physical_io_bytes_proven=False, performance_bounds_proven=False,
        coverage_frontier=['FILE_STABILITY_AND_PATH_IDENTITY_AT_RUN_TIME', 'LAZY_FAULT_TIMING_AND_PAGE_CACHE_BEHAVIOR',
            'NUMA_PLACEMENT_AND_RESIDENT_MEMORY', 'PROVIDER_INTERNAL_EFFECTS_AND_FAILURE_PATHS'],
        target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_fixed_grid_mapped_inputs(reference):
    _need(reference['schema'] == SCHEMA, 'unexpected mapped-input proof schema')
    saved = read_artifact(reference)
    expected = derive_cpu_fixed_grid_mapped_inputs(saved['project'])
    _need(_canonical(saved) == _canonical(expected), 'saved mapped-input proof differs on cold replay')
    return saved
