"""Derive value-independent storage guards for an explicit ATen trial."""

from math import prod

from scar.compiler.aten_native_recipe import normalize_recipe


SCHEMA = 'scar.aten-value-independent-geometry'
CONTRACT_SCHEMA = 'scar.aten-value-independent-geometry-contract'
SCOPE = 'EXPLICIT_PYTHON_CPU_DIFFERENTIAL_TRIAL'
_CONTRACT_FIELDS = {'schema', 'schema_version', 'scope', 'provider_revision', 'operators'}
_OPERATOR_FIELDS = {'schema', 'fixed_strided_float32_cpu_storage', 'value_independent_effects',
                    'replay_safe_no_observable_effects'}
_MAX_STORAGE_BYTES = (1 << 63) - 1


def _declared_operators(contract, torch):
    revision = torch.version.git_version
    if (type(contract) is not dict or contract.keys() != _CONTRACT_FIELDS
            or type(contract['schema_version']) is not int
            or (contract['schema'], contract['schema_version'], contract['scope']) != (CONTRACT_SCHEMA, 1, SCOPE)
            or not revision or contract['provider_revision'] != revision):
        raise ValueError('an explicit geometry contract for the installed ATen revision is required')
    rows = contract['operators']
    if type(rows) is not list or not rows:
        raise ValueError('the geometry contract must declare every graph operator')
    schemas = {}
    for row in rows:
        if (type(row) is not dict or row.keys() != _OPERATOR_FIELDS
                or type(row['schema']) is not str or not row['schema'].startswith('aten::')
                or row['schema'] in schemas or row['fixed_strided_float32_cpu_storage'] is not True
                or row['value_independent_effects'] is not True or row['replay_safe_no_observable_effects'] is not True):
            raise ValueError('each geometry operator needs exact storage and replay-safe effect declarations')
        schemas[row['schema']] = row
    return schemas


def _meta_operator(text, torch):
    declared = torch._C.parse_schema(text)
    dispatch_name = declared.name
    if declared.overload_name:
        dispatch_name += '.' + declared.overload_name
    if not torch._C._dispatch_has_computed_kernel_for_dispatch_key(dispatch_name, 'Meta'):
        raise ValueError(f'geometry trial has no installed Meta route for {dispatch_name}')
    packet = getattr(torch.ops.aten, declared.name.removeprefix('aten::'))
    operator = getattr(packet, declared.overload_name or 'default')
    if str(operator._schema) != text:
        raise ValueError(f'geometry trial Meta callable differs from the registered schema: {dispatch_name}')
    return operator, declared.arguments


def _tensor_geometry(tensor, location, torch):
    if (type(tensor) is not torch.Tensor or tensor.device.type != 'meta' or tensor.dtype is not torch.float32
            or tensor.layout is not torch.strided or tensor.requires_grad or tensor.storage_offset() != 0
            or tensor.is_conj() or tensor.is_neg() or not tensor.is_contiguous()):
        raise ValueError(f'{location}: Meta result is not a dense base float32 Tensor')
    shape, strides = list(tensor.shape), list(tensor.stride())
    if any(type(value) is not int or not 0 <= value <= _MAX_STORAGE_BYTES for value in (*shape, *strides)):
        raise ValueError(f'{location}: Meta result geometry is not a bounded concrete integer shape')
    storage_bytes = tensor.untyped_storage().nbytes()
    if (type(storage_bytes) is not int or not 0 <= storage_bytes <= _MAX_STORAGE_BYTES
            or storage_bytes != 4 * prod(shape)):
        raise ValueError(f'{location}: Meta result does not have exact dense float32 storage')
    return dict(shape=shape, strides=strides, storage_bytes=storage_bytes)


def _check_calibrated_storage(calibration, key, geometries):
    measured = calibration.get(key)
    if (type(measured) is not list or any(type(value) is not int for value in measured)
            or measured != [row['storage_bytes'] for row in geometries]):
        raise ValueError(f'{key}: recorded CPU storage differs from the propagated Meta geometry')


def qualify_trial_geometry(recipe, contract, calibration, torch):
    """Replay exact Meta overloads and reconcile their geometry with CPU storage evidence.

    Meta inference describes geometry only. Value-independent CPU storage and
    effects are declared per operator, including safe replay of the original
    call after native rejection. The trial uses the same shapes and ATen revision.
    """
    declared = _declared_operators(contract, torch)
    recipe = normalize_recipe(recipe, declared.values(), torch)
    schemas = {row['schema'] for row in recipe['operations']}
    if declared.keys() != schemas:
        raise ValueError('the geometry contract must match the exact graph operator set')
    if type(calibration) is not dict:
        raise ValueError('recorded CPU storage calibration is required for geometry qualification')

    values, inputs, operations = {}, [], []
    operators = {schema: _meta_operator(schema, torch) for schema in declared}
    with torch.no_grad():
        for item in recipe['inputs']:
            tensor = torch.empty(item['shape'], device='meta', dtype=torch.float32)
            inputs.append(_tensor_geometry(tensor, item['id'], torch))
            values[item['id']] = tensor
        _check_calibrated_storage(calibration, 'input_storage_bytes', inputs)
        for item in recipe['operations']:
            operator, signature = operators[item['schema']]
            positional, keywords = [], {}
            for argument, parameter in zip(item['arguments'], signature, strict=True):
                operand = values[argument['value']] if 'value' in argument else argument['scalar']
                if parameter.kwarg_only:
                    keywords[parameter.name] = operand
                else:
                    positional.append(operand)
            try:
                result = operator(*positional, **keywords)
            except (RuntimeError, TypeError) as error:
                raise ValueError(f"{item['id']}: installed Meta inference failed for {item['schema']}") from error
            operations.append(dict(id=item['id'], **_tensor_geometry(result, item['id'], torch)))
            values[item['id']] = result
    _check_calibrated_storage(calibration, 'result_storage_bytes', operations)

    return dict(schema=SCHEMA, schema_version=1, scope=SCOPE, provider_revision=contract['provider_revision'],
                basis='EXACT_INSTALLED_META_OVERLOADS_AND_CPU_STORAGE_CALIBRATION',
                effect_basis='EXPLICIT_PER_OPERATOR_DECLARATION', inputs=inputs, operations=operations)
