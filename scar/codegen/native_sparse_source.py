"""Lower the source CSR preparation and restricted scalar tree to C++20."""

from math import isfinite

from scar.optimization.native_ordered_stream import project_native_detached_stream


SOURCE_STREAM = 'scar.native-ordered-source-csr-stream'
_OPS = {'Add': '+', 'Sub': '-', 'Mult': '*', 'Div': '/'}
_EXACT_INTEGER_LIMIT = 2**53


def _scalar(tree, count):
    kind, left, *right = tree
    if not right and kind == 'input' and type(left) is int:
        if left == 0:
            return 'static_cast<std::int64_t>(sequence)', (0, count - 1)
        if left == 1:
            return 'static_cast<std::int64_t>(sparse::count)', (count, count)
    if not right and kind == 'constant':
        if type(left) is int and abs(left) <= _EXACT_INTEGER_LIMIT:
            return f'std::int64_t{{{left}}}', (left, left)
        if type(left) is float and isfinite(left):
            return left.hex(), None
    if kind not in _OPS or len(right) != 1:
        raise ValueError('unsupported native source scale expression')

    a, a_int = _scalar(left, count)
    b, b_int = _scalar(right[0], count)
    operator = _OPS[kind]
    if operator == '/' and (b_int is None or b_int[0] <= 0 <= b_int[1]):
        raise ValueError('native source division requires a provably nonzero exact integer divisor')
    if operator == '/' or a_int is None or b_int is None:
        return f'(static_cast<double>({a}) {operator} static_cast<double>({b}))', None

    low, high = a_int
    other_low, other_high = b_int
    if operator == '+':
        bounds = low + other_low, high + other_high
    elif operator == '-':
        bounds = low - other_high, high - other_low
    else:
        products = low * other_low, low * other_high, high * other_low, high * other_high
        bounds = min(products), max(products)
    if max(map(abs, bounds)) > _EXACT_INTEGER_LIMIT:
        raise ValueError('native source scale exceeds exact integer-to-binary64 range')
    return f'({a} {operator} {b})', bounds


def render_source_scale(tree, count):
    expression, _ = _scalar(tree, count)
    return f'static_cast<double>({expression})'


def project_source_csr_stream(resource_plan, detached):
    compact = project_native_detached_stream(resource_plan, detached)
    header, sizes = compact['header'], compact['bytes']
    render_source_scale(detached['scale'], header['count'])
    abi = compact['input']
    fixed = sizes['input_fixed']
    return dict(compact, schema=SOURCE_STREAM,
        source={**compact['source'], 'contract': 'EXPLICIT_CALLER_CANONICAL_CSR_RHS_AND_SELECTED_ROWS'},
        input=dict(magic='SCARPSR1', structure=abi['structure'],
            original_values=dict(dtype='<f8', count=header['nnz'], range='FINITE'),
            original_rhs=abi['base_rhs'], row_count=dict(format='<I', maximum=header['size']),
            rows=dict(dtype='<i4', order='ARBITRARY_UNIQUE', contract='STORED_DIAGONAL_IN_EVERY_ROW'),
            request=dict(sequence='IMPLICIT_ZERO_BASED', scale='EXACT_NATIVE_SOURCE_EXPRESSION'),
            framing='REQUIRE_EOF_BEFORE_ANY_DISPATCH'),
        implementation={**compact['implementation'],
            'initialization': 'FIND_STORED_DIAGONALS_AND_MASK_SELECTED_ROWS_WITHOUT_ARITHMETIC_REORDERING',
            'scale': 'CXX20_BINARY64_WITHOUT_FP_CONTRACTION'},
        bytes=dict(input_fixed=fixed, selected_row=4, input_complete_max=fixed + 4 * header['size'],
            output_complete=sizes['output_complete'], raw_input_complete=sizes['raw_input_complete'],
            raw_request_bytes_per_worker=sizes['raw_request_bytes_per_worker'],
            compact_request_bytes_per_worker=sizes['compact_request_bytes_per_worker']))
