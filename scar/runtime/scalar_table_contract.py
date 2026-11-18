"""Shared code artifact contract for scalar table compilation and loading."""

from types import CodeType

from scar.trace.code_snapshot import code_key


PROOF = 'scar.scalar-table-cse'
SCHEMA = 'scar.scalar-table-region'
ENTRY_CODE = 'scar.python-scalar-table-region-code'
OFFER = '_scar_scalar_table_offer_v1'
MISS = '_scar_scalar_table_miss_v1'
RESULT = '_scar_scalar_table_result_v1'


def compile_original(raw, recipe):
    if (set(recipe) != {'filename', 'mode', 'flags', 'dont_inherit', 'optimize'}
            or recipe['mode'] != 'exec' or recipe['flags'] != 0 or recipe['dont_inherit'] is not True
            or type(recipe['optimize']) is not int or recipe['optimize'] not in (0, 1, 2)):
        raise ValueError('scalar table region requires an ordinary original Python source compilation')
    return compile(raw, recipe['filename'], 'exec', dont_inherit=True, optimize=recipe['optimize'])


def replace_leaf(root, original_key, revised):
    """Change one leaf while retaining the original module and sibling bytecode."""
    count = 0

    def splice(code):
        nonlocal count
        if code_key(code) == original_key:
            count += 1
            if code.co_qualname != revised.co_qualname or code.co_filename != revised.co_filename:
                raise ValueError('revised code does not belong to the original function')
            return revised
        updated = tuple(splice(child) if type(child) is CodeType else child for child in code.co_consts)
        return code if all(a is b for a, b in zip(code.co_consts, updated, strict=True)) else code.replace(co_consts=updated)

    combined = splice(root)
    if count != 1 or combined.co_code != root.co_code:
        raise ValueError('original main code is not a unique leaf splice')
    return combined
