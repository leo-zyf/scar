"""Normal-completion contract for one exact ndarray C-order copy.

The copied contents are equal at this operation under source stability. The
result is a fresh independent materialization, never a view or an assertion
of equal runtime allocation, object, invocation or logical-version identity.
"""
from types import MappingProxyType


COPY_CONTRACT='numpy.ndarray.copy.content-fresh.v1'
COPY_ATTRS=MappingProxyType({'order':'C','contract':COPY_CONTRACT})
COPY_REQUIRED_GUARDS=(
    'EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR',
    'ARRAY_INPUT_CONTENT_STABLE_DURING_REGION',
)


def copy_attributes():return dict(COPY_ATTRS)


def validate_copy_attributes(attrs):
    """Require the canonical static contract; return a detached attribute map.

    This is format validation, not discharge of the required runtime/profile
    guards or permission to remove an allocation or expose its identity.
    """
    if type(attrs)is not dict or attrs!=COPY_ATTRS:
        raise ValueError('copy requires exact ndarray C-order content/fresh-materialization attributes')
    return copy_attributes()


def copy_semantics():
    return dict(contract=COPY_CONTRACT,relation='copy',equivalence='EXACT_CONTENT_AT_SELECTED_COPY',
        source_content='READ_UNDER_SOURCE_STABILITY',result_content='SAME_CONTENT_ON_NORMAL_COMPLETION',
        materialization='FRESH_INDEPENDENT_C_ORDER_ALLOCATION',aliases_source=False,
        writes='NEW_OUTPUT_ONLY',allocates='FRESH_OUTPUT',may_raise='NUMPY_OR_ALLOCATION_FAILURE',
        equivalence_scope='NORMAL_COMPLETION_NOT_FULL_EXCEPTION_OR_INTERNAL_ALLOCATION_SEQUENCE',
        required_guards=list(COPY_REQUIRED_GUARDS),runtime_value_or_storage_identity_proven=False,
        transformation_authorized=False)


__all__=['COPY_CONTRACT','COPY_ATTRS','COPY_REQUIRED_GUARDS','copy_attributes',
    'validate_copy_attributes','copy_semantics']
