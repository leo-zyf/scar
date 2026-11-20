"""Allocation-relative geometry for views of a live, exact NumPy ndarray.

Only shape and storage metadata enter the witness. Holding an owning object
allows an identity comparison in one observation window. Verifying that its
allocation survives across windows requires a separate lifetime witness.
"""

from dataclasses import dataclass, field
from importlib.machinery import ExtensionFileLoader, ModuleSpec
import sys
from types import ModuleType

from scar.trace.call_binding import _type_metadata
from scar.trace.numpy_ufunc_calls import CORE_MODULES


_IMMUTABLE_TYPE = 1 << 8


@dataclass(frozen=True, slots=True)
class NdarrayViewFrontier:
    reason: str
    shape: tuple[int, ...] | None = None
    byte_strides: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True, eq=False)
class NdarrayView:
    value: object = field(repr=False)
    owner: object = field(repr=False)
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    byte_offset: int
    allocation_nbytes: int
    itemsize: int
    dtype: str
    writeable: bool
    byte_bounds: tuple[int, int]

    def same_storage(self, other):
        """Compare retained owners within the same non-reallocation window."""
        return (type(other) is NdarrayView and self.owner is other.owner
                and self.allocation_nbytes == other.allocation_nbytes)

    def geometry(self):
        """Return only serializable geometry relative to the owning object."""
        return dict(shape=self.shape, byte_strides=self.byte_strides, byte_offset=self.byte_offset,
                    allocation_nbytes=self.allocation_nbytes, itemsize=self.itemsize, dtype=self.dtype,
                    writeable=self.writeable, byte_bounds=self.byte_bounds)


@dataclass(frozen=True, slots=True)
class NdarrayInplaceWitness:
    same_object: bool
    same_owner: bool | None
    same_geometry: bool | None
    frontier: str | None = None


def _provider(kind):
    """Identify the already loaded immutable type without reading app getters."""
    numpy = sys.modules.get('numpy')
    if (sys.implementation.name != 'cpython' or type(numpy) is not ModuleType
            or type(kind) is not type or vars(numpy).get('ndarray') is not kind
            or _type_metadata(kind) != {'module': 'numpy', 'qualname': 'ndarray'}
            or not type.__dict__['__flags__'].__get__(kind) & _IMMUTABLE_TYPE):
        return None
    for module_name in CORE_MODULES:
        module = sys.modules.get(module_name)
        if type(module) is not ModuleType:
            continue
        members = vars(module)
        spec = members.get('__spec__')
        dtype = members.get('dtype')
        if (members.get('ndarray') is kind and isinstance(dtype, type)
                and vars(numpy).get('dtype') is dtype
                and _type_metadata(dtype) == {'module': 'numpy', 'qualname': 'dtype'}
                and type(spec) is ModuleSpec
                and type(spec.loader) is ExtensionFileLoader and spec.name == module_name
                and type(spec.origin) is str and spec.origin == members.get('__file__')):
            return dtype
    return None


def _primitive_dtype(array, dtype_type):
    dtype = array.dtype
    if not any(base is dtype_type for base in type.__dict__['__mro__'].__get__(type(dtype))):
        return None
    # Read the descriptors on the extension base, including for a custom
    # dtype instance. Its Python overrides are outside this witness.
    fields = vars(dtype_type)
    if fields['kind'].__get__(dtype, dtype_type) not in 'biufc':
        return None
    width = fields['itemsize'].__get__(dtype, dtype_type)
    if width <= 0 or fields['hasobject'].__get__(dtype, dtype_type):
        return None
    return width, fields['str'].__get__(dtype, dtype_type)


def ndarray_view_contract(value, *, max_depth=64):
    """Retain exact strides plus the possibly hole-containing byte envelope."""
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError('expected positive ndarray ownership lookup limit')
    kind = type(value)
    dtype_type = _provider(kind)
    if dtype_type is None:
        return NdarrayViewFrontier('EXACT_LOADED_NDARRAY_REQUIRED')
    shape, strides = value.shape, value.strides

    def unresolved(reason):
        return NdarrayViewFrontier(reason, shape, strides)

    primitive = _primitive_dtype(value, dtype_type)
    if primitive is None:
        return unresolved('FIXED_WIDTH_PRIMITIVE_DTYPE_REQUIRED')
    itemsize, dtype = primitive
    owner = value
    for _ in range(max_depth):
        if owner.flags.writebackifcopy:
            return unresolved('WRITEBACK_IF_COPY_OWNER_UNRESOLVED')
        parent = owner.base
        if parent is None:
            break
        if type(parent) is not kind:
            return unresolved('EXTERNAL_OR_SYNTHETIC_BUFFER_OWNER')
        owner = parent
    else:
        return unresolved('OWNERSHIP_CHAIN_LIMIT')
    flags = owner.flags
    if not flags.owndata or not (flags.c_contiguous or flags.f_contiguous):
        return unresolved('CONTIGUOUS_OWNING_ALLOCATION_REQUIRED')
    if _primitive_dtype(owner, dtype_type) is None:
        return unresolved('PRIMITIVE_ALLOCATION_DTYPE_REQUIRED')

    # Pointers are needed transiently to find the relative displacement. They
    # are never returned or compared as persistent storage identities.
    offset = value.__array_interface__['data'][0] - owner.__array_interface__['data'][0]
    if value.size:
        displacements = tuple((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))
        start = offset + sum(min(0, delta) for delta in displacements)
        stop = offset + sum(max(0, delta) for delta in displacements) + itemsize
    else:
        start = stop = offset
    capacity = owner.nbytes
    if not 0 <= start <= stop <= capacity:
        return unresolved('ADDRESSED_BYTES_OUTSIDE_OWNER')
    return NdarrayView(value, owner, shape, strides, offset, capacity, itemsize, dtype,
                       bool(value.flags.writeable), (start, stop))


def inplace_ndarray_witness(before, result):
    """Compare a normal result to a previously observed operand.

    These facts describe the observed object and geometry. They establish
    neither a selected operator implementation nor the bytes it wrote.
    """
    if type(before) is not NdarrayView:
        raise TypeError('in-place witness requires an admitted operand view')
    same_object = before.value is result
    after = ndarray_view_contract(result)
    if type(after) is NdarrayViewFrontier:
        return NdarrayInplaceWitness(same_object, None, None, after.reason)
    return NdarrayInplaceWitness(same_object, before.owner is after.owner,
                                 before.geometry() == after.geometry())
