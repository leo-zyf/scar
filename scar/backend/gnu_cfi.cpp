// GNU CFI v1 numeric, nonowning descriptors on the LP64 ABI.
#include <ISO_Fortran_binding.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>

namespace {
using Index = CFI_index_t;

static_assert(CFI_VERSION == 1 && CFI_MAX_RANK == 15);
static_assert(sizeof(Index) == 8 && sizeof(void *) == 8);
static_assert(sizeof(float) == 4 && sizeof(double) == 8 && sizeof(int) == 4);
static_assert(offsetof(CFI_cdesc_t, dim) == 24 && sizeof(CFI_dim_t) == 24);

bool valid_rank(CFI_rank_t rank) { return rank >= 0 && rank <= CFI_MAX_RANK; }

bool view_attribute(CFI_attribute_t attribute) {
  return attribute == CFI_attribute_other || attribute == CFI_attribute_pointer;
}

Index element_size(CFI_type_t type) {
  if (type == CFI_type_int)
    return sizeof(int);
  if (type == CFI_type_float)
    return sizeof(float);
  if (type == CFI_type_double)
    return sizeof(double);
  return 0;
}
} // namespace

extern "C" int CFI_establish(CFI_cdesc_t *descriptor, void *base,
                             CFI_attribute_t attribute, CFI_type_t type, size_t,
                             CFI_rank_t rank, const Index extents[]) {
  if (!descriptor)
    return CFI_INVALID_DESCRIPTOR;
  if (!valid_rank(rank))
    return CFI_INVALID_RANK;
  if (!view_attribute(attribute))
    return CFI_INVALID_ATTRIBUTE;
  const Index size = element_size(type);
  if (!size)
    return CFI_INVALID_TYPE;

  std::array<CFI_dim_t, CFI_MAX_RANK> dimensions;
  if (base && rank > 0) {
    if (!extents)
      return CFI_INVALID_EXTENT;
    Index stride = size;
    for (int axis = 0; axis < rank; ++axis) {
      dimensions[axis] = {0, extents[axis], stride};
      if (extents[axis] < 0 ||
          __builtin_mul_overflow(stride, extents[axis], &stride)) {
        return CFI_INVALID_EXTENT;
      }
    }
  }

  descriptor->base_addr = base;
  descriptor->elem_len = static_cast<size_t>(size);
  descriptor->version = CFI_VERSION;
  descriptor->rank = rank;
  descriptor->attribute = attribute;
  descriptor->type = type;
  if (base) {
    for (int axis = 0; axis < rank; ++axis)
      descriptor->dim[axis] = dimensions[axis];
  }
  return CFI_SUCCESS;
}

extern "C" int CFI_section(CFI_cdesc_t *result, const CFI_cdesc_t *source,
                           const Index lower[], const Index upper[],
                           const Index strides[]) {
  if (!result || !source || result->version != CFI_VERSION ||
      source->version != CFI_VERSION) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (!source->base_addr)
    return CFI_ERROR_BASE_ADDR_NULL;
  if (!view_attribute(result->attribute) ||
      !view_attribute(source->attribute)) {
    return CFI_INVALID_ATTRIBUTE;
  }
  if (!valid_rank(source->rank) || source->rank == 0 ||
      !valid_rank(result->rank))
    return CFI_INVALID_RANK;
  const Index size = element_size(source->type);
  if (!size || result->type != source->type)
    return CFI_INVALID_TYPE;
  if (result->elem_len != static_cast<size_t>(size) ||
      source->elem_len != result->elem_len) {
    return CFI_INVALID_ELEM_LEN;
  }
  if (!strides)
    return CFI_INVALID_STRIDE;
  const auto retained_rank = std::count_if(
      strides, strides + source->rank, [](Index step) { return step != 0; });
  if (result->rank != retained_rank)
    return CFI_INVALID_RANK;

  // Compute against the original descriptor before publishing the result.
  // A zero stride selects one element and removes its axis from the result.
  std::array<CFI_dim_t, CFI_MAX_RANK> dimensions;
  Index offset = 0;
  int retained = 0;
  for (int axis = 0; axis < source->rank; ++axis) {
    const auto &dim = source->dim[axis];
    Index limit;
    if (dim.extent < 0 ||
        __builtin_add_overflow(dim.lower_bound, dim.extent - 1, &limit)) {
      return CFI_INVALID_EXTENT;
    }
    const Index first = lower ? lower[axis] : dim.lower_bound;
    const Index last = upper ? upper[axis] : limit;
    const Index step = strides[axis];
    if (dim.sm % size)
      return CFI_INVALID_STRIDE;
    if (dim.extent == 0) {
      if (first != dim.lower_bound || last != limit || step != 1)
        return CFI_ERROR_OUT_OF_BOUNDS;
    } else {
      if (first < dim.lower_bound || first > limit || last < dim.lower_bound ||
          last > limit) {
        return CFI_ERROR_OUT_OF_BOUNDS;
      }
      if (!step && first != last)
        return CFI_ERROR_OUT_OF_BOUNDS;
      if ((first < last && step < 0) || (first > last && step > 0))
        return CFI_INVALID_STRIDE;
    }
    Index start_offset;
    if (__builtin_mul_overflow(first - dim.lower_bound, dim.sm,
                               &start_offset) ||
        __builtin_add_overflow(offset, start_offset, &offset)) {
      return CFI_ERROR_OUT_OF_BOUNDS;
    }
    if (!step)
      continue;
    Index stride;
    if (__builtin_mul_overflow(step, dim.sm, &stride))
      return CFI_INVALID_STRIDE;
    dimensions[retained++] = {0, 1 + (last - first) / step, stride};
  }

  result->base_addr = static_cast<std::byte *>(source->base_addr) + offset;
  for (int axis = 0; axis < retained; ++axis)
    result->dim[axis] = dimensions[axis];
  return CFI_SUCCESS;
}

extern "C" int CFI_setpointer(CFI_cdesc_t *result, CFI_cdesc_t *source,
                              const Index lower[]) {
  if (!result || result->version != CFI_VERSION)
    return CFI_INVALID_DESCRIPTOR;
  if (result->attribute != CFI_attribute_pointer)
    return CFI_INVALID_ATTRIBUTE;
  if (!valid_rank(result->rank))
    return CFI_INVALID_RANK;
  const Index size = element_size(result->type);
  if (!size)
    return CFI_INVALID_TYPE;
  if (result->elem_len != static_cast<size_t>(size))
    return CFI_INVALID_ELEM_LEN;
  if (!source) {
    result->base_addr = nullptr;
    return CFI_SUCCESS;
  }
  if (source->version != CFI_VERSION)
    return CFI_INVALID_DESCRIPTOR;
  if (!view_attribute(source->attribute))
    return CFI_INVALID_ATTRIBUTE;
  if (!source->base_addr && source->attribute != CFI_attribute_pointer)
    return CFI_ERROR_BASE_ADDR_NULL;
  if (source->rank != result->rank)
    return CFI_INVALID_RANK;
  if (source->type != result->type)
    return CFI_INVALID_TYPE;
  if (source->elem_len != result->elem_len)
    return CFI_INVALID_ELEM_LEN;

  // A disassociated source may have never initialized its dimensions.
  // Transfer their representation without reading indeterminate field values.
  std::array<CFI_dim_t, CFI_MAX_RANK> dimensions;
  const auto bytes = static_cast<size_t>(source->rank) * sizeof(CFI_dim_t);
  if (bytes)
    std::memcpy(dimensions.data(), source->dim, bytes);
  for (int axis = 0; axis < source->rank; ++axis) {
    auto &dim = dimensions[axis];
    if (lower)
      dim.lower_bound = lower[axis];
    if (source->base_addr) {
      Index last;
      if (dim.extent < 0)
        return CFI_INVALID_EXTENT;
      if (__builtin_add_overflow(dim.lower_bound, dim.extent - 1, &last))
        return CFI_ERROR_OUT_OF_BOUNDS;
      if (dim.sm % size)
        return CFI_INVALID_STRIDE;
    }
  }
  result->base_addr = source->base_addr;
  if (bytes)
    std::memcpy(result->dim, dimensions.data(), bytes);
  return CFI_SUCCESS;
}

extern "C" int CFI_is_contiguous(const CFI_cdesc_t *descriptor) {
  if (!descriptor || descriptor->version != CFI_VERSION ||
      !descriptor->base_addr || !valid_rank(descriptor->rank) ||
      descriptor->rank == 0 || !view_attribute(descriptor->attribute)) {
    return 0;
  }
  Index stride = element_size(descriptor->type);
  if (!stride || descriptor->elem_len != static_cast<size_t>(stride))
    return 0;
  for (int axis = 0; axis < descriptor->rank; ++axis) {
    const auto &dim = descriptor->dim[axis];
    if (dim.extent < 0 || dim.sm != stride ||
        __builtin_mul_overflow(stride, dim.extent, &stride)) {
      return 0;
    }
  }
  return 1;
}
