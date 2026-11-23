#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/views/xbroadcast.hpp>
#include <xtensor/views/xstrided_view.hpp>

namespace scar {
struct array_shape {
  std::size_t rank;
  std::initializer_list<std::int64_t> extents;
};

template <class T> class ndarray {
  using shape_type = std::vector<std::size_t>;
  using strides_type = std::vector<std::ptrdiff_t>;
  using storage_type = xt::xarray<T, xt::layout_type::row_major>;
  struct region {
    std::shared_ptr<storage_type> storage;
    std::size_t offset, size;
    shape_type shape;
    strides_type strides;
  };
  std::shared_ptr<const region> region_;

  explicit ndarray(region r)
      : region_(std::make_shared<region>(std::move(r))) {}

  static shape_type shape(array_shape layout) {
    if (layout.extents.size() != layout.rank)
      throw std::invalid_argument("array shape rank mismatch");
    shape_type result;
    result.reserve(layout.rank);
    for (auto extent : layout.extents) {
      if (!std::in_range<std::size_t>(extent))
        throw std::length_error("invalid array extent");
      result.push_back(static_cast<std::size_t>(extent));
    }
    return result;
  }

  static std::size_t count(const shape_type &shape) {
    if (std::find(shape.begin(), shape.end(), 0) != shape.end())
      return 0;
    constexpr auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(T);
    std::size_t size = 1;
    for (auto extent : shape) {
      if (size > limit / extent)
        throw std::length_error("array byte size overflow");
      size *= extent;
    }
    return size;
  }

  static ndarray allocate(shape_type shape, std::size_t size) {
    auto data =
        std::make_shared<storage_type>(storage_type::from_shape({size}));
    strides_type strides(shape.size());
    if (size)
      xt::compute_strides(shape, xt::layout_type::row_major, strides);
    return ndarray(
        {std::move(data), 0, size, std::move(shape), std::move(strides)});
  }

  void require_rank(std::size_t rank) const {
    if (!region_ || rank != region_->shape.size())
      throw std::invalid_argument("array handle rank mismatch");
  }

  auto expression() const {
    const auto &r = *region_;
    return xt::strided_view<xt::layout_type::row_major>(
        *r.storage, shape_type(r.shape), strides_type(r.strides), r.offset,
        xt::layout_type::row_major);
  }

  std::pair<std::size_t, std::size_t>
  locate(std::size_t rank, std::initializer_list<std::int64_t> indices) const {
    require_rank(rank);
    if (indices.size() > rank)
      throw std::out_of_range("too many array indices");
    auto offset = region_->offset, size = region_->size;
    std::size_t axis = 0;
    for (auto index : indices) {
      const auto extent = static_cast<std::int64_t>(region_->shape[axis]);
      if (index < 0)
        index += extent;
      if (index < 0 || index >= extent)
        throw std::out_of_range("array index outside axis extent");
      offset += static_cast<std::size_t>(index) * region_->strides[axis];
      size /= static_cast<std::size_t>(extent);
      ++axis;
    }
    return {offset, size};
  }

  ndarray suffix(std::size_t axes, std::size_t offset, std::size_t size) const {
    if (!axes)
      return *this;
    const auto &r = *region_;
    return ndarray({r.storage,
                    offset,
                    size,
                    {r.shape.begin() + axes, r.shape.end()},
                    {r.strides.begin() + axes, r.strides.end()}});
  }

public:
  ndarray() = default;

  static ndarray zeros(array_shape layout) {
    auto dims = shape(layout);
    auto size = count(dims);
    auto result = allocate(std::move(dims), size);
    if (size)
      result.region_->storage->fill(T{});
    return result;
  }

  static ndarray literal(array_shape layout, std::initializer_list<T> values) {
    auto dims = shape(layout);
    auto size = count(dims);
    if (size != values.size())
      throw std::invalid_argument("array literal element count mismatch");
    auto result = allocate(std::move(dims), size);
    if (size)
      xt::noalias(*result.region_->storage) =
          xt::adapt(values.begin(), size, xt::no_ownership(), shape_type{size});
    return result;
  }

  ndarray materialize(std::size_t rank, bool negate) const {
    require_rank(rank);
    auto result = allocate(region_->shape, region_->size);
    if (region_->size) {
      auto src = expression(), dst = result.expression();
      if (negate)
        xt::noalias(dst) = -src;
      else
        xt::noalias(dst) = src;
    }
    return result;
  }

  ndarray view(std::size_t rank,
               std::initializer_list<std::int64_t> indices) const {
    auto [offset, size] = locate(rank, indices);
    return suffix(indices.size(), offset, size);
  }

  T load(std::size_t rank, std::initializer_list<std::int64_t> indices) const {
    if (indices.size() != rank)
      throw std::invalid_argument("array scalar load requires every axis");
    return region_element(rank, indices);
  }

  void store(std::size_t rank, std::initializer_list<std::int64_t> indices,
             T value) const {
    if (indices.size() == rank) {
      region_element(rank, indices) = value;
    } else {
      auto dst = view(rank, indices);
      if (dst.region_->size)
        dst.expression().fill(value);
    }
  }

  void store(std::size_t rank, std::initializer_list<std::int64_t> indices,
             ndarray src, std::size_t src_rank) const {
    src.require_rank(src_rank);
    auto dst = view(rank, indices);
    const auto &d = *dst.region_;
    if (src_rank > d.shape.size()) {
      const auto extra = src_rank - d.shape.size();
      const auto &s = *src.region_;
      if (!std::all_of(s.shape.begin(), s.shape.begin() + extra,
                       [](auto n) { return n == 1; }))
        throw std::invalid_argument(
            "source rank cannot broadcast to array target");
      src = src.suffix(extra, s.offset, s.size);
    }
    const auto &s = *src.region_;
    if (!xt::broadcastable(s.shape, d.shape))
      throw std::invalid_argument(
          "source shape cannot broadcast to array target");
    if (!d.size)
      return;
    auto target = dst.expression(), source = src.expression();
    // Prefix-indexed row-major regions occupy contiguous intervals.
    if (d.storage == s.storage && d.offset < s.offset + s.size &&
        s.offset < d.offset + d.size) {
      storage_type snapshot(source);
      xt::noalias(target) = xt::broadcast(snapshot, d.shape);
    } else {
      xt::noalias(target) = xt::broadcast(source, d.shape);
    }
  }

private:
  T &region_element(std::size_t rank,
                    std::initializer_list<std::int64_t> indices) const {
    const auto position = locate(rank, indices);
    return region_->storage->flat(position.first);
  }
};

template <class T> ndarray<T> array_create(array_shape shape) {
  return ndarray<T>::zeros(shape);
}

template <class T> struct array_elements {
  array_shape shape;
  std::initializer_list<T> values;
};
template <class T> ndarray<T> array_literal(array_elements<T> input) {
  return ndarray<T>::literal(input.shape, input.values);
}

template <class T> struct array_operand {
  ndarray<T> value;
  std::size_t rank;
  bool negate = false;
};
template <class T> ndarray<T> array_copy(array_operand<T> input) {
  return input.value.materialize(input.rank, false);
}
template <class T> ndarray<T> array_unary(array_operand<T> input) {
  return input.value.materialize(input.rank, input.negate);
}

template <class T> struct array_location {
  ndarray<T> base;
  std::size_t rank;
  std::initializer_list<std::int64_t> indices;
};
template <class T> ndarray<T> array_view(array_location<T> loc) {
  return loc.base.view(loc.rank, loc.indices);
}
template <class T> T array_load(array_location<T> loc) {
  return loc.base.load(loc.rank, loc.indices);
}

template <class T, class V> struct array_write {
  V value;
  std::size_t source_rank;
  array_location<T> target;
};
template <class T, class V> void array_store(array_write<T, V> write) {
  const auto &loc = write.target;
  if constexpr (std::is_same_v<V, T>)
    loc.base.store(loc.rank, loc.indices, write.value);
  else
    loc.base.store(loc.rank, loc.indices, std::move(write.value),
                   write.source_rank);
}
} // namespace scar
