#pragma once

#include "exceptions.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/core/xstrides.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xbroadcast.hpp>
#include <xtensor/views/xstrided_view.hpp>
#include <xtl/xcomplex.hpp>

namespace scar {
struct array_shape {
  std::size_t rank;
  std::initializer_list<std::int64_t> extents;
};

template <class T> struct array_calculation;

template <class T> class ndarray {
  friend struct array_calculation<T>;
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
        throw std::invalid_argument("negative dimensions are not allowed");
      result.push_back(static_cast<std::size_t>(extent));
    }
    return result;
  }

  [[noreturn]] static void reshape_mismatch(array_shape layout,
                                            std::size_t source_size) {
    std::string axes;
    for (auto extent : layout.extents) {
      if (axes.empty() && extent < 0)
        continue;
      if (!axes.empty())
        axes += ',';
      axes += extent < 0 ? "newaxis" : std::to_string(extent);
    }
    if (layout.rank == 1 && !axes.empty())
      axes += ',';
    throw std::invalid_argument("cannot reshape array of size " +
                                std::to_string(source_size) + " into shape (" +
                                axes + ")");
  }

  static shape_type inferred_shape(array_shape layout, std::size_t size) {
    if (layout.extents.size() != layout.rank)
      throw std::invalid_argument("array shape rank mismatch");
    if (layout.rank > 64)
      throw std::invalid_argument(
          "maximum supported dimension for an ndarray is currently 64, found " +
          std::to_string(layout.rank));
    constexpr auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    shape_type dims;
    dims.reserve(layout.rank);
    std::size_t product = 1;
    std::optional<std::size_t> unknown;
    for (auto extent : layout.extents) {
      if (extent < 0) {
        if (unknown)
          throw std::invalid_argument("can only specify one unknown dimension");
        unknown = dims.size();
        dims.push_back(0);
      } else {
        if (!std::in_range<std::size_t>(extent))
          reshape_mismatch(layout, size);
        const auto axis = static_cast<std::size_t>(extent);
        if (axis && product > limit / axis)
          reshape_mismatch(layout, size);
        product *= axis;
        dims.push_back(axis);
      }
    }
    if (unknown) {
      if (!product || size % product)
        reshape_mismatch(layout, size);
      dims[*unknown] = size / product;
    } else if (size != product)
      reshape_mismatch(layout, size);

    if (!size)
      (void)count(dims);
    return dims;
  }

  static std::size_t count(const shape_type &shape) {
    constexpr auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(T);
    std::size_t size = 1;
    bool empty = false;
    for (auto extent : shape) {
      if (extent == 0) {
        empty = true;
        continue;
      }
      // Source ndarray allocation checks strides even when a different axis is
      // empty.
      if (size > limit / extent)
        throw std::invalid_argument(
            "array is too big; `arr.size * arr.dtype.itemsize` is "
            "larger than the maximum possible size.");
      size *= extent;
    }
    return empty ? 0 : size;
  }

  static strides_type row_strides(const shape_type &shape, std::size_t size) {
    strides_type strides(shape.size());
    if (size)
      xt::compute_strides(shape, xt::layout_type::row_major, strides);
    return strides;
  }

  static ndarray allocate(shape_type shape, std::size_t size) {
    auto data =
        std::make_shared<storage_type>(storage_type::from_shape({size}));
    auto strides = row_strides(shape, size);
    return ndarray(
        {std::move(data), 0, size, std::move(shape), std::move(strides)});
  }

  static ndarray iota_values(std::int64_t length) {
    static_assert(std::is_same_v<T, double> &&
                  std::numeric_limits<T>::is_iec559 &&
                  std::numeric_limits<T>::digits == 53);
    auto dims = shape({1, {length}});
    auto size = count(dims);
    auto result = allocate(std::move(dims), size);
    if (size)
      xt::noalias(*result.region_->storage) = xt::cast<T>(xt::arange(length));
    return result;
  }

  static std::string iota_allocation_message(std::size_t length) {
    // NumPy's f64 allocation diagnostic uses binary units and three significant
    // figures, rounding up to the next unit at 1023.5.
    constexpr const char *units[] = {"bytes", "KiB", "MiB", "GiB",
                                     "TiB",   "PiB", "EiB"};
    const auto bytes = length * sizeof(double);
    auto unit = bytes ? (std::bit_width(bytes) - 1) / 10 : 0;
    auto amount = static_cast<double>(bytes) /
                  static_cast<double>(std::size_t{1} << (10 * unit));
    if (amount >= 1023.5) {
      ++unit;
      amount /= 1024;
    }
    std::ostringstream message;
    message.imbue(std::locale::classic());
    message << "Unable to allocate ";
    if (!unit)
      message << bytes;
    else if (amount < 999.5)
      message << std::showpoint << std::setprecision(3) << amount;
    else
      message << std::showpoint << std::fixed << std::setprecision(0) << amount;
    message << ' ' << units[unit] << " for an array with shape (" << length
            << ",) and data type float64";
    return message.str();
  }

  void require_rank(std::optional<std::size_t> rank) const {
    if (!region_ || (rank && *rank != region_->shape.size()))
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
      const auto position = index < 0 ? index + extent : index;
      if (position < 0 || position >= extent)
        throw std::out_of_range(
            "index " + std::to_string(index) + " is out of bounds for axis " +
            std::to_string(axis) + " with size " + std::to_string(extent));
      offset += static_cast<std::size_t>(position) * region_->strides[axis];
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

  const shape_type &dimensions() const {
    if (!region_)
      throw std::invalid_argument("unbound array handle");
    return region_->shape;
  }

  std::int64_t size(std::optional<std::size_t> rank = std::nullopt) const {
    require_rank(rank);
    if (!std::in_range<std::int64_t>(region_->size))
      throw std::length_error("array size exceeds i64");
    return static_cast<std::int64_t>(region_->size);
  }

  ndarray reshape(std::optional<std::size_t> rank, array_shape layout,
                  bool infer_negative = false) const {
    require_rank(rank);
    const auto &src = *region_;
    auto dims =
        infer_negative ? inferred_shape(layout, src.size) : shape(layout);
    if (!infer_negative && count(dims) != src.size)
      throw std::invalid_argument("array reshape changes element count");
    if (src.size && !xt::do_strides_match(src.shape, src.strides,
                                          xt::layout_type::row_major, true))
      throw std::invalid_argument(
          "array reshape requires a contiguous C-order source");
    auto strides = row_strides(dims, src.size);
    return ndarray({src.storage, src.offset, src.size, std::move(dims),
                    std::move(strides)});
  }

  ndarray broadcast_copy(std::optional<std::size_t> rank,
                         array_shape layout) const {
    require_rank(rank);
    auto dims = shape(layout);
    if (!xt::broadcastable(region_->shape, std::as_const(dims)))
      throw std::invalid_argument(
          "source shape cannot broadcast to requested shape");
    auto size = count(dims);
    auto result = allocate(std::move(dims), size);
    if (size) {
      auto dst = result.expression();
      xt::noalias(dst) = xt::broadcast(expression(), result.region_->shape);
    }
    return result;
  }

  T value_at_flat(std::size_t index) const {
    if (!region_ || index >= region_->size)
      throw std::out_of_range("flat index outside array view");
    return region_->storage->flat(region_->offset + index);
  }

  // A native vector operator reads the selected view and writes a distinct
  // allocation. The two owning handles remain live throughout the call.
  template <class Operator>
  ndarray transform_vector(Operator &&operation) const {
    require_rank(1);
    const auto &src = *region_;
    auto result = allocate({src.size}, src.size);
    auto stride = src.strides.front();
    constexpr auto bytes = static_cast<std::ptrdiff_t>(sizeof(T));
    if (stride > std::numeric_limits<std::ptrdiff_t>::max() / bytes ||
        stride < std::numeric_limits<std::ptrdiff_t>::min() / bytes)
      throw std::length_error("array byte stride overflow");
    const T *input = src.storage->data();
    if (src.offset)
      input += src.offset;
    std::forward<Operator>(operation)(
        input, stride * bytes,
        std::span<T>(result.region_->storage->data(), src.size));
    return result;
  }

  static ndarray zeros(array_shape layout) {
    auto dims = shape(layout);
    auto size = count(dims);
    auto result = allocate(std::move(dims), size);
    if (size)
      result.region_->storage->fill(T{});
    return result;
  }

  static ndarray iota(std::int64_t length) {
    if (length > (std::int64_t{1} << 53))
      throw std::length_error("array_iota exceeds the binary64 index limit");
    return iota_values(length);
  }

  static ndarray iota_rounded(double stop, bool endpoint_overflows_to_empty) {
    static_assert(std::is_same_v<T, double> &&
                  std::numeric_limits<std::ptrdiff_t>::digits == 63);
    constexpr double endpoint = 0x1p63;
    if (std::isnan(stop))
      throw std::invalid_argument("arange: cannot compute length");
    if (stop > endpoint || stop < -endpoint ||
        (stop == endpoint && !endpoint_overflows_to_empty))
      throw std::invalid_argument("Maximum allowed size exceeded");
    // Check the rounded endpoint before narrowing it to a signed index.
    const auto length = stop > 0 && stop < endpoint
                            ? static_cast<std::int64_t>(std::ceil(stop))
                            : std::int64_t{0};
    try {
      return iota_values(length);
    } catch (const std::bad_alloc &) {
      throw program_error("memory_error", iota_allocation_message(length));
    }
  }

  static ndarray literal(array_shape layout, std::initializer_list<T> values) {
    return from_row_major(shape(layout), values);
  }

  template <class U>
  static ndarray from_sequence(const std::shared_ptr<std::vector<U>> &source) {
    const auto size = source->size();
    auto result = allocate({size}, size);
    if (size)
      xt::noalias(*result.region_->storage) = xt::cast<T>(xt::adapt(*source));
    return result;
  }

  static ndarray from_row_major(std::vector<std::size_t> dims,
                                std::span<const T> values) {
    auto size = count(dims);
    if (size != values.size())
      throw std::invalid_argument("array literal element count mismatch");
    auto result = allocate(std::move(dims), size);
    if (size)
      xt::noalias(*result.region_->storage) =
          xt::adapt(values.data(), size, xt::no_ownership(), shape_type{size});
    return result;
  }

  ndarray materialize(std::optional<std::size_t> rank, bool negate) const {
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

template <class T> ndarray<T> array_iota(std::int64_t count) {
  return ndarray<T>::iota(count);
}

template <class T>
ndarray<T> array_iota_rounded(double stop, bool endpoint_overflows_to_empty) {
  return ndarray<T>::iota_rounded(stop, endpoint_overflows_to_empty);
}

template <class T>
std::int64_t array_size(const ndarray<T> &value, std::size_t rank) {
  return value.size(rank);
}

template <class T> std::int64_t array_size(const ndarray<T> &value) {
  return value.size();
}
template <class T> std::int64_t array_rank(const ndarray<T> &value) {
  return static_cast<std::int64_t>(value.dimensions().size());
}
template <class T>
std::int64_t array_extent(const ndarray<T> &value, std::int64_t axis) {
  const auto rank = array_rank(value);
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    throw std::out_of_range("tuple index out of range");
  const auto extent = value.dimensions()[static_cast<std::size_t>(axis)];
  if (!std::in_range<std::int64_t>(extent))
    throw std::overflow_error("array extent exceeds i64");
  return static_cast<std::int64_t>(extent);
}

template <class T> struct array_shape_change {
  ndarray<T> value;
  std::optional<std::size_t> source_rank;
  array_shape shape;
  bool infer_negative = false;
};
template <class T> ndarray<T> array_reshape(array_shape_change<T> input) {
  return input.value.reshape(input.source_rank, input.shape,
                             input.infer_negative);
}
template <class T>
ndarray<T> array_broadcast_copy(array_shape_change<T> input) {
  if (input.infer_negative)
    throw std::invalid_argument("broadcast shape does not infer dimensions");
  return input.value.broadcast_copy(input.source_rank, input.shape);
}

template <class T> struct array_elements {
  array_shape shape;
  std::initializer_list<T> values;
};
template <class T> ndarray<T> array_literal(array_elements<T> input) {
  return ndarray<T>::literal(input.shape, input.values);
}

template <class T, class U>
ndarray<T> array_from_sequence(const std::shared_ptr<std::vector<U>> &source) {
  return ndarray<T>::from_sequence(source);
}

template <class T> struct array_operand {
  ndarray<T> value;
  std::optional<std::size_t> rank;
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

enum class array_math {
  add,
  subtract,
  multiply,
  divide,
  power,
  square,
  sqrt,
  reciprocal,
  positive,
  ones_like
};

struct array_warning {
  std::string_view module, file;
  std::size_t line;
  std::string_view operation, source_line;
};

// Each ufunc observes only its own IEEE flags. Restore both caller flags and
// exception masks even if evaluation throws before producing a result.
class array_fenv {
  std::fenv_t saved_;
  bool active_ = true;

public:
  array_fenv() {
    if (std::feholdexcept(&saved_))
      throw std::runtime_error("cannot isolate floating-point environment");
  }
  array_fenv(const array_fenv &) = delete;
  ~array_fenv() {
    if (active_)
      std::fesetenv(&saved_);
  }
  int finish() {
    int raised = std::fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
    if (std::fesetenv(&saved_))
      throw std::runtime_error("cannot restore floating-point environment");
    active_ = false;
    return raised;
  }
};

inline void report_array_warnings(int flags, const array_warning &site) {
  if (!flags)
    return;
  static std::mutex lock;
  static std::set<
      std::tuple<std::string, std::string, std::size_t, std::string, int>>
      seen;
  const std::lock_guard guard(lock);
  constexpr std::pair<int, std::string_view> classes[] = {
      {FE_DIVBYZERO, "divide by zero"},
      {FE_OVERFLOW, "overflow"},
      {FE_INVALID, "invalid value"}};
  for (auto [bit, reason] : classes) {
    if (!(flags & bit) ||
        !seen.emplace(site.module, site.file, site.line, site.operation, bit)
             .second)
      continue;
    std::string line(site.file);
    line += ":" + std::to_string(site.line) + ": RuntimeWarning: ";
    line += reason;
    line += " encountered in ";
    line += site.operation;
    line += '\n';
    if (!site.source_line.empty()) {
      line += "  ";
      line += site.source_line;
      line += '\n';
    }
    if (std::fwrite(line.data(), 1, line.size(), stderr) != line.size() ||
        std::fflush(stderr))
      throw std::ios_base::failure("cannot write arithmetic diagnostic");
  }
}

template <class T> struct array_number {
  std::variant<T, ndarray<T>> value;
  std::size_t rank;
};

template <class T> struct array_calculation {
  using number = array_number<T>;
  number left, right;
  array_math operation;
  array_warning warning;

private:
  using array = ndarray<T>;
  using shape_type = typename array::shape_type;

  static const shape_type &shape(const number &n) {
    if (const auto *a = std::get_if<array>(&n.value)) {
      a->require_rank(n.rank);
      return a->dimensions();
    }
    if (n.rank)
      throw std::invalid_argument("scalar operand has nonzero rank");
    static const shape_type scalar_shape;
    return scalar_shape;
  }
  static const T *data(const number &n) {
    if (const auto *scalar = std::get_if<T>(&n.value))
      return scalar;
    const auto &r = *std::get<array>(n.value).region_;
    return r.storage->data() + r.offset;
  }
  static auto source(const number &n) { return xt::adapt(data(n), shape(n)); }
  shape_type result_shape() const {
    const auto &l = shape(left), &r = shape(right);
    shape_type result(std::max(l.size(), r.size()), 1);
    try {
      xt::broadcast_shape(l, result);
      xt::broadcast_shape(r, result);
    } catch (const xt::broadcast_error &ex) {
      throw std::invalid_argument(ex.what());
    }
    return result;
  }
  template <class Dest, class Operand>
  void real_operation(Dest &dst, const Operand &l, const Operand &r,
                      const number &rhs) const {
    auto verify_exponent = [&](double value) {
      if (rhs.rank || *data(rhs) != value)
        throw std::invalid_argument(
            "array power specialization exponent mismatch");
    };
    switch (operation) {
    case array_math::divide:
      xt::noalias(dst) = l / r;
      break;
    case array_math::power:
      xt::noalias(dst) = xt::pow(l, r);
      break;
    case array_math::square:
      verify_exponent(2.);
      xt::noalias(dst) = xt::square(l);
      break;
    case array_math::sqrt:
      verify_exponent(0.5);
      xt::noalias(dst) = xt::sqrt(l);
      break;
    case array_math::reciprocal:
      verify_exponent(-1.);
      xt::noalias(dst) = 1. / l;
      break;
    case array_math::positive:
      verify_exponent(1.);
      xt::noalias(dst) = l;
      break;
    case array_math::ones_like:
      verify_exponent(0.);
      dst.fill(1.);
      break;
    default:
      throw std::invalid_argument("unsupported real array operation");
    }
  }

  template <class Dest> void evaluate(Dest &dst, const number &rhs) const {
    array_fenv environment;
    auto l = source(left), r = source(rhs);
    auto dest = xt::noalias(dst);
    switch (operation) {
    case array_math::add:
      dest = l + r;
      break;
    case array_math::subtract:
      dest = l - r;
      break;
    case array_math::multiply:
      if constexpr (std::is_same_v<T, std::complex<double>>) {
        // This profile multiplies the two binary64 components directly.
        // std::complex may repair NaN components by recomputing with infinity.
        using components = xtl::xcomplex<double, double, false>;
        dest = xt::cast<T>(xt::cast<components>(l) * xt::cast<components>(r));
      } else {
        dest = l * r;
      }
      break;
    default:
      if constexpr (std::is_same_v<T, double>)
        real_operation(dst, l, r, rhs);
      else
        throw std::invalid_argument("unsupported complex array operation");
    }
    report_array_warnings(environment.finish(), warning);
  }

public:
  template <class Result> Result fresh() const {
    auto dims = result_shape();
    if constexpr (std::is_same_v<Result, T>) {
      if (!dims.empty())
        throw std::invalid_argument("arithmetic scalar result rank mismatch");
      T value;
      auto dst = xt::adapt(&value, dims);
      evaluate(dst, right);
      return value;
    } else {
      static_assert(std::is_same_v<Result, array>);
      if (dims.empty())
        throw std::invalid_argument("arithmetic array result rank mismatch");
      auto size = array::count(dims);
      auto result = array::allocate(std::move(dims), size);
      if (size) {
        auto dst = result.expression();
        evaluate(dst, right);
      }
      return result;
    }
  }
  std::nullptr_t update() const {
    auto dims = result_shape();
    auto *a = std::get_if<array>(&left.value);
    if (!a || dims != a->dimensions())
      throw std::invalid_argument(
          "broadcast would change the in-place array shape");
    if (!a->region_->size)
      return nullptr;
    auto rhs = right;
    if (auto *b = std::get_if<array>(&rhs.value)) {
      const auto &dst = *a->region_, &src = *b->region_;
      if (dst.storage == src.storage && dst.offset < src.offset + src.size &&
          src.offset < dst.offset + dst.size)
        rhs.value = b->materialize(rhs.rank, false);
    }
    auto dst = a->expression();
    evaluate(dst, rhs);
    return nullptr;
  }
};
template <class Result, class T = double>
Result array_binary(const array_calculation<T> &request) {
  return request.template fresh<Result>();
}
template <class T = double>
std::nullptr_t array_update(const array_calculation<T> &request) {
  return request.update();
}
} // namespace scar
