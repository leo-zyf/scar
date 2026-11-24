#pragma once

#include "exceptions.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace scar {
template <class T> using sequence_ref = std::shared_ptr<std::vector<T>>;

template <class T>
sequence_ref<T> sequence_make(std::initializer_list<T> items) {
  return std::make_shared<std::vector<T>>(items);
}

template <class T> struct sequence_item {
  sequence_ref<T> base;
  T value;
};

template <class T> std::nullptr_t sequence_append(sequence_item<T> item) {
  item.base->push_back(std::move(item.value));
  return nullptr;
}

template <class T> std::int64_t sequence_size(const sequence_ref<T> &base) {
  static_assert(
      std::numeric_limits<typename std::vector<T>::difference_type>::digits <=
      std::numeric_limits<std::int64_t>::digits);
  return static_cast<std::int64_t>(base->size());
}

template <class T> struct sequence_index {
  sequence_ref<T> base;
  std::int64_t index;
};

template <class T> T sequence_get(sequence_index<T> request) {
  const auto size = sequence_size(request.base);
  auto index = request.index;
  if (index < 0)
    index += size;
  if (index < 0 || index >= size)
    throw program_error("index_error", "list index out of range");
  return (*request.base)[static_cast<std::size_t>(index)];
}

struct sequence_pair {
  std::shared_ptr<const void> left, right;
};
inline bool sequence_identity(sequence_pair pair) {
  return pair.left.get() == pair.right.get();
}

template <class T, class Compare>
sequence_ref<T> sequence_sorted_copy(const sequence_ref<T> &base,
                                     Compare less) {
  auto result = std::make_shared<std::vector<T>>(*base);
  std::stable_sort(result->begin(), result->end(), less);
  return result;
}
} // namespace scar
