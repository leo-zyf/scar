#pragma once

#include "text.hpp"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace scar {
struct mapping_state;
using mapping_ref = std::shared_ptr<mapping_state>;
using structured_value = std::variant<std::nullptr_t, bool, std::int64_t,
                                      double, std::string, mapping_ref>;
using mapping_entry = std::pair<std::string, structured_value>;
using mapping_item =
    std::shared_ptr<const std::tuple<std::string, structured_value>>;

struct mapping_state {
  using table_type = std::unordered_map<std::string, structured_value>;
  table_type table;
  // Rehash preserves pointers to unordered_map elements. This profile has no
  // erase operation. Replacing a value retains its position and key.
  std::vector<const table_type::value_type *> order;

  mapping_state() = default;
  mapping_state(const mapping_state &) = delete;
  mapping_state &operator=(const mapping_state &) = delete;

  void put(std::string key, structured_value value) {
    auto [entry, inserted] =
        table.try_emplace(std::move(key), std::move(value));
    if (!inserted) {
      entry->second = std::move(value);
      return;
    }
    try {
      order.push_back(std::addressof(*entry));
    } catch (...) {
      table.erase(entry);
      throw;
    }
  }
};

inline mapping_ref mapping_make(std::initializer_list<mapping_entry> entries) {
  auto result = std::make_shared<mapping_state>();
  result->table.reserve(entries.size());
  result->order.reserve(entries.size());
  for (const auto &[key, value] : entries)
    result->put(key, value);
  return result;
}

struct mapping_lookup {
  mapping_ref base;
  std::string key;
  std::span<const std::uint32_t> nonprintable;
};

inline structured_value mapping_get(mapping_lookup request) {
  const auto found = request.base->table.find(request.key);
  if (found == request.base->table.end())
    throw program_error(
        "key_error",
        value_display::quoted(request.key, false, request.nonprintable));
  return found->second;
}

struct mapping_assignment {
  mapping_ref base;
  std::string key;
  structured_value value;
};

inline std::nullptr_t mapping_put(mapping_assignment request) {
  request.base->put(std::move(request.key), std::move(request.value));
  return nullptr;
}

struct mapping_pair {
  mapping_ref base, source;
};

inline std::nullptr_t mapping_merge(mapping_pair request) {
  if (request.base != request.source)
    for (const auto *entry : request.source->order)
      request.base->put(entry->first, entry->second);
  return nullptr;
}

inline bool mapping_identity(const mapping_pair &pair) {
  return pair.base == pair.source;
}

inline std::int64_t mapping_size(const mapping_ref &base) {
  static_assert(
      std::numeric_limits<decltype(base->order)::difference_type>::digits <=
      std::numeric_limits<std::int64_t>::digits);
  return static_cast<std::int64_t>(base->order.size());
}

inline bool structured_is_mapping(const structured_value &value) {
  return std::holds_alternative<mapping_ref>(value);
}

enum class mapping_access { items, subscript, unpack };

inline mapping_ref structured_as_mapping(const structured_value &value,
                                         mapping_access access) {
  if (const auto map = std::get_if<mapping_ref>(&value))
    return *map;
  constexpr const char *types[] = {"NoneType", "bool", "int", "float", "str"};
  const auto subject = std::string("'") + types[value.index()] + "' object";
  if (access == mapping_access::items)
    throw program_error("attribute_error",
                        subject + " has no attribute 'items'");
  if (access == mapping_access::unpack)
    throw program_error("type_error", subject + " is not a mapping");
  if (std::holds_alternative<std::string>(value))
    throw program_error("type_error",
                        "string indices must be integers, not 'str'");
  throw program_error("type_error", subject + " is not subscriptable");
}

inline bool structured_truth(const structured_value &value) {
  return std::visit(
      [](const auto &item) -> bool {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::nullptr_t>)
          return false;
        else if constexpr (std::is_same_v<T, std::string>)
          return !item.empty();
        else if constexpr (std::is_same_v<T, mapping_ref>)
          return !item->order.empty();
        else
          return item != 0;
      },
      value);
}

template <class Left, class Right> struct structured_text_pair {
  Left left;
  Right right;
};

inline bool structured_text_equal(
    structured_text_pair<structured_value, std::string> pair) {
  auto value = std::get_if<std::string>(&pair.left);
  return value && *value == pair.right;
}

inline bool structured_text_equal(
    structured_text_pair<std::string, structured_value> pair) {
  auto value = std::get_if<std::string>(&pair.right);
  return value && pair.left == *value;
}

struct mapping_items_view {
  mapping_ref base;
};
using mapping_items_ref = std::shared_ptr<const mapping_items_view>;

inline mapping_items_ref mapping_items(mapping_ref base) {
  return std::make_shared<const mapping_items_view>(std::move(base));
}

struct mapping_cursor {
  mapping_ref base;
  std::size_t initial_size, next_index = 0;
  mapping_item current;

  explicit mapping_cursor(mapping_ref map)
      : base(std::move(map)), initial_size(base->order.size()) {}
};
using mapping_iterator_ref = std::shared_ptr<mapping_cursor>;

inline mapping_iterator_ref mapping_iterator(const mapping_items_ref &view) {
  return std::make_shared<mapping_cursor>(view->base);
}

inline bool mapping_next(const mapping_iterator_ref &it) {
  if (!it->base)
    return false;
  if (it->base->order.size() != it->initial_size)
    throw program_error("runtime_error",
                        "dictionary changed size during iteration");
  if (it->next_index == it->initial_size) {
    it->current.reset();
    it->base.reset();
    return false;
  }
  const auto &[key, value] = *it->base->order[it->next_index];
  it->current =
      std::make_shared<const std::tuple<std::string, structured_value>>(key,
                                                                        value);
  ++it->next_index;
  return true;
}

inline mapping_item mapping_current(const mapping_iterator_ref &it) {
  if (!it->current)
    throw std::logic_error("mapping cursor has no current item");
  return it->current;
}
} // namespace scar
