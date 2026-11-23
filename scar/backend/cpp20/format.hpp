#pragma once

#include "array.hpp"
#include "runtime.hpp"
#include <charconv>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace scar {
struct array_format_options {
  int precision;
  std::size_t threshold, edgeitems, linewidth;
  bool suppress;
  std::string_view sign, floatmode;
  bool legacy;
};

namespace text_detail {
struct decimal {
  std::string integer, fraction, exponent;
  bool special = false;
};

inline decimal digits(double value, bool scientific, int precision) {
  if (!std::isfinite(value))
    return {std::isnan(value)     ? "nan"
            : std::signbit(value) ? "-inf"
                                  : "inf",
            {},
            {},
            true};
  char buffer[768];
  const auto mode =
      scientific ? std::chars_format::scientific : std::chars_format::fixed;
  auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value, mode);
  if (converted.ec != std::errc{})
    throw std::runtime_error("array decimal conversion failed");
  std::string raw(buffer, converted.ptr);
  auto e = raw.find('e'), dot = raw.find('.');
  auto end = e == raw.npos ? raw.size() : e;
  if (dot != raw.npos && end - dot - 1 > static_cast<std::size_t>(precision)) {
    converted =
        std::to_chars(buffer, buffer + sizeof(buffer), value, mode, precision);
    if (converted.ec != std::errc{})
      throw std::runtime_error("array decimal rounding failed");
    raw.assign(buffer, converted.ptr);
    e = raw.find('e');
    dot = raw.find('.');
    end = e == raw.npos ? raw.size() : e;
  }
  std::string fraction =
      dot == raw.npos ? "" : raw.substr(dot + 1, end - dot - 1);
  while (!fraction.empty() && fraction.back() == '0')
    fraction.pop_back();
  return {raw.substr(0, dot == raw.npos ? end : dot), std::move(fraction),
          e == raw.npos ? "" : raw.substr(e + 1), false};
}

inline std::vector<std::size_t> shown(std::size_t count, bool summary,
                                      std::size_t edge) {
  std::vector<std::size_t> result;
  if (summary && count > 2 * edge) {
    for (std::size_t i = 0; i < edge; ++i)
      result.push_back(i);
    result.push_back(count); // An omitted interval is not a data index.
    for (std::size_t i = count - edge; i < count; ++i)
      result.push_back(i);
  } else {
    for (std::size_t i = 0; i < count; ++i)
      result.push_back(i);
  }
  return result;
}
} // namespace text_detail

template <class T>
std::string format_array(const ndarray<T> &array,
                         const array_format_options &options) {
  static_assert(std::is_same_v<T, double>);
  if (options.precision != 8 || options.threshold != 1000 ||
      options.edgeitems != 3 || options.linewidth != 75 || options.suppress ||
      options.sign != "-" || options.floatmode != "maxprec" || options.legacy)
    throw std::invalid_argument("unsupported array display profile");
  const auto &dims = array.dimensions();
  if (dims.empty())
    return shortest_binary64(array.value_at_flat(0));
  if (std::find(dims.begin(), dims.end(), 0) != dims.end())
    return "[]";
  std::vector<std::size_t> stride(dims.size(), 1);
  for (std::size_t i = dims.size() - 1; i > 0; --i)
    stride[i - 1] = stride[i] * dims[i];
  const bool summary = stride.front() * dims.front() > options.threshold;
  std::vector<std::vector<std::size_t>> indices;
  for (auto extent : dims)
    indices.push_back(text_detail::shown(extent, summary, options.edgeitems));
  std::map<std::size_t, double> samples;
  std::function<void(std::size_t, std::size_t)> collect = [&](auto axis,
                                                              auto offset) {
    if (axis == dims.size()) {
      samples.emplace(offset, array.value_at_flat(offset));
      return;
    }
    for (auto index : indices[axis])
      if (index != dims[axis])
        collect(axis + 1, offset + index * stride[axis]);
  };
  collect(0, 0);
  double largest = 0., smallest = std::numeric_limits<double>::infinity();
  for (auto [index, value] : samples) {
    static_cast<void>(index);
    if (std::isfinite(value) && value != 0.) {
      largest = std::max(largest, std::abs(value));
      smallest = std::min(smallest, std::abs(value));
    }
  }
  const bool scientific = largest >= 1.e8 || smallest < 1.e-4 ||
                          (largest != 0. && largest / smallest > 1000.);
  std::map<std::size_t, text_detail::decimal> decimals;
  std::size_t left = 0, right = 0, exponent = 0, special = 0;
  bool finite = false;
  for (auto [index, value] : samples) {
    auto repr = text_detail::digits(value, scientific, options.precision);
    if (repr.special)
      special = std::max(special, repr.integer.size());
    else {
      finite = true;
      left = std::max(left, repr.integer.size());
      right = std::max(right, repr.fraction.size());
      exponent = std::max(exponent,
                          repr.exponent.empty() ? 0 : repr.exponent.size() - 1);
    }
    decimals.emplace(index, std::move(repr));
  }
  const std::size_t suffix = scientific ? 2 + exponent : 0;
  std::size_t width = finite ? left + 1 + right + suffix : 0;
  if (special > width) {
    if (finite)
      left += special - width;
    width = special;
  }
  std::map<std::size_t, std::string> fields;
  for (const auto &[index, value] : decimals) {
    if (value.special) {
      fields.emplace(index, std::string(width - value.integer.size(), ' ') +
                                value.integer);
      continue;
    }
    std::string text(left - value.integer.size(), ' ');
    text += value.integer + '.' + value.fraction;
    text.append(right - value.fraction.size(), scientific ? '0' : ' ');
    if (scientific) {
      text += 'e';
      text += value.exponent.front();
      text.append(exponent - value.exponent.size() + 1, '0');
      text += value.exponent.substr(1);
    }
    fields.emplace(index, std::move(text));
  }
  std::function<std::string(std::size_t, std::size_t, std::size_t)> render;
  render = [&](auto axis, auto offset, auto closing) {
    std::string text = "[";
    bool first = true;
    if (axis + 1 == dims.size()) {
      std::size_t column = axis + 1;
      for (auto index : indices[axis]) {
        const auto item = index == dims[axis] ? std::string("...")
                                              : fields.at(offset + index);
        if (!first) {
          if (column + 1 + item.size() + dims.size() > options.linewidth) {
            while (!text.empty() && text.back() == ' ')
              text.pop_back();
            text += '\n';
            text.append(axis + 1, ' ');
            column = axis + 1;
          } else {
            text += ' ';
            ++column;
          }
        }
        text += item;
        column += item.size();
        first = false;
      }
    } else {
      for (std::size_t position = 0; position < indices[axis].size();
           ++position) {
        auto index = indices[axis][position];
        if (!first) {
          text.append(dims.size() - axis - 1, '\n');
          text.append(axis + 1, ' ');
        }
        if (index == dims[axis])
          text += "...";
        else
          text +=
              render(axis + 1, offset + index * stride[axis],
                     position + 1 == indices[axis].size() ? closing + 1 : 1);
        first = false;
      }
    }
    return text + ']';
  };
  return render(0, 0, 1);
}
} // namespace scar
