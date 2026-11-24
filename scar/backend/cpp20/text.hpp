#pragma once

#include "runtime.hpp"
#include <span>

namespace scar {
struct value_format_spec {
  std::string_view fill, align, sign;
  bool negative_zero, alternate, zero;
  std::size_t width;
  std::string_view grouping;
  int precision;
  std::string_view presentation;
  bool sign_explicit, fill_explicit, empty;
};
struct value_format_request {
  text_value value;
  std::string_view conversion;
  value_format_spec spec;
  text_format spelling;
  std::span<const std::uint32_t> nonprintable;
};

namespace value_display {
struct character {
  std::uint32_t code;
  std::size_t bytes;
};
// The byte-preserving UTF-8 ABI maps each ill-formed non-ASCII byte to
// U+DC80..DCFF. A valid sequence always takes precedence over that mapping.
inline character first(std::string_view s) {
  auto b = static_cast<unsigned char>(s.front());
  if (b < 128)
    return {b, 1};
  auto invalid = character{0xdc00u + b, 1};
  unsigned n = b < 0xc2 ? 0 : b < 0xe0 ? 2 : b < 0xf0 ? 3 : b < 0xf5 ? 4 : 0;
  if (!n || s.size() < n)
    return invalid;
  std::uint32_t v = b & (0x7f >> n);
  for (unsigned i = 1; i < n; ++i) {
    auto c = static_cast<unsigned char>(s[i]);
    if ((c & 0xc0) != 0x80)
      return invalid;
    v = (v << 6) | (c & 0x3f);
  }
  if ((n == 3 && v < 0x800) || (n == 4 && v < 0x10000) ||
      (v >= 0xd800 && v <= 0xdfff) || v > 0x10ffff)
    return invalid;
  return {v, n};
}
inline bool codepoint_less(std::string_view left, std::string_view right) {
  while (!left.empty() && !right.empty()) {
    const auto a = first(left);
    const auto b = first(right);
    if (a.code != b.code)
      return a.code < b.code;
    left.remove_prefix(a.bytes);
    right.remove_prefix(b.bytes);
  }
  return left.empty() && !right.empty();
}

inline std::string unicode(std::int64_t n) {
  if (n < 0 || n > 0x10ffff)
    throw program_error("overflow_error", "%c arg not in range(0x110000)");
  if (n < 128)
    return std::string(1, static_cast<char>(n));
  if (n >= 0xdc80 && n <= 0xdcff)
    return std::string(1, static_cast<char>(n - 0xdc00));
  if (n >= 0xd800 && n <= 0xdfff)
    throw program_error("unicode_error",
                        "isolated surrogate outside text transport");
  auto cp = static_cast<std::uint32_t>(n);
  int width = cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
  std::string out(width, '\0');
  for (int i = width - 1; i > 0; --i) {
    out[i] = static_cast<char>(0x80 | (cp & 63));
    cp >>= 6;
  }
  out[0] = static_cast<char>((width == 2   ? 0xc0
                              : width == 3 ? 0xe0
                                           : 0xf0) |
                             cp);
  return out;
}
inline std::string quoted(std::string_view src, bool ascii,
                          std::span<const std::uint32_t> intervals) {
  const char quote =
      src.find('\'') != src.npos && src.find('"') == src.npos ? '"' : '\'';
  std::string out(1, quote);
  constexpr std::string_view hex = "0123456789abcdef";
  while (!src.empty()) {
    auto [cp, n] = first(src);
    if (cp == static_cast<unsigned char>(quote) || cp == '\\') {
      out += '\\';
      out += static_cast<char>(cp);
    } else if (cp == '\t')
      out += "\\t";
    else if (cp == '\n')
      out += "\\n";
    else if (cp == '\r')
      out += "\\r";
    else {
      bool escape = cp < 32 || cp == 127 || (ascii && cp >= 128);
      if (!escape && cp >= 128) {
        auto pos = std::lower_bound(intervals.begin(), intervals.end(), cp);
        escape = pos != intervals.end() &&
                 (*pos == cp || ((pos - intervals.begin()) % 2));
      }
      if (escape) {
        auto digits = cp <= 0xff ? 2 : cp <= 0xffff ? 4 : 8;
        out += digits == 2 ? "\\x" : digits == 4 ? "\\u" : "\\U";
        for (int i = digits - 1; i >= 0; --i)
          out += hex[(cp >> (4 * i)) & 15];
      } else
        out.append(src.substr(0, n));
    }
    src.remove_prefix(n);
  }
  out += quote;
  return out;
}
inline void repeat(std::string &s, std::string_view fill, std::size_t count) {
  if (!count)
    return;
  if (fill.size() > (s.max_size() - s.size()) / count)
    throw std::length_error("formatted text exceeds capacity");
  if (fill.size() == 1) {
    s.append(count, fill.front());
    return;
  }
  s.reserve(s.size() + fill.size() * count);
  for (std::size_t i = 0; i < count; ++i)
    s += fill;
}
inline std::string pad(std::string_view value, std::size_t count,
                       const value_format_spec &s, bool numeric,
                       std::string_view prefix = {}) {
  auto fill = s.zero && !s.fill_explicit ? std::string_view("0") : s.fill;
  auto align =
      s.align == "auto" ? (numeric ? (s.zero ? "=" : ">") : "<") : s.align;
  auto size = count + prefix.size();
  auto spaces = s.width > size ? s.width - size : 0;
  std::string out;
  if (align == ">")
    repeat(out, fill, spaces);
  if (align == "^")
    repeat(out, fill, spaces / 2);
  out += prefix;
  if (align == "=")
    repeat(out, fill, spaces);
  out += value;
  if (align == "<")
    repeat(out, fill, spaces);
  if (align == "^")
    repeat(out, fill, spaces - spaces / 2);
  return out;
}
[[noreturn]] inline void unknown(std::string_view code, std::string_view type) {
  throw program_error("value_error",
                      "Unknown format code '" + std::string(code) +
                          "' for object of type '" + std::string(type) + "'");
}
inline void check_group(const value_format_spec &s, std::string_view code) {
  if (s.grouping.empty())
    return;
  if (code == "c" || code == "s" || code == "n" ||
      ((code == "o" || code == "x" || code == "X" || code == "b") &&
       s.grouping == ","))
    throw program_error("value_error",
                        "Cannot specify '" + std::string(s.grouping) +
                            "' with '" + std::string(code) + "'.");
}
inline std::string text(std::string_view value, const value_format_spec &s) {
  auto code = s.presentation.empty() ? std::string_view("s") : s.presentation;
  check_group(s, code);
  if (code != "s")
    unknown(code, "str");
  if (s.sign_explicit)
    throw program_error("value_error",
                        s.sign == " "
                            ? "Space not allowed in string format specifier"
                            : "Sign not allowed in string format specifier");
  if (s.negative_zero)
    throw program_error(
        "value_error",
        "Negative zero coercion (z) not allowed in string format specifier");
  if (s.alternate)
    throw program_error(
        "value_error",
        "Alternate form (#) not allowed in string format specifier");
  if (s.align == "=")
    throw program_error("value_error",
                        "'=' alignment not allowed in string format specifier");
  if (s.precision < 0 && !s.width)
    return std::string(value);
  std::size_t length = 0, pos = 0,
              limit = s.precision < 0 ? std::string::npos
                                      : static_cast<std::size_t>(s.precision);
  while (pos < value.size() && length < limit) {
    pos += first(value.substr(pos)).bytes;
    ++length;
  }
  return pad(value.substr(0, pos), length, s, false);
}
inline std::string number(std::string magnitude, std::string prefix,
                          const value_format_spec &s, unsigned group,
                          std::size_t digits) {
  if (!s.grouping.empty() && digits) {
    auto grouped = [&](std::size_t n) { return n + (n - 1) / group; };
    const bool zero = (s.zero && !s.fill_explicit) || s.fill == "0";
    auto total = digits;
    if (zero && (s.align == "=" || (s.align == "auto" && s.zero))) {
      auto tail = prefix.size() + magnitude.size() - digits;
      auto width = s.width > tail ? s.width - tail : 0;
      auto high = std::max(total, width);
      while (total < high) {
        auto mid = total + (high - total) / 2;
        if (grouped(mid) < width)
          total = mid + 1;
        else
          high = mid;
      }
    }
    std::string out;
    out.reserve(grouped(total) + magnitude.size() - digits);
    for (std::size_t i = 0; i < total; ++i) {
      if (i && (total - i) % group == 0)
        out += s.grouping;
      out += i < total - digits ? '0' : magnitude[i - (total - digits)];
    }
    out += std::string_view(magnitude).substr(digits);
    magnitude = std::move(out);
  }
  return pad(magnitude, magnitude.size(), s, true, prefix);
}
inline std::string sign(bool negative, const value_format_spec &s) {
  return negative ? "-" : s.sign == "-" ? "" : std::string(s.sign);
}
inline std::string chars(double v, std::chars_format fmt, int precision) {
  char local[64];
  if (precision < static_cast<int>(sizeof(local))) {
    auto [end, ec] =
        std::to_chars(local, local + sizeof(local), v, fmt, precision);
    if (ec == std::errc{})
      return std::string(local, end);
  }
  // 309 integer digits suffice for any finite binary64.
  std::string digits(static_cast<std::size_t>(precision) + 320, '\0');
  auto [end, ec] = std::to_chars(digits.data(), digits.data() + digits.size(),
                                 v, fmt, precision);
  if (ec != std::errc{})
    throw std::runtime_error("binary64 decimal conversion failed");
  digits.resize(static_cast<std::size_t>(end - digits.data()));
  return digits;
}
inline void decimal_point(std::string &digits) {
  if (digits.find('.') != digits.npos)
    return;
  auto exponent = digits.find('e');
  digits.insert(exponent == digits.npos ? digits.size() : exponent, 1, '.');
}
inline std::string significant(double value, int precision, bool alternate,
                               bool unspecified) {
  // One rounding at the requested significant precision drives both digits
  // and exponent selection, including a carry into the next power of ten.
  precision = std::max(1, precision);
  std::string sci = chars(value, std::chars_format::scientific, precision - 1);
  auto e = sci.find('e');
  int power = 0;
  auto exponent = std::string_view(sci).substr(e + 1);
  auto [end, error] = std::from_chars(exponent.data() + 1,
                                      exponent.data() + exponent.size(), power);
  if (error != std::errc{} || end != exponent.data() + exponent.size())
    throw std::runtime_error("invalid binary64 exponent");
  if (exponent.front() == '-')
    power = -power;
  auto digits = sci.substr(0, e);
  if (digits.size() > 1)
    digits.erase(1, 1);
  if (!alternate)
    while (digits.size() > 1 && digits.back() == '0')
      digits.pop_back();
  if (power < -4 || power >= precision - static_cast<int>(unspecified)) {
    if (digits.size() > 1 || alternate)
      digits.insert(1, 1, '.');
    return digits + sci.substr(e);
  }
  auto point = power + 1;
  if (point <= 0)
    return "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
  auto p = static_cast<std::size_t>(point);
  if (p < digits.size())
    digits.insert(p, 1, '.');
  else {
    digits.append(p - digits.size(), '0');
    if (unspecified)
      digits += ".0";
    else if (alternate)
      digits += '.';
  }
  return digits;
}
inline std::string floating(double v, const value_format_spec &s) {
  auto code = s.presentation;
  check_group(s, code);
  if (!code.empty() && std::string_view("fFeEgG%").find(code) == code.npos)
    unknown(code, "float");
  auto upper = code == "F" || code == "E" || code == "G";
  bool negative = std::signbit(v) && !std::isnan(v);
  v = std::abs(v);
  if (code == "%")
    v *= 100.;
  std::string result;
  if (!std::isfinite(v))
    result = std::isnan(v) ? "nan" : "inf";
  else if (code.empty() && s.precision < 0) {
    result = shortest_binary64(v);
    if (s.alternate)
      decimal_point(result);
  } else {
    int p = s.precision < 0 ? 6 : s.precision;
    if (code == "e" || code == "E")
      result = chars(v, std::chars_format::scientific, p);
    else if (code == "f" || code == "F" || code == "%")
      result = chars(v, std::chars_format::fixed, p);
    else
      result = significant(v, p, s.alternate, code.empty());
    if (s.alternate)
      decimal_point(result);
  }
  if (negative && s.negative_zero) {
    auto mantissa = std::string_view(result).substr(0, result.find('e'));
    if (mantissa.find_first_not_of("0.") == mantissa.npos)
      negative = false;
  }
  if (upper)
    for (char &c : result)
      if (c >= 'a' && c <= 'z')
        c -= ('a' - 'A');
  if (code == "%")
    result += '%';
  auto digits = result.find_first_not_of("0123456789");
  if (digits == result.npos)
    digits = result.size();
  return number(std::move(result), sign(negative, s), s, 3, digits);
}
inline std::string integer(std::int64_t value, const value_format_spec &s,
                           std::string_view type) {
  auto code = s.presentation.empty() ? std::string_view("d") : s.presentation;
  check_group(s, code);
  if (std::string_view("eEfFgG%").find(code) != code.npos)
    return floating(static_cast<double>(value), s);
  if (std::string_view("bcdoxX").find(code) == code.npos)
    unknown(code, type);
  if (s.precision >= 0)
    throw program_error("value_error",
                        "Precision not allowed in integer format specifier");
  if (s.negative_zero)
    throw program_error(
        "value_error",
        "Negative zero coercion (z) not allowed in integer format specifier");
  if (code == "c") {
    if (s.sign_explicit)
      throw program_error("value_error",
                          "Sign not allowed with integer format specifier 'c'");
    if (s.alternate)
      throw program_error(
          "value_error",
          "Alternate form (#) not allowed with integer format specifier 'c'");
    return pad(unicode(value), 1, s, true);
  }
  auto base = code == "b" ? 2 : code == "o" ? 8 : code == "d" ? 10 : 16;
  auto magnitude = static_cast<std::uint64_t>(value);
  if (value < 0)
    magnitude = std::uint64_t(0) - magnitude;
  char buf[65];
  auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), magnitude, base);
  if (ec != std::errc{})
    throw std::runtime_error("integer decimal conversion failed");
  std::string digits(buf, end);
  if (code == "X")
    for (char &c : digits)
      if (c >= 'a' && c <= 'f')
        c -= ('a' - 'A');
  auto prefix = sign(value < 0, s);
  if (s.alternate && base != 10) {
    prefix += '0';
    prefix += code;
  }
  return number(std::move(digits), std::move(prefix), s, base == 10 ? 3 : 4,
                static_cast<std::size_t>(end - buf));
}
} // namespace value_display

inline std::string format_value(const value_format_request &request) {
  const auto &v = request.value;
  const auto &s = request.spec;
  auto scalar_string = [&]() { return format_scalar({v, request.spelling}); };
  if (request.conversion != "none") {
    const auto *str = std::get_if<std::string>(&v);
    auto converted =
        str && (request.conversion == "repr" || request.conversion == "ascii")
            ? value_display::quoted(*str, request.conversion == "ascii",
                                    request.nonprintable)
            : scalar_string();
    return value_display::text(converted, s);
  }
  if (auto str = std::get_if<std::string>(&v))
    return value_display::text(*str, s);
  if (s.empty)
    return scalar_string();
  if (auto i = std::get_if<std::int64_t>(&v))
    return value_display::integer(*i, s, "int");
  if (auto b = std::get_if<bool>(&v))
    return value_display::integer(*b, s, "bool");
  if (auto d = std::get_if<double>(&v))
    return value_display::floating(*d, s);
  throw program_error(
      "type_error", "unsupported format string passed to NoneType.__format__");
}
} // namespace scar
