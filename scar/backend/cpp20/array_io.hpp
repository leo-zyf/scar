#pragma once

#include "array.hpp"
#include "host.hpp"
#include "text.hpp"
#include <cerrno>
#include <cstdlib>
#include <locale.h>

namespace scar {
struct text_array_request {
  host::text_stream source_stream;
  array_warning site;
  std::span<const std::uint32_t> nonprintable;
};

namespace text_array_input {
inline bool whitespace(std::uint32_t cp) {
  return (cp >= 9 && cp <= 13) || (cp >= 28 && cp <= 32) || cp == 0x85 ||
         cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
         cp == 0x3000;
}

inline void split_line(std::string_view line,
                       std::vector<std::string_view> &fields) {
  fields.clear();
  std::size_t start = line.npos;
  std::size_t i = 0;
  for (; i < line.size();) {
    const auto [cp, width] = value_display::first(line.substr(i));
    if (cp == '#')
      break;
    if (whitespace(cp)) {
      if (start != line.npos) {
        fields.push_back(line.substr(start, i - start));
        start = line.npos;
      }
    } else if (start == line.npos)
      start = i;
    i += width;
  }
  if (start != line.npos)
    fields.push_back(line.substr(start, i - start));
}

class numeric_locale {
  locale_t value_;

public:
  numeric_locale() : value_(::newlocale(LC_NUMERIC_MASK, "C", nullptr)) {
    if (!value_)
      throw std::system_error(errno, std::generic_category(),
                              "create decimal input locale");
  }
  numeric_locale(const numeric_locale &) = delete;
  ~numeric_locale() { ::freelocale(value_); }
  locale_t get() const { return value_; }
};

// Each field is a view into one NUL-terminated input line. Its delimiter
// terminates decimal conversion, and the end pointer must cover the field.
inline bool number(std::string_view token, double &value) {
  auto magnitude = token;
  if (magnitude.starts_with('+') || magnitude.starts_with('-'))
    magnitude.remove_prefix(1);
  if (magnitude.empty() || magnitude.starts_with("0x") ||
      magnitude.starts_with("0X") ||
      ((magnitude.front() == 'n' || magnitude.front() == 'N') &&
       magnitude.size() != 3))
    return false;

  static const numeric_locale locale;
  const int saved_errno = errno;
  char *end = nullptr;
  value = ::strtod_l(token.data(), &end, locale.get());
  errno = saved_errno;
  return end == token.data() + token.size();
}

inline std::string display_token(std::string_view token,
                                 std::span<const std::uint32_t> intervals) {
  auto result = value_display::quoted(token, false, intervals);
  std::size_t end = 0;
  for (std::size_t chars = 0; end < result.size() && chars < 100; ++chars)
    end += value_display::first(std::string_view(result).substr(end)).bytes;
  result.resize(end);
  return result;
}

inline void warn_empty(const text_array_request &request) {
  const auto &site = request.site;
  const auto &origin = host::stream(request.source_stream).origin();
  if (!origin)
    throw std::logic_error("missing qualified named stream description");

  std::string message(site.operation);
  message += ": input contained no data: \"<_io.TextIOWrapper name=";
  message += value_display::quoted(origin->path, false, request.nonprintable);
  message += " mode='";
  message += origin->mode;
  message += "' encoding='";
  message += origin->encoding;
  message += "'>\"";

  static std::mutex lock;
  static std::set<std::tuple<std::string, std::size_t, std::string>> seen;
  const std::lock_guard guard(lock);
  if (!seen.emplace(site.module, site.line, message).second)
    return;
  std::string report(site.file);
  report +=
      ":" + std::to_string(site.line) + ": UserWarning: " + message + "\n";
  if (!site.source_line.empty()) {
    report += "  ";
    report += site.source_line;
    report += '\n';
  }
  if (std::fwrite(report.data(), 1, report.size(), stderr) != report.size() ||
      std::fflush(stderr))
    throw std::ios_base::failure("cannot write array input warning");
}
} // namespace text_array_input

inline ndarray<double> text_array_read(const text_array_request &request) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);
  std::vector<double> data;
  std::vector<std::string_view> fields;
  std::size_t rows = 0, columns = 0;
  auto &stream = host::stream(request.source_stream);
  for (;;) {
    const auto line = stream.read(-1, true);
    if (line.empty())
      break;
    text_array_input::split_line(line, fields);
    if (fields.empty())
      continue;
    if (!rows)
      columns = fields.size();
    else if (fields.size() != columns)
      throw program_error("value_error",
                          "the number of columns changed from " +
                              std::to_string(columns) + " to " +
                              std::to_string(fields.size()) + " at row " +
                              std::to_string(rows + 1) +
                              "; use `usecols` to select a subset and "
                              "avoid this error");
    for (std::size_t col = 0; col < columns; ++col) {
      double value;
      if (!text_array_input::number(fields[col], value))
        throw program_error("value_error",
                            "could not convert string " +
                                text_array_input::display_token(
                                    fields[col], request.nonprintable) +
                                " to float64 at row " + std::to_string(rows) +
                                ", column " + std::to_string(col + 1) + ".");
      data.push_back(value);
    }
    ++rows;
  }

  std::vector<std::size_t> shape;
  if (rows != 1)
    shape.push_back(rows);
  if (columns > 1)
    shape.push_back(columns);
  auto result = ndarray<double>::from_row_major(std::move(shape), data);
  if (!rows)
    text_array_input::warn_empty(request);
  return result;
}
} // namespace scar
