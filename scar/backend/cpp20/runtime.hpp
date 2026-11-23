#pragma once

#include "exceptions.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <charconv>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fenv.h>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#include <sched.h>
#endif

namespace scar {
enum class text_role { argument, sep, end, flush, stream };
using text_value =
    std::variant<std::string, bool, std::nullptr_t, std::int64_t, double>;
struct text_operand {
  text_role role;
  text_value value;
};
struct text_format {
  std::string_view boolean_true, boolean_false, unit;
  int integer_base;
};
struct text_output {
  std::initializer_list<text_operand> operands;
  text_format format;
};

inline std::string concat_text(std::initializer_list<std::string> parts) {
  std::string result;
  std::size_t size = 0;
  for (const auto &part : parts) {
    if (part.size() > result.max_size() - size)
      throw std::length_error("text concatenation exceeds capacity");
    size += part.size();
  }
  result.reserve(size);
  for (const auto &part : parts)
    result += part;
  return result;
}

inline void flush_stdout() {
  if (std::fflush(stdout) != 0)
    throw std::ios_base::failure("stdout flush failed");
}

inline void write_stdout(std::string_view bytes, bool line_buffered) {
  if (bytes.empty())
    return;
  if (std::fwrite(bytes.data(), 1, bytes.size(), stdout) != bytes.size() ||
      std::ferror(stdout))
    throw std::ios_base::failure("stdout write failed");
  if (line_buffered && bytes.find_first_of("\r\n") != std::string_view::npos)
    flush_stdout();
}

// Format the shortest round-tripping decimal with the declared binary64
// profile.
inline std::string shortest_binary64(double value) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);
  if (std::isnan(value))
    return "nan";
  if (std::isinf(value))
    return std::signbit(value) ? "-inf" : "inf";
  if (value == 0.)
    return std::signbit(value) ? "-0.0" : "0.0";
  char storage[48];
  auto [end, error] =
      std::to_chars(storage, storage + sizeof(storage), std::abs(value),
                    std::chars_format::scientific);
  if (error != std::errc{})
    throw std::runtime_error("binary64 conversion failed");
  std::string_view sci(storage, static_cast<std::size_t>(end - storage));
  const auto e = sci.find('e');
  if (e == sci.npos || e + 2 >= sci.size())
    throw std::runtime_error("invalid binary64 exponent");
  int power = 0;
  auto parsed = std::from_chars(storage + e + 2, end, power);
  if (parsed.ec != std::errc{} || parsed.ptr != end)
    throw std::runtime_error("invalid binary64 exponent");
  if (sci[e + 1] == '-')
    power = -power;
  std::string out = std::signbit(value) ? "-" : "";
  if (power < -4 || power >= 16) {
    out.append(sci);
    return out;
  }
  std::string digits(sci.substr(0, e));
  if (digits.size() > 1)
    digits.erase(1, 1);
  const auto point = power + 1;
  if (point <= 0) {
    out += "0.";
    out.append(-point, '0');
    out += digits;
  } else if (static_cast<std::size_t>(point) >= digits.size()) {
    out += digits;
    out.append(point - digits.size(), '0');
    out += ".0";
  } else {
    digits.insert(point, 1, '.');
    out += digits;
  }
  return out;
}
template <class Value, class Write>
inline void render_scalar(const Value &value, const text_format &format,
                          Write write) {
  if (auto s = std::get_if<std::string>(&value))
    write(*s);
  else if (auto b = std::get_if<bool>(&value))
    write(*b ? format.boolean_true : format.boolean_false);
  else if (std::holds_alternative<std::nullptr_t>(value))
    write(format.unit);
  else if (auto f = std::get_if<double>(&value))
    write(shortest_binary64(*f));
  else if (auto i = std::get_if<std::int64_t>(&value)) {
    char digits[std::numeric_limits<std::int64_t>::digits10 + 3];
    auto [end, error] =
        std::to_chars(digits, digits + sizeof(digits), *i, format.integer_base);
    if (error != std::errc{})
      throw std::ios_base::failure("integer formatting failed");
    write(std::string_view(digits, static_cast<std::size_t>(end - digits)));
  } else
    throw std::invalid_argument("unsupported scalar text representation");
}
struct formatted_scalar {
  text_value value;
  text_format format;
};
inline std::string format_scalar(const formatted_scalar &input) {
  std::string result;
  render_scalar(input.value, input.format,
                [&](std::string_view v) { result.assign(v); });
  return result;
}
inline void write_text_value(const text_value &value, const text_format &format,
                             bool line_buffered) {
  render_scalar(value, format,
                [&](std::string_view s) { write_stdout(s, line_buffered); });
}
template <class Output, class Write, class Flush>
inline std::nullptr_t print_sequence(const Output &output, Write write,
                                     Flush flush_stream) {
  std::string_view separator = " ", ending = "\n";
  bool flush = false;
  for (const auto &operand : output.operands) {
    switch (operand.role) {
    case text_role::sep:
      separator = std::get<std::string>(operand.value);
      break;
    case text_role::end:
      ending = std::get<std::string>(operand.value);
      break;
    case text_role::flush:
      flush = std::get<bool>(operand.value);
      break;
    case text_role::argument:
    case text_role::stream:
      break;
    }
  }
  bool first = true;
  for (const auto &operand : output.operands) {
    if (operand.role != text_role::argument)
      continue;
    if (!first)
      write(separator);
    render_scalar(operand.value, output.format, write);
    first = false;
  }
  write(ending);
  if (flush)
    flush_stream();
  return nullptr;
}
inline std::nullptr_t text_print(const text_output &output) {
  const bool tty = ::isatty(::fileno(stdout)) != 0;
  return print_sequence(
      output, [&](std::string_view s) { write_stdout(s, tty); }, flush_stdout);
}

template <class T, std::size_t N> using array = T[N];
template <class R, class... A> using function = R(A...);
template <class R, class... A> using variadic_function = R(A..., ...);
template <class T, class F>
[[gnu::noinline]] decltype(auto) with_stack_array(std::size_t count, F action) {
  static_assert(std::is_trivially_default_constructible_v<T> &&
                std::is_trivially_destructible_v<T>);
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::bad_array_new_length();
  const auto bytes = (count ? count : 1) * sizeof(T);
  void *memory = __builtin_alloca_with_align(bytes, alignof(T) * CHAR_BIT);
  return action(::new (memory) T[count]);
}
template <class T> T load_bytes(const void *base, std::ptrdiff_t offset) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value;
  std::memcpy(&value, static_cast<const std::byte *>(base) + offset, sizeof(T));
  return value;
}
template <class T>
void store_bytes(void *base, std::ptrdiff_t offset, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::memcpy(static_cast<std::byte *>(base) + offset, &value, sizeof(T));
}
template <class T, class U> T pointer_cast(U value) {
  static_assert(std::is_pointer_v<T>);
  if constexpr (requires { static_cast<T>(value); })
    return static_cast<T>(value);
  else if constexpr (requires { const_cast<T>(value); })
    return const_cast<T>(value);
  else if constexpr (requires { reinterpret_cast<T>(value); })
    return reinterpret_cast<T>(value);
  else
    return static_cast<T>(
        const_cast<void *>(static_cast<const volatile void *>(value)));
}

inline double real_power(double base, double exponent) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);
  if (exponent == 0.0 || base == 1.0)
    return 1.0;
  if (std::isnan(base))
    return base;
  if (std::isnan(exponent))
    return exponent;
  const double magnitude = std::abs(base);
  if (std::isinf(exponent)) {
    if (magnitude == 1.0)
      return 1.0;
    return (exponent > 0.0) == (magnitude > 1.0)
               ? std::numeric_limits<double>::infinity()
               : 0.0;
  }
  const bool negative =
      std::signbit(base) && std::fmod(std::abs(exponent), 2.0) == 1.0;
  if (std::isinf(base)) {
    const double result = exponent > 0.0 ? magnitude : 0.0;
    return negative ? -result : result;
  }
  if (base == 0.0 && exponent < 0.0)
    throw program_error("zero_division_error",
                        "0.0 cannot be raised to a negative power");
  if (base < 0.0 && exponent != std::trunc(exponent))
    throw std::logic_error("real power requires a proven real result");

  double result = magnitude;
  if (magnitude != 0.0 && magnitude != 1.0) {
    errno = 0;
    result = std::pow(magnitude, exponent);
    if (std::isinf(result) && errno == 0)
      errno = ERANGE;
    else if (errno == ERANGE && result == 0.0)
      errno = 0;
    if (const int error = errno) {
      const auto message = std::generic_category().message(error);
      throw program_error(error == ERANGE ? "overflow_error" : "value_error",
                          "(" + std::to_string(error) + ", '" + message + "')");
    }
  }
  return negative ? -result : result;
}

template <class T, class E> T powi(T base, E exponent) {
  static_assert(std::is_integral_v<E>);
  const bool negative = exponent < 0;
  if constexpr (std::is_integral_v<T>) {
    if (negative && base != T{1} && base != static_cast<T>(-1))
      return T{0};
  }
  using U = std::make_unsigned_t<E>;
  U power = static_cast<U>(exponent);
  if (negative)
    power = U{0} - power;
  T result = (power & U{1}) ? base : T{1};
  while ((power >>= 1) != 0) {
    base *= base;
    if (power & U{1})
      result *= base;
  }
  return negative ? T{1} / result : result;
}
template <class T> T sign(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::copysign(std::abs(a), b);
  } else {
    using U = std::make_unsigned_t<T>;
    const U bits = static_cast<U>(a);
    const U magnitude = a < 0 ? U{0} - bits : bits;
    return static_cast<T>(b < 0 ? U{0} - magnitude : magnitude);
  }
}
template <class T> T mod(T a, T b) {
  if constexpr (std::is_integral_v<T>)
    return a % b;
  else
    return std::fmod(a, b);
}
template <class T> T modulo(T a, T b) {
  T remainder;
  if constexpr (std::is_integral_v<T>)
    remainder = a % b;
  else {
    remainder = std::fmod(a, b);
    if (remainder == 0)
      return std::copysign(T{0}, b);
  }
  if (remainder != 0 && ((remainder < 0) != (b < 0)))
    remainder += b;
  return remainder;
}
template <class A, class R> struct unary_function_pointer {
  using argument_type = A;
  using result_type = R;
  R (*function)(A);
  R operator()(A value) const { return function(value); }
};
template <class A, class B, class R> struct binary_function_pointer {
  using first_argument_type = A;
  using second_argument_type = B;
  using result_type = R;
  R (*function)(A, B);
  R operator()(A a, B b) const { return function(a, b); }
};
template <class A, class R> auto legacy_function(R (*f)(A)) {
  return unary_function_pointer<A, R>{f};
}
template <class A, class B, class R> auto legacy_function(R (*f)(A, B)) {
  return binary_function_pointer<A, B, R>{f};
}
template <class P> struct unary_negation {
  using argument_type = typename P::argument_type;
  using result_type = bool;
  P predicate;
  bool operator()(const argument_type &value) const {
    return !predicate(value);
  }
};
template <class P> auto legacy_not(const P &f) { return unary_negation<P>{f}; }
inline int dynamic_fprintf(FILE *sink, const char *format, ...) {
  std::va_list ap;
  va_start(ap, format);
  int result = ::vfprintf(sink, format, ap);
  va_end(ap);
  return result;
}
inline int dynamic_printf(const char *format, ...) {
  std::va_list ap;
  va_start(ap, format);
  int result = ::vprintf(format, ap);
  va_end(ap);
  return result;
}
inline int dynamic_snprintf_va(char *out, std::size_t limit,
                               [[maybe_unused]] std::size_t capacity,
                               const char *format, ...) {
  std::va_list ap;
  va_start(ap, format);
#if defined(__GLIBC__) && __USE_FORTIFY_LEVEL > 0
  int result = ::__vsnprintf_chk(out, limit, __USE_FORTIFY_LEVEL - 1, capacity,
                                 format, ap);
#else
  int result = ::vsnprintf(out, limit, format, ap);
#endif
  va_end(ap);
  return result;
}
[[gnu::always_inline]] inline int dynamic_snprintf(char *out, std::size_t limit,
                                                   const char *format) {
#if defined(__GLIBC__) && __USE_FORTIFY_LEVEL > 0
  auto capacity = __glibc_objsize(out);
#else
  auto capacity = std::size_t{0};
#endif
  return dynamic_snprintf_va(out, limit, capacity, format);
}
inline int dynamic_sprintf_va(char *out, [[maybe_unused]] std::size_t capacity,
                              const char *format, ...) {
  std::va_list ap;
  va_start(ap, format);
#if defined(__GLIBC__) && __USE_FORTIFY_LEVEL > 0
  int result =
      ::__vsprintf_chk(out, __USE_FORTIFY_LEVEL - 1, capacity, format, ap);
#else
  int result = ::vsprintf(out, format, ap);
#endif
  va_end(ap);
  return result;
}
[[gnu::always_inline]] inline int dynamic_sprintf(char *out,
                                                  const char *format) {
#if defined(__GLIBC__) && __USE_FORTIFY_LEVEL > 0
  auto capacity = __glibc_objsize(out);
#else
  auto capacity = std::size_t{0};
#endif
  return dynamic_sprintf_va(out, capacity, format);
}
inline bool fp_traps_masked() {
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  return ::fegetexcept() == 0;
#else
  return false;
#endif
}
#if defined(_OPENMP)
inline bool team_available(int workers) {
  if (omp_in_parallel() || workers > omp_get_max_threads() ||
      workers > omp_get_thread_limit()) {
    return false;
  }
  cpu_set_t allowed;
  return sched_getaffinity(0, sizeof(allowed), &allowed) == 0 &&
         workers <= CPU_COUNT(&allowed);
}
template <class F, class Serial>
inline void map(std::int64_t n, int workers, F body, Serial serial) {
  std::fenv_t initial;
  if (workers <= 1 || workers > n || !team_available(workers) ||
      !fp_traps_masked() || std::fegetenv(&initial) != 0) {
    serial();
    return;
  }
  const int incoming = std::fetestexcept(FE_ALL_EXCEPT);
  int raised = 0;
#pragma omp parallel num_threads(workers) reduction(| : raised)
  {
    std::fenv_t previous;
    std::fegetenv(&previous);
    std::fesetenv(&initial);
#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < n; ++i)
      body(i);
    raised |= std::fetestexcept(FE_ALL_EXCEPT);
    std::fesetenv(&previous);
  }
  raised &= ~incoming;
  if (raised != 0) {
    // Transfer only flags actually raised by a worker. feraiseexcept may also
    // set inexact, so construct the desired flag subset in a held environment.
    std::fenv_t saved;
    std::fexcept_t flags;
    std::feholdexcept(&saved);
    std::feraiseexcept(raised);
    std::fegetexceptflag(&flags, raised);
    std::fesetenv(&saved);
    std::fesetexceptflag(&flags, raised);
  }
}
template <class F> inline void map(std::int64_t n, int workers, F body) {
  map(n, workers, body, [&] {
    for (std::int64_t i = 0; i < n; ++i)
      body(i);
  });
}
#endif

template <std::int64_t Elements, std::int64_t Count, std::size_t Inputs,
          class Kernel>
int binary_f64_entry(int argc, char **argv, Kernel kernel) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
  if (argc != 3)
    return 2;
  constexpr std::streamsize bytes = Elements * sizeof(double);
  std::ifstream source(argv[1], std::ios::binary);
  if (!source)
    return 3;
  std::array<std::vector<double>, Inputs> input;
  for (auto &buffer : input) {
    buffer.resize(Elements);
    if (bytes && !source.read(reinterpret_cast<char *>(buffer.data()), bytes))
      return 3;
  }
  if (source.peek() != std::char_traits<char>::eof() || source.bad())
    return 3;
  std::vector<double> output(Elements);
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    kernel(Elements, Count, input[I].data()..., output.data());
  }(std::make_index_sequence<Inputs>{});
  std::ofstream sink(argv[2], std::ios::binary);
  if (!sink)
    return 4;
  if (bytes)
    sink.write(reinterpret_cast<const char *>(output.data()), bytes);
  sink.close();
  return sink ? 0 : 4;
}
} // namespace scar

#define SCAR_OFFSET_MATCH(Type, field, bits)                                   \
  []<class T>() {                                                              \
    if constexpr (std::is_standard_layout_v<T>)                                \
      return offsetof(T, field) * CHAR_BIT == (bits);                          \
    else                                                                       \
      return true;                                                             \
  }.template operator()<Type>()
