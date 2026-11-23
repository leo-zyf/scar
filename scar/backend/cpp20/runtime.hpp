#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
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
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#include <sched.h>
#endif

namespace scar {
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
                               std::size_t capacity, const char *format, ...) {
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
inline int dynamic_sprintf_va(char *out, std::size_t capacity,
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
