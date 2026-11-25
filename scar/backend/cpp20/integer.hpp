#pragma once
#include "exceptions.hpp"
#include <cmath>
#include <cstdint>
#include <gmpxx.h>
#include <limits>
#include <string>

namespace scar::integers {
using value = mpz_class;
static_assert(std::numeric_limits<long>::digits == 63);
static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits == 53 &&
              std::numeric_limits<double>::max_exponent == 1024);

inline value from_i64(std::int64_t input) {
  return value(static_cast<long>(input));
}
inline value decimal_literal(const std::string &input) {
  return value(input, 10);
}

inline value from_f64(double input, const char *nan_message,
                      const char *inf_message) {
  if (std::isnan(input))
    throw program_error("value_error", nan_message);
  if (std::isinf(input))
    throw program_error("overflow_error", inf_message);
  return value(input);
}

inline double to_binary64(const value &input, const char *overflow_message) {
  const auto bits = mpz_sizeinbase(input.get_mpz_t(), 2);
  if (bits <= 53)
    return input.get_d();
  if (bits > 1024)
    throw program_error("overflow_error", overflow_message);
  const value magnitude = abs(input);
  const auto shift = bits - 53;
  value leading;
  mpz_fdiv_q_2exp(leading.get_mpz_t(), magnitude.get_mpz_t(), shift);
  // GMP's integral-to-double conversion truncates. Round the significand
  // once, using the guard bit and any sticky bits, before exact scaling.
  const bool guard = mpz_tstbit(magnitude.get_mpz_t(), shift - 1) != 0;
  if (guard && (mpz_odd_p(leading.get_mpz_t()) ||
                mpz_scan1(magnitude.get_mpz_t(), 0) < shift - 1))
    ++leading;
  if (bits == 1024 && mpz_sizeinbase(leading.get_mpz_t(), 2) > 53)
    throw program_error("overflow_error", overflow_message);
  const double rounded = std::ldexp(leading.get_d(), static_cast<int>(shift));
  return input < 0 ? -rounded : rounded;
}

inline value add(const value &a, const value &b) { return a + b; }
inline value sub(const value &a, const value &b) { return a - b; }
inline value mul(const value &a, const value &b) { return a * b; }
inline value negative(const value &a) { return -a; }
inline value absolute(const value &a) { return abs(a); }
inline bool eq(const value &a, const value &b) { return a == b; }
inline bool ne(const value &a, const value &b) { return a != b; }
inline bool lt(const value &a, const value &b) { return a < b; }
inline bool le(const value &a, const value &b) { return a <= b; }
inline bool gt(const value &a, const value &b) { return a > b; }
inline bool ge(const value &a, const value &b) { return a >= b; }

inline value floordiv(const value &a, const value &b,
                      const char *zero_message) {
  if (b == 0)
    throw program_error("zero_division_error", zero_message);
  value result;
  mpz_fdiv_q(result.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
  return result;
}
inline value mod(const value &a, const value &b, const char *zero_message) {
  if (b == 0)
    throw program_error("zero_division_error", zero_message);
  value result;
  mpz_fdiv_r(result.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
  return result;
}
inline std::string decimal(const value &input) { return input.get_str(10); }
inline std::int64_t shape_extent(const value &input, const char *message) {
  if (!input.fits_slong_p())
    throw program_error("value_error", message);
  return input.get_si();
}
inline std::int64_t sequence_index(const value &input, const char *message) {
  if (!input.fits_slong_p())
    throw program_error("index_error", message);
  return input.get_si();
}
} // namespace scar::integers
