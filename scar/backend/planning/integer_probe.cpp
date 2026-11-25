#include <climits>
#include <cstdio>
#include <cstring>
#include <gmpxx.h>
#include <limits>

int main() {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);
  static_assert(GMP_NAIL_BITS == 0);
  static_assert(sizeof(mp_limb_t) * CHAR_BIT == GMP_LIMB_BITS);
  mpz_class one(1L);
  if (mpz_cmp_ui(one.get_mpz_t(), 1))
    return 9;
  mpz_t value, divisor, quotient, remainder;
  mpz_inits(value, divisor, quotient, remainder, nullptr);
  mpz_set_si(value, -7);
  mpz_set_ui(divisor, 3);
  mpz_fdiv_qr(quotient, remainder, value, divisor);
  if (mpz_cmp_si(quotient, -3) || mpz_cmp_ui(remainder, 2))
    return 1;
  mpz_set_d(value, -1.75);
  if (mpz_cmp_si(value, -1))
    return 2;
  mpz_abs(value, value);
  mpz_mul(value, value, divisor);
  mpz_add(value, value, divisor);
  mpz_sub(value, value, divisor);
  mpz_neg(value, value);
  if (mpz_cmp_si(value, -3))
    return 3;
  if (mpz_set_str(value, "9007199254740995", 10))
    return 4;
  char decimal[64];
  mpz_get_str(decimal, 10, value);
  if (std::strcmp(decimal, "9007199254740995"))
    return 5;
  unsigned char bytes[32]{};
  std::size_t count = 0;
  mpz_export(bytes, &count, -1, 1, 0, 0, value);
  if (!count || count > sizeof(bytes))
    return 6;
  mpz_import(quotient, count, -1, 1, 0, 0, bytes);
  if (mpz_cmp(value, quotient))
    return 7;
  signed long exponent = 0;
  const auto fraction = mpz_get_d_2exp(&exponent, value);
  if (!(fraction >= 0.5 && fraction < 1.0) || exponent != 54)
    return 8;
  std::printf("%s %d %zu\n", gmp_version, GMP_LIMB_BITS,
              sizeof(void *) * CHAR_BIT);
  mpz_clears(value, divisor, quotient, remainder, nullptr);
}
