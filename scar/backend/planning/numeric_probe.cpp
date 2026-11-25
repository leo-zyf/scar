#include <cmath>

extern "C" {
double scar_strict_before(double x) { return std::sqrt(x); }

[[gnu::optimize("no-math-errno")]] double scar_local_errno(double x) {
  return std::sqrt(x);
}

[[gnu::optimize("no-rounding-math")]] double scar_local_rounding() {
  return 1.0 + 0x1p-53;
}

[[gnu::optimize("no-math-errno", "no-rounding-math")]] double
scar_local_both(double x) {
  return std::sqrt(x);
}

double scar_strict_after(double x) { return std::sqrt(x); }
double scar_strict_rounding() { return 1.0 + 0x1p-53; }
}
