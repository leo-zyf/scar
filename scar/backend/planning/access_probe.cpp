#include <type_traits>

// Compile only: the parameter syntax used by the target printer.
void scar_access_parameter(std::add_pointer_t<double> __restrict__ output,
                           std::add_pointer_t<const double> left,
                           const double *right) {
  output[0] = left[0] + right[0];
}
