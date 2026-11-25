#pragma once

#include "array.hpp"
#include <cmath>
#include <complex>
#include <optional>
// A single submitted transform cannot create an additional worker team.
#define POCKETFFT_NO_MULTITHREADING
#include <pocketfft_hdronly.h>
#undef POCKETFFT_NO_MULTITHREADING

namespace scar {

struct fourier_request {
  ndarray<std::complex<double>> input;
  bool forward;
  std::optional<array_warning> warning;
};

inline ndarray<std::complex<double>> array_fft(const fourier_request &request) {
  static_assert(sizeof(std::complex<double>) == 16 &&
                std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);
  const auto size = request.input.size(1);
  if (!size)
    throw std::invalid_argument(
        "Invalid number of FFT data points (0) specified.");
  std::optional<array_fenv> environment;
  if (request.warning)
    environment.emplace();
  const double scale = 1.0 / std::sqrt(static_cast<double>(size));
  auto transformed = request.input.transform_vector(
      [&](const std::complex<double> *input, std::ptrdiff_t stride,
          std::span<std::complex<double>> output) {
        pocketfft::c2c<double>(
            {output.size()}, {stride},
            {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))}, {0},
            request.forward, input, output.data(), scale, 1);
      });
  if (environment)
    report_array_warnings(environment->finish(), *request.warning);
  return transformed;
}

} // namespace scar
