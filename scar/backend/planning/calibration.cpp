#include <algorithm>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <sched.h>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int count = 1 << 18;
volatile std::uint64_t checksum;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::nano>(Clock::now() - start).count();
}

template <class T> void consume(const std::vector<T> &values) {
  using Bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
  std::uint64_t sum = 0;
  for (T value : values)
    sum += std::bit_cast<Bits>(value);
  checksum = sum;
}

template <class F> void each(int width, F operation) {
  if (width == 1) {
    for (int i = 0; i < count; ++i)
      operation(i);
  } else {
#pragma omp parallel for num_threads(width) schedule(static)
    for (int i = 0; i < count; ++i)
      operation(i);
  }
}

void emit(const std::string &metric, int width, int repetition,
          double duration) {
  if (repetition >= 0)
    std::printf("%d %d %s %.12g\n", width, repetition, metric.c_str(),
                duration);
}

void copy_payload(std::size_t bytes) {
  if (bytes == 0 || bytes % sizeof(std::uint64_t) != 0)
    std::exit(2);
  std::vector<std::uint64_t> source(bytes / sizeof(std::uint64_t));
  std::vector<std::uint64_t> destination(source.size());
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = i;
  // Keep the timed call in the target C library and consume its output after
  // it.
  using Copy = void *(*)(void *, const void *, std::size_t);
  Copy volatile copy = std::memcpy;
  for (int rep = -1; rep < 3; ++rep) {
    source.front() = static_cast<std::uint64_t>(rep + 1);
    auto start = Clock::now();
    copy(destination.data(), source.data(), bytes);
    double duration = elapsed(start);
    consume(destination);
    emit("memcpy_payload_byte", 1, rep, duration / static_cast<double>(bytes));
  }
}

void team(int width) {
  std::vector<int> marks(width);
  for (int rep = -1; rep < 3; ++rep) {
    auto start = Clock::now();
    for (int visit = 0; visit < 8; ++visit) {
      if (width == 1) {
        marks[0] = visit;
      } else {
        cpu_set_t allowed;
        if (omp_in_parallel() || width > omp_get_thread_limit() ||
            width > omp_get_max_threads() ||
            sched_getaffinity(0, sizeof(allowed), &allowed) != 0 ||
            width > CPU_COUNT(&allowed))
          std::exit(3);
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
        if (fegetexcept() != 0)
          std::exit(3);
#endif
        std::fenv_t initial;
        std::fegetenv(&initial);
        const int incoming = std::fetestexcept(FE_ALL_EXCEPT);
        int raised = 0;
#pragma omp parallel num_threads(width) reduction(| : raised)
        {
          std::fenv_t previous;
          std::fegetenv(&previous);
          std::fesetenv(&initial);
          marks[omp_get_thread_num()] = visit;
          raised |= std::fetestexcept(FE_ALL_EXCEPT);
          std::fesetenv(&previous);
        }
        if (raised & ~incoming)
          std::feraiseexcept(raised & ~incoming);
      }
      checksum = marks.back();
    }
    emit("team", width, rep, elapsed(start) / 8);
  }
}

template <class T>
void kernels(const char *name, int width, const std::set<std::string> &wanted) {
  std::string suffix = std::string("_") + name;
  bool needed = (std::is_same_v<T, double> && wanted.contains("byte"));
  for (const auto &key : wanted)
    needed |= key.ends_with(suffix);
  if (!needed)
    return;
  std::vector<T> left(count), right(count), out(count);
  for (int i = 0; i < count; ++i) {
    left[i] = T((i * 13 % 4093) + 67);
    right[i] = T((i * 31 % 127) + 2);
  }

  auto measure = [&](std::string key, double steps, auto operation) {
    if (!wanted.contains(key))
      return;
    for (int rep = -1; rep < 3; ++rep) {
      auto start = Clock::now();
      each(width, [&](int i) { out[i] = operation(left[i], right[i]); });
      double duration = elapsed(start);
      consume(out);
      emit(key, width, rep, duration / (steps * count));
    }
  };
  if constexpr (std::is_same_v<T, double>)
    measure("byte", 2 * sizeof(T), [](T a, T) { return a; });
  measure("arith" + suffix, 4, [](T a, T b) { return (a * b + a) * b + b; });
  measure("div" + suffix, 1, [](T a, T b) { return a / b; });
  measure("const_div" + suffix, 1, [](T a, T) { return a / T(7); });
  measure("shift_div" + suffix, 1, [](T a, T) { return a / T(8); });
  if constexpr (std::is_integral_v<T>) {
    measure("mod" + suffix, 1, [](T a, T b) { return a % b; });
    measure("divmod" + suffix, 1, [](T a, T b) { return a / b + a % b; });
    measure("const_mod" + suffix, 1, [](T a, T) { return a % T(7); });
    measure("shift_mod" + suffix, 1, [](T a, T) { return a % T(8); });
  }
}

void controls(const std::set<std::string> &wanted, std::int64_t period) {
  std::vector<std::int64_t> out(count);
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  if (wanted.contains("fp_guard")) {
    for (int rep = -1; rep < 3; ++rep) {
      auto start = Clock::now();
      for (int i = 0; i < count; ++i)
        out[i] = fegetexcept() == 0;
      const double duration = elapsed(start);
      consume(out);
      emit("fp_guard", 1, rep, duration / count);
    }
  }
#endif
  if (wanted.contains("guard_i64")) {
    std::vector<std::int64_t> sizes(count);
    for (int i = 0; i < count; ++i)
      sizes[i] = i % 127 - 4;
    for (int rep = -1; rep < 3; ++rep) {
      auto start = Clock::now();
      for (int i = 0; i < count; ++i)
        out[i] = sizes[i] > period;
      double duration = elapsed(start);
      consume(out);
      emit("guard_i64", 1, rep, duration / count);
    }
  }
  if (wanted.contains("carry_i64")) {
    for (int rep = -1; rep < 3; ++rep) {
      std::int64_t coordinate = 0;
      auto start = Clock::now();
      for (int i = 0; i < count; ++i) {
        out[i] = coordinate;
        if (++coordinate == period)
          coordinate = 0;
      }
      double duration = elapsed(start);
      consume(out);
      emit("carry_i64", 1, rep, duration / count);
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  int split = 2;
  while (split < argc && std::string(argv[split]) != "--")
    ++split;
  if (split == 2 || split + 1 >= argc)
    return 2;
  const std::set<std::string> wanted(argv + split + 1, argv + argc);
  omp_set_dynamic(0);
  for (int j = 2; j < split; ++j) {
    const int width = std::atoi(argv[j]);
    if (width < 1)
      return 2;
    if (wanted.contains("team")) {
      int actual = 0;
#pragma omp parallel num_threads(width)
      {
#pragma omp single
        actual = omp_get_num_threads();
      }
      if (actual != width)
        return 3;
      team(width);
    }
    kernels<std::int32_t>("i32", width, wanted);
    kernels<std::uint32_t>("u32", width, wanted);
    kernels<std::int64_t>("i64", width, wanted);
    kernels<std::uint64_t>("u64", width, wanted);
    kernels<float>("f32", width, wanted);
    kernels<double>("f64", width, wanted);
    if (width == 1) {
      if (wanted.contains("fp_guard") || wanted.contains("guard_i64") ||
          wanted.contains("carry_i64"))
        controls(wanted, 2 * (split - 1) + 13);
      if (wanted.contains("memcpy_payload_byte"))
        copy_payload(std::stoull(argv[1]));
    }
    std::fflush(stdout);
  }
}
