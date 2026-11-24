#include <cstddef>
#include <cstdint>
#include <cstdio>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <mpi.h>
#include <type_traits>

template <class T> using Ptr = T *;
template <class T> using Const = const T;
template <class T> using Volatile = volatile T;
template <class R, class... Args> using Function = R(Args...);
template <class R, class... Args> using Variadic = R(Args..., ...);

int main() {
  // The generated list contains only checked identifiers and neutral ABI types.
  // Volatile reads keep every symbol reference live without calling it.
#define SCAR_FUNCTION(symbol, ...)                                             \
  {                                                                            \
    static_assert(std::is_same_v<decltype(symbol), __VA_ARGS__>);              \
    auto *volatile reference = &symbol;                                        \
    if (reference == nullptr)                                                  \
      return 1;                                                                \
  }
#define SCAR_TYPE(symbol) static_assert(std::is_object_v<symbol>);
#define SCAR_GLOBAL(symbol, ...)                                               \
  {                                                                            \
    static_assert(std::is_same_v<decltype(symbol), __VA_ARGS__>);              \
    auto *volatile reference = &symbol;                                        \
    if (reference == nullptr)                                                  \
      return 1;                                                                \
  }
#include "bindings.inc"

  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  int length = 0;
  if (MPI_Get_library_version(version, &length) != MPI_SUCCESS || length <= 0 ||
      length > MPI_MAX_LIBRARY_VERSION_STRING)
    return 2;
  const char *reporting_library = "";
#if defined(__linux__)
  Dl_info location{};
  if (dladdr(reinterpret_cast<void *>(&MPI_Get_library_version), &location) ==
      0)
    return 3;
  reporting_library = location.dli_fname;
#endif
  std::printf("%s\n%d.%d\n", reporting_library, MPI_VERSION, MPI_SUBVERSION);
  return std::fwrite(version, 1, static_cast<std::size_t>(length), stdout) ==
                 static_cast<std::size_t>(length)
             ? 0
             : 3;
}
