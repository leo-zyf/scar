#include <climits>
#include <cstdio>
#include <dlfcn.h>
#include <gnu/libc-version.h>
#include <stdlib.h>

int main() {
  static_assert(CHAR_BIT == 8 && sizeof(int) == 4);
  int (*const entry)(const char *) = &::system;
  Dl_info symbol{}, libc{};
  if (!dladdr(reinterpret_cast<void *>(entry), &symbol) ||
      !dladdr(reinterpret_cast<void *>(&gnu_get_libc_version), &libc) ||
      symbol.dli_fbase != libc.dli_fbase || !symbol.dli_fname)
    return 1;
  std::printf("%s\n%s\n%zu\n", symbol.dli_fname, gnu_get_libc_version(),
              CHAR_BIT * sizeof(void *));
}
