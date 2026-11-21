#include <charconv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sys/resource.h>
#include <unistd.h>

namespace {
template <typename Integer>
bool parse_integer(const char* text, Integer& value) {
    const auto end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

int refuse() {
    constexpr char message[] = "SCAR candidate address-space guard could not apply or exec the bound command\n";
    const auto written = write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)written;
    return 125;
}

bool permits(rlim_t inherited, rlim_t requested) {
    return inherited == RLIM_INFINITY || inherited >= requested;
}
}  // namespace

int main(int argc, char* argv[]) {
    std::uint64_t requested = 0;
    int receipt = 0;
    if (argc < 5 || std::strcmp(argv[1], "--exec-address-space-limit") != 0 || argv[4][0] != '/'
            || !parse_integer(argv[2], requested) || requested == 0 || requested >= (std::uint64_t{1} << 63)
            || requested > std::numeric_limits<rlim_t>::max()
            || !parse_integer(argv[3], receipt) || receipt <= STDERR_FILENO) {
        return refuse();
    }

    const auto limit = static_cast<rlim_t>(requested);
    rlimit inherited{};
    if (limit == RLIM_INFINITY || getrlimit(RLIMIT_AS, &inherited) != 0
            || !permits(inherited.rlim_cur, limit) || !permits(inherited.rlim_max, limit)) {
        return refuse();
    }

    const rlimit bound{limit, limit};
    rlimit actual{};
    if (setrlimit(RLIMIT_AS, &bound) != 0 || getrlimit(RLIMIT_AS, &actual) != 0
            || actual.rlim_cur != limit || actual.rlim_max != limit) {
        return refuse();
    }

    // Two 64-bit decimal values and two separators fit in the stack buffer.
    char marker[64];
    const auto soft = std::to_chars(marker, marker + sizeof(marker) - 2, actual.rlim_cur);
    if (soft.ec != std::errc{}) {
        return refuse();
    }
    *soft.ptr = ' ';
    const auto hard = std::to_chars(soft.ptr + 1, marker + sizeof(marker) - 1, actual.rlim_max);
    if (hard.ec != std::errc{}) {
        return refuse();
    }
    auto end = hard.ptr;
    *end++ = '\n';
    if (write(receipt, marker, end - marker) != end - marker || close(receipt) != 0) {
        return refuse();
    }

    execv(argv[4], argv + 4);
    return refuse();
}
