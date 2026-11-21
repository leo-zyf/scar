#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sched.h>
#include <time.h>

namespace {
std::uint64_t monotonic_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return std::uint64_t(value.tv_sec) * 1'000'000'000 + std::uint64_t(value.tv_nsec);
}
}  // namespace

int main(int argc, char* argv[]) {
    const auto entered = monotonic_ns();
    const bool paced = argc == 2 && std::strcmp(argv[1], "--paced-stdin") == 0;
    cpu_set_t affinity{};
    if (entered == 0 || (argc != 1 && !paced) || sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        return 2;
    }
    if (paced) {
        char command[32];
        if (!std::fgets(command, sizeof(command), stdin) || std::strcmp(command, "SCAR_PROBE_EXIT\n") != 0
                || std::puts("SCAR_CPU_LAUNCH_ACK_V1") == EOF || std::fflush(stdout) != 0) {
            return 3;
        }
    }
    const auto leaving = monotonic_ns();
    if (leaving == 0 || std::printf("SCAR_CPU_LAUNCH_CLOCK_V1 %" PRIu64 " %" PRIu64, entered, leaving) < 0) {
        return 4;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &affinity) && std::printf(" %d", cpu) < 0) {
            return 4;
        }
    }
    return std::putchar('\n') == EOF ? 4 : 0;
}
