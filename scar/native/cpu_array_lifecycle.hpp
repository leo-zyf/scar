#pragma once

#include <array>
#include <cstdint>
#include <time.h>
#include <unistd.h>

namespace scar::entry_lifecycle {
enum class Marker { MAIN_ENTER, DISPATCH_BEGIN, DISPATCH_RETURN, PUBLICATION_BEGIN, PUBLICATION_END };

class Witness {
public:
    Witness() { mark(Marker::MAIN_ENTER); }

    void mark(Marker phase) noexcept {
        timespec now{};
        if (clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
            stamps_[static_cast<unsigned>(phase)] =
                std::uint64_t(now.tv_sec) * 1'000'000'000 + std::uint64_t(now.tv_nsec);
        }
    }

    ~Witness() {
        if (stamps_.back()) {
            const auto written = write(198, stamps_.data(), stamps_.size() * sizeof(stamps_[0]));
            (void)written;  // The sampler independently requires one complete record.
        }
    }

private:
    std::array<std::uint64_t, 5> stamps_{};
};
}  // namespace scar::entry_lifecycle
