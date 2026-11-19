#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>

template <std::size_t Count>
class ScarTensorBodyProbe {
public:
    explicit ScarTensorBodyProbe(bool enabled) : start_(enabled ? Clock::now() : Clock::time_point{}) {}

    void setup_complete() { setup_ = Clock::now(); }
    void input_ready() { input_ready_ = Clock::now(); }
    void computed() { computed_ = Clock::now(); }

    auto operator()(std::size_t index, auto&& run) {
        const auto began = Clock::now();
        auto result = std::forward<decltype(run)>(run)();
        operations_[index] += nanos(began, Clock::now());
        return result;
    }

    void write() const {
        const auto published = Clock::now();
        std::cerr << "{\"setup_ns\":" << nanos(start_, setup_)
                  << ",\"read_ns\":" << nanos(setup_, input_ready_)
                  << ",\"loop_ns\":" << nanos(input_ready_, computed_)
                  << ",\"publish_ns\":" << nanos(computed_, published)
                  << ",\"operation_totals_ns\":[";
        for (std::size_t index = 0; index < Count; ++index) {
            if (index) std::cerr << ',';
            std::cerr << operations_[index];
        }
        std::cerr << "]}\n";
    }

private:
    using Clock = std::chrono::steady_clock;

    static std::int64_t nanos(Clock::time_point before, Clock::time_point after) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
    }

    Clock::time_point start_, setup_, input_ready_, computed_;
    std::array<std::int64_t, Count> operations_{};
};
