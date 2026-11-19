#pragma once

#include "cuda_copy_schedule_v1.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

namespace scar::cuda_hybrid_v1::activity {

using Clock = std::chrono::steady_clock;

inline std::int64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

struct Report {
    struct Service {
        bool host;
        std::int64_t ns;
    };

    std::int64_t allocation_ns = 0;
    std::int64_t input_ns = 0;
    std::int64_t setup_ns = 0;
    std::int64_t host_wait_ns = 0;
    std::int64_t join_ns = 0;
    std::int64_t publish_ns = 0;
    std::vector<Service> services;

    void write(std::ostream& out) const {
        out << "{\"schema\":\"scar.hybrid-native-activity-observation\",\"schema_version\":1,\"activities\":[";
        for (std::size_t index = 0; index < services.size(); ++index) {
            if (index) out << ',';
            const auto& service = services[index];
            out << "{\"step_index\":" << index << ",\"clock\":\""
                << (service.host ? "HOST_STEADY" : "CUDA_EVENT") << "\",\"service_ns\":" << service.ns << '}';
        }
        out << "],\"boundary\":{\"allocation_ns\":" << allocation_ns << ",\"input_ns\":" << input_ns
            << ",\"setup_ns\":" << setup_ns << ",\"host_wait_ns\":" << host_wait_ns << ",\"join_ns\":" << join_ns
            << ",\"publish_ns\":" << publish_ns << "}}\n";
    }
};

struct EventInterval {
    EventInterval() = default;
    EventInterval(const EventInterval&) = delete;
    EventInterval& operator=(const EventInterval&) = delete;

    ~EventInterval() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
};

struct NoTiming {
    template <typename Step>
    NoTiming(std::span<const Step>, Report*) {}

    static constexpr Clock::time_point now() { return {}; }
    void setup_done(Clock::time_point) const {}
    void wait_done(Clock::time_point) const {}
    void host_done(std::size_t, Clock::time_point) const {}
    void begin_device(std::size_t, cudaStream_t) const {}
    void end_device(std::size_t, cudaStream_t) const {}
    void join_done(Clock::time_point) const {}
};

class Timing {
public:
    template <typename Step>
    Timing(std::span<const Step> steps, Report* report) : report_(*report), events_(steps.size()) {
        using cuda_copy_v1::detail::check;
        report_.services.reserve(steps.size());
        for (std::size_t index = 0; index < steps.size(); ++index) {
            const bool host = steps[index].host();
            report_.services.push_back({host, 0});
            if (!host) {
                check(cudaEventCreate(&events_[index].begin), "cudaEventCreate (service start)");
                check(cudaEventCreate(&events_[index].end), "cudaEventCreate (service end)");
            }
        }
    }

    static Clock::time_point now() { return Clock::now(); }
    void setup_done(Clock::time_point start) const { report_.setup_ns = elapsed(start); }
    void wait_done(Clock::time_point start) const { report_.host_wait_ns += elapsed(start); }
    void host_done(std::size_t index, Clock::time_point start) const { report_.services[index].ns = elapsed(start); }

    void begin_device(std::size_t index, cudaStream_t stream) const {
        cuda_copy_v1::detail::check(cudaEventRecord(events_[index].begin, stream), "cudaEventRecord (service start)");
    }

    void end_device(std::size_t index, cudaStream_t stream) const {
        cuda_copy_v1::detail::check(cudaEventRecord(events_[index].end, stream), "cudaEventRecord (service end)");
    }

    void join_done(Clock::time_point start) const {
        report_.join_ns = elapsed(start);
        for (std::size_t index = 0; index < events_.size(); ++index) {
            if (report_.services[index].host) continue;
            float ms = 0;
            cuda_copy_v1::detail::check(cudaEventElapsedTime(&ms, events_[index].begin, events_[index].end),
                                        "cudaEventElapsedTime (service interval)");
            const auto ns = std::round(static_cast<double>(ms) * 1'000'000);
            if (!std::isfinite(ns) || ns < 0 || ns >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("invalid CUDA activity duration");
            }
            report_.services[index].ns = static_cast<std::int64_t>(ns);
        }
    }

private:
    Report& report_;
    std::vector<EventInterval> events_;
};

}  // namespace scar::cuda_hybrid_v1::activity
