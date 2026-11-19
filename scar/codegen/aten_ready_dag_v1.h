#pragma once

#include "aten_native_team_v1.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace scar_aten_graph {

// Each lane follows source order. Cross-lane Tensor edges alone publish and
// wait on completion. A release/acquire pair protects the producer's handle.
template <std::size_t Count, std::size_t Edges, class Invoke>
void execute_ready_dag(CpuTeam& team, const std::array<std::uint8_t, Count>& lanes,
                       const std::array<std::size_t, Count + 1>& offsets,
                       const std::array<std::size_t, Edges>& predecessors, Invoke&& invoke) {
    enum Completion : std::uint8_t { pending, complete, cancelled };

    std::array<bool, 3> used{};
    for (const auto lane : lanes) {
        if (lane >= used.size()) throw std::invalid_argument("invalid native DAG lane");
        used[lane] = true;
    }

    std::array<bool, Count> has_remote_reader{};
    for (std::size_t index = 0; index < Count; ++index) {
        for (auto edge = offsets[index]; edge < offsets[index + 1]; ++edge) {
            const auto parent = predecessors[edge];
            if (lanes[parent] != lanes[index]) has_remote_reader[parent] = true;
        }
    }

    std::array<std::atomic<Completion>, Count> state{};
    std::atomic<bool> stopping = false;
    const auto publish = [&](std::size_t index, Completion outcome) noexcept {
        Completion expected = pending;
        if (state[index].compare_exchange_strong(expected, outcome, std::memory_order_release,
                                                std::memory_order_relaxed)) {
            state[index].notify_all();
        }
    };
    const auto abort = [&]() noexcept {
        if (stopping.exchange(true, std::memory_order_acq_rel)) return;
        for (std::size_t index = 0; index < Count; ++index) {
            if (has_remote_reader[index]) publish(index, cancelled);
        }
    };
    const auto run_lane = [&](std::size_t lane) {
        for (std::size_t index = 0; index < Count; ++index) {
            if (lanes[index] != lane) continue;
            for (auto edge = offsets[index]; edge < offsets[index + 1]; ++edge) {
                const auto parent = predecessors[edge];
                if (lanes[parent] == lane) continue;
                state[parent].wait(pending, std::memory_order_acquire);
                if (state[parent].load(std::memory_order_acquire) == cancelled) return;
            }
            if (stopping.load(std::memory_order_acquire)) return;
            invoke(index);
            if (has_remote_reader[index]) publish(index, complete);
        }
    };

    team.run_lanes_with_abort(used, run_lane, abort);
}

}  // namespace scar_aten_graph
