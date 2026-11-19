#pragma once

#include <ATen/Parallel.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/InferenceMode.h>

#include <array>
#include <cfenv>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

namespace scar_aten_graph {

// A synchronous caller and two persistent worker lanes. Every submission joins
// the participating workers before returning or propagating an exception.
class CpuTeam {
public:
    CpuTeam(std::array<int, 3> cpus, bool pin_caller_for_lifetime = true, bool allow_ambient_grad = false)
        : caller_thread_(pthread_self()), cpus_(cpus),
          pin_caller_for_lifetime_(pin_caller_for_lifetime),
          prepared_grad_enabled_(at::GradMode::is_enabled()) {
        if (prepared_grad_enabled_ && !allow_ambient_grad) {
            throw std::runtime_error("native team requires no grad by default");
        }
        check_at_context();
        cpu_set_t planned;
        CPU_ZERO(&planned);
        for (const int cpu : cpus_) {
            if (cpu < 0 || cpu >= CPU_SETSIZE || CPU_ISSET(cpu, &planned)) {
                throw std::invalid_argument("native team requires three distinct Linux CPU IDs");
            }
            CPU_SET(cpu, &planned);
        }
        original_mask_ = current_mask();
        if (!CPU_EQUAL(&original_mask_, &planned)) {
            throw std::runtime_error("native team launch affinity must contain exactly its planned CPUs");
        }
        if (std::fegetenv(&original_fp_) != 0) throw std::runtime_error("read caller floating-point environment");
        try {
            if (pin_caller_for_lifetime_) pin(cpus_[0]);
            for (std::size_t worker = 0; worker < workers_.size(); ++worker) {
                workers_[worker].thread = std::thread([this, worker] { serve(workers_[worker], cpus_[worker + 1]); });
            }
            std::unique_lock lock(mutex_);
            complete_.wait(lock, [this] { return workers_[0].ready && workers_[1].ready; });
            for (const auto& worker : workers_) {
                if (worker.terminal_error) std::rethrow_exception(worker.terminal_error);
            }
        } catch (...) {
            stop();
            if (pin_caller_for_lifetime_) restore();
            throw;
        }
    }

    CpuTeam(const CpuTeam&) = delete;
    CpuTeam& operator=(const CpuTeam&) = delete;

    ~CpuTeam() {
        stop();
        if (pin_caller_for_lifetime_) restore();
    }

    class CallerPlacement {
    public:
        CallerPlacement(const CallerPlacement&) = delete;
        CallerPlacement& operator=(const CallerPlacement&) = delete;

        ~CallerPlacement() {
            if (!team_.pin_caller_for_lifetime_) team_.restore();
        }

    private:
        friend class CpuTeam;

        explicit CallerPlacement(CpuTeam& team) : team_(team) {
            team.check_caller_thread();
            team.check_at_context();
            if (team.pin_caller_for_lifetime_) {
                check_cpu(team.cpus_[0]);
                return;
            }
            const auto admitted = current_mask();
            if (!CPU_EQUAL(&admitted, &team.original_mask_)) {
                throw std::runtime_error("caller affinity has changed since native team preparation");
            }
            try {
                pin(team.cpus_[0]);
            } catch (...) {
                team.restore();
                throw;
            }
        }

        CpuTeam& team_;
    };

    CallerPlacement place_caller() { return CallerPlacement(*this); }
    bool prepared_grad_enabled() const noexcept { return prepared_grad_enabled_; }

    template <class L, class R>
    void run(L&& caller, R&& worker) {
        auto first = [&worker] { std::invoke(std::forward<R>(worker)); };
        submit(std::forward<L>(caller), {make_job(first), Job{}});
    }

    template <class L, class R, class H>
    void run(L&& caller, R&& worker, H&& other_worker) {
        auto first = [&worker] { std::invoke(std::forward<R>(worker)); };
        auto second = [&other_worker] { std::invoke(std::forward<H>(other_worker)); };
        submit(std::forward<L>(caller), {make_job(first), make_job(second)});
    }

    template <class F>
    void run_on_worker(std::size_t lane, F&& function) {
        if (lane < 1 || lane > workers_.size()) throw std::invalid_argument("native team worker lane must be 1 or 2");
        auto worker = [&function] { std::invoke(std::forward<F>(function)); };
        std::array<Job, 2> jobs{};
        jobs[lane - 1] = make_job(worker);
        submit([] {}, jobs);
    }

    // A lane can be waiting for data from another lane. Signal failure even
    // when a worker's context check throws before it can enter the callback.
    template <class Lane, class Abort>
    void run_lanes_with_abort(const std::array<bool, 3>& used, Lane&& run_lane, Abort&& abort) {
        static_assert(std::is_nothrow_invocable_v<Abort&>);
        const auto task = [&run_lane, &abort](std::size_t lane) {
            return [&run_lane, &abort, lane](bool failed) {
                if (failed) std::invoke(abort);
                else std::invoke(run_lane, lane);
            };
        };
        auto first = task(1);
        auto second = task(2);
        submit([&] {
            if (!used[0]) return;
            try {
                std::invoke(run_lane, std::size_t{0});
            } catch (...) {
                std::invoke(abort);
                throw;
            }
        }, {used[1] ? make_interruptible_job(first) : Job{},
            used[2] ? make_interruptible_job(second) : Job{}});
    }

private:
    struct Job {
        void* closure = nullptr;
        void (*invoke)(void*) = nullptr;
        void (*on_failure)(void*) noexcept = nullptr;

        explicit operator bool() const noexcept { return invoke != nullptr; }
        void operator()() const { invoke(closure); }
        void fail() const noexcept { if (on_failure) on_failure(closure); }
    };

    struct Worker {
        Job job;
        bool ready = false;
        std::exception_ptr task_error, terminal_error;
        std::condition_variable pending;
        std::thread thread;
    };

    template <class F>
    static Job make_job(F& closure) {
        return {&closure, [](void* arg) { (*static_cast<F*>(arg))(); }};
    }

    template <class F>
    static Job make_interruptible_job(F& closure) {
        return {&closure, [](void* arg) { (*static_cast<F*>(arg))(false); },
                [](void* arg) noexcept { (*static_cast<F*>(arg))(true); }};
    }

    template <class F>
    void submit(F&& caller, const std::array<Job, 2>& jobs) {
        check_caller_thread();
        check_grad_mode();
        check_cpu(cpus_[0]);
        {
            std::lock_guard lock(mutex_);
            if (active_) throw std::logic_error("native team cannot submit recursively");
            for (const auto& worker : workers_) {
                if (worker.terminal_error) std::rethrow_exception(worker.terminal_error);
            }
            active_ = true;
            for (std::size_t i = 0; i < workers_.size(); ++i) {
                workers_[i].task_error = nullptr;
                workers_[i].job = jobs[i];
            }
        }
        for (std::size_t i = 0; i < workers_.size(); ++i) {
            if (jobs[i]) workers_[i].pending.notify_one();
        }

        std::exception_ptr caller_error;
        try {
            std::invoke(std::forward<F>(caller));
            check_grad_mode();
            check_cpu(cpus_[0]);
        } catch (...) {
            caller_error = std::current_exception();
        }
        std::unique_lock lock(mutex_);
        complete_.wait(lock, [this] { return !workers_[0].job && !workers_[1].job; });
        active_ = false;
        if (caller_error) std::rethrow_exception(caller_error);
        for (const auto& worker : workers_) {
            if (worker.terminal_error) std::rethrow_exception(worker.terminal_error);
            if (worker.task_error) std::rethrow_exception(worker.task_error);
        }
    }

    void check_caller_thread() const {
        if (!pthread_equal(pthread_self(), caller_thread_)) {
            throw std::logic_error("native team must run on its owning caller thread");
        }
    }

    void check_grad_mode() const {
        if (at::GradMode::is_enabled() != prepared_grad_enabled_) {
            throw std::runtime_error("native team grad mode differs from its prepared context");
        }
    }

    void check_at_context() const {
        check_grad_mode();
        if (at::get_num_threads() != 1 || at::get_num_interop_threads() != 1
                || c10::InferenceMode::is_enabled() || std::fegetround() != FE_TONEAREST) {
            throw std::runtime_error("native team requires single-threaded ordinary ATen and round-to-nearest");
        }
    }

    static cpu_set_t current_mask() {
        cpu_set_t mask;
        const int error = pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
        if (error != 0) throw std::system_error(error, std::generic_category(), "read thread affinity");
        return mask;
    }

    static void check_cpu(int cpu) {
        if (sched_getcpu() != cpu) throw std::runtime_error("native team lane is outside its planned CPU");
    }

    static void pin(int cpu) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        const int error = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        if (error != 0) throw std::system_error(error, std::generic_category(), "pin native team lane");
        check_cpu(cpu);
    }

    void serve(Worker& worker, int cpu) {
        try {
            pin(cpu);
            at::init_num_threads();
            const at::ThreadLocalStateGuard torch_context(caller_state_);
            if (std::fesetenv(&original_fp_) != 0) throw std::runtime_error("apply caller floating-point environment");
            check_at_context();

            std::unique_lock lock(mutex_);
            worker.ready = true;
            complete_.notify_one();
            for (;;) {
                worker.pending.wait(lock, [&] { return stopping_ || worker.job; });
                if (stopping_) return;
                const auto job = worker.job;
                lock.unlock();
                std::exception_ptr error;
                try {
                    check_grad_mode();
                    check_cpu(cpu);
                    job();
                    check_grad_mode();
                    check_cpu(cpu);
                } catch (...) {
                    error = std::current_exception();
                    job.fail();
                }
                lock.lock();
                worker.task_error = std::move(error);
                worker.job = {};
                complete_.notify_one();
            }
        } catch (...) {
            std::lock_guard lock(mutex_);
            worker.terminal_error = std::current_exception();
            worker.ready = true;
            worker.job.fail();
            worker.job = {};
            complete_.notify_one();
        }
    }

    void stop() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        for (auto& worker : workers_) worker.pending.notify_one();
        for (auto& worker : workers_) {
            if (worker.thread.joinable()) worker.thread.join();
        }
    }

    void restore() noexcept {
        pthread_setaffinity_np(caller_thread_, sizeof(original_mask_), &original_mask_);
    }

    pthread_t caller_thread_;
    std::array<int, 3> cpus_;
    bool pin_caller_for_lifetime_;
    const bool prepared_grad_enabled_;
    cpu_set_t original_mask_;
    std::fenv_t original_fp_;
    at::ThreadLocalState caller_state_;
    std::array<Worker, 2> workers_;
    std::mutex mutex_;
    std::condition_variable complete_;
    bool stopping_ = false;
    bool active_ = false;
};

}  // namespace scar_aten_graph
