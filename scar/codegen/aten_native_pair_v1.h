#pragma once

#include <ATen/Parallel.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/InferenceMode.h>

#include <cfenv>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace scar_aten_graph {

// A synchronous two-lane executor for a fresh standalone LibTorch process.
// The caller owns the executor, keeps its ATen context fixed, and submits
// independent operations that can access the same inputs read-only.
class CpuPair {
public:
    CpuPair(int caller_cpu, int worker_cpu, bool pin_caller_for_lifetime = true,
            bool allow_ambient_grad = false)
        : caller_thread_(pthread_self()), caller_cpu_(caller_cpu),
          pin_caller_for_lifetime_(pin_caller_for_lifetime),
          prepared_grad_enabled_(at::GradMode::is_enabled()) {
        if (prepared_grad_enabled_ && !allow_ambient_grad) {
            throw std::runtime_error("native pair requires no grad by default");
        }
        check_at_context();
        if (caller_cpu < 0 || caller_cpu >= CPU_SETSIZE
                || worker_cpu < 0 || worker_cpu >= CPU_SETSIZE || caller_cpu == worker_cpu) {
            throw std::invalid_argument("two distinct CPU IDs in the native affinity range are required");
        }
        const int error = pthread_getaffinity_np(caller_thread_, sizeof(original_mask_), &original_mask_);
        if (error != 0) throw std::system_error(error, std::generic_category(), "read caller affinity");
        cpu_set_t launch_mask;
        CPU_ZERO(&launch_mask);
        CPU_SET(caller_cpu, &launch_mask);
        CPU_SET(worker_cpu, &launch_mask);
        if (!CPU_EQUAL(&original_mask_, &launch_mask)) {
            throw std::runtime_error("native pair launch affinity must contain exactly its two planned CPUs");
        }
        if (std::fegetenv(&original_fp_) != 0) throw std::runtime_error("read caller floating-point environment");
        try {
            if (pin_caller_for_lifetime_) pin(caller_cpu);
            worker_ = std::thread([this, worker_cpu] { serve(worker_cpu); });
            std::unique_lock lock(mutex_);
            complete_.wait(lock, [this] { return ready_; });
            if (terminal_error_) std::rethrow_exception(terminal_error_);
        } catch (...) {
            stop();
            if (pin_caller_for_lifetime_) restore();
            throw;
        }
    }

    CpuPair(const CpuPair&) = delete;
    CpuPair& operator=(const CpuPair&) = delete;

    ~CpuPair() {
        stop();
        if (pin_caller_for_lifetime_) restore();
    }

    class CallerPlacement {
    public:
        CallerPlacement(const CallerPlacement&) = delete;
        CallerPlacement& operator=(const CallerPlacement&) = delete;

        ~CallerPlacement() {
            if (!pair_.pin_caller_for_lifetime_) pair_.restore();
        }

    private:
        friend class CpuPair;

        explicit CallerPlacement(CpuPair& pair) : pair_(pair) {
            pair.check_caller_thread();
            pair.check_at_context();
            if (pair.pin_caller_for_lifetime_) {
                check_cpu(pair.caller_cpu_);
                return;
            }
            cpu_set_t admitted;
            const int error = pthread_getaffinity_np(pair.caller_thread_, sizeof(admitted), &admitted);
            if (error != 0) throw std::system_error(error, std::generic_category(), "read caller affinity");
            if (!CPU_EQUAL(&admitted, &pair.original_mask_)) {
                throw std::runtime_error("caller affinity has changed since native pair preparation");
            }
            try {
                pin(pair.caller_cpu_);
            } catch (...) {
                pair.restore();
                throw;
            }
        }

        CpuPair& pair_;
    };

    CallerPlacement place_caller() { return CallerPlacement(*this); }

    bool prepared_grad_enabled() const noexcept { return prepared_grad_enabled_; }

    template <class L, class R>
    void run(L&& caller, R&& worker) {
        check_caller_thread();
        check_grad_mode();
        check_cpu(caller_cpu_);
        auto invoke_worker = [&worker] { std::invoke(std::forward<R>(worker)); };
        {
            std::lock_guard lock(mutex_);
            if (terminal_error_) std::rethrow_exception(terminal_error_);
            if (active_) throw std::logic_error("native pair cannot submit recursively");
            active_ = true;
            worker_error_ = nullptr;
            job_ = &invoke_worker;
            dispatch_ = [](void* closure) { (*static_cast<decltype(invoke_worker)*>(closure))(); };
        }
        pending_.notify_one();

        std::exception_ptr caller_error;
        try {
            std::invoke(std::forward<L>(caller));
            check_grad_mode();
            check_cpu(caller_cpu_);
        } catch (...) {
            caller_error = std::current_exception();
        }
        std::unique_lock lock(mutex_);
        complete_.wait(lock, [this] { return job_ == nullptr; });
        active_ = false;
        if (caller_error) std::rethrow_exception(caller_error);
        if (terminal_error_) std::rethrow_exception(terminal_error_);
        if (worker_error_) std::rethrow_exception(worker_error_);
    }

private:
    void check_caller_thread() const {
        if (!pthread_equal(pthread_self(), caller_thread_)) {
            throw std::logic_error("native pair must run on its owning caller thread");
        }
    }

    void check_grad_mode() const {
        if (at::GradMode::is_enabled() != prepared_grad_enabled_) {
            throw std::runtime_error("native pair grad mode differs from its prepared context");
        }
    }

    void check_at_context() const {
        check_grad_mode();
        if (at::get_num_threads() != 1 || at::get_num_interop_threads() != 1
                || c10::InferenceMode::is_enabled() || std::fegetround() != FE_TONEAREST) {
            throw std::runtime_error("native pair requires single-threaded ordinary ATen and round-to-nearest");
        }
    }

    static void check_cpu(int cpu) {
        if (sched_getcpu() != cpu) throw std::runtime_error("native pair lane is outside its planned CPU");
    }

    static void pin(int cpu) {
        cpu_set_t singleton;
        CPU_ZERO(&singleton);
        CPU_SET(cpu, &singleton);
        const int error = pthread_setaffinity_np(pthread_self(), sizeof(singleton), &singleton);
        if (error != 0) throw std::system_error(error, std::generic_category(), "pin native pair lane");
        check_cpu(cpu);
    }

    void serve(int cpu) {
        try {
            pin(cpu);
            at::init_num_threads();
            const at::ThreadLocalStateGuard torch_context(caller_state_);
            if (std::fesetenv(&original_fp_) != 0) {
                throw std::runtime_error("apply caller floating-point environment");
            }
            check_at_context();

            std::unique_lock lock(mutex_);
            ready_ = true;
            complete_.notify_one();
            for (;;) {
                pending_.wait(lock, [this] { return stopping_ || job_ != nullptr; });
                if (stopping_) return;
                const auto dispatch = dispatch_;
                auto* job = job_;
                lock.unlock();
                std::exception_ptr failure;
                try {
                    check_grad_mode();
                    check_cpu(cpu);
                    dispatch(job);
                    check_grad_mode();
                    check_cpu(cpu);
                } catch (...) {
                    failure = std::current_exception();
                }
                lock.lock();
                worker_error_ = std::move(failure);
                job_ = nullptr;
                complete_.notify_one();
            }
        } catch (...) {
            std::lock_guard lock(mutex_);
            terminal_error_ = std::current_exception();
            ready_ = true;
            job_ = nullptr;
            complete_.notify_one();
        }
    }

    void stop() {
        if (!worker_.joinable()) return;
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        pending_.notify_one();
        worker_.join();
    }

    void restore() noexcept {
        pthread_setaffinity_np(caller_thread_, sizeof(original_mask_), &original_mask_);
    }

    pthread_t caller_thread_;
    int caller_cpu_;
    bool pin_caller_for_lifetime_;
    const bool prepared_grad_enabled_;
    cpu_set_t original_mask_;
    std::fenv_t original_fp_;
    at::ThreadLocalState caller_state_;
    std::mutex mutex_;
    std::condition_variable pending_, complete_;
    bool ready_ = false;
    bool stopping_ = false;
    bool active_ = false;
    void* job_ = nullptr;
    void (*dispatch_)(void*) = nullptr;
    std::exception_ptr terminal_error_, worker_error_;
    std::thread worker_;
};

}  // namespace scar_aten_graph
