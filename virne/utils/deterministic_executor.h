#ifndef VIRNE_UTILS_DETERMINISTIC_EXECUTOR_H
#define VIRNE_UTILS_DETERMINISTIC_EXECUTOR_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || \
    defined(__i386__)
#include <immintrin.h>
#endif

namespace virne::utils
{
namespace detail
{

inline std::size_t& deterministic_executor_depth() noexcept
{
    thread_local std::size_t depth = 0U;
    return depth;
}

class DeterministicExecutorScope
{
public:
    DeterministicExecutorScope() noexcept
    {
        ++deterministic_executor_depth();
    }

    ~DeterministicExecutorScope()
    {
        --deterministic_executor_depth();
    }

    DeterministicExecutorScope(const DeterministicExecutorScope&) = delete;
    DeterministicExecutorScope& operator=(
        const DeterministicExecutorScope&) = delete;
};

// One process-wide executor services the short deterministic batches used by
// the native port.  Calls are serialized at the executor boundary so a task
// can be represented by a single index callback while callers
// outside the parallel section remain fully independent.  Workers are grown
// only to a caller-requested width and live until process shutdown.
class DeterministicExecutor
{
public:
    DeterministicExecutor() = default;
    DeterministicExecutor(const DeterministicExecutor&) = delete;
    DeterministicExecutor& operator=(const DeterministicExecutor&) = delete;

    ~DeterministicExecutor()
    {
        {
            const std::lock_guard<std::mutex> lock(ready_mutex_);
            stopping_.store(true, std::memory_order_release);
        }
        ready_.notify_all();
        for (std::thread& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    template <typename Function>
    void run(std::size_t worker_count, Function&& function)
    {
        if (worker_count <= 1U)
        {
            function(0U);
            return;
        }

        const std::lock_guard<std::mutex> execution_lock(execution_mutex_);
        // Grow the idle pool before publishing stack-backed task state.  If
        // std::thread construction fails, no worker can observe this batch
        // and the exception propagates without a dangling TaskContext.
        ensure_workers(worker_count - 1U);

        if (failures_.size() < worker_count)
        {
            failures_.resize(worker_count);
        }
        std::fill_n(
            failures_.begin(), worker_count, std::exception_ptr{});

        using FunctionType = std::remove_reference_t<Function>;
        struct TaskContext
        {
            FunctionType* function = nullptr;
            std::vector<std::exception_ptr>* failures = nullptr;
        };
        TaskContext context{&function, &failures_};
        const TaskCallback callback =
            +[](void* raw_context, std::size_t worker_index) noexcept
            {
                auto& task = *static_cast<TaskContext*>(raw_context);
                const DeterministicExecutorScope executor_scope;
                try
                {
                    (*task.function)(worker_index);
                }
                catch (...)
                {
                    (*task.failures)[worker_index] =
                        std::current_exception();
                }
            };

        {
            // Publishing the callback and generation under the same mutex
            // used by parked workers prevents a lost wake.  The release
            // generation store also publishes them to workers that observe
            // the batch while briefly spinning between adjacent hot calls.
            const std::lock_guard<std::mutex> ready_lock(ready_mutex_);
            task_context_ = &context;
            task_callback_ = callback;
            active_background_workers_ = worker_count - 1U;
            remaining_background_workers_.store(
                worker_count - 1U, std::memory_order_relaxed);
            generation_.fetch_add(1U, std::memory_order_release);
        }
        ready_.notify_all();

        callback(&context, 0U);

        // Most controller batches finish within a few microseconds.  Give
        // background workers a bounded userspace completion window before
        // paying for a kernel condition-variable round trip.
        for (std::size_t spin = 0U;
             spin < short_spin_iterations &&
             remaining_background_workers_.load(
                 std::memory_order_acquire) != 0U;
             ++spin)
        {
            spin_pause();
        }
        if (remaining_background_workers_.load(
                std::memory_order_acquire) != 0U)
        {
            std::unique_lock<std::mutex> finished_lock(finished_mutex_);
            finished_.wait(finished_lock, [this]
            {
                return remaining_background_workers_.load(
                    std::memory_order_acquire) == 0U;
            });
        }

        // Deterministic worker-index order is also input-block order.
        for (std::size_t worker = 0U; worker < worker_count; ++worker)
        {
            const std::exception_ptr& failure = failures_[worker];
            if (failure)
            {
                std::rethrow_exception(failure);
            }
        }
    }

private:
    using TaskCallback = void (*)(void*, std::size_t) noexcept;
    static constexpr std::size_t short_spin_iterations = 256U;

    static void spin_pause() noexcept
    {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || \
    defined(__i386__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    void ensure_workers(const std::size_t count)
    {
        while (workers_.size() < count)
        {
            const std::size_t worker_index = workers_.size() + 1U;
            const std::size_t initial_generation = generation_.load(
                std::memory_order_acquire);
            workers_.emplace_back(
                [this, worker_index, initial_generation]
                {
                    worker_loop(worker_index, initial_generation);
                });
        }
    }

    void worker_loop(
        std::size_t worker_index,
        std::size_t seen_generation)
    {
        bool previous_batch_was_active = false;
        for (;;)
        {
            // Adjacent mapper/controller batches arrive quickly.  A short,
            // fixed spin keeps their workers off the kernel wait path without
            // consulting host topology or changing caller-selected width.
            // A worker excluded by a narrower subsequent width parks
            // immediately instead of competing with its active siblings.
            std::size_t observed_generation = generation_.load(
                std::memory_order_acquire);
            for (std::size_t spin = 0U;
                 previous_batch_was_active &&
                 observed_generation == seen_generation &&
                 spin < short_spin_iterations;
                 ++spin)
            {
                if (stopping_.load(std::memory_order_acquire))
                {
                    return;
                }
                observed_generation = generation_.load(
                    std::memory_order_acquire);
                if (observed_generation != seen_generation)
                {
                    break;
                }
                spin_pause();
            }

            TaskCallback callback = nullptr;
            void* context = nullptr;
            bool active = false;
            {
                std::unique_lock<std::mutex> ready_lock(ready_mutex_);
                if (generation_.load(std::memory_order_acquire) ==
                    seen_generation)
                {
                    ready_.wait(ready_lock, [this, &seen_generation]
                    {
                        return stopping_.load(std::memory_order_acquire) ||
                            generation_.load(std::memory_order_acquire) !=
                                seen_generation;
                    });
                }
                if (stopping_.load(std::memory_order_acquire))
                {
                    return;
                }
                const std::size_t current_generation = generation_.load(
                    std::memory_order_relaxed);
                if (current_generation == seen_generation)
                {
                    // A spurious wake raced with no publication.  Retry
                    // without touching task state outside the mutex.
                    continue;
                }
                seen_generation = current_generation;
                callback = task_callback_;
                context = task_context_;
                active = worker_index <= active_background_workers_;
            }

            if (!active)
            {
                previous_batch_was_active = false;
                continue;
            }

            previous_batch_was_active = true;
            callback(context, worker_index);
            if (remaining_background_workers_.fetch_sub(
                    1U, std::memory_order_acq_rel) == 1U)
            {
                // Pair with the predicate wait so a completion between the
                // main thread's check and park cannot lose its notification.
                const std::lock_guard<std::mutex> lock(finished_mutex_);
                finished_.notify_one();
            }
        }
    }

    std::mutex execution_mutex_;
    std::mutex ready_mutex_;
    std::mutex finished_mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    TaskCallback task_callback_ = nullptr;
    void* task_context_ = nullptr;
    std::atomic<std::size_t> generation_{0U};
    std::atomic<std::size_t> remaining_background_workers_{0U};
    std::size_t active_background_workers_ = 0U;
    std::atomic<bool> stopping_{false};
    std::vector<std::thread> workers_;
    std::vector<std::exception_ptr> failures_;
};

inline DeterministicExecutor& deterministic_executor()
{
    static DeterministicExecutor executor;
    return executor;
}

} // namespace detail

// Invoke function(begin, end) on deterministic contiguous blocks.  A caller
// width of zero or one is exactly sequential.  minimum_items_per_worker is a
// deterministic workload grain, never a host-derived worker selection.
template <typename Function>
void deterministic_parallel_blocks(
    std::size_t count,
    std::size_t requested_workers,
    std::size_t minimum_items_per_worker,
    Function&& function)
{
    if (count == 0U)
    {
        return;
    }

    // A worker callback may call a lower-level component that also exposes a
    // worker width.  Execute that nested range canonically on the current
    // thread: acquiring the serialized executor again would deadlock, while
    // spawning another layer would oversubscribe and alter error timing.
    if (detail::deterministic_executor_depth() != 0U)
    {
        function(0U, count);
        return;
    }

    const std::size_t grain =
        std::max<std::size_t>(1U, minimum_items_per_worker);
    const std::size_t useful_workers = 1U + (count - 1U) / grain;
    const std::size_t worker_count = requested_workers <= 1U
        ? 1U
        : std::min({requested_workers, count, useful_workers});
    if (worker_count <= 1U)
    {
        function(0U, count);
        return;
    }

    detail::deterministic_executor().run(
        worker_count,
        [&](std::size_t worker)
        {
            const std::size_t begin = count * worker / worker_count;
            const std::size_t end =
                count * (worker + 1U) / worker_count;
            function(begin, end);
        });
}

} // namespace virne::utils

#endif
