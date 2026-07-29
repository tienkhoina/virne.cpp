#include "stats.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct ClockState
{
    std::vector<double> values;
    std::size_t calls = 0;
    std::size_t throw_on = (std::numeric_limits<std::size_t>::max)();
};

struct ScriptedClock
{
    std::shared_ptr<ClockState> state;

    double operator()()
    {
        const std::size_t index = state->calls++;
        if (index == state->throw_on)
        {
            throw std::runtime_error("clock failure");
        }
        return state->values.at(index);
    }
};

struct Record
{
    std::string id;
    std::string output;
    std::string result;
    std::string exception = "none";
    std::size_t calls = 0;
    std::size_t clocks = 0;
    bool identity = false;
    std::size_t lazy_constructed = 0;
    std::size_t lazy_consumed = 0;
};

std::shared_ptr<ClockState> make_clock_state(
    std::initializer_list<double> values)
{
    auto state = std::make_shared<ClockState>();
    state->values.assign(values);
    return state;
}

std::string bytes_to_hex(const std::string& bytes)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(bytes.size() * 2);
    for (const char character : bytes)
    {
        const auto byte = static_cast<unsigned char>(character);
        encoded.push_back(digits[byte >> 4U]);
        encoded.push_back(digits[byte & 0x0fU]);
    }
    return encoded;
}

std::string hex_u64(const std::uint64_t value)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

void print_record(const Record& record)
{
    std::cout << record.id << '\t'
              << bytes_to_hex(record.output) << '\t'
              << record.result << '\t'
              << record.exception << '\t'
              << record.calls << '\t'
              << record.clocks << '\t'
              << (record.identity ? 1 : 0) << '\t'
              << record.lazy_constructed << '\t'
              << record.lazy_consumed << '\n';
}

Record scalar_case(
    std::string id,
    std::string name,
    const double start,
    const double stop,
    const int value)
{
    const auto clock_state = make_clock_state({start, stop});
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        std::move(name),
        [&calls, value] {
            ++calls;
            return value;
        },
        output,
        ScriptedClock{clock_state});
    const int result = timed();
    return Record{
        std::move(id),
        output.str(),
        "int:" + std::to_string(result),
        "none",
        calls,
        clock_state->calls};
}

Record void_case(
    std::string id,
    std::string name,
    const double start,
    const double stop)
{
    const auto clock_state = make_clock_state({start, stop});
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        std::move(name),
        [&calls] { ++calls; },
        output,
        ScriptedClock{clock_state});
    timed();
    return Record{
        std::move(id),
        output.str(),
        "void",
        "none",
        calls,
        clock_state->calls};
}

Record argument_case()
{
    const auto clock_state = make_clock_state({0.0, 1.0});
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "arguments",
        [&calls](const int base, const int scale) {
            ++calls;
            return base * scale + 2;
        },
        output,
        ScriptedClock{clock_state});
    const int result = timed(5, 8);
    return Record{
        "arguments",
        output.str(),
        "int:" + std::to_string(result),
        "none",
        calls,
        clock_state->calls};
}

Record mutation_case()
{
    struct RenameCallable
    {
        std::function<void()> rename;
        std::size_t* calls;

        int operator()()
        {
            ++(*calls);
            rename();
            return 51;
        }
    };

    const auto clock_state = make_clock_state({0.0, 1.0});
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "before",
        RenameCallable{{}, &calls},
        output,
        ScriptedClock{clock_state});
    timed.target().rename = [&timed] {
        timed.set_function_name("after");
    };
    const int result = timed();
    return Record{
        "name_mutation",
        output.str(),
        "int:" + std::to_string(result),
        "none",
        calls,
        clock_state->calls};
}

Record nested_case()
{
    const auto clock_state =
        make_clock_state({0.0, 10.0, 11.0, 3.0});
    std::ostringstream output;
    std::size_t calls = 0;
    auto inner = virne::utils::test_running_time(
        "inner",
        [&calls] {
            ++calls;
            return 1;
        },
        output,
        ScriptedClock{clock_state});
    auto outer = virne::utils::test_running_time(
        "outer",
        [&calls, &inner] {
            ++calls;
            return inner() + 1;
        },
        output,
        ScriptedClock{clock_state});
    const int result = outer();
    return Record{
        "nested",
        output.str(),
        "int:" + std::to_string(result),
        "none",
        calls,
        clock_state->calls};
}

Record identity_case()
{
    struct Object
    {
        int value = 73;
    };

    Object object;
    const auto clock_state = make_clock_state({0.0, 1.0});
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "identity",
        [&calls, &object]() -> Object& {
            ++calls;
            return object;
        },
        output,
        ScriptedClock{clock_state});
    Object& result = timed();
    Record record{
        "identity",
        output.str(),
        "object",
        "none",
        calls,
        clock_state->calls};
    record.identity = &result == &object;
    return record;
}

Record lazy_case()
{
    struct LazyState
    {
        std::size_t constructed = 0;
        std::size_t consumed = 0;
    };

    struct LazyObject
    {
        explicit LazyObject(std::shared_ptr<LazyState> input)
            : state(std::move(input))
        {
            ++state->constructed;
        }

        std::shared_ptr<LazyState> state;
    };

    const auto lazy_state = std::make_shared<LazyState>();
    const auto clock_state = make_clock_state({0.0, 1.0});
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "lazy",
        [&calls, lazy_state] {
            ++calls;
            return LazyObject(lazy_state);
        },
        output,
        ScriptedClock{clock_state});
    LazyObject result = timed();
    (void)result;
    return Record{
        "lazy",
        output.str(),
        "lazy",
        "none",
        calls,
        clock_state->calls,
        false,
        lazy_state->constructed,
        lazy_state->consumed};
}

Record first_clock_exception_case()
{
    const auto clock_state = make_clock_state({});
    clock_state->throw_on = 0;
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "first_clock_exception",
        [&calls] { ++calls; },
        output,
        ScriptedClock{clock_state});
    std::string exception = "missing";
    try
    {
        timed();
    }
    catch (const std::runtime_error&)
    {
        exception = "first_clock";
    }
    return Record{
        "first_clock_exception",
        output.str(),
        "none",
        exception,
        calls,
        clock_state->calls};
}

Record callable_exception_case()
{
    const auto clock_state = make_clock_state({0.0, 1.0});
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "callable_exception",
        [&calls]() -> int {
            ++calls;
            throw std::logic_error("callable failure");
        },
        output,
        ScriptedClock{clock_state});
    std::string exception = "missing";
    try
    {
        (void)timed();
    }
    catch (const std::logic_error&)
    {
        exception = "callable";
    }
    return Record{
        "callable_exception",
        output.str(),
        "none",
        exception,
        calls,
        clock_state->calls};
}

Record second_clock_exception_case()
{
    const auto clock_state = make_clock_state({0.0});
    clock_state->throw_on = 1;
    std::ostringstream output;
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "second_clock_exception",
        [&calls] { ++calls; return 1; },
        output,
        ScriptedClock{clock_state});
    std::string exception = "missing";
    try
    {
        (void)timed();
    }
    catch (const std::runtime_error&)
    {
        exception = "second_clock";
    }
    return Record{
        "second_clock_exception",
        output.str(),
        "none",
        exception,
        calls,
        clock_state->calls};
}

class RejectingBuffer final : public std::streambuf
{
protected:
    std::streamsize xsputn(const char*, std::streamsize) override
    {
        return 0;
    }

    int_type overflow(int_type) override
    {
        return traits_type::eof();
    }
};

Record sink_exception_case()
{
    const auto clock_state = make_clock_state({0.0, 1.0});
    RejectingBuffer buffer;
    std::ostream output(&buffer);
    std::size_t calls = 0;
    auto timed = virne::utils::test_running_time(
        "sink_exception",
        [&calls] { ++calls; return 1; },
        output,
        ScriptedClock{clock_state});
    std::string exception = "missing";
    try
    {
        (void)timed();
    }
    catch (const std::ios_base::failure&)
    {
        exception = "sink";
    }
    return Record{
        "sink_exception",
        "",
        "none",
        exception,
        calls,
        clock_state->calls};
}

int run_differential_cases()
{
    std::cout << "id\toutput_hex\tresult\texception\tcalls\tclocks\tidentity"
                 "\tlazy_constructed\tlazy_consumed\n";

    print_record(scalar_case("ascii_scalar", "ascii", 0.0, 1.25, 42));
    print_record(scalar_case("unicode_name", u8"đồ_thị", 0.0, 1.0, 1));
    print_record(scalar_case("newline_name", "line\nbreak", 0.0, 1.0, 1));
    print_record(scalar_case(
        "nul_name", std::string("nul\0name", 8), 0.0, 1.0, 1));
    print_record(argument_case());
    print_record(void_case("delta_zero", "delta_zero", 0.0, 0.0));
    print_record(void_case(
        "delta_negative_zero", "delta_negative_zero", 0.0, -0.0));
    print_record(void_case("delta_negative", "delta_negative", 2.0, 1.25));
    print_record(void_case(
        "round_even_down", "round_even_down", 0.0, 0.03125));
    print_record(void_case(
        "round_even_up", "round_even_up", 0.0, 0.09375));
    print_record(void_case("large_finite", "large_finite", 0.0, 1.0e20));
    print_record(void_case(
        "nan", "nan", 0.0, (std::numeric_limits<double>::quiet_NaN)()));
    print_record(void_case(
        "positive_infinity",
        "positive_infinity",
        0.0,
        (std::numeric_limits<double>::infinity)()));
    print_record(void_case(
        "negative_infinity",
        "negative_infinity",
        0.0,
        -(std::numeric_limits<double>::infinity)()));
    print_record(mutation_case());
    print_record(nested_case());
    print_record(identity_case());
    print_record(lazy_case());
    print_record(first_clock_exception_case());
    print_record(callable_exception_case());
    print_record(second_clock_exception_case());
    print_record(sink_exception_case());
    return 0;
}

void fnv_update(
    std::uint64_t& hash,
    const char* data,
    const std::size_t size)
{
    for (std::size_t index = 0; index < size; ++index)
    {
        hash ^= static_cast<unsigned char>(data[index]);
        hash *= fnv_prime;
    }
}

void fnv_update_u64(std::uint64_t& hash, const std::uint64_t value)
{
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
    {
        const char byte = static_cast<char>((value >> shift) & 0xffU);
        fnv_update(hash, &byte, 1);
    }
}

class HashBuffer final : public std::streambuf
{
public:
    std::uint64_t hash() const noexcept
    {
        return hash_;
    }

    std::uint64_t bytes() const noexcept
    {
        return bytes_;
    }

protected:
    std::streamsize xsputn(const char* data, const std::streamsize count) override
    {
        if (count <= 0)
        {
            return 0;
        }
        const auto size = static_cast<std::size_t>(count);
        fnv_update(hash_, data, size);
        bytes_ += static_cast<std::uint64_t>(size);
        return count;
    }

    int_type overflow(const int_type character) override
    {
        if (traits_type::eq_int_type(character, traits_type::eof()))
        {
            return traits_type::not_eof(character);
        }
        const char value = traits_type::to_char_type(character);
        fnv_update(hash_, &value, 1);
        ++bytes_;
        return character;
    }

private:
    std::uint64_t hash_ = fnv_offset;
    std::uint64_t bytes_ = 0;
};

std::uint64_t payload_mix(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

struct AlternatingClockState
{
    std::uint64_t calls = 0;
    bool return_stop = false;
};

struct AlternatingClock
{
    std::shared_ptr<AlternatingClockState> state;

    double operator()()
    {
        ++state->calls;
        const double value = state->return_stop ? 1.0 : 0.0;
        state->return_stop = !state->return_stop;
        return value;
    }
};

struct WorkerResult
{
    std::uint64_t return_hash = fnv_offset;
    std::uint64_t output_hash = fnv_offset;
    std::uint64_t output_bytes = 0;
    std::uint64_t calls = 0;
    std::uint64_t clock_calls = 0;
};

struct RunResult
{
    std::uint64_t elapsed_ns = 0;
    std::vector<WorkerResult> workers;
};

std::uint64_t worker_input(
    const std::size_t worker,
    const std::uint64_t iteration) noexcept
{
    return (static_cast<std::uint64_t>(worker) << 32U) ^ iteration;
}

RunResult run_workers(
    const std::size_t worker_count,
    const std::uint64_t iterations,
    const bool wrapped)
{
    RunResult run;
    run.workers.resize(worker_count);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    std::mutex start_mutex;
    std::condition_variable start_condition;
    std::size_t ready_workers = 0;
    std::size_t finished_workers = 0;
    bool start_workers = false;

    for (std::size_t worker = 0; worker < worker_count; ++worker)
    {
        threads.emplace_back([&, worker] {
            {
                std::unique_lock<std::mutex> lock(start_mutex);
                ++ready_workers;
                start_condition.notify_all();
                start_condition.wait(lock, [&start_workers] {
                    return start_workers;
                });
            }

            WorkerResult result;
            if (!wrapped)
            {
                for (std::uint64_t iteration = 0;
                     iteration < iterations;
                     ++iteration)
                {
                    const std::uint64_t value = payload_mix(
                        worker_input(worker, iteration));
                    fnv_update_u64(result.return_hash, value);
                    ++result.calls;
                }
            }
            else
            {
                HashBuffer buffer;
                std::ostream output(&buffer);
                const auto clock_state =
                    std::make_shared<AlternatingClockState>();
                auto timed = virne::utils::test_running_time(
                    "bench",
                    [&result](const std::uint64_t input) {
                        ++result.calls;
                        return payload_mix(input);
                    },
                    output,
                    AlternatingClock{clock_state});
                for (std::uint64_t iteration = 0;
                     iteration < iterations;
                     ++iteration)
                {
                    const std::uint64_t value = timed(
                        worker_input(worker, iteration));
                    fnv_update_u64(result.return_hash, value);
                }
                result.output_hash = buffer.hash();
                result.output_bytes = buffer.bytes();
                result.clock_calls = clock_state->calls;
            }
            run.workers[worker] = result;
            {
                std::lock_guard<std::mutex> lock(start_mutex);
                ++finished_workers;
            }
            start_condition.notify_all();
        });
    }

    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_condition.wait(lock, [&ready_workers, worker_count] {
            return ready_workers == worker_count;
        });
    }
    const auto started = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start_workers = true;
    }
    start_condition.notify_all();
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_condition.wait(lock, [&finished_workers, worker_count] {
            return finished_workers == worker_count;
        });
    }
    const auto stopped = std::chrono::steady_clock::now();
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    run.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            stopped - started)
            .count());
    return run;
}

std::uint64_t combine_hashes(
    const std::vector<WorkerResult>& workers,
    const bool output_hash)
{
    std::uint64_t combined = fnv_offset;
    for (std::size_t index = 0; index < workers.size(); ++index)
    {
        fnv_update_u64(combined, static_cast<std::uint64_t>(index));
        fnv_update_u64(
            combined,
            output_hash ? workers[index].output_hash
                        : workers[index].return_hash);
    }
    return combined;
}

std::uint64_t sum_field(
    const std::vector<WorkerResult>& workers,
    const int field)
{
    std::uint64_t total = 0;
    for (const WorkerResult& worker : workers)
    {
        if (field == 0)
        {
            total += worker.calls;
        }
        else if (field == 1)
        {
            total += worker.clock_calls;
        }
        else
        {
            total += worker.output_bytes;
        }
    }
    return total;
}

std::uint64_t expected_local_output_hash(const std::uint64_t iterations)
{
    static const std::string line("Running time of bench: 1.0000s\n");
    std::uint64_t hash = fnv_offset;
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration)
    {
        fnv_update(hash, line.data(), line.size());
    }
    return hash;
}

int run_benchmark(
    const std::size_t worker_count,
    const std::uint64_t iterations)
{
    if (worker_count == 0 || iterations == 0)
    {
        throw std::invalid_argument("workers and iterations must be positive");
    }

    const RunResult baseline = run_workers(worker_count, iterations, false);
    const RunResult wrapped = run_workers(worker_count, iterations, true);

    const std::uint64_t baseline_return_checksum =
        combine_hashes(baseline.workers, false);
    const std::uint64_t return_checksum =
        combine_hashes(wrapped.workers, false);
    const std::uint64_t output_checksum =
        combine_hashes(wrapped.workers, true);
    const std::uint64_t expected_local_hash =
        expected_local_output_hash(iterations);
    std::vector<WorkerResult> expected_workers(worker_count);
    for (WorkerResult& worker : expected_workers)
    {
        worker.output_hash = expected_local_hash;
    }
    const std::uint64_t expected_output_checksum =
        combine_hashes(expected_workers, true);

    static const std::string line("Running time of bench: 1.0000s\n");
    const std::uint64_t expected_output_bytes =
        static_cast<std::uint64_t>(worker_count) * iterations *
        static_cast<std::uint64_t>(line.size());
    const std::uint64_t expected_calls =
        static_cast<std::uint64_t>(worker_count) * iterations;
    const std::uint64_t calls = sum_field(wrapped.workers, 0);
    const std::uint64_t clock_calls = sum_field(wrapped.workers, 1);
    const std::uint64_t output_bytes = sum_field(wrapped.workers, 2);

    const bool passed =
        baseline_return_checksum == return_checksum &&
        output_checksum == expected_output_checksum &&
        output_bytes == expected_output_bytes && calls == expected_calls &&
        clock_calls == expected_calls * 2U;

    std::cout << "benchmark_version=1\n"
              << "workers=" << worker_count << '\n'
              << "iterations_per_worker=" << iterations << '\n'
              << "baseline_elapsed_ns=" << baseline.elapsed_ns << '\n'
              << "wrapped_elapsed_ns=" << wrapped.elapsed_ns << '\n'
              << "baseline_return_checksum="
              << hex_u64(baseline_return_checksum) << '\n'
              << "return_checksum=" << hex_u64(return_checksum) << '\n'
              << "output_checksum=" << hex_u64(output_checksum) << '\n'
              << "expected_output_checksum="
              << hex_u64(expected_output_checksum) << '\n'
              << "output_bytes=" << output_bytes << '\n'
              << "expected_output_bytes=" << expected_output_bytes << '\n'
              << "calls=" << calls << '\n'
              << "clock_calls=" << clock_calls << '\n'
              << "status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "differential")
        {
            return run_differential_cases();
        }
        if (argc == 4 && std::string(argv[1]) == "benchmark")
        {
            const auto workers = static_cast<std::size_t>(
                std::stoull(argv[2]));
            const auto iterations = static_cast<std::uint64_t>(
                std::stoull(argv[3]));
            return run_benchmark(workers, iterations);
        }

        std::cerr << "usage: stats_harness differential\n"
                     "   or: stats_harness benchmark <workers> <iterations>\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "stats harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
