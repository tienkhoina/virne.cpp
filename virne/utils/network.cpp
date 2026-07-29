#include "network.h"

#include "nx/shortest_paths.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

namespace virne::utils
{

DynamicKey::DynamicKey(std::nullptr_t) noexcept
    : data(std::monostate{})
{
}

DynamicKey::DynamicKey(bool value) noexcept
    : data(value)
{
}

DynamicKey::DynamicKey(int value) noexcept
    : data(static_cast<std::int64_t>(value))
{
}

DynamicKey::DynamicKey(std::int64_t value) noexcept
    : data(value)
{
}

DynamicKey::DynamicKey(double value) noexcept
    : data(value)
{
}

DynamicKey::DynamicKey(const char* value)
    : data(std::string(value))
{
}

DynamicKey::DynamicKey(std::string value)
    : data(std::move(value))
{
}

DynamicValue::DynamicValue(std::nullptr_t) noexcept
    : data(std::monostate{})
{
}

DynamicValue::DynamicValue(bool value) noexcept
    : data(value)
{
}

DynamicValue::DynamicValue(int value) noexcept
    : data(static_cast<std::int64_t>(value))
{
}

DynamicValue::DynamicValue(std::int64_t value) noexcept
    : data(value)
{
}

DynamicValue::DynamicValue(double value) noexcept
    : data(value)
{
}

DynamicValue::DynamicValue(const char* value)
    : data(std::string(value))
{
}

DynamicValue::DynamicValue(std::string value)
    : data(std::move(value))
{
}

DynamicValue::DynamicValue(List value)
    : data(std::move(value))
{
}

DynamicValue::DynamicValue(Dict value)
    : data(std::move(value))
{
}

namespace
{

// Persistent workers remove thread construction from the hot path. Submissions
// are serialized, which keeps the shared executor safe for callers arriving
// from different threads without changing per-call output order.
class DeterministicExecutor
{
public:
    DeterministicExecutor() = default;
    DeterministicExecutor(const DeterministicExecutor&) = delete;
    DeterministicExecutor& operator=(const DeterministicExecutor&) = delete;

    ~DeterministicExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_)
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
        std::lock_guard<std::mutex> execution_lock(execution_mutex_);
        ensure_workers(worker_count - 1U);
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            task_ = std::forward<Function>(function);
            active_background_workers_ = worker_count - 1U;
            completed_background_workers_ = 0;
            ++generation_;
        }
        ready_.notify_all();

        std::exception_ptr foreground_failure;
        try
        {
            task_(0);
        }
        catch (...)
        {
            foreground_failure = std::current_exception();
        }

        std::unique_lock<std::mutex> state_lock(state_mutex_);
        finished_.wait(state_lock, [this] {
            return completed_background_workers_ ==
                   active_background_workers_;
        });
        task_ = {};
        state_lock.unlock();
        if (foreground_failure)
        {
            std::rethrow_exception(foreground_failure);
        }
    }

private:
    void ensure_workers(std::size_t count)
    {
        while (workers_.size() < count)
        {
            const std::size_t worker_index = workers_.size() + 1U;
            std::size_t initial_generation = 0;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                initial_generation = generation_;
            }
            workers_.emplace_back(
                [this, worker_index, initial_generation] {
                    worker_loop(worker_index, initial_generation);
                });
        }
    }

    void worker_loop(
        std::size_t worker_index,
        std::size_t seen_generation)
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        for (;;)
        {
            ready_.wait(lock, [this, &seen_generation] {
                return generation_ != seen_generation || stopping_;
            });
            if (stopping_)
            {
                return;
            }
            seen_generation = generation_;
            if (worker_index > active_background_workers_)
            {
                continue;
            }
            const auto* task = &task_;
            lock.unlock();
            (*task)(worker_index);
            lock.lock();
            ++completed_background_workers_;
            if (completed_background_workers_ ==
                active_background_workers_)
            {
                finished_.notify_one();
            }
        }
    }

    std::mutex execution_mutex_;
    std::mutex state_mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::function<void(std::size_t)> task_;
    std::size_t generation_ = 0;
    std::size_t active_background_workers_ = 0;
    std::size_t completed_background_workers_ = 0;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

DeterministicExecutor& deterministic_executor()
{
    static DeterministicExecutor executor;
    return executor;
}

std::size_t available_worker_width()
{
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        std::size_t available = 0;
        for (std::size_t cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        {
            if (CPU_ISSET(static_cast<int>(cpu), &affinity))
            {
                ++available;
            }
        }
        if (available != 0)
        {
            return available;
        }
    }
#endif
    return std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
}

template <typename Function>
void deterministic_parallel_for(
    std::size_t count,
    std::size_t minimum_items_per_worker,
    std::size_t requested_workers,
    Function&& function)
{
    if (count == 0)
    {
        return;
    }

    std::size_t workers = requested_workers;
    if (workers == 0)
    {
        // The persistent executor makes wider fan-out worthwhile for the
        // independent BFS/GML batches. Cap automatic fan-out at the measured
        // eight-worker optimum and never exceed the visible hardware width.
        workers = std::max<std::size_t>(
            1,
            std::min<std::size_t>(
                8,
                available_worker_width()));
    }
    workers = std::min(workers, count);
    workers = std::min(
        workers,
        (count + minimum_items_per_worker - 1) /
            minimum_items_per_worker);

    if (workers <= 1)
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            function(index);
        }
        return;
    }

    constexpr std::size_t persistent_worker_limit = 8;
    if (workers <= persistent_worker_limit)
    {
        const std::size_t block = (count + workers - 1U) / workers;
        std::vector<std::exception_ptr> failures(workers);
        deterministic_executor().run(
            workers,
            [&](std::size_t worker) {
                const std::size_t begin = worker * block;
                const std::size_t end = std::min(count, begin + block);
                try
                {
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        function(index);
                    }
                }
                catch (...)
                {
                    failures[worker] = std::current_exception();
                }
            });
        for (const auto& failure : failures)
        {
            if (failure)
            {
                std::rethrow_exception(failure);
            }
        }
        return;
    }

    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> failures(workers);
    threads.reserve(workers);
    const std::size_t block = (count + workers - 1) / workers;

    try
    {
        for (std::size_t worker = 0; worker < workers; ++worker)
        {
            const std::size_t begin = worker * block;
            const std::size_t end = std::min(count, begin + block);
            if (begin >= end)
            {
                break;
            }
            threads.emplace_back([&, worker, begin, end] {
                try
                {
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        function(index);
                    }
                }
                catch (...)
                {
                    failures[worker] = std::current_exception();
                }
            });
        }
    }
    catch (...)
    {
        for (auto& thread : threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
        throw;
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
    for (const auto& failure : failures)
    {
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }
}

template <typename GraphType>
BfsTreeLevels get_bfs_tree_level_impl(
    const GraphType& network,
    Vertex source)
{
    const auto lengths =
        nx::single_source_shortest_path_length(
            network, source);

    if (lengths.empty())
    {
        return {};
    }

    std::size_t max_depth = 0;
    for (const auto& [vertex, depth] : lengths)
    {
        (void)vertex;
        max_depth = std::max(max_depth, depth);
    }

    BfsTreeLevels levels(max_depth + 1);
    for (const auto& [vertex, depth] : lengths)
    {
        levels[depth].push_back(vertex);
    }
    return levels;
}

template <typename GraphType>
std::vector<BfsTreeLevels> get_bfs_tree_levels_impl(
    const GraphType& network,
    const std::vector<Vertex>& sources,
    std::size_t worker_count)
{
    std::vector<BfsTreeLevels> results(sources.size());
    deterministic_parallel_for(
        sources.size(),
        1,
        worker_count,
        [&](std::size_t index) {
            results[index] = get_bfs_tree_level_impl(
                network, sources[index]);
        });
    return results;
}

void flatten_into(
    const DynamicValue& value,
    std::vector<DynamicValue>& output)
{
    if (value.is<DynamicValue::List>())
    {
        for (const auto& child : value.as<DynamicValue::List>())
        {
            flatten_into(child, output);
        }
        return;
    }
    if (value.is<DynamicValue::Dict>())
    {
        for (const auto& [key, child] : value.as<DynamicValue::Dict>())
        {
            (void)key;
            flatten_into(child, output);
        }
        return;
    }
    if (value.is<std::monostate>())
    {
        throw std::invalid_argument(
            "flatten_recurrent_dict does not support None/null leaves");
    }
    output.push_back(value);
}

std::string escape_python_string(std::string_view value)
{
    const bool contains_single_quote =
        value.find('\'') != std::string_view::npos;
    const bool contains_double_quote =
        value.find('"') != std::string_view::npos;
    const char quote =
        contains_single_quote && !contains_double_quote ? '"' : '\'';

    std::string result;
    result.reserve(value.size() + 2);
    result.push_back(quote);
    static constexpr char hex_digits[] = "0123456789abcdef";
    for (const unsigned char byte : value)
    {
        switch (byte)
        {
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (byte == static_cast<unsigned char>(quote))
            {
                result.push_back('\\');
                result.push_back(quote);
            }
            else if (byte < 0x20U || byte == 0x7fU)
            {
                result += "\\x";
                result.push_back(hex_digits[byte >> 4]);
                result.push_back(hex_digits[byte & 0x0f]);
            }
            else
            {
                result.push_back(static_cast<char>(byte));
            }
            break;
        }
    }
    result.push_back(quote);
    return result;
}

std::string python_float_string(double value)
{
    if (std::isnan(value))
    {
        return "nan";
    }
    if (std::isinf(value))
    {
        return value < 0.0 ? "-inf" : "inf";
    }
    if (value == 0.0)
    {
        return std::signbit(value) ? "-0.0" : "0.0";
    }

    char buffer[128]{};
    const auto conversion = std::to_chars(
        std::begin(buffer),
        std::end(buffer),
        value,
        std::chars_format::general);

    std::string result;
    if (conversion.ec == std::errc{})
    {
        result.assign(buffer, conversion.ptr);
    }
    else
    {
        std::ostringstream stream;
        stream << std::setprecision(
                      std::numeric_limits<double>::max_digits10)
               << value;
        result = stream.str();
    }

    const bool negative = !result.empty() && result.front() == '-';
    const std::size_t number_begin = negative ? 1U : 0U;
    const std::size_t exponent_position = result.find_first_of("eE");
    const std::string_view mantissa(
        result.data() + number_begin,
        (exponent_position == std::string::npos
             ? result.size()
             : exponent_position) -
            number_begin);
    int exponent_suffix = 0;
    if (exponent_position != std::string::npos)
    {
        exponent_suffix = std::stoi(result.substr(exponent_position + 1));
    }

    const std::size_t point = mantissa.find('.');
    const std::size_t integer_digits =
        point == std::string_view::npos ? mantissa.size() : point;
    std::string digits;
    digits.reserve(mantissa.size());
    for (const char character : mantissa)
    {
        if (character != '.')
        {
            digits.push_back(character);
        }
    }
    const std::size_t first_nonzero = digits.find_first_not_of('0');
    if (first_nonzero == std::string::npos)
    {
        return negative ? "-0.0" : "0.0";
    }
    const int decimal_exponent =
        exponent_suffix + static_cast<int>(integer_digits) - 1 -
        static_cast<int>(first_nonzero);
    digits.erase(0, first_nonzero);

    std::string python;
    if (negative)
    {
        python.push_back('-');
    }
    if (decimal_exponent >= -4 && decimal_exponent < 16)
    {
        if (decimal_exponent < 0)
        {
            python += "0.";
            python.append(
                static_cast<std::size_t>(-decimal_exponent - 1), '0');
            python += digits;
        }
        else
        {
            const std::size_t whole_digits =
                static_cast<std::size_t>(decimal_exponent) + 1U;
            if (digits.size() <= whole_digits)
            {
                python += digits;
                python.append(whole_digits - digits.size(), '0');
                python += ".0";
            }
            else
            {
                python.append(digits, 0, whole_digits);
                python.push_back('.');
                python.append(digits, whole_digits, std::string::npos);
            }
        }
        return python;
    }

    python.push_back(digits.front());
    if (digits.size() > 1)
    {
        python.push_back('.');
        python.append(digits, 1, std::string::npos);
    }
    python.push_back('e');
    python.push_back(decimal_exponent < 0 ? '-' : '+');
    const unsigned int absolute_exponent = static_cast<unsigned int>(
        decimal_exponent < 0 ? -decimal_exponent : decimal_exponent);
    if (absolute_exponent < 10U)
    {
        python.push_back('0');
    }
    python += std::to_string(absolute_exponent);
    return python;
}

std::string python_repr(const DynamicValue& value);

std::string python_str(const DynamicKey& key)
{
    if (key.is<std::monostate>())
    {
        return "None";
    }
    if (key.is<bool>())
    {
        return key.as<bool>() ? "True" : "False";
    }
    if (key.is<std::int64_t>())
    {
        return std::to_string(key.as<std::int64_t>());
    }
    if (key.is<double>())
    {
        return python_float_string(key.as<double>());
    }
    return key.as<std::string>();
}

std::string python_repr(const DynamicKey& key)
{
    if (key.is<std::string>())
    {
        return escape_python_string(key.as<std::string>());
    }
    return python_str(key);
}

bool python_key_equal(const DynamicKey& left, const DynamicKey& right)
{
    if (left.is<std::monostate>() || right.is<std::monostate>())
    {
        return left.is<std::monostate>() && right.is<std::monostate>();
    }
    if (left.is<std::string>() || right.is<std::string>())
    {
        return left.is<std::string>() && right.is<std::string>() &&
               left.as<std::string>() == right.as<std::string>();
    }

    auto integer_value = [](const DynamicKey& key)
        -> std::optional<std::int64_t> {
        if (key.is<bool>())
        {
            return key.as<bool>() ? 1 : 0;
        }
        if (key.is<std::int64_t>())
        {
            return key.as<std::int64_t>();
        }
        return std::nullopt;
    };
    const auto left_integer = integer_value(left);
    const auto right_integer = integer_value(right);
    if (left_integer && right_integer)
    {
        return *left_integer == *right_integer;
    }
    if (left.is<double>() && right.is<double>())
    {
        return left.as<double>() == right.as<double>();
    }

    const std::int64_t integer =
        left_integer ? *left_integer : *right_integer;
    const double floating =
        left.is<double>() ? left.as<double>() : right.as<double>();
    constexpr double int64_lower_bound = -0x1p63;
    constexpr double int64_upper_bound = 0x1p63;
    return std::isfinite(floating) &&
           floating >= int64_lower_bound &&
           floating < int64_upper_bound &&
           std::trunc(floating) == floating &&
           static_cast<std::int64_t>(floating) == integer;
}

std::string python_str(const DynamicValue& value)
{
    if (value.is<std::string>())
    {
        return value.as<std::string>();
    }
    return python_repr(value);
}

std::string python_repr(const DynamicValue& value)
{
    if (value.is<std::monostate>())
    {
        return "None";
    }
    if (value.is<bool>())
    {
        return value.as<bool>() ? "True" : "False";
    }
    if (value.is<std::int64_t>())
    {
        return std::to_string(value.as<std::int64_t>());
    }
    if (value.is<double>())
    {
        return python_float_string(value.as<double>());
    }
    if (value.is<std::string>())
    {
        return escape_python_string(value.as<std::string>());
    }
    if (value.is<DynamicValue::List>())
    {
        std::string result = "[";
        bool first = true;
        for (const auto& child : value.as<DynamicValue::List>())
        {
            if (!first)
            {
                result += ", ";
            }
            first = false;
            result += python_repr(child);
        }
        result += ']';
        return result;
    }

    std::string result = "{";
    bool first = true;
    for (const auto& [key, child] : value.as<DynamicValue::Dict>())
    {
        if (!first)
        {
            result += ", ";
        }
        first = false;
        result += python_repr(key);
        result += ": ";
        result += python_repr(child);
    }
    result += '}';
    return result;
}

DynamicValue* find_value(
    DynamicValue::Dict& dict,
    std::string_view key)
{
    const auto iterator = std::find_if(
        dict.begin(),
        dict.end(),
        [key](const auto& entry) {
            return entry.first.template is<std::string>() &&
                   entry.first.template as<std::string>() == key;
        });
    return iterator == dict.end()
        ? nullptr
        : &iterator->second;
}

std::int64_t python_int(const DynamicValue& value)
{
    if (value.is<std::int64_t>())
    {
        return value.as<std::int64_t>();
    }
    if (value.is<bool>())
    {
        return value.as<bool>() ? 1 : 0;
    }
    if (value.is<double>())
    {
        const double number = value.as<double>();
        constexpr double int64_lower_bound = -0x1p63;
        constexpr double int64_upper_bound = 0x1p63;
        if (!std::isfinite(number) ||
            number < int64_lower_bound ||
            number >= int64_upper_bound)
        {
            throw std::out_of_range(
                "numeric attribute cannot be represented as int64");
        }
        return static_cast<std::int64_t>(number);
    }
    if (value.is<std::string>())
    {
        const std::string& text = value.as<std::string>();
        std::size_t first = 0;
        while (first < text.size() &&
               std::isspace(static_cast<unsigned char>(text[first])))
        {
            ++first;
        }
        std::size_t last = text.size();
        while (last > first &&
               std::isspace(static_cast<unsigned char>(text[last - 1])))
        {
            --last;
        }
        if (first == last)
        {
            throw std::invalid_argument("empty integer attribute");
        }

        bool negative = false;
        if (text[first] == '+' || text[first] == '-')
        {
            negative = text[first] == '-';
            ++first;
        }
        if (first == last)
        {
            throw std::invalid_argument("integer attribute has no digits");
        }

        // Python accepts underscores between decimal digits.  Normalize them
        // before from_chars while rejecting leading, trailing or doubled
        // underscores exactly as int(text) does for this ASCII subset.
        std::string digits;
        digits.reserve(last - first + 1);
        if (negative)
        {
            digits.push_back('-');
        }
        bool previous_was_digit = false;
        for (std::size_t index = first; index < last; ++index)
        {
            const char character = text[index];
            if (character == '_')
            {
                const bool next_is_digit =
                    index + 1 < last &&
                    std::isdigit(static_cast<unsigned char>(text[index + 1]));
                if (!previous_was_digit || !next_is_digit)
                {
                    throw std::invalid_argument(
                        "invalid underscore in integer attribute");
                }
                previous_was_digit = false;
                continue;
            }
            if (!std::isdigit(static_cast<unsigned char>(character)))
            {
                throw std::invalid_argument(
                    "attribute is not a base-10 integer");
            }
            digits.push_back(character);
            previous_was_digit = true;
        }

        std::int64_t converted = 0;
        const auto result = std::from_chars(
            digits.data(),
            digits.data() + digits.size(),
            converted,
            10);
        if (result.ec != std::errc{} ||
            result.ptr != digits.data() + digits.size())
        {
            throw std::invalid_argument(
                "attribute is not a base-10 integer");
        }
        return converted;
    }
    throw std::invalid_argument(
        "attribute value cannot be converted to int");
}

} // namespace

PathLinks path_to_links(
    const std::vector<Vertex>& path,
    std::size_t worker_count)
{
    if (path.size() <= 1)
    {
        throw std::invalid_argument(
            "path_to_links requires at least two vertices");
    }

    PathLinks links(path.size() - 1);
    deterministic_parallel_for(
        links.size(),
        32768,
        // Even with a persistent worker, this contiguous memory-bound copy is
        // faster sequentially at realistic Virne path sizes. Explicit worker
        // counts remain available for unusually large caller-owned batches.
        worker_count == 0 ? 1 : worker_count,
        [&](std::size_t index) {
            links[index] = {path[index], path[index + 1]};
        });
    return links;
}

BfsTreeLevels get_bfs_tree_level(
    const Graph& network,
    Vertex source)
{
    return get_bfs_tree_level_impl(network, source);
}

std::vector<BfsTreeLevels> get_bfs_tree_levels(
    const Graph& network,
    const std::vector<Vertex>& sources,
    std::size_t worker_count)
{
    return get_bfs_tree_levels_impl(
        network, sources, worker_count);
}

std::vector<BfsTreeLevels> get_bfs_tree_levels(
    const DiGraph& network,
    const std::vector<Vertex>& sources,
    std::size_t worker_count)
{
    return get_bfs_tree_levels_impl(
        network, sources, worker_count);
}

BfsTreeLevels get_bfs_tree_level(
    const DiGraph& network,
    Vertex source)
{
    return get_bfs_tree_level_impl(network, source);
}

std::vector<DynamicValue> flatten_recurrent_dict(
    const DynamicValue& recurrent_value)
{
    if (!recurrent_value.is<DynamicValue::List>() &&
        !recurrent_value.is<DynamicValue::Dict>())
    {
        throw std::invalid_argument(
            "flatten_recurrent_dict requires a list or dict root");
    }

    std::vector<DynamicValue> output;
    flatten_into(recurrent_value, output);
    return output;
}

DynamicDictList flatten_dict_list_for_gml(
    const DynamicDictList& dicts,
    std::size_t worker_count)
{
    DynamicDictList output(dicts.size());
    deterministic_parallel_for(
        dicts.size(),
        256,
        worker_count,
        [&](std::size_t index) {
            const auto& dict = dicts[index];
            std::vector<std::pair<DynamicKey, const DynamicValue*>>
                python_mapping;
            python_mapping.reserve(dict.size());
            for (const auto& [key, value] : dict)
            {
                const auto existing = std::find_if(
                    python_mapping.begin(),
                    python_mapping.end(),
                    [&](const auto& entry) {
                        return python_key_equal(entry.first, key);
                    });
                if (existing == python_mapping.end())
                {
                    python_mapping.emplace_back(key, &value);
                }
                else
                {
                    // Python retains the first equal key object but stores the
                    // most recently assigned value.
                    existing->second = &value;
                }
            }

            DynamicValue::Dict clean;
            clean.reserve(python_mapping.size());
            for (const auto& [key, value] : python_mapping)
            {
                const std::string clean_key = python_str(key);
                const auto existing = std::find_if(
                    clean.begin(),
                    clean.end(),
                    [&](const auto& entry) {
                        return entry.first.template is<std::string>() &&
                               entry.first.template as<std::string>() ==
                                   clean_key;
                    });
                if (existing == clean.end())
                {
                    clean.emplace_back(clean_key, python_str(*value));
                }
                else
                {
                    existing->second = python_str(*value);
                }
            }
            output[index] = std::move(clean);
        });
    return output;
}

DynamicDictList& sanitize_attr_setting(
    DynamicDictList& attrs)
{
    for (auto& entry : attrs)
    {
        if (DynamicValue* low = find_value(entry, "low"))
        {
            *low = DynamicValue(python_int(*low));
        }
        if (DynamicValue* high = find_value(entry, "high"))
        {
            *high = DynamicValue(python_int(*high));
        }
    }
    return attrs;
}

} // namespace virne::utils
