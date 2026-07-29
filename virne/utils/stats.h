#pragma once

#include <functional>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace virne::utils
{

struct SystemWallClock
{
    double operator()() const noexcept;
};

namespace detail
{

std::string make_running_time_prefix(
    std::string_view function_name);

void emit_running_time(
    std::ostream& output,
    std::string_view cached_prefix,
    double elapsed_seconds);

} // namespace detail

template <typename Callable, typename Clock = SystemWallClock>
class RunningTimeFunction
{
public:
    RunningTimeFunction(
        std::string function_name,
        Callable callable,
        std::ostream& output,
        Clock clock = {})
        : callable_(std::move(callable)),
          clock_(std::move(clock)),
          output_(&output),
          function_name_(std::move(function_name)),
          cached_prefix_(
              detail::make_running_time_prefix(function_name_))
    {
    }

    template <typename... Args>
    decltype(auto) operator()(Args&&... args)
    {
        const double start = static_cast<double>(
            std::invoke(clock_));

        using Result = std::invoke_result_t<
            Callable&,
            Args&&...>;

        if constexpr (std::is_void_v<Result>)
        {
            std::invoke(
                callable_,
                std::forward<Args>(args)...);
            const double stop = static_cast<double>(
                std::invoke(clock_));
            detail::emit_running_time(
                *output_,
                cached_prefix_,
                stop - start);
            return;
        }
        else if constexpr (std::is_reference_v<Result>)
        {
            Result result = std::invoke(
                callable_,
                std::forward<Args>(args)...);
            const double stop = static_cast<double>(
                std::invoke(clock_));
            detail::emit_running_time(
                *output_,
                cached_prefix_,
                stop - start);
            return std::forward<Result>(result);
        }
        else
        {
            Result result = std::invoke(
                callable_,
                std::forward<Args>(args)...);
            const double stop = static_cast<double>(
                std::invoke(clock_));
            detail::emit_running_time(
                *output_,
                cached_prefix_,
                stop - start);
            return result;
        }
    }

    const std::string& function_name() const noexcept
    {
        return function_name_;
    }

    void set_function_name(
        std::string function_name)
    {
        std::string prefix =
            detail::make_running_time_prefix(function_name);
        function_name_ = std::move(function_name);
        cached_prefix_ = std::move(prefix);
    }

    Callable& target() noexcept
    {
        return callable_;
    }

    const Callable& target() const noexcept
    {
        return callable_;
    }

private:
    Callable callable_;
    Clock clock_;
    std::ostream* output_;
    std::string function_name_;
    std::string cached_prefix_;
};

template <typename Callable>
auto test_running_time(
    std::string function_name,
    Callable&& callable,
    std::ostream& output = std::cout)
{
    using StoredCallable = std::decay_t<Callable>;
    return RunningTimeFunction<StoredCallable, SystemWallClock>(
        std::move(function_name),
        std::forward<Callable>(callable),
        output,
        SystemWallClock{});
}

template <typename Callable, typename Clock>
auto test_running_time(
    std::string function_name,
    Callable&& callable,
    std::ostream& output,
    Clock&& clock)
{
    using StoredCallable = std::decay_t<Callable>;
    using StoredClock = std::decay_t<Clock>;
    return RunningTimeFunction<StoredCallable, StoredClock>(
        std::move(function_name),
        std::forward<Callable>(callable),
        output,
        std::forward<Clock>(clock));
}

} // namespace virne::utils
