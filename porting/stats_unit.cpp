#include "stats.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

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

class TestFailure final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

void require(const bool condition, const std::string& message)
{
    if (!condition)
    {
        throw TestFailure(message);
    }
}

std::shared_ptr<ClockState> make_clock_state(
    std::initializer_list<double> values)
{
    auto state = std::make_shared<ClockState>();
    state->values.assign(values);
    return state;
}

std::string expected_line(
    const std::string& name,
    const std::string& formatted_seconds)
{
    std::string expected("Running time of ");
    expected.append(name);
    expected.append(": ");
    expected.append(formatted_seconds);
    expected.append("s\n");
    return expected;
}

void check_format(
    const std::string& name,
    const double start,
    const double stop,
    const std::string& formatted_seconds)
{
    const auto clock_state = make_clock_state({start, stop});
    std::ostringstream output;
    int calls = 0;
    auto timed = virne::utils::test_running_time(
        name,
        [&calls] { ++calls; },
        output,
        ScriptedClock{clock_state});

    timed();

    require(calls == 1, "format case invoked callable more than once");
    require(clock_state->calls == 2, "format case did not read clock twice");
    require(
        output.str() == expected_line(name, formatted_seconds),
        "formatted output bytes differ for name/seconds case");
}

void test_names_and_formatting()
{
    check_format("ascii", 1.0, 2.25, "1.2500");
    check_format(u8"đồ_thị", 0.0, 1.0, "1.0000");
    check_format("line\nbreak", 0.0, 1.0, "1.0000");
    check_format(std::string("nul\0name", 8), 0.0, 1.0, "1.0000");

    check_format("zero", 0.0, 0.0, "0.0000");
    check_format("negative_zero", 0.0, -0.0, "-0.0000");
    check_format("negative", 2.0, 1.25, "-0.7500");
    check_format("round_even_down", 0.0, 0.03125, "0.0312");
    check_format("round_even_up", 0.0, 0.09375, "0.0938");
    check_format(
        "large",
        0.0,
        1.0e20,
        "100000000000000000000.0000");
    check_format(
        "nan",
        0.0,
        (std::numeric_limits<double>::quiet_NaN)(),
        "nan");
    check_format(
        "positive_infinity",
        0.0,
        (std::numeric_limits<double>::infinity)(),
        "inf");
    check_format(
        "negative_infinity",
        0.0,
        -(std::numeric_limits<double>::infinity)(),
        "-inf");
}

struct NonDefaultMoveOnly
{
    explicit NonDefaultMoveOnly(const int input)
        : value(input)
    {
    }

    NonDefaultMoveOnly() = delete;
    NonDefaultMoveOnly(const NonDefaultMoveOnly&) = delete;
    NonDefaultMoveOnly& operator=(const NonDefaultMoveOnly&) = delete;
    NonDefaultMoveOnly(NonDefaultMoveOnly&&) noexcept = default;
    NonDefaultMoveOnly& operator=(NonDefaultMoveOnly&&) noexcept = default;

    int value;
};

struct RvalueReferenceCallable
{
    int value = 37;

    int&& operator()() noexcept
    {
        return std::move(value);
    }
};

struct ForwardingProbe
{
    int lvalue_calls = 0;
    int rvalue_calls = 0;

    int operator()(std::unique_ptr<int>& value)
    {
        ++lvalue_calls;
        return *value + 1;
    }

    int operator()(std::unique_ptr<int>&& value)
    {
        ++rvalue_calls;
        return *value + 2;
    }
};

void test_return_and_forwarding_contract()
{
    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "scalar",
            [](const int value) { return value * 2; },
            output,
            ScriptedClock{clock_state});
        require(timed(21) == 42, "scalar return differs");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "move_only",
            [] { return std::make_unique<int>(43); },
            output,
            ScriptedClock{clock_state});
        std::unique_ptr<int> result = timed();
        require(result && *result == 43, "move-only result differs");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "non_default",
            [] { return NonDefaultMoveOnly(44); },
            output,
            ScriptedClock{clock_state});
        NonDefaultMoveOnly result = timed();
        require(result.value == 44, "non-default result differs");
    }

    {
        int object = 45;
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "lvalue_reference",
            [&object]() -> int& { return object; },
            output,
            ScriptedClock{clock_state});
        int& result = timed();
        require(&result == &object, "lvalue-reference identity differs");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "rvalue_reference",
            RvalueReferenceCallable{},
            output,
            ScriptedClock{clock_state});
        static_assert(
            std::is_same_v<decltype(timed()), int&&>,
            "rvalue-reference category must be preserved");
        int&& result = timed();
        require(
            &result == &timed.target().value,
            "rvalue-reference identity differs");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        int calls = 0;
        auto timed = virne::utils::test_running_time(
            "void",
            [&calls] { ++calls; },
            output,
            ScriptedClock{clock_state});
        static_assert(
            std::is_void_v<decltype(timed())>,
            "void result must stay void");
        timed();
        require(calls == 1, "void callable invocation count differs");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0, 2.0, 3.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "forwarding",
            ForwardingProbe{},
            output,
            ScriptedClock{clock_state});
        auto value = std::make_unique<int>(10);
        require(timed(value) == 11, "lvalue forwarding differs");
        require(timed(std::move(value)) == 12, "rvalue forwarding differs");
        require(
            timed.target().lvalue_calls == 1 &&
                timed.target().rvalue_calls == 1,
            "argument value category was not preserved");
    }
}

void test_dynamic_name_and_nesting()
{
    {
        struct RenameCallable
        {
            std::function<void()> rename;

            int operator()()
            {
                rename();
                return 51;
            }
        };

        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "before",
            RenameCallable{},
            output,
            ScriptedClock{clock_state});
        timed.target().rename = [&timed] {
            timed.set_function_name("after");
        };

        require(timed() == 51, "renamed callable result differs");
        require(
            timed.function_name() == "after",
            "function_name accessor differs after mutation");
        require(
            output.str() == expected_line("after", "1.0000"),
            "same-call dynamic name was not observed");
    }

    {
        const auto clock_state =
            make_clock_state({0.0, 10.0, 11.0, 3.0});
        std::ostringstream output;
        auto inner = virne::utils::test_running_time(
            "inner",
            [] { return 1; },
            output,
            ScriptedClock{clock_state});
        auto outer = virne::utils::test_running_time(
            "outer",
            [&inner] { return inner() + 1; },
            output,
            ScriptedClock{clock_state});

        require(outer() == 2, "nested result differs");
        require(
            output.str() ==
                expected_line("inner", "1.0000") +
                    expected_line("outer", "3.0000"),
            "nested output order differs");
        require(clock_state->calls == 4, "nested clock count differs");
    }
}

void test_exception_order()
{
    {
        const auto clock_state = make_clock_state({});
        clock_state->throw_on = 0;
        std::ostringstream output;
        int calls = 0;
        auto timed = virne::utils::test_running_time(
            "first_clock",
            [&calls] { ++calls; },
            output,
            ScriptedClock{clock_state});
        try
        {
            timed();
            require(false, "first clock exception did not propagate");
        }
        catch (const std::runtime_error& error)
        {
            require(
                std::string(error.what()) == "clock failure",
                "first clock exception changed");
        }
        require(calls == 0, "callable ran after first clock failure");
        require(clock_state->calls == 1, "first clock count differs");
        require(output.str().empty(), "first clock failure wrote output");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        int calls = 0;
        auto timed = virne::utils::test_running_time(
            "callable",
            [&calls]() -> int {
                ++calls;
                throw std::logic_error("callable failure");
            },
            output,
            ScriptedClock{clock_state});
        try
        {
            (void)timed();
            require(false, "callable exception did not propagate");
        }
        catch (const std::logic_error& error)
        {
            require(
                std::string(error.what()) == "callable failure",
                "callable exception changed");
        }
        require(calls == 1, "throwing callable count differs");
        require(clock_state->calls == 1, "clock ran after callable failure");
        require(output.str().empty(), "callable failure wrote output");
    }

    {
        const auto clock_state = make_clock_state({0.0});
        clock_state->throw_on = 1;
        std::ostringstream output;
        int calls = 0;
        auto timed = virne::utils::test_running_time(
            "second_clock",
            [&calls] { ++calls; return 1; },
            output,
            ScriptedClock{clock_state});
        try
        {
            (void)timed();
            require(false, "second clock exception did not propagate");
        }
        catch (const std::runtime_error& error)
        {
            require(
                std::string(error.what()) == "clock failure",
                "second clock exception changed");
        }
        require(calls == 1, "callable did not finish before second clock");
        require(clock_state->calls == 2, "second clock count differs");
        require(output.str().empty(), "second clock failure wrote output");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        RejectingBuffer buffer;
        std::ostream output(&buffer);
        int calls = 0;
        auto timed = virne::utils::test_running_time(
            "sink",
            [&calls] { ++calls; return 1; },
            output,
            ScriptedClock{clock_state});
        try
        {
            (void)timed();
            require(false, "sink exception did not propagate");
        }
        catch (const std::ios_base::failure&)
        {
        }
        require(calls == 1, "sink failure callable count differs");
        require(clock_state->calls == 2, "sink failure clock count differs");
    }
}

struct CountingCallable
{
    int calls = 0;

    int operator()()
    {
        return ++calls;
    }
};

struct MoveOnlyCallable
{
    explicit MoveOnlyCallable(const int value)
        : value_(std::make_unique<int>(value))
    {
    }

    MoveOnlyCallable(const MoveOnlyCallable&) = delete;
    MoveOnlyCallable& operator=(const MoveOnlyCallable&) = delete;
    MoveOnlyCallable(MoveOnlyCallable&&) noexcept = default;
    MoveOnlyCallable& operator=(MoveOnlyCallable&&) noexcept = default;

    int operator()() const
    {
        return *value_;
    }

private:
    std::unique_ptr<int> value_;
};

void test_ownership_and_lazy_boundary()
{
    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        CountingCallable external;
        auto copied = virne::utils::test_running_time(
            "copied",
            external,
            output,
            ScriptedClock{clock_state});
        static_assert(
            std::is_copy_constructible_v<decltype(copied)>,
            "copyable storage must make wrapper copyable");
        require(copied() == 1, "copied callable result differs");
        require(external.calls == 0, "factory did not own lvalue copy");
        require(copied.target().calls == 1, "target accessor differs");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        CountingCallable external;
        auto referenced = virne::utils::test_running_time(
            "referenced",
            std::ref(external),
            output,
            ScriptedClock{clock_state});
        require(referenced() == 1, "referenced callable result differs");
        require(external.calls == 1, "std::ref identity was not preserved");
        require(
            &referenced.target().get() == &external,
            "target std::ref identity differs");
    }

    {
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto move_only = virne::utils::test_running_time(
            "move_only_callable",
            MoveOnlyCallable(61),
            output,
            ScriptedClock{clock_state});
        static_assert(
            !std::is_copy_constructible_v<decltype(move_only)>,
            "move-only callable wrapper must not become copyable");
        static_assert(
            std::is_move_constructible_v<decltype(move_only)>,
            "move-only callable wrapper must remain movable");
        require(move_only() == 61, "move-only callable result differs");
    }

    {
        struct LazyState
        {
            int constructed = 0;
            int consumed = 0;
        };

        struct LazyValue
        {
            std::shared_ptr<LazyState> state;

            explicit LazyValue(std::shared_ptr<LazyState> input)
                : state(std::move(input))
            {
                ++state->constructed;
            }

            void consume()
            {
                ++state->consumed;
            }
        };

        const auto lazy_state = std::make_shared<LazyState>();
        const auto clock_state = make_clock_state({0.0, 1.0});
        std::ostringstream output;
        auto timed = virne::utils::test_running_time(
            "lazy",
            [lazy_state] { return LazyValue(lazy_state); },
            output,
            ScriptedClock{clock_state});
        LazyValue value = timed();
        require(lazy_state->constructed == 1, "lazy object was not created");
        require(lazy_state->consumed == 0, "lazy work ran inside wrapper");
        value.consume();
        require(lazy_state->consumed == 1, "lazy object could not be consumed");
    }
}

void test_default_wall_clock()
{
    std::ostringstream output;
    auto timed = virne::utils::test_running_time(
        "wall_clock",
        [] { return 71; },
        output);
    require(timed() == 71, "default wall-clock return differs");

    const std::string text = output.str();
    const std::string prefix("Running time of wall_clock: ");
    require(text.rfind(prefix, 0) == 0, "default wall-clock prefix differs");
    require(
        text.size() >= 2 && text.substr(text.size() - 2) == "s\n",
        "default wall-clock suffix differs");
}

} // namespace

int main()
{
    try
    {
        test_names_and_formatting();
        test_return_and_forwarding_contract();
        test_dynamic_name_and_nesting();
        test_exception_order();
        test_ownership_and_lazy_boundary();
        test_default_wall_clock();
        std::cout << "stats unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "stats unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
