#include "attribute/attribute_method.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace
{

namespace attribute = virne::network::attribute;

using attribute::AttributeMethodErrorCode;
using attribute::AttributeMethodException;
using attribute::AttributeMethodOperation;
using attribute::AttributeNumber;
using attribute::ComparisonOperation;
using attribute::ConstraintRestriction;
using attribute::ResourceUpdateOperation;

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

enum class BenchmarkKind : std::uint8_t
{
    resource_add_int,
    resource_sub_double,
    guarded_subtract_int,
    guard_failure_int,
    hard_ge_int,
    hard_le_double,
    hard_eq_double,
    soft_ge_int,
};

struct BenchmarkFixture
{
    std::vector<AttributeNumber> virtual_values;
    std::vector<AttributeNumber> physical_inputs;
    std::vector<AttributeNumber> outputs;
    std::vector<std::uint8_t> flags;
};

std::string hex_encode(std::string_view value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(value.size() * 2U, '0');
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2U] = digits[byte >> 4U];
        result[index * 2U + 1U] = digits[byte & 0x0FU];
    }
    return result;
}

std::uint8_t hex_nibble(char value)
{
    if (value >= '0' && value <= '9')
    {
        return static_cast<std::uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f')
    {
        return static_cast<std::uint8_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F')
    {
        return static_cast<std::uint8_t>(value - 'A' + 10);
    }
    throw std::invalid_argument("invalid hexadecimal input");
}

std::string hex_decode(std::string_view value)
{
    if ((value.size() % 2U) != 0U)
    {
        throw std::invalid_argument("odd hexadecimal input length");
    }
    std::string result(value.size() / 2U, '\0');
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        const std::uint8_t high = hex_nibble(value[index * 2U]);
        const std::uint8_t low = hex_nibble(value[index * 2U + 1U]);
        result[index] = static_cast<char>(
            static_cast<unsigned int>(high) * 16U +
            static_cast<unsigned int>(low));
    }
    return result;
}

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

double double_from_bits(std::uint64_t bits)
{
    double value = 0.0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::string hex_u64(std::uint64_t value)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

AttributeNumber parse_number(std::string_view token)
{
    if (token.size() < 3U || token[1] != ':')
    {
        throw std::invalid_argument("invalid AttributeNumber token");
    }
    const std::string payload(token.substr(2U));
    if (token[0] == 'b')
    {
        if (payload == "0")
        {
            return AttributeNumber{false};
        }
        if (payload == "1")
        {
            return AttributeNumber{true};
        }
        throw std::invalid_argument("invalid boolean AttributeNumber payload");
    }
    if (token[0] == 'i')
    {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(payload, &consumed, 10);
        if (consumed != payload.size())
        {
            throw std::invalid_argument("invalid integer AttributeNumber payload");
        }
        return AttributeNumber{static_cast<std::int64_t>(parsed)};
    }
    if (token[0] == 'f')
    {
        if (payload.size() != 16U)
        {
            throw std::invalid_argument("invalid floating AttributeNumber payload");
        }
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(payload, &consumed, 16);
        if (consumed != payload.size())
        {
            throw std::invalid_argument("invalid floating AttributeNumber payload");
        }
        return AttributeNumber{double_from_bits(static_cast<std::uint64_t>(parsed))};
    }
    throw std::invalid_argument("unknown AttributeNumber lane");
}

std::vector<double> parse_double_bits_csv(std::string_view value)
{
    std::vector<double> result;
    if (value.empty())
    {
        return result;
    }
    std::size_t begin = 0U;
    while (begin <= value.size())
    {
        const std::size_t separator = value.find(',', begin);
        const std::size_t end = separator == std::string_view::npos
            ? value.size()
            : separator;
        const std::string_view token = value.substr(begin, end - begin);
        if (token.size() != 16U)
        {
            throw std::invalid_argument("batch double bit token must have 16 digits");
        }
        std::size_t consumed = 0U;
        const unsigned long long parsed =
            std::stoull(std::string(token), &consumed, 16);
        if (consumed != token.size())
        {
            throw std::invalid_argument("invalid batch double bit token");
        }
        result.push_back(double_from_bits(static_cast<std::uint64_t>(parsed)));
        if (separator == std::string_view::npos)
        {
            break;
        }
        begin = separator + 1U;
        if (begin == value.size())
        {
            throw std::invalid_argument("trailing batch double bit separator");
        }
    }
    return result;
}

std::string serialize_flags_csv(const std::vector<std::uint8_t>& flags)
{
    std::string result;
    if (!flags.empty())
    {
        result.reserve(flags.size() * 2U - 1U);
    }
    for (std::size_t index = 0U; index < flags.size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        if (flags[index] > 1U)
        {
            throw std::runtime_error("batch produced a non-boolean flag");
        }
        result.push_back(flags[index] == 0U ? '0' : '1');
    }
    return result;
}

std::string serialize_double_bits_csv(const std::vector<double>& values)
{
    std::string result;
    if (!values.empty())
    {
        result.reserve(values.size() * 17U - 1U);
    }
    for (std::size_t index = 0U; index < values.size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        result += hex_u64(double_bits(values[index]));
    }
    return result;
}

std::string serialize_number(const AttributeNumber& value)
{
    if (const bool* boolean = std::get_if<bool>(&value))
    {
        return std::string("b:") + (*boolean ? "1" : "0");
    }
    if (const std::int64_t* integer = std::get_if<std::int64_t>(&value))
    {
        return "i:" + std::to_string(*integer);
    }
    return "f:" + hex_u64(double_bits(std::get<double>(value)));
}

std::string error_code_name(AttributeMethodErrorCode value)
{
    switch (value)
    {
    case AttributeMethodErrorCode::invalid_resource_state:
        return "invalid_resource_state";
    case AttributeMethodErrorCode::unsupported_update_operation:
        return "unsupported_update_operation";
    case AttributeMethodErrorCode::insufficient_resource:
        return "insufficient_resource";
    case AttributeMethodErrorCode::non_generative_resource:
        return "non_generative_resource";
    case AttributeMethodErrorCode::missing_generator:
        return "missing_generator";
    case AttributeMethodErrorCode::missing_extrema_field:
        return "missing_extrema_field";
    case AttributeMethodErrorCode::missing_originator:
        return "missing_originator";
    case AttributeMethodErrorCode::unsupported_comparison:
        return "unsupported_comparison";
    case AttributeMethodErrorCode::invalid_restriction:
        return "invalid_restriction";
    case AttributeMethodErrorCode::invalid_numeric_type:
        return "invalid_numeric_type";
    case AttributeMethodErrorCode::numeric_range:
        return "numeric_range";
    case AttributeMethodErrorCode::invalid_batch_shape:
        return "invalid_batch_shape";
    }
    return "unknown";
}

std::string operation_name(AttributeMethodOperation value)
{
    switch (value)
    {
    case AttributeMethodOperation::resolve_update:
        return "resolve_update";
    case AttributeMethodOperation::resolve_comparison:
        return "resolve_comparison";
    case AttributeMethodOperation::resolve_restriction:
        return "resolve_restriction";
    case AttributeMethodOperation::update_resource:
        return "update_resource";
    case AttributeMethodOperation::generate_resource:
        return "generate_resource";
    case AttributeMethodOperation::resolve_extrema:
        return "resolve_extrema";
    case AttributeMethodOperation::generate_extrema:
        return "generate_extrema";
    case AttributeMethodOperation::initialize_constraint:
        return "initialize_constraint";
    case AttributeMethodOperation::calculate_satisfiability:
        return "calculate_satisfiability";
    case AttributeMethodOperation::calculate_satisfiability_batch:
        return "calculate_satisfiability_batch";
    }
    return "unknown";
}

std::string update_operation_name(ResourceUpdateOperation value)
{
    switch (value)
    {
    case ResourceUpdateOperation::add:
        return "add";
    case ResourceUpdateOperation::subtract:
        return "subtract";
    }
    return "unknown";
}

std::string comparison_operation_name(ComparisonOperation value)
{
    switch (value)
    {
    case ComparisonOperation::greater_equal:
        return "greater_equal";
    case ComparisonOperation::less_equal:
        return "less_equal";
    case ComparisonOperation::equal:
        return "equal";
    }
    return "unknown";
}

std::string restriction_name(ConstraintRestriction value)
{
    switch (value)
    {
    case ConstraintRestriction::hard:
        return "hard";
    case ConstraintRestriction::soft:
        return "soft";
    }
    return "unknown";
}

void emit_error(const AttributeMethodException& error)
{
    std::cout << "version=1\n";
    std::cout << "status=error\n";
    std::cout << "error_code=" << error_code_name(error.code()) << '\n';
    std::cout << "operation=" << operation_name(error.operation()) << '\n';
    std::cout << "message_hex=" << hex_encode(error.what()) << '\n';
}

BenchmarkKind benchmark_kind_from_string(std::string_view value)
{
    if (value == "resource_add_int")
    {
        return BenchmarkKind::resource_add_int;
    }
    if (value == "resource_sub_double")
    {
        return BenchmarkKind::resource_sub_double;
    }
    if (value == "guarded_subtract_int")
    {
        return BenchmarkKind::guarded_subtract_int;
    }
    if (value == "guard_failure_int")
    {
        return BenchmarkKind::guard_failure_int;
    }
    if (value == "hard_ge_int")
    {
        return BenchmarkKind::hard_ge_int;
    }
    if (value == "hard_le_double")
    {
        return BenchmarkKind::hard_le_double;
    }
    if (value == "hard_eq_double")
    {
        return BenchmarkKind::hard_eq_double;
    }
    if (value == "soft_ge_int")
    {
        return BenchmarkKind::soft_ge_int;
    }
    throw std::invalid_argument("unknown attribute-method benchmark kind");
}

BenchmarkFixture make_benchmark_fixture(BenchmarkKind kind, std::size_t count)
{
    BenchmarkFixture fixture;
    fixture.virtual_values.reserve(count);
    fixture.physical_inputs.reserve(count);
    fixture.outputs.reserve(count);
    fixture.flags.assign(count, 0U);

    for (std::size_t index = 0; index < count; ++index)
    {
        switch (kind)
        {
        case BenchmarkKind::resource_add_int:
        {
            const std::int64_t virtual_value =
                static_cast<std::int64_t>(index % 17U) - 8;
            const std::int64_t physical_value =
                static_cast<std::int64_t>(index % 101U) - 50;
            fixture.virtual_values.emplace_back(virtual_value);
            fixture.physical_inputs.emplace_back(physical_value);
            fixture.outputs.emplace_back(physical_value);
            break;
        }
        case BenchmarkKind::resource_sub_double:
        {
            const double virtual_value =
                static_cast<double>(static_cast<std::int64_t>(index % 29U) - 14) *
                0.125;
            const double physical_value =
                static_cast<double>(static_cast<std::int64_t>(index % 113U) - 50) *
                0.25;
            fixture.virtual_values.emplace_back(virtual_value);
            fixture.physical_inputs.emplace_back(physical_value);
            fixture.outputs.emplace_back(physical_value);
            break;
        }
        case BenchmarkKind::guarded_subtract_int:
        {
            const std::int64_t virtual_value =
                static_cast<std::int64_t>(index % 7U);
            const std::int64_t physical_value =
                100 + static_cast<std::int64_t>(index % 31U);
            fixture.virtual_values.emplace_back(virtual_value);
            fixture.physical_inputs.emplace_back(physical_value);
            fixture.outputs.emplace_back(physical_value);
            break;
        }
        case BenchmarkKind::guard_failure_int:
        {
            const std::int64_t virtual_value =
                static_cast<std::int64_t>(index % 17U);
            const std::int64_t physical_value =
                static_cast<std::int64_t>(index % 11U);
            fixture.virtual_values.emplace_back(virtual_value);
            fixture.physical_inputs.emplace_back(physical_value);
            fixture.outputs.emplace_back(physical_value);
            break;
        }
        case BenchmarkKind::hard_ge_int:
        case BenchmarkKind::soft_ge_int:
        {
            fixture.virtual_values.emplace_back(
                static_cast<std::int64_t>(index % 97U) - 48);
            fixture.physical_inputs.emplace_back(
                static_cast<std::int64_t>(index % 89U) - 44);
            fixture.outputs.emplace_back(std::int64_t{0});
            break;
        }
        case BenchmarkKind::hard_le_double:
        {
            fixture.virtual_values.emplace_back(
                static_cast<double>(static_cast<std::int64_t>(index % 97U) - 48) *
                0.25);
            fixture.physical_inputs.emplace_back(
                static_cast<double>(static_cast<std::int64_t>(index % 89U) - 44) *
                0.5);
            fixture.outputs.emplace_back(0.0);
            break;
        }
        case BenchmarkKind::hard_eq_double:
        {
            const double virtual_value =
                static_cast<double>(static_cast<std::int64_t>(index % 97U) - 48) *
                0.25;
            const double physical_value = (index % 2U) == 0U
                ? virtual_value
                : virtual_value + 0.25;
            fixture.virtual_values.emplace_back(virtual_value);
            fixture.physical_inputs.emplace_back(physical_value);
            fixture.outputs.emplace_back(0.0);
            break;
        }
        }
    }
    return fixture;
}

void run_benchmark_range(
    BenchmarkKind kind,
    BenchmarkFixture& fixture,
    std::size_t begin,
    std::size_t end)
{
    for (std::size_t index = begin; index < end; ++index)
    {
        switch (kind)
        {
        case BenchmarkKind::resource_add_int:
            fixture.flags[index] = static_cast<std::uint8_t>(
                attribute::update_resource_value(
                    fixture.virtual_values[index],
                    fixture.outputs[index],
                    ResourceUpdateOperation::add,
                    true));
            break;
        case BenchmarkKind::resource_sub_double:
            fixture.flags[index] = static_cast<std::uint8_t>(
                attribute::update_resource_value(
                    fixture.virtual_values[index],
                    fixture.outputs[index],
                    ResourceUpdateOperation::subtract,
                    false));
            break;
        case BenchmarkKind::guarded_subtract_int:
            fixture.flags[index] = static_cast<std::uint8_t>(
                attribute::update_resource_value(
                    fixture.virtual_values[index],
                    fixture.outputs[index],
                    ResourceUpdateOperation::subtract,
                    true));
            break;
        case BenchmarkKind::guard_failure_int:
            try
            {
                fixture.flags[index] = static_cast<std::uint8_t>(
                    attribute::update_resource_value(
                        fixture.virtual_values[index],
                        fixture.outputs[index],
                        ResourceUpdateOperation::subtract,
                        true));
            }
            catch (const AttributeMethodException& error)
            {
                if (error.code() != AttributeMethodErrorCode::insufficient_resource ||
                    error.operation() != AttributeMethodOperation::update_resource)
                {
                    throw;
                }
                fixture.flags[index] = 0U;
            }
            break;
        case BenchmarkKind::hard_ge_int:
        {
            const attribute::SatisfiabilityResult result =
                attribute::calculate_satisfiability_values(
                    fixture.virtual_values[index],
                    fixture.physical_inputs[index],
                    ComparisonOperation::greater_equal,
                    ConstraintRestriction::hard);
            fixture.flags[index] = static_cast<std::uint8_t>(result.flag);
            fixture.outputs[index] = result.offset;
            break;
        }
        case BenchmarkKind::hard_le_double:
        {
            const attribute::SatisfiabilityResult result =
                attribute::calculate_satisfiability_values(
                    fixture.virtual_values[index],
                    fixture.physical_inputs[index],
                    ComparisonOperation::less_equal,
                    ConstraintRestriction::hard);
            fixture.flags[index] = static_cast<std::uint8_t>(result.flag);
            fixture.outputs[index] = result.offset;
            break;
        }
        case BenchmarkKind::hard_eq_double:
        {
            const attribute::SatisfiabilityResult result =
                attribute::calculate_satisfiability_values(
                    fixture.virtual_values[index],
                    fixture.physical_inputs[index],
                    ComparisonOperation::equal,
                    ConstraintRestriction::hard);
            fixture.flags[index] = static_cast<std::uint8_t>(result.flag);
            fixture.outputs[index] = result.offset;
            break;
        }
        case BenchmarkKind::soft_ge_int:
        {
            const attribute::SatisfiabilityResult result =
                attribute::calculate_satisfiability_values(
                    fixture.virtual_values[index],
                    fixture.physical_inputs[index],
                    ComparisonOperation::greater_equal,
                    ConstraintRestriction::soft);
            fixture.flags[index] = static_cast<std::uint8_t>(result.flag);
            fixture.outputs[index] = result.offset;
            break;
        }
        }
    }
}

std::size_t available_cpu_count()
{
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        const int count = CPU_COUNT(&affinity);
        if (count > 0)
        {
            return static_cast<std::size_t>(count);
        }
    }
#endif
    const unsigned int reported = std::thread::hardware_concurrency();
    return reported == 0U ? 1U : static_cast<std::size_t>(reported);
}

std::string affinity_string()
{
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        std::string result;
        for (std::size_t cpu = 0U;
             cpu < static_cast<std::size_t>(CPU_SETSIZE);
             ++cpu)
        {
            if (CPU_ISSET(cpu, &affinity) != 0)
            {
                if (!result.empty())
                {
                    result.push_back(',');
                }
                result += std::to_string(cpu);
            }
        }
        if (!result.empty())
        {
            return result;
        }
    }
#endif
    return "unknown";
}

std::size_t effective_worker_count(std::size_t requested, std::size_t count)
{
    if (count == 0U)
    {
        return 0U;
    }
    const std::size_t desired = requested == 0U
        ? std::min<std::size_t>(8U, available_cpu_count())
        : requested;
    return std::max<std::size_t>(
        1U,
        std::min({desired, count, available_cpu_count()}));
}

void run_benchmark_workers(
    BenchmarkKind kind,
    BenchmarkFixture& fixture,
    std::size_t width)
{
    if (width <= 1U)
    {
        run_benchmark_range(kind, fixture, 0U, fixture.outputs.size());
        return;
    }

    std::vector<std::exception_ptr> errors(width);
    std::vector<std::thread> threads;
    threads.reserve(width - 1U);
    const std::size_t count = fixture.outputs.size();
    const std::size_t base = count / width;
    const std::size_t remainder = count % width;
    std::size_t begin = 0U;

    auto block = [&](std::size_t worker)
    {
        const std::size_t length = base + (worker < remainder ? 1U : 0U);
        const std::size_t end = begin + length;
        const std::size_t block_begin = begin;
        begin = end;
        return std::pair<std::size_t, std::size_t>{block_begin, end};
    };

    const auto caller_block = block(0U);
    try
    {
        for (std::size_t worker = 1U; worker < width; ++worker)
        {
            const auto worker_block = block(worker);
            threads.emplace_back(
                [&, worker, worker_block]
                {
                    try
                    {
                        run_benchmark_range(
                            kind,
                            fixture,
                            worker_block.first,
                            worker_block.second);
                    }
                    catch (...)
                    {
                        errors[worker] = std::current_exception();
                    }
                });
        }
    }
    catch (...)
    {
        for (std::thread& thread : threads)
        {
            thread.join();
        }
        throw;
    }

    try
    {
        run_benchmark_range(
            kind,
            fixture,
            caller_block.first,
            caller_block.second);
    }
    catch (...)
    {
        errors[0] = std::current_exception();
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    for (const std::exception_ptr& error : errors)
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }
}

void fnv_byte(std::uint64_t& checksum, std::uint8_t value)
{
    checksum ^= value;
    checksum *= kFnvPrime;
}

void fnv_u64(std::uint64_t& checksum, std::uint64_t value)
{
    for (std::size_t index = 0; index < sizeof(value); ++index)
    {
        fnv_byte(checksum, static_cast<std::uint8_t>(value & 0xFFU));
        value >>= 8U;
    }
}

void checksum_number(std::uint64_t& checksum, const AttributeNumber& value)
{
    if (const bool* boolean = std::get_if<bool>(&value))
    {
        fnv_byte(checksum, static_cast<std::uint8_t>('b'));
        fnv_u64(checksum, *boolean ? 1U : 0U);
        return;
    }
    if (const std::int64_t* integer = std::get_if<std::int64_t>(&value))
    {
        fnv_byte(checksum, static_cast<std::uint8_t>('i'));
        fnv_u64(checksum, static_cast<std::uint64_t>(*integer));
        return;
    }
    fnv_byte(checksum, static_cast<std::uint8_t>('f'));
    fnv_u64(checksum, double_bits(std::get<double>(value)));
}

std::uint64_t benchmark_checksum(const BenchmarkFixture& fixture)
{
    std::uint64_t checksum = kFnvOffset;
    for (std::size_t index = 0; index < fixture.outputs.size(); ++index)
    {
        fnv_byte(checksum, fixture.flags[index]);
        checksum_number(checksum, fixture.outputs[index]);
    }
    return checksum;
}

int run_double_batch_benchmark(int argc, char** argv)
{
    if (argc != 5)
    {
        throw std::invalid_argument(
            "benchmark_double_batch requires kind count workers");
    }
    const std::string kind_name(argv[2]);
    ComparisonOperation operation = ComparisonOperation::less_equal;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
    if (kind_name == "hard_ge_double")
    {
        operation = ComparisonOperation::greater_equal;
    }
    else if (kind_name == "hard_le_double")
    {
        operation = ComparisonOperation::less_equal;
    }
    else if (kind_name == "hard_eq_double")
    {
        operation = ComparisonOperation::equal;
    }
    else if (kind_name == "soft_ge_double")
    {
        operation = ComparisonOperation::greater_equal;
        restriction = ConstraintRestriction::soft;
    }
    else if (kind_name == "soft_le_double")
    {
        operation = ComparisonOperation::less_equal;
        restriction = ConstraintRestriction::soft;
    }
    else if (kind_name == "soft_eq_double")
    {
        operation = ComparisonOperation::equal;
        restriction = ConstraintRestriction::soft;
    }
    else
    {
        throw std::invalid_argument("unsupported double-batch benchmark kind");
    }
    const std::size_t count = static_cast<std::size_t>(std::stoull(argv[3]));
    const std::size_t requested = static_cast<std::size_t>(std::stoull(argv[4]));
    std::vector<double> virtual_values;
    std::vector<double> physical_values;
    virtual_values.reserve(count);
    physical_values.reserve(count);
    for (std::size_t index = 0U; index < count; ++index)
    {
        virtual_values.push_back(
            static_cast<double>(static_cast<std::int64_t>(index % 97U) - 48)
            * 0.25);
        physical_values.push_back(
            static_cast<double>(static_cast<std::int64_t>(index % 89U) - 44)
            * 0.5);
    }
    std::vector<std::uint8_t> flags(count);
    std::vector<double> offsets(count);
    const std::size_t effective =
        attribute::double_satisfiability_batch_worker_count(
            count, requested);

    const auto start = std::chrono::steady_clock::now();
    attribute::calculate_satisfiability_values_double_batch(
        virtual_values,
        physical_values,
        operation,
        restriction,
        flags,
        offsets,
        requested);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start);

    std::uint64_t checksum = kFnvOffset;
    for (std::size_t index = 0U; index < count; ++index)
    {
        fnv_byte(checksum, flags[index]);
        checksum_number(checksum, AttributeNumber{offsets[index]});
    }
    std::cout << "benchmark_version=1\n";
    std::cout << "kind=" << kind_name << '\n';
    std::cout << "count=" << count << '\n';
    std::cout << "workers=" << requested << '\n';
    std::cout << "effective_workers=" << effective << '\n';
    std::cout << "elapsed_ns=" << elapsed.count() << '\n';
    std::cout << "checksum=" << checksum << '\n';
    std::cout << "output_bytes=" << count * 10U << '\n';
    std::cout << "first_failure=none\n";
    std::cout << "status=PASS\n";
    return 0;
}

int run_double_batch(int argc, char** argv)
{
    if (argc != 7)
    {
        throw std::invalid_argument(
            "batch requires operation restriction workers virtual_bits physical_bits");
    }
    try
    {
        const std::string operation_text(argv[2]);
        const std::string restriction_text(argv[3]);
        const ComparisonOperation operation =
            attribute::comparison_operation_from_string(operation_text);
        const ConstraintRestriction restriction =
            attribute::constraint_restriction_from_string(restriction_text);
        const std::size_t workers =
            static_cast<std::size_t>(std::stoull(argv[4]));
        const std::vector<double> virtual_values =
            parse_double_bits_csv(argv[5]);
        const std::vector<double> physical_values =
            parse_double_bits_csv(argv[6]);
        std::vector<std::uint8_t> flags(virtual_values.size(), 0xA5U);
        std::vector<double> offsets(
            virtual_values.size(), double_from_bits(0x7FF80000000000A5ULL));
        attribute::calculate_satisfiability_values_double_batch(
            virtual_values,
            physical_values,
            operation,
            restriction,
            flags,
            offsets,
            workers);
        std::cout << "version=1\nstatus=ok\nkind=batch\n";
        std::cout << "operation=" << operation_text << '\n';
        std::cout << "restriction=" << restriction_text << '\n';
        std::cout << "workers=" << workers << '\n';
        std::cout << "count=" << virtual_values.size() << '\n';
        std::cout << "flags=" << serialize_flags_csv(flags) << '\n';
        std::cout << "offset_bits=" << serialize_double_bits_csv(offsets) << '\n';
        return 0;
    }
    catch (const AttributeMethodException& error)
    {
        emit_error(error);
        return 0;
    }
}

int run_resolve_update(std::string_view encoded)
{
    try
    {
        const ResourceUpdateOperation value =
            attribute::resource_update_operation_from_string(hex_decode(encoded));
        std::cout << "version=1\nstatus=ok\nkind=resolve_update\nvalue="
                  << update_operation_name(value) << '\n';
        return 0;
    }
    catch (const AttributeMethodException& error)
    {
        emit_error(error);
        return 0;
    }
}

int run_resolve_comparison(std::string_view encoded)
{
    try
    {
        const ComparisonOperation value =
            attribute::comparison_operation_from_string(hex_decode(encoded));
        std::cout << "version=1\nstatus=ok\nkind=resolve_comparison\nvalue="
                  << comparison_operation_name(value) << '\n';
        return 0;
    }
    catch (const AttributeMethodException& error)
    {
        emit_error(error);
        return 0;
    }
}

int run_resolve_restriction(std::string_view encoded)
{
    try
    {
        const ConstraintRestriction value =
            attribute::constraint_restriction_from_string(hex_decode(encoded));
        std::cout << "version=1\nstatus=ok\nkind=resolve_restriction\nvalue="
                  << restriction_name(value) << '\n';
        return 0;
    }
    catch (const AttributeMethodException& error)
    {
        emit_error(error);
        return 0;
    }
}

int run_update(int argc, char** argv)
{
    if (argc != 7)
    {
        throw std::invalid_argument(
            "update requires virtual physical method_hex safe diagnostic_hex");
    }
    try
    {
        const AttributeNumber virtual_value = parse_number(argv[2]);
        AttributeNumber physical_value = parse_number(argv[3]);
        const ResourceUpdateOperation operation =
            attribute::resource_update_operation_from_string(hex_decode(argv[4]));
        const std::string_view safe_text(argv[5]);
        if (safe_text != "0" && safe_text != "1")
        {
            throw std::invalid_argument("safe must be 0 or 1");
        }
        const bool result = attribute::update_resource_value(
            virtual_value,
            physical_value,
            operation,
            safe_text == "1",
            hex_decode(argv[6]));
        std::cout << "version=1\nstatus=ok\nkind=update\nreturn_type=bool\n";
        std::cout << "return_value=" << (result ? "true" : "false") << '\n';
        std::cout << "physical=" << serialize_number(physical_value) << '\n';
        return 0;
    }
    catch (const AttributeMethodException& error)
    {
        emit_error(error);
        return 0;
    }
}

int run_calculate(int argc, char** argv)
{
    if (argc != 6)
    {
        throw std::invalid_argument(
            "calculate requires virtual physical method_hex restriction_hex");
    }
    try
    {
        const AttributeNumber virtual_value = parse_number(argv[2]);
        const AttributeNumber physical_value = parse_number(argv[3]);
        const ComparisonOperation operation =
            attribute::comparison_operation_from_string(hex_decode(argv[4]));
        const ConstraintRestriction restriction =
            attribute::constraint_restriction_from_string(hex_decode(argv[5]));
        const attribute::SatisfiabilityResult result =
            attribute::calculate_satisfiability_values(
                virtual_value, physical_value, operation, restriction);
        std::cout << "version=1\nstatus=ok\nkind=calculate\nflag_type=bool\n";
        std::cout << "flag=" << (result.flag ? "true" : "false") << '\n';
        std::cout << "offset=" << serialize_number(result.offset) << '\n';
        return 0;
    }
    catch (const AttributeMethodException& error)
    {
        emit_error(error);
        return 0;
    }
}

int run_benchmark(int argc, char** argv)
{
    if (argc != 5)
    {
        throw std::invalid_argument("benchmark requires kind count workers");
    }
    const std::string kind_name(argv[2]);
    const BenchmarkKind kind = benchmark_kind_from_string(kind_name);
    const std::size_t count = static_cast<std::size_t>(std::stoull(argv[3]));
    const std::size_t requested = static_cast<std::size_t>(std::stoull(argv[4]));
    const std::size_t effective = effective_worker_count(requested, count);
    BenchmarkFixture fixture = make_benchmark_fixture(kind, count);

    const auto start = std::chrono::steady_clock::now();
    if (effective != 0U)
    {
        run_benchmark_workers(kind, fixture, effective);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start);

    const std::uint64_t checksum = benchmark_checksum(fixture);
    std::string first_failure = "none";
    if (kind == BenchmarkKind::guard_failure_int)
    {
        for (std::size_t index = 0; index < fixture.flags.size(); ++index)
        {
            if (fixture.flags[index] == 0U)
            {
                first_failure = std::to_string(index);
                break;
            }
        }
    }
    std::cout << "benchmark_version=1\n";
    std::cout << "kind=" << kind_name << '\n';
    std::cout << "count=" << count << '\n';
    std::cout << "workers=" << requested << '\n';
    std::cout << "effective_workers=" << effective << '\n';
    std::cout << "elapsed_ns=" << elapsed.count() << '\n';
    std::cout << "checksum=" << checksum << '\n';
    std::cout << "output_bytes=" << count * 10U << '\n';
    std::cout << "first_failure=" << first_failure << '\n';
    std::cout << "status=PASS\n";
    return 0;
}

int run_info()
{
    std::cout << "info_version=1\n";
    std::cout << "compiler=" << __VERSION__ << '\n';
    std::cout << "hardware_concurrency=" << std::thread::hardware_concurrency()
              << '\n';
    std::cout << "available_cpus=" << available_cpu_count() << '\n';
    std::cout << "affinity_cpus=" << affinity_string() << '\n';
    std::cout << "status=PASS\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2)
        {
            throw std::invalid_argument("missing attribute-method harness command");
        }
        const std::string_view command(argv[1]);
        if (command == "resolve_update" && argc == 3)
        {
            return run_resolve_update(argv[2]);
        }
        if (command == "resolve_comparison" && argc == 3)
        {
            return run_resolve_comparison(argv[2]);
        }
        if (command == "resolve_restriction" && argc == 3)
        {
            return run_resolve_restriction(argv[2]);
        }
        if (command == "update")
        {
            return run_update(argc, argv);
        }
        if (command == "calculate")
        {
            return run_calculate(argc, argv);
        }
        if (command == "benchmark")
        {
            return run_benchmark(argc, argv);
        }
        if (command == "benchmark_double_batch")
        {
            return run_double_batch_benchmark(argc, argv);
        }
        if (command == "batch")
        {
            return run_double_batch(argc, argv);
        }
        if (command == "info" && argc == 2)
        {
            return run_info();
        }
        throw std::invalid_argument("unsupported attribute-method harness command");
    }
    catch (const std::exception& error)
    {
        std::cerr << "attribute_method_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
