#include "stats.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace virne::utils
{

double SystemWallClock::operator()() const noexcept
{
    const auto elapsed =
        std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(elapsed).count();
}

namespace detail
{
namespace
{

std::string format_elapsed_seconds(const double elapsed_seconds)
{
    if (std::isnan(elapsed_seconds))
    {
        return "nan";
    }

    if (std::isinf(elapsed_seconds))
    {
        return std::signbit(elapsed_seconds) ? "-inf" : "inf";
    }

    // A finite double needs at most 309 decimal integer digits.  This leaves
    // ample room for its sign, decimal point, four fractional digits, and any
    // implementation bookkeeping required by to_chars.
    std::array<char, 512> buffer{};
    const auto formatted = std::to_chars(
        buffer.data(),
        buffer.data() + buffer.size(),
        elapsed_seconds,
        std::chars_format::fixed,
        4);

    if (formatted.ec != std::errc{})
    {
        throw std::runtime_error(
            "failed to format running-time duration");
    }

    return std::string(buffer.data(), formatted.ptr);
}

} // namespace

std::string make_running_time_prefix(
    const std::string_view function_name)
{
    std::string prefix("Running time of ");
    if (!function_name.empty())
    {
        prefix.append(function_name.data(), function_name.size());
    }
    prefix.append(": ");
    return prefix;
}

void emit_running_time(
    std::ostream& output,
    const std::string_view cached_prefix,
    const double elapsed_seconds)
{
    const std::string formatted =
        format_elapsed_seconds(elapsed_seconds);

    std::string line;
    if (!cached_prefix.empty())
    {
        line.append(cached_prefix.data(), cached_prefix.size());
    }
    line.append(formatted);
    line.append("s\n");

    constexpr auto maximum_stream_size =
        (std::numeric_limits<std::streamsize>::max)();
    if (line.size() > static_cast<std::size_t>(maximum_stream_size))
    {
        throw std::ios_base::failure(
            "running-time output exceeds stream size");
    }

    output.write(
        line.data(),
        static_cast<std::streamsize>(line.size()));
    if (!output)
    {
        throw std::ios_base::failure(
            "failed to write running-time output");
    }
}

} // namespace detail
} // namespace virne::utils
