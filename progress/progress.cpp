// progress.cpp
#include "progress.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <system_error>
#include <type_traits>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace
{
    constexpr std::size_t kBarWidth = 30;
    constexpr double kMinUpdateInterval = 0.05;

    template <typename Integer>
    void append_integer(std::string& output, Integer value)
    {
        std::array<char, std::numeric_limits<Integer>::digits10 + 4> buffer{};
        const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        if (result.ec == std::errc{}) {
            output.append(buffer.data(), result.ptr);
        }
    }

    void append_fixed(std::string& output, double value, int precision)
    {
        std::array<char, 96> buffer{};
        const auto result = std::to_chars(
            buffer.data(), buffer.data() + buffer.size(), value,
            std::chars_format::fixed, precision);
        if (result.ec == std::errc{}) {
            output.append(buffer.data(), result.ptr);
            return;
        }

        // All finite progress values fit above. Keep exceptional values usable
        // on older standard libraries without allocating an iostream.
        const int count = std::snprintf(
            buffer.data(), buffer.size(), "%.*f", precision, value);
        if (count > 0) {
            output.append(buffer.data(), std::min<std::size_t>(
                static_cast<std::size_t>(count), buffer.size() - 1));
        }
    }

    void append_two_digits(std::string& output, std::uint64_t value)
    {
        output.push_back(static_cast<char>('0' + ((value / 10) % 10)));
        output.push_back(static_cast<char>('0' + (value % 10)));
    }

    void append_time(std::string& output, double seconds)
    {
        if (!std::isfinite(seconds) || seconds < 0.0) {
            seconds = 0.0;
        }

        const double maximum_seconds = static_cast<double>(
            std::numeric_limits<std::uint64_t>::max());
        const auto whole_seconds = seconds >= maximum_seconds
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(seconds);
        const std::uint64_t hours = whole_seconds / 3600;
        const std::uint64_t minutes = (whole_seconds / 60) % 60;
        const std::uint64_t remaining_seconds = whole_seconds % 60;

        if (hours > 0) {
            if (hours < 10) {
                output.push_back('0');
            }
            append_integer(output, hours);
            output.push_back(':');
        }
        append_two_digits(output, minutes);
        output.push_back(':');
        append_two_digits(output, remaining_seconds);
    }

    bool is_interactive_terminal() noexcept
    {
#ifdef _WIN32
        const int descriptor = ::_fileno(stdout);
        if (descriptor < 0 || ::_isatty(descriptor) == 0) {
            return false;
        }
        const auto handle = reinterpret_cast<HANDLE>(::_get_osfhandle(descriptor));
        DWORD mode = 0;
        return handle != INVALID_HANDLE_VALUE && ::GetConsoleMode(handle, &mode) != 0;
#else
        return ::isatty(::fileno(stdout)) != 0;
#endif
    }

    void write_spaces(std::ostream& output, std::size_t count)
    {
        static constexpr std::string_view spaces =
            "                                                                ";
        while (count > 0) {
            const std::size_t chunk = std::min(count, spaces.size());
            output.write(spaces.data(), static_cast<std::streamsize>(chunk));
            count -= chunk;
        }
    }
}

Progress::Progress(std::size_t total, const std::string& desc)
    : total_(total)
    , desc_(desc)
    , start_(Clock::now())
    , last_update_time_(start_)
    , next_update_time_(start_ + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(kMinUpdateInterval)))
    , min_update_interval_(kMinUpdateInterval)
    , interactive_terminal_(is_interactive_terminal())
{
    line_buffer_.reserve(desc_.size() + 128);
    postfix_buffer_.reserve(64);
}

Progress::~Progress()
{
    try {
        if (!finished_) {
            finish();
        }
    } catch (...) {}
}

void Progress::set_postfix(const ProgressDict& values)
{
    if (finished_ || postfix_ == values) {
        return;
    }

    postfix_ = values;
    postfix_dirty_ = true;
    display_dirty_ = true;

    // Rendering is intentionally deferred to update()/finish(). Typical loops
    // set metrics immediately before update(); drawing here would format and
    // flush twice and would bypass the update-rate limit.
}

void Progress::format_postfix()
{
    if (!postfix_dirty_) {
        return;
    }

    postfix_keys_.clear();
    postfix_keys_.reserve(postfix_.size());
    for (const auto& entry : postfix_) {
        postfix_keys_.push_back(entry.first);
    }
    std::sort(postfix_keys_.begin(), postfix_keys_.end());

    postfix_buffer_.clear();
    bool first = true;
    for (const auto& key : postfix_keys_) {
        if (!first) {
            postfix_buffer_.append(", ");
        }
        first = false;
        postfix_buffer_.append(key);
        postfix_buffer_.push_back('=');
        std::visit(
            [this](const auto& value) {
                using Value = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, double>) {
                    append_fixed(postfix_buffer_, value, 4);
                } else if constexpr (std::is_same_v<Value, std::string>) {
                    postfix_buffer_.append(value);
                } else {
                    append_integer(postfix_buffer_, value);
                }
            },
            postfix_.at(key));
    }
    postfix_dirty_ = false;
}

void Progress::build_line(std::size_t current, Clock::time_point now)
{
    const double progress = total_ == 0
        ? 1.0
        : static_cast<double>(current) / static_cast<double>(total_);
    const std::size_t filled = std::min<std::size_t>(
        kBarWidth, static_cast<std::size_t>(kBarWidth * progress));

    const auto elapsed_sec = std::chrono::duration<double>(
        now - start_).count();

    const double ips = elapsed_sec > 1e-9
        ? static_cast<double>(current) / elapsed_sec
        : 0.0;
    const double eta = ips > 1e-9
        ? static_cast<double>(total_ - current) / ips
        : 0.0;

    line_buffer_.clear();
    line_buffer_.append(desc_);
    line_buffer_.append(": ");
    const int percentage = static_cast<int>(progress * 100.0);
    if (percentage < 10) {
        line_buffer_.append("  ");
    } else if (percentage < 100) {
        line_buffer_.push_back(' ');
    }
    append_integer(line_buffer_, percentage);
    line_buffer_.append("%|");

    const std::size_t bar_start = line_buffer_.size();
    line_buffer_.append(kBarWidth, ' ');
    std::fill_n(line_buffer_.begin() + static_cast<std::ptrdiff_t>(bar_start), filled, '=');
    if (current < total_ && filled > 0 && filled < kBarWidth) {
        line_buffer_[bar_start + filled - 1] = '>';
    }
    line_buffer_.append("| ");
    append_integer(line_buffer_, current);
    line_buffer_.push_back('/');
    append_integer(line_buffer_, total_);
    line_buffer_.append(" [");
    append_time(line_buffer_, elapsed_sec);
    line_buffer_.push_back('<');
    append_time(line_buffer_, eta);
    line_buffer_.append(", ");
    append_fixed(line_buffer_, ips, 2);
    line_buffer_.append("it/s");

    format_postfix();
    if (!postfix_buffer_.empty()) {
        line_buffer_.append(" | ");
        line_buffer_.append(postfix_buffer_);
    }
}

void Progress::refresh_display(Clock::time_point now)
{
    build_line(current_, now);

    if (interactive_terminal_) {
        std::cout.put('\r');
        std::cout.write(line_buffer_.data(), static_cast<std::streamsize>(line_buffer_.size()));
        if (last_length_ > line_buffer_.size()) {
            write_spaces(std::cout, last_length_ - line_buffer_.size());
        }
        std::cout.flush();
    } else {
        // Logs and redirected streams must never receive cursor-control bytes.
        std::cout.write(line_buffer_.data(), static_cast<std::streamsize>(line_buffer_.size()));
        std::cout.put('\n');
    }

    first_line_printed_ = true;
    last_length_ = line_buffer_.size();
    display_dirty_ = false;
}

bool Progress::should_update(Clock::time_point now) const
{
    if (!display_dirty_) {
        return false;
    }
    return current_ >= total_ || now >= next_update_time_;
}

void Progress::log_error(const std::string& msg) const
{
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    ::localtime_s(&local, &now_c);
#else
    ::localtime_r(&now_c, &local);
#endif
    std::clog << '[' << std::put_time(&local, "%Y-%m-%d %H:%M:%S")
              << "] ERROR: " << msg << '\n';
}

void Progress::update(std::size_t current)
{
    if (finished_) {
        return;
    }

    const std::size_t clamped = std::min(current, total_);
    if (clamped != current_) {
        current_ = clamped;
        display_dirty_ = true;
    }

    const auto now = Clock::now();
    if (!should_update(now)) {
        return;
    }

    refresh_display(now);
    last_update_time_ = now;
    next_update_time_ = now + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(min_update_interval_));
}

bool Progress::update_safe(std::size_t current, bool verbose)
{
    if (finished_) {
        return false;
    }

    if (current > total_) {
        log_error("update_safe: current (" + std::to_string(current) +
                  ") exceeds total (" + std::to_string(total_) + ")");
        current = total_;
    }

    if (current < current_) {
        log_error("update_safe: new current (" + std::to_string(current) +
                  ") < previous (" + std::to_string(current_) + ")");
        return false;
    }

    if (current != current_) {
        current_ = current;
        display_dirty_ = true;
    }

    const auto now = Clock::now();
    const bool need_display = should_update(now);

    if (!need_display) {
        if (verbose) {
            const double elapsed_since_last = std::chrono::duration<double>(
                now - last_update_time_).count();
            std::clog << "[Progress] Skip update: elapsed = " << elapsed_since_last
                      << "s < " << min_update_interval_ << "s\n";
        }
        return true;
    }
    
    refresh_display(now);
    last_update_time_ = now;
    next_update_time_ = now + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(min_update_interval_));

    if (verbose) {
        std::clog << "[Progress] Updated to " << current_ << "/" << total_ << "\n";
    }
    
    return true;
}

void Progress::advance(std::size_t delta)
{
    if (finished_)
        return;

    const std::size_t remaining = total_ - current_;
    update(delta >= remaining ? total_ : current_ + delta);
}

void Progress::finish()
{
    if (finished_) {
        return;
    }
    finished_ = true;

    if (current_ != total_) {
        current_ = total_;
        display_dirty_ = true;
    }

    const auto now = Clock::now();
    if (display_dirty_ || !first_line_printed_) {
        refresh_display(now);
    }
    if (interactive_terminal_ && first_line_printed_) {
        std::cout.put('\n');
    }
    // A completed progress object is a synchronization point even for a
    // redirected, fully buffered stdout stream.
    std::cout.flush();
}

void Progress::close()
{
    finish();
}

std::size_t Progress::current() const noexcept
{
    return current_;
}

std::size_t Progress::total() const noexcept
{
    return total_;
}

bool Progress::finished() const noexcept
{
    return finished_;
}

TqdmProgress::TqdmProgress(
    std::size_t total,
    const std::string& desc)
    : progress_(total, desc)
{
}

void TqdmProgress::update(std::size_t delta)
{
    progress_.advance(delta);
}

void TqdmProgress::set_postfix(const ProgressDict& values)
{
    progress_.set_postfix(values);
}

void TqdmProgress::close()
{
    progress_.finish();
}

void TqdmProgress::finish()
{
    progress_.finish();
}

std::size_t TqdmProgress::current() const noexcept
{
    return progress_.current();
}

std::size_t TqdmProgress::total() const noexcept
{
    return progress_.total();
}

bool TqdmProgress::finished() const noexcept
{
    return progress_.finished();
}
