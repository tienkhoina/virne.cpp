// progress.h
#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

using ProgressValue = std::variant<int, long long, double, std::string>;
using ProgressDict = std::unordered_map<std::string, ProgressValue>;

class Progress
{
public:
    Progress(std::size_t total, const std::string& desc);
    ~Progress();

    // Existing absolute-position API.
    void update(std::size_t current);
    bool update_safe(std::size_t current, bool verbose = false);

    // Incremental API for loops ported from tqdm.update(n). Saturates at total
    // without overflowing size_t.
    void advance(std::size_t delta = 1);

    void set_postfix(const ProgressDict& values);
    void finish();
    void close();

    std::size_t current() const noexcept;
    std::size_t total() const noexcept;
    bool finished() const noexcept;

private:
    using Clock = std::chrono::steady_clock;

    void build_line(std::size_t current, Clock::time_point now);
    void format_postfix();
    bool should_update(Clock::time_point now) const;
    void log_error(const std::string& msg) const;
    void refresh_display(Clock::time_point now);

private:
    std::size_t total_;
    std::string desc_;
    ProgressDict postfix_;
    std::vector<std::string> postfix_keys_;
    std::string postfix_buffer_;
    std::string line_buffer_;
    
    std::size_t current_ = 0;
    std::size_t last_length_ = 0;
    Clock::time_point start_;
    Clock::time_point last_update_time_;
    Clock::time_point next_update_time_;
    double min_update_interval_;
    
    bool interactive_terminal_ = false;
    bool first_line_printed_ = false;
    bool display_dirty_ = true;
    bool postfix_dirty_ = true;
    bool finished_ = false;
};

// Drop-in semantic adapter for the tqdm calls used by the original Virne:
// update(n) is a delta, set_postfix() forwards unchanged, and close() finishes.
// Progress itself intentionally retains update(absolute) for compatibility.
class TqdmProgress
{
public:
    explicit TqdmProgress(
        std::size_t total,
        const std::string& desc = {});

    void update(std::size_t delta = 1);
    void set_postfix(const ProgressDict& values);
    void close();
    void finish();

    std::size_t current() const noexcept;
    std::size_t total() const noexcept;
    bool finished() const noexcept;

private:
    Progress progress_;
};
