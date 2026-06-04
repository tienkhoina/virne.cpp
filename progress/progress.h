// progress.h
#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <variant>

using ProgressValue = std::variant<int, long long, double, std::string>;
using ProgressDict = std::unordered_map<std::string, ProgressValue>;

class Progress
{
public:
    Progress(std::size_t total, const std::string& desc);
    ~Progress();
    
    void update(std::size_t current);
    bool update_safe(std::size_t current, bool verbose = false);
    void set_postfix(const ProgressDict& values);
    void finish();

private:
    std::string build_line(std::size_t current) const;
    std::string format_postfix() const;
    bool should_update(double& elapsed_since_last) const;
    void log_error(const std::string& msg) const;
    void refresh_display();

private:
    std::size_t total_;
    std::string desc_;
    ProgressDict postfix_;
    
    std::size_t current_ = 0;
    std::size_t last_length_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_update_time_;
    double min_update_interval_;
    
    bool first_line_printed_ = false;
    bool finished_ = false;
};