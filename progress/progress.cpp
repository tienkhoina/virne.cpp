// progress.cpp
#include "progress.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <cstdio>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

// ============================================================================
// Hỗ trợ định dạng thời gian
// ============================================================================
namespace
{
    std::string format_time(double seconds)
    {
        if (seconds < 0) seconds = 0;
        const int s = static_cast<int>(seconds);
        const int h = s / 3600;
        const int m = (s % 3600) / 60;
        const int sec = s % 60;

        std::ostringstream oss;
        if (h > 0)
            oss << std::setw(2) << std::setfill('0') << h << ":";
        oss << std::setw(2) << std::setfill('0') << m << ":"
            << std::setw(2) << std::setfill('0') << sec;
        return oss.str();
    }

    bool is_interactive_terminal()
    {
        #ifdef _WIN32
            return false;
        #else
            return isatty(fileno(stdout)) != 0;
        #endif
    }
}

// ============================================================================
// Constructor
// ============================================================================
Progress::Progress(std::size_t total, const std::string& desc)
    : total_(total)
    , desc_(desc)
    , start_(std::chrono::steady_clock::now())
    , last_update_time_(start_)
    , min_update_interval_(0.05)
    , first_line_printed_(false)
{
    if (total_ == 0) {
        log_error("Progress: total is zero, progress will be meaningless.");
    }
}

// ============================================================================
// Destructor
// ============================================================================
Progress::~Progress()
{
    try {
        if (!finished_) {
            finish();
        }
    } catch (...) {}
}

// ============================================================================
// Cập nhật postfix
// ============================================================================
void Progress::set_postfix(const ProgressDict& values)
{
    postfix_ = values;
    // Ngay lập tức cập nhật hiển thị
    if (current_ > 0 || first_line_printed_) {
        refresh_display();
    }
}

// ============================================================================
// Định dạng postfix
// ============================================================================
std::string Progress::format_postfix() const
{
    std::ostringstream oss;
    bool first = true;
    for (const auto& [key, value] : postfix_)
    {
        if (!first) oss << ", ";
        first = false;
        oss << key << "=";
        std::visit([&oss](const auto& v) { 
            oss << std::fixed << std::setprecision(4) << v; 
        }, value);
    }
    return oss.str();
}

// ============================================================================
// Xây dựng dòng thanh tiến trình
// ============================================================================
std::string Progress::build_line(std::size_t current) const
{
    if (total_ == 0) {
        return desc_ + ": 100%|" + std::string(20, '#') + "| ?/? [--:--<--:--, --it/s]";
    }

    constexpr int BAR_WIDTH = 30;  // Tăng lên cho đẹp
    const double progress = static_cast<double>(current) / static_cast<double>(total_);
    const int filled = static_cast<int>(BAR_WIDTH * progress);
    const int safe_filled = (filled < 0) ? 0 : (filled > BAR_WIDTH ? BAR_WIDTH : filled);

    const auto elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_).count();

    const double ips = (elapsed_sec > 1e-9) ? (current / elapsed_sec) : 0.0;
    const double eta = (ips > 1e-9) ? ((total_ - current) / ips) : 0.0;

    std::ostringstream bar_stream;
    for (int i = 0; i < BAR_WIDTH; ++i) {
        bar_stream << (i < safe_filled ? '=' : ' ');
    }
    std::string bar = bar_stream.str();
    
    // Thêm mũi tên ở cuối thanh nếu chưa hoàn thành
    if (current < total_ && safe_filled > 0 && safe_filled < BAR_WIDTH) {
        bar[safe_filled - 1] = '>';
    }

    std::ostringstream oss;
    oss << "\r" << desc_ << ": "
        << std::setw(3) << static_cast<int>(progress * 100.0) << "%|"
        << bar << "| "
        << current << "/" << total_
        << " [" << format_time(elapsed_sec) << "<" << format_time(eta)
        << ", " << std::fixed << std::setprecision(2) << ips << "it/s";

    const std::string postfix_str = format_postfix();
    if (!postfix_str.empty()) {
        oss << " | " << postfix_str;
    }

    oss << "  ";  // Thêm spaces để xóa các ký tự thừa
    return oss.str();
}

// ============================================================================
// Làm mới hiển thị (quan trọng nhất)
// ============================================================================
void Progress::refresh_display()
{
    const std::string line = build_line(current_);
    
    // Sử dụng ANSI escape codes để đảm bảo luôn trên cùng một dòng
    if (first_line_printed_) {
        // Move cursor to beginning of line and clear line
        std::cout << "\033[1A\r\033[2K";
    }
    
    std::cout << line << std::flush;
    first_line_printed_ = true;
    last_length_ = line.size();
}

// ============================================================================
// Kiểm tra có nên update không
// ============================================================================
bool Progress::should_update(double& elapsed_since_last) const
{
    if (last_update_time_.time_since_epoch().count() == 0) {
        elapsed_since_last = min_update_interval_ + 1.0;
        return true;
    }

    auto now = std::chrono::steady_clock::now();
    elapsed_since_last = std::chrono::duration<double>(now - last_update_time_).count();

    return (current_ >= total_) || (elapsed_since_last >= min_update_interval_);
}

// ============================================================================
// Ghi log lỗi
// ============================================================================
void Progress::log_error(const std::string& msg) const
{
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::clog << "["
              << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S")
              << "] ERROR: " << msg << std::endl;
}

// ============================================================================
// update() - phiên bản chính
// ============================================================================
void Progress::update(std::size_t current)
{
    current_ = current;
    
    double elapsed_since_last = 0.0;
    if (!should_update(elapsed_since_last) && current < total_) {
        return;
    }
    
    refresh_display();
    last_update_time_ = std::chrono::steady_clock::now();
}

// ============================================================================
// update_safe() - phiên bản an toàn
// ============================================================================
bool Progress::update_safe(std::size_t current, bool verbose)
{
    if (finished_) return false;
    
    if (total_ == 0) {
        log_error("update_safe: total_ is zero");
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

    current_ = current;
    
    double elapsed_since_last = 0.0;
    bool need_display = should_update(elapsed_since_last);
    
    if (!need_display && current < total_) {
        if (verbose) {
            std::clog << "[Progress] Skip update: elapsed = " << elapsed_since_last
                      << "s < " << min_update_interval_ << "s\n";
        }
        return true;
    }
    
    refresh_display();
    last_update_time_ = std::chrono::steady_clock::now();
    
    if (verbose) {
        std::clog << "[Progress] Updated to " << current_ << "/" << total_ << "\n";
    }
    
    return true;
}

// ============================================================================
// finish() - hoàn thành
// ============================================================================
void Progress::finish()
{
    if (finished_) return;
    finished_ = true;
    
    current_ = total_;
    refresh_display();
    std::cout << std::endl;
}