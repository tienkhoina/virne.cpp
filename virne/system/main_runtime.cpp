#include "main_runtime.h"

#include "offline_system.h"
#include "online_system.h"

#include "../solver/heuristic/registry.h"
#include "../solver/exact/exact_solver.h"
#include "../../random/random_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace virne::system {
namespace {

using Clock = std::chrono::steady_clock;

bool stderr_is_terminal() noexcept {
#if defined(_WIN32)
    return ::_isatty(::_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) != 0;
#endif
}

std::optional<std::size_t> stderr_terminal_columns() noexcept {
#if defined(_WIN32)
    const auto native = ::_get_osfhandle(::_fileno(stderr));
    if (native != -1) {
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (::GetConsoleScreenBufferInfo(
                reinterpret_cast<HANDLE>(native), &info) != 0) {
            const SHORT columns =
                info.srWindow.Right - info.srWindow.Left + 1;
            if (columns > 0) {
                return static_cast<std::size_t>(columns);
            }
        }
    }
#else
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 &&
        size.ws_col != 0U) {
        return static_cast<std::size_t>(size.ws_col);
    }
#endif
    // Some PTY layers expose a zero-sized window while still exporting the
    // shell's width. Do not guess when neither source is available: repeated
    // unknown-width frames could wrap and permanently consume output rows.
    const char* value = nullptr;
#if defined(_WIN32)
    char value_buffer[32]{};
    const DWORD value_length = ::GetEnvironmentVariableA(
        "COLUMNS", value_buffer, static_cast<DWORD>(sizeof(value_buffer)));
    if (value_length > 0U && value_length < sizeof(value_buffer)) {
        value = value_buffer;
    }
#else
    value = std::getenv("COLUMNS");
#endif
    if (value != nullptr) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end != value && *end == '\0' && parsed >= 2UL &&
            parsed <= 10000UL) {
            return static_cast<std::size_t>(parsed);
        }
    }
    return std::nullopt;
}

class ConsoleProgress final : public SystemProgressSink {
public:
    ConsoleProgress(
        MainProgressConfig config,
        std::string_view solver_name,
        std::ostream& output)
        : config_(config),
          solver_name_(solver_name),
          output_(output),
          interactive_(stderr_is_terminal()) {
        if (interactive_) {
            if (const auto columns = stderr_terminal_columns()) {
                terminal_columns_ = *columns;
            }
        }
    }

    ~ConsoleProgress() override {
        if (active_ && interactive_) {
            flush_suffix("\n");
        }
    }

    void begin_epoch(
        std::size_t epoch_index,
        std::size_t total) override {
        if (!config_.enabled) {
            return;
        }
        epoch_index_ = epoch_index;
        total_ = total;
        active_ = true;
        previous_width_ = 0U;
        has_last_rendered_update_ = false;
        start_ = Clock::now();
        last_render_ = start_;
        if (interactive_) {
            render(SystemProgressUpdate{
                epoch_index, 0U, total, 0, 0, 0.0}, false);
        }
    }

    void update(const SystemProgressUpdate& update_value) override {
        if (!config_.enabled || !active_ || !interactive_) {
            return;
        }
        // The systems call end_epoch immediately after their terminal update.
        // Let that call emit the newline-terminated frame in one write/flush
        // instead of flushing the same completed frame twice.
        if (update_value.completed >= update_value.total) {
            return;
        }
        const auto now = Clock::now();
        const auto since_last =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_render_).count();
        const bool interval_elapsed = since_last >= 0 &&
            static_cast<std::uint64_t>(since_last) >=
                config_.minimum_interval_ms;
        if (interval_elapsed) {
            render(update_value, false);
            last_render_ = now;
        }
    }

    void end_epoch(const SystemProgressUpdate& update_value) override {
        if (!config_.enabled || !active_) {
            return;
        }
        if (interactive_ && has_last_rendered_update_ &&
            same_update(last_rendered_update_, update_value)) {
            flush_suffix("\n");
            previous_width_ = 0U;
        } else {
            render(update_value, true);
        }
        active_ = false;
    }

private:
    enum class LineStyle : std::uint8_t {
        full,
        compact,
        essential,
    };

    static bool same_update(
        const SystemProgressUpdate& left,
        const SystemProgressUpdate& right) noexcept {
        return left.epoch_index == right.epoch_index &&
            left.completed == right.completed && left.total == right.total &&
            left.success_count == right.success_count &&
            left.inservice_count == right.inservice_count &&
            left.long_term_r2c_ratio == right.long_term_r2c_ratio;
    }

    std::string format_line(
        const LineStyle style,
        const std::string_view solver,
        const std::size_t bar_width,
        const std::size_t completed,
        const std::size_t total,
        const int percentage,
        const double elapsed_seconds,
        const double rate,
        const double acceptance,
        const double r2c,
        const std::size_t inservice) const {
        const std::size_t filled = std::min(
            bar_width,
            total == 0U
                ? bar_width
                : static_cast<std::size_t>(
                      static_cast<double>(completed) /
                      static_cast<double>(total) *
                      static_cast<double>(bar_width)));

        std::ostringstream line;
        if (style == LineStyle::full) {
            line << "Running with " << solver << " in epoch "
                 << epoch_index_ << ": ";
        } else {
            line << solver << " e" << epoch_index_ << ' ';
        }
        line << std::setw(3) << percentage << "%|"
             << std::string(filled, '#')
             << std::string(bar_width - filled, '-') << "| "
             << completed << '/' << total;

        if (style == LineStyle::full) {
            line << " [" << std::fixed << std::setprecision(2)
                 << elapsed_seconds << "s, " << rate
                 << "it/s, ac=" << acceptance << ", r2c=" << r2c
                 << ", inservice=" << std::setw(5) << std::setfill('0')
                 << inservice << ']';
        } else if (style == LineStyle::compact) {
            line << ' ' << std::fixed << std::setprecision(1)
                 << elapsed_seconds << "s " << rate << "/s"
                 << std::setprecision(2) << " ac=" << acceptance
                 << " r2c=" << r2c << " in=" << inservice;
        } else {
            line << std::fixed << std::setprecision(2)
                 << " ac=" << acceptance << " r2c=" << r2c
                 << " in=" << inservice;
        }
        return line.str();
    }

    std::string fitted_line(
        const LineStyle style,
        const std::string_view solver,
        const std::size_t maximum_visible,
        const std::size_t completed,
        const std::size_t total,
        const int percentage,
        const double elapsed_seconds,
        const double rate,
        const double acceptance,
        const double r2c,
        const std::size_t inservice) const {
        std::size_t bar_width = config_.width;
        std::string result = format_line(
            style,
            solver,
            bar_width,
            completed,
            total,
            percentage,
            elapsed_seconds,
            rate,
            acceptance,
            r2c,
            inservice);
        if (result.size() > maximum_visible && bar_width > 1U) {
            const std::size_t excess = result.size() - maximum_visible;
            bar_width -= std::min(excess, bar_width - 1U);
            result = format_line(
                style,
                solver,
                bar_width,
                completed,
                total,
                percentage,
                elapsed_seconds,
                rate,
                acceptance,
                r2c,
                inservice);
        }
        return result;
    }

    std::string render_line(
        const SystemProgressUpdate& update_value,
        const std::size_t completed,
        const std::size_t total,
        const double fraction,
        const double elapsed_seconds,
        const double rate,
        const double acceptance) const {
        const int percentage = total == 0U || completed >= total
            ? 100
            : static_cast<int>(fraction * 100.0);
        if (!interactive_) {
            return format_line(
                LineStyle::full,
                solver_name_,
                config_.width,
                completed,
                total,
                percentage,
                elapsed_seconds,
                rate,
                acceptance,
                update_value.long_term_r2c_ratio,
                update_value.inservice_count);
        }

        const std::size_t maximum_visible = terminal_columns_ > 1U
            ? terminal_columns_ - 1U
            : 0U;
        const auto make = [&](const LineStyle style,
                              const std::string_view solver) {
            return fitted_line(
                style,
                solver,
                maximum_visible,
                completed,
                total,
                percentage,
                elapsed_seconds,
                rate,
                acceptance,
                update_value.long_term_r2c_ratio,
                update_value.inservice_count);
        };

        std::string result = make(LineStyle::full, solver_name_);
        if (result.size() > maximum_visible) {
            result = make(LineStyle::compact, solver_name_);
        }
        if (result.size() > maximum_visible) {
            result = make(LineStyle::essential, solver_name_);
        }
        if (result.size() > maximum_visible) {
            const std::size_t excess = result.size() - maximum_visible;
            if (solver_name_.size() > excess + 1U) {
                std::string abbreviated = solver_name_.substr(
                    0U, solver_name_.size() - excess - 1U);
                abbreviated.push_back('~');
                result = make(LineStyle::essential, abbreviated);
            }
        }
        if (result.size() > maximum_visible) {
            result.resize(maximum_visible);
        }
        return result;
    }

    void render(const SystemProgressUpdate& update_value, bool final) {
        const std::size_t total = total_ == 0U ? update_value.total : total_;
        const std::size_t completed = total == 0U
            ? 0U
            : std::min(update_value.completed, total);
        const double fraction = total == 0U
            ? 1.0
            : static_cast<double>(completed) / static_cast<double>(total);
        const double elapsed_seconds =
            std::chrono::duration<double>(Clock::now() - start_).count();
        const double rate = elapsed_seconds == 0.0
            ? 0.0
            : static_cast<double>(completed) / elapsed_seconds;

        const double acceptance = update_value.completed == 0U
            ? 0.0
            : static_cast<double>(update_value.success_count) /
                static_cast<double>(update_value.completed);
        if (interactive_) {
            // Refreshing here handles terminal resize and Docker/ConPTY
            // attaching a real window after process start. This is outside
            // the solver hot path and runs only on rate-limited frames.
            if (const auto columns = stderr_terminal_columns()) {
                terminal_columns_ = *columns;
            } else if (terminal_columns_ == 0U) {
                if (!final) {
                    return;
                }
                // An unknown-size TTY gets one compact final frame only, so
                // even a narrow viewport cannot produce refresh-line spam.
                terminal_columns_ = 80U;
            }
        }
        const std::string rendered = render_line(
            update_value,
            completed,
            total,
            fraction,
            elapsed_seconds,
            rate,
            acceptance);
        if (interactive_) {
            const std::size_t maximum_visible = terminal_columns_ > 1U
                ? terminal_columns_ - 1U
                : 0U;
            const std::size_t clear_width =
                std::min(previous_width_, maximum_visible);
            render_buffer_.clear();
            render_buffer_.reserve(
                5U + std::max(rendered.size(), clear_width) +
                static_cast<std::size_t>(final));
            // Clear the current row before drawing it again. The visible
            // frame is one column shorter than the TTY, preventing automatic
            // wrapping from advancing the cursor beyond carriage-return
            // reach.
#if defined(_WIN32)
            render_buffer_.push_back('\r');
            render_buffer_.append(rendered);
            if (clear_width > rendered.size()) {
                render_buffer_.append(
                    clear_width - rendered.size(), ' ');
            }
#else
            render_buffer_.append("\r\x1b[2K");
            render_buffer_.append(rendered);
#endif
            if (final) {
                render_buffer_.push_back('\n');
            }
            flush_buffer();
            previous_width_ = final ? 0U : rendered.size();
        } else if (final) {
            render_buffer_.assign(rendered);
            render_buffer_.push_back('\n');
            flush_buffer();
        }
        last_rendered_update_ = update_value;
        has_last_rendered_update_ = true;
    }

    void flush_suffix(std::string_view suffix) {
        render_buffer_.assign(suffix);
        flush_buffer();
    }

    void flush_buffer() {
        output_.write(
            render_buffer_.data(),
            static_cast<std::streamsize>(render_buffer_.size()));
        output_.flush();
    }

    MainProgressConfig config_;
    std::string solver_name_;
    std::ostream& output_;
    bool interactive_ = false;
    std::size_t terminal_columns_ = 0U;
    bool active_ = false;
    std::size_t epoch_index_ = 0U;
    std::size_t total_ = 0U;
    std::size_t previous_width_ = 0U;
    std::string render_buffer_;
    SystemProgressUpdate last_rendered_update_;
    bool has_last_rendered_update_ = false;
    Clock::time_point start_{};
    Clock::time_point last_render_{};
};

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    const auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    return static_cast<std::uint64_t>(duration.count());
}

network::attribute::CheckingLevel default_checking_level(
    const network::attribute::AttributeFactorySpec& spec) noexcept {
    using network::attribute::AttributeKind;
    using network::attribute::AttributeOwner;
    using network::attribute::CheckingLevel;
    if (spec.owner == AttributeOwner::node) {
        return CheckingLevel::node;
    }
    if (spec.kind == AttributeKind::latency) {
        return CheckingLevel::path;
    }
    return CheckingLevel::link;
}

network::PhysicalNetwork build_physical_network(
    const MainConfig& config,
    RandomContext& random) {
    const bool load = config.load_physical_network &&
        std::filesystem::exists(config.physical_dataset_dir);
    if (load) {
        return network::PhysicalNetwork::load_dataset(
            config.physical_dataset_dir.string(),
            config.physical_dataset_file);
    }
    auto options = config.workers.physical;
    options.seed = config.seed;
    return network::PhysicalNetwork::from_setting(
        config.physical_setting,
        random,
        options);
}

network::VirtualNetworkRequestSimulator build_virtual_simulator(
    const MainConfig& config,
    RandomContext& random) {
    const bool load = config.load_virtual_networks &&
        std::filesystem::exists(config.virtual_dataset_dir);
    if (load) {
        return network::VirtualNetworkRequestSimulator::load_dataset(
            config.virtual_dataset_dir,
            config.workers.simulator.io_workers);
    }
    auto simulator = network::VirtualNetworkRequestSimulator::from_setting(
        config.virtual_setting,
        random,
        config.seed);
    simulator.renew(
        random,
        true,
        true,
        std::nullopt,
        config.workers.simulator);
    return simulator;
}

core::EnvironmentConfig environment_config(
    const MainConfig& config,
    const RuntimeSelections& selections,
    std::optional<std::size_t> stage_index = std::nullopt) {
    core::EnvironmentConfig result;
    result.controller = selections.controller;
    result.counter = selections.counter;
    result.recorder = config.recorder_config;
    if (stage_index.has_value()) {
        result.recorder.run_id += "-stage-" +
            std::to_string(*stage_index);
    }
    result.workers = config.workers.environment;
    result.admission = config.admission;
    return result;
}

void save_datasets(
    const MainConfig& config,
    const network::PhysicalNetwork& physical,
    const network::VirtualNetworkRequestSimulator& simulator) {
    if (config.save_physical_network) {
        physical.save_dataset(
            config.physical_dataset_dir.string(),
            config.physical_dataset_file);
    }
    if (config.save_virtual_networks) {
        simulator.save_dataset(
            config.virtual_dataset_dir,
            config.workers.simulator.io_workers);
    }
}

void save_environment_records(
    const MainConfig& config,
    const core::SolutionStepEnvironment& environment,
    const SystemRunResult& run) {
    if (!config.save_records || run.epochs.empty()) {
        return;
    }
    environment.recorder().save_records(
        config.records_file_name,
        core::RecorderOptions{config.workers.environment.recorder_workers});
    environment.recorder().append_summary(
        run.epochs.back().summary,
        config.summary_file_name);
}

SystemRunConfig system_run_config(const MainConfig& config) {
    SystemRunConfig result;
    result.num_simulations = config.num_simulations;
    result.seed = config.seed;
    result.renew_virtual_networks = config.renew_virtual_networks;
    result.renew_event_schedule = config.renew_event_schedule;
    result.simulator_workers = config.workers.simulator;
    result.capture_solutions = config.capture_solutions;
    return result;
}

std::string_view failure_reason_name(
    core::EnvironmentFailureReason reason) noexcept {
    switch (reason) {
    case core::EnvironmentFailureReason::none:
        return "none";
    case core::EnvironmentFailureReason::early_rejection:
        return "early_rejection";
    case core::EnvironmentFailureReason::placement:
        return "placement";
    case core::EnvironmentFailureReason::routing:
        return "routing";
    case core::EnvironmentFailureReason::unknown:
        return "unknown";
    }
    return "unknown";
}

void write_json_string(std::ostream& output, std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    output.put('"');
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        switch (byte) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (byte < 0x20U) {
                output << "\\u00" << hex[byte >> 4U] << hex[byte & 0x0fU];
            } else {
                output.put(static_cast<char>(byte));
            }
        }
    }
    output.put('"');
}

void write_json_double(std::ostream& output, double value) {
    if (std::isfinite(value)) {
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << value;
    } else if (std::isnan(value)) {
        write_json_string(output, "nan");
    } else if (value < 0.0) {
        write_json_string(output, "-inf");
    } else {
        write_json_string(output, "inf");
    }
}

void write_solution(std::ostream& output, const core::Solution& solution) {
    output << "{\"result\":" << (solution.result ? "true" : "false")
           << ",\"place_result\":"
           << (solution.place_result ? "true" : "false")
           << ",\"route_result\":"
           << (solution.route_result ? "true" : "false")
           << ",\"early_rejection\":"
           << (solution.early_rejection ? "true" : "false")
           << ",\"node_slots\":[";
    bool first = true;
    for (const auto& entry : solution.node_slots.entries()) {
        if (!first) {
            output.put(',');
        }
        first = false;
        output << '[' << entry.key << ',' << entry.value << ']';
    }
    output << "],\"link_paths\":[";
    first = true;
    for (const auto& entry : solution.link_paths.entries()) {
        if (!first) {
            output.put(',');
        }
        first = false;
        output << "[[" << entry.key.source << ',' << entry.key.target
               << "],[";
        bool first_link = true;
        for (const auto link : entry.value) {
            if (!first_link) {
                output.put(',');
            }
            first_link = false;
            output << '[' << link.source << ',' << link.target << ']';
        }
        output << "]]";
    }
    output << "],\"v_net_cost\":";
    write_json_double(output, solution.v_net_cost);
    output << ",\"v_net_revenue\":";
    write_json_double(output, solution.v_net_revenue);
    output << ",\"v_net_r2c_ratio\":";
    write_json_double(output, solution.v_net_r2c_ratio);
    output << ",\"hard_constraint_violation\":";
    write_json_double(
        output, solution.v_net_total_hard_constraint_violation);
    output << ",\"description\":";
    write_json_string(output, solution.description);
    output.put('}');
}

void write_counter_summary(
    std::ostream& output,
    const core::CounterSummary& summary) {
    output << "{\"acceptance_rate\":";
    write_json_double(output, summary.acceptance_rate);
    output << ",\"average_r2c_ratio\":";
    write_json_double(output, summary.average_r2c_ratio);
    output << ",\"long_term_time_r2c_ratio\":";
    write_json_double(output, summary.long_term_time_r2c_ratio);
    output << ",\"success_count\":" << summary.success_count
           << ",\"early_rejection_count\":"
           << summary.early_rejection_count
           << ",\"place_failure_count\":"
           << summary.place_failure_count
           << ",\"route_failure_count\":"
           << summary.route_failure_count
           << ",\"total_cost\":";
    write_json_double(output, summary.total_cost);
    output << ",\"total_revenue\":";
    write_json_double(output, summary.total_revenue);
    output << ",\"total_time_cost\":";
    write_json_double(output, summary.total_time_cost);
    output << ",\"total_time_revenue\":";
    write_json_double(output, summary.total_time_revenue);
    output << ",\"long_term_r2c_ratio\":";
    write_json_double(output, summary.long_term_r2c_ratio);
    output << ",\"total_simulation_time\":";
    write_json_double(output, summary.total_simulation_time);
    output.put('}');
}

void log_run_start(
    core::Logger& logger,
    const MainConfig& config,
    solver::SolverCategory category) {
    std::ostringstream message;
    message << "Use " << config.solver_name << " Solver (Type = "
            << solver::solver_category_name(category) << "), "
            << "mode=" << system_mode_name(config.mode);
    logger.info(message.str());
}

void log_resolved_config(core::Logger& logger, const MainConfig& config) {
    if (config.resolved_config_yaml.empty()) {
        return;
    }
    std::string message;
    message.reserve(
        8U + config.resolved_config_yaml.size() +
        config.resolved_native_config_yaml.size() + 17U);
    message.append("Config:\n");
    message.append(config.resolved_config_yaml);
    if (!config.resolved_native_config_yaml.empty()) {
        message.append("\nNative config:\n");
        message.append(config.resolved_native_config_yaml);
        message.push_back('\n');
    }
    logger.info(message);
}

void log_run_complete(core::Logger& logger, const MainRunReport& report) {
    std::size_t processed = 0U;
    std::size_t accepted = 0U;
    for (const auto& epoch : report.run.epochs) {
        processed += epoch.arrival_steps;
        accepted += epoch.accepted;
    }
    const double acceptance = processed == 0U
        ? 0.0
        : static_cast<double>(accepted) / static_cast<double>(processed);
    const double r2c = report.run.epochs.empty()
        ? 0.0
        : report.run.epochs.back().summary.long_term_r2c_ratio;
    std::ostringstream message;
    message << "Complete: requests=" << processed
            << ", accepted=" << accepted
            << ", ac=" << std::fixed << std::setprecision(2) << acceptance
            << ", r2c=" << r2c
            << ", run_ms="
            << static_cast<double>(report.run_time_ns) / 1.0e6;
    logger.info(message.str());
}

} // namespace

RuntimeSelections runtime_selections_from_virtual_config(
    const network::VirtualNetworkSimulationConfig& config,
    bool reusable) {
    using network::attribute::AttributeKind;
    using network::attribute::CheckingLevel;
    using network::attribute::ConstraintRestriction;

    RuntimeSelections result;
    result.controller.reusable = reusable;
    std::vector<core::controller::ResourceId> node_resources;
    std::vector<core::controller::ResourceId> link_resources;

    for (std::size_t index = 0U;
         index < config.node_attribute_specs.size();
         ++index) {
        const auto& spec = config.node_attribute_specs[index];
        const auto id = static_cast<network::attribute::AttributeRegistryId>(
            index);
        if (spec.kind == AttributeKind::resource) {
            node_resources.push_back(id);
        }
        if (spec.kind == AttributeKind::resource ||
            spec.kind == AttributeKind::position) {
            result.controller.constraints.node_at_node.push_back(id);
            if (spec.restriction == ConstraintRestriction::hard) {
                result.controller.hard_node_constraints.push_back(id);
            }
        }
    }

    for (std::size_t index = 0U;
         index < config.link_attribute_specs.size();
         ++index) {
        const auto& spec = config.link_attribute_specs[index];
        const auto id = static_cast<network::attribute::AttributeRegistryId>(
            index);
        if (spec.kind == AttributeKind::resource) {
            link_resources.push_back(id);
        }
        if (spec.kind != AttributeKind::resource &&
            spec.kind != AttributeKind::latency) {
            continue;
        }
        const CheckingLevel level = spec.checking_level.value_or(
            default_checking_level(spec));
        if (level == CheckingLevel::link) {
            result.controller.constraints.link_at_link.push_back(id);
        } else if (level == CheckingLevel::path) {
            result.controller.constraints.link_at_path.push_back(id);
        }
        if (spec.restriction == ConstraintRestriction::hard) {
            result.controller.hard_link_constraints.push_back(id);
        }
    }
    result.controller.node_resources = node_resources;
    result.controller.link_resources = link_resources;

    // AttributeRegistryId is local to one exact registry.  The controller
    // starts from virtual IDs and binds every dynamic resource name once to
    // the independent physical registry, but CounterSelection is reused for
    // both virtual and physical networks.  Leave its optional filters empty
    // so Counter::prepare selects every typed resource directly in each
    // registry.  Copying virtual IDs into the physical counter is only valid
    // accidentally when both registries have the same layout (for example a
    // single CPU resource); it breaks as soon as physical extrema entries are
    // interleaved with CPU/GPU/RAM resources.
    return result;
}

MainRunReport run_main_config(const MainConfig& config) {
    const auto setup_begin = Clock::now();
    core::Logger logger(config.logger_config);
    RandomContext random(config.seed.value_or(0U));

    // Resolve the one dynamic compatibility name at startup. The selected
    // compact ID and category are fixed before dataset construction; neither
    // a registry lookup nor a string comparison reaches a request loop.
    solver::SolverRegistry registry;
    solver::heuristic::HeuristicSolverRegistryOptions registry_options;
    registry_options.workers = config.workers.node_rank;
    registry_options.bfs = config.bfs_solver_parameters;
    solver::heuristic::register_heuristic_solvers(
        registry,
        random.numpy(),
        random.python(),
        registry_options);
    solver::exact::register_exact_solvers(
        registry,
        random.python(),
        config.exact_solver_parameters);
    solver::exact::register_exact_with_risk_solver(
        registry,
        random.python(),
        config.exact_solver_parameters,
        config.exact_risk_parameters);
    registry.freeze();
    const solver::SolverId selected_solver_id =
        registry.resolve(config.solver_name);
    log_run_start(
        logger,
        config,
        registry.descriptor(selected_solver_id).category);
    log_resolved_config(logger, config);

    network::PhysicalNetwork physical = build_physical_network(config, random);
    network::VirtualNetworkRequestSimulator simulator =
        build_virtual_simulator(config, random);
    save_datasets(config, physical, simulator);

    MainRunReport report;
    report.mode = config.mode;
    report.solver_name = config.solver_name;
    report.physical_nodes = physical.num_nodes();
    report.physical_links = physical.num_links();
    report.virtual_requests = simulator.num_v_nets();
    report.scheduled_events = simulator.num_events();

    const RuntimeSelections selections =
        runtime_selections_from_virtual_config(
            simulator.config(), config.solver_config.reusable);
    core::controller::Controller controller(selections.controller);
    core::Counter counter(selections.counter);
    core::RecorderConfig service_recorder_config = config.recorder_config;
    service_recorder_config.record_dir_name += "-solver-service";
    service_recorder_config.temporary_records = false;
    core::Recorder recorder(
        core::Counter(selections.counter),
        std::move(service_recorder_config));
    // std::cerr is unit-buffered and would flush after every individual
    // insertion.  std::clog targets the same stderr stream but lets the sink
    // issue exactly one explicit flush for each complete progress frame.
    ConsoleProgress progress(config.progress, config.solver_name, std::clog);

    report.solver_id = selected_solver_id;
    std::unique_ptr<solver::Solver> solver_instance = registry.create(
        report.solver_id,
        solver::SolverDependencies{
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger)},
        config.solver_config);

    SystemRunConfig run_config = system_run_config(config);
    run_config.progress = config.progress.enabled ? &progress : nullptr;
    if (config.mode == SystemMode::online) {
        core::SolutionStepEnvironment environment(
            std::move(physical),
            std::move(simulator),
            environment_config(config, selections));
        report.setup_time_ns = elapsed_ns(setup_begin, Clock::now());
        const auto run_begin = Clock::now();
        OnlineSystem system(environment, *solver_instance);
        report.run = system.run(random, run_config);
        report.run_time_ns = elapsed_ns(run_begin, Clock::now());
        save_environment_records(config, environment, report.run);
    } else if (config.mode == SystemMode::offline) {
        report.setup_time_ns = elapsed_ns(setup_begin, Clock::now());
        const auto run_begin = Clock::now();
        OfflineSystem system(physical, simulator, *solver_instance);
        report.run = system.run(OfflineRunConfig{
            config.num_simulations,
            config.capture_solutions,
            config.workers.environment.counter_workers,
            run_config.progress});
        report.run_time_ns = elapsed_ns(run_begin, Clock::now());
    } else if (config.mode == SystemMode::time_window) {
        core::SolutionStepEnvironment environment(
            std::move(physical),
            std::move(simulator),
            environment_config(config, selections));
        TimeWindowRunConfig window_config;
        window_config.system = run_config;
        window_config.window_size = config.time_window_size;
        window_config.window_origin = config.time_window_origin;
        report.setup_time_ns = elapsed_ns(setup_begin, Clock::now());
        const auto run_begin = Clock::now();
        TimeWindowSystem system(environment, *solver_instance);
        TimeWindowRunResult result = system.run(random, window_config);
        report.run_time_ns = elapsed_ns(run_begin, Clock::now());
        report.run = std::move(result.run);
        report.windows = std::move(result.windows);
        save_environment_records(config, environment, report.run);
    } else {
        network::PhysicalNetwork physical_template = physical.clone();
        std::vector<std::unique_ptr<core::SolutionStepEnvironment>>
            environments;
        environments.reserve(config.changeable_stage_count);
        environments.push_back(
            std::make_unique<core::SolutionStepEnvironment>(
                std::move(physical),
                std::move(simulator),
                environment_config(config, selections, 0U)));
        for (std::size_t stage = 1U;
             stage < config.changeable_stage_count;
             ++stage) {
            const std::uint32_t base_seed = config.seed.value_or(0U);
            const auto stage_seed = static_cast<std::uint32_t>(
                base_seed + static_cast<std::uint32_t>(stage));
            RandomContext stage_random(stage_seed);
            auto stage_simulator =
                network::VirtualNetworkRequestSimulator::from_setting(
                    config.virtual_setting,
                    stage_random,
                    stage_seed);
            stage_simulator.renew(
                stage_random,
                true,
                true,
                std::nullopt,
                config.workers.simulator);
            environments.push_back(
                std::make_unique<core::SolutionStepEnvironment>(
                    physical_template.clone(),
                    std::move(stage_simulator),
                    environment_config(config, selections, stage)));
        }
        std::vector<ChangeableStage> stages;
        stages.reserve(environments.size());
        for (const auto& environment : environments) {
            stages.emplace_back(*environment, run_config);
        }
        report.setup_time_ns = elapsed_ns(setup_begin, Clock::now());
        const auto run_begin = Clock::now();
        ChangeableSystem system(std::move(stages), *solver_instance);
        ChangeableRunResult result = system.run(random);
        report.run_time_ns = elapsed_ns(run_begin, Clock::now());
        report.run = std::move(result.run);
        report.stages = std::move(result.stages);
        if (config.save_records) {
            for (std::size_t index = 0U;
                 index < environments.size();
                 ++index) {
                SystemRunResult stage_run;
                const auto& stage = report.stages[index];
                stage_run.epochs.insert(
                    stage_run.epochs.end(),
                    report.run.epochs.begin() +
                        static_cast<std::ptrdiff_t>(stage.first_epoch_index),
                    report.run.epochs.begin() +
                        static_cast<std::ptrdiff_t>(
                            stage.first_epoch_index + stage.num_epochs));
                save_environment_records(
                    config, *environments[index], stage_run);
            }
        }
    }
    log_run_complete(logger, report);
    return report;
}

void write_main_report_json(
    std::ostream& output,
    const MainRunReport& report) {
    output << "{\"mode\":";
    write_json_string(output, system_mode_name(report.mode));
    output << ",\"solver\":";
    write_json_string(output, report.solver_name);
    output << ",\"solver_id\":" << report.solver_id.value
           << ",\"physical_nodes\":" << report.physical_nodes
           << ",\"physical_links\":" << report.physical_links
           << ",\"virtual_requests\":" << report.virtual_requests
           << ",\"scheduled_events\":" << report.scheduled_events
           << ",\"setup_time_ns\":" << report.setup_time_ns
           << ",\"run_time_ns\":" << report.run_time_ns
           << ",\"steps\":[";
    for (std::size_t index = 0U; index < report.run.steps.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }
        const auto& step = report.run.steps[index];
        output << "{\"epoch\":" << step.epoch_index
               << ",\"stage\":" << step.stage_index
               << ",\"window\":" << step.window_index
               << ",\"request_id\":" << step.request_id
               << ",\"event_time\":";
        write_json_double(output, step.event_time);
        output << ",\"accepted\":"
               << (step.accepted ? "true" : "false")
               << ",\"failure_reason\":";
        write_json_string(output, failure_reason_name(step.failure_reason));
        output << ",\"auto_released_events\":"
               << step.auto_released_events << ",\"solution\":";
        write_solution(output, step.solution);
        output.put('}');
    }
    output << "],\"epochs\":[";
    for (std::size_t index = 0U; index < report.run.epochs.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }
        const auto& epoch = report.run.epochs[index];
        output << "{\"epoch\":" << epoch.epoch_index
               << ",\"arrival_steps\":" << epoch.arrival_steps
               << ",\"accepted\":" << epoch.accepted
               << ",\"rejected\":" << epoch.rejected
               << ",\"auto_released_events\":"
               << epoch.auto_released_events << ",\"summary\":";
        write_counter_summary(output, epoch.summary);
        output.put('}');
    }
    output << "],\"stages\":[";
    for (std::size_t index = 0U; index < report.stages.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }
        const auto& stage = report.stages[index];
        output << "{\"stage\":" << stage.stage_index
               << ",\"first_epoch\":" << stage.first_epoch_index
               << ",\"num_epochs\":" << stage.num_epochs
               << ",\"first_step\":" << stage.first_step_index
               << ",\"num_steps\":" << stage.num_steps << '}';
    }
    output << "],\"windows\":[";
    for (std::size_t index = 0U; index < report.windows.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }
        const auto& window = report.windows[index];
        output << "{\"epoch\":" << window.epoch_index
               << ",\"window\":" << window.window_index
               << ",\"first_event_time\":";
        write_json_double(output, window.first_event_time);
        output << ",\"last_event_time\":";
        write_json_double(output, window.last_event_time);
        output << ",\"arrival_steps\":" << window.arrival_steps
               << ",\"accepted\":" << window.accepted
               << ",\"rejected\":" << window.rejected
               << ",\"auto_released_events\":"
               << window.auto_released_events << '}';
    }
    output << "]}\n";
}

void write_main_summary_json(
    std::ostream& output,
    const MainRunReport& report) {
    std::size_t processed = 0U;
    std::size_t accepted = 0U;
    std::size_t rejected = 0U;
    for (const auto& epoch : report.run.epochs) {
        processed += epoch.arrival_steps;
        accepted += epoch.accepted;
        rejected += epoch.rejected;
    }
    const double acceptance = processed == 0U
        ? 0.0
        : static_cast<double>(accepted) / static_cast<double>(processed);
    const double r2c = report.run.epochs.empty()
        ? 0.0
        : report.run.epochs.back().summary.long_term_r2c_ratio;

    output << "{\"mode\":";
    write_json_string(output, system_mode_name(report.mode));
    output << ",\"solver\":";
    write_json_string(output, report.solver_name);
    output << ",\"physical_nodes\":" << report.physical_nodes
           << ",\"physical_links\":" << report.physical_links
           << ",\"virtual_requests\":" << report.virtual_requests
           << ",\"scheduled_events\":" << report.scheduled_events
           << ",\"epochs\":" << report.run.epochs.size()
           << ",\"processed_requests\":" << processed
           << ",\"accepted\":" << accepted
           << ",\"rejected\":" << rejected
           << ",\"acceptance_rate\":";
    write_json_double(output, acceptance);
    output << ",\"long_term_r2c_ratio\":";
    write_json_double(output, r2c);
    output << ",\"setup_time_ns\":" << report.setup_time_ns
           << ",\"run_time_ns\":" << report.run_time_ns << "}\n";
}

} // namespace virne::system
