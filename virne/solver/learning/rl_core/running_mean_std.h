#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::solver::learning::rl_core {

enum class RunningMeanStdErrorCode : std::uint8_t {
    invalid_feature_count,
    sample_length_mismatch,
    moment_length_mismatch,
};

class RunningMeanStdException final : public std::runtime_error {
public:
    RunningMeanStdException(
        RunningMeanStdErrorCode code,
        std::string message);

    RunningMeanStdErrorCode code() const noexcept { return code_; }

private:
    RunningMeanStdErrorCode code_;
};

// Update a float64 running population mean/variance from two sets of moments.
// Each lane is a flattened trailing feature shape; a one-element moment lane
// is broadcast once at this boundary, never looked up in a hot loop.
struct RunningMeanVarMoments {
    std::vector<double> mean;
    std::vector<double> variance;
    double count = 0.0;
};

RunningMeanVarMoments update_mean_var_count_from_moments(
    const std::vector<double>& mean,
    const std::vector<double>& variance,
    double count,
    const std::vector<double>& batch_mean,
    const std::vector<double>& batch_variance,
    double batch_count);

class RunningMeanStd final {
public:
    explicit RunningMeanStd(
        std::size_t feature_count = 1U,
        double epsilon = 1e-4);

    // Samples are row-major [sample_count, feature_count].  The reduction is
    // population variance (ddof=0), matching numpy.var used by Python.
    void update(
        const std::vector<double>& samples,
        std::size_t sample_count);

    void update_from_moments(
        const std::vector<double>& batch_mean,
        const std::vector<double>& batch_variance,
        double batch_count);

    std::size_t feature_count() const noexcept { return mean_.size(); }
    const std::vector<double>& mean() const noexcept { return mean_; }
    const std::vector<double>& variance() const noexcept { return variance_; }
    double count() const noexcept { return count_; }

private:
    std::vector<double> mean_;
    std::vector<double> variance_;
    double count_ = 0.0;
};

}  // namespace virne::solver::learning::rl_core
