#include "running_mean_std.h"

#include <cmath>
#include <limits>
#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

const std::vector<double>& moment_lane(
    const std::vector<double>& lane,
    const std::size_t feature_count,
    std::vector<double>& broadcast) {
    if (lane.size() == feature_count) {
        return lane;
    }
    if (lane.size() == 1U && feature_count != 1U) {
        broadcast.assign(feature_count, lane.front());
        return broadcast;
    }
    throw RunningMeanStdException(
        RunningMeanStdErrorCode::moment_length_mismatch,
        "running mean/variance moment lane has the wrong length");
}

}  // namespace

RunningMeanStdException::RunningMeanStdException(
    const RunningMeanStdErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

RunningMeanVarMoments update_mean_var_count_from_moments(
    const std::vector<double>& mean,
    const std::vector<double>& variance,
    const double count,
    const std::vector<double>& batch_mean,
    const std::vector<double>& batch_variance,
    const double batch_count) {
    if (mean.size() != variance.size() || mean.empty()) {
        throw RunningMeanStdException(
            RunningMeanStdErrorCode::moment_length_mismatch,
            "running mean and variance lanes must be non-empty and equal");
    }
    std::vector<double> mean_broadcast;
    std::vector<double> variance_broadcast;
    const auto& normalized_batch_mean = moment_lane(
        batch_mean, mean.size(), mean_broadcast);
    const auto& normalized_batch_variance = moment_lane(
        batch_variance, mean.size(), variance_broadcast);

    RunningMeanVarMoments result;
    result.mean.resize(mean.size());
    result.variance.resize(mean.size());
    result.count = count + batch_count;
    for (std::size_t index = 0U; index < mean.size(); ++index) {
        const double delta = normalized_batch_mean[index] - mean[index];
        result.mean[index] = mean[index]
            + delta * batch_count / result.count;
        const double m_a = variance[index] * count;
        const double m_b = normalized_batch_variance[index] * batch_count;
        const double m2 = m_a + m_b
            + delta * delta * count * batch_count / result.count;
        result.variance[index] = m2 / result.count;
    }
    return result;
}

RunningMeanStd::RunningMeanStd(
    const std::size_t feature_count,
    const double epsilon)
    : mean_(feature_count, 0.0),
      variance_(feature_count, 1.0),
      count_(epsilon) {
    if (feature_count == 0U) {
        throw RunningMeanStdException(
            RunningMeanStdErrorCode::invalid_feature_count,
            "running mean/variance requires at least one feature");
    }
}

void RunningMeanStd::update(
    const std::vector<double>& samples,
    const std::size_t sample_count) {
    const std::size_t feature_count = mean_.size();
    if (samples.size() != sample_count * feature_count) {
        throw RunningMeanStdException(
            RunningMeanStdErrorCode::sample_length_mismatch,
            "running sample lane does not match sample_count*feature_count");
    }

    std::vector<double> batch_mean(feature_count, 0.0);
    std::vector<double> batch_variance(feature_count, 0.0);
    if (sample_count == 0U) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        batch_mean.assign(feature_count, nan);
        batch_variance.assign(feature_count, nan);
        update_from_moments(batch_mean, batch_variance, 0.0);
        return;
    }

    const double reciprocal = 1.0 / static_cast<double>(sample_count);
    for (std::size_t row = 0U; row < sample_count; ++row) {
        const std::size_t offset = row * feature_count;
        for (std::size_t index = 0U; index < feature_count; ++index) {
            batch_mean[index] += samples[offset + index];
        }
    }
    for (double& value : batch_mean) {
        value *= reciprocal;
    }
    for (std::size_t row = 0U; row < sample_count; ++row) {
        const std::size_t offset = row * feature_count;
        for (std::size_t index = 0U; index < feature_count; ++index) {
            const double delta = samples[offset + index] - batch_mean[index];
            batch_variance[index] += delta * delta;
        }
    }
    for (double& value : batch_variance) {
        value *= reciprocal;
    }
    update_from_moments(batch_mean, batch_variance,
                        static_cast<double>(sample_count));
}

void RunningMeanStd::update_from_moments(
    const std::vector<double>& batch_mean,
    const std::vector<double>& batch_variance,
    const double batch_count) {
    const RunningMeanVarMoments result =
        update_mean_var_count_from_moments(
            mean_, variance_, count_, batch_mean, batch_variance,
            batch_count);
    mean_ = result.mean;
    variance_ = result.variance;
    count_ = result.count;
}

}  // namespace virne::solver::learning::rl_core
