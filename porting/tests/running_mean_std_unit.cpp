#include "../../virne/solver/learning/rl_core/running_mean_std.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl = virne::solver::learning::rl_core;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& message,
    const double tolerance = 1e-12) {
    require(actual.size() == expected.size(), message + " (length)");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        if (std::isnan(expected[index])) {
            require(std::isnan(actual[index]),
                    message + " (expected NaN at " + std::to_string(index)
                        + ")");
            continue;
        }
        require(std::abs(actual[index] - expected[index]) <= tolerance,
                message + " (value at " + std::to_string(index) + ")");
    }
}

void test_initial_state() {
    const rl::RunningMeanStd stats(3U);
    require(stats.feature_count() == 3U, "initial feature count");
    require_close(stats.mean(), {0.0, 0.0, 0.0}, "initial mean");
    require_close(stats.variance(), {1.0, 1.0, 1.0}, "initial variance");
    require(stats.count() == 1e-4, "initial epsilon count");
}

void test_two_batch_update() {
    rl::RunningMeanStd stats(2U, 0.0);
    stats.update({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, 4U);
    require_close(stats.mean(), {4.0, 5.0}, "first batch mean");
    require_close(stats.variance(), {5.0, 5.0}, "first batch variance");
    require(stats.count() == 4.0, "first batch count");

    stats.update({9.0, 10.0, 11.0, 12.0}, 2U);
    require_close(stats.mean(), {6.0, 7.0}, "merged mean");
    require_close(stats.variance(), {35.0 / 3.0, 35.0 / 3.0},
                  "merged variance");
    require(stats.count() == 6.0, "merged count");
}

void test_moment_broadcast_and_free_function() {
    rl::RunningMeanStd stats(3U, 0.0);
    stats.update_from_moments({2.0}, {1.0}, 4.0);
    require_close(stats.mean(), {2.0, 2.0, 2.0}, "broadcast mean");
    require_close(stats.variance(), {1.0, 1.0, 1.0}, "broadcast variance");
    require(stats.count() == 4.0, "broadcast count");

    stats.update_from_moments({4.0, 6.0, 8.0}, {0.5, 1.0, 2.0}, 2.0);
    require_close(stats.mean(), {8.0 / 3.0, 10.0 / 3.0, 4.0},
                  "second moments mean");
    require_close(stats.variance(),
                  {31.0 / 18.0, 41.0 / 9.0, 28.0 / 3.0},
                  "second moments variance");
    require(stats.count() == 6.0, "second moments count");

    const auto result = rl::update_mean_var_count_from_moments(
        {0.0, 0.0}, {1.0, 1.0}, 2.0, {4.0}, {0.0}, 2.0);
    require_close(result.mean, {2.0, 2.0}, "free function mean");
    require_close(result.variance, {4.5, 4.5}, "free function variance");
    require(result.count == 4.0, "free function count");
}

void test_errors() {
    bool feature_count_rejected = false;
    try {
        static_cast<void>(rl::RunningMeanStd(0U));
    } catch (const rl::RunningMeanStdException& error) {
        feature_count_rejected =
            error.code() == rl::RunningMeanStdErrorCode::invalid_feature_count;
    }
    require(feature_count_rejected, "zero feature count error");

    bool sample_length_rejected = false;
    try {
        rl::RunningMeanStd stats(2U);
        stats.update({1.0, 2.0, 3.0}, 2U);
    } catch (const rl::RunningMeanStdException& error) {
        sample_length_rejected =
            error.code() == rl::RunningMeanStdErrorCode::sample_length_mismatch;
    }
    require(sample_length_rejected, "sample length error");

    bool moment_length_rejected = false;
    try {
        rl::RunningMeanStd stats(2U);
        stats.update_from_moments({1.0, 2.0, 3.0}, {1.0}, 1.0);
    } catch (const rl::RunningMeanStdException& error) {
        moment_length_rejected =
            error.code() == rl::RunningMeanStdErrorCode::moment_length_mismatch;
    }
    require(moment_length_rejected, "moment length error");
}

}  // namespace

int main() {
    try {
        test_initial_state();
        test_two_batch_update();
        test_moment_broadcast_and_free_function();
        test_errors();
        std::cout << "running_mean_std_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "running_mean_std_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
