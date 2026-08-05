#include "../../virne/solver/learning/rl_core/rollout_buffer.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl = virne::solver::learning::rl_core;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + " (length)");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require(std::abs(actual[index] - expected[index]) < 1e-12,
                message + " (value at " + std::to_string(index) + ")");
    }
}

rl::RolloutBuffer make_buffer() {
    rl::RolloutBuffer result;
    const std::vector<double> rewards{1.2, -0.4, 0.7, 0.1, -0.3};
    const std::vector<double> values{0.2, 0.3, -0.1, 0.4, 0.05};
    const std::vector<std::uint8_t> dones{0U, 0U, 1U, 0U, 1U};
    for (std::size_t index = 0U; index < rewards.size(); ++index) {
        result.add(
            100U + index,
            static_cast<std::int64_t>(10U + index),
            rewards[index],
            dones[index] != 0U,
            -0.1 * static_cast<double>(index + 1U),
            values[index],
            200U + index,
            true);
    }
    result.extend(rl::RolloutExtensionId::feasibility_flags);
    result.extend("feasibility_budgets");
    result.feasibility_flags = {1.0, 1.0, 0.0, 1.0, 0.0};
    result.feasibility_budgets = {0.5, 0.5, 1.0, 1.5, 2.0};
    return result;
}

void test_lifecycle_split_and_merge() {
    auto buffer = make_buffer();
    require(buffer.size() == 5U, "buffer size");
    require(buffer.curr_idx == 5U, "curr_idx after add");
    require(buffer.observations.front() == 100U, "observation id");
    require(buffer.has_next_observation.back() == 1U, "next observation flag");
    require(buffer.is_extension_enabled(rl::RolloutExtensionId::feasibility_flags),
            "typed extension enabled");
    require(rl::rollout_extension_id_from_name("action_mask")
                == rl::RolloutExtensionId::action_mask,
            "extension lookup");

    const auto selected = buffer.get_subbuffer({4U, 1U});
    require(selected.curr_idx == 0U, "Python-compatible subbuffer curr_idx");
    require(selected.observations == std::vector<rl::ObservationId>{104U, 101U},
            "subbuffer order");
    require_close(selected.feasibility_budgets, {2.0, 0.5},
                  "subbuffer extension");

    const auto ratios = buffer.split_with_ratios({0.4, 0.6});
    require(ratios.size() == 2U, "ratio split count");
    require(ratios[0].size() == 2U && ratios[1].size() == 3U,
            "ratio split sizes");
    require(ratios[1].observations.front() == 102U, "ratio split order");

    const auto instances = buffer.split_with_instance();
    require(instances.size() == 2U, "instance split count");
    require(instances[0].size() == 3U && instances[1].size() == 2U,
            "instance split sizes");

    rl::RolloutBuffer historical;
    for (std::size_t index = 0U; index < 6U; ++index) {
        historical.add(index, static_cast<std::int64_t>(index), 0.0,
                       index == 1U || index == 3U || index == 5U,
                       0.0, 0.0);
    }
    const auto historical_instances = historical.split_with_instance();
    require(historical_instances.size() == 3U,
            "historical instance split count");
    require(historical_instances[0].size() == 2U
                && historical_instances[1].size() == 4U
                && historical_instances[2].size() == 0U,
            "Python absolute-index instance split compatibility");
    require(historical_instances[0].observations
                == std::vector<rl::ObservationId>{0U, 1U}
                && historical_instances[1].observations
                    == std::vector<rl::ObservationId>{2U, 3U, 4U, 5U}
                && historical_instances[2].observations.empty(),
            "Python absolute-index instance split values");

    rl::RolloutBuffer incomplete_tail;
    for (std::size_t index = 0U; index < 5U; ++index) {
        incomplete_tail.add(index, static_cast<std::int64_t>(index), 0.0,
                            index == 1U, 0.0, 0.0);
    }
    const auto complete_instances = incomplete_tail.split_with_instance();
    require(complete_instances.size() == 1U
                && complete_instances.front().observations
                    == std::vector<rl::ObservationId>{0U, 1U},
            "instance split drops incomplete Python tail");

    auto merged = ratios[0];
    const std::size_t old_curr_idx = merged.curr_idx;
    merged.merge(ratios[1]);
    require(merged.size() == 5U, "merge size");
    require(merged.curr_idx == old_curr_idx, "Python-compatible merge curr_idx");
    require(merged.observations == buffer.observations, "merge order");
    require_close(merged.feasibility_flags, buffer.feasibility_flags,
                  "merge extension");

    auto self_merged = make_buffer();
    self_merged.merge(self_merged);
    require(self_merged.size() == 10U, "self merge doubles rows");
    require(self_merged.observations[5] == 100U, "self merge order");

    buffer.clear();
    require(buffer.size() == 0U && buffer.curr_idx == 0U, "clear state");
    require(buffer.feasibility_flags_enabled && buffer.feasibility_budgets_enabled,
            "clear preserves extension schema");

    rl::RolloutBuffer malformed = make_buffer();
    malformed.returns = {1.0};
    bool malformed_threw = false;
    try {
        static_cast<void>(malformed.get_subbuffer({0U}));
    } catch (const rl::RolloutBufferException& error) {
        malformed_threw = error.code() == rl::RolloutBufferErrorCode::length_mismatch;
    }
    require(malformed_threw, "short calculated lane error");

    rl::RolloutBuffer malformed_basic = make_buffer();
    malformed_basic.actions.pop_back();
    bool malformed_basic_threw = false;
    try {
        static_cast<void>(malformed_basic.get_subbuffer({0U}));
    } catch (const rl::RolloutBufferException& error) {
        malformed_basic_threw =
            error.code() == rl::RolloutBufferErrorCode::length_mismatch;
    }
    require(malformed_basic_threw,
            "short basic lane rejected before selected copy");

    rl::RolloutBuffer malformed_extension = make_buffer();
    malformed_extension.feasibility_flags.resize(1U);
    bool malformed_extension_threw = false;
    try {
        static_cast<void>(malformed_extension.get_subbuffer({0U}));
    } catch (const rl::RolloutBufferException& error) {
        malformed_extension_threw =
            error.code() == rl::RolloutBufferErrorCode::length_mismatch;
    }
    require(malformed_extension_threw,
            "short extension lane rejected before selected copy");
}

void test_return_methods() {
    auto gae = make_buffer();
    gae.compute_returns_and_advantages(0.25, 0.91, 0.83,
                                       rl::RolloutBufferMethod::gae);
    require_close(
        gae.returns,
        {1.3319401719999997, 0.11323999999999984, 0.7,
         -0.11885499999999993, -0.3},
        "buffer GAE returns");
    require_close(
        gae.advantages,
        {1.1319401719999997, -0.18676000000000015, 0.7999999999999999,
         -0.518855, -0.35},
        "buffer GAE advantages");

    auto mc = make_buffer();
    mc.compute_returns_and_advantages(0.25, 0.91, 0.83,
                                      rl::RolloutBufferMethod::mc);
    require_close(mc.returns, {1.41567, 0.237, 0.7, -0.173, -0.3},
                  "buffer MC returns");
    require_close(mc.advantages, {0.0, 0.0, 0.0, 0.0, 0.0},
                  "buffer MC advantages");

    auto td = make_buffer();
    td.compute_returns_and_advantages(0.25, 0.91, 0.83,
                                      rl::RolloutBufferMethod::td);
    require_close(td.returns, {1.273, -0.791, 0.8, -0.2545, -0.35},
                  "buffer TD returns");
}

void test_cost_buffer() {
    rl::RolloutBufferWithCost buffer;
    const auto source = make_buffer();
    buffer.merge(source);
    buffer.costs = {1.0, -2.0, 0.5, 2.0, -1.0};
    buffer.compute_cost_returns(0.8, rl::CostReturnMethod::reachability);
    require_close(buffer.cost_returns, {1.0, 0.4, 0.5, 2.0, -1.0},
                  "cost reachability");
    buffer.compute_cost_returns(0.8, rl::CostReturnMethod::cumulative);
    require_close(buffer.costs, {1.0, 0.0, 0.5, 2.0, 0.0},
                  "cost cumulative mutation");
    require_close(
        buffer.cost_returns,
        {1.32, 0.4, 0.5, 2.0, 0.0, 1.0, 0.4, 0.5, 2.0, -1.0},
        "cost repeated-call prepend");

    const auto selected = buffer.get_cost_subbuffer({3U, 0U});
    require_close(selected.costs, {2.0, 1.0}, "cost subbuffer");

    const auto repeated_cost_returns = buffer.cost_returns;
    buffer.merge(buffer);
    std::vector<double> doubled_cost_returns = repeated_cost_returns;
    doubled_cost_returns.insert(doubled_cost_returns.end(),
                                repeated_cost_returns.begin(),
                                repeated_cost_returns.end());
    require_close(buffer.cost_returns, doubled_cost_returns,
                  "cost self merge after repeated return computation");

    rl::RolloutBufferWithCost aggregate;
    rl::RolloutBufferWithCost episode;
    episode.add(7U, 3, 1.0, true, -0.2, 0.4);
    episode.costs = {2.5};
    episode.cost_values = {0.75};
    episode.cost_returns = {2.5};
    episode.baseline_cost_returns = {1.5};
    aggregate.merge(episode);
    require_close(aggregate.costs, {2.5}, "cost merge costs");
    require_close(aggregate.cost_values, {0.75}, "cost merge values");
    require_close(aggregate.cost_returns, {2.5}, "cost merge returns");
    require_close(aggregate.baseline_cost_returns, {1.5},
                  "cost merge baseline returns");

    aggregate.merge(aggregate);
    require_close(aggregate.costs, {2.5, 2.5}, "cost self merge");

    rl::RolloutBufferWithCost malformed;
    malformed.merge(make_buffer());
    malformed.costs = {1.0};
    bool malformed_cost_threw = false;
    try {
        static_cast<void>(malformed.get_cost_subbuffer({0U}));
    } catch (const rl::RolloutBufferException& error) {
        malformed_cost_threw =
            error.code() == rl::RolloutBufferErrorCode::length_mismatch;
    }
    require(malformed_cost_threw,
            "short cost lane rejected even when selected index exists");
}

rl::RolloutBufferWithAvgReturn make_avg_buffer() {
    rl::RolloutBufferWithAvgReturn result;
    const std::vector<double> rewards{1.0, -1.0, 2.0};
    const std::vector<double> values{0.1, 0.2, 0.3};
    for (std::size_t index = 0U; index < rewards.size(); ++index) {
        result.add(index, static_cast<std::int64_t>(index), rewards[index],
                   index == 2U, -0.2, values[index]);
    }
    return result;
}

void test_average_return_buffer() {
    auto gae = make_avg_buffer();
    gae.compute_returns_and_advantages(0.4, 0.9, 0.8,
                                       rl::RolloutBufferMethod::gae);
    require(gae.dones.back() == 0U, "avg GAE clears final done");
    require_close(gae.rewards,
                  {0.33333333333333337, -1.6666666666666665,
                   1.3333333333333335},
                  "avg GAE reward mutation");
    require_close(
        gae.returns,
        {0.08603733333333342, -0.3934666666666665, 1.6933333333333336},
        "avg GAE returns");
    require_close(
        gae.advantages,
        {-0.01396266666666659, -0.5934666666666665, 1.3933333333333335},
        "avg GAE advantages");

    auto td = make_avg_buffer();
    td.compute_returns_and_advantages(0.4, 0.9, 0.8,
                                      rl::RolloutBufferMethod::td);
    require_close(td.rewards, {1.0, -1.0, 2.0}, "avg TD keeps rewards");
    require_close(td.returns,
                  {0.5333333333333334, -1.3666666666666665,
                   1.7333333333333334},
                  "avg TD returns");
    require_close(td.advantages,
                  {0.43333333333333335, -1.5666666666666664,
                   1.4333333333333333},
                  "avg TD advantages");
}

void test_polymorphic_lifecycle_and_compute() {
    rl::RolloutBufferWithCost cost;
    rl::RolloutBuffer* base = &cost;
    base->reserve(8U);
    require(cost.costs.capacity() >= 8U,
            "virtual reserve reaches cost lanes");
    base->add(1U, 2, 3.0, true, -0.1, 0.5);
    cost.costs = {2.0};
    base->merge(*base);
    require(base->size() == 2U && cost.costs == std::vector<double>{2.0, 2.0},
            "virtual self merge preserves cost lanes");
    base->clear();
    require(base->size() == 0U && cost.costs.empty(),
            "base clear dispatches virtual cost reset");

    auto average = make_avg_buffer();
    rl::RolloutBuffer* average_base = &average;
    average_base->compute_returns_and_advantages(
        0.4, 0.9, 0.8, rl::RolloutBufferMethod::td);
    require(average.dones.back() == 0U,
            "virtual compute reaches average-return override");
    require_close(average.returns,
                  {0.5333333333333334, -1.3666666666666665,
                   1.7333333333333334},
                  "polymorphic average-return TD");
}

void test_errors() {
    bool method_threw = false;
    try {
        static_cast<void>(rl::rollout_buffer_method_from_name("unknown"));
    } catch (const rl::RolloutBufferException& error) {
        method_threw = error.code() == rl::RolloutBufferErrorCode::invalid_method;
    }
    require(method_threw, "invalid method error");

    bool ratio_threw = false;
    try {
        static_cast<void>(make_buffer().split_with_ratios({0.4, 0.5}));
    } catch (const rl::RolloutBufferException& error) {
        ratio_threw = error.code() == rl::RolloutBufferErrorCode::invalid_ratio;
    }
    require(ratio_threw, "invalid ratio error");

    bool empty_avg_threw = false;
    try {
        rl::RolloutBufferWithAvgReturn empty;
        empty.compute_returns_and_advantages();
    } catch (const rl::RolloutBufferException& error) {
        empty_avg_threw = error.code() == rl::RolloutBufferErrorCode::length_mismatch;
    }
    require(empty_avg_threw, "empty average buffer error");
}

}  // namespace

int main() {
    try {
        test_lifecycle_split_and_merge();
        test_return_methods();
        test_cost_buffer();
        test_average_return_buffer();
        test_polymorphic_lifecycle_and_compute();
        test_errors();
        std::cout << "rollout_buffer_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rollout_buffer_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
