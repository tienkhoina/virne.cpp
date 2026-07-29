#include "solution.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <optional>
#include <thread>
#include <utility>

namespace virne::core {
namespace {

std::size_t normalized_worker_count(
    const std::size_t requested,
    const std::size_t count) noexcept {
    if (requested <= 1U || count <= 1U) {
        return 1U;
    }
    return std::min(requested, count);
}

template <typename Function>
void parallel_indexed(
    const std::size_t count,
    const std::size_t requested_workers,
    Function&& function) {
    if (count == 0U) {
        return;
    }
    const std::size_t workers =
        normalized_worker_count(requested_workers, count);
    if (workers == 1U) {
        for (std::size_t index = 0U; index < count; ++index) {
            function(index);
        }
        return;
    }

    std::vector<std::exception_ptr> errors(count);
    auto run_range = [&](const std::size_t begin, const std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            try {
                function(index);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        }
    };

    const std::size_t block = count / workers;
    const std::size_t remainder = count % workers;
    std::vector<std::thread> threads;
    threads.reserve(workers - 1U);
    std::size_t begin = 0U;
    std::size_t launched_workers = 0U;
    try {
        for (std::size_t worker = 0U; worker + 1U < workers; ++worker) {
            const std::size_t length = block + (worker < remainder ? 1U : 0U);
            const std::size_t end = begin + length;
            threads.emplace_back(run_range, begin, end);
            begin = end;
            ++launched_workers;
        }
    } catch (...) {
        for (auto& thread : threads) {
            thread.join();
        }
        // A thread creation failure is an execution-policy failure, not a
        // data error. Complete every range that was not launched sequentially.
        for (std::size_t index = begin; index < count; ++index) {
            try {
                function(index);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        }
        for (std::size_t index = 0U; index < count; ++index) {
            if (errors[index]) {
                std::rethrow_exception(errors[index]);
            }
        }
        return;
    }

    static_cast<void>(launched_workers);
    run_range(begin, count);
    for (auto& thread : threads) {
        thread.join();
    }
    for (std::size_t index = 0U; index < count; ++index) {
        if (errors[index]) {
            std::rethrow_exception(errors[index]);
        }
    }
}

} // namespace

SolutionException::SolutionException(
    const SolutionNetworkField field,
    std::string message)
    : std::runtime_error(std::move(message)), field_(field) {}

Solution::Solution(const SolutionMetadata metadata)
    : v_net_id(metadata.v_net_id),
      v_net_lifetime(metadata.v_net_lifetime),
      v_net_arrival_time(metadata.v_net_arrival_time),
      v_net_num_nodes(metadata.v_net_num_nodes),
      v_net_num_edges(metadata.v_net_num_edges) {
    reset();
}

Solution Solution::from_v_net(const network::VirtualNetwork& virtual_network) {
    if (!virtual_network.request_id().has_value()) {
        throw SolutionException(
            SolutionNetworkField::id,
            "virtual network is missing id");
    }
    if (!virtual_network.lifetime().has_value()) {
        throw SolutionException(
            SolutionNetworkField::lifetime,
            "virtual network is missing lifetime");
    }
    if (!virtual_network.arrival_time().has_value()) {
        throw SolutionException(
            SolutionNetworkField::arrival_time,
            "virtual network is missing arrival_time");
    }
    return Solution(SolutionMetadata{
        *virtual_network.request_id(),
        *virtual_network.lifetime(),
        *virtual_network.arrival_time(),
        virtual_network.num_nodes(),
        virtual_network.num_links(),
    });
}

std::vector<Solution> Solution::from_metadata_batch(
    const std::vector<SolutionMetadata>& metadata,
    const std::size_t workers) {
    std::vector<std::optional<Solution>> slots(metadata.size());
    parallel_indexed(metadata.size(), workers, [&](const std::size_t index) {
        slots[index].emplace(metadata[index]);
    });

    std::vector<Solution> result;
    result.reserve(metadata.size());
    for (auto& slot : slots) {
        result.push_back(std::move(*slot));
    }
    return result;
}

void Solution::reset_batch(
    std::vector<Solution>& solutions,
    const std::size_t workers) {
    parallel_indexed(solutions.size(), workers, [&](const std::size_t index) {
        solutions[index].reset();
    });
}

void Solution::reset() {
    result = false;
    node_slots.clear();
    link_paths.clear();
    node_slots_info.clear();
    link_paths_info.clear();

    v_net_cost = 0.0;
    v_net_revenue = 0.0;
    v_net_demand = 0.0;
    v_net_node_demand = 0.0;
    v_net_link_demand = 0.0;
    v_net_node_revenue = 0.0;
    v_net_link_revenue = 0.0;
    v_net_node_cost = 0.0;
    v_net_link_cost = 0.0;
    v_net_path_cost = 0.0;
    v_net_r2c_ratio = 0.0;
    v_net_time_cost = 0.0;
    v_net_time_revenue = 0.0;
    v_net_time_rc_ratio = 0.0;
    description.clear();

    v_net_total_hard_constraint_violation = 0.0;
    v_net_single_step_constraint_offset.clear();
    v_net_constraint_offsets.clear();
    v_net_constraint_violations.clear();
    v_net_single_step_violation_list.clear();
    v_net_single_step_hard_constraint_offset =
        -std::numeric_limits<double>::infinity();
    v_net_max_single_step_hard_constraint_violation =
        -std::numeric_limits<double>::infinity();

    place_result = true;
    route_result = true;
    early_rejection = false;
    revoke_times = 0;
    selected_actions.clear();
    num_interactions = 0;
    v_net_reward = 0.0;
}

bool Solution::is_feasible() const noexcept {
    return result && v_net_total_hard_constraint_violation <= 0.0;
}

std::string Solution::repr() const {
    // The Python f-string escapes both braces and therefore returns this
    // literal text rather than interpolating the dictionary comprehension.
    return "Solution({k: v for k, v in self.__dict__.items()})";
}

} // namespace virne::core
