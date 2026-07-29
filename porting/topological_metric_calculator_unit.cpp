#include "topology/topological_metric_calculator.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using virne::network::MetricColumn;
using virne::network::TopologicalMetricCalculator;
using virne::network::TopologicalMetricOptions;
using virne::network::TopologicalMetrics;

using MetricMember =
    std::optional<MetricColumn> TopologicalMetrics::*;

constexpr std::array<MetricMember, 4> kMetricMembers{
    &TopologicalMetrics::node_degree_centrality,
    &TopologicalMetrics::node_closeness_centrality,
    &TopologicalMetrics::node_eigenvector_centrality,
    &TopologicalMetrics::node_betweenness_centrality};

constexpr std::array<std::string_view, 4> kMetricNames{
    "degree", "closeness", "eigenvector", "betweenness"};

static_assert(
    std::is_same_v<
        typename decltype(MetricColumn::values)::value_type,
        float>,
    "MetricColumn must retain NumPy float32 output semantics");
static_assert(
    MetricColumn::columns() == 1,
    "MetricColumn must represent an (N, 1) array");

void require(
    bool condition,
    const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

std::uint32_t float_bits(float value)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

Graph make_graph(
    std::size_t node_count,
    std::vector<EdgeEndpoints> edges = {})
{
    return Graph(node_count, edges);
}

TopologicalMetricOptions options_for_mask(
    unsigned mask,
    bool normalize,
    std::size_t workers)
{
    TopologicalMetricOptions options;
    options.degree = (mask & 1U) != 0;
    options.closeness = (mask & 2U) != 0;
    options.eigenvector = (mask & 4U) != 0;
    options.betweenness = (mask & 8U) != 0;
    options.normalize = normalize;
    options.worker_count = workers;
    return options;
}

void require_column_bits(
    const std::optional<MetricColumn>& actual,
    const std::vector<std::uint32_t>& expected,
    const std::string& context)
{
    require(actual.has_value(), context + ": metric is absent");
    require(
        actual->columns() == 1,
        context + ": column count is not one");
    require(
        actual->rows() == expected.size(),
        context + ": row count mismatch");
    require(
        actual->values.size() == expected.size(),
        context + ": storage size mismatch");
    for (std::size_t row = 0; row < expected.size(); ++row)
    {
        require(
            float_bits((*actual)[row]) == expected[row],
            context + ": float32 bits differ at row " +
                std::to_string(row));
    }
}

void require_column_equal(
    const std::optional<MetricColumn>& actual,
    const std::optional<MetricColumn>& expected,
    const std::string& context)
{
    require(
        actual.has_value() == expected.has_value(),
        context + ": optional presence mismatch");
    if (!actual.has_value())
    {
        return;
    }
    require(
        actual->columns() == 1 && expected->columns() == 1,
        context + ": invalid shape");
    require(
        actual->rows() == expected->rows(),
        context + ": row count mismatch");
    for (std::size_t row = 0; row < actual->rows(); ++row)
    {
        require(
            float_bits((*actual)[row]) ==
                float_bits((*expected)[row]),
            context + ": float32 bits differ at row " +
                std::to_string(row));
    }
}

void require_metrics_equal(
    const TopologicalMetrics& actual,
    const TopologicalMetrics& expected,
    const std::string& context)
{
    for (std::size_t index = 0;
         index < kMetricMembers.size();
         ++index)
    {
        require_column_equal(
            actual.*kMetricMembers[index],
            expected.*kMetricMembers[index],
            context + "/" + std::string(kMetricNames[index]));
    }
}

void require_presence(
    const TopologicalMetrics& metrics,
    unsigned mask,
    const std::string& context)
{
    for (std::size_t index = 0;
         index < kMetricMembers.size();
         ++index)
    {
        const bool expected =
            (mask & (1U << static_cast<unsigned>(index))) != 0;
        require(
            (metrics.*kMetricMembers[index]).has_value() == expected,
            context + ": wrong presence for " +
                std::string(kMetricNames[index]));
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    std::string_view expected_text,
    const std::string& context)
{
    try
    {
        function();
    }
    catch (const Exception& error)
    {
        require(
            std::string_view(error.what()).find(expected_text) !=
                std::string_view::npos,
            context + ": exception text mismatch: " + error.what());
        return;
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            context + ": wrong exception type: " + error.what());
    }

    throw std::runtime_error(context + ": expected an exception");
}

bool metrics_bit_equal_noexcept(
    const TopologicalMetrics& actual,
    const TopologicalMetrics& expected) noexcept
{
    for (const MetricMember member : kMetricMembers)
    {
        const auto& left = actual.*member;
        const auto& right = expected.*member;
        if (left.has_value() != right.has_value())
        {
            return false;
        }
        if (!left.has_value())
        {
            continue;
        }
        if (left->rows() != right->rows() ||
            left->columns() != right->columns())
        {
            return false;
        }
        for (std::size_t row = 0; row < left->rows(); ++row)
        {
            if (float_bits((*left)[row]) !=
                float_bits((*right)[row]))
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main()
{
    try
    {
        const TopologicalMetrics empty_metrics;
        require_presence(empty_metrics, 0, "typed metrics defaults");

        const TopologicalMetricOptions defaults;
        require(
            defaults.degree && defaults.closeness &&
                defaults.eigenvector && defaults.betweenness &&
                defaults.normalize && defaults.worker_count == 0,
            "calculate option defaults do not match Python");

        const TopologicalMetricOptions degree_only =
            TopologicalMetricOptions::degree_only();
        require(
            degree_only.degree && !degree_only.closeness &&
                !degree_only.eigenvector && !degree_only.betweenness &&
                degree_only.normalize && degree_only.worker_count == 0,
            "constructor degree-only defaults are wrong");

        const Graph path3 = make_graph(
            3,
            {{0, 1}, {1, 2}});

        const TopologicalMetrics path3_normalized =
            TopologicalMetricCalculator::calculate(path3, defaults);
        const std::vector<std::uint32_t> zero_one_zero{
            UINT32_C(0x00000000),
            UINT32_C(0x3f800000),
            UINT32_C(0x00000000)};
        require_column_bits(
            path3_normalized.node_degree_centrality,
            zero_one_zero,
            "path3 normalized degree");
        require_column_bits(
            path3_normalized.node_closeness_centrality,
            zero_one_zero,
            "path3 normalized closeness");
        require_column_bits(
            path3_normalized.node_eigenvector_centrality,
            zero_one_zero,
            "path3 normalized eigenvector");
        require_column_bits(
            path3_normalized.node_betweenness_centrality,
            zero_one_zero,
            "path3 normalized betweenness");

        TopologicalMetricOptions raw_options = defaults;
        raw_options.normalize = false;
        raw_options.worker_count = 1;
        const TopologicalMetrics path3_raw =
            TopologicalMetricCalculator::calculate(path3, raw_options);
        require_column_bits(
            path3_raw.node_degree_centrality,
            {UINT32_C(0x3f000000), UINT32_C(0x3f800000),
             UINT32_C(0x3f000000)},
            "path3 raw degree");
        require_column_bits(
            path3_raw.node_closeness_centrality,
            {UINT32_C(0x3f2aaaab), UINT32_C(0x3f800000),
             UINT32_C(0x3f2aaaab)},
            "path3 raw closeness");
        require_column_bits(
            path3_raw.node_betweenness_centrality,
            {UINT32_C(0x00000000), UINT32_C(0x3f800000),
             UINT32_C(0x00000000)},
            "path3 raw betweenness");
        require(
            path3_raw.node_eigenvector_centrality.has_value() &&
                path3_raw.node_eigenvector_centrality->rows() == 3 &&
                float_bits((*path3_raw.node_eigenvector_centrality)[0]) ==
                    float_bits((*path3_raw.node_eigenvector_centrality)[2]) &&
                (*path3_raw.node_eigenvector_centrality)[1] >
                    (*path3_raw.node_eigenvector_centrality)[0],
            "path3 raw eigenvector shape/symmetry mismatch");

        TopologicalMetricCalculator calculator(path3);
        require(
            &calculator.network == &path3,
            "calculator did not retain the original network identity");
        require_presence(calculator.metrics, 1, "constructor defaults");
        require_metrics_equal(
            calculator.metrics,
            TopologicalMetricCalculator::calculate(path3, degree_only),
            "constructor/static degree-only parity");

        // On this P4 insertion order, NetworkX's symmetric low eigenvector
        // scores differ in double precision but collapse to the same float32.
        // Casting first must therefore produce exact zeros after min/max.
        const Graph cast_before_normalize = make_graph(
            4,
            {{0, 1}, {0, 3}, {2, 3}});
        TopologicalMetricOptions eigen_only = options_for_mask(4, true, 1);
        const TopologicalMetrics cast_result =
            TopologicalMetricCalculator::calculate(
                cast_before_normalize,
                eigen_only);
        require_column_bits(
            cast_result.node_eigenvector_centrality,
            {UINT32_C(0x3f800000), UINT32_C(0x00000000),
             UINT32_C(0x00000000), UINT32_C(0x3f800000)},
            "float32 cast-before-normalize fixture");

        const Graph null_graph = make_graph(0);
        for (const unsigned mask : {0U, 1U, 2U, 8U})
        {
            const TopologicalMetrics result =
                TopologicalMetricCalculator::calculate(
                    null_graph,
                    options_for_mask(mask, true, 4));
            require_presence(result, mask, "empty graph presence");
            for (std::size_t index = 0;
                 index < kMetricMembers.size();
                 ++index)
            {
                const auto& metric = result.*kMetricMembers[index];
                if (metric.has_value())
                {
                    require(
                        metric->rows() == 0 && metric->columns() == 1,
                        "empty graph metric shape mismatch");
                }
            }
        }
        require_throws<std::domain_error>(
            [&] {
                (void)TopologicalMetricCalculator::calculate(
                    null_graph,
                    options_for_mask(4, true, 8));
            },
            "null graph",
            "empty eigenvector parity");
        require_throws<std::domain_error>(
            [&] {
                (void)TopologicalMetricCalculator::calculate(
                    null_graph,
                    defaults);
            },
            "null graph",
            "empty default calculate parity");

        const Graph singleton = make_graph(1);
        const TopologicalMetrics singleton_normalized =
            TopologicalMetricCalculator::calculate(singleton, defaults);
        for (std::size_t index = 0;
             index < kMetricMembers.size();
             ++index)
        {
            require_column_bits(
                singleton_normalized.*kMetricMembers[index],
                {UINT32_C(0x00000000)},
                "singleton normalized/" +
                    std::string(kMetricNames[index]));
        }
        const TopologicalMetrics singleton_raw =
            TopologicalMetricCalculator::calculate(singleton, raw_options);
        require_column_bits(
            singleton_raw.node_degree_centrality,
            {UINT32_C(0x3f800000)},
            "singleton raw degree");
        require_column_bits(
            singleton_raw.node_closeness_centrality,
            {UINT32_C(0x00000000)},
            "singleton raw closeness");
        require_column_bits(
            singleton_raw.node_eigenvector_centrality,
            {UINT32_C(0x3f800000)},
            "singleton raw eigenvector");
        require_column_bits(
            singleton_raw.node_betweenness_centrality,
            {UINT32_C(0x00000000)},
            "singleton raw betweenness");

        const Graph disconnected = make_graph(
            4,
            {{0, 1}, {1, 2}});
        const TopologicalMetrics disconnected_raw =
            TopologicalMetricCalculator::calculate(disconnected, raw_options);
        require_column_bits(
            disconnected_raw.node_degree_centrality,
            {UINT32_C(0x3eaaaaab), UINT32_C(0x3f2aaaab),
             UINT32_C(0x3eaaaaab), UINT32_C(0x00000000)},
            "disconnected raw degree");
        require_column_bits(
            disconnected_raw.node_closeness_centrality,
            {UINT32_C(0x3ee38e39), UINT32_C(0x3f2aaaab),
             UINT32_C(0x3ee38e39), UINT32_C(0x00000000)},
            "disconnected raw closeness");
        require_column_bits(
            disconnected_raw.node_betweenness_centrality,
            {UINT32_C(0x00000000), UINT32_C(0x3eaaaaab),
             UINT32_C(0x00000000), UINT32_C(0x00000000)},
            "disconnected raw betweenness");
        require(
            disconnected_raw.node_eigenvector_centrality.has_value() &&
                disconnected_raw.node_eigenvector_centrality->rows() == 4 &&
                (*disconnected_raw.node_eigenvector_centrality)[1] >
                    (*disconnected_raw.node_eigenvector_centrality)[0] &&
                (*disconnected_raw.node_eigenvector_centrality)[0] ==
                    (*disconnected_raw.node_eigenvector_centrality)[2] &&
                (*disconnected_raw.node_eigenvector_centrality)[3] >= 0.0F,
            "disconnected eigenvector invariants mismatch");

        const Graph self_loop = make_graph(
            4,
            {{0, 0}, {0, 1}, {1, 2}, {2, 3}});
        const TopologicalMetrics self_loop_raw =
            TopologicalMetricCalculator::calculate(self_loop, raw_options);
        require_column_bits(
            self_loop_raw.node_degree_centrality,
            {UINT32_C(0x3f800000), UINT32_C(0x3f2aaaab),
             UINT32_C(0x3f2aaaab), UINT32_C(0x3eaaaaab)},
            "self-loop degree counts loop twice");
        require_column_bits(
            self_loop_raw.node_closeness_centrality,
            {UINT32_C(0x3f000000), UINT32_C(0x3f400000),
             UINT32_C(0x3f400000), UINT32_C(0x3f000000)},
            "self-loop unweighted closeness");
        require_column_bits(
            self_loop_raw.node_betweenness_centrality,
            {UINT32_C(0x00000000), UINT32_C(0x3f2aaaab),
             UINT32_C(0x3f2aaaab), UINT32_C(0x00000000)},
            "self-loop unweighted betweenness");

        const Graph complete5 = make_graph(
            5,
            {{0, 1}, {0, 2}, {0, 3}, {0, 4},
             {1, 2}, {1, 3}, {1, 4},
             {2, 3}, {2, 4}, {3, 4}});
        const TopologicalMetrics complete_normalized =
            TopologicalMetricCalculator::calculate(complete5, defaults);
        for (std::size_t index = 0;
             index < kMetricMembers.size();
             ++index)
        {
            require_column_bits(
                complete_normalized.*kMetricMembers[index],
                std::vector<std::uint32_t>(5, UINT32_C(0x00000000)),
                "constant metric normalization/" +
                    std::string(kMetricNames[index]));
        }

        const Graph path20 = make_graph(
            20,
            [] {
                std::vector<EdgeEndpoints> edges;
                for (Vertex node = 0; node + 1 < 20; ++node)
                {
                    edges.emplace_back(node, node + 1);
                }
                return edges;
            }());
        for (const std::size_t workers : {1U, 8U})
        {
            require_throws<std::runtime_error>(
                [&] {
                    (void)TopologicalMetricCalculator::calculate(
                        path20,
                        options_for_mask(4, true, workers));
                },
                "failed to converge within 100 iterations",
                "P20 eigenvector non-convergence");
        }
        const TopologicalMetrics path20_disabled =
            TopologicalMetricCalculator::calculate(
                path20,
                options_for_mask(0, true, 8));
        require_presence(
            path20_disabled,
            0,
            "disabled metrics must not run eigenvector");

        // Asymmetric, connected, deliberately non-node-major insertion order.
        const Graph option_fixture = make_graph(
            8,
            {{6, 7}, {4, 6}, {3, 5}, {2, 4}, {1, 3},
             {0, 2}, {0, 1}, {2, 3}, {4, 5}, {1, 6}});
        for (const bool normalize : {false, true})
        {
            const TopologicalMetrics full_reference =
                TopologicalMetricCalculator::calculate(
                    option_fixture,
                    options_for_mask(15, normalize, 1));
            for (unsigned mask = 0; mask < 16; ++mask)
            {
                TopologicalMetrics expected;
                for (std::size_t index = 0;
                     index < kMetricMembers.size();
                     ++index)
                {
                    if ((mask & (1U << index)) != 0)
                    {
                        expected.*kMetricMembers[index] =
                            full_reference.*kMetricMembers[index];
                    }
                }

                for (const std::size_t workers :
                     {1U, 2U, 4U, 8U, 0U})
                {
                    const TopologicalMetrics actual =
                        TopologicalMetricCalculator::calculate(
                            option_fixture,
                            options_for_mask(mask, normalize, workers));
                    const std::string context =
                        "option matrix mask=" + std::to_string(mask) +
                        "/normalize=" + std::to_string(normalize) +
                        "/workers=" + std::to_string(workers);
                    require_presence(actual, mask, context);
                    require_metrics_equal(actual, expected, context);
                }
            }
        }

        const TopologicalMetrics concurrent_reference =
            TopologicalMetricCalculator::calculate(
                option_fixture,
                options_for_mask(15, true, 1));
        std::atomic<bool> concurrent_calculate_ok{true};
        std::vector<std::thread> calculation_callers;
        for (std::size_t caller = 0; caller < 6; ++caller)
        {
            calculation_callers.emplace_back([&, caller]
            {
                try
                {
                    const std::array<std::size_t, 4> widths{1, 2, 4, 8};
                    for (std::size_t iteration = 0;
                         iteration < 12;
                         ++iteration)
                    {
                        const TopologicalMetrics current =
                            TopologicalMetricCalculator::calculate(
                                option_fixture,
                                options_for_mask(
                                    15,
                                    true,
                                    widths[(caller + iteration) %
                                           widths.size()]));
                        if (!metrics_bit_equal_noexcept(
                                current,
                                concurrent_reference))
                        {
                            concurrent_calculate_ok.store(false);
                            return;
                        }
                    }
                }
                catch (...)
                {
                    concurrent_calculate_ok.store(false);
                }
            });
        }
        for (auto& caller : calculation_callers)
        {
            caller.join();
        }
        require(
            concurrent_calculate_ok.load(),
            "concurrent calculate changed output or threw");

        TopologicalMetricCalculator::clear_cache();
        require(
            !TopologicalMetricCalculator::get_from_cache("missing"),
            "cache miss must return null");
        require_throws<std::invalid_argument>(
            [] {
                TopologicalMetricCalculator::add_to_cache(
                    "null",
                    nullptr);
            },
            "must not be null",
            "cache must reject null metrics");

        auto cached_a = std::make_shared<TopologicalMetrics>(path3_raw);
        auto cached_b = std::make_shared<TopologicalMetrics>(path3_normalized);
        TopologicalMetricCalculator::add_to_cache("p_net", cached_a);
        auto first_get =
            TopologicalMetricCalculator::get_from_cache("p_net");
        require(
            first_get && first_get.get() == cached_a.get(),
            "cache did not preserve shared identity");
        (*first_get->node_degree_centrality)[0] = 0.25F;
        require(
            float_bits((*cached_a->node_degree_centrality)[0]) ==
                UINT32_C(0x3e800000),
            "cache retrieval did not alias the stored metrics");

        TopologicalMetricCalculator::add_to_cache("p_net", cached_b);
        auto second_get =
            TopologicalMetricCalculator::get_from_cache("p_net");
        require(
            second_get && second_get.get() == cached_b.get() &&
                first_get.get() == cached_a.get(),
            "cache overwrite identity/lifetime mismatch");
        TopologicalMetricCalculator::add_to_cache("", cached_a);
        TopologicalMetricCalculator::add_to_cache("P_NET", cached_a);
        require(
            TopologicalMetricCalculator::get_from_cache("").get() ==
                    cached_a.get() &&
                TopologicalMetricCalculator::get_from_cache("P_NET").get() ==
                    cached_a.get() &&
                TopologicalMetricCalculator::get_from_cache("p_net").get() ==
                    cached_b.get(),
            "cache keys are not empty-safe/case-sensitive");

        TopologicalMetricCalculator::clear_cache();
        require(
            !TopologicalMetricCalculator::get_from_cache("p_net") &&
                second_get.get() == cached_b.get(),
            "cache clear did not remove keys or preserve acquired lifetime");

        std::atomic<bool> start_cache_stress{false};
        std::atomic<bool> cache_stress_ok{true};
        std::vector<std::thread> cache_threads;
        for (std::size_t writer = 0; writer < 2; ++writer)
        {
            cache_threads.emplace_back([&, writer]
            {
                while (!start_cache_stress.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                for (std::size_t iteration = 0;
                     iteration < 1000;
                     ++iteration)
                {
                    TopologicalMetricCalculator::add_to_cache(
                        "race",
                        ((iteration + writer) & 1U) == 0
                            ? cached_a
                            : cached_b);
                }
            });
        }
        for (std::size_t reader = 0; reader < 4; ++reader)
        {
            cache_threads.emplace_back([&]
            {
                while (!start_cache_stress.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                for (std::size_t iteration = 0;
                     iteration < 1500;
                     ++iteration)
                {
                    const auto value =
                        TopologicalMetricCalculator::get_from_cache("race");
                    if (value && value.get() != cached_a.get() &&
                        value.get() != cached_b.get())
                    {
                        cache_stress_ok.store(false);
                    }
                }
            });
        }
        cache_threads.emplace_back([&]
        {
            while (!start_cache_stress.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }
            for (std::size_t iteration = 0;
                 iteration < 250;
                 ++iteration)
            {
                TopologicalMetricCalculator::clear_cache();
            }
        });
        start_cache_stress.store(true, std::memory_order_release);
        for (auto& thread : cache_threads)
        {
            thread.join();
        }
        require(
            cache_stress_ok.load(),
            "concurrent cache returned an invalid object");
        TopologicalMetricCalculator::add_to_cache("race", cached_b);
        require(
            TopologicalMetricCalculator::get_from_cache("race").get() ==
                cached_b.get(),
            "cache unusable after concurrent add/get/clear");
        TopologicalMetricCalculator::clear_cache();

        std::cout << "vne_topological_metric_calculator_unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "vne_topological_metric_calculator_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
