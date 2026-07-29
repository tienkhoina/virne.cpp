#include "network.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid_argument(Function&& function, const std::string& message)
{
    bool caught = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    require(caught, message);
}

template <typename Function>
void require_exception(Function&& function, const std::string& message)
{
    bool caught = false;
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        caught = true;
    }
    require(caught, message);
}

const virne::utils::DynamicValue& value_at(
    const virne::utils::DynamicValue::Dict& dict,
    const std::string& key)
{
    for (const auto& entry : dict)
    {
        if (entry.first.is<std::string>() &&
            entry.first.as<std::string>() == key)
        {
            return entry.second;
        }
    }
    throw std::out_of_range("missing dynamic dict key: " + key);
}

} // namespace

int main()
{
    using virne::utils::DynamicValue;
    using virne::utils::flatten_dict_list_for_gml;
    using virne::utils::flatten_recurrent_dict;
    using virne::utils::get_bfs_tree_level;
    using virne::utils::get_bfs_tree_levels;
    using virne::utils::path_to_links;
    using virne::utils::sanitize_attr_setting;

    try
    {
        const auto links = path_to_links({4, 1, 7, 9});
        require(
            links == virne::utils::PathLinks{{4, 1}, {1, 7}, {7, 9}},
            "path_to_links order mismatch");
        require_invalid_argument(
            [] { (void)path_to_links({1}); },
            "path_to_links must reject one-vertex paths");

        std::vector<Vertex> long_path(100000);
        for (std::size_t index = 0; index < long_path.size(); ++index)
        {
            long_path[index] = index;
        }
        const auto parallel_links = path_to_links(long_path, 4);
        require(parallel_links.size() == 99999, "parallel path link count");
        require(parallel_links.front() == std::pair<Vertex, Vertex>{0, 1},
                "parallel path first link");
        require(parallel_links.back() == std::pair<Vertex, Vertex>{99998, 99999},
                "parallel path last link");
        require(path_to_links(long_path) == path_to_links(long_path, 1),
                "automatic path mode must preserve sequential output");

        const Graph graph(
            7,
            std::vector<EdgeEndpoints>{
                {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 5}});
        const virne::utils::BfsTreeLevels expected_levels{
            {0}, {1, 2}, {3, 4}, {5}};
        require(
            get_bfs_tree_level(graph, 0) == expected_levels,
            "BFS levels or discovery order mismatch");

        const std::vector<Vertex> sources{0, 1, 2, 6};
        require(
            get_bfs_tree_levels(graph, sources, 1) ==
                get_bfs_tree_levels(graph, sources, 4),
            "parallel BFS batch changed deterministic results");
        require(
            get_bfs_tree_levels(graph, {0, 0, 6}, 4).size() == 3,
            "parallel BFS must preserve duplicate sources");
        require(
            get_bfs_tree_levels(graph, {}, 4).empty(),
            "parallel BFS must accept an empty source batch");

        const auto concurrent_expected =
            get_bfs_tree_levels(graph, sources, 1);
        std::atomic<bool> concurrent_ok{true};
        std::vector<std::thread> callers;
        for (std::size_t caller = 0; caller < 4; ++caller)
        {
            callers.emplace_back([&] {
                for (std::size_t iteration = 0; iteration < 20; ++iteration)
                {
                    if (get_bfs_tree_levels(graph, sources) !=
                        concurrent_expected)
                    {
                        concurrent_ok.store(false);
                    }
                }
            });
        }
        for (auto& caller : callers)
        {
            caller.join();
        }
        require(concurrent_ok.load(),
                "shared two-worker executor changed concurrent results");

        const DynamicValue nested(DynamicValue::Dict{
            {"numbers", DynamicValue::List{1, 2.5}},
            {"nested", DynamicValue::Dict{
                {"name", "end"},
                {"flag", true},
                {"negative", -4}}}});
        const std::vector<DynamicValue> expected_flat{
            1, 2.5, "end", true, -4};
        require(
            flatten_recurrent_dict(nested) == expected_flat,
            "recursive flatten order/value mismatch");
        require_invalid_argument(
            [] {
                (void)flatten_recurrent_dict(
                    DynamicValue::List{1, DynamicValue(nullptr)});
            },
            "recursive flatten must reject null leaves");

        const virne::utils::DynamicDictList gml_input{{
            {"label", "cpu"},
            {"active", true},
            {"none", DynamicValue(nullptr)},
            {"ratio", 1.5},
            {"items", DynamicValue::List{1, "x"}},
            {"quote", DynamicValue::List{"a'b"}},
            {"control", DynamicValue::List{std::string("a\0", 2)}},
            {"fixed6", 1e6},
            {"fixed15", 1e15},
            {"scientific16", 1e16},
            {"fixed_negative4", 1e-4},
            {"scientific_negative5", 1e-5},
            {"negative_zero", -0.0},
            {7, "numeric-key"},
            {1, "first-collision"},
            {"1", "last-collision"}},
            {{true, "truth"}, {1, "integer-last"}}};
        const auto gml_output = flatten_dict_list_for_gml(gml_input, 4);
        require(gml_output.size() == 2, "GML output size");
        require(value_at(gml_output[0], "label").as<std::string>() == "cpu",
                "GML string conversion");
        require(value_at(gml_output[0], "active").as<std::string>() == "True",
                "GML bool conversion");
        require(value_at(gml_output[0], "none").as<std::string>() == "None",
                "GML None conversion");
        require(value_at(gml_output[0], "ratio").as<std::string>() == "1.5",
                "GML float conversion");
        require(value_at(gml_output[0], "items").as<std::string>() == "[1, 'x']",
                "GML list conversion");
        require(value_at(gml_output[0], "quote").as<std::string>() ==
                    "[\"a'b\"]",
                "GML Python quote selection");
        require(value_at(gml_output[0], "control").as<std::string>() ==
                    "['a\\x00']",
                "GML Python control-character repr");
        require(value_at(gml_output[0], "fixed6").as<std::string>() ==
                    "1000000.0",
                "GML Python fixed float at 1e6");
        require(value_at(gml_output[0], "fixed15").as<std::string>() ==
                    "1000000000000000.0",
                "GML Python fixed float at 1e15");
        require(value_at(gml_output[0], "scientific16").as<std::string>() ==
                    "1e+16",
                "GML Python scientific float at 1e16");
        require(value_at(gml_output[0], "fixed_negative4").as<std::string>() ==
                    "0.0001",
                "GML Python fixed float at 1e-4");
        require(value_at(gml_output[0], "scientific_negative5").as<std::string>() ==
                    "1e-05",
                "GML Python scientific float at 1e-5");
        require(value_at(gml_output[0], "negative_zero").as<std::string>() ==
                    "-0.0",
                "GML Python negative-zero float");
        require(value_at(gml_output[0], "7").as<std::string>() == "numeric-key",
                "GML numeric-key conversion");
        require(value_at(gml_output[0], "1").as<std::string>() ==
                    "last-collision",
                "GML stringified-key collision must keep the last value");
        require(gml_output[0].size() + 1 == gml_input[0].size(),
                "GML stringified-key collision must not duplicate a key");
        require(gml_output[1].size() == 1,
                "GML Python-equal numeric keys must collapse");
        require(value_at(gml_output[1], "True").as<std::string>() ==
                    "integer-last",
                "GML numeric-key collision must retain first key/last value");

        virne::utils::DynamicDictList settings{
            {{"name", "cpu"}, {"low", " -7 "}, {"high", "+12"}},
            {{"name", "bw"}, {"low", 3.8}, {"high", true}},
            {{"name", "ram"}, {"low", "1_000"}, {"high", "2_000"}}};
        auto* const identity = &settings;
        require(&sanitize_attr_setting(settings) == identity,
                "sanitize_attr_setting must mutate and return the input");
        require(value_at(settings[0], "low").as<std::int64_t>() == -7,
                "sanitize string low");
        require(value_at(settings[0], "high").as<std::int64_t>() == 12,
                "sanitize string high");
        require(value_at(settings[1], "low").as<std::int64_t>() == 3,
                "sanitize float truncation");
        require(value_at(settings[1], "high").as<std::int64_t>() == 1,
                "sanitize bool conversion");
        require(value_at(settings[2], "low").as<std::int64_t>() == 1000,
                "sanitize underscored low");
        require(value_at(settings[2], "high").as<std::int64_t>() == 2000,
                "sanitize underscored high");

        require_invalid_argument(
            [&] {
                virne::utils::DynamicDictList invalid{{{"low", "1.5"}}};
                (void)sanitize_attr_setting(invalid);
            },
            "sanitize must reject non-integer strings");
        require_exception(
            [&] {
                virne::utils::DynamicDictList invalid{{
                    {"low", 0x1p63}}};
                (void)sanitize_attr_setting(invalid);
            },
            "sanitize must safely reject double 2^63");

        virne::utils::DynamicDictList partial{{
            {"low", "4"}, {"high", "invalid"}}};
        require_exception(
            [&] { (void)sanitize_attr_setting(partial); },
            "sanitize invalid high must raise");
        require(value_at(partial[0], "low").as<std::int64_t>() == 4,
                "sanitize must preserve Python partial-mutation order");

        std::cout << "vne_utils_network_unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "vne_utils_network_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
