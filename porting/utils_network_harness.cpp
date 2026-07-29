#include "network.h"

#include "generators/gml_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using virne::utils::BfsTreeLevels;
using virne::utils::DynamicDictList;
using virne::utils::DynamicKey;
using virne::utils::DynamicValue;
using virne::utils::PathLinks;

constexpr std::size_t kPathNodes = 262145;
constexpr std::size_t kBfsSources = 50;
constexpr std::size_t kFlattenBlocks = 128;
constexpr std::size_t kFlattenWidth = 256;
constexpr std::size_t kGmlDicts = 4096;
constexpr std::size_t kSanitizeDicts = 16384;

std::string hex_encode(std::string_view value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2);
    for (const unsigned char byte : value)
    {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 0x0f]);
    }
    return result;
}

std::string encode_path(const PathLinks& links)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < links.size(); ++index)
    {
        if (index != 0)
        {
            stream << ';';
        }
        stream << links[index].first << ',' << links[index].second;
    }
    return stream.str();
}

std::string encode_levels(const BfsTreeLevels& levels)
{
    std::ostringstream stream;
    for (std::size_t depth = 0; depth < levels.size(); ++depth)
    {
        if (depth != 0)
        {
            stream << '|';
        }
        for (std::size_t index = 0; index < levels[depth].size(); ++index)
        {
            if (index != 0)
            {
                stream << ',';
            }
            stream << levels[depth][index];
        }
    }
    return stream.str();
}

std::string encode_dynamic(const DynamicValue& value)
{
    if (value.is<std::monostate>())
    {
        return "n";
    }
    if (value.is<bool>())
    {
        return value.as<bool>() ? "b1" : "b0";
    }
    if (value.is<std::int64_t>())
    {
        return "i" + std::to_string(value.as<std::int64_t>());
    }
    if (value.is<double>())
    {
        std::ostringstream stream;
        stream << 'f' << std::setprecision(17) << value.as<double>();
        return stream.str();
    }
    if (value.is<std::string>())
    {
        return "s" + hex_encode(value.as<std::string>());
    }
    throw std::logic_error("parity encoder expected a scalar DynamicValue");
}

std::string encode_flat(const std::vector<DynamicValue>& values)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
        {
            stream << ';';
        }
        stream << encode_dynamic(values[index]);
    }
    return stream.str();
}

std::string encode_dicts(const DynamicDictList& dicts)
{
    std::ostringstream stream;
    for (std::size_t dict_index = 0; dict_index < dicts.size(); ++dict_index)
    {
        if (dict_index != 0)
        {
            stream << '|';
        }
        for (std::size_t entry_index = 0;
             entry_index < dicts[dict_index].size();
             ++entry_index)
        {
            if (entry_index != 0)
            {
                stream << ';';
            }
            const auto& [key, value] = dicts[dict_index][entry_index];
            if (!key.is<std::string>())
            {
                throw std::logic_error("encoded dict key is not a string");
            }
            stream << hex_encode(key.as<std::string>()) << '='
                   << encode_dynamic(value);
        }
    }
    return stream.str();
}

void emit_parity(std::string_view key, const std::string& value)
{
    std::cout << "PARITY\t" << key << '\t' << value << '\n';
}

void emit_parity()
{
    using virne::utils::flatten_dict_list_for_gml;
    using virne::utils::flatten_recurrent_dict;
    using virne::utils::get_bfs_tree_level;
    using virne::utils::get_bfs_tree_levels;
    using virne::utils::path_to_links;
    using virne::utils::sanitize_attr_setting;

    emit_parity("path", encode_path(path_to_links({4, 1, 7, 9})));

    const std::vector<EdgeEndpoints> edges{
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 5}};
    const Graph graph(7, edges);
    emit_parity("bfs", encode_levels(get_bfs_tree_level(graph, 0)));
    emit_parity("bfs_isolated", encode_levels(get_bfs_tree_level(graph, 6)));

    const std::vector<Vertex> sources{0, 1, 2, 6};
    emit_parity(
        "bfs_parallel_same",
        get_bfs_tree_levels(graph, sources, 1) ==
                get_bfs_tree_levels(graph, sources, 4)
            ? "1"
            : "0");

    const std::vector<EdgeEndpoints> arcs{{0, 1}, {2, 1}, {1, 3}};
    const DiGraph digraph(4, arcs);
    emit_parity("digraph_bfs", encode_levels(get_bfs_tree_level(digraph, 2)));

    const DynamicValue nested(DynamicValue::Dict{
        {"numbers", DynamicValue::List{1, 2.5}},
        {"nested", DynamicValue::Dict{
             {"name", "end"}, {"flag", true}, {"negative", -4}}}});
    emit_parity("flatten", encode_flat(flatten_recurrent_dict(nested)));
    emit_parity(
        "flatten_empty",
        encode_flat(flatten_recurrent_dict(DynamicValue::Dict{})));

    const DynamicDictList gml_input{{
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
        {"positive_inf", std::numeric_limits<double>::infinity()},
        {"negative_inf", -std::numeric_limits<double>::infinity()},
        {"nan", std::numeric_limits<double>::quiet_NaN()},
        {"denorm_min", std::numeric_limits<double>::denorm_min()},
        {"min_normal", std::numeric_limits<double>::min()},
        {"max", std::numeric_limits<double>::max()},
        {"roundtrip", 1.2345678901234567},
        {"both_quotes", DynamicValue::List{std::string("a'\"b\\c\n")}},
        {7, "numeric-key"},
        {1, "first-collision"},
        {"1", "last-collision"}},
        {{true, "truth"}, {1, "integer-last"}}};
    emit_parity(
        "gml", encode_dicts(flatten_dict_list_for_gml(gml_input, 4)));

    DynamicDictList float_corpus;
    float_corpus.reserve(512);
    std::uint64_t float_bits = 0x9e3779b97f4a7c15ULL;
    for (std::size_t index = 0; index < 512; ++index)
    {
        float_bits = float_bits * 6364136223846793005ULL +
                     1442695040888963407ULL;
        double value = 0.0;
        static_assert(sizeof(value) == sizeof(float_bits));
        std::memcpy(&value, &float_bits, sizeof(value));
        float_corpus.push_back({{"value", value}});
    }
    emit_parity(
        "gml_float_corpus",
        encode_dicts(flatten_dict_list_for_gml(float_corpus)));

    DynamicDictList settings{
        {{"name", "cpu"}, {"low", " -7 "}, {"high", "+12"}},
        {{"name", "bw"}, {"low", 3.8}, {"high", true}},
        {{"name", "ram"}, {"low", "1_000"}, {"high", "2_000"}}};
    auto* const identity = &settings;
    emit_parity(
        "sanitize_identity",
        &sanitize_attr_setting(settings) == identity ? "1" : "0");
    emit_parity("sanitize", encode_dicts(settings));

    auto emits_error = [](std::string_view name, auto&& function) {
        try
        {
            function();
            emit_parity(name, "0");
        }
        catch (const std::exception&)
        {
            emit_parity(name, "1");
        }
    };
    emits_error("path_short_error", [] { (void)path_to_links({1}); });
    emits_error("bfs_missing_error", [&] {
        (void)get_bfs_tree_level(graph, 99);
    });
    emits_error("flatten_none_error", [] {
        (void)flatten_recurrent_dict(
            DynamicValue::List{1, DynamicValue(nullptr)});
    });
    emits_error("sanitize_invalid_error", [] {
        DynamicDictList invalid{{{"low", "1.5"}}};
        (void)sanitize_attr_setting(invalid);
    });
}

std::uint64_t checksum_path(const PathLinks& links)
{
    std::uint64_t checksum = 0;
    for (const auto& [u, v] : links)
    {
        checksum += (static_cast<std::uint64_t>(u) + 1U) * 17U;
        checksum += (static_cast<std::uint64_t>(v) + 1U) * 31U;
    }
    return checksum;
}

std::uint64_t checksum_bfs(
    const std::vector<BfsTreeLevels>& batches)
{
    std::uint64_t checksum = 0;
    for (std::size_t batch = 0; batch < batches.size(); ++batch)
    {
        for (std::size_t depth = 0; depth < batches[batch].size(); ++depth)
        {
            for (const Vertex vertex : batches[batch][depth])
            {
                checksum += (batch + 1U) * 1000003U;
                checksum += (depth + 1U) * 1009U;
                checksum += static_cast<std::uint64_t>(vertex);
            }
        }
    }
    return checksum;
}

std::uint64_t checksum_flat(const std::vector<DynamicValue>& values)
{
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        checksum += (index + 1U) *
            static_cast<std::uint64_t>(values[index].as<std::int64_t>() + 1);
    }
    return checksum;
}

void fnv_add(std::uint64_t& hash, std::string_view value)
{
    for (const unsigned char byte : value)
    {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    hash ^= 0xffU;
    hash *= 1099511628211ULL;
}

std::uint64_t checksum_dicts(const DynamicDictList& dicts)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto& dict : dicts)
    {
        for (const auto& [key, value] : dict)
        {
            if (!key.is<std::string>() || !value.is<std::string>())
            {
                throw std::logic_error("GML checksum expects string pairs");
            }
            fnv_add(hash, key.as<std::string>());
            fnv_add(hash, value.as<std::string>());
        }
    }
    return hash;
}

std::uint64_t checksum_sanitized(const DynamicDictList& dicts)
{
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < dicts.size(); ++index)
    {
        for (const auto& [key, value] : dicts[index])
        {
            if (key.is<std::string>() &&
                (key.as<std::string>() == "low" ||
                 key.as<std::string>() == "high"))
            {
                checksum += (index + 1U) *
                    static_cast<std::uint64_t>(
                        value.as<std::int64_t>() + 100000);
            }
        }
    }
    return checksum;
}

struct Measurement
{
    std::vector<std::uint64_t> samples_ns;
    std::uint64_t checksum = 0;
};

template <typename Prepare, typename Action, typename Checksum>
Measurement measure(
    std::size_t warmups,
    std::size_t repetitions,
    Prepare&& prepare,
    Action&& action,
    Checksum&& checksum)
{
    Measurement result;
    result.samples_ns.reserve(repetitions);
    const std::size_t total = warmups + repetitions;
    for (std::size_t sample = 0; sample < total; ++sample)
    {
        prepare();
        const auto start = Clock::now();
        action();
        const auto stop = Clock::now();
        result.checksum = checksum();
        if (sample >= warmups)
        {
            result.samples_ns.push_back(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        stop - start)
                        .count()));
        }
    }
    return result;
}

void emit_benchmark(
    std::string_view name,
    std::size_t workers,
    const Measurement& measurement)
{
    std::cout << "BENCH\t" << name << '\t' << workers << '\t';
    for (std::size_t index = 0;
         index < measurement.samples_ns.size();
         ++index)
    {
        if (index != 0)
        {
            std::cout << ',';
        }
        std::cout << measurement.samples_ns[index];
    }
    std::cout << '\t' << measurement.checksum << '\n';
}

DynamicValue make_flatten_fixture()
{
    DynamicValue::List blocks;
    blocks.reserve(kFlattenBlocks);
    for (std::size_t block = 0; block < kFlattenBlocks; ++block)
    {
        DynamicValue::List values;
        values.reserve(kFlattenWidth + 1);
        for (std::size_t offset = 0; offset < kFlattenWidth; ++offset)
        {
            values.emplace_back(static_cast<std::int64_t>(block + offset));
        }
        values.emplace_back(static_cast<std::int64_t>(block));
        blocks.emplace_back(DynamicValue::Dict{{"values", std::move(values)}});
    }
    return DynamicValue(std::move(blocks));
}

DynamicDictList make_gml_fixture()
{
    DynamicDictList dicts;
    dicts.reserve(kGmlDicts);
    for (std::size_t index = 0; index < kGmlDicts; ++index)
    {
        dicts.push_back({
            {"name", "cpu"},
            {"index", static_cast<std::int64_t>(index)},
            {"active", index % 2 == 0},
            {"ratio", 1.5},
            {"none", DynamicValue(nullptr)},
            {"items", DynamicValue::List{1, "x"}}});
    }
    return dicts;
}

DynamicDictList make_sanitize_fixture()
{
    DynamicDictList dicts;
    dicts.reserve(kSanitizeDicts);
    for (std::size_t index = 0; index < kSanitizeDicts; ++index)
    {
        dicts.push_back({
            {"name", "resource"},
            {"low", std::to_string(index % 97)},
            {"high", std::to_string(100 + index % 193)}});
    }
    return dicts;
}

void emit_benchmark(
    const std::string& fixture_path,
    std::size_t warmups,
    std::size_t repetitions,
    std::size_t workers)
{
    using virne::utils::flatten_dict_list_for_gml;
    using virne::utils::flatten_recurrent_dict;
    using virne::utils::get_bfs_tree_levels;
    using virne::utils::path_to_links;
    using virne::utils::sanitize_attr_setting;

    if (warmups == 0 || repetitions == 0 || workers == 0)
    {
        throw std::invalid_argument(
            "warmups, repetitions and workers must be positive");
    }

    std::cout << "META\tpath_nodes\t" << kPathNodes << '\n'
              << "META\tbfs_sources\t" << kBfsSources << '\n'
              << "META\tflatten_blocks\t" << kFlattenBlocks << '\n'
              << "META\tflatten_width\t" << kFlattenWidth << '\n'
              << "META\tgml_dicts\t" << kGmlDicts << '\n'
              << "META\tsanitize_dicts\t" << kSanitizeDicts << '\n';

    std::vector<Vertex> path(kPathNodes);
    for (std::size_t index = 0; index < path.size(); ++index)
    {
        path[index] = index;
    }
    PathLinks path_result;
    emit_benchmark(
        "path_to_links.st",
        1,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { path_result = path_to_links(path, 1); },
            [&] { return checksum_path(path_result); }));
    emit_benchmark(
        "path_to_links.mt",
        workers,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { path_result = path_to_links(path, workers); },
            [&] { return checksum_path(path_result); }));
    emit_benchmark(
        "path_to_links.auto",
        0,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { path_result = path_to_links(path); },
            [&] { return checksum_path(path_result); }));

    const Graph graph = nx::read_gml(fixture_path, "id");
    std::vector<Vertex> sources;
    sources.reserve(kBfsSources);
    for (std::size_t index = 0; index < kBfsSources; ++index)
    {
        sources.push_back((index * 11U) % graph.num_nodes());
    }
    std::vector<BfsTreeLevels> bfs_result;
    emit_benchmark(
        "get_bfs_tree_level.st",
        1,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { bfs_result = get_bfs_tree_levels(graph, sources, 1); },
            [&] { return checksum_bfs(bfs_result); }));
    emit_benchmark(
        "get_bfs_tree_level.mt",
        workers,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { bfs_result = get_bfs_tree_levels(graph, sources, workers); },
            [&] { return checksum_bfs(bfs_result); }));
    emit_benchmark(
        "get_bfs_tree_level.auto",
        0,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { bfs_result = get_bfs_tree_levels(graph, sources); },
            [&] { return checksum_bfs(bfs_result); }));

    std::vector<Vertex> full_sources(graph.num_nodes());
    for (std::size_t index = 0; index < full_sources.size(); ++index)
    {
        full_sources[index] = index;
    }
    emit_benchmark(
        "get_bfs_tree_level_full.st",
        1,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { bfs_result = get_bfs_tree_levels(graph, full_sources, 1); },
            [&] { return checksum_bfs(bfs_result); }));
    emit_benchmark(
        "get_bfs_tree_level_full.mt",
        workers,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] {
                bfs_result =
                    get_bfs_tree_levels(graph, full_sources, workers);
            },
            [&] { return checksum_bfs(bfs_result); }));
    emit_benchmark(
        "get_bfs_tree_level_full.auto",
        0,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { bfs_result = get_bfs_tree_levels(graph, full_sources); },
            [&] { return checksum_bfs(bfs_result); }));

    const DynamicValue flatten_input = make_flatten_fixture();
    std::vector<DynamicValue> flatten_result;
    emit_benchmark(
        "flatten_recurrent_dict.st",
        1,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { flatten_result = flatten_recurrent_dict(flatten_input); },
            [&] { return checksum_flat(flatten_result); }));

    const DynamicDictList gml_input = make_gml_fixture();
    DynamicDictList gml_result;
    emit_benchmark(
        "flatten_dict_list_for_gml.st",
        1,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { gml_result = flatten_dict_list_for_gml(gml_input, 1); },
            [&] { return checksum_dicts(gml_result); }));
    emit_benchmark(
        "flatten_dict_list_for_gml.mt",
        workers,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { gml_result = flatten_dict_list_for_gml(gml_input, workers); },
            [&] { return checksum_dicts(gml_result); }));
    emit_benchmark(
        "flatten_dict_list_for_gml.auto",
        0,
        measure(
            warmups,
            repetitions,
            [] {},
            [&] { gml_result = flatten_dict_list_for_gml(gml_input); },
            [&] { return checksum_dicts(gml_result); }));

    const DynamicDictList sanitize_base = make_sanitize_fixture();
    DynamicDictList sanitize_result;
    emit_benchmark(
        "sanitize_attr_setting.st",
        1,
        measure(
            warmups,
            repetitions,
            [&] { sanitize_result = sanitize_base; },
            [&] { (void)sanitize_attr_setting(sanitize_result); },
            [&] { return checksum_sanitized(sanitize_result); }));
}

std::size_t parse_size(const char* text, std::string_view name)
{
    const std::string value(text);
    std::size_t consumed = 0;
    const auto result = std::stoull(value, &consumed);
    if (consumed != value.size())
    {
        throw std::invalid_argument(std::string(name) + " is not an integer");
    }
    return static_cast<std::size_t>(result);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "--parity")
        {
            emit_parity();
            return 0;
        }
        if (argc == 6 && std::string_view(argv[1]) == "--benchmark")
        {
            emit_benchmark(
                argv[2],
                parse_size(argv[3], "warmups"),
                parse_size(argv[4], "repetitions"),
                parse_size(argv[5], "workers"));
            return 0;
        }

        std::cerr
            << "usage:\n"
            << "  vne_utils_network_harness --parity\n"
            << "  vne_utils_network_harness --benchmark "
               "<Waxman500.gml> <warmups> <repetitions> <workers>\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "vne_utils_network_harness: " << error.what() << '\n';
        return 1;
    }
}
