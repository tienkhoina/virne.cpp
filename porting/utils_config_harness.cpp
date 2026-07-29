#include "utils_config.h"

#include <yaml-cpp/yaml.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

using attribute::AttributeKind;
using utils::DatasetAttributeSpec;
using utils::DatasetScalar;
using utils::DatasetTopologyKind;
using utils::DistributionKind;
using utils::DistributionSpec;
using utils::RunDirectoryInput;
using utils::RunIdTimestamp;
using utils::SimulationConfigInput;
using utils::SimulationConfigSummary;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct BatchObservation
{
    std::size_t workers = 1U;
    std::uint64_t checksum = 0U;
    std::vector<std::uint64_t> order;
};

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

DistributionSpec uniform_integer(std::int64_t low, std::int64_t high)
{
    DistributionSpec result;
    result.kind = DistributionKind::uniform;
    result.low = DatasetScalar{low};
    result.high = DatasetScalar{high};
    return result;
}

DatasetAttributeSpec dataset_attribute(
    std::uint32_t id,
    std::string name,
    DistributionSpec distribution = {})
{
    DatasetAttributeSpec result;
    result.id = id;
    result.name = std::move(name);
    result.distribution = std::move(distribution);
    return result;
}

SimulationConfigInput make_simulation_input(std::size_t variant)
{
    constexpr std::uint64_t maximum_variant =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) -
        64U;
    if (variant > maximum_variant)
    {
        throw std::invalid_argument("simulation variant exceeds int64 range");
    }
    const std::int64_t value = static_cast<std::int64_t>(variant);

    SimulationConfigInput input;
    input.physical_dataset.save_dir =
        fs::path("oracle") / ("p" + std::to_string(variant));
    input.physical_dataset.topology.num_nodes = 6 + value;
    input.physical_dataset.topology.topology_type =
        DatasetTopologyKind::waxman;
    input.physical_dataset.topology.wm_alpha = DatasetScalar{0.5};
    input.physical_dataset.topology.wm_beta = DatasetScalar{0.2};
    input.physical_dataset.node_attributes = {
        dataset_attribute(0U, "cpu", uniform_integer(1, 9)),
        dataset_attribute(1U, "max_cpu"),
        dataset_attribute(2U, "tag")};
    input.physical_dataset.link_attributes = {
        dataset_attribute(0U, "bw", uniform_integer(2, 10)),
        dataset_attribute(1U, "max_bw")};

    input.virtual_dataset.save_dir =
        fs::path("oracle") / ("v" + std::to_string(variant));
    input.virtual_dataset.num_virtual_networks = 12 + value;
    input.virtual_dataset.size_low = 2;
    input.virtual_dataset.size_high = 5;
    input.virtual_dataset.topology_type = DatasetTopologyKind::random;
    input.virtual_dataset.lifetime.kind = DistributionKind::exponential;
    input.virtual_dataset.lifetime.scale = DatasetScalar{std::int64_t{50}};
    input.virtual_dataset.arrival_lambda = DatasetScalar{0.25};
    input.virtual_dataset.node_attributes = {
        dataset_attribute(0U, "cpu", uniform_integer(0, 4)),
        dataset_attribute(1U, "status"),
        dataset_attribute(2U, "pos"),
        dataset_attribute(3U, "gpu", uniform_integer(1, 3))};
    input.virtual_dataset.link_attributes = {
        dataset_attribute(0U, "bw", uniform_integer(1, 8)),
        dataset_attribute(1U, "status"),
        dataset_attribute(2U, "ltc")};

    input.physical_node_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::extrema,
        AttributeKind::status};
    input.physical_link_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::extrema};
    input.virtual_node_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::status,
        AttributeKind::position,
        AttributeKind::resource};
    input.virtual_link_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::status,
        AttributeKind::latency};
    input.extracted_attribute_kinds.included = {
        true, true, false, false, true};
    input.seed = DatasetScalar{std::int64_t{7 + value}};
    return input;
}

Config make_run_config()
{
    return Config(YAML::Load(R"yaml(
root_dir: oracle-runs
fixed_run_id: config-run-17
experiment:
  save_root_dir: ${root_dir}
  run_id: ${fixed_run_id}
solver:
  solver_name: typed-solver
)yaml"));
}

void hash_byte(std::uint64_t& checksum, std::uint8_t value) noexcept
{
    checksum ^= value;
    checksum *= fnv_prime;
}

void hash_u64(std::uint64_t& checksum, std::uint64_t value) noexcept
{
    for (unsigned int shift = 0U; shift < 64U; shift += 8U)
    {
        hash_byte(
            checksum,
            static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void hash_text(std::uint64_t& checksum, std::string_view value) noexcept
{
    hash_u64(checksum, static_cast<std::uint64_t>(value.size()));
    for (const char raw : value)
    {
        hash_byte(checksum, static_cast<std::uint8_t>(raw));
    }
}

std::uint64_t summary_checksum(const SimulationConfigSummary& summary)
{
    std::uint64_t checksum = fnv_offset;
    hash_text(checksum, summary.p_net_dataset_dir.generic_string());
    hash_text(checksum, summary.v_nets_dataset_dir.generic_string());
    hash_u64(
        checksum,
        static_cast<std::uint64_t>(summary.p_net_setting_num_nodes));
    hash_u64(checksum, summary.p_net_setting_num_node_attrs);
    hash_u64(checksum, summary.p_net_setting_num_link_attrs);
    hash_u64(checksum, summary.p_net_setting_num_node_resource_attrs);
    hash_u64(checksum, summary.p_net_setting_num_link_resource_attrs);
    hash_u64(checksum, summary.p_net_setting_num_node_extrema_attrs);
    hash_u64(checksum, summary.p_net_setting_num_link_extrema_attrs);
    hash_u64(checksum, summary.v_sim_setting_num_node_attrs);
    hash_u64(checksum, summary.v_sim_setting_num_link_attrs);
    hash_u64(checksum, summary.v_sim_setting_num_node_resource_attrs);
    hash_u64(checksum, summary.v_sim_setting_num_link_resource_attrs);
    hash_u64(checksum, summary.v_sim_setting_num_node_non_status_attrs);
    hash_u64(checksum, summary.v_sim_setting_num_link_non_status_attrs);
    hash_u64(
        checksum,
        summary.feature_constructor.num_extracted_p_node_attrs);
    hash_u64(
        checksum,
        summary.feature_constructor.num_extracted_p_link_attrs);
    hash_u64(
        checksum,
        summary.feature_constructor.num_extracted_v_node_attrs);
    hash_u64(
        checksum,
        summary.feature_constructor.num_extracted_v_link_attrs);
    hash_u64(
        checksum,
        static_cast<std::uint64_t>(
            summary.feature_constructor.p_num_nodes));
    return checksum;
}

std::uint64_t summaries_checksum(
    const std::vector<SimulationConfigSummary>& summaries)
{
    std::uint64_t checksum = fnv_offset;
    hash_u64(checksum, static_cast<std::uint64_t>(summaries.size()));
    for (const SimulationConfigSummary& summary : summaries)
    {
        hash_u64(checksum, summary_checksum(summary));
    }
    return checksum;
}

bool summaries_equal(
    const SimulationConfigSummary& left,
    const SimulationConfigSummary& right) noexcept
{
    return
        left.p_net_dataset_dir == right.p_net_dataset_dir &&
        left.v_nets_dataset_dir == right.v_nets_dataset_dir &&
        left.p_net_setting_num_nodes == right.p_net_setting_num_nodes &&
        left.p_net_setting_num_node_attrs ==
            right.p_net_setting_num_node_attrs &&
        left.p_net_setting_num_link_attrs ==
            right.p_net_setting_num_link_attrs &&
        left.p_net_setting_num_node_resource_attrs ==
            right.p_net_setting_num_node_resource_attrs &&
        left.p_net_setting_num_link_resource_attrs ==
            right.p_net_setting_num_link_resource_attrs &&
        left.p_net_setting_num_node_extrema_attrs ==
            right.p_net_setting_num_node_extrema_attrs &&
        left.p_net_setting_num_link_extrema_attrs ==
            right.p_net_setting_num_link_extrema_attrs &&
        left.v_sim_setting_num_node_attrs ==
            right.v_sim_setting_num_node_attrs &&
        left.v_sim_setting_num_link_attrs ==
            right.v_sim_setting_num_link_attrs &&
        left.v_sim_setting_num_node_resource_attrs ==
            right.v_sim_setting_num_node_resource_attrs &&
        left.v_sim_setting_num_link_resource_attrs ==
            right.v_sim_setting_num_link_resource_attrs &&
        left.v_sim_setting_num_node_non_status_attrs ==
            right.v_sim_setting_num_node_non_status_attrs &&
        left.v_sim_setting_num_link_non_status_attrs ==
            right.v_sim_setting_num_link_non_status_attrs &&
        left.feature_constructor.num_extracted_p_node_attrs ==
            right.feature_constructor.num_extracted_p_node_attrs &&
        left.feature_constructor.num_extracted_p_link_attrs ==
            right.feature_constructor.num_extracted_p_link_attrs &&
        left.feature_constructor.num_extracted_v_node_attrs ==
            right.feature_constructor.num_extracted_v_node_attrs &&
        left.feature_constructor.num_extracted_v_link_attrs ==
            right.feature_constructor.num_extracted_v_link_attrs &&
        left.feature_constructor.p_num_nodes ==
            right.feature_constructor.p_num_nodes;
}

void expect_same_order(
    const std::vector<SimulationConfigSummary>& actual,
    const std::vector<SimulationConfigSummary>& expected)
{
    expect(actual.size() == expected.size(), "batch result size mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index)
    {
        expect(
            summaries_equal(actual[index], expected[index]),
            "batch result order or content mismatch");
    }
}

void emit_json_string(std::ostream& output, std::string_view value)
{
    static constexpr char digits[] = "0123456789abcdef";
    output.put('"');
    for (const char raw : value)
    {
        const auto byte = static_cast<unsigned char>(raw);
        switch (byte)
        {
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
            if (byte < 0x20U)
            {
                output << "\\u00"
                       << digits[(byte >> 4U) & 0x0fU]
                       << digits[byte & 0x0fU];
            }
            else
            {
                output.put(raw);
            }
            break;
        }
    }
    output.put('"');
}

void emit_summary_json(
    std::ostream& output,
    std::size_t variant,
    const SimulationConfigSummary& summary)
{
    output << "{\"variant\":" << variant << ",\"p_net_dataset_dir\":";
    emit_json_string(output, summary.p_net_dataset_dir.generic_string());
    output << ",\"v_nets_dataset_dir\":";
    emit_json_string(output, summary.v_nets_dataset_dir.generic_string());
    output
        << ",\"p_net_setting_num_nodes\":"
        << summary.p_net_setting_num_nodes
        << ",\"p_net_setting_num_node_attrs\":"
        << summary.p_net_setting_num_node_attrs
        << ",\"p_net_setting_num_link_attrs\":"
        << summary.p_net_setting_num_link_attrs
        << ",\"p_net_setting_num_node_resource_attrs\":"
        << summary.p_net_setting_num_node_resource_attrs
        << ",\"p_net_setting_num_link_resource_attrs\":"
        << summary.p_net_setting_num_link_resource_attrs
        << ",\"p_net_setting_num_node_extrema_attrs\":"
        << summary.p_net_setting_num_node_extrema_attrs
        << ",\"p_net_setting_num_link_extrema_attrs\":"
        << summary.p_net_setting_num_link_extrema_attrs
        << ",\"v_sim_setting_num_node_attrs\":"
        << summary.v_sim_setting_num_node_attrs
        << ",\"v_sim_setting_num_link_attrs\":"
        << summary.v_sim_setting_num_link_attrs
        << ",\"v_sim_setting_num_node_resource_attrs\":"
        << summary.v_sim_setting_num_node_resource_attrs
        << ",\"v_sim_setting_num_link_resource_attrs\":"
        << summary.v_sim_setting_num_link_resource_attrs
        << ",\"v_sim_setting_num_node_non_status_attrs\":"
        << summary.v_sim_setting_num_node_non_status_attrs
        << ",\"v_sim_setting_num_link_non_status_attrs\":"
        << summary.v_sim_setting_num_link_non_status_attrs
        << ",\"feature_constructor\":{"
        << "\"num_extracted_p_node_attrs\":"
        << summary.feature_constructor.num_extracted_p_node_attrs
        << ",\"num_extracted_p_link_attrs\":"
        << summary.feature_constructor.num_extracted_p_link_attrs
        << ",\"num_extracted_v_node_attrs\":"
        << summary.feature_constructor.num_extracted_v_node_attrs
        << ",\"num_extracted_v_link_attrs\":"
        << summary.feature_constructor.num_extracted_v_link_attrs
        << ",\"p_num_nodes\":"
        << summary.feature_constructor.p_num_nodes
        << "},\"checksum\":" << summary_checksum(summary) << '}';
}

void emit_run_directory_json(
    std::ostream& output,
    const RunDirectoryInput& input)
{
    output << "{\"save_root_dir\":";
    emit_json_string(output, input.save_root_dir.generic_string());
    output << ",\"solver_name\":";
    emit_json_string(output, input.solver_name);
    output << ",\"run_id\":";
    emit_json_string(output, input.run_id);
    output << ",\"path\":";
    emit_json_string(output, utils::get_run_id_dir(input).generic_string());
    output.put('}');
}

void run_differential()
{
    PyRandom random(42U);
    const RunIdTimestamp timestamp{2024, 2, 29, 3, 4, 5};
    const std::string run_id =
        utils::generate_run_id(timestamp, "worker-a", random);
    const std::int64_t continuation = random.randint(0, 9999);

    std::vector<SimulationConfigInput> inputs;
    inputs.reserve(3U);
    for (std::size_t variant = 0U; variant < 3U; ++variant)
    {
        inputs.push_back(make_simulation_input(variant));
    }

    std::vector<SimulationConfigSummary> direct;
    direct.reserve(inputs.size());
    for (const SimulationConfigInput& input : inputs)
    {
        direct.push_back(utils::derive_simulation_config(input));
    }

    std::vector<BatchObservation> batches;
    batches.reserve(4U);
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        const std::vector<SimulationConfigSummary> summaries =
            utils::derive_simulation_configs_batch(inputs, workers);
        expect_same_order(summaries, direct);
        BatchObservation observation;
        observation.workers = workers;
        observation.checksum = summaries_checksum(summaries);
        observation.order.reserve(summaries.size());
        for (const SimulationConfigSummary& summary : summaries)
        {
            observation.order.push_back(summary_checksum(summary));
        }
        batches.push_back(std::move(observation));
    }

    const RunDirectoryInput direct_directory{
        fs::path("direct-root"), "direct-solver", "direct-run"};
    const Config config = make_run_config();
    const RunDirectoryInput config_directory =
        utils::run_directory_input_from_config(config);

    const Config resolved_config(YAML::Load(R"yaml(
copy_value: 17
resolved_copy: ${copy_value}
resolved_text: value-${copy_value}
)yaml"));
    const YAML::Node resolved =
        utils::resolve_config_to_node(resolved_config);
    YAML::Node raw_mapping = YAML::Load("{value: 3}");
    YAML::Node raw_result = utils::resolve_config_to_node(raw_mapping);
    raw_result["value"] = 8;
    bool invalid_mapping_rejected = false;
    try
    {
        static_cast<void>(
            utils::resolve_config_to_node(YAML::Load("[1, 2]")));
    }
    catch (const utils::UtilsConfigException& error)
    {
        invalid_mapping_rejected =
            error.code() ==
                utils::UtilsConfigErrorCode::invalid_mapping_root &&
            error.operation() ==
                utils::UtilsConfigOperation::resolve_config;
    }
    expect(invalid_mapping_rejected, "invalid mapping error drift");

    std::cout << "{\"version\":1,\"mode\":\"differential\",\"run_id\":{";
    std::cout << "\"value\":";
    emit_json_string(std::cout, run_id);
    std::cout << ",\"next_randint_0_9999\":" << continuation << "},";

    std::cout << "\"summaries\":[";
    for (std::size_t index = 0U; index < direct.size(); ++index)
    {
        if (index != 0U)
        {
            std::cout.put(',');
        }
        emit_summary_json(std::cout, index, direct[index]);
    }
    std::cout << "],\"batch\":[";
    for (std::size_t index = 0U; index < batches.size(); ++index)
    {
        if (index != 0U)
        {
            std::cout.put(',');
        }
        const BatchObservation& batch = batches[index];
        std::cout << "{\"workers\":" << batch.workers
                  << ",\"checksum\":" << batch.checksum
                  << ",\"order\":[";
        for (std::size_t item = 0U; item < batch.order.size(); ++item)
        {
            if (item != 0U)
            {
                std::cout.put(',');
            }
            std::cout << batch.order[item];
        }
        std::cout << "]}";
    }
    std::cout << "],\"run_directory\":{\"direct\":";
    emit_run_directory_json(std::cout, direct_directory);
    std::cout << ",\"config\":";
    emit_run_directory_json(std::cout, config_directory);
    std::cout << "},\"config_resolution\":{\"resolved_copy\":"
              << resolved["resolved_copy"].as<int>()
              << ",\"resolved_text\":";
    emit_json_string(
        std::cout, resolved["resolved_text"].as<std::string>());
    std::cout << ",\"raw_alias_value\":"
              << raw_mapping["value"].as<int>()
              << ",\"invalid_mapping_rejected\":true}"
              << ",\"status\":\"PASS\"}\n";
}

std::size_t parse_size(const char* raw, std::string_view name)
{
    const std::string text(raw);
    if (text.empty() || text.front() == '-')
    {
        throw std::invalid_argument(
            "invalid " + std::string(name) + " argument");
    }
    std::size_t consumed = 0U;
    unsigned long long parsed = 0U;
    try
    {
        parsed = std::stoull(text, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(
            "invalid " + std::string(name) + " argument");
    }
    if (consumed != text.size() ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument(
            "invalid " + std::string(name) + " argument");
    }
    return static_cast<std::size_t>(parsed);
}

void run_benchmark(
    std::size_t count,
    std::size_t workers,
    std::size_t repetitions)
{
    if (count == 0U || repetitions == 0U)
    {
        throw std::invalid_argument(
            "benchmark count and repetitions must be positive");
    }
    if (count > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max() - 64))
    {
        throw std::invalid_argument("benchmark count exceeds int64 range");
    }

    std::vector<SimulationConfigInput> inputs;
    inputs.reserve(count);
    for (std::size_t index = 0U; index < count; ++index)
    {
        inputs.push_back(make_simulation_input(index));
    }

    std::vector<SimulationConfigSummary> outputs =
        utils::derive_simulation_configs_batch(inputs, workers);
    const std::uint64_t expected_checksum = summaries_checksum(outputs);

    using Clock = std::chrono::steady_clock;
    std::vector<std::uint64_t> elapsed_ns;
    elapsed_ns.reserve(repetitions);
    for (std::size_t repetition = 0U;
         repetition < repetitions;
         ++repetition)
    {
        const auto start = Clock::now();
        outputs = utils::derive_simulation_configs_batch(inputs, workers);
        const auto finish = Clock::now();
        elapsed_ns.push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                finish - start).count()));
        expect(
            summaries_checksum(outputs) == expected_checksum,
            "benchmark output checksum changed between repetitions");
    }

    std::cout << "{\"version\":1,\"mode\":\"benchmark\",\"count\":"
              << count << ",\"workers\":" << workers
              << ",\"repetitions\":" << repetitions
              << ",\"warmups\":1,\"elapsed_ns\":[";
    for (std::size_t index = 0U; index < elapsed_ns.size(); ++index)
    {
        if (index != 0U)
        {
            std::cout.put(',');
        }
        std::cout << elapsed_ns[index];
    }
    std::cout << "],\"checksum\":" << expected_checksum
              << ",\"entry_count\":" << outputs.size()
              << ",\"status\":\"PASS\"}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1 ||
            (argc == 2 && std::string_view(argv[1]) == "differential"))
        {
            run_differential();
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "benchmark")
        {
            run_benchmark(
                parse_size(argv[2], "count"),
                parse_size(argv[3], "workers"),
                parse_size(argv[4], "repetitions"));
            return 0;
        }
        throw std::invalid_argument(
            "usage: utils_config_harness [differential] | "
            "benchmark <count> <workers> <repetitions>");
    }
    catch (const std::exception& error)
    {
        std::cerr << "utils config harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
