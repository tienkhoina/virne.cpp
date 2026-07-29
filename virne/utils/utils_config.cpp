#include "utils_config.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <ctime>
#include <exception>
#include <limits>
#include <system_error>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace virne::utils
{
namespace
{

using AttributeKind = virne::network::attribute::AttributeKind;

std::size_t attribute_kind_index(AttributeKind kind)
{
    switch (kind)
    {
    case AttributeKind::resource:
        return 0;
    case AttributeKind::extrema:
        return 1;
    case AttributeKind::status:
        return 2;
    case AttributeKind::position:
        return 3;
    case AttributeKind::latency:
        return 4;
    }
    throw UtilsConfigException(
        UtilsConfigErrorCode::invalid_attribute_kind,
        UtilsConfigOperation::derive_simulation_config,
        "invalid attribute kind");
}

void validate_kind_count(
    const SimulationAttributeKinds& attributes,
    std::size_t expected)
{
    if (attributes.kinds.size() != expected)
    {
        throw UtilsConfigException(
            UtilsConfigErrorCode::attribute_kind_count_mismatch,
            UtilsConfigOperation::derive_simulation_config,
            "attribute kind count does not match dataset attributes");
    }
}

void validate_attribute_kind_counts(const SimulationConfigInput& input)
{
    validate_kind_count(
        input.physical_node_attributes,
        input.physical_dataset.node_attributes.size());
    validate_kind_count(
        input.physical_link_attributes,
        input.physical_dataset.link_attributes.size());
    validate_kind_count(
        input.virtual_node_attributes,
        input.virtual_dataset.node_attributes.size());
    validate_kind_count(
        input.virtual_link_attributes,
        input.virtual_dataset.link_attributes.size());

}

struct AttributeCounts
{
    std::size_t resource = 0;
    std::size_t extrema = 0;
    std::size_t non_status = 0;
    std::size_t extracted = 0;
};

AttributeCounts count_attributes(
    const SimulationAttributeKinds& attributes,
    const ExtractedAttributeKinds& extracted)
{
    AttributeCounts result;
    for (const AttributeKind kind : attributes.kinds)
    {
        const std::size_t id = attribute_kind_index(kind);
        result.resource += kind == AttributeKind::resource;
        result.extrema += kind == AttributeKind::extrema;
        result.non_status += kind != AttributeKind::status;
        result.extracted += extracted.included[id];
    }
    return result;
}

SimulationConfigSummary derive_validated_simulation_config(
    const SimulationConfigInput& input)
{
    SimulationConfigSummary result;

    // Preserve the Python helper's observable operation order: physical path,
    // virtual path, then summary fields.
    result.p_net_dataset_dir = get_p_net_dataset_dir_from_setting(
        input.physical_dataset, input.seed);
    result.v_nets_dataset_dir = get_v_nets_dataset_dir_from_setting(
        input.virtual_dataset, input.seed);

    validate_attribute_kind_counts(input);
    const AttributeCounts physical_node = count_attributes(
        input.physical_node_attributes, input.extracted_attribute_kinds);
    const AttributeCounts physical_link = count_attributes(
        input.physical_link_attributes, input.extracted_attribute_kinds);
    const AttributeCounts virtual_node = count_attributes(
        input.virtual_node_attributes, input.extracted_attribute_kinds);
    const AttributeCounts virtual_link = count_attributes(
        input.virtual_link_attributes, input.extracted_attribute_kinds);

    result.p_net_setting_num_nodes = input.physical_dataset.topology.num_nodes;
    result.p_net_setting_num_node_attrs =
        input.physical_node_attributes.kinds.size();
    result.p_net_setting_num_link_attrs =
        input.physical_link_attributes.kinds.size();
    result.p_net_setting_num_node_resource_attrs = physical_node.resource;
    result.p_net_setting_num_link_resource_attrs = physical_link.resource;
    result.p_net_setting_num_node_extrema_attrs = physical_node.extrema;
    result.p_net_setting_num_link_extrema_attrs = physical_link.extrema;

    result.v_sim_setting_num_node_attrs =
        input.virtual_node_attributes.kinds.size();
    result.v_sim_setting_num_link_attrs =
        input.virtual_link_attributes.kinds.size();
    result.v_sim_setting_num_node_resource_attrs = virtual_node.resource;
    result.v_sim_setting_num_link_resource_attrs = virtual_link.resource;
    result.v_sim_setting_num_node_non_status_attrs = virtual_node.non_status;
    result.v_sim_setting_num_link_non_status_attrs = virtual_link.non_status;

    result.feature_constructor.num_extracted_p_node_attrs =
        physical_node.extracted;
    result.feature_constructor.num_extracted_p_link_attrs =
        physical_link.extracted;
    result.feature_constructor.num_extracted_v_node_attrs =
        virtual_node.extracted;
    result.feature_constructor.num_extracted_v_link_attrs =
        virtual_link.extracted;
    result.feature_constructor.p_num_nodes = result.p_net_setting_num_nodes;
    return result;
}

template <typename Function>
void parallel_indexed(
    std::size_t count,
    std::size_t requested_workers,
    Function&& function)
{
    const std::size_t worker_count = requested_workers <= 1
        ? 1
        : std::min(requested_workers, count);
    if (worker_count <= 1)
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            function(index);
        }
        return;
    }

    struct Failure
    {
        std::size_t index = std::numeric_limits<std::size_t>::max();
        std::exception_ptr error;
    };

    std::vector<Failure> failures(worker_count);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    try
    {
        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            threads.emplace_back(
                [&, worker]
                {
                    const std::size_t begin = count * worker / worker_count;
                    const std::size_t end =
                        count * (worker + 1) / worker_count;
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        try
                        {
                            function(index);
                        }
                        catch (...)
                        {
                            failures[worker] =
                                {index, std::current_exception()};
                            break;
                        }
                    }
                });
        }
    }
    catch (...)
    {
        for (std::thread& thread : threads)
        {
            thread.join();
        }
        throw;
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }

    const Failure* first = nullptr;
    for (const Failure& failure : failures)
    {
        if (failure.error && (first == nullptr || failure.index < first->index))
        {
            first = &failure;
        }
    }
    if (first != nullptr)
    {
        std::rethrow_exception(first->error);
    }
}

YAML::Node require_mapping(YAML::Node node)
{
    if (!node.IsMap())
    {
        throw UtilsConfigException(
            UtilsConfigErrorCode::invalid_mapping_root,
            UtilsConfigOperation::resolve_config,
            "config did not resolve to a mapping");
    }
    return node;
}

RunIdTimestamp local_timestamp()
{
    const std::time_t now = std::time(nullptr);
    if (now == static_cast<std::time_t>(-1))
    {
        throw UtilsConfigException(
            UtilsConfigErrorCode::local_time_failure,
            UtilsConfigOperation::generate_run_id,
            "system clock conversion failed");
    }

    std::tm local{};
#if defined(_WIN32)
    if (localtime_s(&local, &now) != 0)
#else
    if (localtime_r(&now, &local) == nullptr)
#endif
    {
        throw UtilsConfigException(
            UtilsConfigErrorCode::local_time_failure,
            UtilsConfigOperation::generate_run_id,
            "local time conversion failed");
    }
    return {
        local.tm_year + 1900,
        local.tm_mon + 1,
        local.tm_mday,
        local.tm_hour,
        local.tm_min,
        local.tm_sec};
}

void append_zero_padded(
    std::string& output,
    std::int64_t value,
    std::size_t minimum_width)
{
    std::array<char, 32> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value);
    if (converted.ec != std::errc{})
    {
        throw std::runtime_error("run ID integer formatting failed");
    }
    const std::size_t length = static_cast<std::size_t>(
        converted.ptr - buffer.data());
    if (value >= 0 && length < minimum_width)
    {
        output.append(minimum_width - length, '0');
    }
    output.append(buffer.data(), length);
}

std::string format_timestamp(const RunIdTimestamp& timestamp)
{
    std::string output;
    output.reserve(15);
    append_zero_padded(output, timestamp.year, 4);
    append_zero_padded(output, timestamp.month, 2);
    append_zero_padded(output, timestamp.day, 2);
    output.push_back('T');
    append_zero_padded(output, timestamp.hour, 2);
    append_zero_padded(output, timestamp.minute, 2);
    append_zero_padded(output, timestamp.second, 2);
    return output;
}

std::string assemble_run_id(
    std::string_view formatted_timestamp,
    std::string_view hostname,
    PyRandom& rng)
{
    std::string output;
    output.reserve(hostname.size() + formatted_timestamp.size() + 7);
    output.append(hostname);
    output.push_back('-');
    output.append(formatted_timestamp);
    output.push_back('-');
    append_zero_padded(output, rng.randint(0, 9999), 4);
    return output;
}

std::string local_hostname()
{
#if defined(_WIN32)
    std::array<char, MAX_COMPUTERNAME_LENGTH + 1> buffer{};
    DWORD length = static_cast<DWORD>(buffer.size());
    if (!GetComputerNameA(buffer.data(), &length))
    {
        throw UtilsConfigException(
            UtilsConfigErrorCode::hostname_failure,
            UtilsConfigOperation::generate_run_id,
            "hostname lookup failed");
    }
    return std::string(buffer.data(), length);
#else
    std::array<char, 256> buffer{};
    if (gethostname(buffer.data(), buffer.size()) != 0)
    {
        throw UtilsConfigException(
            UtilsConfigErrorCode::hostname_failure,
            UtilsConfigOperation::generate_run_id,
            "hostname lookup failed");
    }
    buffer.back() = '\0';
    return std::string(buffer.data());
#endif
}

} // namespace

UtilsConfigException::UtilsConfigException(
    UtilsConfigErrorCode code,
    UtilsConfigOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation)
{
}

UtilsConfigErrorCode UtilsConfigException::code() const noexcept
{
    return code_;
}

UtilsConfigOperation UtilsConfigException::operation() const noexcept
{
    return operation_;
}

std::string generate_run_id(
    const RunIdTimestamp& timestamp,
    std::string_view hostname,
    PyRandom& rng)
{
    const std::string formatted_timestamp = format_timestamp(timestamp);
    return assemble_run_id(formatted_timestamp, hostname, rng);
}

std::string generate_run_id(PyRandom& rng)
{
    const RunIdTimestamp timestamp = local_timestamp();
    const std::string formatted_timestamp = format_timestamp(timestamp);
    const std::string hostname = local_hostname();
    return assemble_run_id(formatted_timestamp, hostname, rng);
}

YAML::Node resolve_config_to_node(const Config& config)
{
    return require_mapping(config.root());
}

YAML::Node resolve_config_to_node(const YAML::Node& config)
{
    return require_mapping(config);
}

SimulationConfigSummary derive_simulation_config(
    const SimulationConfigInput& input)
{
    return derive_validated_simulation_config(input);
}

std::vector<SimulationConfigSummary> derive_simulation_configs_batch(
    const std::vector<SimulationConfigInput>& inputs,
    std::size_t workers)
{
    std::vector<SimulationConfigSummary> results(inputs.size());
    parallel_indexed(
        inputs.size(),
        workers,
        [&](std::size_t index)
        {
            results[index] = derive_validated_simulation_config(inputs[index]);
        });
    return results;
}

std::filesystem::path get_run_id_dir(const RunDirectoryInput& input)
{
    return input.save_root_dir / input.solver_name / input.run_id;
}

RunDirectoryInput run_directory_input_from_config(const Config& config)
{
    // Fixed path strings exist only at this cold dynamic boundary. Each is
    // resolved once; the returned fixed schema has direct fields only.
    return {
        config.get<std::string>("experiment.save_root_dir"),
        config.get<std::string>("solver.solver_name"),
        config.get<std::string>("experiment.run_id")};
}

} // namespace virne::utils
