#pragma once

#include "dataset.h"

#include "attribute/attribute_method.h"
#include "config/config.h"
#include "py_random.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace virne::utils
{

enum class UtilsConfigErrorCode : std::uint8_t
{
    invalid_mapping_root,
    attribute_kind_count_mismatch,
    invalid_attribute_kind,
    local_time_failure,
    hostname_failure,
};

enum class UtilsConfigOperation : std::uint8_t
{
    generate_run_id,
    resolve_config,
    derive_simulation_config,
};

class UtilsConfigException : public std::runtime_error
{
public:
    UtilsConfigException(
        UtilsConfigErrorCode code,
        UtilsConfigOperation operation,
        std::string message);

    UtilsConfigErrorCode code() const noexcept;
    UtilsConfigOperation operation() const noexcept;

private:
    UtilsConfigErrorCode code_;
    UtilsConfigOperation operation_;
};

struct RunIdTimestamp
{
    int year = 1970;
    int month = 1;
    int day = 1;
    int hour = 0;
    int minute = 0;
    int second = 0;
};

struct RunDirectoryInput
{
    std::filesystem::path save_root_dir;
    std::string solver_name;
    std::string run_id;
};

struct SimulationAttributeKinds
{
    std::vector<virne::network::attribute::AttributeKind> kinds;
};

struct ExtractedAttributeKinds
{
    // AttributeKind is the already-resolved compact ID. The five fixed slots
    // replace Python membership/string lookup in every counting loop.
    std::array<bool, 5> included{};
};

struct SimulationConfigInput
{
    PhysicalDatasetSetting physical_dataset;
    VirtualDatasetSetting virtual_dataset;
    SimulationAttributeKinds physical_node_attributes;
    SimulationAttributeKinds physical_link_attributes;
    SimulationAttributeKinds virtual_node_attributes;
    SimulationAttributeKinds virtual_link_attributes;
    ExtractedAttributeKinds extracted_attribute_kinds;
    std::optional<DatasetScalar> seed;
};

struct FeatureConstructorSummary
{
    std::size_t num_extracted_p_node_attrs = 0;
    std::size_t num_extracted_p_link_attrs = 0;
    std::size_t num_extracted_v_node_attrs = 0;
    std::size_t num_extracted_v_link_attrs = 0;
    std::int64_t p_num_nodes = 0;
};

struct SimulationConfigSummary
{
    std::filesystem::path p_net_dataset_dir;
    std::filesystem::path v_nets_dataset_dir;

    std::int64_t p_net_setting_num_nodes = 0;
    std::size_t p_net_setting_num_node_attrs = 0;
    std::size_t p_net_setting_num_link_attrs = 0;
    std::size_t p_net_setting_num_node_resource_attrs = 0;
    std::size_t p_net_setting_num_link_resource_attrs = 0;
    std::size_t p_net_setting_num_node_extrema_attrs = 0;
    std::size_t p_net_setting_num_link_extrema_attrs = 0;

    std::size_t v_sim_setting_num_node_attrs = 0;
    std::size_t v_sim_setting_num_link_attrs = 0;
    std::size_t v_sim_setting_num_node_resource_attrs = 0;
    std::size_t v_sim_setting_num_link_resource_attrs = 0;
    std::size_t v_sim_setting_num_node_non_status_attrs = 0;
    std::size_t v_sim_setting_num_link_non_status_attrs = 0;

    FeatureConstructorSummary feature_constructor;
};

std::string generate_run_id(
    const RunIdTimestamp& timestamp,
    std::string_view hostname,
    PyRandom& rng);

std::string generate_run_id(PyRandom& rng);

YAML::Node resolve_config_to_node(const Config& config);
YAML::Node resolve_config_to_node(const YAML::Node& config);

SimulationConfigSummary derive_simulation_config(
    const SimulationConfigInput& input);

std::vector<SimulationConfigSummary> derive_simulation_configs_batch(
    const std::vector<SimulationConfigInput>& inputs,
    std::size_t workers = 1);

std::filesystem::path get_run_id_dir(const RunDirectoryInput& input);

RunDirectoryInput run_directory_input_from_config(const Config& config);

} // namespace virne::utils
