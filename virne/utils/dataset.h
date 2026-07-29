#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

class NumpyRandomState;

namespace virne::utils
{

enum class DistributionKind : std::uint8_t
{
    none,
    uniform,
    normal,
    exponential,
    poisson,
    customized,
};

enum class DatasetValueKind : std::uint8_t
{
    integer,
    floating,
    boolean,
};

// Kept local to the standard-library-only dataset leaf.  A future setting
// boundary converts the canonical network TopologyType once into this enum.
enum class DatasetTopologyKind : std::uint8_t
{
    path,
    star,
    grid_2d,
    waxman,
    random,
};

enum class DatasetErrorCode : std::uint8_t
{
    invalid_distribution,
    invalid_value_kind,
    invalid_topology,
    missing_parameter,
    invalid_parameter,
    uniform_boolean_uninitialized,
    unsupported_parameter_distribution,
    rng_backend_failure,
    xml_parse_failure,
    xml_schema_failure,
    unknown_endpoint,
    graph_materialization_failure,
    gml_write_failure,
};

enum class DatasetOperation : std::uint8_t
{
    resolve_distribution,
    resolve_topology,
    generate_values,
    cast_values,
    format_parameters,
    format_file_name,
    build_physical_path,
    build_virtual_path,
    parse_xml,
    materialize_graph,
    write_gml,
};

using DatasetAttrId = std::uint32_t;
using DatasetScalar = std::variant<
    std::monostate,
    std::int64_t,
    double,
    bool,
    std::string>;

struct DistributionSpec
{
    DistributionKind kind = DistributionKind::none;
    std::optional<DatasetScalar> low;
    std::optional<DatasetScalar> high;
    std::optional<DatasetScalar> loc;
    std::optional<DatasetScalar> scale;
    std::optional<DatasetScalar> lambda;
    std::optional<DatasetScalar> minimum;
    std::optional<DatasetScalar> maximum;
    bool reciprocal = false;
};

struct DatasetAttributeSpec
{
    DatasetAttrId id = 0;
    std::string name;
    DistributionSpec distribution;
};

struct DatasetFileNameConfig
{
    std::string solver_name;
};

struct OrderedFileNameItem
{
    std::string key;
    DatasetScalar value;
};

struct DatasetFileNameRequest
{
    DatasetFileNameConfig config;
    std::int64_t epoch_id = 0;
    std::vector<OrderedFileNameItem> ordered_items;
};

struct PhysicalTopologyDatasetSpec
{
    std::optional<std::filesystem::path> file_path;
    std::int64_t num_nodes = 0;
    DatasetTopologyKind topology_type = DatasetTopologyKind::waxman;
    DatasetScalar wm_alpha = 0.5;
    DatasetScalar wm_beta = 0.2;
};

struct PhysicalDatasetSetting
{
    std::filesystem::path save_dir;
    PhysicalTopologyDatasetSpec topology;
    std::vector<DatasetAttributeSpec> node_attributes;
    std::vector<DatasetAttributeSpec> link_attributes;
};

struct VirtualDatasetSetting
{
    std::filesystem::path save_dir;
    std::int64_t num_virtual_networks = 0;
    std::int64_t size_low = 0;
    std::int64_t size_high = 0;
    DatasetTopologyKind topology_type = DatasetTopologyKind::random;
    DistributionSpec lifetime;
    DatasetScalar arrival_lambda = 0.0;
    std::vector<DatasetAttributeSpec> node_attributes;
    std::vector<DatasetAttributeSpec> link_attributes;
};

struct PhysicalDatasetPathRequest
{
    PhysicalDatasetSetting setting;
    std::optional<DatasetScalar> seed;
};

struct VirtualDatasetPathRequest
{
    VirtualDatasetSetting setting;
    std::optional<DatasetScalar> seed;
};

struct DistributionRequest
{
    std::size_t count = 0;
    DatasetValueKind value_kind = DatasetValueKind::floating;
    DistributionSpec distribution;
};

struct GeneratedData
{
    DatasetValueKind value_kind = DatasetValueKind::floating;
    std::variant<
        std::vector<std::int64_t>,
        std::vector<double>,
        std::vector<std::uint8_t>> values;
};

class DatasetException : public std::runtime_error
{
public:
    DatasetException(
        DatasetErrorCode code,
        DatasetOperation operation,
        std::string message,
        std::optional<std::size_t> input_index = std::nullopt,
        std::filesystem::path path = {});

    DatasetErrorCode code() const noexcept;
    DatasetOperation operation() const noexcept;
    const std::optional<std::size_t>& input_index() const noexcept;
    const std::filesystem::path& path() const noexcept;

private:
    DatasetErrorCode code_;
    DatasetOperation operation_;
    std::optional<std::size_t> input_index_;
    std::filesystem::path path_;
};

DistributionKind distribution_kind_from_string(std::string_view value);
DatasetValueKind dataset_value_kind_from_string(std::string_view value);
DatasetTopologyKind dataset_topology_kind_from_string(std::string_view value);
std::string_view dataset_topology_kind_name(DatasetTopologyKind value);

std::string format_dataset_scalar(const DatasetScalar& value);
std::vector<DatasetScalar> get_distribution_parameters(
    const DistributionSpec& distribution);
std::string get_parameters_string(
    const std::vector<DatasetScalar>& parameters);
std::optional<double> get_distribution_average(
    const DistributionSpec& distribution,
    DatasetValueKind value_kind) noexcept;

std::string generate_file_name(
    const DatasetFileNameConfig& config,
    std::int64_t epoch_id,
    const std::vector<OrderedFileNameItem>& ordered_items);

std::filesystem::path get_p_net_dataset_dir_from_setting(
    const PhysicalDatasetSetting& setting,
    const std::optional<DatasetScalar>& seed = std::nullopt);
std::filesystem::path get_v_nets_dataset_dir_from_setting(
    const VirtualDatasetSetting& setting,
    const std::optional<DatasetScalar>& seed = std::nullopt);

std::vector<std::string> generate_file_names_batch(
    const std::vector<DatasetFileNameRequest>& requests,
    std::size_t workers = 0);
std::vector<std::filesystem::path> get_p_net_dataset_dirs_batch(
    const std::vector<PhysicalDatasetPathRequest>& requests,
    std::size_t workers = 0);
std::vector<std::filesystem::path> get_v_nets_dataset_dirs_batch(
    const std::vector<VirtualDatasetPathRequest>& requests,
    std::size_t workers = 0);

GeneratedData generate_data_with_distribution(
    const DistributionRequest& request,
    NumpyRandomState& rng,
    std::size_t cast_workers = 0);

} // namespace virne::utils
