#include "dataset.h"

#include <chrono>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using virne::utils::DatasetAttributeSpec;
using virne::utils::DatasetErrorCode;
using virne::utils::DatasetException;
using virne::utils::DatasetFileNameRequest;
using virne::utils::DatasetOperation;
using virne::utils::DatasetScalar;
using virne::utils::DatasetTopologyKind;
using virne::utils::DatasetValueKind;
using virne::utils::DistributionKind;
using virne::utils::DistributionSpec;
using virne::utils::PhysicalDatasetPathRequest;
using virne::utils::PhysicalDatasetSetting;
using virne::utils::VirtualDatasetPathRequest;
using virne::utils::VirtualDatasetSetting;

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

DatasetScalar integer(std::int64_t value)
{
    return DatasetScalar{value};
}

DistributionSpec uniform(std::int64_t low, std::int64_t high)
{
    DistributionSpec result;
    result.kind = DistributionKind::uniform;
    result.low = integer(low);
    result.high = integer(high);
    return result;
}

DatasetAttributeSpec attribute(
    std::uint32_t id,
    std::string name,
    DistributionSpec distribution)
{
    return {id, std::move(name), std::move(distribution)};
}

PhysicalDatasetSetting physical_setting(const fs::path& root)
{
    PhysicalDatasetSetting setting;
    setting.save_dir = root / "p_out";
    setting.topology.num_nodes = 100;
    setting.topology.topology_type = DatasetTopologyKind::waxman;
    setting.topology.wm_alpha = DatasetScalar{0.5};
    setting.topology.wm_beta = DatasetScalar{0.2};
    setting.node_attributes = {
        attribute(1, "cpu", uniform(50, 100)),
        attribute(2, "max_cpu", {})};
    setting.link_attributes = {
        attribute(3, "bw", uniform(50, 100)),
        attribute(4, "max_bw", {})};
    return setting;
}

VirtualDatasetSetting virtual_setting(const fs::path& root)
{
    VirtualDatasetSetting setting;
    setting.save_dir = root / "v_out";
    setting.num_virtual_networks = 1000;
    setting.size_low = 2;
    setting.size_high = 10;
    setting.topology_type = DatasetTopologyKind::random;
    setting.lifetime.kind = DistributionKind::exponential;
    setting.lifetime.scale = integer(500);
    setting.arrival_lambda = DatasetScalar{0.04};
    setting.node_attributes = {attribute(1, "cpu", uniform(0, 20))};
    setting.link_attributes = {attribute(2, "bw", uniform(0, 50))};
    return setting;
}

DatasetFileNameRequest file_name_request(std::size_t index)
{
    return {
        {index % 2 == 0 ? "solver" : "solver-x"},
        static_cast<std::int64_t>(index),
        {{"index", DatasetScalar{static_cast<std::int64_t>(index)}},
         {"even", DatasetScalar{index % 2 == 0}},
         {"raw", DatasetScalar{std::string("x=y/z")}}}};
}

std::string hex_encode(std::string_view value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.resize(value.size() * 2);
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2] = digits[byte >> 4U];
        result[index * 2 + 1] = digits[byte & 0x0FU];
    }
    return result;
}

std::string_view error_code_name(DatasetErrorCode code)
{
    switch (code)
    {
    case DatasetErrorCode::invalid_distribution:
        return "invalid_distribution";
    case DatasetErrorCode::invalid_value_kind:
        return "invalid_value_kind";
    case DatasetErrorCode::invalid_topology:
        return "invalid_topology";
    case DatasetErrorCode::missing_parameter:
        return "missing_parameter";
    case DatasetErrorCode::invalid_parameter:
        return "invalid_parameter";
    case DatasetErrorCode::uniform_boolean_uninitialized:
        return "uniform_boolean_uninitialized";
    case DatasetErrorCode::unsupported_parameter_distribution:
        return "unsupported_parameter_distribution";
    case DatasetErrorCode::rng_backend_failure:
        return "rng_backend_failure";
    case DatasetErrorCode::xml_parse_failure:
        return "xml_parse_failure";
    case DatasetErrorCode::xml_schema_failure:
        return "xml_schema_failure";
    case DatasetErrorCode::unknown_endpoint:
        return "unknown_endpoint";
    case DatasetErrorCode::graph_materialization_failure:
        return "graph_materialization_failure";
    case DatasetErrorCode::gml_write_failure:
        return "gml_write_failure";
    }
    return "invalid_error_enum";
}

std::string_view operation_name(DatasetOperation operation)
{
    switch (operation)
    {
    case DatasetOperation::resolve_distribution:
        return "resolve_distribution";
    case DatasetOperation::resolve_topology:
        return "resolve_topology";
    case DatasetOperation::generate_values:
        return "generate_values";
    case DatasetOperation::cast_values:
        return "cast_values";
    case DatasetOperation::format_parameters:
        return "format_parameters";
    case DatasetOperation::format_file_name:
        return "format_file_name";
    case DatasetOperation::build_physical_path:
        return "build_physical_path";
    case DatasetOperation::build_virtual_path:
        return "build_virtual_path";
    case DatasetOperation::parse_xml:
        return "parse_xml";
    case DatasetOperation::materialize_graph:
        return "materialize_graph";
    case DatasetOperation::write_gml:
        return "write_gml";
    }
    return "invalid_operation_enum";
}

void emit_ok(std::string_view name, std::string_view value)
{
    std::cout << "case=" << name << "|ok|" << hex_encode(value) << '\n';
}

void emit_error(std::string_view name, const DatasetException& error)
{
    std::cout << "case=" << name << "|error|"
              << error_code_name(error.code()) << '|'
              << operation_name(error.operation()) << '\n';
}

template <typename Callable>
void emit_call(std::string_view name, Callable&& callable)
{
    try
    {
        emit_ok(name, callable());
    }
    catch (const DatasetException& error)
    {
        emit_error(name, error);
    }
}

void checksum_byte(std::uint64_t& checksum, unsigned char byte)
{
    checksum ^= static_cast<std::uint64_t>(byte);
    checksum *= kFnvPrime;
}

void checksum_string(std::uint64_t& checksum, std::string_view value)
{
    for (const char character : value)
    {
        checksum_byte(checksum, static_cast<unsigned char>(character));
    }
    checksum_byte(checksum, 0xFFU);
}

std::pair<std::uint64_t, std::size_t> summarize(
    const std::vector<std::string>& values)
{
    std::uint64_t checksum = kFnvOffset;
    std::size_t bytes = 0;
    for (const std::string& value : values)
    {
        checksum_string(checksum, value);
        bytes += value.size();
    }
    return {checksum, bytes};
}

std::pair<std::uint64_t, std::size_t> summarize(
    const std::vector<fs::path>& values)
{
    std::uint64_t checksum = kFnvOffset;
    std::size_t bytes = 0;
    for (const fs::path& path : values)
    {
        const std::string value = path.string();
        checksum_string(checksum, value);
        bytes += value.size();
    }
    return {checksum, bytes};
}

std::string summary_text(const std::pair<std::uint64_t, std::size_t>& summary)
{
    return std::to_string(summary.first) + ":" + std::to_string(summary.second);
}

void run_cases(const fs::path& root)
{
    std::cout << "dataset_core_harness_version=1\n";
    emit_ok("scalar_none", virne::utils::format_dataset_scalar(std::monostate{}));
    emit_ok("scalar_true", virne::utils::format_dataset_scalar(DatasetScalar{true}));
    emit_ok("scalar_false", virne::utils::format_dataset_scalar(DatasetScalar{false}));
    emit_ok("scalar_int_min", virne::utils::format_dataset_scalar(
                                    integer(std::numeric_limits<std::int64_t>::min())));
    emit_ok("scalar_string", virne::utils::format_dataset_scalar(
                                   DatasetScalar{std::string("a-b=c\n")}));
    for (const auto& item : std::vector<std::pair<std::string_view, double>>{
             {"float_one", 1.0},
             {"float_negative_zero", -0.0},
             {"float_million", 1.0e6},
             {"float_e15", 1.0e15},
             {"float_e16", 1.0e16},
             {"float_em4", 1.0e-4},
             {"float_em5", 1.0e-5},
             {"float_inf", std::numeric_limits<double>::infinity()},
             {"float_negative_inf", -std::numeric_limits<double>::infinity()},
             {"float_nan", std::numeric_limits<double>::quiet_NaN()}})
    {
        emit_ok(item.first,
                virne::utils::format_dataset_scalar(DatasetScalar{item.second}));
    }

    const auto parameter_case = [](DistributionSpec distribution)
    {
        return virne::utils::get_parameters_string(
            virne::utils::get_distribution_parameters(distribution));
    };
    emit_call("parameters_none", [&] { return parameter_case({}); });
    DistributionSpec exponential;
    exponential.kind = DistributionKind::exponential;
    exponential.scale = integer(500);
    emit_call("parameters_exponential", [&] { return parameter_case(exponential); });
    DistributionSpec poisson;
    poisson.kind = DistributionKind::poisson;
    poisson.lambda = DatasetScalar{0.04};
    emit_call("parameters_poisson", [&] { return parameter_case(poisson); });
    emit_call("parameters_uniform", [&] { return parameter_case(uniform(-2, 7)); });
    DistributionSpec customized;
    customized.kind = DistributionKind::customized;
    customized.minimum = DatasetScalar{std::string("a")};
    customized.maximum = DatasetScalar{false};
    emit_call("parameters_customized", [&] { return parameter_case(customized); });
    DistributionSpec normal;
    normal.kind = DistributionKind::normal;
    emit_call("parameters_normal_error", [&] { return parameter_case(normal); });
    DistributionSpec missing;
    missing.kind = DistributionKind::uniform;
    missing.low = integer(1);
    emit_call("parameters_missing_error", [&] { return parameter_case(missing); });
    emit_ok(
        "average_stub",
        virne::utils::get_distribution_average(
            normal, DatasetValueKind::floating)
            ? "value"
            : "None");

    emit_ok(
        "filename_empty",
        virne::utils::generate_file_name({"solver"}, 0, {}));
    emit_ok(
        "filename_ordered",
        virne::utils::generate_file_name(
            {"solver"},
            -3,
            {{"alpha", integer(1)},
             {"flag", DatasetScalar{true}},
             {"raw-key", DatasetScalar{std::string("x=y/z")}},
             {"none", DatasetScalar{std::monostate{}}}}));
    emit_ok(
        "filename_special",
        virne::utils::generate_file_name(
            {u8"nghiệm"},
            42,
            {{"line\nkey", DatasetScalar{std::string("x\0y", 3)}},
             {u8"khóa", DatasetScalar{std::string(u8"giá-trị")}}}));

    const PhysicalDatasetSetting physical = physical_setting(root);
    emit_call(
        "physical_default",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(physical)
                .string();
        });
    emit_call(
        "physical_seed_zero",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(
                       physical, integer(0))
                .string();
        });
    emit_call(
        "physical_seed_false",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(
                       physical, DatasetScalar{false})
                .string();
        });
    emit_call(
        "physical_seed_negative",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(
                       physical, integer(-7))
                .string();
        });
    PhysicalDatasetSetting empty_physical = physical;
    empty_physical.node_attributes.clear();
    empty_physical.link_attributes.clear();
    emit_call(
        "physical_empty_attributes",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(empty_physical)
                .string();
        });
    PhysicalDatasetSetting existing = physical;
    existing.topology.file_path = root / "Topo.multi.part.gml";
    emit_call(
        "physical_existing_file",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(existing)
                .string();
        });
    existing.topology.file_path = root / ".hidden.gml";
    emit_call(
        "physical_hidden_file",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(existing)
                .string();
        });
    existing.topology.file_path = root / "missing.gml";
    emit_call(
        "physical_missing_file",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(existing)
                .string();
        });
    existing.topology.file_path = fs::path("None");
    emit_call(
        "physical_none_text_file",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(existing)
                .string();
        });
    existing.topology.file_path = fs::path{};
    emit_call(
        "physical_empty_text_file",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(existing)
                .string();
        });
    PhysicalDatasetSetting path_topology = physical;
    path_topology.topology.num_nodes = 10;
    path_topology.topology.topology_type = DatasetTopologyKind::path;
    path_topology.topology.wm_alpha = integer(1);
    path_topology.topology.wm_beta = integer(2);
    emit_call(
        "physical_path_topology",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(path_topology)
                .string();
        });
    PhysicalDatasetSetting unicode_physical = physical;
    unicode_physical.node_attributes = {
        attribute(9, u8"cpu-động", uniform(1, 2))};
    emit_call(
        "physical_unicode_attribute",
        [&]
        {
            return virne::utils::get_p_net_dataset_dir_from_setting(unicode_physical)
                .string();
        });

    const VirtualDatasetSetting virtual_value = virtual_setting(root);
    emit_call(
        "virtual_default",
        [&]
        {
            return virne::utils::get_v_nets_dataset_dir_from_setting(virtual_value)
                .string();
        });
    emit_call(
        "virtual_seed_false",
        [&]
        {
            return virne::utils::get_v_nets_dataset_dir_from_setting(
                       virtual_value, DatasetScalar{false})
                .string();
        });
    emit_call(
        "virtual_seed_zero",
        [&]
        {
            return virne::utils::get_v_nets_dataset_dir_from_setting(
                       virtual_value, integer(0))
                .string();
        });
    emit_call(
        "virtual_seed_negative",
        [&]
        {
            return virne::utils::get_v_nets_dataset_dir_from_setting(
                       virtual_value, integer(-9))
                .string();
        });
    VirtualDatasetSetting empty_virtual = virtual_value;
    empty_virtual.node_attributes.clear();
    empty_virtual.link_attributes.clear();
    emit_call(
        "virtual_empty_attributes",
        [&]
        {
            return virne::utils::get_v_nets_dataset_dir_from_setting(empty_virtual)
                .string();
        });
    VirtualDatasetSetting invalid_virtual = virtual_value;
    invalid_virtual.lifetime = normal;
    emit_call(
        "virtual_normal_error",
        [&]
        {
            return virne::utils::get_v_nets_dataset_dir_from_setting(invalid_virtual)
                .string();
        });
    VirtualDatasetSetting customized_virtual = virtual_value;
    customized_virtual.lifetime = {};
    customized_virtual.lifetime.kind = DistributionKind::customized;
    customized_virtual.lifetime.minimum = DatasetScalar{std::string("a")};
    customized_virtual.lifetime.maximum = DatasetScalar{false};
    emit_call(
        "virtual_customized_lifetime",
        [&]
        {
            return virne::utils::get_v_nets_dataset_dir_from_setting(
                       customized_virtual)
                .string();
        });

    std::vector<DatasetFileNameRequest> name_requests;
    std::vector<PhysicalDatasetPathRequest> physical_requests;
    std::vector<VirtualDatasetPathRequest> virtual_requests;
    constexpr std::size_t batch_size = 4096;
    name_requests.reserve(batch_size);
    physical_requests.reserve(batch_size);
    virtual_requests.reserve(batch_size);
    for (std::size_t index = 0; index < batch_size; ++index)
    {
        name_requests.push_back(file_name_request(index));
        physical_requests.push_back({physical, integer(static_cast<std::int64_t>(index))});
        virtual_requests.push_back({virtual_value, DatasetScalar{index % 2 == 0}});
    }
    for (const std::size_t workers : {1U, 2U, 4U, 8U, 0U})
    {
        const std::string suffix = workers == 0 ? "auto" : std::to_string(workers);
        emit_ok(
            "batch_filename_w" + suffix,
            summary_text(summarize(
                virne::utils::generate_file_names_batch(name_requests, workers))));
        emit_ok(
            "batch_physical_w" + suffix,
            summary_text(summarize(
                virne::utils::get_p_net_dataset_dirs_batch(
                    physical_requests, workers))));
        emit_ok(
            "batch_virtual_w" + suffix,
            summary_text(summarize(
                virne::utils::get_v_nets_dataset_dirs_batch(
                    virtual_requests, workers))));
    }
    std::cout << "status=PASS\n";
}

void run_float_bits()
{
    std::cout << "dataset_core_float_version=1\n";
    std::string line;
    while (std::getline(std::cin, line))
    {
        std::uint64_t bits = 0;
        const auto parsed = std::from_chars(
            line.data(), line.data() + line.size(), bits, 16);
        if (line.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != line.data() + line.size())
        {
            throw std::invalid_argument("invalid binary64 bit pattern");
        }
        double value = 0.0;
        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        std::cout << hex_encode(
                         virne::utils::format_dataset_scalar(DatasetScalar{value}))
                  << '\n';
    }
    std::cout << "status=PASS\n";
}

struct BenchmarkResult
{
    std::uint64_t elapsed_ns = 0;
    std::uint64_t checksum = 0;
    std::size_t output_bytes = 0;
};

BenchmarkResult run_benchmark(
    std::string_view kind,
    std::size_t operations,
    std::size_t batch_size,
    std::size_t workers,
    const fs::path& root)
{
    using Clock = std::chrono::steady_clock;
    if (kind == "parameters")
    {
        std::vector<DistributionSpec> corpus;
        corpus.push_back({});
        DistributionSpec exponential;
        exponential.kind = DistributionKind::exponential;
        exponential.scale = integer(500);
        corpus.push_back(exponential);
        DistributionSpec poisson;
        poisson.kind = DistributionKind::poisson;
        poisson.lambda = DatasetScalar{0.04};
        corpus.push_back(poisson);
        corpus.push_back(uniform(0, 100));
        std::vector<std::string> outputs(operations);
        const auto start = Clock::now();
        for (std::size_t index = 0; index < operations; ++index)
        {
            outputs[index] = virne::utils::get_parameters_string(
                virne::utils::get_distribution_parameters(
                    corpus[index % corpus.size()]));
        }
        const auto end = Clock::now();
        const auto summary = summarize(outputs);
        return {
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count()),
            summary.first,
            summary.second};
    }

    if (kind == "filename")
    {
        std::vector<DatasetFileNameRequest> requests;
        requests.reserve(batch_size);
        for (std::size_t index = 0; index < batch_size; ++index)
        {
            requests.push_back(file_name_request(index));
        }
        std::vector<std::string> outputs(operations);
        const auto start = Clock::now();
        for (std::size_t index = 0; index < operations; ++index)
        {
            const DatasetFileNameRequest& request = requests[index % requests.size()];
            outputs[index] = virne::utils::generate_file_name(
                request.config, request.epoch_id, request.ordered_items);
        }
        const auto end = Clock::now();
        const auto summary = summarize(outputs);
        return {
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count()),
            summary.first,
            summary.second};
    }

    const PhysicalDatasetSetting physical = physical_setting(root);
    const VirtualDatasetSetting virtual_value = virtual_setting(root);
    if (kind == "physical" || kind == "physical_existing" || kind == "virtual")
    {
        PhysicalDatasetSetting physical_input = physical;
        if (kind == "physical_existing")
        {
            physical_input.topology.file_path = root / "Topo.multi.part.gml";
        }
        std::vector<fs::path> outputs(operations);
        const auto start = Clock::now();
        for (std::size_t index = 0; index < operations; ++index)
        {
            if (kind == "virtual")
            {
                outputs[index] = virne::utils::get_v_nets_dataset_dir_from_setting(
                    virtual_value,
                    integer(static_cast<std::int64_t>(index % 17)));
            }
            else
            {
                outputs[index] = virne::utils::get_p_net_dataset_dir_from_setting(
                    physical_input,
                    integer(static_cast<std::int64_t>(index % 17)));
            }
        }
        const auto end = Clock::now();
        const auto summary = summarize(outputs);
        return {
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count()),
            summary.first,
            summary.second};
    }

    if (kind == "filename_batch")
    {
        std::vector<DatasetFileNameRequest> requests;
        requests.reserve(batch_size);
        for (std::size_t index = 0; index < batch_size; ++index)
        {
            requests.push_back(file_name_request(index));
        }
        std::vector<std::string> outputs;
        const auto start = Clock::now();
        for (std::size_t repeat = 0; repeat < operations; ++repeat)
        {
            outputs = virne::utils::generate_file_names_batch(requests, workers);
        }
        const auto end = Clock::now();
        const auto summary = summarize(outputs);
        return {
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count()),
            summary.first,
            summary.second};
    }

    std::vector<PhysicalDatasetPathRequest> physical_requests;
    std::vector<VirtualDatasetPathRequest> virtual_requests;
    physical_requests.reserve(batch_size);
    virtual_requests.reserve(batch_size);
    for (std::size_t index = 0; index < batch_size; ++index)
    {
        physical_requests.push_back(
            {physical, integer(static_cast<std::int64_t>(index % 17))});
        virtual_requests.push_back(
            {virtual_value, integer(static_cast<std::int64_t>(index % 17))});
    }
    std::vector<fs::path> outputs;
    const auto start = Clock::now();
    for (std::size_t repeat = 0; repeat < operations; ++repeat)
    {
        if (kind == "physical_batch")
        {
            outputs = virne::utils::get_p_net_dataset_dirs_batch(
                physical_requests, workers);
        }
        else if (kind == "virtual_batch")
        {
            outputs = virne::utils::get_v_nets_dataset_dirs_batch(
                virtual_requests, workers);
        }
        else
        {
            throw std::invalid_argument("unsupported benchmark kind");
        }
    }
    const auto end = Clock::now();
    const auto summary = summarize(outputs);
    return {
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count()),
        summary.first,
        summary.second};
}

std::size_t parse_size(const char* text)
{
    const std::string value(text);
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("invalid size argument");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string_view(argv[1]) == "cases")
        {
            run_cases(fs::path(argv[2]));
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "float_bits")
        {
            run_float_bits();
            return 0;
        }
        if (argc == 7 && std::string_view(argv[1]) == "benchmark")
        {
            const std::string kind(argv[2]);
            const std::size_t operations = parse_size(argv[3]);
            const std::size_t batch_size = parse_size(argv[4]);
            const std::size_t workers = parse_size(argv[5]);
            const fs::path root(argv[6]);
            const BenchmarkResult result = run_benchmark(
                kind, operations, batch_size, workers, root);
            std::cout << "benchmark_version=1\n"
                      << "kind=" << kind << '\n'
                      << "workers=" << workers << '\n'
                      << "operations=" << operations << '\n'
                      << "batch_size=" << batch_size << '\n'
                      << "elapsed_ns=" << result.elapsed_ns << '\n'
                      << "checksum=" << result.checksum << '\n'
                      << "output_bytes=" << result.output_bytes << '\n'
                      << "status=PASS\n";
            return 0;
        }
        throw std::invalid_argument("invalid dataset core harness command");
    }
    catch (const std::exception& error)
    {
        std::cerr << "dataset core harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
