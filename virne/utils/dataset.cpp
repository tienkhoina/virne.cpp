#include "dataset.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <exception>
#include <limits>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

namespace virne::utils
{
namespace
{

enum class BatchFamily : std::uint8_t
{
    file_name,
    dataset_path,
};

[[noreturn]] void throw_missing_parameter()
{
    throw DatasetException(
        DatasetErrorCode::missing_parameter,
        DatasetOperation::resolve_distribution,
        "missing distribution parameter");
}

const DatasetScalar& require_parameter(
    const std::optional<DatasetScalar>& value)
{
    if (!value)
    {
        throw_missing_parameter();
    }
    return *value;
}

void append_integer(std::string& output, std::int64_t value)
{
    char buffer[32];
    const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value);
    if (result.ec != std::errc{})
    {
        throw DatasetException(
            DatasetErrorCode::invalid_parameter,
            DatasetOperation::format_parameters,
            "integer formatting failed");
    }
    output.append(buffer, result.ptr);
}

std::string expand_scientific_to_fixed(
    std::string_view mantissa,
    int exponent)
{
    bool negative = false;
    if (!mantissa.empty() && mantissa.front() == '-')
    {
        negative = true;
        mantissa.remove_prefix(1);
    }

    std::string digits;
    digits.reserve(mantissa.size());
    for (const char character : mantissa)
    {
        if (character != '.')
        {
            digits.push_back(character);
        }
    }

    const auto decimal_position = static_cast<std::ptrdiff_t>(exponent) + 1;
    std::string result;
    result.reserve(digits.size() + static_cast<std::size_t>(std::abs(exponent)) + 4);
    if (negative)
    {
        result.push_back('-');
    }
    if (decimal_position <= 0)
    {
        result += "0.";
        result.append(static_cast<std::size_t>(-decimal_position), '0');
        result += digits;
    }
    else if (decimal_position >= static_cast<std::ptrdiff_t>(digits.size()))
    {
        result += digits;
        result.append(
            static_cast<std::size_t>(
                decimal_position - static_cast<std::ptrdiff_t>(digits.size())),
            '0');
        result += ".0";
    }
    else
    {
        const auto split = static_cast<std::size_t>(decimal_position);
        result.append(digits.data(), split);
        result.push_back('.');
        result.append(digits.data() + split, digits.size() - split);
    }
    return result;
}

std::string format_floating(double value)
{
    if (std::isnan(value))
    {
        return "nan";
    }
    if (std::isinf(value))
    {
        return std::signbit(value) ? "-inf" : "inf";
    }

    char buffer[128];
    const auto converted = std::to_chars(
        std::begin(buffer),
        std::end(buffer),
        value,
        std::chars_format::general);
    if (converted.ec != std::errc{})
    {
        throw DatasetException(
            DatasetErrorCode::invalid_parameter,
            DatasetOperation::format_parameters,
            "floating-point formatting failed");
    }

    std::string result(buffer, converted.ptr);
    const std::size_t exponent_marker = result.find_first_of("eE");
    if (exponent_marker != std::string::npos)
    {
        int exponent = 0;
        std::string_view exponent_text(result.data() + exponent_marker + 1,
                                       result.size() - exponent_marker - 1);
        if (!exponent_text.empty() && exponent_text.front() == '+')
        {
            exponent_text.remove_prefix(1);
        }
        const auto parsed = std::from_chars(
            exponent_text.data(),
            exponent_text.data() + exponent_text.size(),
            exponent);
        if (parsed.ec != std::errc{} || parsed.ptr != exponent_text.data() + exponent_text.size())
        {
            throw DatasetException(
                DatasetErrorCode::invalid_parameter,
                DatasetOperation::format_parameters,
                "floating-point exponent formatting failed");
        }
        if (exponent >= -4 && exponent < 16)
        {
            return expand_scientific_to_fixed(
                std::string_view(result.data(), exponent_marker), exponent);
        }
        result[exponent_marker] = 'e';
        return result;
    }

    if (result.find('.') == std::string::npos)
    {
        result += ".0";
    }
    return result;
}

void append_scalar(std::string& output, const DatasetScalar& value)
{
    std::visit(
        [&](const auto& item)
        {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, std::monostate>)
            {
                output += "None";
            }
            else if constexpr (std::is_same_v<Item, std::int64_t>)
            {
                append_integer(output, item);
            }
            else if constexpr (std::is_same_v<Item, double>)
            {
                output += format_floating(item);
            }
            else if constexpr (std::is_same_v<Item, bool>)
            {
                output += item ? "True" : "False";
            }
            else
            {
                output += item;
            }
        },
        value);
}

void append_parameter_string(
    std::string& output,
    const DistributionSpec& distribution)
{
    switch (distribution.kind)
    {
    case DistributionKind::none:
        output += "None";
        return;
    case DistributionKind::exponential:
        append_scalar(output, require_parameter(distribution.scale));
        return;
    case DistributionKind::poisson:
        append_scalar(output, require_parameter(distribution.lambda));
        return;
    case DistributionKind::uniform:
        output.push_back('[');
        append_scalar(output, require_parameter(distribution.low));
        output.push_back('-');
        append_scalar(output, require_parameter(distribution.high));
        output.push_back(']');
        return;
    case DistributionKind::customized:
        output.push_back('[');
        append_scalar(output, require_parameter(distribution.minimum));
        output.push_back('-');
        append_scalar(output, require_parameter(distribution.maximum));
        output.push_back(']');
        return;
    case DistributionKind::normal:
        throw DatasetException(
            DatasetErrorCode::unsupported_parameter_distribution,
            DatasetOperation::resolve_distribution,
            "normal parameters are unsupported by the Python helper");
    }
    throw DatasetException(
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution,
        "invalid distribution enum");
}

std::string format_attributes(
    const std::vector<DatasetAttributeSpec>& attributes)
{
    std::size_t total_size = attributes.size() * 24;
    for (const DatasetAttributeSpec& attribute : attributes)
    {
        total_size += attribute.name.size();
    }

    std::string result;
    result.reserve(total_size);
    for (std::size_t index = 0; index < attributes.size(); ++index)
    {
        if (index != 0)
        {
            result.push_back('-');
        }
        result += attributes[index].name;
        result.push_back('_');
        append_parameter_string(result, attributes[index].distribution);
    }
    return result;
}

void append_seed_suffix(
    std::string& output,
    const std::optional<DatasetScalar>& seed)
{
    if (!seed)
    {
        return;
    }
    output += "-seed_";
    append_scalar(output, *seed);
}

std::size_t effective_worker_count(
    std::size_t requested,
    std::size_t item_count,
    BatchFamily family)
{
    if (item_count < 2)
    {
        return item_count;
    }
    if (requested == 1)
    {
        return 1;
    }
    std::size_t workers = requested;
    if (workers == 0)
    {
        const std::size_t available = std::max<std::size_t>(
            1,
            static_cast<std::size_t>(std::thread::hardware_concurrency()));
        if (family == BatchFamily::file_name)
        {
            // Canonical 1..8 sweeps show allocator/thread overhead dominates
            // through 8,192 short names. Keep that common corpus sequential;
            // reserve five lanes for substantially larger independent batches.
            workers = item_count < 16384 ? 1 : std::min<std::size_t>(5, available);
        }
        else if (item_count < 1024)
        {
            workers = 1;
        }
        else if (item_count < 3072)
        {
            workers = std::min<std::size_t>(4, available);
        }
        else
        {
            workers = std::min<std::size_t>(6, available);
        }
    }
    return std::min(workers, item_count);
}

template <typename Request, typename Result, typename Function>
std::vector<Result> transform_batch(
    const std::vector<Request>& requests,
    std::size_t requested_workers,
    BatchFamily family,
    Function&& function)
{
    std::vector<Result> results(requests.size());
    const auto transform_one =
        [&](std::size_t index)
        {
            try
            {
                results[index] = function(requests[index]);
            }
            catch (const DatasetException& error)
            {
                throw DatasetException(
                    error.code(),
                    error.operation(),
                    error.what(),
                    index,
                    error.path());
            }
        };
    const std::size_t worker_count =
        effective_worker_count(requested_workers, requests.size(), family);
    if (worker_count <= 1)
    {
        for (std::size_t index = 0; index < requests.size(); ++index)
        {
            transform_one(index);
        }
        return results;
    }

    struct WorkerFailure
    {
        std::size_t index = std::numeric_limits<std::size_t>::max();
        std::exception_ptr error;
    };
    std::vector<WorkerFailure> failures(worker_count);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker)
    {
        threads.emplace_back(
            [&, worker]
            {
                const std::size_t begin =
                    requests.size() * worker / worker_count;
                const std::size_t end =
                    requests.size() * (worker + 1) / worker_count;
                for (std::size_t index = begin; index < end; ++index)
                {
                    try
                    {
                        transform_one(index);
                    }
                    catch (...)
                    {
                        failures[worker] = {index, std::current_exception()};
                        break;
                    }
                }
            });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }

    const WorkerFailure* first = nullptr;
    for (const WorkerFailure& failure : failures)
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
    return results;
}

} // namespace

DatasetException::DatasetException(
    DatasetErrorCode code,
    DatasetOperation operation,
    std::string message,
    std::optional<std::size_t> input_index,
    std::filesystem::path path)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      input_index_(input_index),
      path_(std::move(path))
{
}

DatasetErrorCode DatasetException::code() const noexcept
{
    return code_;
}

DatasetOperation DatasetException::operation() const noexcept
{
    return operation_;
}

const std::optional<std::size_t>& DatasetException::input_index() const noexcept
{
    return input_index_;
}

const std::filesystem::path& DatasetException::path() const noexcept
{
    return path_;
}

DistributionKind distribution_kind_from_string(std::string_view value)
{
    if (value == "uniform")
    {
        return DistributionKind::uniform;
    }
    if (value == "normal")
    {
        return DistributionKind::normal;
    }
    if (value == "exponential")
    {
        return DistributionKind::exponential;
    }
    if (value == "poisson")
    {
        return DistributionKind::poisson;
    }
    if (value == "customized")
    {
        return DistributionKind::customized;
    }
    if (value.empty())
    {
        return DistributionKind::none;
    }
    throw DatasetException(
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution,
        "unsupported distribution");
}

DatasetTopologyKind dataset_topology_kind_from_string(std::string_view value)
{
    if (value == "path")
    {
        return DatasetTopologyKind::path;
    }
    if (value == "star")
    {
        return DatasetTopologyKind::star;
    }
    if (value == "grid_2d")
    {
        return DatasetTopologyKind::grid_2d;
    }
    if (value == "waxman")
    {
        return DatasetTopologyKind::waxman;
    }
    if (value == "random")
    {
        return DatasetTopologyKind::random;
    }
    throw DatasetException(
        DatasetErrorCode::invalid_topology,
        DatasetOperation::resolve_topology,
        "unsupported topology");
}

std::string_view dataset_topology_kind_name(DatasetTopologyKind value)
{
    switch (value)
    {
    case DatasetTopologyKind::path:
        return "path";
    case DatasetTopologyKind::star:
        return "star";
    case DatasetTopologyKind::grid_2d:
        return "grid_2d";
    case DatasetTopologyKind::waxman:
        return "waxman";
    case DatasetTopologyKind::random:
        return "random";
    }
    throw DatasetException(
        DatasetErrorCode::invalid_topology,
        DatasetOperation::resolve_topology,
        "invalid topology enum");
}

std::string format_dataset_scalar(const DatasetScalar& value)
{
    std::string result;
    append_scalar(result, value);
    return result;
}

std::vector<DatasetScalar> get_distribution_parameters(
    const DistributionSpec& distribution)
{
    switch (distribution.kind)
    {
    case DistributionKind::none:
        return {};
    case DistributionKind::exponential:
        return {require_parameter(distribution.scale)};
    case DistributionKind::poisson:
        return {require_parameter(distribution.lambda)};
    case DistributionKind::uniform:
        return {
            require_parameter(distribution.low),
            require_parameter(distribution.high)};
    case DistributionKind::customized:
        return {
            require_parameter(distribution.minimum),
            require_parameter(distribution.maximum)};
    case DistributionKind::normal:
        throw DatasetException(
            DatasetErrorCode::unsupported_parameter_distribution,
            DatasetOperation::resolve_distribution,
            "normal parameters are unsupported by the Python helper");
    }
    throw DatasetException(
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution,
        "invalid distribution enum");
}

std::string get_parameters_string(
    const std::vector<DatasetScalar>& parameters)
{
    if (parameters.empty())
    {
        return "None";
    }
    if (parameters.size() == 1)
    {
        return format_dataset_scalar(parameters.front());
    }

    std::string result;
    result.push_back('[');
    for (std::size_t index = 0; index < parameters.size(); ++index)
    {
        if (index != 0)
        {
            result.push_back('-');
        }
        append_scalar(result, parameters[index]);
    }
    result.push_back(']');
    return result;
}

std::optional<double> get_distribution_average(
    const DistributionSpec&,
    DatasetValueKind) noexcept
{
    return std::nullopt;
}

std::string generate_file_name(
    const DatasetFileNameConfig& config,
    std::int64_t epoch_id,
    const std::vector<OrderedFileNameItem>& ordered_items)
{
    std::size_t total_size =
        config.solver_name.size() + ordered_items.size() * 24 + 24;
    for (const OrderedFileNameItem& item : ordered_items)
    {
        total_size += item.key.size() + 2;
        if (const auto* text = std::get_if<std::string>(&item.value))
        {
            total_size += text->size();
        }
    }

    std::string result;
    result.reserve(total_size);
    result += config.solver_name;
    result += "-records-";
    append_integer(result, epoch_id);
    result.push_back('-');
    for (std::size_t index = 0; index < ordered_items.size(); ++index)
    {
        if (index != 0)
        {
            result.push_back('-');
        }
        result += ordered_items[index].key;
        result.push_back('=');
        append_scalar(result, ordered_items[index].value);
    }
    result += ".csv";
    return result;
}

std::filesystem::path get_p_net_dataset_dir_from_setting(
    const PhysicalDatasetSetting& setting,
    const std::optional<DatasetScalar>& seed)
{
    std::string topology;
    bool use_file = false;
    if (setting.topology.file_path)
    {
        const std::filesystem::path& file_path = *setting.topology.file_path;
        const std::string native = file_path.string();
        if (!native.empty() && native != "None")
        {
            std::error_code error;
            use_file = std::filesystem::exists(file_path, error) && !error;
        }
        if (use_file)
        {
            topology = file_path.filename().string();
            const std::size_t first_dot = topology.find('.');
            if (first_dot != std::string::npos)
            {
                topology.resize(first_dot);
            }
        }
    }
    if (!use_file)
    {
        topology.reserve(80);
        append_integer(topology, setting.topology.num_nodes);
        topology.push_back('-');
        topology += dataset_topology_kind_name(setting.topology.topology_type);
        topology += "_[";
        append_scalar(topology, setting.topology.wm_alpha);
        topology.push_back('-');
        append_scalar(topology, setting.topology.wm_beta);
        topology.push_back(']');
    }

    const std::string node_attributes = format_attributes(setting.node_attributes);
    const std::string link_attributes = format_attributes(setting.link_attributes);
    std::string middle;
    middle.reserve(
        topology.size() + node_attributes.size() + link_attributes.size() + 32);
    middle += topology;
    middle.push_back('-');
    middle += node_attributes;
    middle.push_back('-');
    middle += link_attributes;
    append_seed_suffix(middle, seed);
    return setting.save_dir / middle;
}

std::filesystem::path get_v_nets_dataset_dir_from_setting(
    const VirtualDatasetSetting& setting,
    const std::optional<DatasetScalar>& seed)
{
    const std::string node_attributes = format_attributes(setting.node_attributes);
    const std::string link_attributes = format_attributes(setting.link_attributes);

    std::string middle;
    middle.reserve(
        node_attributes.size() + link_attributes.size() + 128);
    append_integer(middle, setting.num_virtual_networks);
    middle += "-[";
    append_integer(middle, setting.size_low);
    middle.push_back('-');
    append_integer(middle, setting.size_high);
    middle += "]-";
    middle += dataset_topology_kind_name(setting.topology_type);
    middle.push_back('-');
    append_parameter_string(middle, setting.lifetime);
    middle.push_back('-');
    append_scalar(middle, setting.arrival_lambda);
    middle.push_back('-');
    middle += node_attributes;
    middle.push_back('-');
    middle += link_attributes;
    append_seed_suffix(middle, seed);
    return setting.save_dir / middle;
}

std::vector<std::string> generate_file_names_batch(
    const std::vector<DatasetFileNameRequest>& requests,
    std::size_t workers)
{
    return transform_batch<DatasetFileNameRequest, std::string>(
        requests,
        workers,
        BatchFamily::file_name,
        [](const DatasetFileNameRequest& request)
        {
            return generate_file_name(
                request.config, request.epoch_id, request.ordered_items);
        });
}

std::vector<std::filesystem::path> get_p_net_dataset_dirs_batch(
    const std::vector<PhysicalDatasetPathRequest>& requests,
    std::size_t workers)
{
    return transform_batch<PhysicalDatasetPathRequest, std::filesystem::path>(
        requests,
        workers,
        BatchFamily::dataset_path,
        [](const PhysicalDatasetPathRequest& request)
        {
            return get_p_net_dataset_dir_from_setting(
                request.setting, request.seed);
        });
}

std::vector<std::filesystem::path> get_v_nets_dataset_dirs_batch(
    const std::vector<VirtualDatasetPathRequest>& requests,
    std::size_t workers)
{
    return transform_batch<VirtualDatasetPathRequest, std::filesystem::path>(
        requests,
        workers,
        BatchFamily::dataset_path,
        [](const VirtualDatasetPathRequest& request)
        {
            return get_v_nets_dataset_dir_from_setting(
                request.setting, request.seed);
        });
}

} // namespace virne::utils
