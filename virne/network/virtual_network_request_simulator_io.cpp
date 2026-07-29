#include "virtual_network_request_simulator.h"
#include "virtual_network_request_simulator_detail.h"

#include "random_context.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace virne::network {
namespace {

namespace fs = std::filesystem;
namespace utils = virne::utils;

[[noreturn]] void throw_setting_error(
    VirtualNetworkSimulatorErrorCode code,
    std::string message) {
    throw VirtualNetworkSimulatorException(
        code,
        VirtualNetworkSimulatorOperation::decode_setting,
        std::move(message));
}

const utils::SettingObject& require_object(
    const utils::SettingValue& value,
    std::string_view context) {
    if (value.kind() != utils::SettingValueKind::object) {
        throw_setting_error(
            VirtualNetworkSimulatorErrorCode::invalid_setting,
            std::string(context) + " must be an object");
    }
    return value.as_object();
}

const utils::SettingValue* find_value(
    const utils::SettingObject& object,
    std::string_view name) {
    const auto id = object.find_key_id(name);
    return id ? &object.at(*id) : nullptr;
}

const utils::SettingValue& require_value(
    const utils::SettingObject& object,
    std::string_view name) {
    const utils::SettingValue* value = find_value(object, name);
    if (value == nullptr) {
        throw_setting_error(
            VirtualNetworkSimulatorErrorCode::missing_setting_field,
            "missing simulator setting field: " + std::string(name));
    }
    return *value;
}

std::int64_t parse_int64_text(
    std::string_view text,
    std::string_view context) {
    std::int64_t result = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw_setting_error(
            VirtualNetworkSimulatorErrorCode::invalid_setting_value,
            std::string(context) + " is not a signed int64");
    }
    return result;
}

std::uint64_t parse_uint64_text(
    std::string_view text,
    std::string_view context) {
    std::uint64_t result = 0U;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw_setting_error(
            VirtualNetworkSimulatorErrorCode::invalid_setting_value,
            std::string(context) + " is not an unsigned integer");
    }
    return result;
}

std::int64_t setting_int64(
    const utils::SettingValue& value,
    std::string_view context) {
    switch (value.kind()) {
    case utils::SettingValueKind::boolean:
        return value.as_bool() ? 1 : 0;
    case utils::SettingValueKind::integer:
        return parse_int64_text(
            value.as_integer().convert_to<std::string>(), context);
    case utils::SettingValueKind::real:
    {
        const double number = value.as_real();
        constexpr double lower = -9223372036854775808.0;
        constexpr double upper = 9223372036854775808.0;
        if (!std::isfinite(number) || number < lower || number >= upper) {
            break;
        }
        return static_cast<std::int64_t>(number);
    }
    case utils::SettingValueKind::string:
        return parse_int64_text(value.as_string(), context);
    case utils::SettingValueKind::null_value:
    case utils::SettingValueKind::list:
    case utils::SettingValueKind::object:
        break;
    }
    throw_setting_error(
        VirtualNetworkSimulatorErrorCode::invalid_setting_value,
        std::string(context) + " is not convertible to int64");
}

std::size_t setting_size(
    const utils::SettingValue& value,
    std::string_view context) {
    std::uint64_t number = 0U;
    if (value.kind() == utils::SettingValueKind::integer) {
        const std::string text =
            value.as_integer().convert_to<std::string>();
        number = parse_uint64_text(text, context);
    } else if (value.kind() == utils::SettingValueKind::string) {
        number = parse_uint64_text(value.as_string(), context);
    } else {
        const std::int64_t signed_value = setting_int64(value, context);
        if (signed_value < 0) {
            throw_setting_error(
                VirtualNetworkSimulatorErrorCode::invalid_setting_value,
                std::string(context) + " must be non-negative");
        }
        number = static_cast<std::uint64_t>(signed_value);
    }
    if (number > std::numeric_limits<std::size_t>::max()) {
        throw_setting_error(
            VirtualNetworkSimulatorErrorCode::invalid_setting_value,
            std::string(context) + " exceeds size_t");
    }
    return static_cast<std::size_t>(number);
}

double parse_double_text(
    const std::string& text,
    std::string_view context) {
    std::size_t consumed = 0U;
    try {
        const double result = std::stod(text, &consumed);
        if (consumed == text.size()) {
            return result;
        }
    } catch (const std::exception&) {
    }
    throw_setting_error(
        VirtualNetworkSimulatorErrorCode::invalid_setting_value,
        std::string(context) + " is not convertible to binary64");
}

double setting_double(
    const utils::SettingValue& value,
    std::string_view context) {
    switch (value.kind()) {
    case utils::SettingValueKind::boolean:
        return value.as_bool() ? 1.0 : 0.0;
    case utils::SettingValueKind::integer:
        return parse_double_text(
            value.as_integer().convert_to<std::string>(), context);
    case utils::SettingValueKind::real:
        return value.as_real();
    case utils::SettingValueKind::string:
        return parse_double_text(value.as_string(), context);
    case utils::SettingValueKind::null_value:
    case utils::SettingValueKind::list:
    case utils::SettingValueKind::object:
        break;
    }
    throw_setting_error(
        VirtualNetworkSimulatorErrorCode::invalid_setting_value,
        std::string(context) + " is not convertible to binary64");
}

bool setting_bool(
    const utils::SettingValue& value,
    std::string_view context) {
    if (value.kind() == utils::SettingValueKind::boolean) {
        return value.as_bool();
    }
    if (value.kind() == utils::SettingValueKind::integer) {
        return setting_int64(value, context) != 0;
    }
    throw_setting_error(
        VirtualNetworkSimulatorErrorCode::invalid_setting_value,
        std::string(context) + " must be boolean");
}

const std::string& setting_string(
    const utils::SettingValue& value,
    std::string_view context) {
    if (value.kind() != utils::SettingValueKind::string) {
        throw_setting_error(
            VirtualNetworkSimulatorErrorCode::invalid_setting_value,
            std::string(context) + " must be a string");
    }
    return value.as_string();
}

utils::DatasetScalar setting_scalar(
    const utils::SettingValue& value,
    std::string_view context) {
    switch (value.kind()) {
    case utils::SettingValueKind::null_value:
        return std::monostate{};
    case utils::SettingValueKind::boolean:
        return value.as_bool();
    case utils::SettingValueKind::integer:
        return setting_int64(value, context);
    case utils::SettingValueKind::real:
        return value.as_real();
    case utils::SettingValueKind::string:
        return value.as_string();
    case utils::SettingValueKind::list:
    case utils::SettingValueKind::object:
        break;
    }
    throw_setting_error(
        VirtualNetworkSimulatorErrorCode::invalid_setting_value,
        std::string(context) + " must be a scalar");
}

AttrValue attr_value_from_setting(
    const utils::SettingValue& value,
    std::string_view context) {
    switch (value.kind()) {
    case utils::SettingValueKind::boolean:
        return value.as_bool();
    case utils::SettingValueKind::integer:
        return setting_int64(value, context);
    case utils::SettingValueKind::real:
        return value.as_real();
    case utils::SettingValueKind::string:
        return value.as_string();
    case utils::SettingValueKind::list:
    {
        std::vector<AttrValue> items;
        items.reserve(value.as_list().size());
        for (const utils::SettingValue& item : value.as_list()) {
            items.push_back(attr_value_from_setting(item, context));
        }
        return make_attr_list(std::move(items));
    }
    case utils::SettingValueKind::object:
    {
        const utils::SettingObject& object = value.as_object();
        std::vector<std::pair<std::string, AttrValue>> entries;
        entries.reserve(object.size());
        for (std::size_t index = 0U; index < object.size(); ++index) {
            const utils::SettingKeyId id{
                static_cast<std::uint32_t>(index)};
            entries.emplace_back(
                object.key_name(id),
                attr_value_from_setting(object.at(id), context));
        }
        return make_attr_object(std::move(entries));
    }
    case utils::SettingValueKind::null_value:
        break;
    }
    throw_setting_error(
        VirtualNetworkSimulatorErrorCode::invalid_setting_value,
        std::string(context) + " contains null, which AttrValue cannot store");
}

SimulationDistribution decode_distribution(
    const utils::SettingValue& value,
    std::string_view context) {
    const utils::SettingObject& object = require_object(value, context);
    SimulationDistribution result;
    result.distribution.kind = utils::distribution_kind_from_string(
        setting_string(
            require_value(object, "distribution"),
            "distribution.distribution"));
    result.value_kind = utils::dataset_value_kind_from_string(
        setting_string(require_value(object, "dtype"), "distribution.dtype"));

    const auto assign_scalar = [&object, &result, context](
        std::string_view key,
        std::optional<utils::DatasetScalar>& destination) {
        if (const utils::SettingValue* item = find_value(object, key)) {
            destination = setting_scalar(
                *item, std::string(context) + "." + std::string(key));
        }
    };
    assign_scalar("low", result.distribution.low);
    assign_scalar("high", result.distribution.high);
    assign_scalar("loc", result.distribution.loc);
    assign_scalar("scale", result.distribution.scale);
    assign_scalar("lam", result.distribution.lambda);
    assign_scalar("minimum", result.distribution.minimum);
    assign_scalar("maximum", result.distribution.maximum);
    if (const utils::SettingValue* reciprocal =
            find_value(object, "reciprocal")) {
        result.distribution.reciprocal =
            setting_bool(*reciprocal, "distribution.reciprocal");
    }
    return result;
}

std::vector<attribute::AttributeFactorySpec> decode_attribute_specs(
    const utils::SettingObject& root,
    std::string_view key) {
    const utils::SettingValue* value = find_value(root, key);
    if (value == nullptr) {
        return {};
    }
    if (value->kind() != utils::SettingValueKind::list) {
        throw_setting_error(
            VirtualNetworkSimulatorErrorCode::invalid_setting_value,
            std::string(key) + " must be a list");
    }
    std::vector<attribute::AttributeFactorySpec> result;
    result.reserve(value->as_list().size());
    for (const utils::SettingValue& item : value->as_list()) {
        result.push_back(attribute::attribute_factory_spec_from_setting(
            require_object(item, key)));
    }
    return result;
}

void decode_topology(
    const utils::SettingValue& value,
    VirtualNetworkSimulationConfig& config) {
    const utils::SettingObject& topology = require_object(value, "topology");
    config.topology_type = TopologyType::Random;
    if (const utils::SettingValue* type = find_value(topology, "type")) {
        config.topology_type = topology_type_from_string(
            setting_string(*type, "topology.type"));
    }
    if (const utils::SettingValue* item = find_value(topology, "m")) {
        config.topology_options.m = setting_int64(*item, "topology.m");
    }
    if (const utils::SettingValue* item = find_value(topology, "n")) {
        config.topology_options.n = setting_int64(*item, "topology.n");
    }
    if (const utils::SettingValue* item =
            find_value(topology, "wm_alpha")) {
        config.topology_options.wm_alpha =
            setting_double(*item, "topology.wm_alpha");
    }
    if (const utils::SettingValue* item =
            find_value(topology, "wm_beta")) {
        config.topology_options.wm_beta =
            setting_double(*item, "topology.wm_beta");
    }
    if (const utils::SettingValue* item =
            find_value(topology, "random_prob")) {
        config.topology_options.random_prob =
            setting_double(*item, "topology.random_prob");
    }
    if (const utils::SettingValue* item =
            find_value(topology, "max_attempts")) {
        config.topology_options.max_attempts =
            setting_size(*item, "topology.max_attempts");
    }
    config.topology_metadata = attr_value_from_setting(value, "topology");
}

utils::SettingValue make_event_setting(
    const VirtualNetworkEvent& event) {
    auto object = std::make_shared<utils::SettingObject>();
    object->reserve(4U);
    object->set_owned(
        "id", utils::SettingValue{static_cast<std::uint64_t>(event.id())});
    object->set_owned(
        "type",
        utils::SettingValue{static_cast<std::int64_t>(event.type())});
    object->set_owned(
        "v_net_id", utils::SettingValue{event.virtual_network_id()});
    object->set_owned("time", utils::SettingValue{event.time()});
    return utils::SettingValue{std::move(object)};
}

utils::SettingDocument make_events_document(
    const std::vector<VirtualNetworkEvent>& events) {
    auto list = std::make_shared<utils::SettingList>();
    list->reserve(events.size());
    for (const VirtualNetworkEvent& event : events) {
        list->push_back(make_event_setting(event));
    }
    return {utils::SettingValue{std::move(list)}};
}

VirtualNetworkEvent decode_event(
    const utils::SettingValue& value,
    std::size_t input_index) {
    const utils::SettingObject& object = require_object(value, "event");
    try {
        const VirtualRequestId request_id = setting_int64(
            require_value(object, "v_net_id"), "event.v_net_id");
        const double time = setting_double(
            require_value(object, "time"), "event.time");
        const std::int64_t raw_type = setting_int64(
            require_value(object, "type"), "event.type");
        const std::size_t id = setting_size(
            require_value(object, "id"), "event.id");
        return VirtualNetworkEvent(
            id,
            static_cast<VirtualEventType>(raw_type),
            request_id,
            time);
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::invalid_event_setting,
            VirtualNetworkSimulatorOperation::load_events,
            "invalid event setting",
            input_index));
    }
}

std::vector<VirtualNetworkEvent> decode_events_document(
    const utils::SettingDocument& document) {
    if (document.root.kind() != utils::SettingValueKind::list) {
        throw VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::invalid_event_setting,
            VirtualNetworkSimulatorOperation::load_events,
            "events setting root must be a list");
    }
    const utils::SettingList& list = document.root.as_list();
    std::vector<VirtualNetworkEvent> result;
    result.reserve(list.size());
    for (std::size_t index = 0U; index < list.size(); ++index) {
        result.push_back(decode_event(list[index], index));
    }
    return result;
}

std::string padded_request_id(VirtualRequestId value) {
    std::string digits;
    if (value >= 0) {
        digits = std::to_string(value);
        if (digits.size() < 5U) {
            digits.insert(0U, 5U - digits.size(), '0');
        }
        return digits;
    }
    const std::uint64_t magnitude =
        static_cast<std::uint64_t>(-(value + 1)) + 1U;
    digits = std::to_string(magnitude);
    if (digits.size() < 4U) {
        digits.insert(0U, 4U - digits.size(), '0');
    }
    return "-" + digits;
}

std::size_t bounded_workers(
    std::size_t requested,
    std::size_t count) noexcept {
    if (requested <= 1U || count <= 1U) {
        return count == 0U ? 0U : 1U;
    }
    return std::min(requested, count);
}

template <typename Callable>
std::vector<std::exception_ptr> run_indexed(
    std::size_t count,
    std::size_t requested_workers,
    Callable&& callable) {
    std::vector<std::exception_ptr> errors(count);
    const std::size_t workers = bounded_workers(requested_workers, count);
    const auto run = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            try {
                callable(index);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        }
    };
    if (workers <= 1U) {
        run(0U, count);
        return errors;
    }

    std::vector<std::thread> threads;
    threads.reserve(workers - 1U);
    try {
        for (std::size_t worker = 1U; worker < workers; ++worker) {
            threads.emplace_back(
                run,
                count * worker / workers,
                count * (worker + 1U) / workers);
        }
    } catch (...) {
        for (std::thread& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        throw;
    }
    run(0U, count / workers);
    for (std::thread& thread : threads) {
        thread.join();
    }
    return errors;
}

std::optional<std::size_t> first_error_index(
    const std::vector<std::exception_ptr>& errors) noexcept {
    for (std::size_t index = 0U; index < errors.size(); ++index) {
        if (errors[index]) {
            return index;
        }
    }
    return std::nullopt;
}

class StageDirectory {
public:
    explicit StageDirectory(fs::path path) : path_(std::move(path)) {}
    StageDirectory(const StageDirectory&) = delete;
    StageDirectory& operator=(const StageDirectory&) = delete;
    ~StageDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

fs::path make_stage_directory(const fs::path& parent) {
    static std::atomic<std::uint64_t> sequence{0U};
    for (;;) {
        const std::uint64_t id =
            sequence.fetch_add(1U, std::memory_order_relaxed);
        fs::path candidate =
            parent / (".virne-save-stage-" + std::to_string(id));
        std::error_code error;
        if (fs::create_directory(candidate, error)) {
            return candidate;
        }
        if (error) {
            throw fs::filesystem_error(
                "create simulator stage directory",
                candidate,
                error);
        }
    }
}

void require_dataset_path(
    bool condition,
    std::string message,
    const fs::path& path) {
    if (!condition) {
        throw VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::invalid_dataset_layout,
            VirtualNetworkSimulatorOperation::load_layout,
            std::move(message),
            std::nullopt,
            path);
    }
}

}  // namespace

VirtualNetworkSimulationConfig virtual_network_simulation_config_from_setting(
    const utils::SettingDocument& setting) {
    const utils::SettingObject& root =
        require_object(setting.root, "simulator setting root");
    VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = setting_size(
        require_value(root, "num_v_nets"), "num_v_nets");
    config.virtual_network_size = decode_distribution(
        require_value(root, "v_net_size"), "v_net_size");
    config.lifetime = decode_distribution(
        require_value(root, "lifetime"), "lifetime");
    config.arrival_rate = decode_distribution(
        require_value(root, "arrival_rate"), "arrival_rate");
    if (const utils::SettingValue* max_latency =
            find_value(root, "max_latency")) {
        config.max_latency =
            decode_distribution(*max_latency, "max_latency");
    }
    decode_topology(require_value(root, "topology"), config);
    config.node_attribute_specs =
        decode_attribute_specs(root, "node_attrs_setting");
    config.link_attribute_specs =
        decode_attribute_specs(root, "link_attrs_setting");

    if (const utils::SettingValue* output = find_value(root, "output")) {
        const utils::SettingObject& output_object =
            require_object(*output, "output");
        if (const utils::SettingValue* save_dir =
                find_value(output_object, "save_dir")) {
            config.output.save_dir =
                setting_string(*save_dir, "output.save_dir");
        }
        if (const utils::SettingValue* name =
                find_value(output_object, "events_file_name")) {
            config.output.events_file_name =
                setting_string(*name, "output.events_file_name");
        }
        if (const utils::SettingValue* name =
                find_value(output_object, "setting_file_name")) {
            config.output.setting_file_name =
                setting_string(*name, "output.setting_file_name");
        }
        config.output_metadata = attr_value_from_setting(*output, "output");
    }
    config.source_setting = detail::clone_setting_document(setting);
    return config;
}

VirtualNetworkRequestSimulator
VirtualNetworkRequestSimulator::from_setting(
    const utils::SettingDocument& setting,
    std::optional<std::uint32_t> seed) {
    return from_setting(setting, global_random_context(), seed);
}

VirtualNetworkRequestSimulator
VirtualNetworkRequestSimulator::from_setting(
    const utils::SettingDocument& setting,
    RandomContext& random,
    std::optional<std::uint32_t> seed) {
    random.set_seed(seed);
    return from_setting(virtual_network_simulation_config_from_setting(setting));
}

void VirtualNetworkRequestSimulator::save_setting(
    const fs::path& path) const {
    if (!config_.source_setting) {
        throw VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::missing_source_setting,
            VirtualNetworkSimulatorOperation::save_setting,
            "typed simulator config has no source setting snapshot",
            std::nullopt,
            path);
    }
    try {
        utils::write_setting_strict(
            *config_.source_setting,
            path.string(),
            utils::SettingMode::write_update);
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::io_failure,
            VirtualNetworkSimulatorOperation::save_setting,
            "failed to save simulator setting",
            std::nullopt,
            path));
    }
}

void VirtualNetworkRequestSimulator::save_dataset(
    const fs::path& directory,
    std::size_t workers) const {
    const fs::path networks_directory = directory / "v_nets";
    try {
        fs::create_directories(directory);
        if (!fs::exists(networks_directory)) {
            fs::create_directory(networks_directory);
        }
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::io_failure,
            VirtualNetworkSimulatorOperation::save_networks,
            "failed to create simulator dataset directories",
            std::nullopt,
            networks_directory));
    }

    std::vector<fs::path> targets;
    targets.reserve(virtual_networks_.size());
    for (std::size_t index = 0U; index < virtual_networks_.size(); ++index) {
        const VirtualRequestId request_id = virtual_networks_[index]
            .request_id().value_or(static_cast<VirtualRequestId>(index));
        targets.push_back(
            networks_directory /
            ("v_net-" + padded_request_id(request_id) + ".gml"));
    }

    if (bounded_workers(workers, virtual_networks_.size()) <= 1U) {
        for (std::size_t index = 0U; index < virtual_networks_.size(); ++index) {
            try {
                virtual_networks_[index].to_gml(targets[index].string());
            } catch (...) {
                std::throw_with_nested(VirtualNetworkSimulatorException(
                    VirtualNetworkSimulatorErrorCode::io_failure,
                    VirtualNetworkSimulatorOperation::save_networks,
                    "failed to save virtual network",
                    index,
                    targets[index]));
            }
        }
    } else {
        StageDirectory stage(make_stage_directory(networks_directory));
        std::vector<fs::path> staged(virtual_networks_.size());
        for (std::size_t index = 0U; index < staged.size(); ++index) {
            std::ostringstream name;
            name << "item-" << std::setfill('0') << std::setw(10) << index
                 << ".gml";
            staged[index] = stage.path() / name.str();
        }
        const std::vector<std::exception_ptr> errors = run_indexed(
            virtual_networks_.size(),
            workers,
            [&](std::size_t index) {
                virtual_networks_[index].to_gml(staged[index].string());
            });
        const std::size_t commit_count =
            first_error_index(errors).value_or(virtual_networks_.size());
        for (std::size_t index = 0U; index < commit_count; ++index) {
            try {
                fs::copy_file(
                    staged[index],
                    targets[index],
                    fs::copy_options::overwrite_existing);
            } catch (...) {
                std::throw_with_nested(VirtualNetworkSimulatorException(
                    VirtualNetworkSimulatorErrorCode::io_failure,
                    VirtualNetworkSimulatorOperation::save_networks,
                    "failed to commit virtual network",
                    index,
                    targets[index]));
            }
        }
        if (commit_count < virtual_networks_.size()) {
            try {
                std::rethrow_exception(errors[commit_count]);
            } catch (...) {
                std::throw_with_nested(VirtualNetworkSimulatorException(
                    VirtualNetworkSimulatorErrorCode::io_failure,
                    VirtualNetworkSimulatorOperation::save_networks,
                    "failed to serialize virtual network",
                    commit_count,
                    targets[commit_count]));
            }
        }
    }

    const fs::path events_path =
        directory / config_.output.events_file_name;
    try {
        const utils::SettingDocument event_document =
            make_events_document(events_);
        utils::write_setting_strict(
            event_document,
            events_path.string(),
            utils::SettingMode::write_update);
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::io_failure,
            VirtualNetworkSimulatorOperation::save_events,
            "failed to save simulator events",
            std::nullopt,
            events_path));
    }
    save_setting(directory / config_.output.setting_file_name);
}

VirtualNetworkRequestSimulator
VirtualNetworkRequestSimulator::load_dataset(
    const fs::path& directory,
    VirtualNetworkDatasetCache& cache,
    std::size_t workers) {
    const std::string raw_directory = directory.string();
    if (raw_directory.find("seed_") != std::string::npos) {
        if (auto cached = cache.find(raw_directory)) {
            return std::move(*cached);
        }
    }

    const fs::path networks_directory = directory / "v_nets";
    const fs::path events_path = directory / "events.yaml";
    const fs::path setting_path = directory / "v_sim_setting.yaml";
    try {
        require_dataset_path(
            fs::exists(directory) && fs::is_directory(directory),
            "dataset directory does not exist",
            directory);
        require_dataset_path(
            fs::exists(networks_directory),
            "v_nets directory does not exist",
            networks_directory);
        require_dataset_path(
            fs::exists(events_path),
            "events.yaml does not exist",
            events_path);
        require_dataset_path(
            fs::exists(setting_path),
            "v_sim_setting.yaml does not exist",
            setting_path);
    } catch (const VirtualNetworkSimulatorException&) {
        throw;
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::io_failure,
            VirtualNetworkSimulatorOperation::load_layout,
            "failed while validating simulator dataset layout",
            std::nullopt,
            directory));
    }

    utils::SettingDocument setting;
    VirtualNetworkSimulationConfig config;
    try {
        setting = utils::read_setting(
            setting_path.string(), utils::SettingMode::read_update);
        config = virtual_network_simulation_config_from_setting(setting);
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::invalid_setting,
            VirtualNetworkSimulatorOperation::load_setting,
            "failed to load simulator setting",
            std::nullopt,
            setting_path));
    }

    std::vector<VirtualNetworkEvent> events;
    try {
        events = decode_events_document(utils::read_setting(
            events_path.string(), utils::SettingMode::read_update));
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::invalid_event_setting,
            VirtualNetworkSimulatorOperation::load_events,
            "failed to load simulator events",
            std::nullopt,
            events_path));
    }

    std::vector<fs::path> network_paths;
    try {
        for (const fs::directory_entry& entry :
             fs::directory_iterator(networks_directory)) {
            network_paths.push_back(entry.path());
        }
        std::sort(
            network_paths.begin(),
            network_paths.end(),
            [](const fs::path& left, const fs::path& right) {
                return left.filename().string() < right.filename().string();
            });
    } catch (...) {
        std::throw_with_nested(VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::io_failure,
            VirtualNetworkSimulatorOperation::load_networks,
            "failed to enumerate virtual network files",
            std::nullopt,
            networks_directory));
    }

    std::vector<std::optional<VirtualNetwork>> slots(network_paths.size());
    const std::vector<std::exception_ptr> errors = run_indexed(
        network_paths.size(),
        workers,
        [&](std::size_t index) {
            BaseNetwork base = BaseNetwork::from_gml(
                network_paths[index].string(), "id");
            slots[index].emplace(std::move(base));
        });
    if (const auto failure = first_error_index(errors)) {
        try {
            std::rethrow_exception(errors[*failure]);
        } catch (...) {
            std::throw_with_nested(VirtualNetworkSimulatorException(
                VirtualNetworkSimulatorErrorCode::io_failure,
                VirtualNetworkSimulatorOperation::load_networks,
                "failed to load virtual network",
                *failure,
                network_paths[*failure]));
        }
    }
    std::vector<VirtualNetwork> networks;
    networks.reserve(slots.size());
    for (auto& slot : slots) {
        networks.push_back(std::move(*slot));
    }

    if (networks.size() > std::numeric_limits<std::size_t>::max() / 2U ||
        networks.size() * 2U != events.size()) {
        throw VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::event_count_mismatch,
            VirtualNetworkSimulatorOperation::validate_dataset,
            "virtual network count must be half the event count",
            std::nullopt,
            directory);
    }

    VirtualNetworkRequestSimulator simulator = from_state(
        std::move(config), std::move(networks), std::move(events));
    cache.store(raw_directory, simulator);
    return simulator.clone();
}

VirtualNetworkRequestSimulator
VirtualNetworkRequestSimulator::load_dataset(
    const fs::path& directory,
    std::size_t workers) {
    return load_dataset(
        directory, global_virtual_network_dataset_cache(), workers);
}

std::optional<VirtualNetworkRequestSimulator>
VirtualNetworkDatasetCache::find(const std::string& raw_directory) const {
    std::shared_ptr<const VirtualNetworkRequestSimulator> value;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(raw_directory);
        if (found == entries_.end()) {
            return std::nullopt;
        }
        value = found->second;
    }
    return value->clone();
}

void VirtualNetworkDatasetCache::store(
    std::string raw_directory,
    const VirtualNetworkRequestSimulator& simulator) {
    auto value = std::make_shared<const VirtualNetworkRequestSimulator>(
        simulator.clone());
    const std::lock_guard<std::mutex> lock(mutex_);
    entries_.insert_or_assign(std::move(raw_directory), std::move(value));
}

std::size_t VirtualNetworkDatasetCache::size() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

VirtualNetworkDatasetCache& global_virtual_network_dataset_cache() {
    static VirtualNetworkDatasetCache cache;
    return cache;
}

}  // namespace virne::network
