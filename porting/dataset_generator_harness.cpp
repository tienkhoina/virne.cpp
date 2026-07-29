#include "dataset_generator.h"

#include "config.h"
#include "random_context.h"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace network = virne::network;

std::string hex_text(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

void emit(const std::string_view name, const std::string& payload) {
    std::cout << "case=" << name << "|ok|" << hex_text(payload) << '\n';
}

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream output;
    output << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return output.str();
}

std::string attr_scalar(const AttrValue& value) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return "i" + std::to_string(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return double_token(*item);
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item ? "b1" : "b0";
    }
    if (const auto* item = std::get_if<std::string>(&value)) {
        return "s" + *item;
    }
    throw std::runtime_error("Generator harness found recursive attribute");
}

Config make_config(
    const std::optional<std::uint32_t> root_seed,
    const std::size_t request_count = 8U) {
    std::ostringstream yaml;
    if (root_seed) {
        yaml << "seed: " << *root_seed << '\n';
    }
    yaml
        << "experiment:\n"
        << "  seed: 23\n"
        << "p_net_setting:\n"
        << "  topology:\n"
        << "    type: path\n"
        << "    num_nodes: 4\n"
        << "    wm_alpha: 0.5\n"
        << "    wm_beta: 0.2\n"
        << "  node_attrs_setting: []\n"
        << "  link_attrs_setting: []\n"
        << "  output:\n"
        << "    save_dir: unused-physical\n"
        << "v_sim_setting:\n"
        << "  num_v_nets: " << request_count << '\n'
        << "  topology:\n"
        << "    type: path\n"
        << "  v_net_size:\n"
        << "    distribution: uniform\n"
        << "    dtype: int\n"
        << "    low: 2\n"
        << "    high: 3\n"
        << "  arrival_rate:\n"
        << "    distribution: poisson\n"
        << "    dtype: float\n"
        << "    lam: 1.0\n"
        << "  lifetime:\n"
        << "    distribution: exponential\n"
        << "    dtype: float\n"
        << "    scale: 2\n"
        << "  node_attrs_setting:\n"
        << "    - name: cpu\n"
        << "      type: resource\n"
        << "      owner: node\n"
        << "      distribution: uniform\n"
        << "      dtype: int\n"
        << "      generative: true\n"
        << "      low: 0\n"
        << "      high: 3\n"
        << "  link_attrs_setting:\n"
        << "    - name: bandwidth\n"
        << "      type: resource\n"
        << "      owner: link\n"
        << "      distribution: uniform\n"
        << "      dtype: int\n"
        << "      generative: true\n"
        << "      low: 0\n"
        << "      high: 5\n"
        << "  output:\n"
        << "    save_dir: unused-virtual\n"
        << "    events_file_name: events.yaml\n"
        << "    setting_file_name: v_sim_setting.yaml\n";
    return Config(YAML::Load(yaml.str()));
}

network::GeneratorWorkers all_workers(const std::size_t width) {
    network::GeneratorWorkers result;
    result.physical_factory_workers = width;
    result.physical_attribute_workers = width;
    result.virtual_simulation.factory_workers = width;
    result.virtual_simulation.arrangement_workers = width;
    result.virtual_simulation.attribute_workers = width;
    result.virtual_simulation.event_workers = width;
    result.virtual_simulation.io_workers = width;
    return result;
}

network::GeneratorWorkers benchmark_workers(const std::size_t width) {
    // Per-request graphs in this fixture have only 2-3 nodes. Keep their
    // factory/attribute lanes explicitly sequential and apply the measured
    // width only to the two large contiguous simulator lanes.
    network::GeneratorWorkers result;
    result.virtual_simulation.arrangement_workers = width;
    result.virtual_simulation.event_workers = width;
    return result;
}

std::string values_payload(
    const network::VirtualNetwork& request,
    const bool node) {
    std::vector<std::vector<AttrValue>> rows;
    if (node) {
        const auto binding = request.bind_node_attribute("cpu");
        if (!binding) {
            return "none";
        }
        rows = network::get_node_attrs_data(
            request, {binding->registry_id}, 1U);
    } else {
        const auto binding = request.bind_link_attribute("bandwidth");
        if (!binding) {
            return "none";
        }
        rows = network::get_link_attrs_data(
            request, {binding->registry_id}, 1U);
    }
    std::string result = "[";
    for (std::size_t index = 0U; index < rows.front().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += attr_scalar(rows.front()[index]);
    }
    result.push_back(']');
    return result;
}

std::string simulator_payload(
    const network::VirtualNetworkRequestSimulator& simulator) {
    std::string result = "V=" + std::to_string(simulator.num_v_nets()) +
        ":" + std::to_string(simulator.num_events()) + ";nets=[";
    for (std::size_t index = 0U; index < simulator.v_nets().size(); ++index) {
        if (index != 0U) {
            result.push_back('/');
        }
        const network::VirtualNetwork& request = simulator.v_nets()[index];
        result += "id=" + std::to_string(request.request_id().value_or(-1));
        result += ",a=" + double_token(request.arrival_time().value_or(-1.0));
        result += ",l=" + double_token(request.lifetime().value_or(-1.0));
        result += ",n=" + std::to_string(request.graph().num_nodes());
        result += ",e=" + std::to_string(request.graph().num_edges());
        result += ",cpu=" + values_payload(request, true);
        result += ",bw=" + values_payload(request, false);
    }
    result += "];events=[";
    for (std::size_t index = 0U; index < simulator.events().size(); ++index) {
        if (index != 0U) {
            result.push_back('/');
        }
        const network::VirtualNetworkEvent& event = simulator.events()[index];
        result += std::to_string(event.id()) + ":" +
            std::to_string(static_cast<unsigned>(event.type())) + ":" +
            std::to_string(event.virtual_network_id()) + ":" +
            double_token(event.time());
    }
    result.push_back(']');
    return result;
}

std::string generated_payload(
    const network::GeneratedDataset& generated,
    RandomContext& random) {
    std::string result = "P=";
    if (generated.physical_network) {
        result += std::to_string(
            generated.physical_network->graph().num_nodes()) + ":" +
            std::to_string(generated.physical_network->graph().num_edges());
    } else {
        result += "none";
    }
    result.push_back(';');
    result += generated.virtual_networks
        ? simulator_payload(*generated.virtual_networks)
        : "V=none";
    result += ";next_py=" + std::to_string(random.python().getrandbits32());
    result += ";next_np=" + std::to_string(random.numpy().next_uint32());
    return result;
}

network::GeneratorSelection selection(
    const bool physical,
    const bool virtuals) {
    network::GeneratorSelection result;
    result.physical_network = physical;
    result.virtual_networks = virtuals;
    return result;
}

void emit_selection_cases() {
    struct Case {
        std::string_view name;
        bool physical;
        bool virtuals;
    };
    for (const Case current : {
             Case{"selection_none", false, false},
             Case{"selection_physical", true, false},
             Case{"selection_virtual", false, true},
             Case{"selection_both", true, true},
         }) {
        const Config config = make_config(41U);
        RandomContext random(777U);
        const auto generated = network::Generator::generate_dataset(
            config,
            random,
            selection(current.physical, current.virtuals),
            all_workers(1U));
        emit(current.name, generated_payload(generated, random));
    }

    for (const std::size_t width : {0U, 2U, 8U}) {
        const Config config = make_config(41U);
        RandomContext random(777U);
        const auto generated = network::Generator::generate_dataset(
            config, random, selection(true, true), all_workers(width));
        emit(
            "selection_both_w" + std::to_string(width),
            generated_payload(generated, random));
    }
}

void emit_seed_cases() {
    {
        const Config config = make_config(std::nullopt);
        RandomContext random(83U);
        const auto generated = network::Generator::generate_dataset(
            config, random, selection(true, true), all_workers(2U));
        emit("absent_seed_both", generated_payload(generated, random));
    }
    {
        const Config config = make_config(17U);
        RandomContext random(0U);
        auto simulator =
            network::Generator::generate_v_nets_dataset_from_config(
                config,
                random,
                network::GeneratorPersistence::memory_only,
                all_workers(2U),
                network::GeneratorSeedMode::composed_experiment_seed);
        network::GeneratedDataset generated;
        generated.virtual_networks.emplace(std::move(simulator));
        emit("composed_seed_virtual", generated_payload(generated, random));
    }
}

void emit_changeable_cases() {
    for (const std::size_t width : {0U, 1U, 2U, 8U}) {
        const Config config = make_config(53U);
        RandomContext random(0U);
        auto simulator =
            network::Generator::generate_changeable_v_nets_dataset_from_config(
                config,
                random,
                network::GeneratorPersistence::memory_only,
                all_workers(width));
        network::GeneratedDataset generated;
        generated.virtual_networks.emplace(std::move(simulator));
        emit(
            "changeable_w" + std::to_string(width),
            generated_payload(generated, random));
    }
}

void emit_error_cases() {
    {
        Config config(YAML::Load("seed: 99\n"));
        RandomContext random(72U);
        bool failed = false;
        try {
            static_cast<void>(
                network::Generator::generate_v_nets_dataset_from_config(
                    config, random));
        } catch (const std::exception&) {
            failed = true;
        }
        emit(
            "missing_subtree",
            "error=" + std::to_string(failed ? 1 : 0) +
                ";next_py=" + std::to_string(random.python().getrandbits32()) +
                ";next_np=" + std::to_string(random.numpy().next_uint32()));
    }
    {
        const Config config = make_config(91U, 6U);
        RandomContext random(0U);
        bool failed = false;
        try {
            static_cast<void>(
                network::Generator::generate_changeable_v_nets_dataset_from_config(
                    config, random));
        } catch (const std::exception&) {
            failed = true;
        }
        emit(
            "invalid_changeable_count",
            "error=" + std::to_string(failed ? 1 : 0) +
                ";next_py=" + std::to_string(random.python().getrandbits32()) +
                ";next_np=" + std::to_string(random.numpy().next_uint32()));
    }
}

std::uint64_t fnv1a(const std::string_view value) {
    std::uint64_t checksum = 14695981039346656037ULL;
    for (const char raw_byte : value) {
        checksum =
            (checksum ^ static_cast<unsigned char>(raw_byte)) *
            1099511628211ULL;
    }
    return checksum;
}

void emit_benchmark_result(
    const std::string_view kind,
    const std::size_t count,
    const std::size_t workers,
    const std::int64_t elapsed_ns,
    const std::string& payload) {
    std::cout
        << "protocol=1\n"
        << "kind=dataset_generator_" << kind << '\n'
        << "count=" << count << '\n'
        << "workers=" << workers << '\n'
        << "elapsed_ns=" << elapsed_ns << '\n'
        << "checksum=" << fnv1a(payload) << '\n'
        << "output_bytes=" << payload.size() << '\n'
        << "entry_count=" << count << '\n'
        << "type_tag=generator_text_v1\n"
        << "status=PASS\n";
}

void run_benchmark(
    const std::string_view kind,
    const std::size_t count,
    const std::size_t workers) {
    const Config config = make_config(67U, count);
    RandomContext random(0U);
    if (kind == "ordinary") {
        const auto begin = std::chrono::steady_clock::now();
        auto generated = network::Generator::generate_dataset(
            config, random, selection(true, true), benchmark_workers(workers));
        const auto end = std::chrono::steady_clock::now();
        const std::string payload = generated_payload(generated, random);
        emit_benchmark_result(
            kind,
            count,
            workers,
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count(),
            payload);
        return;
    }
    if (kind == "changeable") {
        const auto begin = std::chrono::steady_clock::now();
        auto simulator =
            network::Generator::generate_changeable_v_nets_dataset_from_config(
                config,
                random,
                network::GeneratorPersistence::memory_only,
                benchmark_workers(workers));
        const auto end = std::chrono::steady_clock::now();
        network::GeneratedDataset generated;
        generated.virtual_networks.emplace(std::move(simulator));
        const std::string payload = generated_payload(generated, random);
        emit_benchmark_result(
            kind,
            count,
            workers,
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count(),
            payload);
        return;
    }
    throw std::invalid_argument("benchmark kind must be ordinary or changeable");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string_view(argv[1]) == "benchmark") {
            const auto count = static_cast<std::size_t>(std::stoull(argv[3]));
            const auto workers = static_cast<std::size_t>(std::stoull(argv[4]));
            if (count == 0U) {
                throw std::invalid_argument("benchmark count must be positive");
            }
            run_benchmark(argv[2], count, workers);
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "usage: dataset_generator_harness [benchmark "
                "<ordinary|changeable> <count> <workers>]");
        }
        emit_selection_cases();
        emit_seed_cases();
        emit_changeable_cases();
        emit_error_cases();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dataset Generator harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
