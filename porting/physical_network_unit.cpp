#include "physical_network.h"

#include "io/graph_saver.h"
#include "random_context.h"
#include "setting.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using virne::network::PhysicalNetwork;
using virne::network::PhysicalNetworkBuildOptions;
using virne::network::PhysicalTopologyOrigin;
using virne::utils::SettingFormat;
namespace attribute = virne::network::attribute;
namespace network = virne::network;

void expect(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

virne::utils::SettingDocument yaml(const std::string& bytes)
{
    return virne::utils::parse_setting(bytes, SettingFormat::yaml);
}

std::string quote_yaml(const std::filesystem::path& path)
{
    std::string escaped;
    const std::string raw = path.generic_string();
    escaped.reserve(raw.size());
    for (const char value : raw) {
        escaped.push_back(value);
        if (value == '\'') {
            escaped.push_back('\'');
        }
    }
    return "'" + escaped + "'";
}

template <typename Callable>
void expect_any_error(Callable&& callable, const std::string& context)
{
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(context + ": expected exception");
}

template <typename Callable>
void expect_error_containing(
    Callable&& callable,
    const std::string_view fragment,
    const std::string& context)
{
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception& error) {
        expect(
            std::string_view(error.what()).find(fragment) !=
                std::string_view::npos,
            context + ": diagnostic drift: " + error.what());
        return;
    }
    throw std::runtime_error(context + ": expected exception");
}

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        base_ = std::filesystem::temp_directory_path();
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        for (std::uint32_t attempt = 0U; attempt < 256U; ++attempt) {
            const auto candidate =
                base_ /
                ("virne_physical_network_unit_" + std::to_string(stamp) +
                 "_" + std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                path_ = candidate;
                return;
            }
            if (error) {
                throw std::runtime_error(
                    "temporary directory creation failed: " +
                    error.message());
            }
        }
        throw std::runtime_error("unable to reserve a unique temporary root");
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    ~TemporaryDirectory()
    {
        if (!path_.empty() && path_.parent_path() == base_) {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path base_;
    std::filesystem::path path_;
};

std::vector<std::string> registry_names(
    const attribute::NodeAttributeRegistry& registry)
{
    std::vector<std::string> result;
    result.reserve(registry.size());
    for (const auto& entry : registry.entries()) {
        result.push_back(entry.name);
    }
    return result;
}

std::vector<std::string> registry_names(
    const attribute::LinkAttributeRegistry& registry)
{
    std::vector<std::string> result;
    result.reserve(registry.size());
    for (const auto& entry : registry.entries()) {
        result.push_back(entry.name);
    }
    return result;
}

std::vector<std::int64_t> integer_row(
    const std::vector<std::vector<AttrValue>>& rows,
    const std::string& context)
{
    expect(rows.size() == 1U, context + ": row-count drift");
    std::vector<std::int64_t> result;
    result.reserve(rows.front().size());
    for (const AttrValue& value : rows.front()) {
        expect(
            std::holds_alternative<std::int64_t>(value),
            context + ": non-integer value");
        result.push_back(std::get<std::int64_t>(value));
    }
    return result;
}

std::string generated_attribute_config()
{
    return
        "topology:\n"
        "  num_nodes: 5\n"
        "  type: path\n"
        "node_attrs_setting:\n"
        "  - name: cpu\n"
        "    owner: node\n"
        "    type: resource\n"
        "    generative: true\n"
        "    distribution: uniform\n"
        "    dtype: int\n"
        "    low: 1\n"
        "    high: 9\n"
        "link_attrs_setting:\n"
        "  - name: bandwidth\n"
        "    owner: link\n"
        "    type: resource\n"
        "    generative: true\n"
        "    distribution: uniform\n"
        "    dtype: int\n"
        "    low: 10\n"
        "    high: 19\n";
}

struct GenerationSnapshot
{
    std::vector<std::int64_t> cpu;
    std::vector<std::int64_t> bandwidth;
    std::uint32_t python_continuation = 0U;
    std::uint32_t numpy_continuation = 0U;
};

void expect_snapshot_equal(
    const GenerationSnapshot& actual,
    const GenerationSnapshot& expected,
    const std::string& context)
{
    expect(actual.cpu == expected.cpu, context + ": cpu drift");
    expect(
        actual.bandwidth == expected.bandwidth,
        context + ": bandwidth drift");
    expect(
        actual.python_continuation == expected.python_continuation,
        context + ": Python continuation drift");
    expect(
        actual.numpy_continuation == expected.numpy_continuation,
        context + ": NumPy continuation drift");
}

GenerationSnapshot capture_generation(
    const PhysicalNetwork& value,
    RandomContext& random,
    const std::size_t workers)
{
    const auto cpu = value.bind_node_attribute("cpu");
    const auto bandwidth = value.bind_link_attribute("bandwidth");
    expect(cpu.has_value(), "generated cpu binding");
    expect(bandwidth.has_value(), "generated bandwidth binding");

    GenerationSnapshot result;
    result.cpu = integer_row(
        network::get_node_attrs_data(value, {cpu->registry_id}, workers),
        "generated cpu");
    result.bandwidth = integer_row(
        network::get_link_attrs_data(
            value, {bandwidth->registry_id}, workers),
        "generated bandwidth");
    result.python_continuation = random.python().getrandbits32();
    result.numpy_continuation = random.numpy().next_uint32();
    return result;
}

GenerationSnapshot expected_path_generation(RandomContext& random)
{
    GenerationSnapshot result;
    result.cpu = random.numpy().randint(
        std::int64_t{1}, std::int64_t{10}, std::size_t{5});
    result.bandwidth = random.numpy().randint(
        std::int64_t{10}, std::int64_t{20}, std::size_t{4});
    result.python_continuation = random.python().getrandbits32();
    result.numpy_continuation = random.numpy().next_uint32();
    return result;
}

void test_workers_seed_and_continuation()
{
    const auto config = yaml(generated_attribute_config());
    const std::array<std::size_t, 4U> worker_counts{0U, 1U, 2U, 8U};
    std::optional<GenerationSnapshot> baseline;

    for (const std::size_t workers : worker_counts) {
        RandomContext random(999U);
        PhysicalNetworkBuildOptions options;
        options.seed = 77U;
        options.factory_workers = workers;
        options.attribute_workers = workers;
        PhysicalNetwork value =
            PhysicalNetwork::from_setting(config, random, options);
        expect(value.live_num_nodes() == 5U, "worker generated nodes");
        expect(value.live_num_links() == 4U, "worker generated links");
        expect(
            value.build_report().origin == PhysicalTopologyOrigin::generated,
            "worker generated origin");

        GenerationSnapshot current =
            capture_generation(value, random, workers);
        RandomContext expected_random(77U);
        const GenerationSnapshot expected =
            expected_path_generation(expected_random);
        expect_snapshot_equal(
            current, expected,
            "seeded workers=" + std::to_string(workers));
        if (!baseline) {
            baseline = current;
        } else {
            expect_snapshot_equal(
                current, *baseline,
                "worker invariance=" + std::to_string(workers));
        }
    }

    RandomContext continuation(31U);
    RandomContext expected_continuation(31U);
    static_cast<void>(continuation.python().random());
    static_cast<void>(continuation.numpy().random());
    static_cast<void>(expected_continuation.python().random());
    static_cast<void>(expected_continuation.numpy().random());
    PhysicalNetworkBuildOptions no_seed;
    no_seed.factory_workers = 2U;
    no_seed.attribute_workers = 8U;
    PhysicalNetwork continued =
        PhysicalNetwork::from_setting(config, continuation, no_seed);
    expect_snapshot_equal(
        capture_generation(continued, continuation, 8U),
        expected_path_generation(expected_continuation),
        "absent seed preserves continuation");

    PhysicalNetworkBuildOptions global_options;
    global_options.seed = 77U;
    global_options.factory_workers = 2U;
    global_options.attribute_workers = 2U;
    PhysicalNetwork global_value =
        PhysicalNetwork::from_setting(config, global_options);
    RandomContext& global = global_random_context();
    RandomContext expected_global(77U);
    expect_snapshot_equal(
        capture_generation(global_value, global, 2U),
        expected_path_generation(expected_global),
        "global-context overload");

    std::array<std::future<GenerationSnapshot>, 4U> futures;
    for (std::size_t index = 0U; index < worker_counts.size(); ++index) {
        futures[index] = std::async(
            std::launch::async,
            [config, workers = worker_counts[index]] {
                RandomContext random(1234U);
                PhysicalNetworkBuildOptions options;
                options.seed = 77U;
                options.factory_workers = workers;
                options.attribute_workers = workers;
                PhysicalNetwork value = PhysicalNetwork::from_setting(
                    config, random, options);
                return capture_generation(value, random, workers);
            });
    }
    for (std::size_t index = 0U; index < futures.size(); ++index) {
        expect_snapshot_equal(
            futures[index].get(), *baseline,
            "concurrent independent context=" + std::to_string(index));
    }
}

void test_generated_and_missing_num_nodes()
{
    RandomContext random(99U);
    PhysicalNetworkBuildOptions options;
    options.seed = 7U;
    options.factory_workers = 0U;
    options.attribute_workers = 2U;

    PhysicalNetwork network = PhysicalNetwork::from_setting(
        yaml(
            "topology:\n"
            "  num_nodes: 4\n"
            "  type: path\n"
            "node_attrs_setting: []\n"
            "link_attrs_setting: []\n"),
        random,
        options);
    expect(network.live_num_nodes() == 4U, "generated node count");
    expect(network.live_num_links() == 3U, "generated edge count");
    expect(
        network.build_report().origin == PhysicalTopologyOrigin::generated,
        "generated origin");
    expect(
        !network.build_report().requested_file.has_value(),
        "generated file report");

    bool missing = false;
    try {
        (void)PhysicalNetwork::from_setting(
            yaml(
                "topology:\n"
                "  type: path\n"
                "node_attrs_setting: []\n"
                "link_attrs_setting: []\n"),
            random,
            options);
    } catch (const std::invalid_argument& error) {
        missing = std::string(error.what()).find("'num_nodes'") !=
            std::string::npos;
    }
    expect(missing, "missing num_nodes precedence");
}

PhysicalNetwork generate_topology_fixture(
    const std::string& topology_yaml,
    const std::uint32_t seed = 19U)
{
    RandomContext random(999U);
    PhysicalNetworkBuildOptions options;
    options.seed = seed;
    return PhysicalNetwork::from_setting(
        yaml(
            "topology:\n" + topology_yaml +
            "node_attrs_setting: []\n"
            "link_attrs_setting: []\n"),
        random,
        options);
}

void test_topology_options_and_generation_failures()
{
    {
        PhysicalNetwork value = generate_topology_fixture(
            "  num_nodes: 5\n"
            "  type: path\n"
            "  ignored_native_boundary: 123\n");
        expect(
            value.live_num_nodes() == 5U && value.live_num_links() == 4U,
            "path topology and ignored key");
    }
    {
        PhysicalNetwork value = generate_topology_fixture(
            "  num_nodes: 5\n"
            "  type: star\n");
        expect(
            value.live_num_nodes() == 5U && value.live_num_links() == 4U,
            "star topology");
    }
    {
        PhysicalNetwork value = generate_topology_fixture(
            "  num_nodes: 1\n"
            "  type: grid_2d\n"
            "  m: 2\n"
            "  n: 3\n");
        expect(
            value.live_num_nodes() == 6U && value.live_num_links() == 7U,
            "grid topology options");
    }
    {
        PhysicalNetwork value = generate_topology_fixture(
            "  num_nodes: 4\n"
            "  type: random\n"
            "  random_prob: 1.0\n");
        expect(
            value.live_num_nodes() == 4U && value.live_num_links() == 6U,
            "random topology options");
    }
    {
        PhysicalNetwork value = generate_topology_fixture(
            "  num_nodes: 5\n"
            "  type: waxman\n"
            "  wm_alpha: 1.0\n"
            "  wm_beta: 1.0\n",
            23U);
        expect(value.live_num_nodes() == 5U, "waxman node count");
        expect(
            value.live_num_links() >= 4U && value.live_num_links() <= 10U,
            "waxman connected edge count");
    }
    {
        PhysicalNetwork value = generate_topology_fixture(
            "  num_nodes: 4\n",
            29U);
        expect(value.live_num_nodes() == 4U, "default waxman node count");
        expect(value.live_num_links() >= 3U, "default waxman connected");
    }
    {
        PhysicalNetwork value = generate_topology_fixture(
            "  file_path: ''\n"
            "  num_nodes: 3\n"
            "  type: path\n");
        expect(
            value.build_report().origin == PhysicalTopologyOrigin::generated,
            "false file path generates");
        expect(
            !value.build_report().requested_file.has_value(),
            "false file path omitted from report");
    }

    expect_any_error(
        [] {
            static_cast<void>(generate_topology_fixture(
                "  num_nodes: 4\n"
                "  type: grid_2d\n"
                "  n: 2\n"));
        },
        "missing grid dimension");
    expect_any_error(
        [] {
            static_cast<void>(generate_topology_fixture(
                "  num_nodes: 4\n"
                "  type: unsupported\n"));
        },
        "unsupported topology");
    expect_any_error(
        [] {
            RandomContext random(1U);
            static_cast<void>(PhysicalNetwork::from_setting(
                yaml("- not-an-object\n"), random,
                PhysicalNetworkBuildOptions{}));
        },
        "invalid setting root");
    expect_any_error(
        [] {
            RandomContext random(1U);
            static_cast<void>(PhysicalNetwork::from_setting(
                yaml(
                    "topology:\n"
                    "  num_nodes: 3\n"
                    "  type: path\n"
                    "node_attrs_setting:\n"
                    "  - name: invalid_bool_uniform\n"
                    "    owner: node\n"
                    "    type: resource\n"
                    "    generative: true\n"
                    "    distribution: uniform\n"
                    "    dtype: bool\n"
                    "    low: false\n"
                    "    high: true\n"
                    "link_attrs_setting: []\n"),
                random,
                PhysicalNetworkBuildOptions{}));
        },
        "attribute generation failure");
}

void test_gml_success_and_status_registration(
    const std::filesystem::path& root)
{
    const std::filesystem::path gml_path = root / "input.gml";
    Graph graph;
    const auto edge = graph.add_edge(0U, 1U);
    const AttrId loaded_node_id = graph.attr_id("loaded_node");
    const AttrId later_only_id = graph.attr_id("later_only");
    const AttrId loaded_link_id = graph.attr_id("loaded_link");
    const AttrId topology_id = graph.attr_id("topology");
    graph.node_attrs(0U).set(loaded_node_id, std::int64_t{17});
    graph.node_attrs(1U).set(later_only_id, std::int64_t{99});
    graph.edge_attrs(edge).set(loaded_link_id, 3.5);
    graph.graph_attrs().set(topology_id, std::string("loaded-topology"));
    GraphSaver::save_gml(graph, gml_path.string());

    const std::string config =
        "topology:\n"
        "  file_path: " + quote_yaml(gml_path) + "\n"
        "  type: path\n"
        "node_attrs_setting: []\n"
        "link_attrs_setting: []\n";

    RandomContext random(5U);
    PhysicalNetwork network = PhysicalNetwork::from_setting(
        yaml(config), random, PhysicalNetworkBuildOptions{});
    expect(
        network.build_report().origin == PhysicalTopologyOrigin::loaded_gml,
        "loaded origin");
    expect(
        network.build_report().requested_file ==
            std::optional<std::string>{gml_path.generic_string()},
        "loaded requested file");
    expect(
        !network.build_report().gml_error.has_value(),
        "successful load has no GML error");
    expect(network.live_num_nodes() == 2U, "loaded node count");
    expect(network.live_num_links() == 1U, "loaded edge count");
    expect(
        network.node_attributes().size() == 1U,
        "sample node status count=" +
            std::to_string(network.node_attributes().size()));
    expect(
        network.link_attributes().size() == 1U,
        "sample link status count=" +
            std::to_string(network.link_attributes().size()));
    expect(
        network.node_attributes().entries().front().name == "loaded_node",
        "sample node status order");
    expect(
        network.link_attributes().entries().front().name == "loaded_link",
        "sample link status order");
    expect(
        !network.node_attributes().bind("later_only").has_value(),
        "only first node is sampled");
    expect(
        !network.node_attributes().bind("id").has_value(),
        "structural GML id is consumed, not a status attribute");

    const auto node_binding = network.bind_node_attribute("loaded_node");
    const auto link_binding = network.bind_link_attribute("loaded_link");
    expect(node_binding.has_value(), "loaded node binding");
    expect(link_binding.has_value(), "loaded link binding");
    expect(
        std::get<std::int64_t>(
            network.graph().node_attrs(0U).at(node_binding->value_id)) == 17,
        "loaded node value");
    const auto edges = network.graph().edges();
    expect(edges.first != edges.second, "loaded edge exists");
    expect(
        std::get<double>(network.graph().edge_attrs(*edges.first).at(
            link_binding->value_id)) == 3.5,
        "loaded link value");
    expect(
        network.graph_attributes().find("node_attrs_setting") == nullptr,
        "loaded public node setting removed");
    expect(
        network.graph_attributes().find("link_attrs_setting") == nullptr,
        "loaded public link setting removed");
    expect(
        std::get<std::string>(*network.graph_attributes().find("topology")) ==
            "loaded-topology",
        "loaded graph metadata precedence");

    PhysicalNetwork cloned = network.clone();
    expect(
        cloned.build_report().origin == PhysicalTopologyOrigin::loaded_gml,
        "clone report");
    cloned.graph().node_attrs(0U).set(node_binding->value_id, std::int64_t{41});
    expect(
        std::get<std::int64_t>(
            network.graph().node_attrs(0U).at(node_binding->value_id)) == 17,
        "clone graph independence");

    const auto cloned_binding = cloned.bind_node_attribute("loaded_node");
    expect(cloned_binding.has_value(), "clone binding");
    expect(
        cloned_binding->registry_identity == &cloned.node_attributes() &&
            cloned_binding->graph_identity == &cloned.graph(),
        "clone binding identities");
    PhysicalNetwork moved(std::move(cloned));
    expect(
        moved.build_report().origin == PhysicalTopologyOrigin::loaded_gml,
        "move construction report");
    const auto moved_binding = moved.bind_node_attribute("loaded_node");
    expect(
        moved_binding.has_value() &&
            moved_binding->registry_identity == &moved.node_attributes() &&
            moved_binding->graph_identity == &moved.graph(),
        "move construction bindings");
    PhysicalNetwork assigned;
    assigned = std::move(moved);
    const auto assigned_binding = assigned.bind_node_attribute("loaded_node");
    expect(
        assigned_binding.has_value() &&
            assigned_binding->registry_identity ==
                &assigned.node_attributes() &&
            assigned_binding->graph_identity == &assigned.graph(),
        "move assignment bindings");
    expect(
        assigned.build_report().requested_file ==
            std::optional<std::string>{gml_path.generic_string()},
        "move assignment report");
}

void test_configured_status_order_and_metadata(
    const std::filesystem::path& root)
{
    const std::filesystem::path gml_path = root / "ordered-status.gml";
    Graph graph;
    const auto first_edge = graph.add_edge(0U, 1U);
    const auto second_edge = graph.add_edge(1U, 2U);

    const AttrId node_unknown_a = graph.attr_id("node_unknown_a");
    const AttrId configured_node = graph.attr_id("configured_node");
    const AttrId node_unknown_b = graph.attr_id("node_unknown_b");
    const AttrId node_later = graph.attr_id("node_later");
    const AttrId link_unknown_a = graph.attr_id("link_unknown_a");
    const AttrId configured_link = graph.attr_id("configured_link");
    const AttrId link_unknown_b = graph.attr_id("link_unknown_b");
    const AttrId link_later = graph.attr_id("link_later");
    const AttrId topology = graph.attr_id("topology");
    const AttrId loaded_metadata = graph.attr_id("loaded_metadata");
    const AttrId node_settings = graph.attr_id("node_attrs_setting");
    const AttrId link_settings = graph.attr_id("link_attrs_setting");

    graph.node_attrs(0U).set(node_unknown_a, std::int64_t{11});
    graph.node_attrs(0U).set(configured_node, std::int64_t{12});
    graph.node_attrs(0U).set(node_unknown_b, std::int64_t{13});
    graph.node_attrs(1U).set(node_later, std::int64_t{14});
    graph.edge_attrs(first_edge).set(link_unknown_a, 21.0);
    graph.edge_attrs(first_edge).set(configured_link, 22.0);
    graph.edge_attrs(first_edge).set(link_unknown_b, 23.0);
    graph.edge_attrs(second_edge).set(link_later, 24.0);
    graph.graph_attrs().set(topology, std::string{"loaded-topology"});
    graph.graph_attrs().set(loaded_metadata, std::int64_t{9});
    graph.graph_attrs().set(node_settings, std::string{"loaded-node-snapshot"});
    graph.graph_attrs().set(link_settings, std::string{"loaded-link-snapshot"});
    GraphSaver::save_gml(graph, gml_path.string());

    const std::string config =
        "topology:\n"
        "  file_path: " + quote_yaml(gml_path) + "\n"
        "  type: path\n"
        "node_attrs_setting:\n"
        "  - name: configured_node\n"
        "    owner: node\n"
        "    type: status\n"
        "link_attrs_setting:\n"
        "  - name: configured_link\n"
        "    owner: link\n"
        "    type: status\n"
        "graph_attrs_setting:\n"
        "  topology: config-topology\n"
        "  loaded_metadata: 1\n"
        "  config_only: 2\n";
    RandomContext random(71U);
    PhysicalNetwork value = PhysicalNetwork::from_setting(
        yaml(config), random, PhysicalNetworkBuildOptions{});

    expect(
        registry_names(value.node_attributes()) ==
            std::vector<std::string>(
                {"configured_node", "node_unknown_a", "node_unknown_b"}),
        "configured then first-sample node status order");
    expect(
        registry_names(value.link_attributes()) ==
            std::vector<std::string>(
                {"configured_link", "link_unknown_a", "link_unknown_b"}),
        "configured then first-sample link status order");
    expect(
        !value.bind_node_attribute("node_later").has_value(),
        "later node is not sampled");
    expect(
        !value.bind_link_attribute("link_later").has_value(),
        "later edge is not sampled");
    for (const auto& entry : value.node_attributes().entries()) {
        expect(
            entry.attribute->spec().kind == attribute::AttributeKind::status,
            "auto node attribute kind");
    }
    for (const auto& entry : value.link_attributes().entries()) {
        expect(
            entry.attribute->spec().kind == attribute::AttributeKind::status,
            "auto link attribute kind");
    }
    expect(
        value.graph_attributes().find("node_attrs_setting") == nullptr &&
            value.graph_attributes().find("link_attrs_setting") == nullptr,
        "loaded public snapshots erased");
    const AttrValue* actual_topology =
        value.graph_attributes().find("topology");
    const AttrValue* actual_loaded =
        value.graph_attributes().find("loaded_metadata");
    const AttrValue* actual_config =
        value.graph_attributes().find("config_only");
    expect(
        actual_topology != nullptr && actual_loaded != nullptr &&
            actual_config != nullptr,
        "loaded/config metadata presence");
    expect(
        std::get<std::string>(*actual_topology) == "loaded-topology" &&
            std::get<std::int64_t>(*actual_loaded) == 9 &&
            std::get<std::int64_t>(*actual_config) == 2,
        "loaded metadata precedence and config retention");
}

void test_fallback_and_dataset(const std::filesystem::path& root)
{
    const std::filesystem::path broken_path = root / "broken.gml";
    {
        std::ofstream broken(broken_path);
        broken << "not valid gml";
    }
    const std::string config =
        "topology:\n"
        "  file_path: " + quote_yaml(broken_path) + "\n"
        "  num_nodes: 3\n"
        "  type: path\n"
        "node_attrs_setting: []\n"
        "link_attrs_setting: []\n";

    RandomContext random(11U);
    PhysicalNetwork fallback = PhysicalNetwork::from_setting(
        yaml(config), random, PhysicalNetworkBuildOptions{});
    expect(
        fallback.build_report().origin ==
            PhysicalTopologyOrigin::generated_after_gml_error,
        "fallback origin");
    expect(fallback.build_report().gml_error.has_value(), "fallback error");
    expect(
        fallback.build_report().requested_file ==
            std::optional<std::string>{broken_path.generic_string()},
        "fallback requested file");
    expect(fallback.live_num_nodes() == 3U, "fallback node count");
    expect(fallback.live_num_links() == 2U, "fallback edge count");

    const std::filesystem::path missing_path = root / "missing.gml";
    const std::string missing_config =
        "topology:\n"
        "  file_path: " + quote_yaml(missing_path) + "\n"
        "  num_nodes: 4\n"
        "  type: path\n"
        "node_attrs_setting: []\n"
        "link_attrs_setting: []\n";
    PhysicalNetwork missing_file = PhysicalNetwork::from_setting(
        yaml(missing_config), random, PhysicalNetworkBuildOptions{});
    expect(
        missing_file.build_report().origin ==
            PhysicalTopologyOrigin::generated,
        "missing file generates without load-error origin");
    expect(
        missing_file.build_report().requested_file ==
            std::optional<std::string>{missing_path.generic_string()} &&
            !missing_file.build_report().gml_error.has_value(),
        "missing file report");
    expect(
        missing_file.live_num_nodes() == 4U &&
            missing_file.live_num_links() == 3U,
        "missing file generated topology");

    const std::string broken_missing_num =
        "topology:\n"
        "  file_path: " + quote_yaml(broken_path) + "\n"
        "  type: path\n"
        "node_attrs_setting: []\n"
        "link_attrs_setting: []\n";
    expect_error_containing(
        [&] {
            static_cast<void>(PhysicalNetwork::from_setting(
                yaml(broken_missing_num), random,
                PhysicalNetworkBuildOptions{}));
        },
        "num_nodes",
        "fallback generation error replaces malformed GML error");

    const std::filesystem::path dataset_dir = root / "nested" / "dataset";
    fallback.save_dataset(dataset_dir.string());
    expect(
        std::filesystem::exists(dataset_dir / "p_net.gml"),
        "dataset saved");
    PhysicalNetwork loaded = PhysicalNetwork::load_dataset(dataset_dir.string());
    expect(loaded.live_num_nodes() == 3U, "dataset loaded nodes");
    expect(loaded.live_num_links() == 2U, "dataset loaded edges");

    const std::filesystem::path direct_path = root / "direct.gml";
    fallback.to_gml(direct_path.string());
    expect(std::filesystem::exists(direct_path), "direct GML saved");
    PhysicalNetwork direct = PhysicalNetwork::load_dataset(
        root.string(), direct_path.filename().string());
    expect(
        direct.live_num_nodes() == fallback.live_num_nodes() &&
            direct.live_num_links() == fallback.live_num_links(),
        "direct GML round trip");

    const std::filesystem::path custom_dir = root / "custom" / "nested";
    fallback.save_dataset(custom_dir.string(), "physical-custom.gml");
    expect(
        std::filesystem::exists(custom_dir / "physical-custom.gml"),
        "custom dataset file saved");
    PhysicalNetwork custom = PhysicalNetwork::load_dataset(
        custom_dir.string(), "physical-custom.gml");
    expect(
        custom.live_num_nodes() == 3U && custom.live_num_links() == 2U,
        "custom dataset loaded");

    bool missing = false;
    try {
        (void)PhysicalNetwork::load_dataset(
            dataset_dir.string(), "absent.gml");
    } catch (const std::runtime_error& error) {
        missing = std::string(error.what()).find("absent.gml") !=
            std::string::npos;
    }
    expect(missing, "missing dataset error");

    const std::filesystem::path empty_path = root / "empty.gml";
    GraphSaver::save_gml(Graph{}, empty_path.string());
    const std::string empty_config =
        "topology:\n"
        "  file_path: " + quote_yaml(empty_path) + "\n"
        "  num_nodes: 2\n"
        "  type: path\n"
        "node_attrs_setting: []\n"
        "link_attrs_setting: []\n";
    PhysicalNetwork empty_loaded = PhysicalNetwork::from_setting(
        yaml(empty_config), random, PhysicalNetworkBuildOptions{});
    expect(
        empty_loaded.build_report().origin ==
                PhysicalTopologyOrigin::loaded_gml &&
            !empty_loaded.build_report().gml_error.has_value(),
        "empty loaded graph succeeds without sample discovery");
    expect(
        empty_loaded.live_num_nodes() == 0U &&
            empty_loaded.live_num_links() == 0U,
        "empty loaded graph bypasses configured generation");
}

}  // namespace

int main()
{
    try {
        TemporaryDirectory temporary;
        const std::filesystem::path& root = temporary.path();

        test_workers_seed_and_continuation();
        test_generated_and_missing_num_nodes();
        test_topology_options_and_generation_failures();
        test_gml_success_and_status_registration(root);
        test_configured_status_order_and_metadata(root);
        test_fallback_and_dataset(root);

        std::cout << "PhysicalNetwork unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysicalNetwork unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
