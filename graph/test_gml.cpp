#include "generators/gml_loader.h"
#include "io/graph_saver.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <variant>

namespace
{

[[noreturn]] void fail(
    const char* expression,
    int line)
{
    throw std::runtime_error(
        std::string("GML check failed at line ") +
        std::to_string(line) + ": " + expression);
}

#define CHECK(expression) \
    do { if (!(expression)) fail(#expression, __LINE__); } while (false)

std::filesystem::path temporary_path(
    const char* stem)
{
    const auto nonce =
        std::chrono::steady_clock::now()
            .time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
        (std::string(stem) + "-" +
         std::to_string(nonce) + ".gml");
}

std::filesystem::path write_temporary_gml(
    const char* stem,
    const std::string& contents)
{
    const std::filesystem::path path = temporary_path(stem);
    std::ofstream output(path, std::ios::binary);
    output << contents;
    output.close();
    return path;
}

void check_structured_roundtrip()
{
    Graph graph;
    const Vertex first = graph.add_node();
    const Vertex second = graph.add_node();
    graph.node_attrs(first)["id"] = int64_t{10};
    graph.node_attrs(second)["id"] = int64_t{20};
    graph.node_attrs(first)["label"] = std::string("n\"0 & caf\xC3\xA9");
    graph.node_attrs(first)["pos"] = make_attr_list({1.25, 2.5});
    const Edge edge = graph.add_edge(first, second);
    graph.edge_attrs(edge)["note"] =
        std::string("comma, quote \" and\nnewline");

    graph.graph_attrs()["title"] =
        std::string("Virne \"GML\" & metadata\nline 2");
    graph.graph_attrs()["topology"] = make_attr_object({
        {"type", std::string("waxman")},
        {"wm_alpha", 0.5},
        {"wm_beta", 0.2},
    });
    graph.graph_attrs()["node_attrs_setting"] = make_attr_list({
        make_attr_object({
            {"name", std::string("cpu")},
            {"type", std::string("resource")},
            {"low", int64_t{50}},
        }),
        make_attr_object({
            {"name", std::string("max_cpu")},
            {"type", std::string("extrema")},
        }),
    });
    graph.graph_attrs()["empty_list"] = make_attr_list();
    graph.graph_attrs()["single_list"] = make_attr_list({1.0});
    graph.graph_attrs()["real_one"] = 1.0;
    graph.graph_attrs()["small_real"] = 1e-10;
    graph.graph_attrs()["large_real"] = 1e20;
    graph.graph_attrs()["negative_zero"] = -0.0;
    graph.graph_attrs()["literal_backslashes"] =
        std::string(R"(a\tb\nc)");

    const std::filesystem::path path =
        temporary_path("virne-structured");
    GraphSaver::save_gml(graph, path.string());
    Graph loaded = nx::read_gml(path.string());
    Graph loaded_with_runtime_signature =
        nx::read_gml(path.string(), "id");
    CHECK(loaded_with_runtime_signature.num_nodes() == 2);
    bool rejected_label = false;
    try
    {
        static_cast<void>(
            nx::read_gml(path.string(), "label"));
    }
    catch (const std::invalid_argument&)
    {
        rejected_label = true;
    }
    CHECK(rejected_label);
    std::filesystem::remove(path);

    CHECK(loaded.num_nodes() == 2);
    CHECK(loaded.num_edges() == 1);
    CHECK(std::get<int64_t>(loaded.node_attrs(0).at("id")) == 10);
    CHECK(std::get<int64_t>(loaded.node_attrs(1).at("id")) == 20);
    CHECK(std::get<std::string>(loaded.node_attrs(0).at("label")) ==
          "n\"0 & caf\xC3\xA9");
    const AttrList* position = attr_list(loaded.node_attrs(0).at("pos"));
    CHECK(position != nullptr);
    CHECK(position->values.size() == 2);
    CHECK(std::get<double>(position->values[0]) == 1.25);
    CHECK(std::get<double>(position->values[1]) == 2.5);
    CHECK(attr_value_equal(
        graph.graph_attrs().at("title"),
        loaded.graph_attrs().at("title")));
    CHECK(attr_value_equal(
        graph.graph_attrs().at("topology"),
        loaded.graph_attrs().at("topology")));
    CHECK(attr_value_equal(
        graph.graph_attrs().at("node_attrs_setting"),
        loaded.graph_attrs().at("node_attrs_setting")));
    CHECK(attr_list(loaded.graph_attrs().at("empty_list")) != nullptr);
    CHECK(attr_list(loaded.graph_attrs().at("empty_list"))->values.empty());
    const AttrList* singleton =
        attr_list(loaded.graph_attrs().at("single_list"));
    CHECK(singleton != nullptr);
    CHECK(singleton->values.size() == 1);
    CHECK(std::holds_alternative<double>(singleton->values.front()));
    CHECK(std::get<double>(singleton->values.front()) == 1.0);
    CHECK(std::holds_alternative<double>(
        loaded.graph_attrs().at("real_one")));
    CHECK(std::get<double>(loaded.graph_attrs().at("real_one")) == 1.0);
    CHECK(std::get<double>(loaded.graph_attrs().at("small_real")) == 1e-10);
    CHECK(std::get<double>(loaded.graph_attrs().at("large_real")) == 1e20);
    CHECK(std::signbit(
        std::get<double>(loaded.graph_attrs().at("negative_zero"))));
    CHECK(std::get<std::string>(
        loaded.graph_attrs().at("literal_backslashes")) ==
        R"(a\tb\nc)");
    CHECK(std::get<std::string>(
        loaded.edge_attrs(loaded.edge(0, 1)).at("note")) ==
        "comma, quote \" and\nnewline");
}

void check_copy_value_semantics()
{
    Graph original;
    original.add_node();
    original.graph_attrs()["metadata"] = make_attr_object({
        {"values", make_attr_list({int64_t{1}, int64_t{2}})},
    });
    original.node_attrs(0)["position"] =
        make_attr_list({0.25, 0.75});

    Graph copy = original;
    AttrObject* metadata =
        attr_object(*copy.graph_attrs().find("metadata"));
    CHECK(metadata != nullptr);
    AttrList* values = attr_list(*metadata->find("values"));
    CHECK(values != nullptr);
    values->values[0] = int64_t{99};
    attr_list(*copy.node_attrs(0).find("position"))->values[0] = 9.0;
    copy.graph_attrs()["copy_only"] = int64_t{1};

    const AttrObject* original_metadata =
        attr_object(original.graph_attrs().at("metadata"));
    CHECK(std::get<int64_t>(
        attr_list(*original_metadata->find("values"))->values[0]) == 1);
    CHECK(std::get<double>(
        attr_list(original.node_attrs(0).at("position"))->values[0]) ==
        0.25);
    CHECK(!original.attribute_registry().find("copy_only").has_value());

    DiGraph directed;
    directed = original.to_directed();
    attr_list(*directed.node_attrs(0).find("position"))->values[1] = 8.0;
    CHECK(std::get<double>(
        attr_list(original.node_attrs(0).at("position"))->values[1]) ==
        0.75);
}

void check_networkx_named_entities()
{
    const std::filesystem::path path = write_temporary_gml(
        "virne-named-entities",
        "graph [\n  title \"&nbsp;&copy;&apos;\"\n]\n");
    const Graph graph = GmlLoader::load(path.string());
    std::filesystem::remove(path);
    CHECK(std::get<std::string>(graph.graph_attrs().at("title")) ==
          "\xC2\xA0\xC2\xA9&apos;");

    const std::filesystem::path invalid_numeric = write_temporary_gml(
        "virne-invalid-numeric-entities",
        "graph [\n  title \"&#X41;|&#+65;|&#x+41;|&#65;\"\n]\n");
    const Graph numeric = GmlLoader::load(invalid_numeric.string());
    std::filesystem::remove(invalid_numeric);
    CHECK(std::get<std::string>(numeric.graph_attrs().at("title")) ==
          "&#X41;|&#+65;|&#x+41;|A");
}

void check_rejected_gml_cases()
{
    const std::filesystem::path invalid_input = write_temporary_gml(
        "virne-invalid-input-key",
        "graph [\n  not-valid 1\n]\n");
    bool rejected = false;
    try
    {
        static_cast<void>(GmlLoader::load(invalid_input.string()));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    std::filesystem::remove(invalid_input);
    CHECK(rejected);

    for (const std::string malformed_value : {"1e3", "true", "false"})
    {
        const std::filesystem::path malformed = write_temporary_gml(
            "virne-malformed-scalar",
            "graph [\n  value " + malformed_value + "\n]\n");
        rejected = false;
        try
        {
            static_cast<void>(GmlLoader::load(malformed.string()));
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        std::filesystem::remove(malformed);
        CHECK(rejected);
    }

    std::string deeply_nested = "graph [\n";
    for (size_t depth = 0; depth < 257; ++depth)
    {
        deeply_nested += "x [\n";
    }
    for (size_t depth = 0; depth < 257; ++depth)
    {
        deeply_nested += "]\n";
    }
    deeply_nested += "]\n";
    const std::filesystem::path deep = write_temporary_gml(
        "virne-deep-gml",
        deeply_nested);
    rejected = false;
    try
    {
        static_cast<void>(GmlLoader::load(deep.string()));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    std::filesystem::remove(deep);
    CHECK(rejected);

    const std::filesystem::path multigraph = write_temporary_gml(
        "virne-multigraph",
        "graph [\n  multigraph 1\n]\n");
    rejected = false;
    try
    {
        static_cast<void>(GmlLoader::load(multigraph.string()));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    std::filesystem::remove(multigraph);
    CHECK(rejected);

    const std::filesystem::path duplicate = write_temporary_gml(
        "virne-duplicate-edge",
        "graph [\n"
        "  node [ id 0 ]\n"
        "  node [ id 1 ]\n"
        "  edge [ source 0 target 1 ]\n"
        "  edge [ source 1 target 0 ]\n"
        "]\n");
    rejected = false;
    try
    {
        static_cast<void>(GmlLoader::load(duplicate.string()));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    std::filesystem::remove(duplicate);
    CHECK(rejected);

    Graph invalid_key;
    invalid_key.add_node();
    invalid_key.graph_attrs()["not-valid"] = int64_t{1};
    const std::filesystem::path output = write_temporary_gml(
        "virne-invalid-key",
        "ORIGINAL-CONTENT");
    rejected = false;
    try
    {
        GraphSaver::save_gml(invalid_key, output.string());
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    CHECK(rejected);
    std::ifstream preserved_input(output, std::ios::binary);
    const std::string preserved(
        (std::istreambuf_iterator<char>(preserved_input)),
        std::istreambuf_iterator<char>());
    std::filesystem::remove(output);
    CHECK(preserved == "ORIGINAL-CONTENT");

    const std::filesystem::path undirected = write_temporary_gml(
        "virne-undirected-as-directed",
        "graph [\n  directed 0\n]\n");
    rejected = false;
    try
    {
        static_cast<void>(GmlLoader::load_directed(undirected.string()));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    std::filesystem::remove(undirected);
    CHECK(rejected);

    if (std::filesystem::exists("/dev/full"))
    {
        Graph writable;
        writable.add_node();
        rejected = false;
        try
        {
            GraphSaver::save_gml(writable, "/dev/full");
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        CHECK(rejected);
    }

    Graph cyclic;
    cyclic.add_node();
    auto cycle = std::make_shared<AttrList>();
    cycle->values.emplace_back(cycle);
    cyclic.graph_attrs()["cycle"] = cycle;
    rejected = false;
    try
    {
        const Graph copy = cyclic;
        static_cast<void>(copy);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    CHECK(rejected);

    const std::filesystem::path cycle_output =
        temporary_path("virne-cycle");
    rejected = false;
    try
    {
        GraphSaver::save_gml(cyclic, cycle_output.string());
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    std::filesystem::remove(cycle_output);
    CHECK(rejected);
    cycle->values.clear();
}

void check_repository_dataset()
{
    const std::filesystem::path source =
        std::filesystem::path(VIRNE_SOURCE_DIR)
            .parent_path() /
        "virne/datasets/topology/Waxman100.gml";
    Graph graph = GmlLoader::load(source.string());
    CHECK(graph.num_nodes() == 100);
    CHECK(graph.num_edges() == 500);

    const AttrList* node_settings =
        attr_list(graph.graph_attrs().at("node_attrs_setting"));
    CHECK(node_settings != nullptr);
    CHECK(node_settings->values.size() == 2);
    const AttrObject* cpu =
        attr_object(node_settings->values[0]);
    CHECK(cpu != nullptr);
    CHECK(std::get<std::string>(*cpu->find("name")) == "cpu");

    const AttrObject* topology =
        attr_object(graph.graph_attrs().at("topology"));
    CHECK(topology != nullptr);
    CHECK(std::get<std::string>(*topology->find("type")) == "waxman");
    CHECK(std::abs(std::get<double>(*topology->find("wm_alpha")) - 0.5) < 1e-12);

    const AttrList* position =
        attr_list(graph.node_attrs(0).at("pos"));
    CHECK(position != nullptr);
    CHECK(position->values.size() == 2);

    const std::filesystem::path roundtrip =
        temporary_path("virne-waxman100");
    GraphSaver::save_gml(graph, roundtrip.string());
    Graph reloaded = GmlLoader::load(roundtrip.string());
    std::filesystem::remove(roundtrip);
    CHECK(reloaded.num_nodes() == graph.num_nodes());
    CHECK(reloaded.num_edges() == graph.num_edges());
    CHECK(attr_value_equal(
        graph.graph_attrs().at("node_attrs_setting"),
        reloaded.graph_attrs().at("node_attrs_setting")));
    CHECK(attr_value_equal(
        graph.graph_attrs().at("topology"),
        reloaded.graph_attrs().at("topology")));
    CHECK(attr_value_equal(
        graph.node_attrs(0).at("pos"),
        reloaded.node_attrs(0).at("pos")));
}

void check_directed_auto_load()
{
    DiGraph graph;
    graph.add_node();
    graph.add_node();
    graph.add_edge(1, 0);
    graph.graph_attrs()["kind"] = std::string("directed");

    const std::filesystem::path path =
        temporary_path("virne-directed");
    nx::write_gml(graph, path.string());
    LoadedGraph loaded = nx::read_gml_auto(path.string());
    std::filesystem::remove(path);
    CHECK(std::holds_alternative<DiGraph>(loaded));
    const DiGraph& directed = std::get<DiGraph>(loaded);
    CHECK(directed.has_edge(1, 0));
    CHECK(!directed.has_edge(0, 1));
    CHECK(std::get<std::string>(directed.graph_attrs().at("kind")) ==
          "directed");
}

} // namespace

int main()
{
    check_structured_roundtrip();
    check_repository_dataset();
    check_directed_auto_load();
    check_copy_value_semantics();
    check_networkx_named_entities();
    check_rejected_gml_cases();
    return 0;
}
