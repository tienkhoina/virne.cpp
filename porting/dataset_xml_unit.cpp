#include "dataset_xml.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace fs = std::filesystem;

using virne::utils::DatasetErrorCode;
using virne::utils::DatasetException;
using virne::utils::DatasetOperation;
using virne::utils::ParsedXmlTopology;
using virne::utils::XmlEdgeRecord;
using virne::utils::XmlNodeRecord;
using virne::utils::XmlTopologyRequest;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

class TemporaryRoot
{
public:
    TemporaryRoot()
    {
        const auto stamp = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        for (unsigned int attempt = 0; attempt < 100U; ++attempt)
        {
            path_ = fs::temp_directory_path() /
                ("vne_dataset_xml_unit_" + std::to_string(stamp) + "_" +
                 std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(path_, error))
            {
                return;
            }
            if (error)
            {
                throw std::runtime_error(
                    "failed to create XML unit temporary directory: " +
                    error.message());
            }
        }
        throw std::runtime_error("failed to choose XML unit temporary directory");
    }

    TemporaryRoot(const TemporaryRoot&) = delete;
    TemporaryRoot& operator=(const TemporaryRoot&) = delete;

    ~TemporaryRoot()
    {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path& path() const noexcept
    {
        return path_;
    }

private:
    fs::path path_;
};

void write_bytes(const fs::path& path, std::string_view bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("failed to open XML unit fixture");
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output)
    {
        throw std::runtime_error("failed to write XML unit fixture");
    }
}

std::string read_bytes(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to open XML unit output");
    }
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

bool same_node(const XmlNodeRecord& left, const XmlNodeRecord& right)
{
    return left.label == right.label && left.x == right.x && left.y == right.y;
}

bool same_edge(const XmlEdgeRecord& left, const XmlEdgeRecord& right)
{
    return left.label == right.label &&
        left.source_label == right.source_label &&
        left.target_label == right.target_label &&
        left.capacity_st == right.capacity_st &&
        left.capacity_ts == right.capacity_ts &&
        left.cost_st == right.cost_st &&
        left.cost_ts == right.cost_ts;
}

bool same_topology(const ParsedXmlTopology& left, const ParsedXmlTopology& right)
{
    if (left.nodes.size() != right.nodes.size() ||
        left.edges.size() != right.edges.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < left.nodes.size(); ++index)
    {
        if (!same_node(left.nodes[index], right.nodes[index]))
        {
            return false;
        }
    }
    for (std::size_t index = 0; index < left.edges.size(); ++index)
    {
        if (!same_edge(left.edges[index], right.edges[index]))
        {
            return false;
        }
    }
    return true;
}

template <typename Callable>
DatasetException expect_dataset_exception(
    Callable&& callable,
    DatasetErrorCode code,
    DatasetOperation operation)
{
    try
    {
        callable();
    }
    catch (const DatasetException& error)
    {
        expect(error.code() == code, "dataset XML error code mismatch");
        expect(error.operation() == operation, "dataset XML operation mismatch");
        return error;
    }
    throw std::runtime_error("expected DatasetException");
}

const std::string& string_attribute(const AttrMap& attributes, AttrId id)
{
    const auto* value = std::get_if<std::string>(&attributes.at(id));
    if (value == nullptr)
    {
        throw std::runtime_error("dataset XML graph attribute is not a string");
    }
    return *value;
}

std::string base_xml()
{
    return
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<network><networkStructure><nodes>"
        "<node id=\"A\"><coordinates><meta><x>first-x</x></meta>"
        "<x>ignored-x</x><y>1 &amp; 2</y></coordinates></node>"
        "<node id=\"B\"><coordinates><x><![CDATA[3<4]]></x><y>5</y>"
        "</coordinates></node>"
        "</nodes><links>"
        "<link id=\"AB\"><source>A</source><target>B</target>"
        "<additionalModules><addModule><capacity>10</capacity><cost>900</cost>"
        "</addModule><addModule><capacity>20</capacity><cost>800</cost>"
        "</addModule><addModule><capacity>30</capacity><cost>700</cost>"
        "</addModule></additionalModules></link>"
        "</links></networkStructure></network>";
}

void test_parse_records(const fs::path& root)
{
    const fs::path source = root / "base.xml";
    write_bytes(source, base_xml());
    const ParsedXmlTopology parsed = virne::utils::parse_sndlib_xml(source);
    expect(parsed.nodes.size() == 2, "base XML node count mismatch");
    expect(parsed.edges.size() == 1, "base XML edge count mismatch");
    expect(parsed.nodes[0].label == "A", "first node label mismatch");
    expect(parsed.nodes[0].x == "first-x", "first descendant x mismatch");
    expect(parsed.nodes[0].y == "1 & 2", "entity decoding mismatch");
    expect(parsed.nodes[1].x == "3<4", "CDATA decoding mismatch");
    expect(parsed.edges[0].label == "AB", "edge label mismatch");
    expect(parsed.edges[0].source_label == "A", "edge source label mismatch");
    expect(parsed.edges[0].target_label == "B", "edge target label mismatch");
    expect(parsed.edges[0].capacity_st == "10", "first capacity mismatch");
    expect(parsed.edges[0].capacity_ts == "20", "second capacity mismatch");
    expect(parsed.edges[0].cost_st == "10", "capacity-as-cost st bug drift");
    expect(parsed.edges[0].cost_ts == "20", "capacity-as-cost ts bug drift");

    const fs::path utf8 = root / "utf8.xml";
    const std::string utf8_payload =
        std::string("\xEF\xBB\xBF") +
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<network><node id=\"N\xC3\xA9\"><x>\xE6\xBC\xA2</x><y>y</y></node>"
        "</network>";
    write_bytes(utf8, utf8_payload);
    const ParsedXmlTopology utf8_parsed = virne::utils::parse_sndlib_xml(utf8);
    expect(utf8_parsed.nodes.size() == 1, "UTF-8 BOM node count mismatch");
    expect(utf8_parsed.nodes[0].label == "N\xC3\xA9", "UTF-8 label mismatch");
    expect(utf8_parsed.nodes[0].x == "\xE6\xBC\xA2", "UTF-8 value mismatch");

    const fs::path latin1 = root / "latin1.xml";
    const std::string latin1_payload =
        std::string("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>") +
        "<network><node id=\"N" + std::string(1, static_cast<char>(0xE9)) +
        "\"><x>1</x><y>2</y></node></network>";
    write_bytes(latin1, latin1_payload);
    const ParsedXmlTopology latin1_parsed = virne::utils::parse_sndlib_xml(latin1);
    expect(latin1_parsed.nodes.size() == 1, "Latin-1 node count mismatch");
    expect(latin1_parsed.nodes[0].label == "N\xC3\xA9", "Latin-1 conversion mismatch");
}

ParsedXmlTopology duplicate_topology()
{
    ParsedXmlTopology topology;
    topology.nodes = {
        {"A", "0", "0"},
        {"A", "1", "1"},
        {"B", "2", "2"},
    };
    topology.edges = {
        {"AB-first", "A", "B", "10", "20", "10", "20"},
        {"BA-last", "B", "A", "30", "40", "30", "40"},
        {"BB", "B", "B", "50", "60", "50", "60"},
    };
    return topology;
}

void test_materialization()
{
    const Graph graph = virne::utils::materialize_xml_topology(
        "Topo & \xC3\xA9", duplicate_topology());
    expect(graph.num_nodes() == 3, "materialized node count mismatch");
    expect(graph.num_edges() == 2, "simple-edge collapse mismatch");

    const AttrId name_id = graph.attr_id("name");
    const AttrId label_id = graph.attr_id("label");
    const AttrId x_id = graph.attr_id("x");
    const AttrId y_id = graph.attr_id("y");
    const AttrId source_label_id = graph.attr_id("source_label");
    const AttrId target_label_id = graph.attr_id("target_label");
    const AttrId capacity_st_id = graph.attr_id("capacity_st");
    const AttrId capacity_ts_id = graph.attr_id("capacity_ts");
    const AttrId cost_st_id = graph.attr_id("cost_st");
    const AttrId cost_ts_id = graph.attr_id("cost_ts");

    expect(
        string_attribute(graph.graph_attrs(), name_id) == "Topo & \xC3\xA9",
        "graph name mismatch");
    expect(string_attribute(graph.node_attrs(0), label_id) == "A",
           "duplicate node zero label mismatch");
    expect(string_attribute(graph.node_attrs(1), label_id) == "A",
           "duplicate node one label mismatch");
    expect(string_attribute(graph.node_attrs(1), x_id) == "1",
           "node x mismatch");
    expect(string_attribute(graph.node_attrs(2), y_id) == "2",
           "node y mismatch");

    bool found_ab = false;
    bool found_loop = false;
    const auto [begin, end] = graph.edges();
    for (auto iterator = begin; iterator != end; ++iterator)
    {
        const Edge edge = *iterator;
        const Vertex source = graph.source(edge);
        const Vertex target = graph.target(edge);
        const AttrMap& attributes = graph.edge_attrs(edge);
        if ((source == 1 && target == 2) || (source == 2 && target == 1))
        {
            found_ab = true;
            expect(string_attribute(attributes, label_id) == "BA-last",
                   "duplicate edge did not keep last attributes");
            expect(string_attribute(attributes, source_label_id) == "B",
                   "duplicate edge source label mismatch");
            expect(string_attribute(attributes, target_label_id) == "A",
                   "duplicate edge target label mismatch");
            expect(string_attribute(attributes, capacity_st_id) == "30",
                   "duplicate edge first capacity mismatch");
            expect(string_attribute(attributes, capacity_ts_id) == "40",
                   "duplicate edge second capacity mismatch");
            expect(string_attribute(attributes, cost_st_id) == "30",
                   "duplicate edge first cost mismatch");
            expect(string_attribute(attributes, cost_ts_id) == "40",
                   "duplicate edge second cost mismatch");
        }
        if (source == 2 && target == 2)
        {
            found_loop = true;
            expect(string_attribute(attributes, label_id) == "BB",
                   "self-loop label mismatch");
        }
    }
    expect(found_ab, "collapsed duplicate edge missing");
    expect(found_loop, "self-loop missing");
}

void test_error_stages_and_side_effects(const fs::path& root)
{
    const fs::path malformed = root / "malformed.xml";
    write_bytes(malformed, "<network><node></network>");
    const DatasetException malformed_error = expect_dataset_exception(
        [&] { (void)virne::utils::parse_sndlib_xml(malformed); },
        DatasetErrorCode::xml_parse_failure,
        DatasetOperation::parse_xml);
    expect(malformed_error.path() == malformed, "malformed XML path mismatch");
    expect(!malformed_error.input_index().has_value(),
           "scalar parse unexpectedly has input index");

    const fs::path missing_x = root / "missing_x.xml";
    write_bytes(missing_x, "<network><node id=\"A\"><y>2</y></node></network>");
    expect_dataset_exception(
        [&] { (void)virne::utils::parse_sndlib_xml(missing_x); },
        DatasetErrorCode::xml_schema_failure,
        DatasetOperation::parse_xml);

    const fs::path one_capacity = root / "one_capacity.xml";
    write_bytes(
        one_capacity,
        "<network><node id=\"A\"><x>1</x><y>2</y></node>"
        "<link id=\"AA\"><source>A</source><target>A</target>"
        "<capacity>10</capacity></link></network>");
    expect_dataset_exception(
        [&] { (void)virne::utils::parse_sndlib_xml(one_capacity); },
        DatasetErrorCode::xml_schema_failure,
        DatasetOperation::parse_xml);

    ParsedXmlTopology unknown;
    unknown.nodes = {{"A", "1", "2"}};
    unknown.edges = {{"AX", "A", "X", "1", "2", "1", "2"}};
    expect_dataset_exception(
        [&] { (void)virne::utils::materialize_xml_topology("unknown", unknown); },
        DatasetErrorCode::unknown_endpoint,
        DatasetOperation::materialize_graph);

    const fs::path target = root / "unchanged.gml";
    write_bytes(target, "sentinel");
    expect_dataset_exception(
        [&]
        {
            (void)virne::utils::preprocess_xml(
                XmlTopologyRequest{"bad", malformed, target});
        },
        DatasetErrorCode::xml_parse_failure,
        DatasetOperation::parse_xml);
    expect(read_bytes(target) == "sentinel",
           "parse failure changed pre-existing target bytes");

    const fs::path valid = root / "writer_source.xml";
    write_bytes(valid, base_xml());
    const fs::path target_directory = root / "target_directory";
    fs::create_directory(target_directory);
    const DatasetException writer_error = expect_dataset_exception(
        [&]
        {
            (void)virne::utils::preprocess_xml(
                XmlTopologyRequest{"writer", valid, target_directory});
        },
        DatasetErrorCode::gml_write_failure,
        DatasetOperation::write_gml);
    expect(writer_error.path() == target_directory, "writer error path mismatch");
    expect(fs::is_directory(target_directory), "writer failure removed target directory");
}

void test_preprocess_success(const fs::path& root)
{
    const fs::path source = root / "preprocess.xml";
    const fs::path target = root / "preprocess.gml";
    write_bytes(source, base_xml());
    write_bytes(target, "a much longer pre-existing sentinel payload");
    const Graph graph = virne::utils::preprocess_xml(
        XmlTopologyRequest{"Preprocess", source, target});
    expect(graph.num_nodes() == 2, "preprocess node count mismatch");
    expect(graph.num_edges() == 1, "preprocess edge count mismatch");
    const std::string bytes = read_bytes(target);
    expect(!bytes.empty(), "preprocess wrote an empty GML file");
    expect(bytes != "a much longer pre-existing sentinel payload",
           "preprocess did not replace existing target");
}

void test_native_safety_boundaries(const fs::path& root)
{
    const fs::path unsupported = root / "windows1252.xml";
    write_bytes(
        unsupported,
        "<?xml version=\"1.0\" encoding=\"windows-1252\"?>"
        "<network><node id=\"A\"><x>1</x><y>2</y></node></network>");
    expect_dataset_exception(
        [&] { (void)virne::utils::parse_sndlib_xml(unsupported); },
        DatasetErrorCode::xml_parse_failure,
        DatasetOperation::parse_xml);

    const fs::path utf16 = root / "utf16.xml";
    write_bytes(utf16, std::string("\xFF\xFE<\0", 4));
    expect_dataset_exception(
        [&] { (void)virne::utils::parse_sndlib_xml(utf16); },
        DatasetErrorCode::xml_parse_failure,
        DatasetOperation::parse_xml);

    const fs::path custom_entity = root / "custom_entity.xml";
    write_bytes(
        custom_entity,
        "<!DOCTYPE network [<!ENTITY custom \"value\">]>"
        "<network><node id=\"A\"><x>&custom;</x><y>2</y></node></network>");
    expect_dataset_exception(
        [&] { (void)virne::utils::parse_sndlib_xml(custom_entity); },
        DatasetErrorCode::xml_parse_failure,
        DatasetOperation::parse_xml);

    const fs::path alias = root / "source_target_alias.xml";
    write_bytes(alias, base_xml());
    const std::string original = read_bytes(alias);
    expect_dataset_exception(
        [&]
        {
            (void)virne::utils::preprocess_xml(
                XmlTopologyRequest{"alias", alias, alias});
        },
        DatasetErrorCode::gml_write_failure,
        DatasetOperation::write_gml);
    expect(read_bytes(alias) == original, "source/target alias changed XML input");

    const fs::path source = root / "compressed_source.xml";
    const fs::path compressed = root / "unsupported.gml.gz";
    write_bytes(source, base_xml());
    write_bytes(compressed, "sentinel");
    expect_dataset_exception(
        [&]
        {
            (void)virne::utils::preprocess_xml(
                XmlTopologyRequest{"compressed", source, compressed});
        },
        DatasetErrorCode::gml_write_failure,
        DatasetOperation::write_gml);
    expect(read_bytes(compressed) == "sentinel",
           "compressed-target rejection changed existing target");
}

void test_batch_contract(const fs::path& root)
{
    const fs::path first = root / "batch_first.xml";
    const fs::path second = root / "batch_second.xml";
    write_bytes(first, base_xml());
    write_bytes(
        second,
        "<network><node id=\"only\"><x>7</x><y>8</y></node></network>");
    const std::vector<fs::path> paths = {
        first, second, first, second, first, second, first, second,
        first, second, first, second, first, second, first, second,
    };
    const std::vector<ParsedXmlTopology> baseline =
        virne::utils::parse_sndlib_xml_batch(paths, 1);
    expect(baseline.size() == paths.size(), "batch result count mismatch");
    for (const std::size_t workers : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U})
    {
        const std::vector<ParsedXmlTopology> actual =
            virne::utils::parse_sndlib_xml_batch(paths, workers);
        expect(actual.size() == baseline.size(), "worker batch size drift");
        for (std::size_t index = 0; index < baseline.size(); ++index)
        {
            expect(same_topology(actual[index], baseline[index]),
                   "worker batch record drift");
        }
    }
    const std::vector<ParsedXmlTopology> capped =
        virne::utils::parse_sndlib_xml_batch(
            paths, std::numeric_limits<std::size_t>::max());
    expect(capped.size() == baseline.size(), "huge worker request size drift");
    for (std::size_t index = 0; index < baseline.size(); ++index)
    {
        expect(same_topology(capped[index], baseline[index]),
               "huge worker request record drift");
    }

    for (const std::size_t workers : {0U, 1U, 4U, 8U})
    {
        const std::vector<ParsedXmlTopology> empty =
            virne::utils::parse_sndlib_xml_batch({}, workers);
        expect(empty.empty(), "empty batch was not empty");
    }

    const fs::path missing = root / "missing.xml";
    const fs::path malformed = root / "batch_malformed.xml";
    write_bytes(malformed, "<network><node></network>");
    const std::vector<fs::path> errors = {first, missing, malformed, second};
    for (const std::size_t workers : {0U, 1U, 2U, 4U, 8U})
    {
        const DatasetException error = expect_dataset_exception(
            [&] { (void)virne::utils::parse_sndlib_xml_batch(errors, workers); },
            DatasetErrorCode::xml_parse_failure,
            DatasetOperation::parse_xml);
        expect(error.input_index() == std::optional<std::size_t>{1},
               "batch did not select the lowest failing index");
        expect(error.path() == missing, "batch lowest-error path mismatch");
    }
}

} // namespace

int main()
{
    try
    {
        TemporaryRoot root;
        test_parse_records(root.path());
        test_materialization();
        test_error_stages_and_side_effects(root.path());
        test_preprocess_success(root.path());
        test_native_safety_boundaries(root.path());
        test_batch_contract(root.path());
        std::cout << "dataset XML unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "dataset XML unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
