#include "dataset_xml.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

using virne::utils::DatasetErrorCode;
using virne::utils::DatasetException;
using virne::utils::DatasetOperation;
using virne::utils::ParsedXmlTopology;
using virne::utils::XmlTopologyRequest;

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::string hex_encode(std::string_view value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(value.size() * 2, '0');
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2] = digits[byte >> 4U];
        result[index * 2 + 1] = digits[byte & 0x0FU];
    }
    return result;
}

std::string read_bytes(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to read dataset XML harness target");
    }
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::uint64_t fnv1a(std::string_view value)
{
    std::uint64_t checksum = kFnvOffset;
    for (const char item : value)
    {
        checksum ^= static_cast<unsigned char>(item);
        checksum *= kFnvPrime;
    }
    return checksum;
}

std::string error_name(DatasetErrorCode code)
{
    switch (code)
    {
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
    default:
        return "other";
    }
}

std::string operation_name(DatasetOperation operation)
{
    switch (operation)
    {
    case DatasetOperation::parse_xml:
        return "parse_xml";
    case DatasetOperation::materialize_graph:
        return "materialize_graph";
    case DatasetOperation::write_gml:
        return "write_gml";
    default:
        return "other";
    }
}

std::size_t parse_size(const char* text)
{
    const std::string value(text);
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("invalid dataset XML size argument");
    }
    return static_cast<std::size_t>(parsed);
}

void append_hex_field(std::string& output, std::string_view value)
{
    output += hex_encode(value);
}

std::string serialize_topology(const ParsedXmlTopology& topology)
{
    std::string output = "records-v1\nnodes=" +
        std::to_string(topology.nodes.size()) + "\n";
    for (const auto& node : topology.nodes)
    {
        output += "node=";
        append_hex_field(output, node.label);
        output.push_back('|');
        append_hex_field(output, node.x);
        output.push_back('|');
        append_hex_field(output, node.y);
        output.push_back('\n');
    }
    output += "edges=" + std::to_string(topology.edges.size()) + "\n";
    for (const auto& edge : topology.edges)
    {
        output += "edge=";
        append_hex_field(output, edge.label);
        for (const std::string* field : {
                 &edge.source_label,
                 &edge.target_label,
                 &edge.capacity_st,
                 &edge.capacity_ts,
                 &edge.cost_st,
                 &edge.cost_ts})
        {
            output.push_back('|');
            append_hex_field(output, *field);
        }
        output.push_back('\n');
    }
    return output;
}

std::string serialize_batch(const std::vector<ParsedXmlTopology>& topologies)
{
    std::string output = "batch-v1\ncount=" +
        std::to_string(topologies.size()) + "\n";
    for (const ParsedXmlTopology& topology : topologies)
    {
        const std::string item = serialize_topology(topology);
        output += "item-bytes=" + std::to_string(item.size()) + "\n";
        output += item;
    }
    return output;
}

const std::string& string_attribute(const AttrMap& attributes, AttrId id)
{
    const auto* value = std::get_if<std::string>(&attributes.at(id));
    if (value == nullptr)
    {
        throw std::runtime_error("dataset XML graph attribute type drift");
    }
    return *value;
}

std::string serialize_graph(const Graph& graph)
{
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

    std::string output = "graph-v1\ndirected=0\nmultigraph=0\nname=";
    append_hex_field(output, string_attribute(graph.graph_attrs(), name_id));
    output += "\nnodes=" + std::to_string(graph.num_nodes()) + "\n";
    for (Vertex node = 0; node < graph.num_nodes(); ++node)
    {
        const AttrMap& attributes = graph.node_attrs(node);
        output += "node=" + std::to_string(node) + '|';
        append_hex_field(output, string_attribute(attributes, label_id));
        output.push_back('|');
        append_hex_field(output, string_attribute(attributes, x_id));
        output.push_back('|');
        append_hex_field(output, string_attribute(attributes, y_id));
        output.push_back('\n');
    }
    output += "edges=" + std::to_string(graph.num_edges()) + "\n";
    const auto [begin, end] = graph.edges();
    for (auto iterator = begin; iterator != end; ++iterator)
    {
        const Edge edge = *iterator;
        const AttrMap& attributes = graph.edge_attrs(edge);
        output += "edge=" + std::to_string(graph.source(edge)) + '|' +
            std::to_string(graph.target(edge));
        for (const AttrId id : {
                 label_id,
                 source_label_id,
                 target_label_id,
                 capacity_st_id,
                 capacity_ts_id,
                 cost_st_id,
                 cost_ts_id})
        {
            output.push_back('|');
            append_hex_field(output, string_attribute(attributes, id));
        }
        output.push_back('\n');
    }
    return output;
}

void emit_ok(
    std::string_view command,
    std::string_view payload,
    std::optional<std::string_view> gml = std::nullopt)
{
    std::cout << "dataset_xml_harness_version=1\n"
              << "command=" << command << '\n'
              << "status=ok\n"
              << "payload_hex=" << hex_encode(payload) << '\n';
    if (gml.has_value())
    {
        std::cout << "gml_hex=" << hex_encode(*gml) << '\n';
    }
}

void emit_error(
    std::string_view command,
    const DatasetException& error,
    const std::optional<fs::path>& target)
{
    std::string target_kind = "none";
    std::string target_bytes;
    if (target.has_value())
    {
        std::error_code status_error;
        const fs::file_status status = fs::symlink_status(*target, status_error);
        if (status_error || !fs::exists(status))
        {
            target_kind = "missing";
        }
        else if (fs::is_regular_file(status))
        {
            target_kind = "file";
            target_bytes = read_bytes(*target);
        }
        else if (fs::is_directory(status))
        {
            target_kind = "directory";
        }
        else
        {
            target_kind = "other";
        }
    }

    std::cout << "dataset_xml_harness_version=1\n"
              << "command=" << command << '\n'
              << "status=error\n"
              << "error_code=" << error_name(error.code()) << '\n'
              << "operation=" << operation_name(error.operation()) << '\n'
              << "input_index=";
    if (error.input_index().has_value())
    {
        std::cout << *error.input_index();
    }
    else
    {
        std::cout << "none";
    }
    std::cout << '\n'
              << "path_hex=" << hex_encode(error.path().string()) << '\n'
              << "target_kind=" << target_kind << '\n'
              << "target_hex=" << hex_encode(target_bytes) << '\n';
}

void emit_benchmark(
    std::string_view command,
    std::size_t documents,
    std::size_t workers,
    std::size_t repetitions,
    std::uint64_t elapsed_ns,
    std::string_view payload,
    std::optional<std::string_view> graph = std::nullopt,
    std::optional<std::string_view> gml = std::nullopt)
{
    std::cout << "dataset_xml_benchmark_version=1\n"
              << "command=" << command << '\n'
              << "documents=" << documents << '\n'
              << "workers=" << workers << '\n'
              << "repetitions=" << repetitions << '\n'
              << "elapsed_ns=" << elapsed_ns << '\n'
              << "checksum=" << fnv1a(payload) << '\n'
              << "output_bytes=" << payload.size() << '\n';
    if (graph.has_value())
    {
        std::cout << "graph_hex=" << hex_encode(*graph) << '\n';
    }
    if (gml.has_value())
    {
        std::cout << "gml_hex=" << hex_encode(*gml) << '\n';
    }
    std::cout << "status=PASS\n";
}

void run_parse(const fs::path& source)
{
    emit_ok("parse", serialize_topology(virne::utils::parse_sndlib_xml(source)));
}

void run_parse_batch(
    std::size_t workers,
    const std::vector<fs::path>& sources)
{
    emit_ok(
        "parse_batch",
        serialize_batch(virne::utils::parse_sndlib_xml_batch(sources, workers)));
}

void run_materialize(std::string_view name, const fs::path& source)
{
    const ParsedXmlTopology topology = virne::utils::parse_sndlib_xml(source);
    emit_ok(
        "materialize",
        serialize_graph(virne::utils::materialize_xml_topology(name, topology)));
}

void run_preprocess(
    std::string name,
    const fs::path& source,
    const fs::path& target)
{
    const Graph graph = virne::utils::preprocess_xml(
        XmlTopologyRequest{std::move(name), source, target});
    const std::string gml = read_bytes(target);
    emit_ok("preprocess", serialize_graph(graph), gml);
}

void run_parse_benchmark(
    const fs::path& source,
    std::size_t documents,
    std::size_t workers,
    std::size_t repetitions)
{
    if (documents == 0 || repetitions == 0)
    {
        throw std::invalid_argument("parse benchmark counts must be positive");
    }
    const std::vector<fs::path> sources(documents, source);
    std::vector<ParsedXmlTopology> result;
    const auto start = Clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
        result = virne::utils::parse_sndlib_xml_batch(sources, workers);
    }
    const auto end = Clock::now();
    const std::string payload = serialize_batch(result);
    emit_benchmark(
        "benchmark_parse",
        documents,
        workers,
        repetitions,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count()),
        payload);
}

void run_preprocess_benchmark(
    const std::string& name,
    const fs::path& source,
    const fs::path& target,
    std::size_t repetitions)
{
    if (repetitions == 0)
    {
        throw std::invalid_argument("preprocess repetitions must be positive");
    }
    Graph graph;
    const auto start = Clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
        graph = virne::utils::preprocess_xml(
            XmlTopologyRequest{name, source, target});
    }
    const auto end = Clock::now();
    const std::string graph_payload = serialize_graph(graph);
    const std::string gml = read_bytes(target);
    std::string payload = graph_payload;
    payload.push_back('\0');
    payload += gml;
    emit_benchmark(
        "benchmark_preprocess",
        1,
        1,
        repetitions,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count()),
        payload,
        graph_payload,
        gml);
}

} // namespace

int main(int argc, char** argv)
{
    std::string command;
    std::optional<fs::path> error_target;
    try
    {
        if (argc < 2)
        {
            throw std::invalid_argument("missing dataset XML harness command");
        }
        command = argv[1];
        if (command == "parse" && argc == 3)
        {
            run_parse(fs::path(argv[2]));
            return 0;
        }
        if (command == "parse_batch" && argc >= 4)
        {
            const std::size_t workers = parse_size(argv[2]);
            std::vector<fs::path> sources;
            sources.reserve(static_cast<std::size_t>(argc - 3));
            for (int index = 3; index < argc; ++index)
            {
                sources.emplace_back(argv[index]);
            }
            run_parse_batch(workers, sources);
            return 0;
        }
        if (command == "materialize" && argc == 4)
        {
            run_materialize(argv[2], fs::path(argv[3]));
            return 0;
        }
        if (command == "preprocess" && argc == 5)
        {
            error_target = fs::path(argv[4]);
            run_preprocess(argv[2], fs::path(argv[3]), *error_target);
            return 0;
        }
        if (command == "benchmark_parse" && argc == 6)
        {
            run_parse_benchmark(
                fs::path(argv[2]),
                parse_size(argv[3]),
                parse_size(argv[4]),
                parse_size(argv[5]));
            return 0;
        }
        if (command == "benchmark_preprocess" && argc == 6)
        {
            error_target = fs::path(argv[4]);
            run_preprocess_benchmark(
                argv[2], fs::path(argv[3]), *error_target, parse_size(argv[5]));
            return 0;
        }
        throw std::invalid_argument("invalid dataset XML harness command");
    }
    catch (const DatasetException& error)
    {
        emit_error(command, error, error_target);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "dataset XML harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
