#pragma once

#include "dataset.h"
#include "graph.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace virne::utils
{

struct XmlNodeRecord
{
    std::string label;
    std::string x;
    std::string y;
};

struct XmlEdgeRecord
{
    std::string label;
    std::string source_label;
    std::string target_label;
    std::string capacity_st;
    std::string capacity_ts;
    std::string cost_st;
    std::string cost_ts;
};

struct ParsedXmlTopology
{
    std::vector<XmlNodeRecord> nodes;
    std::vector<XmlEdgeRecord> edges;
};

struct XmlTopologyRequest
{
    std::string topology_name;
    std::filesystem::path xml_source_path;
    std::filesystem::path gml_target_path;
};

ParsedXmlTopology parse_sndlib_xml(
    const std::filesystem::path& source_path);

std::vector<ParsedXmlTopology> parse_sndlib_xml_batch(
    const std::vector<std::filesystem::path>& source_paths,
    std::size_t workers = 0);

Graph materialize_xml_topology(
    std::string_view topology_name,
    const ParsedXmlTopology& topology);

Graph preprocess_xml(const XmlTopologyRequest& request);

} // namespace virne::utils
