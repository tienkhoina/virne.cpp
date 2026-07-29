#include "dataset_xml.h"

#include <boost/property_tree/detail/rapidxml.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace virne::utils
{
namespace
{

namespace rapidxml = boost::property_tree::detail::rapidxml;

using XmlNode = rapidxml::xml_node<char>;

enum class XmlEncoding : std::uint8_t
{
    utf8,
    iso_8859_1,
};

struct XmlElementLists
{
    std::vector<XmlNode*> nodes;
    std::vector<XmlNode*> links;
};

struct XmlAttrIds
{
    AttrId name;
    AttrId label;
    AttrId x;
    AttrId y;
    AttrId source_label;
    AttrId target_label;
    AttrId capacity_st;
    AttrId capacity_ts;
    AttrId cost_st;
    AttrId cost_ts;
};

[[noreturn]] void throw_parse_error(
    const std::filesystem::path& path,
    std::string message)
{
    throw DatasetException(
        DatasetErrorCode::xml_parse_failure,
        DatasetOperation::parse_xml,
        std::move(message),
        std::nullopt,
        path);
}

[[noreturn]] void throw_schema_error(
    const std::filesystem::path& path,
    std::string message)
{
    throw DatasetException(
        DatasetErrorCode::xml_schema_failure,
        DatasetOperation::parse_xml,
        std::move(message),
        std::nullopt,
        path);
}

bool ascii_iequals(
    std::string_view lhs,
    std::string_view rhs) noexcept
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index)
    {
        const auto left = static_cast<unsigned char>(lhs[index]);
        const auto right = static_cast<unsigned char>(rhs[index]);
        if (std::tolower(left) != std::tolower(right))
        {
            return false;
        }
    }
    return true;
}

XmlEncoding detect_xml_encoding(
    std::string_view bytes,
    const std::filesystem::path& path)
{
    if (bytes.size() >= 2)
    {
        const auto first = static_cast<unsigned char>(bytes[0]);
        const auto second = static_cast<unsigned char>(bytes[1]);
        if ((first == 0xFFU && second == 0xFEU) ||
            (first == 0xFEU && second == 0xFFU))
        {
            throw_parse_error(path, "unsupported UTF-16 XML encoding");
        }
    }

    std::size_t offset = 0;
    if (bytes.size() >= 3 &&
        static_cast<unsigned char>(bytes[0]) == 0xEFU &&
        static_cast<unsigned char>(bytes[1]) == 0xBBU &&
        static_cast<unsigned char>(bytes[2]) == 0xBFU)
    {
        offset = 3;
    }
    if (bytes.size() - offset < 5 ||
        bytes.compare(offset, 5, "<?xml") != 0)
    {
        return XmlEncoding::utf8;
    }

    const std::size_t declaration_end = bytes.find("?>", offset + 5);
    if (declaration_end == std::string_view::npos)
    {
        return XmlEncoding::utf8;
    }
    std::string_view declaration = bytes.substr(
        offset + 5, declaration_end - (offset + 5));

    std::size_t cursor = 0;
    while (cursor < declaration.size())
    {
        while (cursor < declaration.size() &&
               std::isspace(static_cast<unsigned char>(declaration[cursor])) != 0)
        {
            ++cursor;
        }
        const std::size_t name_begin = cursor;
        while (cursor < declaration.size())
        {
            const unsigned char value =
                static_cast<unsigned char>(declaration[cursor]);
            if (std::isalnum(value) == 0 && value != '_' && value != '-')
            {
                break;
            }
            ++cursor;
        }
        const std::string_view name =
            declaration.substr(name_begin, cursor - name_begin);
        while (cursor < declaration.size() &&
               std::isspace(static_cast<unsigned char>(declaration[cursor])) != 0)
        {
            ++cursor;
        }
        if (cursor >= declaration.size() || declaration[cursor] != '=')
        {
            if (cursor < declaration.size())
            {
                ++cursor;
            }
            continue;
        }
        ++cursor;
        while (cursor < declaration.size() &&
               std::isspace(static_cast<unsigned char>(declaration[cursor])) != 0)
        {
            ++cursor;
        }
        if (cursor >= declaration.size() ||
            (declaration[cursor] != '\'' && declaration[cursor] != '"'))
        {
            continue;
        }
        const char quote = declaration[cursor++];
        const std::size_t value_begin = cursor;
        const std::size_t value_end = declaration.find(quote, cursor);
        if (value_end == std::string_view::npos)
        {
            return XmlEncoding::utf8;
        }
        cursor = value_end + 1;
        if (!ascii_iequals(name, "encoding"))
        {
            continue;
        }

        const std::string_view encoding =
            declaration.substr(value_begin, value_end - value_begin);
        if (ascii_iequals(encoding, "utf-8") ||
            ascii_iequals(encoding, "utf8"))
        {
            return XmlEncoding::utf8;
        }
        if (ascii_iequals(encoding, "iso-8859-1") ||
            ascii_iequals(encoding, "iso_8859-1") ||
            ascii_iequals(encoding, "latin1") ||
            ascii_iequals(encoding, "latin-1"))
        {
            return XmlEncoding::iso_8859_1;
        }
        throw_parse_error(path, "unsupported declared XML encoding");
    }
    return XmlEncoding::utf8;
}

bool valid_utf8(std::string_view value) noexcept
{
    std::size_t index = 0;
    while (index < value.size())
    {
        const auto lead = static_cast<unsigned char>(value[index]);
        if (lead < 0x80U)
        {
            if (lead != 0x09U && lead != 0x0AU && lead != 0x0DU &&
                lead < 0x20U)
            {
                return false;
            }
            ++index;
            continue;
        }

        std::size_t count = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if ((lead & 0xE0U) == 0xC0U)
        {
            count = 2;
            codepoint = lead & 0x1FU;
            minimum = 0x80U;
        }
        else if ((lead & 0xF0U) == 0xE0U)
        {
            count = 3;
            codepoint = lead & 0x0FU;
            minimum = 0x800U;
        }
        else if ((lead & 0xF8U) == 0xF0U)
        {
            count = 4;
            codepoint = lead & 0x07U;
            minimum = 0x10000U;
        }
        else
        {
            return false;
        }
        if (index + count > value.size())
        {
            return false;
        }
        for (std::size_t continuation = 1; continuation < count; ++continuation)
        {
            const auto byte =
                static_cast<unsigned char>(value[index + continuation]);
            if ((byte & 0xC0U) != 0x80U)
            {
                return false;
            }
            codepoint = (codepoint << 6U) | (byte & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFFU ||
            (codepoint >= 0xD800U && codepoint <= 0xDFFFU) ||
            codepoint == 0xFFFEU || codepoint == 0xFFFFU)
        {
            return false;
        }
        index += count;
    }
    return true;
}

std::uint32_t decode_utf8_codepoint(
    std::string_view text,
    std::size_t& index) noexcept
{
    const auto lead = static_cast<unsigned char>(text[index]);
    if (lead < 0x80U)
    {
        ++index;
        return lead;
    }
    std::size_t count = 0;
    std::uint32_t codepoint = 0;
    if ((lead & 0xE0U) == 0xC0U)
    {
        count = 2;
        codepoint = lead & 0x1FU;
    }
    else if ((lead & 0xF0U) == 0xE0U)
    {
        count = 3;
        codepoint = lead & 0x0FU;
    }
    else
    {
        count = 4;
        codepoint = lead & 0x07U;
    }
    for (std::size_t continuation = 1; continuation < count; ++continuation)
    {
        const auto byte =
            static_cast<unsigned char>(text[index + continuation]);
        codepoint = (codepoint << 6U) | (byte & 0x3FU);
    }
    index += count;
    return codepoint;
}

bool valid_xml_codepoint(std::uint32_t value) noexcept
{
    return value == 0x09U || value == 0x0AU || value == 0x0DU ||
           (value >= 0x20U && value <= 0xD7FFU) ||
           (value >= 0xE000U && value <= 0xFFFDU) ||
           (value >= 0x10000U && value <= 0x10FFFFU);
}

void validate_entity_reference(
    std::string_view xml,
    std::size_t& index,
    const std::filesystem::path& path)
{
    const std::size_t semicolon = xml.find(';', index + 1);
    if (semicolon == std::string_view::npos)
    {
        throw_parse_error(path, "unterminated XML entity reference");
    }
    const std::string_view entity =
        xml.substr(index + 1, semicolon - index - 1);
    if (entity == "amp" || entity == "lt" || entity == "gt" ||
        entity == "apos" || entity == "quot")
    {
        index = semicolon + 1;
        return;
    }
    if (entity.empty() || entity.front() != '#')
    {
        throw_parse_error(path, "undefined XML entity reference");
    }

    std::size_t digit = 1;
    unsigned int base = 10;
    if (digit < entity.size() && entity[digit] == 'x')
    {
        base = 16;
        ++digit;
    }
    if (digit == entity.size())
    {
        throw_parse_error(path, "empty numeric XML entity reference");
    }

    std::uint32_t codepoint = 0;
    for (; digit < entity.size(); ++digit)
    {
        const unsigned char byte = static_cast<unsigned char>(entity[digit]);
        unsigned int value = 0;
        if (byte >= '0' && byte <= '9')
        {
            value = static_cast<unsigned int>(byte - '0');
        }
        else if (base == 16U && byte >= 'a' && byte <= 'f')
        {
            value = static_cast<unsigned int>(byte - 'a') + 10U;
        }
        else if (base == 16U && byte >= 'A' && byte <= 'F')
        {
            value = static_cast<unsigned int>(byte - 'A') + 10U;
        }
        else
        {
            throw_parse_error(path, "invalid numeric XML entity reference");
        }
        if (codepoint > (0x10FFFFU - value) / base)
        {
            throw_parse_error(path, "numeric XML entity reference overflow");
        }
        codepoint = codepoint * base + value;
    }
    if (!valid_xml_codepoint(codepoint))
    {
        throw_parse_error(path, "invalid XML character reference");
    }
    index = semicolon + 1;
}

void validate_xml_lexical_content(
    std::string_view xml,
    const std::filesystem::path& path)
{
    for (std::size_t index = 0; index < xml.size();)
    {
        const std::string_view remaining = xml.substr(index);
        std::string_view terminator;
        if (remaining.compare(0, 4, "<!--") == 0)
        {
            terminator = "-->";
            index += 4;
        }
        else if (remaining.compare(0, 9, "<![CDATA[") == 0)
        {
            terminator = "]]>";
            index += 9;
        }
        else if (remaining.compare(0, 2, "<?") == 0)
        {
            terminator = "?>";
            index += 2;
        }
        if (!terminator.empty())
        {
            const std::size_t end = xml.find(terminator, index);
            if (end == std::string_view::npos)
            {
                throw_parse_error(path, "unterminated XML lexical section");
            }
            index = end + terminator.size();
            continue;
        }
        if (remaining.compare(0, 9, "<!DOCTYPE") == 0)
        {
            std::size_t cursor = index + 9;
            std::size_t bracket_depth = 0;
            char quote = '\0';
            for (; cursor < xml.size(); ++cursor)
            {
                const char value = xml[cursor];
                if (quote != '\0')
                {
                    if (value == quote)
                    {
                        quote = '\0';
                    }
                    continue;
                }
                if (value == '\'' || value == '"')
                {
                    quote = value;
                }
                else if (value == '[')
                {
                    ++bracket_depth;
                }
                else if (value == ']' && bracket_depth != 0)
                {
                    --bracket_depth;
                }
                else if (value == '>' && bracket_depth == 0)
                {
                    break;
                }
            }
            if (cursor == xml.size())
            {
                throw_parse_error(path, "unterminated XML document type");
            }
            const std::string_view declaration =
                xml.substr(index, cursor - index + 1);
            if (declaration.find("<!ENTITY") != std::string_view::npos)
            {
                throw_parse_error(
                    path, "custom XML entities are unsupported by this leaf");
            }
            index = cursor + 1;
            continue;
        }
        if (xml[index] == '&')
        {
            validate_entity_reference(xml, index, path);
            continue;
        }
        ++index;
    }
}

std::vector<char> normalize_xml_bytes(
    std::vector<char> bytes,
    XmlEncoding encoding,
    const std::filesystem::path& path)
{
    std::size_t begin = 0;
    if (bytes.size() >= 3 &&
        static_cast<unsigned char>(bytes[0]) == 0xEFU &&
        static_cast<unsigned char>(bytes[1]) == 0xBBU &&
        static_cast<unsigned char>(bytes[2]) == 0xBFU)
    {
        begin = 3;
    }

    if (std::find(bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                  bytes.end(), '\0') != bytes.end())
    {
        throw_parse_error(path, "XML input contains a NUL byte");
    }

    bool needs_copy = begin != 0;
    for (std::size_t index = begin; index < bytes.size() && !needs_copy; ++index)
    {
        const auto value = static_cast<unsigned char>(bytes[index]);
        needs_copy = value == '\r' ||
                     (encoding == XmlEncoding::iso_8859_1 && value >= 0x80U);
    }

    if (!needs_copy)
    {
        if (encoding == XmlEncoding::utf8 &&
            !valid_utf8(std::string_view(bytes.data(), bytes.size())))
        {
            throw_parse_error(path, "XML input is not valid XML UTF-8");
        }
        bytes.push_back('\0');
        return bytes;
    }

    std::vector<char> normalized;
    normalized.reserve((bytes.size() - begin) * 2 + 1);
    for (std::size_t index = begin; index < bytes.size(); ++index)
    {
        const auto value = static_cast<unsigned char>(bytes[index]);
        if (value == '\r')
        {
            if (index + 1 < bytes.size() && bytes[index + 1] == '\n')
            {
                ++index;
            }
            normalized.push_back('\n');
            continue;
        }
        if (encoding == XmlEncoding::iso_8859_1 && value >= 0x80U)
        {
            normalized.push_back(static_cast<char>(0xC0U | (value >> 6U)));
            normalized.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
            continue;
        }
        normalized.push_back(static_cast<char>(value));
    }
    if (!valid_utf8(std::string_view(normalized.data(), normalized.size())))
    {
        throw_parse_error(path, "XML input is not valid XML UTF-8");
    }
    normalized.push_back('\0');
    return normalized;
}

std::vector<char> read_xml_file(
    const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw_parse_error(path, "unable to open XML source");
    }
    const std::streampos end = input.tellg();
    if (end < 0)
    {
        throw_parse_error(path, "unable to determine XML source size");
    }
    const auto unsigned_size = static_cast<std::uintmax_t>(end);
    if (unsigned_size >
        static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw_parse_error(path, "XML source is too large");
    }
    const std::size_t size = static_cast<std::size_t>(unsigned_size);
    std::vector<char> bytes(size);
    input.seekg(0, std::ios::beg);
    if (size != 0)
    {
        input.read(bytes.data(), static_cast<std::streamsize>(size));
        if (!input || static_cast<std::size_t>(input.gcount()) != size)
        {
            throw_parse_error(path, "unable to read complete XML source");
        }
    }
    return bytes;
}

bool node_name_equals(
    const XmlNode& node,
    std::string_view expected) noexcept
{
    return node.type() == rapidxml::node_element &&
           node.name_size() == expected.size() &&
           std::memcmp(node.name(), expected.data(), expected.size()) == 0;
}

void collect_schema_elements(
    XmlNode* parent,
    XmlElementLists& elements)
{
    for (XmlNode* child = parent->first_node(); child != nullptr;
         child = child->next_sibling())
    {
        if (child->type() != rapidxml::node_element)
        {
            continue;
        }
        if (node_name_equals(*child, "node"))
        {
            elements.nodes.push_back(child);
        }
        else if (node_name_equals(*child, "link"))
        {
            elements.links.push_back(child);
        }
        collect_schema_elements(child, elements);
    }
}

XmlNode* first_descendant(
    XmlNode& parent,
    std::string_view name) noexcept
{
    for (XmlNode* child = parent.first_node(); child != nullptr;
         child = child->next_sibling())
    {
        if (child->type() != rapidxml::node_element)
        {
            continue;
        }
        if (node_name_equals(*child, name))
        {
            return child;
        }
        if (XmlNode* nested = first_descendant(*child, name))
        {
            return nested;
        }
    }
    return nullptr;
}

std::string first_child_data(
    XmlNode& element,
    std::string_view field,
    const std::filesystem::path& path)
{
    XmlNode* child = element.first_node();
    if (child == nullptr)
    {
        throw_schema_error(
            path, "missing first child data for " + std::string(field));
    }
    switch (child->type())
    {
    case rapidxml::node_data:
    case rapidxml::node_cdata:
    case rapidxml::node_comment:
    case rapidxml::node_pi:
        return std::string(child->value(), child->value_size());
    default:
        throw_schema_error(
            path, "first child has no data for " + std::string(field));
    }
}

std::string required_descendant_data(
    XmlNode& parent,
    std::string_view field,
    const std::filesystem::path& path)
{
    XmlNode* element = first_descendant(parent, field);
    if (element == nullptr)
    {
        throw_schema_error(
            path, "missing XML element " + std::string(field));
    }
    return first_child_data(*element, field, path);
}

std::array<std::string, 2> first_two_descendant_data(
    XmlNode& parent,
    std::string_view field,
    const std::filesystem::path& path)
{
    std::array<std::string, 2> values;
    std::size_t count = 0;
    const auto visit = [&](const auto& self, XmlNode& current) -> void
    {
        for (XmlNode* child = current.first_node(); child != nullptr && count < 2;
             child = child->next_sibling())
        {
            if (child->type() != rapidxml::node_element)
            {
                continue;
            }
            if (node_name_equals(*child, field))
            {
                values[count++] = first_child_data(*child, field, path);
            }
            if (count < 2)
            {
                self(self, *child);
            }
        }
    };
    visit(visit, parent);
    if (count != 2)
    {
        throw_schema_error(
            path, "fewer than two XML elements named " + std::string(field));
    }
    return values;
}

std::string required_attribute_value(
    XmlNode& element,
    std::string_view attribute_name,
    const std::filesystem::path& path)
{
    for (rapidxml::xml_attribute<char>* attribute = element.first_attribute();
         attribute != nullptr; attribute = attribute->next_attribute())
    {
        if (attribute->name_size() == attribute_name.size() &&
            std::memcmp(attribute->name(), attribute_name.data(),
                        attribute_name.size()) == 0)
        {
            return std::string(attribute->value(), attribute->value_size());
        }
    }
    throw_schema_error(
        path, "missing XML attribute " + std::string(attribute_name));
}

std::size_t available_cpu_count() noexcept
{
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        const int count = CPU_COUNT(&affinity);
        if (count > 0)
        {
            return static_cast<std::size_t>(count);
        }
    }
#endif
    return std::max<std::size_t>(
        1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
}

std::size_t xml_batch_worker_count(
    std::size_t requested,
    const std::vector<std::filesystem::path>& source_paths) noexcept
{
    const std::size_t count = source_paths.size();
    if (count == 0)
    {
        return 0;
    }
    if (requested == 0)
    {
        if (count == 1)
        {
            return 1;
        }

        std::size_t calibrated = 1;
        if (count == 2)
        {
            calibrated = 2;
        }
        else if (count >= 24)
        {
            calibrated = 5;
        }
        else
        {
            // The 2/4/8/16/32 sweep showed that small 132 KiB Brain documents
            // saturate two lanes through eight inputs and four lanes at 16,
            // while the 256-node/2,048-link corpus benefits from four lanes
            // at 4..8 inputs and eight lanes at 16. One representative
            // metadata read avoids a syscall per document.
            std::error_code error;
            const std::uintmax_t bytes =
                std::filesystem::file_size(source_paths.front(), error);
            const bool large_document = !error && bytes >= 196608U;
            if (count >= 9)
            {
                calibrated = large_document ? 8 : 4;
            }
            else
            {
                calibrated = large_document ? 4 : (count >= 8 ? 3 : 2);
            }
        }
        if (calibrated <= 1)
        {
            return 1;
        }
        return std::min({calibrated, count, available_cpu_count()});
    }
    return std::min({requested, count, available_cpu_count()});
}

XmlAttrIds resolve_xml_attr_ids(const Graph& graph)
{
    return {
        graph.attr_id("name"),
        graph.attr_id("label"),
        graph.attr_id("x"),
        graph.attr_id("y"),
        graph.attr_id("source_label"),
        graph.attr_id("target_label"),
        graph.attr_id("capacity_st"),
        graph.attr_id("capacity_ts"),
        graph.attr_id("cost_st"),
        graph.attr_id("cost_ts")};
}

const std::string& string_attribute(
    const AttrMap& attributes,
    AttrId id)
{
    const AttrValue* value = attributes.find(id);
    if (value == nullptr)
    {
        throw std::runtime_error("required XML graph attribute is absent");
    }
    const auto* text = std::get_if<std::string>(value);
    if (text == nullptr)
    {
        throw std::runtime_error("required XML graph attribute is not a string");
    }
    return *text;
}

void append_unsigned_decimal(
    std::string& output,
    std::uint64_t value)
{
    std::array<char, 32> buffer{};
    const auto converted =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (converted.ec != std::errc{})
    {
        throw std::runtime_error("integer formatting failed");
    }
    output.append(buffer.data(), converted.ptr);
}

void append_gml_string(
    std::string& output,
    std::string_view value)
{
    output.push_back('"');
    std::size_t index = 0;
    while (index < value.size())
    {
        const std::uint32_t codepoint = decode_utf8_codepoint(value, index);
        if (codepoint >= 0x20U && codepoint <= 0x7EU &&
            codepoint != static_cast<std::uint32_t>('&') &&
            codepoint != static_cast<std::uint32_t>('"'))
        {
            output.push_back(static_cast<char>(codepoint));
        }
        else
        {
            output += "&#";
            append_unsigned_decimal(output, codepoint);
            output.push_back(';');
        }
    }
    output.push_back('"');
}

void append_key_string_line(
    std::string& output,
    std::string_view indent,
    std::string_view key,
    std::string_view value)
{
    output.append(indent);
    output.append(key);
    output.push_back(' ');
    append_gml_string(output, value);
    output.push_back('\n');
}

void write_networkx_compatible_gml(
    const Graph& graph,
    const std::filesystem::path& target_path)
{
    const std::string extension = target_path.extension().string();
    if (extension == ".gz" || extension == ".bz2")
    {
        throw std::runtime_error(
            "compressed GML targets are outside the native XML leaf");
    }

    const XmlAttrIds ids = resolve_xml_attr_ids(graph);
    const std::string& graph_name =
        string_attribute(graph.graph_attrs(), ids.name);
    if (!valid_utf8(graph_name))
    {
        throw std::runtime_error("topology name is not valid XML-compatible UTF-8");
    }
    std::string output;
    output.reserve(graph.num_nodes() * 96 + graph.num_edges() * 256 + 32);
    output += "graph [\n";
    append_key_string_line(output, "  ", "name", graph_name);

    for (std::size_t node_index = 0; node_index < graph.num_nodes(); ++node_index)
    {
        const Vertex node = static_cast<Vertex>(node_index);
        const AttrMap& attributes = graph.node_attrs(node);
        output += "  node [\n    id ";
        append_unsigned_decimal(output, node_index);
        output += "\n    label \"";
        append_unsigned_decimal(output, node_index);
        output += "\"\n";
        append_key_string_line(
            output, "    ", "x", string_attribute(attributes, ids.x));
        append_key_string_line(
            output, "    ", "y", string_attribute(attributes, ids.y));
        output += "  ]\n";
    }

    for (const auto edge_data : graph.edge_view().data())
    {
        output += "  edge [\n    source ";
        append_unsigned_decimal(output, edge_data.source);
        output += "\n    target ";
        append_unsigned_decimal(output, edge_data.target);
        output.push_back('\n');
        append_key_string_line(
            output, "    ", "label",
            string_attribute(edge_data.attrs, ids.label));
        append_key_string_line(
            output, "    ", "source_label",
            string_attribute(edge_data.attrs, ids.source_label));
        append_key_string_line(
            output, "    ", "target_label",
            string_attribute(edge_data.attrs, ids.target_label));
        append_key_string_line(
            output, "    ", "capacity_st",
            string_attribute(edge_data.attrs, ids.capacity_st));
        append_key_string_line(
            output, "    ", "capacity_ts",
            string_attribute(edge_data.attrs, ids.capacity_ts));
        append_key_string_line(
            output, "    ", "cost_st",
            string_attribute(edge_data.attrs, ids.cost_st));
        append_key_string_line(
            output, "    ", "cost_ts",
            string_attribute(edge_data.attrs, ids.cost_ts));
        output += "  ]\n";
    }
    output += "]\n";

    std::ofstream target(
        target_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!target)
    {
        throw std::runtime_error("unable to open GML target");
    }
    target.write(output.data(), static_cast<std::streamsize>(output.size()));
    target.close();
    if (!target)
    {
        throw std::runtime_error("unable to write complete GML target");
    }
}

bool paths_alias(
    const std::filesystem::path& source,
    const std::filesystem::path& target) noexcept
{
    std::error_code error;
    if (std::filesystem::exists(target, error) && !error)
    {
        if (std::filesystem::equivalent(source, target, error) && !error)
        {
            return true;
        }
    }
    error.clear();
    const std::filesystem::path canonical_source =
        std::filesystem::weakly_canonical(source, error);
    if (error)
    {
        return false;
    }
    error.clear();
    const std::filesystem::path canonical_target =
        std::filesystem::weakly_canonical(target, error);
    return !error && canonical_source == canonical_target;
}

} // namespace

ParsedXmlTopology parse_sndlib_xml(
    const std::filesystem::path& source_path)
{
    try
    {
        std::vector<char> raw = read_xml_file(source_path);
        const XmlEncoding encoding = detect_xml_encoding(
            std::string_view(raw.data(), raw.size()), source_path);
        std::vector<char> buffer =
            normalize_xml_bytes(std::move(raw), encoding, source_path);
        validate_xml_lexical_content(
            std::string_view(buffer.data(), buffer.size() - 1), source_path);

        rapidxml::xml_document<char> document;
        constexpr int flags =
            rapidxml::parse_declaration_node |
            rapidxml::parse_comment_nodes |
            rapidxml::parse_doctype_node |
            rapidxml::parse_pi_nodes |
            rapidxml::parse_validate_closing_tags;
        try
        {
            document.parse<flags>(buffer.data());
        }
        catch (const rapidxml::parse_error& error)
        {
            throw_parse_error(source_path, error.what());
        }

        XmlElementLists elements;
        collect_schema_elements(&document, elements);

        ParsedXmlTopology topology;
        topology.nodes.reserve(elements.nodes.size());
        topology.edges.reserve(elements.links.size());

        // Preserve Python staging: every node record is extracted before any
        // edge record, and no graph exists until both arrays are complete.
        for (XmlNode* node : elements.nodes)
        {
            topology.nodes.push_back({
                required_attribute_value(*node, "id", source_path),
                required_descendant_data(*node, "x", source_path),
                required_descendant_data(*node, "y", source_path)});
        }
        for (XmlNode* link : elements.links)
        {
            std::array<std::string, 2> capacity =
                first_two_descendant_data(*link, "capacity", source_path);
            XmlEdgeRecord edge{
                required_attribute_value(*link, "id", source_path),
                required_descendant_data(*link, "source", source_path),
                required_descendant_data(*link, "target", source_path),
                std::move(capacity[0]),
                std::move(capacity[1]),
                {},
                {}};
            // This intentionally locks the Python bug: cost XML elements are
            // ignored and the two capacity strings are reused as costs.
            edge.cost_st = edge.capacity_st;
            edge.cost_ts = edge.capacity_ts;
            topology.edges.push_back(std::move(edge));
        }
        return topology;
    }
    catch (const std::bad_alloc&)
    {
        throw;
    }
    catch (const DatasetException&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw_parse_error(source_path, error.what());
    }
}

std::vector<ParsedXmlTopology> parse_sndlib_xml_batch(
    const std::vector<std::filesystem::path>& source_paths,
    std::size_t workers)
{
    std::vector<ParsedXmlTopology> results(source_paths.size());
    if (source_paths.empty())
    {
        return results;
    }
    const std::size_t worker_count =
        xml_batch_worker_count(workers, source_paths);
    const auto parse_range = [&](std::size_t worker,
                                 std::vector<std::exception_ptr>& failures)
    {
        const std::size_t begin =
            source_paths.size() * worker / worker_count;
        const std::size_t end =
            source_paths.size() * (worker + 1) / worker_count;
        for (std::size_t index = begin; index < end; ++index)
        {
            try
            {
                results[index] = parse_sndlib_xml(source_paths[index]);
            }
            catch (const DatasetException& error)
            {
                failures[index] = std::make_exception_ptr(DatasetException(
                    error.code(), error.operation(), error.what(), index,
                    error.path()));
            }
            catch (...)
            {
                failures[index] = std::current_exception();
            }
        }
    };

    std::vector<std::exception_ptr> failures(source_paths.size());
    if (worker_count <= 1)
    {
        parse_range(0, failures);
    }
    else
    {
        std::vector<std::thread> threads;
        threads.reserve(worker_count - 1);
        try
        {
            for (std::size_t worker = 1; worker < worker_count; ++worker)
            {
                threads.emplace_back(parse_range, worker, std::ref(failures));
            }
        }
        catch (...)
        {
            for (std::thread& thread : threads)
            {
                thread.join();
            }
            throw;
        }
        parse_range(0, failures);
        for (std::thread& thread : threads)
        {
            thread.join();
        }
    }

    for (const std::exception_ptr& failure : failures)
    {
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }
    return results;
}

Graph materialize_xml_topology(
    std::string_view topology_name,
    const ParsedXmlTopology& topology)
{
    try
    {
        Graph graph;
        const XmlAttrIds ids = resolve_xml_attr_ids(graph);
        std::unordered_map<std::string_view, Vertex> label_to_id;
        label_to_id.reserve(topology.nodes.size());

        for (const XmlNodeRecord& node : topology.nodes)
        {
            const Vertex vertex = graph.add_node();
            AttrMap& attributes = graph.node_attrs(vertex);
            attributes.set(ids.label, node.label);
            attributes.set(ids.x, node.x);
            attributes.set(ids.y, node.y);
            label_to_id.insert_or_assign(node.label, vertex);
        }

        for (const XmlEdgeRecord& edge : topology.edges)
        {
            // Dynamic labels are each resolved exactly once. All graph fields
            // below use the fixed IDs resolved before entering this hot loop.
            const auto source = label_to_id.find(edge.source_label);
            const auto target = label_to_id.find(edge.target_label);
            if (source == label_to_id.end() || target == label_to_id.end())
            {
                throw DatasetException(
                    DatasetErrorCode::unknown_endpoint,
                    DatasetOperation::materialize_graph,
                    "XML link references an unknown endpoint");
            }
            const Edge graph_edge = graph.add_edge(source->second, target->second);
            AttrMap& attributes = graph.edge_attrs(graph_edge);
            attributes.set(ids.label, edge.label);
            attributes.set(ids.source_label, edge.source_label);
            attributes.set(ids.target_label, edge.target_label);
            attributes.set(ids.capacity_st, edge.capacity_st);
            attributes.set(ids.capacity_ts, edge.capacity_ts);
            attributes.set(ids.cost_st, edge.cost_st);
            attributes.set(ids.cost_ts, edge.cost_ts);
        }
        graph.graph_attrs().set(ids.name, std::string(topology_name));
        return graph;
    }
    catch (const std::bad_alloc&)
    {
        throw;
    }
    catch (const DatasetException&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw DatasetException(
            DatasetErrorCode::graph_materialization_failure,
            DatasetOperation::materialize_graph,
            error.what());
    }
}

Graph preprocess_xml(const XmlTopologyRequest& request)
{
    ParsedXmlTopology topology = parse_sndlib_xml(request.xml_source_path);
    Graph graph = materialize_xml_topology(request.topology_name, topology);
    try
    {
        if (paths_alias(request.xml_source_path, request.gml_target_path))
        {
            throw std::runtime_error("XML source and GML target alias");
        }
        write_networkx_compatible_gml(graph, request.gml_target_path);
    }
    catch (const std::bad_alloc&)
    {
        throw;
    }
    catch (const DatasetException&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw DatasetException(
            DatasetErrorCode::gml_write_failure,
            DatasetOperation::write_gml,
            error.what(),
            std::nullopt,
            request.gml_target_path);
    }
    return graph;
}

} // namespace virne::utils
