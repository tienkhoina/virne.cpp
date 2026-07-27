#include "graph_saver.h"

#include "../graph.h"
#include "../attribute.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

namespace
{

constexpr std::string_view kListStart =
    "_networkx_list_start";

bool valid_gml_key(
    std::string_view key)
{
    const auto ascii_alpha =
        [](char value)
        {
            return (value >= 'A' && value <= 'Z') ||
                   (value >= 'a' && value <= 'z');
        };
    const auto ascii_digit =
        [](char value)
        {
            return value >= '0' && value <= '9';
        };

    if (key.empty() || !ascii_alpha(key.front()))
    {
        return false;
    }
    for (const char character : key)
    {
        if (!ascii_alpha(character) &&
            !ascii_digit(character) &&
            character != '_')
        {
            return false;
        }
    }
    return true;
}

std::string format_gml_real(
    double value)
{
    if (std::isnan(value))
    {
        return "NAN";
    }
    if (std::isinf(value))
    {
        return std::signbit(value) ? "-INF" : "+INF";
    }

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(17) << value;
    std::string text = stream.str();
    if (text.find('.') == std::string::npos)
    {
        const size_t exponent = text.find_first_of("eE");
        if (exponent == std::string::npos)
        {
            text += ".0";
        }
        else
        {
            // GML requires a decimal point in every real literal.  Without
            // this, e.g. 1e-10 is tokenized by NetworkX as integer 1 plus an
            // unrelated key named e.
            text.insert(exponent, ".");
        }
    }
    return text;
}

std::string encode_gml_string(
    std::string_view input)
{
    std::string output;
    output.reserve(input.size());

    for (size_t index = 0;
         index < input.size();)
    {
        const unsigned char first =
            static_cast<unsigned char>(input[index]);
        uint32_t codepoint = first;
        size_t width = 1;

        if ((first & 0xE0U) == 0xC0U &&
            index + 1 < input.size())
        {
            codepoint = first & 0x1FU;
            width = 2;
        }
        else if ((first & 0xF0U) == 0xE0U &&
                 index + 2 < input.size())
        {
            codepoint = first & 0x0FU;
            width = 3;
        }
        else if ((first & 0xF8U) == 0xF0U &&
                 index + 3 < input.size())
        {
            codepoint = first & 0x07U;
            width = 4;
        }

        bool valid = true;
        for (size_t part = 1;
             part < width;
             ++part)
        {
            const unsigned char continuation =
                static_cast<unsigned char>(
                    input[index + part]);
            if ((continuation & 0xC0U) != 0x80U)
            {
                valid = false;
                break;
            }
            codepoint =
                (codepoint << 6U) |
                (continuation & 0x3FU);
        }

        if (!valid)
        {
            codepoint = first;
            width = 1;
        }

        if (codepoint < 32U ||
            codepoint > 126U ||
            codepoint == static_cast<uint32_t>('"') ||
            codepoint == static_cast<uint32_t>('&'))
        {
            output += "&#";
            output += std::to_string(codepoint);
            output += ';';
        }
        else
        {
            output.push_back(
                static_cast<char>(codepoint));
        }
        index += width;
    }
    return output;
}

void write_named_value(
    std::ostream& os,
    const std::string& name,
    const AttrValue& value,
    size_t indent,
    bool in_list = false,
    size_t depth = 0)
{
    if (depth > 256)
    {
        throw std::invalid_argument(
            "GML attribute metadata is cyclic or exceeds depth 256");
    }
    if (!valid_gml_key(name))
    {
        throw std::invalid_argument(
            "Invalid GML attribute name: " + name);
    }

    if (const AttrList* list = attr_list(value))
    {
        if (in_list)
        {
            throw std::invalid_argument(
                "Nested GML lists are not supported by NetworkX");
        }
        if (list->values.empty())
        {
            write_named_value(
                os,
                name,
                AttrValue{std::string("[]")},
                indent,
                true,
                depth + 1);
            return;
        }
        if (list->values.size() == 1)
        {
            write_named_value(
                os,
                name,
                AttrValue{std::string(kListStart)},
                indent,
                true,
                depth + 1);
        }
        for (const AttrValue& item : list->values)
        {
            write_named_value(
                os, name, item, indent, true, depth + 1);
        }
        return;
    }

    const std::string padding(indent, ' ');
    if (const AttrObject* object = attr_object(value))
    {
        os << padding << name << " [\n";
        for (const auto& [key, child] : object->entries)
        {
            write_named_value(
                os, key, child, indent + 2, false, depth + 1);
        }
        os << padding << "]\n";
        return;
    }

    os << padding << name << ' ';
    if (const auto* string =
            std::get_if<std::string>(&value))
    {
        os << '"' << encode_gml_string(*string) << '"';
    }
    else if (const auto* boolean =
                 std::get_if<bool>(&value))
    {
        os << (*boolean ? 1 : 0);
    }
    else if (const auto* integer =
                 std::get_if<int64_t>(&value))
    {
        os << *integer;
    }
    else if (const auto* real =
                 std::get_if<double>(&value))
    {
        os << format_gml_real(*real);
    }
    else
    {
        throw std::invalid_argument(
            "Cannot serialize a null structured GML attribute");
    }
    os << '\n';
}

template <typename GraphType>
void save_gml_impl(
    const GraphType& graph,
    const std::string& path,
    bool directed)
{
    // Serialize and validate completely before touching the destination.  A
    // bad attribute must not truncate an existing valid file.
    std::ostringstream out;
    out.imbue(std::locale::classic());

    out << "graph [\n";
    out << "  directed " << (directed ? 1 : 0) << "\n";

    const auto& graph_attributes = graph.graph_attrs();
    for (AttrId attr_id : graph_attributes.attribute_ids())
    {
        const std::string name(graph.attr_name(attr_id));
        if (name == "directed" ||
            name == "multigraph" ||
            name == "node" ||
            name == "edge")
        {
            throw std::invalid_argument(
                "Reserved GML graph attribute name: " + name);
        }
        if (const AttrValue* value =
                graph_attributes.find(attr_id))
        {
            write_named_value(
                out,
                name,
                *value,
                2);
        }
    }

    std::vector<int64_t> node_ids(
        graph.num_nodes());
    std::unordered_set<int64_t> unique_ids;
    const auto id_attr =
        graph.attribute_registry().find("id");
    auto [id_vertex, id_vertex_end] = graph.nodes();
    for (; id_vertex != id_vertex_end; ++id_vertex)
    {
        const Vertex vertex = *id_vertex;
        int64_t serialized_id =
            static_cast<int64_t>(vertex);
        if (id_attr.has_value())
        {
            if (const AttrValue* value =
                    graph.node_attrs(vertex).find(*id_attr))
            {
                const auto* integer =
                    std::get_if<int64_t>(value);
                if (integer == nullptr || *integer < 0)
                {
                    throw std::invalid_argument(
                        "GML node 'id' attribute must be a non-negative integer");
                }
                serialized_id = *integer;
            }
        }
        if (!unique_ids.insert(serialized_id).second)
        {
            throw std::invalid_argument(
                "GML node ids must be unique");
        }
        node_ids[vertex] = serialized_id;
    }

    auto [vit, vend] = graph.nodes();
    for (; vit != vend; ++vit)
    {
        const Vertex v = *vit;
        out << "  node [\n";
        out << "    id " << node_ids[v] << "\n";
        const auto& attrs = graph.node_attrs(v);
        for (AttrId attr_id : attrs.attribute_ids())
        {
            if (graph.attr_name(attr_id) == "id")
            {
                continue;
            }
            if (const AttrValue* value = attrs.find(attr_id))
            {
                write_named_value(
                    out,
                    std::string(graph.attr_name(attr_id)),
                    *value,
                    4);
            }
        }
        out << "  ]\n";
    }

    auto [eit, eend] = graph.edges();
    for (; eit != eend; ++eit)
    {
        const auto e = *eit;
        out << "  edge [\n";
        out << "    source " << node_ids[graph.source(e)] << "\n";
        out << "    target " << node_ids[graph.target(e)] << "\n";
        const auto& attrs = graph.edge_attrs(e);
        for (AttrId attr_id : attrs.attribute_ids())
        {
            const std::string_view name = graph.attr_name(attr_id);
            if (name == "source" || name == "target")
            {
                continue;
            }
            if (const AttrValue* value = attrs.find(attr_id))
            {
                write_named_value(
                    out,
                    std::string(graph.attr_name(attr_id)),
                    *value,
                    4);
            }
        }
        out << "  ]\n";
    }
    out << "]\n";

    const std::string contents = out.str();
    std::ofstream file(
        path,
        std::ios::binary | std::ios::trunc);
    if (!file)
    {
        throw std::runtime_error("Cannot open file: " + path);
    }
    file.write(
        contents.data(),
        static_cast<std::streamsize>(contents.size()));
    file.flush();
    if (!file)
    {
        throw std::runtime_error("Cannot write GML file: " + path);
    }
}

} // namespace

namespace GraphSaver
{

void save_gml(const Graph& graph, const std::string& path)
{
    save_gml_impl(graph, path, false);
}

void save_gml(const DiGraph& graph, const std::string& path)
{
    save_gml_impl(graph, path, true);
}

} // namespace GraphSaver

namespace nx
{

void write_gml(const Graph& graph, const std::string& path)
{
    GraphSaver::save_gml(graph, path);
}

void write_gml(const DiGraph& graph, const std::string& path)
{
    GraphSaver::save_gml(graph, path);
}

} // namespace nx
