#include "graph_saver.h"

#include "../graph.h"
#include "../attribute.h"

#include <fstream>
#include <variant>

namespace
{
    void write_attr(
        std::ostream& os,
        const std::string& name,
        const AttrValue& value)
    {
        os << "    "
           << name
           << " ";

        std::visit(
            [&](const auto& v)
            {
                using T =
                    std::decay_t<decltype(v)>;

                if constexpr (
                    std::is_same_v<T,
                                   std::string>)
                {
                    os << "\""
                       << v
                       << "\"";
                }
                else if constexpr (
                    std::is_same_v<T,
                                bool>)
                {
                    os << (v
                            ? "true"
                            : "false");
                }
                else
                {
                    os << v;
                }
            },
            value);

        os << "\n";
    }
}

namespace GraphSaver
{
    void save_gml(
        const Graph& graph,
        const std::string& path)
    {
        std::ofstream out(path);

        if (!out)
        {
            throw std::runtime_error(
                "Cannot open file: " +
                path);
        }

        out << "graph [\n";
        out << "  directed 0\n";

        auto [vit, vend] =
            graph.nodes();

        for (; vit != vend; ++vit)
        {
            auto v = *vit;

            out << "  node [\n";
            out << "    id "
                << v
                << "\n";

            const auto& attrs =
                graph.node_attrs(v);

            for (auto attr_id :
                 attrs.attribute_ids())
            {
                auto value =
                    attrs.find(attr_id);

                if (!value)
                {
                    continue;
                }

                write_attr(
                    out,
                    std::string(
                        graph.attr_name(
                            attr_id)),
                    *value);
            }

            out << "  ]\n";
        }

        auto [eit, eend] =
            graph.edges();

        for (; eit != eend; ++eit)
        {
            auto e = *eit;

            out << "  edge [\n";

            out << "    source "
                << graph.source(e)
                << "\n";

            out << "    target "
                << graph.target(e)
                << "\n";

            const auto& attrs =
                graph.edge_attrs(e);

            for (auto attr_id :
                 attrs.attribute_ids())
            {
                auto value =
                    attrs.find(attr_id);

                if (!value)
                {
                    continue;
                }

                write_attr(
                    out,
                    std::string(
                        graph.attr_name(
                            attr_id)),
                    *value);
            }

            out << "  ]\n";
        }

        out << "]\n";
    }
}

namespace nx
{
    inline void write_gml(
        const Graph& graph,
        const std::string& path)
    {
        GraphSaver::save_gml(
            graph,
            path);
    }
}