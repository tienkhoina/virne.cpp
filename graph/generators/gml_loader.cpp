#include "gml_loader.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace
{
std::string trim(
    const std::string& s)
{
    size_t b = 0;
    while (b < s.size() &&
           std::isspace(
               static_cast<unsigned char>(
                   s[b])))
    {
        ++b;
    }

    size_t e = s.size();
    while (e > b &&
           std::isspace(
               static_cast<unsigned char>(
                   s[e - 1])))
    {
        --e;
    }

    return s.substr(
        b,
        e - b);
}

std::string unquote(
    std::string s)
{
    s = trim(s);

    if (s.size() >= 2 &&
        ((s.front() == '"' &&
          s.back() == '"') ||
         (s.front() == '\'' &&
          s.back() == '\'')))
    {
        s = s.substr(
            1,
            s.size() - 2);
    }

    return s;
}

AttrValue parse_attr_value(
    const std::string& token)
{
    const std::string s =
        unquote(token);

    if (s == "true")
    {
        return true;
    }

    if (s == "false")
    {
        return false;
    }

    try
    {
        size_t pos = 0;
        long long iv =
            std::stoll(
                s,
                &pos);

        if (pos == s.size())
        {
            return static_cast<int64_t>(
                iv);
        }
    }
    catch (...)
    {
    }

    try
    {
        size_t pos = 0;
        double dv =
            std::stod(
                s,
                &pos);

        if (pos == s.size())
        {
            return dv;
        }
    }
    catch (...)
    {
    }

    return s;
}

bool is_block_start(
    const std::string& line,
    const std::string& key)
{
    return line.find(key) !=
           std::string::npos;
}
}

Graph
GmlLoader::load(
    const std::string& path)
{
    Graph g;

    std::ifstream fin(path);

    if (!fin)
    {
        throw std::runtime_error(
            "cannot open gml file");
    }

    std::unordered_map<
        int,
        Vertex> id_to_vertex;

    std::string line;

    const AttrId id_attr =
        g.attr_id("id");
    const AttrId source_attr =
        g.attr_id("source");
    const AttrId target_attr =
        g.attr_id("target");

    while (std::getline(
        fin,
        line))
    {
        line = trim(line);

        if (line.empty())
        {
            continue;
        }

        //
        // node
        //
        if (is_block_start(
                line,
                "node"))
        {
            int node_id = -1;
            std::vector<
                std::pair<
                    std::string,
                    AttrValue>>
                pending_attrs;

            while (std::getline(
                fin,
                line))
            {
                line = trim(line);

                if (line.empty())
                {
                    continue;
                }

                if (line.find(']') !=
                    std::string::npos)
                {
                    break;
                }

                std::stringstream ss(line);

                std::string key;
                ss >> key;

                std::string value;
                std::getline(ss, value);
                value = trim(value);

                if (key == "id")
                {
                    node_id =
                        std::stoi(value);
                    continue;
                }

                pending_attrs.emplace_back(
                    key,
                    parse_attr_value(value));
            }

            if (node_id >= 0)
            {
                auto v =
                    g.add_node();

                id_to_vertex[node_id] = v;

                g.node_attrs(v).set(
                    id_attr,
                    static_cast<int64_t>(
                        node_id));

                for (const auto& kv :
                     pending_attrs)
                {
                    AttrId attr =
                        g.attr_id(
                            kv.first);

                    g.node_attrs(v).set(
                        attr,
                        kv.second);
                }
            }
        }

        //
        // edge
        //
        else if (is_block_start(
                     line,
                     "edge"))
        {
            int source = -1;
            int target = -1;
            std::vector<
                std::pair<
                    std::string,
                    AttrValue>>
                pending_attrs;

            while (std::getline(
                fin,
                line))
            {
                line = trim(line);

                if (line.empty())
                {
                    continue;
                }

                if (line.find(']') !=
                    std::string::npos)
                {
                    break;
                }

                std::stringstream ss(line);

                std::string key;
                ss >> key;

                std::string value;
                std::getline(ss, value);
                value = trim(value);

                if (key == "source")
                {
                    source =
                        std::stoi(value);
                    continue;
                }
                else if (key == "target")
                {
                    target =
                        std::stoi(value);
                    continue;
                }

                pending_attrs.emplace_back(
                    key,
                    parse_attr_value(value));
            }

            if (source >= 0 &&
                target >= 0)
            {
                auto e =
                    g.add_edge(
                        id_to_vertex.at(source),
                        id_to_vertex.at(target));

                g.edge_attrs(e).set(
                    source_attr,
                    static_cast<int64_t>(
                        source));

                g.edge_attrs(e).set(
                    target_attr,
                    static_cast<int64_t>(
                        target));

                for (const auto& kv :
                     pending_attrs)
                {
                    AttrId attr =
                        g.attr_id(
                            kv.first);

                    g.edge_attrs(e).set(
                        attr,
                        kv.second);
                }
            }
        }
    }

    return g;
}