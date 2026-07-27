#include "gml_loader.h"
#include "gml_html_entities.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace
{

enum class TokenKind
{
    word,
    string,
    left_bracket,
    right_bracket,
    end
};

struct Token
{
    TokenKind kind = TokenKind::end;
    std::string text;
    size_t offset = 0;
};

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

void append_utf8(
    std::string& output,
    uint32_t codepoint)
{
    if (codepoint <= 0x7FU)
    {
        output.push_back(static_cast<char>(codepoint));
    }
    else if (codepoint <= 0x7FFU)
    {
        output.push_back(
            static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(
            static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    else if (codepoint <= 0xFFFFU)
    {
        output.push_back(
            static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(
            static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(
            static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    else if (codepoint <= 0x10FFFFU)
    {
        output.push_back(
            static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(
            static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(
            static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(
            static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    else
    {
        throw std::runtime_error(
            "GML string contains an invalid Unicode code point");
    }
}

std::string decode_gml_string(
    std::string_view input)
{
    std::string output;
    output.reserve(input.size());

    for (size_t index = 0;
         index < input.size();)
    {
        if (input[index] == '&')
        {
            const size_t semicolon =
                input.find(';', index + 1);
            if (semicolon != std::string_view::npos)
            {
                const std::string_view entity =
                    input.substr(
                        index + 1,
                        semicolon - index - 1);
                uint32_t codepoint = 0;
                bool decoded = false;
                if (virne_gml_detail::decode_named_entity(
                        entity,
                        codepoint))
                {
                    decoded = true;
                }
                else if (!entity.empty() &&
                         entity.front() == '#')
                {
                    const bool hexadecimal =
                        entity.size() > 2 &&
                        entity[1] == 'x';
                    const std::string number(
                        entity.substr(hexadecimal ? 2 : 1));
                    bool valid_number = !number.empty();
                    for (const char digit : number)
                    {
                        const bool decimal_digit =
                            digit >= '0' && digit <= '9';
                        const bool hexadecimal_digit =
                            decimal_digit ||
                            (digit >= 'a' && digit <= 'f') ||
                            (digit >= 'A' && digit <= 'F');
                        valid_number = valid_number &&
                            (hexadecimal
                                 ? hexadecimal_digit
                                 : decimal_digit);
                    }
                    if (valid_number)
                    {
                        char* end = nullptr;
                        errno = 0;
                        const unsigned long parsed =
                            std::strtoul(
                                number.c_str(),
                                &end,
                                hexadecimal ? 16 : 10);
                        if (errno == 0 &&
                            end != number.c_str() &&
                            *end == '\0' &&
                            parsed <= 0x10FFFFUL)
                        {
                            codepoint = static_cast<uint32_t>(parsed);
                            decoded = true;
                        }
                    }
                }

                if (decoded)
                {
                    append_utf8(output, codepoint);
                    index = semicolon + 1;
                    continue;
                }
            }
        }
        output.push_back(input[index]);
        ++index;
    }
    return output;
}

class Lexer
{
public:
    explicit Lexer(std::string source)
        : source_(std::move(source))
    {
    }

    Token peek()
    {
        if (!has_lookahead_)
        {
            lookahead_ = read();
            has_lookahead_ = true;
        }
        return lookahead_;
    }

    Token next()
    {
        if (has_lookahead_)
        {
            has_lookahead_ = false;
            return std::move(lookahead_);
        }
        return read();
    }

private:
    std::string source_;
    size_t offset_ = 0;
    bool has_lookahead_ = false;
    Token lookahead_;

    void skip_space_and_comments()
    {
        while (offset_ < source_.size())
        {
            const unsigned char character =
                static_cast<unsigned char>(source_[offset_]);
            if (std::isspace(character))
            {
                ++offset_;
                continue;
            }
            if (source_[offset_] == '#')
            {
                while (offset_ < source_.size() &&
                       source_[offset_] != '\n')
                {
                    ++offset_;
                }
                continue;
            }
            break;
        }
    }

    Token read_string()
    {
        const size_t start = offset_++;
        std::string raw;
        bool closed = false;
        while (offset_ < source_.size())
        {
            const char character = source_[offset_++];
            if (character == '"')
            {
                closed = true;
                break;
            }
            // NetworkX/GML does not interpret C-style backslash escapes.
            // Quotes and non-ASCII bytes are represented as XML entities,
            // so a literal backslash must survive unchanged.
            raw.push_back(character);
        }
        if (!closed)
        {
            throw std::runtime_error(
                "Unterminated GML string at byte " +
                std::to_string(start));
        }
        return {
            TokenKind::string,
            decode_gml_string(raw),
            start};
    }

    Token read()
    {
        skip_space_and_comments();
        if (offset_ >= source_.size())
        {
            return {TokenKind::end, {}, offset_};
        }

        const size_t start = offset_;
        const char character = source_[offset_];
        if (character == '[')
        {
            ++offset_;
            return {TokenKind::left_bracket, "[", start};
        }
        if (character == ']')
        {
            ++offset_;
            return {TokenKind::right_bracket, "]", start};
        }
        if (character == '"')
        {
            return read_string();
        }

        while (offset_ < source_.size())
        {
            const char current = source_[offset_];
            if (std::isspace(
                    static_cast<unsigned char>(current)) ||
                current == '[' || current == ']' ||
                current == '#')
            {
                break;
            }
            ++offset_;
        }
        if (start == offset_)
        {
            throw std::runtime_error(
                "Invalid GML token at byte " +
                std::to_string(start));
        }
        return {
            TokenKind::word,
            source_.substr(start, offset_ - start),
            start};
    }
};

[[noreturn]] void syntax_error(
    const Token& token,
    const std::string& message)
{
    throw std::runtime_error(
        "GML parse error at byte " +
        std::to_string(token.offset) +
        ": " + message);
}

bool valid_real_literal(
    std::string_view text)
{
    size_t begin = 0;
    if (!text.empty() &&
        (text.front() == '+' || text.front() == '-'))
    {
        begin = 1;
    }
    if (text.substr(begin) == "INF")
    {
        return true;
    }

    const size_t exponent = text.find_first_of("eE", begin);
    const size_t mantissa_end =
        exponent == std::string_view::npos
            ? text.size()
            : exponent;
    if (exponent != std::string_view::npos)
    {
        size_t digit = exponent + 1;
        if (digit < text.size() &&
            (text[digit] == '+' || text[digit] == '-'))
        {
            ++digit;
        }
        if (digit == text.size())
        {
            return false;
        }
        for (; digit < text.size(); ++digit)
        {
            if (text[digit] < '0' || text[digit] > '9')
            {
                return false;
            }
        }
    }

    const size_t decimal = text.find('.', begin);
    if (decimal == std::string_view::npos ||
        decimal >= mantissa_end ||
        text.find('.', decimal + 1) != std::string_view::npos)
    {
        return false;
    }
    bool has_digit = false;
    for (size_t index = begin; index < mantissa_end; ++index)
    {
        if (index == decimal)
        {
            continue;
        }
        if (text[index] < '0' || text[index] > '9')
        {
            return false;
        }
        has_digit = true;
    }
    return has_digit;
}

AttrValue parse_scalar(
    const Token& token)
{
    if (token.kind == TokenKind::string)
    {
        return token.text;
    }
    if (token.kind != TokenKind::word)
    {
        syntax_error(token, "expected a scalar value");
    }
    size_t integer_start = 0;
    if (!token.text.empty() &&
        (token.text.front() == '+' ||
         token.text.front() == '-'))
    {
        integer_start = 1;
    }
    bool integer_literal =
        integer_start < token.text.size();
    for (size_t index = integer_start;
         index < token.text.size();
         ++index)
    {
        integer_literal = integer_literal &&
            token.text[index] >= '0' &&
            token.text[index] <= '9';
    }

    char* integer_end = nullptr;
    errno = 0;
    const long long integer =
        std::strtoll(
            token.text.c_str(),
            &integer_end,
            10);
    if (errno == 0 &&
        integer_end != token.text.c_str() &&
        *integer_end == '\0')
    {
        return static_cast<int64_t>(integer);
    }
    if (integer_literal)
    {
        syntax_error(
            token,
            "integer is outside the supported signed 64-bit range");
    }

    if (token.text == "NAN")
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (token.text == "INF" || token.text == "+INF")
    {
        return std::numeric_limits<double>::infinity();
    }
    if (token.text == "-INF")
    {
        return -std::numeric_limits<double>::infinity();
    }

    if (!valid_real_literal(token.text))
    {
        syntax_error(token, "expected an integer, real or quoted string");
    }

    std::istringstream real_stream(token.text);
    real_stream.imbue(std::locale::classic());
    double real = 0.0;
    real_stream >> real;
    if (real_stream && real_stream.peek() == std::char_traits<char>::eof())
    {
        return real;
    }
    syntax_error(token, "expected an integer, real or quoted string");
}

void add_repeated(
    AttrObject& object,
    std::string key,
    AttrValue value)
{
    AttrValue* existing = object.find(key);
    if (existing == nullptr)
    {
        object.entries.emplace_back(
            std::move(key),
            std::move(value));
        return;
    }
    if (AttrList* list = attr_list(*existing))
    {
        list->values.push_back(std::move(value));
        return;
    }
    std::vector<AttrValue> values;
    values.reserve(2);
    values.push_back(std::move(*existing));
    values.push_back(std::move(value));
    *existing = make_attr_list(std::move(values));
}

void normalize_networkx_lists(
    AttrObject& object)
{
    for (auto& [key, value] : object.entries)
    {
        static_cast<void>(key);
        if (AttrObject* child = attr_object(value))
        {
            normalize_networkx_lists(*child);
            continue;
        }

        if (AttrList* list = attr_list(value))
        {
            for (AttrValue& item : list->values)
            {
                if (AttrObject* child = attr_object(item))
                {
                    normalize_networkx_lists(*child);
                }
                else if (const auto* text =
                             std::get_if<std::string>(&item);
                         text != nullptr && *text == "[]")
                {
                    item = make_attr_list();
                }
            }

            if (!list->values.empty())
            {
                const auto* marker =
                    std::get_if<std::string>(
                        &list->values.front());
                if (marker != nullptr && *marker == kListStart)
                {
                    list->values.erase(list->values.begin());
                }
            }
            continue;
        }

        if (const auto* text =
                std::get_if<std::string>(&value);
            text != nullptr && *text == "[]")
        {
            value = make_attr_list();
        }
    }
}

AttrValue parse_object(
    Lexer& lexer,
    size_t depth)
{
    constexpr size_t MaxNestingDepth = 256;
    if (depth > MaxNestingDepth)
    {
        throw std::runtime_error(
            "GML nesting exceeds the supported depth of 256");
    }
    auto object = std::make_shared<AttrObject>();
    while (true)
    {
        const Token key = lexer.next();
        if (key.kind == TokenKind::right_bracket)
        {
            return object;
        }
        if (key.kind == TokenKind::end)
        {
            syntax_error(key, "expected ']' before end of file");
        }
        if (key.kind != TokenKind::word)
        {
            syntax_error(key, "expected an attribute name");
        }
        if (!valid_gml_key(key.text))
        {
            syntax_error(key, "invalid GML attribute name");
        }

        AttrValue value;
        const Token next = lexer.peek();
        if (next.kind == TokenKind::left_bracket)
        {
            lexer.next();
            value = parse_object(lexer, depth + 1);
        }
        else
        {
            value = parse_scalar(lexer.next());
        }
        add_repeated(*object, key.text, std::move(value));
    }
}

AttrObjectPtr parse_document(
    std::string source)
{
    Lexer lexer(std::move(source));
    const Token graph = lexer.next();
    if (graph.kind != TokenKind::word ||
        graph.text != "graph")
    {
        syntax_error(graph, "expected 'graph'");
    }
    const Token bracket = lexer.next();
    if (bracket.kind != TokenKind::left_bracket)
    {
        syntax_error(bracket, "expected '[' after 'graph'");
    }
    AttrValue root = parse_object(lexer, 1);
    const Token trailing = lexer.next();
    if (trailing.kind != TokenKind::end)
    {
        syntax_error(trailing, "unexpected trailing token");
    }
    AttrObjectPtr document =
        std::get<AttrObjectPtr>(std::move(root));
    normalize_networkx_lists(*document);
    return document;
}

int64_t require_integer(
    const AttrObject& object,
    std::string_view key)
{
    const AttrValue* value = object.find(key);
    if (value == nullptr)
    {
        throw std::runtime_error(
            "GML object is missing integer attribute '" +
            std::string(key) + "'");
    }
    if (const auto* integer =
            std::get_if<int64_t>(value))
    {
        return *integer;
    }
    throw std::runtime_error(
        "GML attribute '" + std::string(key) +
        "' must be an integer");
}

bool bool_value(
    const AttrObject& object,
    std::string_view key,
    bool default_value)
{
    const AttrValue* value = object.find(key);
    if (value == nullptr)
    {
        return default_value;
    }
    if (const auto* boolean =
            std::get_if<bool>(value))
    {
        return *boolean;
    }
    if (const auto* integer =
            std::get_if<int64_t>(value))
    {
        return *integer != 0;
    }
    throw std::runtime_error(
        "GML attribute '" + std::string(key) +
        "' must be boolean/integer");
}

template <typename Function>
void for_each_repeated(
    const AttrValue* value,
    Function&& function)
{
    if (value == nullptr)
    {
        return;
    }
    if (const AttrList* list = attr_list(*value))
    {
        for (const AttrValue& item : list->values)
        {
            function(item);
        }
        return;
    }
    function(*value);
}

template <typename GraphType>
GraphType build_graph(
    const AttrObject& document)
{
    GraphType graph;
    std::unordered_map<int64_t, Vertex> id_to_vertex;
    const AttrId id_attr = graph.attr_id("id");

    for (const auto& [name, value] : document.entries)
    {
        if (name != "directed" &&
            name != "multigraph" &&
            name != "node" &&
            name != "edge")
        {
            graph.graph_attrs().set(
                graph.attr_id(name),
                value);
        }
    }

    for_each_repeated(
        document.find("node"),
        [&](const AttrValue& node_value)
        {
            const AttrObject* node = attr_object(node_value);
            if (node == nullptr)
            {
                throw std::runtime_error(
                    "GML 'node' value must be an object");
            }
            const int64_t node_id =
                require_integer(*node, "id");
            if (node_id < 0)
            {
                throw std::runtime_error(
                    "GML node id must be non-negative");
            }
            if (id_to_vertex.find(node_id) !=
                id_to_vertex.end())
            {
                throw std::runtime_error(
                    "Duplicate GML node id " +
                    std::to_string(node_id));
            }
            const Vertex vertex = graph.add_node();
            id_to_vertex.emplace(node_id, vertex);
            graph.node_attrs(vertex).set(
                id_attr, node_id);
            for (const auto& [name, value] : node->entries)
            {
                if (name != "id")
                {
                    graph.node_attrs(vertex).set(
                        graph.attr_id(name), value);
                }
            }
        });

    for_each_repeated(
        document.find("edge"),
        [&](const AttrValue& edge_value)
        {
            const AttrObject* edge_object =
                attr_object(edge_value);
            if (edge_object == nullptr)
            {
                throw std::runtime_error(
                    "GML 'edge' value must be an object");
            }
            const int64_t source_id =
                require_integer(*edge_object, "source");
            const int64_t target_id =
                require_integer(*edge_object, "target");
            const auto source =
                id_to_vertex.find(source_id);
            const auto target =
                id_to_vertex.find(target_id);
            if (source == id_to_vertex.end() ||
                target == id_to_vertex.end())
            {
                throw std::runtime_error(
                    "GML edge references an unknown node");
            }
            if (graph.has_edge(
                    source->second,
                    target->second))
            {
                throw std::runtime_error(
                    "GML contains a duplicate edge");
            }
            const auto edge = graph.add_edge(
                source->second,
                target->second);
            for (const auto& [name, value] :
                 edge_object->entries)
            {
                if (name != "source" && name != "target")
                {
                    graph.edge_attrs(edge).set(
                        graph.attr_id(name), value);
                }
            }
        });
    return graph;
}

struct ParsedGml
{
    AttrObjectPtr document;
    bool directed = false;
};

ParsedGml parse_file(
    const std::string& path)
{
    std::ifstream input(
        path,
        std::ios::binary);
    if (!input)
    {
        throw std::runtime_error(
            "cannot open gml file: " + path);
    }
    std::string source(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    AttrObjectPtr document =
        parse_document(std::move(source));
    if (bool_value(*document, "multigraph", false))
    {
        throw std::runtime_error(
            "GML multigraph input is not supported by Graph/DiGraph");
    }
    return {
        document,
        bool_value(*document, "directed", false)};
}

void validate_label_mode(
    const std::string& label)
{
    // Virne's graph core deliberately exposes contiguous integer Vertex
    // indices.  The original project always calls read_gml(..., label="id").
    // We retain that structural GML id in the node's "id" attribute while
    // mapping storage to a dense Vertex index. Arbitrary string-labelled
    // NetworkX nodes cannot be represented without violating the hot-path
    // index contract, so fail explicitly.
    if (label != "id")
    {
        throw std::invalid_argument(
            "GML label mode must be 'id' for contiguous Virne vertices");
    }
}

} // namespace

Graph GmlLoader::load(
    const std::string& path,
    const std::string& label)
{
    validate_label_mode(label);
    ParsedGml parsed = parse_file(path);
    if (parsed.directed)
    {
        throw std::runtime_error(
            "directed GML requires GmlLoader::load_directed");
    }
    return build_graph<Graph>(*parsed.document);
}

DiGraph GmlLoader::load_directed(
    const std::string& path,
    const std::string& label)
{
    validate_label_mode(label);
    ParsedGml parsed = parse_file(path);
    if (!parsed.directed)
    {
        throw std::runtime_error(
            "undirected GML requires GmlLoader::load");
    }
    return build_graph<DiGraph>(*parsed.document);
}

LoadedGraph GmlLoader::load_auto(
    const std::string& path,
    const std::string& label)
{
    validate_label_mode(label);
    ParsedGml parsed = parse_file(path);
    if (parsed.directed)
    {
        return build_graph<DiGraph>(*parsed.document);
    }
    return build_graph<Graph>(*parsed.document);
}

namespace nx
{

Graph read_gml(
    const std::string& path,
    const std::string& label)
{
    return GmlLoader::load(path, label);
}

DiGraph read_gml_directed(
    const std::string& path,
    const std::string& label)
{
    return GmlLoader::load_directed(path, label);
}

LoadedGraph read_gml_auto(
    const std::string& path,
    const std::string& label)
{
    return GmlLoader::load_auto(path, label);
}

}
