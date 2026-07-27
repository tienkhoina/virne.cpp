#include "csv.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& message)
{
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string& message)
{
    if (!condition)
        fail(message);
}

template <typename Function>
void require_throws(Function&& function, const std::string& expected)
{
    try
    {
        function();
        fail("expected exception containing: " + expected);
    }
    catch (const std::exception& error)
    {
        if (std::string(error.what()).find(expected) == std::string::npos)
            fail("unexpected exception: " + std::string(error.what()));
    }
}

class TempDirectory
{
public:
    TempDirectory()
    {
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        path = fs::temp_directory_path() /
               ("virne-csv-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory()
    {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }

    fs::path path;
};

void write_raw(const fs::path& path, const std::string& contents)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
        fail("cannot create fixture " + path.string());
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

std::string read_raw(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void test_rfc4180_round_trip(const fs::path& root)
{
    using namespace csvio;
    const DataFrame source{
        {"id", "payload", "note", "empty"},
        {
            {"1", "comma,value", "plain", ""},
            {"2", "a \"quoted\" value", "line one\r\nline two", ""},
            {"3", "line one\nline two", "carriage\rreturn", "tail"},
        }};

    const auto file = root / "round-trip.csv";
    write_csv(file.string(), source);
    const auto bytes = read_raw(file);

    require(bytes.find("\r\n") != std::string::npos,
            "writer did not emit RFC 4180 CRLF records");
    require(bytes.find("\"comma,value\"") != std::string::npos,
            "comma-containing field was not quoted");
    require(bytes.find("\"a \"\"quoted\"\" value\"") != std::string::npos,
            "embedded quote was not doubled");

    const auto loaded = read_csv(file.string());
    require(loaded.columns == source.columns, "header did not round-trip");
    require(loaded.rows == source.rows, "quoted values did not round-trip");
}

void test_append_and_schema(const fs::path& root)
{
    using namespace csvio;
    const auto file = root / "append.csv";
    append_csv(file.string(), DataFrame{{"id", "name"}, {{"1", "one"}}});
    append_csv(file.string(), DataFrame{{"id", "name"}, {{"2", "two, too"}}});

    auto loaded = read_csv(file.string());
    require(loaded.nrows() == 2, "append did not retain both rows");
    require(loaded.at(1, "name") == "two, too", "appended quoted cell changed");

    const auto before = read_raw(file);
    require_throws(
        [&] {
            append_csv(
                file.string(),
                DataFrame{{"name", "id"}, {{"bad", "3"}}});
        },
        "schema mismatch");
    require(read_raw(file) == before,
            "schema validation changed the file before throwing");

    const auto unterminated = root / "no-final-newline.csv";
    write_raw(unterminated, "id,name\r\n1,one");
    append_csv(
        unterminated.string(),
        DataFrame{{"id", "name"}, {{"2", "two"}}});
    loaded = read_csv(unterminated.string());
    require(loaded.nrows() == 2 && loaded.at(1, 0) == "2",
            "append did not separate a final unterminated record");
}

void test_schema_and_typed_access(const fs::path& root)
{
    using namespace csvio;
    const DataFrame frame{
        {"id", "score", "accepted"},
        {{" +42 ", "1.25e2", "YES"}, {"-7", "0.5", "false"}}};

    frame.validate();
    require(frame.has_column("score"), "has_column missed an existing name");
    require(!frame.has_column("missing"), "has_column found an unknown name");
    require(frame.column_index("accepted") == 2, "column index is incorrect");
    require(frame.int64_at(0, "id") == 42, "int64 conversion failed");
    require(std::abs(frame.double_at(0, "score") - 125.0) < 1e-12,
            "double conversion failed");
    require(frame.bool_at(0, "accepted"), "true conversion failed");
    require(!frame.bool_at(1, 2), "false conversion failed");

    require_throws(
        [] { csvio::DataFrame{{"x", "x"}, {}}.validate(); },
        "Duplicate");
    require_throws(
        [] { csvio::DataFrame{{"x", "y"}, {{"one"}}}.validate(); },
        "expected 2");
    require_throws(
        [&] { (void)frame.int64_at(0, "score"); },
        "int64");

    const auto bom = root / "bom.csv";
    write_raw(bom, "\xEF\xBB\xBFid,value\r\n1,ok\r\n");
    const auto loaded = read_csv(bom.string());
    require(loaded.columns.front() == "id", "UTF-8 BOM was not stripped");
}

void test_malformed_input(const fs::path& root)
{
    using namespace csvio;
    const auto unterminated = root / "unterminated.csv";
    write_raw(unterminated, "id,value\r\n1,\"missing end");
    require_throws(
        [&] { (void)read_csv(unterminated.string()); },
        "unterminated quoted field");

    const auto stray = root / "stray.csv";
    write_raw(stray, "id,value\r\n1,bad\"quote\r\n");
    require_throws(
        [&] { (void)read_csv(stray.string()); },
        "quote in an unquoted field");

    const auto wrong_width = root / "width.csv";
    write_raw(wrong_width, "id,value\r\n1\r\n");
    require_throws(
        [&] { (void)read_csv(wrong_width.string()); },
        "expected 2");
}

} // namespace

int main()
{
    TempDirectory temporary;
    test_rfc4180_round_trip(temporary.path);
    test_append_and_schema(temporary.path);
    test_schema_and_typed_access(temporary.path);
    test_malformed_input(temporary.path);
    std::cout << "ALL CSV TESTS PASSED\n";
    return 0;
}
