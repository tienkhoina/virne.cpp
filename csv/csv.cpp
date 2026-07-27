#include "csv.h"

#include <tabulate/table.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <variant>

namespace csvio {
namespace {

using Records = std::vector<std::vector<std::string>>;

std::string location_message(
    const std::string& filename,
    std::size_t record,
    const std::string& message)
{
    return filename + ": record " + std::to_string(record) + ": " + message;
}

void finish_record(
    Records& records,
    std::vector<std::string>& record,
    std::string& field)
{
    record.push_back(std::move(field));
    field.clear();
    records.push_back(std::move(record));
    record.clear();
}

// Parse RFC 4180 records directly from the stream so quoted CRLF/newlines do
// not need a special second pass. LF and a lone CR are accepted as input record
// separators; the writer always emits canonical CRLF.
Records parse_records(
    std::istream& input,
    const std::string& filename,
    std::size_t maximum_records = std::numeric_limits<std::size_t>::max())
{
    Records records;
    std::vector<std::string> record;
    std::string field;
    bool in_quotes = false;
    bool after_quote = false;
    bool field_started = false;
    std::size_t record_number = 1;

    while (records.size() < maximum_records)
    {
        const int next = input.get();
        if (next == std::char_traits<char>::eof())
            break;

        const char character = static_cast<char>(next);

        if (in_quotes)
        {
            if (character != '"')
            {
                field.push_back(character);
                continue;
            }

            if (input.peek() == '"')
            {
                input.get();
                field.push_back('"');
                continue;
            }

            in_quotes = false;
            after_quote = true;
            continue;
        }

        if (after_quote)
        {
            if (character == ',')
            {
                record.push_back(std::move(field));
                field.clear();
                after_quote = false;
                field_started = false;
                continue;
            }

            if (character == '\r' || character == '\n')
            {
                if (character == '\r' && input.peek() == '\n')
                    input.get();
                finish_record(records, record, field);
                after_quote = false;
                field_started = false;
                ++record_number;
                continue;
            }

            throw std::runtime_error(location_message(
                filename,
                record_number,
                "unexpected character after a closing quote"));
        }

        if (character == '"')
        {
            if (field_started || !field.empty())
            {
                throw std::runtime_error(location_message(
                    filename,
                    record_number,
                    "quote in an unquoted field"));
            }

            in_quotes = true;
            field_started = true;
            continue;
        }

        if (character == ',')
        {
            record.push_back(std::move(field));
            field.clear();
            field_started = false;
            continue;
        }

        if (character == '\r' || character == '\n')
        {
            if (character == '\r' && input.peek() == '\n')
                input.get();
            finish_record(records, record, field);
            field_started = false;
            ++record_number;
            continue;
        }

        field.push_back(character);
        field_started = true;
    }

    if (in_quotes)
    {
        throw std::runtime_error(location_message(
            filename,
            record_number,
            "unterminated quoted field"));
    }

    // A record terminated by CR/LF has already been committed. Do not create
    // a spurious empty record at EOF, but retain a final empty field after a
    // comma (record is then non-empty).
    if (records.size() < maximum_records &&
        (after_quote || field_started || !field.empty() || !record.empty()))
    {
        finish_record(records, record, field);
    }

    if (input.bad())
        throw std::runtime_error("Failed while reading CSV file: " + filename);

    return records;
}

void strip_utf8_bom(Records& records)
{
    if (records.empty() || records.front().empty())
        return;

    auto& first = records.front().front();
    if (first.size() >= 3 &&
        static_cast<unsigned char>(first[0]) == 0xEF &&
        static_cast<unsigned char>(first[1]) == 0xBB &&
        static_cast<unsigned char>(first[2]) == 0xBF)
    {
        first.erase(0, 3);
    }
}

Records read_records(
    const std::string& filename,
    std::size_t maximum_records = std::numeric_limits<std::size_t>::max())
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open file: " + filename);

    auto records = parse_records(input, filename, maximum_records);
    strip_utf8_bom(records);
    return records;
}

void validate_columns(const std::vector<std::string>& columns)
{
    std::unordered_set<std::string> unique;
    unique.reserve(columns.size());
    for (const auto& column : columns)
    {
        if (!unique.insert(column).second)
            throw std::invalid_argument("Duplicate CSV column: " + column);
    }
}

void write_field(std::ostream& output, const std::string& value)
{
    const bool quote = value.find_first_of(",\"\r\n") != std::string::npos;
    if (!quote)
    {
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
        return;
    }

    output.put('"');
    std::size_t start = 0;
    while (true)
    {
        const auto quote_position = value.find('"', start);
        if (quote_position == std::string::npos)
        {
            output.write(
                value.data() + static_cast<std::ptrdiff_t>(start),
                static_cast<std::streamsize>(value.size() - start));
            break;
        }

        output.write(
            value.data() + static_cast<std::ptrdiff_t>(start),
            static_cast<std::streamsize>(quote_position - start));
        output.write("\"\"", 2);
        start = quote_position + 1;
    }
    output.put('"');
}

void write_record(
    std::ostream& output,
    const std::vector<std::string>& record)
{
    for (std::size_t i = 0; i < record.size(); ++i)
    {
        if (i != 0)
            output.put(',');
        write_field(output, record[i]);
    }
    output.write("\r\n", 2);
}

std::string_view trim_ascii(std::string_view value) noexcept
{
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front())) != 0)
        value.remove_prefix(1);
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back())) != 0)
        value.remove_suffix(1);
    return value;
}

std::runtime_error conversion_error(
    std::size_t row,
    std::size_t column,
    std::string_view target,
    std::string_view value)
{
    return std::runtime_error(
        "CSV cell [" + std::to_string(row) + "," +
        std::to_string(column) + "] cannot be converted to " +
        std::string(target) + ": '" + std::string(value) + "'");
}

bool destination_is_empty(const std::string& filename)
{
    std::error_code error;
    const bool exists = std::filesystem::exists(filename, error);
    if (error)
        throw std::runtime_error("Cannot inspect file: " + filename);
    if (!exists)
        return true;

    const auto size = std::filesystem::file_size(filename, error);
    if (error)
        throw std::runtime_error("Cannot inspect file: " + filename);
    return size == 0;
}

bool destination_has_record_terminator(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open file: " + filename);
    input.seekg(-1, std::ios::end);
    const char last = static_cast<char>(input.get());
    return last == '\r' || last == '\n';
}

} // namespace

std::size_t DataFrame::nrows() const
{
    return rows.size();
}

std::size_t DataFrame::ncols() const
{
    return columns.size();
}

bool DataFrame::empty() const noexcept
{
    return rows.empty();
}

bool DataFrame::has_column(std::string_view name) const noexcept
{
    return std::find(columns.begin(), columns.end(), name) != columns.end();
}

std::size_t DataFrame::column_index(std::string_view name) const
{
    const auto found = std::find(columns.begin(), columns.end(), name);
    if (found == columns.end())
        throw std::out_of_range("Unknown CSV column: " + std::string(name));
    return static_cast<std::size_t>(found - columns.begin());
}

const std::string& DataFrame::at(
    std::size_t row,
    std::size_t column) const
{
    if (row >= rows.size())
        throw std::out_of_range("CSV row index out of range: " + std::to_string(row));
    if (column >= columns.size() || column >= rows[row].size())
        throw std::out_of_range("CSV column index out of range: " + std::to_string(column));
    return rows[row][column];
}

const std::string& DataFrame::at(
    std::size_t row,
    std::string_view column) const
{
    return at(row, column_index(column));
}

std::int64_t DataFrame::int64_at(
    std::size_t row,
    std::size_t column) const
{
    const auto& original = at(row, column);
    auto value = trim_ascii(original);
    bool leading_plus = !value.empty() && value.front() == '+';
    if (leading_plus)
        value.remove_prefix(1);

    std::int64_t result = 0;
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result, 10);
    if (value.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size())
    {
        throw conversion_error(row, column, "int64", original);
    }
    return result;
}

std::int64_t DataFrame::int64_at(
    std::size_t row,
    std::string_view column) const
{
    return int64_at(row, column_index(column));
}

double DataFrame::double_at(
    std::size_t row,
    std::size_t column) const
{
    const auto& original = at(row, column);
    auto value = trim_ascii(original);
    if (!value.empty() && value.front() == '+')
        value.remove_prefix(1);
    double result = 0.0;
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result,
        std::chars_format::general);
    if (value.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size())
    {
        throw conversion_error(row, column, "double", original);
    }
    return result;
}

double DataFrame::double_at(
    std::size_t row,
    std::string_view column) const
{
    return double_at(row, column_index(column));
}

bool DataFrame::bool_at(
    std::size_t row,
    std::size_t column) const
{
    const auto& original = at(row, column);
    auto value = trim_ascii(original);
    std::string normalized;
    normalized.reserve(value.size());
    std::transform(
        value.begin(), value.end(), std::back_inserter(normalized),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });

    if (normalized == "true" || normalized == "1" || normalized == "yes")
        return true;
    if (normalized == "false" || normalized == "0" || normalized == "no")
        return false;
    throw conversion_error(row, column, "bool", original);
}

bool DataFrame::bool_at(
    std::size_t row,
    std::string_view column) const
{
    return bool_at(row, column_index(column));
}

void DataFrame::validate() const
{
    validate_columns(columns);
    for (std::size_t row = 0; row < rows.size(); ++row)
    {
        if (rows[row].size() != columns.size())
        {
            throw std::invalid_argument(
                "CSV row " + std::to_string(row) + " has " +
                std::to_string(rows[row].size()) + " fields; expected " +
                std::to_string(columns.size()));
        }
    }
}

DataFrame read_csv(const std::string& filename)
{
    auto records = read_records(filename);
    if (records.empty())
        return {};

    DataFrame result;
    result.columns = std::move(records.front());
    result.rows.reserve(records.size() - 1);
    for (std::size_t i = 1; i < records.size(); ++i)
        result.rows.push_back(std::move(records[i]));
    result.validate();
    return result;
}

void write_csv(
    const std::string& filename,
    const DataFrame& df)
{
    write_csv(filename, df, CsvWriteOptions{});
}

void write_csv(
    const std::string& filename,
    const DataFrame& df,
    const CsvWriteOptions& options)
{
    df.validate();

    const bool append = options.mode == CsvWriteMode::append;
    const bool empty_destination = append && destination_is_empty(filename);

    if (append && !empty_destination && options.validate_existing_schema)
    {
        const auto header = read_records(filename, 1);
        if (header.empty())
            throw std::runtime_error("Existing CSV has no header: " + filename);
        if (header.front() != df.columns)
            throw std::invalid_argument("CSV schema mismatch while appending: " + filename);
    }

    const bool needs_record_separator =
        append && !empty_destination &&
        !destination_has_record_terminator(filename);

    std::ofstream output(
        filename,
        std::ios::binary |
            (append ? std::ios::app : std::ios::trunc));
    if (!output)
        throw std::runtime_error("Cannot open file: " + filename);

    if (needs_record_separator)
        output.write("\r\n", 2);

    const bool should_write_header =
        options.write_header && (!append || empty_destination);
    if (should_write_header && !df.columns.empty())
        write_record(output, df.columns);

    for (const auto& row : df.rows)
        write_record(output, row);

    if (!output)
        throw std::runtime_error("Failed while writing CSV file: " + filename);
}

void append_csv(
    const std::string& filename,
    const DataFrame& df,
    bool validate_existing_schema)
{
    CsvWriteOptions options;
    options.mode = CsvWriteMode::append;
    options.write_header = true;
    options.validate_existing_schema = validate_existing_schema;
    write_csv(filename, df, options);
}

void print_table(
    const DataFrame& df,
    std::size_t max_rows)
{
    if (df.columns.empty())
        return;

    df.validate();
    tabulate::Table table;
    using TableRow = std::vector<std::variant<
        std::string,
        const char*,
        tabulate::Table>>;

    TableRow header;
    header.reserve(df.columns.size());
    for (const auto& column : df.columns)
        header.push_back(column);
    table.add_row(header);

    const std::size_t limit = std::min(max_rows, df.rows.size());
    for (std::size_t i = 0; i < limit; ++i)
    {
        TableRow row;
        row.reserve(df.rows[i].size());
        for (const auto& value : df.rows[i])
            row.push_back(value);
        table.add_row(row);
    }
    std::cout << table << '\n';
}

} // namespace csvio
