#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace csvio {

struct DataFrame {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;

    std::size_t nrows() const;
    std::size_t ncols() const;
    bool empty() const noexcept;

    // Resolve a column once before a hot row loop, then use at(row, index).
    // column_index() throws std::out_of_range when the name is unknown.
    bool has_column(std::string_view name) const noexcept;
    std::size_t column_index(std::string_view name) const;

    const std::string& at(
        std::size_t row,
        std::size_t column) const;

    const std::string& at(
        std::size_t row,
        std::string_view column) const;

    std::int64_t int64_at(
        std::size_t row,
        std::size_t column) const;

    std::int64_t int64_at(
        std::size_t row,
        std::string_view column) const;

    double double_at(
        std::size_t row,
        std::size_t column) const;

    double double_at(
        std::size_t row,
        std::string_view column) const;

    bool bool_at(
        std::size_t row,
        std::size_t column) const;

    bool bool_at(
        std::size_t row,
        std::string_view column) const;

    // Reject duplicate column names and rows whose widths differ from the
    // header. read_csv()/write_csv()/append_csv() call this automatically.
    void validate() const;
};

enum class CsvWriteMode {
    truncate,
    append
};

struct CsvWriteOptions {
    CsvWriteMode mode = CsvWriteMode::truncate;

    // In append mode the header is written only when the destination is new
    // or empty. Existing headers are never repeated.
    bool write_header = true;

    // In append mode, compare the destination header before writing. Disable
    // only when an external writer deliberately owns the schema.
    bool validate_existing_schema = true;
};

DataFrame read_csv(
    const std::string& filename);

void write_csv(
    const std::string& filename,
    const DataFrame& df);

void write_csv(
    const std::string& filename,
    const DataFrame& df,
    const CsvWriteOptions& options);

// Recorder-oriented shorthand: append rows, emit a header exactly once, and
// reject a different column order before changing the destination file.
void append_csv(
    const std::string& filename,
    const DataFrame& df,
    bool validate_existing_schema = true);

void print_table(
    const DataFrame& df,
    std::size_t max_rows = 10);

}
