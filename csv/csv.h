#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace csvio {

struct DataFrame {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;

    std::size_t nrows() const;
    std::size_t ncols() const;
};

DataFrame read_csv(
    const std::string& filename);

void write_csv(
    const std::string& filename,
    const DataFrame& df);

void print_table(
    const DataFrame& df,
    std::size_t max_rows = 10);

}