#include "csv.h"

#include <tabulate/table.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace csvio {

std::size_t DataFrame::nrows() const
{
    return rows.size();
}

std::size_t DataFrame::ncols() const
{
    return columns.size();
}

DataFrame read_csv(
    const std::string& filename)
{
    DataFrame df;

    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot open file: " +
            filename);
    }

    std::string line;

    if (!std::getline(file, line))
        return df;

    {
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ','))
            df.columns.push_back(cell);
    }

    while (std::getline(file, line))
    {
        std::stringstream ss(line);

        std::string cell;

        std::vector<std::string> row;

        while (std::getline(ss, cell, ','))
            row.push_back(cell);

        df.rows.push_back(
            std::move(row));
    }

    return df;
}

void write_csv(
    const std::string& filename,
    const DataFrame& df)
{
    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error(
            "Cannot open file: " +
            filename);
    }

    for (std::size_t i = 0;
         i < df.columns.size();
         ++i)
    {
        if (i)
            out << ",";

        out << df.columns[i];
    }

    out << '\n';

    for (const auto& row : df.rows)
    {
        for (std::size_t i = 0;
             i < row.size();
             ++i)
        {
            if (i)
                out << ",";

            out << row[i];
        }

        out << '\n';
    }
}

void print_table(
    const DataFrame& df,
    std::size_t max_rows)
{
    if (df.columns.empty())
        return;

    tabulate::Table table;

    tabulate::Table::Row_t header;

    for (const auto& col : df.columns)
        header.push_back(col);

    table.add_row(header);

    const std::size_t limit =
        std::min(max_rows,
                 df.rows.size());

    for (std::size_t i = 0;
         i < limit;
         ++i)
    {
        tabulate::Table::Row_t row;

        for (const auto& value :
             df.rows[i])
        {
            row.push_back(value);
        }

        table.add_row(row);
    }

    std::cout << table << '\n';
}

}