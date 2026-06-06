#include "csv.h"

#include <cassert>
#include <filesystem>
#include <iostream>

int main()
{
    using namespace csvio;

    std::cout
        << "===== CSV STRESS TEST =====\n";

    DataFrame df;

    df.columns = {
        "node_id",
        "node_name",
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
        "pagerank",
        "community",
        "latency_ms",
        "bandwidth_gbps",
        "reward",
        "loss",
        "description"
    };

    constexpr int N = 10000;

    for (int i = 0; i < N; ++i)
    {
        df.rows.push_back({
            std::to_string(i),

            "node_" + std::to_string(i),

            std::to_string(i % 500),

            std::to_string(
                0.123456789012345678 + i),

            std::to_string(
                0.987654321098765432 + i),

            std::to_string(
                123456.789012345678 + i),

            std::to_string(
                999999.999999999999 + i),

            std::to_string(i % 20),

            std::to_string(
                1.234567890123456789 + i),

            std::to_string(
                9876.543210987654321 + i),

            std::to_string(
                0.000000000123456789 + i),

            std::to_string(
                999999999.999999999 + i),

            "very_long_description_for_virtual_network_embedding_node_" +
            std::to_string(i) +
            "_used_for_csv_stress_testing_and_output_validation"
        });
    }

    std::cout
        << "ROWS "
        << df.nrows()
        << '\n';

    std::cout
        << "COLS "
        << df.ncols()
        << '\n';

    const std::string file =
        "csv_big_test.csv";

    std::cout
        << "\nWRITE CSV\n";

    write_csv(file, df);

    assert(
        std::filesystem::exists(file));

    std::cout
        << "WRITE OK\n";

    std::cout
        << "\nREAD CSV\n";

    auto loaded =
        read_csv(file);

    std::cout
        << "ROWS "
        << loaded.nrows()
        << '\n';

    std::cout
        << "COLS "
        << loaded.ncols()
        << '\n';

    assert(
        loaded.nrows() == N);

    assert(
        loaded.ncols() ==
        df.columns.size());

    std::cout
        << "\nHEAD\n";

    print_table(
        loaded,
        10);

    std::cout
        << "\nFIRST CELL\n";

    std::cout
        << loaded.rows.front()[0]
        << '\n';

    std::cout
        << "\nLAST CELL\n";

    std::cout
        << loaded.rows.back().back()
        << '\n';

    std::cout
        << "\nOUTPUT FILE\n";

    std::cout
        << file
        << '\n';

    std::cout
        << "\nALL TESTS PASSED\n";

    return 0;
}