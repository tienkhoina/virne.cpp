#include "csv.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <vector>

int main()
{
    using namespace csvio;

    std::cout << "===== CSV TEST =====\n";

    std::vector<Row> rows = {
        {
            {"node", "0"},
            {"degree", "12"},
            {"score", "0.85"}
        },
        {
            {"node", "1"},
            {"degree", "8"},
            {"score", "0.44"}
        },
        {
            {"node", "2"},
            {"degree", "15"},
            {"score", "0.97"}
        }
    };

    const std::string out_file =
        "csv_test_output.csv";

    std::cout << "\nWRITE CSV\n";

    write_csv(out_file, rows);

    assert(std::filesystem::exists(out_file));

    std::cout << "OK\n";

    std::cout << "\nREAD CSV\n";

    auto loaded =
        read_csv(out_file);

    assert(!loaded.empty());
    assert(loaded.size() == rows.size());

    std::cout
        << "ROWS "
        << loaded.size()
        << '\n';

    std::cout << "\nTABLE VIEW\n";

    print_table(loaded);

    std::cout << "\nFIRST ROW\n";

    for (auto const& kv : loaded.front())
    {
        std::cout
            << kv.first
            << " = "
            << kv.second
            << '\n';
    }

    std::cout
        << "\nOUTPUT FILE "
        << out_file
        << '\n';

    std::cout
        << "\nALL TESTS PASSED\n";

    return 0;
}