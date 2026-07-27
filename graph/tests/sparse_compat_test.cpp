#include "sparse_matrix.h"
#include "nx/sparse.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const char* message)
{
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main()
{
    try
    {
        SparseMatrix matrix(3, 4);
        matrix.add(0, 1, 2.0);
        matrix.add(0, 1, 3.5);
        matrix.add(1, 2, 0.0);
        matrix.add(2, 3, -4.0);

        require(&matrix.tocoo() == &matrix,
                "tocoo must be allocation-free");
        require(matrix.shape() == std::pair<size_t, size_t>{3, 4},
                "shape mismatch");
        require(&matrix.data() == &matrix.value,
                "data must expose COO values");

        const auto dense = matrix.toarray();
        require(dense.size() == 3 && dense[0].size() == 4,
                "dense shape mismatch");
        require(std::abs(dense[0][1] - 5.5) < 1e-12,
                "duplicate COO coordinates must sum");
        require(std::abs(dense[2][3] + 4.0) < 1e-12,
                "dense value mismatch");

        const auto [rows, columns] = matrix.nonzero();
        require(rows == std::vector<size_t>({0, 0, 2}),
                "nonzero row order mismatch");
        require(columns == std::vector<size_t>({1, 1, 3}),
                "nonzero column order mismatch");

        const CSRMatrix duplicate_csr = matrix.to_csr();
        require(duplicate_csr.row_ptr ==
                    std::vector<size_t>({0, 1, 2, 3}),
                "CSR duplicate row pointers mismatch");
        require(duplicate_csr.col_idx ==
                    std::vector<size_t>({1, 2, 3}),
                "CSR columns must be sorted and duplicates coalesced");
        require(duplicate_csr.values ==
                    std::vector<double>({5.5, 0.0, -4.0}),
                "CSR duplicate values must match SciPy coalescing");

        SparseMatrix malformed(2, 2);
        malformed.row.push_back(0);
        require_throws([&] { (void)malformed.toarray(); },
                       "mismatched COO arrays must throw");

        SparseMatrix outside(2, 2);
        outside.add(2, 0, 1.0);
        require_throws([&] { (void)outside.to_csr(); },
                       "out-of-shape COO coordinate must throw");

        Graph graph;
        graph.add_node();
        graph.add_node();
        graph.edge_attrs(graph.add_edge(0, 1))["bw"] = 7.0;
        const CSRMatrix csr =
            nx::to_scipy_sparse_matrix(graph, "bw", "csr");
        require(csr.nnz() == 2,
                "undirected CSR compatibility adapter mismatch");

        Graph ordered_graph;
        for (size_t node = 0; node < 4; ++node)
        {
            ordered_graph.add_node();
        }
        ordered_graph.edge_attrs(
            ordered_graph.add_edge(2, 0))["bw"] = 20.0;
        ordered_graph.edge_attrs(
            ordered_graph.add_edge(3, 1))["bw"] = 31.0;
        ordered_graph.edge_attrs(
            ordered_graph.add_edge(0, 3))["bw"] = 3.0;
        ordered_graph.edge_attrs(
            ordered_graph.add_edge(2, 2))["bw"] = 22.0;

        const SparseMatrix ordered =
            nx::adjacency_matrix(ordered_graph, "bw");
        require(ordered.row ==
                    std::vector<size_t>({0, 0, 1, 2, 2, 3, 3}),
                "COO rows must match SciPy row-major order");
        require(ordered.col ==
                    std::vector<size_t>({2, 3, 3, 0, 2, 0, 1}),
                "COO columns must be sorted inside every row");
        const CSRMatrix ordered_csr = ordered.to_csr();
        require(ordered_csr.row_ptr ==
                    std::vector<size_t>({0, 2, 3, 5, 7}),
                "ordered CSR row pointers mismatch");
        require(ordered_csr.col_idx == ordered.col,
                "ordered CSR columns differ from SciPy");
        require_throws(
            [&] {
                static_cast<void>(
                    nx::to_scipy_sparse_matrix(graph, "bw", "coo"));
            },
            "unsupported sparse format must throw");

        std::cout << "sparse_compat_test: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sparse_compat_test: " << error.what() << '\n';
        return 1;
    }
}
