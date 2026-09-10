#include "sinkhorn.h"

#include <torch/torch.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace virne::solver::learning::neural_network {
namespace {

using torch::indexing::None;
using torch::indexing::Slice;

constexpr double negative_infinity() noexcept {
    return -std::numeric_limits<double>::infinity();
}

struct ResolvedInput {
    torch::Tensor scores;
    std::vector<std::int64_t> nrows;
    std::vector<std::int64_t> ncols;
    bool matrix_input = false;
};

std::vector<std::int64_t> resolve_sizes(
    const std::optional<std::vector<std::int64_t>>& supplied,
    const std::int64_t batch_size,
    const std::int64_t full_size,
    const char* name) {
    if (!supplied.has_value()) {
        return std::vector<std::int64_t>(
            static_cast<std::size_t>(batch_size), full_size);
    }
    if (supplied->size() != static_cast<std::size_t>(batch_size)) {
        throw std::invalid_argument(
            std::string("SinkhornNetwork ") + name
            + " must contain one size per batch item");
    }
    auto resolved = *supplied;
    for (const auto size : resolved) {
        if (size < 0 || size > full_size) {
            throw std::invalid_argument(
                std::string("SinkhornNetwork ") + name
                + " contains an out-of-range size");
        }
    }
    return resolved;
}

ResolvedInput resolve_input(
    const torch::Tensor& input,
    const SinkhornNetworkCallOptions& call) {
    if (!input.defined() || (input.dim() != 2 && input.dim() != 3)) {
        throw std::invalid_argument("input data shape not understood.");
    }
    if (!input.is_floating_point()) {
        throw std::invalid_argument(
            "SinkhornNetwork scores must have a floating-point dtype");
    }
    auto scores = input.dim() == 2 ? input.unsqueeze(0) : input;
    const auto batch_size = scores.size(0);
    return ResolvedInput{
        scores,
        resolve_sizes(call.nrows, batch_size, scores.size(1), "nrows"),
        resolve_sizes(call.ncols, batch_size, scores.size(2), "ncols"),
        input.dim() == 2,
    };
}

torch::Tensor make_log_marginal(
    const std::optional<torch::Tensor>& supplied,
    const torch::Tensor& scores,
    const std::vector<std::int64_t>& active_sizes,
    const std::int64_t full_size,
    const char* name) {
    torch::Tensor marginal;
    if (supplied.has_value()) {
        if (!supplied->defined() || supplied->dim() != 2
            || supplied->size(0) != scores.size(0)
            || supplied->size(1) != full_size) {
            throw std::invalid_argument(
                std::string("SinkhornNetwork ") + name
                + " must have shape [batch, extent]");
        }
        marginal = torch::log(*supplied);
    } else {
        // Python creates these with the global default dtype (float32).
        marginal = torch::zeros(
            {scores.size(0), full_size},
            torch::TensorOptions().device(scores.device()));
    }
    marginal = marginal.clone();
    for (std::int64_t batch = 0; batch < scores.size(0); ++batch) {
        marginal.index_put_(
            {batch, Slice(active_sizes[batch], None)},
            negative_infinity());
    }
    return marginal;
}

void mask_nan_as_negative_infinity(torch::Tensor& value) {
    value = value.masked_fill(
        torch::isnan(value), negative_infinity());
}

void append_dummy_rows(
    torch::Tensor& log_scores,
    torch::Tensor& log_r,
    const std::vector<std::int64_t>& original_nrows,
    std::vector<std::int64_t>& active_nrows,
    const std::vector<std::int64_t>& ncols,
    std::int64_t& dummy_count) {
    const auto original_rows = log_scores.size(1);
    const auto columns = log_scores.size(2);
    if (columns < original_rows) {
        throw std::invalid_argument(
            "SinkhornNetwork dummy_row requires columns >= rows");
    }
    dummy_count = columns - original_rows;
    active_nrows = ncols;
    if (dummy_count > 0) {
        const auto default_options = torch::TensorOptions().device(
            log_scores.device());
        log_scores = torch::cat(
            {log_scores,
             torch::full(
                 {log_scores.size(0), dummy_count, columns},
                 negative_infinity(),
                 default_options)},
            1);
        log_r = torch::cat(
            {log_r,
             torch::full(
                 {log_scores.size(0), dummy_count},
                 negative_infinity(),
                 default_options)},
            1);
    }
    log_scores = log_scores.clone();
    log_r = log_r.clone();
    for (std::int64_t batch = 0; batch < log_scores.size(0); ++batch) {
        log_scores.index_put_(
            {batch,
             Slice(original_nrows[batch], active_nrows[batch]),
             Slice(0, ncols[batch])},
            -100.0);
        log_scores.index_put_(
            {batch, Slice(active_nrows[batch], None), Slice()},
            negative_infinity());
        log_scores.index_put_(
            {batch, Slice(), Slice(ncols[batch], None)},
            negative_infinity());
        log_r.index_put_(
            {batch, Slice(original_nrows[batch], active_nrows[batch])},
            0.0);
    }
}

torch::Tensor finalize_output(
    torch::Tensor log_scores,
    const ResolvedInput& input,
    const std::vector<std::int64_t>& active_nrows,
    const std::int64_t dummy_count,
    const bool dummy_row,
    const bool batched_operation) {
    // The batched Python branch only removes/masks dummy rows when padding
    // actually enlarged the matrix. A square padded fixture retains its rows.
    if (dummy_row && (!batched_operation || dummy_count > 0)) {
        if (dummy_count > 0) {
            log_scores = log_scores.narrow(
                1, 0, log_scores.size(1) - dummy_count);
        }
        log_scores = log_scores.clone();
        for (std::int64_t batch = 0; batch < log_scores.size(0); ++batch) {
            log_scores.index_put_(
                {batch,
                 Slice(input.nrows[batch], active_nrows[batch]),
                 Slice(0, input.ncols[batch])},
                negative_infinity());
        }
    }
    if (input.matrix_input) log_scores = log_scores.squeeze(0);
    return torch::exp(log_scores);
}

}  // namespace

SinkhornNetworkImpl::SinkhornNetworkImpl(SinkhornNetworkOptions options)
    : options_(std::move(options)) {}

torch::Tensor SinkhornNetworkImpl::forward(
    const torch::Tensor& scores,
    const SinkhornNetworkCallOptions& call) {
    return call.dummy ? forward_log_dummy(scores, call)
                      : forward_log(scores, call);
}

torch::Tensor SinkhornNetworkImpl::forward_log(
    const torch::Tensor& scores,
    const SinkhornNetworkCallOptions& call) {
    (void)call.dtype;  // Kept for Python API parity; Python also ignores it.
    auto input = resolve_input(scores, call);
    const auto batch_size = input.scores.size(0);
    auto active_nrows = input.nrows;
    auto log_r = make_log_marginal(
        call.r, input.scores, input.nrows, input.scores.size(1), "r");
    const auto log_c = make_log_marginal(
        call.c, input.scores, input.ncols, input.scores.size(2), "c");
    auto log_scores = input.scores / options_.tau;
    std::int64_t dummy_count = 0;
    if (call.dummy_row) {
        append_dummy_rows(
            log_scores,
            log_r,
            input.nrows,
            active_nrows,
            input.ncols,
            dummy_count);
    }

    if (options_.batched_operation) {
        for (std::int64_t iteration = 0; iteration < options_.max_iter;
             ++iteration) {
            if ((iteration & 1) == 0) {
                const auto sums = torch::logsumexp(log_scores, {2}, true);
                log_scores = log_scores - sums + log_r.unsqueeze(2);
            } else {
                const auto sums = torch::logsumexp(log_scores, {1}, true);
                log_scores = log_scores - sums + log_c.unsqueeze(1);
            }
            mask_nan_as_negative_infinity(log_scores);
        }
        return finalize_output(
            std::move(log_scores),
            input,
            active_nrows,
            dummy_count,
            call.dummy_row,
            options_.batched_operation);
    }

    auto result = torch::full(
        {batch_size, log_scores.size(1), log_scores.size(2)},
        negative_infinity(),
        input.scores.options());
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
        const auto row_count = active_nrows[batch];
        const auto col_count = input.ncols[batch];
        auto item = log_scores.index(
            {batch, Slice(0, row_count), Slice(0, col_count)});
        const auto item_r = log_r.index({batch, Slice(0, row_count)});
        const auto item_c = log_c.index({batch, Slice(0, col_count)});
        for (std::int64_t iteration = 0; iteration < options_.max_iter;
             ++iteration) {
            if ((iteration & 1) == 0) {
                item = item
                       - torch::logsumexp(item, {1}, true)
                       + item_r.unsqueeze(1);
            } else {
                item = item
                       - torch::logsumexp(item, {0}, true)
                       + item_c.unsqueeze(0);
            }
        }
        result.index_put_(
            {batch, Slice(0, row_count), Slice(0, col_count)}, item);
    }
    return finalize_output(
        std::move(result),
        input,
        active_nrows,
        dummy_count,
        call.dummy_row,
        options_.batched_operation);
}

torch::Tensor SinkhornNetworkImpl::forward_log_dummy(
    const torch::Tensor& scores,
    const SinkhornNetworkCallOptions& call) {
    (void)call.dtype;  // Kept for Python API parity; Python also ignores it.
    auto input = resolve_input(scores, call);
    const auto batch_size = input.scores.size(0);
    auto active_nrows = input.nrows;
    auto log_r = make_log_marginal(
        call.r, input.scores, input.nrows, input.scores.size(1), "r");
    const auto log_c = make_log_marginal(
        call.c, input.scores, input.ncols, input.scores.size(2), "c");
    auto log_scores = (input.scores / options_.tau).clone();
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
        log_scores.index_put_(
            {batch, input.nrows[batch] - 1, input.ncols[batch] - 1},
            negative_infinity());
    }
    std::int64_t dummy_count = 0;
    if (call.dummy_row) {
        append_dummy_rows(
            log_scores,
            log_r,
            input.nrows,
            active_nrows,
            input.ncols,
            dummy_count);
    }

    if (options_.batched_operation) {
        for (std::int64_t iteration = 0; iteration < options_.max_iter;
             ++iteration) {
            if ((iteration & 1) == 0) {
                auto sums = torch::logsumexp(log_scores, {2}, true);
                sums = sums.clone();
                for (std::int64_t batch = 0; batch < batch_size; ++batch) {
                    sums.index_put_(
                        {batch, Slice(active_nrows[batch] - 1, None), Slice()},
                        0.0);
                }
                log_scores = log_scores - sums + log_r.unsqueeze(2);
            } else {
                auto sums = torch::logsumexp(log_scores, {1}, true);
                sums = sums.clone();
                for (std::int64_t batch = 0; batch < batch_size; ++batch) {
                    sums.index_put_(
                        {batch, Slice(), Slice(input.ncols[batch] - 1, None)},
                        0.0);
                }
                log_scores = log_scores - sums + log_c.unsqueeze(1);
            }
            mask_nan_as_negative_infinity(log_scores);
        }
        return finalize_output(
            std::move(log_scores),
            input,
            active_nrows,
            dummy_count,
            call.dummy_row,
            options_.batched_operation);
    }

    auto result = torch::full(
        {batch_size, log_scores.size(1), log_scores.size(2)},
        negative_infinity(),
        input.scores.options());
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
        const auto row_count = active_nrows[batch];
        const auto col_count = input.ncols[batch];
        auto item = log_scores.index(
            {batch, Slice(0, row_count), Slice(0, col_count)});
        const auto item_r = log_r.index({batch, Slice(0, row_count)});
        const auto item_c = log_c.index({batch, Slice(0, col_count)});
        for (std::int64_t iteration = 0; iteration < options_.max_iter;
             ++iteration) {
            auto next = item.clone();
            if ((iteration & 1) == 0) {
                const auto sums = torch::logsumexp(item, {1}, true);
                next.index_put_(
                    {Slice(0, -1), Slice()},
                    item.index({Slice(0, -1), Slice()})
                        - sums.index({Slice(0, -1), Slice()})
                        + item_r.index({Slice(0, -1)}).unsqueeze(1));
            } else {
                const auto sums = torch::logsumexp(item, {0}, true);
                next.index_put_(
                    {Slice(), Slice(0, -1)},
                    item.index({Slice(), Slice(0, -1)})
                        - sums.index({Slice(), Slice(0, -1)})
                        + item_c.index({Slice(0, -1)}).unsqueeze(0));
            }
            item = std::move(next);
        }
        result.index_put_(
            {batch, Slice(0, row_count), Slice(0, col_count)}, item);
    }
    return finalize_output(
        std::move(result),
        input,
        active_nrows,
        dummy_count,
        call.dummy_row,
        options_.batched_operation);
}

torch::Tensor logsumexp(
    const torch::Tensor& input,
    std::optional<std::int64_t> dim,
    const bool keepdim) {
    if (!input.defined()) {
        throw std::invalid_argument("logsumexp input must be defined");
    }
    auto values = input;
    if (!dim.has_value()) {
        values = values.view({-1});
        dim = 0;
    }
    const auto maximum = std::get<0>(values.max(*dim, true));
    auto output = maximum
                  + (values - maximum)
                        .exp()
                        .sum(*dim, true)
                        .log();
    return keepdim ? output : output.squeeze(*dim);
}

SinkhornImpl::SinkhornImpl(SinkhornOptions options)
    : options_(std::move(options)) {}

torch::Tensor SinkhornImpl::row_norm(const torch::Tensor& input) const {
    return input
        - ::virne::solver::learning::neural_network::logsumexp(
            input, std::optional<std::int64_t>{2}, true);
}

torch::Tensor SinkhornImpl::col_norm(const torch::Tensor& input) const {
    return input
        - ::virne::solver::learning::neural_network::logsumexp(
            input, std::optional<std::int64_t>{1}, true);
}

torch::Tensor SinkhornImpl::forward(
    const torch::Tensor& input,
    const double eps) {
    auto value = input / options_.tau;
    for (std::int64_t iteration = 0; iteration < options_.sinkhorn_iters;
         ++iteration) {
        value = row_norm(value);
        value = col_norm(value);
    }
    return torch::exp(value) + eps;
}

}  // namespace virne::solver::learning::neural_network
