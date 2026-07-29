#include "attribute_benchmark_manager.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

namespace virne::network::attribute {
namespace {

constexpr std::size_t numpy_float32_lane_count = 16U;

static_assert(sizeof(float) == sizeof(std::uint32_t),
              "Attribute benchmarks require binary32 float storage.");
static_assert(std::numeric_limits<float>::is_iec559,
              "Attribute benchmarks require IEC 60559 float semantics.");

std::uint32_t float_bits(const float value) noexcept {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(const std::uint32_t bits) noexcept {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool is_float32_nan(const float value) noexcept {
    const std::uint32_t bits = float_bits(value);
    return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
           (bits & UINT32_C(0x007fffff)) != 0U;
}

float canonical_float32_nan() noexcept {
    return float_from_bits(UINT32_C(0x7fc00000));
}

double promote_float32(const float value) noexcept {
    // The volatile load forces a real binary32-to-binary64 conversion.  In
    // particular, a signaling NaN from a scalar NumPy tail is quieted while
    // retaining its sign and payload in the same way as float(np.float32).
    volatile float source = value;
    return static_cast<double>(source);
}

class RepeatedRowCursor {
public:
    RepeatedRowCursor(const float* row, const std::size_t columns) noexcept
        : row_(row), columns_(columns) {}

    float next() noexcept {
        const float value = row_[column_];
        ++column_;
        if (column_ == columns_) {
            column_ = 0U;
        }
        return value;
    }

private:
    const float* row_;
    std::size_t columns_;
    std::size_t column_ = 0U;
};

float numpy_float32_max(
    const float* row,
    const std::size_t columns,
    const std::size_t repetitions) noexcept {
    const std::size_t count = columns * repetitions;
    RepeatedRowCursor cursor(row, columns);
    const float first = cursor.next();

    if (count == 1U) {
        return first;
    }

    // NumPy's scalar reduction canonicalizes a NaN in the initial accumulator
    // as soon as there is a second operand.  A later scalar NaN is returned
    // bit-for-bit; promotion to Python float quiets signaling payloads later.
    if (is_float32_nan(first)) {
        return canonical_float32_nan();
    }

    const std::size_t vector_items =
        ((count - 1U) / numpy_float32_lane_count) *
        numpy_float32_lane_count;

    float maximum = first;
    if (vector_items != 0U) {
        std::array<float, numpy_float32_lane_count> lanes{};
        lanes.fill(first);

        for (std::size_t index = 0U; index < vector_items; ++index) {
            const float candidate = cursor.next();
            if (is_float32_nan(candidate)) {
                // The pinned NumPy 2.2.6 16-lane contiguous reduction returns
                // a positive canonical qNaN for a NaN in a vectorized block.
                return canonical_float32_nan();
            }
            float& lane = lanes[index % numpy_float32_lane_count];
            if (candidate >= lane) {
                lane = candidate;
            }
        }

        // This is the tie order of NumPy 2.2.6's 16-lane horizontal maximum.
        // It matters only for equal values with distinct representations,
        // notably +0.0F and -0.0F.
        constexpr std::array<std::size_t, numpy_float32_lane_count>
            horizontal_order{
                12U, 4U, 8U, 0U, 14U, 6U, 10U, 2U,
                13U, 5U, 9U, 1U, 15U, 7U, 11U, 3U};

        maximum = lanes[horizontal_order[0U]];
        for (std::size_t index = 1U; index < horizontal_order.size(); ++index) {
            const float candidate = lanes[horizontal_order[index]];
            if (candidate >= maximum) {
                maximum = candidate;
            }
        }
    }

    const std::size_t tail_items = (count - 1U) - vector_items;
    for (std::size_t index = 0U; index < tail_items; ++index) {
        const float candidate = cursor.next();
        if (is_float32_nan(candidate)) {
            return candidate;
        }
        if (candidate >= maximum) {
            maximum = candidate;
        }
    }
    return maximum;
}

std::size_t available_worker_count() noexcept {
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        const int count = CPU_COUNT(&affinity);
        if (count > 0) {
            return static_cast<std::size_t>(count);
        }
    }
#endif
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0U ? 1U : static_cast<std::size_t>(count);
}

std::size_t effective_worker_count(
    const std::size_t rows,
    const std::size_t configured_workers) noexcept {
    if (rows == 0U || configured_workers <= 1U) {
        return 1U;
    }
    return std::min({rows, configured_workers, available_worker_count()});
}

struct RowReductionInput {
    const float* values = nullptr;
    const std::size_t* active_rows = nullptr;
    std::size_t columns = 0U;
    std::size_t repetitions = 1U;
    double* maxima = nullptr;
};

void reduce_row_block(
    const RowReductionInput input,
    const std::size_t begin,
    const std::size_t end) noexcept {
    for (std::size_t index = begin; index < end; ++index) {
        const std::size_t row_index = input.active_rows[index];
        const float* const row = input.values + row_index * input.columns;
        input.maxima[index] = promote_float32(
            numpy_float32_max(row, input.columns, input.repetitions));
    }
}

std::pair<std::size_t, std::size_t> block_bounds(
    const std::size_t count,
    const std::size_t blocks,
    const std::size_t block) noexcept {
    const std::size_t base = count / blocks;
    const std::size_t remainder = count % blocks;
    const std::size_t begin = block * base + std::min(block, remainder);
    const std::size_t size = base + (block < remainder ? 1U : 0U);
    return {begin, begin + size};
}

void reduce_rows(
    const RowReductionInput input,
    const std::size_t count,
    const std::size_t configured_workers) {
    const std::size_t workers =
        effective_worker_count(count, configured_workers);
    if (workers == 1U) {
        reduce_row_block(input, 0U, count);
        return;
    }

    std::vector<std::thread> threads;
    try {
        threads.reserve(workers - 1U);
    } catch (...) {
        reduce_row_block(input, 0U, count);
        return;
    }

    std::size_t next_block = 1U;
    try {
        for (; next_block < workers; ++next_block) {
            const auto bounds = block_bounds(count, workers, next_block);
            threads.emplace_back(
                [input, bounds]() noexcept {
                    reduce_row_block(input, bounds.first, bounds.second);
                });
        }
    } catch (...) {
        for (std::size_t block = next_block; block < workers; ++block) {
            const auto bounds = block_bounds(count, workers, block);
            reduce_row_block(input, bounds.first, bounds.second);
        }
    }

    const auto first_bounds = block_bounds(count, workers, 0U);
    reduce_row_block(input, first_bounds.first, first_bounds.second);
    for (std::thread& thread : threads) {
        thread.join();
    }
}

struct AttributeBenchmarkCache {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<AttributeBenchmarks>> values;
};

AttributeBenchmarkCache& benchmark_cache() {
    static AttributeBenchmarkCache cache;
    return cache;
}

[[noreturn]] void throw_matrix_shape_error(std::string message) {
    throw AttributeBenchmarkException(
        AttributeBenchmarkErrorCode::invalid_matrix_shape,
        AttributeBenchmarkOperation::validate_prepared_data,
        std::move(message));
}

}  // namespace

AttributeBenchmarkException::AttributeBenchmarkException(
    const AttributeBenchmarkErrorCode code,
    const AttributeBenchmarkOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code), operation_(operation) {}

AttributeBenchmarkErrorCode AttributeBenchmarkException::code() const noexcept {
    return code_;
}

AttributeBenchmarkOperation AttributeBenchmarkException::operation() const noexcept {
    return operation_;
}

AttributeBenchmarkMap::AttributeBenchmarkMap(
    const AttributeBenchmarkMap& other)
    : entries_(other.entries_) {
    rebuild_index();
}

AttributeBenchmarkMap& AttributeBenchmarkMap::operator=(
    const AttributeBenchmarkMap& other) {
    if (this != &other) {
        AttributeBenchmarkMap copy(other);
        entries_.swap(copy.entries_);
        ids_.swap(copy.ids_);
    }
    return *this;
}

std::optional<AttributeBenchmarkId> AttributeBenchmarkMap::bind(
    const std::string_view name) const {
    const auto iterator = ids_.find(name);
    if (iterator == ids_.end()) {
        return std::nullopt;
    }
    return iterator->second;
}

const AttributeBenchmarkEntry& AttributeBenchmarkMap::at(
    const AttributeBenchmarkId id) const {
    const std::size_t index = static_cast<std::size_t>(id);
    if (index >= entries_.size()) {
        throw AttributeBenchmarkException(
            AttributeBenchmarkErrorCode::invalid_benchmark_id,
            AttributeBenchmarkOperation::access_map,
            "Attribute benchmark ID is outside the ordered map.");
    }
    return entries_[index];
}

const double* AttributeBenchmarkMap::find(const std::string_view name) const {
    const auto id = bind(name);
    return id.has_value() ? &entries_[static_cast<std::size_t>(*id)].value : nullptr;
}

const std::vector<AttributeBenchmarkEntry>&
AttributeBenchmarkMap::entries() const noexcept {
    return entries_;
}

void AttributeBenchmarkMap::reserve(const std::size_t size) {
    if (size > static_cast<std::size_t>(
                   std::numeric_limits<AttributeBenchmarkId>::max())) {
        throw AttributeBenchmarkException(
            AttributeBenchmarkErrorCode::benchmark_id_range,
            AttributeBenchmarkOperation::build_map,
            "Attribute benchmark count exceeds the compact ID range.");
    }
    entries_.reserve(size);
    ids_.reserve(size);
}

void AttributeBenchmarkMap::insert_or_assign(
    const std::string_view name,
    const double value) {
    const auto existing = ids_.find(name);
    if (existing != ids_.end()) {
        entries_[static_cast<std::size_t>(existing->second)].value = value;
        return;
    }

    if (entries_.size() >= static_cast<std::size_t>(
                              std::numeric_limits<AttributeBenchmarkId>::max())) {
        throw AttributeBenchmarkException(
            AttributeBenchmarkErrorCode::benchmark_id_range,
            AttributeBenchmarkOperation::build_map,
            "Attribute benchmark count exceeds the compact ID range.");
    }

    const auto id = static_cast<AttributeBenchmarkId>(entries_.size());
    entries_.push_back(AttributeBenchmarkEntry{std::string(name), value});
    ids_.emplace(entries_.back().name, id);
}

void AttributeBenchmarkMap::rebuild_index() {
    ids_.clear();
    ids_.reserve(entries_.size());
    for (std::size_t index = 0U; index < entries_.size(); ++index) {
        ids_.emplace(
            entries_[index].name,
            static_cast<AttributeBenchmarkId>(index));
    }
}

AttributeBenchmarkMap get_attr_benchmarks(
    const PreparedAttributeBenchmarkData& data,
    const std::size_t workers) {
    const AttributeBenchmarkMatrix& matrix = data.matrix;
    if (matrix.rows != 0U &&
        matrix.columns > std::numeric_limits<std::size_t>::max() / matrix.rows) {
        throw_matrix_shape_error(
            "Attribute benchmark matrix dimensions overflow size_t.");
    }
    const std::size_t matrix_size = matrix.rows * matrix.columns;
    if (matrix.values.size() != matrix_size) {
        throw_matrix_shape_error(
            "Attribute benchmark matrix values do not match rows * columns.");
    }
    if (data.column_repetitions == 0U) {
        throw AttributeBenchmarkException(
            AttributeBenchmarkErrorCode::invalid_column_repetitions,
            AttributeBenchmarkOperation::validate_prepared_data,
            "Attribute benchmark column repetitions must be positive.");
    }
    if (matrix.columns != 0U &&
        data.column_repetitions >
            std::numeric_limits<std::size_t>::max() / matrix.columns) {
        throw AttributeBenchmarkException(
            AttributeBenchmarkErrorCode::invalid_column_repetitions,
            AttributeBenchmarkOperation::validate_prepared_data,
            "Attribute benchmark repeated column count overflows size_t.");
    }

    const std::size_t paired_rows =
        std::min(matrix.rows, data.attributes.size());
    std::vector<std::size_t> active_rows;
    std::vector<std::string_view> output_names;
    active_rows.reserve(paired_rows);
    output_names.reserve(paired_rows);

    for (std::size_t row = 0U; row < paired_rows; ++row) {
        const AttributeBenchmarkDescriptor& descriptor = data.attributes[row];
        if (data.extrema_requested &&
            descriptor.kind == AttributeKind::resource) {
            continue;
        }

        active_rows.push_back(row);
        if (data.extrema_requested && descriptor.originator_name.has_value()) {
            output_names.emplace_back(*descriptor.originator_name);
        } else {
            output_names.emplace_back(descriptor.name);
        }
    }

    if (!active_rows.empty() && matrix.columns == 0U) {
        throw AttributeBenchmarkException(
            AttributeBenchmarkErrorCode::empty_retained_row,
            AttributeBenchmarkOperation::reduce_rows,
            "Cannot reduce an empty retained attribute benchmark row.");
    }

    std::vector<double> maxima(active_rows.size());
    if (!active_rows.empty()) {
        const RowReductionInput reduction{
            matrix.values.data(),
            active_rows.data(),
            matrix.columns,
            data.column_repetitions,
            maxima.data()};
        reduce_rows(reduction, active_rows.size(), workers);
    }

    AttributeBenchmarkMap result;
    result.reserve(active_rows.size());
    for (std::size_t index = 0U; index < active_rows.size(); ++index) {
        result.insert_or_assign(output_names[index], maxima[index]);
    }
    return result;
}

AttributeBenchmarkManager::AttributeBenchmarkManager(
    const AttributeBenchmarkRequest& request)
    : benchmarks_(get_benchmarks(request)) {}

const AttributeBenchmarks& AttributeBenchmarkManager::benchmarks() const noexcept {
    return benchmarks_;
}

AttributeBenchmarks AttributeBenchmarkManager::get_benchmarks(
    const AttributeBenchmarkRequest& request) {
    AttributeBenchmarks result;
    if (request.node.has_value()) {
        result.node_attr_benchmarks.emplace(
            get_attr_benchmarks(*request.node, request.workers));
    }
    if (request.link.has_value()) {
        result.link_attr_benchmarks.emplace(
            get_attr_benchmarks(*request.link, request.workers));
    }
    if (request.link_sum.has_value()) {
        result.link_sum_attr_benchmarks.emplace(
            get_attr_benchmarks(*request.link_sum, request.workers));
    }
    return result;
}

void AttributeBenchmarkManager::add_to_cache(
    std::string key,
    std::shared_ptr<AttributeBenchmarks> value) {
    AttributeBenchmarkCache& cache = benchmark_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.values.insert_or_assign(std::move(key), std::move(value));
}

std::shared_ptr<AttributeBenchmarks>
AttributeBenchmarkManager::get_from_cache(const std::string_view key) {
    AttributeBenchmarkCache& cache = benchmark_cache();
    const std::string owned_key(key);
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto iterator = cache.values.find(owned_key);
    return iterator == cache.values.end() ? nullptr : iterator->second;
}

void AttributeBenchmarkManager::clear_cache() {
    AttributeBenchmarkCache& cache = benchmark_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.values.clear();
}

std::size_t AttributeBenchmarkManager::cache_size() {
    AttributeBenchmarkCache& cache = benchmark_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    return cache.values.size();
}

}  // namespace virne::network::attribute
