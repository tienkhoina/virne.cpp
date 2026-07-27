#pragma once

#include "numpy_random_state.h"
#include "py_random.h"

#include <cstdint>
#include <optional>

// Owns Virne's two intentionally independent compatibility streams. Reseeding
// a context applies the same scalar seed to CPython Random and NumPy legacy
// RandomState. This covers only the Python/NumPy part of
// virne.utils.dataset.set_seed; a future LibTorch adapter must seed Torch/CUDA
// and configure its deterministic runtime separately.
class RandomContext
{
public:
    explicit RandomContext(
        std::uint32_t seed_value = 0);

    PyRandom& python() noexcept;

    const PyRandom& python() const noexcept;

    NumpyRandomState& numpy() noexcept;

    const NumpyRandomState& numpy() const noexcept;

    // std::nullopt is the C++ equivalent of set_seed(None): it is a no-op and
    // must not rewind either stream.
    void set_seed(
        std::optional<std::uint32_t> seed_value = std::nullopt);

private:
    PyRandom python_;
    NumpyRandomState numpy_;
};

// Compatibility accessors for code ported from Python's module-global random
// APIs. Prefer passing an explicit RandomContext in new code: the process-wide
// context is mutable and therefore requires external synchronization when
// shared by multiple threads.
RandomContext& global_random_context();

PyRandom& global_py_random();

NumpyRandomState& global_numpy_random();

void set_seed(
    std::optional<std::uint32_t> seed_value = std::nullopt);
