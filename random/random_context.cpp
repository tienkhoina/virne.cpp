#include "random_context.h"

RandomContext::RandomContext(
    std::uint32_t seed_value)
    : python_(seed_value),
      numpy_(seed_value)
{
}

PyRandom&
RandomContext::python() noexcept
{
    return python_;
}

const PyRandom&
RandomContext::python() const noexcept
{
    return python_;
}

NumpyRandomState&
RandomContext::numpy() noexcept
{
    return numpy_;
}

const NumpyRandomState&
RandomContext::numpy() const noexcept
{
    return numpy_;
}

void
RandomContext::set_seed(
    std::optional<std::uint32_t> seed_value)
{
    if (!seed_value.has_value())
    {
        return;
    }

    python_.seed(*seed_value);
    numpy_.seed(*seed_value);
}

RandomContext&
global_random_context()
{
    static RandomContext context(0);
    return context;
}

PyRandom&
global_py_random()
{
    return global_random_context().python();
}

NumpyRandomState&
global_numpy_random()
{
    return global_random_context().numpy();
}

void
set_seed(
    std::optional<std::uint32_t> seed_value)
{
    global_random_context().set_seed(seed_value);
}
