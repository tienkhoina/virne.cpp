#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <algorithm>

class PyRandom
{
private:

    static constexpr int N = 624;
    static constexpr int M = 397;

    static constexpr uint32_t MATRIX_A =
        0x9908b0dfU;

    static constexpr uint32_t UPPER_MASK =
        0x80000000U;

    static constexpr uint32_t LOWER_MASK =
        0x7fffffffU;

    std::array<uint32_t, N> mt{};

    int index_ = N + 1;

private:

    void init_genrand(
        uint32_t s);

    void init_by_array(
        const uint32_t* init_key,
        size_t key_length);

public:

    explicit PyRandom(
        uint64_t seed);

    uint32_t genrand_uint32();

    double random();

    double uniform(
        double a,
        double b);

    uint32_t getrandbits32();

    uint64_t randrange(
        uint64_t stop);

    template<typename T>
    T& choice(
        std::vector<T>& v)
    {
        return v[
            randrange(
                v.size())];
    }

    template<typename T>
    void shuffle(
        std::vector<T>& v)
    {
        for(size_t i=v.size(); i>1; --i)
        {
            size_t j =
                randrange(i);

            std::swap(
                v[i-1],
                v[j]);
        }
    }
};