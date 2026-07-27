#include "py_random.h"

void
PyRandom::init_genrand(
    uint32_t s)
{
    mt[0] = s;

    for(int i=1;i<N;++i)
    {
        mt[i] =
            1812433253U *
            (mt[i-1] ^
            (mt[i-1] >> 30))
            +
            static_cast<uint32_t>(i);
    }

    index_ = N;
}

void
PyRandom::init_by_array(
    const uint32_t* init_key,
    size_t key_length)
{
    init_genrand(
        19650218U);

    size_t i = 1;
    size_t j = 0;

    size_t k =
        (N >
         static_cast<int>(
            key_length))
        ?
        N
        :
        key_length;

    for(; k; --k)
    {
        mt[i] =
            (
                mt[i]
                ^
                (
                    (
                        mt[i-1]
                        ^
                        (
                            mt[i-1] >> 30
                        )
                    )
                    *
                    1664525U
                )
            )
            +
            init_key[j]
            +
            static_cast<uint32_t>(
                j);

        ++i;
        ++j;

        if(i >= N)
        {
            mt[0] =
                mt[N-1];

            i = 1;
        }

        if(j >= key_length)
        {
            j = 0;
        }
    }

    for(k=N-1; k; --k)
    {
        mt[i] =
            (
                mt[i]
                ^
                (
                    (
                        mt[i-1]
                        ^
                        (
                            mt[i-1] >> 30
                        )
                    )
                    *
                    1566083941U
                )
            )
            -
            static_cast<uint32_t>(
                i);

        ++i;

        if(i >= N)
        {
            mt[0] =
                mt[N-1];

            i = 1;
        }
    }

    mt[0] =
        0x80000000U;
}

PyRandom::PyRandom(
    uint64_t seed_value)
{
    seed(seed_value);
}

void
PyRandom::seed(
    uint64_t seed_value)
{
    std::vector<uint32_t> key;

    uint64_t n = seed_value;

    do
    {
        key.push_back(
            static_cast<uint32_t>(
                n &
                0xffffffffULL));

        n >>= 32;
    }
    while(n);

    init_by_array(
        key.data(),
        key.size());
}

uint32_t
PyRandom::genrand_uint32()
{
    uint32_t y;

    static const uint32_t
    mag01[2] =
    {
        0x0U,
        MATRIX_A
    };

    if(index_ >= N)
    {
        int kk;

        for(
            kk=0;
            kk < N-M;
            ++kk)
        {
            y =
                (
                    mt[kk]
                    &
                    UPPER_MASK
                )
                |
                (
                    mt[kk+1]
                    &
                    LOWER_MASK
                );

            mt[kk] =
                mt[kk+M]
                ^
                (y >> 1)
                ^
                mag01[
                    y & 1U];
        }

        for(
            ;
            kk < N-1;
            ++kk)
        {
            y =
                (
                    mt[kk]
                    &
                    UPPER_MASK
                )
                |
                (
                    mt[kk+1]
                    &
                    LOWER_MASK
                );

            mt[kk] =
                mt[
                    kk +
                    (M-N)]
                ^
                (y >> 1)
                ^
                mag01[
                    y & 1U];
        }

        y =
            (
                mt[N-1]
                &
                UPPER_MASK
            )
            |
            (
                mt[0]
                &
                LOWER_MASK
            );

        mt[N-1] =
            mt[M-1]
            ^
            (y >> 1)
            ^
            mag01[
                y & 1U];

        index_ = 0;
    }

    y =
        mt[index_++];

    y ^= (y >> 11);
    y ^= (y << 7)
        & 0x9d2c5680U;
    y ^= (y << 15)
        & 0xefc60000U;
    y ^= (y >> 18);

    return y;
}

double
PyRandom::random()
{
    uint32_t a =
        genrand_uint32()
        >> 5;

    uint32_t b =
        genrand_uint32()
        >> 6;

    return
        (
            a *
            67108864.0
            +
            b
        )
        *
        (
            1.0 /
            9007199254740992.0
        );
}

double
PyRandom::uniform(
    double a,
    double b)
{
    return
        a +
        (b-a)
        *
        random();
}

uint32_t
PyRandom::getrandbits32()
{
    return
        genrand_uint32();
}

uint64_t
PyRandom::getrandbits(
    int k)
{
    if (k < 0 || k > 64)
    {
        throw std::invalid_argument(
            "getrandbits supports 0 <= k <= 64");
    }

    if (k == 0)
    {
        return 0;
    }

    // CPython emits the first MT word into the least-significant limb.  The
    // final partial word is shifted down before it becomes the high limb.
    if (k <= 32)
    {
        return static_cast<uint64_t>(
            genrand_uint32() >> (32 - k));
    }

    uint64_t result =
        static_cast<uint64_t>(
            genrand_uint32());

    const int high_bits =
        k - 32;

    uint64_t high =
        static_cast<uint64_t>(
            genrand_uint32());

    if (high_bits < 32)
    {
        high >>= (32 - high_bits);
    }

    return result | (high << 32);
}

uint64_t
PyRandom::randrange(
    uint64_t stop)
{
    if(stop == 0)
    {
        throw std::invalid_argument(
            "empty range for randrange()");
    }

    // Python's _randbelow_with_getrandbits uses stop.bit_length(), including
    // the extra rejection bit for exact powers of two.  That state
    // consumption is essential for choice()/shuffle() parity.
    int k = 0;

#if defined(__GNUC__) || defined(__clang__)
    k = 64 - __builtin_clzll(stop);
#else
    uint64_t n = stop;
    while(n)
    {
        ++k;
        n >>= 1;
    }
#endif

    while(true)
    {
        uint64_t r =
            getrandbits(k);

        if(r < stop)
        {
            return r;
        }
    }
}

int64_t
PyRandom::randrange(
    int64_t start,
    int64_t stop)
{
    return randrange(
        start,
        stop,
        1);
}

int64_t
PyRandom::randrange(
    int64_t start,
    int64_t stop,
    int64_t step)
{
    if (step == 0)
    {
        throw std::invalid_argument(
            "zero step for randrange()");
    }

    using Wide = __int128_t;
    using UWide = __uint128_t;

    const Wide wide_start =
        static_cast<Wide>(start);
    const Wide width =
        static_cast<Wide>(stop)
        - wide_start;
    const Wide wide_step =
        static_cast<Wide>(step);

    Wide count = 0;

    if (wide_step > 0)
    {
        if (width <= 0)
        {
            throw std::invalid_argument(
                "empty range for randrange()");
        }

        count =
            (width - 1) / wide_step + 1;
    }
    else
    {
        if (width >= 0)
        {
            throw std::invalid_argument(
                "empty range for randrange()");
        }

        count =
            ((-width) - 1) / (-wide_step) + 1;
    }

    if (static_cast<UWide>(count)
        > static_cast<UWide>(
            std::numeric_limits<uint64_t>::max()))
    {
        throw std::overflow_error(
            "randrange width exceeds the fixed 64-bit compatibility boundary");
    }

    const uint64_t offset =
        randrange(
            static_cast<uint64_t>(count));

    const Wide result =
        wide_start
        + wide_step
        * static_cast<Wide>(offset);

    return static_cast<int64_t>(result);
}

int64_t
PyRandom::randint(
    int64_t a,
    int64_t b)
{
    if (b < a)
    {
        throw std::invalid_argument(
            "empty range for randint()");
    }

    using Wide = __int128_t;
    using UWide = __uint128_t;

    const Wide width =
        static_cast<Wide>(b)
        - static_cast<Wide>(a)
        + 1;

    if (static_cast<UWide>(width)
        > static_cast<UWide>(
            std::numeric_limits<uint64_t>::max()))
    {
        throw std::overflow_error(
            "randint interval exceeds the fixed 64-bit compatibility boundary");
    }

    const uint64_t offset =
        randrange(
            static_cast<uint64_t>(width));

    return static_cast<int64_t>(
        static_cast<Wide>(a)
        + static_cast<Wide>(offset));
}
