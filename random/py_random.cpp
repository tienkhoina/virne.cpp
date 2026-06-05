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
    uint64_t seed)
{
    std::vector<uint32_t> key;

    uint64_t n = seed;

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
    uint64_t result = 0;
    int bits = 0;

    while(bits < k)
    {
        result =
            (result << 32)
            |
            static_cast<uint64_t>(
                genrand_uint32());

        bits += 32;
    }

    if(bits > k)
    {
        result >>=
            (bits - k);
    }

    return result;
}

uint64_t
PyRandom::randrange(
    uint64_t stop)
{
    if(stop == 0)
    {
        return 0;
    }

    int k = 0;

    uint64_t n =
        stop - 1;

    while(n)
    {
        ++k;
        n >>= 1;
    }

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