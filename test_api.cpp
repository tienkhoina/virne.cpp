#include "progress/progress.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>

int main()
{
    Progress p(
        1000,
        "Running with nea_rank in epoch 0");

    std::mt19937 rng(42);

    double ac = 0.0;
    double r2c = 0.0;
    double reward = 0.0;
    double loss = 1.0;

    int inservice = 0;

    for (size_t i = 1;
         i <= 1000;
         ++i)
    {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                10));

        std::normal_distribution<double>
            noise(0.0, 0.01);

        ac +=
            (0.95 - ac) * 0.01 +
            noise(rng);

        ac =
            std::clamp(
                ac,
                0.0,
                1.0);

        r2c +=
            (0.85 - r2c) * 0.01 +
            noise(rng);

        r2c =
            std::clamp(
                r2c,
                0.0,
                1.5);

        reward +=
            (2.0 - reward) * 0.01 +
            noise(rng);

        loss *=
            0.999;

        if (i % 5 == 0)
        {
            ++inservice;
        }

        p.set_postfix(
        {
            {"ac", ac},
            {"r2c", r2c},
            {"reward", reward},
            {"loss", loss},
            {"inservice", inservice}
        });

        p.update(i);
    }

    p.finish();

    return 0;
}