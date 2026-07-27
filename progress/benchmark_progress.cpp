#include "progress.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <streambuf>

namespace
{
    class NullBuffer final : public std::streambuf
    {
    protected:
        int_type overflow(int_type character) override { return character; }
        std::streamsize xsputn(const char*, std::streamsize count) override { return count; }
    };

    template <typename Function>
    double measure_ms(Function&& function)
    {
        const auto start = std::chrono::steady_clock::now();
        function();
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }

    template <typename Function>
    double median_ms(Function&& function)
    {
        std::array<double, 5> samples{};
        for (double& sample : samples) {
            sample = measure_ms(function);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }
}

int main()
{
    constexpr std::size_t update_iterations = 2'000'000;
    constexpr std::size_t postfix_iterations = 100'000;

    NullBuffer sink;
    std::streambuf* const previous = std::cout.rdbuf(&sink);

    const double update_ms = median_ms([] {
        Progress progress(update_iterations + 1, "updates");
        for (std::size_t i = 0; i < update_iterations; ++i) {
            progress.update(i);
        }
        progress.finish();
    });

    const double postfix_ms = median_ms([] {
        Progress progress(postfix_iterations, "postfix");
        ProgressDict values{{"epoch", 0}, {"loss", 1.0}, {"state", std::string("train")}};
        for (std::size_t i = 0; i < postfix_iterations; ++i) {
            values["epoch"] = static_cast<int>(i);
            values["loss"] = static_cast<double>(postfix_iterations - i) /
                             static_cast<double>(postfix_iterations);
            progress.set_postfix(values);
            progress.update(i + 1);
        }
        progress.finish();
    });

    const double advance_ms = median_ms([] {
        Progress progress(update_iterations, "advance");
        for (std::size_t i = 0; i < update_iterations; ++i) {
            progress.advance();
        }
        progress.finish();
    });

    const double tqdm_ms = median_ms([] {
        TqdmProgress progress(update_iterations, "tqdm");
        for (std::size_t i = 0; i < update_iterations; ++i) {
            progress.update();
        }
        progress.close();
    });

    std::cout.rdbuf(previous);
    std::cout << "samples=5 statistic=median\n";
    std::cout << "update_calls=" << update_iterations
              << " elapsed_ms=" << update_ms
              << " throughput_mops=" << update_iterations / update_ms / 1000.0 << '\n';
    std::cout << "advance_calls=" << update_iterations
              << " elapsed_ms=" << advance_ms
              << " throughput_mops=" << update_iterations / advance_ms / 1000.0 << '\n';
    std::cout << "tqdm_update_calls=" << update_iterations
              << " elapsed_ms=" << tqdm_ms
              << " throughput_mops=" << update_iterations / tqdm_ms / 1000.0 << '\n';
    std::cout << "postfix_update_calls=" << postfix_iterations
              << " elapsed_ms=" << postfix_ms
              << " throughput_mops=" << postfix_iterations / postfix_ms / 1000.0 << '\n';
    return 0;
}
