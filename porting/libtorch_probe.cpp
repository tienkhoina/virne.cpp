#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef VIRNE_LIBTORCH_VERSION
#define VIRNE_LIBTORCH_VERSION "unknown"
#endif

namespace
{

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Digest
{
    std::uint64_t value = fnv_offset;
    std::uint64_t bytes = 0U;

    void append_byte(const std::uint8_t byte)
    {
        value ^= byte;
        value *= fnv_prime;
        ++bytes;
    }

    void append_u64(const std::uint64_t input)
    {
        for (std::size_t shift = 0U; shift < 64U; shift += 8U)
        {
            append_byte(static_cast<std::uint8_t>(input >> shift));
        }
    }

    void append_double(const double input)
    {
        std::uint64_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(input));
        std::memcpy(&bits, &input, sizeof(bits));
        append_u64(bits);
    }
};

struct Arguments
{
    int threads = 1;
    std::string device = "cpu";
};

[[noreturn]] void usage_error(const std::string_view message)
{
    throw std::invalid_argument(
        std::string(message) +
        "\nusage: vne_libtorch_probe [--threads N] [--device cpu|cuda|auto]");
}

Arguments parse_arguments(const int argc, char** argv)
{
    Arguments result;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (argument == "--threads")
        {
            if (++index >= argc)
            {
                usage_error("--threads requires a positive integer");
            }
            try
            {
                const auto parsed = std::stoll(argv[index]);
                if (parsed <= 0 || parsed > std::numeric_limits<int>::max())
                {
                    usage_error("--threads requires a positive integer");
                }
                result.threads = static_cast<int>(parsed);
            }
            catch (const std::exception&)
            {
                usage_error("--threads requires a positive integer");
            }
        }
        else if (argument == "--device")
        {
            if (++index >= argc)
            {
                usage_error("--device requires cpu, cuda, or auto");
            }
            result.device = argv[index];
            if (result.device != "cpu" && result.device != "cuda" &&
                result.device != "auto")
            {
                usage_error("--device requires cpu, cuda, or auto");
            }
        }
        else
        {
            usage_error("unknown argument");
        }
    }
    return result;
}

std::string json_escape(const std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    for (const char character : value)
    {
        if (character == '\\' || character == '"')
        {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    return result;
}

} // namespace

int main(const int argc, char** argv)
{
    try
    {
        const Arguments arguments = parse_arguments(argc, argv);
        // Configure both pools before the first tensor operation.  This is
        // process-global in LibTorch, so callers should create one probe (or
        // one worker-owned runtime) per process rather than mutate it in a
        // hot loop.
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(arguments.threads);
        torch::manual_seed(0U);

        const bool cuda_available = torch::cuda::is_available();
        if (arguments.device == "cuda" && !cuda_available)
        {
            throw std::runtime_error("CUDA was requested but is unavailable");
        }
        const std::string device_name =
            arguments.device == "cuda" ||
                (arguments.device == "auto" && cuda_available)
            ? "cuda"
            : "cpu";
        const torch::Device device(
            device_name == "cuda" ? torch::kCUDA : torch::kCPU);

        torch::NoGradGuard no_grad;
        const auto options = torch::TensorOptions()
            .dtype(torch::kFloat64)
            .device(device);
        const auto input = torch::arange(0.0, 12.0, options).reshape({3, 4});
        const auto output = torch::matmul(input, input.transpose(0, 1))
                                .to(torch::kCPU)
                                .contiguous();

        Digest digest;
        const auto* values = output.data_ptr<double>();
        for (int index = 0; index < output.numel(); ++index)
        {
            digest.append_double(values[index]);
        }
        const double sum = output.sum().item<double>();
        std::uint64_t sum_bits = 0U;
        std::memcpy(&sum_bits, &sum, sizeof(sum_bits));

        std::cout << "{\"libtorch_version\":\""
                  << json_escape(VIRNE_LIBTORCH_VERSION)
                  << "\",\"device\":\"" << device_name
                  << "\",\"cuda_available\":"
                  << (cuda_available ? "true" : "false")
                  << ",\"threads\":" << torch::get_num_threads()
                  << ",\"interop_threads\":"
                  << torch::get_num_interop_threads()
                  << ",\"shape\":[3,3]"
                  << ",\"dtype\":\"float64\""
                  << ",\"checksum\":\"" << digest.value << "\""
                  << ",\"output_bytes\":" << digest.bytes
                  << ",\"sum_bits\":\"" << std::hex << sum_bits
                  << "\"}\n" << std::dec;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "libtorch probe error: " << error.what() << '\n';
        return 2;
    }
}
