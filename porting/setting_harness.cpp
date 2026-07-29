#include "setting.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using virne::utils::SettingDocument;
using virne::utils::SettingException;
using virne::utils::SettingFormat;
using virne::utils::SettingKeyId;
using virne::utils::SettingValueKind;

volatile std::uint64_t benchmark_sink = 0;

std::string read_stdin()
{
    std::ostringstream stream;
    stream << std::cin.rdbuf();
    return stream.str();
}

SettingFormat parse_format(std::string_view text)
{
    if (text == "json")
    {
        return SettingFormat::json;
    }
    if (text == "yaml")
    {
        return SettingFormat::yaml;
    }
    throw std::invalid_argument("format must be json or yaml");
}

std::size_t parse_size(std::string_view text, std::string_view name)
{
    if (text.empty())
    {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(std::string(text), &consumed, 10);
    if (consumed != text.size() ||
        value > static_cast<unsigned long long>(
                    std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument(std::string(name) + " is invalid");
    }
    return static_cast<std::size_t>(value);
}

char hex_digit(unsigned value)
{
    return "0123456789abcdef"[value & 0x0FU];
}

std::string hex_encode(std::string_view bytes)
{
    std::string encoded;
    encoded.resize(bytes.size() * 2U);
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        const auto byte = static_cast<unsigned char>(bytes[index]);
        encoded[index * 2U] = hex_digit(static_cast<unsigned>(byte >> 4U));
        encoded[index * 2U + 1U] = hex_digit(static_cast<unsigned>(byte));
    }
    return encoded;
}

std::uint64_t fnv1a(
    std::string_view bytes,
    std::uint64_t hash = 14695981039346656037ULL)
{
    for (const char character : bytes)
    {
        const auto byte = static_cast<unsigned char>(character);
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<std::string> split_nul(std::string_view bytes)
{
    std::vector<std::string> items;
    std::size_t begin = 0;
    while (begin <= bytes.size())
    {
        const std::size_t end = bytes.find('\0', begin);
        if (end == std::string_view::npos)
        {
            items.emplace_back(bytes.substr(begin));
            break;
        }
        items.emplace_back(bytes.substr(begin, end - begin));
        begin = end + 1U;
    }
    return items;
}

void print_success(std::string_view bytes)
{
    std::cout << "{\"ok\":true,\"output_hex\":\""
              << hex_encode(bytes)
              << "\",\"output_size\":" << bytes.size() << "}\n";
}

void print_batch_success(const std::vector<std::string>& outputs)
{
    std::uint64_t checksum = 14695981039346656037ULL;
    std::cout << "{\"ok\":true,\"outputs_hex\":[";
    for (std::size_t index = 0; index < outputs.size(); ++index)
    {
        if (index != 0U)
        {
            std::cout << ',';
        }
        std::cout << '\"' << hex_encode(outputs[index]) << '\"';
        checksum = fnv1a(outputs[index], checksum);
        checksum ^= static_cast<std::uint64_t>(outputs[index].size());
        checksum *= 1099511628211ULL;
    }
    std::cout << "],\"count\":" << outputs.size()
              << ",\"checksum\":" << checksum << "}\n";
}

std::uint64_t elapsed_nanoseconds(Clock::time_point begin)
{
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin);
    return static_cast<std::uint64_t>(elapsed.count());
}

void run_transform(
    SettingFormat input_format,
    SettingFormat output_format)
{
    const SettingDocument document =
        virne::utils::parse_setting(read_stdin(), input_format);
    print_success(virne::utils::dump_setting(document, output_format));
}

void run_batch(
    SettingFormat input_format,
    SettingFormat output_format,
    std::size_t workers)
{
    const std::vector<std::string> inputs = split_nul(read_stdin());
    const auto documents = virne::utils::parse_setting_batch(
        inputs, input_format, workers);
    print_batch_success(virne::utils::dump_setting_batch(
        documents, output_format, workers));
}

void run_benchmark(
    SettingFormat format,
    std::size_t workers,
    std::size_t rounds,
    std::size_t batch_size,
    std::size_t id_iterations)
{
    if (rounds == 0U || batch_size == 0U)
    {
        throw std::invalid_argument("rounds and batch_size must be positive");
    }

    const std::string input = read_stdin();
    const SettingDocument seed = virne::utils::parse_setting(input, format);
    const std::string seed_dump = virne::utils::dump_setting(seed, format);
    const std::vector<std::string> inputs(batch_size, input);
    const std::vector<SettingDocument> seed_documents(batch_size, seed);

    // Warm all code paths outside the measured intervals.
    static_cast<void>(virne::utils::parse_setting(input, format));
    static_cast<void>(virne::utils::dump_setting(seed, format));
    static_cast<void>(virne::utils::parse_setting_batch(inputs, format, workers));
    static_cast<void>(
        virne::utils::dump_setting_batch(seed_documents, format, workers));

    std::uint64_t checksum = fnv1a(seed_dump);
    SettingDocument last_document;
    auto begin = Clock::now();
    for (std::size_t round = 0; round < rounds; ++round)
    {
        last_document = virne::utils::parse_setting(input, format);
        checksum ^= static_cast<std::uint64_t>(last_document.root.kind());
    }
    const std::uint64_t parse_ns = elapsed_nanoseconds(begin);

    std::string last_dump;
    begin = Clock::now();
    for (std::size_t round = 0; round < rounds; ++round)
    {
        last_dump = virne::utils::dump_setting(seed, format);
        checksum ^= static_cast<std::uint64_t>(last_dump.size() + round);
    }
    const std::uint64_t dump_ns = elapsed_nanoseconds(begin);
    checksum = fnv1a(last_dump, checksum);

    std::vector<SettingDocument> parsed_batch;
    begin = Clock::now();
    for (std::size_t round = 0; round < rounds; ++round)
    {
        parsed_batch =
            virne::utils::parse_setting_batch(inputs, format, workers);
        checksum ^= static_cast<std::uint64_t>(parsed_batch.size() + round);
    }
    const std::uint64_t batch_parse_ns = elapsed_nanoseconds(begin);

    std::vector<std::string> dumped_batch;
    begin = Clock::now();
    for (std::size_t round = 0; round < rounds; ++round)
    {
        dumped_batch = virne::utils::dump_setting_batch(
            seed_documents, format, workers);
        checksum ^= static_cast<std::uint64_t>(dumped_batch.size() + round);
    }
    const std::uint64_t batch_dump_ns = elapsed_nanoseconds(begin);
    for (const std::string& output : dumped_batch)
    {
        checksum = fnv1a(output, checksum);
    }

    std::uint64_t id_access_ns = 0;
    std::size_t performed_id_accesses = 0;
    if (seed.root.kind() == SettingValueKind::object &&
        !seed.root.as_object().empty() && id_iterations != 0U)
    {
        const auto& object = seed.root.as_object();
        const SettingKeyId id{0U};
        begin = Clock::now();
        for (std::size_t index = 0; index < id_iterations; ++index)
        {
            checksum ^=
                static_cast<std::uint64_t>(object.at(id).kind()) + index;
        }
        id_access_ns = elapsed_nanoseconds(begin);
        performed_id_accesses = id_iterations;
    }

    benchmark_sink = checksum;
    std::cout
        << "{\"ok\":true"
        << ",\"format\":\""
        << (format == SettingFormat::json ? "json" : "yaml") << '\"'
        << ",\"workers\":" << workers
        << ",\"rounds\":" << rounds
        << ",\"batch_size\":" << batch_size
        << ",\"parse\":{\"total_ns\":" << parse_ns
        << ",\"operations\":" << rounds << '}'
        << ",\"dump\":{\"total_ns\":" << dump_ns
        << ",\"operations\":" << rounds << '}'
        << ",\"batch_parse\":{\"total_ns\":" << batch_parse_ns
        << ",\"operations\":" << (rounds * batch_size) << '}'
        << ",\"batch_dump\":{\"total_ns\":" << batch_dump_ns
        << ",\"operations\":" << (rounds * batch_size) << '}'
        << ",\"id_access\":{\"total_ns\":" << id_access_ns
        << ",\"operations\":" << performed_id_accesses << '}'
        << ",\"checksum\":" << checksum
        << ",\"output_size\":" << seed_dump.size()
        << "}\n";
}

void print_usage()
{
    std::cout
        << "setting_harness protocol v1\n"
        << "  transform <input-format> <output-format>\n"
        << "    stdin: one raw document; stdout: JSON with exact output_hex\n"
        << "  batch <input-format> <output-format> <workers>\n"
        << "    stdin: NUL-separated raw documents; stdout: JSON outputs_hex[]\n"
        << "  benchmark <format> <workers> <rounds> <batch-size> "
           "[id-iterations]\n"
        << "    stdin: one raw document; stdout: JSON timing rows in ns\n"
        << "Formats are json or yaml. Worker 0 requests automatic width.\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "--help")
        {
            print_usage();
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "transform")
        {
            run_transform(parse_format(argv[2]), parse_format(argv[3]));
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "batch")
        {
            run_batch(
                parse_format(argv[2]),
                parse_format(argv[3]),
                parse_size(argv[4], "workers"));
            return 0;
        }
        if ((argc == 6 || argc == 7) &&
            std::string_view(argv[1]) == "benchmark")
        {
            run_benchmark(
                parse_format(argv[2]),
                parse_size(argv[3], "workers"),
                parse_size(argv[4], "rounds"),
                parse_size(argv[5], "batch-size"),
                argc == 7
                    ? parse_size(argv[6], "id-iterations")
                    : std::size_t{1000000});
            return 0;
        }

        print_usage();
        return 64;
    }
    catch (const SettingException& error)
    {
        std::cout << "{\"ok\":false,\"error_code\":"
                  << static_cast<unsigned>(error.code())
                  << ",\"message_hex\":\"" << hex_encode(error.what())
                  << "\"}\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        std::cout << "{\"ok\":false,\"error_code\":-1"
                  << ",\"message_hex\":\"" << hex_encode(error.what())
                  << "\"}\n";
        return 2;
    }
}
