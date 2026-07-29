#include "../virne/solver/rank/python310_timsort.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    using virne::solver::rank::LinkRankEntry;
    using virne::solver::rank::LinkRanking;
    using virne::solver::rank::detail::python310_timsort_reverse;

    std::size_t count = 0U;
    while (std::cin >> count) {
        LinkRanking ranking;
        ranking.reserve(count);
        for (std::size_t index = 0U; index < count; ++index) {
            std::uint32_t edge_id = 0U;
            std::string raw_bits;
            if (!(std::cin >> edge_id >> raw_bits)) {
                throw std::runtime_error("truncated Timsort probe case");
            }
            const std::uint64_t bits = std::stoull(raw_bits, nullptr, 16);
            double value = 0.0;
            static_assert(sizeof(value) == sizeof(bits));
            std::memcpy(&value, &bits, sizeof(value));
            ranking.push_back(LinkRankEntry{
                edge_id,
                static_cast<Vertex>(edge_id + 1000U),
                static_cast<Vertex>(edge_id * 3U + 7U),
                value});
        }

        python310_timsort_reverse(ranking);
        for (std::size_t index = 0U; index < ranking.size(); ++index) {
            if (index != 0U) {
                std::cout << ',';
            }
            std::uint64_t bits = 0U;
            std::memcpy(&bits, &ranking[index].value, sizeof(bits));
            std::cout << ranking[index].edge_id << ':'
                      << ranking[index].source << ':'
                      << ranking[index].target << ':';
            const std::ios_base::fmtflags flags = std::cout.flags();
            std::cout << std::hex << bits;
            std::cout.flags(flags);
        }
        std::cout << '\n';
    }
}
