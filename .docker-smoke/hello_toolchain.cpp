// Temporary compiler smoke test for the VirneCpp container toolchain.
// Not part of the build; it only proves g++/cmake work inside the container and
// that artifacts land in the bind-mounted host directory.

#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

int main()
{
    std::vector<int> values(10);

    std::iota(values.begin(), values.end(), 1);

    const long long sum =
        std::accumulate(values.begin(), values.end(), 0LL);

    std::string where = "unknown";

    if (std::FILE* mounts = std::fopen("/proc/1/cgroup", "r"))
    {
        where = "container";
        std::fclose(mounts);
    }

    std::printf("hello from VirneCpp toolchain\n");
    std::printf("  compiler          : GCC %d.%d.%d\n",
                __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    std::printf("  libstdc++         : _GLIBCXX_RELEASE=%d __GLIBCXX__=%d\n",
                _GLIBCXX_RELEASE, __GLIBCXX__);
    std::printf("  __cplusplus       : %ld\n",
                static_cast<long>(__cplusplus));
    std::printf("  runtime           : %s\n", where.c_str());
    std::printf("  sum(1..10)        : %lld\n", sum);

    return sum == 55 ? 0 : 1;
}
