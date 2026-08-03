# Dependency policy

Production C++ dependencies are never installed into the OS, Conda, or a
global compiler prefix. They are workspace-vendored below `libs/` and linked
explicitly by the top-level CMake project. Here, "workspace-vendored" means
local source trees at pinned paths; it does not mean those large trees are
tracked by Git.

| Dependency | Pinned version | Local path | Link mode |
|---|---:|---|---|
| Boost | 1.85.0 (`BOOST_VERSION=108500`) | `libs/boost` | headers/interface target `virne_boost` |
| yaml-cpp | 0.8.0 | `libs/yaml-cpp` | local static target `yaml-cpp` |
| tabulate | 1.4.0 | `libs/tabulate` | headers/interface target `virne_tabulate` |
| OR-Tools reference source | 9.15 | `libs/ortools-src` | audit/reference only; not linked |
| OR-Tools C++ (Linux) | 9.15.6755 | `libs/ortools` | imported shared target `virne_ortools` |
| OR-Tools C++ (Windows) | 9.15.6755 | `libs/ortools-win` | imported DLL/import-library target `virne_ortools` |

The production CMake files deliberately do not call `find_package` for these
libraries.  Add future C++ dependencies under `libs/`, pin their version here,
and expose an explicit local target; do not install them into the environment.

`/libs/` is intentionally a workspace-local payload and is ignored by Git.
The prepared workspace already contains all listed local payloads. A fresh clone
must populate the exact paths above from the following archives and verify the
SHA-256; it must not fall back to OS/Conda packages. The same hashes are stored
in the repository-owned, machine-readable `DEPENDENCIES.sha256` manifest.

| Dependency | Source archive | SHA-256 |
|---|---|---|
| Boost 1.85.0 | `https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz` | `be0d91732d5b0cc6fbb275c7939974457e79b54d6f07ce2e3dfdd68bef883b0b` |
| yaml-cpp 0.8.0 | `https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz` | `fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16` |
| tabulate 1.4.0 | `https://github.com/p-ranav/tabulate/archive/refs/tags/v1.4.tar.gz` | `c20cdc3175526a069e932136a7cbdf6f27b137bdb4fc5f574eb5a497228c8e11` |
| OR-Tools source 9.15 | `https://github.com/google/or-tools/releases/download/v9.15/or-tools-9.15.tar.gz` | `599c870319bb127441d92c452d8f79bca46ca6fd295c1deb8031ed303a361311` |
| OR-Tools Ubuntu 22.04 C++ 9.15.6755 | `https://github.com/google/or-tools/releases/download/v9.15/or-tools_amd64_ubuntu-22.04_cpp_v9.15.6755.tar.gz` | `0b30114d7c05f0596286bf3ef8d02adcf5f45be3b39273490e6bb74a2a9bd1ea` |
| OR-Tools Visual Studio 2022 C++ 9.15.6755 | `https://github.com/google/or-tools/releases/download/v9.15/or-tools_x64_VisualStudio2022_cpp_v9.15.6755.zip` | `43429c741641c8b495ee77e44ea00f0f4524519495fd2edaf929003aa2b2ea30` |

The complete reconstruction procedure from the repository root is:

```bash
mkdir -p .deps-cache libs
curl -fL https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz \
  -o .deps-cache/boost_1_85_0.tar.gz
curl -fL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz \
  -o .deps-cache/yaml-cpp-0.8.0.tar.gz
curl -fL https://github.com/p-ranav/tabulate/archive/refs/tags/v1.4.tar.gz \
  -o .deps-cache/tabulate-1.4.tar.gz
curl -fL https://github.com/google/or-tools/releases/download/v9.15/or-tools-9.15.tar.gz \
  -o .deps-cache/or-tools-9.15.tar.gz
curl -fL https://github.com/google/or-tools/releases/download/v9.15/or-tools_amd64_ubuntu-22.04_cpp_v9.15.6755.tar.gz \
  -o .deps-cache/or-tools_amd64_ubuntu-22.04_cpp_v9.15.6755.tar.gz
curl -fL https://github.com/google/or-tools/releases/download/v9.15/or-tools_x64_VisualStudio2022_cpp_v9.15.6755.zip \
  -o .deps-cache/or-tools_x64_VisualStudio2022_cpp_v9.15.6755.zip
(cd .deps-cache && sha256sum -c ../DEPENDENCIES.sha256)

tar -xzf .deps-cache/boost_1_85_0.tar.gz -C libs
tar -xzf .deps-cache/yaml-cpp-0.8.0.tar.gz -C libs
tar -xzf .deps-cache/tabulate-1.4.tar.gz -C libs
mv libs/boost_1_85_0 libs/boost
mv libs/yaml-cpp-0.8.0 libs/yaml-cpp
mv libs/tabulate-1.4 libs/tabulate
tar -xzf .deps-cache/or-tools-9.15.tar.gz -C libs
mv libs/or-tools-9.15 libs/ortools-src
tar -xzf .deps-cache/or-tools_amd64_ubuntu-22.04_cpp_v9.15.6755.tar.gz -C libs
mv libs/or-tools_x86_64_Ubuntu-22.04_cpp_v9.15.6755 libs/ortools
unzip -q .deps-cache/or-tools_x64_VisualStudio2022_cpp_v9.15.6755.zip -d libs
mv libs/or-tools_x64_VisualStudio2022_cpp_v9.15.6755 libs/ortools-win
```

These commands are for a fresh clone where the destination directories
do not yet exist. They download and unpack only the pinned source/binary
payloads into repository-local paths; they do not invoke a package manager or
install into any environment.

The repository-local `.venv` is a test-only exception requested for the
NetworkX oracle. The oracle baseline is CPython 3.10, NetworkX 3.4.2, NumPy
1.26.4 and SciPy 1.15.3; it is never linked into, imported by, or required by
production C++ binaries. See `benchmarks/requirements.txt`. The prepared
workspace uses pip 25.1.1 inside that `.venv`.

Boost upgrades are special: graph hot loops intentionally depend on the
1.85.0 adjacency-list layout.  `graph/graph_types.h` has a compile-time version
guard, and the upgrade procedure in `graph/API.md` is mandatory.

## Accepted implementation-layout pins

The Release compatibility baseline is **GCC 11.4.0 + libstdc++ 11**
(`_GLIBCXX_RELEASE=11`, `__GLIBCXX__=20230528`). This toolchain is not a
vendored C++ dependency, but its exact layout matters to the optional Random
fast path in `random/numpy_random_state.cpp`: for GNU libstdc++, that code
derives from `std::vector` and advances `_M_impl._M_finish` after beginning the
lifetime of trivial output scalars. Its large-`randint` pipeline also reuses
the lower half of disjoint `vector<int64_t>` chunks for ordered `uint32_t`
offsets before workers widen them backwards in place; Release uses
`-fno-strict-aliasing` for this accepted representation hack. Other
standard-library implementations use the portable, zero-initializing fallback
and do not select that GNU-only pipeline.

Boost 1.85.0 is enforced by `static_assert`. When GNU libstdc++ is selected,
the Random direct-output fast path is enabled only when the compiler is GCC
11.4 and the library reports `_GLIBCXX_RELEASE=11`, `__GLIBCXX__=20230528`;
other GNU and non-GNU standard libraries use the portable fallback. Therefore
any accepted GCC/libstdc++ change is an explicit compatibility review, not a
routine toolchain bump: update the guard, inspect the vector layout, then pass
Random's bit-exact differential suite, its large-output case, ASan/UBSan and
the Random benchmark before accepting the new baseline. The accepted risk and
fallback are documented in `random/README.md`; the graph-side Boost risks are
documented in `graph/API.md`.
