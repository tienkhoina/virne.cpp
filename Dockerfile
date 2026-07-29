# VirneCpp toolchain image.
#
# Scope: compilers/build tools only. The production C++ dependencies
# (Boost 1.85.0, yaml-cpp 0.8.0, tabulate 1.4.0) are NOT installed here on
# purpose: DEPENDENCIES.md forbids resolving them from an OS/Conda prefix, they
# must live under the workspace-local libs/ tree. The test-only CPython/
# NetworkX oracle is likewise not part of this image.
#
# Baseline pinned by DEPENDENCIES.md: GCC 11.4.0 + libstdc++ 11
# (_GLIBCXX_RELEASE=11, __GLIBCXX__=20230528). Ubuntu 22.04 is the distro that
# ships exactly that toolchain layout.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gcc-11 \
        g++-11 \
        cpp-11 \
        libstdc++-11-dev \
        cmake \
        make \
        ninja-build \
        pkg-config \
        binutils \
        gdb \
        git \
        curl \
        ca-certificates \
        tar \
        gzip \
        xz-utils \
        coreutils \
 && rm -rf /var/lib/apt/lists/*

# Make the pinned compiler the default cc/c++ so `cmake -B build` with no
# toolchain flags selects GCC 11 instead of some other version.
RUN update-alternatives --install /usr/bin/gcc  gcc  /usr/bin/gcc-11 110 \
 && update-alternatives --install /usr/bin/g++  g++  /usr/bin/g++-11 110 \
 && update-alternatives --install /usr/bin/cc   cc   /usr/bin/gcc-11 110 \
 && update-alternatives --install /usr/bin/c++  c++  /usr/bin/g++-11 110

# Fail the image build if the toolchain is not the accepted baseline. The
# Random fast path in random/numpy_random_state.cpp derives from the exact
# libstdc++ 11 vector layout, so a silent compiler drift must not reach a
# developer container.
RUN set -eux; \
    printf '%s\n' \
      '#include <version>' \
      '#include <cstdio>' \
      '#if !defined(__GNUC__) || __GNUC__ != 11 || __GNUC_MINOR__ != 4' \
      '#  error "expected GCC 11.4.x"' \
      '#endif' \
      '#if !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE != 11' \
      '#  error "expected _GLIBCXX_RELEASE == 11"' \
      '#endif' \
      '#if !defined(__GLIBCXX__) || __GLIBCXX__ != 20230528' \
      '#  error "expected __GLIBCXX__ == 20230528"' \
      '#endif' \
      'int main() {' \
      '  std::printf("gcc %d.%d.%d _GLIBCXX_RELEASE=%d __GLIBCXX__=%d\\n",' \
      '              __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__,' \
      '              _GLIBCXX_RELEASE, __GLIBCXX__);' \
      '  return 0;' \
      '}' > /tmp/toolchain_guard.cpp; \
    g++ -std=c++17 -O2 /tmp/toolchain_guard.cpp -o /tmp/toolchain_guard; \
    /tmp/toolchain_guard; \
    rm -f /tmp/toolchain_guard.cpp /tmp/toolchain_guard

ENV CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11

# The repository is bind-mounted here; nothing is COPYed into the image, so all
# builds and artifacts land in the host directory.
WORKDIR /work

CMD ["/bin/bash"]
