# LibTorch probe — 2026-08-04

## Dependency/install

- Archive: `libtorch-shared-with-deps-2.6.0+cpu.zip`
- URL: `https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.6.0%2Bcpu.zip`
- SHA-256: `ad2901049e4d660097f1f54470d60c5afd3de1c293800fd1ae39ac3f9c7d2578`
- Installed path: `libs/libtorch`
- Windows install path: `libs/libtorch-win`
- Windows archive SHA-256: `f2c2e46073848a8e0150984ef26af7c112149a61401063dd4b1f12b7905dac41`
- Windows payload integrity: `build-version=2.6.0+cpu`, Torch CMake config,
  import libraries, and runtime DLLs present; native Windows compile/run still
  requires the MSVC/clang-cl toolchain.
- Torch build version: `2.6.0+cpu`
- ABI: `_GLIBCXX_USE_CXX11_ABI=0`, inherited from the archive's CMake target

## Native gate

The isolated GCC 11 Release target was configured with
`VIRNE_ENABLE_LIBTORCH=ON` and built successfully. `vne_libtorch_probe` passed
CTest with one intra-op/inter-op worker. Its deterministic CPU output was:

```json
{"libtorch_version":"2.6.0+cpu","device":"cpu","cuda_available":false,"threads":1,"interop_threads":1,"shape":[3,3],"dtype":"float64","checksum":"15003413940045531672","output_bytes":72,"sum_bits":"4091b80000000000"}
```

The fixed matrix values are `[14, 38, 62; 38, 126, 214; 62, 214, 366]`;
the digest is independent of the LibTorch thread width for this case.

## Python/original comparison

`porting/compare_libtorch_probe.py` implements the exact Python expression and
compares shape, dtype, byte count, FNV checksum, and IEEE-754 sum bits. The
oracle image was verified earlier as Python 3.10.20 with Torch `2.6.0+cu124`;
the new comparator must be run in that image because the host Anaconda Torch is
2.2.2+cpu and is not the pinned oracle. Docker Desktop's Linux engine became
unavailable (`dockerBackendApiServer`/`Access is denied`) after the native build,
so no Python-oracle PASS is claimed in this artifact. Re-run the documented
command once the engine is healthy; a mismatch exits non-zero and writes the
full Python/C++ payloads.

The frozen environment gate remains valid and independent of Torch: the
existing 9/9 differential artifact has checksum
`17358322786803582063` and final physical checksum
`5251282115348753471` for workers `1/2/8`.
