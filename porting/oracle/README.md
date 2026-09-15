# Python oracles

## CPU ML reconstruction (2026-09-08)

The deleted toolchain and oracle images have been reconstructed. From the
`virne.cpp` directory:

```powershell
docker build --tag virne-cpp-toolchain:gcc11.4 .
docker build --tag virne-python-oracle:py310-nonml porting/oracle
docker build --file porting/oracle/Dockerfile.ml `
  --tag virne-python-oracle:py310-torch26-pyg27-cpu `
  --tag virne-cpu:latest porting/oracle
docker compose up -d --no-build
```

The ML image derives from the non-ML image and shares its layers. Both ML
tags resolve to one image ID. Build contexts exclude source/dependency
payloads; the workspace and original Python checkout are mounted at runtime.
Pip caching is disabled and only binary wheels are installed. There is no
CUDA runtime, compiler, copied LibTorch archive or copied repository in the
ML oracle image. The production C++ compiler continues to use `libs/libtorch`.

The ML oracle pins Torch `2.6.0+cpu`, PyG `2.7.0`, torch_scatter
`2.1.2+pt26cpu`, torch_sparse `0.6.18+pt26cpu`, SciPy `1.15.3` and pip
`25.1.1`. The Dockerfile checks package dependencies and imports as part of
building the image. This is an environment check, not a rerun of frozen
component tests.

The `virne-cpu:latest` name preserves the old ML commands but now selects CPU
wheels. It is not the old CUDA-backed image and does not restore every
optional package from that historical image. Old timing artifacts stay frozen;
new measurements record this CPU environment separately.

Example for an unfinished/new component only:

```powershell
docker run --rm --network none `
  --mount 'type=bind,source=D:\Virne to cpp\virne.cpp,target=/work' `
  --mount 'type=bind,source=D:\Virne to cpp\virne,target=/virne,readonly' `
  --workdir /work virne-cpu:latest python porting/compare_neural_network_remaining.py `
  --native build-ml-linux/porting/vne_neural_network_remaining_harness
```

The `/work` and `/virne` mount layout preserves the comparator's relative
Python-source lookup. `virne-cpp-dev` uses the existing Compose definition;
no duplicate keepalive image/container is needed.

Installed packages are recorded in `requirements-installed-2026-09-08.txt`.
After removing unused Docker build caches, `docker system df` reported three
distinct images totaling 1.736 GB, zero build cache and no local volumes.
Individual image sizes include shared layers and must not be summed to infer
host disk use. Docker virtual-disk file compaction was not performed.

## Non-ML reference

This image is the minimal reference runtime for porting the original Python
implementations in `virne/utils/network.py` and `virne/network/topology/`.
It deliberately does not install the `virne` project, Torch, PyG, solvers, or
system code.

The versions match the existing `virne-cpu:latest` reference image:

- CPython 3.10.20
- NumPy 2.2.6
- NetworkX 3.4.2

The original Python checkout must remain outside the image and be mounted
read-only when an oracle is run.

## Build

From the directory containing the sibling `virne` and `virne.cpp` checkouts:

```powershell
docker build --pull --tag virne-python-oracle:py310-nonml .\virne.cpp\porting\oracle
```

The base image is pinned to the registry digest resolved on 2026-07-27. The
Python packages are exact-pinned in `requirements-nonml.txt`.

## Load leaf modules without importing ML

Do not use `import virne`, `import virne.network`, or `import virne.utils` in an
oracle harness. Their package initializers eagerly reach solver or utility
modules that import Torch and other ML dependencies.

Load the independent source files directly instead:

```python
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SOURCE_ROOT = Path('/src/virne/virne')


def load_leaf(module_name: str, relative_path: str):
    spec = spec_from_file_location(module_name, SOURCE_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {relative_path}')
    module = module_from_spec(spec)
    # Registration before exec_module is required by dataclasses.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


utils_network = load_leaf(
    '_virne_oracle_utils_network',
    'utils/network.py',
)
topology_generator = load_leaf(
    '_virne_oracle_topology_generator',
    'network/topology/topology_generator.py',
)
topological_metrics = load_leaf(
    '_virne_oracle_topological_metrics',
    'network/topology/topological_metric_calculator.py',
)

assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
```

These three leaf modules have no runtime dependency on package-relative imports,
so direct loading preserves their implementation while avoiding every
`__init__.py` side effect.

## Run isolated

The following PowerShell command mounts the original checkout read-only, turns
off networking, makes the container root filesystem read-only, and fixes common
numeric-library thread counts:

```powershell
$pythonSource = (Resolve-Path .\virne).Path
docker run --rm `
  --network none `
  --read-only `
  --cpuset-cpus=2 `
  --tmpfs /tmp:rw,noexec,nosuid,size=64m `
  --mount "type=bind,source=$pythonSource,target=/src/virne,readonly" `
  --env OMP_NUM_THREADS=1 `
  --env OPENBLAS_NUM_THREADS=1 `
  --env MKL_NUM_THREADS=1 `
  --env NUMEXPR_NUM_THREADS=1 `
  virne-python-oracle:py310-nonml `
  python -c "import numpy, networkx; print(numpy.__version__, networkx.__version__)"
```

Use one container invocation for a complete benchmark batch. The Python harness
should warm up each case and collect repeated samples with
`time.perf_counter_ns()`; the C++ side should time the same workload with
`std::chrono::steady_clock`. Compare outputs first, then report median, p95, and
the C++/Python ratio. Keep kernel time separate from parsing and serialization
time, use identical inputs and seeds, and do not include image or container
startup in either result.

## Full CPU RL pipeline

Use the reusable image `virne-python-oracle:py310-rl-full-cpu` for the complete
Python application path. It adds Hydra, OR-Tools, scikit-learn, higher,
colorlog, TensorBoard, W&B and the Torch 2.6 CPU / PyG 2.7 wheels to the pinned
CPU base. The native checkpoint converter
`porting/convert_cpp_checkpoint.py` emits a Python `state_dict` without editing
the original Python checkout. The installation and loaded one-request smoke
are recorded in `porting/results/python_pipeline_install_2026-09-15.json`.
