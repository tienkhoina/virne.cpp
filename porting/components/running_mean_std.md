# Running mean and variance

## Status

This leaf ports the `RunningMeanStd` and
`update_mean_var_count_from_moments` utilities from
`virne/solver/learning/utils.py` as a fixed, float64 C++ API.  The state is a
dense feature lane (`mean`, `variance`, `count`); no string-keyed map or hidden
worker is used in the update loop.

## API

Header: `virne/solver/learning/rl_core/running_mean_std.h`

- `RunningMeanStd(feature_count = 1, epsilon = 1e-4)` initializes zero means,
  unit population variances, and the Python epsilon count.
- `update(samples, sample_count)` consumes row-major
  `[sample_count, feature_count]` values and computes population moments
  (`ddof=0`) before merging them into the running state.
- `update_from_moments(batch_mean, batch_variance, batch_count)` merges a
  prepared moment batch. A one-element mean/variance lane is broadcast once at
  this boundary; all hot arithmetic uses direct vector indices.
- The free `update_mean_var_count_from_moments` function returns a new typed
  moment triple without mutating the caller's lanes.
- Invalid feature counts, sample lengths, and moment lengths raise
  `RunningMeanStdException` with a typed `RunningMeanStdErrorCode`.

The recurrence and sequential reduction preserve Python's mutation order.
Independent statistics objects may be updated by caller-owned workers; one
object is intentionally not internally synchronized.

## Verification

```powershell
cmake --build build-rollout-win-msvc --config Release `
  --target vne_running_mean_std_unit vne_running_mean_std_harness --parallel 1
ctest --test-dir build-rollout-win-msvc -C Release `
  -R '^vne_running_mean_std_unit$' --output-on-failure
python porting/compare_running_mean_std.py `
  --native .\build-rollout-win-msvc\porting\Release\vne_running_mean_std_harness.exe `
  --benchmark --threads 1 --warmups 2 --repetitions 5 --iterations 25 `
  --json-output porting/results/running_mean_std_differential_YYYY-MM-DD.json
```

The Windows MSVC Release CTest passed and the comparator reports `4/4` exact
state cases plus all three typed error cases.  The one `4096 x 8` benchmark
uses Virne's actual NumPy path (`np.array`, `np.mean`, `np.var(ddof=0)`), not a
pure-Python substitute.  With one warmup, three samples, and five iterations,
native median time was `25,300 ns/call` versus Python `237,360 ns/call`
(`9.382x` native speedup).  The shared output checksum is
`15486242895814385222`; reports are stored in
`porting/results/running_mean_std_differential_2026-08-05.json` and
`porting/results/running_mean_std_benchmark_2026-08-05.json`.
