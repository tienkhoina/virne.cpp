# Component API: `core.Logger`

State: **COMPLETE / FROZEN (NON-ML)** on 2026-07-29.

Python oracle: `../virne/virne/core/logger.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`274189AF7211ABA134C4696B2DA8190F743B2F59D4287CF56843510F0B2C7083`,
5,651 bytes and 136 physical lines.

The separable non-ML leaf is standard console/file logging plus ordered metric
CSV persistence. `utils.config` supplies the completed run-directory rule and
the frozen CSV writer owns RFC4180/schema behavior. Python `colorlog` output is
represented directly with its five fixed ANSI level codes. OmegaConf is only
constructor decoding and is replaced by direct fields.

WandB and TensorBoard are eager optional Python imports. Native production
does not import or link WandB, Torch, TensorBoard, RL, solver, system, plotting,
or CUDA. A generic cold `LoggerSink` interface is prepared so a future optional
module can own those dependencies without changing the core logger.

## Fixed schema and ID rule

Backend selection, level, paths, options, operations, errors, event, and metric
row are direct fields/enums. The two native backends are direct booleans, never
a string set/map. Metric names are genuinely dynamic: each name is registered
once and resolves to object-local `LoggerMetricId`; repeated metric formatting
and CSV row loops use only dense ID-indexed slots. The interval-display keyword
flag is precomputed once during registration, so the log loop performs no
substring/name lookup.

## Stable non-ML C++ API

```cpp
enum class LoggerLevel : std::uint8_t {
    debug, info, warning, error, critical
};

struct LoggerBackends { bool console = true; bool file = false; };
struct LoggerOptions { std::size_t workers = 1U; };

struct LoggerConfig {
    std::filesystem::path save_root_dir;
    std::string solver_name;
    std::string run_id;
    std::string log_dir_name = "logs";
    std::string log_file_name = "run.log";
    LoggerBackends backends;
    LoggerLevel level = LoggerLevel::warning;
    std::size_t log_show_interval = 1U;
    std::string project_name;
    std::string experiment_name;
};

struct LoggerMetricId { std::uint32_t value; };
struct LoggerMetricRow {
    std::optional<std::int64_t> step;
    std::vector<std::optional<double>> values;
};
struct LoggerEntry {
    std::string message;
    LoggerLevel level = LoggerLevel::info;
    std::optional<LoggerMetricRow> metrics;
};

struct LoggerEventView {
    std::string_view message;
    LoggerLevel level;
    std::optional<std::int64_t> step;
    const std::vector<std::optional<double>>* metric_values;
};

class LoggerSink {
public:
    virtual ~LoggerSink() = default;
    virtual void write(const Logger&, const LoggerEventView&) = 0;
    virtual void close() noexcept = 0;
};

enum class LoggerErrorCode : std::uint8_t;
enum class LoggerOperation : std::uint8_t;
class LoggerException : public std::runtime_error;

class Logger {
public:
    explicit Logger(LoggerConfig);
    ~Logger();

    LoggerMetricId register_metric(std::string_view name);
    std::vector<LoggerMetricId> register_metrics(
        const std::vector<std::string>& names);
    void log(const LoggerEntry&);
    void log_batch(const std::vector<LoggerEntry>&, LoggerOptions = {});

    void debug(std::string_view message);
    void info(std::string_view message);
    void warning(std::string_view message);
    void error(std::string_view message);
    void critical(std::string_view message);

    void attach_sink(std::shared_ptr<LoggerSink>);
    void close() noexcept;

    const LoggerConfig& config() const noexcept;
    const std::filesystem::path& log_dir() const noexcept;
    const std::optional<std::filesystem::path>& log_file_path() const noexcept;
    const std::vector<std::string>& metric_names() const noexcept;
};
```

## Observable behavior

- Constructor validates all path components, creates the run log directory,
  and opens the file backend in append mode. File backend with an empty name is
  a typed error before logging.
- Standard output is filtered by configured severity. Non-empty messages use
  exact `colorlog`-compatible bytes: ANSI level color, uppercase level padded
  to eight columns, reset, one space, message, newline. Console writes to
  stderr; file uses the same colored format as Python.
- A metric row requires the current dense schema width. When either standard
  backend exists it appends `update_time` plus registered metric columns to
  `training_info.csv`, with header once and exact ordered schema validation.
- For a nonzero step divisible by `log_show_interval`, one standard line is
  emitted at the entry level. Only metric IDs whose name registration matched
  `loss`, `prob`, `return`, `penalty`, or `value` participate; formatting is
  signed four-decimal and preserves ID order.
- Empty message suppresses only the first standard line. External sinks still
  receive the typed event, matching Python optional-backend behavior.
- `close` is idempotent, flushes/closes the file and calls every attached sink
  once. The destructor never throws.
- Missing step, a zero interval, or a missing selected progress value fails
  after the standard message and metric CSV row but before the progress line
  and optional sinks, preserving Python's observable partial order.

Schema growth is allowed until the first metric CSV write, then rejected before
the schema/file changes. This typed safety rule replaces Python's silent
header/value misalignment. Arbitrary objects, string-keyed per-call mappings,
global root-logger handler mutation, custom logging levels/formatters, encoding
plugins, and filesystem behavior outside the configured directory are explicit
boundaries.

## Parallel and performance contract

Single-entry logging is ordered and serialized. `log_batch` may format
independent fixed entries into pre-sized slots with caller workers (`0/1`
sequential, wider values bounded by entry count), then writes console/file/CSV
and invokes sinks strictly in input order. The lowest input error is surfaced;
no host-derived worker policy is embedded. Logger instance methods are
internally synchronized; ordering across concurrent independent callers is the
order in which they acquire the logger, while one batch remains contiguous.

The accepted compact benchmark used a pre-registered eight-ID dense metric
schema and 4,096 batch entries at workers `1/2/8`, one warm-up and three
samples. Directory creation, schema registration, process startup, file
reading, checksum work, and cleanup stayed outside timing. It gated exact CSV
bytes/rows/schema and an empty standard log, measuring C++ speedups of 14.878x,
16.951x, and 13.689x. The protocol is frozen in
`porting/results/logger_benchmark_2026-07-29.json`; do not rerun or edit it.

## Deferred boundary

WandB initialization/log/finish, TensorBoard `SummaryWriter`, Torch, RL reward
and feature data, training orchestration, solver/system, plotting, network
transport, and backend-specific config remain out of scope. `project_name` and
`experiment_name` are retained only as fixed metadata for a future optional
sink; core Logger never interprets them.

## Frozen handoff

The exact Logger source hash is
`274189AF7211ABA134C4696B2DA8190F743B2F59D4287CF56843510F0B2C7083`.
The AST-isolated differential passed 9/9 shared cases, including native workers
`1/2/8`; strict GCC 11, the focused unit suite, ASan/UBSan/leak checks,
concurrent callers, and targeted frozen-integrity CTest passed. See
`porting/results/logger_2026-07-29.md` for the compact API/performance handoff.
