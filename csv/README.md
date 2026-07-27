# CSV contract

`csv_lib` reads and writes RFC 4180-style CSV without an external runtime.
The existing two-argument `read_csv()` and `write_csv()` APIs remain valid.

## Correctness

- The parser is stream based and preserves commas, doubled quotes, CRLF, lone
  CR/LF, and embedded newlines inside quoted fields.
- The writer quotes only when required, doubles quotes, and emits CRLF record
  terminators.
- UTF-8 BOM input is accepted. Malformed quoting, duplicate column names, and
  rows with a different width from the header produce exceptions.
- `append_csv()` writes the header only for a new or empty destination and
  validates the existing ordered schema before modifying the file. It also
  separates an existing final record that has no newline.

This is sufficient for Recorder values such as serialized `node_slots` and
`link_paths`, which commonly contain commas, quotes, or line breaks.

## Typed Counter access

Use `column_index()` once before a hot loop, then access by numeric index:

```cpp
const auto revenue = frame.column_index("revenue");
double total = 0.0;
for (std::size_t row = 0; row < frame.nrows(); ++row)
    total += frame.double_at(row, revenue);
```

`int64_at()`, `double_at()`, and `bool_at()` use strict conversions and include
the row/column position in conversion errors. Name-based overloads are useful
outside hot loops.

## Verification

```bash
ctest --test-dir build -R '^csv_rfc4180$' --output-on-failure
cmake --build build --target csv_benchmark -j2
build/csv/csv_benchmark
.venv/bin/python csv/benchmark_csv_python.py
```

The benchmark reports the median of five 200,000-row round trips and remains
outside the default build. The Python baseline uses only the standard-library
`csv` module and writes the same columns, values, quoting policy, and CRLF line
terminators, so it does not install pandas or any package into the environment.

### Reference Release run (2026-07-24 08:56 UTC)

Both implementations produced the same 14,833,394-byte file. Each number is
the median of five samples on the current workspace host.

| Operation | C++ | Python `csv` | C++ speedup |
|---|---:|---:|---:|
| Write 200,000 rows | 202.99 ms | 610.75 ms | 3.01x |
| Read 200,000 rows | 456.55 ms | 920.13 ms | 2.02x |

The C++ throughput was 985,249 rows/s for writing and 438,073 rows/s for
reading. Host load affects timings; rerun the commands above for release
decisions on another machine.
