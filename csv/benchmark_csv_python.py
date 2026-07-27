#!/usr/bin/env python3
"""Stdlib csv baseline matching benchmark_csv.cpp (no third-party install)."""

import csv
import os
import statistics
import tempfile
import time


ROW_COUNT = 200_000
SAMPLES = 5


def make_rows():
    return [
        [
            str(i),
            "true" if i % 3 == 0 else "false",
            f"{i / 17.0:.6f}",
            "{0: 4, 1: 9, 2: 12}",
            "[(0, 1), (1, 7), (7, 9)]",
        ]
        for i in range(ROW_COUNT)
    ]


def median_ms(function):
    samples = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return statistics.median(samples)


def main():
    columns = ["id", "accepted", "score", "node_slots", "link_paths"]
    rows = make_rows()
    path = os.path.join(tempfile.gettempdir(), "virne-csv-python-benchmark.csv")

    def write():
        with open(path, "w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\r\n")
            writer.writerow(columns)
            writer.writerows(rows)

    checksum = 0

    def read():
        nonlocal checksum
        with open(path, "r", encoding="utf-8", newline="") as source:
            loaded = list(csv.reader(source))
        checksum += len(loaded) - 1

    write_ms = median_ms(write)
    byte_count = os.path.getsize(path)
    read_ms = median_ms(read)
    os.remove(path)
    mib = byte_count / (1024.0 * 1024.0)

    print(
        f"rows={ROW_COUNT} bytes={byte_count} samples={SAMPLES} "
        f"statistic=median checksum={checksum}"
    )
    print("operation,elapsed_ms,rows_per_second,mib_per_second")
    print(
        f"write,{write_ms:.2f},{ROW_COUNT * 1000.0 / write_ms:.2f},"
        f"{mib * 1000.0 / write_ms:.2f}"
    )
    print(
        f"read,{read_ms:.2f},{ROW_COUNT * 1000.0 / read_ms:.2f},"
        f"{mib * 1000.0 / read_ms:.2f}"
    )


if __name__ == "__main__":
    main()
