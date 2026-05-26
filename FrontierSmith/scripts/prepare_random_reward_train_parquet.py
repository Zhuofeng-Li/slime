#!/usr/bin/env python3
"""Copy a VERL parquet and tag its rows for random-reward training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Source VERL parquet")
    parser.add_argument("--output", type=Path, required=True, help="Tagged output parquet")
    parser.add_argument("--data-source", default="frontiercs_random_reward")
    args = parser.parse_args()

    table = pq.read_table(args.input)
    data_source = pa.array([args.data_source] * table.num_rows, type=pa.string())
    field = pa.field("data_source", pa.string())

    idx = table.schema.get_field_index("data_source")
    if idx >= 0:
        table = table.set_column(idx, field, data_source)
    else:
        table = table.append_column(field, data_source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output)
    print(f"Saved {table.num_rows} rows with data_source={args.data_source!r} -> {args.output}")


if __name__ == "__main__":
    main()
