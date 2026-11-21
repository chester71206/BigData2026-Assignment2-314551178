"""Remove future-visible rows and verify that a past graph is identical."""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from activity_features import ActivityData, ROW_TIMES, TABLE_NAMES  # noqa: E402
from graph_dataset import build_snapshot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/extended/db")
    parser.add_argument("--cutoff", default="2026-10-17 16:00:00")
    args = parser.parse_args()
    cohort = [int(v) for v in (ROOT / "scripts/cohort.txt").read_text().split()]
    full = ActivityData(args.db)
    original, _, _, _ = build_snapshot(full, cohort, args.cutoff, True)
    with tempfile.TemporaryDirectory() as temp:
        for name in TABLE_NAMES:
            table = full.tables[name]
            table.loc[table[ROW_TIMES[name]].lt(pd.Timestamp(args.cutoff))].to_parquet(
                Path(temp) / f"{name}.parquet", index=False)
        restricted = ActivityData(temp)
        rebuilt, _, _, _ = build_snapshot(restricted, cohort, args.cutoff, True)
    assert original.node_types == rebuilt.node_types
    assert original.edge_types == rebuilt.edge_types
    for kind in original.node_types:
        assert torch.equal(original[kind].x, rebuilt[kind].x), kind
    for edge in original.edge_types:
        assert torch.equal(original[edge].edge_index, rebuilt[edge].edge_index), edge
    assert torch.equal(original.query_index, rebuilt.query_index)
    print(f"PASS: all node features, query indices and {len(original.edge_types)} edge types "
          f"unchanged after deleting rows first visible at/after {args.cutoff} UTC")


if __name__ == "__main__":
    main()
