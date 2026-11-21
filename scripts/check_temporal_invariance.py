"""Check that removing all rows first visible after a cutoff changes no features."""

import argparse
import tempfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from activity_features import ActivityData, ROW_TIMES, TABLE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/extended/db"))
    parser.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp("2026-11-07 16:00:00"))
    args = parser.parse_args()
    users = [int(v) for v in Path("scripts/cohort.txt").read_text().split()]
    full = ActivityData(args.db)
    original = full.features(users, args.cutoff)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in TABLE_NAMES:
            table = full.tables[name]
            table.loc[table[ROW_TIMES[name]].lt(args.cutoff)].to_parquet(root / f"{name}.parquet", index=False)
        restricted = ActivityData(root)
        comparison = restricted.features(users, args.cutoff)
    for name in original:
        assert_frame_equal(original[name], comparison[name], check_exact=True)
        print(f"{name}: {original[name].shape[1]} features unchanged")
    print(f"PASS: no feature changed after removing rows first visible at/after {args.cutoff} UTC")


if __name__ == "__main__":
    main()
