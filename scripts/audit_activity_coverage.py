"""Summarize observed cohort labels and raw archive size by seven-day window."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from activity_features import ACTIVITIES, ActivityData
from run_lightgbm import archive_hours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/extended/db"))
    parser.add_argument("--raw", type=Path, default=Path("raw_data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/metrics"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = ActivityData(args.db)
    users = [int(value) for value in Path("scripts/cohort.txt").read_text().split()]
    raw = archive_hours(args.raw)
    sizes = {}
    for path in args.raw.glob("*.json.gz"):
        day, hour = path.name.removesuffix(".json.gz").rsplit("-", 1)
        sizes[pd.Timestamp(f"{day} {int(hour):02d}:00:00")] = path.stat().st_size
    rows = []
    for start in pd.date_range("2026-09-05 16:00:00", "2026-11-07 16:00:00", freq="7D"):
        end = start + pd.Timedelta(days=7)
        hours = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
        label = data.labels(users, start)
        row = {
            "week_start_utc": str(start), "week_end_utc": str(end),
            "raw_hours_present": sum(hour in sizes for hour in hours),
            "raw_gb": sum(sizes.get(hour, 0) for hour in hours) / 1e9,
        }
        row.update({name: int(label[name].sum()) for name in ACTIVITIES})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "activity_coverage.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for activity in ACTIVITIES:
        axes[0].plot(frame["week_start_utc"].str[:10], frame[activity], marker="o", label=activity)
    axes[0].set_ylabel("Observed positive cohort users")
    axes[0].legend(ncol=3)
    axes[0].set_title("Observed activity coverage; downward shifts may reflect archive gaps")
    axes[1].bar(frame["week_start_utc"].str[:10], frame["raw_gb"])
    axes[1].set_ylabel("Raw archive GB in week")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(args.out.parent / "figures" / "activity_coverage.png", dpi=160)
    plt.close(fig)
    print(frame.to_string(index=False))
    print("Raw range:", raw["first_hour_utc"], "to", raw["last_hour_utc"])


if __name__ == "__main__":
    main()
