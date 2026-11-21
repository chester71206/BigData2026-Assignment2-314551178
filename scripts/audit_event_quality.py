"""Audit six observed event types using only the eight preprocessed tables."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from activity_features import ACTIVITIES, ActivityData


def raw_daily(raw_dir, days):
    sizes = {}
    hours = {}
    for path in raw_dir.glob("*.json.gz"):
        try:
            day, hour = path.name.removesuffix(".json.gz").rsplit("-", 1)
            when = pd.Timestamp(day)
            int(hour)
        except ValueError:
            continue
        sizes[when] = sizes.get(when, 0) + path.stat().st_size
        hours[when] = hours.get(when, 0) + 1
    return (pd.Series(sizes).reindex(days, fill_value=0) / 1e6,
            pd.Series(hours).reindex(days, fill_value=0))


def longest_run(mask):
    runs = (mask != mask.shift()).cumsum()
    return max((int(group.sum()) for _, group in mask.groupby(runs) if group.iloc[0]), default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/extended/db"))
    parser.add_argument("--raw", type=Path, default=Path("raw_data"))
    parser.add_argument("--out", type=Path, default=Path("quality_study"))
    parser.add_argument("--start", type=pd.Timestamp, default=pd.Timestamp("2026-09-01"))
    parser.add_argument("--end", type=pd.Timestamp,
                        help="last UTC day to audit; defaults to latest day with 24 raw files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = ActivityData(args.db)
    first = args.start.normalize()
    if args.end is None:
        per_day = {}
        for path in args.raw.glob("*.json.gz"):
            try:
                day, hour = path.name.removesuffix(".json.gz").rsplit("-", 1)
                int(hour)
            except ValueError:
                continue
            per_day[day] = per_day.get(day, 0) + 1
        full_days = [pd.Timestamp(day) for day, count in per_day.items() if count == 24]
        if not full_days:
            raise ValueError("No complete UTC day found in raw data")
        last = max(full_days)
    else:
        last = args.end.normalize()
    days = pd.date_range(first, last, freq="D")
    event = data.events.copy()
    event["day"] = event["visible_at"].dt.floor("D")
    event = event.loc[event["day"].between(first, last)]
    counts = (event.groupby(["day", "activity"]).size().unstack(fill_value=0)
              .reindex(days, fill_value=0).reindex(columns=ACTIVITIES, fill_value=0))
    users = (event.groupby(["day", "activity"])["UserID"].nunique().unstack(fill_value=0)
             .reindex(days, fill_value=0).reindex(columns=ACTIVITIES, fill_value=0))
    raw_mb, raw_hours = raw_daily(args.raw, days)
    daily = pd.DataFrame({"day_utc": days, "raw_hours": raw_hours.to_numpy(),
                          "raw_MB": raw_mb.to_numpy()})
    for activity in ACTIVITIES:
        daily[f"{activity}_rows"] = counts[activity].to_numpy()
        daily[f"{activity}_users"] = users[activity].to_numpy()
        others = counts.drop(columns=[activity]).sum(axis=1)
        daily[f"{activity}_zero_gap"] = (
            counts[activity].eq(0).to_numpy()
            & (daily["raw_hours"].to_numpy() == 24)
            & others.gt(100).to_numpy()
        )
    daily.to_csv(args.out / "daily_activity_quality.csv", index=False)

    monthly = []
    for activity in ACTIVITIES:
        for month in ("2026-09", "2026-10", "2026-11"):
            part = daily.loc[daily["day_utc"].dt.strftime("%Y-%m").eq(month)]
            monthly.append({
                "month": month, "activity": activity, "days": len(part),
                "median_daily_rows": float(part[f"{activity}_rows"].median()),
                "total_rows": int(part[f"{activity}_rows"].sum()),
                "zero_gap_days": int(part[f"{activity}_zero_gap"].sum()),
                "median_raw_MB": float(part["raw_MB"].median()),
            })
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(args.out / "monthly_activity_quality.csv", index=False)

    weekly = []
    week_starts = pd.date_range("2026-09-05 16:00:00", last + pd.Timedelta(days=1), freq="7D")
    for start in week_starts:
        if start + pd.Timedelta(days=7) > last + pd.Timedelta(days=1):
            continue
        end = start + pd.Timedelta(days=7)
        overlapping = daily.loc[
            daily["day_utc"].ge(start.floor("D")) & daily["day_utc"].lt(end.ceil("D"))
        ]
        row = {"week_start_utc": str(start), "week_end_utc": str(end)}
        row.update({f"{activity}_gap_days": int(overlapping[f"{activity}_zero_gap"].sum())
                    for activity in ACTIVITIES})
        weekly.append(row)
    weekly = pd.DataFrame(weekly)
    weekly.to_csv(args.out / "weekly_gap_flags.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for activity, ax in zip(ACTIVITIES, axes.flat):
        ax.plot(days, counts[activity].to_numpy(), linewidth=1.3)
        gaps = daily.loc[daily[f"{activity}_zero_gap"], "day_utc"]
        ax.scatter(gaps, [0] * len(gaps), color="red", s=25, label="zero despite other activity")
        ax.set_title(activity)
        ax.set_ylabel("Visible rows per UTC day")
        if len(gaps):
            ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out / "daily_activity_quality.png", dpi=160)
    plt.close(fig)

    print("Daily row medians by month:")
    print(monthly.pivot(index="activity", columns="month", values="median_daily_rows").to_string())
    print("Zero-gap days by type:")
    for activity in ACTIVITIES:
        mask = daily[f"{activity}_zero_gap"]
        print(activity, int(mask.sum()), "longest run", longest_run(mask),
              "dates", ",".join(daily.loc[mask, "day_utc"].dt.strftime("%Y-%m-%d")))


if __name__ == "__main__":
    main()
