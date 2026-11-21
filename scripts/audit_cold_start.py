"""Compare fixed 0 versus fixed 1 for users unseen before each holdout."""

import argparse
from pathlib import Path

import pandas as pd

from activity_features import ACTIVITIES, ActivityData
from evaluate import scores
from run_relational_study import FOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/extended/db"))
    parser.add_argument("--study", type=Path, default=Path("relational_study"))
    parser.add_argument("--test", type=Path, default=Path("submission/test_input.csv"))
    args = parser.parse_args()
    data = ActivityData(args.db)
    known = data.users.set_index("github_user_id")["first_seen_at"]
    rows = []
    for fold in FOLDS:
        name, cutoff = fold["name"], pd.Timestamp(fold["eval"])
        pred = pd.read_csv(args.study / "predictions" / f"{name}_C_user_temporal.csv")
        truth = pd.read_csv(args.study / "predictions" / f"{name}_truth.csv")
        unseen = ~pred["UserID"].map(known).lt(cutoff).fillna(False)
        for activity in ACTIVITIES:
            values = {}
            for default in (0, 1):
                changed = pred.copy()
                changed.loc[unseen, activity] = default
                values[default] = scores(changed, truth)["per_event"][activity]
            rows.append({
                "fold": name, "activity": activity, "scored_in_fold": (
                    activity != "push" or name == "sep26_six"
                ),
                "unseen_users": int(unseen.sum()),
                "observed_positives_among_unseen": int(truth.loc[unseen, activity].sum()),
                "f1_default_zero": values[0], "f1_default_one": values[1],
            })
    output = args.study / "metrics" / "cold_start_strategy.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    missing = pd.read_csv(args.test)["UserID"].isin(data.users["github_user_id"])
    print("Current test users with no Users row:", int((~missing).sum()))
    print(pd.DataFrame(rows).to_string(index=False))
    print("Wrote", output)


if __name__ == "__main__":
    main()
