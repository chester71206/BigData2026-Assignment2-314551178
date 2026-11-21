"""Reproduce predictions from saved LightGBM checkpoints and thresholds."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from activity_features import ACTIVITIES, ActivityData
from run_lightgbm import GROUPS, archive_hours, assemble, make_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/extended/db"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--raw", type=Path, default=Path("raw_data"))
    parser.add_argument("--test", type=Path, default=Path("submission/test_input.csv"))
    parser.add_argument("--prediction-time", type=pd.Timestamp)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions/reproduced_prediction.csv"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads((args.artifacts / "metrics" / "run_config.json").read_text())
    cutoff = args.prediction_time or pd.Timestamp(config["prediction_time_utc"])
    latest = pd.Timestamp(archive_hours(args.raw)["last_hour_utc"]) + pd.Timedelta(hours=1)
    if latest > pd.Timestamp(config["official_cutoff_utc"]):
        raise ValueError("Raw files extend into test week; use a pre-cutoff snapshot")
    if cutoff > latest or cutoff > pd.Timestamp(config["official_cutoff_utc"]):
        raise ValueError("Prediction cutoff exceeds available raw data or official cutoff")
    data = ActivityData(args.db)
    users = [int(value) for value in Path("scripts/cohort.txt").read_text().split()]
    test_users = pd.read_csv(args.test)["UserID"].to_numpy(dtype="int64")
    groups = data.features(users, cutoff)
    winner = config["best_experiment"]
    prediction = {}
    if winner == "A_copy_last_week":
        for activity in ACTIVITIES:
            prediction[activity] = (
                groups["temporal"].loc[test_users, f"tmp_{activity}_7d"] > 0
            ).to_numpy(dtype="int8")
    else:
        frame = assemble(groups, GROUPS[winner]).loc[test_users]
        thresholds = pd.read_csv(args.artifacts / "metrics" / "tuning.csv").set_index(
            ["experiment", "activity"]
        )
        for activity in ACTIVITIES:
            path = args.artifacts / "models" / f"best_{activity}.txt"
            if path.exists():
                model = lgb.Booster(model_file=str(path))
                probability = model.predict(frame, num_threads=args.threads)
            else:
                constant = json.loads(
                    (args.artifacts / "models" / f"best_{activity}.json").read_text()
                )["constant_probability"]
                probability = np.full(len(frame), constant)
            threshold = float(thresholds.loc[(winner, activity), "threshold"])
            prediction[activity] = (probability >= threshold).astype("int8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = make_output(test_users, prediction)
    observed_users = set(data.users.loc[
        data.users["first_seen_at"].lt(cutoff), "github_user_id"
    ])
    output.loc[~output["UserID"].isin(observed_users), ACTIVITIES] = 0
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(test_users)} predictions to {args.output}; cutoff={cutoff} UTC")


if __name__ == "__main__":
    main()
