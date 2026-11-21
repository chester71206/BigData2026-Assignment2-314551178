"""Reproduce a final-refresh CSV from saved checkpoints and thresholds."""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from activity_features import ACTIVITIES, ActivityData
from run_lightgbm import assemble, make_output
from train_final_lightgbm import GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("final_refresh"))
    parser.add_argument("--test", type=Path, default=Path("submission/test_input.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.artifacts / "metrics/final_config.json").read_text())
    details = pd.read_csv(args.artifacts / "metrics/final_model_details.csv")
    cutoff = pd.Timestamp(config["prediction_time_utc"])
    data = ActivityData(args.db)
    cohort = [int(value) for value in Path("scripts/cohort.txt").read_text().split()]
    test = pd.read_csv(args.test)["UserID"].to_numpy(dtype=np.int64)
    features = data.features(cohort, cutoff)
    predicted = {}
    for _, row in details.iterrows():
        activity = row["activity"]
        x = assemble(features, GROUPS[row["arm"]]).loc[test]
        model_file = args.artifacts / "models" / f"best_{activity}.txt"
        if model_file.exists():
            probability = lgb.Booster(model_file=str(model_file)).predict(
                x, num_threads=config["threads"])
        else:
            constant_file = args.artifacts / "models" / f"best_{activity}.json"
            probability = np.full(len(test),
                                  json.loads(constant_file.read_text())["constant_probability"])
        predicted[activity] = (probability >= row["threshold"]).astype(np.int8)
    output = make_output(test, predicted)
    visible = set(data.users.loc[data.users["first_seen_at"].lt(cutoff), "github_user_id"])
    output.loc[~output["UserID"].isin(visible), ACTIVITIES] = 0
    target = args.output or args.artifacts / "predictions/reproduced_prediction.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    print(f"Wrote {len(output)} rows to {target}; prediction_time={cutoff} UTC")


if __name__ == "__main__":
    main()
