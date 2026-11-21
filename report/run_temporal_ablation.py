"""Incremental temporal ablation on the final LightGBM chronological folds.

Uses the same users, labels, split weeks, LightGBM search, threshold selection,
and teacher evaluator as scripts/train_final_lightgbm.py. Results stay in report/.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from activity_features import ACTIVITIES, ActivityData  # noqa: E402
from run_lightgbm import choose_threshold, fit_one, make_output, predict_model  # noqa: E402
from train_final_lightgbm import BACKTESTS, WEEKS  # noqa: E402
from evaluate import scores  # noqa: E402

OUT = ROOT / "report" / "temporal_ablation"
EVENTS = ACTIVITIES[:-1]  # push has documented late coverage gaps
WINDOWS = [7, 14, 30]


def temporal_columns():
    c7 = ["tmp_events_7d", "tmp_active_days_7d"] + [f"tmp_{a}_7d" for a in ACTIVITIES]
    c14 = ["tmp_events_14d", "tmp_active_days_14d"] + [f"tmp_{a}_14d" for a in ACTIVITIES]
    c30 = ["tmp_events_30d", "tmp_active_days_30d"] + [f"tmp_{a}_30d" for a in ACTIVITIES]
    longer = c14 + c30
    recency = ["tmp_any_recency_days"] + [f"tmp_{a}_recency_days" for a in ACTIVITIES]
    trend = ["tmp_events_week_delta"] + [f"tmp_{a}_week_delta" for a in ACTIVITIES]
    return {
        "U_history": [],
        "U_plus_7d": c7,
        "U_plus_7_14d": c7 + c14,
        "U_plus_7_14_30d": c7 + longer,
        "U_plus_windows_recency": c7 + longer + recency,
        "U_plus_windows_recency_trend": c7 + longer + recency + trend,
    }


def frame(group, columns):
    result = group["user"].copy()
    if columns:
        absent = set(columns) - set(group["temporal"].columns)
        if absent:
            raise ValueError(f"Missing temporal columns: {absent}")
        result = pd.concat([result, group["temporal"][columns]], axis=1)
    # Match run_lightgbm.assemble exactly; float dtype can change tree splits.
    return result.astype("float32")


def main():
    for folder in ("metrics", "logs", "predictions"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("temporal_ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    only = os.environ.get("TEMPORAL_ABLATION_ONLY")
    for handler in (logging.StreamHandler(), logging.FileHandler(OUT / "logs/train.log",
                                                               mode="a" if only else "w")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    begin = time.monotonic()
    data = ActivityData(ROOT / "data/through_2026-11-20-05/db")
    cohort = np.array([int(v) for v in (ROOT / "scripts/cohort.txt").read_text().split()], dtype=np.int64)
    test = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
    features = {week: data.features(cohort, week) for week in WEEKS[:9]}
    labels = {week: data.labels(cohort, week) for week in WEEKS[:9]}
    all_arms = temporal_columns()
    arms = {only: all_arms[only]} if only else all_arms
    if only:
        rows = pd.read_csv(OUT / "metrics/scores.csv").to_dict("records")
        event_rows = pd.read_csv(OUT / "metrics/per_activity.csv").to_dict("records")
        tune_rows = pd.read_csv(OUT / "metrics/tuning.csv").to_dict("records")
        if any(row["arm"] == only for row in rows):
            raise ValueError(f"Arm {only} already exists; refusing to duplicate it")
    else:
        rows, event_rows, tune_rows = [], [], []
    for fold in BACKTESTS:
        name, train, tune, ev = fold["name"], fold["train"], fold["tune"], fold["eval"]
        truth = make_output(test, labels[ev].loc[test, ACTIVITIES])
        truth_path = OUT / "predictions" / f"{name}_truth.csv"
        truth.to_csv(truth_path, index=False)
        for arm, cols in arms.items():
            x_train = pd.concat([frame(features[w], cols) for w in train], ignore_index=True)
            x_tune = frame(features[tune], cols)
            x_eval = frame(features[ev], cols).loc[test]
            pred = {"push": np.zeros(len(test), dtype=np.int8)}
            for activity in EVENTS:
                y_train = np.concatenate([labels[w][activity].to_numpy() for w in train])
                y_tune = labels[tune][activity].to_numpy()
                model, constant, _, rounds, loss, tune_prob = fit_one(
                    x_train, y_train, x_tune, y_tune, 4, logger,
                    f"{name}/{arm}/{activity}")
                threshold, tune_f1 = choose_threshold(tune_prob, y_tune)
                prob = predict_model(model, constant, x_eval, 4)
                pred[activity] = (prob >= threshold).astype(np.int8)
                tune_rows.append({"fold": name, "arm": arm, "activity": activity,
                                  "threshold": threshold, "tune_f1": tune_f1,
                                  "tune_logloss": loss, "best_rounds": rounds})
            predicted = make_output(test, pred)
            pred_path = OUT / "predictions" / f"{name}_{arm}.csv"
            predicted.to_csv(pred_path, index=False)
            proc = subprocess.run([
                sys.executable, str(ROOT / "scripts/evaluate.py"),
                "--pred", str(pred_path), "--truth", str(truth_path), "--skip", "push",
            ], text=True, capture_output=True, check=True)
            (OUT / "logs" / f"evaluate_{name}_{arm}.log").write_text(proc.stdout)
            result = scores(predicted, truth, EVENTS)
            rows.append({"fold": name, "arm": arm, "feature_count": x_eval.shape[1],
                         "macro_f1": result["macro_f1"], "micro_f1": result["micro_f1"],
                         "score": result["score"]})
            event_rows.extend({"fold": name, "arm": arm, "activity": activity,
                               "f1": result["per_event"][activity]} for activity in EVENTS)
            pd.DataFrame(rows).to_csv(OUT / "metrics/scores.csv", index=False)
            pd.DataFrame(event_rows).to_csv(OUT / "metrics/per_activity.csv", index=False)
            pd.DataFrame(tune_rows).to_csv(OUT / "metrics/tuning.csv", index=False)
            logger.info("RESULT %s %s Score=%.4f elapsed=%.1fs", name, arm,
                        result["score"], time.monotonic() - begin)
    summary = (pd.DataFrame(rows).groupby("arm", sort=False)["score"]
               .agg(["mean", "std", "min", "max"]))
    summary.to_csv(OUT / "metrics/summary.csv")
    (OUT / "metrics/run.json").write_text(json.dumps({
        "source_db": "data/through_2026-11-20-05/db",
        "folds": BACKTESTS, "test_user_count": len(test), "cohort_count": len(cohort),
        "scored_activities": EVENTS, "excluded_activity": "push",
        "arms": {key: list(cols) for key, cols in all_arms.items()},
        "seconds_last_command": time.monotonic() - begin,
        "incremental_only_arm": only,
    }, indent=2))
    logger.info("DONE elapsed=%.1fs\n%s", time.monotonic() - begin, summary)


if __name__ == "__main__":
    main()
