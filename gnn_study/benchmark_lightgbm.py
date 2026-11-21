"""Time the existing LightGBM fitting code on the same chronological folds."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from activity_features import ACTIVITIES, ActivityData  # noqa: E402
from run_lightgbm import assemble, fit_one, predict_model  # noqa: E402
from run_relational_study import FOLDS  # noqa: E402


class QuietLog:
    def info(self, *_args, **_kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/extended/db")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "gnn_study/metrics/lightgbm_benchmark.csv")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    data = ActivityData(args.db)
    cohort = [int(v) for v in (ROOT / "scripts/cohort.txt").read_text().split()]
    test = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
    starts = sorted({t for fold in FOLDS for t in fold["train"] + [fold["tune"], fold["eval"]]})
    features = {t: data.features(cohort, t) for t in starts}
    labels = {t: data.labels(cohort, t) for t in starts}
    gap = pd.read_csv(ROOT / "quality_study/weekly_gap_flags.csv").set_index("week_start_utc")
    process = psutil.Process()
    rows = []
    for fold in FOLDS:
        weeks = fold["train"] + [fold["tune"], fold["eval"]]
        active = ACTIVITIES[:-1] if gap.loc[weeks, "push_gap_days"].sum() else ACTIVITIES
        for arm, groups in (("B_user", ("user",)),
                            ("C_user_temporal", ("user", "temporal")),
                            ("F_repo_interactions", ("user", "temporal", "repo_only", "interaction"))):
            train_x = pd.concat([assemble(features[t], groups) for t in fold["train"]],
                                ignore_index=True)
            tune_x = assemble(features[fold["tune"]], groups)
            eval_x = assemble(features[fold["eval"]], groups).loc[test]
            train_start = time.perf_counter()
            peak = process.memory_info().rss
            models = []
            for activity in active:
                train_y = np.concatenate([labels[t][activity].to_numpy() for t in fold["train"]])
                tune_y = labels[fold["tune"]][activity].to_numpy()
                models.append(fit_one(train_x, train_y, tune_x, tune_y,
                                      args.threads, QuietLog(),
                                      f"{fold['name']}/{arm}/{activity}"))
                peak = max(peak, process.memory_info().rss)
            train_seconds = time.perf_counter() - train_start
            pred_start = time.perf_counter()
            for model, constant, *_ in models:
                predict_model(model, constant, eval_x, args.threads)
            inference_seconds = time.perf_counter() - pred_start
            rows.append({"fold": fold["name"], "arm": arm, "train_seconds": train_seconds,
                         "inference_seconds": inference_seconds,
                         "peak_rss_gb": peak / 1e9,
                         "trained_binary_models": len(active),
                         "note": "features precomputed; two LightGBM configs per activity"})
            print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
