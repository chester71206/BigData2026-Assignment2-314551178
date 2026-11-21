"""Paired user bootstrap for five-activity model differences across folds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from activity_features import ACTIVITIES
from evaluate import scores
from run_relational_study import FOLDS


def score_arrays(pred, truth):
    tp = ((pred == 1) & (truth == 1)).sum(axis=0)
    fp = ((pred == 1) & (truth == 0)).sum(axis=0)
    fn = ((pred == 0) & (truth == 1)).sum(axis=0)
    denominators = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominators, out=np.ones_like(tp, dtype=float),
                   where=denominators != 0)
    micro_den = int(2 * tp.sum() + fp.sum() + fn.sum())
    micro = float(2 * tp.sum() / micro_den) if micro_den else 1.0
    return float((f1.mean() + micro) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("relational_study"))
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    comparisons = [
        ("C_user_temporal", "B_user", "temporal_minus_user"),
        ("R_repo_volume", "C_user_temporal", "repo_minus_temporal"),
        ("I_interactions", "C_user_temporal", "interaction_minus_temporal"),
        ("F_repo_interactions", "C_user_temporal", "combined_minus_temporal"),
    ]
    rows = []
    for fold in FOLDS:
        name = fold["name"]
        truth_df = pd.read_csv(args.study / "predictions" / f"{name}_truth.csv")
        truth = truth_df[ACTIVITIES[:-1]].to_numpy(dtype="int8")
        arrays = {}
        for arm in {arm for left, right, _ in comparisons for arm in (left, right)}:
            df = pd.read_csv(args.study / "predictions" / f"{name}_{arm}.csv")
            if not df["UserID"].equals(truth_df["UserID"]):
                raise ValueError(f"{name}/{arm}: UserID order mismatch")
            arrays[arm] = df[ACTIVITIES[:-1]].to_numpy(dtype="int8")
            expected = scores(df, truth_df, ACTIVITIES[:-1])["score"]
            if abs(score_arrays(arrays[arm], truth) - expected) > 1e-12:
                raise ValueError("Bootstrap score differs from teacher score")
        for left, right, label in comparisons:
            observed = score_arrays(arrays[left], truth) - score_arrays(arrays[right], truth)
            deltas = np.empty(args.replicates)
            for i in range(args.replicates):
                index = rng.integers(0, len(truth), len(truth))
                deltas[i] = (score_arrays(arrays[left][index], truth[index])
                             - score_arrays(arrays[right][index], truth[index]))
            rows.append({
                "fold": name, "comparison": label, "delta_score": observed,
                "ci_low_2_5": float(np.quantile(deltas, 0.025)),
                "ci_high_97_5": float(np.quantile(deltas, 0.975)),
                "bootstrap_pr_positive": float((deltas > 0).mean()),
                "replicates": args.replicates,
            })
    result = pd.DataFrame(rows)
    result.to_csv(args.study / "metrics" / "bootstrap_differences.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
