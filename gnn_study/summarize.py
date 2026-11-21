"""Consolidate GNN seeds, existing LightGBM scores, and relation masking plots."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "gnn_study")
    args = parser.parse_args()
    out = args.study
    base = pd.read_csv(out / "metrics/gnn_fold_metrics.csv")
    seeds = [base.loc[base["arm"].eq("C_hetero")]]
    for seed in (7, 123):
        seeds.append(pd.read_csv(out / f"stability_seed{seed}/metrics/gnn_fold_metrics.csv"))
    stability = pd.concat(seeds, ignore_index=True)
    stability.to_csv(out / "metrics/seed_stability.csv", index=False)
    light = pd.read_csv(ROOT / "relational_study/metrics/fold_metrics.csv")
    light = light.loc[light["arm"].isin(["B_user", "C_user_temporal", "F_repo_interactions"]),
                       ["fold", "arm", "score_5_without_push"]].copy()
    light["model_family"] = "LightGBM"
    gnn = base[["fold", "arm", "score_5_without_push"]].copy()
    gnn["model_family"] = "GNN"
    pd.concat([light, gnn], ignore_index=True).to_csv(
        out / "metrics/gnn_vs_lightgbm.csv", index=False)
    ablation = pd.read_csv(out / "metrics/relation_ablation.csv")
    rank = (ablation.groupby(["source", "relation"], as_index=False)["drop_when_masked"]
            .mean().sort_values("drop_when_masked", ascending=False))
    rank["edge"] = rank["source"] + "." + rank["relation"]
    rank.to_csv(out / "metrics/relation_masking_rank.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    selected = rank.head(12).iloc[::-1]
    ax.barh(selected["edge"], 100 * selected["drop_when_masked"])
    ax.set_xlabel("Mean five-activity Score decrease when edge type is masked (pp)")
    ax.set_title("Held-out inference masking; model is not retrained")
    fig.tight_layout()
    fig.savefig(out / "figures/relation_masking.png", dpi=160)
    plt.close(fig)
    print("Stability:\n", stability.groupby("fold")["score_5_without_push"]
          .agg(["mean", "std", "min", "max"]).to_string())
    print("Top masked relations:\n", rank[["edge", "drop_when_masked"]]
          .head(8).to_string(index=False))


if __name__ == "__main__":
    main()
