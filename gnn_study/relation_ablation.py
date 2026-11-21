"""Measure held-out score change after masking each FK relation at inference."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from activity_features import ACTIVITIES  # noqa: E402
from evaluate import scores  # noqa: E402
from run_lightgbm import make_output  # noqa: E402
from run_relational_study import FOLDS  # noqa: E402
from train_gnn import HeteroSAGE, evaluate_teacher  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "gnn_study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    test = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
    relation_table = pd.read_csv(args.study / "metrics/graph_relations.csv")
    results = []
    for fold in FOLDS:
        name, cutoff = fold["name"], fold["eval"]
        stamp = pd.Timestamp(cutoff).strftime("%Y%m%d%H")
        graph = torch.load(args.study / "snapshots" / f"{stamp}.pt", weights_only=False)
        ids = graph.cohort_ids.numpy()
        test_index = pd.Index(ids).get_indexer(test)
        checkpoint = torch.load(args.study / "models" / f"{name}_C_hetero_seed{args.seed}.pt",
                                weights_only=False)
        model = HeteroSAGE(graph.metadata(), graph["Users"].x.shape[1], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state"])
        model.eval()
        thresholds = json.loads((args.study / "logs" /
                                  f"{name}_C_hetero_seed{args.seed}_thresholds.json").read_text())
        truth = pd.read_csv(args.study / "predictions" / f"{name}_truth.csv")
        baseline = pd.read_csv(args.study / "predictions" / f"{name}_C_hetero.csv")
        base_score = scores(baseline, truth, ACTIVITIES[:-1])["score"]
        active = list(thresholds)
        unseen = graph["Users"].x[graph.query_index, 12].numpy()[test_index] == 0
        for _, row in relation_table.loc[relation_table["cutoff_utc"].eq(cutoff)].iterrows():
            source, field, target = row["source"], row["relation"], row["target"]
            forward = (source, f"{field}_to_{target}", target)
            reverse = (target, f"rev_{source}_{field}", source)
            edges = {key: value for key, value in graph.edge_index_dict.items()
                     if key not in (forward, reverse)}
            with torch.no_grad():
                prob = torch.sigmoid(model(graph, edge_index_override=edges))[test_index].numpy()
            predicted = {a: (prob[:, j] >= thresholds[a]["threshold"]).astype(np.int8)
                         if a in active else np.zeros(len(test), dtype=np.int8)
                         for j, a in enumerate(ACTIVITIES)}
            for values in predicted.values():
                values[unseen] = 0
            output = make_output(test, predicted)
            label = f"C_without_{source}_{field}"
            result = evaluate_teacher(args.study, name, label, output, truth,
                                      ["push"] if "push" not in active else [])
            five = scores(output, truth, ACTIVITIES[:-1])["score"]
            results.append({"fold": name, "source": source, "relation": field,
                            "target": target, "edges_removed": int(row["edges"]),
                            "baseline_score_5": base_score,
                            "masked_score_5": five,
                            "drop_when_masked": base_score - five,
                            "teacher_score": result["score"]})
            print(name, source, field, f"delta={base_score-five:+.4f}", flush=True)
    pd.DataFrame(results).to_csv(args.study / "metrics/relation_ablation.csv", index=False)


if __name__ == "__main__":
    main()
