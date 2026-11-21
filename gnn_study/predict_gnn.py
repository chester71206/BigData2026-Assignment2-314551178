"""Reproduce a historical GNN prediction from its checkpoint and thresholds."""

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
from run_lightgbm import make_output  # noqa: E402
from train_gnn import HeteroSAGE, TwoLayerGCN, homogeneous  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "gnn_study")
    parser.add_argument("--fold", default="oct17_five")
    parser.add_argument("--arm", choices=["A_gcn", "B_gcn_temporal", "C_hetero"],
                        default="C_hetero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint_path = args.study / "models" / f"{args.fold}_{args.arm}_seed{args.seed}.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    cutoff = checkpoint["fold"]["eval"]
    stamp = pd.Timestamp(cutoff).strftime("%Y%m%d%H")
    graph = torch.load(args.study / "snapshots" / f"{stamp}.pt", weights_only=False)
    test = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
    index = pd.Index(graph.cohort_ids.numpy()).get_indexer(test)
    assert (index >= 0).all()
    if args.arm == "C_hetero":
        model = HeteroSAGE(graph.metadata(), graph["Users"].x.shape[1], checkpoint["hidden"])
        model_input = graph
    else:
        model = TwoLayerGCN(graph["Users"].x.shape[1], checkpoint["hidden"])
        model_input = homogeneous(graph)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(model_input, args.arm != "A_gcn"))[index].numpy()
    thresholds = json.loads((args.study / "logs" /
                             f"{args.fold}_{args.arm}_seed{args.seed}_thresholds.json").read_text())
    predicted = {a: (probabilities[:, j] >= thresholds[a]["threshold"]).astype(np.int8)
                 if a in thresholds else np.zeros(len(test), dtype=np.int8)
                 for j, a in enumerate(ACTIVITIES)}
    unseen = graph["Users"].x[graph.query_index, 12].numpy()[index] == 0
    for values in predicted.values():
        values[unseen] = 0
    output = make_output(test, predicted)
    path = args.output or (args.study / "predictions" /
                           f"reproduced_{args.fold}_{args.arm}_seed{args.seed}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Wrote {len(output)} users, {unseen.sum()} cold-start defaults: {path}")


if __name__ == "__main__":
    main()
