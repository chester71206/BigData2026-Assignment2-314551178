"""Train chronological two-layer GCN and relation-aware GraphSAGE experiments."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from activity_features import ACTIVITIES, ActivityData  # noqa: E402
from evaluate import scores  # noqa: E402
from run_lightgbm import choose_threshold, make_output  # noqa: E402
from run_relational_study import FOLDS  # noqa: E402


class TwoLayerGCN(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, len(ACTIVITIES))
        self.dropout = dropout

    def forward(self, graph, temporal):
        x = graph.x if temporal else graph.x.clone()
        if not temporal:
            x[:, 9 + 18:] = 0
        x = F.relu(self.conv1(x, graph.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, graph.edge_index))
        return self.head(x[graph.query_index])


class HeteroSAGE(nn.Module):
    def __init__(self, metadata, input_dim, hidden=32, dropout=0.25,
                 omit_relation=None):
        super().__init__()
        node_types, edge_types = metadata
        self.project = nn.ModuleDict({kind: nn.Linear(input_dim, hidden) for kind in node_types})
        eligible = [e for e in edge_types if omit_relation is None or
                    (e[1] != omit_relation and e[1] != f"rev_{e[0]}_{omit_relation}")]
        self.convs = nn.ModuleList([
            HeteroConv({edge: SAGEConv((hidden, hidden), hidden) for edge in eligible},
                       aggr="sum") for _ in range(2)
        ])
        self.head = nn.Linear(hidden, len(ACTIVITIES))
        self.dropout = dropout

    def forward(self, graph, temporal=True, edge_index_override=None):
        x = {kind: F.relu(layer(graph[kind].x)) for kind, layer in self.project.items()}
        edges = edge_index_override if edge_index_override is not None else graph.edge_index_dict
        for conv in self.convs:
            updated = conv(x, edges)
            x = {kind: F.dropout(F.relu(updated.get(kind, x[kind])),
                                 p=self.dropout, training=self.training)
                 for kind in x}
        return self.head(x["Users"][graph.query_index])


def homogeneous(graph):
    homo = graph.to_homogeneous(node_attrs=["x"])
    users_offset = 0
    for name in graph.node_types:
        if name == "Users":
            break
        users_offset += graph[name].num_nodes
    homo.query_index = graph.query_index + users_offset
    return homo


def evaluate_teacher(out, fold, arm, pred, truth, skip):
    import subprocess
    pred_file = out / "predictions" / f"{fold}_{arm}.csv"
    truth_file = out / "predictions" / f"{fold}_truth.csv"
    pred.to_csv(pred_file, index=False)
    truth.to_csv(truth_file, index=False)
    command = [sys.executable, str(ROOT / "scripts/evaluate.py"),
               "--pred", str(pred_file), "--truth", str(truth_file)]
    if skip:
        command += ["--skip", *skip]
    result = subprocess.run(command, capture_output=True, text=True)
    (out / "logs" / f"{fold}_{arm}_evaluate.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return scores(pred, truth, [a for a in ACTIVITIES if a not in skip])


def train_one(fold, arm, seed, graphs, labels, test_index, active, out,
              epochs=24, patience=5, hidden=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    starts = fold["train"]
    tune = fold["tune"]
    val = fold["eval"]
    first_graph = graphs[starts[0]]
    dim = first_graph["Users"].x.shape[1]
    if arm in ("A_gcn", "B_gcn_temporal"):
        model = TwoLayerGCN(dim, hidden)
        inputs = {t: homogeneous(graphs[t]) for t in starts + [tune, val]}
        temporal = arm == "B_gcn_temporal"
    else:
        model = HeteroSAGE(first_graph.metadata(), dim, hidden)
        inputs = {t: graphs[t] for t in starts + [tune, val]}
        temporal = True
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    active_index = [ACTIVITIES.index(a) for a in active]
    y = {t: torch.from_numpy(labels[t].to_numpy(dtype=np.float32, copy=True))
         for t in starts + [tune]}
    train_y = torch.cat([y[t][:, active_index] for t in starts])
    positive = train_y.sum(0)
    pos_weight = ((len(train_y) - positive) / positive.clamp(min=1)).clamp(max=10)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    history = []
    peak_rss = process.memory_info().rss
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses = []
        for t in starts:
            logits = model(inputs[t], temporal)[:, active_index]
            losses.append(loss_fn(logits, y[t][:, active_index]))
        train_loss = torch.stack(losses).mean()
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            tune_logits = model(inputs[tune], temporal)[:, active_index]
            tune_loss = float(loss_fn(tune_logits, y[tune][:, active_index]))
        peak_rss = max(peak_rss, process.memory_info().rss)
        history.append({"epoch": epoch, "train_loss": float(train_loss),
                        "tune_loss": tune_loss})
        if tune_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad = tune_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start_time
    path = out / "models" / f"{fold['name']}_{arm}_seed{seed}.pt"
    torch.save({"state": best_state, "arm": arm, "seed": seed, "fold": fold,
                "hidden": hidden, "best_epoch": best_epoch}, path)
    model.eval()
    with torch.no_grad():
        tune_prob = torch.sigmoid(model(inputs[tune], temporal)).numpy()
        inference_start = time.perf_counter()
        eval_prob = torch.sigmoid(model(inputs[val], temporal))[test_index].numpy()
        inference_seconds = time.perf_counter() - inference_start
    thresholds = {}
    pred = np.zeros((len(test_index), len(ACTIVITIES)), dtype=np.int8)
    for a in active:
        j = ACTIVITIES.index(a)
        threshold, tune_f1 = choose_threshold(tune_prob[:, j], y[tune][:, j].numpy())
        thresholds[a] = {"threshold": float(threshold), "tune_f1": float(tune_f1)}
        pred[:, j] = (eval_prob[:, j] >= threshold).astype(np.int8)
    # Match the established cold-start policy for users without a visible Users row.
    unseen = inputs[val].x[inputs[val].query_index, 12].numpy() == 0 if arm in (
        "A_gcn", "B_gcn_temporal") else (
            inputs[val]["Users"].x[inputs[val].query_index, 12].numpy() == 0)
    pred[unseen[test_index], :] = 0
    pd.DataFrame(history).to_csv(out / "logs" / f"{fold['name']}_{arm}_seed{seed}_epochs.csv", index=False)
    (out / "logs" / f"{fold['name']}_{arm}_seed{seed}_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n")
    return pred, {"best_epoch": best_epoch, "epochs_run": len(history),
                  "tune_loss": best_loss, "train_seconds": train_seconds,
                  "inference_seconds": inference_seconds,
                  "peak_rss_gb": peak_rss / 1e9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/extended/db")
    parser.add_argument("--out", type=Path, default=ROOT / "gnn_study")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--arms", nargs="+", default=["A_gcn", "B_gcn_temporal", "C_hetero"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    for sub in ("metrics", "logs", "models", "predictions", "figures"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    cohort = [int(v) for v in (ROOT / "scripts/cohort.txt").read_text().split()]
    test = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
    test_index = pd.Index(cohort).get_indexer(test)
    assert (test_index >= 0).all()
    data = ActivityData(args.db)
    times = sorted({t for fold in FOLDS for t in fold["train"] + [fold["tune"], fold["eval"]]})
    graphs = {}
    labels = {}
    for t in times:
        stamp = pd.Timestamp(t).strftime("%Y%m%d%H")
        graphs[t] = torch.load(args.out / "snapshots" / f"{stamp}.pt", weights_only=False)
        labels[t] = data.labels(cohort, t)
    gap = pd.read_csv(ROOT / "quality_study/weekly_gap_flags.csv").set_index("week_start_utc")
    metrics_file = args.out / "metrics/gnn_fold_metrics.csv"
    events_file = args.out / "metrics/gnn_per_activity.csv"
    rows = pd.read_csv(metrics_file).to_dict("records") if metrics_file.exists() else []
    per_event = pd.read_csv(events_file).to_dict("records") if events_file.exists() else []
    for arm in args.arms:
        for fold in FOLDS:
            name = fold["name"]
            if any(row["fold"] == name and row["arm"] == arm and row["seed"] == args.seed
                   for row in rows):
                print(f"skip existing {arm} {name} seed={args.seed}", flush=True)
                continue
            weeks = fold["train"] + [fold["tune"], fold["eval"]]
            push_bad = int(gap.loc[weeks, "push_gap_days"].sum()) > 0
            active = ACTIVITIES[:-1] if push_bad else ACTIVITIES
            predicted, runtime = train_one(fold, arm, args.seed, graphs, labels,
                                           test_index, active, args.out,
                                           epochs=args.epochs)
            truth = make_output(test, labels[fold["eval"]].loc[test, ACTIVITIES])
            prediction = make_output(test, {a: predicted[:, j] for j, a in enumerate(ACTIVITIES)})
            skipped = ["push"] if push_bad else []
            result = evaluate_teacher(args.out, name, arm, prediction, truth, skipped)
            five = scores(prediction, truth, ACTIVITIES[:-1])
            rows.append({"fold": name, "arm": arm, "seed": args.seed,
                         "n_scored": len(active), "macro_f1": result["macro_f1"],
                         "micro_f1": result["micro_f1"], "score": result["score"],
                         "score_5_without_push": five["score"], **runtime})
            per_event.extend({"fold": name, "arm": arm, "activity": a,
                              "f1": result["per_event"].get(a, np.nan),
                              "truth_positives": int(truth[a].sum()),
                              "predicted_positives": int(prediction[a].sum())}
                             for a in ACTIVITIES)
            print(f"{arm} {name}: five={five['score']:.4f} score={result['score']:.4f} "
                  f"epochs={runtime['epochs_run']} train={runtime['train_seconds']:.1f}s",
                  flush=True)
            pd.DataFrame(rows).to_csv(metrics_file, index=False)
            pd.DataFrame(per_event).to_csv(events_file, index=False)
    light = pd.read_csv(ROOT / "relational_study/metrics/fold_metrics.csv")
    light = light.loc[light["arm"].isin(["B_user", "C_user_temporal", "F_repo_interactions"])]
    gnn = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for arm in gnn["arm"].unique():
        part = gnn.loc[gnn["arm"].eq(arm)]
        ax.plot(part["fold"], part["score_5_without_push"], marker="o", label=arm)
    for arm in light["arm"].unique():
        part = light.loc[light["arm"].eq(arm)]
        ax.plot(part["fold"], part["score_5_without_push"], linestyle="--", marker="s", label=arm)
    ax.set_ylabel("Five-activity Score (push excluded)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "figures/gnn_vs_lightgbm.png", dpi=160)
    plt.close(fig)
    (args.out / "metrics/run_config.json").write_text(json.dumps({
        "folds": FOLDS, "arms": sorted(gnn["arm"].unique().tolist()), "seed": args.seed,
        "epochs": args.epochs, "threads": args.threads,
        "device": "cpu", "torch_version": torch.__version__}, indent=2) + "\n")


if __name__ == "__main__":
    main()
