"""Leakage-aware heterogeneous snapshots from the instructor's eight tables."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from activity_features import ActivityData, ROW_TIMES  # noqa: E402
from preprocess import TABLES  # noqa: E402
from run_relational_study import FOLDS  # noqa: E402

TYPES = tuple(TABLES)
SAFE_FKS = {
    name: {field: target for field, target in spec["fks"].items()
           if (name, field) != ("PullRequests", "merged_by_id")}
    for name, spec in TABLES.items()
}


def transform_features(frame):
    """Fixed transform learned from no future data; signed counts retain direction."""
    values = frame.to_numpy(dtype=np.float32, copy=True)
    names = frame.columns
    for j, name in enumerate(names):
        if name.endswith("_share") or name == "usr_seen":
            continue
        values[:, j] = np.sign(values[:, j]) * np.log1p(np.abs(values[:, j]))
    return values


def build_snapshot(data, cohort, cutoff, use_temporal, test_ids=None):
    cutoff = pd.Timestamp(cutoff)
    visible = {
        name: table.loc[table[ROW_TIMES[name]].lt(cutoff)].copy()
        for name, table in data.tables.items()
    }
    # Query anchors for cohort members not yet observed have no source row or edge.
    actual_users = visible["Users"]["github_user_id"].to_numpy(dtype=np.int64)
    virtual = np.setdiff1d(np.asarray(cohort, dtype=np.int64), actual_users)
    user_ids = np.concatenate([actual_users, virtual])
    query = pd.Index(user_ids).get_indexer(cohort)
    assert (query >= 0).all()
    feature_groups = data.features(cohort, cutoff)
    use = ["user", "temporal"] if use_temporal else ["user"]
    user_features = pd.concat([feature_groups[g] for g in use], axis=1)
    dim = 8 + 1 + user_features.shape[1]
    graph = HeteroData()
    index = {}
    for kind, name in enumerate(TYPES):
        table = visible[name]
        key = TABLES[name]["pk"]
        keys = table[key].to_numpy(dtype=np.int64)
        if name == "Users":
            count = len(user_ids)
            index[name] = pd.Index(keys)
        else:
            count = len(table)
            index[name] = pd.Index(keys)
        x = np.zeros((count, dim), dtype=np.float32)
        x[:, kind] = 1.0
        age = (cutoff - table[ROW_TIMES[name]]).dt.total_seconds().to_numpy() / 86400
        x[:len(table), 8] = np.log1p(np.maximum(age, 0)).astype(np.float32)
        if name == "Users":
            x[query, 9:] = transform_features(user_features)
        graph[name].x = torch.from_numpy(x)
        graph[name].num_nodes = count
    rows = []
    degree = {name: np.zeros(graph[name].num_nodes, dtype=np.int64) for name in TYPES}
    for source, fks in SAFE_FKS.items():
        source_table = visible[source]
        for field, target in fks.items():
            target_index = index[target]
            target_pos = target_index.get_indexer(source_table[field])
            source_pos = np.flatnonzero(target_pos >= 0).astype(np.int64)
            target_pos = target_pos[source_pos].astype(np.int64)
            edges = torch.from_numpy(np.stack([source_pos, target_pos]))
            relation = f"{field}_to_{target}"
            graph[source, relation, target].edge_index = edges
            graph[target, f"rev_{source}_{field}", source].edge_index = edges.flip(0)
            np.add.at(degree[source], source_pos, 1)
            np.add.at(degree[target], target_pos, 1)
            rows.append({"cutoff_utc": str(cutoff), "source": source,
                         "relation": field, "target": target, "edges": len(source_pos),
                         "null_or_invisible_fk": len(source_table) - len(source_pos)})
    stats = []
    for name in TYPES:
        n = graph[name].num_nodes
        actual = len(visible[name])
        stats.append({"cutoff_utc": str(cutoff), "node_type": name, "nodes": n,
                      "observed_nodes": actual, "isolated_nodes": int((degree[name] == 0).sum()),
                      "isolated_fraction": float((degree[name] == 0).mean()) if n else 0})
    obs_users = len(np.intersect1d(actual_users, cohort))
    observed_test = (len(np.intersect1d(actual_users, test_ids))
                     if test_ids is not None else None)
    summary = {"cutoff_utc": str(cutoff), "nodes": sum(s["nodes"] for s in stats),
               "observed_nodes": sum(s["observed_nodes"] for s in stats),
               "directed_fk_edges": sum(row["edges"] for row in rows),
               "message_edges_with_reverse": 2 * sum(row["edges"] for row in rows),
               "isolated_fraction": sum(s["isolated_nodes"] for s in stats) /
               sum(s["nodes"] for s in stats),
               "observed_cohort_users": obs_users,
               "observed_cohort_fraction": obs_users / len(cohort),
               "observed_test_users": observed_test,
               "observed_test_fraction": (observed_test / len(test_ids)
                                          if test_ids is not None else None)}
    graph.query_index = torch.from_numpy(query.astype(np.int64))
    graph.cohort_ids = torch.tensor(cohort, dtype=torch.long)
    graph.cutoff_utc = str(cutoff)
    return graph, rows, stats, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/extended/db")
    parser.add_argument("--out", type=Path, default=ROOT / "gnn_study")
    args = parser.parse_args()
    (args.out / "snapshots").mkdir(parents=True, exist_ok=True)
    data = ActivityData(args.db)
    cohort = [int(v) for v in (ROOT / "scripts/cohort.txt").read_text().split()]
    test_ids = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
    cutoffs = sorted({time for fold in FOLDS for time in
                      fold["train"] + [fold["tune"], fold["eval"]]})
    relations, nodes, summaries = [], [], []
    for cutoff in cutoffs:
        graph, rel, node, summary = build_snapshot(data, cohort, cutoff, True, test_ids)
        stamp = pd.Timestamp(cutoff).strftime("%Y%m%d%H")
        torch.save(graph, args.out / "snapshots" / f"{stamp}.pt")
        relations.extend(rel)
        nodes.extend(node)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.out / "metrics").mkdir(exist_ok=True)
    pd.DataFrame(relations).to_csv(args.out / "metrics/graph_relations.csv", index=False)
    pd.DataFrame(nodes).to_csv(args.out / "metrics/graph_nodes.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out / "metrics/graph_statistics.csv", index=False)


if __name__ == "__main__":
    main()
