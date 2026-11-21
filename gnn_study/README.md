# GNN Study

This study uses the existing instructor-generated `data/extended/db/*.parquet` files. It never downloads GH Archive again and does not modify `scripts/preprocess.py` or `scripts/evaluate.py`.

## Reproduce in a new output directory

From the repository root:

```bash
.venv/bin/python -m pip install -r gnn_study/requirements.txt
.venv/bin/python gnn_study/graph_dataset.py --db data/extended/db --out gnn_study/rebuild
.venv/bin/python gnn_study/train_gnn.py --db data/extended/db --out gnn_study/rebuild --arms A_gcn B_gcn_temporal --threads 4
.venv/bin/python gnn_study/train_gnn.py --db data/extended/db --out gnn_study/rebuild --arms C_hetero --threads 4
.venv/bin/python gnn_study/predict_gnn.py --study gnn_study/rebuild --fold oct17_five --arm C_hetero
.venv/bin/python gnn_study/check_graph_invariance.py --db data/extended/db
```

`train_gnn.py` reuses the exact historical folds from `scripts/run_relational_study.py`. It saves a checkpoint and threshold file per fold and model. The stored CSVs were produced with seed 42 and 24 maximum epochs; the best epoch is selected on the threshold week by BCE loss, with patience 5. Test weeks are read only for final evaluation. The five-activity comparison excludes push in all three weeks. The September 26 fold additionally reports six activities. Cold-start users without a visible `Users` row receive six zeros.

The main stored artifacts are under `metrics/`, `models/`, `logs/`, `predictions/`, `snapshots/`, and `figures/`. For additional seed runs or relation masking, use the existing snapshots:

```bash
.venv/bin/python gnn_study/relation_ablation.py --study gnn_study/rebuild
.venv/bin/python gnn_study/benchmark_lightgbm.py --out gnn_study/rebuild/metrics/lightgbm_benchmark.csv
mkdir -p gnn_study/rebuild/stability_seed7 gnn_study/rebuild/stability_seed123
ln -s ../snapshots gnn_study/rebuild/stability_seed7/snapshots
ln -s ../snapshots gnn_study/rebuild/stability_seed123/snapshots
.venv/bin/python gnn_study/train_gnn.py --out gnn_study/rebuild/stability_seed7 --arms C_hetero --seed 7 --threads 4
.venv/bin/python gnn_study/train_gnn.py --out gnn_study/rebuild/stability_seed123 --arms C_hetero --seed 123 --threads 4
.venv/bin/python gnn_study/summarize.py --study gnn_study/rebuild
```

The relation masking script needs the seed-42 C checkpoints in the given study directory. It removes one FK and its reverse edge during inference, without retraining; the result is a dependency probe, not a causal estimate.

See [gnn_study_summary.md](gnn_study_summary.md) for results and limitations.
