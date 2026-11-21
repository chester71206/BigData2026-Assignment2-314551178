# GitHub activity prediction — NYCU Big Data Assignment 2

This repository predicts whether each of 4,465 GitHub users will perform six activities during the seven days beginning **2026-11-21 16:00 UTC**. The six binary output columns are `open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, and `push`. Student ID: **314551178**.

The submission model uses six independent LightGBM classifiers. It is trained on chronological weekly snapshots built from the eight Parquet tables produced by the course-provided `scripts/preprocess.py`. The repository also contains temporal and relational feature ablations and a separate GNN study. The course-provided preprocessing and evaluation scripts are unchanged.

**Data source:** [GH Archive](https://www.gharchive.org/) hourly event files. No GitHub API or external user data is used for model features. Raw archives and generated Parquet tables are excluded from Git because the raw files occupy about **29 GiB**; the commands below reconstruct them.

## Results at a glance

The committed run uses 1,947 consecutive UTC hourly files from **2026-09-01 00:00 through 2026-11-21 02:00**. Its feature cutoff is **2026-11-21 03:00 UTC**, before the official test week. The CSV is therefore a *provisional* prediction; official test labels and a test score are unavailable.

Historical model selection used four chronological holdouts. The table reports the course evaluator's `Score = (Macro-F1 + Micro-F1) / 2` over the **same five activities, excluding Push** because Push has systematic archive gaps. Values are percentages; each holdout uses an earlier week for tuning and a later week for evaluation.

| Feature set | Sep 26 | Oct 10 | Oct 17 | Oct 31 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy last week | 37.82 | 35.32 | 34.60 | 34.48 | 35.56 |
| User history | **45.24** | 39.41 | 38.59 | 40.57 | **40.95** |
| User + temporal | 37.57 | 39.98 | 40.94 | 41.27 | 39.94 |
| User + temporal + original relations | 37.75 | 40.50 | **41.80** | 40.48 | 40.13 |
| User + temporal + repository volume | 38.00 | **40.96** | 41.13 | 40.76 | 40.21 |
| User + temporal + interactions | 38.00 | 39.47 | 39.94 | 40.21 | 39.41 |
| User + temporal + repository volume + interactions | 38.09 | 40.74 | 40.09 | **41.98** | 40.23 |

The four-fold mean selects **User history (18 features)**. Temporal and relational features improve some individual weeks but do not improve this mean. Push is selected separately from two earlier holdouts whose label weeks have no full-day zero-Push gaps. See [the run summary](final_submission_summary.md), [per-activity metrics](final_submission/metrics/backtest_per_activity.csv), [model selection data](final_submission/metrics/model_selection.csv), and [training log](final_submission/logs/train.log).

The committed [prediction CSV](submission/output_314551178.csv) has all 4,465 test users and passed `scripts/evaluate.py`'s format check. For 146 users with no visible `Users` row at the feature cutoff, the model applies the documented all-zero default. These users are present in the CSV.

## Requirements

- Python **3.13**, `curl`, and enough free disk space for approximately 29 GiB of compressed GH Archive files plus working space. Linux commands are shown below.
- Run commands from the repository root. `requirements.txt` pins the packages used by the LightGBM pipeline.
- Downloading the full archive is the expensive step. The downloader skips existing nonempty files, so reuse a populated `raw_data/` directory if available.

## Reproduce the submitted LightGBM run

### 1. Install dependencies and download the exact UTC range

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python scripts/download_gharchive.py --start 2026-09-01 --end 2026-11-21 --end-hour 2 --out raw_data
```

The downloader should have 1,947 hourly `.json.gz` files. A missing hour causes training coverage checks to fail. All dates and times in this README are UTC unless marked otherwise.

### 2. Build the eight course tables and audit coverage

```bash
.venv/bin/python scripts/preprocess.py --raw raw_data --out data/reproduction
.venv/bin/python scripts/audit_event_quality.py --db data/reproduction/db --raw raw_data --out runs/reproduction/quality
```

The course script writes `Users`, `Repositories`, `PullRequests`, `Issues`, `IssueComments`, `PRReviews`, `PRReviewComments`, and `Pushes` as `data/reproduction/db/*.parquet`. See [SCHEMA.md](SCHEMA.md) for primary keys, foreign keys, and timestamp meanings. The quality audit writes daily counts and `weekly_gap_flags.csv`; training refuses label weeks with full-day gaps in the five main activities and uses only safe weeks for Push.

### 3. Run chronological backtests, select models, and refit

```bash
.venv/bin/python scripts/train_final_lightgbm.py \
  --db data/reproduction/db \
  --raw raw_data \
  --quality runs/reproduction/quality/weekly_gap_flags.csv \
  --out runs/reproduction \
  --threads 4 \
  --prediction-time "2026-11-21 03:00:00"
```

Use a new `--out` directory for each run; the trainer will not overwrite a completed run. The explicit prediction time reproduces this submission even if `raw_data/` later contains newer hours. The script trains on the 9,000-person course cohort, selects feature groups on four historical holdouts, uses the last complete labeled week to choose LightGBM rounds and one threshold per activity, then refits on all eligible labeled weeks. Five activities use ten weeks beginning Sep 5 through Nov 7; Push uses four earlier safe weeks. Random train/test splitting is not used.

Training writes `runs/reproduction/metrics/`, `models/`, `logs/`, `figures/`, and `predictions/provisional_prediction.csv`. Historical prediction and truth files in `predictions/` are evaluated by the course's `scripts/evaluate.py`; individual evaluator outputs are in `logs/`.

### 4. Reproduce inference and check the output

```bash
.venv/bin/python scripts/predict_final_lightgbm.py \
  --db data/reproduction/db \
  --artifacts runs/reproduction \
  --output runs/reproduction/predictions/reproduced_prediction.csv
.venv/bin/python scripts/evaluate.py \
  --pred runs/reproduction/predictions/reproduced_prediction.csv \
  --input submission/test_input.csv
cmp runs/reproduction/predictions/reproduced_prediction.csv submission/output_314551178.csv
```

The expected format check is `Format check: no errors (4465 users)`. The `cmp` command checks byte-for-byte equality with the committed CSV. `--input` checks format and user coverage; it cannot compute an official test score without test labels.

The output header is exactly:

```text
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
```

`submission/test_input.csv`'s `UserID` corresponds to `Users.github_user_id`, **not** the internal `Users.user_id`.

## Modeling and data-quality notes

- Each sample is `(UserID, prediction_time)`. Features use only events and rows visible **before** that time; labels use the following seven days. For example, a PR with an earlier `created_at` is still excluded from a snapshot if its `first_seen_at` is later. Later state fields such as `merged_at` and `closed_at` are excluded from features.
- The feature code is [`scripts/activity_features.py`](scripts/activity_features.py). It constructs user activity history, 7/14/30-day counts and recency, repository activity, and PR/issue/review interactions. The [relational study](relational_study_summary.md) tests whether richer joins provide observable signal; the [temporal ablation](report/temporal_ablation_summary.md) separates short windows, longer windows, recency, and trend.
- All raw hourly files may exist while an activity type is still missing. The [quality audit](final_submission/quality/daily_activity_quality.csv) found **15 full UTC days with zero Push rows**. Those missing observations are not treated as reliable negative Push labels. Cross-week model comparison uses the same five non-Push classes.
- The [GNN study](gnn_study/README.md) compares two-layer GCN, GCN with temporal features, and a heterogeneous SAGE model on three historical weeks. Its graph snapshots can be regenerated from the eight tables; the large snapshot files are not stored in Git. The tested GNNs did not exceed the LightGBM scores on those weeks.

## Repository outputs

| Path | Contents |
| --- | --- |
| [`submission/output_314551178.csv`](submission/output_314551178.csv) | Submission-format prediction for 4,465 users |
| [`final_submission/`](final_submission/) | Selected models, backtest metrics, evaluator and training logs, plots, saved predictions, quality audit |
| [`final_submission_summary.md`](final_submission_summary.md) | Exact cutoff, selected settings, results, limitations |
| [`report/Asg2_314551178.pdf`](report/Asg2_314551178.pdf) | 16-slide English report |
| [`report/Assignment2_report.pptx`](report/Assignment2_report.pptx) | Editable slides; [report build instructions](report/README.md) |
| [`gnn_study/`](gnn_study/) | Graph construction, GNN experiments, checkpoints, metrics and analysis |
| [`artifacts/`](artifacts/) and [`final_refresh/`](final_refresh/) | Earlier experiments retained for comparison; the submitted run is `final_submission/` |

The report and CSV are prepared here. The course also requires a student-recorded video and separate E3 uploads; an MP4 is not included in this repository.
