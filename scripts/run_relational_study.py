"""Compare repository volume with explicit reviewer–repository–PR interactions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from activity_features import ACTIVITIES, ActivityData
from evaluate import scores
from run_lightgbm import (
    OFFICIAL_CUTOFF, archive_hours, assemble, check_coverage, choose_threshold,
    fit_one, make_output, predict_model, setup_log,
)

FOLDS = [
    {
        "name": "sep26_six",
        "train": ["2026-09-05 16:00:00", "2026-09-12 16:00:00"],
        "tune": "2026-09-19 16:00:00", "eval": "2026-09-26 16:00:00",
    },
    {
        "name": "oct10_five",
        "train": ["2026-09-12 16:00:00", "2026-09-19 16:00:00",
                  "2026-09-26 16:00:00"],
        "tune": "2026-10-03 16:00:00", "eval": "2026-10-10 16:00:00",
    },
    {
        "name": "oct17_five",
        "train": ["2026-09-19 16:00:00", "2026-09-26 16:00:00",
                  "2026-10-03 16:00:00"],
        "tune": "2026-10-10 16:00:00", "eval": "2026-10-17 16:00:00",
    },
]
ARMS = {
    "A_copy": (),
    "B_user": ("user",),
    "C_user_temporal": ("user", "temporal"),
    "R_repo_volume": ("user", "temporal", "repo_only"),
    "I_interactions": ("user", "temporal", "interaction"),
    "F_repo_interactions": ("user", "temporal", "repo_only", "interaction"),
}


def observed_relations(data, cutoff):
    t = pd.Timestamp(cutoff)
    prs = data.tables["PullRequests"].loc[
        data.tables["PullRequests"]["first_seen_at"].lt(t),
        ["pr_id", "author_id", "repo_id"],
    ]
    reviews = data.tables["PRReviews"].loc[
        data.tables["PRReviews"]["first_seen_at"].lt(t)
        & data.tables["PRReviews"]["submitted_at"].ge(t - pd.Timedelta(days=30))
        & data.tables["PRReviews"]["submitted_at"].lt(t),
        ["pr_id", "reviewer_id", "repo_id"],
    ].dropna(subset=["reviewer_id"])
    joined = reviews.merge(prs[["pr_id", "author_id"]], on="pr_id", how="inner")
    known = joined.dropna(subset=["author_id"])
    pairs = known.groupby(["reviewer_id", "author_id"]).size()
    reviewer_repo = reviews.groupby(["reviewer_id", "repo_id"])["pr_id"].nunique()
    by_pr = reviews.groupby("pr_id")["reviewer_id"].nunique()
    comments = data.tables["PRReviewComments"].loc[
        data.tables["PRReviewComments"]["created_at"].ge(t - pd.Timedelta(days=30))
        & data.tables["PRReviewComments"]["created_at"].lt(t),
        ["pr_id", "author_id"],
    ]
    same_pr = comments.merge(
        reviews[["pr_id", "reviewer_id"]].drop_duplicates(),
        left_on=["pr_id", "author_id"], right_on=["pr_id", "reviewer_id"], how="inner",
    )
    return {
        "cutoff_utc": str(t), "reviews_30d": len(reviews),
        "reviews_joined_to_visible_pr": len(joined),
        "reviews_with_known_pr_author": len(known),
        "known_author_fraction": len(known) / len(reviews) if len(reviews) else 0,
        "unique_reviewer_author_pairs": len(pairs),
        "repeated_reviewer_author_pairs": int((pairs >= 2).sum()),
        "reviewer_repo_pairs": len(reviewer_repo),
        "reviewer_repo_pairs_with_2plus_prs": int((reviewer_repo >= 2).sum()),
        "prs_with_2plus_reviewers": int((by_pr >= 2).sum()),
        "review_code_comments_same_pr": len(same_pr),
        "users_with_review_code_comments_same_pr": same_pr["author_id"].nunique(),
    }


def teacher_eval(out, fold, arm, prediction, truth, skipped):
    prefix = f"{fold}_{arm}"
    pred_path = out / "predictions" / f"{prefix}.csv"
    truth_path = out / "predictions" / f"{fold}_truth.csv"
    prediction.to_csv(pred_path, index=False)
    truth.to_csv(truth_path, index=False)
    cmd = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
           "--pred", str(pred_path), "--truth", str(truth_path)]
    if skipped:
        cmd += ["--skip", *skipped]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out / "logs" / f"{prefix}_evaluate.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{prefix}: {result.stderr}")
    return scores(prediction, truth, [a for a in ACTIVITIES if a not in skipped])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/extended/db"))
    parser.add_argument("--raw", type=Path, default=Path("raw_data"))
    parser.add_argument("--test", type=Path, default=Path("submission/test_input.csv"))
    parser.add_argument("--quality", type=Path, default=Path("quality_study/weekly_gap_flags.csv"))
    parser.add_argument("--out", type=Path, default=Path("relational_study"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (args.out / "metrics" / "fold_metrics.csv").exists():
        parser.error("Completed output exists; choose a new --out directory")
    for folder in ("metrics", "logs", "models", "predictions", "figures"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    log = setup_log(args.out)
    audit = archive_hours(args.raw)
    if pd.Timestamp(audit["last_hour_utc"]) + pd.Timedelta(hours=1) > OFFICIAL_CUTOFF:
        raise ValueError("Raw data extends into the official test week")
    starts = [t for fold in FOLDS for t in fold["train"] + [fold["tune"], fold["eval"]]]
    check_coverage(audit, starts)
    quality = pd.read_csv(args.quality).set_index("week_start_utc")
    data = ActivityData(args.db)
    cohort = [int(value) for value in Path("scripts/cohort.txt").read_text().split()]
    test = pd.read_csv(args.test)["UserID"].to_numpy(dtype="int64")
    times = sorted(set(starts))
    features, labels = {}, {}
    for t in times:
        features[t] = data.features(cohort, t)
        labels[t] = data.labels(cohort, t)
        log.info("Loaded %s with positives %s", t, labels[t].sum().to_dict())
    pd.DataFrame(observed_relations(data, t) for t in
                 ["2026-09-26 16:00:00", "2026-10-10 16:00:00",
                  "2026-10-17 16:00:00", "2026-11-16 16:00:00"]).to_csv(
                      args.out / "metrics" / "relation_observability.csv", index=False)
    coverage_rows = []
    for cutoff in ("2026-09-26 16:00:00", "2026-10-10 16:00:00",
                   "2026-10-17 16:00:00", "2026-11-16 16:00:00"):
        groups = features.get(cutoff) or data.features(cohort, cutoff)
        for group_name in ("repo_only", "interaction"):
            frame = groups[group_name].loc[test]
            coverage_rows.append({
                "cutoff_utc": cutoff, "group": group_name,
                "users_with_any_nonzero": int(frame.ne(0).any(axis=1).sum()),
                "total_test_users": len(test),
            })
            coverage_rows.extend({
                "cutoff_utc": cutoff, "group": column,
                "users_with_any_nonzero": int(frame[column].ne(0).sum()),
                "total_test_users": len(test),
            } for column in frame.columns)
    pd.DataFrame(coverage_rows).to_csv(
        args.out / "metrics" / "feature_observability.csv", index=False
    )

    rows, event_rows, tune_rows, importance_rows, cold_rows = [], [], [], [], []
    seen = data.users.set_index("github_user_id")["first_seen_at"]
    for fold in FOLDS:
        name, tune, ev = fold["name"], fold["tune"], fold["eval"]
        week_starts = fold["train"] + [tune, ev]
        push_gaps = sum(int(quality.loc[week, "push_gap_days"]) for week in week_starts)
        scored = ACTIVITIES if push_gaps == 0 else ACTIVITIES[:-1]
        skipped = [a for a in ACTIVITIES if a not in scored]
        log.info("Fold %s: scored=%s; push zero-gap days across train/tune/eval=%d",
                 name, scored, push_gaps)
        truth = make_output(test, labels[ev].loc[test, ACTIVITIES])
        unseen = ~seen.reindex(test).lt(pd.Timestamp(ev)).fillna(False).to_numpy()
        cold_truth = truth.loc[unseen, ACTIVITIES]
        cold_rows.extend({
            "fold": name, "activity": activity, "unseen_users": int(unseen.sum()),
            "observed_positives": int(cold_truth[activity].sum()),
            "observed_rate": float(cold_truth[activity].mean()),
        } for activity in ACTIVITIES)
        for arm, groups in ARMS.items():
            predicted, thresholds = {}, {}
            if not groups:
                for activity in ACTIVITIES:
                    predicted[activity] = (
                        features[ev]["temporal"].loc[test, f"tmp_{activity}_7d"].gt(0)
                        .to_numpy(dtype="int8")
                    )
                    thresholds[activity] = 1
            else:
                train_x = pd.concat([assemble(features[t], groups) for t in fold["train"]],
                                    ignore_index=True)
                tune_x = assemble(features[tune], groups)
                eval_x = assemble(features[ev], groups).loc[test]
                for activity in scored:
                    train_y = np.concatenate([labels[t][activity].to_numpy() for t in fold["train"]])
                    tune_y = labels[tune][activity].to_numpy()
                    model, constant, params, rounds, loss, tune_prob = fit_one(
                        train_x, train_y, tune_x, tune_y, args.threads, log,
                        f"{name}/{arm}/{activity}",
                    )
                    threshold, tune_f1 = choose_threshold(tune_prob, tune_y)
                    eval_prob = predict_model(model, constant, eval_x, args.threads)
                    predicted[activity] = (eval_prob >= threshold).astype("int8")
                    thresholds[activity] = threshold
                    if model is not None:
                        model.save_model(str(args.out / "models" / f"{name}_{arm}_{activity}.txt"))
                        importance_rows.extend(
                            {"fold": name, "arm": arm, "activity": activity,
                             "feature": feature, "gain": float(gain)}
                            for feature, gain in zip(
                                train_x.columns, model.feature_importance(importance_type="gain")
                            )
                        )
                    tune_rows.append({
                        "fold": name, "arm": arm, "activity": activity,
                        "threshold": threshold, "tune_f1": tune_f1,
                        "logloss": loss, "best_iteration": rounds,
                    })
                for activity in skipped:
                    predicted[activity] = np.zeros(len(test), dtype="int8")
                    thresholds[activity] = np.nan
            prediction = make_output(test, predicted)
            result = teacher_eval(args.out, name, arm, prediction, truth, skipped)
            five = scores(prediction, truth, ACTIVITIES[:-1])
            rows.append({
                "fold": name, "arm": arm, "n_scored": len(scored),
                "skipped": ",".join(skipped), "macro_f1": result["macro_f1"],
                "micro_f1": result["micro_f1"], "score": result["score"],
                "score_5_without_push": five["score"],
            })
            for activity in ACTIVITIES:
                event_rows.append({
                    "fold": name, "arm": arm, "activity": activity,
                    "f1": result["per_event"].get(activity, np.nan),
                    "truth_positives": int(truth[activity].sum()),
                    "predicted_positives": int(prediction[activity].sum()),
                    "threshold": thresholds[activity],
                })
            log.info("%s/%s Score=%.4f (scored=%d), five-class=%.4f",
                     name, arm, result["score"], len(scored), five["score"])

    result = pd.DataFrame(rows)
    result.to_csv(args.out / "metrics" / "fold_metrics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.out / "metrics" / "per_activity.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(args.out / "metrics" / "tuning.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(args.out / "metrics" / "feature_importance.csv", index=False)
    pd.DataFrame(cold_rows).to_csv(args.out / "metrics" / "cold_start_observed.csv", index=False)
    (result.groupby("arm")["score_5_without_push"].agg(mean="mean", std="std")
     .reset_index().to_csv(args.out / "metrics" / "five_class_summary.csv", index=False))
    fig, ax = plt.subplots(figsize=(11, 5))
    for arm in ARMS:
        subset = result.loc[result["arm"].eq(arm)]
        ax.plot(subset["fold"], subset["score_5_without_push"], marker="o", label=arm)
    ax.set_ylabel("Five-activity Score (push excluded)")
    ax.set_title("Relation feature comparison across chronological holdouts")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "fold_comparison.png", dpi=160)
    plt.close(fig)
    focused = (importance.loc[importance["feature"].str.startswith(("int_", "rel_repo_"))]
               .groupby("feature")["gain"].sum().nlargest(18).sort_values())
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(focused.index, focused.values)
    ax.set_xlabel("Summed LightGBM gain")
    ax.set_title("Repository volume vs explicit interaction features")
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "relation_feature_importance.png", dpi=160)
    plt.close(fig)
    (args.out / "metrics" / "study_config.json").write_text(json.dumps({
        "folds": FOLDS, "arms": ARMS, "raw_last_hour_utc": audit["last_hour_utc"],
        "scoring_policy": "push omitted in a fold if any train/tune/eval week includes a full UTC zero-push day",
        "threads": args.threads,
    }, indent=2) + "\n")
    print(result.pivot(index="arm", columns="fold", values="score_5_without_push").to_string())


if __name__ == "__main__":
    main()
