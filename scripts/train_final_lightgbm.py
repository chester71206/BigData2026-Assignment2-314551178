"""Chronological backtests, last-week tuning, and quality-aware LightGBM refit."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from activity_features import ACTIVITIES, ActivityData
from evaluate import scores
from run_lightgbm import (OFFICIAL_CUTOFF, archive_hours, assemble,
                          check_coverage, choose_threshold, fit_one,
                          make_output, predict_model)

WEEKS = [str(t) for t in pd.date_range("2026-09-05 16:00:00",
                                      "2026-11-07 16:00:00", freq="7D")]
BACKTESTS = [
    {"name": "sep26", "train": WEEKS[:2], "tune": WEEKS[2], "eval": WEEKS[3]},
    {"name": "oct10", "train": WEEKS[:4], "tune": WEEKS[4], "eval": WEEKS[5]},
    {"name": "oct17", "train": WEEKS[:5], "tune": WEEKS[5], "eval": WEEKS[6]},
    {"name": "oct31", "train": WEEKS[:7], "tune": WEEKS[7], "eval": WEEKS[8]},
]
GROUPS = {
    "B_user": ("user",),
    "C_user_temporal": ("user", "temporal"),
    "D_original_relational": ("user", "temporal", "relational"),
    "R_repo_volume": ("user", "temporal", "repo_only"),
    "I_interactions": ("user", "temporal", "interaction"),
    "F_repo_interactions": ("user", "temporal", "repo_only", "interaction"),
}


def setup_logger(out):
    logger = logging.getLogger("final_lightgbm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(out / "logs/train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def teacher_evaluate(out, name, predicted, truth, scored):
    pred_path = out / "predictions" / f"{name}.csv"
    truth_path = out / "predictions" / f"{name}_truth.csv"
    predicted.to_csv(pred_path, index=False)
    truth.to_csv(truth_path, index=False)
    skipped = [a for a in ACTIVITIES if a not in scored]
    command = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
               "--pred", str(pred_path), "--truth", str(truth_path)]
    if skipped:
        command += ["--skip", *skipped]
    result = subprocess.run(command, text=True, capture_output=True)
    (out / "logs" / f"{name}_evaluate.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return scores(predicted, truth, scored)


def fit_activity(features, labels, group_names, train_weeks, tune_week,
                 activity, threads, logger, tag):
    x_train = pd.concat([assemble(features[t], group_names) for t in train_weeks],
                        ignore_index=True)
    y_train = np.concatenate([labels[t][activity].to_numpy() for t in train_weeks])
    x_tune = assemble(features[tune_week], group_names)
    y_tune = labels[tune_week][activity].to_numpy()
    model, constant, params, rounds, loss, tune_prob = fit_one(
        x_train, y_train, x_tune, y_tune, threads, logger, tag)
    threshold, tune_f1 = choose_threshold(tune_prob, y_tune)
    return model, constant, params, rounds, loss, threshold, tune_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("raw_data"))
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("submission/test_input.csv"))
    parser.add_argument("--out", type=Path, default=Path("final_refresh"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prediction-time", type=pd.Timestamp)
    args = parser.parse_args()
    if (args.out / "metrics/final_config.json").exists():
        parser.error("Completed run exists: choose a new --out directory")
    for folder in ("metrics", "models", "logs", "predictions", "figures"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out)
    audit = archive_hours(args.raw)
    check_coverage(audit, WEEKS)
    latest_cutoff = pd.Timestamp(audit["last_hour_utc"]) + pd.Timedelta(hours=1)
    prediction_time = args.prediction_time or latest_cutoff
    if prediction_time > latest_cutoff or prediction_time > OFFICIAL_CUTOFF:
        raise ValueError("Prediction time exceeds available or official pre-test data")
    quality = pd.read_csv(args.quality).set_index("week_start_utc")
    if not set(WEEKS).issubset(quality.index):
        raise ValueError("Quality audit lacks one or more complete label weeks")
    five = ACTIVITIES[:-1]
    if int(quality.loc[WEEKS, [f"{a}_gap_days" for a in five]].to_numpy().sum()):
        raise ValueError("A five-activity label week has a full-day zero gap")
    push_safe = [t for t in WEEKS if int(quality.loc[t, "push_gap_days"]) == 0]
    if push_safe != WEEKS[:4]:
        raise ValueError(f"Unexpected push-safe weeks: {push_safe}")
    data = ActivityData(args.db)
    cohort = np.array([int(v) for v in Path("scripts/cohort.txt").read_text().split()],
                      dtype=np.int64)
    test = pd.read_csv(args.test)["UserID"].to_numpy(dtype=np.int64)
    if not set(test).issubset(set(cohort)):
        raise ValueError("Test users not all present in cohort")
    features = {t: data.features(cohort, t) for t in WEEKS}
    labels = {t: data.labels(cohort, t) for t in WEEKS}
    logger.info("Loaded %d label weeks, cohort=%d, test=%d, latest=%s",
                len(WEEKS), len(cohort), len(test), latest_cutoff)
    rows, events, tuning = [], [], []
    for fold in BACKTESTS:
        name, tune, ev = fold["name"], fold["tune"], fold["eval"]
        truth = make_output(test, labels[ev].loc[test, ACTIVITIES])
        baseline = {a: features[ev]["temporal"].loc[test, f"tmp_{a}_7d"].gt(0)
                    .to_numpy(dtype=np.int8) for a in ACTIVITIES}
        baseline["push"][:] = 0
        arms = {"A_copy": make_output(test, baseline)}
        for arm, group_names in GROUPS.items():
            x_eval = assemble(features[ev], group_names).loc[test]
            predicted = {"push": np.zeros(len(test), dtype=np.int8)}
            for activity in five:
                model, constant, params, rounds, loss, threshold, tune_f1 = fit_activity(
                    features, labels, group_names, fold["train"], tune,
                    activity, args.threads, logger, f"{name}/{arm}/{activity}")
                probability = predict_model(model, constant, x_eval, args.threads)
                predicted[activity] = (probability >= threshold).astype(np.int8)
                tuning.append({"fold": name, "arm": arm, "activity": activity,
                               "threshold": threshold, "tune_f1": tune_f1,
                               "tune_logloss": loss, "best_rounds": rounds})
            arms[arm] = make_output(test, predicted)
        for arm, pred in arms.items():
            result = teacher_evaluate(args.out, f"{name}_{arm}", pred, truth, five)
            rows.append({"fold": name, "arm": arm, "n_scored": 5,
                         "macro_f1": result["macro_f1"],
                         "micro_f1": result["micro_f1"], "score": result["score"]})
            events.extend({"fold": name, "arm": arm, "activity": a,
                           "f1": result["per_event"][a],
                           "truth_positives": int(truth[a].sum()),
                           "predicted_positives": int(pred[a].sum())} for a in five)
            logger.info("%s/%s five-class Score=%.4f", name, arm, result["score"])
        pd.DataFrame(rows).to_csv(args.out / "metrics/backtest_scores.csv", index=False)
    pd.DataFrame(events).to_csv(args.out / "metrics/backtest_per_activity.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.out / "metrics/backtest_tuning.csv", index=False)
    summary = (pd.DataFrame(rows).query("arm != 'A_copy'").groupby("arm")["score"]
               .agg(["mean", "std"]).sort_values("mean", ascending=False))
    summary.to_csv(args.out / "metrics/model_selection.csv")
    winner = summary.index[0]
    logger.info("Five-class method selected from four independent holdouts: %s", winner)

    # Push has only four label weeks with no complete UTC zero-push day.
    push_folds = [
        {"name": "push_sep19", "train": WEEKS[:1], "tune": WEEKS[1], "eval": WEEKS[2]},
        {"name": "push_sep26", "train": WEEKS[:2], "tune": WEEKS[2], "eval": WEEKS[3]},
    ]
    push_rows = []
    for fold in push_folds:
        truth = make_output(test, labels[fold["eval"]].loc[test, ACTIVITIES])
        for arm in ("B_user", "C_user_temporal", "F_repo_interactions"):
            group_names = GROUPS[arm]
            model, constant, _, _, _, threshold, _ = fit_activity(
                features, labels, group_names, fold["train"], fold["tune"],
                "push", args.threads, logger, f"{fold['name']}/{arm}/push")
            x_eval = assemble(features[fold["eval"]], group_names).loc[test]
            probability = predict_model(model, constant, x_eval, args.threads)
            pred = {a: np.zeros(len(test), dtype=np.int8) for a in ACTIVITIES}
            pred["push"] = (probability >= threshold).astype(np.int8)
            output = make_output(test, pred)
            result = teacher_evaluate(args.out, f"{fold['name']}_{arm}",
                                      output, truth, ["push"])
            push_rows.append({"fold": fold["name"], "arm": arm,
                              "push_f1": result["per_event"]["push"],
                              "truth_positives": int(truth["push"].sum()),
                              "predicted_positives": int(output["push"].sum())})
    pd.DataFrame(push_rows).to_csv(args.out / "metrics/push_backtests.csv", index=False)
    push_winner = (pd.DataFrame(push_rows).groupby("arm")["push_f1"].mean()
                   .sort_values(ascending=False).index[0])
    logger.info("Push method selected from two safe holdouts: %s", push_winner)

    final_tune = WEEKS[-1]
    final_train = WEEKS[:-1]
    prediction_groups = data.features(cohort, prediction_time)
    final = {}
    final_rows = []
    for activity in ACTIVITIES:
        arm = push_winner if activity == "push" else winner
        group_names = GROUPS[arm]
        train_weeks = push_safe[:-1] if activity == "push" else final_train
        tune_week = push_safe[-1] if activity == "push" else final_tune
        model, constant, params, rounds, loss, threshold, tune_f1 = fit_activity(
            features, labels, group_names, train_weeks, tune_week,
            activity, args.threads, logger, f"final_tune/{arm}/{activity}")
        all_weeks = train_weeks + [tune_week]
        x_all = pd.concat([assemble(features[t], group_names) for t in all_weeks],
                          ignore_index=True)
        y_all = np.concatenate([labels[t][activity].to_numpy() for t in all_weeks])
        x_prediction = assemble(prediction_groups, group_names).loc[test]
        if len(np.unique(y_all)) < 2:
            constant = float(np.mean(y_all))
            probability = np.full(len(test), constant)
            (args.out / "models" / f"best_{activity}.json").write_text(
                json.dumps({"constant_probability": constant}) + "\n")
        else:
            refit = lgb.train(params, lgb.Dataset(x_all, label=y_all),
                              num_boost_round=max(1, rounds))
            refit.save_model(str(args.out / "models" / f"best_{activity}.txt"))
            probability = refit.predict(x_prediction, num_threads=args.threads)
        final[activity] = (probability >= threshold).astype(np.int8)
        final_rows.append({"activity": activity, "arm": arm,
                           "train_weeks": len(all_weeks),
                           "last_label_week_start_utc": all_weeks[-1],
                           "tune_week_start_utc": tune_week,
                           "threshold": threshold, "tune_f1": tune_f1,
                           "tune_logloss": loss, "best_rounds": rounds,
                           "training_positives": int(y_all.sum())})
    output = make_output(test, final)
    observed = set(data.users.loc[data.users["first_seen_at"].lt(prediction_time),
                                  "github_user_id"])
    cold = ~output["UserID"].isin(observed)
    output.loc[cold, ACTIVITIES] = 0
    pred_path = args.out / "predictions/provisional_prediction.csv"
    output.to_csv(pred_path, index=False)
    format_check = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("evaluate.py")),
         "--pred", str(pred_path), "--input", str(args.test)],
        text=True, capture_output=True)
    (args.out / "logs/format_check.log").write_text(
        format_check.stdout + format_check.stderr)
    if format_check.returncode:
        raise RuntimeError(format_check.stderr)
    pd.DataFrame(final_rows).to_csv(args.out / "metrics/final_model_details.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    score_frame = pd.DataFrame(rows)
    for arm in ["A_copy"] + list(GROUPS):
        part = score_frame.loc[score_frame["arm"].eq(arm)]
        ax.plot(part["fold"], part["score"], marker="o", label=arm)
    ax.set_ylabel("Five-activity Score (push excluded)")
    ax.set_title("Chronological method selection before final-week tuning")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "figures/backtests.png", dpi=160)
    plt.close(fig)
    (args.out / "metrics/final_config.json").write_text(json.dumps({
        "raw_first_hour_utc": audit["first_hour_utc"],
        "raw_last_hour_utc": audit["last_hour_utc"],
        "prediction_time_utc": str(prediction_time),
        "official_cutoff_utc": str(OFFICIAL_CUTOFF),
        "backtests": BACKTESTS,
        "five_class_method": winner,
        "five_class_final_tune_week_utc": final_tune,
        "five_class_training_weeks_utc": WEEKS,
        "push_method": push_winner,
        "push_safe_training_weeks_utc": push_safe,
        "cold_start_users": int(cold.sum()),
        "cold_start_strategy": "all_zero_if_no_visible_Users_row",
        "cohort_size": len(cohort), "test_size": len(test),
        "lightgbm_version": lgb.__version__, "threads": args.threads,
    }, indent=2) + "\n")
    logger.info("Saved %s, cold=%d, format=%s", pred_path, int(cold.sum()),
                format_check.stdout.strip())


if __name__ == "__main__":
    main()
