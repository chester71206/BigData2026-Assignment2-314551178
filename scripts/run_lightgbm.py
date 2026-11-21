"""Run temporal LightGBM ablations and make a provisional/final prediction."""

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from activity_features import ACTIVITIES, ActivityData
from evaluate import scores

TRAIN_WEEKS = [
    "2026-09-12 16:00:00", "2026-09-19 16:00:00",
    "2026-09-26 16:00:00",
]
TUNE_WEEK = "2026-10-03 16:00:00"
EVAL_WEEK = "2026-10-10 16:00:00"
OFFICIAL_CUTOFF = pd.Timestamp("2026-11-21 16:00:00")
GROUPS = {
    "A_copy_last_week": (),
    "B_user": ("user",),
    "C_user_temporal": ("user", "temporal"),
    "D_user_temporal_relational": ("user", "temporal", "relational"),
}
CONFIGS = [
    {"num_leaves": 15, "min_data_in_leaf": 60, "lambda_l2": 2.0, "feature_fraction": 0.9},
    {"num_leaves": 31, "min_data_in_leaf": 120, "lambda_l2": 5.0, "feature_fraction": 0.8},
]


def setup_log(out):
    log = logging.getLogger("experiments")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(out / "logs" / "training.log")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def archive_hours(raw):
    names = sorted(raw.glob("*.json.gz"))
    hours = {}
    for path in names:
        try:
            stem = path.name.removesuffix(".json.gz")
            day, hour = stem.rsplit("-", 1)
            timestamp = pd.Timestamp(f"{day} {int(hour):02d}:00:00")
        except (ValueError, TypeError):
            continue
        hours[timestamp] = path.stat().st_size
    if not hours:
        raise ValueError(f"No raw GH Archive files in {raw}")
    first, last = min(hours), max(hours)
    expected = pd.date_range(first, last, freq="h")
    missing = [str(hour) for hour in expected if hour not in hours]
    sizes = pd.Series(hours)
    low = sizes[sizes < 1_000_000].sort_index()
    return {
        "first_hour_utc": str(first),
        "last_hour_utc": str(last),
        "expected_hours": len(expected),
        "present_hours": len(hours),
        "missing_hours": missing,
        "hours_under_1mb": {str(k): int(v) for k, v in low.items()},
        "compressed_bytes": int(sizes.sum()),
    }


def check_coverage(audit, label_starts):
    if audit["missing_hours"]:
        raise ValueError(f"Missing GH Archive hours: {audit['missing_hours'][:20]}")
    if audit["hours_under_1mb"]:
        bad = list(audit["hours_under_1mb"])[:20]
        raise ValueError(f"Suspiciously small GH Archive hours (<1 MB): {bad}")
    end = max(pd.Timestamp(t) for t in label_starts) + pd.Timedelta(days=7)
    last = pd.Timestamp(audit["last_hour_utc"]) + pd.Timedelta(hours=1)
    if last < end:
        raise ValueError(f"Label week ends at {end}, but raw data only covers through {last}")


def assemble(groups, names):
    return pd.concat([groups[name] for name in names], axis=1).astype("float32")


def f1_for_threshold(prob, truth, threshold):
    predicted = prob >= threshold
    positives = truth == 1
    tp = np.count_nonzero(predicted & positives)
    fp = np.count_nonzero(predicted & ~positives)
    fn = np.count_nonzero(~predicted & positives)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def choose_threshold(prob, truth):
    if not np.any(truth):
        return 1.0, f1_for_threshold(prob, truth, 1.0)
    candidates = np.unique(np.r_[
        np.linspace(0.02, 0.98, 49), np.quantile(prob, np.linspace(0.01, 0.99, 99)),
    ])
    candidates = candidates[(candidates > 0) & (candidates <= 1)]
    ranked = sorted(
        ((f1_for_threshold(prob, truth, float(t)), float(t)) for t in candidates),
        key=lambda item: (-item[0], abs(item[1] - 0.5)),
    )
    return ranked[0][1], ranked[0][0]


def fit_one(x_train, y_train, x_tune, y_tune, threads, log, tag):
    if len(np.unique(y_train)) < 2:
        probability = float(np.mean(y_train))
        log.info("%s constant class=%s", tag, probability)
        return None, probability, {}, 0, probability, np.full(len(x_tune), probability)
    best = None
    for index, config in enumerate(CONFIGS):
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "verbosity": -1, "seed": 42,
            "num_threads": threads, "deterministic": True,
            "force_col_wise": True, **config,
        }
        train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
        tune_set = lgb.Dataset(x_tune, label=y_tune, reference=train_set, free_raw_data=False)
        model = lgb.train(
            params, train_set, num_boost_round=250,
            valid_sets=[tune_set],
            callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
        )
        prob = model.predict(x_tune, num_threads=threads)
        clipped = np.clip(prob, 1e-8, 1 - 1e-8)
        loss = float(-np.mean(y_tune * np.log(clipped) + (1 - y_tune) * np.log(1 - clipped)))
        log.info("%s config=%d tune_logloss=%.6f best_iteration=%d", tag, index, loss, model.best_iteration)
        if best is None or loss < best[0]:
            best = (loss, model, params, model.best_iteration, prob)
    loss, model, params, rounds, prob = best
    return model, None, params, rounds, loss, prob


def predict_model(model, constant, frame, threads):
    if model is None:
        return np.full(len(frame), constant)
    return model.predict(frame, num_threads=threads)


def make_output(users, values):
    result = pd.DataFrame(values, index=pd.Index(users, name="UserID"))
    result = result[ACTIVITIES].astype("int8").reset_index()
    return result


def teacher_evaluate(out, tag, predicted, truth, log):
    pred_file = out / "predictions" / f"{tag}_validation.csv"
    truth_file = out / "predictions" / "validation_truth.csv"
    predicted.to_csv(pred_file, index=False)
    truth.to_csv(truth_file, index=False)
    process = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("evaluate.py")),
         "--pred", str(pred_file), "--truth", str(truth_file)],
        capture_output=True, text=True,
    )
    (out / "logs" / f"evaluate_{tag}.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"Teacher evaluate.py failed for {tag}: {process.stderr}")
    log.info("%s teacher Score: %s", tag, process.stdout.strip().splitlines()[-1])
    return scores(predicted, truth)


def save_plots(out, metrics, importance):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25
    for shift, field, label in ((-1, "macro_f1", "Macro-F1"), (0, "micro_f1", "Micro-F1"), (1, "score", "Score")):
        ax.bar(x + shift * width, metrics[field], width, label=label)
    ax.set_xticks(x, ["A: Copy", "B: User", "C: +Time", "D: +Relations"])
    ax.set_ylim(0, max(0.1, metrics[["macro_f1", "micro_f1", "score"]].to_numpy().max() * 1.15))
    ax.set_ylabel("F1 / Score")
    ax.set_title("Temporal holdout ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "ablation_comparison.png", dpi=160)
    plt.close(fig)

    top = importance.groupby("feature")["gain"].sum().nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Summed LightGBM gain across six classifiers")
    ax.set_title("Feature importance: full relational model")
    fig.tight_layout()
    fig.savefig(out / "figures" / "feature_importance.png", dpi=160)
    plt.close(fig)


def save_best_importance(out, importance, winner):
    top = (importance.loc[importance["experiment"].eq(winner)]
           .groupby("feature")["gain"].sum().nlargest(20).sort_values())
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Summed LightGBM gain across six classifiers")
    ax.set_title(f"Feature importance: selected model {winner}")
    fig.tight_layout()
    fig.savefig(out / "figures" / "best_feature_importance.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/db"))
    parser.add_argument("--raw", type=Path, default=Path("raw_data"))
    parser.add_argument("--test", type=Path, default=Path("submission/test_input.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--prediction-time", type=pd.Timestamp)
    parser.add_argument("--train-weeks", nargs="+", default=TRAIN_WEEKS)
    parser.add_argument("--threshold-week", default=TUNE_WEEK)
    parser.add_argument("--evaluation-week", default=EVAL_WEEK)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    train_weeks = [str(pd.Timestamp(value)) for value in args.train_weeks]
    tune_week = str(pd.Timestamp(args.threshold_week))
    eval_week = str(pd.Timestamp(args.evaluation_week))
    all_starts = [pd.Timestamp(value) for value in train_weeks + [tune_week, eval_week]]
    if all_starts != sorted(all_starts) or len(set(all_starts)) != len(all_starts):
        parser.error("training, threshold and evaluation weeks must be unique and chronological")
    if (args.out / "metrics" / "run_config.json").exists():
        parser.error(f"{args.out} already contains a completed run; choose a new --out directory")
    for folder in ("logs", "models", "metrics", "figures", "predictions"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    log = setup_log(args.out)

    audit = archive_hours(args.raw)
    (args.out / "metrics" / "data_coverage.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    check_coverage(audit, train_weeks + [tune_week, eval_week])
    latest_cutoff = pd.Timestamp(audit["last_hour_utc"]) + pd.Timedelta(hours=1)
    if latest_cutoff > OFFICIAL_CUTOFF:
        raise ValueError("Raw files extend into the official test week; rebuild tables from pre-cutoff files only")
    prediction_time = args.prediction_time or latest_cutoff
    if prediction_time > latest_cutoff or prediction_time > OFFICIAL_CUTOFF:
        raise ValueError("Prediction time exceeds available raw data or official cutoff")
    log.info("Data range %s to %s UTC; prediction cutoff %s UTC",
             audit["first_hour_utc"], audit["last_hour_utc"], prediction_time)

    data = ActivityData(args.db)
    users = np.array([int(value) for value in Path("scripts/cohort.txt").read_text().split()], dtype="int64")
    test_users = pd.read_csv(args.test)["UserID"].to_numpy(dtype="int64")
    if not set(test_users).issubset(set(users)):
        raise ValueError("Test users are not all in cohort.txt")
    log.info("Cohort=%d, test users=%d, normalized events=%d", len(users), len(test_users), len(data.events))
    log.info(
        "Late-visible action rows excluded from features at eval cutoff: %d",
        int((data.events["event_at"].lt(pd.Timestamp(eval_week))
             & data.events["visible_at"].ge(pd.Timestamp(eval_week))).sum()),
    )
    times = train_weeks + [tune_week, eval_week]
    features, labels = {}, {}
    for time in times:
        groups = data.features(users, time)
        answer = data.labels(users, time)
        features[time], labels[time] = groups, answer
        log.info("Week %s: positives=%s", time, answer[ACTIVITIES].sum().to_dict())
    eval_truth = make_output(test_users, labels[eval_week].loc[test_users, ACTIVITIES])
    metric_rows, per_event_rows, tune_rows, importance_rows = [], [], [], []
    fitted = {}
    for tag, names in GROUPS.items():
        if not names:
            predictions = {
                activity: (features[eval_week]["temporal"].loc[test_users, f"tmp_{activity}_7d"] > 0).astype("int8").to_numpy()
                for activity in ACTIVITIES
            }
            thresholds = {activity: 1 for activity in ACTIVITIES}
        else:
            x_train = pd.concat([assemble(features[time], names) for time in train_weeks], ignore_index=True)
            x_tune = assemble(features[tune_week], names)
            x_eval = assemble(features[eval_week], names).loc[test_users]
            predictions, thresholds, fitted[tag] = {}, {}, {}
            for activity in ACTIVITIES:
                y_train = np.concatenate([labels[time][activity].to_numpy() for time in train_weeks])
                y_tune = labels[tune_week][activity].to_numpy()
                model, constant, params, rounds, loss, tune_prob = fit_one(
                    x_train, y_train, x_tune, y_tune, args.threads, log, f"{tag}/{activity}"
                )
                threshold, tune_f1 = choose_threshold(tune_prob, y_tune)
                eval_prob = predict_model(model, constant, x_eval, args.threads)
                predictions[activity] = (eval_prob >= threshold).astype("int8")
                thresholds[activity] = threshold
                fitted[tag][activity] = {
                    "model": model, "constant": constant, "params": params, "rounds": rounds,
                }
                if model is not None:
                    model.save_model(str(args.out / "models" / f"{tag}_{activity}_eval.txt"))
                    gains = model.feature_importance(importance_type="gain")
                    importance_rows.extend(
                        {"experiment": tag, "activity": activity, "feature": feature, "gain": float(gain)}
                        for feature, gain in zip(x_train.columns, gains)
                    )
                tune_rows.append({
                    "experiment": tag, "activity": activity, "threshold": threshold,
                    "tune_f1": tune_f1, "tune_logloss": loss, "best_iteration": rounds,
                    "train_positives": int(y_train.sum()), "tune_positives": int(y_tune.sum()),
                })
                log.info("%s/%s threshold=%.4f tune_F1=%.4f eval_predicted_positive=%d",
                         tag, activity, threshold, tune_f1, int(predictions[activity].sum()))
        eval_prediction = make_output(test_users, predictions)
        score = teacher_evaluate(args.out, tag, eval_prediction, eval_truth, log)
        metric_rows.append({
            "experiment": tag, "macro_f1": score["macro_f1"],
            "micro_f1": score["micro_f1"], "score": score["score"],
        })
        for activity in ACTIVITIES:
            per_event_rows.append({
                "experiment": tag, "activity": activity,
                "f1": score["per_event"][activity],
                "truth_positives": int(eval_truth[activity].sum()),
                "predicted_positives": int(eval_prediction[activity].sum()),
                "threshold": thresholds[activity],
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out / "metrics" / "ablation.csv", index=False)
    pd.DataFrame(per_event_rows).to_csv(args.out / "metrics" / "per_activity.csv", index=False)
    for tag in GROUPS:
        pd.DataFrame(row for row in per_event_rows if row["experiment"] == tag).to_csv(
            args.out / "metrics" / f"{tag}.csv", index=False
        )
    pd.DataFrame(tune_rows).to_csv(args.out / "metrics" / "tuning.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(args.out / "metrics" / "feature_importance.csv", index=False)
    save_plots(args.out, metrics, importance.loc[
        importance["experiment"].eq("D_user_temporal_relational")
    ])
    winner = metrics.sort_values("score", ascending=False).iloc[0]["experiment"]
    log.info("Best holdout experiment: %s", winner)
    save_best_importance(args.out, importance, winner)

    # Refit the selected feature group with all historical labels available by
    # the provisional cutoff, using fixed hyperparameters and tree counts.
    provisional_groups = data.features(users, prediction_time)
    if winner == "A_copy_last_week":
        (args.out / "models" / "best_baseline.json").write_text(
            json.dumps({"method": "copy_last_week", "window_days": 7}) + "\n"
        )
        final = {
            activity: (provisional_groups["temporal"].loc[test_users, f"tmp_{activity}_7d"] > 0).astype("int8").to_numpy()
            for activity in ACTIVITIES
        }
    else:
        names = GROUPS[winner]
        x_all = pd.concat([assemble(features[time], names) for time in times], ignore_index=True)
        x_pred = assemble(provisional_groups, names).loc[test_users]
        final = {}
        chosen = pd.DataFrame(tune_rows).set_index(["experiment", "activity"])
        for activity in ACTIVITIES:
            y_all = np.concatenate([labels[time][activity].to_numpy() for time in times])
            info = fitted[winner][activity]
            threshold = float(chosen.loc[(winner, activity), "threshold"])
            if len(np.unique(y_all)) < 2:
                probability = np.full(len(x_pred), float(np.mean(y_all)))
                (args.out / "models" / f"best_{activity}.json").write_text(
                    json.dumps({"constant_probability": float(np.mean(y_all))}) + "\n"
                )
            else:
                model = lgb.train(
                    info["params"], lgb.Dataset(x_all, label=y_all),
                    num_boost_round=max(1, info["rounds"]),
                )
                model.save_model(str(args.out / "models" / f"best_{activity}.txt"))
                probability = model.predict(x_pred, num_threads=args.threads)
            final[activity] = (probability >= threshold).astype("int8")
    output = make_output(test_users, final)
    # With no observed Users row there is no individual history to rank these
    # users. Use a deterministic all-zero default for the six activities.
    observed_users = set(data.users.loc[
        data.users["first_seen_at"].lt(prediction_time), "github_user_id"
    ])
    missing_history = ~output["UserID"].isin(observed_users)
    output.loc[missing_history, ACTIVITIES] = 0
    log.info("Cold-start default: all zero for %d users absent from Users at cutoff",
             int(missing_history.sum()))
    pred_file = args.out / "predictions" / "provisional_prediction.csv"
    output.to_csv(pred_file, index=False)
    format_check = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("evaluate.py")),
         "--pred", str(pred_file), "--input", str(args.test)],
        capture_output=True, text=True,
    )
    (args.out / "logs" / "provisional_format_check.log").write_text(
        format_check.stdout + format_check.stderr
    )
    if format_check.returncode:
        raise RuntimeError(format_check.stderr)
    log.info("%s", format_check.stdout.strip())
    (args.out / "metrics" / "run_config.json").write_text(json.dumps({
        "train_weeks_utc": train_weeks, "threshold_week_utc": tune_week,
        "evaluation_week_utc": eval_week, "prediction_time_utc": str(prediction_time),
        "official_cutoff_utc": str(OFFICIAL_CUTOFF),
        "best_experiment": winner, "cohort_size": len(users),
        "test_size": len(test_users), "cold_start_users": int(missing_history.sum()),
        "cold_start_strategy": "all_zero_when_no_Users_row_before_cutoff",
        "lightgbm_version": lgb.__version__,
        "threads": args.threads,
    }, indent=2) + "\n")
    log.info("Wrote %s", pred_file)


if __name__ == "__main__":
    main()
