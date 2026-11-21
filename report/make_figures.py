"""Produce monochrome figures for the course presentation."""

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report/assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def schema():
    # Keep every FK arrow, but put PK/FK names inside nodes. The slide's
    # companion join example explains selected routes without 20 edge labels.
    def subscript(name):
        base, suffix = name.rsplit("_", 1)
        return f"{base}<SUB>{suffix}</SUB>"

    def node_label(table, primary, fk_lines):
        lines = [table, f"PK  {subscript(primary)}"]
        lines += ["FK  " + " &#183; ".join(subscript(fk) for fk in row)
                  for row in fk_lines]
        cells = "".join(
            f'<TR><TD ALIGN="CENTER"><FONT FACE="DejaVu Sans Mono">{line}</FONT></TD></TR>'
            for line in lines
        )
        return f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="0">{cells}</TABLE>'

    nodes = {
        "Users": ("user_id", []),
        "Repositories": ("repo_id", []),
        "PullRequests": ("pr_id", [["repo_id", "author_id"], ["merged_by_id"]]),
        "Issues": ("issue_id", [["repo_id", "author_id"],
                                ["pull_request_id", "assignee_id"]]),
        "IssueComments": ("comment_id", [["issue_id", "repo_id", "author_id"]]),
        "PRReviews": ("review_id", [["pr_id", "repo_id", "reviewer_id"]]),
        "PRReviewComments": ("comment_id", [["pr_id", "repo_id", "author_id"],
                                           ["pull_request_review_id", "in_reply_to_id"]]),
        "Pushes": ("push_id", [["user_id", "repo_id"]]),
    }
    source = r'''digraph Schema {
      graph [rankdir=LR, bgcolor="white", splines=ortho, nodesep=0.90,
             ranksep=0.90, outputorder=edgesfirst, pad=0.12];
      node [shape=box, style="rounded,filled", fillcolor="white", color="black",
            penwidth=1.2, fontname="DejaVu Sans Mono", fontsize=12, margin="0.14,0.10"];
      edge [color="#666666", penwidth=0.85, arrowsize=0.55];
      Users [label=<@Users@>, penwidth=2];
      Repositories [label=<@Repositories@>, penwidth=2];
      PullRequests [label=<@PullRequests@>];
      Issues [label=<@Issues@>];
      IssueComments [label=<@IssueComments@>];
      PRReviews [label=<@PRReviews@>];
      PRReviewComments [label=<@PRReviewComments@>];
      Pushes [label=<@Pushes@>];
      {rank=same; PRReviewComments; IssueComments;}
      {rank=same; PRReviews; Issues;}
      {rank=same; PullRequests; Pushes;}
      {rank=same; Users; Repositories;}
      PRReviewComments -> IssueComments [style=invis, weight=10];
      PRReviews -> Issues [style=invis, weight=10];
      PullRequests -> Pushes [style=invis, weight=10];
      Users -> Repositories [style=invis, weight=10];
      PullRequests -> Repositories [minlen=1, style=dashed, color="#999999"];
      PullRequests -> Users;
      PullRequests -> Users;
      Issues -> Repositories [style=dashed, color="#999999"];
      Issues -> Users;
      Issues -> PullRequests;
      Issues -> Users;
      IssueComments -> Issues;
      IssueComments -> Repositories [style=dashed, color="#999999"];
      IssueComments -> Users;
      PRReviews -> PullRequests;
      PRReviews -> Repositories [style=dashed, color="#999999"];
      PRReviews -> Users;
      PRReviewComments -> PullRequests;
      PRReviewComments -> Repositories [style=dashed, color="#999999"];
      PRReviewComments -> Users;
      PRReviewComments -> PRReviews;
      PRReviewComments -> PRReviewComments [constraint=false];
      Pushes -> Users;
      Pushes -> Repositories [style=dashed, color="#999999"];
    }'''
    for name, (primary, fk_lines) in nodes.items():
        source = source.replace(f"@{name}@", node_label(name, primary, fk_lines))
    path = ASSETS / "schema_graph_bw.dot"
    path.write_text(source)
    subprocess.run(["dot", "-Tpng", "-Gdpi=260", str(path), "-o",
                    str(ASSETS / "schema_graph_bw.png")], check=True)


def framework():
    dot = r'''digraph Pipeline {
      graph [rankdir=LR, bgcolor="white", nodesep=0.38, ranksep=0.46,
             fontname="Helvetica", pad="0.12"];
      node [shape=box, style="rounded", color="black", fontname="Helvetica",
            fontsize=12, margin="0.14,0.10"];
      edge [color="black", arrowsize=0.75];
      raw [label="GH Archive\nhourly files"];
      prep [label="Instructor\npreprocess.py"];
      db [label="Eight Parquet\ntables"];
      feat [label="Visible-at-t joins\nUser / Temporal / Relation"];
      fold [label="Four chronological\nbacktests"];
      select [label="Select method\nby mean five-class Score"];
      tune [label="Last complete week\nstop + thresholds"];
      refit [label="Refit all eligible\nlabeled weeks"];
      out [label="Six classifiers\n4,465-row CSV"];
      quality [label="Coverage audit\npush-safe week mask"];
      raw -> prep -> db -> feat -> fold -> select -> tune -> refit -> out;
      db -> quality -> fold;
      quality -> tune;
      quality -> refit;
    }'''
    path = ASSETS / "framework_bw.dot"
    path.write_text(dot)
    subprocess.run(["dot", "-Tpng", "-Gdpi=240", str(path), "-o",
                    str(ASSETS / "framework_bw.png")], check=True)


def framework_grid():
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    items = [
        (0.25, 2.85, "1  GH Archive\n    hourly files"),
        (3.55, 2.85, "2  Instructor\n    eight Parquet tables"),
        (6.85, 2.85, "3  Visible-at-t\n    feature snapshots"),
        (6.85, 1.55, "4  Coverage audit\n    push-safe labels"),
        (3.55, 1.55, "5  Four rolling\n    holdouts"),
        (0.25, 1.55, "6  Select feature set\n    by five-class Score"),
        (0.25, 0.25, "7  Last-week tuning\n    rounds + thresholds"),
        (3.55, 0.25, "8  Refit all eligible\n    labeled weeks"),
        (6.85, 0.25, "9  Six classifiers\n    4,465-row CSV"),
    ]
    for x, y, label in items:
        ax.add_patch(FancyBboxPatch((x, y), 2.85, 0.85,
                                   boxstyle="round,pad=0.05,rounding_size=0.06",
                                   facecolor="white", edgecolor="black", linewidth=1.1))
        ax.text(x + 0.15, y + 0.43, label, va="center", ha="left", fontsize=10.5)
    arrows = [
        ((3.10, 3.28), (3.50, 3.28)), ((6.40, 3.28), (6.80, 3.28)),
        ((8.28, 2.83), (8.28, 2.48)), ((6.80, 1.98), (6.45, 1.98)),
        ((3.50, 1.98), (3.15, 1.98)), ((1.68, 1.53), (1.68, 1.18)),
        ((3.10, 0.68), (3.50, 0.68)), ((6.40, 0.68), (6.80, 0.68)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.1})
    fig.tight_layout(pad=0.1)
    fig.savefig(ASSETS / "framework_grid.png", dpi=260)
    plt.close(fig)


def model_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    def box(x, y, w, h, label, weight=1.15, size=10.5, family="DejaVu Sans"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.04,rounding_size=0.055",
                                   facecolor="white", edgecolor="black", linewidth=weight))
        ax.text(x + w / 2, y + h / 2, label, va="center", ha="center",
                fontsize=size, fontfamily=family)

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0})

    box(0.25, 3.30, 9.5, 0.57,
        "8 relational tables  |  Users · Repositories · PRs · Issues · Reviews · Comments · Pushes",
        size=11)
    box(0.4, 2.20, 2.7, 0.63, "User history\n18 features", weight=1.75)
    box(3.65, 2.20, 2.7, 0.63, "Temporal\n7/14/30d · recency · trend")
    box(6.90, 2.20, 2.7, 0.63, "Relational\nrepo volume · interactions")
    for x in (1.75, 5.0, 8.25):
        arrow((x, 3.27), (x, 2.87))
        arrow((x, 2.17), (5.0, 1.68))
    box(3.60, 1.05, 2.80, 0.60, "Chronological ablation\nselect User history", weight=1.4)
    arrow((6.45, 1.35), (6.85, 1.35))
    box(6.90, 1.05, 2.70, 0.60, "Six LightGBM\nbinary classifiers", weight=1.75)
    arrow((8.25, 1.00), (8.25, 0.70))
    outputs = (r"6 outputs: $\mathrm{open}_{\mathrm{pr}}$ · "
               r"$\mathrm{open}_{\mathrm{issue}}$ · $\mathrm{issue}_{\mathrm{comment}}$"
               "\n"
               r"$\mathrm{pr}_{\mathrm{review}}$ · "
               r"$\mathrm{pr}_{\mathrm{review\_comment}}$ · $\mathrm{push}$")
    box(0.40, 0.05, 9.20, 0.61, outputs, size=10.1)
    ax.text(0.40, 0.90,
            "Training strategy: four rolling holdouts → tune rounds and thresholds → refit eligible labeled weeks",
            fontsize=9.1, va="center")
    fig.tight_layout(pad=0.06)
    fig.savefig(ASSETS / "model_architecture_bw.png", dpi=260)
    plt.close(fig)


def scores():
    data = pd.read_csv(ROOT / "final_submission/metrics/backtest_scores.csv")
    order = ["sep26", "oct10", "oct17", "oct31"]
    styles = {
        "A_copy": ("Copy last week", "#aaaaaa", ":", "x"),
        "B_user": ("User", "#111111", "-", "o"),
        "C_user_temporal": ("User + temporal", "#555555", "--", "s"),
        "F_repo_interactions": ("User + temporal + relations", "#777777", "-.", "^"),
    }
    fig, ax = plt.subplots(figsize=(9.1, 3.7))
    for arm, (label, color, line, marker) in styles.items():
        part = data.loc[data.arm.eq(arm)].set_index("fold").loc[order]
        ax.plot(order, part.score * 100, label=label, color=color,
                linestyle=line, marker=marker, linewidth=1.8, markersize=6)
    ax.set_ylim(32, 47)
    ax.set_ylabel("Five-activity Score (%)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(ASSETS / "backtest_bw.png", dpi=250)
    plt.close(fig)


def quality():
    data = pd.read_csv(ROOT / "final_submission/quality/daily_activity_quality.csv",
                       parse_dates=["day_utc"])
    fig, ax = plt.subplots(figsize=(9.1, 3.25))
    ax.plot(data.day_utc, data.push_rows, color="black", linewidth=1.4)
    gaps = data.loc[data.push_zero_gap]
    ax.scatter(gaps.day_utc, gaps.push_rows, color="black", marker="x", s=35,
               label="Full UTC day with zero observed pushes")
    ax.set_ylabel("Visible Pushes rows / UTC day")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ASSETS / "push_quality_bw.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    schema()
    framework()
    framework_grid()
    model_architecture()
    scores()
    quality()
