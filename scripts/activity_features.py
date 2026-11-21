"""Leakage-aware features and labels from the eight instructor-produced tables."""

from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITIES = [
    "open_pr", "open_issue", "issue_comment",
    "pr_review", "pr_review_comment", "push",
]
TABLE_NAMES = [
    "Users", "Repositories", "PullRequests", "Issues",
    "IssueComments", "PRReviews", "PRReviewComments", "Pushes",
]
ROW_TIMES = {
    "Users": "first_seen_at", "Repositories": "first_seen_at",
    "PullRequests": "first_seen_at", "Issues": "first_seen_at",
    "IssueComments": "created_at", "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at", "Pushes": "created_at",
}
SPECS = [
    ("open_pr", "PullRequests", "author_id", "created_at", "pr_id"),
    ("open_issue", "Issues", "author_id", "created_at", "issue_id"),
    ("issue_comment", "IssueComments", "author_id", "created_at", "issue_id"),
    ("pr_review", "PRReviews", "reviewer_id", "submitted_at", "pr_id"),
    ("pr_review_comment", "PRReviewComments", "author_id", "created_at", "pr_id"),
    ("push", "Pushes", "user_id", "created_at", "push_id"),
]


def add_counts(result, source, by, name):
    """Map a grouped count onto the fixed UserID index."""
    result[name] = source.groupby(by, observed=True).size().reindex(result.index, fill_value=0)


def add_nunique(result, source, by, value, name):
    result[name] = source.groupby(by, observed=True)[value].nunique().reindex(result.index, fill_value=0)


class ActivityData:
    def __init__(self, db):
        db = Path(db)
        self.tables = {name: pd.read_parquet(db / f"{name}.parquet") for name in TABLE_NAMES}
        self.users = self.tables["Users"]
        self.user_code = self.users.set_index("user_id")["github_user_id"]
        self.repo_seen = self.tables["Repositories"].set_index("repo_id")["first_seen_at"]
        self.events = self._make_events()

    def _make_events(self):
        frames = []
        for activity, table_name, actor, action_time, entity in SPECS:
            table = self.tables[table_name]
            if activity == "open_issue":
                table = table.loc[table["is_pull_request"].eq(False)]
            row_time = ROW_TIMES[table_name]
            frame = table[[actor, "repo_id", entity, action_time, row_time]].copy()
            frame.columns = ["user_id", "repo_id", "entity_id", "event_at", "visible_at"]
            frame["UserID"] = frame["user_id"].map(self.user_code)
            frame["activity"] = activity
            frame = frame.dropna(subset=["UserID", "event_at", "visible_at"])
            frames.append(frame[["UserID", "user_id", "repo_id", "entity_id",
                                 "event_at", "visible_at", "activity"]])
        events = pd.concat(frames, ignore_index=True)
        events["UserID"] = events["UserID"].astype("int64")
        return events

    def labels(self, users, start):
        """One week of observable labels; data coverage is checked by the caller."""
        start = pd.Timestamp(start)
        end = start + pd.Timedelta(days=7)
        frame = self.events.loc[
            self.events["event_at"].ge(start)
            & self.events["event_at"].lt(end)
            & self.events["visible_at"].lt(end)
        ]
        result = pd.DataFrame(0, index=pd.Index(users, name="UserID"), columns=ACTIVITIES, dtype="int8")
        for activity in ACTIVITIES:
            ids = frame.loc[frame["activity"].eq(activity), "UserID"].unique()
            result.loc[result.index.isin(ids), activity] = 1
        return result

    def features(self, users, cutoff):
        """Feature groups use only rows visible strictly before cutoff."""
        cutoff = pd.Timestamp(cutoff)
        index = pd.Index(users, name="UserID")
        base = pd.DataFrame(index=index)
        temporal = pd.DataFrame(index=index)
        relational = pd.DataFrame(index=index)
        interaction = pd.DataFrame(index=index)
        available = self.events.loc[
            self.events["visible_at"].lt(cutoff) & self.events["event_at"].lt(cutoff)
        ].copy()
        available = available.loc[available["UserID"].isin(index)]
        available["days_ago"] = (cutoff - available["event_at"]).dt.total_seconds() / 86400

        # User features: cumulative historical behavior and participation.
        add_counts(base, available, "UserID", "usr_events_total")
        add_nunique(base, available, "UserID", "repo_id", "usr_repositories_total")
        available["event_day"] = available["event_at"].dt.floor("D")
        add_nunique(base, available, "UserID", "event_day", "usr_active_days_total")
        seen = self.users.set_index("github_user_id")["first_seen_at"].reindex(index)
        base["usr_seen"] = seen.lt(cutoff).fillna(False).astype("int8")
        base["usr_observed_days"] = ((cutoff - seen).dt.total_seconds() / 86400).clip(lower=0).fillna(0)
        for activity in ACTIVITIES:
            subset = available.loc[available["activity"].eq(activity)]
            add_counts(base, subset, "UserID", f"usr_{activity}_total")
            base[f"usr_{activity}_share"] = (
                base[f"usr_{activity}_total"] / base["usr_events_total"].clip(lower=1)
            )
        base["usr_events_per_day"] = (
            base["usr_events_total"] / base["usr_observed_days"].clip(lower=1)
        )

        # Temporal features use action time AND the row's visibility time.
        for days in (7, 14, 30):
            window = available.loc[available["days_ago"].lt(days)]
            add_counts(temporal, window, "UserID", f"tmp_events_{days}d")
            add_nunique(temporal, window, "UserID", "event_day", f"tmp_active_days_{days}d")
            for activity in ACTIVITIES:
                subset = window.loc[window["activity"].eq(activity)]
                add_counts(temporal, subset, "UserID", f"tmp_{activity}_{days}d")
        temporal["tmp_any_recency_days"] = (
            available.groupby("UserID")["days_ago"].min().reindex(index).fillna(999)
        )
        for activity in ACTIVITIES:
            subset = available.loc[available["activity"].eq(activity)]
            temporal[f"tmp_{activity}_recency_days"] = (
                subset.groupby("UserID")["days_ago"].min().reindex(index).fillna(999)
            )
            recent = temporal[f"tmp_{activity}_7d"]
            previous = temporal[f"tmp_{activity}_14d"] - recent
            temporal[f"tmp_{activity}_week_delta"] = recent - previous
            temporal[f"tmp_{activity}_weekly_rate_30d"] = (
                temporal[f"tmp_{activity}_30d"] / 30 * 7
            )
        temporal["tmp_events_week_delta"] = (
            2 * temporal["tmp_events_7d"] - temporal["tmp_events_14d"]
        )

        # Repository neighborhood, with the user's own events subtracted.
        recent30 = available.loc[available["days_ago"].lt(30) & available["repo_id"].notna()].copy()
        visible_repos = self.tables["Repositories"].loc[
            self.tables["Repositories"]["first_seen_at"].lt(cutoff),
            ["repo_id", "first_seen_at"],
        ]
        recent30 = recent30.merge(visible_repos, on="repo_id", how="inner")
        repo_total = recent30.groupby("repo_id").size().rename("repo_total")
        repo_users = recent30.groupby("repo_id")["UserID"].nunique().rename("repo_users")
        own = recent30.groupby(["UserID", "repo_id"]).size().rename("own").reset_index()
        own = own.join(repo_total, on="repo_id").join(repo_users, on="repo_id")
        own = own.merge(visible_repos, on="repo_id", how="left")
        own["neighbor_events"] = own["repo_total"] - own["own"]
        own["repo_age_days"] = (cutoff - own["first_seen_at"]).dt.total_seconds() / 86400
        for column, agg in (
            ("neighbor_events", "sum"), ("neighbor_events", "mean"),
            ("neighbor_events", "max"), ("repo_users", "mean"),
            ("repo_age_days", "mean"),
        ):
            name = f"rel_repo_{column}_{agg}_30d"
            relational[name] = own.groupby("UserID")[column].agg(agg).reindex(index).fillna(0)
        add_nunique(relational, recent30, "UserID", "repo_id", "rel_active_repos_30d")

        # Count interactions with visible PR and issue rows only.
        prs = self.tables["PullRequests"].loc[
            self.tables["PullRequests"]["first_seen_at"].lt(cutoff),
            ["pr_id", "author_id", "repo_id", "is_cross_repo", "created_at"],
        ].copy()
        issues = self.tables["Issues"].loc[
            self.tables["Issues"]["first_seen_at"].lt(cutoff),
            ["issue_id", "author_id", "is_pull_request", "assignee_id", "pull_request_id"],
        ].copy()
        prs["pr_author_code"] = prs["author_id"].map(self.user_code)
        issues["issue_author_code"] = issues["author_id"].map(self.user_code)
        issues["assignee_code"] = issues["assignee_id"].map(self.user_code)
        add_counts(
            relational, prs.loc[prs["is_cross_repo"].eq(True)], "pr_author_code",
            "rel_cross_repo_prs_total",
        )
        add_counts(relational, issues, "assignee_code", "rel_issues_assigned_total")

        reviews = self.tables["PRReviews"].loc[
            self.tables["PRReviews"]["first_seen_at"].lt(cutoff)
            & self.tables["PRReviews"]["submitted_at"].ge(cutoff - pd.Timedelta(days=30))
            & self.tables["PRReviews"]["submitted_at"].lt(cutoff),
            ["review_id", "pr_id", "reviewer_id"],
        ].merge(prs[["pr_id", "repo_id", "pr_author_code"]], on="pr_id", how="inner")
        reviews["reviewer_code"] = reviews["reviewer_id"].map(self.user_code)
        add_counts(relational, reviews, "pr_author_code", "rel_pr_reviews_received_30d")
        add_nunique(
            relational, reviews, "pr_author_code", "reviewer_id",
            "rel_distinct_reviewers_received_30d",
        )
        add_nunique(
            relational, reviews, "reviewer_code", "pr_author_code",
            "rel_distinct_pr_authors_reviewed_30d",
        )

        comments = self.tables["IssueComments"].loc[
            self.tables["IssueComments"]["created_at"].ge(cutoff - pd.Timedelta(days=30))
            & self.tables["IssueComments"]["created_at"].lt(cutoff),
            ["issue_id", "author_id"],
        ].merge(
            issues[["issue_id", "issue_author_code", "is_pull_request"]],
            on="issue_id", how="inner",
        )
        add_counts(relational, comments, "issue_author_code", "rel_issue_comments_received_30d")
        add_nunique(
            relational, comments, "issue_author_code", "author_id",
            "rel_distinct_issue_commenters_30d",
        )
        add_counts(
            relational, comments.loc[comments["is_pull_request"].eq(True)],
            "issue_author_code", "rel_pr_discussion_comments_received_30d",
        )

        review_comments = self.tables["PRReviewComments"].loc[
            self.tables["PRReviewComments"]["created_at"].ge(cutoff - pd.Timedelta(days=30))
            & self.tables["PRReviewComments"]["created_at"].lt(cutoff),
            ["pr_id", "author_id", "pull_request_review_id"],
        ].merge(prs[["pr_id", "pr_author_code"]], on="pr_id", how="inner")
        add_counts(
            relational, review_comments, "pr_author_code",
            "rel_pr_code_comments_received_30d",
        )
        visible_review_ids = set(
            self.tables["PRReviews"].loc[
                self.tables["PRReviews"]["first_seen_at"].lt(cutoff), "review_id"
            ]
        )
        linked = review_comments.loc[
            review_comments["pull_request_review_id"].isin(visible_review_ids)
        ].copy()
        linked["commenter_code"] = linked["author_id"].map(self.user_code)
        add_counts(
            relational, linked, "commenter_code",
            "rel_code_comments_linked_to_visible_reviews_30d",
        )

        # Reviewer -> PR -> repository interactions. Most PR authors are absent
        # from the archive; these features retain reviewer/repository paths even
        # when author_id is null. Every joined PR/review is visible before cutoff.
        vr = reviews.dropna(subset=["reviewer_code", "pr_id", "repo_id"]).copy()
        add_nunique(interaction, vr, "reviewer_code", "pr_id", "int_reviewed_prs_30d")
        add_nunique(interaction, vr, "reviewer_code", "repo_id", "int_reviewed_repos_30d")
        reviewer_repo = (vr.groupby(["reviewer_code", "repo_id"], observed=True)
                         .agg(reviews=("review_id", "size"), prs=("pr_id", "nunique"))
                         .reset_index())
        repo_review_count = vr.groupby("repo_id", observed=True).size().rename("repo_reviews")
        reviewer_repo = reviewer_repo.join(repo_review_count, on="repo_id")
        reviewer_repo["peer_reviews"] = reviewer_repo["repo_reviews"] - reviewer_repo["reviews"]
        for column, aggregate in (("reviews", "max"), ("prs", "max"),
                                  ("peer_reviews", "mean"), ("peer_reviews", "max")):
            interaction[f"int_reviewer_repo_{column}_{aggregate}_30d"] = (
                reviewer_repo.groupby("reviewer_code")[column].agg(aggregate)
                .reindex(index).fillna(0)
            )
        pr_reviewer_count = vr.groupby("pr_id")["reviewer_id"].nunique().rename("reviewer_count")
        vr = vr.join(pr_reviewer_count, on="pr_id")
        interaction["int_pr_coreviewers_mean_30d"] = (
            vr.assign(other_reviewers=(vr["reviewer_count"] - 1).clip(lower=0))
            .groupby("reviewer_code")["other_reviewers"].mean().reindex(index).fillna(0)
        )
        add_nunique(interaction, vr.loc[vr["reviewer_count"].gt(1)],
                    "reviewer_code", "pr_id", "int_multi_reviewer_prs_30d")
        add_counts(interaction, vr.loc[vr["pr_author_code"].notna()],
                   "reviewer_code", "int_reviews_with_known_author_30d")
        author_pairs = (vr.dropna(subset=["pr_author_code"])
                        .groupby(["reviewer_code", "pr_author_code"], observed=True)
                        .size().rename("times").reset_index())
        add_counts(interaction, author_pairs.loc[author_pairs["times"].ge(2)],
                   "reviewer_code", "int_repeat_author_pairs_30d")

        # A review and a code comment by the same user on the same PR make an
        # observed two-edge interaction, regardless of whether PR author is known.
        reviewed_pairs = vr[["reviewer_id", "pr_id"]].drop_duplicates()
        same_pr_code = review_comments.merge(
            reviewed_pairs, left_on=["author_id", "pr_id"],
            right_on=["reviewer_id", "pr_id"], how="inner",
        )
        same_pr_code["reviewer_code"] = same_pr_code["author_id"].map(self.user_code)
        add_counts(interaction, same_pr_code, "reviewer_code", "int_review_code_comments_same_pr_30d")
        add_nunique(interaction, same_pr_code, "reviewer_code", "pr_id",
                    "int_review_code_comment_prs_30d")

        # PR conversation issue -> PR -> reviewer, with the issue row visible.
        pr_issues = issues.loc[issues["is_pull_request"].eq(True),
                               ["issue_id", "pull_request_id"]].dropna(subset=["pull_request_id"])
        discussion = self.tables["IssueComments"].loc[
            self.tables["IssueComments"]["created_at"].ge(cutoff - pd.Timedelta(days=30))
            & self.tables["IssueComments"]["created_at"].lt(cutoff),
            ["issue_id", "author_id"],
        ].merge(pr_issues, on="issue_id", how="inner")
        same_pr_discussion = discussion.merge(
            reviewed_pairs, left_on=["author_id", "pull_request_id"],
            right_on=["reviewer_id", "pr_id"], how="inner",
        )
        same_pr_discussion["reviewer_code"] = same_pr_discussion["author_id"].map(self.user_code)
        add_counts(interaction, same_pr_discussion, "reviewer_code",
                   "int_review_discussion_comments_same_pr_30d")

        authored_repos = prs.loc[
            prs["author_id"].notna() & prs["created_at"].ge(cutoff - pd.Timedelta(days=30))
            & prs["created_at"].lt(cutoff), ["author_id", "repo_id"],
        ].drop_duplicates()
        # reviewer_code is anonymized GitHub ID; use the row-number reviewer_id
        # for the FK join to authored PRs.
        reviewer_repo_ids = vr[["reviewer_id", "repo_id", "reviewer_code"]].drop_duplicates()
        overlap = reviewer_repo_ids.merge(
            authored_repos, left_on=["reviewer_id", "repo_id"],
            right_on=["author_id", "repo_id"], how="inner",
        )
        add_nunique(interaction, overlap, "reviewer_code", "repo_id",
                    "int_reviewer_authored_repo_overlap_30d")

        repo_columns = [name for name in relational if name.startswith("rel_repo_") or name == "rel_active_repos_30d"]
        groups = {"user": base, "temporal": temporal, "relational": relational,
                  "repo_only": relational[repo_columns].copy(), "interaction": interaction}
        for group in groups.values():
            group.fillna(0, inplace=True)
            if not np.isfinite(group.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite feature at {cutoff}")
        return groups
