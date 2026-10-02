from __future__ import annotations

import pandas as pd


def temporal_leave_one_out(
    ratings: pd.DataFrame,
    min_interactions: int = 5,
    positive_threshold: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each sufficiently active user, reserve the latest positive interaction for test.
    All earlier interactions remain in training. This avoids future-to-past leakage.
    """
    r = ratings.sort_values(["user_id", "timestamp"]).copy()
    counts = r.groupby("user_id").size()
    eligible = counts[counts >= min_interactions].index
    r = r[r.user_id.isin(eligible)]

    positives = r[r.rating >= positive_threshold]
    test_idx = positives.groupby("user_id")["timestamp"].idxmax()
    test = r.loc[test_idx].copy()
    train = r.drop(index=test_idx).copy()

    # Keep only users whose test item is genuinely unseen after removing it.
    duplicated = train.merge(
        test[["user_id", "item_id"]], on=["user_id", "item_id"], how="inner"
    )[["user_id"]].drop_duplicates()["user_id"]
    if len(duplicated):
        test = test[~test.user_id.isin(duplicated)]
        train = train[train.user_id.isin(test.user_id.unique())]

    return train.reset_index(drop=True), test.reset_index(drop=True)


def seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(uid): set(group.item_id.astype(int))
        for uid, group in train.groupby("user_id")
    }
