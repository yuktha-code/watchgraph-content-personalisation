from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNNRecommender:
    """Mean-centered item-item collaborative filtering."""

    def __init__(self, top_neighbors: int = 40):
        self.top_neighbors = top_neighbors

    def fit(self, train: pd.DataFrame) -> "ItemKNNRecommender":
        matrix = train.pivot_table(
            index="user_id", columns="item_id", values="rating", aggfunc="mean"
        )
        self.user_ids_ = list(matrix.index.astype(int))
        self.item_ids_ = list(matrix.columns.astype(int))
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_index_ = {m: i for i, m in enumerate(self.item_ids_)}

        user_means = matrix.mean(axis=1)
        centered = matrix.sub(user_means, axis=0).fillna(0.0)
        self.user_means_ = {int(k): float(v) for k, v in user_means.items()}
        self.ratings_ = matrix

        # Similarity between movie columns.
        self.similarity_ = cosine_similarity(centered.T)
        np.fill_diagonal(self.similarity_, 0.0)
        return self

    def _score_one(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_index_ or item_id not in self.item_index_:
            return self.user_means_.get(user_id, 0.0)

        row = self.ratings_.loc[user_id]
        rated = row.dropna()
        if rated.empty:
            return self.user_means_.get(user_id, 0.0)

        target_idx = self.item_index_[item_id]
        pairs = []
        for seen_item, rating in rated.items():
            seen_item = int(seen_item)
            if seen_item not in self.item_index_:
                continue
            sim = float(self.similarity_[target_idx, self.item_index_[seen_item]])
            if sim > 0:
                pairs.append((sim, float(rating)))

        if not pairs:
            return self.user_means_.get(user_id, float(rated.mean()))

        pairs.sort(reverse=True, key=lambda x: x[0])
        pairs = pairs[: self.top_neighbors]
        sims = np.array([x[0] for x in pairs], dtype=float)
        vals = np.array([x[1] for x in pairs], dtype=float)
        denom = np.abs(sims).sum()
        if denom <= 1e-12:
            return self.user_means_.get(user_id, float(rated.mean()))
        return float((sims @ vals) / denom)

    def score_items(self, user_id: int, candidate_items: list[int]) -> dict[int, float]:
        return {int(i): self._score_one(int(user_id), int(i)) for i in candidate_items}
