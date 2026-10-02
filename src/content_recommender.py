from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .ingestion import GENRES


class ContentRecommender:
    """Genre-profile recommender using the MovieLens metadata."""

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame) -> "ContentRecommender":
        self.movies_ = movies.set_index("item_id").copy()
        self.genre_matrix_ = self.movies_[GENRES].astype(float)
        self.train_ = train.copy()
        self.global_mean_ = float(train.rating.mean())

        self.user_profiles_ = {}
        for uid, group in train.groupby("user_id"):
            group = group[group.item_id.isin(self.genre_matrix_.index)]
            if group.empty:
                continue
            X = self.genre_matrix_.loc[group.item_id].to_numpy(dtype=float)
            weights = (group.rating.to_numpy(dtype=float) - self.global_mean_)
            if np.allclose(weights, 0):
                weights = np.ones_like(weights)
            profile = (X * weights[:, None]).sum(axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            self.user_profiles_[int(uid)] = profile
        return self

    def score_items(self, user_id: int, candidate_items: list[int]) -> dict[int, float]:
        profile = self.user_profiles_.get(int(user_id))
        if profile is None:
            return {int(i): 0.0 for i in candidate_items}

        valid = [int(i) for i in candidate_items if int(i) in self.genre_matrix_.index]
        if not valid:
            return {int(i): 0.0 for i in candidate_items}

        X = self.genre_matrix_.loc[valid].to_numpy(dtype=float)
        scores = cosine_similarity(X, profile.reshape(1, -1)).ravel()
        out = {i: float(s) for i, s in zip(valid, scores)}
        return {int(i): out.get(int(i), 0.0) for i in candidate_items}
