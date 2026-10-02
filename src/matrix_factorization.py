from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    """
    Latent-factor recommender using TruncatedSVD on a user-item preference matrix.

    Ratings are mean-centered per user before factorization. Prediction scores are
    reconstructed from user and item latent representations and shifted by the
    user's historical mean rating.
    """

    def __init__(self, n_components: int = 40, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, train: pd.DataFrame) -> "SVDRecommender":
        users = sorted(train.user_id.astype(int).unique())
        items = sorted(train.item_id.astype(int).unique())

        self.user_ids_ = users
        self.item_ids_ = items
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {m: i for i, m in enumerate(items)}

        user_means = train.groupby("user_id")["rating"].mean().to_dict()
        self.user_means_ = {int(k): float(v) for k, v in user_means.items()}
        self.global_mean_ = float(train.rating.mean())

        rows, cols, vals = [], [], []
        for row in train.itertuples(index=False):
            uid = int(row.user_id)
            iid = int(row.item_id)
            rows.append(self.user_index_[uid])
            cols.append(self.item_index_[iid])
            vals.append(float(row.rating) - self.user_means_[uid])

        matrix = csr_matrix(
            (np.asarray(vals, dtype=float), (rows, cols)),
            shape=(len(users), len(items)),
        )

        n_comp = min(
            self.n_components,
            max(2, min(matrix.shape) - 1),
        )
        self.svd_ = TruncatedSVD(
            n_components=n_comp,
            random_state=self.random_state,
        )
        self.user_factors_ = self.svd_.fit_transform(matrix)
        self.item_factors_ = self.svd_.components_.T
        self.explained_variance_ratio_ = float(
            self.svd_.explained_variance_ratio_.sum()
        )
        return self

    def _score_one(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_index_:
            return self.global_mean_
        if item_id not in self.item_index_:
            return self.user_means_.get(user_id, self.global_mean_)

        u = self.user_factors_[self.user_index_[user_id]]
        v = self.item_factors_[self.item_index_[item_id]]
        return float(
            self.user_means_.get(user_id, self.global_mean_) + np.dot(u, v)
        )

    def score_items(self, user_id: int, candidate_items: list[int]) -> dict[int, float]:
        return {
            int(i): self._score_one(int(user_id), int(i))
            for i in candidate_items
        }
