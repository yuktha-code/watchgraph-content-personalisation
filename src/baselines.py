from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityRecommender:
    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        stats = train.groupby("item_id").agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "size"),
        )
        # Bayesian-style shrinkage to reduce tiny-sample inflation.
        global_mean = float(train.rating.mean())
        m = float(stats.rating_count.quantile(0.75))
        stats["score"] = (
            (stats.rating_count / (stats.rating_count + m)) * stats.mean_rating
            + (m / (stats.rating_count + m)) * global_mean
        )
        self.scores_ = stats["score"].to_dict()
        self.default_ = min(self.scores_.values()) if self.scores_ else 0.0
        return self

    def score_items(self, user_id: int, candidate_items: list[int]) -> dict[int, float]:
        return {int(i): float(self.scores_.get(int(i), self.default_)) for i in candidate_items}
