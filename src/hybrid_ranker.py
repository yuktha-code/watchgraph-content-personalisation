from __future__ import annotations

import numpy as np


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


class HybridRanker:
    def __init__(
        self,
        popularity,
        collaborative,
        content,
        w_pop: float = 0.15,
        w_cf: float = 0.55,
        w_content: float = 0.30,
    ):
        total = w_pop + w_cf + w_content
        self.popularity = popularity
        self.collaborative = collaborative
        self.content = content
        self.weights = (w_pop / total, w_cf / total, w_content / total)

    def score_items(self, user_id: int, candidate_items: list[int]) -> dict[int, float]:
        pop = _minmax(self.popularity.score_items(user_id, candidate_items))
        cf = _minmax(self.collaborative.score_items(user_id, candidate_items))
        con = _minmax(self.content.score_items(user_id, candidate_items))
        wp, wc, wt = self.weights
        return {
            int(i): wp * pop.get(int(i), 0.0)
            + wc * cf.get(int(i), 0.0)
            + wt * con.get(int(i), 0.0)
            for i in candidate_items
        }

    def recommend(self, user_id: int, candidate_items: list[int], k: int = 10) -> list[int]:
        scores = self.score_items(user_id, candidate_items)
        return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]
