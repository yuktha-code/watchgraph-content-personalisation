from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from .ingestion import load_movielens
from .preprocessing import temporal_leave_one_out, seen_items
from .baselines import PopularityRecommender
from .collaborative_filtering import ItemKNNRecommender
from .content_recommender import ContentRecommender
from .hybrid_ranker import HybridRanker


def ranking_metrics(ranked: list[int], relevant: set[int], k: int) -> tuple[float, float, float]:
    topk = ranked[:k]
    hits = [1 if item in relevant else 0 for item in topk]
    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, ndcg


def evaluate_model(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: list[int],
    k: int = 10,
    max_users: int | None = None,
) -> dict[str, float]:
    seen = seen_items(train)
    rows = []
    users = list(test.user_id.astype(int).unique())
    if max_users:
        users = users[:max_users]

    for uid in users:
        relevant = set(test.loc[test.user_id == uid, "item_id"].astype(int))
        candidates = [i for i in all_items if i not in seen.get(uid, set())]
        scores = model.score_items(uid, candidates)
        ranked = [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        p, r, n = ranking_metrics(ranked, relevant, k)
        rows.append((p, r, n))

    arr = np.asarray(rows, dtype=float)
    return {
        f"precision@{k}": float(arr[:, 0].mean()) if len(arr) else 0.0,
        f"recall@{k}": float(arr[:, 1].mean()) if len(arr) else 0.0,
        f"ndcg@{k}": float(arr[:, 2].mean()) if len(arr) else 0.0,
        "evaluated_users": int(len(rows)),
    }


def main(k: int = 10) -> dict:
    ratings, movies = load_movielens()
    train, test = temporal_leave_one_out(ratings)
    all_items = sorted(movies.item_id.astype(int).unique())

    pop = PopularityRecommender().fit(train)
    cf = ItemKNNRecommender(top_neighbors=40).fit(train)
    content = ContentRecommender().fit(train, movies)
    hybrid = HybridRanker(pop, cf, content)

    results = {}
    for name, model in {
        "popularity": pop,
        "item_knn": cf,
        "content": content,
        "hybrid": hybrid,
    }.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, train, test, all_items, k=k)
        print(results[name])

    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = pd.DataFrame(results).T
    summary.to_csv("results/metrics.csv")
    print("\nSaved results/metrics.json and results/metrics.csv")
    print(summary)
    return results


if __name__ == "__main__":
    main()
