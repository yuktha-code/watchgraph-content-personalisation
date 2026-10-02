from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .ingestion import load_movielens
from .baselines import PopularityRecommender
from .matrix_factorization import SVDRecommender


def temporal_train_val_test(ratings: pd.DataFrame, min_interactions: int = 8, positive_threshold: float = 4.0):
    r = ratings.sort_values(["user_id", "timestamp"]).copy()
    counts = r.groupby("user_id").size()
    eligible = counts[counts >= min_interactions].index
    r = r[r.user_id.isin(eligible)].copy()

    positives = r[r.rating >= positive_threshold].copy()
    pos_counts = positives.groupby("user_id").size()
    eligible = pos_counts[pos_counts >= 2].index
    r = r[r.user_id.isin(eligible)].copy()
    positives = positives[positives.user_id.isin(eligible)].copy()

    positives["rank"] = positives.groupby("user_id")["timestamp"].rank(method="first", ascending=False)
    test_idx = positives.loc[positives["rank"] == 1].index
    val_idx = positives.loc[positives["rank"] == 2].index

    test = r.loc[test_idx].copy()
    val = r.loc[val_idx].copy()
    train = r.drop(index=test_idx.union(val_idx)).copy()

    users = sorted(set(train.user_id) & set(val.user_id) & set(test.user_id))
    return (
        train[train.user_id.isin(users)].reset_index(drop=True),
        val[val.user_id.isin(users)].reset_index(drop=True),
        test[test.user_id.isin(users)].reset_index(drop=True),
    )


def seen_map(df: pd.DataFrame):
    return {int(uid): set(g.item_id.astype(int)) for uid, g in df.groupby("user_id")}


def norm_dict(d):
    if not d:
        return {}
    keys = list(d)
    vals = np.array([d[k] for k in keys], dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {int(k): 0.0 for k in keys}
    vals = (vals - lo) / (hi - lo)
    return {int(k): float(v) for k, v in zip(keys, vals)}


def metric_row(ranked, relevant, k=10):
    top = ranked[:k]
    hits = [1 if i in relevant else 0 for i in top]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum(h / math.log2(j + 2) for j, h in enumerate(hits))
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(j + 2) for j in range(ideal))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, ndcg


def precompute_component_scores(pop, svd, train, heldout, all_items):
    seen = seen_map(train)
    cache = {}
    for n, uid in enumerate(sorted(heldout.user_id.astype(int).unique()), start=1):
        candidates = [i for i in all_items if i not in seen.get(uid, set())]
        pop_s = norm_dict(pop.score_items(uid, candidates))
        svd_s = norm_dict(svd.score_items(uid, candidates))
        relevant = set(heldout.loc[heldout.user_id == uid, "item_id"].astype(int))
        cache[uid] = (candidates, pop_s, svd_s, relevant)
        if n % 200 == 0:
            print(f"  precomputed {n} users")
    return cache


def evaluate_alpha(cache, alpha_svd, k=10):
    rows = []
    catalog = set()
    for _, (candidates, pop_s, svd_s, relevant) in cache.items():
        ranked = sorted(
            candidates,
            key=lambda i: alpha_svd * svd_s.get(i, 0.0) + (1.0 - alpha_svd) * pop_s.get(i, 0.0),
            reverse=True,
        )
        catalog.update(ranked[:k])
        rows.append(metric_row(ranked, relevant, k))
    arr = np.asarray(rows, dtype=float)
    return {
        f"precision@{k}": float(arr[:, 0].mean()),
        f"recall@{k}": float(arr[:, 1].mean()),
        f"ndcg@{k}": float(arr[:, 2].mean()),
        "evaluated_users": int(len(rows)),
        "catalog_count": int(len(catalog)),
    }


def main(k=10):
    ratings, movies = load_movielens()
    train, val, test = temporal_train_val_test(ratings)
    all_items = sorted(movies.item_id.astype(int).unique())

    print(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}, users={test.user_id.nunique():,}")

    print("\nFitting validation models...")
    pop = PopularityRecommender().fit(train)
    svd = SVDRecommender(n_components=40, random_state=42).fit(train)

    print("Precomputing validation component scores...")
    val_cache = precompute_component_scores(pop, svd, train, val, all_items)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    val_results = {}
    best_alpha = None
    best_ndcg = -1.0

    print("\nValidation alpha search")
    for a in alphas:
        m = evaluate_alpha(val_cache, a, k=k)
        val_results[str(a)] = m
        print(f"alpha_svd={a:.2f}", m)
        if m[f"ndcg@{k}"] > best_ndcg:
            best_ndcg = m[f"ndcg@{k}"]
            best_alpha = a

    print(f"\nBest alpha_svd={best_alpha:.2f}")

    print("\nRefitting on train + validation...")
    train2 = pd.concat([train, val], ignore_index=True)
    pop2 = PopularityRecommender().fit(train2)
    svd2 = SVDRecommender(n_components=40, random_state=42).fit(train2)

    print("Precomputing final test scores...")
    test_cache = precompute_component_scores(pop2, svd2, train2, test, all_items)

    popularity = evaluate_alpha(test_cache, 0.0, k=k)
    svd_only = evaluate_alpha(test_cache, 1.0, k=k)
    tuned = evaluate_alpha(test_cache, best_alpha, k=k)

    for d in [popularity, svd_only, tuned]:
        d["catalog_coverage"] = d.pop("catalog_count") / len(all_items)

    out = {
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
            "users": int(test.user_id.nunique()),
        },
        "validation_alpha_results": val_results,
        "best_alpha_svd": float(best_alpha),
        "test_results": {
            "popularity": popularity,
            "svd": svd_only,
            "tuned_svd_popularity": tuned,
        },
        "svd_explained_variance_ratio": float(svd2.explained_variance_ratio_),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/v2_fast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(out["test_results"]).T.to_csv("results/v2_fast_test_metrics.csv")

    print("\nFINAL TEST RESULTS")
    print(pd.DataFrame(out["test_results"]).T)
    print("\nSaved results/v2_fast_metrics.json")
    return out


if __name__ == "__main__":
    main()
