from __future__ import annotations

from functools import lru_cache
from fastapi import FastAPI, HTTPException

from src.ingestion import load_movielens
from src.preprocessing import temporal_leave_one_out, seen_items
from src.baselines import PopularityRecommender
from src.collaborative_filtering import ItemKNNRecommender
from src.content_recommender import ContentRecommender
from src.hybrid_ranker import HybridRanker

app = FastAPI(title="WatchGraph Recommendation API", version="1.0")


@lru_cache(maxsize=1)
def build_system():
    ratings, movies = load_movielens()
    train, _ = temporal_leave_one_out(ratings)
    pop = PopularityRecommender().fit(train)
    cf = ItemKNNRecommender(top_neighbors=40).fit(train)
    content = ContentRecommender().fit(train, movies)
    hybrid = HybridRanker(pop, cf, content)
    return train, movies, hybrid


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    train, movies, hybrid = build_system()
    users = set(train.user_id.astype(int))
    if user_id not in users:
        raise HTTPException(status_code=404, detail="Unknown MovieLens user_id")

    seen = seen_items(train).get(user_id, set())
    candidates = [int(i) for i in movies.item_id if int(i) not in seen]
    ranked = hybrid.recommend(user_id, candidates, k=k)

    lookup = movies.set_index("item_id")
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "item_id": int(i),
                "title": str(lookup.loc[i, "title"]),
                "year": None if lookup.loc[i, "year"] != lookup.loc[i, "year"] else int(lookup.loc[i, "year"]),
            }
            for i in ranked
        ],
    }
