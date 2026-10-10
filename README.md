# WatchGraph — Personalized Content Recommendation & Ranking Engine

[![Data pipeline and ranking smoke tests](https://github.com/yuktha-code/watchgraph-content-personalisation/actions/workflows/ci.yml/badge.svg)](https://github.com/yuktha-code/watchgraph-content-personalisation/actions/workflows/ci.yml)

WatchGraph is an independent recommender-systems project built around the public **MovieLens 100K** dataset from **GroupLens Research**. The project treats MovieLens as a proxy for OTT user-content interactions because real streaming watch-history data is proprietary.

## What the project implements

- automatic MovieLens 100K download and parsing
- leakage-aware per-user temporal holdout
- Bayesian-shrunk popularity baseline
- mean-centred item-item collaborative filtering
- content-based ranking from movie genre metadata
- hybrid score fusion across popularity, collaborative and content models
- offline ranking evaluation with Precision@K, Recall@K and NDCG@K
- FastAPI endpoint for serving top-K recommendations
- an additional SVD-based ranking experiment with separate temporal train/validation/test splits in `run_v2_fast.py`
- offline ingestion, temporal-split and ranking metric regression tests in `tests/`

## Data

Source: GroupLens MovieLens 100K  
URL used by the ingestion script: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`

The dataset contains 100,000 timestamped ratings from 943 users across 1,682 movies. The project does not claim that MovieLens is JioHotstar, Netflix or any proprietary OTT dataset.

## Validation design

For each sufficiently active user, the latest high-rating interaction is reserved as the held-out relevant item. Earlier interactions stay in training. This produces a chronological leave-one-out evaluation and prevents the model from learning from interactions occurring after the test event.

The ranking task asks: **does the held-out relevant title appear in the user's top-K unseen recommendations?**

## Models

1. **Popularity baseline** — Bayesian-style shrinkage of mean ratings by rating volume.
2. **Item-KNN collaborative filtering** — item-item cosine similarity over mean-centred user ratings.
3. **Content model** — a user preference vector over MovieLens genre features.
4. **Hybrid ranker** — min-max normalised score fusion across popularity, collaborative filtering and content relevance.

## Metrics

- `Precision@10`
- `Recall@10`
- `NDCG@10`

The baseline and hybrid model metrics are in `results/metrics.json` and `results/metrics.csv`. A separate SVD experiment and its temporal validation/test metrics are in `results/v2_fast_metrics.json` and `results/v2_fast_test_metrics.csv`; results are from MovieLens, not production traffic.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

The first run downloads MovieLens automatically.

The included GitHub Actions workflow runs four offline regression tests without downloading MovieLens. To reproduce the full benchmarks locally, run `python run_pipeline.py` and `python run_v2_fast.py` with the documented dataset download.

## API

```powershell
uvicorn api.app:app --reload
```

Then open:

`http://127.0.0.1:8000/recommend/1?k=10`

## Repository structure

```text
WatchGraph/
├── api/
│   └── app.py
├── data/
├── dashboard/
├── notebooks/
├── results/
├── src/
│   ├── baselines.py
│   ├── collaborative_filtering.py
│   ├── content_recommender.py
│   ├── evaluation.py
│   ├── hybrid_ranker.py
│   ├── ingestion.py
│   └── preprocessing.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Interview explanation of the data source

> I used the public MovieLens 100K dataset from GroupLens as a proxy for OTT user-content interactions because real streaming watch-history data is proprietary. The ratings include timestamps, which allowed me to create a temporal holdout rather than a random split. I then evaluated recommendation quality as a ranking problem using Precision@K, Recall@K and NDCG rather than relying only on rating-prediction error.

## Scope and limitations

MovieLens ratings are explicit feedback, whereas modern streaming platforms rely heavily on implicit signals such as impressions, clicks, watch time, completion, skips and search events. The genre-based content model is intentionally lightweight. A production extension would add richer text/image embeddings, implicit-feedback matrix factorisation, candidate-generation/reranking stages, diversity constraints, freshness and online experimentation.

## Technical references

Public recommendation-system repositories and documentation can be useful for studying architecture and evaluation. This repository is structured as an independent implementation rather than a renamed copy of a downloaded project.
