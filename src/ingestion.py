from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download_movielens(data_dir: str | Path = "data") -> Path:
    """Download and extract MovieLens 100K from GroupLens if it is absent."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "ml-100k"
    if extracted.exists():
        return extracted

    archive = data_dir / "ml-100k.zip"
    if not archive.exists():
        print(f"Downloading MovieLens 100K from {MOVIELENS_URL}")
        urlretrieve(MOVIELENS_URL, archive)

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(data_dir)
    return extracted


def load_movielens(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movie metadata into tidy DataFrames."""
    base = download_movielens(data_dir)

    ratings = pd.read_csv(
        base / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)

    movie_cols = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url"
    ] + GENRES
    movies = pd.read_csv(
        base / "u.item",
        sep="|",
        names=movie_cols,
        encoding="latin-1",
        engine="python",
    )
    movies["year"] = pd.to_numeric(
        movies["release_date"].astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
    return ratings, movies


if __name__ == "__main__":
    r, m = load_movielens()
    print(f"ratings={len(r):,}, users={r.user_id.nunique():,}, movies={m.item_id.nunique():,}")
