"""Offline regression tests for public-dataset ingestion and ranking logic."""
import math
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.ingestion import load_movielens
from src.preprocessing import temporal_leave_one_out
from src.v2_fast_pipeline import temporal_train_val_test
from src.evaluation import ranking_metrics

class PipelineSmokeTests(unittest.TestCase):
    def test_local_movielens_parse_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "ml-100k"
            directory.mkdir()
            (directory / "u.data").write_text("1\t10\t5\t876000000\n2\t20\t3\t876000100\n", encoding="utf-8")
            movie_record = ["10", "Example Title", "01-Jan-1997", "", "http://example.invalid"] + ["0"] * 19
            (directory / "u.item").write_text("|".join(movie_record) + "\n", encoding="latin-1")
            ratings, movies = load_movielens(tmp)
            self.assertEqual(len(ratings), 2)
            self.assertEqual(len(movies), 1)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(ratings["timestamp"]))
            self.assertEqual(int(movies.loc[0, "year"]), 1997)

    def test_temporal_holdout_does_not_include_future_train_events(self):
        df = pd.DataFrame([
            {"user_id": uid, "item_id": n + uid * 100, "rating": 5, "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=n)}
            for uid in (1, 2) for n in range(8)
        ])
        train, test = temporal_leave_one_out(df)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(train), 14)
        self.assertTrue(all(test.groupby("user_id").size() == 1))
        for uid in (1, 2):
            self.assertLess(train.loc[train.user_id == uid, "timestamp"].max(),
                            test.loc[test.user_id == uid, "timestamp"].iloc[0])

    def test_train_validation_test_are_chronological(self):
        df = pd.DataFrame([
            {"user_id": uid, "item_id": n + uid * 100, "rating": 5, "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=n)}
            for uid in (1, 2) for n in range(8)
        ])
        train, val, test = temporal_train_val_test(df)
        self.assertEqual((len(train), len(val), len(test)), (12, 2, 2))
        for uid in (1, 2):
            t = train.loc[train.user_id == uid, "timestamp"].max()
            v = val.loc[val.user_id == uid, "timestamp"].iloc[0]
            z = test.loc[test.user_id == uid, "timestamp"].iloc[0]
            self.assertLess(t, v)
            self.assertLess(v, z)

    def test_ranking_metrics(self):
        precision, recall, ndcg = ranking_metrics([1, 2, 3], {2}, k=3)
        self.assertAlmostEqual(precision, 1 / 3)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(ndcg, 1 / math.log2(3))

if __name__ == "__main__":
    unittest.main()
