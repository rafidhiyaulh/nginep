import math

import pytest

from nginep.metrics import aspect_report, mean_ndcg_at_k, ndcg_at_k


class TestNdcg:
    def test_perfect_ranking_is_1(self):
        # already sorted descending -> actual DCG == ideal DCG
        assert ndcg_at_k([3, 2, 1, 0], k=4) == pytest.approx(1.0)

    def test_worst_ranking_is_less_than_1(self):
        assert ndcg_at_k([0, 1, 2, 3], k=4) < 1.0

    def test_empty_list_is_zero(self):
        assert ndcg_at_k([], k=5) == 0.0

    def test_all_zero_relevance_is_zero(self):
        # IDCG is 0 too -> defined as 0, not a divide-by-zero crash
        assert ndcg_at_k([0, 0, 0], k=3) == 0.0

    def test_k_smaller_than_list_only_considers_top_k(self):
        # only the top-1 matters at k=1, and it's already the best item
        assert ndcg_at_k([5, 0, 0], k=1) == pytest.approx(1.0)

    def test_matches_hand_computed_value(self):
        # relevances in ranked order: [3, 2, 3, 0, 1, 2] (a textbook NDCG example)
        rels = [3, 2, 3, 0, 1, 2]
        dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)))
        assert ndcg_at_k(rels, k=6) == pytest.approx(dcg / idcg)


class TestMeanNdcg:
    def test_averages_across_queries(self):
        perfect = [3, 2, 1]
        worst = [1, 2, 3]
        mean = mean_ndcg_at_k([perfect, worst], k=3)
        expected = (ndcg_at_k(perfect, 3) + ndcg_at_k(worst, 3)) / 2
        assert mean == pytest.approx(expected)

    def test_empty_input_is_zero(self):
        assert mean_ndcg_at_k([], k=5) == 0.0


class TestAspectReport:
    def test_perfect_predictions_give_f1_of_1(self):
        y = ["pos", "neg", "neut", "pos", "neg"]
        result = aspect_report(y, y)
        assert result["macro_f1"] == pytest.approx(1.0)

    def test_all_wrong_gives_low_f1(self):
        y_true = ["pos", "pos", "pos"]
        y_pred = ["neg", "neg", "neg"]
        result = aspect_report(y_true, y_pred, labels=["pos", "neg", "neut"])
        assert result["macro_f1"] < 0.5

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            aspect_report(["pos", "neg"], ["pos"])

    def test_report_has_per_class_breakdown(self):
        result = aspect_report(["pos", "neg"], ["pos", "pos"], labels=["pos", "neg"])
        assert "pos" in result["report"]
        assert "neg" in result["report"]
