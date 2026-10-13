import pytest
from legal_rag_fr.eval import recall_at_k, mrr, ndcg_at_k

@pytest.mark.parametrize("ranked, relevant, k, recall_at_k_expected", [
    ([1, 2, 2], {2}, 3, 1.0),
    ([1, 0], {0,1}, 2, 1.0),
])

def test_hand_values(ranked, relevant, k, recall_at_k_expected):
    assert recall_at_k(ranked, relevant, k) == pytest.approx(recall_at_k_expected)

def test_eval_at_k():
    assert recall_at_k( [2,0,9,1,3], {2,9}, k=5) == 1.0


def test_mrr():
    assert mrr([5, 0, 2], {0,2}) == 0.5

def test_mrr_no_relevant():
     assert mrr([0,1], {9}) == 0.0


def test_eval_nul_set():
      with pytest.raises(ValueError):
            recall_at_k([1,2], {}, k=3)

def test_mrr_nul_set():
      with pytest.raises(ValueError):
            mrr([1,2], {})

def test_ndcg_at_k_basics():
     assert ndcg_at_k(['a', 'b', 'x'], {'a', 'b'}, k=3) == 1.0
     assert ndcg_at_k(['c', 'a'], {'a'}, k=5) == pytest.approx(0.6309, rel=1e-4)
     assert ndcg_at_k(['a', 'c', 'a'], {'a'}, k=3) == 1.0
     assert ndcg_at_k(['x', 'a', 'b'], {'a', 'b'}, k=3) == pytest.approx(0.6934, rel=1e-4)
     assert ndcg_at_k([], {'b'}, k=3) == 0.0
     assert ndcg_at_k(['a'], {}, k=3) == 0.0
     assert ndcg_at_k(['c','d','a'], {'a'}, k=2) == 0.0

def test_ndcg_monotone():
     assert ndcg_at_k(['h', 'e', 'd', 'a', 'b', 'x'], {'a', 'b'}, k=3) < ndcg_at_k(['h', 'e', 'a', 'b', 'x', 'd'], {'a', 'b'}, k=3)
     assert ndcg_at_k(['d', 'a', 'b', 'x'], {'a', 'b'}, k=3) < ndcg_at_k(['a', 'b', 'x', 'd'], {'a', 'b'}, k=3)
          