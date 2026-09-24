"""Tests W1-D4 — rrf (lessons/0004, exercice §6).

Contrat pinné :
  rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]
  score(d) = somme sur les listes de 1 / (k + rang(d)), rang à partir de 1

Valeur de contrôle (calculable de tête, leçon §4) :
  rrf([["A","B","C","D"], ["C","A","E"]], k=60)
    A = 1/61 + 1/62 ≈ 0.032522   C = 1/61 + 1/63 ≈ 0.032266
    B = 1/62          D = 1/64          E = 1/63
  ordre fusionné : A > C > B > E > D
"""

import pytest

from legal_rag_fr.rrf import rrf


def test_hand_value_lesson():
    scores = rrf([["A", "B", "C", "D"], ["C", "A", "E"]], k=60)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["C"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["B"] == pytest.approx(1 / 62)
    assert scores["D"] == pytest.approx(1 / 64)
    assert scores["E"] == pytest.approx(1 / 63)
    assert set(scores) == {"A", "B", "C", "D", "E"}


def test_fused_ordering():
    scores = rrf([["A", "B", "C", "D"], ["C", "A", "E"]], k=60)
    order = [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    assert order == ["A", "C", "B", "E", "D"]


def test_single_list():
    scores = rrf([["x", "y"]])
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["y"] == pytest.approx(1 / 62)


def test_k_parameter_changes_weights():
    scores = rrf([["x", "y"]], k=1)
    assert scores["x"] == pytest.approx(1 / 2)
    assert scores["y"] == pytest.approx(1 / 3)


def test_empty_ranking_gives_empty():
    assert rrf([[]]) == {}
