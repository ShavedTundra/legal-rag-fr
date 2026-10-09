"""Tests W1-D4 — DenseIndex (lessons/0004, exercice §6).

Contrat pinné :
  - fit(texts) -> M ndarray (n, 384) float32, lignes de norme 1
  - search(query, k=5) -> list[(index, score)] décroissant, k borné à n
  - search avant fit -> ValueError (garde explicite, pas un side-effect)
  - modèle chargé eager dans __init__

Valeurs de contrôle vérifiées au modèle réel le 2026-09-23 :
  - query maladie -> doc0 top-1 à 0.6701
  - query exacte corpus[2] -> top=2, score 0.9999998
  - query licenciement (riche) -> doc3 (rupture) 0.5747 > doc2 (marché)
"""

import numpy as np
import pytest
pytest.importorskip("sentence_transformers")

from legal_rag_fr.dense import DenseIndex  # noqa: E402

CORPUS = [
    "Le contrat de travail est suspendu pendant la maladie.",            # 0
    "La maladie suspend le contrat de travail du salarié.",              # 1
    "Les pommes coûtent trois euros le kilo au marché.",                 # 2
    "La rupture du contrat à l'initiative de l'employeur obéit à des règles strictes.",  # 3
    "Le préavis est dû sauf faute grave du salarié.",                    # 4
    "Le conseil de prud'hommes est compétent en cas de litige.",         # 5
]

Q_MALADIE = "que devient le contrat quand le salarié est malade ?"
Q_LICENCIEMENT = "Un employeur souhaite licencier un salarié, quelle règle s'applique ?"


@pytest.fixture(scope="module")
def index():
    idx = DenseIndex()
    idx.fit(CORPUS)
    return idx


def test_fit_shape_norms_dtype(index):
    M = index.matrix
    assert isinstance(M, np.ndarray)
    assert M.shape == (6, 384)
    assert M.dtype == np.float32
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-4)


def test_search_semantic_maladie_top1(index):
    results = index.search(Q_MALADIE, k=1)
    assert len(results) == 1
    i, s = results[0]
    assert i == 0
    assert s > 0.6  # vérifié : 0.6701


def test_search_mismatch_lexical(index):
    """Le dense retrouve la rupture (doc3, zéro mot partagé avec la query
    reformulée) et la distingue nettement du hors-sujet (doc2 marché)."""
    top3 = index.search(Q_LICENCIEMENT, k=3)
    ranks = [i for i, _ in top3]
    assert 3 in ranks            # doc3 dans le top-3 (vérifié : 2e à 0.5747)
    all_scores = dict(index.search(Q_LICENCIEMENT, k=6))
    assert all_scores[3] > all_scores[2]  # rupture > prix des pommes


def test_search_exact_query_is_one(index):
    """Requête = document mot pour mot : cosinus(x, x) = 1, top-1 garanti."""
    results = index.search(CORPUS[2], k=1)
    i, s = results[0]
    assert i == 2
    assert s > 0.999


def test_search_k_results_descending(index):
    results = index.search(Q_MALADIE, k=4)
    assert len(results) == 4
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_k_clamped_to_n(index):
    results = index.search(Q_MALADIE, k=100)
    assert len(results) == 6


def test_search_before_fit_raises():
    idx = DenseIndex()
    with pytest.raises(ValueError):
        idx.search("n'importe quoi", k=2)


def test_search_deterministic(index):
    r1 = index.search(Q_MALADIE, k=6)
    r2 = index.search(Q_MALADIE, k=6)
    assert r1 == r2
