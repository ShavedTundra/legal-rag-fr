import pytest
pytest.importorskip("sentence_transformers")
from legal_rag_fr.rerank import Reranker # noqa: E402
DOCS = [...]

@pytest.fixture(scope="module")
def reranker():
    return Reranker()
                                                                                                                                                                                                                           
D_PROSPECTION = "La prospection commerciale de consommateurs par des professionnels par message sur un service de communications interpersonnelles, par courrier électronique, est interdite sans consentement préalable."                                                                                                           
D_PROPRIETE = "La propriété est le droit de jouir et disposer des choses de la manière la plus absolue."                                                          
D_VENTE = "La vente est une convention par laquelle l'un s'oblige à livrer une chose, l'autre à la payer."
D_MANDAT = "Le mandat ou procuration est un acte par lequel une personne donne pouvoir à une autre de faire quelque chose en son nom."

def test_rerank_semantic_top1(reranker):  
    result = reranker.rerank("publicité par sms sans accord préalable", [D_PROPRIETE, D_PROSPECTION, D_VENTE], top_k=3)                                               
    assert result[0][0] == 1                                                                                                                                          
    assert result[0][1] == pytest.approx(2.0, abs=3.0)   # valeur pin : 2.007

def test_rerank_rescue(reranker):
    docs = [D_VENTE, D_PROPRIETE, D_PROSPECTION]
    assert reranker.rerank("publicité par sms sans accord préalable", docs, top_k=3)[0][0] == 2

def test_scores_descending(reranker):
    docs = [D_PROPRIETE, D_VENTE, D_VENTE,D_PROSPECTION]
    r = reranker.rerank("vendre un bien", docs, top_k=4)
    scores = [s for _, s in r]
    assert scores == sorted(scores, reverse=True)

def test_top_k_respected_and_clamped(reranker):
    assert len(reranker.rerank("mandat", [D_PROPRIETE, D_MANDAT, D_VENTE], top_k=2)) == 2
    assert len(reranker.rerank("mandat", [D_PROPRIETE, D_MANDAT, D_VENTE], top_k=100)) == 3

def test_indices_are_input_positions(reranker):
    r = reranker.rerank("mandat", [D_PROPRIETE, D_MANDAT, D_VENTE], top_k=3)
    assert sorted(i for i,_ in r) == [0, 1, 2]

def test_control_values(reranker):
    r = reranker.rerank("publicité par sms sans accord préalable", [D_PROPRIETE, D_PROSPECTION], top_k=2)
    d = dict(r)
    assert d[1] > d[0]

def test_empty_docs(reranker):
    assert reranker.rerank("mandat", [], top_k=3) == []

def test_deterministic(reranker):
    q = "publicité par sms sans accord préalable"
    docs = [D_PROPRIETE, D_PROSPECTION, D_VENTE]
    assert reranker.rerank(q, docs, top_k=3) == reranker.rerank(q, docs, top_k=3)
