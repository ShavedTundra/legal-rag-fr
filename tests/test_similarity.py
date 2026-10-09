import pytest
from legal_rag_fr.similarity import cosine, semantic_search
import numpy as np

@pytest.mark.parametrize("u,v, expected", [
    ([1, 2, 2], [2, 4, 4], 1.0),
    ([1, 0], [0, 1], 0.0),
    ([3, 4], [4, 3], 0.96),
    ([0,0], [1,2], 0.0),
    ([2*x for x in [2,1]], [3,2], cosine([2,1],[3,2])),
])

def test_hand_values(u, v, expected):
    assert cosine(u, v) == pytest.approx(expected)

def test_scale_invariance():                                                                                                                                      
       u, v = [1, 2, 2], [3, 1, 0]                                                                                                                                   
       assert cosine([2*x for x in u], v) == pytest.approx(cosine(u, v)) 

def test_dim_mismatch_raises():
      with pytest.raises(ValueError):
            cosine([1,2], [5,6,7,8])

DOCS = [                                                                                                                                                          
       [0.2, 0.8, 1.2],   # D1 licenciement                                                                                                                          
       [1.3, 0.4, 0.05],  # D2 CDI                                                                                                                                   
       [0.4, 2.0, 0.2],   # D3 bonne foi (verbeux)                                                                                                                   
       [0.9, 0.1, 1.0],   # D4 résolution                                                                                                                            
       [0.1, 1.1, 0.0],   # D5 durée du travail                                                                                                                      
   ]  

def test_sem_sanction():
    sem_res = semantic_search([0.0, 0.0, 2.0], DOCS)
    ranking = sorted(range(len(sem_res)), key= lambda i: -sem_res[i])
    assert ranking == [0, 3, 2, 1, 4]
    assert sem_res[4] == pytest.approx(0.0)

def test_numpy_faceoff():
    rng = np.random.default_rng(42)
    docs = rng.random((50,50))
    query = rng.random(50)
    for doc in docs:
        ref = np.dot(query, doc) / (np.linalg.norm(query) * np.linalg.norm(doc))
        assert cosine(list(query), list(doc)) == pytest.approx(ref, rel=1e-9)
