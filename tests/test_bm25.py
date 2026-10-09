from legal_rag_fr.bm25 import tokenize, BM25
import pytest
from rank_bm25 import BM25Okapi

def test_tokenize_basic():
    assert tokenize("Le contrat de l'employé - durée indéterminée!")== [
        "le", "contrat", "de", "l'employé", "durée", "indéterminée",
    ]

def test_tokenize_empty():
    assert tokenize("!!! - ...") == []

DOCS = [
    "Le contrat de travail",
    "La durée du contrat",
    "Le louange de choses",
]

def test_stats():
    bm = BM25(DOCS)
    assert bm.N == 3
    assert bm.avgdl == 4.0
    assert bm.df["contrat"] == 2 # in D1 & D2
    assert bm.df["louange"] == 1
    assert bm.doc_term_freqs[1]["durée"] == 1
    assert bm.doc_term_freqs[2]["durée"] == 0
    assert bm.doc_lens == [4, 4, 4]

def test_scores():
    bm = BM25(DOCS)
    scores = bm.get_scores("contrat travail")
    assert scores[0] == pytest.approx(0.5534, abs=1e-3)
    assert scores[1] == pytest.approx(0.04257, abs=1e-3)                                                                                                          
    assert scores[2] == pytest.approx(0.0)   

## Out of Bounds K1 tests
def test_k1_out_of_bounds_raises():
    for bad in (0.1, 4.0, 10.0):
        with pytest.raises(ValueError):
            BM25(DOCS, k1=bad)

def test_k1_bounds_edges_accepted():
    BM25(DOCS, k1=0.5)
    BM25(DOCS, k1=3.0)

## Compare with Okapi from rank_bm25 library
CORPUS = [                                                                                                                                                        
       "Le contrat est un accord de volontés entre plusieurs parties destiné à créer des obligations légales.",                                                      
       "Le contrat oblige les parties à exécuter leurs engagements tels qu'ils ont été convenus.",                                                                   
       "Le contrat est formé par la rencontre d'une offre et de l'acceptation de cette offre.",                                                                      
       "Le consentement doit être libre et éclairé; le dol, l'erreur et la violence vicient le contrat.",                                                            
       "La violence physique ou morale entraine la nullité du contrat lorsqu'elle détermine le consentement.",                                                       
       "Le contrat ne peut nuire à des tiers; il ne produit d'effet qu'entre les parties contractantes.",                                                            
       "Les contrats sont synallagmatiques lorsque les parties s'obligent réciproquement les unes envers les autres.",                                               
       "Le contrat à titre gratuit est celui qui procure un avantage à une partie sans rien exiger en retour.",                                                      
       "La bonne foi doit gouverner l'exécution du contrat comme sa formation.",                                                                                     
       "En cas d'inexécution, le créancier peut demander l'exécution forcée ou la résolution du contrat.",                                                           
       "Le présent contrat a pour objet de définir la nature et le volume des prestations de transport que l'opérateur de transport confie de façon régulière et significative au sous-traitant",
       "et de fixer les conditions dans lesquelles ces opérations sont exécutées.", 
       "L'opérateur de transport informe le sous-traitant des changements dans la nature des marchandises transportées quand celles-ci font l'objet d'une réglementation particulière.",
       "Le chiffre d'affaires minimum sur lequel l'opérateur de transport s'engage envers le sous-traitant s'élève à EUR.",                                                                                                          
   ]

QUERIES = [                                                                                                                                                       
       "nullité du consentement",                                                                                                                                    
       "exécution forcée du contrat",                                                                                                                                
       "contrat gratuit",                                                                                                                                            
       "bons père de famille",   # pick queries that hit YOUR 10 articles too                                                                                        
       "effets à l'égard des tiers", 
       "chiffre d'affaires",
       "le créancier",                                                                                                                                
   ]

def top5(scores : list[float]):
    return sorted(range(len(CORPUS)), key= lambda i: (-scores[i], i))[:5]
    

def test_matches_rank_bm25():
    tokenized = [tokenize(doc) for doc in CORPUS]
    ref = BM25Okapi(tokenized, k1=1.5, b=0.75)
    mine = BM25(CORPUS, k1=1.5, b=0.75)
    
    for q in QUERIES:
        ref_scores = ref.get_scores(tokenize(q))
        my_scores = mine.get_scores(q)
    
        assert top5(my_scores) == top5(ref_scores), q
        for i in top5(ref_scores):
            assert abs(my_scores[i] - ref_scores[i]) / max(abs(ref_scores[i]), 1e-9) < 0.01, (q, i)