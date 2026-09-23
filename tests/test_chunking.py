import copy                                                                                                                                                      
import pytest                                                                                                                                                    
from legal_rag_fr.chunking import chunk_fixed, chunk_by_paragraph                                                                                                             
                                                                                                                                                                
# -- chunk_fixed ---------------------------------------------------------                                                                                       
                                                                                                                                                                
def test_control_value():                                                                                                                                        
    """Lesson §5: 9 mots, size 5, overlap 2 → 3 chunks."""                                                                                                       
    words = "le contrat de travail est exécuté de bonne foi".split()                                                                                             
    chunks = chunk_fixed(words, 5, 2)                                                                                                                            
    assert chunks == [                                                                                                                                           
        ["le", "contrat", "de", "travail", "est"],   # chunk 1: mots 0-4                                                                                                                               
        ["travail", "est", "exécuté", "de", "bonne"],   # chunk 2: mots 3-7                                                                                                                               
        ["de", "bonne", "foi"]   # chunk 3: mots 6-8                                                                                                                               
    ]                                                                                                                                                            
                                                                                                                                                                
def test_guard_rejects_bad_overlap():                                                                                                                            
    """overlap >= size → ValueError (comme k₁ en L1)."""                                                                                                         
    with pytest.raises(ValueError):                                                                                                                              
        chunk_fixed("a b c".split(), size=1, overlap=2)   # pick a bad pair                                                                                  
                                                                                                                                                                
def test_overlap_property():                                                                                                                                     
    """Propriété §5 : les `overlap` derniers mots du chunk i                                                                                                     
    = les `overlap` premiers mots du chunk i+1."""                                                                                                               
    words = "un préavis est dû sauf faute grave du salarié en cas de litige".split()  # 12 mots                                                                  
    overlap = 2                                                                                                                                                  
    chunks = chunk_fixed(words, size=4, overlap=overlap)                                                                                                         
    for i in range(len(chunks) - 1):                                                                                                                             
        assert chunks[i][2] == chunks[i+1][0]                                                                                                                
                                                                                                                                                                
# -- chunk_by_paragraph --------------------------------------------------                                                                                       
                                                                                                                                                                
P1 = "le contrat est exécuté de bonne foi".split()          # 7                                                                                                  
P2 = "un préavis est dû sauf faute grave".split()           # 7                                                                                                  
P3 = "le juge apprécie la bonne foi des parties".split()    # 8                                                                                                  
                                                                                                                                                                
def test_para_merge():                                                                                                                                           
    """p1+p2 fusionnent (14 ≤ 14), p3 seul."""                                                                                                                   
    chunks = chunk_by_paragraph([P1, P2, P3], size=14)                                                                                                          
    assert len(chunks) == 2                                                                                                                                   
    assert chunks[0] == P1 + P2          # fusion, ordre gardé                                                                                                   
    assert chunks[1] == P3                                                                                                                                      
                                                                                                                                                                
def test_para_zero_loss():                                                                                                                                       
    """Union ordonnée des chunks = texte original, zéro duplication."""                                                                                          
    chunks = chunk_by_paragraph([P1, P2, P3], size=10)                                                                                                           
    flat = [w for c in chunks for w in c]                                                                                                                        
    assert flat == P1 + P2 + P3                                                                                                                                           
                                                                                                                                                                
def test_para_oversized_emitted_whole():                                                                                                                         
    """Alinéa seul > size → rendu tel quel, non coupé."""                                                                                                        
    p4 = "la rupture du contrat obéit à des règles strictes et le salarié peut contester".split()  # 14                                                          
    chunks = chunk_by_paragraph([p4], size=10)                                                                                                                   
    assert chunks == [p4]               # et PAS de chunk vide avant                                                                                             
                                                                                                                                                                
def test_para_no_empty_chunks():                                                                                                                                 
    """Régression bug 1 : jamais de chunk vide dans out."""                                                                                                      
    p4 = "a b c d e f g h i j k l m n".split()  # 14 mots                                                                                                        
    chunks = chunk_by_paragraph([p4], size=5)                                                                                                                    
    assert all(c != [] for c in chunks)                                                                                                                              
                                                                                                                                                                
def test_para_input_not_mutated():                                                                                                                               
    """Régression bug 2 : l'entrée reste intacte après l'appel."""                                                                                               
    paragraphs = [["a","b","c","d"], ["e","f","g","h"], ["i"]]                                                                                                   
    snapshot = copy.deepcopy(paragraphs)                                                                                                                         
    chunk_by_paragraph(paragraphs, size=5)                                                                                                                       
    assert paragraphs == snapshot