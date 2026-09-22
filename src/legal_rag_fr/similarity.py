import math

def cosine(u: list[float], v: list[float]) -> float:
        if len(u) != len(v):
              raise ValueError("The 2 vectors need to be of same size")
        
        u_norm = math.sqrt(sum(x**2 for x in u))
        v_norm = math.sqrt(sum(y**2 for y in v))

        if u_norm == 0 or v_norm == 0:
              return 0.0

        u_normed = [x / u_norm for x in u]
        v_normed = [y / v_norm for y in v]
        
              
        cosine = [x*y for x,y in zip(u_normed, v_normed)]

        return sum(cosine)

def semantic_search(query: list[float], docs: list[list[float]]) -> list[float]:
    res = []
    for doc in docs:
        s = cosine(query, doc)
        res.append(s)
    
    return res