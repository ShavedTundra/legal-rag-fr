
def rrf(ranks: list[list[int]], k: int=60) -> dict[str, float]:
    res = dict()
    for rank in ranks:
        for (idx, val) in enumerate(rank):
            if val in res.keys():
                res[val] += 1 / (k + (idx+1))
            else:
                res[val] = 1 / (k + (idx+1))

    return res