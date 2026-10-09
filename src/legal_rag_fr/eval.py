def recall_at_k(ranked : list[int], relevant: set[int], k: int) -> float:
    """
        Recall at k : | top_k INTERSECT relevant | / | relevant |
        recall = true_positive / (true_positive + false_negative)
    """
    if len(relevant) == 0:
        raise ValueError("Need to input a set of relevant docs")
    hits = set([i for i in ranked[:k] if i in relevant])
    return len(hits) / len(relevant)
    

def mrr(ranked: list[int], relevant: set[int]) -> float:
    """
    MRR (mean reciproqual rank) : 1 / rank of 1st relevant doc 
    """
    if len(relevant) == 0:
        raise ValueError("Need to input a set of relevant docs")
    if not any(x in relevant for x in ranked):
        return 0.0
    else:
        return 1.0 / (min([index for index, val in enumerate(ranked) if val in relevant]) + 1.0) # list.index returns the 0-based index of the 1st occurence of the value inside list