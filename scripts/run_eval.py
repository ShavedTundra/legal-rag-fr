import json, time
from pathlib import Path
from statistics import fmean, mean

from legal_rag_fr.bm25 import BM25
from legal_rag_fr.rrf import rrf
from legal_rag_fr.eval import recall_at_k, mrr

try:
    from legal_rag_fr.dense import DenseIndex
    DENSE_OK = True
except ImportError:
    DENSE_OK = False

K = 5
DATA = Path(__file__).resolve().parents[1] / "evals" / "eval_set.json"

with open('evals/eval_set.json') as f:
    data = json.load(f)


def top_k(scores, k=K):
    return sorted(range(len(scores)), key= lambda i: (-scores[i], i))[:k]

def make_bm25(corpus) -> list[float]:
    bm25 = BM25(corpus)
    def rank(query):
        return top_k(bm25.get_scores(query))

    return rank


def make_dense(corpus):
    idx = DenseIndex()
    idx.fit(corpus)
    def rank(query):
        res = idx.search(query, k=K)
        return [int(i) for i,_ in res]
    return rank

def make_rrf(bm_rank, dense_rank):
    def rank(query): 
        fused = rrf([bm_rank(query), dense_rank(query)], k=60)
        return [int(i) for i, _ in sorted(fused.items(), key=lambda x: -x[1])][:K]
    return rank


corpus = data["corpus"]
queries = data["queries"]
bm_rank = make_bm25(corpus)
rankers = {"BM25": bm_rank}

if DENSE_OK:
    t0 = time.perf_counter()
    dense_rank = make_dense(corpus)
    print(f"chargement MiniLM + encodage corpus : {time.perf_counter() - t0:.1f} s")

    rankers["Dense"] = dense_rank
    rankers["RRF"] = make_rrf(bm_rank, dense_rank)
else:
    print("sentence-transformers absent - BM25 seulement (uv sync --extra dense)")

REPS = 5

for name, rank in rankers.items():
    print(f"\n{name}")
    for _ in range(5):
        rank(queries[0]["query"])
    recalls, rrs, times = [], [], []
    for q in queries:
        samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ranked = rank(q["query"])
            samples.append(time.perf_counter() - t0) #seconds, float

        times.append(mean(samples))
        relevant = set(q["relevant"])
        recalls.append(recall_at_k(ranked, relevant, k=K))
        rrs.append(mrr(ranked, relevant))
        print(f"  {q["query"]:<28} top5={ranked}  "
              f"recall={recalls[-1]:.2f} rr={rrs[-1]:.2f}")

    print(f"{name: <5} recall@5={fmean(recalls):.3f}  "
          f"MRR={fmean(rrs):.3f}  {fmean(times)*1000:.2f} ms/query")
