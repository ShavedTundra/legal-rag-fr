from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str= "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, docs: list[str], top_k: int=5) -> list[tuple[int, float]]:
        scores = self.model.predict([(query, d) for d in docs])
        order = sorted(range(len(docs)), key= lambda i: (-float(scores[i]), i))[:top_k]
        return [(i, float(scores[i])) for i in order]

