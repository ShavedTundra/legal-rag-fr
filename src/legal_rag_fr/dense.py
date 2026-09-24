from sentence_transformers import SentenceTransformer
import numpy as np

class DenseIndex:
    def __init__(self, model_name: str="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.matrix = None

    def fit(self, texts: list[list[str]]) -> list[list[float]]:
        M = self.model.encode(texts, normalize_embeddings=True)
        self.matrix = M
    
    def search(self, query: str, k: int) -> list[(int, float)]:
        if self.matrix is None:
            raise ValueError("Index vide, appelle fit() d'abord")
        q = self.model.encode([query], normalize_embeddings=True)
        scores = self.matrix @ q[0]
        idxs = np.argsort(scores)[::-1][:k]
        idx_scores = []
        for idx in idxs:
            idx_scores.append([idx, scores[idx]]) 

        return idx_scores