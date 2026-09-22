import re
from collections import Counter 
import math
import statistics
## Second round, solo this time
TOKEN_RE = re.compile(r"[a-zàâäéèêëîïôöùûüç']+")

def tokenize(text: str) -> list[str]:
    if not type(text) is str:
        raise TypeError("Input a string")
    lower_cased = text.lower()

    return TOKEN_RE.findall(lower_cased)

class BM25:
    def __init__(self, corpus: list[str], k1: float=1.5, b: float=0.75):
        if not 0.5 <= k1 <= 3.0:
            raise ValueError(f"k1 needs to be between 0.5 and 3, got {k1}")
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(doc) for doc in corpus]
        self.doc_term_freqs = [Counter(token) for token in self.doc_tokens]
        self.doc_lens = [len(doc_token) for doc_token in self.doc_tokens]
        self.N = len(corpus)
        self.avgdl = sum(self.doc_lens) / self.N
        self.df = Counter()
        for tokens in self.doc_tokens:
            self.df.update(set(tokens))
        
        raw_idf = {t: math.log((self.N - f + 0.5) / (f+0.5)) for t,f in self.df.items()}
        avg_idf = statistics.mean(raw_idf.values())
        self.idf = {t: (w if w > 0 else avg_idf*0.25) for t, w in raw_idf.items()}

    def get_scores(self, query: str) -> list[float]:
        scores = []
        for tf_counter, dl in zip(self.doc_term_freqs, self.doc_lens):
            s=0.0
            for term in tokenize(query):
                f = tf_counter.get(term, 0)
                if f==0:
                    continue
                s+= (self.idf[term]*f*(self.k1+1)) / (f+self.k1*(1-self.b + self.b*(dl/self.avgdl)))

            scores.append(s)
        
        return scores