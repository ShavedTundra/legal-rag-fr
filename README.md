# legal-rag-fr

Moteur de recherche hybride sur le droit français — miniature type Doctrine.
Lexical (BM25 from scratch) + dense (SentenceTransformers) + fusion RRF + harness d'éval.

MIT — voir [LICENSE](LICENSE).

## Dev

```bash
uv sync                          # venv + deps (dev group inclus)
uv run pytest                    # tests (dense skippé sans l'extra)
uv run ruff check .              # lint

uv sync --extra dense            # + SentenceTransformers (recherche dense)
uv run pytest                    # tests complets (44)
uv run python scripts/run_eval.py  # benchmark BM25 vs Dense vs RRF
```

## Modules

- `legal_rag_fr.bm25` — BM25 from scratch (Okapi, idf Lucene), validé contre `rank_bm25` : même top-5, écart < 1 %. Tokenisation française (accents gérés).
- `legal_rag_fr.similarity` — cosinus et `semantic_search` sur vecteurs denses.
- `legal_rag_fr.chunking` — découpage à fenêtre fixe et par paragraphe, avec recouvrement.
- `legal_rag_fr.dense` — `DenseIndex` : embeddings MiniLM (384 d, normalisés), chargement eager, garde `ValueError` si `search` avant `fit`.
- `legal_rag_fr.rrf` — fusion Reciprocal Rank Fusion (k = 60) de plusieurs classements.
- `legal_rag_fr.eval` — métriques `recall_at_k` et `mrr`, pures et testées.
- `scripts/run_eval.py` — harness : charge `evals/eval_set.json` (corpus de 14 articles — 10 Code civil + 4 contrat de transport — et 6 requêtes labellisées), exécute les 3 configurations, affiche métriques agrégées + détail par requête.

## Métriques

Corpus de 14 documents, 6 requêtes annotées à la main (jugement de pertinence sur le texte seul).
Latence = médiane de 5 exécutions par requête, après échauffement (Apple Silicon, CPU).

| Config | recall@5 | MRR | Latence (ms/requête) |
|---|---|---|---|
| BM25 | 1.000 | 1.000 | 0.01 |
| Dense (MiniLM) | 1.000 | 0.917 | 5.70 |
| Hybride (RRF) | 1.000 | 1.000 | ≈5.5 |

Chargement MiniLM + encodage du corpus : ~3,5 s (une fois, hors latence par requête).
La fusion RRF ajoute un coût négligeable (< 0,3 ms) sur le dense seul.

**Lecture** : sur ce corpus, chaque requête a un ancrage lexical direct — BM25 est donc
difficile à battre en rappel. Le dense trébuche sur « effets à l'égard des tiers »
(il classe l'article transport en tête, rr = 0,50) ; la fusion RRF rétablit le bon
article en position 1 — l'hybride corrige la miss du dense au prix de la latence du dense.

**Limites** (honnêteté de micro-benchmark) : 14 documents et 6 requêtes ne mesurent pas
la généralisation ; les requêtes paraphrasées (sans recouvrement lexical) mettraient en
évidence l'avantage du dense. Micro-benchmark indicatif, pas une revendication de performance.
