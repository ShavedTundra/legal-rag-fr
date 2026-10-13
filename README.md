# legal-rag-fr

Moteur de recherche hybride sur le droit français.
Lexical (BM25 from scratch) + dense (SentenceTransformers) + fusion RRF + reranking cross-encoder + harness d'éval.

MIT — voir [LICENSE](LICENSE).

## Dev

```bash
uv sync                          # venv + deps (dev group inclus)
uv run pytest                    # tests (dense/rerank skippés sans l'extra)
uv run ruff check .              # lint

uv sync --extra dense            # + SentenceTransformers (dense + rerank)
uv run pytest                    # tests complets (54)
uv run python scripts/run_eval.py  # benchmark BM25 vs Dense vs RRF vs Rerank
```

## Modules

- `legal_rag_fr.bm25` — BM25 from scratch (Okapi, idf Lucene), validé contre `rank_bm25` : même top-5, écart < 1 %. Tokenisation française (accents gérés).
- `legal_rag_fr.similarity` — cosinus et `semantic_search` sur vecteurs denses.
- `legal_rag_fr.chunking` — découpage à fenêtre fixe et par paragraphe, avec recouvrement.
- `legal_rag_fr.dense` — `DenseIndex` : embeddings MiniLM multilingue (384 d, normalisés), chargement eager, garde `ValueError` si `search` avant `fit`.
- `legal_rag_fr.rrf` — fusion Reciprocal Rank Fusion (k = 60) de plusieurs classements.
- `legal_rag_fr.rerank` — `Reranker` : cross-encoder `BAAI/bge-reranker-v2-m3` (multilingue), rescore des paires (requête, document), sortie top-k triée par score décroissant. Modèle anglais (ms-marco) écarté : il démarait les documents pertinents français (recall sous la fusion seule).
- `legal_rag_fr.eval` — métriques `recall_at_k`, `mrr`, `ndcg_at_k`, pures et testées.
- `scripts/run_eval.py` — harness : charge `evals/legifrance_sample.json` + `evals/eval_set_paraphrase.json`, exécute les 4 configurations, affiche métriques agrégées + détail par requête.

## Méthodologie

**Corpus** : 23 articles Légifrance (`evals/legifrance_sample.json`) — 13 Code civil, 5 Code de la consommation, 5 Code de procédure civile — identifiants type `cc-544`.

**Requêtes** : 8 requêtes paraphrasées (`evals/eval_set_paraphrase.json`) : formulation naturelle, vocabulaire distinct du texte cible, donc sans ancrage lexical direct. Chaque requête pointe 1–2 articles pertinents (11 paires au total). Un jeu légué à ancrage lexical (`evals/eval_set.json`, 6 requêtes, corpus de 14 docs) est conservé pour référence.

**Pipeline évalué** :
1. BM25 et dense (MiniLM multilingue, 384 d) classent chacun le corpus, profondeur 10 (`RETRIEVAL_K`).
2. Fusion RRF (k = 60) des deux classements.
3. Rerank : les 20 premiers de la fusion (`n_candidates = 20`) sont rescorés par le cross-encoder bge-reranker-v2-m3, sortie top-5.

**Métriques** : recall@5 (fraction des articles pertinets présents dans le top-5) et MRR (inverse du rang du premier article pertinent). Latence = moyenne de 5 exécutions par requête après échauffement (Apple Silicon, CPU).

## Résultats

| Config | recall@5 | MRR | Latence (ms/requête) |
|---|---|---|---|
| BM25 | 0.438 | 0.260 | 0.03 |
| Dense (MiniLM) | 0.812 | 0.708 | 7.4 |
| Hybride (RRF) | 0.812 | 0.463 | 6.2 |
| RRF + Rerank (bge-v2-m3) | 1.000 | 0.938 | ≈900 |

Chargement des modèles + encodage du corpus : quelques secondes, une fois, hors latence par requête.
La fusion RRF ajoute un coût négligeable (< 1 ms) sur le dense seul ; le rerank domine la latence (~0,9 s/requête, 20 candidats sur CPU).

**Lecture** : sans ancrage lexical, BM25 s'effondre (0.438) — le dense porte le rappel
seul. La fusion RRF conserve le recall du dense mais dégrade le MRR (les candidats
croisés des deux branches repositionnent des distracteurs au-dessus du premier
pertinent). Le reranking cross-encoder rescore les candidats indépendamment et replace
l'article pertinent en tête sur 7 requêtes sur 8 (MRR 0.938), rappel parfait. Les deux
requêtes à double annotation (« prérogative abandonnée », « repousser l'échéance »)
n'atteignent le reranker que grâce à la profondeur de retrieval 10 et à la fenêtre de
20 candidats — à profondeur 5, leurs seconds articles restent hors pool.

**Limites** (honnêteté de micro-benchmark) : 23 documents et 8 requêtes ne mesurent pas
la généralisation. `RETRIEVAL_K = 10` et `n_candidates = 20` ont été réglés sur CE jeu
(choisis à partir de ses deux misses exacts) : le recall 1.000 est donc optimiste, à
re-valider sur un jeu élargi (20–30 requêtes) avant tout autre réglage. Latence rerank
CPU incompatible interactif sans quantification (int8/ONNX).
