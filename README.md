# legal-rag-fr

Moteur de recherche hybride sur le droit français.
Lexical (BM25 from scratch) + dense (SentenceTransformers) + fusion RRF + reranking cross-encoder + harness d'éval.

MIT — voir [LICENSE](LICENSE).

[![CI](https://github.com/ShavedTundra/legal-rag-fr/actions/workflows/ci.yml/badge.svg)](https://github.com/ShavedTundra/legal-rag-fr/actions/workflows/ci.yml)

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

**Corpus** : 59 articles Légifrance (`evals/legifrance_sample.json`) — 28 Code civil, 13 Code de la consommation, 10 Code de procédure civile, 8 Code pénal — identifiants type `cc-544`. ~24 articles sans requête associée servent de distracteurs.

**Requêtes** : 35 requêtes paraphrasées (`evals/eval_set_paraphrase.json`) : formulation naturelle, vocabulaire distinct du texte cible — l'absence d'ancrage lexical est vérifiée par un contrôle automatique de recouvrement de tokens (accents neutralisés, mots outils exclus). Chaque requête pointe 1–2 articles pertinents (39 paires au total). Un jeu légué à ancrage lexical (`evals/eval_set.json`, 6 requêtes, corpus de 14 docs) est conservé pour référence.

**Pipeline évalué** :
1. BM25 et dense (MiniLM multilingue, 384 d) classent chacun le corpus, profondeur 10 (`RETRIEVAL_K`).
2. Fusion RRF (k = 60) des deux classements.
3. Rerank : les 20 premiers de la fusion (`n_candidates = 20`) sont rescorés par le cross-encoder bge-reranker-v2-m3, sortie top-5.

**Métriques** : recall@5 (fraction des articles pertinets présents dans le top-5) et MRR (inverse du rang du premier article pertinent). Latence = moyenne de 5 exécutions par requête après échauffement (Apple Silicon, CPU).

## Résultats

| Config | recall@5 | MRR | Latence (ms/requête) |
|---|---|---|---|
| BM25 | 0.157 | 0.136 | 0.1 |
| Dense (MiniLM) | 0.671 | 0.493 | 6.3 |
| Hybride (RRF) | 0.500 | 0.277 | 6.2 |
| RRF + Rerank (bge-v2-m3) | 0.814 | 0.776 | ≈1000 |

Chargement des modèles + encodage du corpus : quelques secondes, une fois, hors latence par requête.
La fusion RRF ajoute un coût négligeable (< 1 ms) sur le dense seul ; le rerank domine la latence (~0,9 s/requête, 20 candidats sur CPU).

**Lecture** : sans ancrage lexical, BM25 s'effondre (0.157 — 20 articles pertinents sur 39 hors top-5). Le dense porte le rappel seul. **Nouveau constat v0.3 : la fusion RRF dégrade ici le dense** (recall 0.500 < 0.671) — fusionner un classement lexical pauvre avec un bon classement dense injecte des distracteurs ; la fusion n'est rentable que si les deux branches sont à peu près comparables. Le reranking cross-encoder rescore les candidats indépendamment : +0.14 recall et MRR 0.493 → 0.776 vs dense seul, article pertinent en tête sur 25 requêtes sur 35. 5 requêtes restent ratées au rerank (ex. définitions très courtes comme le vol, ou paraphrases abstraites « maîtrise d'un patrimoine ») — le pool de 20 candidats ne contient pas toujours l'article pertinent.

**Limites** (honnêteté de micro-benchmark) : 59 documents et 35 requêtes mesurent mieux la généralisation que la v0.2 (23/8) mais restent un micro-benchmark. `RETRIEVAL_K = 10` et `n_candidates = 20` ont été réglés sur le jeu v0.2 (8 requêtes) et NON re-réglés sur la v0.3 — les 5 misses du rerank sont donc partiellement un artefact de ces réglages hérités ; prochaine étape : re-régler sur v0.3 puis re-valider sur un jeu élargi. Latence rerank CPU incompatible interactif sans quantification (int8/ONNX).
