# legal-rag-fr

Moteur de recherche hybride sur le droit français — miniature type Doctrine.
Lexical (BM25 from scratch) + dense (SentenceTransformers) + fusion RRF + harness d'éval.

## Dev

```bash
uv sync                # venv + deps (dev group inclus)
uv run pytest          # tests
uv run ruff check .    # lint
```

## Modules

- `legal_rag_fr.bm25` — BM25 from scratch, validé contre `rank_bm25` (même top-5, écart < 1 %).

## Métriques (à remplir — W1-D7)

| Config | recall@5 | MRR | Latence (ms) |
|---|---|---|---|
| BM25 | — | — | — |
| Dense | — | — | — |
| Hybride (RRF) | — | — | — |
| Hybride + rerank | — | — | — |
