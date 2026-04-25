# RAG Strategies — Field Survey + What This Repo Ships

> If you're looking for *every* RAG technique implemented as a tutorial, the canonical reference is **[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)** (42+ techniques, MIT, Jupyter notebooks). This doc takes a different cut: a survey of the field with explicit "do we ship this and why," plus a comparative evaluation of the strategies we *do* integrate into our self-hosted production pipeline.

## Two orthogonal axes

The retrieval pipeline has two independent knobs:

| Axis | Env var | Values | What it controls |
|---|---|---|---|
| Retrieval mode | `RETRIEVAL_MODE` | `dense` / `bm25` / `hybrid` | The *algorithm* used to fetch candidates from the index |
| RAG strategy | `RAG_STRATEGY` | `direct` / `hyde` / `multi_query` / `step_back` | How the user's query is *transformed and orchestrated* before retrieval |

A query with `RAG_STRATEGY=hyde` and `RETRIEVAL_MODE=hybrid` will first generate a hypothetical answer (HyDE), then run hybrid (BM25+dense+RRF) on it. Strategies and modes compose freely.

## Strategies shipped in this repo

Each ships as a single module under `src/enterprise_rag/retrieval/strategies/` exposing `NAME` and `retrieve(query) -> list[dict]`. They share `_rerank_and_tag()` (cross-encoder rerank against the *original* query) so the final passages always answer what the user actually asked.

### `direct` — naive RAG (baseline)
Retrieve with the user's query, rerank, return. The classic RAG recipe. Fast, no extra LLM calls, decent baseline. Ship as default.

### `hyde` — Hypothetical Document Embeddings
**Paper:** [Gao et al. 2022](https://arxiv.org/abs/2212.10496) — "Precise Zero-Shot Dense Retrieval without Relevance Labels"

**Mechanism.** LLM generates a plausible *answer* to the question. We embed and retrieve based on that hypothetical answer, then rerank with the original query. Bridges the question/answer representation gap in dense retrieval — dense embeddings tend to put answers close to other answers, not to questions.

**When it helps.** Dense-heavy retrieval, especially when the corpus is densely written prose (papers, manuals) and questions are abstract. Less effective for keyword-driven corpora where BM25 already does the work.

**Cost.** +1 LLM call per query (~200 tokens output). With our self-hosted Llama 3.1 8B FP8 + spec decoding this is ~1.5s.

### `multi_query` — Multi-Query / RAG-Fusion
**Sources:** [LangChain MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/), [RAG-Fusion (Raudaschl 2023)](https://github.com/Raudaschl/rag-fusion)

**Mechanism.** Generate N (default 4) paraphrases of the query, retrieve passages for each, fuse rankings with Reciprocal Rank Fusion (k=60). The original query is also included to preserve recall on exact phrasing. Final fused list goes through cross-encoder rerank.

**When it helps.** Vocabulary mismatch — when documents use different terminology than the user. Broad/ambiguous questions benefit from the variant coverage.

**Cost.** +1 LLM call (~256 tokens) for variant generation, then 5 retrievals instead of 1. Mostly a parallelizable cost in retrieval; LLM call is the bottleneck (~2s).

### `step_back` — Step-Back Prompting
**Paper:** [Zheng et al. 2023](https://arxiv.org/abs/2310.06117) — "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"

**Mechanism.** Generate a more abstract version of the question (the "step-back" question), retrieve for both abstract and original, RRF-fuse. The abstract retrieval pulls in higher-level context (definitions, principles); the original anchors specifics.

**When it helps.** Multi-hop or context-dependent questions where the literal answer requires background knowledge. Less impact on direct factual lookups.

**Cost.** +1 LLM call (~128 tokens), 2 retrievals.

## Strategies referenced but *not* shipped

These are documented for completeness — when each fits, what it would take to add. We deferred them because either (a) they require infra we don't have (web search, a fine-tuned model), (b) they're a different category of system entirely (graph DB), or (c) the marginal value vs the cost wasn't worth this round.

### `crag` — Corrective RAG
**Paper:** [Yan et al. 2024](https://arxiv.org/abs/2401.15884)

A retrieval-evaluator scores retrieved passages; if quality is poor, the system rewrites the query and falls back to web search to augment. Three states: *correct* (use as-is), *incorrect* (web fallback), *ambiguous* (combine).

**Why deferred.** Web-search fallback requires a search API integration (Tavily/Bing/SerpAPI) that's a meaningful addition to the dependency surface and operational story. Worth adding once we have a use case where the corpus is incomplete by design.

### `self_rag` — Self-RAG
**Paper:** [Asai et al. 2023](https://arxiv.org/abs/2310.11511)

The model is *fine-tuned* to emit reflection tokens (`[Retrieve]`, `[Relevant]`, `[Supported]`, `[Useful]`) that adaptively decide when to retrieve and how to weigh retrieved evidence. Strong on selective retrieval — short factual answers don't pay the retrieval tax.

**Why deferred.** Requires either a Self-RAG-fine-tuned model checkpoint or substantial fine-tuning work. We're not running a custom-trained model; integrating a third-party Self-RAG checkpoint is a separate decision.

### `adaptive_rag` — Adaptive RAG
**Paper:** [Jeong et al. 2024](https://arxiv.org/abs/2403.14403)

A small classifier picks one of {no-retrieval, single-step, multi-step} based on query complexity. Routes simple questions through cheap paths and complex multi-hop questions through iterative retrieval.

**Why deferred.** Worth adding once we have the iterative-retrieval strategy below — adaptive routing presupposes multiple strategies to route between. Our current set is too small to benefit.

### `flare` — Forward-Looking Active REtrieval
**Paper:** [Jiang et al. 2023](https://arxiv.org/abs/2305.06983)

The model generates the answer one sentence at a time; for each sentence, if it's low-confidence, the system retrieves additional context before continuing. Iterative retrieval interleaved with generation.

**Why deferred.** Requires token-level confidence (logprobs) and fine-grained streaming control inside the generation loop. Fits a different pipeline architecture than ours; sizable refactor.

### `graphrag` — GraphRAG
**Paper / system:** [Microsoft Research, 2024](https://github.com/microsoft/graphrag)

Build a knowledge graph from the corpus offline (entities + relations + community summaries), retrieve via graph traversal at query time. Strong on holistic "summarize the corpus" questions that pure embedding retrieval can't answer.

**Why deferred.** Graph construction is a *separate* offline pipeline — entity extraction, deduplication, community detection. Different category of system; would warrant its own codebase rather than a strategy plug-in.

### `raptor` — RAPTOR
**Paper:** [Sarthi et al. 2024](https://arxiv.org/abs/2401.18059)

Hierarchical clustering of chunks at index time produces a tree of summaries at multiple abstraction levels. Retrieval can pull from any level — leaf chunks for specifics, cluster summaries for overviews.

**Why deferred.** Index-time technique that requires re-indexing all documents. Pairs naturally with very large or hierarchical corpora; our current focus is single-doc RAG quality.

### `contextual_retrieval` — Contextual Retrieval (Anthropic)
**Source:** [Anthropic blog, 2024](https://www.anthropic.com/news/contextual-retrieval)

For each chunk, prepend a short LLM-generated context describing the chunk's place in the document (~50 tokens) before embedding. Improves retrieval recall by ~50% on Anthropic's evals.

**Why deferred (but easy follow-up).** Cheap to implement — modify `ingestion/parser.py` to call the LLM during chunking and stash the context in `chunk.metadata`. Mainly held back to keep this round's scope tight.

### `parent_document` — Parent-Document Retriever
**Source:** [LangChain pattern](https://python.langchain.com/docs/how_to/parent_document_retriever/)

Index small chunks for precise retrieval, but return their parent (full section / full doc) as context to the LLM. Best of both: tight match scores from small chunks, full context for the LLM.

**Why deferred.** Requires storing parent-child relationships in payload; small refactor. Easy follow-up.

### `colbert` / late-interaction retrieval
**Paper:** [Khattab & Zaharia 2020](https://arxiv.org/abs/2004.12832), tooling: [RAGatouille](https://github.com/bclavie/RAGatouille)

Token-level interaction between query and document tokens (each token has its own vector; matching uses MaxSim). Higher quality than single-vector dense retrieval, slower.

**Why deferred.** Requires a different vector store (ColBERT-aware); BGE-M3 already includes a multi-vector mode but Qdrant indexing for that is non-trivial. Worth revisiting when retrieval recall is the bottleneck.

### `self_query` — Self-Querying Retriever
**Source:** [LangChain pattern](https://python.langchain.com/docs/how_to/self_query/)

LLM extracts structured metadata filters from a natural-language query ("papers by Smith from 2023 about RAG" → `{author: "Smith", year: 2023, topic: "RAG"}`). Filters are applied to the vector search.

**Why deferred.** Requires a metadata schema for the indexed corpus. Our chunks currently store `source`, `page`, `section_path` — useful for citations but not rich enough for self-querying to shine. Add this when we ingest corpora with strong structured metadata.

## Other techniques worth knowing

| Technique | One-liner | Reference |
|---|---|---|
| Reliable RAG | Add a verification pass that checks each citation actually supports the answer | [NirDiamant](https://github.com/NirDiamant/RAG_Techniques) |
| Proposition Chunking | Split documents into atomic factual propositions (sentence-level) before embedding | Chen et al. 2023 |
| Semantic Chunking | Embedding-similarity-based chunk boundaries instead of fixed sizes | Greg Kamradt's pattern |
| Hierarchical Indices | Two-level index: abstract summaries + leaf chunks; query routes to summary first | various |
| Reranking with Cohere | Drop-in cross-encoder reranker via Cohere Rerank API | [Cohere](https://cohere.com/rerank) |
| Reciprocal Rank Fusion | Score = `Σ 1/(60+rank)` across N rankings — used inside multi_query and hybrid | Cormack et al. 2009 |
| Memory-Augmented RAG (MemoRAG) | Long-term memory module that summarizes past interactions | Qian et al. 2024 |
| Multi-modal RAG | Embed images alongside text via CLIP/ColPali; retrieve mixed-modality | various |

## Which strategy to pick

A rough decision tree based on workload:

```
Is the corpus large + holistic (need to "summarize everything")?
  └── Yes → GraphRAG (separate system)
  └── No → continue ↓
Are queries factual + use the same vocab as documents?
  └── Yes → direct (cheapest, no LLM rewrite tax)
  └── No → continue ↓
Is the corpus densely written prose (papers, books, manuals)?
  └── Yes → hyde (bridges question/answer gap)
  └── No → continue ↓
Are users likely to phrase queries differently than documents?
  └── Yes → multi_query (vocabulary coverage via variants)
  └── No → continue ↓
Are queries complex / require background context?
  └── Yes → step_back (abstract + concrete retrieval)
  └── No → direct
```

In practice: ship `direct` as the default, A/B against `multi_query` and `hyde` on your eval set, pick whichever wins by RAGAS scores. A 3-5% lift in `context_recall` is meaningful in production.

## Comparative RAGAS evaluation on this repo's corpus

Run `python evals/runner.py --label <strategy>` after setting `RAG_STRATEGY=<strategy>` in `.env` and recreating the API container. Latest results (HR-interview Q&A gold set, 14 questions, judged by local Llama 3.1 8B FP8):

> The full numbers will be filled in by the eval runs. Until then, see `evals/results/`. The `direct` baseline lives at `evals/results/20260425T172044-baseline.json` (answer_relevancy ≈ 0.79).

Once we have all four strategies measured the table will live here as the headline output of this doc.

## Implementation pattern (for future strategies)

Add a new strategy in three steps:

1. Create `src/enterprise_rag/retrieval/strategies/<name>.py` with:
   ```python
   from .base import _base_retrieve, _rerank_and_tag
   NAME = "<name>"
   def retrieve(query: str) -> list[dict]:
       # 1. transform query (call _llm.rewrite if needed)
       # 2. candidates = _base_retrieve(transformed_query)  (or fuse multiple)
       # 3. return _rerank_and_tag(query, candidates, strategy=NAME)
   ```
2. Add `<name>` to the auto-register loop in `strategies/__init__.py`
3. Document it here, including paper/source, mechanism, when it helps, and the cost

The interface is deliberately function-shaped (not class-shaped) so adding a strategy is one short module and one line in `__init__.py`. Class hierarchies wait until we have ≥10 strategies sharing real state.

## Further reading

- **[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)** — the canonical "all techniques" curriculum (42+ notebooks)
- **[langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)** — Lance Martin's video series + notebooks
- **[Microsoft GraphRAG](https://microsoft.github.io/graphrag/)** — official reference for that approach
- **[Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)** — well-written blog on the index-time augmentation
- **[Awesome-RAG (Danielskry)](https://github.com/Danielskry/Awesome-RAG)** — curated link list, regularly updated
- **[Survey: Retrieval-Augmented Generation for LLMs](https://arxiv.org/abs/2312.10997)** — Gao et al. 2023, broad academic survey
