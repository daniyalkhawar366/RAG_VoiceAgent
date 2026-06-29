# Technical Writeup — Genesis CPO RAG Voice Agent

## Approach

The task asked for a voice agent grounded in live inventory. I broke this into three independent concerns and assembled them myself without high-level orchestration frameworks (no LangChain, no LlamaIndex, no Vapi/Retell).

**Scraping** — Playwright drives a headless Chromium browser to walk the paginated inventory, visiting each detail page and parsing the DOM's `innerText` line-by-line. This produces a structured JSON document per listing with price, engine variant, colour, body type, features, and a stable URL slug used as the vector DB document ID.

**RAG layer** — Documents are embedded and stored in a local ChromaDB instance using the default `all-MiniLM-L6-v2` sentence-transformer model. Retrieval is a two-stage pipeline:

1. A regex-based **Natural Language Query Parser** intercepts each query before it hits the vector store. It extracts structured metadata constraints (`price_sar ≤ X`, `body_type = SUV`, etc.) and compiles them into ChromaDB `where` clauses so that arithmetic comparisons are 100% accurate rather than left to embedding similarity.
2. The vector search then runs *only over the pre-filtered subset*, giving us genuine hybrid retrieval without a BM25 library: keyword-derived metadata gating + semantic embedding within the filtered results.

A **similarity distance gate** (L2 threshold = 0.85) discards weak matches before they reach the LLM, ensuring the agent says "I couldn't find that in our inventory" rather than hallucinating a plausible answer from an irrelevant listing.

**Voice loop** — A `ConversationSession` object maintains full turn history, the active vehicle in context, and active search filters. On each turn: (1) audio is recorded with dynamic VAD calibrated to the room's ambient noise floor; (2) transcribed via a direct PCM POST to the Google Web Speech API; (3) an LLM call classifies intent into one of nine categories, routing the query appropriately (skip retrieval on AFFIRMATION/SMALLTALK, run retrieval on INVENTORY\_SEARCH, return a static response on OUT\_OF\_SCOPE without touching the LLM at all); (4) a streaming Groq call feeds a sentence-splitter pipeline that synthesises audio with `edge-tts` and begins playback before generation is complete, cutting perceived latency significantly.

---

## Key Choices and Trade-offs

| Decision | Chosen | Traded Off |
|---|---|---|
| Vector DB | ChromaDB (local file) | Pinecone / Weaviate cloud — sub-millisecond retrieval, zero cost, offline-capable |
| LLM inference | Groq Llama 3.3 70B | GPT-4o / Claude — Groq's ~500 tok/s throughput is essential for a fluid voice loop |
| TTS | edge-tts (Microsoft Neural) | ElevenLabs / OpenAI — free, no billing setup, high fidelity |
| STT | Direct PCM POST to Google Web Speech v2 | Whisper API / Deepgram — no API key required for demo, but unofficial and without custom-vocabulary support |
| Embedding model | all-MiniLM-L6-v2 (local) | OpenAI `text-embedding-3-small` — free, offline, no roundtrip latency |

---

## Assumptions

- The inventory is Genesis CPO vehicles only; the agent is explicitly scoped to decline all other brands.
- Prices are listed in Saudi Riyals (SAR) and are treated as ground truth from the scraped data.
- The deployment target is a single-machine demo environment, so local ChromaDB and local embeddings are appropriate. A distributed deployment would use a hosted vector store.
- English is the only supported language for this iteration.
- Mileage entries labelled "REGISTER YOUR INTEREST" are treated as 0 km / not specified in the index document, rather than being dropped entirely.

---

## What I'd Improve With More Time

1. **Cross-encoder reranking** — After initial vector retrieval, pass the top-10 candidates through a `cross-encoder/ms-marco-MiniLM-L-6-v2` model to reorder by fine-grained relevance before sending to the LLM.
2. **True BM25 hybrid search** — Combine ChromaDB vector scores with a `rank_bm25` keyword score using Reciprocal Rank Fusion, improving exact-string retrieval (e.g., VIN search, exact trim names).
3. **Scheduled scraping with incremental re-indexing** — An APScheduler job would re-scrape at a configurable interval and re-embed only modified listings, keeping the vector store fresh without full rebuilds.
4. **Query-result caching** — A hash-based in-memory cache for repeated queries (e.g., "what's your cheapest car?" asked by multiple callers) to avoid redundant embedding and retrieval work.
5. **Whisper API or Deepgram STT** — Replace the unofficial Google Speech v2 endpoint with a production-grade STT service that supports custom vocabulary (Genesis model names), speaker diarisation, and multilingual input.
6. **Confidence scoring** — Expose the retrieval distance as a confidence signal and use it to trigger a clarifying question ("I found some close matches, but could you confirm — were you asking about the GV80 or the G80?") instead of committing to a potentially wrong answer.
7. **Arabic language support** — Given the SAR currency and likely Saudi market, adding multilingual STT + translation before retrieval would significantly widen the agent's reach.
8. **Retrieval metrics** — Instrument the pipeline with Precision@K and latency-per-stage logging (STT, intent, RAG, LLM first token, TTS) so performance regressions are caught before demos.
